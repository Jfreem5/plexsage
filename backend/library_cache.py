"""Local SQLite cache for Plex library track metadata.

This module provides fast local access to track data by caching Plex library
metadata in a SQLite database. It eliminates the 2+ minute cold start time
for large libraries by syncing once and loading from cache thereafter.
"""

import json
import logging
import random
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backend.models import OWNER_USER_KEY

# The join keys live in the Plex-facing module: this cache's ``track_keys`` table and
# plexclean's datasource tables spell the same key, and one key has one spelling.
from backend.plex_client import album_key, artist_key, song_key

logger = logging.getLogger(__name__)

# Database location
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "library_cache.db"

# Patterns for detecting live recordings (same as plex_client.py)
DATE_PATTERN = r"\d{4}[-/]\d{2}[-/]\d{2}"
LIVE_KEYWORDS = r"\b(?:live|concert|sbd|bootleg)\b"

# Batch size for sync operations (smaller = more frequent progress updates)
SYNC_BATCH_SIZE = 500

# Shortest artist fragment considered for a substring match; "AC" or "a-ha" style
# names are only ever matched exactly, so "abba" cannot drag in "Abba Danza".
ARTIST_MATCH_MIN_LEN = 3

# Cap on the artist roster handed back as filter suggestions (a library can hold
# thousands of artists; the UI only ever needs the most-played / closest names).
MAX_ARTIST_OPTIONS = 60

# Bucket width used to interpolate track-length percentiles on the cache path.
DURATION_BUCKET_MS = 15_000

# Module-level sync state (in-memory for progress tracking)
_sync_state = {
    "is_syncing": False,
    "phase": None,  # "fetching_albums", "fetching", or "processing"
    "current": 0,
    "total": 0,
    "error": None,
}

# Lock to prevent race conditions when starting sync
_sync_lock = threading.Lock()

# Track if schema has been initialized
_schema_initialized = False
_schema_lock = threading.Lock()


def _is_live_version(title: str, album: str) -> bool:
    """Check if track appears to be a live recording based on title/album."""
    for text in [title, album]:
        if re.search(DATE_PATTERN, text):
            return True
        if re.search(LIVE_KEYWORDS, text, re.IGNORECASE):
            return True
    return False


def _normalize_name(name: str) -> str:
    """Lowercase *name* and collapse whitespace (artist matching is forgiving)."""
    return " ".join(name.lower().split())


def _matches_artist(track_artist: str, wanted: list[str]) -> bool:
    """Whether *track_artist* is one of the *wanted* artist names.

    Compares case-insensitively with containment in both directions, so "beatles"
    matches "The Beatles" and "Beatles" matches "The Beatles (Remastered)". Names
    shorter than :data:`ARTIST_MATCH_MIN_LEN` only match exactly.
    """
    if not wanted:
        return True
    have = _normalize_name(track_artist or "")
    for candidate in wanted:
        want = _normalize_name(candidate or "")
        if not want:
            continue
        if want == have:
            return True
        if len(want) >= ARTIST_MATCH_MIN_LEN and want in have:
            return True
        if len(have) >= ARTIST_MATCH_MIN_LEN and have in want:
            return True
    return False


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with WAL mode enabled.

    Returns:
        sqlite3.Connection with row_factory set for dict-like access
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for concurrent reads during writes
    conn.execute("PRAGMA journal_mode=WAL")
    # Set busy timeout for lock contention
    conn.execute("PRAGMA busy_timeout=5000")
    # Enable foreign keys (good practice)
    conn.execute("PRAGMA foreign_keys=ON")

    return conn


def init_schema(conn: sqlite3.Connection) -> bool:
    """Initialize database schema if not exists.

    Args:
        conn: Database connection

    Returns:
        True if a schema migration was applied (existing tracks need re-sync),
        False if schema was already up-to-date or freshly created.
    """
    conn.executescript("""
        -- Tracks table: cached Plex track metadata
        CREATE TABLE IF NOT EXISTS tracks (
            rating_key TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            album TEXT NOT NULL,
            duration_ms INTEGER,
            year INTEGER,
            genres TEXT,
            user_rating INTEGER,
            is_live BOOLEAN,
            parent_rating_key TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes for common query patterns
        CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
        CREATE INDEX IF NOT EXISTS idx_tracks_year ON tracks(year);
        CREATE INDEX IF NOT EXISTS idx_tracks_is_live ON tracks(is_live);

        -- Sync state: single-row metadata table
        CREATE TABLE IF NOT EXISTS sync_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            plex_server_id TEXT,
            last_sync_at TIMESTAMP,
            track_count INTEGER DEFAULT 0,
            sync_duration_ms INTEGER
        );

        -- Ensure sync_state has exactly one row
        INSERT OR IGNORE INTO sync_state (id) VALUES (1);

        -- Results table: persistent storage for generated playlists and recommendations
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            prompt TEXT NOT NULL,
            snapshot JSON NOT NULL,
            track_count INTEGER NOT NULL,
            artist TEXT,
            art_rating_key TEXT,
            playlist_id TEXT,
            playlist_title TEXT,
            playlist_owner TEXT,
            playlist_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_results_type_created ON results(type, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at DESC);

        -- Dimension cache: LLM-extracted musical dimensions for a seed track.
        -- Keyed by rating key + analysis model + metadata fingerprint so a
        -- model swap or edited metadata invalidates the entry automatically.
        CREATE TABLE IF NOT EXISTS track_dimensions (
            rating_key TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            fingerprint TEXT NOT NULL DEFAULT '',
            dimensions JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rating_key, model, fingerprint)
        );

        -- Per-Plex-account ratings. ``tracks.user_rating`` holds only the ratings
        -- of the account whose token we connect with (the household admin), so a
        -- filter asked in another member's stars lives here instead: one row per
        -- (account, track), scanned from that account's own Plex connection. Read
        -- as ``COALESCE(this, tracks.user_rating)``, which is what makes the admin
        -- account the documented fallback for tracks a member never rated.
        CREATE TABLE IF NOT EXISTS user_ratings (
            user_key TEXT NOT NULL,
            rating_key TEXT NOT NULL,
            rating INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_key, rating_key)
        );
        CREATE INDEX IF NOT EXISTS idx_user_ratings_key ON user_ratings(user_key, rating_key);

        -- Which accounts have actually been scanned: a missing line means "this
        -- account's ratings are not loaded yet", which is different from "loaded
        -- but they rated nothing".
        CREATE TABLE IF NOT EXISTS rating_sync (
            user_key TEXT PRIMARY KEY,
            rated_count INTEGER DEFAULT 0,
            total_seen INTEGER DEFAULT 0,
            last_sync_at TIMESTAMP
        );

        -- Join keys of the library's own rows, computed from title/album/artist on
        -- demand (:func:`ensure_track_keys`) rather than stored on ``tracks`` so a
        -- schema upgrade here never asks an existing library to re-sync. The Spotify
        -- import matcher is what reads this table: it asks Plex-internal titles by
        -- exact song key, then by artist/album with a title-prefix GLOB.
        CREATE TABLE IF NOT EXISTS track_keys (
            rating_key TEXT NOT NULL,
            match_key TEXT NOT NULL,
            ck_album TEXT NOT NULL,
            ck_artist TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rating_key)
        );

        -- The Spotify import's fallback rungs ask by artist or album with a
        -- title-prefix GLOB, so index the two equality parts of those queries
        -- (GLOB itself cannot be indexed).
        CREATE INDEX IF NOT EXISTS idx_track_keys_match_key ON track_keys(match_key);
        CREATE INDEX IF NOT EXISTS idx_track_keys_ck_album ON track_keys(ck_album);
        CREATE INDEX IF NOT EXISTS idx_track_keys_ck_artist ON track_keys(ck_artist);

        -- How many library rows the key table above covers (stale keys are rebuilt,
        -- never trusted). This stamp used to ride in the datasource bookkeeping
        -- table, which is gone: plexclean owns third-party facts, so the only
        -- reason this cache keeps keys is the Spotify import's wish list.
        CREATE TABLE IF NOT EXISTS key_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            keyed_tracks INTEGER DEFAULT -1
        );
        INSERT OR IGNORE INTO key_state (id) VALUES (1);

        -- Wish list of the Spotify playlists imported so far: the rows a playlist
        -- asked for that this library cannot play, kept so they can be gotten later.
        -- One row per (playlist, slot): re-importing a playlist REPLACES its rows
        -- rather than stacking them, and other playlists' wishes stay intact.
        CREATE TABLE IF NOT EXISTS missing_songs (
            playlist_ref TEXT NOT NULL,
            position INTEGER NOT NULL DEFAULT 0,
            playlist_name TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            album TEXT NOT NULL DEFAULT '',
            artist TEXT NOT NULL DEFAULT '',
            duration_ms INTEGER NOT NULL DEFAULT 0,
            spotify_track_id TEXT NOT NULL DEFAULT '',
            spotify_url TEXT NOT NULL DEFAULT '',
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (playlist_ref, position)
        );
    """)

    # Migration: add parent_rating_key column if missing (for pre-existing databases)
    migrated = False
    try:
        conn.execute("ALTER TABLE tracks ADD COLUMN parent_rating_key TEXT")
        migrated = True
        logger.info("Migration applied: added parent_rating_key column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add view_count and last_viewed_at columns for familiarity tracking
    try:
        conn.execute("ALTER TABLE tracks ADD COLUMN view_count INTEGER DEFAULT 0")
        migrated = True
        logger.info("Migration applied: added view_count column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        conn.execute("ALTER TABLE tracks ADD COLUMN last_viewed_at TEXT")
        migrated = True
        logger.info("Migration applied: added last_viewed_at column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add subtitle column to results table
    try:
        conn.execute("ALTER TABLE results ADD COLUMN subtitle TEXT")
        logger.info("Migration applied: added subtitle column to results")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: link history rows to the Plex playlist they were saved as.
    # Deliberately not folded into `migrated`: these columns are on `results`,
    # so they say nothing about the `tracks` schema and must not ask for a re-sync.
    for _results_column in ("playlist_id", "playlist_title", "playlist_owner", "playlist_url"):
        try:
            conn.execute(f"ALTER TABLE results ADD COLUMN {_results_column} TEXT")
            logger.info("Migration applied: added %s column to results", _results_column)
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Index on parent_rating_key (must come after migration adds the column)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_parent_key ON tracks(parent_rating_key)")

    # Third-party datasources moved to plexclean, so a DB upgraded from an older
    # plexsage keeps its now-dead tables; drop them so the cache has exactly the
    # shape above. Deliberately not folded into `migrated`: dropping tables says
    # nothing about the `tracks` schema, so it must not ask for a re-sync.
    for _dead_table in ("external_data", "external_state"):
        try:
            conn.execute(f"DROP TABLE {_dead_table}")
            logger.info("Dropped the %s table: datasources belong to plexclean", _dead_table)
        except sqlite3.OperationalError:
            pass  # Already gone (a fresh DB never had it)

    conn.commit()
    return migrated


# Whether a migration was applied on startup (signals need for re-sync)
_migration_applied = False


def ensure_db_initialized() -> sqlite3.Connection:
    """Ensure database exists and schema is initialized.

    Returns:
        Initialized database connection
    """
    global _schema_initialized, _migration_applied
    conn = get_db_connection()

    # Only initialize schema once per process; lock prevents races on startup
    if not _schema_initialized:
        with _schema_lock:
            if not _schema_initialized:
                _migration_applied = init_schema(conn)
                _schema_initialized = True

    return conn


def rebuild_track_keys(conn: sqlite3.Connection) -> int:
    """Recompute the join key of every cached track; returns keys written.

    The key is plexsage's own normalization of title, album and artist
    (:func:`backend.plex_client.song_key`), which is the spelling plexclean's
    third-party tools compute the same way, so one key has one spelling across the
    two repos. Keeping these keys in their own table instead of columns on
    ``tracks`` is what lets an existing library pick the feature up without a
    schema migration that would demand a full re-sync.
    """
    conn.execute("DELETE FROM track_keys")
    rows = conn.execute("SELECT rating_key, title, album, artist FROM tracks").fetchall()
    written = 0
    batch: list[tuple[str, str, str, str]] = []
    for row in rows:
        key = song_key(row["title"], row["album"], row["artist"])
        batch.append(
            (
                row["rating_key"],
                key,
                album_key(row["album"]),
                artist_key(row["artist"]),
            )
        )
        if len(batch) >= 5000:
            written += _write_keys(conn, batch)
            batch = []
    written += _write_keys(conn, batch)
    conn.execute("UPDATE key_state SET keyed_tracks = ?", (written,))
    conn.commit()
    return written


def _write_keys(conn: sqlite3.Connection, batch) -> int:
    """Insert one batch of computed keys; empty batches cost nothing."""
    if not batch:
        return 0
    conn.executemany(
        "INSERT INTO track_keys (rating_key, match_key, ck_album, ck_artist) VALUES (?, ?, ?, ?)",
        batch,
    )
    return len(batch)


def ensure_track_keys(conn: sqlite3.Connection) -> int:
    """Keys for the library rows, rebuilt when they no longer line up with tracks.

    A count mismatch is enough to catch the two ways rows change here (a sync adds
    or drops tracks; a datasource refresh rebuilds them outright), and rebuilding
    costs a second for tens of thousands of rows, so the stale case is repaired
    rather than trusted.
    """
    try:
        keyed = conn.execute("SELECT keyed_tracks FROM key_state").fetchone()["keyed_tracks"]
    except sqlite3.Error:
        return 0
    total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    keyed = int(keyed or -1)
    if keyed != total:
        return rebuild_track_keys(conn)
    return keyed


def get_sync_state() -> dict[str, Any]:
    """Get current sync state from database and in-memory state.

    Returns:
        Dict with track_count, synced_at, is_syncing, sync_progress, error
    """
    conn = ensure_db_initialized()
    try:
        row = conn.execute(
            "SELECT plex_server_id, last_sync_at, track_count, sync_duration_ms "
            "FROM sync_state WHERE id = 1"
        ).fetchone()

        # Snapshot sync state under lock for consistent reads
        with _sync_lock:
            ss = dict(_sync_state)

        result = {
            "track_count": row["track_count"] if row else 0,
            "synced_at": row["last_sync_at"] if row else None,
            "plex_server_id": row["plex_server_id"] if row else None,
            "sync_duration_ms": row["sync_duration_ms"] if row else None,
            "is_syncing": ss["is_syncing"],
            "sync_progress": None,
            "error": ss["error"],
        }

        if ss["is_syncing"]:
            result["sync_progress"] = {
                "phase": ss["phase"],
                "current": ss["current"],
                "total": ss["total"],
            }

        return result
    finally:
        conn.close()


def get_cached_tracks() -> list[dict[str, Any]]:
    """Get all tracks from cache.

    Returns:
        List of track dicts with all fields
    """
    conn = ensure_db_initialized()
    try:
        rows = conn.execute(
            "SELECT rating_key, title, artist, album, duration_ms, year, "
            "genres, user_rating, is_live FROM tracks"
        ).fetchall()

        tracks = []
        for row in rows:
            track = dict(row)
            # Parse genres JSON
            if track["genres"]:
                track["genres"] = json.loads(track["genres"])
            else:
                track["genres"] = []
            tracks.append(track)

        return tracks
    finally:
        conn.close()


def get_tracks_by_filters(
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    artists: list[str] | None = None,
    exclude_genres: list[str] | None = None,
    min_rating: int = 0,
    min_duration_ms: int = 0,
    max_duration_ms: int = 0,
    exclude_live: bool = True,
    limit: int = 0,
    exclude_artists: list[str] | None = None,
    exclude_decades: list[str] | None = None,
    min_year: int = 0,
    max_year: int = 0,
    min_plays: int = 0,
    max_plays: int = 0,
    unheard_only: bool = False,
    exclude_compilations: bool = False,
    live_only: bool = False,
    rated_only: bool = False,
    unrated_only: bool = False,
    max_rating: int = 0,
    played_after: str = "",
    played_before: str = "",
    rating_user: str = "",
) -> list[dict[str, Any]]:
    """Get tracks from cache matching filter criteria.

    Args:
        genres: List of genre names to include (OR matching)
        decades: List of decades like "1990s" (OR matching)
        artists: Artist names to restrict to (OR matching, case-insensitive
                 containment so "beatles" finds "The Beatles")
        exclude_genres: Genre names to drop (a track carrying one is excluded)
        min_rating: Minimum user rating (0-10, 0 = no filter)
        min_duration_ms: Minimum track length in milliseconds (0 = no filter)
        max_duration_ms: Maximum track length in milliseconds (0 = no filter)
        exclude_live: Whether to exclude live recordings
        limit: Max tracks to return (0 = no limit)
        exclude_artists: Artist names to drop (same containment rules as
                 ``artists``, e.g. "no Autolectric" style bans)
        exclude_decades: Decades like "1980s" to drop; undated tracks survive
        min_year: Oldest acceptable release year (0 = no lower bound)
        max_year: Newest acceptable release year (0 = no upper bound)
        min_plays: Minimum Plex play count (0 = no floor)
        max_plays: Maximum Plex play count (0 = no ceiling)
        unheard_only: Require a zero play count ("hidden gems")
        exclude_compilations: Drop tracks from Various Artists compilations
        live_only: Keep only live recordings ("just the live takes"); this wins
                   over ``exclude_live`` so the two cannot cancel out to nothing
        rated_only: Keep only tracks the user rated (``unrated_only``'s opposite)
        unrated_only: Keep only tracks the user never rated
        max_rating: User-rating ceiling (0-10, 0 = no ceiling); unrated tracks
                    stay because they are not above it
        played_after / played_before: Inclusive last-played window as
                 ``YYYY-MM-DD`` dates ("" = unbounded on that side)

    Returns:
        List of matching track dicts

    ``rating_user`` names whose ratings the rating facets mean; the rows' own
    ``user_rating`` column still carries the connecting account's stars.
    """
    conn = ensure_db_initialized()
    try:
        sql_and_params = _filter_conditions(
            decades=decades,
            exclude_decades=exclude_decades,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            min_plays=min_plays,
            max_plays=max_plays,
            unheard_only=unheard_only,
            exclude_compilations=exclude_compilations,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            exclude_live=exclude_live,
            live_only=live_only,
            rated_only=rated_only,
            unrated_only=unrated_only,
            max_rating=max_rating,
            played_after=played_after,
            played_before=played_before,
            rating_user=rating_user,
        )
        if sql_and_params is None:
            where_clause, params = "1=1", []
        else:
            where_clause, params = sql_and_params
        query = f"SELECT * FROM tracks WHERE {where_clause}"

        # Genre, artist and exclusion-by-name filtering happen in Python (genres
        # are JSON and artist names need containment matching), so a SQL LIMIT
        # would drop valid candidates. Sample after filtering instead.
        python_filters = bool(genres) or bool(artists) or bool(exclude_genres) or bool(exclude_artists)

        if limit > 0 and not python_filters:
            query += " ORDER BY RANDOM() LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()
        tracks = []

        for row in rows:
            track = dict(row)
            # Parse genres JSON
            if track["genres"]:
                track["genres"] = json.loads(track["genres"])
            else:
                track["genres"] = []

            # Genre/artist/exclusion filtering (JSON field + name containment)
            if genres:
                track_genres = [g.lower() for g in track["genres"]]
                if not any(g.lower() in track_genres for g in genres):
                    continue

            if exclude_genres:
                track_genres = [g.lower() for g in track["genres"]]
                if any(g.lower() in track_genres for g in exclude_genres):
                    continue

            if artists and not _matches_artist(track["artist"], artists):
                continue

            if exclude_artists and _matches_artist(track["artist"], exclude_artists):
                continue

            tracks.append(track)

        # Apply limit after genre filtering with random sampling
        if limit > 0 and python_filters and len(tracks) > limit:
            tracks = random.sample(tracks, limit)

        return tracks
    finally:
        conn.close()


def get_metadata_batch(rating_keys: list[str], rating_user: str = "") -> dict[str, dict[str, Any]]:
    """Fetch ranking metadata for a batch of tracks.

    Used by rule-based pre-ranking, which needs the user rating and the last
    played timestamp for a whole candidate pool without loading every column.

    Args:
        rating_keys: Rating keys to look up.
        rating_user: Plex account whose ratings the ``user_rating`` value must
            carry, so pre-ranking's bonus ranks the stars the request was made
            against rather than the connecting account's. Tracks that account
            never rated keep the admin column, which is the same fallback the
            SQL filters apply.

    Returns:
        Mapping of rating_key -> {"user_rating", "view_count", "last_viewed_at"}.
        Missing keys are simply absent from the result.
    """
    if not rating_keys:
        return {}

    conn = ensure_db_initialized()
    try:
        result: dict[str, dict[str, Any]] = {}
        if rating_user:
            rating_column = (
                "COALESCE((SELECT rating FROM user_ratings "
                "WHERE user_key = ? AND rating_key = tracks.rating_key), user_rating)"
            )
        else:
            rating_column = "user_rating"

        # SQLite's bound-variable limit is 999 by default; chunk to stay under it.
        chunk_size = 900
        for start in range(0, len(rating_keys), chunk_size):
            batch = rating_keys[start:start + chunk_size]
            placeholders = ",".join("?" for _ in batch)
            params = ([rating_user] if rating_user else []) + list(batch)
            rows = conn.execute(
                f"SELECT rating_key, {rating_column} AS rating_effective, view_count, last_viewed_at "
                f"FROM tracks WHERE rating_key IN ({placeholders})",
                params,
            ).fetchall()
            for row in rows:
                result[row["rating_key"]] = {
                    "user_rating": row["rating_effective"],
                    "view_count": row["view_count"],
                    "last_viewed_at": row["last_viewed_at"],
                }
        return result
    finally:
        conn.close()


def has_user_ratings(user_key: str) -> bool:
    """True when that Plex account's ratings have already been scanned in.

    An account with no rows here is not "this member rated nothing": it means
    their ratings were never loaded, and the effective rating expression
    silently falls back to the admin column for every track. The status table
    (:func:`user_ratings_status`) is what distinguishes the two cases, so this
    only gates the scan.

    ``SELECT EXISTS (SELECT 1 FROM ...)`` is the spelling SQLite accepts; the
    scalar ``EXISTS(table)`` form is a syntax error, and because any
    :class:`sqlite3.Error` is reported as "no ratings", that wrong form would
    turn the scan (and the cost accounting that reads it) off even with rows
    stored.
    """
    conn = ensure_db_initialized()
    try:
        row = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM user_ratings WHERE user_key = ?)", (user_key,)
        ).fetchone()
        return bool(row[0])
    except sqlite3.Error as e:
        logger.warning("Cannot check stored ratings for '%s': %s", user_key, e)
        return False
    finally:
        conn.close()


def user_ratings_status() -> dict[str, Any]:
    """Summary of the per-account ratings store, for the status route.

    Returns:
        Dict with ``total_tracks``, the accounts already scanned (``synced_users``)
        and per-account ``users`` lines (rated count, library size at scan time and
        the scan stamp).
    """
    conn = ensure_db_initialized()
    try:
        total = conn.execute("SELECT COUNT(1) FROM tracks").fetchone()[0]
        rows = conn.execute(
            "SELECT user_key, rated_count, total_seen, last_sync_at FROM rating_sync"
        ).fetchall()
        users = [
            {
                "user_key": row["user_key"],
                "rated_count": row["rated_count"],
                "total_seen": row["total_seen"],
                "last_sync_at": row["last_sync_at"],
            }
            for row in rows
        ]
        return {"total_tracks": total, "synced_users": [u["user_key"] for u in users], "users": users}
    except sqlite3.Error as e:
        logger.warning("Cannot read ratings status: %s", e)
        return {"total_tracks": 0, "synced_users": [], "users": []}
    finally:
        conn.close()


def sync_user_ratings(
    user_key: str,
    plex_client: Any,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Scan one Plex account's own star ratings into the cache; rows stored.

    ``tracks.user_rating`` holds only the ratings of the account whose token we
    connect with (the household admin), so a filter asked in another member's
    stars needs this second, much smaller store: one row per account per track
    they rated, read through the ``COALESCE`` in :func:`_filter_conditions`,
    which is what makes the admin column the fallback for tracks they never
    rated.

    Args:
        user_key: Plex account to scan (a household member's exact title, or a
            shared friend's username). ``__owner__`` and "" name the admin, whose
            ratings are already the ``user_rating`` column: nothing to store.
        plex_client: Connected :class:`plex_client.PlexClient` (the admin's), which
            is the only connection that may ``switchUser``.
        on_progress: Optional callback(current, total) while the rows are written.

    Returns:
        Rows stored (0 when the account is the admin or rated nothing).

    Raises:
        PlexQueryError: unknown account or the scan as it failed. Silently
            returning 0 here would read as "this member rated nothing" and drop
            every one of their rated tracks from a ``min_rating`` filter.
    """
    key = (user_key or "").strip()
    if not key or key == OWNER_USER_KEY:
        # The admin's stars are already every track's ``user_rating`` column.
        return 0

    rated = plex_client.collect_user_ratings(key)

    conn = ensure_db_initialized()
    try:
        # Replace, never stack: a rescanned account can un-rate a track.
        conn.execute("DELETE FROM user_ratings WHERE user_key = ?", (key,))
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            # updated_at carries a DEFAULT, so an INSERT that names it would need
            # the stamp repeated 50k times.
            "INSERT INTO user_ratings(user_key, rating_key, rating) VALUES (?, ?, ?)",
            ((key, rating_key, stars) for rating_key, stars in rated),
        )
        seen = conn.execute("SELECT COUNT(1) FROM tracks").fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO rating_sync"
            "(user_key, rated_count, total_seen, last_sync_at) VALUES (?, ?, ?, ?)",
            (key, len(rated), seen, now),
        )
        conn.commit()
        if on_progress:
            on_progress(len(rated), len(rated))
        return len(rated)
    except sqlite3.Error as e:
        conn.rollback()
        logger.warning("Cannot store ratings for '%s': %s", key, e)
        return 0
    finally:
        conn.close()


def ensure_user_ratings(user_key: str, plex_client: Any) -> int:
    """Load an account's ratings the first time a filter asks for them.

    Returns:
        Rows stored, 0 when they were already cached (the common case: this runs
        per request, and only the first one pays for the scan).
    """
    key = (user_key or "").strip()
    if not key or key == OWNER_USER_KEY or has_user_ratings(key):
        return 0
    return sync_user_ratings(key, plex_client)


def get_dimensions(
    rating_key: str,
    model: str = "",
    fingerprint: str = "",
) -> list[dict[str, Any]] | None:
    """Return cached musical dimensions for a seed track, or None on a miss.

    Args:
        rating_key: Plex rating key of the seed track.
        model: Analysis model that produced the entry.
        fingerprint: Hash of the track metadata the entry depends on.

    Returns:
        List of dimension dicts, or None if nothing is cached for this key.
    """
    conn = ensure_db_initialized()
    try:
        row = conn.execute(
            "SELECT dimensions FROM track_dimensions "
            "WHERE rating_key = ? AND model = ? AND fingerprint = ?",
            (rating_key, model, fingerprint),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["dimensions"])
    finally:
        conn.close()


def save_dimensions(
    rating_key: str,
    dimensions: list[dict[str, Any]],
    model: str = "",
    fingerprint: str = "",
) -> None:
    """Cache musical dimensions for a seed track.

    Args:
        rating_key: Plex rating key of the seed track.
        dimensions: Dimension dicts as returned by the analysis model.
        model: Analysis model that produced them.
        fingerprint: Hash of the track metadata they were derived from.
    """
    conn = ensure_db_initialized()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO track_dimensions "
            "(rating_key, model, fingerprint, dimensions) VALUES (?, ?, ?, ?)",
            (rating_key, model, fingerprint, json.dumps(dimensions)),
        )
        conn.commit()
    finally:
        conn.close()


def clear_dimensions() -> None:
    """Delete all cached seed-track dimensions."""
    conn = ensure_db_initialized()
    try:
        conn.execute("DELETE FROM track_dimensions")
        conn.commit()
    finally:
        conn.close()


def clear_cache() -> None:
    """Clear all cached tracks and reset sync state."""
    conn = ensure_db_initialized()
    try:
        conn.execute("DELETE FROM tracks")
        conn.execute(
            "UPDATE sync_state SET last_sync_at = NULL, track_count = 0, "
            "sync_duration_ms = NULL WHERE id = 1"
        )
        conn.commit()
        logger.info("Cache cleared")
    finally:
        conn.close()


def is_cache_stale(max_age_hours: int = 24) -> bool:
    """Check if cache is older than max_age_hours.

    Args:
        max_age_hours: Maximum cache age in hours

    Returns:
        True if cache is stale or empty
    """
    state = get_sync_state()
    if not state["synced_at"]:
        return True

    try:
        # Parse ISO timestamp
        synced_at = datetime.fromisoformat(state["synced_at"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - synced_at).total_seconds() / 3600
        return age_hours > max_age_hours
    except (ValueError, TypeError):
        return True


def check_server_changed(current_server_id: str) -> bool:
    """Check if Plex server has changed since last sync.

    Args:
        current_server_id: Current Plex server's machineIdentifier

    Returns:
        True if server changed (cache should be cleared)
    """
    state = get_sync_state()
    cached_server_id = state.get("plex_server_id")

    if not cached_server_id:
        return False  # First sync, no change

    return cached_server_id != current_server_id


def sync_library(
    plex_client: Any,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Sync tracks from Plex to local cache.

    This is a blocking synchronous operation. For async usage, wrap in
    asyncio.to_thread() or run in a thread pool.

    Args:
        plex_client: PlexClient instance with active connection
        on_progress: Optional callback(current, total) for progress updates

    Returns:
        Dict with success, track_count, duration_ms, error
    """
    global _sync_state

    # Use lock to prevent race condition between check and set
    with _sync_lock:
        if _sync_state["is_syncing"]:
            return {"success": False, "error": "Sync already in progress"}

        _sync_state = {
            "is_syncing": True,
            "phase": "fetching_albums",
            "current": 0,
            "total": 0,
            "error": None,
        }

    start_time = time.time()
    conn = None

    try:
        # Get server ID for cache validation
        server_id = plex_client.get_machine_identifier()
        if not server_id:
            raise ValueError("Could not get Plex server identifier")

        # Check if server changed - clear cache if so
        if check_server_changed(server_id):
            logger.info("Plex server changed, clearing cache")
            clear_cache()

        conn = ensure_db_initialized()

        # Clear existing tracks and reset sync state to avoid stale "cache available"
        # signal if sync fails partway through
        conn.execute("DELETE FROM tracks")
        conn.execute("UPDATE sync_state SET track_count = 0 WHERE id = 1")
        conn.commit()

        # Phase 1: Fetch albums for genre/year mapping
        logger.info("Fetching album metadata from Plex...")
        album_metadata = plex_client.get_all_albums_metadata()
        logger.info("Got metadata for %d albums", len(album_metadata))

        # Phase 2: Fetch all tracks from Plex
        with _sync_lock:
            _sync_state["phase"] = "fetching"
        logger.info("Fetching all tracks from Plex (this may take 30-60s)...")
        all_tracks = plex_client.get_all_raw_tracks()
        total = len(all_tracks)
        logger.info("Got %d tracks from Plex", total)

        with _sync_lock:
            _sync_state["total"] = total
            _sync_state["phase"] = "processing"

        # Phase 3: Process tracks in batches with album metadata lookup
        synced_count = 0
        batch_data = []

        for i, track in enumerate(all_tracks):
            # Extract track data
            title = track.title
            album = getattr(track, "parentTitle", "") or ""
            artist = getattr(track, "grandparentTitle", "") or "Unknown Artist"

            # Look up genres and year from album metadata using parentRatingKey
            parent_key = str(getattr(track, "parentRatingKey", ""))
            album_data = album_metadata.get(parent_key, {})
            genres = album_data.get("genres", [])
            year = album_data.get("year")

            # Extract play history data
            view_count = getattr(track, "viewCount", 0) or 0
            last_viewed_at_raw = getattr(track, "lastViewedAt", None)
            last_viewed_at = last_viewed_at_raw.isoformat() if last_viewed_at_raw else None

            batch_data.append((
                str(track.ratingKey),
                title,
                artist,
                album,
                track.duration or 0,
                year,
                json.dumps(genres),  # Store genres as JSON array
                getattr(track, "userRating", None),
                _is_live_version(title, album),
                parent_key,
                view_count,
                last_viewed_at,
            ))

            # Insert and update progress every SYNC_BATCH_SIZE tracks
            if len(batch_data) >= SYNC_BATCH_SIZE:
                conn.executemany(
                    "INSERT OR REPLACE INTO tracks "
                    "(rating_key, title, artist, album, duration_ms, year, genres, "
                    "user_rating, is_live, parent_rating_key, view_count, last_viewed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch_data,
                )
                synced_count += len(batch_data)
                batch_data = []

                # Update progress (single field, atomic under CPython GIL)
                with _sync_lock:
                    _sync_state["current"] = synced_count
                if on_progress:
                    on_progress(synced_count, total)

                logger.info("Synced %d/%d tracks", synced_count, total)

                # Commit every batch to allow concurrent reads (WAL mode)
                conn.commit()

        # Insert remaining tracks
        if batch_data:
            conn.executemany(
                "INSERT OR REPLACE INTO tracks "
                "(rating_key, title, artist, album, duration_ms, year, genres, "
                "user_rating, is_live, parent_rating_key, view_count, last_viewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch_data,
            )
            synced_count += len(batch_data)
            with _sync_lock:
                _sync_state["current"] = synced_count

        # Final commit
        conn.commit()

        # Update sync state
        duration_ms = int((time.time() - start_time) * 1000)
        synced_at = datetime.now(timezone.utc).isoformat()

        conn.execute(
            "UPDATE sync_state SET plex_server_id = ?, last_sync_at = ?, "
            "track_count = ?, sync_duration_ms = ? WHERE id = 1",
            (server_id, synced_at, synced_count, duration_ms),
        )
        conn.commit()

        logger.info("Sync complete: %d tracks in %dms", synced_count, duration_ms)

        # Clear migration flag — new columns are now populated
        global _migration_applied
        _migration_applied = False

        return {
            "success": True,
            "track_count": synced_count,
            "duration_ms": duration_ms,
        }

    except Exception as e:
        logger.exception("Sync failed: %s", e)
        with _sync_lock:
            _sync_state["error"] = str(e)
        return {"success": False, "error": str(e)}

    finally:
        with _sync_lock:
            _sync_state["is_syncing"] = False
            _sync_state["phase"] = None
            _sync_state["current"] = 0
            _sync_state["total"] = 0
        if conn:
            conn.close()


def get_sync_progress() -> dict[str, Any]:
    """Get current sync progress (for polling).

    Returns:
        Dict with is_syncing, phase, current, total, error
    """
    with _sync_lock:
        return dict(_sync_state)


def _is_compilation(artist: str | None, album: str | None) -> bool:
    """Heuristic for multi-artist "best of" style albums, usable on both paths.

    Plex has no compilation flag, so the only signal available without a schema
    change is the placeholder artist. An album filed under Various Artists is a
    compilation by definition; a Various-Artists track on an album that names a
    real artist is a soundtrack or tribute appearance, which is not excluded.
    """
    return (artist or "").strip().lower() == "various artists"


def _decade_start_year(decade: str) -> int | None:
    """First year of a decade label like "1990s"/"1990", or None if unparseable."""
    try:
        return int(str(decade).strip().rstrip("s"))
    except ValueError:
        return None


def _filter_conditions(
    decades: list[str] | None,
    exclude_decades: list[str] | None,
    min_year: int = 0,
    max_year: int = 0,
    min_rating: int = 0,
    min_plays: int = 0,
    max_plays: int = 0,
    unheard_only: bool = False,
    exclude_compilations: bool = False,
    min_duration_ms: int = 0,
    max_duration_ms: int = 0,
    exclude_live: bool = True,
    live_only: bool = False,
    rated_only: bool = False,
    unrated_only: bool = False,
    max_rating: int = 0,
    played_after: str = "",
    played_before: str = "",
    rating_user: str = "",
) -> tuple[str, list[Any]] | None:
    """SQL WHERE body for the filters SQLite can answer exactly.

    Returns the clause text plus bind parameters, or None when nothing is
    filtered so the caller can skip the clause. Genre and artist matching stays
    in Python (genres are a JSON column and artist names need the containment
    rules in :func:`_matches_artist`), but everything numeric is decided here so
    ``count_tracks_by_filters`` and ``get_tracks_by_filters`` cannot drift apart
    when a filter is added.

    The newer facets read columns Plex already fills: ``is_live`` (a ``live_only``
    request), ``user_rating`` (``rated_only`` / ``unrated_only`` / a ``max_rating``
    ceiling) and ``last_viewed_at`` (the ``played_after`` / ``played_before``
    window).

    ``rating_user`` names a Plex household account whose ratings those stars mean
    instead of the connecting account's. ``tracks.user_rating`` only ever holds the
    latter, so the rating tests read an effective value: the account's own row in
    ``user_ratings`` when there is one, otherwise the admin column, which is the
    fallback the feature promises for tracks that account never rated, and correlates on ``tracks.rating_key``: the outer FROM must stay
    named ``tracks`` in both callers below.

    """
    conditions: list[str] = []
    params: list[Any] = []

    # Whose stars the rating tests below mean. A bare name inside the subquery
    # resolves to ``user_ratings``, so the OUTER row has to be table-qualified.
    # An account whose ratings were never loaded joins to nothing, and COALESCE
    # then hands back the admin column: the fallback, without a special case.
    rating_expr = "user_rating"
    rating_params: list[Any] = []
    if rating_user:
        rating_expr = (
            "COALESCE((SELECT rating FROM user_ratings "
            "WHERE user_key = ? AND rating_key = tracks.rating_key), user_rating)"
        )
        rating_params = [rating_user]

    if live_only:
        # "only live takes". Authoritative over exclude_live: a caller that left
        # the default drop-live flag on would otherwise match nothing at all.
        conditions.append("is_live = 1")
    elif exclude_live:
        conditions.append("is_live = 0")

    if rated_only:
        # Plex files an unrated track as NULL, never as 0, so "music I have rated"
        # is a NOT NULL test rather than a rating floor of 1.
        conditions.append(f"{rating_expr} IS NOT NULL")
        params.extend(rating_params)
    elif unrated_only:
        conditions.append(f"{rating_expr} IS NULL")
        params.extend(rating_params)

    if max_rating > 0:
        # A ceiling bans music the user already loved; a track they never
        # rated is not above it, so the NULL branch survives like it does for
        # a banned decade.
        conditions.append(f"({rating_expr} IS NULL OR {rating_expr} <= ?)")
        params.extend(rating_params)
        params.append(max_rating)

    if min_rating > 0:
        conditions.append(f"{rating_expr} >= ?")
        params.extend(rating_params)
        params.append(min_rating)

    if min_plays > 0:
        conditions.append("view_count >= ?")
        params.append(min_plays)

    if unheard_only:
        # "hidden gems": never played. 0 means unbounded elsewhere in this file,
        # so a play ceiling cannot express this and it gets its own flag.
        conditions.append("COALESCE(view_count, 0) = 0")
    elif max_plays > 0:
        conditions.append("COALESCE(view_count, 0) <= ?")
        params.append(max_plays)

    if min_year > 0:
        conditions.append("year >= ?")
        params.append(min_year)

    if max_year > 0:
        conditions.append("year <= ?")
        params.append(max_year)

    if decades:
        decade_conditions = []
        for decade in decades:
            start_year = _decade_start_year(decade)
            if start_year is None:
                continue
            decade_conditions.append(f"(year >= {start_year} AND year < {start_year + 10})")
        if decade_conditions:
            conditions.append("(" + " OR ".join(decade_conditions) + ")")

    if exclude_decades:
        # Tracks with an unknown year are not inside a banned decade, so keep
        # the NULL branch; otherwise a decade ban would silently drop undated
        # music.
        for decade in exclude_decades:
            start_year = _decade_start_year(decade)
            if start_year is None:
                continue
            conditions.append(
                f"(year IS NULL OR year < {start_year} OR year >= {start_year + 10})"
            )

    if exclude_compilations:
        # Plex has no compilation flag; a track filed under the "Various Artists"
        # placeholder is from a multi-artist "best of" album.
        conditions.append("COALESCE(artist, '') != 'Various Artists'")

    if min_duration_ms > 0:
        conditions.append("duration_ms >= ?")
        params.append(min_duration_ms)

    if max_duration_ms > 0:
        conditions.append("duration_ms <= ?")
        params.append(max_duration_ms)

    if played_after or played_before:
        # The cache holds Plex's lastViewedAt as an ISO timestamp
        # ("2026-03-02T07:57:35"), so the window compares only the leading
        # YYYY-MM-DD date: both bounds are inclusive by day, and a never-played
        # row is NULL here and fails either bound, which is what a "played
        # recently" request wants.
        dated = "SUBSTR(last_viewed_at, 1, 10)"
        if played_after:
            conditions.append(f"{dated} >= ?")
            params.append(played_after)
        if played_before:
            conditions.append(f"{dated} <= ?")
            params.append(played_before)

    if not conditions:
        return None
    return " AND ".join(conditions), params


def count_tracks_by_filters(
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    artists: list[str] | None = None,
    exclude_genres: list[str] | None = None,
    min_rating: int = 0,
    min_duration_ms: int = 0,
    max_duration_ms: int = 0,
    exclude_live: bool = True,
    exclude_artists: list[str] | None = None,
    exclude_decades: list[str] | None = None,
    min_year: int = 0,
    max_year: int = 0,
    min_plays: int = 0,
    max_plays: int = 0,
    unheard_only: bool = False,
    exclude_compilations: bool = False,
    live_only: bool = False,
    rated_only: bool = False,
    unrated_only: bool = False,
    max_rating: int = 0,
    played_after: str = "",
    played_before: str = "",
    rating_user: str = "",
) -> int:
    """Count tracks matching filter criteria without fetching full data.

    Args:
        genres: List of genre names to include (OR matching)
        decades: List of decades like "1990s" (OR matching)
        artists: Artist names to restrict to (OR matching)
        exclude_genres: Genre names to drop (OR matching)
        min_rating: Minimum user rating (0-10, 0 = no filter)
        min_duration_ms: Minimum track length (0 = no filter)
        max_duration_ms: Maximum track length (0 = no filter)
        exclude_live: Whether to exclude live recordings
        exclude_artists: Artist names to drop (case-insensitive containment)
        exclude_decades: Decades like "1980s" to drop, e.g. "no 80s hair metal"
        min_year: Oldest acceptable release year (0 = no lower bound)
        max_year: Newest acceptable release year (0 = no upper bound)
        min_plays: Minimum Plex play count (0 = no floor)
        max_plays: Maximum Plex play count (0 = no ceiling)
        unheard_only: Require a zero play count ("hidden gems")
        exclude_compilations: Drop tracks from Various Artists albums
        live_only: Keep only live recordings ("only live takes"); overrides
                   ``exclude_live`` so the pair cannot cancel out to zero rows
        rated_only: Require the track to carry a user rating
        unrated_only: Require the track to carry no user rating
        max_rating: User-rating ceiling (0-10, 0 = no ceiling); unrated rows stay
        played_after / played_before: Inclusive last-played window as
                 ``YYYY-MM-DD`` dates ("" = unbounded on that side)
        rating_user: Plex account whose ratings the rating facets mean ("" = the
                 connecting account, whose stars are the ``user_rating`` column)

    Returns:
        Count of matching tracks, or -1 if cache is empty
    """
    state = get_sync_state()
    if state["track_count"] == 0:
        return -1  # Cache empty, signal to use Plex

    conn = ensure_db_initialized()
    try:
        sql_and_params = _filter_conditions(
            decades=decades,
            exclude_decades=exclude_decades,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            min_plays=min_plays,
            max_plays=max_plays,
            unheard_only=unheard_only,
            exclude_compilations=exclude_compilations,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            exclude_live=exclude_live,
            live_only=live_only,
            rated_only=rated_only,
            unrated_only=unrated_only,
            max_rating=max_rating,
            played_after=played_after,
            played_before=played_before,
            rating_user=rating_user,
        )
        if sql_and_params is None:
            where_clause, params = "1=1", []
        else:
            where_clause, params = sql_and_params

        # Genre, artist and exclusion-by-name matching happen in Python (genres
        # are a JSON column and artist names need containment rules), so the
        # shortcut COUNT(*) query is only valid when none of those are set.
        python_filters = bool(genres) or bool(artists) or bool(exclude_genres) or bool(exclude_artists)

        if not python_filters:
            query = f"SELECT COUNT(*) FROM tracks WHERE {where_clause}"
            return conn.execute(query, params).fetchone()[0]

        query = f"SELECT genres, artist FROM tracks WHERE {where_clause}"
        rows = conn.execute(query, params).fetchall()

        count = 0
        genres_lower = [g.lower() for g in genres] if genres else []
        excluded_lower = [g.lower() for g in exclude_genres] if exclude_genres else []
        for row in rows:
            track_genres_lower = (
                [g.lower() for g in json.loads(row["genres"])] if row["genres"] else []
            )
            if genres_lower and not any(g in track_genres_lower for g in genres_lower):
                continue
            if excluded_lower and any(g in track_genres_lower for g in excluded_lower):
                continue
            if artists and not _matches_artist(row["artist"], artists):
                continue
            if exclude_artists and _matches_artist(row["artist"], exclude_artists):
                continue
            count += 1

        return count
    finally:
        conn.close()


# Facet -> the value that switches it off, so a tier can clear it. Named here (not
# inline in :func:`relax_filters_for_target`) because these must stay the same
# neutral values :func:`count_tracks_by_filters` and :func:`get_tracks_by_filters`
# treat as "no filter"; a facet absent from this table is never relaxed.
FILTER_NEUTRALS: dict[str, Any] = {
    "genres": None,
    "decades": None,
    "artists": None,
    "exclude_genres": None,
    "exclude_artists": None,
    "exclude_decades": None,
    "exclude_compilations": False,
    "exclude_live": False,
    "live_only": False,
    "min_rating": 0,
    "max_rating": 0,
    "rated_only": False,
    "unrated_only": False,
    "min_plays": 0,
    "max_plays": 0,
    "unheard_only": False,
    "played_after": "",
    "played_before": "",
    "min_year": 0,
    "max_year": 0,
    "min_duration_ms": 0,
    "max_duration_ms": 0,
    # Known but never in a tier: switching whose ratings are read cannot widen a
    # pool (it re-reads the same rows), so the ladder must never "drop" it.
    "rating_user": "",
}
# The order facts are given up in: the library's own bans and floors come off
# before its genre/artist identity, which is what says *which music* the user
# meant and is therefore given up last (see ``identity_floor``). Every tier is
# cumulative on the previous one and is named, so a playlist can state which
# filters it actually honoured. The last rung is named because
# :func:`relax_filters_for_target` treats it differently: it is the only tier
# that stops saying *which music* the user meant, so it is not offered unless
# the pool is smaller than the playlist itself (see ``identity_floor``).
IDENTITY_TIER = "no genre, decade or artist whitelist"
RELAXATION_TIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("no bans", ("exclude_genres", "exclude_artists", "exclude_decades")),
    ("no rating or play floors",
     ("min_rating", "max_rating", "rated_only", "unrated_only",
      "min_plays", "max_plays", "unheard_only", "played_after", "played_before")),
    ("no year or duration windows",
     ("min_year", "max_year", "min_duration_ms", "max_duration_ms")),
    ("compilations and live recordings allowed", ("exclude_compilations", "exclude_live")),
    (IDENTITY_TIER, ("genres", "decades", "artists", "live_only")),
)


def relax_filters_for_target(
    facets: dict,
    target: int,
    count: int | None = None,
    identity_floor: int = 0,
) -> dict:
    """Drop filters, cheapest first, until the cache can supply *target* candidates.

    A pool smaller than the AI step needs makes a playlist thin, not broken (the
    curator prompt now asks for what it was offered), but it does mean the strictest
    reading of the request found almost nothing - a lone "best of all time" rating
    floor once cut 337 Holiday tracks to 5. This walks :data:`RELAXATION_TIERS`
    cumulatively, counting after each tier, and stops at the first tier that reaches
    *target*, so the filters stay as specific as the cache can honour.

    Args:
        facets: The filter kwargs the caller already has (the same dict it would
                splat into :func:`count_tracks_by_filters` / :func:`get_tracks_by_filters`).
                Unknown keys raise, so a renamed facet cannot be silently ignored.
        target: Candidates wanted (usually what :func:`generator.compute_track_budget`
                wants). 0 or a negative value means "no ladder": the filters are
                returned unchanged.
        count: The already-measured count of *facets*, when the caller has it, so
               the first count is not paid twice.
        identity_floor: How small the pool has to get before :data:`IDENTITY_TIER`
                may drop the genre/decade/artist whitelist. Usually the requested
                playlist size: a request whose filters already reach that many
                tracks is answered, it is just narrower than the AI budget, and
                replacing 80 Bossa Nova tracks with the whole library would answer
                a different question. 0 (the default) lets that tier run whenever
                the ladder runs at all.

    Returns:
        Dict with:
        - ``filters``: the relaxed facets, ready to splat into either function
        - ``matching``: candidates that pool has
        - ``asked``: candidates the filters as given had, before any tier ran
        - ``target``: what was asked for
        - ``tier``: the tier reached ("as asked" when nothing was relaxed)
        - ``relaxed``: every tier that actually changed something, in order
        - ``short_by``: how far the loosest reading still falls short (0 usually)
    """
    unknown = sorted(set(facets) - set(FILTER_NEUTRALS))
    if unknown:
        raise ValueError(f"unknown filter facets: {unknown}; relaxable ones are {sorted(FILTER_NEUTRALS)}")

    working = dict(facets)
    tried: list[str] = []
    if target is not None and target <= 0:
        measured = count if count is not None else count_tracks_by_filters(**working)
        return {
            "filters": working,
            "matching": measured,
            "asked": measured,
            "target": target,
            "tier": "as asked",
            "relaxed": tried,
            "short_by": 0,
        }

    measured = count if count is not None else count_tracks_by_filters(**working)
    if measured < 0:
        # -1 means the cache is empty, so the caller is looking at Plex and no tier
        # here changes anything: the live query applies its own filters and cannot
        # ladder against counts it never measured. Relaxing on top of that would
        # silently widen the request to the whole library.
        return {
            "filters": working,
            "matching": measured,
            "asked": measured,
            "target": target,
            "tier": "as asked",
            "relaxed": tried,
            "short_by": 0,
        }

    asked = measured
    if measured >= target:
        return {
            "filters": working,
            "matching": measured,
            "asked": asked,
            "target": target,
            "tier": "as asked",
            "relaxed": tried,
            "short_by": 0,
        }

    for label, names in RELAXATION_TIERS:
        # A floor of 0 means the caller has no playlist size to protect.
        if label == IDENTITY_TIER and 0 < identity_floor <= measured:
            break  # the pool already answers the request; keep it about this music
        before = dict(working)
        for name in names:
            # Only facets the caller actually set: a key that is missing already
            # counts as neutral, and adding it here would report a tier that
            # changed nothing as one that did.
            if name in working and working[name] != FILTER_NEUTRALS[name]:
                working[name] = FILTER_NEUTRALS[name]
        if working == before:
            continue  # that tier asks nothing of this request

        count_of_tier = count_tracks_by_filters(**working)
        tried.append(label)
        if count_of_tier > measured:
            measured = count_of_tier
        if count_of_tier >= target:
            return {
                "filters": working,
                "matching": count_of_tier,
                "asked": asked,
                "target": target,
                "tier": label,
                "relaxed": tried,
                "short_by": 0,
            }

    return {
        "filters": working,
        "matching": measured,
        "asked": asked,
        "target": target,
        "tier": tried[-1] if tried else "as asked",
        "relaxed": tried,
        "short_by": max(0, target - measured),
    }


def has_cached_tracks() -> bool:
    """Check if cache has any tracks.

    Returns:
        True if cache is populated
    """
    state = get_sync_state()
    return state["track_count"] > 0


def needs_resync() -> bool:
    """Check if a schema migration was applied that requires a re-sync.

    Returns:
        True if a migration was applied and sync hasn't completed yet.
        Safe for fresh DBs: _migration_applied is False when CREATE TABLE
        already includes all columns (ALTER TABLE no-ops).
    """
    return _migration_applied


def get_album_candidates(
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    exclude_live: bool = True,
) -> list[dict[str, Any]]:
    """Get album candidates aggregated from cached tracks.

    Groups tracks by parent_rating_key to produce album-level data.
    Supports genre/decade filtering.

    Args:
        genres: Optional genre filter (OR matching)
        decades: Optional decade filter (OR matching, e.g. "1990s")
        exclude_live: Exclude live recordings (default True)

    Returns:
        List of album dicts with parent_rating_key, album, album_artist,
        year, genres, decade, track_count, track_rating_keys.
    """
    conn = ensure_db_initialized()
    try:
        conditions = ["parent_rating_key IS NOT NULL", "parent_rating_key != ''"]
        if exclude_live:
            conditions.append("is_live = 0")
        params: list[Any] = []

        if decades:
            decade_conditions = []
            for decade in decades:
                try:
                    start_year = int(decade.rstrip("s"))
                except ValueError:
                    continue
                end_year = start_year + 9
                decade_conditions.append("(year >= ? AND year <= ?)")
                params.extend([start_year, end_year])
            if decade_conditions:
                conditions.append(f"({' OR '.join(decade_conditions)})")

        where_clause = " AND ".join(conditions)
        query = (
            f"SELECT rating_key, title, artist, album, year, genres, parent_rating_key "
            f"FROM tracks WHERE {where_clause} "
            f"ORDER BY parent_rating_key, rating_key"
        )

        rows = conn.execute(query, params).fetchall()

        # Aggregate tracks into albums
        albums: dict[str, dict[str, Any]] = {}
        for row in rows:
            prk = row["parent_rating_key"]
            track_genres = json.loads(row["genres"]) if row["genres"] else []

            if prk not in albums:
                # Derive decade from year
                year = row["year"]
                decade = ""
                if year:
                    decade_start = (year // 10) * 10
                    decade = f"{decade_start}s"

                albums[prk] = {
                    "parent_rating_key": prk,
                    "album": row["album"],
                    # artist column stores grandparentTitle (album artist), not track artist
                    "album_artist": row["artist"],
                    "year": year,
                    "genres": [],
                    "decade": decade,
                    "track_count": 0,
                    "track_rating_keys": [],
                    "_genre_set": set(),
                }

            album = albums[prk]
            album["track_count"] += 1
            album["track_rating_keys"].append(row["rating_key"])
            for g in track_genres:
                if g not in album["_genre_set"]:
                    album["_genre_set"].add(g)
                    album["genres"].append(g)

        # Apply genre filter in Python (genres stored as JSON)
        result = []
        genres_lower = [g.lower() for g in genres] if genres else None
        for album in albums.values():
            # Remove internal tracking set
            del album["_genre_set"]

            if genres_lower:
                album_genres_lower = [g.lower() for g in album["genres"]]
                if not any(g in album_genres_lower for g in genres_lower):
                    continue

            result.append(album)

        return result
    finally:
        conn.close()


def get_cached_genre_decade_stats() -> dict[str, list[dict[str, Any]]]:
    """Get genre and decade stats from the local cache.

    Returns genre/decade lists derived from cached tracks, avoiding a
    round-trip to the Plex server.

    Returns:
        Dict with 'genres' and 'decades' lists, each containing
        {'name': str, 'count': int} dicts sorted by name.
    """
    conn = ensure_db_initialized()
    try:
        rows = conn.execute("SELECT genres, year FROM tracks").fetchall()

        genre_counts: dict[str, int] = {}
        decade_counts: dict[str, int] = {}

        for row in rows:
            # Tally genres
            if row["genres"]:
                for g in json.loads(row["genres"]):
                    genre_counts[g] = genre_counts.get(g, 0) + 1

            # Tally decades
            year = row["year"]
            if year:
                decade_start = (year // 10) * 10
                decade_name = f"{decade_start}s"
                decade_counts[decade_name] = decade_counts.get(decade_name, 0) + 1

        genres = sorted(
            [{"name": name, "count": count} for name, count in genre_counts.items()],
            key=lambda x: x["name"],
        )
        decades = sorted(
            [{"name": name, "count": count} for name, count in decade_counts.items()],
            key=lambda x: x["name"],
        )

        return {"genres": genres, "decades": decades}
    finally:
        conn.close()


def get_cached_artists(
    limit: int = MAX_ARTIST_OPTIONS,
    exclude_live: bool = True,
) -> list[dict[str, Any]]:
    """Artist roster from the local cache, most-tracked first.

    Groups the cached tracks by name (case-insensitive) so a library that
    spells the same act two ways is offered once. Returns an empty list for an
    empty cache.

    Args:
        limit: Maximum artists to return (0 = no limit).
        exclude_live: Skip tracks flagged as live recordings.

    Returns:
        List of ``{'name': str, 'count': int}`` dicts sorted by count desc,
        then name.
    """
    conn = ensure_db_initialized()
    try:
        conditions = ["artist IS NOT NULL", "artist != ''"]
        if exclude_live:
            conditions.append("is_live = 0")
        where_clause = " AND ".join(conditions)
        query = (
            "SELECT artist, COUNT(*) FROM tracks "
            f"WHERE {where_clause} GROUP BY artist COLLATE NOCASE"
        )
        params: list[Any] = []
        if limit > 0:
            # Over-fetch so dropping a duplicate spelling cannot drop a real
            # contender from the top of the roster.
            query += " LIMIT ?"
            params.append(limit * 2)
        rows = conn.execute(query, params).fetchall()

        roster: list[dict[str, Any]] = [
            {"name": row["artist"], "count": row[1] or 0} for row in rows
        ]
        roster.sort(key=lambda item: (-item["count"], item["name"].lower()))
        return roster[:limit] if limit > 0 else roster
    finally:
        conn.close()


def get_cached_duration_stats(
    exclude_live: bool = True,
    bucket_ms: int = DURATION_BUCKET_MS,
) -> dict[str, Any]:
    """Track-length distribution from the local cache.

    Exact min/max plus percentiles interpolated over ``bucket_ms`` buckets, which
    is enough to phrase a duration suggestion without shipping every length.

    Args:
        exclude_live: Skip tracks flagged as live recordings.
        bucket_ms: Bucket width used for the percentile walk.

    Returns:
        Dict with ``total``, ``min_ms``, ``p25_ms``, ``p50_ms``, ``p75_ms``,
        ``max_ms`` (all 0 when the cache holds no durations).
    """
    conn = ensure_db_initialized()
    try:
        conditions = ["duration_ms > 0"]
        if exclude_live:
            conditions.append("is_live = 0")
        where_clause = " AND ".join(conditions)

        total, min_ms, max_ms = conn.execute(
            f"SELECT COUNT(*), MIN(duration_ms), MAX(duration_ms) "
            f"FROM tracks WHERE {where_clause}"
        ).fetchone()
        if not total or not max_ms:
            return {
                "total": 0,
                "min_ms": 0,
                "p25_ms": 0,
                "p50_ms": 0,
                "p75_ms": 0,
                "max_ms": 0,
            }

        bucket_expr = f"(duration_ms / {bucket_ms}) * {bucket_ms}"
        buckets = conn.execute(
            f"SELECT {bucket_expr}, COUNT(*) FROM tracks WHERE {where_clause} "
            f"GROUP BY {bucket_expr} ORDER BY CAST({bucket_expr} AS INTEGER)"
        ).fetchall()

        edges = [(int(row[0]), row[1] or 0) for row in buckets]
        return {
            "total": total,
            "min_ms": int(min_ms or 0),
            "p25_ms": _percentile_ms(edges, 0.25, bucket_ms),
            "p50_ms": _percentile_ms(edges, 0.50, bucket_ms),
            "p75_ms": _percentile_ms(edges, 0.75, bucket_ms),
            "max_ms": int(max_ms or 0),
        }
    finally:
        conn.close()


def _percentile_ms(
    edges: list[tuple[int, int]], fraction: float, bucket_ms: int
) -> int:
    """Interpolate the *fraction* percentile from ``(bucket_start, count)`` pairs."""
    if not edges:
        return 0
    total = sum(count for _, count in edges)
    if total <= 0:
        return 0
    target = max(1, round(total * fraction))
    seen = 0
    for (start, count), (nxt_start, _) in zip(
        edges, [*edges[1:], (edges[-1][0] + bucket_ms, 0)]
    ):
        if seen + count >= target:
            width = max(nxt_start - start, 1)
            offset = (target - seen) / max(count, 1)
            return int(start + width * min(max(offset - 0.5, 0.0), 1.0))
        seen += count
    return int(edges[-1][0])


def get_cached_play_stats(exclude_live: bool = True) -> dict[str, Any]:
    """Summary of Plex play history and release years, for prompt facets.

    Plex's ``viewCount`` is already in the cache (``view_count``), so the local
    path can answer "how familiar is this music?" exactly: how many tracks were
    never played at all, how the play counts spread, and the last-played dates
    that make a "heard recently" window checkable. The release-year span
    comes along because a prompt can only ask for a plausible window once the
    caller knows what the library actually covers.

    The rating and live counts ride along because they answer the same question
    — "is there anything left if I ask for this?" — and each needs its own scan:
    ``rated`` is how many tracks carry a user rating (so a rating floor or
    ceiling or "unrated only" is worth suggesting at all), and ``live`` counts
    the live recordings regardless of *exclude_live*, which is what tells the
    caller a "only live takes" request has a pool.

    Args:
        exclude_live: Skip tracks flagged as live recordings.

    Returns:
        Dict with ``total``, ``unheard`` (tracks with a zero play count),
        ``played``, ``plays`` (five-number summary ``[min, p25, median, p75,
        max]`` of ``view_count``), ``years`` (``[oldest, newest]`` release
        years, empty when no track is dated), ``last_played`` (``[oldest, newest]``
        ISO dates the library was last heard on, empty when nothing is played),
        ``rated`` (tracks carrying a user rating) and ``live`` (live recordings
        in the whole library, counted even when *exclude_live* dropped them).
    """
    conn = ensure_db_initialized()
    try:
        where_clause = "is_live = 0" if exclude_live else "1=1"
        year_clause = "year IS NOT NULL AND " + where_clause

        total, unheard, played = conn.execute(
            f"SELECT COUNT(*), "
            f"SUM(CASE WHEN COALESCE(view_count, 0) = 0 THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN COALESCE(view_count, 0) > 0 THEN 1 ELSE 0 END) "
            f"FROM tracks WHERE {where_clause}"
        ).fetchone()
        if not total:
            return {
                "total": 0,
                "unheard": 0,
                "played": 0,
                "plays": [],
                "years": [],
                "last_played": [],
                "rated": 0,
                "live": 0,
            }

        oldest, newest = conn.execute(
            f"SELECT MIN(year), MAX(year) FROM tracks WHERE {year_clause}"
        ).fetchone()

        # Last-heard dates are ISO timestamps of a fixed width, so MIN/MAX are
        # also the chronologically oldest and newest; they are trimmed to dates
        # because that is the granularity the played_after/played_before filter
        # compares on.
        first_heard, last_heard = conn.execute(
            "SELECT MIN(last_viewed_at), MAX(last_viewed_at) "
            "FROM tracks WHERE last_viewed_at IS NOT NULL"
        ).fetchone()

        rated = conn.execute(
            f"SELECT COUNT(user_rating) FROM tracks WHERE {where_clause}"
        ).fetchone()[0]
        live = conn.execute(
            "SELECT COUNT(*) FROM tracks WHERE is_live = 1"
        ).fetchone()[0]

        # Play counts are a tiny integer alphabet (most tracks sit at 0), so the
        # histogram is small enough to walk exactly instead of bucketing.
        histogram = conn.execute(
            f"SELECT COALESCE(view_count, 0), COUNT(*) FROM tracks "
            f"WHERE {where_clause} GROUP BY COALESCE(view_count, 0) "
            f"ORDER BY COALESCE(view_count, 0)"
        ).fetchall()
        edges = [(int(row[0]), row[1] or 0) for row in histogram]
        return {
            "total": total,
            "unheard": unheard or 0,
            "played": played or 0,
            "plays": _play_summary(edges),
            "years": [int(oldest), int(newest)] if oldest is not None else [],
            "last_played": [
                str(first_heard)[:10],
                str(last_heard)[:10],
            ]
            if first_heard
            else [],
            "rated": rated or 0,
            "live": live or 0,
        }
    finally:
        conn.close()


def _play_summary(edges: list[tuple[int, int]]) -> list[int]:
    """Five-number summary ``[min, p25, median, p75, max]`` of a play histogram."""
    if not edges:
        return []
    total = sum(count for _, count in edges)
    if total <= 0:
        return []

    def at(fraction: float) -> int:
        target = max(1, round(total * fraction))
        seen = 0
        for value, count in edges:
            if seen + count >= target:
                return value
            seen += count
        return edges[-1][0]

    return [
        edges[0][0],
        at(0.25),
        at(0.50),
        at(0.75),
        edges[-1][0],
    ]


def get_album_familiarity(
    parent_rating_keys: list[str] | None = None,
) -> dict[str, dict]:
    """Get familiarity data for albums aggregated from cached track play history.

    Classifies each album as:
    - "unplayed": 0 total plays across all tracks
    - "well-loved": avg plays per track >= 3
    - "light": some plays but avg < 3

    Args:
        parent_rating_keys: Optional list of album keys to query.
            If None, returns all albums.

    Returns:
        Dict mapping parent_rating_key -> {"level": str, "last_viewed_at": str|None}
    """
    conn = ensure_db_initialized()
    try:
        query = (
            "SELECT parent_rating_key, "
            "SUM(view_count) AS total_plays, "
            "AVG(view_count) AS avg_plays, "
            "MAX(last_viewed_at) AS last_viewed "
            "FROM tracks "
            "WHERE parent_rating_key IS NOT NULL AND parent_rating_key != '' "
        )
        params: list[str] = []

        if parent_rating_keys is not None:
            placeholders = ",".join("?" for _ in parent_rating_keys)
            query += f"AND parent_rating_key IN ({placeholders}) "
            params.extend(parent_rating_keys)

        query += "GROUP BY parent_rating_key"

        rows = conn.execute(query, params).fetchall()

        result: dict[str, dict] = {}
        for row in rows:
            total_plays = row["total_plays"] or 0
            avg_plays = row["avg_plays"] or 0

            if total_plays == 0:
                level = "unplayed"
            elif avg_plays >= 3:
                level = "well-loved"
            else:
                level = "light"

            result[row["parent_rating_key"]] = {
                "level": level,
                "last_viewed_at": row["last_viewed"],
            }

        return result
    finally:
        conn.close()


# =============================================================================
# Results Persistence
# =============================================================================


def save_result(
    result_type: str,
    title: str,
    prompt: str,
    snapshot: dict,
    track_count: int,
    artist: str | None = None,
    art_rating_key: str | None = None,
    subtitle: str | None = None,
) -> str:
    """Save a generated result and return its unique ID.

    Args:
        result_type: "prompt_playlist", "seed_playlist", or "album_recommendation"
        title: Display title for the result
        prompt: Original user prompt
        snapshot: Full serialized response (GenerateResponse or RecommendGenerateResponse)
        track_count: Number of tracks in the result
        artist: Primary artist (for album recs)
        art_rating_key: Rating key for thumbnail art
        subtitle: Pre-computed subtitle for history feed cards

    Returns:
        16-char hex ID for the saved result
    """
    conn = ensure_db_initialized()
    try:
        # Generate collision-resistant ID with INSERT OR IGNORE to handle
        # concurrent inserts that race past the existence check.
        for _ in range(10):
            result_id = secrets.token_hex(8)
            cursor = conn.execute(
                """INSERT OR IGNORE INTO results (id, type, title, prompt, snapshot, track_count, artist, art_rating_key, subtitle)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result_id, result_type, title, prompt, json.dumps(snapshot), track_count, artist, art_rating_key, subtitle),
            )
            if cursor.rowcount > 0:
                break
        else:
            raise RuntimeError("Failed to generate unique result ID after 10 attempts")
        conn.commit()
        logger.info("Saved result %s (type=%s, tracks=%d)", result_id, result_type, track_count)
        return result_id
    finally:
        conn.close()


def link_result_playlist(
    result_id: str,
    playlist_id: str,
    playlist_title: str = "",
    playlist_owner: str = "",
    playlist_url: str = "",
) -> bool:
    """Point a saved result at the Plex playlist it was written to.

    Called once a history row has been saved to Plex, so the history feed can
    show which Plex playlist a generated playlist became. Only the primary
    playlist is recorded; ``copy_to`` accounts get their own separate playlists
    and are reported in the save response instead.

    Args:
        result_id: Result id returned by :func:`save_result`.
        playlist_id: Plex rating key of the created playlist.
        playlist_title: Title the playlist has in Plex.
        playlist_owner: Display name of the account that owns it.
        playlist_url: Web URL for the playlist, when the server publishes one.

    Returns:
        True when a history row was updated, False when ``result_id`` is unknown.
    """
    if not result_id or not playlist_id:
        return False
    conn = ensure_db_initialized()
    try:
        cursor = conn.execute(
            "UPDATE results SET playlist_id = ?, playlist_title = ?, playlist_owner = ?, playlist_url = ? "
            "WHERE id = ?",
            (
                str(playlist_id),
                playlist_title or "",
                playlist_owner or "",
                playlist_url or "",
                result_id,
            ),
        )
        conn.commit()
        linked = cursor.rowcount > 0
        if linked:
            logger.info("Linked result %s to Plex playlist %s", result_id, playlist_id)
        else:
            logger.warning("Result %s not found while linking playlist %s", result_id, playlist_id)
        return linked
    finally:
        conn.close()


def get_result(result_id: str) -> dict[str, Any] | None:
    """Fetch a single result by ID, including its snapshot.

    Returns:
        Dict with all columns (snapshot parsed from JSON), or None if not found.
    """
    conn = ensure_db_initialized()
    try:
        row = conn.execute(
            "SELECT id, type, title, prompt, snapshot, track_count, artist, art_rating_key, subtitle, "
            "playlist_id, playlist_title, playlist_owner, playlist_url, created_at FROM results WHERE id = ?",
            (result_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "type": row["type"],
            "title": row["title"],
            "prompt": row["prompt"],
            "snapshot": json.loads(row["snapshot"]),
            "track_count": row["track_count"],
            "artist": row["artist"],
            "art_rating_key": row["art_rating_key"],
            "subtitle": row["subtitle"],
            "playlist_id": row["playlist_id"],
            "playlist_title": row["playlist_title"],
            "playlist_owner": row["playlist_owner"],
            "playlist_url": row["playlist_url"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def list_results(
    result_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """List results ordered by created_at DESC, without snapshots.

    Args:
        result_type: Optional type filter (can be comma-separated for multiple)
        limit: Max results to return
        offset: Pagination offset

    Returns:
        Tuple of (list of result dicts without snapshot, total count)
    """
    conn = ensure_db_initialized()
    try:
        where_clause = ""
        params: list[Any] = []

        if result_type:
            types = [t.strip() for t in result_type.split(",") if t.strip()]
            placeholders = ",".join("?" for _ in types)
            where_clause = f"WHERE type IN ({placeholders})"
            params = types

        # Get total count
        total = conn.execute(
            f"SELECT COUNT(*) FROM results {where_clause}", params
        ).fetchone()[0]

        # Get page
        rows = conn.execute(
            f"""SELECT id, type, title, prompt, track_count, artist, art_rating_key, subtitle,
                playlist_id, playlist_title, playlist_owner, playlist_url, created_at
                FROM results {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

        results = [
            {
                "id": row["id"],
                "type": row["type"],
                "title": row["title"],
                "prompt": row["prompt"],
                "track_count": row["track_count"],
                "artist": row["artist"],
                "art_rating_key": row["art_rating_key"],
                "subtitle": row["subtitle"],
                "playlist_id": row["playlist_id"],
                "playlist_title": row["playlist_title"],
                "playlist_owner": row["playlist_owner"],
                "playlist_url": row["playlist_url"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return results, total
    finally:
        conn.close()


def delete_result(result_id: str) -> bool:
    """Delete a result by ID.

    Returns:
        True if a row was deleted, False if not found.
    """
    conn = ensure_db_initialized()
    try:
        cursor = conn.execute("DELETE FROM results WHERE id = ?", (result_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted result %s", result_id)
        return deleted
    finally:
        conn.close()


# =============================================================================
# Spotify Import Wish List
# =============================================================================


def record_missing_songs(
    playlist_ref: str,
    playlist_name: str,
    rows: list[dict],
) -> int:
    """Replace one imported playlist's wish list with *rows*; return rows written.

    Re-importing replaces rather than stacks, so a wish list never grows across
    re-runs of the same playlist, while other playlists' wishes stay intact.
    Each row needs at least ``position``/``title``; ``position`` is the 1-based
    slot in the Spotify playlist, which keys the table together with the ref.

    Args:
        playlist_ref: The playlist's canonical id (what imports key on).
        playlist_name: Display name of the playlist, for reading the wish list.
        rows: Missed Spotify rows as dicts (``position``, ``title``, ``album``,
              ``artist``, ``duration_ms``, ``spotify_track_id``, ``spotify_url``).

    Returns:
        Number of wish-list rows now stored for this playlist.
    """
    conn = ensure_db_initialized()
    try:
        conn.execute("DELETE FROM missing_songs WHERE playlist_ref = ?", (playlist_ref,))
        written = 0
        for row in rows:
            try:
                position = int(row.get("position") or 0)
            except (TypeError, ValueError):
                position = 0
            if position <= 0:
                continue
            cursor = conn.execute(
                "INSERT OR REPLACE INTO missing_songs ("
                "playlist_ref, position, playlist_name, title, album, artist, "
                "duration_ms, spotify_track_id, spotify_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    playlist_ref,
                    position,
                    playlist_name or "",
                    str(row.get("title") or ""),
                    str(row.get("album") or ""),
                    str(row.get("artist") or ""),
                    int(row.get("duration_ms") or 0),
                    str(row.get("spotify_track_id") or ""),
                    str(row.get("spotify_url") or ""),
                ),
            )
            written += cursor.rowcount
        conn.commit()
        logger.info("Recorded %d wish-list rows for playlist %s", written, playlist_ref)
        return written
    finally:
        conn.close()


def list_missing_songs(limit: int = 200, offset: int = 0) -> dict[str, Any]:
    """Stored wish list: songs imported playlists asked for but the library lacks.

    Newest playlist import first, playlist slots in order; the per-playlist
    summary counts every row regardless of the page slice.

    Returns:
        Dict with ``missing`` (page of row dicts, ``recorded_at`` as text),
        ``playlists`` (one ``{playlist_ref, playlist_name, count}`` per
        playlist in the table, newest first) and ``total`` (row count).
    """
    conn = ensure_db_initialized()
    try:
        total = conn.execute("SELECT COUNT(*) FROM missing_songs").fetchone()[0]
        rows = conn.execute(
            "SELECT playlist_ref, position, playlist_name, title, album, artist, "
            "duration_ms, spotify_track_id, spotify_url, recorded_at "
            "FROM missing_songs ORDER BY recorded_at DESC, playlist_ref ASC, position ASC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {
            "missing": [
                {
                    "position": row["position"],
                    "title": row["title"],
                    "album": row["album"],
                    "artist": row["artist"],
                    "duration_ms": row["duration_ms"],
                    "spotify_track_id": row["spotify_track_id"],
                    "spotify_url": row["spotify_url"],
                    "playlist_ref": row["playlist_ref"],
                    "playlist_name": row["playlist_name"],
                    "recorded_at": str(row["recorded_at"] or ""),
                }
                for row in rows
            ],
            "playlists": [
                {"playlist_ref": row["playlist_ref"], "playlist_name": row["playlist_name"], "count": row["count"]}
                for row in conn.execute(
                    "SELECT playlist_ref, MAX(playlist_name) AS playlist_name, COUNT(1) AS count "
                    "FROM missing_songs GROUP BY playlist_ref ORDER BY MAX(recorded_at) DESC"
                ).fetchall()
            ],
            "total": int(total or 0),
        }
    except sqlite3.Error as e:
        logger.warning("Failed to read the wish list: %s", e)
        return {"missing": [], "playlists": [], "total": 0}
    finally:
        conn.close()
