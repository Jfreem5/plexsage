"""SQLite-backed track metadata store for fast, local library queries.

Mirrors the Plex music library into a local SQLite file so that filtering,
searching, and pre-ranking can be served without repeated Plex HTTP calls.

Stores user ratings, play counts, last-played timestamps, genres, live
detection results, and (optionally) per-track embeddings.

Uses only the Python standard library (sqlite3) — no external dependencies.
"""

import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from backend.models import Track

logger = logging.getLogger(__name__)

# Default DB location: alongside the backend package
_DEFAULT_DB_PATH = Path(__file__).parent / "plexsage.db"

# How long a synced library is considered fresh (30 minutes).
SYNC_INTERVAL_SECONDS = 30 * 60

# SQLite caps the number of bound parameters; chunk large IN () clauses.
_SQL_CHUNK = 500


def _chunks(items: Sequence[Any], size: int = _SQL_CHUNK) -> Iterable[Sequence[Any]]:
    """Yield *items* in fixed-size slices."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _decade_to_range(decade: str) -> tuple[int | None, int | None]:
    """Convert a decade label (e.g. "1990s", "90s") to an inclusive year range.

    Returns ``(None, None)`` when the label can't be parsed.
    """
    match = re.match(r"(\d{2,4})", decade or "")
    if not match:
        return None, None
    value = int(match.group(1))
    if value < 100:
        # Two-digit decade: 30-99 -> 1900s, 0-29 -> 2000s
        value = 1900 + value if value >= 30 else 2000 + value
    start = (value // 10) * 10
    return start, start + 9


def _build_fts_match(query: str) -> str:
    """Turn a free-text query into a safe FTS5 prefix-match expression."""
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    return " ".join(f'"{token}"*' for token in tokens)


class TrackDB:
    """SQLite-backed store for track metadata used by library queries.

    Schema:
        tracks (
            rating_key   TEXT PRIMARY KEY,
            title        TEXT,
            artist       TEXT,
            album        TEXT,
            year         INTEGER,
            duration_ms  INTEGER,
            is_live      INTEGER,   -- pre-computed at sync time
            user_rating  INTEGER,   -- 0-10, 0 = unrated
            play_count   INTEGER,   -- total plays
            last_played  TEXT,      -- ISO timestamp or NULL
            embedding    BLOB,      -- optional similarity vector
            updated_at   TEXT       -- last sync timestamp
        )
        track_genres (
            rating_key   TEXT,
            genre        TEXT,
            PRIMARY KEY (rating_key, genre)
        )
        track_dimensions (
            rating_key   TEXT PRIMARY KEY,
            dimensions   TEXT,      -- JSON array of Dimension objects
            model        TEXT,      -- analysis model that produced them
            fingerprint  TEXT,      -- hash of the source track metadata
            analyzed_at  TEXT
        )
        meta (key TEXT PRIMARY KEY, value TEXT)
        tracks_fts   -- FTS5 index over title/artist/album (when available)

    Indexes:
        idx_tracks_artist   ON tracks(artist)
        idx_tracks_year     ON tracks(year)
        idx_tracks_rating   ON tracks(user_rating)
        idx_tracks_lastplay ON tracks(last_played)
        idx_genres_genre    ON track_genres(genre)
    """

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the track database.

        Args:
            db_path: Path to the SQLite file. Defaults to plexsage.db
                     alongside this module.
        """
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        # Writes are serialized with this lock; each thread owns its own
        # connection (see _get_conn). Re-entrant because write methods call
        # one another (e.g. sync_from_plex -> upsert_tracks).
        self._lock = threading.RLock()
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._generation = 0
        self._fts_enabled = False
        self._init_db()

    def _init_db(self) -> None:
        """Create tables, run migrations, and build the FTS index."""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS tracks (
                    rating_key   TEXT PRIMARY KEY,
                    title        TEXT NOT NULL,
                    artist       TEXT NOT NULL,
                    album        TEXT NOT NULL DEFAULT '',
                    year         INTEGER,
                    duration_ms  INTEGER NOT NULL DEFAULT 0,
                    is_live      INTEGER NOT NULL DEFAULT 0,
                    user_rating  INTEGER NOT NULL DEFAULT 0,
                    play_count   INTEGER NOT NULL DEFAULT 0,
                    last_played  TEXT,
                    embedding    BLOB,
                    updated_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS track_genres (
                    rating_key   TEXT NOT NULL,
                    genre        TEXT NOT NULL,
                    PRIMARY KEY (rating_key, genre)
                );

                CREATE TABLE IF NOT EXISTS track_dimensions (
                    rating_key   TEXT PRIMARY KEY,
                    dimensions   TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    fingerprint  TEXT,
                    analyzed_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tracks_artist   ON tracks(artist);
                CREATE INDEX IF NOT EXISTS idx_tracks_year     ON tracks(year);
                CREATE INDEX IF NOT EXISTS idx_tracks_rating   ON tracks(user_rating);
                CREATE INDEX IF NOT EXISTS idx_tracks_lastplay ON tracks(last_played);
                CREATE INDEX IF NOT EXISTS idx_genres_genre    ON track_genres(genre);
            """)

            self._migrate()
            # Covering index for the filtered track listing: the sort
            # expressions lead (so ORDER BY ... COLLATE NOCASE is served in
            # index order instead of a temp b-tree) and every selected and
            # filtered column trails (so the scan needs no table lookups).
            # Created after _migrate() because it references is_live.
            self._conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_tracks_nocase_sort
                   ON tracks(artist COLLATE NOCASE, album COLLATE NOCASE, title COLLATE NOCASE,
                             rating_key, year, duration_ms, is_live, user_rating)"""
            )
            self._init_fts()
            self._conn.commit()
            logger.info("TrackDB initialized at %s", self._db_path)

    def _migrate(self) -> None:
        """Add columns introduced after the initial schema."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(tracks)")}
        if "embedding" not in columns:
            self._conn.execute("ALTER TABLE tracks ADD COLUMN embedding BLOB")
            logger.info("TrackDB migration: added tracks.embedding column")
        if "is_live" not in columns:
            self._conn.execute(
                "ALTER TABLE tracks ADD COLUMN is_live INTEGER NOT NULL DEFAULT 0"
            )
            logger.info("TrackDB migration: added tracks.is_live column")

    def _init_fts(self) -> None:
        """Create the FTS5 index if SQLite supports it, and backfill if empty.

        Falls back to ``LIKE`` queries when FTS5 is unavailable.
        """
        try:
            self._conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS tracks_fts USING fts5(
                       rating_key UNINDEXED,
                       title,
                       artist,
                       album,
                       tokenize='unicode61'
                   )"""
            )
            self._fts_enabled = True
        except sqlite3.OperationalError as exc:
            self._fts_enabled = False
            logger.warning("FTS5 unavailable (%s); falling back to LIKE search", exc)
            return

        has_rows = self._conn.execute("SELECT 1 FROM tracks_fts LIMIT 1").fetchone()
        if has_rows is None:
            track_count = self._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            if track_count:
                logger.info("TrackDB: backfilling FTS index for %d tracks", track_count)
                self.rebuild_fts()

    @property
    def _conn(self) -> sqlite3.Connection:
        """The calling thread's connection (see :meth:`_get_conn`)."""
        return self._get_conn()

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection to the database file."""
        conn = sqlite3.connect(
            str(self._db_path),
            # Statements run on the owning thread; this only allows close() to
            # tidy up connections opened by worker threads.
            check_same_thread=False,
            timeout=30,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Return this thread's connection, opening one on first use.

        A sqlite3 connection is not safe for concurrent multi-statement work,
        so each thread owns its own instead of sharing one across the
        ``asyncio.to_thread`` workers (WAL lets them progress independently).
        Connections are re-created after :meth:`close`.
        """
        local = self._local
        if getattr(local, "generation", None) != self._generation:
            conn = self._connect()
            local.conn = conn
            local.generation = self._generation
            with self._lock:
                self._connections.append(conn)
        return local.conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Run a write transaction: serialized across threads, rolled back on error.

        The lock keeps two writers from interleaving (or committing) each
        other's half-applied batch on their separate connections.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def _upsert_row(
        self,
        track: Track,
        is_live: bool,
        user_rating: int,
        play_count: int,
        last_played: str | None,
        now: str,
        maintain_fts: bool = True,
    ) -> None:
        """Upsert one track row and its genres/FTS entry (no commit).

        Uses ``ON CONFLICT DO UPDATE`` so that columns not written here
        (notably ``embedding``) survive re-syncs.

        Args:
            maintain_fts: When False, skip the per-row FTS rewrite. Bulk
                callers set this so they can rewrite FTS for the whole batch
                at once via :meth:`_replace_fts_bulk` (see that method for why
                the per-row path is quadratic).
        """
        self._conn.execute(
            """INSERT INTO tracks
               (rating_key, title, artist, album, year, duration_ms,
                is_live, user_rating, play_count, last_played, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(rating_key) DO UPDATE SET
                   title=excluded.title,
                   artist=excluded.artist,
                   album=excluded.album,
                   year=excluded.year,
                   duration_ms=excluded.duration_ms,
                   is_live=excluded.is_live,
                   user_rating=excluded.user_rating,
                   play_count=excluded.play_count,
                   last_played=excluded.last_played,
                   updated_at=excluded.updated_at""",
            (
                track.rating_key,
                track.title,
                track.artist,
                track.album,
                track.year,
                track.duration_ms,
                1 if is_live else 0,
                user_rating,
                play_count,
                last_played,
                now,
            ),
        )

        self._conn.execute("DELETE FROM track_genres WHERE rating_key = ?", (track.rating_key,))
        for genre in track.genres:
            self._conn.execute(
                "INSERT OR IGNORE INTO track_genres (rating_key, genre) VALUES (?, ?)",
                (track.rating_key, genre),
            )

        if maintain_fts and self._fts_enabled:
            self._conn.execute("DELETE FROM tracks_fts WHERE rating_key = ?", (track.rating_key,))
            self._conn.execute(
                "INSERT INTO tracks_fts (rating_key, title, artist, album) VALUES (?, ?, ?, ?)",
                (track.rating_key, track.title, track.artist, track.album),
            )

    def _replace_fts_bulk(self, tracks: list[Track]) -> None:
        """Rewrite the FTS rows for *tracks* in bulk (no commit).

        ``rating_key`` is declared ``UNINDEXED`` in the FTS5 table, so a
        per-row ``DELETE ... WHERE rating_key = ?`` degrades to a full scan of
        the index — O(n) per track, i.e. O(n²) for a whole-library sync (a
        10k-track sync spent ~4s here versus ~30ms for one bulk rewrite).
        Deleting the batch in a single chunked statement and re-inserting with
        ``executemany`` keeps a full sync linear.
        """
        if not self._fts_enabled or not tracks:
            return

        for chunk in _chunks([t.rating_key for t in tracks]):
            placeholders = ",".join("?" * len(chunk))
            self._conn.execute(
                f"DELETE FROM tracks_fts WHERE rating_key IN ({placeholders})", chunk
            )

        self._conn.executemany(
            "INSERT INTO tracks_fts (rating_key, title, artist, album) VALUES (?, ?, ?, ?)",
            [(t.rating_key, t.title, t.artist, t.album) for t in tracks],
        )

    def upsert_track(self, track: Track, is_live: bool = False,
                     user_rating: int = 0, play_count: int = 0,
                     last_played: str | None = None) -> None:
        """Insert or update a single track's metadata.

        Args:
            track: The Track model (provides rating_key, title, artist, etc.)
            is_live: Whether this is a live recording
            user_rating: User's rating 0-10 (0 = unrated)
            play_count: Total number of plays
            last_played: ISO timestamp of last play, or None
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._write():
            self._upsert_row(track, is_live, user_rating, play_count, last_played, now)

    def upsert_tracks(self, tracks: list[Track],
                      is_live_flags: list[bool] | None = None,
                      user_ratings: list[int] | None = None,
                      play_counts: list[int] | None = None,
                      last_played: list[str | None] | None = None) -> int:
        """Bulk upsert multiple tracks.

        Args:
            tracks: List of Track models
            is_live_flags: Optional per-track live flags (defaults to False)
            user_ratings: Optional per-track ratings (defaults to 0)
            play_counts: Optional per-track play counts (defaults to 0)
            last_played: Optional per-track last-played timestamps

        Returns:
            Number of tracks upserted
        """
        if not tracks:
            return 0

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with self._write():
            for i, track in enumerate(tracks):
                is_live = is_live_flags[i] if is_live_flags and i < len(is_live_flags) else False
                rating = user_ratings[i] if user_ratings and i < len(user_ratings) else 0
                plays = play_counts[i] if play_counts and i < len(play_counts) else 0
                last_play = last_played[i] if last_played and i < len(last_played) else None
                self._upsert_row(track, is_live, rating, plays, last_play, now, maintain_fts=False)

            self._replace_fts_bulk(tracks)

        logger.info("Upserted %d tracks into TrackDB", len(tracks))
        return len(tracks)

    def sync_from_plex(self, tracks: list[Track],
                       is_live_flags: list[bool] | None = None) -> int:
        """Sync a batch of tracks from Plex into the database.

        Upserts all provided tracks, preserving any existing
        user_rating / play_count / last_played values that aren't
        overridden by the new data.

        Args:
            tracks: List of Track models from Plex
            is_live_flags: Optional per-track live detection results

        Returns:
            Number of tracks synced
        """
        if not tracks:
            return 0

        existing = self.get_metadata_batch([t.rating_key for t in tracks])

        user_ratings = []
        play_counts = []
        last_played = []

        for t in tracks:
            meta = existing.get(t.rating_key)
            user_ratings.append(meta["user_rating"] if meta else 0)
            play_counts.append(meta["play_count"] if meta else 0)
            last_played.append(meta["last_played"] if meta else None)

        return self.upsert_tracks(
            tracks,
            is_live_flags=is_live_flags,
            user_ratings=user_ratings,
            play_counts=play_counts,
            last_played=last_played,
        )

    def prune_except(self, rating_keys: Iterable[str]) -> int:
        """Delete tracks not present in *rating_keys* (library removals).

        Args:
            rating_keys: The complete set of rating keys currently in Plex.

        Returns:
            Number of tracks removed.
        """
        keep = {str(k) for k in rating_keys}
        if not keep:
            return 0

        with self._write() as conn:
            existing = [row[0] for row in conn.execute("SELECT rating_key FROM tracks")]
            removed = [key for key in existing if key not in keep]
            if not removed:
                return 0

            for chunk in _chunks(removed):
                placeholders = ",".join("?" * len(chunk))
                conn.execute(f"DELETE FROM track_genres WHERE rating_key IN ({placeholders})", chunk)
                conn.execute(f"DELETE FROM track_dimensions WHERE rating_key IN ({placeholders})", chunk)
                if self._fts_enabled:
                    conn.execute(f"DELETE FROM tracks_fts WHERE rating_key IN ({placeholders})", chunk)
                conn.execute(f"DELETE FROM tracks WHERE rating_key IN ({placeholders})", chunk)

        logger.info("Pruned %d tracks removed from Plex", len(removed))
        return len(removed)

    def set_embedding(self, rating_key: str, embedding: bytes | None) -> None:
        """Store (or clear) the similarity vector for a track."""
        with self._write() as conn:
            conn.execute(
                "UPDATE tracks SET embedding = ? WHERE rating_key = ?",
                (embedding, rating_key),
            )

    def get_embeddings(self, rating_keys: list[str] | None = None) -> dict[str, bytes]:
        """Return stored embeddings as ``{rating_key: blob}``."""
        conn = self._get_conn()
        if rating_keys:
            result: dict[str, bytes] = {}
            for chunk in _chunks(list(rating_keys)):
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"""SELECT rating_key, embedding FROM tracks
                        WHERE embedding IS NOT NULL
                          AND rating_key IN ({placeholders})""",
                    chunk,
                ).fetchall()
                result.update({rk: blob for rk, blob in rows})
            return result

        rows = conn.execute(
            "SELECT rating_key, embedding FROM tracks WHERE embedding IS NOT NULL"
        ).fetchall()
        return {rk: blob for rk, blob in rows}

    # ------------------------------------------------------------------
    # Cached track analysis (seed-track dimensions)
    # ------------------------------------------------------------------

    def save_dimensions(
        self,
        rating_key: str,
        dimensions: list[dict[str, Any]],
        model: str,
        fingerprint: str | None = None,
    ) -> None:
        """Cache the LLM-generated dimensions for a track.

        Args:
            rating_key: Track the dimensions belong to.
            dimensions: The Dimension objects as JSON-serializable dicts.
            model: Analysis model that produced them.
            fingerprint: Hash of the source metadata used for invalidation.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._write() as conn:
            conn.execute(
                """INSERT INTO track_dimensions
                   (rating_key, dimensions, model, fingerprint, analyzed_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(rating_key) DO UPDATE SET
                       dimensions=excluded.dimensions,
                       model=excluded.model,
                       fingerprint=excluded.fingerprint,
                       analyzed_at=excluded.analyzed_at""",
                (rating_key, json.dumps(dimensions), model, fingerprint, now),
            )

    def get_dimensions(
        self,
        rating_key: str,
        model: str | None = None,
        fingerprint: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Return cached dimensions, or None on a miss/stale entry.

        A cached entry is treated as stale when *model* differs from the stored
        model, or when *fingerprint* differs from the stored metadata hash.
        Passing None for either skips that check.

        Args:
            rating_key: Track to look up.
            model: Current analysis model; a mismatch invalidates the cache.
            fingerprint: Current metadata hash; a mismatch invalidates the cache.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT dimensions, model, fingerprint FROM track_dimensions WHERE rating_key = ?",
            (rating_key,),
        ).fetchone()
        if row is None:
            return None

        dimensions_json, stored_model, stored_fingerprint = row
        if model is not None and stored_model != model:
            return None
        if fingerprint is not None and stored_fingerprint != fingerprint:
            return None

        try:
            result = json.loads(dimensions_json)
        except (TypeError, ValueError):
            logger.warning("Corrupt cached dimensions for %s; ignoring", rating_key)
            return None
        return result if isinstance(result, list) else None

    def delete_dimensions(self, rating_key: str) -> None:
        """Drop the cached analysis for a single track."""
        with self._write() as conn:
            conn.execute("DELETE FROM track_dimensions WHERE rating_key = ?", (rating_key,))

    def clear_dimensions(self) -> None:
        """Drop every cached track analysis."""
        with self._write() as conn:
            conn.execute("DELETE FROM track_dimensions")

    def get_dimension_count(self) -> int:
        """Number of tracks with cached dimensions."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM track_dimensions").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Sync metadata
    # ------------------------------------------------------------------

    def set_last_sync(self, timestamp: float | None = None) -> None:
        """Record the time of the most recent successful sync."""
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_sync', ?)",
                (str(timestamp if timestamp is not None else time.time()),),
            )

    def get_last_sync(self) -> float | None:
        """Return the epoch time of the last sync, or None if never synced."""
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM meta WHERE key = 'last_sync'").fetchone()
        if not row or row[0] is None:
            return None
        try:
            return float(row[0])
        except (TypeError, ValueError):
            return None

    def needs_sync(self, max_age_seconds: int = SYNC_INTERVAL_SECONDS) -> bool:
        """Whether the DB is empty or older than *max_age_seconds*."""
        if self.is_empty():
            return True
        last_sync = self.get_last_sync()
        if last_sync is None:
            return True
        return (time.time() - last_sync) > max_age_seconds

    def is_empty(self) -> bool:
        """Whether the tracks table has no rows."""
        conn = self._get_conn()
        return conn.execute("SELECT 1 FROM tracks LIMIT 1").fetchone() is None

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def _rows_to_tracks(self, rows: list[tuple]) -> list[Track]:
        """Convert ``(rating_key, title, artist, album, year, duration_ms, is_live)`` rows.

        Batch-loads genres for all rows in a single query. ``is_live`` is the
        flag pre-computed during sync, so no regex runs here.
        """
        if not rows:
            return []

        conn = self._get_conn()
        keys = [row[0] for row in rows]
        genre_map: dict[str, list[str]] = {key: [] for key in keys}
        for chunk in _chunks(keys):
            placeholders = ",".join("?" * len(chunk))
            for rk, genre in conn.execute(
                f"SELECT rating_key, genre FROM track_genres WHERE rating_key IN ({placeholders})",
                chunk,
            ):
                genre_map[rk].append(genre)

        return [
            Track(
                rating_key=rating_key,
                title=title,
                artist=artist,
                album=album,
                duration_ms=duration_ms or 0,
                year=year,
                genres=genre_map.get(rating_key, []),
                is_live=bool(is_live),
                art_url=f"/api/art/{rating_key}",
            )
            for rating_key, title, artist, album, year, duration_ms, is_live in rows
        ]

    def _build_where(
        self,
        genres: list[str] | None,
        decades: list[str] | None,
        exclude_live: bool,
        min_rating: int,
    ) -> tuple[list[str], list[Any]]:
        """Build SQL WHERE clauses/params for the standard filter set."""
        where: list[str] = []
        params: list[Any] = []

        if genres:
            placeholders = ",".join("?" * len(genres))
            where.append(
                "EXISTS (SELECT 1 FROM track_genres g "
                f"WHERE g.rating_key = t.rating_key AND g.genre IN ({placeholders}))"
            )
            params.extend(genres)

        if decades:
            decade_clauses = []
            for decade in decades:
                start, end = _decade_to_range(decade)
                if start is None:
                    continue
                decade_clauses.append("(t.year >= ? AND t.year <= ?)")
                params.extend([start, end])
            if decade_clauses:
                where.append("(" + " OR ".join(decade_clauses) + ")")

        if exclude_live:
            where.append("t.is_live = 0")

        if min_rating > 0:
            where.append("t.user_rating >= ?")
            params.append(min_rating)

        return where, params

    def _filter_query(self, where: list[str]) -> str:
        """SQL for a filtered track listing (see :meth:`get_tracks_by_filters`).

        The ``ORDER BY`` matches the leading columns of
        ``idx_tracks_nocase_sort``; keep the two in step.
        """
        sql = (
            "SELECT t.rating_key, t.title, t.artist, t.album, t.year, t.duration_ms, t.is_live "
            "FROM tracks t"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        return sql + (
            " ORDER BY t.artist COLLATE NOCASE, t.album COLLATE NOCASE, t.title COLLATE NOCASE"
        )

    def get_tracks_by_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        exclude_live: bool = True,
        min_rating: int = 0,
        limit: int | None = None,
    ) -> list[Track]:
        """Return tracks matching filter criteria using SQL (no Plex call)."""
        conn = self._get_conn()
        where, params = self._build_where(genres, decades, exclude_live, min_rating)

        sql = self._filter_query(where)
        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return self._rows_to_tracks(rows)

    def count_tracks_by_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        exclude_live: bool = True,
        min_rating: int = 0,
    ) -> int:
        """Count tracks matching filter criteria using SQL."""
        conn = self._get_conn()
        where, params = self._build_where(genres, decades, exclude_live, min_rating)
        sql = "SELECT COUNT(*) FROM tracks t"
        if where:
            sql += " WHERE " + " AND ".join(where)
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    def get_random_tracks(self, count: int, exclude_live: bool = True) -> list[Track]:
        """Return up to *count* random tracks without hitting Plex."""
        conn = self._get_conn()
        sql = "SELECT rating_key, title, artist, album, year, duration_ms, is_live FROM tracks"
        if exclude_live:
            sql += " WHERE is_live = 0"
        sql += " ORDER BY RANDOM() LIMIT ?"
        rows = conn.execute(sql, (max(0, count),)).fetchall()
        return self._rows_to_tracks(rows)

    def search_tracks(self, query: str, limit: int = 20) -> list[Track]:
        """Search tracks by title, artist, or album.

        Uses the FTS5 index when available, falling back to ``LIKE``.
        """
        query = (query or "").strip()
        if not query:
            return []

        conn = self._get_conn()

        if self._fts_enabled:
            match = _build_fts_match(query)
            if match:
                try:
                    rows = conn.execute(
                        """SELECT t.rating_key, t.title, t.artist, t.album, t.year, t.duration_ms, t.is_live
                           FROM tracks_fts f
                           JOIN tracks t ON t.rating_key = f.rating_key
                           WHERE tracks_fts MATCH ?
                           ORDER BY f.rank
                           LIMIT ?""",
                        (match, limit),
                    ).fetchall()
                    return self._rows_to_tracks(rows)
                except sqlite3.OperationalError:
                    logger.exception("FTS query failed; falling back to LIKE")

        like = f"%{query.lower()}%"
        rows = conn.execute(
            """SELECT rating_key, title, artist, album, year, duration_ms, is_live
               FROM tracks
               WHERE lower(title) LIKE ? OR lower(artist) LIKE ? OR lower(album) LIKE ?
               ORDER BY artist COLLATE NOCASE, title COLLATE NOCASE
               LIMIT ?""",
            (like, like, like, limit),
        ).fetchall()
        return self._rows_to_tracks(rows)

    def get_live_flags(self, rating_keys: list[str]) -> dict[str, bool]:
        """Return the pre-computed live flag for each known rating key.

        Lets callers reuse the sync-time detection instead of re-running the
        regex on raw Plex objects.
        """
        if not rating_keys:
            return {}
        conn = self._get_conn()
        result: dict[str, bool] = {}
        for chunk in _chunks(list(rating_keys)):
            placeholders = ",".join("?" * len(chunk))
            for rk, is_live in conn.execute(
                f"SELECT rating_key, is_live FROM tracks WHERE rating_key IN ({placeholders})",
                chunk,
            ):
                result[rk] = bool(is_live)
        return result

    def get_library_stats(self) -> dict[str, Any]:
        """Return ``{total_tracks, genres, decades}`` computed with SQL.

        Genres and decades include track counts, unlike the Plex filter-choices
        API which omits them.
        """
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

        genres = [
            {"name": name, "count": count}
            for name, count in conn.execute(
                "SELECT genre, COUNT(*) FROM track_genres "
                "GROUP BY genre ORDER BY genre COLLATE NOCASE"
            )
        ]

        decades = [
            {"name": f"{decade}s", "count": count}
            for decade, count in conn.execute(
                "SELECT (year / 10) * 10 AS decade, COUNT(*) FROM tracks "
                "WHERE year IS NOT NULL AND year > 0 "
                "GROUP BY decade ORDER BY decade"
            )
        ]

        return {"total_tracks": total, "genres": genres, "decades": decades}

    def rebuild_fts(self) -> int:
        """Rebuild the FTS index from the tracks table. Returns row count."""
        if not self._fts_enabled:
            return 0
        with self._write() as conn:
            conn.execute("DELETE FROM tracks_fts")
            conn.execute(
                "INSERT INTO tracks_fts (rating_key, title, artist, album) "
                "SELECT rating_key, title, artist, album FROM tracks"
            )
            row = conn.execute("SELECT COUNT(*) FROM tracks_fts").fetchone()
        return row[0] if row else 0

    def get_metadata(self, rating_key: str) -> dict[str, Any] | None:
        """Get metadata for a single track.

        Returns:
            Dict with user_rating, play_count, last_played, is_live, genres
            or None if not found
        """
        conn = self._get_conn()
        row = conn.execute(
            """SELECT user_rating, play_count, last_played, is_live
               FROM tracks WHERE rating_key = ?""",
            (rating_key,),
        ).fetchone()

        if row is None:
            return None

        genres = [
            r[0] for r in conn.execute(
                "SELECT genre FROM track_genres WHERE rating_key = ?",
                (rating_key,),
            ).fetchall()
        ]

        return {
            "user_rating": row[0],
            "play_count": row[1],
            "last_played": row[2],
            "is_live": bool(row[3]),
            "genres": genres,
        }

    def get_metadata_batch(self, rating_keys: list[str]) -> dict[str, dict[str, Any]]:
        """Get metadata for multiple tracks in one query.

        Args:
            rating_keys: List of rating keys to look up

        Returns:
            Dict mapping rating_key → metadata dict
        """
        if not rating_keys:
            return {}

        conn = self._get_conn()
        result: dict[str, dict[str, Any]] = {}

        for chunk in _chunks(list(rating_keys)):
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""SELECT rating_key, user_rating, play_count, last_played, is_live
                    FROM tracks WHERE rating_key IN ({placeholders})""",
                chunk,
            ).fetchall()
            for row in rows:
                result[row[0]] = {
                    "user_rating": row[1],
                    "play_count": row[2],
                    "last_played": row[3],
                    "is_live": bool(row[4]),
                    "genres": [],
                }

        # Fetch genres for all keys in one query
        if result:
            for chunk in _chunks(list(result.keys())):
                placeholders = ",".join("?" * len(chunk))
                genre_rows = conn.execute(
                    f"""SELECT rating_key, genre FROM track_genres
                        WHERE rating_key IN ({placeholders})""",
                    chunk,
                ).fetchall()
                for rk, genre in genre_rows:
                    if rk in result:
                        result[rk]["genres"].append(genre)

        return result

    def get_track_count(self) -> int:
        """Get total number of tracks in the database."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()
        return row[0] if row else 0

    def get_stale_tracks(self, max_age_hours: int = 24) -> list[str]:
        """Get rating keys of tracks not updated within max_age_hours.

        Useful for identifying tracks that need re-syncing.
        """
        conn = self._get_conn()
        cutoff = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() - max_age_hours * 3600),
        )
        rows = conn.execute(
            "SELECT rating_key FROM tracks WHERE updated_at < ?",
            (cutoff,),
        ).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        """Clear all data from the database."""
        with self._write() as conn:
            conn.execute("DELETE FROM tracks")
            conn.execute("DELETE FROM track_genres")
            conn.execute("DELETE FROM track_dimensions")
            conn.execute("DELETE FROM meta")
            if self._fts_enabled:
                conn.execute("DELETE FROM tracks_fts")
        logger.info("TrackDB cleared")

    def close(self) -> None:
        """Close every connection opened by this database.

        Threads that touch the DB afterwards transparently open a new
        connection (see :meth:`_get_conn`).
        """
        with self._lock:
            self._generation += 1
            connections, self._connections = self._connections, []
            for conn in connections:
                try:
                    conn.close()
                except Exception:
                    logger.exception("Failed to close TrackDB connection")
            logger.info("TrackDB connections closed")


# Global instance
_track_db: TrackDB | None = None


def get_track_db() -> TrackDB | None:
    """Get the global TrackDB instance."""
    return _track_db


def init_track_db(db_path: str | Path | None = None) -> TrackDB:
    """Initialize or reinitialize the global TrackDB instance."""
    global _track_db
    if _track_db:
        _track_db.close()
    _track_db = TrackDB(db_path)
    return _track_db
