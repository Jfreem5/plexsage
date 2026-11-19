"""Import Spotify playlists into Plex: read the rows, match against the library.

A pasted playlist ref (``open.spotify.com/playlist/…`` URL, ``spotify:playlist:<id>``
or a bare id) names the playlist; the rows behind it come by one of two paths, chosen
by what the config provides:

* **api** — ``SPOTIFY_CLIENT_ID``/``SPOTIFY_CLIENT_SECRET`` (a free Spotify for
  Developers app pair) fetch fresh rows from api.spotify.com via the client-credentials
  grant, using the ``requests`` dependency plexapi already brings in — no new package.
* **dump** — ``spotify.parquet_dir`` points at a one-directory-per-table Annas/Spotify
  parquet dump; the ``playlists``/``playlist_tracks`` tables resolve the playlist id
  from that snapshot, which needs duckdb if the environment has it (the same
  duckdb-inside-the-helper convention the other optional readers here use: its
  absence is a named error, never a crash of the rest of the app).

Matching walks the cache's join-key table (:func:`library_cache.ensure_track_keys`)
with the same normalization the cache writes those keys with (:func:`plex_client.song_key`,
:func:`plex_client.album_key`, :func:`plex_client.artist_key`): exact
song key, then title+artist (a different album is the same song's other version),
then title+album (compilations credit artists differently), preferring the
candidate whose duration sits closest to Spotify's. Without a populated cache the
live-Plex fallback searches per row instead. Everything the library cannot answer
is reported AND written to the cache DB's ``missing_songs`` table (by
:func:`library_cache.record_missing_songs`) so the wish list can be acted on later.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

from backend import library_cache
from backend.plex_client import album_key, artist_key, norm, song_key

logger = logging.getLogger(__name__)

# Where the two fetch paths live
SPOTIFY_ACCOUNTS_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

SPOTIFY_PAGE_SIZE = 50  # API page size while paging a playlist
MAX_PLAYLIST_TRACKS = 1000  # Rows read from one playlist, either path
SPOTIFY_SEARCH_LIMIT = 12  # Plex candidates fetched per row on the live-Plex path
CACHE_KEY_SCAN_LIMIT = 400  # Library rows considered per fallback query

# ids are alnum; inlining one into the dump query is safe only because of this
PLAYLIST_ID_RE = re.compile(r"[A-Za-z0-9]{8,64}")

# Cached client-credentials token: (token, expiry epoch). Spotify hands back ~1h
# life; the entry is dropped at that point and fetched again on the next call.
_token_cache: tuple[str, float] | None = None


class SpotifyImportError(Exception):
    """Raised when a Spotify playlist cannot be fetched or resolved."""


def parse_playlist_ref(ref: str) -> tuple[str, str]:
    """Extract ``(playlist_id, canonical_url)`` from *ref*.

    Accepts an ``open.spotify.com`` playlist URL (including the ``/user/x/playlist/y``
    form), a ``spotify:playlist:<id>`` URI, or a bare id.
    """
    text = (ref or "").strip()
    if not text:
        raise SpotifyImportError("playlist_ref must name the Spotify playlist (URL or id)")
    if text.lower().startswith("spotify:"):
        parts = text.split(":")
        if len(parts) == 3 and parts[1].lower() == "playlist" and PLAYLIST_ID_RE.fullmatch(parts[2]):
            playlist_id = parts[2]
        else:
            raise SpotifyImportError(f"not a Spotify playlist URI: {text}")
    elif "/" in text or text.lower().startswith("http"):
        match = re.search(r"/playlist/([A-Za-z0-9]+)", text)
        if not match:
            raise SpotifyImportError(
                f"no playlist id found in {text} (expected .../playlist/<id>)"
            )
        playlist_id = match.group(1)
    elif PLAYLIST_ID_RE.fullmatch(text):
        playlist_id = text
    else:
        raise SpotifyImportError(f"not a Spotify playlist id: {text}")
    return playlist_id, f"https://open.spotify.com/playlist/{playlist_id}"


def api_access_token(client_id: str, client_secret: str) -> str:
    """Client-credentials token for the configured app pair (cached until expiry)."""
    global _token_cache
    now = time.time()
    if _token_cache and _token_cache[1] > now + 30:
        return _token_cache[0]
    try:
        response = requests.post(
            SPOTIFY_ACCOUNTS_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=15,
        )
    except requests.RequestException as e:
        raise SpotifyImportError(f"Spotify credentials endpoint unreachable: {e}") from e
    if response.status_code != 200:
        raise SpotifyImportError(
            f"Spotify credentials rejected (HTTP {response.status_code}): "
            "check SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET or the spotify: config section"
        )
    body = response.json()
    token = str(body.get("access_token") or "")
    if not token:
        raise SpotifyImportError("Spotify credentials endpoint returned no access_token")
    expires_in = int(body.get("expires_in") or 3600)
    _token_cache = (token, now + expires_in)
    return token


def _api_get(url: str, token: str, params: dict[str, Any]) -> dict[str, Any]:
    """One authenticated GET against the Spotify API; raise on any non-200."""
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=20)
    except requests.RequestException as e:
        raise SpotifyImportError(f"Spotify API unreachable: {e}") from e
    if response.status_code != 200:
        raise SpotifyImportError(f"Spotify API returned HTTP {response.status_code} for {url}")
    return response.json()


def fetch_playlist_api(playlist_id: str, client_id: str, client_secret: str) -> dict[str, Any]:
    """Fresh playlist rows from api.spotify.com (public playlists only).

    Returns:
        Dict with ``name``, ``owner``, ``tracks`` (row dicts, 1-based positions)
        and ``warnings``.
    """
    token = api_access_token(client_id, client_secret)
    fields = "id,name,description,owner(display_name),tracks(total,href,next,item)"
    try:
        body = _api_get(f"{SPOTIFY_API_BASE}/playlists/{playlist_id}", token, {"fields": fields})
    except SpotifyImportError as e:
        if "HTTP 401" not in str(e):
            raise
        # A rotated or mistaken cached token 401s; fetch a fresh one and retry once.
        _token_cache_clear()
        token = api_access_token(client_id, client_secret)
        body = _api_get(f"{SPOTIFY_API_BASE}/playlists/{playlist_id}", token, {"fields": fields})
    tracks_body = body.get("tracks") or {}
    warnings: list[str] = []
    tracks: list[dict[str, Any]] = []
    episodes_skipped = 0
    url = tracks_body.get("href") or f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks"
    while url and len(tracks) < MAX_PLAYLIST_TRACKS:
        page = _api_get(url, token, {"limit": SPOTIFY_PAGE_SIZE})
        for item in page.get("items") or []:
            if item.get("type") == "episode":
                episodes_skipped += 1
                continue
            row = _api_row(item, len(tracks) + 1)
            if row is None:
                continue
            tracks.append(row)
        next_url = page.get("next")
        if tracks_body.get("total") and len(tracks) >= MAX_PLAYLIST_TRACKS and next_url:
            warnings.append(
                f"playlist has {tracks_body['total']} rows; stopped reading at {MAX_PLAYLIST_TRACKS}"
            )
        url = next_url if len(tracks) < MAX_PLAYLIST_TRACKS else None
    if episodes_skipped:
        warnings.append(f"{episodes_skipped} podcast episodes skipped (playlists mix in episodes)")
    return {
        "name": str(body.get("name") or ""),
        "owner": str((body.get("owner") or {}).get("display_name") or ""),
        "tracks": tracks,
        "warnings": warnings,
    }


def _token_cache_clear() -> None:
    """Drop the cached client-credentials token (it was just rejected)."""
    global _token_cache
    _token_cache = None


def _api_row(item: dict[str, Any], fallback_position: int) -> dict[str, Any] | None:
    """One playlist item as a row dict, or None when it names no song."""
    track = item.get("track") or item.get("item")
    if not isinstance(track, dict) or not track.get("name"):
        return None
    artists = " & ".join(
        str(a.get("name")) for a in (track.get("artists") or []) if a.get("name")
    )
    album = track.get("album") or {}
    track_id = str(track.get("id") or "")
    position = item.get("index")
    position = position + 1 if isinstance(position, int) else fallback_position
    return {
        "position": position,
        "title": str(track.get("name") or ""),
        "album": str(album.get("name") or ""),
        "artist": artists or str(track.get("artist_name") or ""),
        "duration_ms": int(track.get("duration_ms") or 0),
        "spotify_track_id": track_id,
        "spotify_url": f"https://open.spotify.com/track/{track_id}" if track_id else "",
    }


def fetch_playlist_dump(playlist_id: str, parquet_dir: str) -> dict[str, Any]:
    """Playlist rows from the local parquet dump (snapshot coverage only)."""
    root = Path(parquet_dir)
    for table in ("playlists", "playlist_tracks", "tracks", "albums"):
        if not (root / table).is_dir():
            raise SpotifyImportError(
                f"{root} is not the one-directory-per-table parquet dump "
                "(needs playlists/, playlist_tracks/, tracks/ and albums/ parts inside)"
            )
    try:
        import duckdb
    except ImportError:
        raise SpotifyImportError(
            "resolving a Spotify playlist from the local dump needs duckdb installed "
            "(or configure SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET to fetch from the API)"
        )
    # parse_playlist_ref guarantees an alnum id, so inlining it into the query is safe.
    quoted = f"'{playlist_id}'"
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE want AS SELECT rowid AS proid, name, owner_display_name "
            f"FROM read_parquet('{root}/playlists/*.parquet') WHERE id = {quoted};"
        )
        head = con.execute("SELECT proid, name, owner_display_name FROM want").fetchall()
        if not head:
            raise SpotifyImportError(
                f"playlist {playlist_id} is not in the local dump "
                "(only playlists the snapshot captured are)"
            )
        _, playlist_name, playlist_owner = head[0]
        con.execute(
            "CREATE TABLE entry AS SELECT pt.position, pt.track_rowid, "
            "pt.name_if_is_local, pt.album_name_if_is_local, "
            "pt.artists_name_if_is_local, pt.duration_ms_if_is_local "
            f"FROM read_parquet('{root}/playlist_tracks/*.parquet') AS pt "
            "JOIN want ON pt.playlist_rowid = want.proid "
            "WHERE pt.is_episode = 0;"
        )
        # track_artists carries one row per credited artist, so GROUP BY the slot
        # and let ANY() name one credit: the fallback match below tries every
        # artist of a row anyway, so any single one still lands the song.
        rows = con.execute(
            "SELECT e.position + 1 AS pos, "
            "COALESCE(ANY(tr.name), ANY(e.name_if_is_local), '') AS title, "
            "COALESCE(ANY(al.name), ANY(e.album_name_if_is_local), '') AS album, "
            "COALESCE(ANY(art.name), ANY(e.artists_name_if_is_local), '') AS artist, "
            "COALESCE(ANY(tr.duration_ms), ANY(e.duration_ms_if_is_local), 0) AS duration_ms "
            "FROM entry AS e "
            f"LEFT JOIN read_parquet('{root}/tracks/*.parquet') AS tr ON tr.rowid = e.track_rowid "
            f"LEFT JOIN read_parquet('{root}/albums/*.parquet') AS al ON al.rowid = tr.album_rowid "
            f"LEFT JOIN read_parquet('{root}/track_artists/*.parquet') AS ta ON ta.track_rowid = e.track_rowid "
            f"LEFT JOIN read_parquet('{root}/artists/*.parquet') AS art ON art.rowid = ta.artist_rowid "
            "GROUP BY e.position ORDER BY e.position"
        ).fetchall()
    finally:
        con.close()
    tracks = [
        {
            "position": int(row[0] or 0),
            "title": str(row[1] or ""),
            "album": str(row[2] or ""),
            "artist": str(row[3] or ""),
            "duration_ms": int(row[4] or 0),
            "spotify_track_id": "",
            "spotify_url": "",
        }
        for row in rows[:MAX_PLAYLIST_TRACKS]
    ]
    warnings: list[str] = []
    if len(rows) > MAX_PLAYLIST_TRACKS:
        warnings.append(
            f"playlist has {len(rows)} rows in the dump; stopped reading at {MAX_PLAYLIST_TRACKS}"
        )
    warnings.append(
        "rows come from the local parquet snapshot: it covers only playlists and tracks "
        "it captured, and dump rows carry no track ids, so missing songs are named without links"
    )
    return {
        "name": str(playlist_name or ""),
        "owner": str(playlist_owner or ""),
        "tracks": tracks,
        "warnings": warnings,
    }


def fetch_playlist(playlist_ref: str, config: Any) -> dict[str, Any]:
    """Read *playlist_ref*'s rows by the configured path ("api" or "dump").

    Credentials win when present; without them a configured parquet dump resolves
    the playlist from the local snapshot. Neither configured is an error that
    names both remedies, so the response never has to guess.
    """
    playlist_id, playlist_url = parse_playlist_ref(playlist_ref)
    spotify = config.spotify
    if spotify.client_id and spotify.client_secret:
        source = "api"
        body = fetch_playlist_api(playlist_id, spotify.client_id, spotify.client_secret)
    elif spotify.parquet_dir:
        source = "dump"
        body = fetch_playlist_dump(playlist_id, spotify.parquet_dir)
    else:
        raise SpotifyImportError(
            "No Spotify import path configured: set SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET "
            "(or a spotify: section in config.yaml) to fetch fresh rows from the Spotify API, "
            "or set SPOTIFY_PARQUET_DIR / spotify.parquet_dir to resolve the playlist "
            "from a local Annas/Spotify parquet dump"
        )
    return {
        "playlist_id": playlist_id,
        "playlist_url": playlist_url,
        "source": source,
        "name": body["name"],
        "owner": body["owner"],
        "tracks": body["tracks"],
        "warnings": body["warnings"],
    }


def _pick_candidate(candidates: list[dict[str, Any]], want_ms: int) -> dict[str, Any]:
    """The candidate whose duration sits closest to *want_ms* (first on a tie).

    Unknown durations (0 either side) cannot separate candidates, so the query
    order decides those; a Spotify row that knows its length and a library row
    that knows its own pick the same-recording version over a deluxe twin.
    """
    if not candidates:
        return {}
    if want_ms <= 0:
        return candidates[0]
    known = [c for c in candidates if c.get("duration_ms")]
    pool = known or candidates
    return min(pool, key=lambda c: abs((c.get("duration_ms") or 0) - want_ms))


def _match_rows(tracks: list[dict[str, Any]], select) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The three-rung walk over *tracks*; *select* runs one SQL against the cache.

    Shared by the cache path: exact song key, then title+artist per credited
    artist, then title+album. ``match_kind`` records which rung answered.
    """
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    claimed: dict[str, dict[str, Any]] = {}  # song key -> chosen library row (duplicates reuse)

    exact_sql = (
        "SELECT t.rating_key, t.title, t.artist, t.album, t.duration_ms "
        "FROM track_keys AS k JOIN tracks AS t ON t.rating_key = k.rating_key "
        "WHERE k.match_key = ? LIMIT ?"
    )
    version_sql = (
        "SELECT t.rating_key, t.title, t.artist, t.album, t.duration_ms "
        "FROM track_keys AS k JOIN tracks AS t ON t.rating_key = k.rating_key "
        "WHERE k.ck_artist = ? AND k.match_key GLOB ? LIMIT ?"
    )
    album_sql = (
        "SELECT t.rating_key, t.title, t.artist, t.album, t.duration_ms "
        "FROM track_keys AS k JOIN tracks AS t ON t.rating_key = k.rating_key "
        "WHERE k.ck_album = ? AND k.match_key GLOB ? LIMIT ?"
    )

    for row in tracks:
        title_n = norm(row.get("title"))
        key = song_key(row.get("title"), row.get("album"), row.get("artist"))
        if key in claimed:
            reuse = dict(claimed[key])
            reuse["position"] = int(row.get("position") or 0)
            matched.append(reuse)
            continue
        want_ms = int(row.get("duration_ms") or 0)
        # norm leaves letters/digits/spaces only, so the pattern below cannot
        # carry a GLOB wildcard of its own.
        title_prefix = f"{title_n}|*"

        candidates = select(exact_sql, (key, 10))
        kind = "exact"
        if not candidates:
            artists = [artist_key(row.get("artist"))]
            for credited in str(row.get("artist") or "").split("&"):
                folded = artist_key(credited)
                if folded and folded not in artists:
                    artists.append(folded)
            for artist_part in artists:
                candidates = select(version_sql, (artist_part, title_prefix, CACHE_KEY_SCAN_LIMIT))
                if candidates:
                    kind = "version"
                    break
        album_n = album_key(row.get("album"))
        if not candidates and album_n:
            candidates = select(album_sql, (album_n, title_prefix, CACHE_KEY_SCAN_LIMIT))
            if candidates:
                kind = "album"

        chosen = _pick_candidate(candidates, want_ms) if candidates else {}
        if not chosen:
            missing.append(dict(row))
            continue
        chosen["match_kind"] = kind
        claimed[key] = chosen
        matched.append({**chosen, "position": int(row.get("position") or 0)})
    return matched, missing


def match_cached(tracks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match Spotify rows against the local cache; return ``(matched, missing)``.

    Exact song key first, then title+artist (other album = another version of the
    same song, tried for every credited artist so "A & B" still lands when Plex
    lists "B feat. A"), then title+album (compilations credit artists
    differently). ``match_kind`` says which rung answered, so the caller can show
    the near misses instead of implying an exact match.
    """
    conn = library_cache.ensure_db_initialized()
    try:
        library_cache.ensure_track_keys(conn)

        def select(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
            return [
                {
                    "rating_key": row["rating_key"],
                    "title": row["title"],
                    "artist": row["artist"],
                    "album": row["album"],
                    "duration_ms": int(row["duration_ms"] or 0),
                }
                for row in conn.execute(sql, params)
            ]

        return _match_rows(tracks, select)
    finally:
        conn.close()


def match_live(tracks: list[dict[str, Any]], plex: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match Spotify rows on the live-Plex fallback (no populated cache).

    Plex full-text search answers per Spotify row (:meth:`PlexClient.search_tracks`
    is title-matched server-side); the rung labels mirror the cache path, and
    ``match_kind`` is "search" for whichever comparison landed first, since the
    Plex path cannot ask the three key queries the cache branch can.
    """
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    claimed: dict[str, dict[str, Any]] = {}

    for row in tracks:
        key = song_key(row.get("title"), row.get("album"), row.get("artist"))
        if key in claimed:
            reuse = dict(claimed[key])
            reuse["position"] = int(row.get("position") or 0)
            matched.append(reuse)
            continue
        want_ms = int(row.get("duration_ms") or 0)
        title_n = norm(row.get("title"))
        album_n = norm(row.get("album"))
        artist_parts = [artist_key(row.get("artist"))]
        for credited in str(row.get("artist") or "").split("&"):
            folded = artist_key(credited)
            if folded and folded not in artist_parts:
                artist_parts.append(folded)

        try:
            candidates = plex.search_tracks(str(row.get("title") or ""), limit=SPOTIFY_SEARCH_LIMIT)
        except Exception:
            logger.exception("Live-Plex search for '%s' failed", row.get("title"))
            candidates = []

        exact: list[dict[str, Any]] = []
        near: list[dict[str, Any]] = []
        for candidate in candidates:
            cand = {
                "rating_key": candidate.rating_key,
                "title": candidate.title,
                "artist": candidate.artist,
                "album": candidate.album,
                "duration_ms": candidate.duration_ms,
            }
            if song_key(cand["title"], cand["album"], cand["artist"]) == key:
                exact.append(cand)
            elif norm(cand["title"]) == title_n and (
                artist_key(cand["artist"]) in artist_parts
                or (album_n and album_key(cand["album"]) == album_n)
            ):
                near.append(cand)
        # Plex's search is full-text, so it answers rows the key rungs would not
        # have asked about; label every landing "search" rather than claim a rung.
        pool = exact or near
        chosen = _pick_candidate(pool, want_ms) if pool else {}
        if not chosen:
            missing.append(dict(row))
            continue
        chosen["match_kind"] = "search"
        claimed[key] = chosen
        matched.append({**chosen, "position": int(row.get("position") or 0)})
    return matched, missing
