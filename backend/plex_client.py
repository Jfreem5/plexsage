"""Plex server client for library queries and playlist management."""

import hashlib
import logging
import re
import threading
import time
from typing import Any

from plexapi.exceptions import NotFound, Unauthorized
from plexapi.playqueue import PlayQueue
from plexapi.server import PlexServer
from requests.exceptions import ConnectionError, Timeout
from unidecode import unidecode

from backend.models import (
    OWNER_USER_KEY,
    PlexClientInfo,
    PlexPlaylistInfo,
    PlexUserInfo,
    Track,
)

logger = logging.getLogger(__name__)


class TrackCache:
    """In-memory cache for filtered track results with TTL."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 50):
        """Initialize cache with TTL in seconds (default 5 minutes) and max entries."""
        self._cache: dict[str, tuple[list[Track], float]] = {}
        self._ttl = ttl_seconds
        self._max_entries = max_entries

    def _make_key(
        self,
        genres: list[str] | None,
        decades: list[str] | None,
        exclude_live: bool,
        min_rating: int,
        artists: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        min_duration_ms: int = 0,
        max_duration_ms: int = 0,
    ) -> str:
        """Create deterministic cache key from filter params."""
        key_data = {
            "genres": sorted(genres or []),
            "decades": sorted(decades or []),
            "exclude_live": exclude_live,
            "min_rating": min_rating,
            "artists": sorted(artists or []),
            "exclude_genres": sorted(exclude_genres or []),
            "min_duration_ms": min_duration_ms,
            "max_duration_ms": max_duration_ms,
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()

    def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if not self._cache:
            return
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]
        logger.info("Evicted oldest cache entry (key=%s)", oldest_key[:8])

    def get(
        self,
        genres: list[str] | None,
        decades: list[str] | None,
        exclude_live: bool,
        min_rating: int,
        artists: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        min_duration_ms: int = 0,
        max_duration_ms: int = 0,
    ) -> list[Track] | None:
        """Get cached tracks if available and not expired."""
        key = self._make_key(
            genres,
            decades,
            exclude_live,
            min_rating,
            artists,
            exclude_genres,
            min_duration_ms,
            max_duration_ms,
        )
        if key in self._cache:
            tracks, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                logger.info("Cache hit for filters (key=%s)", key[:8])
                return tracks
            else:
                logger.info("Cache expired for filters (key=%s)", key[:8])
                del self._cache[key]
        return None

    def set(
        self,
        genres: list[str] | None,
        decades: list[str] | None,
        exclude_live: bool,
        min_rating: int,
        tracks: list[Track],
        artists: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        min_duration_ms: int = 0,
        max_duration_ms: int = 0,
    ) -> None:
        """Cache tracks with current timestamp."""
        key = self._make_key(
            genres,
            decades,
            exclude_live,
            min_rating,
            artists,
            exclude_genres,
            min_duration_ms,
            max_duration_ms,
        )

        # Evict oldest if at capacity (and not updating existing key)
        if key not in self._cache and len(self._cache) >= self._max_entries:
            self._evict_oldest()

        self._cache[key] = (tracks, time.time())
        logger.info("Cached %d tracks (key=%s)", len(tracks), key[:8])

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Track cache cleared")


# Global cache instance
_track_cache = TrackCache()


def get_track_cache() -> TrackCache:
    """Get the global track cache instance."""
    return _track_cache


class PlexQueryError(Exception):
    """Raised when a Plex library query fails."""


# Fuzzy matching threshold (0-100)
FUZZ_THRESHOLD = 60

# Patterns for detecting live recordings
DATE_PATTERN = r"\d{4}[-/]\d{2}[-/]\d{2}"
LIVE_KEYWORDS = r"\b(?:live|concert|sbd|bootleg)\b"

# Cap on the artist roster offered as filter suggestions (mirrors the identical
# constant in library_cache.py; a library can hold thousands of artists).
MAX_ARTIST_OPTIONS = 60

# Random sample size used to estimate duration percentiles on the Plex path.
DURATION_SAMPLE_SIZE = 400

# Shape returned whenever duration stats cannot be measured.
_EMPTY_DURATION_STATS = {
    "total": 0,
    "min_ms": 0,
    "p25_ms": 0,
    "p50_ms": 0,
    "p75_ms": 0,
    "max_ms": 0,
}


def simplify_string(s: str) -> str:
    """Normalize string for fuzzy comparison."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)  # Remove punctuation
    s = unidecode(s)  # Normalize unicode (café → cafe)
    return s


#: Join-key normalisation for the local cache's join tables. plexclean vendors the same
#: five functions in ``plexclean/text.py`` — the one-key/one-spelling rule (brief §5.1)
#: is what lets plexclean's enrichment legs and this repo's Spotify-import matcher agree
#: on a key without either one paraphrasing the other.
ARTIST_CONNECTOR = re.compile(r"&|\band\b", re.IGNORECASE)
KEY_PARTS = re.compile(r"\s{2,}")


def norm(value: Any) -> str:
    """Normalized text of a key part: simplify_string, whitespace tidied."""
    if not value:
        return ""
    return KEY_PARTS.sub(" ", simplify_string(str(value))).strip()


def norm_artist(value: Any) -> str:
    """Artist text collapsed so "&" and "and" spellings share one key."""
    if not value:
        return ""
    return norm(ARTIST_CONNECTOR.sub(" ", str(value)))


def song_key(title: Any, album: Any, artist: Any) -> str:
    """Join key of one song: ``norm(title)|norm(album)|norm_artist(artist)``."""
    return f"{norm(title)}|{norm(album)}|{norm_artist(artist)}"


def album_key(album: Any) -> str:
    """Join key of one album release: ``norm(album)``."""
    return norm(album)


def artist_key(artist: Any) -> str:
    """Join key of one artist: ``norm_artist(artist)``."""
    return norm_artist(artist)


def normalize_artist(name: str) -> list[str]:
    """Return variations of artist name for matching."""
    variations = [name]
    if " and " in name.lower():
        variations.append(name.replace(" and ", " & ").replace(" And ", " & "))
    elif " & " in name:
        variations.append(name.replace(" & ", " and "))
    return variations


# Shortest artist fragment considered for a substring match; "AC" or "a-ha" style
# names are only ever matched exactly, so "abba" cannot drag in "Abba Danza".
ARTIST_MATCH_MIN_LEN = 3

# How much deeper to sample Plex when filters that only apply in Python are set,
# so a rare artist or a short track is not squeezed out of a thin random slice.
POST_FILTER_SAMPLE_FACTOR = 8


def _sample_percentile(sorted_values: list[int], fraction: float) -> int:
    """Nearest-rank percentile of an already-sorted list of millisecond values."""
    if not sorted_values:
        return 0
    rank = min(max(round(len(sorted_values) * fraction), 1), len(sorted_values))
    return int(sorted_values[rank - 1])


def _track_genres(track: Any) -> list[str]:
    """Lowercased genre tags on a raw Plex track (empty when the server omitted them)."""
    if not hasattr(track, "genres"):
        return []
    tags = []
    for g in track.genres:
        tag = g.tag if hasattr(g, "tag") else str(g)
        if tag:
            tags.append(tag.lower())
    return tags


def _is_compilation_track(track: Any) -> bool:
    """Whether a raw Plex track belongs to a multi-artist "best of" album.

    Plex has no compilation flag; the album artist (``grandparentTitle``) is the
    "Various Artists" placeholder exactly on those releases. The track's own
    artist tags are checked too because some releases carry the placeholder
    there instead, and the local cache path has only that column to work from.
    """
    if getattr(track, "grandparentTitle", None):
        return track.grandparentTitle.strip().lower() == "various artists"
    for artist in getattr(track, "artists", []) or []:
        name = artist.title if hasattr(artist, "title") else str(artist)
        if name and name.strip().lower() == "various artists":
            return True
    return False


def _banned_artist_hit(track: Any, banned: list[str]) -> bool:
    """Whether a raw Plex track should be dropped for an artist exclusion.

    Both the album artist (``grandparentTitle``, what the include-artist filter
    already matches) and the track's own artist tags are tested, so a ban does
    not depend on which of the two the server happened to fill in.
    """
    names = [getattr(track, "grandparentTitle", None) or ""]
    names += [
        artist.title if hasattr(artist, "title") else str(artist)
        for artist in (getattr(track, "artists", None) or [])
    ]
    return any(_matches_artist(name, banned) for name in names if name)


def _decade_banned(track: Any, banned: list[str]) -> bool:
    """Whether a raw Plex track falls inside one of the *banned* decade labels.

    Undated tracks are kept: a decade the prompt bans cannot contain a release
    whose year the server never reported, so dropping those would hide music the
    user still asked for.
    """
    if not banned:
        return False
    year = getattr(track, "year", None)
    if not year:
        return False
    starts = set()
    for label in banned:
        digits = str(label).strip().rstrip("s")
        if digits.isdigit():
            starts.add(int(digits))
    return any(start <= int(year) < start + 10 for start in starts)


def _played_outside_window(track: Any, played_after: str, played_before: str) -> bool:
    """Whether a raw Plex track was last heard outside the inclusive date window.

    Plex keeps ``lastViewedAt`` as a datetime on the raw track object; a never-played
    track leaves it unset, and such a row is outside every window — which is what a
    "heard in the last six months" request wants. Comparison is by whole day
    (``YYYY-MM-DD``), the granularity the filter is phrased at.
    """
    if not played_after and not played_before:
        return False
    stamp = getattr(track, "lastViewedAt", None)
    if stamp is None:
        return True
    day = (stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp))[:10]
    if played_after and day < played_after:
        return True
    return bool(played_before and day > played_before)


def _rating_mismatched(
    track: Any, rated_only: bool, unrated_only: bool, max_rating: int
) -> bool:
    """Whether a raw Plex track fails the rated/unrated flags or a rating ceiling.

    Plex leaves ``userRating`` unset when the listener never rated a track, so
    "unrated only" is its own test rather than a rating floor of 0, and an
    unrated row survives a ceiling because it is not above it (same NULL rule
    :func:`_decade_banned` applies to banned decades).
    """
    rating = getattr(track, "userRating", None)
    if rated_only and rating is None:
        return True
    if unrated_only and rating is not None:
        return True
    if max_rating > 0 and rating is not None:
        try:
            return float(rating) > max_rating
        except (TypeError, ValueError):
            return True
    return False


def _not_live_only(track: Any) -> bool:
    """Whether a raw Plex track must be dropped by a "only live takes" request.

    The same heuristics :func:`is_live_version` uses to drop live rows are used
    here to keep only those, since Plex has no live flag either.
    """
    return not is_live_version(track)


def _matches_artist(track_artist: str, wanted: list[str]) -> bool:
    """Whether *track_artist* is one of the *wanted* artist names.

    Compares with :func:`simplify_string` (case- and punctuation-insensitive,
    accent-folded) and containment in both directions, so "beatles" matches
    "The Beatles" and "The Beatles (Remastered)" matches "Beatles". Names shorter
    than :data:`ARTIST_MATCH_MIN_LEN` only match exactly.
    """
    if not wanted:
        return True
    have = simplify_string(track_artist or "")
    for candidate in wanted:
        for variant in normalize_artist(candidate or ""):
            want = simplify_string(variant)
            if not want:
                continue
            if want == have:
                return True
            if len(want) >= ARTIST_MATCH_MIN_LEN and want in have:
                return True
            if len(have) >= ARTIST_MATCH_MIN_LEN and have in want:
                return True
    return False


def is_live_version(track: Any) -> bool:
    """Check if track appears to be a live recording.

    Args:
        track: Plex track object (raw Plex object or Track model)

    Returns:
        True if track appears to be a live version
    """
    # Use parentTitle first - it's already cached on the track object
    # This avoids a network call per track (track.album() does HTTP request)
    album_title = getattr(track, 'parentTitle', '') or ''

    # Only call track.album() if parentTitle is empty and album() exists
    if not album_title and callable(getattr(track, 'album', None)):
        album = track.album()
        album_title = album.title if album else ""

    track_title = track.title

    for text in [album_title, track_title]:
        if re.search(DATE_PATTERN, text):
            return True
        if re.search(LIVE_KEYWORDS, text, re.IGNORECASE):
            return True

    return False


class PlexClient:
    """Client for interacting with Plex server."""

    # Cooldown between reconnection attempts (seconds)
    RECONNECT_COOLDOWN = 30

    # How long a "logged in as another user" connection is reused (seconds)
    USER_SERVER_TTL = 900

    def __init__(self, url: str, token: str, music_library: str = "Music"):
        """Initialize Plex client.

        Args:
            url: Plex server URL
            token: Plex authentication token
            music_library: Name of the music library section
        """
        self.url = url
        self.token = token
        self.music_library_name = music_library
        self._server: PlexServer | None = None
        self._library = None
        self._error: str | None = None
        self._last_reconnect_attempt: float = time.time()
        self._reconnect_lock = threading.Lock()
        self._scratch_lock = threading.Lock()
        # Plex account/user caches (for creating playlists as other users)
        self._account: Any = None
        self._users_by_key: dict[str, Any] = {}
        self._user_servers: dict[str, tuple[Any, float]] = {}
        self._user_lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        """Attempt to connect to Plex server."""
        if not self.url or not self.token:
            self._error = "Plex URL and token are required"
            return

        # Any cached account/user connections belong to the old server object
        with self._user_lock:
            self._account = None
            self._users_by_key = {}
            self._user_servers.clear()

        try:
            self._server = PlexServer(self.url, self.token, timeout=30)
            self._library = self._server.library.section(self.music_library_name)
            self._error = None
        except Unauthorized:
            self._error = "Invalid Plex token - unauthorized"
            self._server = None
            self._library = None
        except NotFound:
            self._error = f"Music library '{self.music_library_name}' not found"
            self._library = None
        except ConnectionError:
            self._error = f"Cannot connect to Plex server at {self.url}"
            self._server = None
            self._library = None
        except Timeout:
            self._error = "Connection to Plex server timed out"
            self._server = None
            self._library = None
        except Exception as e:
            self._error = f"Plex connection error: {str(e)}"
            self._server = None
            self._library = None

    def is_connected(self) -> bool:
        """Check if connected to Plex server with valid library.

        If not connected, attempts to reconnect (with cooldown to avoid hammering).
        Thread-safe to prevent multiple simultaneous reconnection attempts.
        """
        if self._server is not None and self._library is not None:
            return True

        # Not connected - try to reconnect if cooldown has passed
        now = time.time()
        with self._reconnect_lock:
            # Re-check connection inside lock (may have connected while waiting)
            if self._server is not None and self._library is not None:
                return True

            if now - self._last_reconnect_attempt >= self.RECONNECT_COOLDOWN:
                self._last_reconnect_attempt = now
                logger.info("Attempting to reconnect to Plex server...")
                self._connect()

        return self._server is not None and self._library is not None

    def get_machine_identifier(self) -> str | None:
        """Get the Plex server's machine identifier."""
        if not self._server:
            return None
        return self._server.machineIdentifier

    def get_error(self) -> str | None:
        """Get the last error message if any."""
        return self._error

    def get_music_libraries(self) -> list[str]:
        """Get list of music library names."""
        if not self._server:
            return []

        try:
            sections = self._server.library.sections()
            return [s.title for s in sections if s.type == "artist"]
        except Exception:
            return []

    def get_library_total_tracks(self) -> int:
        """Get total track count in the music library.

        Returns:
            Total number of tracks, or 0 if not connected
        """
        if not self._library:
            return 0

        try:
            return self._library.totalViewSize(libtype="track")
        except Exception as e:
            logger.exception("Failed to get library track count: %s", e)
            return 0

    def get_all_raw_tracks(self) -> list[Any]:
        """Get all raw track objects from the library.

        This fetches all tracks in a single API call. For large libraries
        (30k+ tracks), this may take 30-60 seconds.

        Returns:
            List of raw Plex track objects
        """
        if not self._library:
            return []

        try:
            logger.info("Fetching all tracks from Plex (this may take a while)...")
            return self._library.search(libtype="track")
        except Exception as e:
            logger.exception("Failed to get all tracks: %s", e)
            return []

    def get_all_albums_metadata(self) -> dict[str, dict[str, Any]]:
        """Fetch all albums and return mapping of rating_key -> metadata.

        Returns:
            Dict mapping album rating_key (as string) to dict with 'genres' and 'year'
        """
        if not self._library:
            return {}

        try:
            logger.info("Fetching all albums for metadata mapping...")
            albums = self._library.search(libtype="album")
            album_metadata = {
                str(album.ratingKey): {
                    "genres": [g.tag for g in album.genres],
                    "year": getattr(album, "year", None),
                }
                for album in albums
            }
            logger.info("Got metadata for %d albums", len(album_metadata))
            return album_metadata
        except Exception as e:
            logger.exception("Failed to get album metadata: %s", e)
            return {}

    def get_library_stats(self) -> dict[str, Any]:
        """Get statistics about the music library.

        Returns:
            Dict with total_tracks, genres, and decades
        """
        if not self._library:
            return {"total_tracks": 0, "genres": [], "decades": []}

        try:
            # Get genres using filter choices API (fast) - works at track level
            # Note: listFilterChoices doesn't provide counts, so we omit them
            genre_choices = self._library.listFilterChoices("genre", libtype="track")
            genres = [
                {"name": g.title, "count": None}
                for g in genre_choices
            ]
            genres = sorted(genres, key=lambda x: x["name"])

            # Get decades using filter choices API at album level
            # (decade filter only exists for albums, not tracks)
            decade_choices = self._library.listFilterChoices("decade", libtype="album")
            decades = []
            for d in decade_choices:
                name = d.title
                if name and not name.endswith('s'):
                    name = f"{name}s"
                decades.append({
                    "name": name,
                    "count": None
                })
            decades = sorted(decades, key=lambda x: x["name"])

            # Get total track count efficiently
            # Use totalSize from search response metadata
            total_tracks = self._library.totalViewSize(libtype="track")

            return {
                "total_tracks": total_tracks,
                "genres": genres,
                "decades": decades,
            }
        except Exception as e:
            return {"total_tracks": 0, "genres": [], "decades": [], "error": str(e)}

    def get_all_tracks(self) -> list[Track]:
        """Get all tracks from the library."""
        if not self._library:
            return []

        try:
            plex_tracks = self._library.search(libtype="track")
            return [self._convert_track(t) for t in plex_tracks]
        except Exception:
            return []

    def get_tracks_by_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        exclude_live: bool = True,
        min_rating: int = 0,
        limit: int = 0,
        artists: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        min_duration_ms: int = 0,
        max_duration_ms: int = 0,
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
    ) -> list[Track]:
        """Get tracks matching filter criteria.

        Plex can express the include-genre / decade / rating-floor / year / play-count
        filters, so the artist, artist-exclusion, decade-exclusion, genre-
        exclusion, compilation and duration-window filters are applied to the
        results here instead (same treatment as the live-version step). The new
        facets also run here: Plex has no "only live" filter, no rated/unrated
        test and no last-heard window.

        Args:
            genres: List of genre names to include
            decades: List of decades (e.g., ["1990s", "2000s"])
            exclude_live: Whether to exclude live recordings
            min_rating: Minimum user rating (0-10, 0 = no filter)
            limit: Max tracks to return (0 = no limit). When set, uses random
                   server-side sampling for efficiency with large libraries.
            artists: Artist names to restrict to (case-insensitive containment)
            exclude_genres: Genre names to drop from the results
            min_duration_ms: Minimum track length in milliseconds (0 = no floor)
            max_duration_ms: Maximum track length in milliseconds (0 = no ceiling)
            exclude_artists: Artist names to drop (same containment matching)
            exclude_decades: Decades like "1980s" to drop (undated tracks kept)
            min_year: Oldest acceptable release year (0 = no lower bound)
            max_year: Newest acceptable release year (0 = no upper bound)
            min_plays: Minimum Plex play count (0 = no floor)
            max_plays: Maximum Plex play count (0 = no ceiling)
            unheard_only: Require a zero play count ("hidden gems")
            exclude_compilations: Drop tracks from Various Artists albums
            live_only: Keep only live recordings; this outranks ``exclude_live``
                       so the pair cannot cancel out to an empty pool
            rated_only: Keep only tracks the listener rated
            unrated_only: Keep only tracks the listener never rated
            max_rating: User-rating ceiling (0-10, 0 = no ceiling); unrated rows stay
            played_after / played_before: Inclusive last-heard window as
                    ``YYYY-MM-DD`` dates ("" = unbounded on that side)

        Returns:
            List of matching Track objects
        """
        if not self._library:
            return []

        # Filters Plex cannot express (artist names, not-genre, not-decade,
        # compilations, duration, only-live, rated/unrated, last-heard window).
        post_filters = (
            bool(artists)
            or bool(exclude_artists)
            or bool(exclude_genres)
            or bool(exclude_decades)
            or exclude_compilations
            or live_only
            or rated_only
            or unrated_only
            or max_rating > 0
            or bool(played_after)
            or bool(played_before)
            or min_duration_ms > 0
            or max_duration_ms > 0
        )

        try:
            filters = self._build_filters(
                genres,
                decades,
                min_rating,
                min_year=min_year,
                max_year=max_year,
                min_plays=min_plays,
                max_plays=max_plays,
                unheard_only=unheard_only,
            )

            # When limit is set, use server-side random sampling for efficiency.
            # Client-side filters need headroom in that sample or a rare artist
            # / short song could be missed entirely, so fetch deeper.
            if limit > 0:
                fetch_count = int(limit * 1.3) if exclude_live else limit
                if post_filters:
                    fetch_count = max(fetch_count, limit * POST_FILTER_SAMPLE_FACTOR)
                plex_tracks = self._library.search(
                    libtype="track",
                    sort="random",
                    limit=fetch_count,
                    **filters,
                )
            else:
                plex_tracks = self._library.search(libtype="track", **filters)

            # Post-filter for live versions (can't be done server-side)
            if live_only:
                # "only live takes" outranks the drop-live default, matching the
                # cache path, so the pair cannot cancel out to nothing.
                plex_tracks = [t for t in plex_tracks if not _not_live_only(t)]
            elif exclude_live:
                plex_tracks = [t for t in plex_tracks if not is_live_version(t)]

            # Decade bans, compilations, rated/unrated tests, a rating ceiling and
            # a last-heard window need the raw Plex fields (the album artist and
            # the play history are not carried onto the converted Track).
            if exclude_decades:
                plex_tracks = [
                    t for t in plex_tracks if not _decade_banned(t, exclude_decades)
                ]
            if exclude_compilations:
                plex_tracks = [t for t in plex_tracks if not _is_compilation_track(t)]
            if rated_only or unrated_only or max_rating > 0:
                plex_tracks = [
                    t
                    for t in plex_tracks
                    if not _rating_mismatched(t, rated_only, unrated_only, max_rating)
                ]
            if played_after or played_before:
                plex_tracks = [
                    t
                    for t in plex_tracks
                    if not _played_outside_window(t, played_after, played_before)
                ]

            tracks = [self._convert_track(t) for t in plex_tracks]

            # Artist, genre-exclusion and duration windows are Python-side too:
            # Plex has no not-genre filter and its duration filter is unreliable
            # across servers, so this mirrors is_live_version's post-filtering.
            if artists:
                tracks = [t for t in tracks if _matches_artist(t.artist, artists)]
            if exclude_artists:
                tracks = [
                    t for t in tracks if not _matches_artist(t.artist, exclude_artists)
                ]
            if exclude_genres:
                dropped = [g.lower() for g in exclude_genres]
                tracks = [
                    t for t in tracks
                    if not any(g.lower() in dropped for g in t.genres)
                ]
            if min_duration_ms > 0:
                tracks = [t for t in tracks if t.duration_ms >= min_duration_ms]
            if max_duration_ms > 0:
                tracks = [t for t in tracks if t.duration_ms > 0 and t.duration_ms <= max_duration_ms]

            # Apply limit after every post-filter
            if limit > 0:
                tracks = tracks[:limit]

            return tracks
        except Exception as e:
            logger.exception("Failed to query Plex library with filters: %s", filters)
            raise PlexQueryError(f"Failed to query Plex library: {e}") from e

    def count_tracks_by_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        exclude_live: bool = True,
        min_rating: int = 0,
        artists: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        min_duration_ms: int = 0,
        max_duration_ms: int = 0,
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
    ) -> int:
        """Count matching tracks without converting to Track objects.

        This is faster than get_tracks_by_filters() when only the count is needed.
        The genre / decade / rating / year / play-count filters go to Plex, while
        the artist, artist-exclusion, genre-exclusion, decade-exclusion,
        compilation, duration, only-live, rated/unrated and last-heard-window
        filters cannot be handed to Plex (its track-level genre filter offers
        only = and != operators, and it has no live flag, no rated test and no
        last-heard window), so they are applied to the raw results here. Genre
        exclusion is best-effort on this path: Plex attaches genre tags to only
        part of its search results, whereas the local cache path is exact.

        Args:
            genres: List of genre names to include
            decades: List of decades (e.g., ["1990s", "2000s"])
            exclude_live: Whether to exclude live recordings
            min_rating: Minimum user rating (0-10, 0 = no filter)
            artists: Artist names to restrict to (case-insensitive containment)
            exclude_genres: Genre names to drop from the results
            min_duration_ms: Minimum track length in milliseconds (0 = no floor)
            max_duration_ms: Maximum track length in milliseconds (0 = no ceiling)
            exclude_artists: Artist names to drop (same containment matching)
            exclude_decades: Decades like "1980s" to drop (undated tracks kept)
            min_year: Oldest acceptable release year (0 = no lower bound)
            max_year: Newest acceptable release year (0 = no upper bound)
            min_plays: Minimum Plex play count (0 = no floor)
            max_plays: Maximum Plex play count (0 = no ceiling)
            unheard_only: Require a zero play count ("hidden gems")
            exclude_compilations: Drop tracks from Various Artists albums
            live_only: Keep only live recordings (outranks ``exclude_live``)
            rated_only: Require the listener to have rated the track
            unrated_only: Require the listener to have left the track unrated
            max_rating: User-rating ceiling (0-10, 0 = no ceiling); unrated rows stay
            played_after / played_before: Inclusive last-heard window as
                    ``YYYY-MM-DD`` dates ("" = unbounded on that side)

        Returns:
            Count of matching tracks, or -1 on error
        """
        if not self._library:
            return -1

        post_filters = (
            bool(artists)
            or bool(exclude_artists)
            or bool(exclude_genres)
            or bool(exclude_decades)
            or exclude_compilations
            or live_only
            or rated_only
            or unrated_only
            or max_rating > 0
            or bool(played_after)
            or bool(played_before)
            or min_duration_ms > 0
            or max_duration_ms > 0
        )

        try:
            filters = self._build_filters(
                genres,
                decades,
                min_rating,
                min_year=min_year,
                max_year=max_year,
                min_plays=min_plays,
                max_plays=max_plays,
                unheard_only=unheard_only,
            )

            # Fast path: nothing to apply beyond the server-side filters
            if not filters and not exclude_live and not post_filters:
                return self._library.totalViewSize(libtype="track")

            # Get raw Plex tracks (no conversion to Track objects)
            plex_tracks = self._library.search(libtype="track", **filters)

            if live_only:
                # Count live rows only (the inverse of the drop-live step)
                plex_tracks = [t for t in plex_tracks if not _not_live_only(t)]
            elif exclude_live:
                # Count non-live tracks without full conversion
                # is_live_version uses parentTitle which is already cached
                plex_tracks = [t for t in plex_tracks if not is_live_version(t)]

            if exclude_decades:
                plex_tracks = [
                    t for t in plex_tracks if not _decade_banned(t, exclude_decades)
                ]
            if exclude_compilations:
                plex_tracks = [t for t in plex_tracks if not _is_compilation_track(t)]
            if rated_only or unrated_only or max_rating > 0:
                plex_tracks = [
                    t
                    for t in plex_tracks
                    if not _rating_mismatched(t, rated_only, unrated_only, max_rating)
                ]
            if played_after or played_before:
                plex_tracks = [
                    t
                    for t in plex_tracks
                    if not _played_outside_window(t, played_after, played_before)
                ]

            if artists:
                plex_tracks = [
                    t for t in plex_tracks
                    if _matches_artist(t.grandparentTitle or "", artists)
                ]
            if exclude_artists:
                plex_tracks = [
                    t for t in plex_tracks if not _banned_artist_hit(t, exclude_artists)
                ]
            if exclude_genres:
                dropped = [g.lower() for g in exclude_genres]
                plex_tracks = [
                    t for t in plex_tracks
                    if not any(g in dropped for g in _track_genres(t))
                ]
            if min_duration_ms > 0:
                plex_tracks = [
                    t for t in plex_tracks if (getattr(t, "duration", None) or 0) >= min_duration_ms
                ]
            if max_duration_ms > 0:
                plex_tracks = [
                    t for t in plex_tracks
                    if 0 < (getattr(t, "duration", None) or 0) <= max_duration_ms
                ]

            return len(plex_tracks)
        except Exception as e:
            logger.exception("Failed to count tracks with filters: %s", e)
            return -1

    def _build_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        min_rating: int = 0,
        min_year: int = 0,
        max_year: int = 0,
        min_plays: int = 0,
        max_plays: int = 0,
        unheard_only: bool = False,
    ) -> dict[str, Any]:
        """Build Plex filter kwargs from filter parameters.

        A recording-year window and a play-count window go server-side because
        Plex's library section exposes both (``year`` and ``viewCount``, the
        latter supporting the ``>=``/``<=`` operators). An exact single year uses
        the plain ``year`` filter, and "never played" is ``viewCount=0`` rather
        than a ceiling because 0 means unbounded in this codebase.

        Args:
            genres: List of genre names to include
            decades: List of decades (e.g., ["1990s", "2000s"])
            min_rating: Minimum user rating (0-10, 0 = no filter)
            min_year: Oldest acceptable release year (0 = no lower bound)
            max_year: Newest acceptable release year (0 = no upper bound)
            min_plays: Minimum Plex play count (0 = no floor)
            max_plays: Maximum Plex play count (0 = no ceiling)
            unheard_only: Require a zero play count ("hidden gems")

        Returns:
            Dict of filter kwargs for Plex search
        """
        filters = {}

        if genres:
            filters['genre'] = genres

        if decades:
            # Convert decades like "1980s" to decade values "1980"
            decade_values = []
            for d in decades:
                if d.endswith('s'):
                    decade_values.append(d[:-1])
                else:
                    decade_values.append(d)
            if decade_values:
                filters['decade'] = decade_values

        if min_rating > 0:
            filters['userRating>>='] = min_rating

        if min_year > 0 and min_year == max_year:
            filters['year'] = str(min_year)
        else:
            if min_year > 0:
                filters['year>='] = str(min_year)
            if max_year > 0:
                filters['year<='] = str(max_year)

        if unheard_only:
            filters['viewCount'] = 0
        elif min_plays > 0:
            filters['viewCount>='] = str(min_plays)

        if not unheard_only and max_plays > 0:
            filters['viewCount<='] = str(max_plays)

        return filters

    def get_genres(self) -> list[dict[str, Any]]:
        """Get list of genres with track counts."""
        stats = self.get_library_stats()
        return stats.get("genres", [])

    def get_decades(self) -> list[dict[str, Any]]:
        """Get list of decades with track counts."""
        stats = self.get_library_stats()
        return stats.get("decades", [])

    def get_all_artists(
        self,
        limit: int = MAX_ARTIST_OPTIONS,
        exclude_live: bool = True,
    ) -> list[dict[str, Any]]:
        """Artist roster for the live-Plex fallback path.

        Plex has no per-artist track count on the section endpoint, so every
        entry carries ``count=None``; the local cache path
        (:meth:`library_cache.get_cached_artists`) is the one that reports
        counts. Names come back in Plex's own order, capped at *limit*.

        Args:
            limit: Maximum artists to return (0 = no limit).
            exclude_live: Accepted for call-site symmetry; Plex's artist hub
                          already aggregates live and studio recordings.

        Returns:
            List of ``{'name': str, 'count': None}`` dicts, or an empty list
            when the query fails.
        """
        if not self._library:
            return []

        try:
            kwargs = {"maxresults": limit} if limit > 0 else {}
            artists = self._library.searchArtists(**kwargs)
            roster = [{"name": a.title, "count": None} for a in artists if a.title]
            roster.sort(key=lambda item: item["name"].lower())
            return roster
        except Exception as e:
            logger.exception("Failed to query Plex artist roster: %s", e)
            return []

    def get_duration_stats(
        self,
        exclude_live: bool = True,
        sample_size: int = DURATION_SAMPLE_SIZE,
    ) -> dict[str, Any]:
        """Track-length distribution from a random sample of the library.

        Plex does not expose duration percentiles (and its track-level duration
        filter is not offered by every server), so this reads a bounded random
        sample. The local cache path computes the same shape exactly.

        Args:
            exclude_live: Drop tracks that look like live recordings.
            sample_size: How many random tracks to measure.

        Returns:
            Dict with ``total``, ``min_ms``, ``p25_ms``, ``p50_ms``, ``p75_ms``,
            ``max_ms`` (all 0 when the sample produced no durations).
        """
        if not self._library:
            return dict(_EMPTY_DURATION_STATS)

        try:
            total = self._library.totalViewSize(libtype="track")
            fetch_count = int(sample_size * 1.3) if exclude_live else sample_size
            plex_tracks = self._library.search(
                libtype="track",
                sort="random",
                limit=fetch_count,
            )
            durations = []
            for t in plex_tracks:
                if exclude_live and is_live_version(t):
                    continue
                ms = getattr(t, "duration", None) or 0
                if ms > 0:
                    durations.append(int(ms))

            if not durations:
                return dict(_EMPTY_DURATION_STATS)

            durations.sort()
            return {
                "total": int(total or len(durations)),
                "min_ms": durations[0],
                "p25_ms": _sample_percentile(durations, 0.25),
                "p50_ms": _sample_percentile(durations, 0.50),
                "p75_ms": _sample_percentile(durations, 0.75),
                "max_ms": durations[-1],
            }
        except Exception as e:
            logger.exception("Failed to measure Plex track durations: %s", e)
            return dict(_EMPTY_DURATION_STATS)

    def get_random_tracks(
        self,
        count: int,
        exclude_live: bool = True,
    ) -> list[Track]:
        """Get random tracks from the library without loading all tracks.

        Uses Plex's random sort with limit for efficient sampling.

        Args:
            count: Number of random tracks to fetch
            exclude_live: Whether to exclude live recordings

        Returns:
            List of random Track objects
        """
        if not self._library:
            return []

        try:
            # Fetch more than needed to account for live version filtering
            fetch_count = int(count * 1.3) if exclude_live else count

            plex_tracks = self._library.search(
                libtype="track",
                sort="random",
                limit=fetch_count,
            )

            if exclude_live:
                plex_tracks = [t for t in plex_tracks if not is_live_version(t)]

            tracks = [self._convert_track(t) for t in plex_tracks[:count]]
            return tracks
        except Exception as e:
            logger.exception("Failed to get random tracks: %s", e)
            raise PlexQueryError(f"Failed to get random tracks: {e}") from e

    def search_tracks(self, query: str, limit: int = 20) -> list[Track]:
        """Search for tracks by title or artist.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching Track objects
        """
        if not self._library:
            return []

        try:
            # Search tracks
            results = self._library.searchTracks(title=query, limit=limit)

            # Also search by artist if we have few results
            if len(results) < limit:
                artist_results = self._library.search(libtype="track", limit=limit)
                # Filter by artist name
                artist_matches = [
                    t for t in artist_results
                    if query.lower() in (t.grandparentTitle or "").lower()
                ]
                # Combine and deduplicate
                seen_keys = {t.ratingKey for t in results}
                for t in artist_matches:
                    if t.ratingKey not in seen_keys:
                        results.append(t)
                        seen_keys.add(t.ratingKey)

            return [self._convert_track(t) for t in results[:limit]]
        except Exception:
            return []

    def get_track_by_key(self, rating_key: str) -> Track | None:
        """Get a single track by rating key.

        Args:
            rating_key: Plex rating key

        Returns:
            Track object or None if not found
        """
        if not self._server:
            return None

        try:
            item = self._server.fetchItem(int(rating_key))
            return self._convert_track(item)
        except Exception:
            return None

    def get_thumb_path(self, rating_key: str) -> str | None:
        """Get the raw Plex thumb path for a track.

        Walks the Plex hierarchy: track thumb → album thumb → artist thumb.
        Compilation/soundtrack albums often lack track and album art, so
        falling back to the artist thumb avoids blank covers.

        Args:
            rating_key: Plex rating key

        Returns:
            Thumb path (e.g., '/library/metadata/123/thumb/456') or None
        """
        if not self._server:
            return None

        try:
            item = self._server.fetchItem(int(rating_key))
            return (
                getattr(item, "thumb", None)
                or getattr(item, "parentThumb", None)
                or getattr(item, "grandparentThumb", None)
            )
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Plex user resolution (playlists owned by other Plex accounts)
    # -------------------------------------------------------------------------

    def _get_account(self, refresh: bool = False) -> Any:
        """Return the MyPlexAccount for the configured token (cached).

        Returns None when not connected. Raises if the token cannot reach plex.tv.
        """
        if not self._server:
            return None

        with self._user_lock:
            if self._account is None or refresh:
                self._account = self._server.myPlexAccount()
                self._users_by_key = {}
                self._user_servers.clear()
            return self._account

    @staticmethod
    def _user_key(user: Any) -> str:
        """Stable key for a Plex user (home users have no username, only a title)."""
        key = (
            getattr(user, "username", None)
            or getattr(user, "title", None)
            or str(getattr(user, "id", "") or "")
        )
        return key.strip()

    @staticmethod
    def _is_managed(user: Any) -> bool:
        """True for Plex managed (kids) profiles, which cannot own playlists."""
        if getattr(user, "managed", False):
            return True
        data = getattr(user, "_data", None)
        profile = data.get("restrictionProfile") if hasattr(data, "get") else None
        return bool(profile)

    def _load_users(self, refresh: bool = False) -> dict[str, Any]:
        """Map user key -> Plex user object for accounts that can use this server."""
        with self._user_lock:
            cached = self._users_by_key
        if cached and not refresh:
            return cached

        account = self._get_account(refresh=refresh)
        if account is None:
            return {}

        machine_id = self.get_machine_identifier()
        users: dict[str, Any] = {}
        for user in account.users():
            servers = getattr(user, "servers", None) or []
            if not any(getattr(s, "machineIdentifier", None) == machine_id for s in servers):
                continue
            key = self._user_key(user)
            if key:
                users[key] = user

        with self._user_lock:
            self._users_by_key = users
        return users

    def get_users(self) -> list[PlexUserInfo]:
        """List Plex accounts that can own playlists (owner first, then alphabetical)."""
        if not self._server:
            return []

        owner_title = "Owner"
        try:
            account = self._get_account()
            if account is not None:
                owner_title = (
                    getattr(account, "title", None)
                    or getattr(account, "username", None)
                    or "Owner"
                )
        except Exception as e:
            logger.warning("Failed to read Plex account (user list limited to owner): %s", e)

        owner = PlexUserInfo(key=OWNER_USER_KEY, title=owner_title, is_owner=True)
        others: list[PlexUserInfo] = []
        try:
            for key, user in self._load_users().items():
                others.append(
                    PlexUserInfo(
                        key=key,
                        title=getattr(user, "title", None) or key,
                        user_id=getattr(user, "id", None),
                        is_home=bool(getattr(user, "home", False)),
                        is_managed=self._is_managed(user),
                        has_pin=bool(getattr(user, "protected", False)),
                    )
                )
        except Exception as e:
            logger.warning("Failed to list Plex users: %s", e)

        others.sort(key=lambda u: u.title.lower())
        return [owner] + others

    def _server_for_user(self, user: str | None, refresh: bool = False) -> Any:
        """Return a Plex connection acting as `user` (OWNER_USER_KEY = our own token).

        Raises:
            PlexQueryError: if the user is unknown or the login as that user fails.
        """
        if not self._server:
            raise PlexQueryError("Not connected to Plex")

        key = (user or OWNER_USER_KEY).strip() or OWNER_USER_KEY
        if key == OWNER_USER_KEY:
            return self._server

        cached = self._user_servers.get(key)
        if cached and not refresh and time.time() - cached[1] < self.USER_SERVER_TTL:
            return cached[0]

        plex_user = self._load_users(refresh=refresh).get(key)
        if plex_user is None:
            raise PlexQueryError(
                f"Unknown Plex user '{key}'. Reload the user list — "
                "only accounts shared with this server can own playlists."
            )

        try:
            # Admin-only: logs in to the same server with that user's token
            server = self._server.switchUser(plex_user)
        except Exception as e:
            logger.exception("Failed to connect to Plex as '%s'", key)
            raise PlexQueryError(f"Could not connect to Plex as '{key}': {e}") from e

        with self._user_lock:
            self._user_servers[key] = (server, time.time())
        return server

    def _account_id(self, server: Any) -> int | None:
        """Plex account id behind a connection, used to spot duplicate targets."""
        account = getattr(server, "account", None)
        account_id = getattr(account, "id", None)
        return account_id if isinstance(account_id, int) else None

    def _account_for_key(self, key: str) -> int | None:
        """Plex account id a user key resolves to, for spotting one account twice.

        `PlexServer.account` is a method, not the owning account, so a connection
        cannot say which account it was made for; the plex.tv user objects can.
        """
        try:
            account = self._get_account() if key == OWNER_USER_KEY else self._load_users().get(key)
        except Exception as e:
            logger.warning("Cannot resolve account id for '%s': %s", key, e)
            return None
        if account is None:
            return None
        account_id = getattr(account, "id", None)
        return account_id if isinstance(account_id, int) else None

    def _owner_label(self, user: str | None) -> str:
        """Human-readable name of the account that owns a playlist."""
        key = (user or OWNER_USER_KEY).strip() or OWNER_USER_KEY
        if key == OWNER_USER_KEY:
            try:
                account = self._get_account()
                return (
                    getattr(account, "title", None)
                    or getattr(account, "username", None)
                    or "Owner"
                )
            except Exception:
                return "Owner"

        plex_user = self._load_users().get(key)
        if plex_user is not None:
            return getattr(plex_user, "title", None) or key
        return key

    def collect_user_ratings(self, user: str | None = None) -> list[tuple[str, int]]:
        """Every (rating_key, stars) this Plex account has given the music library.

        Star ratings are per account: a household member's ratings exist only on
        the connection made as that member (admin ``switchUser``, cached by
        :meth:`_server_for_user`), never on the admin's own walk, which is why
        :meth:`get_all_raw_tracks` cannot see them. The scan is the same one
        :meth:`get_all_raw_tracks` pays for (30-60s on a 50k-track library), so
        callers cache the result in the library cache rather than rescanning.

        Raises:
            PlexQueryError: the account is unknown, the login as it fails, or the
                library scan fails (an empty list would otherwise read as "this
                account rated nothing at all", which clears a member's filters).
        """
        server = self._server_for_user(user)
        try:
            section = server.library.section(self.music_library_name)
            tracks = section.search(libtype="track")
        except Exception as e:
            logger.exception("Failed to scan the library for account ratings")
            raise PlexQueryError(f"Could not scan ratings for this account: {e}") from e

        rated: list[tuple[str, int]] = []
        for track in tracks:
            stars = getattr(track, "userRating", None)
            if stars is None:
                continue
            try:
                # plexapi hands the field back as a float-ish value.
                value = int(round(float(stars)))
            except (TypeError, ValueError):
                continue
            if value > 0:
                rated.append((str(track.ratingKey), value))
        return rated

    def _create_playlist_on(
        self, server: Any, name: str, rating_keys: list[str], description: str
    ) -> dict[str, Any]:
        """Fetch tracks and create a playlist on one server connection."""
        items = []
        skipped_keys = []
        for key in rating_keys:
            try:
                item = server.fetchItem(int(key))
                items.append(item)
            except Exception as e:
                logger.warning("Failed to fetch track %s for playlist: %s", key, e)
                skipped_keys.append(key)

        if skipped_keys:
            logger.info(
                "Playlist '%s': skipped %d of %d tracks",
                name,
                len(skipped_keys),
                len(rating_keys),
            )

        if not items:
            return {
                "success": False,
                "error": "No valid tracks found",
                "tracks_added": 0,
                "tracks_skipped": len(skipped_keys),
            }

        playlist = server.createPlaylist(name, items=items)

        # Set description/summary if provided
        if description:
            try:
                playlist.edit(summary=description)
                logger.info("Set playlist description: %d chars", len(description))
            except Exception as e:
                logger.warning("Failed to set playlist description: %s", e)

        return {
            "success": True,
            "playlist_id": str(playlist.ratingKey),
            "playlist_url": self._build_playlist_url(playlist.ratingKey),
            "tracks_added": len(items),
            "tracks_skipped": len(skipped_keys),
        }

    def _create_copies(
        self,
        name: str,
        rating_keys: list[str],
        description: str,
        copy_to: list[str] | None,
        primary_server: Any = None,
    ) -> list[dict[str, Any]]:
        """Create the same playlist inside each additional user's own account.

        Blank keys, the owner placeholder, and any key that resolves to the same
        account the main playlist was written to are skipped (a copy there would
        only duplicate the playlist that was just created).
        """
        copies: list[dict[str, Any]] = []
        primary_account = self._account_id(primary_server)

        for key in copy_to or []:
            key = (key or "").strip()
            if not key or key == OWNER_USER_KEY:
                continue

            label = self._owner_label(key)
            try:
                server = self._server_for_user(key)
                if primary_account is not None and self._account_id(server) == primary_account:
                    logger.info("Skipping redundant copy of '%s' for '%s'", name, label)
                    continue
                copy = self._create_playlist_on(server, name, rating_keys, description)
            except PlexQueryError as e:
                copies.append({"user": label, "success": False, "error": str(e)})
                continue
            except Exception as e:
                logger.exception("Failed to copy playlist '%s' to '%s'", name, key)
                copies.append({"user": label, "success": False, "error": str(e)})
                continue
            copies.append({"user": label, **copy})
        return copies

    def create_playlist(
        self,
        name: str,
        rating_keys: list[str],
        description: str = "",
        user: str | None = None,
        copy_to: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a playlist in Plex, optionally owned by another Plex user.

        Args:
            name: Playlist name
            rating_keys: List of track rating keys
            description: Playlist description/summary (optional)
            user: Key of the user who should own the playlist (None/'__owner__' = our account)
            copy_to: Extra user keys that each get their own copy of the playlist

        Returns:
            Dict with success status, playlist_id, owner name, and per-user copies
        """
        if not self._server:
            return {"success": False, "error": "Not connected to Plex"}

        try:
            server = self._server_for_user(user)
        except PlexQueryError as e:
            return {"success": False, "error": str(e)}

        try:
            result = self._create_playlist_on(server, name, rating_keys, description)
        except Exception as e:
            logger.exception("Failed to create playlist '%s'", name)
            result = {"success": False, "error": str(e)}

        result["owner"] = self._owner_label(user)
        result["copies"] = self._create_copies(
            name, rating_keys, description, copy_to, primary_server=server
        )
        return result

    def move_playlist(
        self,
        playlist_id: str,
        to_user: str,
        from_user: str | None = None,
        delete_source: bool = False,
        name: str = "",
        description: str | None = None,
        copy_to: list[str] | None = None,
    ) -> dict[str, Any]:
        """Recreate a playlist inside another Plex user's account (Home admin only).

        Plex's "Grant Access" (Single Item Access) does hand a playlist to another account, but that
        grant is made at plex.tv, only the Home admin can make it, and the receiver only *sees* the
        result: the PMS REST API has no route for it (`/playlists/{id}/users` 404s) and plexapi has no
        client for it. So the source playlist's tracks are re-fetched and written as a new playlist
        that the destination owns and can edit. `delete_source`
        False keeps the original (a copy), True removes it (a move). Other household
        accounts need no token of their own: the admin connection logs in to the same
        server as that member (`switchUser`), which only the Plex Home admin can do.

        Args:
            playlist_id: Rating key of the source playlist, or its exact Plex title
            to_user: Key of the account that should own the new playlist
            from_user: Key of the account that owns it now (None/'__owner__' = our own)
            delete_source: True deletes the original once the copy exists
            name: Name for the new playlist (default: the source playlist's title)
            description: Summary for the new playlist (default: the source's summary)
            copy_to: Extra user keys that each get their own copy as well

        Returns:
            Dict with success, from, owner, playlist_id, playlist_url, tracks_added,
            tracks_skipped, source_deleted, copies and error. A failed `delete_source`
            leaves success True with source_deleted False and the reason in error.
        """
        if not self._server:
            return {"success": False, "error": "Not connected to Plex"}

        base = {
            "success": False,
            "from": self._owner_label(from_user),
            "owner": self._owner_label(to_user),
            "playlist_id": None,
            "playlist_url": None,
            "tracks_added": None,
            "tracks_skipped": None,
            "source_deleted": False,
            "copies": [],
            "error": None,
        }

        query = (playlist_id or "").strip()
        if not query:
            return {**base, "error": "No source playlist given"}

        src_key = (from_user or OWNER_USER_KEY).strip() or OWNER_USER_KEY
        dst_key = (to_user or "").strip() or OWNER_USER_KEY
        if src_key == dst_key:
            return {**base, "error": "Source and destination are the same Plex account"}

        try:
            src_server = self._server_for_user(src_key)
            dst_server = self._server_for_user(dst_key)
        except PlexQueryError as e:
            return {**base, "error": str(e)}

        src_id = self._account_for_key(src_key)
        if src_id is not None and src_id == self._account_for_key(dst_key):
            return {**base, "error": "Both keys resolve to the same Plex account"}

        source = None
        if query.isdigit():
            try:
                source = src_server.fetchItem(int(query))
            except Exception as e:
                return {
                    **base,
                    "error": f"Rating key {query} is not a playlist of {base['from']}: {e}",
                }
        else:
            try:
                siblings = src_server.playlists()
            except Exception as e:
                logger.exception("Failed to list playlists while moving '%s'", query)
                return {
                    **base,
                    "error": f"Could not list playlists of {base['from']}: {e}",
                }
            matches = [p for p in siblings if p.title == query]
            if len(matches) > 1:
                found = sorted(str(p.ratingKey) for p in matches)
                return {
                    **base,
                    "error": f"Several playlists are named '{query}'; pass one rating key in {found}",
                }
            if not matches:
                titles = sorted(p.title for p in siblings)
                return {
                    **base,
                    "error": f"No playlist named {query!r} belongs to {base['from']} "
                    f"(Plex spells apostrophes as U+2019): {titles}",
                }
            source = matches[0]

        try:
            leaves = source.items()
        except Exception as e:
            return {**base, "error": f"Could not read the source playlist: {e}"}
        keys = [str(t.ratingKey) for t in leaves if getattr(t, "type", None) == "track"]

        new_name = name.strip() or source.title
        summary = description if description is not None else (getattr(source, "summary", "") or "")

        made = self._create_playlist_on(dst_server, new_name, keys, summary)
        if not made.get("success"):
            return {**base, "tracks_skipped": made.get("tracks_skipped"),
                    "error": made.get("error") or "No tracks could be re-fetched"}

        delete_error = None
        source_deleted = False
        if delete_source:
            try:
                source.delete()
                source_deleted = True
            except Exception as e:
                logger.exception("Failed to delete the source playlist of '%s'", new_name)
                delete_error = f"The copy exists but the original was not deleted: {e}"

        # _create_copies compares connections, which cannot name their account on
        # plexapi 4.18, so drop a copy that would land beside the one just made.
        extras = [k for k in copy_to or [] if ((k or "").strip() or OWNER_USER_KEY) != dst_key]

        return {
            **base,
            "success": True,
            "playlist_id": made["playlist_id"],
            "playlist_url": self._build_playlist_url(made["playlist_id"]),
            "tracks_added": made["tracks_added"],
            "tracks_skipped": made["tracks_skipped"],
            "source_deleted": source_deleted,
            "copies": self._create_copies(new_name, keys, summary, extras, primary_server=dst_server),
            "error": delete_error,
        }

    _MOBILE_KEYWORDS = {"ios", "android", "iphone", "ipad", "ipod", "tvos"}

    @classmethod
    def _is_mobile_client(cls, product: str, platform: str) -> bool:
        """Check if a client is a mobile/TV device that needs an active session."""
        tokens = set(re.split(r"[\s,/]+", f"{product} {platform}".lower()))
        return bool(tokens & cls._MOBILE_KEYWORDS)

    def get_clients(self) -> list[PlexClientInfo]:
        """Get online Plex clients capable of playback.

        Discovers clients via two methods:
        1. server.clients() — local GDM-discovered clients
        2. myPlexAccount().resources() — cloud-connected players (Plexamp, etc.)

        Returns:
            List of PlexClientInfo for clients with playback capability.
            Unresponsive clients are silently excluded.
        """
        if not self._server:
            return []

        result = []
        seen_ids: set[str] = set()

        # Method 1: Local GDM discovery
        try:
            local_clients = self._server.clients()
        except Exception as e:
            logger.warning("Failed to get local Plex clients: %s", e)
            local_clients = []

        for client in local_clients:
            capabilities = getattr(client, "protocolCapabilities", None) or []
            if isinstance(capabilities, str):
                capabilities = [c.strip() for c in capabilities.split(",")]
            if "playback" not in capabilities:
                continue

            try:
                is_playing = client.isPlayingMedia(includePaused=True)
            except Exception:
                logger.warning("Client '%s' unresponsive, skipping", getattr(client, "title", "unknown"))
                continue

            mid = client.machineIdentifier
            seen_ids.add(mid)
            result.append(PlexClientInfo(
                client_id=mid,
                name=client.title,
                product=client.product,
                platform=client.platform,
                is_playing=is_playing,
                is_mobile=self._is_mobile_client(client.product, client.platform),
            ))

        local_count = len(result)

        # Method 2: Cloud-connected players via account resources
        try:
            account = self._server.myPlexAccount()
            resources = account.resources()

            # Pre-fetch sessions once for all cloud resources
            playing_ids: set[str] = set()
            try:
                for session in self._server.sessions():
                    if session.player:
                        playing_ids.add(session.player.machineIdentifier)
            except Exception:
                pass

            for resource in resources:
                provides = getattr(resource, "provides", "") or ""
                if "player" not in provides:
                    continue
                # Skip if already found via local discovery
                client_id = resource.clientIdentifier
                if client_id in seen_ids:
                    continue
                # Only include online resources
                if not getattr(resource, "presence", False):
                    continue

                seen_ids.add(client_id)
                product = getattr(resource, "product", "Unknown")
                platform = getattr(resource, "platform", "Unknown") or getattr(resource, "platformVersion", "Unknown")
                result.append(PlexClientInfo(
                    client_id=client_id,
                    name=resource.name,
                    product=product,
                    platform=platform,
                    is_playing=client_id in playing_ids,
                    is_mobile=self._is_mobile_client(product, platform),
                ))
        except Exception as e:
            logger.warning("Failed to query account resources for players: %s", e)

        logger.debug("Client discovery: %d found (%d local, %d cloud)", len(result), local_count, len(result) - local_count)
        return result

    def get_playlists(self, user: str | None = None) -> list[PlexPlaylistInfo]:
        """Get audio playlists from the Plex server.

        Args:
            user: Key of the Plex user whose playlists to list (None = our account)

        Returns:
            List of PlexPlaylistInfo sorted alphabetically by title.
        """
        try:
            server = self._server_for_user(user)
        except PlexQueryError as e:
            logger.warning("Cannot list playlists: %s", e)
            return []

        try:
            playlists = server.playlists(playlistType="audio")
            result = [
                PlexPlaylistInfo(
                    rating_key=str(playlist.ratingKey),
                    title=playlist.title,
                    track_count=playlist.leafCount,
                )
                for playlist in playlists
                if not playlist.smart and not playlist.radio
            ]
            return sorted(result, key=lambda p: p.title.lower())
        except Exception as e:
            logger.exception("Failed to get playlists: %s", e)
            return []

    def update_playlist(
        self,
        playlist_id: str,
        rating_keys: list[str],
        mode: str = "replace",
        description: str = "",
        user: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing Plex playlist by replacing or appending tracks.

        Handles the __scratch__ sentinel for auto-creating "MediaSage - Now Playing".

        Args:
            playlist_id: Rating key of target playlist, or "__scratch__" sentinel
            rating_keys: List of track rating keys to add
            mode: 'replace' or 'append'
            description: Optional playlist description/summary
            user: Key of the user who owns the playlist (None/'__owner__' = our account)

        Returns:
            Dict with success, tracks_added, tracks_skipped, duplicates_skipped,
            playlist_url, owner, error
        """
        try:
            server = self._server_for_user(user)
        except PlexQueryError as e:
            return {"success": False, "error": str(e)}
        owner = self._owner_label(user)

        try:
            # Handle __scratch__ sentinel — find or create "MediaSage - Now Playing"
            if playlist_id == "__scratch__":
                with self._scratch_lock:
                    scratch_playlist = None
                    try:
                        for p in server.playlists(playlistType="audio"):
                            if p.title == "MediaSage - Now Playing":
                                scratch_playlist = p
                                break
                    except Exception as e:
                        logger.warning("Failed to search for scratch playlist: %s", e)

                    if scratch_playlist is None:
                        # Create new scratch playlist with the provided tracks
                        items = []
                        tracks_skipped = 0
                        for key in rating_keys:
                            try:
                                items.append(server.fetchItem(int(key)))
                            except Exception:
                                tracks_skipped += 1

                        if not items:
                            return {
                                "success": False,
                                "error": "No valid tracks found",
                                "owner": owner,
                            }

                        playlist = server.createPlaylist(
                            "MediaSage - Now Playing", items=items
                        )

                        if description:
                            try:
                                playlist.edit(summary=description)
                            except Exception as e:
                                logger.warning("Failed to set playlist description: %s", e)

                        playlist_url = self._build_playlist_url(playlist.ratingKey)
                        return {
                            "success": True,
                            "tracks_added": len(items),
                            "tracks_skipped": tracks_skipped,
                            "duplicates_skipped": 0,
                            "playlist_url": playlist_url,
                            "owner": owner,
                        }
                    else:
                        # Use the existing scratch playlist
                        playlist_id = str(scratch_playlist.ratingKey)

            # Fetch the target playlist
            playlist = server.fetchItem(int(playlist_id))

            # Fetch new track objects
            items = []
            tracks_skipped = 0
            duplicates_skipped = 0

            if mode == "replace":
                # Fetch new items FIRST to avoid data loss if add fails
                for key in rating_keys:
                    try:
                        items.append(server.fetchItem(int(key)))
                    except Exception:
                        tracks_skipped += 1

                if not items:
                    return {
                        "success": False,
                        "error": "No valid tracks found to replace with",
                        "tracks_added": 0,
                        "tracks_skipped": tracks_skipped,
                        "duplicates_skipped": 0,
                        "owner": owner,
                    }

                # Add new items FIRST, then remove old ones.
                # This avoids data loss: if addItems fails, old tracks remain.
                # If removeItems fails, user has duplicates (recoverable) not emptiness.
                existing_items = playlist.items()
                playlist.addItems(items)
                warning = None
                if existing_items:
                    try:
                        playlist.removeItems(existing_items)
                    except Exception as e:
                        logger.warning("Failed to remove old items during replace: %s", e)
                        warning = "Replaced tracks were added but old tracks could not be removed. Playlist may contain duplicates."

            elif mode == "append":
                warning = None
                # Build set of existing rating keys for deduplication
                existing_keys = {
                    str(item.ratingKey) for item in playlist.items()
                }

                for key in rating_keys:
                    if key in existing_keys:
                        duplicates_skipped += 1
                        continue
                    try:
                        items.append(server.fetchItem(int(key)))
                    except Exception:
                        tracks_skipped += 1

                if items:
                    playlist.addItems(items)

            else:
                return {"success": False, "error": f"Unknown update mode: {mode}", "owner": owner}

            # Update description if provided
            if description:
                try:
                    playlist.edit(summary=description)
                except Exception as e:
                    logger.warning("Failed to set playlist description: %s", e)

            playlist_url = self._build_playlist_url(playlist.ratingKey)
            result = {
                "success": True,
                "tracks_added": len(items),
                "tracks_skipped": tracks_skipped,
                "duplicates_skipped": duplicates_skipped,
                "playlist_url": playlist_url,
                "owner": owner,
            }
            if warning:
                result["warning"] = warning
            return result
        except Exception as e:
            logger.exception("Failed to update playlist '%s'", playlist_id)
            return {"success": False, "error": str(e), "owner": owner}

    def _build_playlist_url(self, rating_key: int) -> str | None:
        """Build the Plex web app URL for a playlist."""
        machine_id = self.get_machine_identifier()
        if not machine_id:
            return None
        return (
            f"{self.url}/web/index.html#!/server/{machine_id}"
            f"/playlist?key=%2Fplaylists%2F{rating_key}"
        )

    def play_queue(
        self, rating_keys: list[str], client_id: str, mode: str = "replace"
    ) -> dict[str, Any]:
        """Create a play queue and start playback on a Plex client.

        Args:
            rating_keys: List of track rating keys
            client_id: machineIdentifier of target client
            mode: 'replace' (new queue) or 'play_next' (add after current)

        Returns:
            Dict with success, client_name, client_product, tracks_queued, error
        """
        if not self._server:
            return {"success": False, "error": "Not connected to Plex"}

        # Find the target client — try local GDM first, then cloud resources
        target_client = None
        try:
            for c in self._server.clients():
                if c.machineIdentifier == client_id:
                    target_client = c
                    break
        except Exception:
            pass

        # Fall back to cloud-connected resource
        if not target_client:
            try:
                account = self._server.myPlexAccount()
                for resource in account.resources():
                    if resource.clientIdentifier == client_id:
                        target_client = resource.connect()
                        break
            except Exception as e:
                logger.warning("Failed to connect to cloud client %s: %s", client_id, e)

        if not target_client:
            return {
                "success": False,
                "error": "Device couldn't be reached. Try starting playback on the device first, then re-open the picker.",
                "error_code": "not_found",
            }

        # Proxy through server for reliable communication
        try:
            target_client.proxyThroughServer(value=True)
        except Exception:
            pass  # Best effort — some clients don't support this

        # Fetch track objects
        tracks = []
        tracks_skipped = 0
        for key in rating_keys:
            try:
                item = self._server.fetchItem(int(key))
                tracks.append(item)
            except Exception as e:
                tracks_skipped += 1
                logger.warning("Failed to fetch track %s for play queue: %s", key, e)

        if not tracks:
            return {"success": False, "error": "No valid tracks found"}

        try:
            if mode == "replace":
                play_queue = PlayQueue.create(
                    self._server,
                    items=tracks,
                    startItem=tracks[0],
                    includeRelated=0,
                )
                target_client.playMedia(play_queue)
            elif mode == "play_next":
                # Get current play queue from client timeline entries
                # timelines() returns a list; timeline is a single object/None
                play_queue_id = None
                try:
                    for entry in target_client.timelines():
                        if entry.type == "music" and getattr(entry, "playQueueID", None):
                            play_queue_id = entry.playQueueID
                            break
                except Exception as e:
                    logger.warning("Failed to get timelines from client %s: %s", client_id, e)
                    return {"success": False, "error": "Could not read active queue from client"}
                if not play_queue_id:
                    return {
                        "success": False,
                        "error": "No active play queue on this client",
                    }
                existing_queue = PlayQueue.get(
                    self._server, play_queue_id, own=True
                )
                # Add in reverse order so tracks play in intended order
                tracks_queued = 0
                reversed_tracks = list(reversed(tracks))
                for i, track in enumerate(reversed_tracks):
                    try:
                        is_last = i == len(reversed_tracks) - 1
                        existing_queue.addItem(track, playNext=True, refresh=is_last)
                        tracks_queued += 1
                    except Exception:
                        logger.warning("Failed to add track %s to queue", track.ratingKey)

            else:
                return {"success": False, "error": f"Unknown play queue mode: {mode}"}

            actual_queued = tracks_queued if mode == "play_next" else len(tracks)
            return {
                "success": True,
                "client_name": target_client.title,
                "client_product": target_client.product,
                "tracks_queued": actual_queued,
                "tracks_skipped": tracks_skipped,
            }
        except ConnectionError:
            return {
                "success": False,
                "error": f"Client '{target_client.title}' went offline during playback",
            }
        except Exception as e:
            logger.exception("Failed to create play queue on '%s'", target_client.title)
            return {"success": False, "error": str(e)}

    def _convert_track(self, plex_track: Any) -> Track:
        """Convert a Plex track object to our Track model."""
        # Get genres
        genres = []
        if hasattr(plex_track, "genres"):
            genres = [
                g.tag if hasattr(g, "tag") else str(g)
                for g in plex_track.genres
            ]

        # Get year from album or track
        year = getattr(plex_track, "parentYear", None) or getattr(plex_track, "year", None)

        # Build art URL (will be proxied through our API)
        art_url = f"/api/art/{plex_track.ratingKey}" if plex_track.ratingKey else None

        return Track(
            rating_key=str(plex_track.ratingKey),
            title=plex_track.title,
            artist=plex_track.grandparentTitle or "Unknown Artist",
            album=plex_track.parentTitle or "Unknown Album",
            duration_ms=plex_track.duration or 0,
            year=year,
            genres=genres,
            art_url=art_url,
        )


# Global client instance
_plex_client: PlexClient | None = None


def get_plex_client() -> PlexClient | None:
    """Get the current Plex client instance."""
    return _plex_client


def init_plex_client(url: str, token: str, music_library: str = "Music") -> PlexClient:
    """Initialize or reinitialize the Plex client."""
    global _plex_client
    _plex_client = PlexClient(url, token, music_library)
    return _plex_client
