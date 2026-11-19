"""Pydantic models for MediaSage API contracts and internal data structures."""

from typing import Literal

import re

from pydantic import BaseModel, Field, field_validator, model_validator

# Sentinel user key meaning "the Plex account MediaSage itself is connected as"
OWNER_USER_KEY = "__owner__"

# A last-heard window is phrased as a bare ISO date so it lines up with the
# library cache's ``last_viewed_at`` column, which is an ISO timestamp that the
# filter compares by whole day ("2026-03-02T07:57:35" → "2026-03-02").
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _iso_date_error(value: str, field: str) -> str | None:
    """Error text when *value* is not a real ``YYYY-MM-DD`` date; "" means unset."""
    if not value:
        return None
    if not ISO_DATE_RE.fullmatch(value):
        return f"{field} must be an ISO date (YYYY-MM-DD)"
    month, day = int(value[5:7]), int(value[8:10])
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return f"{field} is not a real calendar date"
    return None


def _familiarity_conflict(filters) -> str | None:
    """Contradiction message when a request's familiarity facets fight each other.

    Shared by :class:`FilterPreviewRequest` and :class:`GenerateRequest`, which carry
    the same filter fields. ``live_only`` is deliberately absent here: both filter
    layers treat it as authoritative over ``exclude_live``, so asking for live rows
    alongside the default drop-live flag means "keep the live rows", not nothing.
    """
    for name in ("played_after", "played_before"):
        error = _iso_date_error(getattr(filters, name), name)
        if error:
            return error
    if filters.played_after and filters.played_before and filters.played_after > filters.played_before:
        return "played_after cannot be after played_before"
    # Never-played tracks have no last-heard date to sit inside a window.
    if filters.unheard_only and (filters.played_after or filters.played_before):
        return "unheard_only contradicts a played_after/played_before window"
    if filters.rated_only and filters.unrated_only:
        return "rated_only contradicts unrated_only"
    # An unrated row carries no rating, so it cannot clear a rating floor.
    if filters.unrated_only and filters.min_rating > 0:
        return "unrated_only contradicts a min_rating floor"
    if filters.min_rating and filters.max_rating and filters.min_rating > filters.max_rating:
        return "min_rating cannot exceed max_rating"
    return None


def album_key(artist: str, album: str, lower: bool = True) -> str:
    """Build composite key for artist+album lookups."""
    if lower:
        return f"{artist.lower()}|||{album.lower()}"
    return f"{artist}|||{album}"


# =============================================================================
# Core Entities
# =============================================================================


class Track(BaseModel):
    """A music track from the Plex library."""

    rating_key: str
    title: str
    artist: str
    album: str
    duration_ms: int
    year: int | None = None
    genres: list[str] = []
    is_live: bool | None = None
    art_url: str | None = None

    @property
    def duration_formatted(self) -> str:
        """Return duration as M:SS format."""
        minutes = self.duration_ms // 60000
        seconds = (self.duration_ms % 60000) // 1000
        return f"{minutes}:{seconds:02d}"


class Dimension(BaseModel):
    """A musical dimension identified from a seed track."""

    id: str
    label: str
    description: str
    category: str = ""  # Descriptor family ("mood", "tempo", "instrumentation", ...)


class FilterSet(BaseModel):
    """Filters applied to narrow track selection."""

    genres: list[str] = []
    decades: list[str] = []
    artists: list[str] = []
    exclude_genres: list[str] = []
    exclude_artists: list[str] = []  # "anything but Nickelback" → drop these artists
    exclude_decades: list[str] = []  # "not the 80s" → drop these decades
    min_year: int = 0  # 0 = no floor ("recorded in 1977", "released after 2015")
    max_year: int = 0  # 0 = no ceiling ("before 1980")
    min_plays: int = 0  # 0 = no floor ("most played", "on repeat")
    max_plays: int = 0  # 0 = no ceiling ("not the stuff I blast")
    unheard_only: bool = False  # "hidden gems" → tracks never played
    exclude_compilations: bool = False  # "studio albums only" → drop Various Artists
    track_count: int = 25
    exclude_live: bool = True
    min_rating: int = 0
    min_duration_ms: int = 0
    max_duration_ms: int = 0
    live_only: bool = False  # "only live takes" → keep just the live rows
    rated_only: bool = False  # "music I have rated" → rows carrying a user rating
    unrated_only: bool = False  # "stuff I never judged" → rows without a user rating
    max_rating: int = 0  # 0 = no ceiling; "nothing I gave 9s to" caps Plex's 0-10 scale
    played_after: str = ""  # ISO YYYY-MM-DD last-heard floor ("" = unbounded)
    played_before: str = ""  # ISO YYYY-MM-DD last-heard ceiling ("" = unbounded)

    @field_validator("track_count")
    @classmethod
    def validate_track_count(cls, v: int) -> int:
        if v not in [15, 25, 50, 100]:
            raise ValueError("track_count must be 15, 25, 50, or 100")
        return v


class Playlist(BaseModel):
    """A generated playlist with tracks and metadata."""

    name: str
    tracks: list[Track]
    source_prompt: str | None = None
    seed_track_key: str | None = None
    selected_dimensions: list[str] | None = None

    @property
    def duration_total(self) -> int:
        """Total duration in milliseconds."""
        return sum(t.duration_ms for t in self.tracks)

    @property
    def track_count(self) -> int:
        return len(self.tracks)


# =============================================================================
# Configuration Models
# =============================================================================


class PlexConfig(BaseModel):
    """Plex server connection settings."""

    url: str
    token: str
    music_library: str = "Music"


class LLMConfig(BaseModel):
    """LLM provider settings."""

    provider: Literal["anthropic", "openai", "gemini", "ollama", "custom"]
    api_key: str = ""  # Optional for local providers
    model_analysis: str
    model_generation: str
    smart_generation: bool = False
    # Local provider settings
    ollama_url: str = "http://localhost:11434"
    ollama_context_window: int = 32768  # Detected from model, can be overridden
    custom_url: str = ""
    custom_context_window: int = 32768
    # HTTP behavior for LLM API calls. Local inference servers (LM Studio,
    # MLX, Ollama) can need many minutes to finish one generation, so raise
    # the timeout and set max_retries to 0 for them (a retry re-runs the
    # entire slow generation from scratch).
    request_timeout: float = 600.0
    max_retries: int = 2

    @field_validator("ollama_context_window", "custom_context_window")
    @classmethod
    def validate_context_window(cls, v: int) -> int:
        if v < 512:
            raise ValueError("Context window must be at least 512 tokens")
        if v > 2000000:
            raise ValueError("Context window cannot exceed 2,000,000 tokens")
        return v


class DefaultsConfig(BaseModel):
    """Default values for UI."""

    track_count: int = 25


class SpotifyConfig(BaseModel):
    """The optional Spotify side of playlist import.

    Two fetch paths, picked by what is configured: with ``client_id``/``client_secret``
    (a free Spotify for Developers app pair) the playlist comes fresh from
    api.spotify.com; without them, ``parquet_dir`` points at the one-directory-per-table
    Annas/Spotify parquet dump, one directory per table, which resolves only the
    playlists that snapshot captured. An install that configures neither
    keeps every other route working; the import routes say so instead of guessing.
    """

    client_id: str = ""
    client_secret: str = ""
    parquet_dir: str = ""  # dump level with playlists/playlist_tracks/tracks parts inside

    @field_validator("client_id", "client_secret", "parquet_dir")
    @classmethod
    def strip_values(cls, v: str) -> str:
        return (v or "").strip()


class AppConfig(BaseModel):
    """Root configuration object.

    ``spotify`` is the only optional section: an installation that leaves it empty
    keeps every existing route and prompt behaving exactly as before.
    """

    plex: PlexConfig
    llm: LLMConfig
    defaults: DefaultsConfig = DefaultsConfig()
    spotify: SpotifyConfig = SpotifyConfig()


# =============================================================================
# API Request/Response Models
# =============================================================================


class GenreCount(BaseModel):
    """Genre with track count."""

    name: str
    count: int | None = None


class DecadeCount(BaseModel):
    """Decade with track count."""

    name: str
    count: int | None = None


class ArtistCount(BaseModel):
    """Artist with track count (the artist roster offered alongside genres/decades)."""

    name: str
    count: int | None = None


class LibraryStatsResponse(BaseModel):
    """Library statistics response."""

    total_tracks: int
    genres: list[GenreCount]
    decades: list[DecadeCount]


class FilterHint(BaseModel):
    """One prompt phrase and the library filters it produced."""

    hint: str  # Prompt phrase ("chill", "90s", "shoegaze")
    matches: list[str] = []  # Library filters it produced; empty when nothing matched


class AnalyzePromptRequest(BaseModel):
    """Request to analyze a natural language prompt."""

    prompt: str = Field(..., min_length=1, max_length=2000)


class AnalyzePromptResponse(BaseModel):
    """Response from prompt analysis.

    Filter suggestions come in two tiers:

    * **Hard filters** (``suggested_*``) can be enforced against library metadata
      — genres, decades, artists, genre/artist/decade exclusions, a recording-year
      window, a play-count window (familiarity lives in Plex's ``viewCount``), a
      last-heard window (Plex's ``lastViewedAt``), an unheard-only flag, a
      compilation drop, whether live rows stay in the pool, a rated/unrated test,
      a rating floor and ceiling, and a duration window. They feed
      :class:`FilterPreviewRequest` and :class:`GenerateRequest`.
    * **Soft cues** (``style_cues``) are mood/energy/tempo/instrument phrases the
      library has no column for. They steer the generation prompt instead.

    The ``available_*`` measurements say which of those filters the library can
    actually answer, so a "heard recently" or "unrated only" suggestion is only
    made when the library has the rows for it.
    """

    suggested_genres: list[str]
    suggested_decades: list[str]
    available_genres: list[GenreCount]
    available_decades: list[DecadeCount]
    suggested_artists: list[str] = []  # Library artists named in the prompt
    suggested_excluded_genres: list[str] = []  # "no synth-pop" → drop these genres
    suggested_exclude_artists: list[str] = []  # "anything but Nickelback" → drop these artists
    suggested_exclude_decades: list[str] = []  # "not the 80s" → drop these decades
    suggested_min_year: int = 0  # 0 = no floor ("recorded in 1977", "last five years")
    suggested_max_year: int = 0  # 0 = no ceiling ("recorded before 1980")
    suggested_min_plays: int = 0  # 0 = no floor on Plex's play counter ("most played")
    suggested_max_plays: int = 0  # 0 = no ceiling ("not the tracks I blast")
    suggested_unheard_only: bool = False  # "hidden gems" → only never-played tracks
    suggested_exclude_compilations: bool = False  # "studio albums only" → drop Various Artists
    suggested_min_rating: int = 0  # 0 = no floor; prompt asked for well-rated music
    suggested_min_duration_ms: int = 0  # 0 = no floor ("long songs", "epics")
    suggested_max_duration_ms: int = 0  # 0 = no ceiling ("short songs", "under 3 minutes")
    suggested_live_only: bool = False  # "only live takes" → keep just the live rows
    suggested_exclude_live: bool = True  # live rows stay unless "only live" was asked for
    suggested_rated_only: bool = False  # "music I have rated" → rows carrying a user rating
    suggested_unrated_only: bool = False  # "stuff I never judged" → rows without a user rating
    suggested_max_rating: int = 0  # 0 = no ceiling ("nothing I gave 9s to")
    suggested_played_after: str = ""  # ISO YYYY-MM-DD last-heard floor ("" = unbounded)
    suggested_played_before: str = ""  # ISO YYYY-MM-DD last-heard ceiling ("" = unbounded)
    style_cues: list[str] = []  # Soft steering: mood / energy / tempo / instrument phrases
    available_artists: list[ArtistCount] = []  # Closest library artists (capped sample)
    available_durations_ms: list[int] = []  # Five-number summary: [shortest, p25, median, p75, longest]
    available_plays: list[int] = []  # Five-number summary of play counts: [min, p25, median, p75, max]
    unheard_tracks: int = -1  # Tracks with view_count = 0 in the library (-1 = unknown)
    year_span: list[int] = []  # [oldest, newest] release years the library holds
    available_last_played: list[str] = []  # [oldest, newest] last-heard dates ("" = never played)
    rated_tracks: int = -1  # Tracks carrying a user rating (-1 = unknown)
    live_tracks: int = -1  # Live recordings the library holds (-1 = unknown)
    reasoning: str
    matched_hints: list[FilterHint] = []  # Evidence: prompt phrase → filters chosen
    excluded_genres: list[str] = []  # Library genres considered and left out
    token_count: int = 0
    estimated_cost: float = 0.0
    method: str = "llm"  # "rules", "embeddings", or "llm"
    confidence: float = 0.0


class AnalyzeTrackRequest(BaseModel):
    """Request to analyze a seed track for dimensions."""

    rating_key: str


class AnalyzeTrackResponse(BaseModel):
    """Response from track analysis."""

    track: Track
    dimensions: list[Dimension]
    token_count: int = 0
    estimated_cost: float = 0.0
    cached: bool = False


class FilterPreviewRequest(BaseModel):
    """Request to preview filter results."""

    genres: list[str] = []
    decades: list[str] = []
    artists: list[str] = []  # Restrict to these artists (OR matching)
    exclude_genres: list[str] = []  # Drop tracks carrying these genres
    exclude_artists: list[str] = []  # Drop these artists ("anything but Nickelback")
    exclude_decades: list[str] = []  # Drop these decades ("not the 80s")
    min_year: int = 0  # 0 = no floor ("recorded in 1977", "last five years")
    max_year: int = 0  # 0 = no ceiling ("before 1980")
    min_plays: int = 0  # 0 = no floor on Plex's play counter ("most played")
    max_plays: int = 0  # 0 = no ceiling ("deep cuts, not the rotation")
    unheard_only: bool = False  # Only tracks never played ("hidden gems")
    exclude_compilations: bool = False  # Drop Various Artists / compilation rows
    track_count: int = 25
    max_tracks_to_ai: int = 500  # 0 = no limit
    min_rating: int = 0  # 0 = any, 2/4/6/8/10 = minimum rating (Plex uses 0-10)
    min_duration_ms: int = 0  # 0 = any length
    max_duration_ms: int = 0  # 0 = no ceiling ("short songs for a run")
    live_only: bool = False  # "only live takes" → keep just the live rows
    rated_only: bool = False  # "music I have rated" → rows carrying a user rating
    unrated_only: bool = False  # "stuff I never judged" → rows without a user rating
    max_rating: int = 0  # 0 = no ceiling, 2/4/6/8/10 = rating cap (Plex uses 0-10)
    played_after: str = ""  # ISO YYYY-MM-DD: last heard on/after this date
    played_before: str = ""  # ISO YYYY-MM-DD: last heard on/before this date
    exclude_live: bool = True
    prompt: str | None = None  # Used to size the dynamic AI track budget
    # Plex account whose star ratings the rating facets read ("" = the account we
    # connect with). A household member's stars live on their own connection, so
    # the first request naming one pays for that scan; tracks they never rated
    # keep the admin's rating, which is the documented fallback.
    rating_user: str = ""

    @field_validator("max_duration_ms", "min_duration_ms", "min_year", "max_year")
    @classmethod
    def validate_duration_window(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Duration bounds are milliseconds above zero (0 = unbounded)")
        return v

    @field_validator("min_plays", "max_plays")
    @classmethod
    def validate_play_window(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Play-count bounds are whole plays above zero (0 = unbounded)")
        return v

    @field_validator("max_rating")
    @classmethod
    def validate_rating_ceiling(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError("max_rating is a Plex rating cap between 0 and 10")
        return v

    @model_validator(mode="after")
    def validate_duration_window_order(self) -> "FilterPreviewRequest":
        if self.min_duration_ms and self.max_duration_ms and self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms cannot exceed max_duration_ms")
        if self.min_year and self.max_year and self.min_year > self.max_year:
            raise ValueError("min_year cannot exceed max_year")
        if self.min_plays and self.max_plays and self.min_plays > self.max_plays:
            raise ValueError("min_plays cannot exceed max_plays")
        if self.unheard_only and self.min_plays > 0:
            raise ValueError("unheard_only contradicts a min_plays floor")
        return self

    @model_validator(mode="after")
    def validate_familiarity_window(self) -> "FilterPreviewRequest":
        conflict = _familiarity_conflict(self)
        if conflict:
            raise ValueError(conflict)
        return self

    @model_validator(mode="after")
    def normalize_rating_user(self) -> "FilterPreviewRequest":
        # The UI reuses the Plex roster's own key for the connecting account, and
        # the cache's ``user_rating`` column IS that account: "" and ``__owner__``
        # are the same thing, so normalize to one spelling.
        if self.rating_user.strip() == OWNER_USER_KEY:
            self.rating_user = ""
        return self


class FilterPreviewResponse(BaseModel):
    """Response with filter preview stats."""

    matching_tracks: int  # -1 if unknown; on the cache branch this is the pool the
                          # run would actually use, i.e. after filter relaxation
    tracks_to_send: int  # How many will actually be sent to AI
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float
    # What the filters as asked matched, and which tiers had to give up. The cache
    # branch widens filters when they leave fewer candidates than the AI budget wants
    # (:func:`library_cache.relax_filters_for_target`), so the preview and the run
    # agree on the pool. Defaults keep a response that never relaxed anything valid.
    matching_as_asked: int = 0  # Candidates the strict filters had (0 = not measured)
    relaxed_tiers: list[str] = []  # Tiers that changed something, in order
    relaxed_to: str = ""  # Tier reached ("" = nothing relaxed, e.g. the Plex branch)
    # Whose ratings the rating facets actually read, and why a requested account
    # could not be used: a household member's stars need their own connection, and
    # the live-Plex fallback has none, so a preview may quietly answer in the
    # admin's stars. "" means the admin account.
    rating_user: str = ""  # Account used for the rating facets ("" = the admin's)
    rating_error: str = ""  # Set when the requested account's ratings were not used


class SeedTrackInput(BaseModel):
    """Seed track input for generation."""

    rating_key: str
    selected_dimensions: list[str]


class GenerateRequest(BaseModel):
    """Request to generate a playlist."""

    prompt: str | None = None
    seed_track: SeedTrackInput | None = None
    additional_notes: str | None = None
    refinement_answers: list[str | None] | None = None
    genres: list[str]
    decades: list[str]
    artists: list[str] = []  # Restrict to these artists (OR matching)
    exclude_genres: list[str] = []  # Drop tracks carrying these genres
    exclude_artists: list[str] = []  # Drop these artists ("anything but Nickelback")
    exclude_decades: list[str] = []  # Drop these decades ("not the 80s")
    min_year: int = 0  # 0 = no floor ("recorded in 1977", "last five years")
    max_year: int = 0  # 0 = no ceiling ("before 1980")
    min_plays: int = 0  # 0 = no floor on Plex's play counter ("most played")
    max_plays: int = 0  # 0 = no ceiling ("deep cuts, not the rotation")
    unheard_only: bool = False  # Only tracks never played ("hidden gems")
    exclude_compilations: bool = False  # Drop Various Artists / compilation rows
    track_count: int = 25
    exclude_live: bool = True
    min_rating: int = 0  # 0 = any, 2/4/6/8/10 = minimum rating
    min_duration_ms: int = 0  # 0 = any length
    max_duration_ms: int = 0  # 0 = no ceiling
    live_only: bool = False  # "only live takes" → keep just the live rows
    rated_only: bool = False  # "music I have rated" → rows carrying a user rating
    unrated_only: bool = False  # "stuff I never judged" → rows without a user rating
    max_rating: int = 0  # 0 = no ceiling, 2/4/6/8/10 = rating cap (Plex uses 0-10)
    played_after: str = ""  # ISO YYYY-MM-DD: last heard on/after this date
    played_before: str = ""  # ISO YYYY-MM-DD: last heard on/before this date
    max_tracks_to_ai: int = 500  # 0 = no limit
    style_cues: list[str] = []  # Soft steering phrases from prompt analysis (no DB column)
    # Plex account whose star ratings the rating facets read ("" = the account we
    # connect with). See :attr:`FilterPreviewRequest.rating_user` for the fallback
    # rules and the scan this pays for.
    rating_user: str = ""

    @field_validator("min_duration_ms", "max_duration_ms", "min_year", "max_year")
    @classmethod
    def validate_duration_window(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Duration bounds are milliseconds above zero (0 = unbounded)")
        return v

    @field_validator("min_plays", "max_plays")
    @classmethod
    def validate_play_window(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Play-count bounds are whole plays above zero (0 = unbounded)")
        return v

    @field_validator("max_rating")
    @classmethod
    def validate_rating_ceiling(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError("max_rating is a Plex rating cap between 0 and 10")
        return v

    @model_validator(mode="after")
    def validate_duration_window_order(self) -> "GenerateRequest":
        if self.min_duration_ms and self.max_duration_ms and self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms cannot exceed max_duration_ms")
        if self.min_year and self.max_year and self.min_year > self.max_year:
            raise ValueError("min_year cannot exceed max_year")
        if self.min_plays and self.max_plays and self.min_plays > self.max_plays:
            raise ValueError("min_plays cannot exceed max_plays")
        if self.unheard_only and self.min_plays > 0:
            raise ValueError("unheard_only contradicts a min_plays floor")
        return self

    @model_validator(mode="after")
    def validate_familiarity_window(self) -> "GenerateRequest":
        conflict = _familiarity_conflict(self)
        if conflict:
            raise ValueError(conflict)
        return self

    @model_validator(mode="after")
    def check_flow(self) -> "GenerateRequest":
        if not self.prompt and not self.seed_track:
            raise ValueError("Either prompt or seed_track must be provided")
        return self

    @model_validator(mode="after")
    def normalize_rating_user(self) -> "GenerateRequest":
        if self.rating_user.strip() == OWNER_USER_KEY:
            self.rating_user = ""
        return self


class GenerateResponse(BaseModel):
    """Response from playlist generation."""

    tracks: list[Track]
    token_count: int
    estimated_cost: float
    # Curator narrative fields
    playlist_title: str = ""
    narrative: str = ""
    track_reasons: dict[str, str] = {}
    # Which filters the cache could not honour: the pool was widened tier by tier
    # from ``matching_as_asked`` candidates up to the size the candidate budget
    # wanted (see :func:`library_cache.relax_filters_for_target`). Empty on the
    # live-Plex branch and whenever the filters already reached the target.
    matching_as_asked: int = 0
    relaxed_tiers: list[str] = []
    relaxed_to: str = ""
    # Whose stars the rating facets actually filtered by, and why a requested
    # account could not be used (the live-Plex fallback has no member connection).
    rating_user: str = ""
    rating_error: str = ""


def _validate_rating_keys(v: list[str]) -> list[str]:
    """Validate a list of Plex rating keys (must be non-empty, all numeric)."""
    if not v:
        raise ValueError("At least one track is required")
    for key in v:
        if not key.isdigit():
            raise ValueError(f"Invalid rating key: {key}")
    return v


def _truncate_description(v: str) -> str:
    """Truncate description to 2000 chars for Plex compatibility."""
    return v[:2000] if v else v


def _validate_result_id(v: str | None) -> str | None:
    """Normalize an optional history-row id ("" → absent)."""
    if v is None:
        return None
    cleaned = v.strip()
    if not cleaned:
        return None
    if not 8 <= len(cleaned) <= 16 or not all(c in "0123456789abcdef" for c in cleaned):
        raise ValueError("result_id must be an 8-16 character hex result id")
    return cleaned


class PlexUserInfo(BaseModel):
    """A Plex account that can own playlists on this server."""

    key: str  # Sent back to the API ('__owner__' or the Plex username/title)
    title: str  # Display name (home user title, or username for shared friends)
    user_id: int | None = None
    is_owner: bool = False
    is_home: bool = False
    is_managed: bool = False  # Managed (kids) profiles cannot create their own playlists
    has_pin: bool = False  # PIN-protected user; an admin can still act for them


class PlexPlaylistCopy(BaseModel):
    """Result of creating a copy of a playlist for an additional user."""

    user: str
    success: bool
    playlist_id: str | None = None
    playlist_url: str | None = None
    tracks_added: int | None = None
    tracks_skipped: int | None = None
    error: str | None = None


def _validate_user_keys(v: list[str]) -> list[str]:
    """Normalize a list of Plex user keys, dropping blanks and duplicates."""
    seen: list[str] = []
    for item in v:
        key = (item or "").strip()
        if key and key not in seen:
            seen.append(key)
    return seen


class SavePlaylistRequest(BaseModel):
    """Request to save a playlist to Plex."""

    name: str
    rating_keys: list[str]
    description: str = ""  # Playlist description (narrative) saved to Plex
    user: str = OWNER_USER_KEY  # Owner of the playlist written to Plex
    copy_to: list[str] = []  # Additional users to receive their own copy (owner key dropped)
    result_id: str | None = None  # History row to link to the created Plex playlist

    @field_validator("user")
    @classmethod
    def normalize_user(cls, v: str) -> str:
        return (v or "").strip() or OWNER_USER_KEY

    @field_validator("copy_to")
    @classmethod
    def validate_copy_to(cls, v: list[str]) -> list[str]:
        return _validate_user_keys(v)

    @model_validator(mode="after")
    def drop_redundant_owner_copy(self) -> "SavePlaylistRequest":
        # The owning account already receives the primary playlist, so a copy aimed
        # at the same Plex key would write a second identical playlist. Dedupe here
        # rather than on the Plex side: a Plex connection cannot name its account
        # (PlexServer.account is a zero-arg method), so plex_client has no reliable
        # account id to compare. Keys are exact Plex titles, so only an identical
        # key is recognized — user="__owner__" plus copy_to=["<home user title>"]
        # still writes two playlists.
        self.copy_to = [key for key in self.copy_to if key != self.user]
        return self

    @field_validator("result_id")
    @classmethod
    def normalize_result_id(cls, v: str | None) -> str | None:
        return _validate_result_id(v)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Playlist name cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def truncate_description(cls, v: str) -> str:
        return _truncate_description(v)

    @field_validator("rating_keys")
    @classmethod
    def validate_rating_keys(cls, v: list[str]) -> list[str]:
        return _validate_rating_keys(v)


class SavePlaylistResponse(BaseModel):
    """Response from saving a playlist."""

    success: bool
    playlist_id: str | None = None
    playlist_url: str | None = None
    error: str | None = None
    tracks_added: int | None = None
    tracks_skipped: int | None = None
    owner: str | None = None  # Display name of the account that owns the playlist
    copies: list[PlexPlaylistCopy] = []  # Copies created for other Plex users
    linked_result_id: str | None = None  # History row now pointing at this playlist


# =============================================================================
# Instant Queue Models (005)
# =============================================================================


class PlexPlaylistInfo(BaseModel):
    """Lightweight playlist info for the picker."""

    rating_key: str
    title: str
    track_count: int


class PlexClientInfo(BaseModel):
    """Online Plex client info."""

    client_id: str
    name: str
    product: str
    platform: str
    is_playing: bool
    is_mobile: bool = False


class UpdatePlaylistRequest(BaseModel):
    """Request to update an existing playlist."""

    playlist_id: str
    rating_keys: list[str]
    mode: Literal["replace", "append"]
    description: str = ""
    user: str = OWNER_USER_KEY  # Owner of the playlist being updated

    @field_validator("user")
    @classmethod
    def normalize_user(cls, v: str) -> str:
        return (v or "").strip() or OWNER_USER_KEY

    @field_validator("description")
    @classmethod
    def truncate_description(cls, v: str) -> str:
        return _truncate_description(v)

    @field_validator("playlist_id")
    @classmethod
    def validate_playlist_id(cls, v: str) -> str:
        if v != "__scratch__" and not v.isdigit():
            raise ValueError("playlist_id must be '__scratch__' or a numeric rating key")
        return v

    @field_validator("rating_keys")
    @classmethod
    def validate_rating_keys(cls, v: list[str]) -> list[str]:
        return _validate_rating_keys(v)


class UpdatePlaylistResponse(BaseModel):
    """Response from updating a playlist."""

    success: bool
    tracks_added: int = 0
    tracks_skipped: int = 0
    duplicates_skipped: int = 0
    playlist_url: str | None = None
    warning: str | None = None
    error: str | None = None
    owner: str | None = None  # Display name of the account that owns the playlist


class PlayQueueRequest(BaseModel):
    """Request to create a play queue."""

    rating_keys: list[str]
    client_id: str
    mode: Literal["replace", "play_next"] = "replace"

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("client_id cannot be empty")
        return v

    @field_validator("rating_keys")
    @classmethod
    def validate_rating_keys(cls, v: list[str]) -> list[str]:
        return _validate_rating_keys(v)


class PlayQueueResponse(BaseModel):
    """Response from play queue creation."""

    success: bool
    client_name: str | None = None
    client_product: str | None = None
    tracks_queued: int = 0
    tracks_skipped: int = 0
    error: str | None = None


class ConfigResponse(BaseModel):
    """Config without secrets for display."""

    version: str
    plex_url: str
    plex_connected: bool
    plex_token_set: bool  # True if token is configured (without revealing it)
    music_library: str | None
    llm_provider: str
    llm_configured: bool
    llm_api_key_set: bool  # True if API key is configured (without revealing it)
    model_analysis: str  # The analysis model being used
    model_generation: str  # The generation model being used
    max_tracks_to_ai: int  # Recommended max tracks for this model
    max_albums_to_ai: int  # Recommended max albums for this model
    cost_per_million_input: float  # Cost per million input tokens for generation model
    cost_per_million_output: float  # Cost per million output tokens for generation model
    analysis_cost_per_million_input: float = 0.0  # Cost per million input tokens for analysis model
    analysis_cost_per_million_output: float = 0.0  # Cost per million output tokens for analysis model
    defaults: DefaultsConfig
    # Local provider fields
    ollama_url: str = "http://localhost:11434"
    ollama_context_window: int = 32768
    custom_url: str = ""
    custom_context_window: int = 32768
    is_local_provider: bool = False
    provider_from_env: bool = False  # True if LLM_PROVIDER env var is overriding UI


class UpdateConfigRequest(BaseModel):
    """Partial config update."""

    plex_url: str | None = None
    plex_token: str | None = None
    music_library: str | None = None
    llm_provider: str | None = None
    llm_api_key: str | None = None
    model_analysis: str | None = None
    model_generation: str | None = None
    # Local provider fields
    ollama_url: str | None = None
    ollama_context_window: int | None = None
    custom_url: str | None = None
    custom_context_window: int | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    plex_connected: bool
    llm_configured: bool


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None


# =============================================================================
# Per-Plex-account rating models (whose stars a filter reads)
# =============================================================================


class PlexUserRatingsStatus(BaseModel):
    """One Plex household account's ratings, as cached locally."""

    user_key: str  # Exact Plex title ('Becky', 'The Boys') or shared username
    rated_count: int = 0  # Tracks that account rated
    total_seen: int = 0  # Library size when the scan ran
    last_sync_at: str | None = None


class UserRatingsStatusResponse(BaseModel):
    """State of the per-account rating store.

    The admin account never appears: its stars are the ``user_rating`` column
    synced with the library, which is also the fallback for tracks a member
    never rated.
    """

    total_tracks: int = 0
    users: list[PlexUserRatingsStatus] = []
    synced_users: list[str] = []  # Keys already scanned, so the rest need a scan


class RefreshUserRatingsRequest(BaseModel):
    """Re-scan Plex accounts' ratings into the cache."""

    users: list[str] = []  # Empty = every account already cached

    @field_validator("users")
    @classmethod
    def validate_users(cls, v: list[str]) -> list[str]:
        return _validate_user_keys(v)


class RefreshUserRatingsResponse(BaseModel):
    """Per-account outcome of a rating rescan."""

    refreshed: list[str] = []  # Accounts whose ratings were rewritten
    errors: dict[str, str] = {}  # Account -> why its scan failed
    status: UserRatingsStatusResponse


# =============================================================================
# Agent curation memory (data/agent_song_reasoning.db)
# =============================================================================


class ReasoningTagStat(BaseModel):
    """One mined selection-reason tag with how many tracks carry it."""

    tag: str
    count: int = 0


class ReasoningStatusResponse(BaseModel):
    """State of the agent's curation memory (the reasoning DB)."""

    runs: int = 0  # Generation runs recorded
    selections: int = 0  # Stored (track, reason) picks
    tracks_with_reasons: int = 0  # Distinct tracks with at least one stored reason
    tagged_tracks: int = 0  # Distinct tracks carrying mined tags
    tags: list[ReasoningTagStat] = []  # The vocabulary the agent curated with so far
    last_run_at: str | None = None
    error: str | None = None


# =============================================================================
# Ollama API Models
# =============================================================================


class OllamaModel(BaseModel):
    """A model available in Ollama."""

    name: str
    size: int = 0
    modified_at: str = ""


class OllamaModelInfo(BaseModel):
    """Detailed info about an Ollama model."""

    name: str
    context_window: int
    context_detected: bool = True  # False if using fallback default
    parameter_size: str | None = None


class OllamaModelsResponse(BaseModel):
    """Response from listing Ollama models."""

    models: list[OllamaModel] = []
    error: str | None = None


class OllamaStatus(BaseModel):
    """Connection status for Ollama."""

    connected: bool
    model_count: int = 0
    error: str | None = None


# =============================================================================
# Library Cache Models
# =============================================================================


class SyncProgress(BaseModel):
    """Progress details when sync is running."""

    phase: str | None = None  # "fetching_albums", "fetching", or "processing"
    current: int
    total: int


class LibraryCacheStatusResponse(BaseModel):
    """Response from GET /api/library/status."""

    track_count: int
    synced_at: str | None = None
    is_syncing: bool
    sync_progress: SyncProgress | None = None
    error: str | None = None
    plex_connected: bool
    needs_resync: bool = False


class SyncTriggerResponse(BaseModel):
    """Response from POST /api/library/sync."""

    started: bool
    blocking: bool = False


# =============================================================================
# Recommendation Models (006)
# =============================================================================


class AlbumCandidate(BaseModel):
    """An album from the user's Plex library, aggregated from cached tracks."""

    parent_rating_key: str
    album: str
    album_artist: str
    year: int | None = None
    genres: list[str] = []
    decade: str = ""
    track_count: int = 0
    track_rating_keys: list[str] = []


class ClarifyingQuestion(BaseModel):
    """A question generated by the LLM to refine the recommendation."""

    question_text: str
    options: list[str]
    dimension: str


class SommelierPitch(BaseModel):
    """The editorial writeup for a recommendation."""

    hook: str = ""
    context: str = ""
    listening_guide: str = ""
    connection: str = ""
    short_pitch: str = ""
    full_text: str = ""


class AlbumRecommendation(BaseModel):
    """The output of the recommendation pipeline."""

    rank: str  # "primary" or "secondary"
    album: str
    artist: str
    year: int | None = None
    rating_key: str | None = None
    track_rating_keys: list[str] = []
    art_url: str | None = None
    pitch: SommelierPitch = SommelierPitch()
    research_available: bool = False


class ResearchData(BaseModel):
    """External research fetched for grounding the pitch."""

    musicbrainz_id: str | None = None
    release_date: str | None = None
    label: str | None = None
    track_listing: list[str] = []
    credits: dict[str, str] = {}
    genre_tags: list[str] = []
    wikipedia_summary: str | None = None
    review_links: list[str] = []
    review_texts: list[str] = []
    cover_art_url: str | None = None
    earliest_release_mbid: str | None = None


class ExtractedFacts(BaseModel):
    """Structured facts extracted from research sources by LLM."""

    origin_story: str = ""
    personnel: list[str] = []
    musical_style: str = ""
    vocal_approach: str = ""
    cultural_context: str = ""
    track_highlights: str = ""
    common_misconceptions: str = ""
    source_coverage: str = ""
    track_listing: list[str] = []  # Authoritative list from MusicBrainz, not LLM-extracted

    def to_text(self, include_track_listing: bool = True) -> str:
        """Format facts as labeled text block for LLM prompts."""
        parts = []
        for label, value in [
            ("Origin", self.origin_story),
            ("Personnel", ", ".join(self.personnel) if self.personnel else ""),
            ("Musical style", self.musical_style),
            ("Vocal approach", self.vocal_approach),
            ("Cultural context", self.cultural_context),
            ("Track highlights", self.track_highlights),
            ("Common misconceptions", self.common_misconceptions),
            ("Source coverage", self.source_coverage),
        ]:
            if value:
                parts.append(f"- {label}: {value}")
        if include_track_listing and self.track_listing:
            parts.append("- Track listing: " + ", ".join(self.track_listing))
        return "\n".join(parts)


class PitchIssue(BaseModel):
    """A factual issue found during pitch validation."""

    claim: str
    problem: str
    correction: str


class PitchValidation(BaseModel):
    """Result of validating a pitch against research data."""

    valid: bool
    issues: list[PitchIssue] = []


class TasteProfile(BaseModel):
    """Aggregate view of the user's library for discovery mode."""

    genre_distribution: dict[str, int] = {}
    decade_distribution: dict[str, int] = {}
    top_artists: list[str] = []
    total_albums: int = 0
    owned_albums: list[dict[str, str]] = []


class RecommendSessionState(BaseModel):
    """Transient state maintained during a recommendation session."""

    mode: Literal["library", "discovery"] = "library"
    prompt: str = ""
    filters: dict[str, list[str]] = {}
    questions: list[ClarifyingQuestion] = []
    answers: list[str | None] = []
    answer_texts: list[str] = []
    album_candidates: list[AlbumCandidate] = []
    taste_profile: TasteProfile | None = None
    familiarity_pref: Literal["any", "comfort", "rediscover", "hidden_gems"] = "any"
    previously_recommended: list[str] = []  # "artist|||album" keys shown in prior rounds
    # Cost accumulators (reset each generation round)
    total_tokens: int = 0
    total_cost: float = 0.0


class AnalyzePromptFiltersRequest(BaseModel):
    """Request to analyze a prompt and suggest genre/decade filters."""

    prompt: str
    genres: list[str] = []
    decades: list[str] = []


class AnalyzePromptFiltersResponse(BaseModel):
    """Response with suggested genre/decade pre-selections."""

    genres: list[str] = []
    decades: list[str] = []
    matched_hints: list[FilterHint] = []  # Evidence: prompt phrase → filters chosen
    excluded_genres: list[str] = []  # Available genres the model left out on purpose
    reasoning: str = ""


class RecommendQuestionsRequest(BaseModel):
    """Request to generate clarifying questions."""

    prompt: str = Field(..., min_length=1, max_length=2000)


class RecommendQuestionsResponse(BaseModel):
    """Response with clarifying questions."""

    questions: list[ClarifyingQuestion]
    session_id: str
    token_count: int = 0
    estimated_cost: float = 0.0


class RecommendSwitchModeRequest(BaseModel):
    """Request to switch a recommendation session to a different mode."""

    session_id: str
    mode: Literal["library", "discovery"]


class RecommendSwitchModeResponse(BaseModel):
    """Response after switching recommendation mode."""

    session_id: str


class RecommendGenerateRequest(BaseModel):
    """Request to generate album recommendations."""

    session_id: str
    answers: list[str | None]
    answer_texts: list[str] = []
    mode: Literal["library", "discovery"] = "library"
    genres: list[str] = []
    decades: list[str] = []
    familiarity_pref: Literal["any", "comfort", "rediscover", "hidden_gems"] = "any"
    max_albums: int = 2500

    @field_validator("max_albums")
    @classmethod
    def validate_max_albums(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_albums must be non-negative")
        return min(v, 50000)


class RecommendGenerateResponse(BaseModel):
    """Response with album recommendations."""

    recommendations: list[AlbumRecommendation]
    token_count: int = 0
    estimated_cost: float = 0.0
    research_warning: str | None = None


class AlbumPreviewResponse(BaseModel):
    """Response from album preview endpoint."""

    matching_albums: int
    albums_to_send: int
    estimated_input_tokens: int = 0
    estimated_cost: float = 0.0


# =============================================================================
# Results Persistence Models
# =============================================================================


class ResultListItem(BaseModel):
    """A saved result summary for history list (no snapshot)."""

    id: str
    type: str
    title: str
    prompt: str
    track_count: int
    artist: str | None = None
    art_rating_key: str | None = None
    subtitle: str | None = None
    created_at: str
    playlist_id: str | None = None  # Plex playlist this row was saved as
    playlist_title: str | None = None
    playlist_owner: str | None = None
    playlist_url: str | None = None


class ResultListResponse(BaseModel):
    """Paginated list of saved results."""

    results: list[ResultListItem]
    total: int


class ResultDetail(BaseModel):
    """Full saved result including snapshot for rendering."""

    id: str
    type: str
    title: str
    prompt: str
    track_count: int
    artist: str | None = None
    art_rating_key: str | None = None
    subtitle: str | None = None
    created_at: str
    playlist_id: str | None = None
    playlist_title: str | None = None
    playlist_owner: str | None = None
    playlist_url: str | None = None
    snapshot: dict


# =============================================================================
# Setup/Onboarding Models
# =============================================================================


class SetupStatusResponse(BaseModel):
    """Full onboarding checklist state."""

    data_dir_writable: bool
    process_uid: int = 0
    process_gid: int = 0
    data_dir: str = ""
    plex_connected: bool
    plex_error: str | None = None
    plex_from_env: bool = False
    music_libraries: list[str] = []
    llm_configured: bool
    llm_provider: str = ""
    llm_from_env: bool = False
    library_synced: bool
    track_count: int = 0
    is_syncing: bool = False
    sync_progress: SyncProgress | None = None
    setup_complete: bool


class ValidatePlexRequest(BaseModel):
    """Request to validate Plex credentials during setup."""

    plex_url: str
    plex_token: str
    music_library: str = "Music"


class ValidatePlexResponse(BaseModel):
    """Response from Plex validation."""

    success: bool
    error: str | None = None
    server_name: str | None = None
    music_libraries: list[str] = []


class ValidateAIRequest(BaseModel):
    """Request to validate AI provider credentials during setup."""

    provider: str
    api_key: str = ""
    ollama_url: str = ""
    custom_url: str = ""


class ValidateAIResponse(BaseModel):
    """Response from AI provider validation."""

    success: bool
    error: str | None = None
    provider_name: str = ""


class SetupCompleteResponse(BaseModel):
    """Response from marking setup as complete."""

    success: bool


# =============================================================================
# Spotify Playlist Import Models
# =============================================================================


class SpotifyMatchedTrack(BaseModel):
    """One library track that answered a Spotify playlist row."""

    rating_key: str
    title: str
    artist: str
    album: str
    duration_ms: int = 0
    position: int = 0  # 1-based slot of the Spotify row it came from
    match_kind: str = "exact"  # exact song key | version (album-free) | album (artist-free) | search (live Plex)


class SpotifyMissingSong(BaseModel):
    """One Spotify playlist row the library had no answer for.

    Also the shape stored in the cache's ``missing_songs`` table, which is the
    wish list: everything an import could not play stays there so the songs
    can be gotten later.
    """

    position: int = 0  # 1-based slot within its Spotify playlist
    title: str = ""
    album: str = ""
    artist: str = ""
    duration_ms: int = 0
    spotify_track_id: str = ""
    spotify_url: str = ""
    playlist_ref: str = ""  # The playlist import that missed this song
    playlist_name: str = ""  # Its display name, for reading the wish list
    recorded_at: str = ""  # When it was written ("" until persisted)


class SpotifyImportPreviewRequest(BaseModel):
    """Ask what a Spotify playlist would match, without writing anything."""

    playlist_ref: str  # open.spotify.com URL, spotify:playlist:<id>, or a bare id

    @field_validator("playlist_ref")
    @classmethod
    def validate_playlist_ref(cls, v: str) -> str:
        ref = (v or "").strip()
        if not ref:
            raise ValueError("playlist_ref must name the Spotify playlist (URL or id)")
        return ref


class SpotifyImportPreviewResponse(BaseModel):
    """What a Spotify playlist matched against the library; nothing written."""

    playlist_name: str = ""
    playlist_owner: str = ""  # The Spotify playlist's owner, when it names one
    spotify_playlist_url: str = ""  # Canonical open.spotify.com URL
    source: str = ""  # "api" (fresh fetch) or "dump" (local parquet snapshot)
    total_tracks: int = 0  # Spotify rows read (episodes and bare locals dropped)
    matched: list[SpotifyMatchedTrack] = []
    missing: list[SpotifyMissingSong] = []
    warnings: list[str] = []


class SpotifyImportRequest(BaseModel):
    """Import a Spotify playlist as a Plex playlist; missing songs are recorded."""

    playlist_ref: str  # open.spotify.com URL, spotify:playlist:<id>, or a bare id
    name: str = ""  # Plex playlist name (default: the Spotify playlist's name)
    description: str = ""
    user: str = OWNER_USER_KEY  # Owner of the playlist written to Plex
    copy_to: list[str] = []  # Additional users to receive their own copy

    @field_validator("playlist_ref")
    @classmethod
    def validate_playlist_ref(cls, v: str) -> str:
        ref = (v or "").strip()
        if not ref:
            raise ValueError("playlist_ref must name the Spotify playlist (URL or id)")
        return ref

    @field_validator("user")
    @classmethod
    def normalize_user(cls, v: str) -> str:
        return (v or "").strip() or OWNER_USER_KEY

    @field_validator("copy_to")
    @classmethod
    def validate_copy_to(cls, v: list[str]) -> list[str]:
        return _validate_user_keys(v)

    @field_validator("description")
    @classmethod
    def truncate_description(cls, v: str) -> str:
        return _truncate_description(v)

    @model_validator(mode="after")
    def drop_redundant_owner_copy(self) -> "SpotifyImportRequest":
        # Same rule as SavePlaylistRequest: the owning account gets the primary
        # playlist, so a copy aimed at the same Plex key would write a second
        # identical playlist.
        self.copy_to = [key for key in self.copy_to if key != self.user]
        return self


class SpotifyImportResponse(BaseModel):
    """An imported Spotify playlist as written to Plex, plus its wish list."""

    success: bool
    error: str | None = None
    playlist_name: str = ""
    playlist_owner: str = ""  # The Spotify playlist's owner, when it names one
    spotify_playlist_url: str = ""  # Canonical open.spotify.com URL
    source: str = ""  # "api" or "dump"
    total_tracks: int = 0
    matched: list[SpotifyMatchedTrack] = []
    missing: list[SpotifyMissingSong] = []
    matched_count: int = 0
    missing_count: int = 0
    missing_recorded: int = 0  # Wish-list rows written to the cache DB for this playlist
    result_id: str | None = None  # History row saved for this import
    playlist_id: str | None = None  # Plex playlist created
    playlist_url: str | None = None  # Plex web URL for it
    tracks_added: int | None = None
    tracks_skipped: int | None = None
    owner: str | None = None  # Display name of the Plex account that owns it
    copies: list[PlexPlaylistCopy] = []
    warnings: list[str] = []


class SpotifyMissingSummary(BaseModel):
    """One imported playlist and how many of its songs the library lacks."""

    playlist_ref: str = ""
    playlist_name: str = ""
    count: int = 0


class SpotifyMissingListResponse(BaseModel):
    """The stored wish list: songs imported playlists asked for but lacked."""

    missing: list[SpotifyMissingSong] = []
    playlists: list[SpotifyMissingSummary] = []
    total: int = 0
