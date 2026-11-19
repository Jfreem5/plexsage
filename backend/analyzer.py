"""Prompt analysis and seed track dimension extraction."""

import hashlib
import logging
from typing import Any

from backend.llm_client import get_llm_client
from backend.models import (
    AnalyzePromptResponse,
    AnalyzeTrackResponse,
    ArtistCount,
    Dimension,
    FilterHint,
    GenreCount,
    DecadeCount,
    Track,
)
from backend.plex_client import get_plex_client
from backend.prompt_classifier import classify_prompt
from backend import library_cache

logger = logging.getLogger(__name__)

# Bumped whenever TRACK_ANALYSIS_SYSTEM changes what the dimension cache should
# hold. It is folded into the cache fingerprint, so raising it retires the entries
# the old prompt produced instead of serving them forever.
DIMENSION_PROMPT_VERSION = 2

# Dimensions the seed-track analysis asks for (and accepts).
DIMENSION_MIN = 8
DIMENSION_MAX = 12

# Cap on how much prompt evidence is echoed back to the UI.
MAX_HINT_EVIDENCE = 8


PROMPT_ANALYSIS_SYSTEM = """You are a music expert helping to create playlists from a user's music library.

Analyze the user's prompt and suggest the filters that would help find matching tracks.
Some filters can be enforced against library metadata; mood phrases cannot.

Return a JSON object with:
- genres: Array of genre names copied from the available genre list (e.g., ["Alternative", "Rock", "Indie"])
- decades: Array of decade strings copied from the available decade list (e.g., ["1990s", "2000s"])
- artists: Array of artist names copied from the available artist list, ONLY when the request
    names specific performers; otherwise []
- exclude_genres: Array of available genre names the request explicitly rules out
    ("rock but no synth-pop" → ["Synth-pop"]); never repeat a genre already in "genres"
- exclude_artists: Array of available artist names the request explicitly rules out ("prog but not
    Yes" → ["Yes"]); never repeat a name already in "artists"
- exclude_decades: Array of available decade names the request rules out ("modern, not the 2010s");
    never repeat a decade already in "decades"
- min_year: 0 unless the request pins a release date, in which case the oldest acceptable release
    year ("recorded in 1977" → 1977, "between 1975 and 1985" → 1975, "the last five years" → five
    years before today); a decade word alone is NOT a year
- max_year: 0 unless the request caps the release date ("before 1980" → 1980); must not be below "min_year"
- min_plays: 0 unless the request asks for familiar music ("most played", "on repeat", "well loved"),
    in which case a floor on Plex's per-track play counter (typically 3)
- max_plays: 0 unless the asks for music sitting outside its rotation ("deep cuts", "not what I blast"),
    in which case a ceiling on that play counter (typically 2)
- unheard_only: true only when the request wants music that has never been played at all
    ("hidden gems", "stuff I forgot I own"); then leave both play bounds at 0
- exclude_compilations: true when the request wants original albums instead of "best of" sets
    ("studio albums only", "no compilations", "not a Various Artists set")
- min_rating: 0 unless the request asks for well-rated / favourite music, in which case the
    user-rating floor on Plex's 0-10 scale (typically 8)
- min_duration_ms: 0 unless the request asks for longer tracks ("long songs", "over 6 minutes") — track length in milliseconds
- max_duration_ms: 0 unless the request asks for shorter tracks ("short songs", "under 3 minutes") — track length in milliseconds
- style_cues: Array of short mood / energy / tempo / instrumentation phrases the request evokes that the
    library has no column for (e.g. ["slower tempo, warm acoustic textures"]); 2-6 phrases, or [] if none
- reasoning: 2-4 sentences saying which part of the prompt led to each filter, and why
    any plausible genre was left out
- hint_matches: Array of {"hint": "phrase copied from the request", "matches": ["Genre", ...]}
    for every phrase that mattered — genre, decade, artist, year, or how familiar the music should
    be. Include a phrase with an empty "matches" array when the request named something the library
    has no equivalent for.
- excluded_genres: Array of genre names from the available list you considered and left out
    (this is evidence about the choice, not a filter to apply)

Be specific about genres, decades, artists, dates, familiarity and lengths. Consider:
- Mood/atmosphere (melancholy, upbeat, energetic) → style_cues, plus the genres that carry it
- Era references (90s, classic, modern) → decades; an exact year or a year span → the year fields
- Genre keywords (alternative, jazz, electronic) → genres
- Named performers and "greatest hits of X" → artists
- Explicit exclusions ("no", "without", "anything but") → exclude_genres and exclude_artists; a
  banned name always wins over the same name mentioned positively
- Familiarity requests ("most played", "on repeat") → min_plays; ("deep cuts", "off rotation") →
  max_plays; ("hidden gems", "never listened to") → unheard_only
- Date requests ("recorded in 1977", "the last five years", "older than 1980") → the year fields
- Album-collection requests ("studio albums only", "no compilations") → exclude_compilations
- Length requests ("short songs", "epics", "under four minutes") → the duration fields
- Quality requests ("well rated", "my favourites") → min_rating

Return ONLY valid JSON, no markdown formatting."""


TRACK_ANALYSIS_SYSTEM = """You are a music expert analyzing a song to identify its distinctive characteristics.

Given a track's title, artist, album, and year, identify 8-12 specific musical dimensions that make
this track unique. These dimensions will help the user explore similar music.

Aim for breadth: cover at least four different descriptor categories, and prefer several within the
categories that best describe this track (a tempo-driven track may carry three tempo statements).

For each dimension, provide:
- id: A short identifier (e.g., "mood", "tempo", "instrumentation")
- label: A specific, evocative label (NOT generic like "the mood" - be specific like
  "The melancholy, bittersweet mood")
- description: 1-2 sentences saying what makes this dimension true of THIS recording - name the
  concrete evidence (instrument, tempo range, production texture, vocal character)
- category: One of "mood", "energy", "tempo", "era", "instrumentation", "vocals",
  "production", "genre lineage", "scene"

Make dimensions SPECIFIC to this track, not generic. Bad: "The genre". Good: "90s British alternative
rock with Britpop influences". Do not repeat a dimension: each label must say something different.

Return a JSON object with:
{
  "dimensions": [
    {"id": "mood", "label": "The melancholy, introspective mood", "description": "...", "category": "mood"},
    ...
  ]
}

Return ONLY valid JSON, no markdown formatting."""


def _duration_phrase(summary: list[int]) -> str:
    """Render a five-number duration summary as minutes for the LLM prompt."""
    if not summary:
        return "unknown"
    labels = ["shortest", "p25", "median", "p75", "longest"]
    parts = []
    for label, ms in zip(labels, summary):
        if not ms:
            continue
        parts.append(f"{label} {ms / 60000:.1f} min")
    return ", ".join(parts) or "unknown"


def _library_facets(
    plex_client,
) -> tuple[list[ArtistCount], list[int], list[int], int, list[int]]:
    """Artist roster, track lengths, play counts and year span of the library.

    The local cache answers first because it holds exact per-artist track counts,
    the Plex play counter (``view_count``) and every release year, so all three
    numeric summaries are exact there. Plex is sampled only when the cache is
    empty: its artist hub exposes no track count and its search API has no way
    to ask "how many tracks were never played", so the play facets and the year
    span stay unknown on that path and the caller must not suggest a window from
    them.

    Args:
        plex_client: Connected :class:`PlexClient`, used for the fallback.

    Returns:
        ``(available_artists, available_durations_ms, available_plays,
        unheard_tracks, year_span)``. The durations and the play counts are
        five-number summaries (``[min, p25, median, p75, max]``, empty when the
        library has none), ``unheard_tracks`` is the count of never-played
        tracks or -1 when unknown, and ``year_span`` is ``[oldest, newest]``
        release years (empty when nothing is dated).
    """
    roster: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    play_stats: dict[str, Any] = {}
    if library_cache.has_cached_tracks():
        try:
            roster = library_cache.get_cached_artists(limit=library_cache.MAX_ARTIST_OPTIONS)
            stats = library_cache.get_cached_duration_stats()
            play_stats = library_cache.get_cached_play_stats()
        except Exception:
            logger.warning("Artist/duration/play roster unavailable from cache", exc_info=True)
            roster, stats, play_stats = [], {}, {}

    if not roster:
        roster = plex_client.get_all_artists(limit=library_cache.MAX_ARTIST_OPTIONS)
        stats = plex_client.get_duration_stats()

    artists = [
        ArtistCount(name=item["name"], count=item.get("count"))
        for item in roster
        if item.get("name")
    ]
    summary = [
        int(stats.get(key) or 0)
        for key in ("min_ms", "p25_ms", "p50_ms", "p75_ms", "max_ms")
    ]
    plays = [int(value) for value in (play_stats.get("plays") or [])][:5]
    unheard = play_stats.get("unheard")
    years = [int(value) for value in (play_stats.get("years") or [])][:2]
    return (
        artists,
        summary if any(summary) else [],
        plays,
        int(unheard) if isinstance(unheard, int) else -1,
        years,
    )


def _library_names(raw: Any, available: list[str]) -> list[str]:
    """Keep only *raw* entries the library really has, in library casing."""
    lookup = {name.lower(): name for name in available}
    kept: list[str] = []
    for item in (raw if isinstance(raw, list) else []):
        name = lookup.get(str(item).strip().lower())
        if name and name not in kept:
            kept.append(name)
    return kept


def _duration_bound(raw: Any) -> int:
    """Coerce an LLM-supplied duration bound to non-negative milliseconds."""
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _rating_floor(raw: Any) -> int:
    """Coerce an LLM-supplied rating floor into Plex's 0-10 range."""
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, value))


def _year_bound(raw: Any) -> int:
    """Coerce an LLM-supplied release year to a plausible year (0 = unbounded)."""
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return 0
    if value < 1:
        return 0
    return value if value <= 2999 else 0


def _play_bound(raw: Any) -> int:
    """Coerce an LLM-supplied play-count bound to a whole number of plays."""
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _bool_flag(raw: Any) -> bool:
    """Coerce an LLM-supplied flag; the model sometimes answers "yes"/"no"."""
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "yes", "1"}
    return bool(raw)


def _play_phrase(summary: list[int]) -> str:
    """Render a five-number play-count summary for the LLM prompt."""
    if not summary:
        return "unknown"
    labels = ["least played", "p25", "median", "p75", "most played"]
    parts = [f"{label} {value}" for label, value in zip(labels, summary)]
    return ", ".join(parts) or "unknown"


def _year_span_phrase(span: list[int]) -> str:
    """Render the oldest/newest release years the library holds."""
    if not span:
        return "unknown"
    return " to ".join(str(int(year)) for year in span)


def analyze_prompt(prompt: str) -> AnalyzePromptResponse:
    """Analyze a natural language prompt to suggest filters.

    Classification is attempted locally first (keyword rules, then embedding
    similarity).  The LLM is only called as a fallback when the local
    confidence is low, so the common case costs zero tokens.

    Suggestions come in two tiers: hard filters (genres, decades, artists,
    genre/artist/decade exclusions, a recording-year window, a play-count
    window, an unheard-only flag, a compilation drop, a rating floor and a
    duration window) that can be enforced against library metadata, and soft
    ``style_cues`` that only steer the generation prompt.

    Args:
        prompt: User's playlist description

    Returns:
        AnalyzePromptResponse with suggested and available filters, the local
        library's play-count and year-span measurements that make a date or
        familiarity request checkable, plus the prompt phrases that produced
        them (``matched_hints``) and the library genres that were considered
        and left out (``excluded_genres``).

    Raises:
        ValueError: If LLM response cannot be parsed
        RuntimeError: If required clients are not initialized
    """
    plex_client = get_plex_client()

    if not plex_client:
        raise RuntimeError("Plex client not initialized")

    # Get library stats for available filters
    stats = plex_client.get_library_stats()
    available_genres = [GenreCount(**g) for g in stats.get("genres", [])]
    available_decades = [DecadeCount(**d) for d in stats.get("decades", [])]

    # Artists, track lengths, play counts and the year span come from the local
    # cache when it is populated (Plex's genre/decade lists above stay server-
    # side: they are the exact vocabularies the filter endpoints accept).
    available_artists, available_durations, available_plays, unheard_tracks, year_span = _library_facets(
        plex_client
    )
    artist_names = [a.name for a in available_artists]

    # Primary path: local rules + embeddings (no LLM call)
    classification = classify_prompt(
        prompt,
        [g.name for g in available_genres],
        [d.name for d in available_decades],
        available_artists=artist_names,
    )
    if classification.confident:
        logger.info(
            "Classified prompt locally via %s (confidence %.2f)",
            classification.method,
            classification.confidence,
        )
        return AnalyzePromptResponse(
            suggested_genres=classification.genres,
            suggested_decades=classification.decades,
            available_genres=available_genres,
            available_decades=available_decades,
            suggested_artists=classification.artists,
            suggested_excluded_genres=classification.negated_genres,
            suggested_exclude_artists=classification.exclude_artists,
            suggested_exclude_decades=classification.exclude_decades,
            suggested_min_year=classification.min_year,
            suggested_max_year=classification.max_year,
            suggested_min_plays=classification.min_plays,
            suggested_max_plays=classification.max_plays,
            suggested_unheard_only=classification.unheard_only,
            suggested_exclude_compilations=classification.exclude_compilations,
            suggested_min_rating=classification.min_rating,
            suggested_min_duration_ms=classification.min_duration_ms,
            suggested_max_duration_ms=classification.max_duration_ms,
            style_cues=classification.style_cues,
            available_artists=available_artists,
            available_durations_ms=available_durations,
            available_plays=available_plays,
            unheard_tracks=unheard_tracks,
            year_span=year_span,
            reasoning=classification.reasoning,
            matched_hints=_hints_from_mapping(classification.matched_hints),
            excluded_genres=list(classification.excluded_genres),
            token_count=0,
            estimated_cost=0.0,
            method=classification.method,
            confidence=classification.confidence,
        )

    # Fallback path: full LLM analysis
    llm_client = get_llm_client()
    if not llm_client:
        raise RuntimeError("LLM client not initialized")

    logger.info(
        "Local classification confidence too low (%.2f); falling back to LLM",
        classification.confidence,
    )

    analysis_prompt = f"""User's playlist request: "{prompt}"

Available genres in their library:
{', '.join(f"{g.name} ({g.count})" if g.count else g.name for g in available_genres[:30])}

Available decades in their library:
{', '.join(f"{d.name} ({d.count})" if d.count else d.name for d in available_decades)}

Available artists in their library (a capped sample):
{', '.join(artist_names[:40])}

Track lengths in their library, shortest to longest (minutes): {_duration_phrase(available_durations)}
Whole plays per track in their library, least to most played: {_play_phrase(available_plays)}
Tracks in their library they have never played: {unheard_tracks if unheard_tracks >= 0 else "unknown"}
Release years their library covers: {_year_span_phrase(year_span)}
Suggest genres, decades and artists from the available options that best match the
user's request, plus anything the request rules out (genres, artists and decades),
any release date it implies, how familiar the picks should be, any quality bar it
implies, whether to drop Various Artists compilations, and any length it implies.
Only suggest a year window that fits the years their library actually covers, and
only ask for unheard or well-played tracks when the play counts show there are any.
Name the phrases in the request that drove each choice, and say which available
genres you considered and left out."""

    # Call LLM
    response = llm_client.analyze(analysis_prompt, PROMPT_ANALYSIS_SYSTEM)

    # Parse response
    data = llm_client.parse_json_response(response)

    # Filter suggestions to only include available options
    available_genre_names = {g.name for g in available_genres}
    available_decade_names = {d.name for d in available_decades}

    suggested_genres = [g for g in data.get("genres", []) if g in available_genre_names]
    suggested_decades = [d for d in data.get("decades", []) if d in available_decade_names]
    suggested_excluded = [
        g for g in _library_names(data.get("exclude_genres"), list(available_genre_names))
        if g not in suggested_genres
    ]
    # A stated ban wins over a positive mention, same as the local rules do.
    banned_artists = _library_names(data.get("exclude_artists"), artist_names)
    banned_artist_set = set(banned_artists)
    suggested_artists = [
        a
        for a in _library_names(data.get("artists"), artist_names)
        if a not in suggested_excluded and a not in banned_artist_set
    ]
    banned_artists = [a for a in banned_artists if a not in suggested_artists]
    suggested_excluded_decades = [
        d
        for d in (data.get("exclude_decades") or [])
        if d in available_decade_names and d not in suggested_decades
    ]
    style_cues = [
        str(c).strip()[:120] for c in (data.get("style_cues") or []) if str(c).strip()
    ][:8]
    min_duration_ms = _duration_bound(data.get("min_duration_ms"))
    max_duration_ms = _duration_bound(data.get("max_duration_ms"))
    if min_duration_ms and max_duration_ms and min_duration_ms > max_duration_ms:
        min_duration_ms, max_duration_ms = max_duration_ms, min_duration_ms
    suggested_rating = _rating_floor(data.get("min_rating"))
    min_year = _year_bound(data.get("min_year"))
    max_year = _year_bound(data.get("max_year"))
    if min_year and max_year and min_year > max_year:
        min_year, max_year = max_year, min_year
    min_plays = _play_bound(data.get("min_plays"))
    max_plays = _play_bound(data.get("max_plays"))
    if min_plays and max_plays and min_plays > max_plays:
        min_plays, max_plays = max_plays, min_plays
    unheard_only = _bool_flag(data.get("unheard_only"))
    if unheard_only:
        # 0 means unbounded everywhere, so a play floor cannot travel with it.
        min_plays = max_plays = 0
    exclude_compilations = _bool_flag(data.get("exclude_compilations"))

    return AnalyzePromptResponse(
        suggested_genres=suggested_genres,
        suggested_decades=suggested_decades,
        available_genres=available_genres,
        available_decades=available_decades,
        suggested_artists=suggested_artists,
        suggested_excluded_genres=suggested_excluded,
        suggested_exclude_artists=banned_artists,
        suggested_exclude_decades=suggested_excluded_decades,
        suggested_min_year=min_year,
        suggested_max_year=max_year,
        suggested_min_plays=min_plays,
        suggested_max_plays=max_plays,
        suggested_unheard_only=unheard_only,
        suggested_exclude_compilations=exclude_compilations,
        suggested_min_rating=suggested_rating,
        suggested_min_duration_ms=min_duration_ms,
        suggested_max_duration_ms=max_duration_ms,
        style_cues=style_cues,
        available_artists=available_artists,
        available_durations_ms=available_durations,
        available_plays=available_plays,
        unheard_tracks=unheard_tracks,
        year_span=year_span,
        reasoning=data.get("reasoning", ""),
        matched_hints=_hints_from_llm(
            data.get("hint_matches"),
            available_genre_names | available_decade_names | set(artist_names),
        ),
        excluded_genres=[
            g for g in data.get("excluded_genres", []) if g in available_genre_names
        ],
        token_count=response.total_tokens,
        estimated_cost=response.estimated_cost(),
        method="llm",
        confidence=0.0,
    )


def _track_fingerprint(track: Track) -> str:
    """Hash the track metadata that the analysis prompt depends on.

    Cached dimensions are invalidated whenever any of these fields change.
    :data:`DIMENSION_PROMPT_VERSION` is folded in so a changed analysis prompt
    retires the entries the previous prompt produced.
    """
    payload = "\x1f".join(
        [
            f"v{DIMENSION_PROMPT_VERSION}",
            track.title or "",
            track.artist or "",
            track.album or "",
            str(track.year) if track.year else "",
            ",".join(sorted(track.genres or [])),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_dimensions(data: dict[str, Any]) -> list[Dimension]:
    """Build Dimension models from a parsed LLM response.

    Beyond :data:`DIMENSION_MAX` the extra entries are dropped, and duplicate
    ids get a suffix so one malformed response cannot collapse the list.
    """
    if not isinstance(data, dict):
        return []
    used: set[str] = set()
    dimensions: list[Dimension] = []
    for i, d in enumerate(data.get("dimensions", [])):
        if len(dimensions) >= DIMENSION_MAX:
            break
        if not isinstance(d, dict):
            continue
        dim_id = str(d.get("id") or f"dim_{i}")
        while dim_id in used:
            dim_id = f"{dim_id}_2"
        used.add(dim_id)
        dimensions.append(
            Dimension(
                id=dim_id,
                label=str(d.get("label") or "Unknown dimension"),
                description=str(d.get("description", "")),
                category=str(d.get("category") or ""),
            )
        )
    if len(dimensions) < DIMENSION_MIN:
        logger.info("Seed-track analysis returned only %d dimensions", len(dimensions))
    return dimensions


def _hints_from_mapping(mapping: dict[str, list[str]]) -> list[FilterHint]:
    """Convert local classifier evidence (keyword → genres) into the API shape."""
    return [
        FilterHint(hint=keyword, matches=list(genres))
        for keyword, genres in list(mapping.items())[:MAX_HINT_EVIDENCE]
    ]


def _hints_from_llm(raw: Any, available: set[str]) -> list[FilterHint]:
    """Keep only hint phrases whose matches are genres the library really has."""
    hints: list[FilterHint] = []
    for entry in (raw if isinstance(raw, list) else [])[:MAX_HINT_EVIDENCE]:
        if not isinstance(entry, dict):
            continue
        phrase = str(entry.get("hint") or "").strip()
        if not phrase:
            continue
        matches = [m for m in (entry.get("matches") or []) if m in available]
        hints.append(FilterHint(hint=phrase, matches=matches))
    return hints


def _analysis_model(llm_client) -> str:
    """Return the configured analysis model name ("" when unknown)."""
    return str(getattr(getattr(llm_client, "config", None), "model_analysis", "") or "")


def _load_cached_dimensions(
    rating_key: str, model: str, fingerprint: str
) -> list[dict[str, Any]] | None:
    """Read cached dimensions, treating any cache failure as a miss."""
    try:
        return library_cache.get_dimensions(
            rating_key, model=model, fingerprint=fingerprint
        )
    except Exception:
        logger.warning("Dimension cache unavailable for track %s", rating_key, exc_info=True)
        return None


def _store_cached_dimensions(
    rating_key: str, dimensions: list[dict[str, Any]], model: str, fingerprint: str
) -> None:
    """Best-effort write of dimensions to the local cache."""
    try:
        library_cache.save_dimensions(
            rating_key, dimensions, model=model, fingerprint=fingerprint
        )
    except Exception:
        logger.warning("Failed to cache dimensions for track %s", rating_key, exc_info=True)


def analyze_track(track: Track, use_cache: bool = True) -> AnalyzeTrackResponse:
    """Analyze a seed track to extract musical dimensions.

    Results are cached in the local library database keyed by rating key,
    analysis model, and a metadata fingerprint. Re-analyzing the same track
    with the same model costs zero tokens.

    Args:
        track: Track to analyze
        use_cache: When False, always call the LLM (used for cache-busting).

    Returns:
        AnalyzeTrackResponse with track and dimensions

    Raises:
        ValueError: If LLM response cannot be parsed
        RuntimeError: If LLM client is not initialized
    """
    llm_client = get_llm_client()

    if not llm_client:
        raise RuntimeError("LLM client not initialized")

    fingerprint = _track_fingerprint(track)
    model = _analysis_model(llm_client)

    if use_cache:
        cached = _load_cached_dimensions(track.rating_key, model, fingerprint)
        if cached:
            logger.info("Dimension cache hit for track %s (%s)", track.rating_key, model)
            return AnalyzeTrackResponse(
                track=track,
                dimensions=_parse_dimensions({"dimensions": cached}),
                token_count=0,
                estimated_cost=0.0,
                cached=True,
            )

    # Build analysis prompt
    analysis_prompt = f"""Analyze this track:
Title: {track.title}
Artist: {track.artist}
Album: {track.album}
Year: {track.year or "Unknown"}
Genres: {", ".join(track.genres) if track.genres else "Unknown"}

Identify 8-12 specific musical dimensions that make this track distinctive.
Cover at least four descriptor categories and say what concrete evidence in
this recording supports each one."""

    # Call LLM
    response = llm_client.analyze(analysis_prompt, TRACK_ANALYSIS_SYSTEM)

    # Parse response
    data = llm_client.parse_json_response(response)

    dimensions = _parse_dimensions(data)

    if use_cache and dimensions:
        _store_cached_dimensions(
            track.rating_key,
            [d.model_dump() for d in dimensions],
            model,
            fingerprint,
        )

    return AnalyzeTrackResponse(
        track=track,
        dimensions=dimensions,
        token_count=response.total_tokens,
        estimated_cost=response.estimated_cost(),
        cached=False,
    )
