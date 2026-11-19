"""Playlist generation with library validation."""

import json
import logging
import math
import time
from collections.abc import Generator
from datetime import datetime

from backend.llm_client import get_llm_client
from backend.models import GenerateResponse, Track
from backend.plex_client import PlexQueryError, get_plex_client
from backend.prompt_classifier import (
    extract_decade_hints,
    extract_genre_hint_groups,
    extract_genre_hints,
    is_genre_keyword,
)
from backend import library_cache
from backend import agent_reasoning

logger = logging.getLogger(__name__)

# Maximum number of candidate tracks ever sent to the LLM after pre-ranking.
PRE_RANK_LIMIT = 500


# =============================================================================
# Dynamic Track Budget
# =============================================================================
#
# How many *options* the LLM needs depends on how specific the request is. A
# concrete genre/decade filter already narrows the library, so the LLM can pick
# from fewer candidates; a vague mood/activity prompt needs a wider pool. The
# budget interpolates between BUDGET_FLOOR and the ceiling (the user's cap, the
# pre-rank limit, or the candidate pool) using a 0-1 specificity score.

BUDGET_FLOOR = 60          # Never send fewer candidate tracks than this
MIN_TRACKS_PER_PICK = 6    # Scale the floor with the requested playlist size

_SPECIFICITY_EXPLICIT_GENRE = 0.6  # One or two explicit genre filters
_SPECIFICITY_BROAD_GENRE = 0.4     # Many explicit genre filters (broader)
_SPECIFICITY_DECADE = 0.2          # An explicit decade filter
_SPECIFICITY_ARTIST = 0.3          # Naming performers narrows the library hard
_SPECIFICITY_EXCLUSION = 0.2       # Ruling a genre/artist/decade out narrows it too
_SPECIFICITY_WINDOW = 0.2          # A duration, year or play-count window
_SPECIFICITY_FLAG = 0.1            # A yes/no narrowing request (unheard, no compilations)


def _prompt_specificity(
    prompt: str | None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    artists: list[str] | None = None,
    exclude_genres: list[str] | None = None,
    min_duration_ms: int = 0,
    max_duration_ms: int = 0,
    min_rating: int = 0,
    exclude_artists: list[str] | None = None,
    exclude_decades: list[str] | None = None,
    min_year: int = 0,
    max_year: int = 0,
    min_plays: int = 0,
    max_plays: int = 0,
    unheard_only: bool = False,
    exclude_compilations: bool = False,
) -> float:
    """Score how *specific* a request is, from 0.0 (vague) to 1.0 (narrow).

    Concrete genre/decade cues raise the score; so do named artists, genre/artist/
    decade exclusions, a duration/year/play-count window, a rating floor and the
    yes-no narrowing requests (unheard-only, drop compilations), since all of
    those pre-narrow the pool the LLM has to choose from. A mood/activity prompt
    with no such cue stays low so the LLM gets a wider pool of options.

    The year and play-count windows share the duration window's band: they trim
    the candidate pool just as hard. A yes/no flag scores lower because it only
    removes one class of rows (never-played tracks, or the Various-Artists ones).

    Args:
        prompt: Natural-language request (``None`` for the seed-track flow).
        genres: Explicitly selected genre filters.
        decades: Explicitly selected decade filters.
        artists: Explicitly selected artist filters.
        exclude_genres / exclude_artists / exclude_decades: Names the request
            rules out.
        min_duration_ms / max_duration_ms: Track-length window in force.
        min_rating: User-rating floor in force (0 = none).
        min_year / max_year: Release-year window in force (0 = unbounded).
        min_plays / max_plays: Plex play-count window in force (0 = unbounded).
        unheard_only: Only never-played tracks.
        exclude_compilations: Various-Artists compilations dropped.

    Returns:
        Specificity score, capped at 1.0.
    """
    specificity = 0.0

    if prompt:
        prompt_lower = prompt.lower()
        keyword_groups = extract_genre_hint_groups(prompt_lower)
        genre_keywords = [k for k in keyword_groups if is_genre_keyword(k)]
        mood_keywords = [k for k in keyword_groups if not is_genre_keyword(k)]

        if genre_keywords:
            specificity += 0.5 * min(1.0, len(genre_keywords) / 2)
        if extract_decade_hints(prompt_lower):
            specificity += 0.25
        elif mood_keywords and not genre_keywords:
            specificity += 0.15

    if genres:
        explicit = (
            _SPECIFICITY_EXPLICIT_GENRE
            if len(genres) <= 2
            else _SPECIFICITY_BROAD_GENRE
        )
        specificity = max(specificity, explicit)

    if decades:
        specificity += _SPECIFICITY_DECADE

    if artists:
        specificity += _SPECIFICITY_ARTIST
    if exclude_genres or exclude_artists or exclude_decades:
        specificity += _SPECIFICITY_EXCLUSION
    if (
        min_duration_ms
        or max_duration_ms
        or min_rating
        or min_year
        or max_year
        or min_plays
        or max_plays
        or unheard_only
    ):
        specificity += _SPECIFICITY_WINDOW
    if exclude_compilations:
        specificity += _SPECIFICITY_FLAG

    return min(1.0, specificity)


def compute_track_budget(
    prompt: str | None = None,
    requested_max: int = 0,
    track_count: int = 25,
    pool_size: int | None = None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    artists: list[str] | None = None,
    exclude_genres: list[str] | None = None,
    min_duration_ms: int = 0,
    max_duration_ms: int = 0,
    min_rating: int = 0,
    exclude_artists: list[str] | None = None,
    exclude_decades: list[str] | None = None,
    min_year: int = 0,
    max_year: int = 0,
    min_plays: int = 0,
    max_plays: int = 0,
    unheard_only: bool = False,
    exclude_compilations: bool = False,
) -> int:
    """Return how many candidate tracks to send to the LLM for this request.

    The ceiling is the tighter of :data:`PRE_RANK_LIMIT`, the user's
    ``requested_max`` (``0`` = no user limit), and *pool_size* when known. The
    result slides from the ceiling (vague prompt) down toward the floor
    (specific prompt), where the floor scales with *track_count* so the LLM
    always has enough options for the requested playlist size.

    Args:
        prompt: Natural-language request (``None`` for the seed-track flow).
        requested_max: The user's/``max_tracks_to_ai`` cap (``0`` = no limit).
        track_count: Number of tracks the user asked for.
        pool_size: Size of the candidate pool, if known.
        genres: Explicitly selected genre filters.
        decades: Explicitly selected decade filters.
        artists: Explicitly selected artist filters.
        exclude_genres / exclude_artists / exclude_decades: Filters that drop
            named genres, artists and decades.
        min_duration_ms / max_duration_ms: Track-length window in force.
        min_rating: User-rating floor in force (0 = none).
        min_year / max_year: Release-year window in force (0 = unbounded).
        min_plays / max_plays: Plex play-count window in force (0 = unbounded).
        unheard_only: Only never-played tracks.
        exclude_compilations: Various-Artists compilations dropped.

    Returns:
        Number of tracks to keep after pre-ranking (0 only if the pool is empty).
    """
    ceiling = PRE_RANK_LIMIT
    if requested_max and requested_max > 0:
        ceiling = min(ceiling, requested_max)
    if pool_size is not None:
        ceiling = min(ceiling, pool_size)

    floor = max(BUDGET_FLOOR, track_count * MIN_TRACKS_PER_PICK)
    if ceiling <= floor:
        return max(0, ceiling)

    specificity = _prompt_specificity(
        prompt,
        genres,
        decades,
        artists,
        exclude_genres,
        min_duration_ms,
        max_duration_ms,
        min_rating,
        exclude_artists,
        exclude_decades,
        min_year,
        max_year,
        min_plays,
        max_plays,
        unheard_only,
        exclude_compilations,
    )
    budget = ceiling - (ceiling - floor) * specificity
    return max(floor, min(ceiling, round(budget)))


# =============================================================================
# Rule-Based Pre-Ranking
# =============================================================================

# Genre/decade hint dictionaries live in backend.prompt_classifier so that
# prompt analysis and pre-ranking share a single source of truth.


def _year_to_decade(year: int) -> str:
    """Convert a year to its decade string (e.g., 1985 -> "1980s")."""
    decade = (year // 10) * 10
    return f"{decade}s"


def _parse_timestamp(ts: str | None) -> float | None:
    """Parse an ISO timestamp string to a Unix epoch float. Returns None on failure."""
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _recency_score(last_played_ts: float | None, now: float, half_life_hours: float = 72) -> float:
    """Return 0-1 recency score. Recently played = higher.

    Uses exponential decay with a configurable half-life (default 72 hours).
    A track played 72 hours ago scores 0.5; 144 hours ago scores 0.25, etc.
    """
    if last_played_ts is None:
        return 0.0
    age_hours = max(0.0, (now - last_played_ts) / 3600)
    return math.exp(-age_hours / half_life_hours * math.log(2))


def pre_rank_tracks(
    tracks: list[Track],
    prompt: str | None = None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    max_tracks: int = PRE_RANK_LIMIT,
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
) -> list[Track]:
    """Score and rank tracks using rule-based heuristics, returning the top N.

    Scoring components (total max = 100, plus bonuses and penalties):
        1. Genre match   (0-40)  -- how well the track's genres match prompt hints
        2. Decade match  (0-20)  -- whether the track's year falls in hinted decades
        3. User rating   (0-25)  -- from the local library cache (0-10 scale)
        4. Recency       (0-15)  -- exponential decay on last played (72h half-life)
        5. Artist match  (0-30)  -- the track is one of the named performers
        6. Length window (+-10)  -- inside vs. outside the requested duration window
        7. Excluded genre (-40)  -- the track carries a genre the request ruled out
        8. Year window   (+-10)  -- inside vs. outside the requested release years
        9. Banned artist (-30)   -- the track is one of the excluded performers
        10. Banned decade (-20)  -- the track's year falls in an excluded decade
        11. Play window  (+-10)  -- inside vs. outside the familiarity request
        12. Curation affinity (0-15) -- how much the agent's own stored
            selection reasons for the track (mined tags, see
            :mod:`backend.agent_reasoning`) overlap the request's hints

    Components 5-11 mostly matter on the live-Plex path, where those filters can
    only be applied partially; on the cache path the pool is already narrowed.
    The play-count component needs the local cache's ``view_count``, so a track
    the cache has no row for simply does not move on that axis.

    If no prompt hints are found, the function falls back to rating + recency
    only, so the LLM still gets a sensible ordering.

    Args:
        tracks: Candidate tracks (already filtered by the Plex client/cache).
        prompt: The user's natural-language prompt (used for hint extraction).
        genres: Optional explicit genre filters -- additional hard hints.
        decades: Optional explicit decade filters -- additional hard hints.
        max_tracks: Maximum number of tracks to return (0 = no limit).
        artists: Optional explicit artist filters -- strong preference bonus.
        exclude_genres: Genres the request rules out -- applied as a penalty so a
                        stray match cannot displace a clean candidate.
        min_duration_ms / max_duration_ms: Requested track-length window.
        exclude_artists: Performers the request rules out -- same penalty shape
                         as the artist-match bonus.
        exclude_decades: Decades the request rules out -- same penalty shape as
                         the decade-match bonus.
        min_year / max_year: Requested release-year window (0 = unbounded).
        min_plays / max_plays: Requested Plex play-count window (0 = unbounded).
        unheard_only: Prefer tracks with a zero play count.

    Returns:
        A new list of up to *max_tracks* tracks, sorted by score descending.
        Ties are broken by rating_key for stable, deterministic ordering.
    """
    if not tracks:
        return []
    if max_tracks <= 0 or len(tracks) <= max_tracks:
        return tracks  # No need to rank if we're sending all

    # --- Extract hints from prompt and/or explicit filters ---
    prompt_lower = (prompt or "").lower()
    hinted_genres = extract_genre_hints(prompt_lower)
    hinted_decades = extract_decade_hints(prompt_lower)

    # Merge in explicit filter genres/decades as additional hints
    for g in genres or []:
        hinted_genres.add(g.lower())
    for d in decades or []:
        hinted_decades.add(d)

    excluded_lower = [g.lower() for g in exclude_genres or []]
    window_open = min_duration_ms > 0 or max_duration_ms > 0
    year_open = min_year > 0 or max_year > 0
    play_open = min_plays > 0 or max_plays > 0 or unheard_only

    # --- Fetch cache metadata in one batch ---
    metadata: dict[str, dict] = {}
    try:
        metadata = library_cache.get_metadata_batch([t.rating_key for t in tracks])
    except Exception:
        logger.warning("Pre-rank metadata unavailable; ranking without ratings", exc_info=True)

    # --- Fetch the agent's mined selection-reason tags in one batch -------
    # Past curation is a filter signal: a track whose own stored reasons speak
    # the current request's language has already survived this bar once.
    past_tags: dict[str, set[str]] = {}
    try:
        past_tags = agent_reasoning.get_reason_tags([t.rating_key for t in tracks])
    except Exception:
        logger.warning("Pre-rank curation tags unavailable; ranking without affinity", exc_info=True)
    hinted_tags: set[str] = hinted_genres | hinted_decades

    now = time.time()
    scored: list[tuple[float, Track]] = []

    for track in tracks:
        score = 0.0

        # 1. Genre match (0-40)
        if hinted_genres:
            track_genres = {g.lower() for g in track.genres}
            matched = hinted_genres & track_genres
            if matched:
                # Proportional: how many of the hinted genres does this track have?
                score += min(len(matched) / len(hinted_genres), 1.0) * 40
            else:
                # Substring / partial match: does any track genre contain a hint
                # or vice versa? Gives a smaller bonus so near-misses still rank
                # above completely unrelated tracks.
                for tg in track_genres:
                    for hg in hinted_genres:
                        if hg in tg or tg in hg:
                            score += 10  # partial match bonus (capped at 10)
                            break
                    if score >= 10:
                        break

        # 2. Decade match (0-20)
        if hinted_decades and track.year:
            track_decade = _year_to_decade(track.year)
            if track_decade in hinted_decades:
                score += 20

        # 12. Curation affinity (0-15)
        if past_tags and hinted_tags:
            mine = past_tags.get(track.rating_key)
            if mine:
                overlap = mine & hinted_tags
                if overlap:
                    # Proportional like the genre match: reasons that speak
                    # more of the request's vocabulary move the track further
                    # into the AI pool.
                    score += min(len(overlap) / len(hinted_tags), 1.0) * 15

        # 5. Artist match (0-30)
        if artists and library_cache._matches_artist(track.artist, artists):
            score += 30

        # 6. Length window (+-10)
        if window_open:
            length = track.duration_ms or 0
            inside = (min_duration_ms <= 0 or length >= min_duration_ms) and (
                max_duration_ms <= 0 or (length > 0 and length <= max_duration_ms)
            )
            score += 10 if inside else -10

        # 8. Recording-year window (+-10)
        if year_open and track.year:
            inside = (min_year <= 0 or track.year >= min_year) and (
                max_year <= 0 or track.year <= max_year
            )
            score += 10 if inside else -10

        # 10. Banned decade (-20)
        if exclude_decades and track.year and _year_to_decade(track.year) in set(exclude_decades):
            score -= 20

        # 9. Banned artist (-30)
        if exclude_artists and library_cache._matches_artist(track.artist, exclude_artists):
            score -= 30

        # 7. Excluded genre (-40)
        if excluded_lower and any(g in {t.lower() for t in track.genres} for g in excluded_lower):
            score -= 40

        # 3. User rating (0-25)  +  4. Recency (0-15)  +  11. Familiarity window (+-10)
        meta = metadata.get(track.rating_key)
        if meta:
            rating = meta.get("user_rating", 0) or 0
            score += (rating / 10) * 25

            last_played_ts = _parse_timestamp(meta.get("last_viewed_at"))
            recency = _recency_score(last_played_ts, now)
            score += recency * 15

            if play_open:
                # Plex's play counter lives in the cache; a track with no row
                # there has no opinion to give, so it stays where it ranked.
                plays = meta.get("view_count") or 0
                if unheard_only:
                    score += 10 if plays == 0 else -10
                else:
                    inside = (min_plays <= 0 or plays >= min_plays) and (
                        max_plays <= 0 or plays <= max_plays
                    )
                    score += 10 if inside else -10

        scored.append((score, track))

    # Sort by score descending, break ties by rating_key for deterministic ordering
    scored.sort(key=lambda x: (-x[0], x[1].rating_key))

    return [t for _, t in scored[:max_tracks]]


def generate_narrative(
    track_selections: list[dict],
    llm_client,
    user_request: str = "",
) -> tuple[str, str]:
    """Generate a creative title and narrative for the playlist.

    Args:
        track_selections: List of track dicts with artist, title, album, reason
        llm_client: LLM client instance
        user_request: Original user prompt/request for context

    Returns:
        Tuple of (playlist_title with date, narrative)
        On failure, returns ("{Mon YYYY} Playlist", "")
    """
    # Build input for Query 2: track list with reasons
    tracks_with_reasons = "\n".join(
        f"- {sel.get('artist', 'Unknown')} - \"{sel.get('title', 'Unknown')}\": {sel.get('reason', 'Selected for this playlist')}"
        for sel in track_selections[:15]  # Limit to first 15 for context efficiency
    )

    # Include user request for context
    if user_request:
        narrative_prompt = f"User's request: {user_request}\n\nSelected tracks:\n{tracks_with_reasons}"
    else:
        narrative_prompt = f"Selected tracks:\n{tracks_with_reasons}"

    # Get current month/year for title suffix
    date_suffix = datetime.now().strftime("%b %Y")
    fallback_title = f"{date_suffix} Playlist"

    try:
        # Use analysis model for better creative writing quality
        response = llm_client.analyze(narrative_prompt, NARRATIVE_SYSTEM)
        result = llm_client.parse_json_response(response)

        # Handle array-wrapped responses (some LLMs wrap in [])
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        if not isinstance(result, dict):
            logger.warning("Narrative response not a dict: %s", type(result).__name__)
            return fallback_title, ""

        raw_title = result.get("title", "").strip()

        # Try common alternate keys for narrative
        narrative = (
            result.get("narrative")
            or result.get("description")
            or result.get("text")
            or result.get("content")
            or ""
        ).strip()

        # Log if we got title but no narrative (helps debug)
        if raw_title and not narrative:
            logger.warning("Narrative missing from response. Keys: %s", list(result.keys()))

        # Append date to title
        if raw_title:
            playlist_title = f"{raw_title} - {date_suffix}"
        else:
            playlist_title = fallback_title

        return playlist_title, narrative

    except Exception as e:
        logger.warning("Narrative generation failed: %s", e)
        return fallback_title, ""


def _cached_track_to_model(cached: dict) -> Track:
    """Convert a cached track dict to a Track model."""
    return Track(
        rating_key=cached["rating_key"],
        title=cached["title"],
        artist=cached["artist"],
        album=cached["album"],
        duration_ms=cached.get("duration_ms") or 0,
        year=cached.get("year"),
        genres=cached.get("genres") or [],
        art_url=f"/api/art/{cached['rating_key']}",
    )


def _past_reasons_for(tracks: list[Track]) -> dict[str, str]:
    """Stored curation reasons replayed for *tracks* ({} when the memory is empty).

    The agent's own earlier argument for a track, not a third-party fact, so the
    candidate legend words it as the agent's own judgement. Budgeted per
    line and per candidate set by :mod:`backend.agent_reasoning`.
    """
    if not tracks or not agent_reasoning.has_reasoning_data():
        return {}
    reasons = agent_reasoning.get_previous_reasons(
        [t.rating_key for t in tracks][: agent_reasoning.MAX_TRACKS_WITH_PAST_REASON]
    )
    return {key: reason[: agent_reasoning.MAX_PAST_REASON_CHARS] for key, reason in reasons.items()}


# =============================================================================
# Candidate Pool Relaxation
# =============================================================================

# The facet names :func:`compute_track_budget` reads. ``exclude_live`` says which
# tracks qualify but says nothing about how specific the request is, so it stays
# out of the budget's kwargs.
BUDGET_FACETS = (
    "genres",
    "decades",
    "artists",
    "exclude_genres",
    "exclude_artists",
    "exclude_decades",
    "min_year",
    "max_year",
    "min_plays",
    "max_plays",
    "unheard_only",
    "exclude_compilations",
    "min_duration_ms",
    "max_duration_ms",
    "min_rating",
)


def candidate_facets(
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    artists: list[str] | None = None,
    exclude_genres: list[str] | None = None,
    exclude_artists: list[str] | None = None,
    exclude_decades: list[str] | None = None,
    min_year: int = 0,
    max_year: int = 0,
    min_plays: int = 0,
    max_plays: int = 0,
    unheard_only: bool = False,
    exclude_compilations: bool = False,
    min_rating: int = 0,
    min_duration_ms: int = 0,
    max_duration_ms: int = 0,
    exclude_live: bool = True,
    rating_user: str = "",
) -> dict:
    """Spell the cache-branch filters once, as one splittable dict.

    Counting and fetching must agree on the facet names and on the neutral value
    for "off", so both read this dict instead of repeating the argument list; the
    relaxation ladder in turn refuses any key :data:`library_cache.FILTER_NEUTRALS`
    does not know how to clear.

    ``rating_user`` is spelled here too, but it is never in a relaxation tier:
    switching whose ratings are read re-reads the same rows, so it cannot widen a
    pool, and dropping it would answer a member's rating floor with the admin's
    column while still claiming the filters were honoured.

    Returns:
        The filter kwargs :func:`_get_tracks_from_cache_or_plex` takes, with empty
        lists spelled ``None`` as the cache functions treat them.
    """
    return {
        "genres": genres or None,
        "decades": decades or None,
        "artists": artists or None,
        "exclude_genres": exclude_genres or None,
        "exclude_artists": exclude_artists or None,
        "exclude_decades": exclude_decades or None,
        "min_year": min_year,
        "max_year": max_year,
        "min_plays": min_plays,
        "max_plays": max_plays,
        "unheard_only": unheard_only,
        "exclude_compilations": exclude_compilations,
        "min_rating": min_rating,
        "min_duration_ms": min_duration_ms,
        "max_duration_ms": max_duration_ms,
        "exclude_live": exclude_live,
        "rating_user": rating_user or "",
    }


def relax_candidate_pool(
    facets: dict,
    prompt: str | None = None,
    track_count: int = 25,
    requested_max: int = 0,
    count: int | None = None,
) -> dict:
    """Widen the cache filters until the AI step's candidate budget is reachable.

    The target is the budget measured against an unbounded pool: the pool itself is
    what is being looked for, so feeding its own size back in would stop the ladder
    before it started. The ladder gives up the library's bans, then its rating and
    play floors, then year and duration windows, then compilations and live rows,
    and only at the end the genre/decade/artist whitelist - and that last
    one only when the pool cannot even fill the requested playlist, so a narrow
    request that is already answered stays narrow.

    Returns:
        :func:`library_cache.relax_filters_for_target`'s dict (``filters``,
        ``matching``, ``asked``, ``target``, ``tier``, ``relaxed``, ``short_by``).
    """
    want = compute_track_budget(
        prompt=prompt,
        requested_max=requested_max,
        track_count=track_count,
        **{name: facets[name] for name in BUDGET_FACETS if name in facets},
    )
    return library_cache.relax_filters_for_target(
        facets, want, count=count, identity_floor=track_count
    )


def _get_tracks_from_cache_or_plex(
    plex_client,
    genres: list[str] | None,
    decades: list[str] | None,
    exclude_live: bool,
    min_rating: int,
    max_tracks_to_ai: int,
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
    rating_user: str = "",
) -> list[Track]:
    """Get tracks from cache if available, otherwise from Plex.

    Args:
        plex_client: Connected Plex client, used when the cache is empty.
        genres: Genre filter (OR matching).
        decades: Decade filter (OR matching).
        exclude_live: Drop live recordings.
        min_rating: User-rating floor (0 = none).
        max_tracks_to_ai: Candidate cap (0 = library default).
        artists: Artist filter (OR matching).
        exclude_genres: Genres to drop.
        min_duration_ms / max_duration_ms: Track-length window (0 = unbounded).
        exclude_artists: Artist names to drop.
        exclude_decades: Decades to drop (undated tracks survive).
        min_year / max_year: Release-year window (0 = unbounded).
        min_plays / max_plays: Plex play-count window (0 = unbounded).
        unheard_only: Require a zero play count.
        exclude_compilations: Drop the Various-Artists rows.
        rating_user: Plex account whose stars the rating facets read — "" or
            ``__owner__`` means the connecting admin's ``user_rating`` column, a
            household member's exact title or a shared friend's username reads the
            cache's ``user_ratings`` rows instead. Cache branch only: a live Plex
            walk sees one account's stars and cannot name another's.

    Returns:
        List of Track objects
    """
    has_filters = bool(
        genres
        or decades
        or artists
        or exclude_genres
        or exclude_artists
        or exclude_decades
        or min_rating > 0
        or min_duration_ms > 0
        or max_duration_ms > 0
        or min_year > 0
        or max_year > 0
        or min_plays > 0
        or max_plays > 0
        or unheard_only
        or exclude_compilations
    )
    effective_limit = max_tracks_to_ai if max_tracks_to_ai > 0 else 2000

    # Try cache first
    if library_cache.has_cached_tracks():
        logger.info("Using cached tracks for generation")
        wanted_ratings = rating_user or ""
        if wanted_ratings:
            # A member's stars are not on the admin's walk, so their rows are loaded
            # here once and every later request pays nothing. A failed scan may not
            # read as "this member rated nothing": the effective rating is
            # COALESCE(user_ratings, tracks.user_rating), so missing rows would answer
            # every track with the admin's column and a rating floor would then drop
            # exactly the tracks the member had rated.
            try:
                library_cache.ensure_user_ratings(wanted_ratings, plex_client)
            except PlexQueryError as exc:
                logger.warning(
                    "Rating facets fall back to the admin's stars: %s", exc
                )
                wanted_ratings = ""
        cached_tracks = library_cache.get_tracks_by_filters(
            genres=genres,
            decades=decades,
            artists=artists,
            exclude_genres=exclude_genres,
            min_rating=min_rating,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            exclude_live=exclude_live,
            limit=effective_limit,
            exclude_artists=exclude_artists,
            exclude_decades=exclude_decades,
            min_year=min_year,
            max_year=max_year,
            min_plays=min_plays,
            max_plays=max_plays,
            unheard_only=unheard_only,
            exclude_compilations=exclude_compilations,
            rating_user=wanted_ratings,
        )
        return [_cached_track_to_model(t) for t in cached_tracks]

    # Fall back to Plex. rating_user cannot be honoured here: another account's
    # stars exist only in the cache's user_ratings table, and a live Plex walk sees
    # the connecting admin's column, so the facets stay as asked against that one.
    logger.info("Cache empty, fetching from Plex")
    if not has_filters:
        return plex_client.get_random_tracks(
            count=effective_limit,
            exclude_live=exclude_live,
        )
    return plex_client.get_tracks_by_filters(
        genres=genres,
        decades=decades,
        exclude_live=exclude_live,
        min_rating=min_rating,
        limit=effective_limit,
        artists=artists,
        exclude_genres=exclude_genres,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        exclude_artists=exclude_artists,
        exclude_decades=exclude_decades,
        min_year=min_year,
        max_year=max_year,
        min_plays=min_plays,
        max_plays=max_plays,
        unheard_only=unheard_only,
        exclude_compilations=exclude_compilations,
    )


def generate_playlist_stream(
    prompt: str | None = None,
    seed_track: Track | None = None,
    selected_dimensions: list[str] | None = None,
    additional_notes: str | None = None,
    refinement_answers: list[str | None] | None = None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    track_count: int = 25,
    exclude_live: bool = True,
    min_rating: int = 0,
    max_tracks_to_ai: int = 500,
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
    style_cues: list[str] | None = None,
    rating_user: str = "",
) -> Generator[str, None, None]:
    """Generate a playlist with streaming progress updates.

    The hard filters (genres, decades, artists, genre/artist/decade exclusions,
    a release-year window, a play-count window, an unheard-only flag, a
    compilation drop, a rating floor and a duration window) narrow which tracks
    are candidates; ``style_cues`` are the soft half of an analysis — mood,
    tempo and texture phrases the library has no column for — so they are
    written into the curator prompt instead of enforced against metadata. ``rating_user``
    names whose stars a rating floor reads: "" is the connecting admin's ``user_rating``
    column, any other Plex key reads that account's ``user_ratings`` rows (scanned into the
    cache on the first ask) and is honoured on the cache branch only — a live Plex walk sees
    one account's stars and cannot name another's.

    Yields SSE-formatted events with progress updates and final result.
    """
    def emit(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        logger.info("Starting playlist generation (streaming)")
        llm_client = get_llm_client()
        plex_client = get_plex_client()

        if not llm_client:
            yield emit("error", {"message": "LLM client not initialized"})
            return
        if not plex_client:
            yield emit("error", {"message": "Plex client not initialized"})
            return

        has_filters = bool(
            genres
            or decades
            or artists
            or exclude_genres
            or exclude_artists
            or exclude_decades
            or min_rating > 0
            or min_duration_ms > 0
            or max_duration_ms > 0
            or min_year > 0
            or max_year > 0
            or min_plays > 0
            or max_plays > 0
            or unheard_only
            or exclude_compilations
        )

        # Step 1: Size the pool, then fetch it. The strictest reading of a request
        # is not always what the cache can honour: one rating floor the analyzer
        # invented ("best of all time" became min_rating >= 8) once cut 337 Holiday
        # tracks to 5, which reads back as a thin playlist rather than a broken one.
        # On the cache branch only, the ladder in
        # :mod:`backend.library_cache` gives up the least specific constraint first
        # until the candidate budget is reachable, and the tier names are reported
        # so the playlist can say which filters it actually honoured. The live-Plex
        # fallback keeps the filters exactly as asked: it has no counts to measure
        # against.
        using_cache = library_cache.has_cached_tracks()
        facets = candidate_facets(
            genres=genres,
            decades=decades,
            artists=artists,
            exclude_genres=exclude_genres,
            exclude_artists=exclude_artists,
            exclude_decades=exclude_decades,
            min_year=min_year,
            max_year=max_year,
            min_plays=min_plays,
            max_plays=max_plays,
            unheard_only=unheard_only,
            exclude_compilations=exclude_compilations,
            min_rating=min_rating,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            exclude_live=exclude_live,
            rating_user=rating_user,
        )
        ladder = None
        if using_cache:
            ladder = relax_candidate_pool(
                facets,
                prompt=prompt,
                track_count=track_count,
                requested_max=max_tracks_to_ai,
            )
            facets = ladder["filters"]
            if ladder["tier"] != "as asked":
                yield emit(
                    "progress",
                    {
                        "step": "relaxing",
                        "message": (
                            f"Only {ladder['asked']:,} tracks matched as asked "
                            f"(the AI step wants {ladder['target']:,}), so widening "
                            f"the filters to {ladder['tier']} "
                            f"[{ladder['matching']:,} tracks]"
                        ),
                    },
                )
            yield emit("progress", {"step": "fetching", "message": "Loading tracks from cache..."})
        elif not has_filters:
            yield emit("progress", {"step": "fetching", "message": "Sampling random tracks from library..."})
        else:
            yield emit("progress", {"step": "fetching", "message": "Fetching tracks from library..."})

        logger.info(
            "Fetching tracks: genres=%s, decades=%s, artists=%s, exclude_genres=%s, "
            "exclude_artists=%s, exclude_decades=%s, year=%s..%s, plays=%s..%s, "
            "unheard_only=%s, exclude_compilations=%s, min_rating=%s, duration=%s..%s, "
            "using_cache=%s",
            genres,
            decades,
            artists,
            exclude_genres,
            exclude_artists,
            exclude_decades,
            min_year,
            max_year,
            min_plays,
            max_plays,
            unheard_only,
            exclude_compilations,
            min_rating,
            min_duration_ms,
            max_duration_ms,
            using_cache,
        )
        if ladder:
            logger.info(
                "Filter relaxation: asked=%s, target=%s, tier=%s, relaxed=%s, pool=%s, short_by=%s",
                ladder["asked"],
                ladder["target"],
                ladder["tier"],
                ladder["relaxed"],
                ladder["matching"],
                ladder["short_by"],
            )
        try:
            filtered_tracks = _get_tracks_from_cache_or_plex(
                plex_client=plex_client,
                max_tracks_to_ai=max_tracks_to_ai,
                **facets,
            )
        except PlexQueryError as e:
            yield emit("error", {"message": f"Plex server error: {e}"})
            return

        logger.info("Got %d tracks", len(filtered_tracks))

        if not filtered_tracks:
            yield emit("error", {"message": "No tracks match the selected filters. Try broadening your selection."})
            return

        # Step 1b: Size the candidate set and pre-rank locally.
        # A specific genre/decade/artist request can pick from a smaller pool, while a
        # vague mood prompt keeps a wider one. This trimming costs no tokens.
        budget = compute_track_budget(
            prompt=prompt,
            requested_max=max_tracks_to_ai,
            track_count=track_count,
            pool_size=len(filtered_tracks),
            genres=genres,
            decades=decades,
            artists=artists,
            exclude_genres=exclude_genres,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            min_rating=min_rating,
            exclude_artists=exclude_artists,
            exclude_decades=exclude_decades,
            min_year=min_year,
            max_year=max_year,
            min_plays=min_plays,
            max_plays=max_plays,
            unheard_only=unheard_only,
            exclude_compilations=exclude_compilations,
        )
        if budget > 0 and len(filtered_tracks) > budget:
            pool_size = len(filtered_tracks)
            yield emit(
                "progress",
                {"step": "pre_ranking", "message": f"Pre-ranking {pool_size} tracks locally..."},
            )
            filtered_tracks = pre_rank_tracks(
                filtered_tracks,
                prompt=prompt,
                genres=genres,
                decades=decades,
                max_tracks=budget,
                artists=artists,
                exclude_genres=exclude_genres,
                min_duration_ms=min_duration_ms,
                max_duration_ms=max_duration_ms,
                exclude_artists=exclude_artists,
                exclude_decades=exclude_decades,
                min_year=min_year,
                max_year=max_year,
                min_plays=min_plays,
                max_plays=max_plays,
                unheard_only=unheard_only,
            )
            logger.info(
                "Pre-ranked candidates from %d to %d (budget=%d)",
                pool_size,
                len(filtered_tracks),
                budget,
            )

        # Step 2: Report track count (sampling already done server-side)
        if has_filters:
            yield emit("progress", {"step": "filtering", "message": f"Using {len(filtered_tracks)} tracks..."})
        else:
            yield emit("progress", {"step": "filtering", "message": f"Using {len(filtered_tracks)} random tracks..."})

        # Step 3: Build track list
        yield emit("progress", {"step": "preparing", "message": f"Preparing {len(filtered_tracks)} tracks for AI..."})

        # A "Curated before:" tail is this agent's own earlier argument for a track
        # (stored by record_run in the reasoning DB), reused so the model does not
        # have to re-derive it. Without stored memory no tail is written.
        past_reasons = _past_reasons_for(filtered_tracks)
        if past_reasons:
            legend = []
            legend.append(
                "Lines marked \"Curated before:\" carry the reason this library's curator "
                "picked that track in an earlier playlist; it is the agent's own judgement, "
                "honour it when the request fits and ignore it when the request asks for something else."
            )
            generation_parts_note = " ".join(legend)
        else:
            generation_parts_note = ""
        track_list = "\n".join(
            f"{i+1}. {t.artist} - {t.title} ({t.album}, {t.year or 'Unknown year'})"
            + (f" | Curated before: {past_reasons[t.rating_key]}" if past_reasons.get(t.rating_key) else "")
            for i, t in enumerate(filtered_tracks)
        )

        # Build the generation prompt
        generation_parts = []

        if prompt:
            generation_parts.append(f"User's request: {prompt}")

        if style_cues:
            # The library has no column for these, so they steer the picks rather
            # than narrow the pool: the curator still chooses from the same list.
            cues = [str(c).strip()[:120] for c in style_cues if str(c).strip()][:8]
            if cues:
                generation_parts.append(
                    "Style cues from the request (prefer tracks that also satisfy these): "
                    + "; ".join(cues)
                )

        if seed_track:
            generation_parts.append(
                f"Seed track: {seed_track.title} by {seed_track.artist} "
                f"(from {seed_track.album}, {seed_track.year or 'Unknown year'})"
            )
            if selected_dimensions:
                generation_parts.append(f"Explore these dimensions: {', '.join(selected_dimensions)}")

        if additional_notes:
            generation_parts.append(f"Additional notes: {additional_notes}")

        if refinement_answers:
            answered = [a for a in refinement_answers if a]
            if answered:
                generation_parts.append(f"User preferences: {', '.join(answered)}")

        if generation_parts_note:
            generation_parts.append(generation_parts_note)

        # Ask for what the model was actually offered. Told to "Select 200" out of a
        # 5-line list, a small local curator answers with an empty array instead of the
        # five it can honour (measured against the mlx_lm endpoint: "Select 12" of 5 ->
        # '[]' at 2 output tokens, "Select 5" -> all five), which reads back as a valid
        # run that matched nothing. A playlist can never be longer than the pool, so the
        # count is clamped here rather than left for the model to refuse.
        requested_picks = min(track_count, len(filtered_tracks))
        generation_parts.append(f"\nSelect {requested_picks} tracks from this library:\n{track_list}")

        generation_prompt = "\n\n".join(generation_parts)

        # Step 4: Call LLM
        yield emit("progress", {"step": "ai_working", "message": "AI is curating your playlist..."})

        logger.info("Calling LLM with prompt length: %d chars", len(generation_prompt))
        response = llm_client.generate(generation_prompt, GENERATION_SYSTEM)
        logger.info("LLM response received: %d input, %d output tokens", response.input_tokens, response.output_tokens)

        # Step 5: Parse response
        yield emit("progress", {"step": "parsing", "message": "Parsing AI selections..."})

        track_selections = llm_client.parse_json_response(response)

        if not isinstance(track_selections, list):
            yield emit("error", {"message": "LLM returned invalid track selection format"})
            return

        if not track_selections:
            # The JSON was well formed and empty, so the curator refused the job rather
            # than failing to answer - worth naming, since the run below "succeeds".
            logger.warning(
                "Curator selected nothing from %d candidates (asked for %d)",
                len(filtered_tracks),
                requested_picks,
            )

        # Step 6: Match tracks
        yield emit("progress", {"step": "matching", "message": f"Matching {len(track_selections)} selections to library..."})

        matched_tracks: list[Track] = []
        used_keys: set[str] = set()
        track_reasons: dict[str, str] = {}

        if seed_track:
            used_keys.add(seed_track.rating_key)

        for selection in track_selections:
            if len(matched_tracks) >= track_count:
                break

            artist = selection.get("artist", "")
            title = selection.get("title", "")
            reason = selection.get("reason", "")

            for track in filtered_tracks:
                if track.rating_key in used_keys:
                    continue

                if _tracks_match(artist, title, track):
                    matched_tracks.append(track)
                    used_keys.add(track.rating_key)
                    if reason:
                        track_reasons[track.rating_key] = reason
                    break

        # Step 7: Generate narrative. The liner note names 3-4 specific songs, so it is
        # written from the tracks that actually made the playlist rather than from
        # whatever the model named: an unmatched selection list otherwise advertises
        # songs this playlist does not contain (a 0-track run described 'White Christmas'
        # and 'Jingle Bell Rock' for a playlist with no tracks at all).
        narrated = [
            {
                "artist": t.artist,
                "title": t.title,
                "album": t.album,
                "reason": track_reasons.get(t.rating_key, ""),
            }
            for t in matched_tracks
        ]
        yield emit("progress", {"step": "narrative", "message": "Writing playlist narrative..."})

        if narrated:
            playlist_title, narrative = generate_narrative(narrated, llm_client, prompt or "")
        else:
            logger.warning(
                "Nothing matched the library (%d candidates, %d selections); "
                "skipping the narrative model",
                len(filtered_tracks),
                len(track_selections),
            )
            playlist_title, narrative = f"{datetime.now().strftime('%b %Y')} Playlist", ""
        logger.info("Generated narrative: title='%s', narrative_len=%d", playlist_title, len(narrative))

        # Emit narrative event for frontend
        yield emit("narrative", {
            "playlist_title": playlist_title,
            "narrative": narrative,
            "track_reasons": track_reasons,
            "user_request": prompt or "",
        })

        # Step 8: Complete
        logger.info("Track matching complete. Matched %d tracks", len(matched_tracks))
        logger.info("Emitting 'Playlist ready!' progress event")
        yield emit("progress", {"step": "complete", "message": "Playlist ready!"})

        logger.info("Building GenerateResponse: tokens=%s, cost=%s",
                    getattr(response, 'total_tokens', 'N/A'),
                    response.estimated_cost() if response else 'N/A')

        try:
            result = GenerateResponse(
                tracks=matched_tracks,
                token_count=response.total_tokens,
                estimated_cost=response.estimated_cost(),
                playlist_title=playlist_title,
                narrative=narrative,
                track_reasons=track_reasons,
                matching_as_asked=ladder["asked"] if ladder else 0,
                relaxed_tiers=ladder["relaxed"] if ladder else [],
                relaxed_to=ladder["tier"] if ladder and ladder["relaxed"] else "",
            )
            logger.info("GenerateResponse built successfully with %d tracks", len(result.tracks))
        except Exception as e:
            logger.exception("Failed to build GenerateResponse: %s", e)
            yield emit("error", {"message": f"Failed to build response: {e}"})
            return

        # Send tracks in batches to work around iOS Safari dropping large SSE events.
        # Safari Mobile has undocumented buffering limits that can cause large events
        # to be silently dropped. Batching keeps each event small (~2KB).
        tracks_data = [t.model_dump(mode="json") for t in result.tracks]
        batch_size = 5
        for i in range(0, len(tracks_data), batch_size):
            batch = tracks_data[i:i + batch_size]
            logger.info("Emitting track batch %d-%d", i, i + len(batch))
            yield emit("tracks", {"batch": batch, "index": i})

        # Save result to history before emitting the final event
        result_id = None
        reasoning_recorded = 0
        try:
            result_type = "seed_playlist" if seed_track else "prompt_playlist"
            # Title: always use the LLM-generated playlist title
            result_title = playlist_title
            # Use first track's rating key for card thumbnail
            first_art_key = matched_tracks[0].rating_key if matched_tracks else None
            # Subtitle: seed playlists show origin track, prompt playlists show prompt + count
            if seed_track:
                result_subtitle = f"From: {seed_track.title} by {seed_track.artist} \u00b7 {len(matched_tracks)} tracks"
            elif prompt:
                result_subtitle = f"{prompt} \u00b7 {len(matched_tracks)} tracks"
            else:
                result_subtitle = f"{len(matched_tracks)} tracks"
            result_id = library_cache.save_result(
                result_type=result_type,
                title=result_title,
                prompt=prompt or "",
                snapshot=result.model_dump(mode="json"),
                track_count=len(matched_tracks),
                art_rating_key=first_art_key,
                subtitle=result_subtitle,
            )
        except Exception as e:
            logger.warning("Failed to save result: %s", e)

        # Curation memory: keep each pick's reason in the reasoning DB so later
        # runs can replay the argument beside the candidate (reuse) and rank
        # candidates by the mined tags (affinity). A memory-store failure must
        # not fail the run: the playlist is already built and saved.
        try:
            reasoned = [
                (t.rating_key, track_reasons[t.rating_key])
                for t in matched_tracks
                if track_reasons.get(t.rating_key)
            ]
            if reasoned:
                reasoning_recorded = agent_reasoning.record_run(
                    selections=reasoned,
                    prompt=prompt or "",
                    playlist_title=playlist_title,
                    result_id=result_id or "",
                    source="seed" if seed_track else "prompt",
                )
                logger.info("Stored curation reasoning for %d tracks", reasoning_recorded)
        except Exception as e:
            logger.warning("Failed to store curation reasoning: %s", e)

        # Emit small complete event with just metadata
        logger.info("Emitting complete event")
        complete_data = {
            "track_count": len(result.tracks),
            "token_count": result.token_count,
            "estimated_cost": result.estimated_cost,
            "playlist_title": result.playlist_title,
            "narrative": result.narrative,
            "track_reasons": result.track_reasons,
            "matching_as_asked": result.matching_as_asked,
            "relaxed_tiers": result.relaxed_tiers,
            "relaxed_to": result.relaxed_to,
            # Picks whose reason landed in the curation memory (replayable on
            # the next run through GET /api/reasoning/status)
            "reasoning_recorded": reasoning_recorded,
        }
        if result_id:
            complete_data["result_id"] = result_id
        yield emit("complete", complete_data)
        logger.info("Complete event emitted successfully")

        # Trailing padding to push complete event through network buffers (iOS Safari fix)
        # SSE comments (lines starting with ':') are ignored by the parser but help flush buffers
        yield ": heartbeat\n\n"

    except Exception as e:
        logger.exception("Error during playlist generation")
        yield emit("error", {"message": str(e)})


GENERATION_SYSTEM = """You are a music curator creating a playlist from a user's music library.

You will be given:
1. A description of what the user wants (prompt, seed track dimensions, or both)
2. A numbered list of tracks that are available in their library

Your task is to select tracks that best match the user's request. For each track, include a brief reason (1 sentence) explaining why it fits.

Guidelines:
- Select tracks that fit the mood, era, style, and other aspects of the request
- Vary the selection - don't pick too many tracks from the same artist or album
- Consider the flow of the playlist - how tracks will sound in sequence
- If using a seed track, don't include the seed track itself in the results

Return ONLY a JSON array like:
[
  {"artist": "Artist Name", "album": "Album Name", "title": "Track Title", "reason": "Brief explanation of why this track fits."},
  ...
]

No markdown formatting, no explanations - just the JSON array."""


NARRATIVE_SYSTEM = """You are a music connoisseur writing a brief liner note for a playlist.

Given the user's original request and the track selections (with reasons), create:
1. A creative playlist title (2-5 words, evocative, do NOT include any date)
2. A brief narrative (3 sentences, under 400 characters) that:
   - Reflects the mood or theme the user asked for
   - Mentions 3-4 specific songs by name (use single quotes around song names, e.g. 'Skinny Love')

Sound like a passionate music lover. Be concise.

Return ONLY valid JSON:
{"title": "Creative Title Here", "narrative": "Your brief narrative with 'song names' in single quotes..."}

No markdown formatting, no explanations - just the JSON object."""


def _tracks_match(llm_artist: str, llm_title: str, library_track: Track) -> bool:
    """Check if LLM selection matches a library track.

    Uses fuzzy matching to handle slight variations in naming.
    """
    from rapidfuzz import fuzz
    from backend.plex_client import simplify_string, normalize_artist, FUZZ_THRESHOLD

    # Compare titles
    simplified_llm_title = simplify_string(llm_title)
    simplified_lib_title = simplify_string(library_track.title)

    if fuzz.ratio(simplified_llm_title, simplified_lib_title) < FUZZ_THRESHOLD:
        return False

    # Compare artists (with variations)
    for artist_variant in normalize_artist(llm_artist):
        simplified_artist = simplify_string(artist_variant)
        simplified_lib_artist = simplify_string(library_track.artist)
        if fuzz.ratio(simplified_artist, simplified_lib_artist) >= FUZZ_THRESHOLD:
            return True

    return False
