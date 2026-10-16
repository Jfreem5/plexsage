"""Playlist generation with library validation."""

import json
import logging
import math
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime

from backend.llm_client import get_llm_client
from backend.models import FilterSet, GenerateResponse, Track
from backend.plex_client import PlexQueryError, get_plex_client, get_track_cache
from backend.prompt_classifier import (
    extract_decade_hints,
    extract_genre_hint_groups,
    extract_genre_hints,
    is_genre_keyword,
)
from backend.track_db import get_track_db

logger = logging.getLogger(__name__)

# Maximum number of tracks sent to the LLM after pre-ranking
PRE_RANK_LIMIT = 500

# Random sample size when a request has no filters (avoids loading the library)
SAMPLE_SIZE = 2000


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


def _prompt_specificity(
    prompt: str | None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
) -> float:
    """Score how *specific* a request is, from 0.0 (vague) to 1.0 (narrow).

    Concrete genre/decade cues raise the score; a mood/activity prompt with no
    genre cue stays low so the LLM gets a wider pool of options.
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

    return min(1.0, specificity)


def compute_track_budget(
    prompt: str | None = None,
    requested_max: int = 0,
    track_count: int = 25,
    pool_size: int | None = None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
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

    specificity = _prompt_specificity(prompt, genres, decades)
    budget = ceiling - (ceiling - floor) * specificity
    return max(floor, min(ceiling, round(budget)))


# =============================================================================
# Rule-Based Pre-Ranking
# =============================================================================

# Genre/decade hint dictionaries live in backend.prompt_classifier so that
# prompt analysis and pre-ranking share a single source of truth.


def _year_to_decade(year: int) -> str:
    """Convert a year to its decade string (e.g., 1985 → "1980s")."""
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
    filters: FilterSet | None = None,
    max_tracks: int = 500,
) -> list[Track]:
    """Score and rank tracks using rule-based heuristics, returning the top N.

    Scoring components (total max = 100):
        1. Genre match   (0-40)  — how well the track's genres match prompt hints
        2. Decade match  (0-20)  — whether the track's year falls in hinted decades
        3. User rating   (0-25)  — from the local SQLite track DB (0-10 scale)
        4. Recency       (0-15)  — exponential decay on last_played (72h half-life)

    If no prompt hints are found, the function falls back to rating + recency
    only, so the LLM still gets a sensible ordering.

    Args:
        tracks: Candidate tracks (already filtered by the Plex client).
        prompt: The user's natural-language prompt (used for hint extraction).
        filters: Optional FilterSet (genres/decades from the UI) — used as
                 additional hard hints when the prompt is absent.
        max_tracks: Maximum number of tracks to return (0 = no limit).

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
    if filters:
        for g in filters.genres:
            hinted_genres.add(g.lower())
        for d in filters.decades:
            hinted_decades.add(d)

    # --- Fetch DB metadata in one batch ---
    db = get_track_db()
    metadata: dict[str, dict] = {}
    if db:
        metadata = db.get_metadata_batch([t.rating_key for t in tracks])

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
                # or vice versa?  Gives a smaller bonus so near-misses still rank
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

        # 3. User rating (0-25)  +  4. Recency (0-15)
        meta = metadata.get(track.rating_key)
        if meta:
            rating = meta.get("user_rating", 0) or 0
            score += (rating / 10) * 25

            last_played_raw = meta.get("last_played")
            last_played_ts = _parse_timestamp(last_played_raw)
            recency = _recency_score(last_played_ts, now)
            score += recency * 15

        scored.append((score, track))

    # Sort by score descending, break ties by rating_key for deterministic ordering
    scored.sort(key=lambda x: (-x[0], x[1].rating_key))

    return [t for _, t in scored[:max_tracks]]


@dataclass
class _PreparedGeneration:
    """Candidates and prompt produced by the shared preparation stage."""

    candidates: list[Track]
    prompt: str


def _noop_emit(event: str, data: dict) -> str:
    """Progress sink for callers that don't stream (the non-SSE path)."""
    return ""


def _collect_preparation(
    preparation: Generator[str, None, _PreparedGeneration],
) -> _PreparedGeneration:
    """Drive a preparation generator to completion, discarding progress events."""
    try:
        while True:
            next(preparation)
    except StopIteration as stop:
        return stop.value


def _prepare_generation(
    emit: Callable[[str, dict], str],
    plex_client,
    *,
    prompt: str | None,
    seed_track: Track | None,
    selected_dimensions: list[str] | None,
    additional_notes: str | None,
    genres: list[str] | None,
    decades: list[str] | None,
    track_count: int,
    exclude_live: bool,
    min_rating: int,
    max_tracks_to_ai: int,
) -> Generator[str, None, _PreparedGeneration]:
    """Fetch candidates, mirror them locally, pre-rank, and build the LLM prompt.

    Shared by the streaming and non-streaming generation paths. Progress events
    are formatted with *emit* and yielded; the pre-ranked candidates and the
    finished prompt are returned when the generator completes.

    Raises:
        PlexQueryError: The Plex API could not serve the request.
        ValueError: No track matched the selection.
    """
    has_filters = genres or decades or min_rating > 0

    # Step 1: Fetch tracks. Unfiltered requests sample randomly (so repeat
    # generations differ); filtered requests reuse the short-lived cache.
    if not has_filters:
        yield emit("progress", {"step": "fetching", "message": "Sampling random tracks from library..."})
        logger.info("No filters applied, fetching %d random tracks", SAMPLE_SIZE)
        filtered_tracks = plex_client.get_random_tracks(
            count=SAMPLE_SIZE,
            exclude_live=exclude_live,
        )
        logger.info("Got %d random tracks", len(filtered_tracks))
    else:
        track_cache = get_track_cache()
        cached_tracks = track_cache.get(genres, decades, exclude_live, min_rating)

        if cached_tracks is not None:
            yield emit("progress", {"step": "cache_hit", "message": "Using cached library results..."})
            filtered_tracks = cached_tracks
            logger.info("Using %d cached tracks", len(filtered_tracks))
        else:
            yield emit("progress", {"step": "fetching", "message": "Fetching tracks from library..."})
            logger.info("Fetching tracks with filters: genres=%s, decades=%s, min_rating=%s", genres, decades, min_rating)
            filtered_tracks = plex_client.get_tracks_by_filters(
                genres=genres,
                decades=decades,
                exclude_live=exclude_live,
                min_rating=min_rating,
            )
            # Cache for potential future use
            track_cache.set(genres, decades, exclude_live, min_rating, filtered_tracks)
            logger.info("Found %d tracks matching filters", len(filtered_tracks))

    if not filtered_tracks:
        raise ValueError("No tracks match the selected filters. Try broadening your selection.")

    # Step 2: Mirror a Plex-served batch locally, then pre-rank. A read served
    # by the local DB is already mirrored, so only re-sync on a Plex fallback.
    if not plex_client.db_available():
        db = get_track_db()
        if db:
            db.sync_from_plex(filtered_tracks)

    llm_limit = compute_track_budget(
        prompt=prompt,
        requested_max=max_tracks_to_ai,
        track_count=track_count,
        pool_size=len(filtered_tracks),
        genres=genres,
        decades=decades,
    )
    logger.info("Pre-ranking %d candidates, retaining %d", len(filtered_tracks), llm_limit)
    yield emit("progress", {"step": "pre_ranking", "message": f"Ranking {len(filtered_tracks)} candidates…", "candidates": len(filtered_tracks), "retained": llm_limit})
    pre_rank_filters = FilterSet(genres=genres or [], decades=decades or []) if has_filters else None
    candidates = pre_rank_tracks(filtered_tracks, prompt, filters=pre_rank_filters, max_tracks=llm_limit)

    # Step 3: Build the prompt from the pre-ranked candidates.
    yield emit("progress", {"step": "preparing", "message": f"Preparing {len(candidates)} tracks for AI..."})

    track_list = "\n".join(
        f"{i+1}. {t.artist} - {t.title} ({t.album}, {t.year or 'Unknown year'})"
        for i, t in enumerate(candidates)
    )

    generation_parts = []

    if prompt:
        generation_parts.append(f"User's request: {prompt}")

    if seed_track:
        generation_parts.append(
            f"Seed track: {seed_track.title} by {seed_track.artist} "
            f"(from {seed_track.album}, {seed_track.year or 'Unknown year'})"
        )
        if selected_dimensions:
            generation_parts.append(f"Explore these dimensions: {', '.join(selected_dimensions)}")

    if additional_notes:
        generation_parts.append(f"Additional notes: {additional_notes}")

    generation_parts.append(f"\nSelect {track_count} tracks from this library:\n{track_list}")

    return _PreparedGeneration(
        candidates=candidates,
        prompt="\n\n".join(generation_parts),
    )


def _match_selections(
    candidates: list[Track],
    selections: list[dict],
    track_count: int,
    seed_track: Track | None,
) -> list[Track]:
    """Resolve the LLM's ``{artist, title}`` picks against the candidates.

    Indexes the candidates once (exact match first, then trigram-pruned fuzzy)
    and returns the matching tracks in selection order, excluding the seed.
    """
    matched_tracks: list[Track] = []
    used_keys: set[str] = set()

    if seed_track:
        used_keys.add(seed_track.rating_key)

    matcher = TrackMatcher(candidates)

    for selection in selections:
        if len(matched_tracks) >= track_count:
            break

        track = matcher.match(
            selection.get("artist", ""),
            selection.get("title", ""),
            used_keys,
        )
        if track is not None:
            matched_tracks.append(track)
            used_keys.add(track.rating_key)

    return matched_tracks


def generate_playlist_stream(
    prompt: str | None = None,
    seed_track: Track | None = None,
    selected_dimensions: list[str] | None = None,
    additional_notes: str | None = None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    track_count: int = 25,
    exclude_live: bool = True,
    min_rating: int = 0,
    max_tracks_to_ai: int = 500,
) -> Generator[str, None, None]:
    """Generate a playlist with streaming progress updates.

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

        try:
            prepared = yield from _prepare_generation(
                emit,
                plex_client,
                prompt=prompt,
                seed_track=seed_track,
                selected_dimensions=selected_dimensions,
                additional_notes=additional_notes,
                genres=genres,
                decades=decades,
                track_count=track_count,
                exclude_live=exclude_live,
                min_rating=min_rating,
                max_tracks_to_ai=max_tracks_to_ai,
            )
        except PlexQueryError as e:
            yield emit("error", {"message": f"Plex server error: {e}"})
            return

        # Step 4: Call LLM
        yield emit("progress", {"step": "ai_working", "message": "AI is curating your playlist..."})

        logger.info("Calling LLM with prompt length: %d chars", len(prepared.prompt))
        response = llm_client.generate(prepared.prompt, GENERATION_SYSTEM)
        logger.info("LLM response received: %d input, %d output tokens", response.input_tokens, response.output_tokens)

        # Step 5: Parse response
        yield emit("progress", {"step": "parsing", "message": "Parsing AI selections..."})

        track_selections = llm_client.parse_json_response(response)

        if not isinstance(track_selections, list):
            yield emit("error", {"message": "LLM returned invalid track selection format"})
            return

        # Step 6: Match tracks
        yield emit("progress", {"step": "matching", "message": f"Matching {len(track_selections)} selections to library..."})

        matched_tracks = _match_selections(
            prepared.candidates, track_selections, track_count, seed_track
        )

        # Step 7: Complete
        yield emit("progress", {"step": "complete", "message": "Playlist ready!"})

        result = GenerateResponse(
            tracks=matched_tracks,
            token_count=response.total_tokens,
            estimated_cost=response.estimated_cost(),
        )

        yield emit("complete", result.model_dump(mode="json"))

    except Exception as e:
        logger.exception("Error during playlist generation")
        yield emit("error", {"message": str(e)})


GENERATION_SYSTEM = """You are a music curator creating a playlist from a user's music library.

You will be given:
1. A description of what the user wants (prompt, seed track dimensions, or both)
2. A numbered list of tracks that are available in their library

Your task is to select tracks that best match the user's request. Return your selections as a JSON array of objects with artist, album, and title.

Guidelines:
- Select tracks that fit the mood, era, style, and other aspects of the request
- Vary the selection - don't pick too many tracks from the same artist or album
- Consider the flow of the playlist - how tracks will sound in sequence
- If using a seed track, don't include the seed track itself in the results

Return ONLY a JSON array like:
[
  {"artist": "Artist Name", "album": "Album Name", "title": "Track Title"},
  ...
]

No markdown formatting, no explanations - just the JSON array."""


def generate_playlist(
    prompt: str | None = None,
    seed_track: Track | None = None,
    selected_dimensions: list[str] | None = None,
    additional_notes: str | None = None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    track_count: int = 25,
    exclude_live: bool = True,
    min_rating: int = 0,
    max_tracks_to_ai: int = 500,
) -> GenerateResponse:
    """Generate a playlist using the filter-first approach.

    Args:
        prompt: Natural language description (prompt-first flow)
        seed_track: Seed track for finding similar music (seed flow)
        selected_dimensions: Dimension IDs selected by user (seed flow)
        additional_notes: Extra user preferences
        genres: Genre filters to apply
        decades: Decade filters to apply
        track_count: Number of tracks to generate
        exclude_live: Whether to exclude live recordings
        min_rating: Minimum track rating (0-10, 0 = no filter)
        max_tracks_to_ai: Max tracks to send to AI (0 = no limit)

    Returns:
        GenerateResponse with tracks, token count, and estimated cost

    Raises:
        ValueError: If no tracks match filters or LLM response invalid
        RuntimeError: If clients not initialized
    """
    logger.info("Starting playlist generation")
    llm_client = get_llm_client()
    plex_client = get_plex_client()

    if not llm_client:
        raise RuntimeError("LLM client not initialized")
    if not plex_client:
        raise RuntimeError("Plex client not initialized")

    # Fetch → mirror → pre-rank → prompt is shared with the streaming path.
    preparation = _prepare_generation(
        _noop_emit,
        plex_client,
        prompt=prompt,
        seed_track=seed_track,
        selected_dimensions=selected_dimensions,
        additional_notes=additional_notes,
        genres=genres,
        decades=decades,
        track_count=track_count,
        exclude_live=exclude_live,
        min_rating=min_rating,
        max_tracks_to_ai=max_tracks_to_ai,
    )
    try:
        prepared = _collect_preparation(preparation)
    except PlexQueryError as e:
        raise RuntimeError(f"Plex server error while fetching tracks: {e}") from e

    # Call LLM
    logger.info("Calling LLM with prompt length: %d chars", len(prepared.prompt))
    response = llm_client.generate(prepared.prompt, GENERATION_SYSTEM)
    logger.info("LLM response received: %d input, %d output tokens", response.input_tokens, response.output_tokens)

    # Parse response
    logger.debug("Parsing JSON response")
    track_selections = llm_client.parse_json_response(response)

    if not isinstance(track_selections, list):
        raise ValueError("LLM returned invalid track selection format")

    matched_tracks = _match_selections(
        prepared.candidates, track_selections, track_count, seed_track
    )

    return GenerateResponse(
        tracks=matched_tracks,
        token_count=response.total_tokens,
        estimated_cost=response.estimated_cost(),
    )


def _trigrams(s: str) -> frozenset[str]:
    """Return the set of character trigrams in *s*.

    Strings shorter than three characters collapse to a single gram (the
    string itself) so they still participate in the index.
    """
    if not s:
        return frozenset()
    if len(s) < 3:
        return frozenset({s})
    return frozenset(s[i : i + 3] for i in range(len(s) - 2))


def _simplified_artist_variants(artist: str) -> list[str]:
    """Normalized artist variants, simplified for fuzzy comparison."""
    from backend.plex_client import normalize_artist, simplify_string

    return [simplify_string(v) for v in normalize_artist(artist)]


def _fuzzy_track_match(
    llm_title_simple: str,
    llm_artist_variants: list[str],
    lib_title_simple: str,
    lib_artist_simple: str,
) -> bool:
    """Fuzzy-compare pre-simplified title/artist strings."""
    from rapidfuzz import fuzz
    from backend.plex_client import FUZZ_THRESHOLD

    if fuzz.ratio(llm_title_simple, lib_title_simple) < FUZZ_THRESHOLD:
        return False

    return any(
        fuzz.ratio(variant, lib_artist_simple) >= FUZZ_THRESHOLD
        for variant in llm_artist_variants
    )


def _tracks_match(llm_artist: str, llm_title: str, library_track: Track) -> bool:
    """Check whether a single LLM selection matches a library track.

    Convenience wrapper that simplifies the inputs on the fly. Batch matching
    should use :class:`TrackMatcher`, which precomputes these values once.
    """
    from backend.plex_client import simplify_string

    return _fuzzy_track_match(
        simplify_string(llm_title),
        _simplified_artist_variants(llm_artist),
        simplify_string(library_track.title),
        simplify_string(library_track.artist),
    )


# Above this fraction of the candidate list, unioning trigram posting lists
# costs more than it saves, so the matcher falls back to a linear scan (which
# still benefits from precomputed simplified keys).
_CANDIDATE_FRACTION = 0.25
_MIN_CANDIDATE_CAP = 50
_EMPTY: frozenset[int] = frozenset()


class TrackMatcher:
    """Resolve LLM track selections against a fixed candidate list.

    Precomputes simplified title/artist strings, an exact-match dictionary, and
    a trigram inverted index once per generation. Each selection is then
    resolved with roughly ``O(k)`` fuzzy comparisons (``k`` = tracks sharing a
    trigram with the selection) instead of scanning every candidate.

    Resolution order:
      1. Normalized exact match (``(artist, title)``), including ``&``/``and``
         artist variants.
      2. Fuzzy match over trigram candidates.
      3. Full-scan fallback, so a genuine match is never dropped when trigrams
         don't overlap (e.g. very short titles).

    The candidate list is already a small, pre-ranked in-memory set, so an
    in-memory index is preferred over a SQLite FTS round-trip: it avoids I/O and
    preserves the pre-rank ordering used for deterministic tie-breaking.
    """

    def __init__(self, tracks: list[Track]) -> None:
        from backend.plex_client import normalize_artist, simplify_string

        self._tracks = tracks
        self._simple_titles: list[str] = []
        self._simple_artists: list[str] = []
        self._exact: dict[tuple[str, str], list[int]] = {}
        self._index: dict[str, set[int]] = {}
        self._candidate_cap = max(
            _MIN_CANDIDATE_CAP, int(len(tracks) * _CANDIDATE_FRACTION)
        )

        for i, track in enumerate(tracks):
            simple_title = simplify_string(track.title)
            simple_artist = simplify_string(track.artist)
            self._simple_titles.append(simple_title)
            self._simple_artists.append(simple_artist)

            for variant in normalize_artist(track.artist):
                # The original variant simplifies to simple_artist already.
                simple_variant = (
                    simple_artist if variant == track.artist else simplify_string(variant)
                )
                key = (simple_variant, simple_title)
                self._exact.setdefault(key, []).append(i)

            # Index titles only: _fuzzy_track_match requires a title match
            # first, so title trigrams are the precise blocking key.
            for gram in _trigrams(simple_title):
                self._index.setdefault(gram, set()).add(i)

    def _fuzzy(self, index: int, simple_title: str, variants: list[str]) -> bool:
        """Fuzzy-compare a selection against a precomputed candidate."""
        return _fuzzy_track_match(
            simple_title,
            variants,
            self._simple_titles[index],
            self._simple_artists[index],
        )

    def _candidates(self, simple_title: str) -> set[int] | None:
        """Return title-trigram candidates, or None if the overlap is too broad."""
        found: set[int] = set()
        for gram in _trigrams(simple_title):
            found |= self._index.get(gram, _EMPTY)
            if len(found) > self._candidate_cap:
                return None
        return found

    def match(self, artist: str, title: str, used: set[str]) -> Track | None:
        """Return the best unused library track for an LLM selection, or None."""
        from backend.plex_client import simplify_string

        simple_title = simplify_string(title)
        variants = _simplified_artist_variants(artist)

        # 1) Normalized exact match.
        for variant in variants:
            for index in self._exact.get((variant, simple_title), ()):
                track = self._tracks[index]
                if track.rating_key not in used:
                    return track

        # 2) Fuzzy match over trigram candidates.
        candidates = self._candidates(simple_title)
        if candidates is None:
            # Overlap too broad to prune: scan every track in library order.
            for index, track in enumerate(self._tracks):
                if track.rating_key in used:
                    continue
                if self._fuzzy(index, simple_title, variants):
                    return track
            return None

        for index in sorted(candidates):
            track = self._tracks[index]
            if track.rating_key in used:
                continue
            if self._fuzzy(index, simple_title, variants):
                return track

        # 3) Fallback: scan tracks trigrams didn't surface so no match is lost.
        for index, track in enumerate(self._tracks):
            if index in candidates or track.rating_key in used:
                continue
            if self._fuzzy(index, simple_title, variants):
                return track

        return None
