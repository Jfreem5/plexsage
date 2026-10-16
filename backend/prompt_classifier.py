"""Rule- and embedding-based prompt classification (no LLM required).

``backend.analyzer.analyze_prompt`` used to send every user prompt to the LLM
just to map e.g. "something chill for a rainy day" onto
``["Ambient", "Post-Rock"]`` / ``["2010s"]``.  That is a classification task,
so this module does it locally and the LLM is only consulted as a *fallback*
when confidence is low.

Pipeline
--------
1. **Keyword rules** — :data:`MOOD_GENRE_MAP` and :data:`DECADE_HINTS` map
   common mood / activity / era words to candidate genres and decades.
   Cheap, exact, and covers the majority of prompts.
2. **Embedding similarity** — the prompt and every available library genre
   are turned into vectors and compared by cosine similarity.  This catches
   genre words the rules don't know (e.g. "shoegaze", "post-rock").
3. **Confidence** — each stage returns a confidence score; the analyzer
   accepts the result when it clears :data:`CONFIDENCE_THRESHOLD` and
   otherwise falls back to the LLM.

The embeddings are deterministic feature-hashed bag-of-words vectors built
with the standard library only — no model download, no network call, and no
extra dependency.  :func:`_embed` is the single seam to swap in a real
embedding model later without touching the public API.
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable


# =============================================================================
# Keyword → genre mapping
# =============================================================================
#
# Keys are matched as substrings of the lowercased prompt.  Values are
# candidate genres; they are mapped onto the genres that actually exist in the
# user's library before being returned.

MOOD_GENRE_MAP: dict[str, list[str]] = {
    # Mood / activity
    "chill": ["Ambient", "Lo-Fi", "Jazz", "Downtempo", "Chillout"],
    "chillout": ["Chillout", "Downtempo", "Ambient"],
    "relax": ["Ambient", "Classical", "Jazz", "New Age"],
    "relaxing": ["Ambient", "Classical", "Jazz", "New Age"],
    "calm": ["Ambient", "Classical", "New Age"],
    "mellow": ["Ambient", "Lo-Fi", "Jazz", "Soul"],
    "party": ["EDM", "Pop", "Hip-Hop", "Dance", "House"],
    "dance": ["Dance", "EDM", "House", "Electronic", "Pop"],
    "upbeat": ["Pop", "Dance", "Funk", "Disco"],
    "energetic": ["Rock", "EDM", "Punk", "Metal"],
    "happy": ["Pop", "Funk", "Disco", "Reggae"],
    "sad": ["Ballad", "Acoustic", "Folk", "Classical"],
    "melancholy": ["Alternative", "Indie", "Ballad", "Blues"],
    "workout": ["Rock", "EDM", "Hip-Hop", "Metal"],
    "gym": ["Rock", "EDM", "Hip-Hop", "Metal"],
    "focus": ["Classical", "Ambient", "Lo-Fi", "Jazz"],
    "study": ["Classical", "Ambient", "Lo-Fi", "Jazz"],
    "sleep": ["Ambient", "Classical", "New Age", "Lo-Fi"],
    "reading": ["Classical", "Ambient", "Jazz", "Lo-Fi"],
    "cooking": ["Jazz", "Bossa Nova", "Soul", "Reggae"],
    "road trip": ["Rock", "Classic Rock", "Alternative", "Pop"],
    "driving": ["Rock", "Alternative", "Indie", "Pop"],
    "morning": ["Pop", "Jazz", "Folk", "Acoustic"],
    "evening": ["Jazz", "Soul", "Blues", "Ambient"],
    "summer": ["Pop", "Reggae", "Funk", "Disco"],
    "winter": ["Classical", "Jazz", "Ambient", "Festive"],
    # Genre keywords
    "ambient": ["Ambient", "Downtempo", "New Age"],
    "lo-fi": ["Lo-Fi", "Chillhop", "Downtempo"],
    "lofi": ["Lo-Fi", "Chillhop", "Downtempo"],
    "chillhop": ["Chillhop", "Lo-Fi"],
    "jazz": ["Jazz", "Smooth Jazz", "Bebop", "Fusion"],
    "classical": ["Classical", "Orchestral", "Chamber"],
    "rock": ["Rock", "Alternative", "Punk", "Metal", "Indie"],
    "classic rock": ["Classic Rock", "Rock"],
    "hard rock": ["Hard Rock", "Rock", "Metal"],
    "alternative": ["Alternative", "Indie", "Grunge"],
    "alt rock": ["Alternative", "Rock"],
    "indie": ["Indie", "Indie Rock", "Alternative"],
    "grunge": ["Grunge", "Alternative"],
    "punk": ["Punk", "Alternative", "Emo"],
    "emo": ["Emo", "Alternative", "Punk"],
    "metal": ["Metal", "Heavy Metal", "Thrash", "Death Metal"],
    "hip hop": ["Hip-Hop", "Rap", "R&B"],
    "hip-hop": ["Hip-Hop", "Rap", "R&B"],
    "rap": ["Rap", "Hip-Hop"],
    "r&b": ["R&B", "Soul", "Funk"],
    "rnb": ["R&B", "Soul", "Funk"],
    "pop": ["Pop", "Dance-Pop", "Indie Pop"],
    "country": ["Country", "Folk", "Bluegrass"],
    "folk": ["Folk", "Indie Folk", "Acoustic"],
    "bluegrass": ["Bluegrass", "Country"],
    "blues": ["Blues", "Delta Blues", "Chicago Blues"],
    "reggae": ["Reggae", "Ska", "Dancehall"],
    "ska": ["Ska", "Reggae"],
    "electronic": ["Electronic", "EDM", "House", "Techno", "Trance"],
    "edm": ["EDM", "House", "Techno", "Electronic"],
    "house": ["House", "EDM", "Dance"],
    "techno": ["Techno", "Electronic"],
    "trance": ["Trance", "Electronic"],
    "dubstep": ["Dubstep", "EDM"],
    "soul": ["Soul", "R&B", "Funk"],
    "funk": ["Funk", "Disco", "Soul"],
    "disco": ["Disco", "Funk", "Dance"],
    "latin": ["Latin", "Salsa", "Bossa Nova"],
    "bossa nova": ["Bossa Nova", "Latin", "Jazz"],
    "world": ["World", "Latin", "Reggae"],
    "shoegaze": ["Shoegaze", "Dream Pop"],
    "dream pop": ["Dream Pop", "Shoegaze"],
    "post-rock": ["Post-Rock", "Post Rock"],
    "post rock": ["Post-Rock", "Post Rock"],
    "trip hop": ["Trip-Hop", "Downtempo"],
    "trip-hop": ["Trip-Hop", "Downtempo"],
    # Instrument hints whose spelling doesn't appear in genre descriptions
    "saxophone": ["Jazz", "Smooth Jazz"],
    "sax": ["Jazz", "Smooth Jazz"],
}

# Decade / era keywords → decade strings (matching Plex's "1980s" format).
DECADE_HINTS: dict[str, list[str]] = {
    "60s": ["1960s"], "sixties": ["1960s"], "1960s": ["1960s"],
    "70s": ["1970s"], "seventies": ["1970s"], "1970s": ["1970s"],
    "80s": ["1980s"], "eighties": ["1980s"], "1980s": ["1980s"],
    "90s": ["1990s"], "nineties": ["1990s"], "1990s": ["1990s"],
    "00s": ["2000s"], "2000s": ["2000s"], "y2k": ["1990s", "2000s"],
    "10s": ["2010s"], "2010s": ["2010s"],
    "20s": ["2020s"], "2020s": ["2020s"],
    "classic": ["1970s", "1980s", "1990s"],
    "modern": ["2010s", "2020s"],
    "current": ["2010s", "2020s"],
    "recent": ["2010s", "2020s"],
    "oldies": ["1960s", "1970s", "1980s"],
    "retro": ["1970s", "1980s", "1990s"],
}

# Descriptions embedded for genre similarity matching.  They deliberately
# include mood / activity synonyms so that lexical similarity can stand in for
# semantic similarity without shipping a model.  Aliases are resolved by
# normalizing the key (lowercased, non-alphanumerics stripped).
GENRE_DESCRIPTIONS: dict[str, str] = {
    "ambient": "calm atmospheric instrumental soundscapes, relaxing, dreamy, "
               "spacious, chill, meditative, meditation, quiet, peaceful, "
               "rainy, gloomy, overcast, background music for sleep and focus",
    "downtempo": "downtempo mellow chill electronic beats, laid-back, smooth, "
                 "relaxing, low tempo",
    "chillout": "chillout lounge downtempo relaxing chill mellow background",
    "lo-fi": "lo-fi chill instrumental hip hop beats, mellow, relaxed, cozy, "
             "study and focus music",
    "chillhop": "chillhop lo-fi jazzy instrumental beats, relaxed, smooth",
    "jazz": "jazz improvisation, swing, smooth, saxophone, piano, bebop, "
            "sophisticated evening music",
    "smooth jazz": "smooth jazz mellow relaxing saxophone easy listening",
    "bebop": "bebop fast jazz improvisation, trumpet, saxophone",
    "fusion": "jazz fusion electric instrumental, funk rock jazz blend",
    "classical": "classical orchestral symphony, strings, piano, chamber, "
                 "elegant, relaxing, focus music",
    "orchestral": "orchestral cinematic strings, symphony, epic, film score",
    "chamber": "chamber classical strings, intimate acoustic ensemble",
    "new age": "new age relaxing meditative meditation ambient piano, calm, "
               "quiet, peaceful, spa, healing",
    "electronic": "electronic synth, dance, techno, house, edm, atmospheric "
                  "digital production",
    "edm": "edm electronic dance music, high energy, festival, party, drops, "
           "club",
    "house": "house electronic dance, four on the floor, club, party, groovy",
    "techno": "techno electronic dance, repetitive, driving, club, industrial",
    "trance": "trance electronic euphoric, melodic, uplifting, club, energy",
    "dubstep": "dubstep bass heavy electronic, drops, wobble, club",
    "dance": "dance club pop electronic upbeat party groove",
    "pop": "pop catchy melodic mainstream upbeat radio friendly hooks",
    "dance-pop": "dance pop upbeat catchy club friendly",
    "indie pop": "indie pop upbeat jangly melodic guitar",
    "synth-pop": "synth pop electronic retro upbeat 80s",
    "rock": "rock guitars, drums, band, energetic, loud, classic rock and roll",
    "classic rock": "classic rock guitars, 70s 80s arena band, anthemic",
    "hard rock": "hard rock loud guitars, driving drums, aggressive, powerful",
    "alternative": "alternative rock indie guitars, 90s, moody, melancholic, "
                   "grunge",
    "indie": "indie alternative guitar driven, jangly, diy, college radio",
    "indie rock": "indie rock guitars, energetic, anthemic, alternative",
    "grunge": "grunge 90s distortion, raw guitars, angst, heavy, alternative",
    "punk": "punk fast energetic raw guitars, rebellious, loud, short songs",
    "emo": "emo emotional alternative punk, angsty, melodic",
    "metal": "metal heavy distorted guitars, aggressive, loud, intense, "
             "thrash and death metal",
    "heavy metal": "heavy metal loud distorted guitars, powerful, aggressive",
    "thrash": "thrash metal fast aggressive riffs, intense",
    "death metal": "death metal brutal heavy, growls, intense",
    "hip-hop": "hip hop rap beats, urban, rhyming, groovy, party, chill",
    "rap": "rap hip hop rhymes, beats, urban, confident",
    "trap": "trap hip hop heavy bass, hi-hats, 808, modern",
    "r&b": "r&b rhythm and blues, smooth soulful vocals, romantic, groovy",
    "soul": "soul r&b emotional vocals, groovy, warm, classic",
    "funk": "funk groovy bass, syncopated rhythm, danceable, horns",
    "disco": "disco dance party groovy 70s, four on the floor, celebratory",
    "country": "country acoustic guitars, storytelling, americana, folk, "
               "heartland",
    "folk": "folk acoustic singer songwriter, intimate, storytelling, guitar",
    "bluegrass": "bluegrass acoustic strings, banjo, fiddle, country",
    "americana": "americana roots country folk rock blend",
    "blues": "blues guitar, soulful, melancholy, raw, emotional, delta",
    "delta blues": "delta blues raw acoustic guitar, emotional, southern",
    "chicago blues": "chicago blues electric guitar, band, urban, soulful",
    "reggae": "reggae caribbean island groove, relaxed, offbeat, ska and "
              "dancehall",
    "ska": "ska upbeat brass, offbeat, reggae punk hybrid",
    "dancehall": "dancehall reggae electronic, upbeat, caribbean",
    "latin": "latin salsa rhythms, percussion, dance, festive",
    "salsa": "salsa latin dance, brass, percussion, energetic",
    "bossa nova": "bossa nova brazilian soft jazz, gentle acoustic, relaxing",
    "world": "world music global fusion, traditional, diverse rhythms",
    "shoegaze": "shoegaze reverb drenched guitars, dreamy, hazy, atmospheric, "
                "wall of sound",
    "dream pop": "dream pop ethereal dreamy vocals, atmospheric, shoegaze",
    "post-rock": "post-rock instrumental cinematic builds, atmospheric "
                 "guitars, crescendos, expansive, moody",
    "trip-hop": "trip hop downtempo beats, moody, atmospheric, hip hop and "
                "electronic blend",
    "ballad": "ballad slow emotional song, tender, romantic, sad",
    "acoustic": "acoustic unplugged guitar, intimate, warm, soft",
    "festive": "festive holiday christmas cheerful joyful",
    "gospel": "gospel choir, spiritual, uplifting, soulful",
    "opera": "opera classical vocals, dramatic, orchestral",
    "soundtrack": "soundtrack film score cinematic instrumental",
}


# =============================================================================
# Embedding helpers
# =============================================================================

# Confidence at or above which the analyzer accepts the local classification
# and skips the LLM entirely.
CONFIDENCE_THRESHOLD = 0.55

_EMBED_DIM = 512
_EMB_MIN_SIM = 0.20       # below this, an embedding match is not trusted
_EMB_REL_BAND = 0.6       # keep matches within 60% of the best score

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize(name: str) -> str:
    """Lowercase *name* and strip everything that isn't alphanumeric."""
    return _NORMALIZE_RE.sub("", name.lower())


@functools.lru_cache(maxsize=4096)
def _embed(text: str) -> tuple[float, ...]:
    """Return an L2-normalized feature-hashed bag-of-words vector for *text*.

    Uses word unigrams and bigrams so multi-word genres ("post rock") match.
    The standard library's hashing is not stable across processes, so a
    cryptographic hash is used to keep vectors deterministic.
    """
    vec = [0.0] * _EMBED_DIM
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]
    features = list(tokens)
    features.extend(f"{a}\x1f{b}" for a, b in zip(tokens, tokens[1:]))

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        vec[h % _EMBED_DIM] += 1.0 if (h >> 63) & 1 else -1.0

    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return tuple(vec)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity of two normalized vectors, clamped to [0, 1]."""
    return max(0.0, sum(x * y for x, y in zip(a, b)))


_GENRE_DESC_INDEX = {_normalize(k): v for k, v in GENRE_DESCRIPTIONS.items()}


def _genre_description(genre: str) -> str:
    """Return the embedded description for *genre*, or the name itself."""
    return _GENRE_DESC_INDEX.get(_normalize(genre), genre)


# =============================================================================
# Hint extraction
# =============================================================================


def extract_genre_hint_groups(prompt_lower: str) -> dict[str, list[str]]:
    """Return ``{keyword: [genres]}`` for keywords found in the prompt.

    Keeping the groups (rather than a flat set) lets the confidence score
    count how many *distinct prompt keywords* fired, which is a far better
    signal than the number of genres they expand to.
    """
    return {
        keyword: [g.lower() for g in genres]
        for keyword, genres in MOOD_GENRE_MAP.items()
        if keyword in prompt_lower
    }


def extract_genre_hints(prompt_lower: str) -> set[str]:
    """Extract lowercased genre hints from an already-lowercased prompt."""
    return {
        genre
        for genres in extract_genre_hint_groups(prompt_lower).values()
        for genre in genres
    }


def extract_decade_hints(prompt_lower: str) -> set[str]:
    """Extract decade strings (e.g. ``{"1980s"}``) from a lowercased prompt."""
    hinted: set[str] = set()
    for keyword, decades in DECADE_HINTS.items():
        if keyword in prompt_lower:
            hinted.update(decades)
    return hinted


def is_genre_keyword(keyword: str) -> bool:
    """Whether *keyword* names a genre rather than a mood/activity.

    Used to judge how *specific* a prompt is: a concrete genre ("jazz",
    "shoegaze") already narrows the library, whereas a mood/activity word
    ("chill", "road trip") leaves more room for the LLM to choose.
    """
    return _normalize(keyword) in _GENRE_DESC_INDEX


# =============================================================================
# Mapping hints onto the user's library
# =============================================================================


def _map_candidate(candidate: str, available_by_norm: dict[str, str]) -> str | None:
    """Map a single candidate genre onto an available library genre."""
    cnorm = _normalize(candidate)
    if not cnorm:
        return None

    if cnorm in available_by_norm:
        return available_by_norm[cnorm]

    substring_hit = next(
        (name for anorm, name in available_by_norm.items()
         if cnorm in anorm or anorm in cnorm),
        None,
    )
    if substring_hit:
        return substring_hit

    best_ratio, best_name = 0.0, None
    for anorm, name in available_by_norm.items():
        ratio = difflib.SequenceMatcher(None, cnorm, anorm).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, name
    if best_name and best_ratio >= 0.85:
        return best_name
    return None


def match_genres_to_available(
    candidate_genres: Iterable[str], available: list[str]
) -> list[str]:
    """Map candidate genre names onto genres that exist in *available*.

    Matching order: normalized exact → substring containment → fuzzy
    (``difflib`` ratio ≥ 0.85).  Results are returned in library order so the
    output is deterministic.
    """
    available_by_norm = {_normalize(name): name for name in available}
    targets = {
        name
        for candidate in candidate_genres
        if (name := _map_candidate(candidate, available_by_norm))
    }
    return [name for name in available if name in targets]


def _dedupe(items: Iterable[str]) -> list[str]:
    """Case-insensitive de-duplication preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# =============================================================================
# Confidence + public classification API
# =============================================================================


def _rule_confidence(
    keyword_hits: int, matched_genres: list[str], matched_decades: list[str]
) -> float:
    """Confidence that the keyword rules produced a useful suggestion.

    Based on how many distinct prompt keywords matched (not how many genres
    those keywords expand to), so broad hint groups don't inflate confidence.
    """
    if matched_genres:
        confidence = 0.65 + min(0.2, 0.1 * (keyword_hits - 1))
        if matched_decades:
            confidence += 0.15
    elif matched_decades:
        confidence = 0.6
    else:
        confidence = 0.0
    return round(min(confidence, 0.95), 4)


def _embedding_suggest(
    prompt: str, available_genres: list[str]
) -> tuple[list[str], float]:
    """Return embedding-matched genres and a confidence score."""
    prompt = prompt.strip()
    if not prompt or not available_genres:
        return [], 0.0

    prompt_vec = _embed(prompt)
    scored = [
        (_cosine(prompt_vec, _embed(_genre_description(genre))), genre)
        for genre in available_genres
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    top = scored[0][0]
    if top < _EMB_MIN_SIM:
        return [], 0.0

    floor = max(_EMB_MIN_SIM, top * _EMB_REL_BAND)
    genres = [genre for score, genre in scored if score >= floor][:6]
    confidence = min(0.9, 0.55 + 0.4 * top)
    return genres, round(confidence, 4)


@dataclass(frozen=True)
class PromptClassification:
    """Result of local prompt classification.

    Attributes:
        genres: Suggested genres that exist in the user's library.
        decades: Suggested decades that exist in the user's library.
        confidence: 0-1 confidence that the suggestion is useful.
        method: ``"rules"``, ``"embeddings"``, or ``"none"``.
        reasoning: Human-readable explanation, surfaced in the API response.
    """

    genres: list[str] = field(default_factory=list)
    decades: list[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "none"
    reasoning: str = ""

    @property
    def confident(self) -> bool:
        """Whether confidence clears :data:`CONFIDENCE_THRESHOLD`."""
        return self.confidence >= CONFIDENCE_THRESHOLD and bool(self.genres or self.decades)


def classify_prompt(
    prompt: str,
    available_genres: list[str],
    available_decades: list[str],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> PromptClassification:
    """Classify *prompt* into library genres/decades without an LLM call.

    Tries keyword rules first, then embedding similarity.  When neither clears
    *threshold* the returned ``method`` is ``"none"`` and the caller is
    expected to fall back to the LLM.

    Args:
        prompt: The user's natural-language request.
        available_genres: Genre names present in the user's library.
        available_decades: Decade names present in the user's library.
        threshold: Minimum confidence to accept a local result.

    Returns:
        A :class:`PromptClassification`.
    """
    prompt_lower = prompt.lower()
    hint_groups = extract_genre_hint_groups(prompt_lower)
    hinted_decades = extract_decade_hints(prompt_lower)

    # Map each hint group onto the library, counting how many distinct prompt
    # keywords actually produced a usable genre.
    available_by_norm = {_normalize(name): name for name in available_genres}
    matched_targets: set[str] = set()
    keyword_hits = 0
    for genres in hint_groups.values():
        hit = False
        for genre in genres:
            name = _map_candidate(genre, available_by_norm)
            if name:
                matched_targets.add(name)
                hit = True
        if hit:
            keyword_hits += 1
    matched_genres = [name for name in available_genres if name in matched_targets]

    decade_lookup = {d.lower(): d for d in available_decades}
    matched_decades = _dedupe(
        decade_lookup[d.lower()] for d in hinted_decades if d.lower() in decade_lookup
    )

    rule_conf = _rule_confidence(keyword_hits, matched_genres, matched_decades)
    if rule_conf >= threshold:
        return PromptClassification(
            genres=matched_genres,
            decades=matched_decades,
            confidence=rule_conf,
            method="rules",
            reasoning=_describe("keyword rules", matched_genres, matched_decades, rule_conf),
        )

    emb_genres, emb_conf = _embedding_suggest(prompt, available_genres)
    if emb_conf >= threshold:
        genres = _dedupe([*matched_genres, *emb_genres])
        confidence = max(rule_conf, emb_conf)
        return PromptClassification(
            genres=genres,
            decades=matched_decades,
            confidence=confidence,
            method="embeddings",
            reasoning=_describe("embedding similarity", genres, matched_decades, confidence),
        )

    confidence = max(rule_conf, emb_conf)
    return PromptClassification(
        genres=matched_genres,
        decades=matched_decades,
        confidence=confidence,
        method="none",
        reasoning=(
            "Low confidence from keyword rules and embeddings "
            f"({confidence:.2f}); falling back to the LLM."
        ),
    )


def _describe(
    technique: str, genres: list[str], decades: list[str], confidence: float
) -> str:
    """Build the reasoning string for a confident classification."""
    parts = []
    if genres:
        parts.append("genres " + ", ".join(genres))
    if decades:
        parts.append("decades " + ", ".join(decades))
    matched = " and ".join(parts) if parts else "no filters"
    return (
        f"Matched {matched} from the prompt via {technique} "
        f"(confidence {confidence:.2f}); no LLM call needed."
    )


__all__ = [
    "CONFIDENCE_THRESHOLD",
    "DECADE_HINTS",
    "GENRE_DESCRIPTIONS",
    "MOOD_GENRE_MAP",
    "PromptClassification",
    "classify_prompt",
    "extract_decade_hints",
    "extract_genre_hint_groups",
    "extract_genre_hints",
    "is_genre_keyword",
    "match_genres_to_available",
]
