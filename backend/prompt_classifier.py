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
from datetime import date, timedelta
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

# =============================================================================
# Extra facets beyond genre/decade
# =============================================================================
#
# The prompt can also ask for an artist, rule a genre out, state a length or a
# quality bar, and describe a mood that no library column can express. The first
# four become hard filters (they map onto library metadata); the last one is a
# soft cue that only steers the generation prompt.

# Phrases that rule a genre out ("rock but no metal", "everything except dubstep").
NEGATION_CUES = (
    "no",
    "none of",
    "not",
    "without",
    "except",
    "avoid",
    "avoiding",
    "exclude",
    "excluding",
    "minus",
    "free of",
    "free from",
    "devoid of",
    "rather than",
    "instead of",
    "anything but",
    "but not",
    "drop",
    "skip",
)

# Clause scan window after a negation cue; long enough for "any 80s synth-pop",
# short enough that the next clause is not swallowed.
NEGATION_SCAN_CHARS = 80

# Soft mood/texture vocabulary. Nothing in the library has a column for these,
# so they are echoed to the generator as steering language rather than turned
# into a DB filter. Values are the phrase the curator prompt is shown.
STYLE_CUES: dict[str, str] = {
    "slow, quiet, and intimate": ("slow", "quiet", "intimate", "gentle", "soft", "lull", "sparse"),
    "calm and downtempo": ("chill", "calm", "relaxed", "relaxing", "mellow", "laid back", "laid-back", "downtempo"),
    "driving and high-energy": ("driving", "energetic", "energy", "high energy", "high-energy", "intense", "pumping", "workout", "gym"),
    "warm and groovy": ("groovy", "groove", "funky", "soulful", "warm", "syncopated"),
    "loud and aggressive": ("loud", "aggressive", "heavy", "hard-hitting", "angry", "brutal", "raucous"),
    "dreamy and atmospheric": ("dreamy", "atmospheric", "hazy", "ethereal", "spacious", "reverb", "wall of sound"),
    "acoustic and singer-songwriter": ("acoustic", "unplugged", "singer-songwriter", "singer songwriter", "stripped back"),
    "bright and upbeat": ("upbeat", "bright", "happy", "sunny", "joyful", "feel-good", "feel good"),
    "melancholy and reflective": ("melancholy", "melancholic", "sad", "somber", "reflective", "aching", "heartbroken"),
    "tempo-forward and urgent": ("fast", "rapid", "urgent", "driving tempo", "rambunctious", "run"),
    "polished, produced pop": ("polished", "radio", "produced", "mainstream", "chart"),
    "raw, lo-fi texture": ("lo-fi", "lofi", "lo fi", "home recorded", "tape", "demo", "raw"),
}

# Cap on the artists a prompt can name (a 200-word prompt naming 40 bands is
# not a useful playlist request).
MAX_ARTIST_SUGGESTIONS = 6

# Track-length words with no number in the prompt ("short songs", "epics").
SHORT_TRACK_MS = 150_000  # ≤ 2:30
LONG_TRACK_MS = 300_000  # ≥ 5:00
EPIC_TRACK_MS = 600_000  # ≥ 10:00

DURATION_UPPER_WORDS = (
    "short songs", "short song", "shorter songs", "shorter song", "shorter track",
    "short tracks", "short tracks", "radio edit", "radio edits", "brief",
    "shorter", "short",
)
DURATION_LOWER_WORDS = (
    "long songs", "long song", "longer songs", "longer tracks", "long tracks",
    "album tracks", "long-form", "long form", "extended",
)
DURATION_EPIC_WORDS = ("epic", "epics", "epic length", "marathon", "suites")

# Numbered duration patterns: "under 3 minutes", "over ten minutes"...
_DURATION_MAX_RE = re.compile(
    r"\b(?:under|less than|fewer than|below|up to|no (?:longer|more) than|at most|"
    r"maximum|shorter than|within|short of)\s+(\d+(?:\.\d+)?)\s*"
    r"(minutes?|mins?|seconds?|secs?|s|m)\b"
)
_DURATION_MIN_RE = re.compile(
    r"\b(?:over|more than|longer than|at least|above|min(?:imum)?|longer|"
    r"north of)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?|s|m)\b"
)
_DURATION_RANGE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?|s|m)\s*(?:-|to|through|until)\s*"
    r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?|s|m)\b"
)
_DURATION_UNIT_MS = {
    "m": 60_000, "min": 60_000, "mins": 60_000, "minute": 60_000, "minutes": 60_000,
    "s": 1_000, "sec": 1_000, "secs": 1_000, "second": 1_000, "seconds": 1_000,
}

# Quality-bar vocabulary → user rating floor (Plex rates 0-10).
RATING_STRONG_WORDS = (
    "well rated", "well-rated", "highly rated", "high rated", "highest rated",
    "best rated", "top rated", "top-rated", "thumbs up", "thumbs-up",
    "critically acclaimed",
)
RATING_SOFT_WORDS = ("favorites", "favourites", "liked", "loved", "ones i love")
RATING_STRONG_FLOOR = 8
RATING_SOFT_FLOOR = 6

# Familiarity bar — the library's tracks table mirrors Plex's play counter
# (``viewCount``), so a "hidden gems" or "most played" request becomes an exact
# filter on the cache path. Floors stay below the counts a typical library holds:
# 94% of the tracks on this machine have never been played at all.
UNHEARD_WORDS = (
    "never played", "never heard", "never listened", "never spun", "never pressed",
    "haven't heard", "have not heard", "haven't listened", "unheard",
    "hidden gem", "hidden gems", "never on rotation", "not on rotation",
)
PLAY_FLOOR_WORDS = (
    "most played", "most listened", "most played tracks", "on repeat",
    "frequently played", "often played", "heavily played", "well loved", "well-loved",
)
PLAY_CEILING_WORDS = (
    "not the most played", "off rotation", "off the rotation",
    "not on my rotation", "rarely played", "seldom played", "light rotation",
)
MIN_PLAY_FLOOR = 3  # "well-loved" floor, mirrors library_cache.get_album_familiarity
MAX_PLAY_CEILING = 2  # "off rotation" ceiling: at most two plays

# Only-live requests: the library flags live rows (``is_live``), so "only live
# takes" becomes an exact filter. The wording must carry an exclusivity word next
# to a live word, because "no live versions" asks the opposite (which the drop-live
# default already honours) and a bare "live" ("a live band feel") must not flip the
# pool to only-live.
LIVE_ONLY_WORDS = (
    "only live",
    "only the live",
    "just live",
    "just the live",
    "nothing but live",
    "stick to live",
    "live versions only",
    "live version only",
    "live takes only",
    "live recordings only",
    "live album only",
    "live cuts only",
)

# A "nothing I rated highly" request caps Plex's 0-10 user rating instead of
# flooring it. A stated ceiling outranks the floor vocabulary below, since the
# same phrasing contains "rated" and "favourites".
RATING_CEILING_WORDS = (
    "nothing i rated",
    "nothing i have rated",
    "nothing i've rated",
    "not what i rated",
    "not the ones i rated",
    "not rated highly",
    "nothing rated highly",
    "not my highest rated",
    "not my most rated",
    "under-rated only",
)
RATING_CEILING_CAP = 4  # "at most 4/10": bans music the listener loved, keeps unrated

# Rated vs. unrated rows. Plex leaves an unrated track as NULL rather than 0, so
# these are presence tests that SQLite answers exactly (see the is_live /
# user_rating rules in library_cache._filter_conditions).
RATED_ONLY_WORDS = (
    "i rated",
    "i have rated",
    "i've rated",
    "i rate",
    "already rated",
    "that i gave a rating",
    "my rated",
)
UNRATED_ONLY_WORDS = (
    "unrated",
    "not yet rated",
    "never rated",
    "not rated yet",
    "haven't rated",
    "have not rated",
    "i never rated",
    "without a rating",
)

# Last-heard window ("heard in the last six months", "played since 2024"). Plex
# fills lastViewedAt, which the cache keeps, so a familiarity-by-recency request
# is enforceable. A play verb must appear in the same clause, or "recorded in
# 1977" and "from the last five years" (a release-year request) would read as one.
_PLAY_VERB = r"(?:play(?:s|ed|ing)?|hear(?:s|ing|d)?|listen(?:s|ing|ed)?|listened to|spin(?:s|ned)?|spun|stream(?:s|ed|ing)?|rotat(?:e|ion|ing))"
_PLAYED_RECENT_RE = re.compile(
    rf"\b{_PLAY_VERB}\b[^.;!?]{{0,24}}?\b(?:in\s+)?(?:the\s+|a\s+)?"
    rf"(last|past|most\s+recent|final|recent)\s+"
    rf"(?:(\d{{1,2}}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few)\s+)?"
    rf"(days?|weeks?|months?|years?)\b"
)
_PLAYED_SINCE_RE = re.compile(
    rf"\b{_PLAY_VERB}\b[^.;!?]{{0,24}}?\bsince\s+((?:19|20)\d{{2}})\b"
)
_PLAYED_BEFORE_RE = re.compile(
    rf"\b(?:haven'?t|have\s+not|had\s+not|has\s+not|never|not|no|stopped|barely|hardly"
    rf"|rarely|seldom)\s+(?:{_PLAY_VERB})\b[^.;!?]{{0,20}}?"
    rf"\b(?:since|after|later than)\s+((?:19|20)\d{{2}})\b"
)
_PLAYED_RECENT_WORDS = (
    "recently played",
    "recently heard",
    "played recently",
    "heard recently",
    "played lately",
    "hearing it lately",
    "still on rotation",
    "still spinning",
)
_PLAYED_UNIT_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}
_PLAYED_SPAN_WORDS = {"couple": 2, "few": 3}
_PLAY_VERB_RE = re.compile(rf"\b{_PLAY_VERB}\b")
RECENT_PLAY_DEFAULT_DAYS = 60  # "recently played" with no number in it
_PLAYED_YEAR_MIN_BOUND = 1850  # implausible years are ignored, like release years

# Compilations. Nothing in the cache flags an album as a compilation, but plexapi
# fills the album artist with "Various Artists" for exactly those rows, which is
# what the filter drops (see :func:`library_cache.get_tracks_by_filters`).
# Compilations. Only exclusion phrasings are listed: "compilation" on its own
# can just as well ask FOR one ("80s hits compilation"), and turning that into a
# filter would drop exactly the tracks the prompt wanted.
COMPILATION_WORDS = (
    "no compilations", "no compilation", "not compilations", "not a compilation",
    "avoid compilations", "avoid compilation", "skip compilations",
    "exclude compilations", "without compilations", "non-compilation",
    "no various artists", "not various artists",
    "studio albums only", "studio album only", "only studio albums", "only studio album",
    "original albums only", "original album only",
)

# Release-year window: Plex already offers decades, so these patterns only add the
# finer grain decades cannot express ("recorded in 1977", "between 1975 and 1985",
# "from 1990 to 1999", "the last three years").
# Bare four-digit years are read as eras: a playlist prompt writing "1977" means
# the recording year. ``1977s`` style decade spellings are left to DECADE_HINTS.
_YEAR_RANGE_RE = re.compile(
    r"\b(?:between|from)\s+((?:19|20)\d{2})\s*(?:and|to|-|–)\s*((?:19|20)\d{2})\b"
)
_YEAR_MIN_RE = re.compile(
    r"\b(?:since|after|later than|newer than|released after|recorded after)\s+((?:19|20)\d{2})\b"
)
_YEAR_MAX_RE = re.compile(
    r"\b(?:before|prior to|older than|up to|released before|recorded before|earlier than)\s+((?:19|20)\d{2})\b"
)
_YEAR_CUE_RE = re.compile(
    r"\b(?:recorded|released|recorded in|released in|cut|pressed|made|from|in)\s+(?:the\s+)?((?:19|20)\d{2})\b"
)
_YEAR_ANY_RE = re.compile(r"((?:19|20)\d{2})(?!\w)")
_RECENT_WINDOW_RE = re.compile(
    r"\b(?:last|past|final|most recent)\s+"
    r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few)\s*years?\b"
)
_RECENT_WINDOW_SPAN = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "few": 3,
}
_YEAR_MIN_BOUND = 1850  # implausible years are ignored rather than turned into a filter

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
_EMB_NEAR_MISS_BAND = 0.45  # reported as "considered but left out"

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


def _words(text: str) -> str:
    """Lowercase *text* and collapse every run of non-alphanumerics to one space."""
    return " ".join(text.lower().split())


def match_artists(prompt: str, available_artists: list[str]) -> list[str]:
    """Library artists the prompt names.

    Matching is word-boundary based on the normalized names, so "beatles" finds
    "The Beatles" (an article-prefixed name also matches without its prefix) and
    "costello" finds "Elvis Costello". Only artists present in *available_artists*
    can be returned, and longer names win first so "Gang of Four" is preferred
    over "Four".
    """
    if not available_artists:
        return []
    haystack = f" {_words(prompt)} "
    found: list[str] = []
    for name in sorted(available_artists, key=lambda item: (-len(item), item.lower())):
        needle = _words(name)
        if not needle:
            continue
        candidates = [needle]
        for article in ("the ", "a ", "an "):
            if needle.startswith(article):
                candidates.append(needle[len(article):])
        if any(f" {cand} " in haystack for cand in candidates):
            found.append(name)
        if len(found) >= MAX_ARTIST_SUGGESTIONS:
            break
    return found


def extract_negated_genres(
    prompt_lower: str, available_genres: list[str]
) -> list[str]:
    """Library genres the prompt rules out ("rock but no metal" → ``["Metal"]``).

    Each :data:`NEGATION_CUES` occurrence opens an up to :data:`NEGATION_SCAN_CHARS`
    character window, cut at the next clause boundary; library genre names inside
    that window are treated as negated. Only genres that exist in the library are
    returned, in library order.
    """
    if not available_genres or not prompt_lower:
        return []
    negated: set[str] = set()
    for cue in NEGATION_CUES:
        pattern = rf"\b{re.escape(cue)}\b"
        for hit in re.finditer(pattern, prompt_lower):
            window = prompt_lower[hit.end(): hit.end() + NEGATION_SCAN_CHARS]
            window = re.split(r"[.;,!?]", window)[0]
            spaced = f" {re.sub(r'[^a-z0-9]+', ' ', window).strip()} "
            for genre in available_genres:
                name = re.sub(r"[^a-z0-9]+", " ", genre.lower()).strip()
                if len(name) >= 3 and f" {name} " in spaced:
                    negated.add(genre)
    return [g for g in available_genres if g in negated]


def extract_style_cues(prompt_lower: str) -> list[str]:
    """Soft steering phrases from :data:`STYLE_CUES` that the prompt evokes.

    These are not filters — the library has no column for mood or texture — so
    they are handed to the generator as description language.
    """
    return [
        cue
        for cue, triggers in STYLE_CUES.items()
        if any(_phrase_hit(prompt_lower, trigger) for trigger in triggers)
    ]


@functools.lru_cache(maxsize=512)
def _phrase_hit(prompt: str, phrase: str) -> bool:
    """Whether *phrase* occurs in *prompt* as whole words."""
    return re.search(rf"\b{re.escape(phrase)}\b", prompt) is not None


def _duration_from_unit(value: str, unit: str) -> int:
    """Milliseconds for *value* expressed in *unit* ("3" + "minutes")."""
    multiplier = _DURATION_UNIT_MS.get(unit.rstrip(".")) or _DURATION_UNIT_MS.get(unit[:2], 60_000)
    return int(float(value) * multiplier)


# A floor word flips into a ceiling when a negation opens the phrase: "nothing
# over 3 minutes" and "no songs longer than 5 minutes" ask for a cap, not a
# floor, and reading them the other way round would enforce the opposite filter.
DURATION_NEGATORS = (
    "nothing",
    "no songs",
    "no tracks",
    "no song",
    "no track",
    "not",
    "never",
    "without",
    "avoid",
    "under",
    "below",
    "less",
    "minus",
    "except",
    "skip",
    "exclude",
)


def _duration_negated(prompt_lower: str, start: int) -> bool:
    """Whether the number phrase starting at *start* is opened by a negation.

    Only the clause the phrase lives in is scanned, so "no metal, over 3
    minutes" keeps its floor: the ban belongs to another clause.
    """
    head = prompt_lower[max(0, start - 24) : start]
    head = re.split(r"[.;,!]", head)[-1]
    return any(_phrase_hit(head, word) for word in DURATION_NEGATORS)


def extract_duration_window(prompt_lower: str) -> tuple[int, int]:
    """``(min_duration_ms, max_duration_ms)`` the prompt asks for.

    Numbers win over words ("under 4 minutes" beats "long songs"); a window the
    prompt contradicts with itself is dropped. Returns zeros when nothing applies.
    """
    lo = hi = 0
    range_hit = _DURATION_RANGE_RE.search(prompt_lower)
    if range_hit:
        first = _duration_from_unit(range_hit.group(1), range_hit.group(2))
        second = _duration_from_unit(range_hit.group(3), range_hit.group(4))
        lo, hi = min(first, second), max(first, second)
    upper = _DURATION_MAX_RE.search(prompt_lower)
    if upper:
        hi = hi or _duration_from_unit(upper.group(1), upper.group(2))
    lower = _DURATION_MIN_RE.search(prompt_lower)
    if lower:
        value = _duration_from_unit(lower.group(1), lower.group(2))
        if _duration_negated(prompt_lower, lower.start()):
            hi = hi or value
        else:
            lo = lo or value
    if not (lo or hi):
        if any(_phrase_hit(prompt_lower, word) for word in DURATION_EPIC_WORDS):
            lo = EPIC_TRACK_MS
        elif any(_phrase_hit(prompt_lower, word) for word in DURATION_LOWER_WORDS):
            lo = LONG_TRACK_MS
        elif any(_phrase_hit(prompt_lower, word) for word in DURATION_UPPER_WORDS):
            hi = SHORT_TRACK_MS
    if lo and hi and lo > hi:
        return 0, 0
    return lo, hi


def extract_rating_floor(prompt_lower: str) -> int:
    """User-rating floor the prompt implies ("best-rated tracks" → 8)."""
    if any(word in prompt_lower for word in RATING_STRONG_WORDS):
        return RATING_STRONG_FLOOR
    if any(word in prompt_lower for word in RATING_SOFT_WORDS):
        return RATING_SOFT_FLOOR
    if re.search(r"\b(?<!un)rated\b", prompt_lower):
        return RATING_STRONG_FLOOR
    return 0


def extract_negated_artists(
    prompt_lower: str, available_artists: list[str]
) -> list[str]:
    """Library artists the prompt rules out ("prog but not Yes" → ``["Yes"]``).

    Same negation-window scan as :func:`extract_negated_genres`: each
    :data:`NEGATION_CUES` occurrence opens a window cut at the next clause
    boundary, and library artist names inside it are treated as banned. Names
    are matched with the word-boundary rules :func:`match_artists` uses, so "not
    the beatles" still finds "The Beatles". Only artists present in the library
    can be returned, in library order.
    """
    if not available_artists or not prompt_lower:
        return []
    negated: set[str] = set()
    for cue in NEGATION_CUES:
        pattern = rf"\b{re.escape(cue)}\b"
        for hit in re.finditer(pattern, prompt_lower):
            window = prompt_lower[hit.end() : hit.end() + NEGATION_SCAN_CHARS]
            window = re.split(r"[.;,!?]", window)[0]
            spaced = f" {re.sub(r'[^a-z0-9]+', ' ', window).strip()} "
            for name in available_artists:
                needle = _words(name)
                if len(needle) < 3 or f" {needle} " not in spaced:
                    continue
                negated.add(name)
    return [a for a in available_artists if a in negated]


def extract_negated_decades(prompt_lower: str, available_decades: list[str]) -> list[str]:
    """Library decades the prompt rules out ("modern but not the 2010s")."""
    if not available_decades or not prompt_lower:
        return []
    banned: set[str] = set()
    for cue in NEGATION_CUES:
        pattern = rf"\b{re.escape(cue)}\b"
        for hit in re.finditer(pattern, prompt_lower):
            window = prompt_lower[hit.end() : hit.end() + NEGATION_SCAN_CHARS]
            window = re.split(r"[.;,!?]", window)[0]
            for token, decades in DECADE_HINTS.items():
                if not re.search(rf"\b{re.escape(token)}\b", window):
                    continue
                banned.update(decades)
    lookup = {d.lower(): d for d in available_decades}
    return [lookup[d.lower()] for d in sorted(banned) if d.lower() in lookup]


def extract_play_window(prompt_lower: str) -> tuple[int, int, bool]:
    """Play-count window the prompt implies: ``(min_plays, max_plays, unheard)``.

    Plex already counts plays (``viewCount``), so "most played" becomes a floor
    and "off rotation" a ceiling. Unheard is a separate flag rather than a
    ceiling of 0 because 0 means unbounded everywhere in this codebase. The
    phrases are checked most-specific first: "not the most played" contains the
    floor word "most played", so a ceiling must be read before a floor or the
    negation turns into the opposite filter.
    """
    unheard = any(word in prompt_lower for word in UNHEARD_WORDS)
    if unheard:
        return 0, 0, True
    if any(word in prompt_lower for word in PLAY_CEILING_WORDS):
        return 0, MAX_PLAY_CEILING, False
    if any(word in prompt_lower for word in PLAY_FLOOR_WORDS):
        return MIN_PLAY_FLOOR, 0, False
    return 0, 0, False


def _play_verb_in_clause(prompt_lower: str, start: int) -> bool:
    """Whether the clause opened at *start* names a play verb.

    Only the clause the phrase sits in is scanned, cut at the nearest sentence or
    clause boundary, so "rock from the last five years" stays a release-year request
    while "music I played in the last five years" becomes a last-heard window
    (see :func:`extract_last_played_window`).
    """
    head = prompt_lower[max(0, start - 40) : start]
    head = re.split(r"[.;,!]", head)[-1]
    return _PLAY_VERB_RE.search(head) is not None


def _clamp_year(value: int) -> int:
    """Return *value* when it is a plausible release year, else 0 (no bound)."""
    if value < _YEAR_MIN_BOUND:
        return 0
    return value


def extract_year_window(prompt_lower: str) -> tuple[int, int]:
    """Release-year window the prompt asks for: ``(min_year, max_year)``.

    Decades already cover the coarse grain, so this adds the finer one: "recorded
    in 1977" pins a single year, "between 1975 and 1985" spans a range and "the
    last five years" is anchored to today. Phrases with no plausible year in them
    produce ``(0, 0)`` (unbounded), and implausible four-digit numbers such as a
    track count are ignored rather than turned into a filter.
    """
    match = _YEAR_RANGE_RE.search(prompt_lower)
    if match:
        low, high = int(match.group(1)), int(match.group(2))
        if low > high:
            low, high = high, low
        return _clamp_year(low), _clamp_year(high)

    low = _clamp_year(int(_YEAR_MIN_RE.search(prompt_lower).group(1))) \
        if _YEAR_MIN_RE.search(prompt_lower) else 0
    high = _clamp_year(int(_YEAR_MAX_RE.search(prompt_lower).group(1))) \
        if _YEAR_MAX_RE.search(prompt_lower) else 0
    if low or high:
        return low, high

    match = _RECENT_WINDOW_RE.search(prompt_lower)
    if match:
        # "played in the last five years" is a familiarity request, not a release
        # date; extract_last_played_window owns that reading.
        if _play_verb_in_clause(prompt_lower, match.start()):
            return 0, 0
        token = match.group(1)
        span = int(token) if token.isdigit() else _RECENT_WINDOW_SPAN.get(token, 0)
        if 1 <= span <= 60:
            return date.today().year - span, 0
        return 0, 0

    for pattern in (_YEAR_CUE_RE, _YEAR_ANY_RE):
        match = pattern.search(prompt_lower)
        if match:
            year = _clamp_year(int(match.group(1)))
            if year:
                return year, year
    return 0, 0


def extract_compilation_exclusion(prompt_lower: str) -> bool:
    """Whether the prompt asks for original albums rather than compilations."""
    return any(word in prompt_lower for word in COMPILATION_WORDS)


def _last_played_phrase(prompt_lower: str) -> str:
    """The prompt phrase that opened a last-heard window, for evidence text."""
    for pattern in (_PLAYED_BEFORE_RE, _PLAYED_SINCE_RE, _PLAYED_RECENT_RE):
        hit = pattern.search(prompt_lower)
        if hit:
            return hit.group(0).strip()
    return _phrase_matched(prompt_lower, _PLAYED_RECENT_WORDS)


def extract_live_request(prompt_lower: str) -> bool:
    """Whether the prompt asks for *only* live rows ("just the live takes").

    The drop-live default already honours "no live versions", so only an
    exclusivity phrasing is turned into the ``live_only`` filter.
    """
    return any(word in prompt_lower for word in LIVE_ONLY_WORDS)


def extract_rating_presence(prompt_lower: str) -> tuple[bool, bool]:
    """``(rated_only, unrated_only)`` the prompt asks for.

    Asking for rows the listener never rated wins when both presences appear
    ("rated songs, plus whatever is still unrated"), since a stated ban is the
    more specific instruction of the two.
    """
    unrated = any(word in prompt_lower for word in UNRATED_ONLY_WORDS)
    rated = not unrated and any(word in prompt_lower for word in RATED_ONLY_WORDS)
    return rated, unrated


def extract_rating_window(prompt_lower: str) -> tuple[int, int]:
    """``(min_rating, max_rating)`` the prompt implies; 0 means unbounded.

    A stated ceiling ("not the ones I rated highly") is the only reading that
    makes sense, and it outranks the floor vocabulary: that phrasing contains
    "rated" and "favourites", which would otherwise set a floor and then
    contradict the cap.
    """
    if any(word in prompt_lower for word in RATING_CEILING_WORDS):
        return 0, RATING_CEILING_CAP
    return extract_rating_floor(prompt_lower), 0


def extract_last_played_window(prompt_lower: str) -> tuple[str, str]:
    """``(played_after, played_before)`` as ISO ``YYYY-MM-DD`` dates ("" = unbounded).

    Plex fills ``lastViewedAt`` and the library cache keeps it, so a familiarity-by-
    recency request is enforceable exactly. Only a *play* verb licenses the window:
    "recorded in 1977" and "from the last five years" belong to the release-year
    facet. Months and years are counted as 30 and 365 days because the window is
    phrased by day.

    A negated "haven't heard it since 2023" caps how recently the track was heard
    and is read before the plain "played since 2024", which floors it.
    """
    negated = _PLAYED_BEFORE_RE.search(prompt_lower)
    if negated:
        year = int(negated.group(1))
        if year < _PLAYED_YEAR_MIN_BOUND:
            return "", ""
        cutoff = date(year, 1, 1) - timedelta(days=1)
        return "", cutoff.isoformat()

    floored = _PLAYED_SINCE_RE.search(prompt_lower)
    if floored:
        year = int(floored.group(1))
        if year < _PLAYED_YEAR_MIN_BOUND:
            return "", ""
        return date(year, 1, 1).isoformat(), ""

    recent = _PLAYED_RECENT_RE.search(prompt_lower)
    if recent:
        unit_days = _PLAYED_UNIT_DAYS.get(recent.group(3), 0)
        spans = {**_RECENT_WINDOW_SPAN, "couple": 2}
        token = recent.group(2)
        if not token:
            span = 1
        elif token.isdigit():
            span = int(token)
        else:
            span = spans.get(token, 0)
        if unit_days and 1 <= span <= 60:
            after = date.today() - timedelta(days=span * unit_days)
            return after.isoformat(), ""

    if any(word in prompt_lower for word in _PLAYED_RECENT_WORDS):
        after = date.today() - timedelta(days=RECENT_PLAY_DEFAULT_DAYS)
        return after.isoformat(), ""
    return "", ""


def _phrase_matched(prompt_lower: str, words: tuple[str, ...] | list[str]) -> str:
    """First phrase from *words* present in *prompt_lower*, for evidence text."""
    return next((word for word in words if word in prompt_lower), "")


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


def _facet_confidence(
    artists: bool,
    negated: bool,
    duration: bool,
    rating: int,
    cues: bool,
    excluded_artists: bool = False,
    excluded_decades: bool = False,
    plays: bool = False,
    unheard: bool = False,
    year: bool = False,
    compilations: bool = False,
    live: bool = False,
    rated: bool = False,
    ceiling: bool = False,
    last_played: bool = False,
) -> float:
    """Confidence contributed by the non-genre facets on their own.

    A named artist or an explicit "no synth-pop" narrows the library as surely
    as a genre word does, so those prompts can skip the LLM even when no genre
    keyword fires. Soft mood cues are deliberately scored below
    :data:`CONFIDENCE_THRESHOLD`: they describe a feeling the library cannot
    filter on, so on their own they must not stop the caller from consulting the
    embedding stage or the LLM.

    The year window and the play-count facets are the newest signals: a stated
    year or "hidden gems" is an explicit, enforceable instruction, so it scores
    like the other hard filters. Dropping compilations narrows less (it only
    removes the Various-Artists rows), so it sits just under a rating floor. The
    still-newer facets score the same way because each is an exact SQL test now:
    a last-heard window ties to a stated year, "only live" and a rated/unrated
    request to "hidden gems", and a rating ceiling to a named ban because it only
    removes rows the listener already loved.
    """
    scores = []
    if artists:
        scores.append(0.80)
    if last_played:
        scores.append(0.78)
    if unheard:
        scores.append(0.78)
    if live or rated:
        scores.append(0.76)
    if year:
        scores.append(0.76)
    if plays:
        scores.append(0.76)
    if excluded_artists:
        scores.append(0.74)
    if ceiling:
        scores.append(0.72)
    if negated:
        scores.append(0.70)
    if excluded_decades:
        scores.append(0.70)
    if duration:
        scores.append(0.68)
    if compilations:
        scores.append(0.66)
    if rating:
        scores.append(0.62)
    if cues:
        scores.append(0.45)
    return max(scores) if scores else 0.0


def _combine_confidence(primary: float, facet: float) -> float:
    """Blend rule/embedding confidence with the evidence the facets carried."""
    if facet <= 0:
        return primary
    if primary <= 0:
        return facet
    return round(min(max(primary, facet) + 0.05, 0.95), 4)


def _embedding_suggest(
    prompt: str, available_genres: list[str]
) -> tuple[list[str], float, list[str]]:
    """Return embedding-matched genres, a confidence score, and near misses.

    The third value lists genres that scored close to the prompt but outside the
    selection band, which is the evidence shown for "considered and left out".
    """
    prompt = prompt.strip()
    if not prompt or not available_genres:
        return [], 0.0, []

    prompt_vec = _embed(prompt)
    scored = [
        (_cosine(prompt_vec, _embed(_genre_description(genre))), genre)
        for genre in available_genres
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    top = scored[0][0]
    if top < _EMB_MIN_SIM:
        return [], 0.0, []

    floor = max(_EMB_MIN_SIM, top * _EMB_REL_BAND)
    genres = [genre for score, genre in scored if score >= floor][:6]
    confidence = min(0.9, 0.55 + 0.4 * top)
    selected = set(genres)
    excluded = [
        genre
        for score, genre in scored
        if genre not in selected and score >= _EMB_NEAR_MISS_BAND * top
    ][:8]
    return genres, round(confidence, 4), excluded


@dataclass(frozen=True)
class PromptClassification:
    """Result of local prompt classification.

    Attributes:
        genres: Suggested genres that exist in the user's library.
        decades: Suggested decades that exist in the user's library.
        artists: Library artists the prompt named (empty without a roster).
        negated_genres: Library genres the prompt rules out ("no metal").
        style_cues: Soft mood/texture phrases (no library column backs these).
        min_duration_ms / max_duration_ms: Track-length window the prompt asked
            for; 0 means unbounded.
        min_rating: User-rating floor implied by the prompt; 0 means none.
        exclude_artists / exclude_decades: Library artists and decades the prompt
            rules out; enforced as hard exclusions.
        min_year / max_year: Release-year window the prompt asked for; 0 means
            unbounded on that side.
        min_plays / max_plays: Plex play-count window ("most played" is a floor,
            "off rotation" a ceiling); 0 means unbounded.
        unheard_only: Only never-played tracks ("hidden gems").
        exclude_compilations: Drop the Various-Artists rows ("studio albums only").
        live_only: Only live recordings ("just the live takes"). This clears
            ``exclude_live`` so the pair cannot cancel out to an empty pool.
        exclude_live: Whether live rows are kept out of the pool; the analysis says
            False when the prompt asked for live takes.
        rated_only / unrated_only: Rows chosen by whether the listener rated them
            (Plex leaves an unrated track NULL, never 0).
        min_rating / max_rating: User-rating window on Plex's 0-10 scale; 0 means
            unbounded on that side.
        played_after / played_before: Last-heard window as ISO ``YYYY-MM-DD`` dates
            ("" = unbounded), enforced against Plex's ``lastViewedAt``.
        confidence: 0-1 confidence that the suggestion is useful.
        method: ``"rules"``, ``"embeddings"``, or ``"none"``.
        reasoning: Human-readable explanation, surfaced in the API response.
        matched_hints: Prompt keyword → library genres that keyword produced.
            Keywords that matched nothing in the library map to an empty list.
            Genre/decade evidence only — artists, cues and length windows are
            reported in their own fields.
        excluded_genres: Library genres that were scored and deliberately left
            out of the suggestion (embedding near misses).
    """

    genres: list[str] = field(default_factory=list)
    decades: list[str] = field(default_factory=list)
    artists: list[str] = field(default_factory=list)
    negated_genres: list[str] = field(default_factory=list)
    exclude_artists: list[str] = field(default_factory=list)
    exclude_decades: list[str] = field(default_factory=list)
    style_cues: list[str] = field(default_factory=list)
    min_duration_ms: int = 0
    max_duration_ms: int = 0
    min_rating: int = 0
    min_year: int = 0
    max_year: int = 0
    min_plays: int = 0
    max_plays: int = 0
    unheard_only: bool = False
    exclude_compilations: bool = False
    live_only: bool = False
    exclude_live: bool = True
    rated_only: bool = False
    unrated_only: bool = False
    max_rating: int = 0
    played_after: str = ""
    played_before: str = ""
    confidence: float = 0.0
    method: str = "none"
    reasoning: str = ""
    matched_hints: dict[str, list[str]] = field(default_factory=dict)
    excluded_genres: list[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        """Whether confidence clears :data:`CONFIDENCE_THRESHOLD`."""
        narrowed = bool(
            self.genres
            or self.decades
            or self.artists
            or self.negated_genres
            or self.exclude_artists
            or self.exclude_decades
            or self.min_duration_ms
            or self.max_duration_ms
            or self.min_rating
            or self.max_rating
            or self.min_year
            or self.max_year
            or self.min_plays
            or self.max_plays
            or self.unheard_only
            or self.exclude_compilations
            or self.live_only
            or self.rated_only
            or self.unrated_only
            or self.played_after
            or self.played_before
        )
        return self.confidence >= CONFIDENCE_THRESHOLD and narrowed


def classify_prompt(
    prompt: str,
    available_genres: list[str],
    available_decades: list[str],
    threshold: float = CONFIDENCE_THRESHOLD,
    available_artists: list[str] | None = None,
) -> PromptClassification:
    """Classify *prompt* into library genres/decades without an LLM call.

    Tries keyword rules first, then embedding similarity.  When neither clears
    *threshold* the returned ``method`` is ``"none"`` and the caller is
    expected to fall back to the LLM.

    Beyond genres and decades the prompt can also name artists, rule genres,
    artists and decades out, ask for a recording-year window, a play-count
    window or never-played tracks ("hidden gems"), a quality bar, a length, and
    describe a mood; those come back in ``artists`` / ``exclude_artists`` /
    ``exclude_decades`` / ``min_year`` / ``max_year`` / ``min_plays`` /
    ``max_plays`` / ``unheard_only`` / ``exclude_compilations`` /
    ``negated_genres`` / ``style_cues`` / the duration window / ``min_rating``.
    The newest facets read columns Plex already fills: only-live rows
    (``live_only``), a last-heard window (``played_after`` / ``played_before``),
    rated vs. unrated rows (``rated_only`` / ``unrated_only``) and a rating
    ceiling (``max_rating``). Each counts as extra evidence, raising confidence.

    Contradictory readings are resolved the same way the request models raise on
    them: a last-heard window outranks "hidden gems" (never-played rows have no
    last-heard date), an unrated request outranks a rating floor, and a stated
    ceiling outranks the floor words the same phrasing contains. Asking for live
    rows also clears ``exclude_live``, since both filter layers treat ``live_only``
    as authoritative.

    Args:
        prompt: The user's natural-language request.
        available_genres: Genre names present in the user's library.
        available_decades: Decade names present in the user's library.
        threshold: Minimum confidence to accept a local result.
        available_artists: Artist names present in the library; without a
                           roster no artist can be recognised.

    Returns:
        A :class:`PromptClassification`.
    """
    prompt_lower = prompt.lower()
    hint_groups = extract_genre_hint_groups(prompt_lower)
    hinted_decades = extract_decade_hints(prompt_lower)

    # Map each hint group onto the library, counting how many distinct prompt
    # keywords actually produced a usable genre. The per-keyword mapping is kept
    # as evidence so the response can say *why* each filter was chosen.
    available_by_norm = {_normalize(name): name for name in available_genres}
    matched_targets: set[str] = set()
    keyword_hits = 0
    hint_evidence: dict[str, list[str]] = {}
    for keyword, candidates in hint_groups.items():
        mapped = []
        for genre in candidates:
            name = _map_candidate(genre, available_by_norm)
            if name:
                mapped.append(name)
                matched_targets.add(name)
        hint_evidence[keyword] = _dedupe(mapped)
        if mapped:
            keyword_hits += 1
    # Facets that need no keyword map: artists, exclusions, length window,
    # rating floor, soft cues. Exclusions are read first because an explicit
    # "no metal" outranks the Metal that the word "rock" would otherwise add.
    matched_artists = match_artists(prompt_lower, available_artists or [])
    negated_genres = extract_negated_genres(prompt_lower, available_genres)
    negated_set = set(negated_genres)
    excluded_artists = extract_negated_artists(prompt_lower, available_artists or [])
    # A stated ban wins over a positive mention, same as negated genres do:
    # "progressive but not Yes" must not also ask for Yes.
    banned_artists = set(excluded_artists)
    matched_artists = [a for a in matched_artists if a not in banned_artists]
    excluded_decades = extract_negated_decades(prompt_lower, available_decades)
    style_cues = extract_style_cues(prompt_lower)
    min_duration_ms, max_duration_ms = extract_duration_window(prompt_lower)
    rating_floor = extract_rating_floor(prompt_lower)
    min_plays, max_plays, unheard_only = extract_play_window(prompt_lower)
    min_year, max_year = extract_year_window(prompt_lower)
    exclude_compilations = extract_compilation_exclusion(prompt_lower)
    live_only = extract_live_request(prompt_lower)
    rated_only, unrated_only = extract_rating_presence(prompt_lower)
    rating_floor, rating_ceiling = extract_rating_window(prompt_lower)
    played_after, played_before = extract_last_played_window(prompt_lower)
    if played_after or played_before:
        # "haven't heard anything since 2023" is played-but-stale: the window is
        # the more specific reading, and a never-played row has no last-heard
        # date to sit inside one (the request models raise on this pair too).
        unheard_only = False
    if unrated_only:
        # An unrated row carries no rating, so it cannot clear a rating floor.
        rating_floor = 0
    # Evidence for the facets no genre backs: the phrase that opened each, keyed
    # to the filter field it set, so the UI can explain the suggestion.
    facet_evidence: dict[str, list[str]] = {}
    if live_only:
        phrase = _phrase_matched(prompt_lower, LIVE_ONLY_WORDS)
        if phrase:
            facet_evidence[phrase] = ["live_only"]
    if unrated_only:
        phrase = _phrase_matched(prompt_lower, UNRATED_ONLY_WORDS)
        if phrase:
            facet_evidence[phrase] = ["unrated_only"]
    if rated_only:
        phrase = _phrase_matched(prompt_lower, RATED_ONLY_WORDS)
        if phrase:
            facet_evidence[phrase] = ["rated_only"]
    if rating_ceiling:
        phrase = _phrase_matched(prompt_lower, RATING_CEILING_WORDS)
        if phrase:
            facet_evidence[phrase] = ["max_rating"]
    if played_after or played_before:
        phrase = _last_played_phrase(prompt_lower)
        if phrase:
            facet_evidence[phrase] = ["played_after", "played_before"]
    hint_evidence = {**hint_evidence, **facet_evidence}

    matched_genres = [
        name for name in available_genres if name in matched_targets and name not in negated_set
    ]

    decade_lookup = {d.lower(): d for d in available_decades}
    matched_decades = _dedupe(
        decade_lookup[d.lower()] for d in hinted_decades if d.lower() in decade_lookup
    )
    # "not the 80s" and "only the 80s" cannot both hold; a stated ban wins.
    matched_decades = [d for d in matched_decades if d not in set(excluded_decades)]

    facet_conf = _facet_confidence(
        bool(matched_artists),
        bool(negated_genres),
        bool(min_duration_ms or max_duration_ms),
        rating_floor,
        bool(style_cues),
        excluded_artists=bool(excluded_artists),
        excluded_decades=bool(excluded_decades),
        plays=bool(min_plays or max_plays),
        unheard=unheard_only,
        year=bool(min_year or max_year),
        compilations=exclude_compilations,
        live=live_only,
        rated=rated_only or unrated_only,
        ceiling=bool(rating_ceiling),
        last_played=bool(played_after or played_before),
    )

    rule_conf = _rule_confidence(keyword_hits, matched_genres, matched_decades)
    if rule_conf + facet_conf >= threshold:
        confidence = _combine_confidence(rule_conf, facet_conf)
        return PromptClassification(
            genres=matched_genres,
            decades=matched_decades,
            artists=matched_artists,
            negated_genres=negated_genres,
            style_cues=style_cues,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            min_rating=rating_floor,
            min_year=min_year,
            max_year=max_year,
            min_plays=min_plays,
            max_plays=max_plays,
            unheard_only=unheard_only,
            exclude_compilations=exclude_compilations,
            exclude_artists=excluded_artists,
            exclude_decades=excluded_decades,
            live_only=live_only,
            exclude_live=not live_only,
            rated_only=rated_only,
            unrated_only=unrated_only,
            max_rating=rating_ceiling,
            played_after=played_after,
            played_before=played_before,
            confidence=confidence,
            method="rules",
            reasoning=_describe(
                "keyword rules",
                matched_genres,
                matched_decades,
                confidence,
                hint_evidence,
                artists=matched_artists,
                negated=negated_genres,
                duration=(min_duration_ms, max_duration_ms),
                rating=rating_floor,
                cues=style_cues,
                banned_artists=excluded_artists,
                banned_decades=excluded_decades,
                year=(min_year, max_year),
                plays=(min_plays, max_plays, unheard_only),
                compilations=exclude_compilations,
                rating_max=rating_ceiling,
                rated=rated_only,
                unrated=unrated_only,
                live_only=live_only,
                last_played=(played_after, played_before),
            ),
            matched_hints=hint_evidence,
            excluded_genres=[],
        )

    emb_genres, emb_conf, emb_excluded = _embedding_suggest(prompt, available_genres)
    if emb_conf + facet_conf >= threshold:
        genres = [
            g for g in _dedupe([*matched_genres, *emb_genres]) if g not in negated_set
        ]
        confidence = _combine_confidence(max(rule_conf, emb_conf), facet_conf)
        excluded = _dedupe(g for g in emb_excluded if g not in genres and g not in negated_genres)
        return PromptClassification(
            genres=genres,
            decades=matched_decades,
            artists=matched_artists,
            negated_genres=negated_genres,
            style_cues=style_cues,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            min_rating=rating_floor,
            min_year=min_year,
            max_year=max_year,
            min_plays=min_plays,
            max_plays=max_plays,
            unheard_only=unheard_only,
            exclude_compilations=exclude_compilations,
            exclude_artists=excluded_artists,
            exclude_decades=excluded_decades,
            live_only=live_only,
            exclude_live=not live_only,
            rated_only=rated_only,
            unrated_only=unrated_only,
            max_rating=rating_ceiling,
            played_after=played_after,
            played_before=played_before,
            confidence=confidence,
            method="embeddings",
            reasoning=_describe(
                "embedding similarity",
                genres,
                matched_decades,
                confidence,
                hint_evidence,
                artists=matched_artists,
                negated=negated_genres,
                duration=(min_duration_ms, max_duration_ms),
                rating=rating_floor,
                cues=style_cues,
                banned_artists=excluded_artists,
                banned_decades=excluded_decades,
                year=(min_year, max_year),
                plays=(min_plays, max_plays, unheard_only),
                compilations=exclude_compilations,
                rating_max=rating_ceiling,
                rated=rated_only,
                unrated=unrated_only,
                live_only=live_only,
                last_played=(played_after, played_before),
            ),
            matched_hints=hint_evidence,
            excluded_genres=excluded,
        )

    confidence = _combine_confidence(max(rule_conf, emb_conf), facet_conf)
    return PromptClassification(
        genres=matched_genres,
        decades=matched_decades,
        artists=matched_artists,
        negated_genres=negated_genres,
        style_cues=style_cues,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        min_rating=rating_floor,
        min_year=min_year,
        max_year=max_year,
        min_plays=min_plays,
        max_plays=max_plays,
        unheard_only=unheard_only,
        exclude_compilations=exclude_compilations,
        exclude_artists=excluded_artists,
        exclude_decades=excluded_decades,
        live_only=live_only,
        exclude_live=not live_only,
        rated_only=rated_only,
        unrated_only=unrated_only,
        max_rating=rating_ceiling,
        played_after=played_after,
        played_before=played_before,
        confidence=confidence,
        method="none",
        reasoning=(
            "Low confidence from keyword rules and embeddings "
            f"({confidence:.2f}); falling back to the LLM."
        ),
        matched_hints=hint_evidence,
        excluded_genres=_dedupe(emb_excluded),
    )


def _format_duration(ms: int) -> str:
    """Render milliseconds as "m:ss" for the reasoning text."""
    if not ms:
        return "any length"
    seconds = round(ms / 1000)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _describe(
    technique: str,
    genres: list[str],
    decades: list[str],
    confidence: float,
    hint_evidence: dict[str, list[str]] | None = None,
    excluded: list[str] | None = None,
    artists: list[str] | None = None,
    negated: list[str] | None = None,
    duration: tuple[int, int] = (0, 0),
    rating: int = 0,
    cues: list[str] | None = None,
    banned_artists: list[str] | None = None,
    banned_decades: list[str] | None = None,
    year: tuple[int, int] = (0, 0),
    plays: tuple[int, int, bool] = (0, 0, False),
    compilations: bool = False,
    rating_max: int = 0,
    rated: bool = False,
    unrated: bool = False,
    live_only: bool = False,
    last_played: tuple[str, str] = ("", ""),
) -> str:
    """Build the reasoning string for a confident classification.

    Names the prompt phrases that did the work, the filters they produced, and
    any library filters that were scored and left out, so the UI can explain the
    suggestion instead of just asserting it.
    """
    parts = []
    if genres:
        parts.append("genres " + ", ".join(genres))
    if decades:
        parts.append("decades " + ", ".join(decades))
    if artists:
        parts.append("artists " + ", ".join(artists))
    matched = " and ".join(parts) if parts else "no filters"

    detail = [
        (
            f"Matched {matched} from the prompt via {technique} "
            f"(confidence {confidence:.2f}); no LLM call needed."
        )
    ]

    if hint_evidence:
        phrases = [f"'{k}' → {', '.join(v)}" for k, v in hint_evidence.items() if v]
        missed = [f"'{k}' (nothing in your library)" for k, v in hint_evidence.items() if not v]
        if phrases:
            detail.append("Prompt phrases: " + "; ".join(phrases[:6]))
        if missed:
            detail.append(
                f"No library genre for: {', '.join(missed[:4])}; "
                "those phrases were ignored."
            )
    if excluded:
        detail.append(
            "Close but left out: " + ", ".join(excluded[:4])
            + " — scored below the filters above them."
        )
    if negated:
        detail.append(
            "Asked to leave out: " + ", ".join(negated[:6])
            + " — dropped as a hard exclusion."
        )
    if banned_artists:
        detail.append(
            "Artists excluded: " + ", ".join(banned_artists[:6])
            + " — the prompt named them as things to avoid."
        )
    if banned_decades:
        detail.append(
            "Decades excluded: " + ", ".join(banned_decades[:6])
            + " — the prompt asked not to use them."
        )
    lo, hi = duration
    if lo or hi:
        window = f"{_format_duration(lo)} to {_format_duration(hi)}" if lo and hi else (
            f"at least {_format_duration(lo)}" if lo else f"up to {_format_duration(hi)}"
        )
        detail.append(f"Length window: {window}.")
    y_lo, y_hi = year
    if y_lo or y_hi:
        years = f"{y_lo} to {y_hi}" if y_lo and y_hi else (f"from {y_lo} on" if y_lo else f"up to {y_hi}")
        detail.append(f"Recording years: {years}.")
    p_lo, p_hi, unheard = plays
    if unheard:
        detail.append("Plays: only tracks you have never played.")
    elif p_lo or p_hi:
        plays_window = (
            f"{p_lo} to {p_hi} plays" if p_lo and p_hi
            else f"at least {p_lo} plays" if p_lo
            else f"at most {p_hi} plays"
        )
        detail.append(f"Plays: {plays_window}.")
    if compilations:
        detail.append(
            "Compilations: dropped the Various-Artists rows, keeping original albums."
        )
    if rating:
        detail.append(f"Quality floor: user rating ≥ {rating}/10.")
    if rating_max:
        detail.append(
            f"Quality ceiling: user rating ≤ {rating_max}/10 — the prompt banned "
            "music the listener already rated higher than that; unrated rows stay."
        )
    if unrated:
        detail.append("Rating presence: only tracks you have never rated.")
    elif rated:
        detail.append("Rating presence: only tracks you have rated.")
    if live_only:
        detail.append(
            "Live rows: only recordings flagged as live; the drop-live default was "
            "cleared so the pair cannot cancel out to nothing."
        )
    p_after, p_before = last_played
    if p_after or p_before:
        window = (
            f"{p_after} to {p_before}"
            if p_after and p_before
            else f"heard on or after {p_after}"
            if p_after
            else f"heard on or before {p_before}"
        )
        detail.append(f"Last heard: {window} (Plex's lastViewedAt).")
    if cues:
        detail.append(
            "Mood cues (no library column, they steer the picks instead): "
            + "; ".join(cues[:4])
            + "."
        )

    return " ".join(detail)


__all__ = [
    "CONFIDENCE_THRESHOLD",
    "DECADE_HINTS",
    "GENRE_DESCRIPTIONS",
    "MOOD_GENRE_MAP",
    "PromptClassification",
    "classify_prompt",
    "extract_decade_hints",
    "extract_compilation_exclusion",
    "extract_duration_window",
    "extract_genre_hint_groups",
    "extract_genre_hints",
    "extract_negated_artists",
    "extract_negated_decades",
    "extract_negated_genres",
    "extract_play_window",
    "extract_rating_floor",
    "extract_style_cues",
    "extract_year_window",
    "is_genre_keyword",
    "match_artists",
    "match_genres_to_available",
]
