# PlexSage Backend — Navigation Map

> FastAPI backend for a Plex playlist generator powered by LLMs.
> This map lets you jump straight to the right file/section without re-scanning.

---

## File Overview

| File | Purpose | Key Exports |
|------|---------|-------------|
| `main.py` | FastAPI app, all HTTP routes, lifespan, static serving | `app` |
| `config.py` | Config loading (env → user yaml → base yaml → defaults) | `get_config`, `update_config_values`, `refresh_config` |
| `models.py` | All Pydantic models (entities, config, API req/resp, Ollama) | `Track`, `AppConfig`, `LLMConfig`, `PlexConfig`, request/response models |
| `plex_client.py` | Plex server client, library queries, playlist creation, caching | `PlexClient`, `get_plex_client`, `init_plex_client`, `TrackCache` |
| `track_db.py` | Local SQLite mirror of the library (filtering, FTS search, stats) | `TrackDB`, `get_track_db`, `init_track_db` |
| `llm_client.py` | Unified LLM client (Anthropic/OpenAI/Gemini/Ollama/custom), cost & context helpers | `LLMClient`, `get_llm_client`, `init_llm_client`, `estimate_cost_for_model` |
| `prompt_classifier.py` | Local prompt → genre/decade classification (keyword rules + hashing embeddings); LLM fallback trigger | `classify_prompt`, `PromptClassification`, `MOOD_GENRE_MAP`, `DECADE_HINTS` |
| `analyzer.py` | Prompt → filter suggestions (local first, LLM fallback); seed track → dimension extraction | `analyze_prompt`, `analyze_track` |
| `generator.py` | Playlist generation (sync + SSE streaming), exact + trigram-pruned fuzzy track matching | `generate_playlist`, `generate_playlist_stream`, `TrackMatcher` |
| `__init__.py` | Package marker, version | `__version__` |

---

## Architecture at a Glance

```
HTTP Request
    │
    ▼
main.py (FastAPI routes)
    │
    ├── config.py ──────────► AppConfig (Plex + LLM + Defaults)
    │
    ├── plex_client.py ─────► PlexClient (library queries, playlists, art)
    │       ├── TrackCache (TTL in-memory cache for filtered results)
    │       └── track_db.py ─► TrackDB (SQLite mirror: reads served locally)
    │
    ├── llm_client.py ──────► LLMClient (provider-agnostic completion)
    │       ├── prompt_classifier.py  (keywords + embeddings, no LLM)
    │       ├── analyzer.py  (prompt analysis via classifier, LLM fallback; track dimensions)
    │       └── generator.py (playlist generation, fuzzy matching)
    │
    └── models.py ──────────► Pydantic schemas (shared contracts)
```

---

## API Endpoints (all in `main.py`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/health` | `health_check` | Plex + LLM status |
| GET | `/api/config` | `get_configuration` | Current config (no secrets) |
| POST | `/api/config` | `update_configuration` | Update config, reinit clients |
| GET | `/api/ollama/status` | `ollama_status` | Ollama connection check |
| GET | `/api/ollama/models` | `ollama_models` | List Ollama models |
| GET | `/api/ollama/model-info` | `ollama_model_info` | Model details (context window) |
| GET | `/api/library/stats` | `get_library_stats` | Total tracks, genres, decades |
| GET | `/api/library/search` | `search_library` | Search tracks by query |
| POST | `/api/library/sync` | `sync_library` | Mirror Plex library into local SQLite DB |
| POST | `/api/analyze/prompt` | `analyze_prompt` | Prompt → suggested filters |
| POST | `/api/analyze/track` | `analyze_track` | Seed track → dimensions |
| POST | `/api/filter/preview` | `preview_filters` | Filter count + cost estimate |
| POST | `/api/generate` | `generate_playlist` | Generate playlist (sync) |
| POST | `/api/generate/stream` | `generate_playlist_sse` | Generate playlist (SSE) |
| POST | `/api/playlist` | `save_playlist` | Save playlist to Plex |
| GET | `/api/art/{rating_key}` | `get_album_art` | Album art proxy (hides token) |
| GET | `/` | `serve_index` | Serve frontend `index.html` |
| — | `/static` | — | Static file mount (frontend) |

---

## Key Flows

### 1. Prompt-First Flow
```
User types prompt
  → POST /api/analyze/prompt
    → analyzer.analyze_prompt()
      → plex_client.get_library_stats()      (available genres/decades)
      → prompt_classifier.classify_prompt()  (keyword rules → embeddings)
         ├── confident? → return, 0 tokens   (method "rules" | "embeddings")
         └── low confidence → llm_client.analyze()  (method "llm")
  → User picks filters
  → POST /api/filter/preview             (count + cost)
  → POST /api/generate
    → generator.generate_playlist()
      → plex_client.get_tracks_by_filters()  (or random sample)
      → generator.compute_track_budget()    (dynamic candidate count from prompt)
      → generator.pre_rank_tracks()          (rules; uses prompt_classifier hints)
      → llm_client.generate()              (selects tracks)
      → TrackMatcher.match()               (exact, then trigram-pruned fuzzy)
  → POST /api/playlist                    (save to Plex)
```

### 2. Seed-Track Flow
```
User picks a track
  → POST /api/analyze/track
    → analyzer.analyze_track()
      → track_db.get_dimensions()          (cache hit? model + metadata fingerprint)
      → llm_client.analyze()             (extracts 5-7 dimensions; cached in track_dimensions)
  → User picks dimensions
  → POST /api/generate (with seed_track)
    → generator.generate_playlist()
      → same as above, but prompt includes seed + dimensions
```

### 3. SSE Streaming (generator.py)
`generate_playlist_stream()` yields events:
`progress` (fetching → filtering → preparing → ai_working → parsing → matching → complete) or `error` or `complete`

---

## Config Priority Chain (`config.py`)

```
1. Environment variables        (highest)   e.g. PLEX_URL, LLM_PROVIDER, ANTHROPIC_API_KEY
2. config.user.yaml             (UI-saved)  written by update_config_values()
3. config.yaml                  (base)
4. Hardcoded defaults           (lowest)
```

- `MODEL_DEFAULTS` dict maps provider → default analysis/generation models
- Provider auto-detection: gemini > openai > anthropic (by API key presence)
- `update_config_values()` persists to `config.user.yaml` and reinitializes clients

---

## LLM Providers (`llm_client.py`)

| Provider | Client | Notes |
|----------|--------|-------|
| `anthropic` | `anthropic.Anthropic` | `messages.create()` |
| `openai` | `openai.OpenAI` | `chat.completions.create()` |
| `gemini` | `google.genai.Client` | Retry logic for truncation (MAX_TOKENS) |
| `ollama` | `httpx` (direct) | `/api/generate`, 600s timeout |
| `custom` | `openai.OpenAI` (base_url) | Any OpenAI-compatible endpoint |

### Helper Functions
- `estimate_cost_for_model(model, in, out, config)` → USD (0 for local)
- `get_max_tracks_for_model(model, config)` → max tracks from context window
- `get_model_context_limit(model, config)` → tokens (128k fallback)
- `get_model_cost(model, config)` → `{input, output}` per million tokens
- `list_ollama_models(url)` / `get_ollama_model_info(url, name)` / `get_ollama_status(url)`

### Cost Table (`MODEL_COSTS`)
| Model | Input $/M | Output $/M |
|-------|-----------|------------|
| claude-sonnet-4-5 | 3.00 | 15.00 |
| claude-haiku-4-5 | 0.25 | 1.25 |
| gpt-4.1 | 2.00 | 8.00 |
| gpt-4.1-mini | 0.40 | 1.60 |
| gemini-2.5-pro | 1.25 | 5.00 |
| gemini-2.5-flash | 0.075 | 0.30 |

---

## Prompt Classifier (`prompt_classifier.py`)

Local replacement for the LLM prompt-analysis call. Returns a
`PromptClassification(genres, decades, confidence, method, reasoning)`.

| Stage | What it does | Method value |
|-------|--------------|--------------|
| Keyword rules | `MOOD_GENRE_MAP` (mood/activity/genre → genres) + `DECADE_HINTS` (era words → decades). Hints are mapped onto actual library genres via exact → substring → fuzzy (`difflib`). | `"rules"` |
| Embeddings | Feature-hashed bag-of-words vectors (stdlib `hashlib`; word unigrams + bigrams). Prompt vs each library genre's `GENRE_DESCRIPTIONS` text, cosine similarity. Catches genre words rules don't know (e.g. "shoegaze"). | `"embeddings"` |
| Fallback | Confidence below `CONFIDENCE_THRESHOLD` (0.55) → analyzer calls the LLM. | `"none"` → `"llm"` |

- No external dependencies and no network calls; `_embed()` is the seam to drop in a real model later.
- Confidence is based on the number of distinct prompt *keywords* that matched (or the top embedding similarity), so broad hint groups don't inflate it.
- `generator.pre_rank_tracks()` reuses `extract_genre_hints` / `extract_decade_hints` from this module as the single source of truth for the hint dictionaries.

---

## Dynamic Track Budget (`generator.py`)

`compute_track_budget()` decides how many pre-ranked candidates the LLM sees,
trading fewer options (cheaper) against enough variety. It interpolates between
a floor and a ceiling using a 0-1 specificity score:

- **Ceiling** = the tighter of `PRE_RANK_LIMIT` (500), the user's
  `max_tracks_to_ai` (`0` = no limit), and the candidate pool size.
- **Floor** = `max(BUDGET_FLOOR, track_count × MIN_TRACKS_PER_PICK)` so the LLM
  always has enough options for the requested playlist size.
- **Specificity** (`_prompt_specificity`) rises with concrete genre keywords
  (`is_genre_keyword`), decade hints, and explicitly selected genre/decade
  filters; a bare mood/activity prompt stays low. `score 0 → ceiling`,
  `score 1 → floor`.

Both `generate_playlist`/`generate_playlist_stream` and the
`/api/filter/preview` endpoint (optional `prompt` field) call it, so the preview
matches the real request. With no prompt or filters the ceiling is unchanged, so
the seed flow behaves exactly as before.

---

## Track Matching (`generator.py`)

Resolves the LLM's `{artist, title}` selections against the pre-ranked candidate
list. `TrackMatcher` indexes the candidates **once per generation** (simplified
title/artist strings, an exact-match dict, and a title-trigram inverted index),
then resolves each selection in roughly `O(k)` instead of scanning all `n`
tracks with `rapidfuzz`:

1. **Exact match** — normalized `(artist, title)` lookup, including `&`/`and`
   artist variants. O(1), resolves most selections.
2. **Trigram candidates** — tracks sharing a title trigram with the selection,
   fuzzy-compared via `_fuzzy_track_match()`. Titles are the blocking key because
   the title must match first. When the overlap is too broad (more than ~25% of
   candidates), it falls back to a linear scan, which still benefits from the
   precomputed keys.
3. **Full-scan fallback** — covers tracks trigrams didn't surface (e.g. titles
   shorter than a trigram), so a genuine match is never dropped.

`_tracks_match()` remains as a single-pair convenience wrapper. The candidate
list is already a small pre-ranked in-memory set, so an in-memory index is
preferred over a SQLite FTS round-trip: it avoids I/O and preserves the
pre-rank ordering used for deterministic tie-breaking.

---

## Plex Client (`plex_client.py`)

### `PlexClient` Methods
| Method | Purpose |
|--------|---------|
| `_connect()` | Connect via `PlexServer`, grab library section |
| `is_connected()` | Check server + library are valid |
| `get_library_stats()` | Genres/decades + counts (from TrackDB if synced, else Plex filter choices) |
| `get_all_tracks()` | All tracks as `Track` models |
| `get_tracks_by_filters()` | Filters + live exclusion (from TrackDB if synced, else Plex) |
| `count_tracks_by_filters()` | Fast count (from TrackDB if synced, else Plex) |
| `get_random_tracks()` | Random sample (from TrackDB if synced, else Plex sort) |
| `search_tracks()` | Title/artist/album search (TrackDB FTS if synced, else Plex) |
| `get_track_by_key()` | Single track by ratingKey |
| `get_thumb_path()` | Raw thumb path for art proxy |
| `create_playlist()` | Create playlist, returns URL + counts |
| `sync_track_db()` | Mirror the whole library into `TrackDB` (ratings, plays, live flags) |
| `ensure_synced()` | Sync only if the DB is empty or stale |
| `_get_track_db()` | Return the populated TrackDB, or None to use the API |
| `_convert_track()` | Plex object → `Track` model |

### `TrackCache`
- In-memory, TTL 300s, max 50 entries
- Key = MD5 of (genres, decades, exclude_live, min_rating)
- Used by `generator.py` to avoid re-fetching on repeat generations

### `TrackDB` (`track_db.py`)
- File-backed SQLite mirror of the library (`backend/plexsage.db`); stdlib only.
- **Serving reads:** `PlexClient` consults the DB first when it has rows, so
  `get_library_stats`, `get_tracks_by_filters`, `count_tracks_by_filters`,
  `get_random_tracks`, and `search_tracks` avoid Plex HTTP round-trips. Empty or
  erroring DB → transparent fallback to the Plex API.
- **Sync:** `PlexClient.sync_track_db()` fetches every track once (rating,
  play count, last-played, live flag) and prunes removed tracks. The app runs it
  on startup and every `SYNC_INTERVAL_SECONDS` (30 min) via a background task in
  the `main.py` lifespan; `POST /api/library/sync` triggers it on demand.
- **Tables:** `tracks` (incl. nullable `embedding` BLOB), `track_genres`,
  `track_dimensions` (cached seed-track analysis: dimensions JSON + model +
  metadata fingerprint), `tracks_fts` (FTS5 over title/artist/album), `meta`
  (last sync time).
- **Migrations:** missing columns are added on startup (`_migrate`); the FTS
  index is backfilled when empty.

### Live Detection (`is_live_version`)
- Checks `parentTitle` (album) and `title` for date patterns or keywords (live, concert, sbd, bootleg)
- **Pre-computed once at sync time** and stored in `tracks.is_live`; DB queries filter with `WHERE is_live = 0` instead of post-filtering in Python.
- The flag is surfaced as `Track.is_live` on DB reads and via `TrackDB.get_live_flags(rating_keys)`. The Plex API fallback (only used when the DB is empty) detects once per fetched track and stores it on the returned `Track`.

---

## Models (`models.py`) — Quick Reference

### Core Entities
- **`Track`** — rating_key, title, artist, album, duration_ms, year, genres, is_live, art_url
- **`Dimension`** — id, label, description (from seed track analysis)
- **`FilterSet`** — genres, decades, track_count (15/25/40), exclude_live
- **`Playlist`** — name, tracks, source_prompt, seed_track_key, selected_dimensions

### Config
- **`PlexConfig`** — url, token, music_library
- **`LLMConfig`** — provider, api_key, model_analysis, model_generation, smart_generation, ollama/custom settings
- **`DefaultsConfig`** — track_count (default 25)
- **`AppConfig`** — plex + llm + defaults

### API Request/Response (grouped by endpoint)
- Health: `HealthResponse`
- Config: `ConfigResponse`, `UpdateConfigRequest`
- Library: `LibraryStatsResponse`, `GenreCount`, `DecadeCount`
- Analysis: `AnalyzePromptRequest/Response` (response includes `method` and `confidence`), `AnalyzeTrackRequest/Response`
- Filter: `FilterPreviewRequest/Response`
- Generate: `GenerateRequest`, `GenerateResponse`, `SeedTrackInput`
- Playlist: `SavePlaylistRequest/Response`
- Ollama: `OllamaModel`, `OllamaModelInfo`, `OllamaModelsResponse`, `OllamaStatus`

---

## Where to Look for Common Tasks

| Task | File | Section/Function |
|------|------|-----------------|
| Add a new API endpoint | `main.py` | Add route + import model from `models.py` |
| Add a new LLM provider | `llm_client.py` | Add `_complete_<provider>()` + branch in `_complete()` |
| Change cost estimates | `llm_client.py` | `MODEL_COSTS` dict |
| Change context limits | `llm_client.py` | `MODEL_CONTEXT_LIMITS` dict |
| Add a new filter type | `plex_client.py` | `_build_filters()` + `get_tracks_by_filters()` |
| Change live detection | `plex_client.py` | `is_live_version()`, `LIVE_KEYWORDS`, `DATE_PATTERN` |
| Change live filtering | `track_db.py` | `tracks.is_live`, `_build_where()`, `get_live_flags()` |
| Change fuzzy matching | `generator.py` | `TrackMatcher`, `_fuzzy_track_match()`, `_trigrams()`, `FUZZ_THRESHOLD` in `plex_client.py` |
| Change the dynamic AI track budget | `generator.py` | `compute_track_budget()`, `_prompt_specificity()`, `BUDGET_FLOOR`, `MIN_TRACKS_PER_PICK` |
| Change generation prompt | `generator.py` | `GENERATION_SYSTEM` |
| Change analysis prompts | `analyzer.py` | `PROMPT_ANALYSIS_SYSTEM`, `TRACK_ANALYSIS_SYSTEM` |
| Change prompt classification rules/descriptions | `prompt_classifier.py` | `MOOD_GENRE_MAP`, `DECADE_HINTS`, `GENRE_DESCRIPTIONS`, `CONFIDENCE_THRESHOLD` |
| Add a new config field | `models.py` + `config.py` | Add to `LLMConfig`/`PlexConfig` + load in `load_config()` |
| Change SSE events | `generator.py` | `generate_playlist_stream()` |
| Change caching behavior | `plex_client.py` | `TrackCache` class |
| Change the library mirror / sync schedule | `track_db.py` + `plex_client.py` | `TrackDB` schema, `sync_track_db()`, `SYNC_INTERVAL_SECONDS` |
| Change searchable fields | `track_db.py` | `tracks_fts` virtual table, `search_tracks()` |
| Change config priority | `config.py` | `load_config()` |
| Add a new Pydantic model | `models.py` | Add class, import in `main.py` |

---

## Dependencies (from imports)

- `fastapi`, `starlette` — HTTP framework
- `pydantic` — data models
- `httpx` — async HTTP (art proxy, Ollama)
- `anthropic`, `openai`, `google-genai` — LLM SDKs
- `plexapi` — Plex server API
- `rapidfuzz` — fuzzy string matching
- `unidecode` — unicode normalization
- `pyyaml`, `python-dotenv` — config loading
- `requests` — (via plexapi)
- `sqlite3`, `hashlib`, `difflib` — stdlib (local TrackDB + prompt classifier)

---

## Global Singletons

| Variable | Module | Type |
|----------|--------|------|
| `_config` | `config.py` | `AppConfig` |
| `_plex_client` | `plex_client.py` | `PlexClient` |
| `_llm_client` | `llm_client.py` | `LLMClient` |
| `_track_cache` | `plex_client.py` | `TrackCache` |
| `_track_db` | `track_db.py` | `TrackDB` |

All initialized in `main.py` lifespan or via `init_*` functions.
</contents>