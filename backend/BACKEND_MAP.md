# PlexSage Backend — Navigation Map

> FastAPI backend for a Plex playlist generator powered by LLMs.
> This map lets you jump straight to the right file/section without re-scanning.

---

## File Overview

| File | Purpose | Key Exports |
|------|---------|-------------|
| `main.py` | FastAPI app, all HTTP routes, lifespan, static serving | `app` |
| `config.py` | Config loading (env → user yaml → base yaml → defaults) | `get_config`, `update_config_values`, `load_user_yaml_config`, `save_user_config` |
| `models.py` | All Pydantic models (entities, config, API req/resp, library cache, recommend, Ollama) | `Track`, `AppConfig`, `LLMConfig`, `PlexConfig`, request/response models |
| `plex_client.py` | Plex server client, library queries, playlist/queue creation, art, TTL cache | `PlexClient`, `get_plex_client`, `init_plex_client`, `TrackCache`, `is_live_version` |
| `library_cache.py` | Local SQLite cache: library mirror, sync progress, filters/search/stats, saved results, seed-track dimensions | `sync_library`, `get_tracks_by_filters`, `count_tracks_by_filters`, `save_result`, `get_dimensions`, `save_dimensions`, `get_metadata_batch` |
| `llm_client.py` | Unified LLM client (Anthropic/OpenAI/Gemini/Ollama/custom), cost & context helpers | `LLMClient`, `get_llm_client`, `init_llm_client`, `estimate_cost_for_model` |
| `prompt_classifier.py` | Local prompt → genre/decade classification (keyword rules + hashing embeddings); LLM fallback trigger | `classify_prompt`, `PromptClassification`, `extract_genre_hints`, `extract_decade_hints` |
| `analyzer.py` | Prompt → filter suggestions (local first, LLM fallback); seed track → dimension extraction (cached) | `analyze_prompt`, `analyze_track` |
| `generator.py` | Streaming playlist generation, dynamic candidate budget, rule-based pre-ranking, fuzzy track matching | `generate_playlist_stream`, `compute_track_budget`, `pre_rank_tracks`, `_tracks_match` |
| `recommender.py` | Album-recommendation pipeline (4 LLM calls) with in-memory session state | `RecommendationPipeline` |
| `music_research.py` | MusicBrainz + Wikipedia lookups to ground pitches in verifiable facts | `MusicResearchClient` |
| `version.py` | Version detection from git tags or env | `get_version` |
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
    │       └── TrackCache (TTL in-memory cache for filtered results)
    │
    ├── library_cache.py ───► SQLite cache (data/library_cache.db)
    │       ├── tracks / sync_state / results / track_dimensions
    │       └── served locally: filters, counts, search, stats, candidates
    │
    ├── llm_client.py ──────► LLMClient (provider-agnostic completion)
    │       ├── prompt_classifier.py  (keywords + embeddings, no LLM)
    │       ├── analyzer.py  (prompt analysis via classifier, LLM fallback; track dimensions)
    │       ├── generator.py (streaming generation, budget, pre-rank, matching)
    │       ├── recommender.py (album recommendation pipeline)
    │       └── music_research.py (MusicBrainz/Wikipedia grounding)
    │
    └── models.py ──────────► Pydantic schemas (shared contracts)
```

---

## API Endpoints (all in `main.py`)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/health` | `health_check` | Plex + LLM status |
| GET | `/api/setup/status` | `setup_status` | First-run wizard state |
| POST | `/api/setup/validate-plex` | `validate_plex` | Test Plex credentials |
| POST | `/api/setup/validate-ai` | `validate_ai` | Test LLM credentials |
| POST | `/api/setup/complete` | `setup_complete` | Persist wizard config |
| GET | `/api/config` | `get_configuration` | Current config (no secrets) |
| POST | `/api/config` | `update_configuration` | Update config, reinit clients |
| GET | `/api/ollama/status` | `ollama_status` | Ollama connection check |
| GET | `/api/ollama/models` | `ollama_models` | List Ollama models |
| GET | `/api/ollama/model-info` | `ollama_model_info` | Model details (context window) |
| GET | `/api/library/status` | `get_library_status` | Cache status + sync progress |
| POST | `/api/library/sync` | `trigger_library_sync` | Start a background library sync |
| GET | `/api/library/stats` | `get_library_stats` | Total tracks, genres, decades (Plex) |
| GET | `/api/library/stats/cached` | `get_library_stats_cached` | Genre/decade chips from the cache |
| GET | `/api/library/search` | `search_library` | Search tracks by query |
| POST | `/api/analyze/prompt` | `analyze_prompt` | Prompt → suggested filters |
| POST | `/api/analyze/track` | `analyze_track` | Seed track → dimensions (cached) |
| POST | `/api/filter/preview` | `preview_filters` | Filter count + cost estimate |
| POST | `/api/generate/stream` | `generate_playlist_sse` | Generate playlist (SSE) |
| POST | `/api/playlist` | `save_playlist` | Save playlist to Plex |
| POST | `/api/playlist/update` | `update_playlist` | Update an existing Plex playlist |
| GET | `/api/plex/clients` | `list_plex_clients` | Available Plex players |
| GET | `/api/plex/playlists` | `list_plex_playlists` | Existing Plex playlists |
| POST | `/api/play-queue` | `play_queue` | Start playback on a Plex client |
| GET | `/api/recommend/albums/preview` | `recommend_albums_preview` | Album candidates |
| POST | `/api/recommend/analyze-prompt` | `recommend_analyze_prompt` | Recommend-mode prompt analysis |
| POST | `/api/recommend/questions` | `recommend_questions` | Follow-up questions |
| POST | `/api/recommend/switch-mode` | `recommend_switch_mode` | Switch recommendation mode |
| POST | `/api/recommend/generate` | `recommend_generate` | Run the recommendation pipeline |
| GET | `/api/results` | `list_results` | Saved playlist/result history |
| GET | `/api/results/{result_id}` | `get_result` | One saved result |
| DELETE | `/api/results/{result_id}` | `delete_result` | Delete a saved result |
| GET | `/api/art/{rating_key}` | `get_album_art` | Album art proxy (hides token) |
| GET | `/api/external-art` | `get_external_art` | Proxy external art (Cover Art Archive) |
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
  → POST /api/filter/preview               (count + cost; budget uses prompt)
  → POST /api/generate/stream
    → generator.generate_playlist_stream()
      → library_cache.get_tracks_by_filters()  (or Plex / random sample)
      → generator.compute_track_budget()     (dynamic candidate count)
      → generator.pre_rank_tracks()          (rules; rating + recency from cache)
      → llm_client.generate()                (selects tracks)
      → _tracks_match()                      (rapidfuzz title/artist match)
      → library_cache.save_result()          (history)
  → POST /api/playlist                       (save to Plex)
```

### 2. Seed-Track Flow
```
User picks a track
  → POST /api/analyze/track
    → analyzer.analyze_track()
      → library_cache.get_dimensions()       (hit? model + metadata fingerprint)
      → llm_client.analyze()                 (extracts 5-7 dimensions; cached in track_dimensions)
  → User picks dimensions
  → POST /api/generate/stream (with seed_track)
    → generator.generate_playlist_stream()
      → same as above, but the prompt includes the seed + chosen dimensions
```

### 3. Generation (generator.py)

There is a single generation path, `generate_playlist_stream()` (SSE,
`/api/generate/stream`). It emits `progress` events
(`fetching → filtering → pre_ranking → preparing → ai_working → parsing →
matching → narrative → complete`), batched `tracks` events (5 per event, to work
around iOS Safari SSE buffering limits), a `narrative` event, and a final
`complete` event (or `error`).

Candidates come from `_get_tracks_from_cache_or_plex()` (local cache first, Plex
fallback), then `compute_track_budget()` sizes the pool and `pre_rank_tracks()`
trims it locally before any tokens are spent.

### 4. Library Sync (library_cache.py)

```
POST /api/library/sync  → background thread: library_cache.sync_library(plex_client)
                        → progress polled via GET /api/library/status
startup / migration     → automatic re-sync in the main.py lifespan
```

`sync_library()` walks albums/tracks in batches (`SYNC_BATCH_SIZE` = 500) and
upserts each batch, so peak memory tracks the batch size rather than the whole
library. Live detection (`_is_live_version`) is computed once per track at sync
and stored in `tracks.is_live`.

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
| `ollama` | `httpx` (direct) | `/api/generate`, long timeout |
| `custom` | `openai.OpenAI` (base_url) | Any OpenAI-compatible endpoint |

### Helper Functions
- `estimate_cost_for_model(model, in, out, config)` → USD (0 for local)
- `get_max_tracks_for_model(model, config)` → max tracks from context window
- `get_model_context_limit(model, config)` → tokens
- `get_model_cost(model, config)` → `{input, output}` per million tokens
- `list_ollama_models(url)` / `get_ollama_model_info(url, name)` / `get_ollama_status(url)`

Responses are parsed with `json_repair` so slightly malformed LLM JSON still
yields a usable object.

---

## Prompt Classifier (`prompt_classifier.py`)

Local replacement for the LLM prompt-analysis call. Returns a
`PromptClassification(genres, decades, confidence, method, reasoning)`.

| Stage | What it does | Method value |
|-------|--------------|--------------|
| Keyword rules | Mood/activity/genre hint groups → genres + era words → decades. Hints are mapped onto actual library genres via exact → substring → fuzzy (`difflib`). | `"rules"` |
| Embeddings | Feature-hashed bag-of-words vectors (stdlib `hashlib`; word unigrams + bigrams). Prompt vs each library genre description, cosine similarity. Catches genre words the rules don't know (e.g. "shoegaze"). | `"embeddings"` |
| Fallback | Confidence below `CONFIDENCE_THRESHOLD` (0.55) → analyzer calls the LLM. | `"none"` → `"llm"` |

- No external dependencies and no network calls; `_embed()` is the seam to drop in a real model later.
- Confidence is based on the number of distinct prompt *keywords* that matched (or the top embedding similarity), so broad hint groups don't inflate it.
- `generator.pre_rank_tracks()` and `generator._prompt_specificity()` reuse
  `extract_genre_hints` / `extract_decade_hints` / `extract_genre_hint_groups`
  from this module as the single source of truth for the hint dictionaries.

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

Both `generate_playlist_stream()` and the `/api/filter/preview` endpoint
(optional `prompt` field) call it, so the preview matches the real request.

---

## Rule-Based Pre-Ranking (`generator.py`)

`pre_rank_tracks()` scores the candidate pool and keeps the top `budget` tracks
before the LLM sees them (total max = 100):

1. **Genre match (0-40)** — overlap with prompt/filter genre hints (proportional,
   plus a small partial/substring bonus for near misses).
2. **Decade match (0-20)** — track year falls in a hinted decade.
3. **User rating (0-25)** — `library_cache.get_metadata_batch()` `user_rating`.
4. **Recency (0-15)** — exponential decay (72 h half-life) on `last_viewed_at`.

Ties break on `rating_key` for deterministic ordering. If the pool is already
within budget, the list is returned untouched (no metadata fetch).

---

## Track Matching (`generator.py`)

`_tracks_match()` resolves each LLM `{artist, title}` selection against the
candidate list using `rapidfuzz.fuzz.ratio` on `simplify_string()`-normalized
titles, then on artist variants from `normalize_artist()`; anything below
`FUZZ_THRESHOLD` (in `plex_client.py`) is rejected. The candidate list is a small
pre-ranked in-memory set, so a linear scan is acceptable.

---

## Plex Client (`plex_client.py`)

### `PlexClient` Methods
| Method | Purpose |
|--------|---------|
| `_connect()` | Connect via `PlexServer`, grab library section |
| `is_connected()` | Check server + library are valid |
| `get_library_stats()` | Genres/decades + counts |
| `get_all_tracks()` / `get_all_albums_metadata()` | Library as `Track` models |
| `get_tracks_by_filters()` | Filters + live exclusion |
| `count_tracks_by_filters()` | Fast count |
| `get_random_tracks()` | Random sample |
| `search_tracks()` | Title/artist/album search |
| `get_track_by_key()` | Single track by ratingKey |
| `get_thumb_path()` | Raw thumb path for art proxy |
| `create_playlist()` / `update_playlist()` | Playlist write-back |
| `get_clients()` / `get_playlists()` / `play_queue()` | Playback control |
| `_convert_track()` | Plex object → `Track` model |

### `TrackCache`
- In-memory, TTL 300s, max 50 entries
- Key = MD5 of (genres, decades, exclude_live, min_rating)
- Used to avoid re-fetching on repeat generations

---

## Local Library Cache (`library_cache.py`)

- File-backed SQLite cache at `data/library_cache.db`; stdlib only.
- **Connections:** each call opens its own connection (`get_db_connection()`) with
  WAL + `busy_timeout`, so readers run concurrently with a writer.
- **Tables:** `tracks` (rating_key, title, artist, album, duration_ms, year,
  genres JSON, user_rating, is_live, parent_rating_key, view_count,
  last_viewed_at), `sync_state` (single row: server id, last sync, count),
  `results` (saved playlists/recommendations: snapshot JSON, title, subtitle,
  art key), `track_dimensions` (seed-track analysis: dimensions JSON keyed by
  rating_key + model + metadata fingerprint).
- **Reads served locally:** `has_cached_tracks()`, `get_cached_tracks()`,
  `get_tracks_by_filters()`, `count_tracks_by_filters()`,
  `get_cached_genre_decade_stats()`, `get_album_candidates()`,
  `get_album_familiarity()`, `get_metadata_batch()` (rating/recency for
  pre-ranking). `get_tracks_by_filters` applies genre filtering in Python (the
  `genres` column is JSON) and random-samples when a limit is set.
- **Dimension cache:** `get_dimensions()` / `save_dimensions()` back
  `analyzer.analyze_track()`; a model swap or edited metadata changes the
  fingerprint and misses the cache.
- **Sync:** `sync_library()` walks the library in `SYNC_BATCH_SIZE` (500) batches
  with progress in `_sync_state`, exposed by `get_sync_state()` /
  `get_sync_progress()`. `needs_resync()` and `is_cache_stale()` gate startup
  syncs; `check_server_changed()` handles a swapped Plex server.
- **Results:** `save_result()` / `get_result()` / `list_results()` /
  `delete_result()` persist generated playlists and recommendations for the
  history UI.

### Live Detection (`is_live_version`)
- Checks the album (`parentTitle`) and title for date patterns or keywords (live, concert, sbd, bootleg).
- **Pre-computed once at sync time** and stored in `tracks.is_live`; cache queries filter with `WHERE is_live = 0` instead of post-filtering in Python.

---

## Models (`models.py`) — Quick Reference

### Core Entities
- **`Track`** — rating_key, title, artist, album, duration_ms, year, genres, is_live, art_url
- **`Dimension`** — id, label, description (from seed track analysis)
- **`FilterSet`** — genres, decades, track_count, exclude_live
- **`Playlist`** — name, tracks, source_prompt, seed_track_key, selected_dimensions

### Config
- **`PlexConfig`** — url, token, music_library
- **`LLMConfig`** — provider, api_key, model_analysis, model_generation, smart_generation, ollama/custom settings
- **`DefaultsConfig`** — track_count (default 25)
- **`AppConfig`** — plex + llm + defaults

### API Request/Response (grouped by endpoint)
- Health: `HealthResponse`
- Setup: `SetupStatusResponse`, `ValidatePlexResponse`, `ValidateAIResponse`, `SetupCompleteResponse`
- Config: `ConfigResponse`, `UpdateConfigRequest`
- Library: `LibraryStatsResponse`, `LibraryCacheStatusResponse`, `SyncProgress`, `SyncTriggerResponse`, `GenreCount`, `DecadeCount`
- Analysis: `AnalyzePromptRequest/Response` (response includes `method` and `confidence`), `AnalyzeTrackRequest/Response` (includes `cached`)
- Filter: `FilterPreviewRequest` (optional `prompt`), `FilterPreviewResponse`
- Generate: `GenerateRequest`, `GenerateResponse`, `SeedTrackInput`
- Playlist: `SavePlaylistRequest/Response`, `UpdatePlaylistRequest/Response`
- Plex: `PlexClientInfo`, `PlexPlaylistInfo`, `PlayQueueRequest/Response`
- Recommend: `AlbumPreviewResponse`, `AnalyzePromptFiltersResponse`, `RecommendQuestionsResponse`, `RecommendSwitchModeResponse`
- Results: `ResultListResponse`, `ResultDetail`
- Ollama: `OllamaModel`, `OllamaModelInfo`, `OllamaModelsResponse`, `OllamaStatus`

---

## Where to Look for Common Tasks

| Task | File | Section/Function |
|------|------|-----------------|
| Add a new API endpoint | `main.py` | Add route + import model from `models.py` |
| Add a new LLM provider | `llm_client.py` | Add `_complete_<provider>()` + branch in `_complete()` |
| Change cost estimates | `llm_client.py` | `MODEL_COSTS` dict |
| Change context limits | `llm_client.py` | `MODEL_CONTEXT_LIMITS` dict |
| Add a new filter type | `plex_client.py` + `library_cache.py` | `_build_filters()` / `get_tracks_by_filters()` |
| Change live detection | `plex_client.py` / `library_cache.py` | `is_live_version()`, `_is_live_version()` |
| Change fuzzy matching | `generator.py` | `_tracks_match()`, `FUZZ_THRESHOLD` in `plex_client.py` |
| Change the dynamic AI track budget | `generator.py` | `compute_track_budget()`, `_prompt_specificity()`, `BUDGET_FLOOR`, `MIN_TRACKS_PER_PICK` |
| Change pre-ranking rules | `generator.py` | `pre_rank_tracks()` |
| Change generation prompt | `generator.py` | `GENERATION_SYSTEM` |
| Change analysis prompts | `analyzer.py` | `PROMPT_ANALYSIS_SYSTEM`, `TRACK_ANALYSIS_SYSTEM` |
| Change prompt classification rules/descriptions | `prompt_classifier.py` | hint dictionaries, `GENRE_DESCRIPTIONS`, `CONFIDENCE_THRESHOLD` |
| Add a new config field | `models.py` + `config.py` | Add to `LLMConfig`/`PlexConfig` + load in `load_config()` |
| Change SSE events | `generator.py` | `generate_playlist_stream()` |
| Change caching behavior | `plex_client.py` | `TrackCache` class |
| Change the library sync / schedule | `library_cache.py` | `sync_library()`, `SYNC_BATCH_SIZE`, `needs_resync()` |
| Change the seed-track dimension cache | `library_cache.py` + `analyzer.py` | `get_dimensions()` / `save_dimensions()` |
| Change searchable fields | `library_cache.py` | `get_tracks_by_filters()`, `get_cached_tracks()` |
| Change the recommendation pipeline | `recommender.py` | `RecommendationPipeline` |
| Change config priority | `config.py` | `load_config()` |
| Add a new Pydantic model | `models.py` | Add class, import in `main.py` |

---

## Dependencies (from imports)

- `fastapi`, `starlette` — HTTP framework
- `pydantic`, `pydantic-settings` — data models / settings
- `httpx` — async HTTP (art proxy, Ollama)
- `anthropic`, `openai`, `google-genai` — LLM SDKs
- `plexapi` — Plex server API
- `rapidfuzz` — fuzzy string matching
- `json-repair` — tolerant LLM JSON parsing
- `unidecode` — unicode normalization
- `readability-lxml`, `lxml` — HTML cleanup for research scraping
- `pyyaml`, `python-dotenv` — config loading
- `requests` — (via plexapi)
- `sqlite3`, `hashlib`, `difflib` — stdlib (local library cache + prompt classifier)

---

## Global Singletons

| Variable | Module | Type |
|----------|--------|------|
| `_config` | `config.py` | `AppConfig` |
| `_plex_client` | `plex_client.py` | `PlexClient` |
| `_llm_client` | `llm_client.py` | `LLMClient` |
| `_track_cache` | `plex_client.py` | `TrackCache` |
| `_sync_state`, `_schema_initialized` | `library_cache.py` | sync progress / schema flag |

All initialized in `main.py` lifespan or via `init_*` functions.
