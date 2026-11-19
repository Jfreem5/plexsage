# PlexSage Backend — Navigation Map

> FastAPI backend for a Plex playlist generator powered by LLMs.
> This map lets you jump straight to the right file/section without re-scanning.
>
> `../INTERFACES.md` (repo root) holds the generated contract surface — route table with
> request/response models, module signatures + constants, DB schema, frontend surface. Read that
> for *what* the files expose; read this for *where* things live and how flows walk.

---

## File Overview

| File | Purpose | Key Exports |
|------|---------|-------------|
| `main.py` | FastAPI app, all HTTP routes, lifespan, static serving | `app` |
| `config.py` | Config loading (env → user yaml → base yaml → defaults) | `get_config`, `update_config_values`, `load_user_yaml_config`, `save_user_config` |
| `models.py` | All Pydantic models (entities, config, API req/resp, library cache, recommend, Ollama) | `Track`, `AppConfig`, `LLMConfig`, `PlexConfig`, request/response models |
| `plex_client.py` | Plex server client, library queries, playlist/queue creation, art, TTL cache, the vendored plexclean join-key normalizers | `PlexClient`, `get_plex_client`, `init_plex_client`, `TrackCache`, `is_live_version`, `norm`, `norm_artist`, `song_key`, `album_key`, `artist_key`, `collect_user_ratings` |
| `library_cache.py` | Local SQLite cache: library mirror, sync progress, filters/search/stats, saved results (linked to their Plex playlist), seed-track dimensions, library join keys, per-account **star ratings**, **Spotify wish list** | `sync_library`, `get_tracks_by_filters`, `count_tracks_by_filters`, `relax_filters_for_target`, `ensure_track_keys`, `sync_user_ratings`, `ensure_user_ratings`, `user_ratings_status`, `save_result`, `link_result_playlist`, `get_dimensions`, `save_dimensions`, `get_metadata_batch`, `record_missing_songs`, `list_missing_songs` |
| `spotify_import.py` | Optional Spotify playlist import: fetch rows (Spotify API client-credentials, else the local parquet dump via duckdb) and match them against the library's join keys | `parse_playlist_ref`, `fetch_playlist`, `match_cached`, `match_live`, `SpotifyImportError` |
| `llm_client.py` | Unified LLM client (Anthropic/OpenAI/Gemini/Ollama/custom), cost & context helpers | `LLMClient`, `get_llm_client`, `init_llm_client`, `estimate_cost_for_model` |
| `prompt_classifier.py` | Local prompt → genre/decade classification (keyword rules + hashing embeddings); reports which phrases matched and which genres were rejected | `classify_prompt`, `PromptClassification`, `extract_genre_hints`, `extract_decade_hints` |
| `analyzer.py` | Prompt → filter suggestions **with per-phrase evidence** (local first, LLM fallback); seed track → 8-12 categorized dimensions (cached) | `analyze_prompt`, `analyze_track` |
| `generator.py` | Streaming playlist generation, dynamic candidate budget, rule-based pre-ranking, fuzzy track matching | `generate_playlist_stream`, `compute_track_budget`, `pre_rank_tracks`, `_tracks_match` |
| `agent_reasoning.py` | The curator's "why this track" memory in its **own** SQLite DB (`data/agent_song_reasoning.db`): `runs` + per-track `selections` reasons + mined `tags`; `generator` writes each run and replays earlier reasons beside candidate lines, tags into pre-ranking | `record_run`, `get_previous_reasons`, `get_reason_tags`, `keys_with_tags`, `has_reasoning_data`, `reasoning_status` |
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
    ├── config.py ──────────► AppConfig (Plex + LLM + Defaults + optional spotify)
    │
    ├── plex_client.py ─────► PlexClient (library queries, playlists, art)
    │       └── TrackCache (TTL in-memory cache for filtered results)
    │
    ├── library_cache.py ───► SQLite cache (data/library_cache.db)
    │       ├── tracks / sync_state / results / track_dimensions
    │       ├── track_keys / key_state  (library join keys + their staleness stamp)
    │       ├── missing_songs  (Spotify wish list)
    │       │     └── spotify_import.py (Spotify API / parquet dump → matched + missing rows)
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
| GET | `/api/library/ratings/status` | `get_user_ratings_status` | Which Plex accounts the cache holds stars for |
| POST | `/api/library/ratings/refresh` | `refresh_user_ratings` | Rescan those accounts' stars (admin `switchUser`) |
| GET | `/api/reasoning/status` | `get_reasoning_status` | Curation memory: runs, reasons, mined tags |
| GET | `/api/library/stats` | `get_library_stats` | Total tracks, genres, decades (Plex) |
| GET | `/api/library/stats/cached` | `get_library_stats_cached` | Genre/decade chips from the cache |
| GET | `/api/library/search` | `search_library` | Search tracks by query |
| POST | `/api/analyze/prompt` | `analyze_prompt` | Prompt → suggested filters |
| POST | `/api/analyze/track` | `analyze_track` | Seed track → dimensions (cached) |
| POST | `/api/filter/preview` | `preview_filters` | Filter count + cost estimate (widens tiers on the cache branch and reports them) |
| POST | `/api/import/spotify/preview` | `preview_spotify_import` | Match a Spotify playlist against the library, write nothing |
| POST | `/api/import/spotify` | `import_spotify_playlist` | Plex playlist of the matches + wish-list rows + history row |
| GET | `/api/import/spotify/missing` | `get_spotify_wish_list` | The stored wish list (+ per-playlist summaries) |
| POST | `/api/generate/stream` | `generate_playlist_sse` | Generate playlist (SSE) |
| POST | `/api/playlist` | `save_playlist` | Save playlist to Plex (links the history row when `result_id` is sent) |
| POST | `/api/playlist/update` | `update_playlist` | Update an existing Plex playlist |
| GET | `/api/plex/clients` | `list_plex_clients` | Available Plex players |
| GET | `/api/plex/playlists` | `list_plex_playlists` | Existing Plex playlists |
| POST | `/api/play-queue` | `play_queue` | Start playback on a Plex client |
| GET | `/api/recommend/albums/preview` | `recommend_albums_preview` | Album candidates |
| POST | `/api/recommend/analyze-prompt` | `recommend_analyze_prompt` | Recommend-mode prompt analysis |
| POST | `/api/recommend/questions` | `recommend_questions` | Follow-up questions |
| POST | `/api/recommend/switch-mode` | `recommend_switch_mode` | Switch recommendation mode |
| POST | `/api/recommend/generate` | `recommend_generate` | Run the recommendation pipeline |
| GET | `/api/results` | `list_results` | Saved playlist/result history (+ Plex playlist link) |
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
      → generator.relax_candidate_pool()   (cache branch: widen until the budget fits)
        → library_cache.relax_filters_for_target()
  → POST /api/generate/stream
    → generator.generate_playlist_stream()
      → generator.relax_candidate_pool()   (same ladder, so the run matches the preview)
      → library_cache.get_tracks_by_filters()  (or Plex / random sample)
      → generator.compute_track_budget()     (dynamic candidate count)
      → generator.pre_rank_tracks()          (rules; rating + recency from cache)
      → llm_client.generate()                (selects tracks)
      → _tracks_match()                      (rapidfuzz title/artist match)
      → library_cache.save_result()          (history)
  → POST /api/playlist                       (save to Plex)
    → library_cache.link_result_playlist()   (stamps the Plex playlist on the row)
```

### 2. Seed-Track Flow
```
User picks a track
  → POST /api/analyze/track
    → analyzer.analyze_track()
      → library_cache.get_dimensions()       (hit? model + metadata fingerprint)
      → llm_client.analyze()                 (extracts 8-12 dimensions; cached in track_dimensions)
  → User picks dimensions
  → POST /api/generate/stream (with seed_track)
    → generator.generate_playlist_stream()
      → same as above, but the prompt includes the seed + chosen dimensions
```

### 3. Generation (generator.py)

There is a single generation path, `generate_playlist_stream()` (SSE,
`/api/generate/stream`). It emits `progress` events
(`relaxing → fetching → filtering → pre_ranking → preparing → ai_working → parsing →
matching → narrative → complete`), batched `tracks` events (5 per event, to work
around iOS Safari SSE buffering limits), a `narrative` event, and a final
`complete` event (or `error`). `relaxing` appears only when the cache had
to widen the filters, and names the tiers it gave up.

Candidates come from `_get_tracks_from_cache_or_plex()` (local cache first, Plex
fallback). On the cache branch `candidate_facets()` spells the filter kwargs once
and `relax_candidate_pool()` widens them with `relax_filters_for_target()` until the
pool reaches what `compute_track_budget()` wants (measured against an unbounded pool,
since the pool is what is being looked for); the live-Plex branch keeps the filters
as asked because it has no counts to ladder against. The pre-rank hints and the budget still use the **asked** facets,
so widening only adds candidates, it does not re-score them.

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

```
GET  /api/library/ratings/status    → library_cache.user_ratings_status()      (pure SQLite, no Plex walk)
POST /api/library/ratings/refresh   → per account: library_cache.sync_user_ratings(key, admin client)
                                    → plex_client.collect_user_ratings(user)   (admin switchUser, cached conn)
GET  /api/reasoning/status          → agent_reasoning.reasoning_status()       (data/agent_song_reasoning.db)
```

Per-account stars are a separate store, not a re-sync: `sync_user_ratings` replaces that
account's `user_ratings` rows plus its one `rating_sync` line, and the filter SQL reads
`COALESCE(user_ratings, tracks.user_rating)` — so the admin's column is the fallback and a
member who has not been scanned cannot be filtered by. A rating scan is the same 30-60 s
walk as `get_all_raw_tracks`, so `preview_filters` pays it once per account through
`ensure_user_ratings` and every later request costs nothing.

---

### 5. Spotify Playlist Import (`spotify_import.py`)

```
POST /api/import/spotify           (playlist_ref)
  → 503 unless Plex is connected                     (checked BEFORE any fetch)
  → spotify_import.fetch_playlist(ref, config)       (502 on SpotifyImportError)
      ├── credentials set  → api_access_token() → paging /playlists/<id>/tracks   (source "api")
      ├── parquet_dir set  → duckdb over playlists/ + playlist_tracks/ + ...      (source "dump")
      └── neither          → error naming both remedies
  → has_cached_tracks() ? match_cached(tracks) : match_live(tracks, plex)
      → (matched with match_kind, missing)
  → dedupe rating_keys (playlist order, first copy wins)
  → library_cache.record_missing_songs(id, name, missing)      ← ALWAYS, even empty
  → library_cache.save_result("spotify_import", …)             → result_id
  ├── nothing matched → success:false, no Plex playlist, wish list still written
  └── plex_client.create_playlist(name, keys, description, user, copy_to)
        → link_result_playlist(result_id, playlist_id, …)      → history row shows the Plex playlist

GET /api/import/spotify/missing    → library_cache.list_missing_songs(limit, offset)
```

`POST /api/import/spotify/preview` runs the same fetch + match and writes nothing, so the UI can show
`match_kind` per row before anything is created.

---

## Config Priority Chain (`config.py`)

```
1. Environment variables        (highest)   e.g. PLEX_URL, LLM_PROVIDER, ANTHROPIC_API_KEY
2. config.user.yaml             (UI-saved)  written by update_config_values()
3. config.yaml                  (base)
4. Hardcoded defaults           (lowest)
```

- `MODEL_DEFAULTS` dict maps provider → default analysis/generation models
- `spotify:` is optional too (`client_id`, `client_secret`, `parquet_dir`), overridable one by one by
  `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` / `SPOTIFY_PARQUET_DIR`; credentials beat `parquet_dir`
  when both are set, and `update_config_values()` keeps the section as-is (the UI cannot patch it)
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
`PromptClassification(genres, decades, confidence, method, reasoning, matched_hints,
excluded_genres)`. `matched_hints` maps each prompt phrase that was consulted to the
library filters it produced (an empty list means the phrase matched nothing);
`excluded_genres` lists library genres that were scored and left out, so the UI can
say *why* a genre was not suggested.

| Stage | What it does | Method value |
|-------|--------------|--------------|
| Keyword rules | Mood/activity/genre hint groups → genres + era words → decades. Hints are mapped onto actual library genres via exact → substring → fuzzy (`difflib`). | `"rules"` |
| Embeddings | Feature-hashed bag-of-words vectors (stdlib `hashlib`; word unigrams + bigrams). Prompt vs each library genre description, cosine similarity. Catches genre words the rules don't know (e.g. "shoegaze"); near misses inside `_EMB_NEAR_MISS_BAND` (0.45) go to `excluded_genres` instead of being dropped silently. | `"embeddings"` |
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
| `create_playlist()` / `update_playlist()` | Playlist write-back (`user=`, `copy_to=`) |
| `move_playlist()` | Re-own an existing playlist in another account (`switchUser`) |
| `get_users()` / `_server_for_user()` | Plex accounts that can own playlists (Home admin only) |
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
  art key, plus the Plex playlist it was saved as — `playlist_id`,
  `playlist_title`, `playlist_owner`, `playlist_url`), `track_dimensions` (seed-track
  analysis: dimensions JSON keyed by rating_key + model + metadata fingerprint
  **and `DIMENSION_PROMPT_VERSION`**), `track_keys` (the library's own join keys, rebuilt
  when stale) and `key_state` (single row: the keyed-track count that stamp lives in).
- **Reads served locally:** `has_cached_tracks()`, `get_cached_tracks()`,
  `get_tracks_by_filters()`, `count_tracks_by_filters()`, `relax_filters_for_target()`,
  `get_cached_genre_decade_stats()`, `get_album_candidates()`,
  `get_album_familiarity()`, `get_metadata_batch()` (rating/recency for
  pre-ranking). `get_tracks_by_filters` applies genre filtering in Python (the
  `genres` column is JSON) and random-samples when a limit is set.
- **Dimension cache:** `get_dimensions()` / `save_dimensions()` back
  `analyzer.analyze_track()`; a model swap or edited metadata changes the
  fingerprint and misses the cache. So does `DIMENSION_PROMPT_VERSION` (in
  `analyzer.py`) — bump it whenever `TRACK_ANALYSIS_SYSTEM` changes shape.
- **Sync:** `sync_library()` walks the library in `SYNC_BATCH_SIZE` (500) batches
  with progress in `_sync_state`, exposed by `get_sync_state()` /
  `get_sync_progress()`. `needs_resync()` and `is_cache_stale()` gate startup
  syncs; `check_server_changed()` handles a swapped Plex server.
- **Filter relaxation ladder:** `relax_filters_for_target(facets, target, count=None, identity_floor=0)`
  walks `RELAXATION_TIERS` (5, cumulative, least damaging first:
  `no bans` → `no rating or play floors` → `no year or duration windows` →
  `compilations and live recordings allowed` → `IDENTITY_TIER` =
  `no genre, decade or artist whitelist`) and stops at the first tier whose count
  reaches `target`, returning `{filters, matching, asked, target, tier, relaxed, short_by}`.
  A `target` ≤ 0 or a measured `-1` (empty cache → the caller is on the live-Plex
  branch) returns the facets unchanged, and the identity tier is skipped whenever
  `0 < identity_floor <= measured` — callers pass the requested playlist size, so a
  request whose filters already fill that many tracks is answered as asked instead
  of swapped for the whole library. `FILTER_NEUTRALS` spells every "off" value, and
  only facets the caller actually set are touched, so no tier is reported for a
  change that did nothing.
- **Results:** `save_result()` / `get_result()` / `list_results()` /
  `delete_result()` persist generated playlists and recommendations for the
  history UI. `link_result_playlist(result_id, playlist_id, ...)` stamps the Plex
  playlist a generation was saved as onto its history row (called from
  `POST /api/playlist` when the request carries `result_id`); rows saved before
  this feature keep NULL playlist columns.

### Live Detection (`is_live_version`)
- Checks the album (`parentTitle`) and title for date patterns or keywords (live, concert, sbd, bootleg).
- **Pre-computed once at sync time** and stored in `tracks.is_live`; cache queries filter with `WHERE is_live = 0` instead of post-filtering in Python.

---

## Spotify Playlist Import (`spotify_import.py`)

Optional feature: paste a Spotify playlist (URL / `spotify:playlist:<id>` / bare id), get a Plex playlist of
the songs this library can play, and a wish list of the songs it cannot — persisted in `missing_songs` so a
later run (or the companion plexclean checkout) can go get them.

- **Two fetch paths, picked by config:** `spotify.client_id`+`client_secret` → fresh rows from api.spotify.com
  (client-credentials grant, cached token until expiry, retried once on HTTP 401, using the `requests`
  dependency plexapi already brings in); else `spotify.parquet_dir` → the one-directory-per-table Annas/Spotify
  dump read with duckdb (imported inside the helper, so its absence is a named error, never a crash);
  neither configured → `SpotifyImportError` naming both remedies, which `main.py` turns into HTTP 502.
  Credentials win because they are the only path that sees a playlist as it is now.
- **Layout guard before the import:** `fetch_playlist_dump` insists on `playlists/`, `playlist_tracks/`,
  `tracks/`, `albums/` under `parquet_dir` before duckdb is even imported, so a wrong path says so.
  Episodes are skipped on both paths; `MAX_PLAYLIST_TRACKS = 1000` caps either.
- **Matching reuses the join-key convention**, not a second one: `match_cached` asks `track_keys`
  (built by `library_cache.ensure_track_keys()` from plex_client's `song_key`/`album_key`/`artist_key`) — exact song key → `ck_artist` + `match_key GLOB '<title>|*'`
  per `&`-split credited artist (a different album is another version of the same song) → `ck_album` + the same
  title glob (compilations credit artists differently). `match_kind` records which rung landed it;
  `_pick_candidate` takes the duration-closest candidate, which is how a deluxe twin loses. The glob is safe
  because `norm` leaves only letters/digits/spaces.
- **Cache branch decides, Plex branch falls back:** routes call `match_cached` when
  `library_cache.has_cached_tracks()` and `match_live` (Plex full-text search, `SPOTIFY_SEARCH_LIMIT` per row)
  otherwise; the live path labels every landing `search` since search cannot name a rung.
- **The wish list replaces, never stacks:** `record_missing_songs(playlist_ref, name, rows)` DELETEs that
  playlist's rows then inserts, so re-importing a fully-matched playlist clears its stale wish (the route calls
  it **always**, even with nothing missing); PK `(playlist_ref, position)` and `position <= 0` rows are dropped.
  It is a plain `CREATE TABLE IF NOT EXISTS` in `init_schema()`, deliberately outside `migrated`.
- **History row before the Plex write:** the import saves its `spotify_import` result first so the created
  playlist can be linked back by `link_result_playlist` (same order as the generation flow), and so a
  nothing-matched run still leaves an auditable record beside the wish rows it wrote. Nothing-matched returns
  `success: false` with no empty Plex playlist.
- **Dump rows carry no Spotify ids**, so `spotify_track_id`/`spotify_url` are empty there and the wish list
  names songs without links — hence the warning `fetch_playlist_dump` always appends.

---

## Models (`models.py`) — Quick Reference

### Core Entities
- **`Track`** — rating_key, title, artist, album, duration_ms, year, genres, is_live, art_url
- **`Dimension`** — id, label, description, category (descriptor family: mood/tempo/instrumentation…)
- **`FilterHint`** — hint, `matches` (one prompt phrase → the library filters it produced)
- **`FilterSet`** — genres, decades, track_count, exclude_live
- **`Playlist`** — name, tracks, source_prompt, seed_track_key, selected_dimensions

### Config
- **`PlexConfig`** — url, token, music_library
- **`LLMConfig`** — provider, api_key, model_analysis, model_generation, smart_generation, ollama/custom settings
- **`DefaultsConfig`** — track_count (default 25)
- **`SpotifyConfig`** — client_id, client_secret, parquet_dir (all optional, all stripped)
- **`AppConfig`** — plex + llm + defaults + spotify

### API Request/Response (grouped by endpoint)
- Health: `HealthResponse`
- Setup: `SetupStatusResponse`, `ValidatePlexResponse`, `ValidateAIResponse`, `SetupCompleteResponse`
- Config: `ConfigResponse`, `UpdateConfigRequest`
- Library: `LibraryStatsResponse`, `LibraryCacheStatusResponse`, `SyncProgress`, `SyncTriggerResponse`, `GenreCount`, `DecadeCount`
- Analysis: `AnalyzePromptRequest/Response` (response includes `method`, `confidence`,
  `matched_hints: [FilterHint]` and `excluded_genres` — evidence, not just a verdict),
  `AnalyzeTrackRequest/Response` (includes `cached`; dimensions carry `category`)
- Filter: `FilterPreviewRequest` (optional `prompt`),
  `FilterPreviewResponse` (adds `matching_as_asked=0` / `relaxed_tiers=[]` / `relaxed_to=""`: the pool
  `matching_tracks` is post-ladder, `matching_as_asked` is the strict count)
- Generate: `GenerateRequest` (also takes `style_cues`), `GenerateResponse` (same tier reporting:
  `matching_as_asked=0` / `relaxed_tiers=[]` / `relaxed_to=""`; `matching_as_asked` stays 0 on the Plex
  branch because nothing was counted), `SeedTrackInput`
- Spotify import: `SpotifyImportPreviewRequest/Response`, `SpotifyImportRequest/Response` (reuses the
  `SavePlaylistRequest` validators), `SpotifyMatchedTrack` (`match_kind` ∈ exact|version|album|search),
  `SpotifyMissingSong` (wire row = `missing_songs` row), `SpotifyMissingSummary`,
  `SpotifyMissingListResponse` (`missing` page + `playlists` summaries + whole-table `total`)
- Playlist: `SavePlaylistRequest/Response`, `UpdatePlaylistRequest/Response` (both carry `user`/`copy_to` keys,
  `OWNER_USER_KEY` = our own account); `SavePlaylistRequest` also takes `result_id`
  (history row to link, 8-16 hex or `None`), and its `model_validator(mode="after")` drops
  a `copy_to` entry equal to `user` (that would write a second identical playlist);
  `SavePlaylistResponse` answers `linked_result_id` when a row was stamped
- Plex: `PlexClientInfo`, `PlexPlaylistInfo`, `PlexUserInfo`, `PlexPlaylistCopy`, `PlayQueueRequest/Response`
- Recommend: `AlbumPreviewResponse`, `AnalyzePromptFiltersResponse` (also reports `matched_hints` / `excluded_genres`; `analyze_prompt_filters` drops invented or repeated filters and caps both lists at 8), `RecommendQuestionsResponse`, `RecommendSwitchModeResponse`
- Results: `ResultListResponse` (`ResultListItem`) and `ResultDetail` — both carry `playlist_id`/`playlist_title`/`playlist_owner`/`playlist_url`
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
| Relax filters that leave too few candidates | `library_cache.py` + `generator.py` | `relax_filters_for_target()`, `RELAXATION_TIERS`, `IDENTITY_TIER`; `relax_candidate_pool()` / `candidate_facets()` are how the routes call it |
| Change pre-ranking rules | `generator.py` | `pre_rank_tracks()` |
| Change generation prompt | `generator.py` | `GENERATION_SYSTEM` |
| Change prompt analysis prompts | `analyzer.py` | `PROMPT_ANALYSIS_SYSTEM` (asks for `hint_matches`/`excluded_genres`), `TRACK_ANALYSIS_SYSTEM` (8-12 dimensions + categories) |
| Change how much seed-track analysis reports | `analyzer.py` | `DIMENSION_MIN`/`DIMENSION_MAX` (8-12), `MAX_HINT_EVIDENCE`, `DIMENSION_PROMPT_VERSION` (cache invalidation) |
| Change prompt-evidence reporting | `prompt_classifier.py` | `PromptClassification.matched_hints`/`excluded_genres`, `_describe()`, `_EMB_NEAR_MISS_BAND` |
| Link a history row to its Plex playlist | `library_cache.py` | `link_result_playlist()` (called by `save_playlist` in `main.py`) |
| Change prompt classification rules/descriptions | `prompt_classifier.py` | hint dictionaries, `GENRE_DESCRIPTIONS`, `CONFIDENCE_THRESHOLD` |
| Add a new config field | `models.py` + `config.py` | Add to `LLMConfig`/`PlexConfig` + load in `load_config()` |
| Change SSE events | `generator.py` | `generate_playlist_stream()` |
| Change caching behavior | `plex_client.py` | `TrackCache` class |
| Change the library sync / schedule | `library_cache.py` | `sync_library()`, `SYNC_BATCH_SIZE`, `needs_resync()` |
| Change how a Spotify playlist's rows are fetched | `spotify_import.py` | `fetch_playlist()` (credentials → API, else `parquet_dir` → dump), `fetch_playlist_api()` (paging + token cache), `fetch_playlist_dump()` (duckdb inside, layout guard), `MAX_PLAYLIST_TRACKS` |
| Change how imported songs match the library | `spotify_import.py` | `match_cached()` (exact → version → album rungs over `track_keys`), `_pick_candidate()` (duration-closest), `match_live()` (Plex fallback), `CACHE_KEY_SCAN_LIMIT` |
| Change the "songs I don't have" wish list | `library_cache.py` | `missing_songs` table, `record_missing_songs()` (replaces per playlist), `list_missing_songs()` |
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
- **optional, not required:** `duckdb` — `spotify_import.fetch_playlist_dump()` imports it inside
  the helper (behind a layout guard), so a machine without it gets a named error rather than an
  `ImportError`, and it stays out of `requirements.in` / `requirements-dev.in` on purpose

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
