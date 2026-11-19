# INTERFACES.md — PlexSage file/surface reference

Machine-readable contract surface for every file in this repo. Other agents/apps should read **this**
instead of opening sources. Everything below was derived from the code (parsed, not prose); if you
change a signature, update the matching row here.

Companion docs (they answer *why*, this answers *what*):

- `backend/BACKEND_MAP.md` — navigation map, flows, "where to look" table.
- `backend/BACKEND_IMPROVEMENTS.md` — why there is exactly one cache module (`library_cache.py`).
- `AGENTS.md` — local-only (gitignored) repo conventions + traps.

---

## 0. Shape of the app

```
frontend/  hand-written HTML/CSS/JS, no build step
   └─> backend/main.py  FastAPI app: every route lives here (42 routes)
          ├─ backend/models.py        every Pydantic request/response contract (nothing in main.py)
          ├─ backend/config.py        config precedence + persistence
          ├─ backend/version.py       version from git tag / env
          ├─ backend/plex_client.py   live Plex server (plexapi)
          │     └─ backend/library_cache.py  SQLite mirror of that library (the local-first switch)
          │           └─ backend/spotify_import.py  Spotify playlist rows → Plex playlist + wish list
          ├─ backend/llm_client.py    Anthropic/OpenAI/Gemini/Ollama/custom + cost & context math
          │     ├─ backend/analyzer.py            prompt → filters, seed track → dimensions
          │     │     └─ backend/prompt_classifier.py  LLM-free fallback for the above
          │     ├─ backend/generator.py           playlist generation (SSE)
          │     └─ backend/recommender.py         album recommendations (SSE)
          │           └─ backend/music_research.py  MusicBrainz/Wikipedia/Cover Art grounding
          ├─ backend/agent_reasoning.py   the curator's own selection reasons (its own DB file)
          ├─ data/library_cache.db    one state file for the cache (no second cache module)
          └─ data/agent_song_reasoning.db  curation memory lives in its own file, never joined against `tracks`
```

`backend/__init__.py` is a package marker re-exporting `__version__` from `version.py`.
`backend` is importable from repo root, so app/docs tests import `backend.main:app`.

---

## 1. HTTP routes (all declared in `backend/main.py`)

Request/response types are the Pydantic classes in `backend/models.py` — the contract, not a sketch.
`→ ∅` means no body. `SSE` means `StreamingResponse` of `text/event-stream` frames `event: <name>\ndata: <json>`.

| Method + path | Body model | Response model | Notes |
|---|---|---|---|
| `GET /api/health` | ∅ | `HealthResponse` | `plex_connected`, `llm_configured` only |
| `GET /api/setup/status` | ∅ | `SetupStatusResponse` | onboarding checklist; reads cache state |
| `POST /api/setup/validate-plex` | `ValidatePlexRequest` | `ValidatePlexResponse` | live Plex handshake; saves on success |
| `POST /api/setup/validate-ai` | `ValidateAIRequest` | `ValidateAIResponse` | live provider ping; saves on success |
| `POST /api/setup/complete` | ∅ | `SetupCompleteResponse` | marks onboarding done |
| `GET /api/config` | ∅ | `ConfigResponse` | secrets never serialized; `*_set` booleans instead |
| `POST /api/config` | `UpdateConfigRequest` | `ConfigResponse` | persists to `data/config.user.yaml` |
| `GET /api/ollama/status` | ∅ (query `url`) | `OllamaStatus` | pings `<url>/api/models` |
| `GET /api/ollama/models` | ∅ (query `url`) | `OllamaModelsResponse` | |
| `GET /api/ollama/model-info` | ∅ (query `model`, `url`) | `OllamaModelInfo \| None` | probes context window |
| `GET /api/library/status` | ∅ | `LibraryCacheStatusResponse` | DB read + `is_syncing`/`sync_progress` |
| `POST /api/library/sync` | ∅ | `SyncTriggerResponse` | schedules `library_cache.sync_library` on a thread |
| `GET /api/library/ratings/status` | ∅ | `UserRatingsStatusResponse` | **SQLite** only: which Plex accounts the cache holds stars for; the admin never appears (its column is every track's `user_rating`), and a missing account is "not scanned yet", not "rated nothing" — read back by the settings view's `memory-ratings` panel (§5) |
| `POST /api/library/ratings/refresh` | `RefreshUserRatingsRequest` | `RefreshUserRatingsResponse` | rescans those accounts' stars through the admin's `switchUser` connection (503 when Plex is not connected); empty `users` = every already-cached account, `__owner__` is skipped, a failed scan lands in `errors` next to the fresh `status` — called by `rescan-ratings-btn` (§5) |
| `GET /api/reasoning/status` | ∅ | `ReasoningStatusResponse` | **SQLite**, other DB: `agent_reasoning.reasoning_status()` over `data/agent_song_reasoning.db` — runs, stored reasons, mined tag vocabulary; read back by the same panel's `memory-reasoning` div |
| `GET /api/library/stats` | ∅ | `LibraryStatsResponse` | **live Plex** round-trip (genres/decade counts) |
| `GET /api/library/stats/cached` | ∅ | `LibraryStatsResponse` | **SQLite** round-trip-free variant |
| `GET /api/library/search` | ∅ (query `q`) | `list[Track]` | live Plex search |
| `POST /api/analyze/prompt` | `AnalyzePromptRequest` | `AnalyzePromptResponse` | local classifier first, LLM fallback; returns evidence (`matched_hints`, `excluded_genres`) |
| `POST /api/analyze/track` | `AnalyzeTrackRequest` | `AnalyzeTrackResponse` | 8-12 dimensions, DB-cached (`cached: true` on hit) |
| `POST /api/filter/preview` | `FilterPreviewRequest` | `FilterPreviewResponse` | branches on `library_cache.has_cached_tracks()`; the SQLite branch widens tiers (`relaxed_tiers`/`relaxed_to`/`matching_as_asked`) so the count is the pool a run will use; a `rating_user` other than `__owner__` is scanned in first (`library_cache.ensure_user_ratings`) and the account actually read is echoed back in `rating_user`, with `rating_error` naming why it could not be |
| `POST /api/import/spotify/preview` | `SpotifyImportPreviewRequest` | `SpotifyImportPreviewResponse` | rows from api.spotify.com when credentials are set, else the local parquet dump; match rungs `exact`/`version`/`album` (cache) or `search` (live Plex); writes nothing |
| `POST /api/import/spotify` | `SpotifyImportRequest` | `SpotifyImportResponse` | writes the Plex playlist of what matched, replaces that playlist's `missing_songs` wish rows (always, even empty), saves a `spotify_import` history row + links it |
| `GET /api/import/spotify/missing` | ∅ (query `limit`, `offset`) | `SpotifyMissingListResponse` | **SQLite** wish list page + per-playlist summaries (`total` counts the whole table) |
| `POST /api/generate/stream` | `GenerateRequest` | **SSE** | events: `progress`, `tracks`, `narrative`, `complete`, `error`; tier widening is reported on the cache path, and `rating_user` is threaded to the cache query there (a failed scan falls back to the admin's stars with a logged warning, since a missing row reads as a real zero) |
| `POST /api/playlist` | `SavePlaylistRequest` | `SavePlaylistResponse` | writes Plex playlist; links history row when `result_id` sent |
| `GET /api/plex/clients` | ∅ | `list[PlexClientInfo]` | live Plex |
| `POST /api/play-queue` | `PlayQueueRequest` | `PlayQueueResponse` | live Plex |
| `GET /api/plex/playlists` | ∅ (query `user`) | `list[PlexPlaylistInfo]` | live Plex |
| `GET /api/plex/users` | ∅ | `list[PlexUserInfo]` | plex.tv roster: owner, Home members, shared friends |
| `POST /api/playlist/update` | `UpdatePlaylistRequest` | `UpdatePlaylistResponse` | `mode: replace\|append`; no re-link of history rows |
| `GET /api/recommend/albums/preview` | ∅ (query `genres`, `decades`, `max_albums`) | `AlbumPreviewResponse` | DB branch like `/filter/preview` |
| `POST /api/recommend/analyze-prompt` | `AnalyzePromptFiltersRequest` | `AnalyzePromptFiltersResponse` | evidence (`matched_hints`, `excluded_genres`) |
| `POST /api/recommend/questions` | `RecommendQuestionsRequest` | `RecommendQuestionsResponse` | creates a session (`session_id`) |
| `POST /api/recommend/switch-mode` | `RecommendSwitchModeRequest` | `RecommendSwitchModeResponse` | keeps answers |
| `POST /api/recommend/generate` | `RecommendGenerateRequest` | **SSE** | events: `progress`, `albums`, `pitch`, `complete`, `error` |
| `GET /api/results` | ∅ (query `type`, `limit`, `offset`) | `ResultListResponse` | **SQLite** `results` table, no snapshots |
| `GET /api/results/{result_id}` | ∅ | `ResultDetail` | **SQLite**, includes `snapshot` JSON |
| `DELETE /api/results/{result_id}` | ∅ | 204 | removes the row |
| `GET /api/art/{rating_key}` | ∅ | image bytes | proxies Plex thumb so the browser never sees the token |
| `GET /api/external-art` | ∅ (query `url`) | image bytes | HTTPS-only, allowlist `_EXTERNAL_ART_DOMAINS = {coverartarchive.org, archive.org}` |
| `GET /` | ∅ | `HTMLResponse` | serves `frontend/index.html`, cache-busting `/static/*?v=<version>` |

Static assets: `app.mount("/static", StaticFiles(directory=frontend_path))` inside `lifespan`, so
`/static/style.css` and `/static/app.js` are plain file reads (`frontend_path` = `backend/../frontend`,
falling back to `/app/frontend` in Docker).

Non-route startup behavior (`lifespan`): instantiates `plex_client.init_plex_client` + `llm_client.init_llm_client`
from `config.get_config()`, calls `library_cache.ensure_db_initialized()`, and auto-starts `sync_library()`
**only** when `init_schema()` reported a `tracks`/`track_dimensions` migration (`needs_resync()`); a
`results`-only migration deliberately does not.

---

## 2. `backend/models.py` — every contract (grouped)

Constant: `OWNER_USER_KEY = "__owner__"` (the sentinel meaning "the account this token belongs to").
`album_key(artist, album, lower=True)` here is a composite artist+album lookup key — a different
function from `plex_client.album_key(album)`, the plexclean join key.

### Entities (wire shapes)

| Class | Fields | Notes |
|---|---|---|
| `Track` | `rating_key, title, artist, album, duration_ms, year, genres, is_live, art_url` | has `__str__`-style duration formatter |
| `Dimension` | `id, label, description, category=""` | `category` ∈ `mood, energy, tempo, era, instrumentation, vocals, production, genre lineage, scene` (from `analyzer` prompt) |
| `FilterSet` | `genres, decades, track_count=25, exclude_live=True` | validator: `track_count` ∈ 15/25/50/100 |
| `Playlist` | `name, tracks, source_prompt, seed_track_key, selected_dimensions` | response-side shape |
| `AlbumCandidate` | `parent_rating_key, album, album_artist, year, genres, decade, track_count, track_rating_keys` | built from cached tracks |
| `ClarifyingQuestion` | `question_text, options, dimension` | |
| `SommelierPitch` | `hook, context, listening_guide, connection, short_pitch, full_text` | |
| `AlbumRecommendation` | `rank, album, artist, year, rating_key, track_rating_keys, art_url, pitch, research_available` | |
| `ResearchData` | `musicbrainz_id, release_date, label, track_listing, credits, genre_tags, wikipedia_summary, review_links, review_texts, cover_art_url, earliest_release_mbid` | |
| `ExtractedFacts` | `origin_story, personnel, musical_style, vocal_approach, cultural_context, track_highlights, common_misconceptions, source_coverage, track_listing` | LLM-structured facts |
| `PitchIssue` / `PitchValidation` | `claim, problem, correction` / `valid, issues` | pitch fact-check |
| `TasteProfile` | `genre_distribution, decade_distribution, top_artists, total_albums, owned_albums` | discovery mode |
| `RecommendSessionState` | `mode, prompt, filters, questions, answers, answer_texts, album_candidates, taste_profile, familiarity_pref, previously_recommended, total_tokens, total_cost` | transient, held in `recommender` |
| `FilterHint` | `hint, matches` | evidence: one prompt phrase → filters it produced |

### Config + diagnostics

`PlexConfig{url, token, music_library}` · `LLMConfig{provider, api_key, model_analysis, model_generation,
smart_generation, ollama_url, ollama_context_window, custom_url, custom_context_window, request_timeout, max_retries}`
(context-window validators: ≥ 512, ≤ 2 000 000) · `DefaultsConfig{track_count}` · `SpotifyConfig{client_id, client_secret, parquet_dir}` (all stripped, all optional) · `AppConfig{plex, llm, defaults, spotify}`.

`ConfigResponse` deliberately has no `token`/`api_key` fields — only `plex_token_set`, `llm_api_key_set` booleans.
`UpdateConfigRequest` is the partial set the UI can patch: `plex_url, plex_token, music_library, llm_provider,
llm_api_key, model_analysis, model_generation, ollama_url, ollama_context_window, custom_url, custom_context_window`.
`HealthResponse{status, plex_connected, llm_configured}` · `ErrorResponse{error, detail}`.

### Library cache + stats

`LibraryStatsResponse{total_tracks, genres: [GenreCount{name,count}], decades: [DecadeCount{name,count}]}`
· `LibraryCacheStatusResponse{track_count, synced_at, is_syncing, sync_progress: SyncProgress{phase,current,total}, error, plex_connected, needs_resync}`
· `SyncTriggerResponse{started, blocking}`. `FilterPreviewRequest{genres, decades, artists, exclude_genres, exclude_artists,
exclude_decades, min_year, max_year, min_plays, max_plays, unheard_only, exclude_compilations, track_count, max_tracks_to_ai,
min_rating, max_rating, min_duration_ms, max_duration_ms, live_only, rated_only, unrated_only, played_after, played_before,
exclude_live, prompt, rating_user}` — every facet defaults off/empty, and `rating_user` names the Plex account whose stars the
rating facets read ("" = the connecting admin's). → `FilterPreviewResponse{matching_tracks, tracks_to_send, estimated_input_tokens,
estimated_output_tokens, estimated_cost, matching_as_asked=0, relaxed_tiers=[], relaxed_to="", rating_user="", rating_error=""}`:
the tier trio reports the widening (see `relax_filters_for_target`), the rating pair says whose stars were actually used
(`rating_error` is set when that account's ratings were never loaded, so the floor silently fell back to the admin's column).

### Per-account ratings + agent reasoning (both routed)

`PlexUserRatingsStatus{user_key, rated_count=0, total_seen=0, last_sync_at=None}` → `UserRatingsStatusResponse{total_tracks=0,
users=[], synced_users=[]}` (also nested in the refresh response) · `RefreshUserRatingsRequest{users=[]}` (validator
`_validate_user_keys`, so `__owner__` / exact Plex title / friend username) → `RefreshUserRatingsResponse{refreshed=[],
errors: dict[user_key → why its scan failed], status: UserRatingsStatusResponse}` · `ReasoningTagStat{tag, count=0}` →
`ReasoningStatusResponse{runs=0, selections=0, tracks_with_reasons=0, tagged_tracks=0, tags=[], last_run_at=None, error=None}`.
Three routes answer them (`GET /api/library/ratings/status`, `POST /api/library/ratings/refresh`, `GET /api/reasoning/status`,
§1), and the settings view's `memory-stats-section` reads both status routes back and offers the rescan (§5), so a member
whose stars were never loaded is visible rather than silently averaged into the admin's.
`errors` exists because a scan that failed and returned "0 rows" would read as *this member rated nothing* — and since the
effective rating is `COALESCE(user_ratings, tracks.user_rating)`, that would drop every track they had rated from a floor.

### Spotify playlist import (optional)

`SpotifyImportPreviewRequest{playlist_ref}` (non-empty; accepts `open.spotify.com/…/playlist/<id>` URL,
`spotify:playlist:<id>` or a bare alnum id) → `SpotifyImportPreviewResponse{playlist_name, playlist_owner,
spotify_playlist_url, source, total_tracks, matched: [SpotifyMatchedTrack], missing: [SpotifyMissingSong], warnings}`.
`source` is `api` (fresh rows, needs the credentials pair) or `dump` (the local Annas/Spotify parquet snapshot named by
`spotify.parquet_dir`); neither configured → HTTP 502 naming both remedies. `SpotifyMatchedTrack{rating_key, title,
artist, album, duration_ms, position, match_kind}` where `match_kind` ∈ `exact` (song key) / `version` (title+artist,
another album) / `album` (title+album, another artist credit) on the cache branch, `search` on the live-Plex branch.
`SpotifyMissingSong{position, title, album, artist, duration_ms, spotify_track_id, spotify_url, playlist_ref,
playlist_name, recorded_at}` is both the wire shape and the wish-list row (dump rows carry no track ids, so those
fields are empty on that path).
`SpotifyImportRequest{playlist_ref, name, description, user=OWNER_USER_KEY, copy_to=[]}` reuses the `SavePlaylistRequest`
validators (`normalize_user`, `_validate_user_keys`, `_truncate_description`, `drop_redundant_owner_copy`) →
`SpotifyImportResponse{success, error, playlist_name, playlist_owner, spotify_playlist_url, source, total_tracks,
matched, missing, matched_count, missing_count, missing_recorded, result_id, playlist_id, playlist_url, tracks_added,
tracks_skipped, owner, copies, warnings}` — `missing_recorded` counts wish rows written, and `success` is false with no
`playlist_id` when nothing matched (the wish list is still written and the history row still saved).
`SpotifyMissingSummary{playlist_ref, playlist_name, count}` → `SpotifyMissingListResponse{missing, playlists, total}`.

### Prompt analysis

`AnalyzePromptRequest{prompt: 1..2000}` → `AnalyzePromptResponse{suggested_genres, suggested_decades,
available_genres, available_decades, reasoning, matched_hints: [FilterHint], excluded_genres, token_count, estimated_cost,
method, confidence}`. `method` distinguishes the local classifier from the LLM: the classifier answers
without a token, the LLM path fills `token_count`/`estimated_cost`.
`AnalyzeTrackRequest{rating_key}` → `AnalyzeTrackResponse{track, dimensions, token_count, estimated_cost, cached}`.
Recommend side: `AnalyzePromptFiltersRequest{prompt, genres, decades}` → `AnalyzePromptFiltersResponse{genres, decades, matched_hints, excluded_genres, reasoning}`.

### Generation + Plex write/playback

`SeedTrackInput{rating_key, selected_dimensions}`; `GenerateRequest{prompt, seed_track, additional_notes,
refinement_answers, genres, decades, track_count=25, exclude_live=True, min_rating=0, max_tracks_to_ai=500, …,
style_cues, rating_user}`
with `check_flow`: **at least one of `prompt` / `seed_track` required**.
`GenerateResponse{tracks, token_count, estimated_cost, playlist_title, narrative, track_reasons: dict[rating_key → reason],
matching_as_asked=0, relaxed_tiers=[], relaxed_to=""}` — same tier reporting as the preview; `relaxed_tiers` is what the run
actually gave up, so the liner note and the UI can say which filters were honoured (`matching_as_asked` is 0 on the live-Plex branch).

`SavePlaylistRequest{name, rating_keys, description, user=OWNER_USER_KEY, copy_to=[], result_id=None}` —
validators, in order: `user` blank→`__owner__`; `copy_to` drops blank/dup keys (`_validate_user_keys`);
`drop_redundant_owner_copy` (model-level) removes a `copy_to` entry equal to `user`; `result_id` must be 8-16 hex chars
(`""`→`None` via `_validate_result_id`); `name` non-empty and stripped; `description` truncated to 2000 chars;
`rating_keys` non-empty and all numeric (`_validate_rating_keys`).
→ `SavePlaylistResponse{success, playlist_id, playlist_url, error, tracks_added, tracks_skipped, owner,
copies: [PlexPlaylistCopy{user, success, playlist_id, playlist_url, tracks_added, tracks_skipped, error}], linked_result_id}`.
`linked_result_id` is the history row this save pointed at the playlist for — `None` if no `result_id` was sent or the row was unknown.

`UpdatePlaylistRequest{playlist_id, rating_keys, mode: replace|append, description, user}` — `playlist_id` must be
`"__scratch__"` or numeric — → `UpdatePlaylistResponse{success, tracks_added, tracks_skipped, duplicates_skipped, playlist_url, warning, error, owner}`.
`PlayQueueRequest{rating_keys, client_id, mode: replace|play_next}` → `PlayQueueResponse{success, client_name, client_product, tracks_queued, tracks_skipped, error}`.
`PlexPlaylistInfo{rating_key, title, track_count}` · `PlexClientInfo{client_id, name, product, platform, is_playing, is_mobile}`
· `PlexUserInfo{key, title, user_id, is_owner, is_home, is_managed, has_pin}` (`key` is what you send back: `__owner__`,
exact Plex title, or friend username).

### Recommendation flow

`RecommendQuestionsRequest{prompt: 1..2000}` → `RecommendQuestionsResponse{questions, session_id, token_count, estimated_cost}` ·
`RecommendSwitchModeRequest{session_id, mode: library|discovery}` → `{session_id}` ·
`RecommendGenerateRequest{session_id, answers: [str|null], answer_texts, mode, genres, decades, familiarity_pref: any|comfort|rediscover|hidden_gems, max_albums=2500}`
(`max_albums` clamped to ≤ 50000, negative rejected) → `RecommendGenerateResponse{recommendations, token_count, estimated_cost, research_warning}` ·
`AlbumPreviewResponse{matching_albums, albums_to_send, estimated_input_tokens, estimated_cost}`.

### History rows + setup

`ResultListItem{id, type, title, prompt, track_count, artist, art_rating_key, subtitle, created_at, playlist_id,
playlist_title, playlist_owner, playlist_url}` → `ResultListResponse{results, total}`;
`ResultDetail` = the same plus `snapshot`. The four `playlist_*` fields are the Plex linkage (see §4).
`SetupStatusResponse{data_dir_writable, process_uid, process_gid, data_dir, plex_connected, plex_error, plex_from_env,
music_libraries, llm_configured, llm_provider, llm_from_env, library_synced, track_count, is_syncing, sync_progress, setup_complete}` ·
`ValidatePlexRequest{plex_url, plex_token, music_library}` → `ValidatePlexResponse{success, error, server_name, music_libraries}` ·
`ValidateAIRequest{provider, api_key, ollama_url, custom_url}` → `ValidateAIResponse{success, error, provider_name}` ·
`SetupCompleteResponse{success}`. Ollama: `OllamaModel{name,size,modified_at}` · `OllamaModelInfo{name,context_window,context_detected,parameter_size}`
· `OllamaModelsResponse{models,error}` · `OllamaStatus{connected,model_count,error}`.

---

## 3. Python surface — public functions per module

Signature shorthand: defaults shown; `async` marked. Private (`_name`) helpers omitted everywhere else, but the
ones named here are load-bearing.

### `agent_reasoning.py` (the curator's "why this track" memory, its own DB)

```
CONSTANTS  DATA_DIR = Path(__file__).parent.parent / "data"   DB_PATH = DATA_DIR / "agent_song_reasoning.db"
           MAX_REASON_CHARS = 280  MAX_PAST_REASON_CHARS = 160  MAX_TRACKS_WITH_PAST_REASON = 400
           MAX_TAGS_PER_REASON = 12  MAX_TAG_CHARS = 48  MAX_STATUS_TAGS = 40  QUERY_PARAM_CHUNK = 900
def ensure_db_initialized()  record_run(selections, prompt="", playlist_title="", result_id="", source="prompt") -> rows
def get_previous_reasons(rating_keys) -> {rating_key: reason}  get_reason_tags(rating_keys) -> {rating_key: {tag}}
def keys_with_tags(tags, match="any") -> set  has_reasoning_data() -> bool  reasoning_status() -> dict
private: _clean_tag  _mine_tags (a reason → mined vocabulary)  _connect  _init_tables (runs + selections + tags)
```

`generator` writes one row per pick through `record_run` (picks whose curator gave no reason are skipped) and reads the
store back two ways: `_past_reasons_for` replays the earlier reason beside a candidate line (capped by
`MAX_TRACKS_WITH_PAST_REASON` lines and `MAX_PAST_REASON_CHARS` each), and `pre_rank_tracks` folds `get_reason_tags`
into scoring, so a track the agent keeps explaining the same way ranks up. `keys_with_tags` answers "what else did the
agent say this about?" and batches rating keys `QUERY_PARAM_CHUNK` at a time to stay under SQLite's parameter cap.
`result_id` rides in as a string link, not a foreign key — the reasoning DB and the library cache stay separate files.
`has_reasoning_data()` gates both replays, so a fresh install pays nothing.

### `analyzer.py`

```
CONSTANTS  DIMENSION_PROMPT_VERSION = 2      # folded into the dimension-cache fingerprint
           DIMENSION_MIN = 8  DIMENSION_MAX = 12  MAX_HINT_EVIDENCE = 8
           PROMPT_ANALYSIS_SYSTEM  TRACK_ANALYSIS_SYSTEM   # the two prompt texts
def analyze_prompt(prompt) -> AnalyzePromptResponse    # ValueError on unparsable LLM JSON, RuntimeError if Plex/LLM unset
def analyze_track(track, use_cache=True) -> AnalyzeTrackResponse   # use_cache=False forces an LLM call
private: _parse_dimensions (clamps to 8-12, dedupes labels, keeps category)
         _track_fingerprint (metadata + DIMENSION_PROMPT_VERSION)  _hints_from_mapping  _hints_from_llm
         _library_facets (cache-or-Plex roster: `get_cached_artists`/`get_cached_duration_stats`/`get_cached_play_stats`,
                          falls back to `plex_client.get_all_artists`/`get_duration_stats` when the cache is empty)
         _library_names  _duration_bound  _rating_floor  _year_bound  _play_bound  _bool_flag (classifier → facet)
         _duration_phrase  _play_phrase  _year_span_phrase (the prompt's library-shape lines)
         _analysis_model  _load_cached_dimensions  _store_cached_dimensions
```

`analyze_prompt` calls `prompt_classifier.classify_prompt` first and only spends an LLM call when the local
pass is under `CONFIDENCE_THRESHOLD`. Bumping `DIMENSION_PROMPT_VERSION` invalidates every `track_dimensions`
row — never delete the table to "clear stale entries".

### `config.py`

```
CONSTANT USER_CONFIG_PATH = Path("data/config.user.yaml")
         MODEL_DEFAULTS = {anthropic, openai, gemini, ollama, custom -> analysis/generation model names}
def deep_merge(base, override)  remove_empty_values(d)
def load_yaml_config(config_path=None)  load_user_yaml_config()
class ConfigSaveError                        # raised when user yaml cannot be written
def save_user_config(updates)  get_env_or_yaml(env_key, yaml_value, default=None)
def load_config(config_path=None)  get_config()  refresh_config(config_path=None)
def update_config_values(updates)
```

`config.py` calls `load_dotenv()` **at import**, so importing any backend module reads a local `.env`.
Precedence: process env (incl. `.env`) > `data/config.user.yaml` (UI-saved) > `config.yaml` > code defaults.
The base YAML path is **not** overridable by `PLEXSAGE_CONFIG` — `backend/config.py` never reads that
variable; it is a notebook-only convention (`manual_backend.ipynb` uses it to locate the profile file).
An app started from elsewhere gets `config.yaml` relative to its cwd, so the way to point the app at
another file is a copy of it under `data/config.user.yaml`.
The `spotify:` section (`client_id`, `client_secret`, `parquet_dir`) fills `AppConfig.spotify`, overridable by
`SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` and `SPOTIFY_PARQUET_DIR`; `update_config_values` keeps the section
as-is (the UI cannot patch it), and an install that configures neither keeps every other route unchanged
— only the import routes answer, and they answer with an error naming both remedies.

### `generator.py`

```
CONSTANTS  PRE_RANK_LIMIT = 500  BUDGET_FLOOR = 60  MIN_TRACKS_PER_PICK = 6  BUDGET_FACETS (the 15 facet names compute_track_budget reads)
           _SPECIFICITY_EXPLICIT_GENRE = 0.6  _SPECIFICITY_BROAD_GENRE = 0.4  _SPECIFICITY_DECADE = 0.2
           GENERATION_SYSTEM  NARRATIVE_SYSTEM
def compute_track_budget(prompt=None, requested_max=0, track_count=25, pool_size=None, genres=None, decades=None)
def pre_rank_tracks(tracks, prompt=None, genres=None, decades=None, max_tracks=PRE_RANK_LIMIT)
def generate_narrative(track_selections, llm_client, user_request="")
def candidate_facets(genres=None, decades=None, artists=None, …, exclude_live=True, rating_user="")  → the 17 cache filter kwargs, empty lists spelled None
def relax_candidate_pool(facets, prompt=None, track_count=25, requested_max=0, count=None)  → relax_filters_for_target(facets, compute_track_budget(…, pool_size=None), identity_floor=track_count)   # cache branch only
async def generate_playlist_stream(prompt=None, seed_track=None, selected_dimensions=None, additional_notes=None,
                                   refinement_answers=None, genres=None, decades=None, track_count=25,
                                   exclude_live=True, min_rating=0, max_tracks_to_ai=500, …, rating_user="")
private: _tracks_match (fuzzy artist/album/title → rating_key)  _cached_track_to_model  _past_reasons_for
```

`_get_tracks_from_cache_or_plex` counts the cache facets as "has filters": the cache query takes all of them, while an
empty cache answers a filterless request with a random Plex sample and a filtered one with plexapi's search — the
facets as asked, since Plex has no counts for `relax_candidate_pool` to widen against. `rating_user` rides the same
dict but is in **no** tier: `FILTER_NEUTRALS` knows how to clear it, yet whose stars are read cannot widen a pool, and
dropping it would answer a member's rating floor with the admin's column while still claiming the request was honoured.
On this path a failed `library_cache.ensure_user_ratings` degrades to the admin's stars with a logged warning — the
place that must not guess silently is the preview, which says so in `rating_error` instead. The curator prompt asks for
`min(track_count, len(filtered_tracks))` picks: a small local model told to "Select 200" out of 5 offered lines
answers `[]` (2 output tokens) instead of the 5 it can honour, which parsed as a valid-but-empty run. The
narrative is written from the **matched** tracks, so it cannot name songs the playlist does not contain; with
nothing matched it skips the narrative model and returns the `"{Mon YYYY} Playlist"` fallback with no blurb.

### `library_cache.py`

```
CONSTANTS  DATA_DIR = Path(__file__).parent.parent / "data"   DB_PATH = DATA_DIR / "library_cache.db"
           DATE_PATTERN  LIVE_KEYWORDS  SYNC_BATCH_SIZE = 500  ARTIST_MATCH_MIN_LEN = 3
           MAX_ARTIST_OPTIONS = 60  DURATION_BUCKET_MS = 15_000
def get_db_connection()  init_schema(conn)  ensure_db_initialized()
def get_sync_state()  get_cached_tracks()  get_tracks_by_filters(candidate_facets' 17 kwargs + limit=0,
                    live_only/rated_only/unrated_only=False, max_rating=0, played_after=played_before="",
                    rating_user="")  get_metadata_batch(rating_keys, rating_user="")
def get_dimensions(rating_key, model="", fingerprint="")  save_dimensions(rating_key, dimensions, model="", fingerprint="")
def clear_dimensions()  clear_cache()  is_cache_stale(max_age_hours=24)  check_server_changed(current_server_id)
def sync_library(plex_client, on_progress=None)  get_sync_progress()
def count_tracks_by_filters(the same 23 facets, no limit) -> int  has_cached_tracks()
def relax_filters_for_target(facets: dict, target: int, count=None, identity_floor=0)
        -> {filters, matching, asked, target, tier, relaxed, short_by}
CONSTANTS  IDENTITY_TIER = "no genre, decade or artist whitelist"  # RELAXATION_TIERS' last label
CONSTANTS  FILTER_NEUTRALS (facet -> its off value)  RELAXATION_TIERS (5, least specific first)
def needs_resync()  get_album_candidates(genres=None, decades=None, exclude_live=True)
def get_cached_genre_decade_stats()  get_cached_artists(limit=MAX_ARTIST_OPTIONS, exclude_live=True)
def get_cached_duration_stats(exclude_live=True, bucket_ms=DURATION_BUCKET_MS)  get_cached_play_stats(exclude_live=True)
def get_album_familiarity(parent_rating_keys=None)
def has_user_ratings(user_key) -> bool  user_ratings_status() -> {total_tracks, users, synced_users}
def sync_user_ratings(user_key, plex_client, on_progress=None) -> int  ensure_user_ratings(user_key, plex_client) -> int
def save_result(result_type, title, prompt, snapshot, track_count, artist=None, art_rating_key=None, subtitle=None)
def link_result_playlist(result_id, playlist_id, playlist_title="", playlist_owner="", playlist_url="") -> bool
def get_result(result_id)  list_results(result_type=None, limit=20, offset=0)  delete_result(result_id)
def rebuild_track_keys(conn)  ensure_track_keys(conn) -> keyed_tracks
def record_missing_songs(playlist_ref, playlist_name, rows) -> rows_written   # replaces that playlist's wish rows
def list_missing_songs(limit=200, offset=0) -> {missing, playlists, total}    # page + per-playlist summaries
private: _filter_conditions(…, rating_user)  _write_keys(conn, batch)  _matches_artist  _is_compilation
         _decade_start_year  _is_live_version  _normalize_name  _percentile_ms  _play_summary
```

`link_result_playlist` returns `False` on an unknown `result_id` or blank `playlist_id`; it only ever touches the
primary playlist row. `has_cached_tracks()` is the local-first switch every route branches on.

`relax_filters_for_target` walks `RELAXATION_TIERS` **cumulatively** (bans → rating/play floors → year+duration
windows → compilation/live bans → `IDENTITY_TIER`) and stops at the first tier whose count reaches `target`, so the
pool stays as specific as the cache can honour. `asked` is the count of the filters as given, `tier`/`relaxed` name
where it ended and every tier that changed something, `short_by` says how far even the loosest reading fell short.
Two guards keep it honest: a `target` ≤ 0 or a measured count of `-1` (empty cache → the caller is on the live-Plex
branch) return the facets unchanged, and `IDENTITY_TIER` is skipped whenever `0 < identity_floor <= measured` — callers
pass the requested playlist size, so a request whose filters already fill that many tracks is answered as asked rather
than swapped for the whole library. Tiers only touch facets the caller actually set (`FILTER_NEUTRALS` spells every
"off" value), so no neutral keys are injected and no tier is reported for a change that did nothing. `rating_user` is
the one facet in `FILTER_NEUTRALS` that appears in **no** tier — reading another account's stars re-reads the same
rows, so it cannot widen a pool and must not be dropped from a request that asked for them.

`get_tracks_by_filters`/`count_tracks_by_filters` share one filter grammar: every numeric, year, duration, rating and
play-window facet is decided once in `_filter_conditions` (one SQL WHERE body plus bind params, `None` when nothing is
filtered) so a count and a fetch cannot drift apart, and only genre/artist matching stays in Python (genres are a JSON
column, artist names need the containment rules in `_matches_artist`). The library's join keys live in `track_keys` —
`song_key`/`album_key`/`artist_key` of every cached row, spelled by `backend.plex_client` so plexclean's third-party
tools compute the same string — and `key_state.keyed_tracks` says how many rows that table covers. `ensure_track_keys`
compares that stamp with `COUNT(tracks)` and calls `rebuild_track_keys` when they disagree, so a sync that added or
dropped tracks repairs the keys instead of serving a stale set; only `spotify_import` asks for them, and they are a
table of their own rather than columns on `tracks` so a pre-existing library gains the feature without a schema
migration that would demand a full re-sync.

Per-account ratings: `has_user_ratings(user_key)` only says that account was scanned, and `user_ratings_status()` is
what separates "not loaded yet" from "loaded and rated nothing" — the effective rating is `COALESCE(user_ratings,
tracks.user_rating)`, so a never-scanned account silently reads the connecting admin's stars for every track.
`sync_user_ratings` writes the account's stars and its `rating_sync` line; `ensure_user_ratings` scans on the first
ask and costs nothing afterwards. The routes do reach them now: `preview_filters` resolves `rating_user` through
`ensure_user_ratings` (skipping `__owner__`, which needs no rows) and reports the account it actually used in
`FilterPreviewResponse.rating_user` plus `rating_error` when the scan failed or the cache was empty; `generate_playlist_stream`
threads the same field into `candidate_facets`/`get_tracks_by_filters`/`count_tracks_by_filters`, and a failed scan there
falls back to `""` with a logged warning rather than pretending the member rated nothing. Only the SQLite branch can honour
it — `plex_client.get_tracks_by_filters` has no `rating_user`, because another account's stars exist only in this cache.

### `llm_client.py`

```
CONSTANTS  MODEL_COSTS  MODEL_CONTEXT_LIMITS  TOKENS_PER_TRACK = 40  TOKENS_PER_ALBUM = 25
           MAX_OUTPUT_TOKENS = 8192   # max_tokens sent to anthropic + OpenAI-compatible endpoints
class LLMResponse(content, input_tokens, output_tokens, model)
def estimate_cost_for_model(model, input_tokens, output_tokens, config=None)
class LLMClient: analyze(prompt, system)  generate(prompt, system)  parse_json_response(response)
def get_llm_client()  init_llm_client(config)
def get_max_tracks_for_model(model, buffer_percent=0.1, config=None)  get_max_albums_for_model(...)
def get_model_context_limit(model, config=None)  get_model_cost(model, config=None)
def list_ollama_models(ollama_url, timeout=5.0)  get_ollama_model_info(ollama_url, model_name, timeout=5.0)
def get_ollama_status(ollama_url, timeout=5.0)
```

### `music_research.py`

```
CONSTANTS  USER_AGENT  MB_BASE_URL = "https://musicbrainz.org/ws/2"  MB_RATE_LIMIT = 1.0
           WIKIPEDIA_API  COVER_ART_BASE  RG_* match-scoring weights
           _WIKIPEDIA_DROP_KEYWORDS (sections stripped from an article extract)  _WIKIPEDIA_MAX_CHARS = 8000
class MusicResearchClient (all async):
    search_album(artist, album, year=None)  lookup_release_group(mbid)  lookup_release(release_mbid)
    fetch_wikipedia_summary(wikipedia_url)  resolve_wikidata_to_wikipedia(wikidata_url)
    fetch_cover_art(release_mbid, release_group_mbid=None)  fetch_review_text(url)
    research_album(artist, album, full=True, year=None)  close()
```

### `plex_client.py`

```
class TrackCache: get(genres, decades, exclude_live, min_rating)  set(...)  clear()   # TTL 300 s
def get_track_cache()
class PlexQueryError
CONSTANTS  FUZZ_THRESHOLD = 60  DATE_PATTERN  LIVE_KEYWORDS  ARTIST_MATCH_MIN_LEN = 3
           MAX_ARTIST_OPTIONS = 60  DURATION_SAMPLE_SIZE = 400  POST_FILTER_SAMPLE_FACTOR = 8
           ARTIST_CONNECTOR = /&|\band\b/i  KEY_PARTS = /\s{2,}/   # plexclean's norm/norm_artist spelling
def norm(value)  norm_artist(value)  song_key(title, album, artist)  album_key(album)  artist_key(artist)
def simplify_string(s)  normalize_artist(name)  is_live_version(track)
private: _sample_percentile  _track_genres  _is_compilation_track  _banned_artist_hit  _decade_banned
         _played_outside_window  _rating_mismatched  _not_live_only  _matches_artist
class PlexClient (class attrs RECONNECT_COOLDOWN, USER_SERVER_TTL = 900, _MOBILE_KEYWORDS):
    is_connected()  get_machine_identifier()  get_error()  get_music_libraries()  get_library_total_tracks()
    get_all_raw_tracks()  get_all_albums_metadata()  get_library_stats()  get_all_tracks()
    get_tracks_by_filters(all cache facets + limit=0, but **no** `rating_user` — another account's stars exist only
                          in the cache)  count_tracks_by_filters(same facets, no limit)
    get_genres()  get_decades()  get_random_tracks(count, exclude_live=True)
    get_all_artists(limit=MAX_ARTIST_OPTIONS, exclude_live=True)  get_duration_stats(exclude_live=True, sample_size=DURATION_SAMPLE_SIZE)
    search_tracks(query, limit=20)  get_track_by_key(rating_key)  get_thumb_path(rating_key)
    get_users()  collect_user_ratings(user=None) → [(rating_key, stars)] for that account
    create_playlist(name, rating_keys, description="", user=None, copy_to=None)
    move_playlist(playlist_id, to_user, from_user=None, delete_source=False, name="", description=None, copy_to=None)
    get_clients()  get_playlists(user=None)  update_playlist(playlist_id, rating_keys, mode="replace", description="", user=None)
    play_queue(rating_keys, client_id, mode="replace")
def get_plex_client()  init_plex_client(url, token, music_library="Music")
```

Ownership walk: `get_users()` keys are `__owner__`, the **exact** Plex title for Home members (`Becky`, `BluJay`,
`The Boys`), and the username for shared friends (`leosmith484` for display title "Leo Smith"). Case-sensitive.
`create_playlist(user=<key>)` already places the playlist via admin `switchUser`, and `plexapi.copyToUser` is the
same path — do not add a second ownership mechanism. `PlexServer.account` is a zero-arg *method*, so any code that
calls `server.account` as a property gets `None` on real connections.
`collect_user_ratings(user=<key>)` is the same walk made as that account, which is why a member's stars never appear on
the admin's `get_all_raw_tracks()` pass — and that pass costs 30-60 s on a 50 k-track library, which is why `library_cache`
stores its rows rather than letting a filter rescan per request.
`norm`/`norm_artist` (and the `song_key`/`album_key`/`artist_key` built from them) are the **join keys**: `library_cache.track_keys`
stores exactly those spellings, `spotify_import` asks them, and plexclean's third-party tooling computes the same strings,
so one key has one spelling across the repos — do not re-implement a second normalization.

### `prompt_classifier.py`

```
CONSTANT CONFIDENCE_THRESHOLD = 0.55
         _EMBED_DIM = 512  _EMB_MIN_SIM = 0.2  _EMB_REL_BAND = 0.6  _EMB_NEAR_MISS_BAND = 0.45
         _TOKEN_RE  _NORMALIZE_RE  _GENRE_DESC_INDEX (normalized GENRE_DESCRIPTIONS)
def extract_genre_hint_groups(prompt_lower)      # {keyword: [genres]}
def extract_genre_hints(prompt_lower)  extract_decade_hints(prompt_lower)  is_genre_keyword(keyword)
def match_genres_to_available(candidate_genres, available)
class PromptClassification(genres, decades, confidence, method, reasoning, matched_hints, excluded_genres)
def classify_prompt(prompt, available_genres, available_decades, threshold=CONFIDENCE_THRESHOLD)
private: _embedding_suggest returns (genres, confidence, excluded) for near-misses inside
         _EMB_NEAR_MISS_BAND = 0.45; _describe() renders "Prompt phrases:" / "No library genre for:" / "Close but left out:"
```

### `recommender.py`

```
CONSTANTS  ALBUM_ARTIST_MIN_SCORE = 70  ALBUM_COMBINED_MIN_SCORE = 70
           DIMENSION_LIBRARY (12 named dimensions)  SESSION_EXPIRY = 1800  MAX_SESSIONS = 100
           FAMILIARITY_SELECTION_DIRECTIVES  FAMILIARITY_PITCH_GUIDANCE (comfort/rediscover/hidden_gems)
def format_answers_for_selection(answers, answer_texts)  format_answers_for_pitch(answers, answer_texts)
class RecommendationPipeline:
    create_session(session_state)  migrate_sessions_from(other)  get_session(session_id)  delete_session(session_id)
    update_session_questions/answers/generate_state(session_id, ...)   get_session_costs(session_id)
    update_previously_recommended(session_id, new_keys)
    gap_analysis(prompt, session_id)  analyze_prompt_filters(prompt, available_genres, available_decades)
    generate_questions(prompt, dimension_ids, session_id)
    select_albums(prompt, answers, answer_texts, album_candidates, session_id, familiarity_pref="any", familiarity_data=None, previously_recommended=None)
    write_pitches(recommendations, prompt, answers, answer_texts, session_id, research=None, familiarity_pref="any", familiarity_data=None, extracted_facts=None)
    validate_pitch(pitch, facts, session_id)  rewrite_pitch(rec, facts, validation, prompt, answers_str, session_id)
    build_taste_profile(album_candidates)  select_discovery_albums(...)  validate_discovery_album(rec, research, prompt, session_id)
    extract_facts(artist, album, research, session_id)
```

Sessions are in-memory only (`SESSION_EXPIRY` 30 min, `MAX_SESSIONS` 100): restarts lose them, and `main.py`
holds one pipeline singleton (`_get_pipeline()`).

### `spotify_import.py`

```
CONSTANTS  SPOTIFY_ACCOUNTS_TOKEN_URL = accounts.spotify.com/api/token  SPOTIFY_API_BASE = api.spotify.com/v1
           SPOTIFY_PAGE_SIZE = 50  MAX_PLAYLIST_TRACKS = 1000  SPOTIFY_SEARCH_LIMIT = 12
           CACHE_KEY_SCAN_LIMIT = 400  PLAYLIST_ID_RE = [A-Za-z0-9]{8,64}
           _token_cache: tuple[str, float] | None     # client-credentials token + expiry epoch
class SpotifyImportError                             # every fetch/match refusal
def parse_playlist_ref(ref) -> (playlist_id, canonical_url)   # URL / spotify:playlist:<id> / bare id
def api_access_token(client_id, client_secret) -> str          # cached until expiry, retried once on HTTP 401
def fetch_playlist_api(playlist_id, client_id, client_secret) -> {name, owner, tracks, warnings}
def fetch_playlist_dump(playlist_id, parquet_dir) -> {name, owner, tracks, warnings}   # duckdb, imported inside
def fetch_playlist(playlist_ref, config) -> {playlist_id, playlist_url, source, name, owner, tracks, warnings}
def match_cached(tracks) -> (matched, missing)         # three join rungs against track_keys
def match_live(tracks, plex) -> (matched, missing)     # live-Plex search fallback
private: _api_get (Bearer GET)  _api_row (one item → row dict, None when it names no song)
         _token_cache_clear  _pick_candidate (duration-closest of the candidates)
         _match_rows (the rung walk, shared by match_cached)
```

`fetch_playlist` picks the path by what `config.spotify` carries: credentials → `api`, else `parquet_dir` →
`dump`, else a `SpotifyImportError` naming both remedies (the route turns it into HTTP 502). `fetch_playlist_dump`
guards the one-directory-per-table layout **before** importing duckdb, so a machine with no duckdb still gets a
named error rather than an `ImportError` (duckdb is imported **inside** the helper, after that guard, so the module stays
optional); dump rows carry no
Spotify ids, so `spotify_track_id`/`spotify_url` stay empty. Episodes are skipped on both paths.
`match_cached` asks the cache's `track_keys` table: exact `match_key`, then `ck_artist` + `match_key GLOB '<title>|*'`
per credited artist (`&`-split, so "A & B" lands when Plex lists "B feat. A"), then `ck_album` + the same title glob;
the duration-closest candidate wins, which is how a deluxe-edition twin loses. `match_live` only runs when
`has_cached_tracks()` is false and labels every landing `search`, since Plex's full-text search cannot name a rung.

### `version.py`

`FALLBACK_VERSION = "dev"`; `get_version()` reads `APP_VERSION` env, else `git describe --tags`, else `"dev"`.
Releases are git tags `vMAJOR.MINOR.PATCH`; CI derives the image version the same way.

### `main.py`

Public surface is the route table in §1 plus `lifespan(app)` and the module singletons it fills:
`get_plex_client()` / `get_llm_client()` / `_get_pipeline()` / `get_track_cache()` / `get_config()`.
Load-bearing private constants here, since routes gate on them:

```
_VALID_RESULT_TYPES = {prompt_playlist, seed_playlist, album_recommendation, spotify_import}  # the results.type column
_RESULT_ID_RE = ^[0-9a-f]{8,16}$                                              # path-param validation for /api/results/{id}
_EXTERNAL_ART_DOMAINS = {coverartarchive.org, archive.org}                   # /api/external-art allowlist
```

---

## 4. SQLite state — `data/library_cache.db`

Path is derived from module location, **not** cwd: `DATA_DIR = Path(__file__).parent.parent / "data"` ⇒
always repo-root `data/library_cache.db`. It is gitignored (`*.db`): the file is machine state, so never quote
its row counts as a repo fact. Connections open in WAL mode (`get_db_connection()`). `agent_reasoning.py` derives
its own path the same way (`DATA_DIR / "agent_song_reasoning.db"`), so the two DBs stay independent files and one
feature's schema change never touches the other's tables.

| Table | Columns | Notes |
|---|---|---|
| `tracks` | `rating_key* (PK), title, artist, album, duration_ms, year, genres, user_rating, is_live, parent_rating_key, updated_at` | `genres` is JSON text; indexes on `artist`, `year`, `is_live`, `parent_rating_key` |
| `sync_state` | `id* (PK CHECK = 1), plex_server_id, last_sync_at, track_count, sync_duration_ms` | single row, `INSERT OR IGNORE` |
| `results` | `id* (PK), type, title, prompt, snapshot (JSON), track_count, artist, art_rating_key, subtitle, playlist_id, playlist_title, playlist_owner, playlist_url, created_at` | indexes `(type, created_at DESC)` and `(created_at DESC)`; the four `playlist_*` columns are the history→Plex link (NULL = unlinked). `id` is `secrets.token_hex(8)` = 16 hex chars; `type` ∈ `_VALID_RESULT_TYPES` |
| `track_dimensions` | PK `(rating_key, model, fingerprint)`, plus `dimensions (JSON), created_at` | cache of seed-track dimensions; fingerprint embeds `DIMENSION_PROMPT_VERSION` |
| `user_ratings` | `user_key, rating_key, rating, updated_at` + PK `(user_key, rating_key)` | per-Plex-account stars — `tracks.user_rating` only carries the connecting admin account, so a filter asked in a member's stars reads here; the effective rating is `COALESCE(user_ratings, tracks.user_rating)`, which is what makes the admin account the fallback for tracks a member never rated |
| `rating_sync` | `user_key* (PK), rated_count=0, total_seen=0, last_sync_at` | which accounts have actually been scanned: a missing line means "this account's ratings are not loaded yet", which is **not** the same as "loaded, rated nothing" (`has_user_ratings` vs `user_ratings_status` distinguish the two) |
| `track_keys` | `rating_key* (PK), match_key, ck_album, ck_artist, updated_at` | the library's own join keys, rebuilt by `rebuild_track_keys`/`ensure_track_keys` — deliberately **not** an ALTER on `tracks`, so it never sets `migrated`/`needs_resync()`; also indexed on `match_key`, `ck_album`, `ck_artist` for the import's fallback rungs |
| `key_state` | `id* (PK CHECK = 1), keyed_tracks = -1` | single row: how many `tracks` rows the `track_keys` table covers; a mismatch is rebuilt by `ensure_track_keys`, never trusted |
| `missing_songs` | `playlist_ref, position, playlist_name, title, album, artist, duration_ms, spotify_track_id, spotify_url, recorded_at` + PK `(playlist_ref, position)` | the Spotify wish list: rows an imported playlist asked for that this library cannot play; re-importing REPLACES a playlist's rows (`record_missing_songs` DELETEs then inserts), and a slot with `position <= 0` is never stored |

Migration rule (`init_schema()`): `ALTER TABLE` adds missing columns for old DBs — `tracks` gains
`parent_rating_key`, `view_count`, `last_viewed_at`; `results` gains `subtitle` then the four `playlist_*` columns.
Only the `tracks` ALTERs set the `migrated` flag that drives `needs_resync()` and the lifespan auto-sync (dimension
detail is folded into the fingerprint instead of ALTERed); the `results` ALTERs are **deliberately excluded** so a
results-shape change must not ask for a library re-sync. `init_schema()` also **drops** `external_data` and
`external_state` from DBs that an older plexsage left behind — third-party facts live in the sibling plexclean
checkout now — and that DROP is kept out of `migrated` for the same reason: removing tables says nothing about the
`tracks` shape. New columns belong on an existing table, not in a new table or a link endpoint.
`key_state` (with `track_keys`) and the Spotify wish list (`missing_songs`) are plain `CREATE TABLE IF NOT EXISTS`
entries in the same script and never set `migrated`; an old DB gains them on first open, and its `track_keys` are
filled the next time `ensure_track_keys` runs — which is the Spotify import's matcher, not the filter routes.

The second file, `data/agent_song_reasoning.db`, has three tables (`agent_reasoning._init_tables`): `runs`
(`run_id* PK = secrets.token_hex(8), source='prompt', prompt='', playlist_title='', result_id='', track_count=0,
created_at`), `selections` (`run_id, rating_key, reason, created_at`, PK `(run_id, rating_key)`, indexed on
`rating_key`) and `tags` (`rating_key, tag, created_at`, PK `(rating_key, tag)` — so a tag the curator reused costs no
extra row). All three are `IF NOT EXISTS` with no ALTER and no migration flag: nothing here is ever joined against
`tracks`, and `runs.result_id` is a text link to a `results` row in the **other** file.

## 5. Frontend surface — no build step, hand-written

| File | Shape |
|---|---|
| `frontend/index.html` (~1031 lines) | one static DOM of id'd panels; `GET /` serves it with cache-busted `/static/style.css?v=` + `/static/app.js?v=` |
| `frontend/style.css` (~3945 lines) | dark Plexamp styling (`#1a1a1a` / `#e5a00d`); panels are targeted by class name from JS |
| `frontend/app.js` (~6294 lines, 171 top-level `function`/`const`/`window.` entries by the grep in §7) | all behavior; talks to the API via one `apiCall(endpoint, options = {})` wrapper |

```
app.js constants: focusManager (modal focus trap)  OWNER_USER_KEY = '__owner__'
                  OWNERSHIP_AS_USER = 'as_user'  OWNERSHIP_COPY_TO = 'copy_to_user'
state (single mutable object): view/mode/step, prompt, questions/questionAnswers/questionTexts,
    seedTrack/dimensions/selectedDimensions/additionalNotes,
    availableGenres/availableDecades/selectedGenres/selectedDecades/trackCount/excludeLive/maxTracksToAI/minRating,
    playlist/playlistName/tokenCount/estimatedCost/resultId,
    playlistTitle/narrative/trackReasons/userRequest, sessionTokens/sessionCost, loading/error, config,
    lastFilterPreview/lastRelaxation, selectedTrackKey, plexClients/_pendingClientId, saveMode/selectedPlaylistId/plexPlaylists,
    plexUsers/plexUsersLoaded/plexUser/playlistOwnership/ratingUser, rec {…}, setup {…}
window.* exports (used by index.html inline handlers):
    artPlaceholderHtml hideError hideSuccess hideSuccessModal dismissSuccessModal
    dismissClientPicker dismissPlayChoice dismissPlaySuccess dismissUpdateSuccess closeBottomSheet
```

Function families (name prefixes are the navigation aid): `fetch*/update*/render*` paint panels,
`handle*` are event handlers, `dismiss*`/`open*` drive modals, `show*/hide*` toggle banners + loading,
`_*` are internal step helpers (`_processStepQueue`, `_applyStepUpdate`). Views are
`home | create | recommend | settings`; create steps are `input | refine | dimensions | filters | results`;
recommend steps are `prompt | refine | setup | results`. Two invariants hold in these files and are worth re-checking
after any edit (both once had duplicates): every `id="…"` in `index.html` is unique — a repeated id means
`getElementById` silently drives only the first copy, so a later panel's chip is dead — and no top-level function name
is declared twice in `app.js` (a re-declaration replaces the earlier body, which is how a "Rated By" populate helper
came to exist three times beside two identical `filter-section` blocks).

The frontend never sees the Plex token: album art always comes from `/api/art/{rating_key}` or `/api/external-art`.
It also sends `result_id` when saving, so `POST /api/playlist` can link the history row; the history card renders
a `.history-card-playlist` badge anchored at `playlist_url` with `.history-card-playlist-owner` showing the owner.
When the cache had to widen the filters, the UI says so instead of implying the strict reading:
`updateFilterPreviewDisplay` appends `· widened from N as asked: tier → tier` to the count line from
`state.lastFilterPreview.relaxed_tiers`, `renderNarrativeBox` prints a `.narrative-note` line (`var(--warning)`) from
`state.lastRelaxation` (set from the `complete` event's `relaxed_tiers`/`matching_as_asked`, `null` when nothing was
given up), and the SSE `progress` step `relaxing` maps to `preparing` in `PLAYLIST_STEP_MAP`.
The account a rating facet filters by is explicit, too: `updateFilterPreview` and `handleGenerate` both send
`rating_user` only when `state.ratingUser` is not `__owner__`, so the admin's column stays the default and a member's
stars are opt-in — and because the run names the same account the preview counted, the candidate set is the one the
count line advertised. The chip that sets `state.ratingUser` is remembered in `localStorage`
(`mediasage-rating-user`) and reset to `__owner__` when the account is no longer in `state.plexUsers`. The two local
stores now have a panel of their own in the settings view — `memory-stats-section` (`index.html`), whose `memory-ratings`
+ `memory-reasoning` divs `renderRatingsAndMemorySection()` fills at the end of `loadSettings` from
`fetchUserRatingsStatus` (`GET /api/library/ratings/status`) and `fetchReasoningStatus` (`GET /api/reasoning/status`):
per-account rated counts and scan stamps, an explicit `.memory-warning` line when the account the chips asked for is not
in `synced_users` (i.e. a ★+ floor is really reading the admin's stars), and the memory's runs / reasons / mined-tag
vocabulary. It is deliberately not guarded on `plex_connected` — both stores are local, and the ratings one exists even
with no Plex connection. `rescan-ratings-btn` (same section, wired in `setupEventListeners`) is the only caller of
`refreshUserRatings` (`POST /api/library/ratings/refresh`): it names every account but `__owner__` from `fetchPlexUsers`
and re-renders the panel from the response, so a scan that failed for a member says which one rather than blanking the
line.

## 6. Data files + secrets

| Path | Role | Git |
|---|---|---|
| `config.yaml` (repo root) | base config | gitignored |
| `data/config.user.yaml` | UI-saved overrides (`USER_CONFIG_PATH`) | gitignored (under `data/`) |
| `.env` | env-layer source, read at `config.py` import | gitignored |
| `data/library_cache.db` | SQLite state for the library cache (§4) | gitignored (`data/*.db`) |
| `data/agent_song_reasoning.db` | SQLite state for the agent-reasoning memory (§4, `agent_reasoning.py`) | gitignored (`data/*.db`) |
| `data/external/` + `data/*.parquet` / `*.jsonl` | leftover ignore rules for a user's third-party dump; nothing in the app reads them any more (the Spotify import's `parquet_dir` is an absolute path outside the repo) | gitignored |
| `config.example.yaml` | the only committed config template | tracked |

Secrets for the notebook live outside any repo, in `~/Library/Application Support/PlexSage/config.yaml`
(reached with `PLEXSAGE_CONFIG`); never add secrets to the notebook or to tracked files.

## 7. Commands + test baseline (verified)

```bash
uvicorn backend.main:app --reload --port 5765      # UI :5765, docs /docs, health /api/health
uv run --with-requirements requirements.in --with-requirements requirements-dev.in pytest tests/ -q
#   same prefix + tests/test_config.py -q   or   -k 'filter_preview'
uvx ruff check .            # default rules; scope to files you touched, never --fix repo-wide
```

No pytest/ruff/typecheck config, no Makefile, no CI test job. `.github/workflows/docker-publish.yml` only
builds + pushes images. Neither system `python3` nor `~/mlx_env` has pytest/ruff; `uv` is on PATH and deps
are pinned in `requirements.in` / `requirements-dev.in` (`requirements*.txt` are the compiled pins for the
README's `pip install -r`, compiled with `--python-version=3.14`).

Suite today: **512 tests → 6 failed, 506 passed** with the unpinned command above;
**5 failed, 507 passed** with the compiled pins. The known-broken set is fixed, not a regression:
3 `test_llm_client.py` init-signature tests (they assert the pre-`timeout`/`max_retries` client call), 2 `test_api.py`
filter-preview tests that assume an empty library, and `test_post_config_updates_llm_provider` (fails only against
newest SDKs). `tests/test_spotify_import.py` (24 tests) and `tests/test_agent_reasoning.py` (30 tests) are green in both
runs, as are the 4 `TestSpotifyConfig` tests in `test_config.py`. The facet + relaxation suites are green in both runs (`TestFilterRelaxation` in
`test_library_cache.py`, `TestFilterPreviewRelaxation` in `test_api.py`, `TestCandidatePoolRelaxation` in
`test_generator.py`, and the facet sets `TestFacetFilters`/`TestYearWindowFacet`/`TestPlayWindowFacet`/`TestBanFacet`/
`TestStreamFacets`): they fake `count_tracks_by_filters`, so no count in them is the local DB's.
The ratings store now has two green suites: `TestUserRatings` (8 of the 85 in `test_library_cache.py`) runs the
methods against canned clients — it fakes `plex_client.collect_user_ratings`, so no Plex walk and no `switchUser`
happens in the suite, and its `unknown_plex` client proves a failed scan raises `PlexQueryError` instead of answering
0 rows — and `TestRatingsAndReasoningEndpoints` (7 of the 65 in `test_api.py`) answers the three routes from a canned
`rating_sync` state + a canned `reasoning_status()` payload, so no test depends on this machine's DB. `agent_reasoning`
is covered too: its 30 tests (`TestSchemaCreation`/`TestRecordRun`/`TestReplay`/`TestTagQueries`/`TestStatusRead`) run
`record_run`/`get_previous_reasons`/`get_reason_tags`/`keys_with_tags`/`reasoning_status` against a throwaway DB — the
fixture repatches `DATA_DIR` **and** `DB_PATH` plus `_schema_initialized`, since `DB_PATH` is bound at import and a
fixture that misses it writes into the real `data/agent_song_reasoning.db`.

Tests carry process state: `tests/conftest.py` autouse-clears the module-level `TrackCache`; startup auto-syncs
when `needs_resync()` was set by a schema migration (`test_setup.py` patches that out). Because the local DB here
is populated, new API tests must patch `library_cache.has_cached_tracks` / `count_tracks_by_filters` /
`user_ratings_status` / `agent_reasoning.reasoning_status` / `link_result_playlist` — otherwise expected numbers become
machine-dependent (`TestUserRatings` and `TestRatingsAndReasoningEndpoints` both do exactly that).
`test_spotify_import.py` sidesteps that instead: its `temp_db` fixture repatches both `DATA_DIR` **and** `DB_PATH`
(patching only `DATA_DIR` writes into the real cache), and its fetch tests canned `requests` responses plus a fake
`duckdb` module in `sys.modules`, so no network and no NAS read happens in the suite. The uv test env has **no**
pyarrow/pandas/polars/duckdb — which is exactly what makes the parquet-dump import path raise its named
`SpotifyImportError` there instead of an `ImportError`; `~/mlx_env` has pyarrow, pandas and duckdb (no polars).

## 8. Re-deriving this file after an edit

The tables above are generated, not hand-typed, so re-check any row you touch:

```bash
grep -n '@app\.' backend/main.py                                    # route table (§1)
python3 - <<'EOF'                                                   # module surface (§3), signatures + constants
import ast, pathlib
for p in sorted(pathlib.Path("backend").glob("*.py")):
    print(f"### {p.name}")
    for n in ast.parse(p.read_text()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
            kind = "async def" if isinstance(n, ast.AsyncFunctionDef) else "def"
            print(f"{kind} {n.name}({ast.unparse(n.args)})")
        elif isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
            print(f"class {n.name}")
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.isupper():
            print(f"{n.targets[0].id} = {ast.unparse(n.value)}")
EOF
grep -n 'CREATE TABLE\|ALTER TABLE' backend/library_cache.py         # DB schema (§4)
grep -n '^function \|^const \|^window\.' frontend/app.js            # frontend surface (§5)
python3 - <<'EOF'                                                   # the two §5 frontend invariants
import re, collections
html = open("frontend/index.html").read()
print([k for k, v in collections.Counter(re.findall(r'id="([^"]+)"', html)).items() if v > 1])
js = open("frontend/app.js").read()
print([k for k, v in collections.Counter(re.findall(r'^(?:async\s+)?function\s+(\w+)', js, re.M)).items() if v > 1])
EOF
```

Rules for edits: new contracts go in `models.py`; new columns go on the existing table with an `ALTER`
migration outside `init_schema()`'s `migrated` return when they don't need a re-sync; new dimension detail
raises `DIMENSION_MIN`/`DIMENSION_MAX` and bumps `DIMENSION_PROMPT_VERSION` rather than clearing the cache table.

## 9. Pointers

- `backend/BACKEND_MAP.md` — flows (how a request walks through), "Where to Look for Common Tasks", dependency graph.
- `backend/BACKEND_IMPROVEMENTS.md` — why `track_db.py` was deleted and what carried onto `library_cache.py`; do not add a second cache module.
- `AGENTS.md` — repo conventions, notebook traps, Plex account ids, commit/tag conventions. It is gitignored, so this file is the committed mirror of the *surface* part of it, not a copy.
- `README.md` — user-facing "API Reference" and install/config guides (`docs/` holds the Synology + Gemini guides).



