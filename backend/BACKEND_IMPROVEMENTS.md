Based on the backend architecture, here are the key areas where you can reduce LLM effort and improve filtering/search:

1. Two-Stage Generation (Reduce Tracks Sent to LLM)
Current: generator.py sends up to 2000 tracks (or max_tracks_to_ai) as a numbered list to the LLM, which then picks 15-40. The LLM must "read" all of them.

Opportunity: Add a pre-ranking stage before the LLM call:


Apply
Filtered tracks (e.g., 2000)
    → [NEW] Embedding/rule-based pre-rank to top 200-500
    → LLM selects 25 from the smaller set
This could cut input tokens by 75-90% on the most expensive call. Approaches:

Embedding similarity: Pre-compute embeddings for each track (title + artist + album + genres). When a prompt comes in, embed the prompt and take the top-N most similar tracks.
Rule-based scoring: Score tracks by genre match, decade match, rating, and recency. Only send the top scorers.
Hybrid: Use rules for hard filters (genre/decade), then embeddings for soft ranking (mood, vibe).
Where to add: generator.py — between the "filtering" step and the "preparing" step. You'd add a pre_rank_tracks(tracks, prompt) function.

2. Replace LLM Prompt Analysis with Embeddings + Rules
✅ Implemented: see `backend/prompt_classifier.py`. `analyzer.analyze_prompt()` now classifies via keyword rules, then local embedding similarity, and only calls the LLM when confidence is low (`method: none`). Response includes `method`/`confidence`; the `/api/analyze/prompt` endpoint no longer requires an LLM to be configured.

Current: analyzer.analyze_prompt() makes a full LLM call just to map "something chill for a rainy day" → ["Ambient", "Post-Rock"] + ["2010s"].

Opportunity: This is a classification/mapping task that doesn't need a full LLM:

Keyword → genre mapping: A curated dictionary (e.g., "chill" → Ambient, Lo-Fi, Jazz; "party" → EDM, Pop, Hip-Hop). Covers 80% of prompts with zero LLM cost.
Embedding match: Embed the prompt, compare against embedded genre descriptions. Pick top matches.
Fallback to LLM: Only call the LLM if the rule/embedding confidence is low.
Where to add: New module prompt_classifier.py or extend analyzer.py. The LLM call becomes a fallback, not the primary path.

3. Local Track Database (Replace Repeated Plex API Calls)
✅ Implemented: `backend/track_db.py` now serves filtering, FTS search, stats, and random sampling from a file-backed SQLite mirror with zero Plex round-trips. `PlexClient` reads from the DB when populated (falling back to the API otherwise); `sync_track_db()` mirrors the library on startup and every 30 min, plus `POST /api/library/sync`.

Current: Every get_tracks_by_filters(), get_library_stats(), search_tracks() hits the Plex HTTP API. The TrackCache (5-min TTL, 50 entries) helps but is in-memory and short-lived.

Opportunity: A local SQLite database (zero infra, file-based) that mirrors your library:


Apply
CREATE TABLE tracks (
    rating_key TEXT PRIMARY KEY,
    title TEXT,
    artist TEXT,
    album TEXT,
    year INTEGER,
    duration_ms INTEGER,
    is_live INTEGER,          -- pre-computed
    user_rating INTEGER,
    play_count INTEGER,
    last_played TEXT,
    embedding BLOB            -- vector for similarity search
);

CREATE TABLE track_genres (
    rating_key TEXT,
    genre TEXT,
    PRIMARY KEY (rating_key, genre)
);

CREATE INDEX idx_artist ON tracks(artist);
CREATE INDEX idx_year ON tracks(year);
CREATE INDEX idx_rating ON tracks(user_rating);
Benefits:

Instant filtering — SQL WHERE genre IN (...) AND year BETWEEN ... AND user_rating >= 8 is milliseconds vs. Plex API round-trips
Full-text search — search_tracks() becomes a SQL FTS query instead of Plex's limited search
Persistent cache — No TTL expiry; sync on a schedule (e.g., every 30 min or on library change)
Enables embeddings — Store a vector per track for similarity search
Complex queries — "Tracks by artists who also have albums in the 90s" is a JOIN, not an LLM call
Where to add: New module track_db.py with a sync_from_plex() function. plex_client.py methods would check the DB first, fall back to Plex API.

4. Cache LLM Analysis Results Per Track
✅ Implemented: `track_db.py` has a `track_dimensions` table (rating_key, dimensions JSON, model, metadata fingerprint, analyzed_at). `analyzer.analyze_track()` checks it before calling the LLM and returns `cached=true` with zero tokens on a hit; entries are invalidated when the analysis model or the track's metadata changes. `prune_except()`/`clear()` drop stale rows.

Current: analyze_track() calls the LLM every time you select a seed track. If you analyze the same track twice, you pay twice.

Opportunity: Store dimensions in the track database:


Apply
CREATE TABLE track_dimensions (
    rating_key TEXT PRIMARY KEY,
    dimensions JSON,          -- the 5-7 dimensions
    analyzed_at TEXT,
    model TEXT               -- which model generated them
);
Check DB before calling LLM
Invalidate when the model changes or track metadata changes
This is a pure win — same track, same model = same dimensions
5. Pre-Compute Live Detection
✅ Implemented: `tracks.is_live` is computed once per track in `sync_track_db()`. `get_tracks_by_filters`/`count_tracks_by_filters`/`get_random_tracks` filter with `WHERE is_live = 0` (no Python post-filtering). The flag is exposed on `Track.is_live` and via `TrackDB.get_live_flags()`; legacy DBs gain the column automatically. The Plex API fallback detects once per fetched track.

Current: is_live_version() runs regex on every track every time it's fetched. It's cheap but repeated.

Opportunity: Store is_live as a boolean column in the track DB (computed once during sync). Filtering becomes WHERE is_live = 0 instead of post-filtering in Python.

6. Fuzzy Matching Optimization
✅ Implemented: `generator.py` now uses `TrackMatcher`, which indexes the pre-ranked candidates once per generation: a normalized exact-match dict (`(artist, title)` incl. `&`/`and` variants), then a title-trigram inverted index that prunes fuzzy comparisons to a small candidate set (`O(k)`), with a linear-scan fallback when the overlap is too broad or trigrams are too short. `_tracks_match()` remains as a single-pair wrapper. Benchmark: ~5× faster on realistic, diverse libraries; near parity on pathological duplicate-heavy libraries.

Current: _tracks_match() in generator.py is O(n×m) — for each LLM selection, it scans all filtered tracks with rapidfuzz. With 500 tracks and 25 selections, that's 12,500 fuzzy comparisons.

Opportunity:

Exact match first: Build a dict of {(artist.lower(), title.lower()): track}. Check exact match before fuzzy.
Trigram index: Pre-compute trigrams for titles/artists. Only fuzzy-compare candidates that share trigrams.
Database FTS: If using the track DB, use SQLite FTS5 for the initial candidate set, then fuzzy-rank the small result.
This reduces the matching step from O(n×m) to roughly O(n×k) where k is a small candidate set.

7. Smart max_tracks_to_ai Calculation
✅ Implemented: `generator.compute_track_budget()` sizes the pre-ranked candidate set dynamically from a 0-1 prompt-specificity score (concrete genre/decade cues and explicit genre filters → fewer options; vague mood/activity prompts → more). It interpolates between a floor (scaled by `track_count`) and the ceiling (`max_tracks_to_ai`, `PRE_RANK_LIMIT`, or the candidate pool). Both generation paths and `/api/filter/preview` (new optional `prompt` field) use it, so the preview matches the real request. With no prompt/filters the ceiling is unchanged (`500`), keeping the seed flow identical.

Current: get_max_tracks_for_model() calculates a static max based on context window. The UI shows this as a fixed number.

Opportunity: Make it dynamic based on the actual prompt complexity:

Short prompt ("jazz") → need fewer tracks (genre filter already narrows it)
Vague prompt ("something for a road trip") → need more tracks for the LLM to have options
If using the two-stage pre-ranking (item 1), the effective max can be lower since the LLM only sees pre-ranked tracks

8. Bulk Library Sync Performance
✅ Implemented: `TrackDB.upsert_tracks()` rewrites the FTS5 index for the whole batch at once (`_replace_fts_bulk`) instead of per row. `rating_key` is declared `UNINDEXED` in FTS5, so the old per-row `DELETE ... WHERE rating_key = ?` degraded to a full index scan — O(n) per track, i.e. O(n²) per sync. Measured: a 10,000-track sync dropped from ~4.35s to ~78ms, and 30,000 tracks now sync in ~0.25s (linear). This is on the startup/30-min sync path and on every generation's `sync_from_plex()`.

9. Album Art Proxy
✅ Implemented: `GET /api/art/{rating_key}` now performs a single Plex item lookup (it previously fetched the same item twice — once to validate, once for the thumb) and returns `Cache-Control: public, max-age=86400`, so browsers reuse art instead of re-proxying it through the backend on every render.

Summary: Impact vs. Effort
Improvement	LLM Cost Reduction	Effort	Complexity
Two-stage pre-ranking (embeddings)	60-90% on generation	Medium	Medium
Rule-based prompt classifier	100% on analysis (most prompts)	Low	Low
Local SQLite track DB	0% (but enables everything else)	Medium	Low
Cache track dimensions	100% on repeat analyses	Low	Low
Pre-compute is_live	0% (but faster filtering)	Low	Low
Fuzzy match optimization	0% (but faster)	Low	Low
Dynamic max_tracks	20-40% on generation	Low	Low
The highest-leverage combination is: SQLite track DB (foundation) + two-stage pre-ranking (biggest cost cut) + rule-based prompt classifier (eliminates the cheapest-but-unnecessary LLM call). Together these could reduce total LLM spend by 70-80% while improving speed and reliability.


Remaining candidates
#	Opportunity	Impact	Effort
A	✅ Implemented: both generation paths only mirror a batch into the DB when the read fell back to the Plex API — `PlexClient.db_available()` reports whether reads are being served locally, so a DB-served generation skips the extra metadata read + write + commit	lower latency/write load	Low
B	✅ Implemented: both paths share `_prepare_generation()` (fetch → mirror → budget → pre-rank → prompt) and `_match_selections()`; the streaming wrapper forwards progress events with `yield from`, the sync wrapper drains them (`_collect_preparation`). The non-streaming path now uses the same `TrackCache` as streaming instead of re-querying Plex.	removes drift	Medium
C	✅ Implemented: `main.py` builds every `ConfigResponse` through `_config_response(config)` (with `_is_llm_configured(config)`), shared by `GET`/`POST /api/config` and `/api/health` instead of three copies	maintainability	Low
D	✅ Implemented: `TrackDB` gives each thread its own connection (WAL), so a concurrent reader never observes a writer's half-applied batch; writes run through `_write()`, which serializes them with a re-entrant lock and rolls back on error. `close()` closes all connections and later use reopens transparently.	robustness	Low–Med
E	✅ Implemented: `sync_track_db()` walks the library in `SYNC_PAGE_SIZE` (500) pages using `container_start`/`container_size` (plus `maxresults`, which caps plexapi's internal loop to one page per request) and upserts each page as it arrives, so peak memory tracks the page size instead of the whole library. Pruning still needs the set of rating keys seen.	memory on huge libraries	Medium
F	get_tracks_by_filters sorts by artist/album/title COLLATE NOCASE with no matching index → temp sort each call	query latency	Low
G	Frontend now re-hits /api/filter/preview on every slider change when a prompt is set (no debounce)	network chatter
