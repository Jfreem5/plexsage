# Backend Improvements — Status

Ideas for cutting LLM cost/latency and repeated Plex API calls, originally
tracked while building the `local_ai` branch.

`main` has since landed 116 commits that independently added a local SQLite
library cache (`backend/library_cache.py`) plus a recommender and setup flow.
The branch was merged into `main` and now **converges on `library_cache.py`** as
the single local cache; our duplicate `track_db.py` was retired. This file
records what carried over.

## Carried over (re-applied on top of `main`)

- **Local prompt classification** — `backend/prompt_classifier.py`: keyword
  rules, then stdlib feature-hashed embeddings, then the LLM only at low
  confidence. `analyzer.analyze_prompt()` is local-first and returns
  `method`/`confidence`. ✅
- **Seed-track dimension cache** — now stored in `library_cache.track_dimensions`
  (keyed by rating key + analysis model + metadata fingerprint).
  `AnalyzeTrackResponse.cached` reports a hit; zero tokens on a repeat. ✅
- **Dynamic AI track budget** — `generator.compute_track_budget()` +
  `_prompt_specificity()`. Used by `generate_playlist_stream()` and by
  `/api/filter/preview` (optional `prompt` field) so the preview matches the
  real request. ✅
- **Rule-based pre-ranking** — `generator.pre_rank_tracks()`: genre/decade match
  plus user rating and recency from `library_cache.get_metadata_batch()`. Trims
  the candidate pool to the budget before any tokens are spent. ✅
- **Album art proxy caching** — `GET /api/art/{rating_key}` returns
  `Cache-Control: public, max-age=86400`. ✅

## Provided by `main`'s `library_cache.py` (our duplicate removed)

- **Local track DB / reads served locally** — `library_cache.py`
  (`data/library_cache.db`) serves filters, counts, search, stats, and album
  candidates with no Plex round-trips; sync runs on startup and on demand via
  `POST /api/library/sync` with progress polled from `GET /api/library/status`.
- **Pre-computed live detection** — `tracks.is_live` is computed once at sync and
  filtered in SQL (`WHERE is_live = 0`).
- **Batched sync** — `sync_library()` walks the library in `SYNC_BATCH_SIZE`
  (500) batches, so peak memory tracks the batch size.

## Superseded / not carried over

- **Embedding-based pre-ranking** (two-stage, item 1) — only the rule-based
  pre-ranking carried over; track embeddings are not stored or used for ranking.
- **Fuzzy-match index** (`TrackMatcher`, item 6) — `main` matches with
  `_tracks_match()` + `rapidfuzz`; the trigram-pruned matcher was not carried over.
- **Sync/config/generation-prep refactors** (A, B, C) — `main` restructured
  these: a single streaming generation path, shared config-response helpers, and
  a background sync with progress reporting.
- **Thread-safety and covering-index work** (D, F) — `main`'s cache uses
  per-call WAL connections and its own index set instead.

## Still open

- **Debounce the filter preview** (G) — the frontend re-hits
  `/api/filter/preview` on every slider change when a prompt is set.
