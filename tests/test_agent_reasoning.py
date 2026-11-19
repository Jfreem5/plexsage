"""Tests for the curator's curation memory (`backend/agent_reasoning.py`).

The memory lives in its **own** SQLite file, so every test here runs against a
throwaway copy: the DB path is derived from module location and `DB_PATH` is bound
at import, so a fixture has to repatch `DB_PATH` **and** `DATA_DIR` (repatching
only `DATA_DIR` writes into the real `data/agent_song_reasoning.db` — the same
trap that once let a fixture replace the real library cache). `_schema_initialized`
is a process-wide flag too, so it is reset per test.

Nothing here needs Plex or an LLM: runs are canned `(rating_key, reason)` pairs, and
the mined vocabulary is asserted against `backend.prompt_classifier`'s own keyword
maps — this module deliberately invents no keyword list of its own.
"""

import sqlite3
from datetime import datetime

import pytest

from backend import agent_reasoning


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    """Point the reasoning DB at a throwaway file and reset its schema flag."""
    db_file = tmp_path / "agent_song_reasoning.db"
    monkeypatch.setattr(agent_reasoning, "DATA_DIR", tmp_path)
    monkeypatch.setattr(agent_reasoning, "DB_PATH", db_file)
    monkeypatch.setattr(agent_reasoning, "_schema_initialized", False)
    yield db_file


def count_rows(db_path, table: str) -> int:
    """Rows in *table*, -1 when the table is gone (a malformed DB)."""
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(f"SELECT COUNT(1) FROM {table}").fetchone()[0]
    except sqlite3.Error:
        return -1
    finally:
        conn.close()


def read_rows(db_path, sql: str) -> list:
    """Rows from *sql*, nameable like the module's own connections make them."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(sql))
    finally:
        conn.close()


class TestSchemaCreation:
    """Three tables, in a file of its own."""

    def test_first_call_creates_the_file_and_tables(self, memory_db):
        """Opening the DB is what creates it; the three tables land together."""
        conn = agent_reasoning.ensure_db_initialized()
        conn.close()

        assert memory_db.exists()
        assert [count_rows(memory_db, name) for name in ("runs", "selections", "tags")] == [0, 0, 0]

    def test_opening_it_again_recreates_nothing(self, memory_db, monkeypatch):
        """The CREATEs are `IF NOT EXISTS`, so a second schema pass is harmless."""
        agent_reasoning.ensure_db_initialized()
        agent_reasoning.record_run([("k1", "A chill ambient reason")])
        monkeypatch.setattr(agent_reasoning, "_schema_initialized", False)
        agent_reasoning.ensure_db_initialized()

        assert count_rows(memory_db, "selections") == 1

    def test_the_reasoning_file_is_not_the_library_cache(self, memory_db):
        """Two independent DBs: a `tracks` schema change cannot reach this one."""
        from backend import library_cache

        assert agent_reasoning.DB_PATH != library_cache.DB_PATH
        assert agent_reasoning.DB_PATH.name == "agent_song_reasoning.db"


class TestRecordRun:
    """One row per run, one per (track, reason), mined tags beside them."""

    def test_the_return_value_counts_picks_not_rows(self, memory_db):
        """The caller logs picks stored, so a run row must not join the count."""
        assert agent_reasoning.record_run([("k1", "chill"), ("k2", "an 80s closer")]) == 2
        assert count_rows(memory_db, "selections") == 2
        assert count_rows(memory_db, "runs") == 1

    def test_a_pick_without_a_reason_is_skipped(self, memory_db):
        """The reason is the whole payload: a blank one stores nothing at all."""
        assert agent_reasoning.record_run([("k1", "   "), ("", "kept"), ("k2", "kept")]) == 1

        keys = [row["rating_key"] for row in read_rows(memory_db, "SELECT rating_key FROM selections")]
        assert keys == ["k2"]

    def test_no_reason_means_no_run_row(self, memory_db):
        """An empty run must not leave a `runs` line claiming zero picks."""
        agent_reasoning.ensure_db_initialized()
        assert agent_reasoning.record_run([]) == 0

        assert count_rows(memory_db, "runs") == 0

    def test_the_run_line_keeps_its_context(self, memory_db):
        """Prompt, title, source and the cache result id land on the run row."""
        agent_reasoning.record_run(
            [("k1", "A reason")],
            prompt="late-night driving",
            playlist_title="November 2026 Playlist",
            result_id="abc12345",
            source="seed",
        )

        row = read_rows(memory_db, "SELECT source, prompt, playlist_title, track_count FROM runs")[0]

        assert row["source"] == "seed"
        assert row["prompt"] == "late-night driving"
        assert row["playlist_title"] == "November 2026 Playlist"
        assert row["track_count"] == 1

    def test_result_id_is_a_string_link_not_a_foreign_key(self, memory_db):
        """The two DBs stay separate files, so the id is kept as text."""
        agent_reasoning.record_run([("k1", "A reason")], result_id="abc12345")
        conn = sqlite3.connect(str(memory_db))
        stored = conn.execute("SELECT result_id FROM runs").fetchone()[0]
        conn.close()

        assert stored == "abc12345"

    def test_two_runs_never_merge(self, memory_db):
        """Each generation gets its own run_id, so history stays per-run."""
        agent_reasoning.record_run([("k1", "A chill reason")])
        agent_reasoning.record_run([("k1", "A different chill reason")])

        assert count_rows(memory_db, "runs") == 2
        assert count_rows(memory_db, "selections") == 2

    def test_reason_text_is_trimmed_and_capped(self, memory_db):
        """Model prose cannot bloat the DB: stored reasons are bounded."""
        rambling = "x" * (agent_reasoning.MAX_REASON_CHARS + 400)
        agent_reasoning.record_run([("k1", f"  {rambling}  ")])
        conn = sqlite3.connect(str(memory_db))
        stored = conn.execute("SELECT reason FROM selections").fetchone()[0]
        conn.close()

        assert len(stored) == agent_reasoning.MAX_REASON_CHARS

    def test_tags_share_the_classifier_vocabulary(self, memory_db):
        """Tags are what the classifier already knows, so hints and tags compare."""
        tags = agent_reasoning._mine_tags("An 80s chill closer with ambient synth")

        assert "chill" in tags
        assert "ambient" in tags
        assert "1980s" in tags

    def test_a_reason_contributes_no_more_than_the_cap(self, memory_db):
        """Both caps hold: MAX_TAGS_PER_REASON tags, each MAX_TAG_CHARS long."""
        tags = agent_reasoning._mine_tags(
            "chill relax calm ambient classical new age jazz lo-fi downtempo chillout " + "z" * 200
        )

        assert len(tags) <= agent_reasoning.MAX_TAGS_PER_REASON
        assert all(len(tag) <= agent_reasoning.MAX_TAG_CHARS for tag in tags)

    def test_tag_spelling_is_normalized(self, memory_db):
        """Case and runs of whitespace collapse, so a reused tag has one spelling."""
        assert agent_reasoning._clean_tag("  Classic   ROCK  ") == "classic rock"
        assert agent_reasoning._clean_tag(None) == ""

    def test_a_reused_tag_costs_no_extra_row(self, memory_db):
        """PK (rating_key, tag) plus INSERT OR IGNORE: the vocabulary stays small."""
        reason = "An 80s chill closer"
        mined = agent_reasoning._mine_tags(reason)
        agent_reasoning.record_run([("k1", reason)])
        agent_reasoning.record_run([("k1", reason)])

        stored = {(row["rating_key"], row["tag"]) for row in read_rows(memory_db, "SELECT rating_key, tag FROM tags")}

        assert stored == {("k1", tag) for tag in mined}
        assert len(mined) == len(stored)


class TestReplay:
    """What a candidate line replays: the newest stored argument per track."""

    def test_only_the_asked_keys_answer(self, memory_db):
        """A track the memory never saw is absent, not an empty excuse."""
        agent_reasoning.record_run([("k1", "A chill reason")])

        assert agent_reasoning.get_previous_reasons(["k1", "k2"]) == {"k1": "A chill reason"}

    def test_repeated_and_blank_keys_are_tolerated(self, memory_db):
        """Candidate lists carry dupes and blanks; the IN-clause must not."""
        agent_reasoning.record_run([("k1", "A chill reason")])

        assert agent_reasoning.get_previous_reasons(["k1", "k1", "", None]) == {"k1": "A chill reason"}

    def test_the_newest_reason_wins(self, memory_db, monkeypatch):
        """A track curated again argues again; the older argument is not replayed."""
        agent_reasoning.record_run([("k1", "the newer argument")])

        class Older:
            def now(*args, **kwargs):
                return datetime(2020, 1, 1)

        monkeypatch.setattr(agent_reasoning, "datetime", Older)
        agent_reasoning.record_run([("k1", "the older argument")])

        assert agent_reasoning.get_previous_reasons(["k1"]) == {"k1": "the newer argument"}

    def test_the_in_clause_is_chunked(self, memory_db, monkeypatch):
        """Rating keys are asked QUERY_PARAM_CHUNK at a time, under SQLite's cap."""
        monkeypatch.setattr(agent_reasoning, "QUERY_PARAM_CHUNK", 2)
        agent_reasoning.record_run([(key, "An 80s chill reason") for key in ("k1", "k2", "k3", "k4", "k5")])

        found = agent_reasoning.get_previous_reasons(["k1", "k2", "k3", "k4", "k5"])

        assert set(found) == {"k1", "k2", "k3", "k4", "k5"}

    def test_the_line_is_cut_by_the_caller_not_here(self, memory_db):
        """Storage keeps MAX_REASON_CHARS whole; only generator trims to replay."""
        long_reason = "y" * agent_reasoning.MAX_REASON_CHARS
        agent_reasoning.record_run([("k1", long_reason)])

        assert len(agent_reasoning.get_previous_reasons(["k1"])["k1"]) == agent_reasoning.MAX_REASON_CHARS
        assert agent_reasoning.MAX_PAST_REASON_CHARS < agent_reasoning.MAX_REASON_CHARS


class TestTagQueries:
    """`get_reason_tags` feeds pre-ranking; `keys_with_tags` answers "where else?"."""

    def test_tags_are_a_set_per_key(self, memory_db):
        """One key collects its whole mined vocabulary, in a set."""
        agent_reasoning.record_run([("k1", "An 80s chill closer")])

        found = agent_reasoning.get_reason_tags(["k1"])

        assert found["k1"] >= {"chill", "1980s"}

    def test_unqueried_keys_stay_out_of_the_map(self, memory_db):
        """Scoring must not invent empty entries for tracks with no history."""
        agent_reasoning.record_run([("k1", "An 80s chill closer")])

        assert set(agent_reasoning.get_reason_tags(["k1", "k2"])) == {"k1"}

    def test_any_matches_loosely_and_all_needs_every_tag(self, memory_db):
        """"any" is "curated once in this language"; "all" is the same argument."""
        agent_reasoning.record_run([("k1", "an 80s chill closer"), ("k2", "a chill closer")])

        assert agent_reasoning.keys_with_tags(["1980s", "chill"]) == {"k1", "k2"}
        assert agent_reasoning.keys_with_tags(["1980s", "chill"], match="all") == {"k1"}

    def test_queried_tags_are_cleaned_before_the_query(self, memory_db):
        """The caller may say " AMBIENT " in any casing; the store is lowercase."""
        agent_reasoning.record_run([("k1", "An 80s Ambient closer")])

        assert agent_reasoning.keys_with_tags(["  AMBIENT  "]) == {"k1"}

    def test_no_tags_ask_nothing(self, memory_db):
        """An empty query must not become an unbounded scan."""
        agent_reasoning.record_run([("k1", "An 80s chill closer")])

        assert agent_reasoning.keys_with_tags([]) == set()
        assert agent_reasoning.keys_with_tags(["   "]) == set()

    def test_the_gate_reads_selections_not_tags(self, memory_db):
        """`has_reasoning_data()` gates replay and affinity, and starts off."""
        assert agent_reasoning.has_reasoning_data() is False

        agent_reasoning.record_run([("k1", "An 80s chill reason")])

        assert agent_reasoning.has_reasoning_data() is True


class TestStatusRead:
    """What `GET /api/reasoning/status` serializes."""

    def test_an_empty_memory_still_answers(self, memory_db):
        """A fresh install reports zeroes and no error, so the route never 500s."""
        payload = agent_reasoning.reasoning_status()

        assert payload["runs"] == 0
        assert payload["selections"] == 0
        assert payload["tags"] == []
        assert payload["last_run_at"] is None
        assert payload["error"] is None

    def test_the_counts_are_rows_not_runs(self, memory_db):
        """Runs, picks, distinct reasoned tracks and distinct tagged tracks."""
        agent_reasoning.record_run([("k1", "An 80s reason"), ("k2", "A chill reason")])
        payload = agent_reasoning.reasoning_status()

        assert payload["runs"] == 1
        assert payload["selections"] == 2
        assert payload["tracks_with_reasons"] == 2
        assert payload["tagged_tracks"] == 2
        assert payload["last_run_at"] is not None

    def test_the_vocabulary_ranks_by_reuse(self, memory_db):
        """A tag used across tracks outranks a one-off, and the list is ordered."""
        agent_reasoning.record_run([("k1", "an 80s chill closer"), ("k2", "a chill closer")])
        payload = agent_reasoning.reasoning_status()
        counts = {entry["tag"]: entry["count"] for entry in payload["tags"]}
        reported = [entry["count"] for entry in payload["tags"]]

        assert counts["chill"] == 2
        assert reported == sorted(reported, reverse=True)

    def test_the_status_cap_bounds_the_vocabulary(self, memory_db, monkeypatch):
        """MAX_STATUS_TAGS is a report cap, so the panel stays one screen tall."""
        monkeypatch.setattr(agent_reasoning, "MAX_STATUS_TAGS", 1)
        agent_reasoning.record_run([("k1", "an 80s chill jazz reason")])
        payload = agent_reasoning.reasoning_status()

        assert len(payload["tags"]) == 1

    def test_an_unreadable_memory_says_so(self, memory_db):
        """A broken DB is named, not a blank panel: the route has an `error` field."""
        conn = agent_reasoning.ensure_db_initialized()
        conn.execute("DROP TABLE runs")
        conn.close()

        payload = agent_reasoning.reasoning_status()

        assert "runs" in payload["error"]
        assert payload["runs"] == 0
