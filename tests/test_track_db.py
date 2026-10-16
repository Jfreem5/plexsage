"""Tests for the SQLite-backed local track database (backend.track_db)."""

import sqlite3
import time

import pytest

from backend.models import Track
from backend.track_db import TrackDB, _build_fts_match, _decade_to_range


def _sample_tracks() -> list[Track]:
    """A small library covering genres, decades, ratings, and a live track."""
    return [
        Track(
            rating_key="1",
            title="Fake Plastic Trees",
            artist="Radiohead",
            album="The Bends",
            duration_ms=290000,
            year=1995,
            genres=["Alternative", "Rock"],
        ),
        Track(
            rating_key="2",
            title="Black",
            artist="Pearl Jam",
            album="Ten",
            duration_ms=340000,
            year=1991,
            genres=["Grunge", "Rock"],
        ),
        Track(
            rating_key="3",
            title="So What",
            artist="Miles Davis",
            album="Kind of Blue",
            duration_ms=545000,
            year=1959,
            genres=["Jazz"],
        ),
        Track(
            rating_key="4",
            title="Champagne Supernova - Live",
            artist="Oasis",
            album="Live at Knebworth 1996",
            duration_ms=460000,
            year=1996,
            genres=["Britpop", "Rock"],
        ),
    ]


@pytest.fixture
def db(tmp_path) -> TrackDB:
    """A fresh, empty track database in a temp directory."""
    return TrackDB(tmp_path / "test.db")


@pytest.fixture
def seeded(db) -> TrackDB:
    """A database populated with the sample library."""
    db.upsert_tracks(
        _sample_tracks(),
        is_live_flags=[False, False, False, True],
        user_ratings=[9, 8, 10, 7],
        play_counts=[5, 2, 20, 0],
        last_played=[None, None, "2024-01-01T00:00:00Z", None],
    )
    return db


class TestUpsertAndMetadata:
    """Writing tracks and reading their metadata back."""

    def test_upsert_increments_count(self, seeded):
        assert seeded.get_track_count() == 4

    def test_get_metadata_returns_rating_and_genres(self, seeded):
        meta = seeded.get_metadata("1")
        assert meta["user_rating"] == 9
        assert meta["play_count"] == 5
        assert set(meta["genres"]) == {"Alternative", "Rock"}
        assert meta["is_live"] is False

    def test_get_metadata_missing_returns_none(self, seeded):
        assert seeded.get_metadata("does-not-exist") is None

    def test_get_metadata_batch(self, seeded):
        batch = seeded.get_metadata_batch(["1", "3"])
        assert set(batch) == {"1", "3"}
        assert batch["3"]["user_rating"] == 10

    def test_clear_removes_everything(self, seeded):
        seeded.clear()
        assert seeded.get_track_count() == 0
        assert seeded.get_library_stats() == {
            "total_tracks": 0,
            "genres": [],
            "decades": [],
        }

    def test_embedding_roundtrip(self, seeded):
        seeded.set_embedding("1", b"\x01\x02\x03")
        assert seeded.get_embeddings(["1"]) == {"1": b"\x01\x02\x03"}
        assert set(seeded.get_embeddings()) == {"1"}

    def test_upsert_preserves_embedding(self, seeded):
        """A re-sync must not wipe stored vectors."""
        seeded.set_embedding("1", b"\x01\x02\x03")
        seeded.upsert_tracks([_sample_tracks()[0]])
        assert seeded.get_embeddings(["1"]) == {"1": b"\x01\x02\x03"}


class TestFiltering:
    """SQL filtering by genre, decade, rating, and live status."""

    def test_filter_by_genre(self, seeded):
        results = seeded.get_tracks_by_filters(genres=["Jazz"])
        assert [t.title for t in results] == ["So What"]

    def test_filter_by_decade(self, seeded):
        results = seeded.get_tracks_by_filters(decades=["1990s"], exclude_live=False)
        assert {t.title for t in results} == {
            "Fake Plastic Trees",
            "Black",
            "Champagne Supernova - Live",
        }

    def test_filter_by_min_rating(self, seeded):
        results = seeded.get_tracks_by_filters(min_rating=9)
        assert {t.title for t in results} == {"Fake Plastic Trees", "So What"}

    def test_exclude_live_default(self, seeded):
        results = seeded.get_tracks_by_filters(genres=["Rock"])
        assert "Champagne Supernova - Live" not in {t.title for t in results}

    def test_include_live(self, seeded):
        results = seeded.get_tracks_by_filters(genres=["Rock"], exclude_live=False)
        assert "Champagne Supernova - Live" in {t.title for t in results}

    def test_combined_filters(self, seeded):
        results = seeded.get_tracks_by_filters(
            genres=["Rock"], decades=["1990s"], min_rating=9, exclude_live=True
        )
        assert [t.title for t in results] == ["Fake Plastic Trees"]

    def test_limit(self, seeded):
        results = seeded.get_tracks_by_filters(limit=1)
        assert len(results) == 1

    def test_no_filters_excludes_live_by_default(self, seeded):
        assert len(seeded.get_tracks_by_filters()) == 3
        assert len(seeded.get_tracks_by_filters(exclude_live=False)) == 4

    def test_count_matches_filter(self, seeded):
        assert seeded.count_tracks_by_filters(genres=["Rock"]) == 2
        assert seeded.count_tracks_by_filters(genres=["Rock"], exclude_live=False) == 3

    def test_results_include_art_url(self, seeded):
        track = seeded.get_tracks_by_filters(genres=["Jazz"])[0]
        assert track.art_url == "/api/art/3"

    def test_random_tracks_respects_live_filter(self, seeded):
        titles = {t.title for t in seeded.get_random_tracks(10, exclude_live=True)}
        assert "Champagne Supernova - Live" not in titles


class TestSearch:
    """Full-text search over title, artist, and album."""

    def test_search_by_title_prefix(self, seeded):
        assert [t.title for t in seeded.search_tracks("plast")] == ["Fake Plastic Trees"]

    def test_search_by_artist(self, seeded):
        titles = {t.title for t in seeded.search_tracks("radio")}
        assert titles == {"Fake Plastic Trees"}

    def test_search_by_album(self, seeded):
        titles = {t.title for t in seeded.search_tracks("kind of blue")}
        assert titles == {"So What"}

    def test_search_no_match(self, seeded):
        assert seeded.search_tracks("zzzzzz") == []

    def test_search_empty_query(self, seeded):
        assert seeded.search_tracks("") == []

    def test_search_respects_limit(self, seeded):
        assert len(seeded.search_tracks("rock", limit=1)) <= 1


class TestBulkFts:
    """Bulk upserts rewrite the FTS index without duplicating or losing rows."""

    @staticmethod
    def _fts_count(db: TrackDB) -> int:
        return db._get_conn().execute("SELECT COUNT(*) FROM tracks_fts").fetchone()[0]

    @pytest.fixture(autouse=True)
    def _require_fts(self, db):
        if not db._fts_enabled:
            pytest.skip("FTS5 not available in this SQLite build")

    def test_bulk_upsert_indexes_every_track(self, db):
        tracks = [
            Track(rating_key=str(i), title=f"Song {i}", artist=f"Artist {i}",
                  album=f"Album {i}", duration_ms=1000, year=2000, genres=["Rock"])
            for i in range(200)
        ]
        db.upsert_tracks(tracks)

        assert len(db.search_tracks("Song 137")) == 1
        assert self._fts_count(db) == 200

    def test_bulk_upsert_updates_stale_fts_entry(self, db):
        old = Track(rating_key="1", title="Old Title", artist="A", album="X",
                    duration_ms=1, year=2000, genres=["Rock"])
        db.upsert_tracks([old])

        new = old.model_copy(update={"title": "Brand New Title"})
        db.upsert_tracks([new])

        assert db.search_tracks("Old") == []
        assert [t.title for t in db.search_tracks("Brand")] == ["Brand New Title"]

    def test_bulk_upsert_does_not_duplicate_fts_rows(self, db):
        tracks = [
            Track(rating_key=str(i), title=f"Song {i}", artist=f"Artist {i}",
                  album="A", duration_ms=1, year=2000, genres=["Rock"])
            for i in range(50)
        ]
        db.upsert_tracks(tracks)
        db.upsert_tracks(tracks)  # re-sync the same library

        assert self._fts_count(db) == db.get_track_count()


class TestLibraryStats:
    """Aggregate stats computed in SQL."""

    def test_total_tracks(self, seeded):
        assert seeded.get_library_stats()["total_tracks"] == 4

    def test_genre_counts(self, seeded):
        genres = {g["name"]: g["count"] for g in seeded.get_library_stats()["genres"]}
        assert genres["Rock"] == 3
        assert genres["Jazz"] == 1
        assert genres["Alternative"] == 1

    def test_decade_counts(self, seeded):
        decades = {d["name"]: d["count"] for d in seeded.get_library_stats()["decades"]}
        assert decades["1990s"] == 3
        assert decades["1950s"] == 1


class TestSyncMetadata:
    """Sync freshness tracking."""

    def test_needs_sync_when_empty(self, db):
        assert db.needs_sync() is True

    def test_needs_sync_without_timestamp(self, seeded):
        assert seeded.needs_sync() is True

    def test_fresh_after_last_sync(self, seeded):
        seeded.set_last_sync()
        assert seeded.needs_sync() is False

    def test_stale_after_interval(self, seeded):
        seeded.set_last_sync(time.time() - 100)
        assert seeded.needs_sync(max_age_seconds=10) is True

    def test_get_last_sync_default_none(self, db):
        assert db.get_last_sync() is None


class TestPrune:
    """Removing tracks that no longer exist in Plex."""

    def test_prune_removes_missing_tracks(self, seeded):
        removed = seeded.prune_except(["1", "3"])
        assert removed == 2
        assert seeded.get_track_count() == 2
        assert seeded.get_metadata("2") is None

    def test_prune_keeps_everything_when_unchanged(self, seeded):
        removed = seeded.prune_except(["1", "2", "3", "4"])
        assert removed == 0
        assert seeded.get_track_count() == 4

    def test_prune_with_empty_input_is_noop(self, seeded):
        assert seeded.prune_except([]) == 0
        assert seeded.get_track_count() == 4


class TestSyncFromPlex:
    """sync_from_plex must not clobber existing rating/play data."""

    def test_sync_preserves_existing_metadata(self, seeded):
        # Track "1" already has rating 9 / plays 5 from the fixture.
        seeded.sync_from_plex([_sample_tracks()[0]])
        meta = seeded.get_metadata("1")
        assert meta["user_rating"] == 9
        assert meta["play_count"] == 5

    def test_sync_inserts_new_tracks_with_defaults(self, db):
        db.sync_from_plex([_sample_tracks()[0]])
        meta = db.get_metadata("1")
        assert meta["user_rating"] == 0
        assert meta["play_count"] == 0
        assert db.get_track_count() == 1


class TestTrackDimensions:
    """Caching of LLM-generated seed-track dimensions."""

    def test_save_and_get(self, db):
        dims = [{"id": "mood", "label": "Melancholy", "description": "Introspective"}]
        db.save_dimensions("1", dims, model="model-a", fingerprint="fp")
        assert db.get_dimensions("1", model="model-a", fingerprint="fp") == dims

    def test_get_missing_returns_none(self, db):
        assert db.get_dimensions("nope") is None

    def test_model_change_invalidates(self, db):
        db.save_dimensions("1", [{"id": "mood"}], model="model-a", fingerprint="fp")
        assert db.get_dimensions("1", model="model-b", fingerprint="fp") is None
        # The stored row is untouched; a matching lookup still succeeds.
        assert db.get_dimensions("1", model="model-a", fingerprint="fp") == [{"id": "mood"}]

    def test_fingerprint_change_invalidates(self, db):
        db.save_dimensions("1", [{"id": "mood"}], model="model-a", fingerprint="fp1")
        assert db.get_dimensions("1", model="model-a", fingerprint="fp2") is None

    def test_lookup_without_checks_returns_entry(self, db):
        db.save_dimensions("1", [{"id": "mood"}], model="model-a", fingerprint="fp")
        assert db.get_dimensions("1") == [{"id": "mood"}]

    def test_save_overwrites_existing(self, db):
        db.save_dimensions("1", [{"id": "old"}], model="model-a")
        db.save_dimensions("1", [{"id": "new"}], model="model-b")
        assert db.get_dimensions("1")[0]["id"] == "new"
        assert db.get_dimension_count() == 1

    def test_delete_dimensions(self, db):
        db.save_dimensions("1", [{"id": "mood"}], model="model-a")
        db.delete_dimensions("1")
        assert db.get_dimensions("1") is None

    def test_clear_dimensions(self, db):
        db.save_dimensions("1", [{"id": "mood"}], model="model-a")
        db.save_dimensions("2", [{"id": "era"}], model="model-a")
        db.clear_dimensions()
        assert db.get_dimension_count() == 0

    def test_corrupt_json_is_ignored(self, db):
        db.save_dimensions("1", [{"id": "mood"}], model="model-a")
        conn = db._get_conn()
        conn.execute(
            "UPDATE track_dimensions SET dimensions = ? WHERE rating_key = ?",
            ("not valid json", "1"),
        )
        conn.commit()
        assert db.get_dimensions("1") is None

    def test_prune_removes_dimensions_for_removed_tracks(self, seeded):
        seeded.save_dimensions("1", [{"id": "mood"}], model="model-a")
        seeded.save_dimensions("2", [{"id": "era"}], model="model-a")
        seeded.prune_except(["1"])
        assert seeded.get_dimensions("1") is not None
        assert seeded.get_dimensions("2") is None
        assert seeded.get_dimension_count() == 1

    def test_clear_removes_dimensions(self, seeded):
        seeded.save_dimensions("1", [{"id": "mood"}], model="model-a")
        seeded.clear()
        assert seeded.get_dimension_count() == 0


class TestLiveDetection:
    """is_live is pre-computed at sync time and read back without regex."""

    def test_flag_roundtrips_through_filters(self, seeded):
        by_key = {
            t.rating_key: t
            for t in seeded.get_tracks_by_filters(exclude_live=False)
        }
        assert by_key["4"].is_live is True
        assert by_key["1"].is_live is False

    def test_search_carries_live_flag(self, seeded):
        assert seeded.search_tracks("champagne")[0].is_live is True

    def test_random_carries_live_flag(self, seeded):
        by_key = {
            t.rating_key: t for t in seeded.get_random_tracks(10, exclude_live=False)
        }
        assert by_key["4"].is_live is True

    def test_get_live_flags_batch(self, seeded):
        assert seeded.get_live_flags(["1", "4", "missing"]) == {"1": False, "4": True}

    def test_get_live_flags_empty(self, seeded):
        assert seeded.get_live_flags([]) == {}

    def test_defaults_to_not_live(self, db):
        """Tracks upserted without flags are treated as non-live."""
        db.upsert_tracks([_sample_tracks()[0]])
        assert db.get_tracks_by_filters()[0].is_live is False


class TestMigrations:
    """Older DB files are upgraded in place."""

    def test_legacy_db_gains_is_live_column(self, tmp_path):
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.execute(
            """CREATE TABLE tracks (
                   rating_key  TEXT PRIMARY KEY,
                   title       TEXT NOT NULL,
                   artist      TEXT NOT NULL,
                   album       TEXT NOT NULL DEFAULT '',
                   year        INTEGER,
                   duration_ms INTEGER NOT NULL DEFAULT 0,
                   user_rating INTEGER NOT NULL DEFAULT 0,
                   play_count  INTEGER NOT NULL DEFAULT 0,
                   last_played TEXT,
                   updated_at  TEXT NOT NULL
               )"""
        )
        conn.commit()
        conn.close()

        db = TrackDB(path)
        columns = {row[1] for row in db._get_conn().execute("PRAGMA table_info(tracks)")}
        assert {"is_live", "embedding"} <= columns

        # And the migrated column is usable end-to-end.
        db.upsert_tracks([_sample_tracks()[0]], is_live_flags=[False])
        assert db.get_track_count() == 1
        assert db.get_tracks_by_filters()[0].is_live is False


class TestHelpers:
    """Module-level parsing helpers."""

    @pytest.mark.parametrize(
        "decade,expected",
        [
            ("1990s", (1990, 1999)),
            ("90s", (1990, 1999)),
            ("2000s", (2000, 2009)),
            ("10s", (2010, 2019)),
            ("1980", (1980, 1989)),
            ("nonsense", (None, None)),
        ],
    )
    def test_decade_to_range(self, decade, expected):
        assert _decade_to_range(decade) == expected

    def test_build_fts_match_prefixes_tokens(self):
        assert _build_fts_match("Fake Plastic") == '"Fake"* "Plastic"*'

    def test_build_fts_match_strips_specials(self):
        assert _build_fts_match("rock's roll!") == '"rock"* "s"* "roll"*'
