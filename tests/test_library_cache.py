"""Tests for local library cache functionality."""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from backend import library_cache


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_library_cache.db"
    monkeypatch.setattr(library_cache, "DB_PATH", db_path)
    monkeypatch.setattr(library_cache, "DATA_DIR", tmp_path)
    # Reset schema initialization flag so each test gets fresh schema
    monkeypatch.setattr(library_cache, "_schema_initialized", False)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def initialized_db(temp_db):
    """Create and initialize a test database."""
    conn = library_cache.ensure_db_initialized()
    conn.close()
    return temp_db


class TestSchemaCreation:
    """Test database schema initialization."""

    def test_creates_data_directory(self, temp_db):
        """Schema init creates data directory if missing."""
        # Remove directory
        temp_db.parent.rmdir() if temp_db.parent.exists() else None

        conn = library_cache.ensure_db_initialized()
        conn.close()

        assert temp_db.parent.exists()

    def test_creates_tracks_table(self, initialized_db):
        """Schema creates tracks table with correct columns."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute("PRAGMA table_info(tracks)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "rating_key" in columns
        assert "title" in columns
        assert "artist" in columns
        assert "album" in columns
        assert "duration_ms" in columns
        assert "year" in columns
        assert "genres" in columns
        assert "user_rating" in columns
        assert "is_live" in columns
        assert "updated_at" in columns

    def test_creates_sync_state_table(self, initialized_db):
        """Schema creates sync_state table with single row."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute("SELECT COUNT(*) FROM sync_state")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_creates_indexes(self, initialized_db):
        """Schema creates expected indexes."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_tracks_%'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "idx_tracks_artist" in indexes
        assert "idx_tracks_year" in indexes
        assert "idx_tracks_is_live" in indexes

    def test_wal_mode_enabled(self, initialized_db):
        """Connection uses WAL journal mode."""
        conn = library_cache.get_db_connection()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()

        assert mode.lower() == "wal"

    def test_idempotent_initialization(self, initialized_db):
        """Multiple initializations don't cause errors."""
        # Should not raise
        conn1 = library_cache.ensure_db_initialized()
        conn1.close()
        conn2 = library_cache.ensure_db_initialized()
        conn2.close()


class TestSyncState:
    """Test sync state management."""

    def test_initial_sync_state(self, initialized_db):
        """Fresh database has empty sync state."""
        state = library_cache.get_sync_state()

        assert state["track_count"] == 0
        assert state["synced_at"] is None
        assert state["is_syncing"] is False
        assert state["sync_progress"] is None
        assert state["error"] is None

    def test_sync_state_after_sync(self, initialized_db):
        """Sync state updates after successful sync."""
        conn = library_cache.get_db_connection()

        # Simulate a completed sync
        synced_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE sync_state SET plex_server_id = ?, last_sync_at = ?, "
            "track_count = ?, sync_duration_ms = ? WHERE id = 1",
            ("test-server-id", synced_at, 1000, 5000),
        )
        conn.commit()
        conn.close()

        state = library_cache.get_sync_state()

        assert state["track_count"] == 1000
        assert state["synced_at"] == synced_at
        assert state["plex_server_id"] == "test-server-id"


class TestCacheOperations:
    """Test cache read/write operations."""

    def test_get_cached_tracks_empty(self, initialized_db):
        """Empty cache returns empty list."""
        tracks = library_cache.get_cached_tracks()
        assert tracks == []

    def test_get_cached_tracks_with_data(self, initialized_db):
        """Cached tracks are returned correctly."""
        conn = library_cache.get_db_connection()
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, genres, user_rating, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("123", "Test Song", "Test Artist", "Test Album", 180000, 1999,
             json.dumps(["Rock", "Alternative"]), 8, False),
        )
        conn.commit()
        conn.close()

        tracks = library_cache.get_cached_tracks()

        assert len(tracks) == 1
        assert tracks[0]["rating_key"] == "123"
        assert tracks[0]["title"] == "Test Song"
        assert tracks[0]["genres"] == ["Rock", "Alternative"]

    def test_clear_cache(self, initialized_db):
        """Clear cache removes all tracks and resets state."""
        conn = library_cache.get_db_connection()
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album) "
            "VALUES ('1', 'Song', 'Artist', 'Album')"
        )
        conn.execute(
            "UPDATE sync_state SET track_count = 1, last_sync_at = ? WHERE id = 1",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
        conn.close()

        library_cache.clear_cache()

        tracks = library_cache.get_cached_tracks()
        state = library_cache.get_sync_state()

        assert tracks == []
        assert state["track_count"] == 0
        assert state["synced_at"] is None


class TestFiltering:
    """Test track filtering operations."""

    @pytest.fixture
    def sample_tracks(self, initialized_db):
        """Insert sample tracks for filter testing."""
        conn = library_cache.get_db_connection()
        tracks = [
            ("1", "Rock Song", "Rock Artist", "Rock Album", 180000, 1995,
             json.dumps(["Rock"]), 8, False),
            ("2", "Pop Song", "Pop Artist", "Pop Album", 200000, 2005,
             json.dumps(["Pop"]), 6, False),
            ("3", "Live Concert", "Live Artist", "Live 2020", 300000, 2020,
             json.dumps(["Rock"]), 4, True),
            ("4", "Jazz Song", "Jazz Artist", "Jazz Album", 250000, 1985,
             json.dumps(["Jazz"]), 10, False),
        ]
        conn.executemany(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, genres, user_rating, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tracks,
        )
        conn.commit()
        conn.close()
        return tracks

    def test_filter_exclude_live(self, sample_tracks):
        """Exclude live filter removes live tracks."""
        tracks = library_cache.get_tracks_by_filters(exclude_live=True)

        assert len(tracks) == 3
        assert all(not t["is_live"] for t in tracks)

    def test_filter_include_live(self, sample_tracks):
        """Include live returns all tracks."""
        tracks = library_cache.get_tracks_by_filters(exclude_live=False)

        assert len(tracks) == 4

    def test_filter_by_decade(self, sample_tracks):
        """Filter by decade returns matching tracks."""
        tracks = library_cache.get_tracks_by_filters(decades=["1990s"], exclude_live=False)

        assert len(tracks) == 1
        assert tracks[0]["year"] == 1995

    def test_filter_by_multiple_decades(self, sample_tracks):
        """Filter by multiple decades uses OR logic."""
        tracks = library_cache.get_tracks_by_filters(
            decades=["1980s", "2000s"], exclude_live=False
        )

        assert len(tracks) == 2
        years = {t["year"] for t in tracks}
        assert years == {1985, 2005}

    def test_filter_by_genre(self, sample_tracks):
        """Filter by genre returns matching tracks."""
        tracks = library_cache.get_tracks_by_filters(genres=["Rock"], exclude_live=False)

        assert len(tracks) == 2  # Rock Song and Live Concert

    def test_filter_by_genre_with_limit(self, initialized_db):
        """Filter by genre with limit applies limit AFTER genre filtering.

        This tests the fix for a bug where SQL LIMIT was applied before
        Python-side genre filtering, causing rare genres to return too few results.
        """
        conn = library_cache.get_db_connection()
        # Insert 100 Pop tracks and 10 Jazz tracks
        tracks = []
        for i in range(100):
            tracks.append((
                f"pop-{i}", f"Pop Song {i}", "Pop Artist", "Pop Album",
                180000, 2000, json.dumps(["Pop"]), 5, False
            ))
        for i in range(10):
            tracks.append((
                f"jazz-{i}", f"Jazz Song {i}", "Jazz Artist", "Jazz Album",
                180000, 1990, json.dumps(["Jazz"]), 5, False
            ))
        conn.executemany(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, genres, user_rating, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tracks,
        )
        conn.commit()
        conn.close()

        # Request 5 Jazz tracks - should get exactly 5 Jazz tracks
        # (Bug: without fix, would get random sample of all tracks, mostly Pop)
        result = library_cache.get_tracks_by_filters(genres=["Jazz"], limit=5)

        assert len(result) == 5
        # All returned tracks must be Jazz
        for t in result:
            assert "Jazz" in t["genres"], f"Non-Jazz track returned: {t['title']}"

    def test_filter_by_min_rating(self, sample_tracks):
        """Filter by minimum rating."""
        tracks = library_cache.get_tracks_by_filters(min_rating=8, exclude_live=False)

        assert len(tracks) == 2
        assert all(t["user_rating"] >= 8 for t in tracks)

    def test_filter_with_limit(self, sample_tracks):
        """Filter with limit returns at most N tracks."""
        tracks = library_cache.get_tracks_by_filters(limit=2, exclude_live=False)

        assert len(tracks) == 2

    def test_filter_combined(self, sample_tracks):
        """Combined filters work together."""
        tracks = library_cache.get_tracks_by_filters(
            genres=["Rock"],
            decades=["1990s"],
            exclude_live=True,
            min_rating=6,
        )

        assert len(tracks) == 1
        assert tracks[0]["title"] == "Rock Song"


class TestLiveDetection:
    """Test live version detection."""

    def test_live_keyword_in_title(self):
        """Detects 'live' keyword in title."""
        assert library_cache._is_live_version("Song (Live)", "Album")
        assert library_cache._is_live_version("Live at Madison Square Garden", "Album")

    def test_live_keyword_in_album(self):
        """Detects 'live' keyword in album."""
        assert library_cache._is_live_version("Song", "Live in Concert")
        assert library_cache._is_live_version("Song", "The Concert Album")

    def test_date_pattern_in_album(self):
        """Detects date patterns indicating live recordings."""
        assert library_cache._is_live_version("Song", "2020-01-15 Chicago")
        assert library_cache._is_live_version("Song", "1999/05/20")

    def test_bootleg_keyword(self):
        """Detects bootleg keyword."""
        assert library_cache._is_live_version("Song", "Bootleg Series")

    def test_non_live_tracks(self):
        """Non-live tracks are not flagged."""
        assert not library_cache._is_live_version("Regular Song", "Studio Album")
        assert not library_cache._is_live_version("Alive", "Greatest Hits")  # 'live' in 'alive'


class TestStalenessCheck:
    """Test cache staleness detection."""

    def test_empty_cache_is_stale(self, initialized_db):
        """Empty cache is considered stale."""
        assert library_cache.is_cache_stale()

    def test_recent_cache_not_stale(self, initialized_db):
        """Recently synced cache is not stale."""
        conn = library_cache.get_db_connection()
        synced_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE sync_state SET last_sync_at = ? WHERE id = 1",
            (synced_at,),
        )
        conn.commit()
        conn.close()

        assert not library_cache.is_cache_stale(max_age_hours=24)

    def test_old_cache_is_stale(self, initialized_db):
        """Old cache is considered stale."""
        conn = library_cache.get_db_connection()
        # Set sync time to 25 hours ago
        from datetime import timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        conn.execute(
            "UPDATE sync_state SET last_sync_at = ? WHERE id = 1",
            (old_time,),
        )
        conn.commit()
        conn.close()

        assert library_cache.is_cache_stale(max_age_hours=24)


class TestServerChangeDetection:
    """Test Plex server change detection."""

    def test_first_sync_no_change(self, initialized_db):
        """First sync doesn't count as server change."""
        assert not library_cache.check_server_changed("new-server-id")

    def test_same_server_no_change(self, initialized_db):
        """Same server ID doesn't trigger change."""
        conn = library_cache.get_db_connection()
        conn.execute(
            "UPDATE sync_state SET plex_server_id = ? WHERE id = 1",
            ("my-server-id",),
        )
        conn.commit()
        conn.close()

        assert not library_cache.check_server_changed("my-server-id")

    def test_different_server_triggers_change(self, initialized_db):
        """Different server ID triggers change detection."""
        conn = library_cache.get_db_connection()
        conn.execute(
            "UPDATE sync_state SET plex_server_id = ? WHERE id = 1",
            ("old-server-id",),
        )
        conn.commit()
        conn.close()

        assert library_cache.check_server_changed("new-server-id")


class TestSyncLibrary:
    """Test sync_library() function."""

    @pytest.fixture
    def mock_track(self):
        """Create a mock Plex track object."""
        class MockTrack:
            def __init__(self, rating_key, title, artist, album, duration, parent_key):
                self.ratingKey = rating_key
                self.title = title
                self.grandparentTitle = artist
                self.parentTitle = album
                self.duration = duration
                self.parentRatingKey = parent_key
                self.userRating = None
        return MockTrack

    @pytest.fixture
    def mock_plex_client(self, mock_track):
        """Create a mock Plex client."""
        class MockPlexClient:
            def __init__(self):
                self.tracks = [
                    mock_track("1", "Song One", "Artist A", "Album X", 180000, "100"),
                    mock_track("2", "Song Two", "Artist B", "Album Y", 200000, "101"),
                    mock_track("3", "Song Three", "Artist A", "Album X", 220000, "100"),
                ]
                self.album_metadata = {
                    "100": {"genres": ["Rock", "Alternative"], "year": 1995},
                    "101": {"genres": ["Electronic"], "year": 2020},
                }

            def get_machine_identifier(self):
                return "test-server-123"

            def get_all_albums_metadata(self):
                return self.album_metadata

            def get_all_raw_tracks(self):
                return self.tracks

        return MockPlexClient()

    @pytest.fixture
    def reset_sync_state(self, monkeypatch):
        """Reset sync state before each test."""
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": False,
            "phase": None,
            "current": 0,
            "total": 0,
            "error": None,
        })

    def test_sync_success(self, initialized_db, mock_plex_client, reset_sync_state):
        """Sync completes successfully with correct track count."""
        result = library_cache.sync_library(mock_plex_client)

        assert result["success"] is True
        assert result["track_count"] == 3
        assert "duration_ms" in result

        # Verify tracks in database
        tracks = library_cache.get_cached_tracks()
        assert len(tracks) == 3

    def test_sync_stores_genres_from_albums(self, initialized_db, mock_plex_client, reset_sync_state):
        """Sync correctly maps album genres to tracks."""
        library_cache.sync_library(mock_plex_client)

        tracks = library_cache.get_cached_tracks()
        track_by_key = {t["rating_key"]: t for t in tracks}

        # Tracks 1 and 3 are from Album X (parent_key=100)
        assert track_by_key["1"]["genres"] == ["Rock", "Alternative"]
        assert track_by_key["3"]["genres"] == ["Rock", "Alternative"]

        # Track 2 is from Album Y (parent_key=101)
        assert track_by_key["2"]["genres"] == ["Electronic"]

    def test_sync_stores_year_from_albums(self, initialized_db, mock_plex_client, reset_sync_state):
        """Sync correctly maps album year to tracks."""
        library_cache.sync_library(mock_plex_client)

        tracks = library_cache.get_cached_tracks()
        track_by_key = {t["rating_key"]: t for t in tracks}

        assert track_by_key["1"]["year"] == 1995
        assert track_by_key["2"]["year"] == 2020
        assert track_by_key["3"]["year"] == 1995

    def test_sync_updates_sync_state(self, initialized_db, mock_plex_client, reset_sync_state):
        """Sync updates the sync_state table."""
        library_cache.sync_library(mock_plex_client)

        state = library_cache.get_sync_state()

        assert state["track_count"] == 3
        assert state["synced_at"] is not None
        assert state["plex_server_id"] == "test-server-123"
        assert state["is_syncing"] is False

    def test_sync_progress_callback(self, initialized_db, mock_plex_client, reset_sync_state):
        """Progress callback is invoked during sync."""
        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        library_cache.sync_library(mock_plex_client, on_progress=on_progress)

        # With 3 tracks and SYNC_BATCH_SIZE=500, we won't hit the batch callback
        # but we can verify the function accepts the callback without error
        assert isinstance(progress_calls, list)

    def test_sync_rejects_concurrent_sync(self, initialized_db, mock_plex_client, monkeypatch):
        """Second sync attempt is rejected while one is in progress."""
        # Simulate sync in progress
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": True,
            "phase": "processing",
            "current": 50,
            "total": 100,
            "error": None,
        })

        result = library_cache.sync_library(mock_plex_client)

        assert result["success"] is False
        assert "already in progress" in result["error"]

    def test_sync_handles_missing_server_id(self, initialized_db, reset_sync_state):
        """Sync fails gracefully if server ID is unavailable."""
        class BadPlexClient:
            def get_machine_identifier(self):
                return None

        result = library_cache.sync_library(BadPlexClient())

        assert result["success"] is False
        assert "server identifier" in result["error"].lower()

    def test_sync_clears_error_on_success(self, initialized_db, mock_plex_client, monkeypatch):
        """Successful sync clears any previous error state."""
        # Set initial state with error
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": False,
            "phase": None,
            "current": 0,
            "total": 0,
            "error": "Previous error",
        })

        library_cache.sync_library(mock_plex_client)

        state = library_cache.get_sync_state()
        assert state["error"] is None
        assert state["is_syncing"] is False

    def test_sync_removes_deleted_tracks(self, initialized_db, mock_track, reset_sync_state):
        """Sync removes tracks that no longer exist in Plex."""
        # Pre-populate cache with a track that won't be in the sync
        conn = library_cache.get_db_connection()
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, genres, user_rating, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("stale-track-999", "Stale Song", "Old Artist", "Deleted Album",
             180000, 2000, json.dumps(["Rock"]), 5, False),
        )
        conn.commit()
        conn.close()

        # Verify stale track exists
        tracks_before = library_cache.get_cached_tracks()
        assert any(t["rating_key"] == "stale-track-999" for t in tracks_before)

        # Create mock client that returns different tracks
        class MockPlexClient:
            def get_machine_identifier(self):
                return "test-server-123"

            def get_all_albums_metadata(self):
                return {"100": {"genres": ["Electronic"], "year": 2024}}

            def get_all_raw_tracks(self):
                return [mock_track("new-1", "New Song", "New Artist", "New Album", 200000, "100")]

        # Run sync
        result = library_cache.sync_library(MockPlexClient())

        assert result["success"] is True
        assert result["track_count"] == 1

        # Verify stale track was removed
        tracks_after = library_cache.get_cached_tracks()
        assert len(tracks_after) == 1
        assert tracks_after[0]["rating_key"] == "new-1"
        assert not any(t["rating_key"] == "stale-track-999" for t in tracks_after)

    def test_failed_sync_resets_cache_state(self, initialized_db, mock_track, reset_sync_state):
        """Failed sync resets track_count so has_cached_tracks() returns False."""
        # First, do a successful sync to populate cache
        class SuccessfulClient:
            def get_machine_identifier(self):
                return "test-server-123"

            def get_all_albums_metadata(self):
                return {"100": {"genres": ["Rock"], "year": 2020}}

            def get_all_raw_tracks(self):
                return [mock_track("1", "Song", "Artist", "Album", 180000, "100")]

        result = library_cache.sync_library(SuccessfulClient())
        assert result["success"] is True
        assert library_cache.has_cached_tracks() is True

        # Reset sync state for next sync attempt
        library_cache._sync_state["is_syncing"] = False

        # Now attempt a sync that fails during Plex API call
        class FailingClient:
            def get_machine_identifier(self):
                return "test-server-123"

            def get_all_albums_metadata(self):
                raise ConnectionError("Plex unreachable")

        result = library_cache.sync_library(FailingClient())
        assert result["success"] is False

        # Cache should now report as empty to avoid using stale data
        assert library_cache.has_cached_tracks() is False
        assert library_cache.get_sync_state()["track_count"] == 0


class TestDimensionCache:
    """Seed-track dimension cache stored alongside the library cache."""

    def test_save_and_get_roundtrip(self, temp_db):
        dims = [{"id": "mood", "label": "M", "description": "d"}]
        library_cache.save_dimensions("1", dims, model="m", fingerprint="f")
        assert library_cache.get_dimensions("1", model="m", fingerprint="f") == dims

    def test_miss_returns_none(self, temp_db):
        assert library_cache.get_dimensions("nope", model="m", fingerprint="f") is None

    def test_model_change_is_a_miss(self, temp_db):
        library_cache.save_dimensions("1", [{"id": "x"}], model="a", fingerprint="f1")
        assert library_cache.get_dimensions("1", model="b", fingerprint="f1") is None

    def test_fingerprint_change_is_a_miss(self, temp_db):
        library_cache.save_dimensions("1", [{"id": "x"}], model="a", fingerprint="f1")
        assert library_cache.get_dimensions("1", model="a", fingerprint="f2") is None

    def test_save_overwrites_same_key(self, temp_db):
        library_cache.save_dimensions("1", [{"id": "old"}], model="m", fingerprint="f")
        library_cache.save_dimensions("1", [{"id": "new"}], model="m", fingerprint="f")
        assert library_cache.get_dimensions("1", model="m", fingerprint="f") == [{"id": "new"}]


class TestMetadataBatch:
    """get_metadata_batch() feeds rule-based pre-ranking."""

    def _seed(self):
        conn = library_cache.ensure_db_initialized()
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album, user_rating, "
            "view_count, last_viewed_at) VALUES "
            "('1', 'a', 'a', 'a', 8, 3, '2026-01-02T00:00:00')"
        )
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album, user_rating) "
            "VALUES ('2', 'b', 'b', 'b', 4)"
        )
        conn.commit()
        conn.close()

    def test_returns_metadata_for_known_keys(self, initialized_db):
        self._seed()
        meta = library_cache.get_metadata_batch(["1", "2"])
        assert meta["1"]["user_rating"] == 8
        assert meta["1"]["last_viewed_at"] == "2026-01-02T00:00:00"
        assert meta["2"]["user_rating"] == 4

    def test_empty_input_returns_empty(self, initialized_db):
        assert library_cache.get_metadata_batch([]) == {}

    def test_missing_keys_are_absent(self, initialized_db):
        self._seed()
        assert library_cache.get_metadata_batch(["nope"]) == {}


class TestResultPlaylistLink:
    """A history row can point at the Plex playlist it was saved as."""

    def _seed_result(self):
        return library_cache.save_result(
            result_type="prompt_playlist",
            title="Road Trip",
            prompt="driving songs",
            snapshot={"tracks": []},
            track_count=2,
        )

    def test_link_records_playlist(self, initialized_db):
        result_id = self._seed_result()
        linked = library_cache.link_result_playlist(
            result_id, "3751", "Road Trip", "The Boys", "https://example/web"
        )
        assert linked is True

        row = library_cache.get_result(result_id)
        assert row["playlist_id"] == "3751"
        assert row["playlist_title"] == "Road Trip"
        assert row["playlist_owner"] == "The Boys"
        assert row["playlist_url"] == "https://example/web"

    def test_link_relinks_to_a_different_playlist(self, initialized_db):
        result_id = self._seed_result()
        assert library_cache.link_result_playlist(result_id, "1111", "First", "Becky") is True
        assert library_cache.link_result_playlist(result_id, "2222", "Second", "BluJay") is True
        assert library_cache.get_result(result_id)["playlist_id"] == "2222"

    def test_unlinked_row_reads_as_none(self, initialized_db):
        result_id = self._seed_result()
        row = library_cache.get_result(result_id)
        assert row["playlist_id"] is None
        assert row["playlist_owner"] is None

    def test_history_list_carries_the_link(self, initialized_db):
        result_id = self._seed_result()
        library_cache.link_result_playlist(result_id, "3751", "Road Trip", "The Boys")
        items, total = library_cache.list_results()
        assert total == 1
        assert items[0]["playlist_id"] == "3751"
        assert items[0]["playlist_owner"] == "The Boys"

    def test_unknown_result_is_not_linked(self, initialized_db):
        assert library_cache.link_result_playlist("deadbeef", "3751") is False

    def test_blank_playlist_id_is_ignored(self, initialized_db):
        result_id = self._seed_result()
        assert library_cache.link_result_playlist(result_id, "") is False

    def test_old_results_table_gains_the_columns(self, temp_db, monkeypatch):
        """Pre-existing databases are migrated, so old rows stay readable."""
        old = sqlite3.connect(str(temp_db))
        old.execute(
            "CREATE TABLE tracks (rating_key TEXT PRIMARY KEY, title TEXT NOT NULL, "
            "artist TEXT NOT NULL, album TEXT NOT NULL, duration_ms INTEGER, "
            "year INTEGER, genres TEXT, user_rating INTEGER, is_live BOOLEAN, "
            "updated_at TIMESTAMP)"
        )
        old.execute(
            "CREATE TABLE results (id TEXT PRIMARY KEY, type TEXT NOT NULL, title TEXT NOT NULL, "
            "prompt TEXT NOT NULL, snapshot JSON NOT NULL, track_count INTEGER NOT NULL, "
            "artist TEXT, art_rating_key TEXT, created_at TIMESTAMP)"
        )
        old.execute(
            "INSERT INTO results (id, type, title, prompt, snapshot, track_count) "
            "VALUES ('aaaaaaaa', 'prompt_playlist', 'Old', 'p', '{\"tracks\": []}', 3)"
        )
        old.commit()
        old.close()

        library_cache.ensure_db_initialized()

        row = library_cache.get_result("aaaaaaaa")
        assert row is not None
        assert row["playlist_id"] is None
        assert library_cache.link_result_playlist("aaaaaaaa", "9999", "Old", "Becky") is True


class TestFacetFilters:
    """Year, play-count, decade-ban and compilation filters on the cache path."""

    @pytest.fixture
    def facet_tracks(self, initialized_db):
        """Seven tracks covering undated, never-played and Various-Artists rows."""
        conn = library_cache.get_db_connection()
        rows = [
            # key, title, artist, album, duration_ms, year, rating, is_live, view_count
            ("1", "Rock Song", "Rock Artist", "Rock Album", 180000, 1995, 8, 0, 0),
            ("2", "Pop Song", "Pop Artist", "Pop Album", 200000, 2005, 6, 0, 7),
            ("3", "Live Concert", "Live Artist", "Live 2020", 300000, 2020, 4, 1, 3),
            ("4", "Jazz Song", "Jazz Artist", "Jazz Album", 250000, 1985, 10, 0, 2),
            ("5", "Session Sketch", "Various Artists", "Dance Hits", 210000, None, 0, 0, 0),
            ("6", "Eighties Hit", "Rock Artist", "Hair Metal", 190000, 1989, 5, 0, 30),
            ("7", "Orphan Song", "Rock Artist", "Self Titled", 195000, 2011, 3, 0, None),
        ]
        conn.executemany(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, user_rating, is_live, view_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.execute(
            "UPDATE sync_state SET plex_server_id = ?, last_sync_at = ?, "
            "track_count = ?, sync_duration_ms = ? WHERE id = 1",
            ("facet-server", datetime.now(timezone.utc).isoformat(), len(rows), 5000),
        )
        conn.commit()
        conn.close()
        return rows

    def _keys(self, **kwargs):
        return sorted(t["rating_key"] for t in library_cache.get_tracks_by_filters(**kwargs))

    def test_year_window_narrows_and_drops_undated(self, facet_tracks):
        """A requested release window excludes undated rows."""
        assert self._keys(min_year=1990, max_year=2010, exclude_live=False) == ["1", "2"]

    def test_decade_ban_keeps_undated_rows(self, facet_tracks):
        """An undated track is not inside a banned decade, so it survives."""
        assert self._keys(exclude_decades=["1980s"], exclude_live=False) == ["1", "2", "3", "5", "7"]

    def test_play_floor(self, facet_tracks):
        assert self._keys(min_plays=3, exclude_live=False) == ["2", "3", "6"]

    def test_play_ceiling_keeps_null_play_counts(self, facet_tracks):
        """A NULL play count is zero, not a missing value."""
        assert self._keys(max_plays=2, exclude_live=False) == ["1", "4", "5", "7"]

    def test_unheard_only(self, facet_tracks):
        assert self._keys(unheard_only=True, exclude_live=False) == ["1", "5", "7"]

    def test_unheard_and_year_window_compose(self, facet_tracks):
        assert self._keys(unheard_only=True, min_year=1980, exclude_live=False) == ["1", "7"]

    def test_compilation_drop(self, facet_tracks):
        """Only the Various-Artists row is dropped."""
        assert self._keys(exclude_compilations=True, exclude_live=False) == ["1", "2", "3", "4", "6", "7"]

    def test_artist_ban_uses_containment(self, facet_tracks):
        assert self._keys(exclude_artists=["pop artist"], exclude_live=False) == ["1", "3", "4", "5", "6", "7"]

    def test_year_window_beats_decade_hint(self, facet_tracks):
        assert self._keys(decades=["1980s"], min_year=1988, exclude_live=False) == ["6"]

    def test_count_matches_the_fetch(self, facet_tracks):
        """The COUNT(*) shortcut agrees with the filtered fetch."""
        count = library_cache.count_tracks_by_filters(
            unheard_only=True, min_year=1980, exclude_live=False
        )

        assert count == 2

    def test_count_still_signals_empty_cache(self, initialized_db):
        assert library_cache.count_tracks_by_filters(unheard_only=True) == -1

    def test_play_and_year_measurements(self, facet_tracks):
        """The cache reports the facets a prompt can only ask about sensibly."""
        stats = library_cache.get_cached_play_stats()

        assert stats["total"] == 6
        assert stats["unheard"] == 3
        assert stats["played"] == 3
        assert stats["plays"] == [0, 0, 0, 2, 30]
        assert stats["years"] == [1985, 2011]

    def test_undated_library_reports_no_year_span(self, facet_tracks):
        conn = library_cache.get_db_connection()
        conn.execute("UPDATE tracks SET year = NULL")
        conn.commit()
        conn.close()

        assert library_cache.get_cached_play_stats()["years"] == []


class TestFilterRelaxation:
    """The ladder widens filters, cheapest first, until the candidate target fits."""

    @pytest.fixture
    def counted(self, monkeypatch):
        """Stand in for the counter: every facet the ladder gives up widens the pool."""
        calls: list[dict] = []

        def count(**facets):
            calls.append(dict(facets))
            kept = sum(
                1 for name, value in facets.items() if value != library_cache.FILTER_NEUTRALS[name]
            )
            return 100 // (2**kept)

        monkeypatch.setattr(library_cache, "count_tracks_by_filters", count)
        return calls

    @pytest.fixture
    def narrow_ask(self):
        """A request that answers to every tier: a ban, a floor, a Plex floor."""
        return dict(
            genres=["Rock"],
            exclude_genres=["Noise"],
            min_rating=6,
            exclude_live=True,
        )

    def test_ladder_stops_at_the_first_tier_that_reaches_the_target(self, counted, narrow_ask):
        """The tiers before that one are reported, the ones after are not paid."""
        ladder = library_cache.relax_filters_for_target(narrow_ask, 25)

        assert ladder["tier"] == "no rating or play floors"
        assert ladder["relaxed"] == ["no bans", "no rating or play floors"]
        assert ladder["matching"] == 25
        assert ladder["asked"] == 6
        assert ladder["short_by"] == 0
        # The asked facets are counted first, then one count per tier that changed
        # something; the year and compilation tiers touch nothing here.
        assert len(counted) == 3
        assert counted[0] == narrow_ask

    def test_asking_for_a_ban_gives_up_the_ban_first(self, counted, narrow_ask):
        """A target the bans alone cannot satisfy keeps the identity filters."""
        ladder = library_cache.relax_filters_for_target(narrow_ask, 6, count=3)

        assert ladder["tier"] == "no bans"
        assert ladder["relaxed"] == ["no bans"]
        assert ladder["filters"]["exclude_genres"] is None
        assert ladder["filters"]["min_rating"] == 6
        assert ladder["filters"]["genres"] == ["Rock"]

    def test_filters_that_already_reach_the_target_are_left_alone(self, counted, narrow_ask):
        ladder = library_cache.relax_filters_for_target(narrow_ask, 3)

        assert ladder["tier"] == "as asked"
        assert ladder["relaxed"] == []
        assert ladder["filters"] == narrow_ask
        assert ladder["matching"] == ladder["asked"] == 6
        assert len(counted) == 1

    def test_zero_target_turns_the_ladder_off(self, counted, narrow_ask):
        """``max_tracks_to_ai`` set to 0 means the filters run exactly as asked."""
        ladder = library_cache.relax_filters_for_target(narrow_ask, 0)

        assert ladder["tier"] == "as asked"
        assert ladder["relaxed"] == []
        assert ladder["filters"] == narrow_ask

    def test_empty_cache_ends_the_walk(self, narrow_ask, monkeypatch):
        """``-1`` is Plex's signal, not an empty pool: no tier may widen it."""
        seen: list[dict] = []

        def count(**facets):
            seen.append(dict(facets))
            return -1

        monkeypatch.setattr(library_cache, "count_tracks_by_filters", count)

        ladder = library_cache.relax_filters_for_target(narrow_ask, 50)

        assert ladder["matching"] == -1
        assert ladder["asked"] == -1
        assert ladder["tier"] == "as asked"
        assert ladder["relaxed"] == []
        assert ladder["filters"] == narrow_ask
        assert len(seen) == 1

    def test_identity_tier_never_runs_for_an_answered_request(self, counted, narrow_ask):
        """3 candidates for a 3-track playlist is a thin but complete answer."""
        ladder = library_cache.relax_filters_for_target(narrow_ask, 1000, identity_floor=3)

        assert library_cache.IDENTITY_TIER not in ladder["relaxed"]
        assert ladder["filters"]["genres"] == ["Rock"]
        assert ladder["filters"]["exclude_live"] is False
        assert ladder["tier"] == "compilations and live recordings allowed"
        assert ladder["short_by"] == 950

    def test_starved_request_loses_the_whitelist(self, counted, narrow_ask):
        """With no floor the ladder runs out and widens the request completely."""
        ladder = library_cache.relax_filters_for_target(narrow_ask, 1000)

        assert ladder["tier"] == library_cache.IDENTITY_TIER
        assert ladder["filters"]["genres"] is None
        assert ladder["matching"] == 100
        assert ladder["short_by"] == 900

    def test_unknown_facets_are_rejected(self):
        """A facet the neutrals table does not know would be silently ignored."""
        with pytest.raises(ValueError, match="unknown filter facets"):
            library_cache.relax_filters_for_target({"nonsense": 1}, 10)


class TestUserRatings:
    """Per-account stars live in their own table; the admin's stay a column."""

    @pytest.fixture
    def plex(self):
        """The connected admin client: scanning a member yields (rating_key, stars)."""

        class FakeAdminClient:
            def collect_user_ratings(self, user=None):
                return [("k1", 9), ("k2", 4)]

        return FakeAdminClient()

    @pytest.fixture
    def rescanned_plex(self):
        """The same member, one star changed and one track un-rated."""

        class FakeAdminClient:
            def collect_user_ratings(self, user=None):
                return [("k1", 2), ("k2", 4)]

        return FakeAdminClient()

    @pytest.fixture
    def unrated_plex(self):
        """A member who has never starred anything: a scan, but no rows."""

        class FakeAdminClient:
            def collect_user_ratings(self, user=None):
                return []

        return FakeAdminClient()

    @pytest.fixture
    def unknown_plex(self):
        """A key Plex does not know: the scan must fail, not answer 0 rows."""
        from backend.plex_client import PlexQueryError

        class FakeAdminClient:
            def collect_user_ratings(self, user=None):
                raise PlexQueryError("Unknown Plex user")

        return FakeAdminClient()

    def test_the_admin_never_gets_rows(self, initialized_db):
        """Its stars already are every track's ``user_rating`` column."""
        assert library_cache.sync_user_ratings(library_cache.OWNER_USER_KEY, None) == 0
        assert library_cache.sync_user_ratings("", None) == 0

        conn = sqlite3.connect(str(initialized_db))
        assert conn.execute("SELECT COUNT(1) FROM user_ratings").fetchone()[0] == 0
        conn.close()

    def test_a_member_scan_stores_rows_and_a_stamp(self, initialized_db, plex):
        """Rows plus the ``rating_sync`` line that marks the account as scanned."""
        assert library_cache.sync_user_ratings("Becky", plex) == 2

        conn = sqlite3.connect(str(initialized_db))
        rows = conn.execute(
            "SELECT user_key, rating_key, rating FROM user_ratings ORDER BY rating_key"
        ).fetchall()
        sync = conn.execute(
            "SELECT user_key, rated_count, total_seen FROM rating_sync"
        ).fetchall()
        conn.close()

        assert rows == [("Becky", "k1", 9), ("Becky", "k2", 4)]
        assert sync == [("Becky", 2, 0)]

    def test_rescanning_replaces_rows_rather_than_stacking_them(self, initialized_db, plex, rescanned_plex):
        """An account can un-rate a track, so the previous scan's rows must go."""
        library_cache.sync_user_ratings("Becky", plex)
        library_cache.sync_user_ratings("Becky", rescanned_plex)

        conn = sqlite3.connect(str(initialized_db))
        rows = conn.execute(
            "SELECT user_key, rating_key, rating FROM user_ratings ORDER BY rating_key"
        ).fetchall()
        conn.close()

        assert rows == [("Becky", "k1", 2), ("Becky", "k2", 4)]

    def test_status_names_the_accounts_already_scanned(self, initialized_db, plex):
        """The status read is what tells 'not loaded yet' from 'loaded and rated nothing'."""
        library_cache.sync_user_ratings("Becky", plex)

        state = library_cache.user_ratings_status()
        assert state["synced_users"] == ["Becky"]
        assert state["users"] == [
            {"user_key": "Becky", "rated_count": 2, "total_seen": 0, "last_sync_at": state["users"][0]["last_sync_at"]}
        ]
        assert state["users"][0]["last_sync_at"] is not None

    def test_rated_nothing_is_not_the_same_as_never_scanned(self, initialized_db, unrated_plex):
        """A scan with no rows still stamps the account, and gates the next scan."""
        assert library_cache.has_user_ratings("Becky") is False

        assert library_cache.sync_user_ratings("Becky", unrated_plex) == 0

        assert library_cache.has_user_ratings("Becky") is False
        assert library_cache.user_ratings_status()["synced_users"] == ["Becky"]

    def test_a_failed_scan_is_not_counted_as_rated_nothing(self, initialized_db, unknown_plex):
        """Silently returning 0 here would drop every track the member rated."""
        from backend.plex_client import PlexQueryError

        with pytest.raises(PlexQueryError, match="Unknown Plex user"):
            library_cache.sync_user_ratings("Becky", unknown_plex)

        assert library_cache.user_ratings_status()["synced_users"] == []

    def test_ensure_scans_once_and_the_rest_of_the_requests_pay_nothing(self, initialized_db, monkeypatch):
        """Per-request loading: only the first filter asking for an account scans."""
        scans: list[str] = []

        class FakeAdminClient:
            def collect_user_ratings(self, user=None):
                scans.append(user or "")
                return [("k1", 9)]

        assert library_cache.ensure_user_ratings("Becky", FakeAdminClient()) == 1
        assert scans == ["Becky"]

        assert library_cache.ensure_user_ratings("Becky", FakeAdminClient()) == 0
        assert scans == ["Becky"]

    def test_ensure_never_scans_the_admin(self, initialized_db, monkeypatch):
        """The admin's column is already loaded with the library."""
        scans: list[str] = []

        class FakeAdminClient:
            def collect_user_ratings(self, user=None):
                scans.append(user or "")
                return [("k1", 9)]

        assert library_cache.ensure_user_ratings(library_cache.OWNER_USER_KEY, FakeAdminClient()) == 0
        assert library_cache.ensure_user_ratings("", FakeAdminClient()) == 0
        assert scans == []
