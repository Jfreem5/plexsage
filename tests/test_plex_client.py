"""Tests for Plex client."""

import time
from unittest.mock import MagicMock, patch

from backend.models import Track
from backend.plex_client import TrackCache, is_live_version


class TestPlexClientConnection:
    """Tests for Plex connection handling."""

    def test_connect_with_valid_credentials(self, mocker):
        """Should connect successfully with valid credentials."""
        from backend.plex_client import PlexClient

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "valid-token", "Music")
            assert client.is_connected() is True

    def test_connect_with_invalid_token(self, mocker):
        """Should handle invalid token gracefully."""
        from backend.plex_client import PlexClient
        from plexapi.exceptions import Unauthorized

        with patch("backend.plex_client.PlexServer", side_effect=Unauthorized("Invalid token")):
            client = PlexClient("http://localhost:32400", "invalid-token", "Music")
            assert client.is_connected() is False
            assert "invalid" in client.get_error().lower() or "unauthorized" in client.get_error().lower()

    def test_connect_with_unreachable_server(self, mocker):
        """Should handle unreachable server gracefully."""
        from backend.plex_client import PlexClient
        from requests.exceptions import ConnectionError

        with patch("backend.plex_client.PlexServer", side_effect=ConnectionError("Connection refused")):
            client = PlexClient("http://unreachable:32400", "token", "Music")
            assert client.is_connected() is False
            assert client.get_error() is not None

    def test_connect_with_missing_library(self, mocker):
        """Should handle missing music library."""
        from backend.plex_client import PlexClient
        from plexapi.exceptions import NotFound

        mock_server = MagicMock()
        mock_server.library.section.side_effect = NotFound("Library not found")

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "NonexistentLibrary")
            assert client.is_connected() is False
            assert "library" in client.get_error().lower() or "not found" in client.get_error().lower()


class TestPlexClientLibraryStats:
    """Tests for library statistics."""

    def test_get_library_stats(self, mocker):
        """Should return library statistics."""
        from backend.plex_client import PlexClient

        # Mock genre filter choices
        mock_genre1 = MagicMock()
        mock_genre1.title = "Rock"
        mock_genre2 = MagicMock()
        mock_genre2.title = "Alternative"

        # Mock decade filter choices
        mock_decade1 = MagicMock()
        mock_decade1.title = "1990"
        mock_decade2 = MagicMock()
        mock_decade2.title = "2000"

        mock_library = MagicMock()
        mock_library.totalViewSize.return_value = 10
        mock_library.listFilterChoices.side_effect = lambda field, libtype: (
            [mock_genre1, mock_genre2] if field == "genre" else [mock_decade1, mock_decade2]
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            stats = client.get_library_stats()

            assert stats["total_tracks"] == 10
            assert len(stats["genres"]) == 2
            assert len(stats["decades"]) == 2


class TestPlexClientMusicLibraries:
    """Tests for music library listing."""

    def test_get_music_libraries(self, mocker):
        """Should return list of music libraries."""
        from backend.plex_client import PlexClient

        mock_section1 = MagicMock()
        mock_section1.title = "Music"
        mock_section1.type = "artist"

        mock_section2 = MagicMock()
        mock_section2.title = "Movies"
        mock_section2.type = "movie"

        mock_section3 = MagicMock()
        mock_section3.title = "Jazz Collection"
        mock_section3.type = "artist"

        mock_server = MagicMock()
        mock_server.library.sections.return_value = [mock_section1, mock_section2, mock_section3]
        mock_server.library.section.return_value = MagicMock()

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            libraries = client.get_music_libraries()

            assert "Music" in libraries
            assert "Jazz Collection" in libraries
            assert "Movies" not in libraries


class TestPlexClientTrackSearch:
    """Tests for track search functionality."""

    def test_search_tracks_by_title(self, mocker):
        """Should find tracks by title."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_track.ratingKey = "123"
        mock_track.title = "Fake Plastic Trees"
        mock_track.grandparentTitle = "Radiohead"
        mock_track.parentTitle = "The Bends"
        mock_track.duration = 290000
        mock_track.parentYear = 1995
        mock_track.genres = [MagicMock(tag="Alternative")]

        mock_library = MagicMock()
        mock_library.searchTracks.return_value = [mock_track]
        mock_library.search.return_value = []

        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            results = client.search_tracks("Fake Plastic")

            assert len(results) == 1
            assert results[0].title == "Fake Plastic Trees"
            assert results[0].artist == "Radiohead"

    def test_search_tracks_by_artist(self, mocker):
        """Should find tracks by artist name."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_track.ratingKey = "456"
        mock_track.title = "Creep"
        mock_track.grandparentTitle = "Radiohead"
        mock_track.parentTitle = "Pablo Honey"
        mock_track.duration = 238000
        mock_track.parentYear = 1993
        mock_track.genres = []

        mock_library = MagicMock()
        mock_library.searchTracks.return_value = []
        mock_library.search.return_value = [mock_track]

        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            results = client.search_tracks("Radiohead")

            assert len(results) == 1
            assert results[0].artist == "Radiohead"

    def test_search_tracks_returns_formatted_results(self, mocker):
        """Search results should include album art URLs."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_track.ratingKey = "789"
        mock_track.title = "Black"
        mock_track.grandparentTitle = "Pearl Jam"
        mock_track.parentTitle = "Ten"
        mock_track.duration = 340000
        mock_track.parentYear = 1991
        mock_track.genres = [MagicMock(tag="Grunge")]

        mock_library = MagicMock()
        mock_library.searchTracks.return_value = [mock_track]
        mock_library.search.return_value = []

        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            results = client.search_tracks("Black")

            assert len(results) == 1
            assert results[0].art_url == "/api/art/789"


class TestPlexClientPlaylistCreation:
    """Tests for playlist creation."""

    def test_create_playlist_success(self, mocker):
        """Should create playlist successfully."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_playlist = MagicMock()
        mock_playlist.ratingKey = "999"

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.return_value = mock_track
        mock_server.createPlaylist.return_value = mock_playlist

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            result = client.create_playlist("Test Playlist", ["1", "2", "3"])

            assert result["success"] is True
            assert result["playlist_id"] == "999"

    def test_create_playlist_handles_invalid_tracks(self, mocker):
        """Should skip invalid track keys gracefully."""
        from backend.plex_client import PlexClient

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.side_effect = Exception("Not found")

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            result = client.create_playlist("Test Playlist", ["invalid"])

            assert result["success"] is False
            assert "error" in result

    def test_create_playlist_returns_playlist_url(self, mocker):
        """Should return playlist URL when successful."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_playlist = MagicMock()
        mock_playlist.ratingKey = "999"

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.return_value = mock_track
        mock_server.createPlaylist.return_value = mock_playlist
        mock_server.machineIdentifier = "abc123def456"

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            result = client.create_playlist("Test Playlist", ["1", "2", "3"])

            assert result["success"] is True
            assert result["playlist_url"] is not None
            assert "abc123def456" in result["playlist_url"]
            assert "999" in result["playlist_url"]


class TestTrackCache:
    """Tests for track caching functionality."""

    def _make_track(self, rating_key: str, title: str = "Test Track") -> Track:
        """Helper to create a test track."""
        return Track(
            rating_key=rating_key,
            title=title,
            artist="Test Artist",
            album="Test Album",
            duration_ms=180000,
        )

    def test_cache_miss_returns_none(self):
        """Should return None for uncached filters."""
        cache = TrackCache()
        result = cache.get(["Rock"], ["1990s"], True, 0)
        assert result is None

    def test_cache_hit_returns_tracks(self):
        """Should return cached tracks for matching filters."""
        cache = TrackCache()
        tracks = [self._make_track("1"), self._make_track("2")]

        cache.set(["Rock"], ["1990s"], True, 0, tracks)
        result = cache.get(["Rock"], ["1990s"], True, 0)

        assert result is not None
        assert len(result) == 2
        assert result[0].rating_key == "1"

    def test_cache_expired_returns_none(self):
        """Should return None for expired entries."""
        cache = TrackCache(ttl_seconds=0)  # Immediate expiration
        tracks = [self._make_track("1")]

        cache.set(["Rock"], ["1990s"], True, 0, tracks)
        time.sleep(0.01)  # Ensure TTL expires
        result = cache.get(["Rock"], ["1990s"], True, 0)

        assert result is None

    def test_different_filters_are_separate_entries(self):
        """Should cache separately for different filter combinations."""
        cache = TrackCache()
        rock_tracks = [self._make_track("1", "Rock Song")]
        jazz_tracks = [self._make_track("2", "Jazz Song")]

        cache.set(["Rock"], [], True, 0, rock_tracks)
        cache.set(["Jazz"], [], True, 0, jazz_tracks)

        rock_result = cache.get(["Rock"], [], True, 0)
        jazz_result = cache.get(["Jazz"], [], True, 0)

        assert rock_result[0].title == "Rock Song"
        assert jazz_result[0].title == "Jazz Song"

    def test_key_generation_is_consistent(self):
        """Should generate same key regardless of genre/decade order."""
        cache = TrackCache()
        tracks = [self._make_track("1")]

        # Set with one order
        cache.set(["Rock", "Alternative"], ["1990s", "2000s"], True, 0, tracks)

        # Get with different order - should still hit
        result = cache.get(["Alternative", "Rock"], ["2000s", "1990s"], True, 0)

        assert result is not None

    def test_max_entries_evicts_oldest(self):
        """Should evict oldest entry when at capacity."""
        cache = TrackCache(max_entries=2)

        cache.set(["Rock"], [], True, 0, [self._make_track("1")])
        time.sleep(0.01)
        cache.set(["Jazz"], [], True, 0, [self._make_track("2")])
        time.sleep(0.01)
        cache.set(["Pop"], [], True, 0, [self._make_track("3")])  # Should evict Rock

        assert cache.get(["Rock"], [], True, 0) is None  # Evicted
        assert cache.get(["Jazz"], [], True, 0) is not None
        assert cache.get(["Pop"], [], True, 0) is not None

    def test_clear_removes_all_entries(self):
        """Should remove all entries on clear."""
        cache = TrackCache()
        cache.set(["Rock"], [], True, 0, [self._make_track("1")])
        cache.set(["Jazz"], [], True, 0, [self._make_track("2")])

        cache.clear()

        assert cache.get(["Rock"], [], True, 0) is None
        assert cache.get(["Jazz"], [], True, 0) is None

    def test_updating_existing_key_does_not_evict(self):
        """Should not evict when updating an existing entry."""
        cache = TrackCache(max_entries=2)

        cache.set(["Rock"], [], True, 0, [self._make_track("1")])
        cache.set(["Jazz"], [], True, 0, [self._make_track("2")])

        # Update Rock - should not trigger eviction
        cache.set(["Rock"], [], True, 0, [self._make_track("1-updated")])

        assert cache.get(["Rock"], [], True, 0) is not None
        assert cache.get(["Jazz"], [], True, 0) is not None


def _make_plex_track(
    rating_key="123",
    title="Fake Plastic Trees",
    artist="Radiohead",
    album="The Bends",
    year=1995,
    duration=290000,
    genres=("Alternative", "Rock"),
    user_rating=9,
    view_count=5,
    last_viewed_at=1700000000,
):
    """Build a MagicMock that quacks like a Plex track object."""
    track = MagicMock()
    track.ratingKey = rating_key
    track.title = title
    track.grandparentTitle = artist
    track.parentTitle = album
    track.parentYear = year
    track.duration = duration
    track.genres = [MagicMock(tag=g) for g in genres]
    track.userRating = user_rating
    track.viewCount = view_count
    track.lastViewedAt = last_viewed_at
    return track


class TestPlexClientTrackDB:
    """Reads should be served from the local TrackDB when it is populated."""

    def _client(self):
        from backend.plex_client import PlexClient

        mock_library = MagicMock()
        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library
        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
        return client, mock_library

    def _make_db(self, tmp_path, populate=True):
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "plex.db")
        if populate:
            db.upsert_tracks(
                [
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
                ],
                is_live_flags=[False, False],
                user_ratings=[9, 8],
            )
            db.set_last_sync()
        return db

    def test_filters_use_track_db(self, tmp_path):
        """get_tracks_by_filters should not touch Plex when the DB has data."""
        client, mock_library = self._client()
        db = self._make_db(tmp_path)

        with patch("backend.plex_client.get_track_db", return_value=db):
            results = client.get_tracks_by_filters(genres=["Rock"])

        assert {t.title for t in results} == {"Fake Plastic Trees", "Black"}
        mock_library.search.assert_not_called()

    def test_count_uses_track_db(self, tmp_path):
        client, mock_library = self._client()
        db = self._make_db(tmp_path)

        with patch("backend.plex_client.get_track_db", return_value=db):
            count = client.count_tracks_by_filters(genres=["Rock"])

        assert count == 2
        mock_library.search.assert_not_called()

    def test_search_uses_track_db(self, tmp_path):
        client, mock_library = self._client()
        db = self._make_db(tmp_path)

        with patch("backend.plex_client.get_track_db", return_value=db):
            results = client.search_tracks("radio")

        assert [t.title for t in results] == ["Fake Plastic Trees"]
        mock_library.searchTracks.assert_not_called()

    def test_library_stats_use_track_db(self, tmp_path):
        client, mock_library = self._client()
        db = self._make_db(tmp_path)

        with patch("backend.plex_client.get_track_db", return_value=db):
            stats = client.get_library_stats()

        assert stats["total_tracks"] == 2
        # Genre counts come from the DB, which Plex's filter-choices API lacks.
        assert {g["name"]: g["count"] for g in stats["genres"]}["Rock"] == 2
        mock_library.listFilterChoices.assert_not_called()

    def test_falls_back_to_plex_when_db_empty(self, tmp_path):
        """An unpopulated DB must not be treated as authoritative."""
        client, mock_library = self._client()
        empty_db = self._make_db(tmp_path, populate=False)

        mock_library.search.return_value = [_make_plex_track()]

        with patch("backend.plex_client.get_track_db", return_value=empty_db):
            results = client.get_tracks_by_filters(genres=["Rock"])

        mock_library.search.assert_called_once()
        assert [t.title for t in results] == ["Fake Plastic Trees"]

    def test_falls_back_to_plex_when_db_absent(self):
        client, mock_library = self._client()
        mock_library.search.return_value = [_make_plex_track()]

        with patch("backend.plex_client.get_track_db", return_value=None):
            results = client.get_tracks_by_filters(genres=["Rock"])

        mock_library.search.assert_called_once()
        assert len(results) == 1

    def test_sync_track_db_populates_metadata(self, tmp_path):
        client, mock_library = self._client()
        db = self._make_db(tmp_path, populate=False)
        mock_library.search.return_value = [_make_plex_track()]

        with patch("backend.plex_client.get_track_db", return_value=db):
            count = client.sync_track_db(force=True)

        assert count == 1
        assert db.get_track_count() == 1
        meta = db.get_metadata("123")
        assert meta["user_rating"] == 9
        assert meta["play_count"] == 5
        assert meta["last_played"] is not None
        assert meta["is_live"] is False
        assert set(meta["genres"]) == {"Alternative", "Rock"}

    def test_db_path_carries_precomputed_live_flag(self, tmp_path):
        """DB reads expose the sync-time flag and honour exclude_live in SQL."""
        client, mock_library = self._client()
        db = self._make_db(tmp_path)
        db.upsert_tracks(
            [
                Track(
                    rating_key="9",
                    title="Live Cut",
                    artist="Oasis",
                    album="Live at Knebworth 1996",
                    duration_ms=200000,
                    year=1996,
                    genres=["Rock"],
                )
            ],
            is_live_flags=[True],
        )

        with patch("backend.plex_client.get_track_db", return_value=db):
            with_live = client.get_tracks_by_filters(genres=["Rock"], exclude_live=False)
            without_live = client.get_tracks_by_filters(genres=["Rock"], exclude_live=True)

        flags = {t.rating_key: t.is_live for t in with_live}
        assert flags["9"] is True
        assert flags["1"] is False
        assert "9" not in {t.rating_key for t in without_live}
        mock_library.search.assert_not_called()

    def test_db_path_does_not_recompute_live(self, tmp_path):
        """Serving from the DB must not run the live-detection regex."""
        client, _ = self._client()
        db = self._make_db(tmp_path)

        with patch("backend.plex_client.get_track_db", return_value=db):
            with patch("backend.plex_client.is_live_version") as spy:
                results = client.get_tracks_by_filters(genres=["Rock"])

        assert results
        spy.assert_not_called()

    def test_api_fallback_detects_live_once_per_track(self):
        """On the API path, detection runs once per fetched track and is stored."""
        client, mock_library = self._client()
        mock_library.search.return_value = [
            _make_plex_track(rating_key="1", title="Song", album="Some Album"),
            _make_plex_track(rating_key="2", title="Song", album="Live at Foo"),
        ]

        with patch("backend.plex_client.get_track_db", return_value=None):
            with patch(
                "backend.plex_client.is_live_version", wraps=is_live_version
            ) as spy:
                results = client.get_tracks_by_filters(genres=["Rock"])

        assert [t.rating_key for t in results] == ["1"]
        assert results[0].is_live is False
        assert spy.call_count == 2  # once per fetched track

    def test_sync_track_db_skips_when_fresh(self, tmp_path):
        client, mock_library = self._client()
        db = self._make_db(tmp_path)  # populated + last_sync = now

        with patch("backend.plex_client.get_track_db", return_value=db):
            count = client.sync_track_db()

        assert count == 0
        mock_library.search.assert_not_called()

    def test_ensure_synced_triggers_when_stale(self, tmp_path):
        client, mock_library = self._client()
        db = self._make_db(tmp_path)
        db.set_last_sync(time.time() - 1000)
        mock_library.search.return_value = [_make_plex_track()]

        with patch("backend.plex_client.get_track_db", return_value=db):
            count = client.ensure_synced(max_age_seconds=10)

        assert count == 1
        mock_library.search.assert_called_once()

    def test_sync_prunes_removed_tracks(self, tmp_path):
        """Tracks gone from Plex are removed from the mirror."""
        client, mock_library = self._client()
        db = self._make_db(tmp_path)  # has keys 1 and 2
        mock_library.search.return_value = [_make_plex_track(rating_key="1")]

        with patch("backend.plex_client.get_track_db", return_value=db):
            client.sync_track_db(force=True)

        assert db.get_track_count() == 1
        assert db.get_metadata("2") is None

