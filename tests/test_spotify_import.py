"""Tests for the Spotify playlist import chain.

No network and no duckdb here: the fetch paths run against canned ``requests``
responses and a fake duckdb connection, and matching runs against a throwaway
cache DB, so no number in these tests is machine state.
"""

import sys
import types
from unittest.mock import MagicMock, Mock, patch

import pytest

from backend import library_cache, spotify_import
from backend.models import AppConfig, LLMConfig, PlexConfig, Track


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the cache at a throwaway database and data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(library_cache, "DATA_DIR", data_dir)
    monkeypatch.setattr(library_cache, "DB_PATH", data_dir / "library_cache.db")
    monkeypatch.setattr(library_cache, "_schema_initialized", False)
    yield data_dir


@pytest.fixture
def cached_library(temp_db):
    """Six library rows covering the match rungs and their near misses."""
    conn = library_cache.ensure_db_initialized()
    conn.executemany(
        "INSERT INTO tracks (rating_key, title, artist, album, duration_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            ("1", "Fake Plastic Trees", "Radiohead", "The Bends", 290000),
            ("2", "Say It Ain't So", "Weezer", "Weezer (Blue Album)", 268000),
            ("3", "Come Together", "The Beatles", "Abbey Road", 300000),
            ("4", "All of the Lights", "Kanye West & Rihanna", "My Beautiful Darking", 400000),
            ("5", "Silent Shout", "The Avalanches", "Since I Left You", 290000),
            ("6", "Clocks", "Coldplay", "A Rush of Madness", 400000),
        ],
    )
    conn.commit()
    conn.close()
    return temp_db


def api_rows(items: list[dict], next_url: str | None = None) -> dict:
    """One API page body with the given items."""
    return {"items": items, "next": next_url, "total": len(items)}


SPOTIFY_ROW = {
    "position": 1,
    "title": "Fake Plastic Trees",
    "album": "The Bends",
    "artist": "Radiohead",
    "duration_ms": 290000,
    "spotify_track_id": "abc123",
    "spotify_url": "https://open.spotify.com/track/abc123",
}


class TestPlaylistRefParsing:
    """Every accepted spelling of a playlist reference lands the same id."""

    def test_url_uri_and_bare_id_agree(self):
        assert spotify_import.parse_playlist_ref(
            "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=x"
        ) == (
            "37i9dQZF1DXcBWIGoYBM5M",
            "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
        )
        assert spotify_import.parse_playlist_ref(
            "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M"
        )[0] == "37i9dQZF1DXcBWIGoYBM5M"
        assert spotify_import.parse_playlist_ref("37i9dQZF1DXcBWIGoYBM5M")[0] == (
            "37i9dQZF1DXcBWIGoYBM5M"
        )

    def test_user_playlist_path_still_yields_the_playlist_id(self):
        assert spotify_import.parse_playlist_ref(
            "https://open.spotify.com/user/someone/playlist/37i9dQZF1DXcBWIGoYBM5M/save"
        )[0] == "37i9dQZF1DXcBWIGoYBM5M"

    def test_non_playlist_refs_are_rejected_with_names(self):
        for bad in ("", "https://open.spotify.com/album/abcdef0123456789abcdef"):
            with pytest.raises(spotify_import.SpotifyImportError) as err:
                spotify_import.parse_playlist_ref(bad)
            if bad:
                assert bad in str(err.value)


class TestApiFetch:
    """Canned requests responses: paging, episode skip, and errors."""

    @pytest.fixture(autouse=True)
    def fake_http(self, monkeypatch):
        """Canned token + playlist endpoints, no network."""
        monkeypatch.setattr(spotify_import, "_token_cache", None)
        page_url = "https://api.spotify.com/v1/playlists/p1/tracks"
        pages = {
            "https://api.spotify.com/v1/playlists/p1": {
                "name": "Canned",
                "owner": {"display_name": "Owner"},
                "tracks": {"total": 3, "href": page_url, "next": None},
            },
            page_url: api_rows(
                [
                    {"index": 0, "type": "track", "track": {
                        "name": "One", "id": "t1", "duration_ms": 100,
                        "artists": [{"name": "A"}], "album": {"name": "AL"},
                    }},
                    {"index": 1, "type": "episode", "track": {"name": "Skip me"}},
                    {"index": 2, "type": "track", "track": {
                        "name": "Local file", "id": None, "duration_ms": 0,
                        "artists": [], "album": None,
                    }},
                ]
            ),
        }

        def get(url, **_):
            body = pages[url]
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = body
            return response

        def post(url, **_):
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"access_token": "tok", "expires_in": 3600}
            return response

        monkeypatch.setattr(spotify_import.requests, "post", post)
        monkeypatch.setattr(spotify_import.requests, "get", get)

    def test_pages_become_rows_positions_and_names(self):
        body = spotify_import.fetch_playlist_api("p1", "cid", "secret")
        assert body["name"] == "Canned"
        assert body["owner"] == "Owner"
        assert [t["title"] for t in body["tracks"]] == ["One", "Local file"]
        assert [t["position"] for t in body["tracks"]] == [1, 3]
        assert body["tracks"][0]["spotify_url"] == "https://open.spotify.com/track/t1"
        assert any("episode" in w for w in body["warnings"])

    def test_token_is_cached_across_calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            spotify_import.requests, "post", lambda url, **_: calls.append(url) or MagicMock(
                status_code=200, json=MagicMock(return_value={"access_token": "tok", "expires_in": 3600})
            )
        )
        spotify_import.api_access_token("cid", "secret")
        spotify_import.api_access_token("cid", "secret")
        assert calls == ["https://accounts.spotify.com/api/token"]

    def test_error_status_raises_named_error(self):
        def get(url, **_):
            response = MagicMock()
            response.status_code = 404
            return response

        with patch.object(spotify_import.requests, "get", get):
            with pytest.raises(spotify_import.SpotifyImportError) as err:
                spotify_import.fetch_playlist_api("missing", "cid", "secret")
        assert "HTTP 404" in str(err.value)


class TestDumpFetch:
    """The dump path: layout guard, duckdb absence, and canned rows."""

    def test_wrong_layout_named(self, tmp_path):
        with pytest.raises(spotify_import.SpotifyImportError) as err:
            spotify_import.fetch_playlist_dump("a" * 22, str(tmp_path))
        assert "one-directory-per-table" in str(err.value)

    def test_duckdb_absence_is_an_error_not_a_crash(self, tmp_path):
        for name in ("playlists", "playlist_tracks", "tracks", "albums"):
            (tmp_path / name).mkdir()
        with pytest.raises(spotify_import.SpotifyImportError) as err:
            spotify_import.fetch_playlist_dump("a" * 22, str(tmp_path))
        assert "duckdb" in str(err.value)

    def test_canned_duckdb_rows_become_rows_dict(self, tmp_path):
        for name in ("playlists", "playlist_tracks", "tracks", "albums"):
            (tmp_path / name).mkdir()
        results = [
            None,  # CREATE want
            [(7, "Canned list", "Owner")],  # SELECT want
            None,  # CREATE entry
            [
                (1, "Fake Plastic Trees", "The Bends", "Radiohead", 290000),
                (2, "Nope Nope", "No", "Nope", 1),
            ],
        ]

        class FakeCon:
            def __init__(self):
                self.i = 0

            def execute(self, sql):
                out = results[self.i]
                self.i += 1
                return MagicMock(fetchall=lambda: out)

            def close(self):
                pass

        fake = types.ModuleType("duckdb")
        fake.connect = lambda: FakeCon()
        with patch.dict(sys.modules, {"duckdb": fake}):
            body = spotify_import.fetch_playlist_dump("a" * 22, str(tmp_path))
        assert body["name"] == "Canned list"
        assert body["owner"] == "Owner"
        assert [t["title"] for t in body["tracks"]] == ["Fake Plastic Trees", "Nope Nope"]
        assert any("snapshot" in w for w in body["warnings"])

    def test_missing_playlist_named(self, tmp_path):
        for name in ("playlists", "playlist_tracks", "tracks", "albums"):
            (tmp_path / name).mkdir()
        results = [None, [], None, []]

        class FakeCon:
            def __init__(self):
                self.i = 0

            def execute(self, sql):
                out = results[self.i]
                self.i += 1
                return MagicMock(fetchall=lambda: out)

            def close(self):
                pass

        fake = types.ModuleType("duckdb")
        fake.connect = lambda: FakeCon()
        with patch.dict(sys.modules, {"duckdb": fake}):
            with pytest.raises(spotify_import.SpotifyImportError) as err:
                spotify_import.fetch_playlist_dump("a" * 22, str(tmp_path))
        assert "not in the local dump" in str(err.value)


class TestMatchCached:
    """The three join rungs against a throwaway cache DB."""

    @pytest.fixture(autouse=True)
    def seeded(self, cached_library):
        """Rows come from the seeded temp library."""

    def song(self, **over):
        row = dict(SPOTIFY_ROW)
        row.update(over)
        return row

    def rows_are_matched_by_rung(self):
        matched, missing = spotify_import.match_cached([
            self.song(),  # exact
            self.song(title="Say It Ain't So", album="Weezer (Deluxe)", artist="Weezer", position=2),
            self.song(title="Clocks", album="A Rush of Madness", artist="Coldplay (Remastered)", position=3),
            self.song(title="All of the Lights", album="My Beautiful Darking", artist="Rihanna & Kanye West", position=4),
            self.song(title="Never Here", position=5),
        ])
        kinds = {m["match_kind"] for m in matched}
        assert kinds == {"exact", "version", "album"}
        assert len(missing) == 1
        assert missing[0]["title"] == "Never Here"
        assert matched[0]["rating_key"] == "1"

    def all_lights_credits_land_via_split_artists(self):
        matched, missing = spotify_import.match_cached([
            self.song(title="All of the Lights", album="My Beautiful Darking", artist="Kanye West & Rihanna", position=1),
        ])
        assert matched[0]["rating_key"] == "4"
        assert matched[0]["match_kind"] == "exact"

    def duplicate_rows_reuse_without_double_matching(self):
        row = self.song()
        row2 = dict(row, position=9)
        matched, missing = spotify_import.match_cached([row, row2])
        assert len(matched) == 2
        assert {m["rating_key"] for m in matched} == {"1"}
        assert not missing


class TestWishList:
    """record_missing_songs replaces per playlist; list reads it back."""

    @pytest.fixture(autouse=True)
    def seeded(self, cached_library):
        """The wish list lives in the same throwaway DB as the library."""

    def wish_rows(self):
        return [
            {"position": 1, "title": "Missing One", "album": "AL", "artist": "A", "duration_ms": 100},
            {"position": 2, "title": "Missing Two", "album": "AL", "artist": "B", "duration_ms": 200},
            {"position": 0, "title": "No slot"},
        ]

    def test_record_counts_and_skips_unslotted(self):
        written = library_cache.record_missing_songs("pl1", "List", self.wish_rows())
        assert written == 2

    def test_reimport_replaces_instead_of_stacking(self):
        library_cache.record_missing_songs("pl1", "List", self.wish_rows())
        library_cache.record_missing_songs("pl1", "Renewed", [self.wish_rows()[0]])
        payload = library_cache.list_missing_songs()
        assert payload["total"] == 1
        assert payload["missing"][0]["playlist_name"] == "Renewed"

    def test_clearing_a_now_fully_matched_playlist(self):
        library_cache.record_missing_songs("pl1", "List", self.wish_rows())
        written = library_cache.record_missing_songs("pl1", "List", [])
        payload = library_cache.list_missing_songs()
        assert written == 0
        assert payload["total"] == 0
        assert payload["playlists"] == []

    def test_summaries_count_whole_table_not_page(self):
        library_cache.record_missing_songs("pl1", "One", self.wish_rows())
        library_cache.record_missing_songs("pl2", "Two", [self.wish_rows()[0]])
        page = library_cache.list_missing_songs(limit=1)
        assert page["total"] == 3
        assert len(page["missing"]) == 1
        assert {s["playlist_ref"]: s["count"] for s in page["playlists"]} == {"pl1": 2, "pl2": 1}


class TestImportRoutes:
    """Route wiring with canned fetch/match answers: writes and errors."""


def canned_payload():
    return {
        "playlist_id": "pl" + "x" * 20,
        "playlist_url": "https://open.spotify.com/playlist/" + "pl" + "x" * 20,
        "source": "api",
        "name": "Canned list",
        "owner": "Owner",
        "tracks": [dict(SPOTIFY_ROW)],
        "warnings": [],
    }


CANNED_MATCH = {
    "rating_key": "1",
    "title": "Fake Plastic Trees",
    "artist": "Radiohead",
    "album": "The Bends",
    "duration_ms": 290000,
    "position": 1,
    "match_kind": "exact",
}
CANNED_MISSING = {
    "position": 5,
    "title": "Gone",
    "album": "AL",
    "artist": "A",
    "duration_ms": 1,
    "spotify_track_id": "",
    "spotify_url": "",
}


def plex_rows(*specs: tuple) -> Mock:
    """Fake Plex client whose search_tracks answers with Track objects."""

    def search(query: str, **_):
        return [
            Track(rating_key=k, title=t, artist=a, album=al, duration_ms=d)
            for k, t, a, al, d in specs
            if t.split()[0].lower() in query.lower()
        ]

    mock = Mock()
    mock.search_tracks = search
    return mock


class TestMatchLive:
    """The live-Plex fallback: whatever Plex's search answers, keyed locally."""

    def test_exact_key_beats_a_near_candidate(self):
        plex = plex_rows(
            ("9", "Fake Plastic Trees", "Radiohead", "Other Album", 300000),
            ("1", "Fake Plastic Trees", "Radiohead", "The Bends", 290000),
        )
        matched, missing = spotify_import.match_live([dict(SPOTIFY_ROW)], plex)
        assert matched[0]["rating_key"] == "1"
        assert matched[0]["match_kind"] == "search"
        assert not missing

    def test_duration_picks_the_closer_version(self):
        plex = plex_rows(
            ("9", "Fake Plastic Trees", "Radiohead", "The Bends Deluxe", 320000),
            ("1", "Fake Plastic Trees", "Radiohead", "The Bends", 290001),
        )
        matched, _ = spotify_import.match_live([dict(SPOTIFY_ROW)], plex)
        assert matched[0]["rating_key"] == "1"

    def test_unsearchable_row_is_missing_not_fatal(self):
        plex = plex_rows()
        matched, missing = spotify_import.match_live([dict(SPOTIFY_ROW)], plex)
        assert not matched
        assert [m["title"] for m in missing] == ["Fake Plastic Trees"]

    def test_failing_search_leaves_the_row_missing(self):
        plex = Mock()
        plex.search_tracks = Mock(side_effect=RuntimeError("server went away"))
        matched, missing = spotify_import.match_live([dict(SPOTIFY_ROW)], plex)
        assert not matched
        assert [m["title"] for m in missing] == ["Fake Plastic Trees"]


class TestPreviewRoute:
    def _client(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        return TestClient(app)

    def test_preview_reports_match_rungs(self, monkeypatch):
        monkeypatch.setattr("backend.main.get_config", lambda: MagicMock())
        monkeypatch.setattr("backend.main.fetch_playlist", lambda *_: canned_payload())
        monkeypatch.setattr("backend.main.library_cache.has_cached_tracks", lambda: True)
        monkeypatch.setattr(
            "backend.main.match_cached", lambda tracks: ([CANNED_MATCH], [CANNED_MISSING])
        )
        response = self._client().post(
            "/api/import/spotify/preview", json={"playlist_ref": "spotify:playlist:abc"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "api"
        assert body["playlist_name"] == "Canned list"
        assert body["matched"][0]["match_kind"] == "exact"
        assert body["missing"][0]["playlist_ref"] == canned_payload()["playlist_id"]

    def test_unconfigured_path_is_a_named_502(self, monkeypatch):
        monkeypatch.setattr(
            "backend.main.get_config",
            lambda: AppConfig(
                plex=PlexConfig(url="", token=""),
                llm=LLMConfig(provider="gemini", model_analysis="", model_generation=""),
            ),
        )
        response = self._client().post(
            "/api/import/spotify/preview", json={"playlist_ref": "not-a-playlist!"}
        )
        assert response.status_code == 502
        assert "playlist id" in response.json()["detail"]


class TestImportRoute:
    def _client(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        return TestClient(app)

    def _patch_fetch(self, monkeypatch):
        monkeypatch.setattr("backend.main.get_config", lambda: MagicMock())
        monkeypatch.setattr("backend.main.fetch_playlist", lambda *_: canned_payload())
        monkeypatch.setattr("backend.main.library_cache.has_cached_tracks", lambda: True)

    def test_nothing_matched_writes_the_wish_only(self, monkeypatch):
        plex = MagicMock()
        monkeypatch.setattr("backend.main.get_plex_client", lambda: plex)
        self._patch_fetch(monkeypatch)
        monkeypatch.setattr(
            "backend.main.match_cached", lambda tracks: ([], [CANNED_MISSING])
        )
        monkeypatch.setattr(
            "backend.library_cache.record_missing_songs", lambda *_: 1
        )
        monkeypatch.setattr(
            "backend.library_cache.save_result", lambda *args, **kwargs: "abcd1234"
        )
        response = self._client().post(
            "/api/import/spotify", json={"playlist_ref": "spotify:playlist:abc"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert body["matched_count"] == 0
        assert body["missing_recorded"] == 1
        assert body["result_id"] == "abcd1234"
        plex.create_playlist.assert_not_called()

    def test_matched_rows_become_one_deduped_playlist(self, monkeypatch):
        plex = MagicMock()
        plex.create_playlist.return_value = {
            "success": True,
            "playlist_id": "99",
            "playlist_url": "http://plex/web/99",
            "tracks_added": 1,
            "tracks_skipped": 0,
            "owner": "Owner",
            "copies": [],
        }
        monkeypatch.setattr("backend.main.get_plex_client", lambda: plex)
        self._patch_fetch(monkeypatch)
        # The same song twice in the playlist: one Plex entry.
        monkeypatch.setattr(
            "backend.main.match_cached",
            lambda tracks: ([CANNED_MATCH, dict(CANNED_MATCH, position=7)], []),
        )
        monkeypatch.setattr("backend.library_cache.record_missing_songs", lambda *_: 0)
        monkeypatch.setattr(
            "backend.library_cache.save_result", lambda *args, **kwargs: "abcd1234"
        )
        monkeypatch.setattr("backend.library_cache.link_result_playlist", lambda *_: True)
        response = self._client().post(
            "/api/import/spotify", json={"playlist_ref": "spotify:playlist:abc"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["matched_count"] == 1
        assert body["playlist_id"] == "99"
        assert body["matched"][1]["position"] == 7
        plex.create_playlist.assert_called_once()

    def test_import_needs_connected_plex_before_fetch(self, monkeypatch):
        monkeypatch.setattr("backend.main.get_plex_client", lambda: None)
        response = self._client().post("/api/import/spotify", json={"playlist_ref": "x"})
        assert response.status_code == 503


class TestWishListRoute:
    def _client(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        return TestClient(app)

    def test_wish_list_survives_the_import(self, temp_db, monkeypatch):
        written = library_cache.record_missing_songs(
            "plwish", "My list", [dict(CANNED_MISSING)]
        )
        assert written == 1
        response = self._client().get("/api/import/spotify/missing?limit=5")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["missing"][0]["title"] == "Gone"
        assert body["playlists"] == [
            {"playlist_ref": "plwish", "playlist_name": "My list", "count": 1}
        ]
