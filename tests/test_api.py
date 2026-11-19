"""Tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from fastapi.testclient import TestClient

from backend.models import DefaultsConfig, PlexUserInfo


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    # Import here to avoid module-level import issues
    from backend.main import app
    return TestClient(app)


def create_mock_config(
    plex_url="http://test:32400",
    plex_token="token",
    music_library="Music",
    llm_provider="anthropic",
    llm_api_key="key",
    model_analysis="claude-sonnet-4-5",
    model_generation="claude-haiku-4-5",
    track_count=25,
    ollama_url="http://localhost:11434",
    custom_url="",
    custom_context_window=32768,
):
    """Create a properly structured mock config."""
    mock = MagicMock()
    mock.plex.url = plex_url
    mock.plex.token = plex_token
    mock.plex.music_library = music_library
    mock.llm.provider = llm_provider
    mock.llm.api_key = llm_api_key
    mock.llm.model_analysis = model_analysis
    mock.llm.model_generation = model_generation
    mock.llm.ollama_url = ollama_url
    mock.llm.custom_url = custom_url
    mock.llm.custom_context_window = custom_context_window
    mock.defaults = DefaultsConfig(track_count=track_count)
    return mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_status(self, client):
        """Should return health status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_plex_client") as mock_plex:
                mock_config.return_value = create_mock_config()
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert data["status"] == "healthy"

    def test_health_check_shows_plex_status(self, client):
        """Should show Plex connection status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_plex_client") as mock_plex:
                mock_config.return_value = create_mock_config()
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "plex_connected" in data
                assert data["plex_connected"] is True

    def test_health_check_shows_llm_status(self, client):
        """Should show LLM configuration status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_plex_client") as mock_plex:
                mock_config.return_value = create_mock_config(llm_api_key="key")
                mock_plex.return_value = None  # No Plex client

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "llm_configured" in data
                assert data["llm_configured"] is True


class TestConfigEndpoints:
    """Tests for configuration endpoints."""

    def test_get_config_returns_safe_values(self, client):
        """GET /api/config should return config without secrets."""
        with patch("backend.main.get_config") as mock_get_config:
            with patch("backend.main.get_plex_client") as mock_plex:
                mock_get_config.return_value = create_mock_config(
                    plex_url="http://test:32400",
                    plex_token="secret-token",
                    llm_provider="anthropic",
                    llm_api_key="secret-api-key",
                )
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/config")

                assert response.status_code == 200
                data = response.json()

                # Should include URL but not token
                assert data["plex_url"] == "http://test:32400"
                assert "secret-token" not in str(data)

                # Should show provider but not API key
                assert data["llm_provider"] == "anthropic"
                assert "api_key" not in data
                assert "secret-api-key" not in str(data)

    def test_post_config_validates_plex_url(self, client):
        """POST /api/config should validate Plex URL format."""
        with patch("backend.main.update_config_values") as mock_update:
            with patch("backend.main.get_plex_client") as mock_plex:
                with patch("backend.main.init_plex_client"):
                    mock_config = create_mock_config(plex_url="http://new-server:32400")
                    mock_update.return_value = mock_config
                    mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                    response = client.post(
                        "/api/config",
                        json={"plex_url": "http://new-server:32400"}
                    )

                    assert response.status_code == 200

    def test_post_config_updates_llm_provider(self, client):
        """POST /api/config should allow changing LLM provider."""
        with patch("backend.main.update_config_values") as mock_update:
            with patch("backend.main.get_plex_client") as mock_plex:
                with patch("backend.main.init_plex_client"):
                    mock_config = create_mock_config(llm_provider="openai")
                    mock_update.return_value = mock_config
                    mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                    response = client.post(
                        "/api/config",
                        json={"llm_provider": "openai"}
                    )

                    assert response.status_code == 200

    def test_local_provider_configured_without_api_key(self, client):
        """A local provider URL alone counts as configured, on both endpoints."""
        with patch("backend.main.get_config") as mock_get_config:
            with patch("backend.main.get_plex_client") as mock_plex:
                mock_get_config.return_value = create_mock_config(
                    llm_provider="ollama", llm_api_key=""
                )
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                health = client.get("/api/health").json()
                config = client.get("/api/config").json()

        assert health["llm_configured"] is True
        assert config["llm_configured"] is True
        assert config["llm_api_key_set"] is False
        assert config["is_local_provider"] is True


class TestIndexPage:
    """Tests for index page serving."""

    def test_index_returns_response(self, client):
        """Should return some response for root path."""
        response = client.get("/")
        # Either returns HTML or JSON message
        assert response.status_code == 200


class TestOllamaEndpoints:
    """Tests for Ollama API endpoints."""

    def test_ollama_status_connected(self, client):
        """GET /api/ollama/status should return connected status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=True,
                    model_count=3,
                    error=None,
                )

                response = client.get("/api/ollama/status")

                assert response.status_code == 200
                data = response.json()
                assert data["connected"] is True
                assert data["model_count"] == 3

    def test_ollama_status_not_connected(self, client):
        """GET /api/ollama/status should return error when not connected."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=False,
                    model_count=0,
                    error="Connection refused",
                )

                response = client.get("/api/ollama/status")

                assert response.status_code == 200
                data = response.json()
                assert data["connected"] is False
                assert data["error"] == "Connection refused"

    def test_ollama_models_list(self, client):
        """GET /api/ollama/models should return list of models."""
        from backend.models import OllamaModel, OllamaModelsResponse

        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.list_ollama_models") as mock_models:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_models.return_value = OllamaModelsResponse(
                    models=[
                        OllamaModel(name="llama3:8b", size=4661224676, modified_at="2024-01-15T00:00:00Z"),
                        OllamaModel(name="mistral:latest", size=3825819904, modified_at="2024-01-14T00:00:00Z"),
                    ],
                    error=None,
                )

                response = client.get("/api/ollama/models")

                assert response.status_code == 200
                data = response.json()
                assert len(data["models"]) == 2
                assert data["models"][0]["name"] == "llama3:8b"

    def test_ollama_model_info(self, client):
        """GET /api/ollama/model-info should return model details."""
        from backend.models import OllamaModelInfo

        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_model_info") as mock_info:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_info.return_value = OllamaModelInfo(
                    name="llama3:8b",
                    context_window=8192,
                    parameter_size="8B",
                )

                response = client.get("/api/ollama/model-info?model=llama3:8b")

                assert response.status_code == 200
                data = response.json()
                assert data["name"] == "llama3:8b"
                assert data["context_window"] == 8192

    def test_ollama_model_info_not_found(self, client):
        """GET /api/ollama/model-info should return 404 for unknown model."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_model_info") as mock_info:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_info.return_value = None  # Model not found

                response = client.get("/api/ollama/model-info?model=nonexistent")

                assert response.status_code == 404

    def test_ollama_status_with_custom_url(self, client):
        """GET /api/ollama/status should accept custom URL parameter."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=True,
                    model_count=1,
                    error=None,
                )

                response = client.get("/api/ollama/status?url=http://custom-host:11434")

                assert response.status_code == 200
                # Verify the custom URL was passed
                mock_status.assert_called_once_with("http://custom-host:11434")


class TestLibrarySyncEndpoint:
    """Tests for the local library sync endpoint."""

    def test_sync_requires_plex(self, client):
        """POST /api/library/sync should 503 when Plex is not connected."""
        with patch("backend.main.get_plex_client", return_value=None):
            response = client.post("/api/library/sync")

        assert response.status_code == 503

    def test_sync_starts_background_sync(self, client):
        """POST /api/library/sync kicks off a background sync and reports it."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            with patch(
                "backend.main.library_cache.get_sync_progress",
                return_value={"is_syncing": False},
            ):
                with patch("backend.main.library_cache.sync_library"):
                    response = client.post("/api/library/sync")

        assert response.status_code == 200
        assert response.json() == {"started": True, "blocking": False}

    def test_sync_conflicts_when_already_running(self, client):
        """POST /api/library/sync should 409 while a sync is already running."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            with patch(
                "backend.main.library_cache.get_sync_progress",
                return_value={"is_syncing": True},
            ):
                response = client.post("/api/library/sync")

        assert response.status_code == 409


class TestRatingsAndReasoningEndpoints:
    """Per-account stars + curation memory: both stores are read back, never re-scanned here."""

    @pytest.fixture
    def scanned(self):
        """Canned ``rating_sync`` state, so no test depends on this machine's DB."""
        return {
            "total_tracks": 100,
            "synced_users": ["Becky"],
            "users": [
                {
                    "user_key": "Becky",
                    "rated_count": 7,
                    "total_seen": 100,
                    "last_sync_at": "2026-10-31T00:00:00",
                }
            ],
        }

    def test_ratings_status_names_the_scanned_accounts(self, client, scanned):
        """GET /api/library/ratings/status is pure SQLite: no Plex handshake needed."""
        with patch("backend.main.library_cache.user_ratings_status", return_value=scanned):
            response = client.get("/api/library/ratings/status")

        assert response.status_code == 200
        assert response.json() == {
            "total_tracks": 100,
            "users": [
                {
                    "user_key": "Becky",
                    "rated_count": 7,
                    "total_seen": 100,
                    "last_sync_at": "2026-10-31T00:00:00",
                }
            ],
            "synced_users": ["Becky"],
        }

    def test_refresh_requires_plex_connected(self, client, scanned):
        """A rescan is a Plex walk, so it 503s before any account is touched."""
        from backend.plex_client import PlexQueryError

        with (
            patch("backend.main.get_plex_client", return_value=None),
            patch("backend.main.library_cache.user_ratings_status", return_value=scanned),
            patch("backend.main.library_cache.sync_user_ratings", side_effect=PlexQueryError("would crash")),
        ):
            response = client.post("/api/library/ratings/refresh", json={})

        assert response.status_code == 503

    def test_refresh_rescans_every_already_cached_account(self, client, scanned):
        """An empty body means the accounts already stored, not the whole household."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True

        with (
            patch("backend.main.get_plex_client", return_value=mock_plex),
            patch("backend.main.library_cache.user_ratings_status", return_value=scanned),
            patch("backend.main.library_cache.sync_user_ratings", return_value=7),
        ):
            response = client.post("/api/library/ratings/refresh", json={})

        assert response.status_code == 200
        assert response.json()["refreshed"] == ["Becky"]
        assert response.json()["errors"] == {}
        assert response.json()["status"]["synced_users"] == ["Becky"]

    def test_refresh_targets_only_the_named_account(self, client, scanned):
        """Naming accounts in the body keeps other playlists' wishes' owners alone."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True

        with (
            patch("backend.main.get_plex_client", return_value=mock_plex),
            patch("backend.main.library_cache.user_ratings_status", return_value=scanned),
            patch("backend.main.library_cache.sync_user_ratings", return_value=3),
        ):
            response = client.post("/api/library/ratings/refresh", json={"users": ["The Boys"]})

        assert response.status_code == 200
        assert response.json()["refreshed"] == ["The Boys"]

    def test_refresh_skips_the_admin_account(self, client, scanned):
        """``__owner__`` has no rows to rewrite: its stars are every track's column."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True

        with (
            patch("backend.main.get_plex_client", return_value=mock_plex),
            patch("backend.main.library_cache.user_ratings_status", return_value=scanned),
            patch("backend.main.library_cache.sync_user_ratings") as synced,
        ):
            response = client.post("/api/library/ratings/refresh", json={"users": ["__owner__"]})

        assert response.status_code == 200
        assert response.json()["refreshed"] == []
        synced.assert_not_called()

    def test_a_failed_rescan_is_named_not_counted_as_zero(self, client, scanned):
        """A crashed scan must not read as "this member rated nothing"."""
        from backend.plex_client import PlexQueryError

        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True

        with (
            patch("backend.main.get_plex_client", return_value=mock_plex),
            patch("backend.main.library_cache.user_ratings_status", return_value=scanned),
            patch("backend.main.library_cache.sync_user_ratings", side_effect=PlexQueryError("Unknown Plex user")),
        ):
            response = client.post("/api/library/ratings/refresh", json={"users": ["Becky"]})

        assert response.status_code == 200
        assert response.json()["refreshed"] == []
        assert response.json()["errors"] == {"Becky": "Unknown Plex user"}
        assert response.json()["status"]["synced_users"] == ["Becky"]

    def test_reasoning_status_reports_the_curation_memory(self, client):
        """GET /api/reasoning/status reads the other DB, untouched by a library sync."""
        with patch(
            "backend.main.agent_reasoning.reasoning_status",
            return_value={
                "runs": 2,
                "selections": 40,
                "tracks_with_reasons": 33,
                "tagged_tracks": 12,
                "tags": [{"tag": "deep cut", "count": 5}],
                "last_run_at": "2026-10-30T00:00:00",
            },
        ):
            response = client.get("/api/reasoning/status")

        assert response.status_code == 200
        assert response.json() == {
            "runs": 2,
            "selections": 40,
            "tracks_with_reasons": 33,
            "tagged_tracks": 12,
            "tags": [{"tag": "deep cut", "count": 5}],
            "last_run_at": "2026-10-30T00:00:00",
            "error": None,
        }


class TestFilterPreviewRelaxation:
    """POST /api/filter/preview widens the cache filters it cannot satisfy."""

    @staticmethod
    def _narrow_cache(**facets):
        """A cache that has plenty only once the floors and windows are off."""
        if any(
            facets.get(name)
            for name in ("min_rating", "max_rating", "min_plays", "max_plays")
        ):
            return 4
        if any(facets.get(name) for name in ("min_year", "max_year")):
            return 6
        return 2500

    def _preview(self, client, payload, counted=None):
        """Post a preview against a cache whose counts depend only on the facets."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        counter = counted or self._narrow_cache

        with patch("backend.main.get_config", return_value=create_mock_config()):
            with patch("backend.main.get_plex_client", return_value=mock_plex):
                with patch("backend.main.library_cache.has_cached_tracks", return_value=True):
                    with patch(
                        "backend.library_cache.count_tracks_by_filters",
                        side_effect=counter,
                    ):
                        return client.post("/api/filter/preview", json=payload)

    def test_preview_relaxes_and_names_the_tiers(self, client):
        """The pool a run will use is reported, along with the tiers given up."""
        response = self._preview(
            client,
            {
                "prompt": "soundtracks",
                "min_rating": 7,
                "min_year": 2015,
                "max_year": 2018,
                "track_count": 25,
                "max_tracks_to_ai": 200,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["matching_as_asked"] == 4
        assert data["matching_tracks"] == 2500
        assert data["relaxed_tiers"] == [
            "no rating or play floors",
            "no year or duration windows",
        ]
        assert data["relaxed_to"] == "no year or duration windows"

    def test_preview_gives_up_a_ban_before_the_floors(self, client):
        """Bans are given up before the rating floors and year windows."""
        response = self._preview(
            client,
            {
                "prompt": "modern prog, not the 2010s",
                "exclude_decades": ["2010s"],
                "min_rating": 7,
                "min_year": 2015,
                "max_year": 2018,
                "track_count": 25,
                "max_tracks_to_ai": 200,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["matching_as_asked"] == 4
        assert data["matching_tracks"] == 2500
        assert data["relaxed_tiers"] == [
            "no bans",
            "no rating or play floors",
            "no year or duration windows",
        ]
        assert data["relaxed_to"] == "no year or duration windows"

    def test_preview_reports_nothing_given_up_when_filters_suffice(self, client):
        """A pool the filters as given already fills is reported unchanged."""
        response = self._preview(
            client,
            {"prompt": "best of the 90s", "genres": ["Jazz"], "track_count": 25},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["matching_as_asked"] == 2500
        assert data["matching_tracks"] == 2500
        assert data["relaxed_tiers"] == []
        assert data["relaxed_to"] == ""

    def test_preview_never_relaxes_the_plex_branch(self, client):
        """With no cache the filters stay as asked, since Plex cannot count tiers."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.count_tracks_by_filters.return_value = 31

        with patch("backend.main.get_config", return_value=create_mock_config()):
            with patch("backend.main.get_plex_client", return_value=mock_plex):
                with patch("backend.main.library_cache.has_cached_tracks", return_value=False):
                    response = client.post(
                        "/api/filter/preview",
                        json={"prompt": "obscure", "genres": ["Prog"], "track_count": 50},
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["matching_tracks"] == 31
        assert data["matching_as_asked"] == 31
        assert data["relaxed_tiers"] == []
        assert data["relaxed_to"] == ""

    def test_specific_request_stays_specific(self, client):
        """A pool that already fills the playlist keeps its genre whitelist."""
        from backend.library_cache import IDENTITY_TIER

        def counted(**facets):
            return 40 if facets["genres"] else 4000

        response = self._preview(
            client,
            {
                "prompt": "prog opera",
                "genres": ["Prog"],
                "track_count": 25,
                "max_tracks_to_ai": 200,
            },
            counted=counted,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["matching_as_asked"] == 40
        assert data["matching_tracks"] == 40
        assert IDENTITY_TIER not in data["relaxed_tiers"]


class TestFilterPreviewEndpoint:

    def _post_preview(self, client, payload):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.count_tracks_by_filters.return_value = payload.pop("_matching")
        with patch("backend.main.get_config", return_value=create_mock_config()):
            with patch("backend.main.get_plex_client", return_value=mock_plex):
                return client.post("/api/filter/preview", json=payload)

    def test_specific_prompt_sends_fewer_tracks(self, client):
        response = self._post_preview(client, {
            "_matching": 10000,
            "genres": ["Jazz"],
            "decades": [],
            "track_count": 25,
            "max_tracks_to_ai": 500,
            "prompt": "jazz",
        })

        assert response.status_code == 200
        data = response.json()
        assert 0 < data["tracks_to_send"] < 500
        assert data["tracks_to_send"] < data["matching_tracks"]

    def test_no_signal_uses_full_ceiling(self, client):
        from backend.generator import PRE_RANK_LIMIT

        response = self._post_preview(client, {
            "_matching": 10000,
            "genres": [],
            "decades": [],
            "track_count": 25,
            "max_tracks_to_ai": 500,
        })

        assert response.status_code == 200
        assert response.json()["tracks_to_send"] == PRE_RANK_LIMIT

    def test_pool_size_caps_tracks_to_send(self, client):
        response = self._post_preview(client, {
            "_matching": 40,
            "genres": ["Jazz"],
            "decades": [],
            "track_count": 25,
            "max_tracks_to_ai": 500,
            "prompt": "jazz",
        })

        assert response.status_code == 200
        assert response.json()["tracks_to_send"] == 40

    def test_empty_library_sends_zero(self, client):
        response = self._post_preview(client, {
            "_matching": 0,
            "genres": ["Jazz"],
            "decades": [],
            "track_count": 25,
            "max_tracks_to_ai": 500,
            "prompt": "jazz",
        })

        assert response.status_code == 200
        assert response.json()["tracks_to_send"] == 0


class TestAlbumArtEndpoint:
    """GET /api/art/{rating_key} proxies album art from Plex."""

    @staticmethod
    def _http_client(content=b"JPEGDATA", status=200, content_type="image/jpeg"):
        response = MagicMock()
        response.status_code = status
        response.content = content
        response.headers = {"content-type": content_type}

        http = MagicMock()
        http.__aenter__ = AsyncMock(return_value=http)
        http.__aexit__ = AsyncMock(return_value=False)
        http.get = AsyncMock(return_value=response)
        return http

    def _get(self, client, http):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.get_thumb_path.return_value = "/library/metadata/1/thumb/2"
        with patch("backend.main.get_config", return_value=create_mock_config()):
            with patch("backend.main.get_plex_client", return_value=mock_plex):
                with patch("backend.main.httpx.AsyncClient", return_value=http):
                    response = client.get("/api/art/123")
        return response, mock_plex

    def test_art_is_fetched_with_one_item_lookup(self, client):
        response, mock_plex = self._get(client, self._http_client())

        assert response.status_code == 200
        assert response.content == b"JPEGDATA"
        # A single Plex item lookup, not the old validate-then-thumb double fetch.
        mock_plex.get_thumb_path.assert_called_once_with("123")
        mock_plex.get_track_by_key.assert_not_called()

    def test_art_is_cacheable(self, client):
        response, _ = self._get(client, self._http_client())
        assert "max-age" in response.headers.get("cache-control", "")

    def test_missing_thumb_returns_404(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.get_thumb_path.return_value = None
        with patch("backend.main.get_config", return_value=create_mock_config()):
            with patch("backend.main.get_plex_client", return_value=mock_plex):
                response = client.get("/api/art/123")

        assert response.status_code == 404


class TestPlaylistUserEndpoints:
    """Tests for choosing which Plex account a playlist is written to."""

    def test_users_endpoint_lists_selectable_accounts(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.get_users.return_value = [
            PlexUserInfo(key="__owner__", title="freem716", is_owner=True),
            PlexUserInfo(key="amy", title="Amy", user_id=42),
        ]

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.get("/api/plex/users")

        assert response.status_code == 200
        users = response.json()
        assert [u["key"] for u in users] == ["__owner__", "amy"]
        assert users[0]["is_owner"] is True
        assert users[1]["is_managed"] is False

    def test_users_endpoint_requires_plex_connection(self, client):
        with patch("backend.main.get_plex_client", return_value=None):
            response = client.get("/api/plex/users")

        assert response.status_code == 503

    def test_playlists_endpoint_forwards_user(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.get_playlists.return_value = []

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.get("/api/plex/playlists", params={"user": "amy"})

        assert response.status_code == 200
        mock_plex.get_playlists.assert_called_once_with("amy")

    def test_playlists_endpoint_defaults_to_connected_account(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.get_playlists.return_value = []

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.get("/api/plex/playlists")

        assert response.status_code == 200
        mock_plex.get_playlists.assert_called_once_with("__owner__")

    def test_save_playlist_forwards_user_and_copy_to(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {
            "success": True,
            "playlist_id": "77",
            "playlist_url": "http://plex/web/index#!/playlist/77",
            "tracks_added": 2,
            "tracks_skipped": 0,
            "owner": "Amy",
            "copies": [
                {
                    "user": "Leo",
                    "success": True,
                    "playlist_id": "78",
                    "playlist_url": "http://plex/web/index#!/playlist/78",
                    "tracks_added": 2,
                    "tracks_skipped": 0,
                }
            ],
        }

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.post("/api/playlist", json={
                "name": "Road Trip",
                "rating_keys": ["1", "2"],
                "user": "amy",
                "copy_to": ["leo"],
            })

        assert response.status_code == 200
        args = mock_plex.create_playlist.call_args[0]
        assert args[0] == "Road Trip"
        assert args[3] == "amy"
        assert args[4] == ["leo"]

        body = response.json()
        assert body["owner"] == "Amy"
        assert body["copies"][0]["user"] == "Leo"
        assert body["copies"][0]["playlist_id"] == "78"

    def test_save_playlist_defaults_to_the_connected_account(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {"success": True, "playlist_id": "77"}

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.post("/api/playlist", json={
                "name": "Road Trip",
                "rating_keys": ["1", "2"],
            })

        assert response.status_code == 200
        args = mock_plex.create_playlist.call_args[0]
        assert args[3] == "__owner__"
        assert args[4] == []
        assert response.json()["copies"] == []

    def test_save_playlist_deduplicates_copy_targets(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {"success": True, "playlist_id": "77"}

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.post("/api/playlist", json={
                "name": "Road Trip",
                "rating_keys": ["1"],
                "user": "",
                "copy_to": ["amy", "amy", "  ", ""],
            })

        assert response.status_code == 200
        args = mock_plex.create_playlist.call_args[0]
        assert args[3] == "__owner__"
        assert args[4] == ["amy"]

    def test_save_playlist_links_history_row_to_the_created_playlist(self, client):
        """A request carrying result_id stamps the new Plex playlist on that row."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {
            "success": True,
            "playlist_id": "77",
            "playlist_url": "http://plex/web/index#!/playlist/77",
            "tracks_added": 2,
            "tracks_skipped": 0,
            "owner": "Amy",
        }

        with patch("backend.main.get_plex_client", return_value=mock_plex), patch(
            "backend.main.library_cache.link_result_playlist"
        ) as mock_link:
            mock_link.return_value = True
            response = client.post("/api/playlist", json={
                "name": "Road Trip",
                "rating_keys": ["1", "2"],
                "user": "amy",
                "result_id": "abc12345",
            })

        assert response.status_code == 200
        assert mock_link.call_args[0] == (
            "abc12345",
            "77",
            "Road Trip",
            "Amy",
            "http://plex/web/index#!/playlist/77",
        )
        assert response.json()["linked_result_id"] == "abc12345"

    def test_save_playlist_link_reports_an_unknown_result_id(self, client):
        """An absent history row leaves the playlist unlinked, not an error."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {"success": True, "playlist_id": "77"}

        with patch("backend.main.get_plex_client", return_value=mock_plex), patch(
            "backend.main.library_cache.link_result_playlist"
        ) as mock_link:
            mock_link.return_value = False
            response = client.post("/api/playlist", json={
                "name": "Road Trip",
                "rating_keys": ["1"],
                "result_id": "abcdef01",
            })

        assert response.status_code == 200
        assert response.json()["linked_result_id"] is None

    def test_save_playlist_skips_link_without_result_id(self, client):
        """Saving without a result_id must not touch the results table."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {"success": True, "playlist_id": "77"}

        with patch("backend.main.get_plex_client", return_value=mock_plex), patch(
            "backend.main.library_cache.link_result_playlist"
        ) as mock_link:
            response = client.post("/api/playlist", json={
                "name": "Road Trip",
                "rating_keys": ["1", "2"],
            })

        assert response.status_code == 200
        mock_link.assert_not_called()
        assert response.json()["linked_result_id"] is None

    def test_save_playlist_result_id_rejects_a_malformed_value(self, client):
        """A non-hex result_id is refused by the request model."""
        response = client.post(
            "/api/playlist",
            json={"name": "Road Trip", "rating_keys": ["1"], "result_id": "not-hex"},
        )

        assert response.status_code == 422

    def test_save_playlist_copy_to_the_owner_is_dropped(self, client):
        """A copy aimed at the owning account would be a second identical playlist."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {"success": True, "playlist_id": "77"}

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.post(
                "/api/playlist",
                json={
                    "name": "Road Trip",
                    "rating_keys": ["1"],
                    "user": "amy",
                    "copy_to": ["amy"],
                },
            )

        assert response.status_code == 200
        assert mock_plex.create_playlist.call_args[0][4] == []

    def test_save_playlist_copy_to_owner_key_dropped_for_default_user(self, client):
        """The active account owns the primary playlist, so __owner__ is not copied."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {"success": True, "playlist_id": "77"}

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.post(
                "/api/playlist",
                json={"name": "Road Trip", "rating_keys": ["1"], "copy_to": ["__owner__"]},
            )

        assert response.status_code == 200
        assert mock_plex.create_playlist.call_args[0][3] == "__owner__"
        assert mock_plex.create_playlist.call_args[0][4] == []

    def test_save_playlist_other_copies_are_kept(self, client):
        """Only the owning key is dropped; other accounts still get their own copy."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.create_playlist.return_value = {"success": True, "playlist_id": "77"}

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.post(
                "/api/playlist",
                json={
                    "name": "Road Trip",
                    "rating_keys": ["1"],
                    "user": "The Boys",
                    "copy_to": ["Becky", "The Boys", "Becky"],
                },
            )

        assert response.status_code == 200
        assert mock_plex.create_playlist.call_args[0][3] == "The Boys"
        assert mock_plex.create_playlist.call_args[0][4] == ["Becky"]

    def test_update_playlist_forwards_user(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.update_playlist.return_value = {
            "success": True,
            "tracks_added": 1,
            "tracks_skipped": 0,
            "duplicates_skipped": 0,
            "playlist_url": "http://plex/web/index#!/playlist/10",
            "owner": "Amy",
        }

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.post("/api/playlist/update", json={
                "playlist_id": "10",
                "rating_keys": ["101"],
                "mode": "append",
                "user": "amy",
            })

        assert response.status_code == 200
        args = mock_plex.update_playlist.call_args[0]
        assert args[3] == ""        # description keeps its default
        assert args[4] == "amy"
        assert response.json()["owner"] == "Amy"

    def test_update_playlist_for_owner_placeholder(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.update_playlist.return_value = {
            "success": True,
            "tracks_added": 1,
            "tracks_skipped": 0,
            "duplicates_skipped": 0,
            "owner": "freem716",
        }

        with patch("backend.main.get_plex_client", return_value=mock_plex):
            response = client.post("/api/playlist/update", json={
                "playlist_id": "__scratch__",
                "rating_keys": ["101"],
                "mode": "replace",
            })

        assert response.status_code == 200
        assert mock_plex.update_playlist.call_args[0][4] == "__owner__"


# ===================================================================
# Section 12b: Year / play / ban facets are forwarded by the routes
# ===================================================================


class TestYearWindowFacet:
    """POST /api/filter/preview must forward the year bounds it now carries."""

    def _preview(self, client, payload, matching):
        """Force the Plex branch so the count never depends on the local DB."""
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.count_tracks_by_filters.return_value = matching
        with patch("backend.main.get_plex_client", return_value=mock_plex):
            with patch("backend.main.get_config", return_value=create_mock_config()):
                with patch("backend.main.library_cache.has_cached_tracks", return_value=False):
                    with patch("backend.main.compute_track_budget", return_value=matching):
                        return mock_plex, client.post(
                            "/api/filter/preview", json=payload
                        )

    def test_year_window_reaches_plex(self, client):
        """min_year/max_year are forwarded as keyword arguments."""
        mock_plex, response = self._preview(
            client, {"genres": ["Rock"], "min_year": 1990, "max_year": 2010}, 7
        )
        assert response.status_code == 200
        assert mock_plex.count_tracks_by_filters.call_args[1]["min_year"] == 1990
        assert mock_plex.count_tracks_by_filters.call_args[1]["max_year"] == 2010

    def test_plex_receives_no_local_db_read(self, client):
        """With no cache the route must ask Plex, not the SQLite count."""
        mock_plex, response = self._preview(client, {"genres": ["Rock"]}, 4)
        assert response.status_code == 200
        assert mock_plex.count_tracks_by_filters.call_args[1]["min_year"] == 0

    def test_inverted_year_window_is_rejected(self, client):
        """min_year > max_year cannot match anything, so FastAPI says 422."""
        response = client.post(
            "/api/filter/preview",
            json={"genres": ["Rock"], "min_year": 2000, "max_year": 1990},
        )
        assert response.status_code == 422

    def test_negative_year_is_rejected(self, client):
        """Negative years are a client error, not a silent clamp."""
        response = client.post("/api/filter/preview", json={"genres": ["Rock"], "min_year": -5})
        assert response.status_code == 422


class TestPlayWindowFacet:
    """Play-count facets reach Plex and contradicting windows are refused."""

    def _preview(self, client, payload, matching):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.count_tracks_by_filters.return_value = matching
        with patch("backend.main.get_plex_client", return_value=mock_plex):
            with patch("backend.main.get_config", return_value=create_mock_config()):
                with patch("backend.main.library_cache.has_cached_tracks", return_value=False):
                    with patch("backend.main.compute_track_budget", return_value=matching):
                        return mock_plex, client.post(
                            "/api/filter/preview", json=payload
                        )

    def test_play_window_reaches_plex(self, client):
        mock_plex, response = self._preview(
            client, {"genres": ["Rock"], "min_plays": 3, "max_plays": 10}, 9
        )
        assert response.status_code == 200
        assert mock_plex.count_tracks_by_filters.call_args[1]["min_plays"] == 3
        assert mock_plex.count_tracks_by_filters.call_args[1]["max_plays"] == 10

    def test_unheard_flag_reaches_plex(self, client):
        mock_plex, response = self._preview(client, {"genres": ["Rock"], "unheard_only": True}, 5)
        assert response.status_code == 200
        assert mock_plex.count_tracks_by_filters.call_args[1]["unheard_only"] is True

    def test_compilation_exclusion_reaches_plex(self, client):
        mock_plex, response = self._preview(
            client, {"genres": ["Rock"], "exclude_compilations": True}, 5
        )
        assert response.status_code == 200
        assert mock_plex.count_tracks_by_filters.call_args[1]["exclude_compilations"] is True

    def test_unheard_plus_min_plays_is_rejected(self, client):
        """Never-played and at-least-N-plays contradict, so FastAPI says 422."""
        response = client.post(
            "/api/filter/preview",
            json={"genres": ["Rock"], "unheard_only": True, "min_plays": 1},
        )
        assert response.status_code == 422

    def test_inverted_play_window_is_rejected(self, client):
        response = client.post(
            "/api/filter/preview",
            json={"genres": ["Rock"], "min_plays": 8, "max_plays": 2},
        )
        assert response.status_code == 422

    def test_negative_play_count_is_rejected(self, client):
        response = client.post("/api/filter/preview", json={"genres": ["Rock"], "min_plays": -1})
        assert response.status_code == 422


class TestBanFacet:
    """Banned artists and decades are forwarded and reported back."""

    def test_bans_reach_plex(self, client):
        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.count_tracks_by_filters.return_value = 6
        with patch("backend.main.get_plex_client", return_value=mock_plex):
            with patch("backend.main.get_config", return_value=create_mock_config()):
                with patch("backend.main.library_cache.has_cached_tracks", return_value=False):
                    with patch("backend.main.compute_track_budget", return_value=6):
                        response = client.post(
                            "/api/filter/preview",
                            json={
                                "genres": ["Rock"],
                                "exclude_artists": ["Yes"],
                                "exclude_decades": ["1980s"],
                            },
                        )
        assert response.status_code == 200
        forwarded = mock_plex.count_tracks_by_filters.call_args[1]
        assert forwarded["exclude_artists"] == ["Yes"]
        assert forwarded["exclude_decades"] == ["1980s"]


class TestStreamFacets:
    """POST /api/generate/stream must forward the new facets to the generator."""

    def test_generated_playlist_forwards_the_facets(self, client):
        """Every wide-FilterSet field must arrive at the track fetch."""
        from backend.models import Track
        from backend.llm_client import LLMResponse

        mock_plex = MagicMock()
        mock_plex.is_connected.return_value = True
        mock_plex.get_tracks_by_filters.return_value = [
            Track(
                rating_key=f"yr{i}",
                title=f"Track {i}",
                artist="The Yearners",
                album="Years",
                duration_ms=200000,
                year=year,
                genres=["Rock"],
                art_url=f"/api/art/yr{i}",
            )
            for i, year in enumerate(range(1990, 2000))
        ]

        llm_client = MagicMock()
        llm_client.generate.return_value = LLMResponse(
            content="[]", input_tokens=10, output_tokens=5, model="claude-haiku-4-5"
        )
        llm_client.parse_json_response.return_value = [
            {"artist": "The Yearners", "title": f"Track {i}"} for i in range(2)
        ]

        with (
            patch("backend.main.get_plex_client", return_value=mock_plex),
            patch("backend.main.get_llm_client", return_value=llm_client),
            patch("backend.generator.get_plex_client", return_value=mock_plex),
            patch("backend.generator.get_llm_client", return_value=llm_client),
            patch("backend.main.get_config", return_value=create_mock_config()),
            # Force the Plex branch so the fetch cannot read the local SQLite DB
            patch("backend.main.library_cache.has_cached_tracks", return_value=False),
        ):
            response = client.post(
                "/api/generate/stream",
                json={
                    "prompt": "obscure year window",
                    "genres": ["Rock"],
                    "decades": [],
                    "min_year": 1990,
                    "max_year": 2000,
                    "exclude_compilations": True,
                    "min_plays": 2,
                    "max_plays": 0,
                    "unheard_only": False,
                    "exclude_artists": ["Omitted"],
                    "exclude_decades": ["1970s"],
                },
            )
            assert response.status_code == 200

        forwarded = mock_plex.get_tracks_by_filters.call_args[1]
        assert forwarded["min_year"] == 1990
        assert forwarded["max_year"] == 2000
        assert forwarded["min_plays"] == 2
        assert forwarded["exclude_compilations"] is True
        assert forwarded["exclude_artists"] == ["Omitted"]
        assert forwarded["exclude_decades"] == ["1970s"]
        assert forwarded["unheard_only"] is False

    def test_contradicting_stream_windows_are_rejected(self, client):
        """The generate route shares the same validators as the preview route."""
        response = client.post(
            "/api/generate/stream",
            json={
                "prompt": "hidden gems that got played",
                "genres": ["Rock"],
                "decades": [],
                "unheard_only": True,
                "min_plays": 4,
            },
        )
        assert response.status_code == 422
