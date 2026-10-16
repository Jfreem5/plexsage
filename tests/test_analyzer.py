"""Tests for prompt and track analysis."""

import json
import pytest
from unittest.mock import MagicMock, patch

from backend.prompt_classifier import PromptClassification

# Forces analyzer.analyze_prompt() down the LLM fallback path so the LLM
# handling can be tested independently of the local classifier.
LOW_CONFIDENCE = PromptClassification(
    genres=[], decades=[], confidence=0.0, method="none", reasoning="forced fallback"
)


def _plex_stats(genres, decades):
    return {"genres": genres, "decades": decades}


class TestPromptAnalysis:
    """Tests for prompt analysis."""

    def test_analyze_prompt_uses_rules_without_llm(self, mocker):
        """A clear mood prompt should be classified locally, costing no tokens."""
        from backend.analyzer import analyze_prompt

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            with patch("backend.analyzer.get_plex_client") as mock_plex:
                mock_plex.return_value.get_library_stats.return_value = _plex_stats(
                    [{"name": "Ambient", "count": 100}, {"name": "Jazz", "count": 200}],
                    [{"name": "2010s", "count": 150}],
                )

                result = analyze_prompt("something chill for a rainy day")

                assert result.method == "rules"
                assert result.confidence > 0
                assert result.token_count == 0
                assert result.estimated_cost == 0.0
                assert "Ambient" in result.suggested_genres
                # The LLM must not have been consulted at all.
                mock_llm.return_value.analyze.assert_not_called()

    def test_analyze_prompt_falls_back_to_llm(self, mocker):
        """Low-confidence prompts should use the LLM fallback."""
        from backend.analyzer import analyze_prompt
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content=json.dumps({
                "genres": ["Alternative", "Rock"],
                "decades": ["1990s"],
                "reasoning": "The prompt suggests 90s alternative rock."
            }),
            input_tokens=100,
            output_tokens=50,
            model="test-model"
        )

        with patch("backend.analyzer.classify_prompt", return_value=LOW_CONFIDENCE):
            with patch("backend.analyzer.get_llm_client") as mock_llm:
                mock_client = MagicMock()
                mock_client.analyze.return_value = mock_response
                mock_client.parse_json_response.return_value = json.loads(mock_response.content)
                mock_llm.return_value = mock_client

                with patch("backend.analyzer.get_plex_client") as mock_plex:
                    mock_plex.return_value.get_library_stats.return_value = _plex_stats(
                        [{"name": "Alternative", "count": 100}, {"name": "Rock", "count": 200}],
                        [{"name": "1990s", "count": 150}],
                    )

                    result = analyze_prompt("melancholy 90s alternative")

                    assert result.method == "llm"
                    assert "Alternative" in result.suggested_genres
                    assert "1990s" in result.suggested_decades
                    mock_client.analyze.assert_called_once()

    def test_analyze_prompt_handles_malformed_response(self, mocker):
        """Should handle malformed LLM JSON responses gracefully."""
        from backend.analyzer import analyze_prompt
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content="Not valid JSON at all",
            input_tokens=100,
            output_tokens=50,
            model="test-model"
        )

        with patch("backend.analyzer.classify_prompt", return_value=LOW_CONFIDENCE):
            with patch("backend.analyzer.get_llm_client") as mock_llm:
                mock_client = MagicMock()
                mock_client.analyze.return_value = mock_response
                mock_client.parse_json_response.side_effect = ValueError("Invalid JSON")
                mock_llm.return_value = mock_client

                with patch("backend.analyzer.get_plex_client") as mock_plex:
                    mock_plex.return_value.get_library_stats.return_value = _plex_stats(
                        [{"name": "Rock", "count": 100}],
                        [{"name": "1990s", "count": 100}],
                    )

                    with pytest.raises(ValueError):
                        analyze_prompt("test prompt")

    def test_analyze_prompt_returns_available_filters(self, mocker):
        """Should return available genres and decades from library."""
        from backend.analyzer import analyze_prompt

        with patch("backend.analyzer.get_llm_client"):
            with patch("backend.analyzer.get_plex_client") as mock_plex:
                mock_plex.return_value.get_library_stats.return_value = _plex_stats(
                    [
                        {"name": "Rock", "count": 500},
                        {"name": "Jazz", "count": 200},
                        {"name": "Classical", "count": 100},
                    ],
                    [
                        {"name": "1980s", "count": 300},
                        {"name": "1990s", "count": 400},
                        {"name": "2000s", "count": 200},
                    ],
                )

                result = analyze_prompt("rock music from the 90s")

                # Should include all available options from library
                assert len(result.available_genres) == 3
                assert len(result.available_decades) == 3
                assert "Rock" in result.suggested_genres
                assert "1990s" in result.suggested_decades

    def test_analyze_prompt_without_llm_still_works(self, mocker):
        """Rule-based analysis should not require an LLM client to be configured."""
        from backend.analyzer import analyze_prompt

        with patch("backend.analyzer.get_llm_client", return_value=None):
            with patch("backend.analyzer.get_plex_client") as mock_plex:
                mock_plex.return_value.get_library_stats.return_value = _plex_stats(
                    [{"name": "EDM", "count": 100}, {"name": "Pop", "count": 200}],
                    [{"name": "2020s", "count": 150}],
                )

                result = analyze_prompt("party music")

                assert result.method == "rules"
                assert result.token_count == 0


class TestFilterSuggestions:
    """Tests for filter suggestion logic."""

    def test_filters_suggest_matching_library_genres(self, mocker):
        """LLM suggestions should be filtered to genres that exist in the library."""
        from backend.analyzer import analyze_prompt
        from backend.llm_client import LLMResponse

        # LLM suggests "Alt Rock" but library has "Grunge"
        mock_response = LLMResponse(
            content=json.dumps({
                "genres": ["Alt Rock", "Grunge"],
                "decades": ["1990s"],
                "reasoning": "90s alternative rock"
            }),
            input_tokens=100,
            output_tokens=50,
            model="test-model"
        )

        with patch("backend.analyzer.classify_prompt", return_value=LOW_CONFIDENCE):
            with patch("backend.analyzer.get_llm_client") as mock_llm:
                mock_client = MagicMock()
                mock_client.analyze.return_value = mock_response
                mock_client.parse_json_response.return_value = json.loads(mock_response.content)
                mock_llm.return_value = mock_client

                with patch("backend.analyzer.get_plex_client") as mock_plex:
                    mock_plex.return_value.get_library_stats.return_value = _plex_stats(
                        [
                            {"name": "Alternative", "count": 500},
                            {"name": "Grunge", "count": 100},
                        ],
                        [{"name": "1990s", "count": 400}],
                    )

                    result = analyze_prompt("90s alt rock")

                    # Grunge should be in suggestions (exact library match)
                    assert "Grunge" in result.suggested_genres
                    # "Alt Rock" is not a library genre and must be dropped
                    assert "Alt Rock" not in result.suggested_genres


class TestTrackAnalysis:
    """Tests for seed track dimension extraction."""

    def test_analyze_track_extracts_dimensions(self, mocker):
        """Should extract musical dimensions from a track."""
        from backend.analyzer import analyze_track
        from backend.llm_client import LLMResponse
        from backend.models import Track

        track = Track(
            rating_key="1",
            title="Fake Plastic Trees",
            artist="Radiohead",
            album="The Bends",
            duration_ms=290000,
            year=1995,
            genres=["Alternative", "Rock"],
        )

        mock_response = LLMResponse(
            content=json.dumps({
                "dimensions": [
                    {"id": "mood", "label": "Melancholy, bittersweet mood", "description": "Emotional and reflective"},
                    {"id": "era", "label": "Mid-90s British alternative", "description": "Britpop era sound"},
                    {"id": "vocals", "label": "Falsetto-tinged vocals", "description": "Thom Yorke's distinctive style"},
                ]
            }),
            input_tokens=100,
            output_tokens=80,
            model="test-model"
        )

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.analyze.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            result = analyze_track(track)

            assert result.track.rating_key == "1"
            assert len(result.dimensions) == 3
            assert result.dimensions[0].id == "mood"
            assert "melancholy" in result.dimensions[0].label.lower()

    def test_analyze_track_returns_specific_labels(self, mocker):
        """Dimension labels should be specific, not generic."""
        from backend.analyzer import analyze_track
        from backend.llm_client import LLMResponse
        from backend.models import Track

        track = Track(
            rating_key="2",
            title="Black",
            artist="Pearl Jam",
            album="Ten",
            duration_ms=340000,
            year=1991,
            genres=["Grunge", "Rock"],
        )

        mock_response = LLMResponse(
            content=json.dumps({
                "dimensions": [
                    {"id": "mood", "label": "The raw, aching heartbreak", "description": "Loss and longing"},
                    {"id": "vocals", "label": "Eddie Vedder's baritone intensity", "description": "Powerful delivery"},
                ]
            }),
            input_tokens=100,
            output_tokens=60,
            model="test-model"
        )

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.analyze.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            result = analyze_track(track)

            # Labels should be specific, not just "the mood" or "the vocals"
            for dim in result.dimensions:
                assert len(dim.label) > 10  # Should be descriptive
                assert dim.label.lower() != f"the {dim.id}"  # Not generic


class TestTrackAnalysisCache:
    """analyze_track() should reuse cached dimensions per track/model."""

    def _track(self, **overrides):
        from backend.models import Track

        data = dict(
            rating_key="1",
            title="Fake Plastic Trees",
            artist="Radiohead",
            album="The Bends",
            duration_ms=290000,
            year=1995,
            genres=["Alternative", "Rock"],
        )
        data.update(overrides)
        return Track(**data)

    def _llm_response(self):
        from backend.llm_client import LLMResponse

        return LLMResponse(
            content=json.dumps(
                {
                    "dimensions": [
                        {
                            "id": "mood",
                            "label": "Melancholy, bittersweet mood",
                            "description": "Introspective and reflective",
                        },
                        {
                            "id": "era",
                            "label": "Mid-90s British alternative",
                            "description": "Britpop-era sound",
                        },
                    ]
                }
            ),
            input_tokens=100,
            output_tokens=80,
            model="analysis-model",
        )

    def _mock_llm(self, model="analysis-model"):
        client = MagicMock()
        client.config.model_analysis = model
        client.analyze.return_value = self._llm_response()
        client.parse_json_response.return_value = json.loads(self._llm_response().content)
        return client

    def test_second_analysis_hits_cache(self, tmp_path):
        """Same track + same model should not call the LLM twice."""
        from backend.analyzer import analyze_track
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "analyzer.db")
        client = self._mock_llm()

        with patch("backend.analyzer.get_llm_client", return_value=client):
            with patch("backend.analyzer.get_track_db", return_value=db):
                first = analyze_track(self._track())
                second = analyze_track(self._track())

        assert first.cached is False
        assert first.token_count > 0
        assert second.cached is True
        assert second.token_count == 0
        assert second.estimated_cost == 0.0
        assert [d.id for d in second.dimensions] == [d.id for d in first.dimensions]
        assert client.analyze.call_count == 1

    def test_model_change_invalidates_cache(self, tmp_path):
        """Switching analysis models must re-analyze the track."""
        from backend.analyzer import analyze_track
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "analyzer.db")

        client_a = self._mock_llm(model="model-a")
        with patch("backend.analyzer.get_llm_client", return_value=client_a):
            with patch("backend.analyzer.get_track_db", return_value=db):
                analyze_track(self._track())

        client_b = self._mock_llm(model="model-b")
        with patch("backend.analyzer.get_llm_client", return_value=client_b):
            with patch("backend.analyzer.get_track_db", return_value=db):
                result = analyze_track(self._track())

        assert client_a.analyze.call_count == 1
        assert client_b.analyze.call_count == 1
        assert result.cached is False

    def test_metadata_change_invalidates_cache(self, tmp_path):
        """Changed track metadata must re-analyze (fingerprint mismatch)."""
        from backend.analyzer import analyze_track
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "analyzer.db")
        client = self._mock_llm()

        with patch("backend.analyzer.get_llm_client", return_value=client):
            with patch("backend.analyzer.get_track_db", return_value=db):
                analyze_track(self._track())
                result = analyze_track(self._track(album="A Different Album"))

        assert client.analyze.call_count == 2
        assert result.cached is False

    def test_cache_does_not_apply_without_database(self):
        """With no DB configured, analysis behaves exactly as before."""
        from backend.analyzer import analyze_track

        client = self._mock_llm()
        with patch("backend.analyzer.get_llm_client", return_value=client):
            with patch("backend.analyzer.get_track_db", return_value=None):
                analyze_track(self._track())
                analyze_track(self._track())

        assert client.analyze.call_count == 2

    def test_use_cache_disabled_bypasses_cache(self, tmp_path):
        """use_cache=False forces a fresh LLM call even on a cache hit."""
        from backend.analyzer import analyze_track
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "analyzer.db")
        client = self._mock_llm()

        with patch("backend.analyzer.get_llm_client", return_value=client):
            with patch("backend.analyzer.get_track_db", return_value=db):
                analyze_track(self._track())
                analyze_track(self._track(), use_cache=False)

        assert client.analyze.call_count == 2

    def test_cached_dimensions_reconstruct_models(self, tmp_path):
        """Cached JSON should rebuild valid Dimension models."""
        from backend.analyzer import analyze_track
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "analyzer.db")
        client = self._mock_llm()

        with patch("backend.analyzer.get_llm_client", return_value=client):
            with patch("backend.analyzer.get_track_db", return_value=db):
                analyze_track(self._track())
                second = analyze_track(self._track())

        assert [d.id for d in second.dimensions] == ["mood", "era"]
        assert second.dimensions[0].label == "Melancholy, bittersweet mood"
