"""Tests for playlist generation."""

import json

import pytest
from unittest.mock import MagicMock, patch


def _parse_sse_events(generator):
    """Parse SSE events from generate_playlist_stream into (event, data) tuples."""
    events = []
    for raw in generator:
        if raw.startswith(":"):
            continue  # SSE comment (heartbeat)
        for line in raw.strip().split("\n"):
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                events.append((event_type, json.loads(line[len("data: ") :])))
    return events


class TestPlaylistGeneration:
    """Tests for playlist generation (streaming)."""

    def test_generate_validates_tracks_against_library(self, mocker, mock_plex_tracks):
        """Generated playlist should only contain tracks from library."""
        from backend.generator import generate_playlist_stream
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content=json.dumps(
                [
                    {
                        "artist": "Radiohead",
                        "album": "The Bends",
                        "title": "Fake Plastic Trees",
                    },
                    {"artist": "Pearl Jam", "album": "Ten", "title": "Black"},
                ]
            ),
            input_tokens=1000,
            output_tokens=100,
            model="test-model",
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = mock_response
            mock_client.analyze.return_value = LLMResponse(
                content='{"title": "Test", "narrative": "Test narrative."}',
                input_tokens=100,
                output_tokens=50,
                model="test-model",
            )
            mock_client.parse_json_response.side_effect = [
                json.loads(mock_response.content),
                {"title": "Test", "narrative": "Test narrative."},
            ]
            mock_llm.return_value = mock_client

            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = mock_plex_tracks[
                    :5
                ]
                mock_plex.return_value = mock_plex_client

                with patch(
                    "backend.generator.library_cache.has_cached_tracks",
                    return_value=False,
                ):
                    with patch(
                        "backend.generator.library_cache.save_result",
                        return_value="abc123",
                    ):
                        events = _parse_sse_events(
                            generate_playlist_stream(
                                prompt="90s alternative",
                                genres=["Alternative", "Rock"],
                                decades=["1990s"],
                                track_count=25,
                                exclude_live=True,
                            )
                        )

                        # Collect track rating keys from track batch events
                        track_keys = []
                        for etype, data in events:
                            if etype == "tracks":
                                track_keys.extend(
                                    t["rating_key"] for t in data["batch"]
                                )

                        library_keys = {t.rating_key for t in mock_plex_tracks}
                        for key in track_keys:
                            assert key in library_keys

    def test_generate_handles_empty_filter_results(self, mocker):
        """Should emit error event when no tracks match filters."""
        from backend.generator import generate_playlist_stream

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_llm.return_value = MagicMock()

            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = []
                mock_plex.return_value = mock_plex_client

                with patch(
                    "backend.generator.library_cache.has_cached_tracks",
                    return_value=False,
                ):
                    events = _parse_sse_events(
                        generate_playlist_stream(
                            prompt="nonexistent genre",
                            genres=["Nonexistent"],
                            decades=["1800s"],
                            track_count=25,
                            exclude_live=True,
                        )
                    )

                    error_events = [d for t, d in events if t == "error"]
                    assert len(error_events) == 1
                    assert "No tracks" in error_events[0]["message"]

    def test_fuzzy_matching_finds_similar_titles(self, mocker, mock_plex_tracks):
        """Should fuzzy match LLM responses to library tracks."""
        from backend.generator import generate_playlist_stream
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content=json.dumps(
                [
                    {
                        "artist": "Radiohead",
                        "album": "The Bends",
                        "title": "Fake Plastic Tree",
                    },
                ]
            ),
            input_tokens=1000,
            output_tokens=100,
            model="test-model",
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = mock_response
            mock_client.analyze.return_value = LLMResponse(
                content='{"title": "Test", "narrative": "Test."}',
                input_tokens=100,
                output_tokens=50,
                model="test-model",
            )
            mock_client.parse_json_response.side_effect = [
                json.loads(mock_response.content),
                {"title": "Test", "narrative": "Test."},
            ]
            mock_llm.return_value = mock_client

            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = mock_plex_tracks[
                    :5
                ]
                mock_plex.return_value = mock_plex_client

                with patch(
                    "backend.generator.library_cache.has_cached_tracks",
                    return_value=False,
                ):
                    with patch(
                        "backend.generator.library_cache.save_result",
                        return_value="abc123",
                    ):
                        events = _parse_sse_events(
                            generate_playlist_stream(
                                prompt="radiohead",
                                genres=["Alternative"],
                                decades=["1990s"],
                                track_count=25,
                                exclude_live=True,
                            )
                        )

                        # Should complete without error
                        error_events = [d for t, d in events if t == "error"]
                        assert len(error_events) == 0


class TestTrackMatching:
    """Tests for track matching utilities."""

    def test_simplify_string_removes_punctuation(self):
        """Should remove punctuation from strings."""
        from backend.plex_client import simplify_string

        assert simplify_string("Don't Stop") == "dont stop"
        assert simplify_string("Rock & Roll") == "rock  roll"
        assert simplify_string("(Remastered)") == "remastered"

    def test_simplify_string_normalizes_unicode(self):
        """Should normalize unicode characters."""
        from backend.plex_client import simplify_string

        assert simplify_string("Café") == "cafe"
        assert simplify_string("Motörhead") == "motorhead"

    def test_normalize_artist_handles_and_variations(self):
        """Should handle 'and' vs '&' variations."""
        from backend.plex_client import normalize_artist

        variations = normalize_artist("Simon & Garfunkel")
        assert "Simon & Garfunkel" in variations
        assert "Simon and Garfunkel" in variations

        variations = normalize_artist("Tom and Jerry")
        assert "Tom and Jerry" in variations
        assert "Tom & Jerry" in variations


class TestNarrativeGeneration:
    """Tests for curator narrative generation."""

    def test_generate_narrative_returns_title_and_narrative(self, mocker):
        """Should generate creative title and narrative from track selections."""
        from backend.generator import generate_narrative
        from backend.llm_client import LLMResponse

        track_selections = [
            {
                "artist": "Radiohead",
                "title": "Fake Plastic Trees",
                "reason": "Melancholic atmosphere",
            },
            {"artist": "Pearl Jam", "title": "Black", "reason": "Emotional depth"},
        ]

        mock_response = LLMResponse(
            content='{"title": "Rainstorm Reverie", "narrative": "This playlist weaves through Radiohead\'s Fake Plastic Trees and Pearl Jam\'s Black for a moody journey."}',
            input_tokens=500,
            output_tokens=50,
            model="test-model",
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        mock_client.parse_json_response.return_value = {
            "title": "Rainstorm Reverie",
            "narrative": "This playlist weaves through Radiohead's Fake Plastic Trees and Pearl Jam's Black for a moody journey.",
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert "Rainstorm Reverie" in title
        # Title should include date suffix
        assert " - " in title
        assert "Fake Plastic Trees" in narrative or len(narrative) > 0

    def test_generate_narrative_fallback_on_failure(self, mocker):
        """Should return fallback title on LLM failure."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.generate.side_effect = Exception("LLM error")

        title, narrative = generate_narrative(track_selections, mock_client)

        # Should return fallback title with date
        assert "Playlist" in title
        assert narrative == ""

    def test_generate_narrative_passes_through_long_narrative(self, mocker):
        """Should pass through narrative without truncation (LLM prompt guides length)."""
        from backend.generator import generate_narrative
        from backend.llm_client import LLMResponse

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        long_narrative = "A" * 600
        mock_response = LLMResponse(
            content=json.dumps({"title": "Test", "narrative": long_narrative}),
            input_tokens=500,
            output_tokens=50,
            model="test-model",
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        mock_client.parse_json_response.return_value = {
            "title": "Test",
            "narrative": long_narrative,
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        # No truncation - LLM prompt guides length instead
        assert narrative == long_narrative

    def test_generate_narrative_handles_array_wrapped_response(self, mocker):
        """Should handle array-wrapped JSON responses from some LLMs."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        # Some LLMs wrap their response in an array like [{...}]
        mock_client.parse_json_response.return_value = [
            {"title": "Wrapped Title", "narrative": "This is wrapped in an array."}
        ]

        title, narrative = generate_narrative(track_selections, mock_client)

        assert "Wrapped Title" in title
        assert narrative == "This is wrapped in an array."

    def test_generate_narrative_handles_alternate_key_names(self, mocker):
        """Should try alternate keys like description, text, content."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        # LLM uses "description" instead of "narrative"
        mock_client.parse_json_response.return_value = {
            "title": "Alt Key Test",
            "description": "Using description key instead of narrative.",
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert "Alt Key Test" in title
        assert narrative == "Using description key instead of narrative."

    def test_generate_narrative_handles_text_key(self, mocker):
        """Should fall back to 'text' key for narrative."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = {
            "title": "Text Key Test",
            "text": "Using text key.",
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert narrative == "Using text key."

    def test_generate_narrative_handles_content_key(self, mocker):
        """Should fall back to 'content' key for narrative."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = {
            "title": "Content Key Test",
            "content": "Using content key.",
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert narrative == "Using content key."

    def test_generate_narrative_empty_array_returns_fallback(self, mocker):
        """Should handle empty array response gracefully."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = []

        title, narrative = generate_narrative(track_selections, mock_client)

        # Should return fallback
        assert "Playlist" in title
        assert narrative == ""

    def test_generate_narrative_prefers_narrative_key_over_alternatives(self, mocker):
        """Should prefer 'narrative' key when multiple keys present."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = {
            "title": "Priority Test",
            "narrative": "Primary value",
            "description": "Should not use this",
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert narrative == "Primary value"

    def test_generate_narrative_empty_string_uses_fallback_key(self, mocker):
        """Should try alternate keys when narrative key is empty string."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = {
            "title": "Empty Primary Test",
            "narrative": "",
            "description": "Fallback description used",
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert narrative == "Fallback description used"


class TestLiveVersionFiltering:
    """Tests for live version detection."""

    def test_is_live_version_detects_live_keyword(self):
        """Should detect 'live' in track or album title."""
        from backend.plex_client import is_live_version

        class MockTrack:
            def __init__(self, title, album_title):
                self.title = title
                self._album_title = album_title

            def album(self):
                return MagicMock(title=self._album_title)

        assert is_live_version(MockTrack("Song - Live", "Album")) is True
        assert (
            is_live_version(MockTrack("Song", "Live at Madison Square Garden")) is True
        )
        assert is_live_version(MockTrack("Song", "Album")) is False

    def test_is_live_version_detects_concert_keyword(self):
        """Should detect 'concert' in track or album title."""
        from backend.plex_client import is_live_version

        class MockTrack:
            def __init__(self, title, album_title):
                self.title = title
                self._album_title = album_title

            def album(self):
                return MagicMock(title=self._album_title)

        assert is_live_version(MockTrack("Song", "Concert Recording")) is True

    def test_is_live_version_detects_date_patterns(self):
        """Should detect date patterns in album titles."""
        from backend.plex_client import is_live_version

        class MockTrack:
            def __init__(self, title, album_title):
                self.title = title
                self._album_title = album_title

            def album(self):
                return MagicMock(title=self._album_title)

        assert is_live_version(MockTrack("Song", "2023-05-15 Show")) is True
        assert is_live_version(MockTrack("Song", "1999/12/31 New Years")) is True
        assert is_live_version(MockTrack("Song", "Regular Album 2023")) is False


def _rank_track(key, genres, year):
    from backend.models import Track

    return Track(
        rating_key=key,
        title=f"Title {key}",
        artist="Artist",
        album="Album",
        duration_ms=1,
        year=year,
        genres=genres,
    )


class TestDynamicTrackBudget:
    """The AI candidate budget shrinks for narrow requests and widens for vague ones."""

    def test_specific_prompt_budget_below_ceiling(self):
        from backend.generator import PRE_RANK_LIMIT, compute_track_budget

        budget = compute_track_budget(
            prompt="jazz",
            requested_max=500,
            track_count=25,
            pool_size=10000,
            genres=["Jazz"],
        )
        assert 0 < budget < PRE_RANK_LIMIT

    def test_vague_prompt_uses_full_ceiling(self):
        from backend.generator import PRE_RANK_LIMIT, compute_track_budget

        budget = compute_track_budget(
            prompt=None, requested_max=500, track_count=25, pool_size=10000
        )
        assert budget == PRE_RANK_LIMIT

    def test_pool_size_caps_the_budget(self):
        from backend.generator import compute_track_budget

        assert (
            compute_track_budget(
                prompt="jazz",
                requested_max=500,
                track_count=25,
                pool_size=40,
                genres=["Jazz"],
            )
            == 40
        )

    def test_empty_pool_yields_zero(self):
        from backend.generator import compute_track_budget

        assert (
            compute_track_budget(
                prompt="jazz", requested_max=500, track_count=25, pool_size=0
            )
            == 0
        )

    def test_requested_max_caps_the_ceiling(self):
        from backend.generator import compute_track_budget

        assert (
            compute_track_budget(
                prompt=None, requested_max=100, track_count=25, pool_size=10000
            )
            == 100
        )


class TestPreRanking:
    """Local rule-based pre-ranking trims the candidate pool without tokens."""

    def test_prefers_genre_and_decade_match(self):
        from backend.generator import pre_rank_tracks

        tracks = [
            _rank_track("1", ["Pop"], 2001),
            _rank_track("2", ["Jazz"], 1995),
            _rank_track("3", ["Jazz"], 1959),
        ]
        ranked = pre_rank_tracks(tracks, prompt="jazz from the 90s", max_tracks=1)
        assert [t.rating_key for t in ranked] == ["2"]

    def test_returns_input_when_under_budget(self):
        from backend.generator import pre_rank_tracks

        tracks = [_rank_track("1", ["Pop"], 2001)]
        assert pre_rank_tracks(tracks, prompt="jazz", max_tracks=500) is tracks

    def test_ties_are_broken_deterministically(self):
        from backend.generator import pre_rank_tracks

        tracks = [_rank_track("2", ["Rock"], 1995), _rank_track("1", ["Rock"], 1995)]
        ranked = pre_rank_tracks(tracks, prompt=None, max_tracks=1)
        assert [t.rating_key for t in ranked] == ["1"]

    def test_user_rating_breaks_ties(self, tmp_path, monkeypatch):
        from backend import library_cache
        from backend.generator import pre_rank_tracks

        monkeypatch.setattr(library_cache, "DB_PATH", tmp_path / "rank.db")
        monkeypatch.setattr(library_cache, "DATA_DIR", tmp_path)
        monkeypatch.setattr(library_cache, "_schema_initialized", False)

        conn = library_cache.ensure_db_initialized()
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album, user_rating) "
            "VALUES ('1', 'a', 'a', 'a', 2)"
        )
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album, user_rating) "
            "VALUES ('2', 'b', 'b', 'b', 10)"
        )
        conn.commit()
        conn.close()

        tracks = [_rank_track("1", ["Jazz"], 1995), _rank_track("2", ["Jazz"], 1995)]
        ranked = pre_rank_tracks(tracks, prompt="jazz", max_tracks=1)
        assert [t.rating_key for t in ranked] == ["2"]

    def test_metadata_failure_is_tolerated(self):
        from backend.generator import pre_rank_tracks

        tracks = [_rank_track("1", ["Jazz"], 1995), _rank_track("2", ["Rock"], 1995)]
        with patch(
            "backend.library_cache.get_metadata_batch",
            side_effect=RuntimeError("no db"),
        ):
            ranked = pre_rank_tracks(tracks, prompt="jazz", max_tracks=1)
        assert [t.rating_key for t in ranked] == ["1"]

    def test_stream_pre_ranks_large_candidate_pool(self):
        """A pool larger than the budget gets trimmed before reaching the LLM."""
        from backend.generator import generate_playlist_stream
        from backend.llm_client import LLMResponse
        from backend.models import Track

        pool = [
            Track(
                rating_key=str(i),
                title=f"T{i}",
                artist="Artist",
                album="Album",
                duration_ms=1,
                year=1990 + (i % 20),
                genres=["Rock"],
            )
            for i in range(600)
        ]

        with patch("backend.generator.get_llm_client") as mock_llm:
            client = MagicMock()
            client.generate.return_value = LLMResponse(
                content=json.dumps(
                    [{"artist": "Artist", "album": "Album", "title": "T1"}]
                ),
                input_tokens=10,
                output_tokens=5,
                model="test-model",
            )
            client.analyze.return_value = LLMResponse(
                content='{"title": "Test", "narrative": "Test."}',
                input_tokens=5,
                output_tokens=5,
                model="test-model",
            )
            client.parse_json_response.side_effect = [
                json.loads(client.generate.return_value.content),
                {"title": "Test", "narrative": "Test."},
            ]
            mock_llm.return_value = client

            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex.return_value.get_tracks_by_filters.return_value = pool
                with patch(
                    "backend.generator.library_cache.has_cached_tracks",
                    return_value=False,
                ):
                    with patch(
                        "backend.generator.library_cache.get_metadata_batch",
                        return_value={},
                    ):
                        with patch(
                            "backend.generator.library_cache.save_result",
                            return_value="id",
                        ):
                            events = _parse_sse_events(
                                generate_playlist_stream(
                                    track_count=25, genres=["Rock"]
                                )
                            )

        steps = [d.get("step") for _, d in events if _ == "progress"]
        assert "pre_ranking" in steps

        generation_prompt = client.generate.call_args[0][0]
        # Fewer than the raw 600 candidates are sent to the LLM.
        assert generation_prompt.count("\n") < 600


class TestFacetAwareBudget:
    """The new hard filters narrow the pool, so the AI budget shrinks for them."""

    def test_year_window_shrinks_the_budget(self):
        from backend.generator import compute_track_budget

        vague = compute_track_budget(
            prompt="deep cuts", requested_max=500, track_count=25, pool_size=10000
        )
        dated = compute_track_budget(
            prompt="deep cuts",
            requested_max=500,
            track_count=25,
            pool_size=10000,
            min_year=1980,
            max_year=1989,
        )

        assert dated < vague

    def test_play_window_shrinks_the_budget(self):
        from backend.generator import compute_track_budget

        plain = compute_track_budget(
            prompt="nothing specific",
            requested_max=500,
            track_count=25,
            pool_size=10000,
        )
        unheard = compute_track_budget(
            prompt="nothing specific",
            requested_max=500,
            track_count=25,
            pool_size=10000,
            unheard_only=True,
        )

        assert unheard < plain

    def test_yes_no_flag_narrows_less_than_a_window(self):
        from backend.generator import compute_track_budget

        flag = compute_track_budget(
            prompt=None,
            requested_max=500,
            track_count=25,
            pool_size=10000,
            exclude_compilations=True,
        )
        window = compute_track_budget(
            prompt=None,
            requested_max=500,
            track_count=25,
            pool_size=10000,
            min_plays=1,
        )

        assert flag > window

    def test_year_window_scores_like_a_duration_window(self):
        from backend.generator import compute_track_budget

        by_year = compute_track_budget(
            prompt=None,
            requested_max=500,
            track_count=25,
            pool_size=10000,
            min_year=1980,
        )
        by_length = compute_track_budget(
            prompt=None,
            requested_max=500,
            track_count=25,
            pool_size=10000,
            min_duration_ms=180_000,
        )

        assert by_year == by_length

    def test_exclusion_lists_narrow_the_pool(self):
        from backend.generator import compute_track_budget

        plain = compute_track_budget(
            prompt=None, requested_max=500, track_count=25, pool_size=10000
        )
        banned = compute_track_budget(
            prompt=None,
            requested_max=500,
            track_count=25,
            pool_size=10000,
            exclude_artists=["Nickelback"],
            exclude_decades=["1980s"],
        )

        assert banned < plain


def _facet_track(key, artist="Artist", year=2000, duration_ms=200_000):
    from backend.models import Track

    return Track(
        rating_key=key,
        title=f"Title {key}",
        artist=artist,
        album="Album",
        duration_ms=duration_ms,
        year=year,
        genres=["Rock"],
    )


class TestFacetAwarePreRanking:
    """Pre-ranking honours the year, familiarity and ban filters."""

    @pytest.fixture
    def ranked_db(self, tmp_path, monkeypatch):
        from backend import library_cache

        monkeypatch.setattr(library_cache, "DB_PATH", tmp_path / "rank.db")
        monkeypatch.setattr(library_cache, "DATA_DIR", tmp_path)
        monkeypatch.setattr(library_cache, "_schema_initialized", False)

        conn = library_cache.ensure_db_initialized()
        for key, plays in (("1", 5), ("2", 0)):
            conn.execute(
                "INSERT INTO tracks (rating_key, title, artist, album, user_rating, "
                "view_count) VALUES (?, 'a', 'a', 'a', 0, ?)",
                (key, plays),
            )
        conn.commit()
        conn.close()

    def test_unheard_only_prefers_the_never_played(self, ranked_db):
        from backend.generator import pre_rank_tracks

        tracks = [_facet_track("1"), _facet_track("2")]
        ranked = pre_rank_tracks(tracks, max_tracks=1, unheard_only=True)

        assert [t.rating_key for t in ranked] == ["2"]

    def test_play_floor_prefers_the_familiar(self, ranked_db):
        from backend.generator import pre_rank_tracks

        tracks = [_facet_track("1"), _facet_track("2")]
        ranked = pre_rank_tracks(tracks, max_tracks=1, min_plays=3)

        assert [t.rating_key for t in ranked] == ["1"]

    def test_tracks_without_a_cache_row_ignore_the_play_axis(self, ranked_db):
        from backend.generator import pre_rank_tracks

        tracks = [_facet_track("1"), _facet_track("3")]
        ranked = pre_rank_tracks(tracks, max_tracks=1, unheard_only=True)

        assert [t.rating_key for t in ranked] == ["3"]

    def test_year_window_demotes_an_older_candidate(self):
        from backend.generator import pre_rank_tracks

        tracks = [_facet_track("1", year=1985), _facet_track("2", year=1995)]
        ranked = pre_rank_tracks(tracks, max_tracks=1, min_year=1990)

        assert [t.rating_key for t in ranked] == ["2"]

    def test_banned_artist_loses_despite_a_genre_match(self):
        from backend.generator import pre_rank_tracks

        tracks = [
            _facet_track("1", artist="Nickelback"),
            _facet_track("2", artist="Metallica"),
        ]
        ranked = pre_rank_tracks(
            tracks, prompt="metal", max_tracks=1, exclude_artists=["Nickelback"]
        )

        assert [t.rating_key for t in ranked] == ["2"]

    def test_banned_decade_loses(self):
        from backend.generator import pre_rank_tracks

        tracks = [_facet_track("1", year=1985), _facet_track("2", year=1995)]
        ranked = pre_rank_tracks(tracks, max_tracks=1, exclude_decades=["1980s"])

        assert [t.rating_key for t in ranked] == ["2"]


class TestCandidatePoolRelaxation:
    """The candidate pool the AI chooses from widens when the filters are too narrow."""

    def test_candidate_facets_spell_off_the_caches_way(self):
        """Empty lists become None so counting and fetching agree."""
        from backend.generator import candidate_facets

        facets = candidate_facets(genres=["Prog"], artists=[], exclude_live=False)

        assert facets["genres"] == ["Prog"]
        assert facets["artists"] is None
        assert facets["exclude_live"] is False

    def test_relaxation_keeps_the_genres_it_was_given(self):
        """Only tiers the request actually set are walked, so the whitelist survives."""
        from backend.generator import candidate_facets, relax_candidate_pool

        seen: list[tuple] = []

        def counted(**filters):
            seen.append(tuple(sorted(n for n, v in filters.items() if v)))
            return {1: 40, 2: 120, 3: 900}.get(len(seen), 900)

        with patch("backend.generator.library_cache.count_tracks_by_filters", side_effect=counted):
            result = relax_candidate_pool(
                candidate_facets(
                    genres=["Rock"], min_rating=7, min_year=2015, max_year=2018, exclude_live=False
                ),
                prompt="rock",
                track_count=25,
            )

        assert seen[0] == ("genres", "max_year", "min_rating", "min_year")
        assert result["relaxed"] == ["no rating or play floors", "no year or duration windows"]
        assert result["matching"] == 900
        assert result["filters"]["genres"] == ["Rock"]
        assert result["filters"]["min_rating"] == 0

    def test_relaxation_stops_at_the_candidate_budget(self):
        """One tier is enough once it reaches the number of candidates wanted."""
        from backend.generator import candidate_facets, compute_track_budget, relax_candidate_pool

        with patch("backend.generator.library_cache.count_tracks_by_filters", return_value=900):
            want = compute_track_budget(prompt="rock anthems", track_count=25, genres=["Rock"])
            result = relax_candidate_pool(
                candidate_facets(genres=["Rock"], exclude_live=False),
                prompt="rock anthems",
                track_count=25,
            )

        assert result["target"] == want
        assert result["relaxed"] == []
        assert result["tier"] == "as asked"

    def test_specific_request_is_not_swapped_for_the_library(self):
        """The whitelist is only given up when the pool cannot fill the playlist."""
        from backend.generator import candidate_facets, relax_candidate_pool

        seen: list[tuple] = []

        def counted(**filters):
            seen.append(tuple(sorted(n for n, v in filters.items() if v)))
            return 80 if len(seen) == 1 else 60000

        with patch("backend.generator.library_cache.count_tracks_by_filters", side_effect=counted):
            result = relax_candidate_pool(
                candidate_facets(genres=["Bossa Nova"], exclude_live=False),
                prompt="bossa nova",
                track_count=40,
            )

        assert result["matching"] == 80
        assert result["filters"]["genres"] == ["Bossa Nova"]
        assert result["relaxed"] == []

    def test_identity_tier_runs_when_the_pool_cannot_fill_the_playlist(self):
        """Fewer tracks than the playlist wants is the only way to widen the whitelist."""
        from backend.generator import candidate_facets, relax_candidate_pool

        seen: list[tuple] = []

        def counted(**filters):
            seen.append(tuple(sorted(n for n, v in filters.items() if v)))
            return 12 if len(seen) == 1 else 6000

        with patch("backend.generator.library_cache.count_tracks_by_filters", side_effect=counted):
            result = relax_candidate_pool(
                candidate_facets(genres=["Prog"], artists=["Camel"], exclude_live=False),
                prompt="prog opera",
                track_count=25,
            )

        assert result["relaxed"] == ["no genre, decade or artist whitelist"]
        assert result["filters"]["genres"] is None
        assert result["matching"] == 6000
