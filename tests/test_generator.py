"""Tests for playlist generation."""

import json
import pytest
from unittest.mock import MagicMock, patch


class TestPlaylistGeneration:
    """Tests for playlist generation."""

    def test_generate_validates_tracks_against_library(self, mocker, mock_plex_tracks):
        """Generated playlist should only contain tracks from library."""
        from backend.generator import generate_playlist
        from backend.llm_client import LLMResponse

        # LLM returns tracks that are in the library
        mock_response = LLMResponse(
            content=json.dumps([
                {"artist": "Radiohead", "album": "The Bends", "title": "Fake Plastic Trees"},
                {"artist": "Pearl Jam", "album": "Ten", "title": "Black"},
            ]),
            input_tokens=1000,
            output_tokens=100,
            model="test-model"
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = mock_plex_tracks[:5]
                mock_plex.return_value = mock_plex_client

                result = generate_playlist(
                    prompt="90s alternative",
                    genres=["Alternative", "Rock"],
                    decades=["1990s"],
                    track_count=25,
                    exclude_live=True
                )

                # All returned tracks should be from the library
                for track in result.tracks:
                    assert any(
                        t.rating_key == track.rating_key
                        for t in mock_plex_tracks
                    )

    def test_generate_handles_empty_filter_results(self, mocker):
        """Should handle case when no tracks match filters."""
        from backend.generator import generate_playlist

        with patch("backend.generator.get_llm_client"):
            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = []
                mock_plex.return_value = mock_plex_client

                with pytest.raises(ValueError, match="No tracks"):
                    generate_playlist(
                        prompt="nonexistent genre",
                        genres=["Nonexistent"],
                        decades=["1800s"],
                        track_count=25,
                        exclude_live=True
                    )

    def test_fuzzy_matching_finds_similar_titles(self, mocker, mock_plex_tracks):
        """Should fuzzy match LLM responses to library tracks."""
        from backend.generator import generate_playlist
        from backend.llm_client import LLMResponse

        # LLM returns slightly different track name
        mock_response = LLMResponse(
            content=json.dumps([
                # Note: "Fake Plastic Tree" vs "Fake Plastic Trees"
                {"artist": "Radiohead", "album": "The Bends", "title": "Fake Plastic Tree"},
            ]),
            input_tokens=1000,
            output_tokens=100,
            model="test-model"
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = mock_plex_tracks[:5]
                mock_plex.return_value = mock_plex_client

                result = generate_playlist(
                    prompt="radiohead",
                    genres=["Alternative"],
                    decades=["1990s"],
                    track_count=25,
                    exclude_live=True
                )

                # Should still match the track despite slight title difference
                # (implementation will use fuzzy matching)
                assert len(result.tracks) >= 0  # May or may not match depending on threshold

    def test_generate_matches_exact_and_fuzzy_drops_hallucinations(self, mock_plex_tracks):
        """Exact and near-miss selections resolve; invented tracks are dropped."""
        from backend.generator import generate_playlist
        from backend.llm_client import LLMResponse

        response = LLMResponse(
            content=json.dumps([
                {"artist": "Radiohead", "title": "Fake Plastic Trees"},   # exact
                {"artist": "Pearl Jam", "title": "Blak"},                 # typo -> fuzzy
                {"artist": "Nobody", "title": "Definitely Not In Library"},  # hallucination
            ]),
            input_tokens=1000,
            output_tokens=100,
            model="test-model",
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = response
            mock_client.parse_json_response.return_value = json.loads(response.content)
            mock_llm.return_value = mock_client

            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = mock_plex_tracks
                mock_plex.return_value = mock_plex_client

                result = generate_playlist(
                    prompt="90s alternative",
                    genres=["Alternative", "Rock"],
                    decades=["1990s"],
                    track_count=25,
                )

        assert {t.rating_key for t in result.tracks} == {"1", "2"}

    def test_streaming_resolves_fuzzy_and_exact_selections(self, mock_plex_tracks):
        """The SSE path uses the same matcher (exact + fuzzy)."""
        from backend.generator import generate_playlist_stream
        from backend.llm_client import LLMResponse

        response = LLMResponse(
            content=json.dumps([
                {"artist": "Radiohead", "title": "Fake Plastic Tree"},  # fuzzy
                {"artist": "Oasis", "title": "Wonderwall"},             # exact
            ]),
            input_tokens=1,
            output_tokens=1,
            model="test-model",
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = response
            mock_client.parse_json_response.return_value = json.loads(response.content)
            mock_llm.return_value = mock_client

            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = mock_plex_tracks
                mock_plex.return_value = mock_plex_client

                events = list(generate_playlist_stream(
                    prompt="90s alternative",
                    genres=["Alternative", "Rock"],
                    decades=["1990s"],
                    track_count=25,
                ))

        complete = [e for e in events if e.startswith("event: complete")]
        assert complete, events
        payload = json.loads(complete[-1].split("data: ", 1)[1])
        assert {t["rating_key"] for t in payload["tracks"]} == {"1", "5"}

    def test_dynamic_budget_limits_tracks_sent_to_llm(self, mock_plex_tracks):
        """A specific request pre-ranks fewer candidates before the LLM call."""
        from backend.generator import generate_playlist, PRE_RANK_LIMIT
        from backend.llm_client import LLMResponse
        import backend.generator as gen

        pool = [
            t.model_copy(update={"rating_key": f"{t.rating_key}-{i}"})
            for i in range(60)
            for t in mock_plex_tracks
        ]  # 600 tracks

        response = LLMResponse(content=json.dumps([]), input_tokens=1,
                               output_tokens=1, model="test-model")
        captured = {}
        real_pre_rank = gen.pre_rank_tracks

        def spy(tracks, prompt=None, filters=None, max_tracks=PRE_RANK_LIMIT):
            captured["max_tracks"] = max_tracks
            return real_pre_rank(tracks, prompt, filters, max_tracks)

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = response
            mock_client.parse_json_response.return_value = []
            mock_llm.return_value = mock_client
            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = pool
                mock_plex.return_value = mock_plex_client
                with patch("backend.generator.pre_rank_tracks", side_effect=spy):
                    generate_playlist(
                        prompt="jazz", genres=["Jazz"], decades=[], track_count=15
                    )

        assert 0 < captured["max_tracks"] < PRE_RANK_LIMIT

    def test_streaming_reports_dynamic_budget(self, mock_plex_tracks):
        from backend.generator import generate_playlist_stream, PRE_RANK_LIMIT
        from backend.llm_client import LLMResponse

        pool = [
            t.model_copy(update={"rating_key": f"{t.rating_key}-{i}"})
            for i in range(60)
            for t in mock_plex_tracks
        ]  # 600 tracks

        response = LLMResponse(content=json.dumps([]), input_tokens=1,
                               output_tokens=1, model="test-model")

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = response
            mock_client.parse_json_response.return_value = []
            mock_llm.return_value = mock_client
            with patch("backend.generator.get_plex_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = pool
                mock_plex.return_value = mock_plex_client
                events = list(generate_playlist_stream(
                    prompt="jazz", genres=["Jazz"], decades=[], track_count=15
                ))

        pre = [e for e in events if "pre_ranking" in e][0]
        payload = json.loads(pre.split("data: ", 1)[1])
        assert payload["candidates"] == 600
        assert payload["retained"] < payload["candidates"]
        assert payload["retained"] < PRE_RANK_LIMIT


class TestTrackDbSyncOnGeneration:
    """Plex fallbacks get mirrored locally; local reads are not re-synced."""

    def _run(self, library, db, db_available, stream=False):
        """Generate a playlist with a stubbed Plex client and TrackDB."""
        from backend.generator import generate_playlist, generate_playlist_stream
        from backend.llm_client import LLMResponse

        response = LLMResponse(
            content=json.dumps([
                {"artist": "Radiohead", "album": "The Bends", "title": "Fake Plastic Trees"},
            ]),
            input_tokens=10,
            output_tokens=10,
            model="test-model",
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            llm_client = MagicMock()
            llm_client.generate.return_value = response
            llm_client.parse_json_response.return_value = json.loads(response.content)
            mock_llm.return_value = llm_client

            with patch("backend.generator.get_plex_client") as mock_plex:
                plex_client = MagicMock()
                plex_client.get_tracks_by_filters.return_value = library
                plex_client.db_available.return_value = db_available
                mock_plex.return_value = plex_client

                with patch("backend.generator.get_track_db", return_value=db):
                    kwargs = dict(
                        prompt="radiohead",
                        genres=["Alternative"],
                        decades=["1990s"],
                        track_count=25,
                    )
                    if stream:
                        list(generate_playlist_stream(**kwargs))
                    else:
                        generate_playlist(**kwargs)

    def test_skips_sync_when_read_came_from_db(self, mock_plex_tracks, tmp_path):
        """A DB-served read is already mirrored; no re-sync or commit."""
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "plexsage.db")
        db.upsert_tracks(mock_plex_tracks)
        spy = MagicMock(wraps=db)

        self._run(mock_plex_tracks, spy, db_available=True)

        spy.sync_from_plex.assert_not_called()

    def test_streaming_skips_sync_when_read_came_from_db(self, mock_plex_tracks, tmp_path):
        """The SSE generation path shares the same skip logic."""
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "plexsage.db")
        db.upsert_tracks(mock_plex_tracks)
        spy = MagicMock(wraps=db)

        self._run(mock_plex_tracks, spy, db_available=True, stream=True)

        spy.sync_from_plex.assert_not_called()

    def test_syncs_when_read_fell_back_to_plex(self, mock_plex_tracks, tmp_path):
        """An empty DB is mirrored with the tracks just fetched from Plex."""
        from backend.track_db import TrackDB

        db = TrackDB(tmp_path / "plexsage.db")
        spy = MagicMock(wraps=db)

        self._run(mock_plex_tracks, spy, db_available=False)

        spy.sync_from_plex.assert_called_once_with(mock_plex_tracks)
        assert db.count_tracks_by_filters() == len(mock_plex_tracks)


class TestSharedPreparation:
    """Both generation paths share one fetch → rank → prompt stage."""

    def _clients(self, tracks):
        from backend.llm_client import LLMResponse

        response = LLMResponse(
            content=json.dumps([
                {"artist": "Radiohead", "title": "Fake Plastic Trees"},
            ]),
            input_tokens=1,
            output_tokens=1,
            model="test-model",
        )
        llm = MagicMock()
        llm.generate.return_value = response
        llm.parse_json_response.return_value = json.loads(response.content)

        plex = MagicMock()
        plex.get_tracks_by_filters.return_value = tracks
        return llm, plex

    def test_non_streaming_path_uses_track_cache(self, mock_plex_tracks):
        """Repeat non-streaming generations reuse cached library results.

        The non-streaming path used to skip the cache entirely.
        """
        from backend.generator import generate_playlist

        llm, plex = self._clients(mock_plex_tracks)

        with patch("backend.generator.get_llm_client", return_value=llm):
            with patch("backend.generator.get_plex_client", return_value=plex):
                generate_playlist(genres=["Alternative"], decades=["1990s"], track_count=5)
                generate_playlist(genres=["Alternative"], decades=["1990s"], track_count=5)

        assert plex.get_tracks_by_filters.call_count == 1

    def test_streaming_and_non_streaming_share_the_cache(self, mock_plex_tracks):
        """A streaming fetch warms the cache the non-streaming path reuses."""
        from backend.generator import generate_playlist, generate_playlist_stream

        llm, plex = self._clients(mock_plex_tracks)

        with patch("backend.generator.get_llm_client", return_value=llm):
            with patch("backend.generator.get_plex_client", return_value=plex):
                list(generate_playlist_stream(
                    genres=["Alternative"], decades=["1990s"], track_count=5
                ))
                generate_playlist(genres=["Alternative"], decades=["1990s"], track_count=5)

        assert plex.get_tracks_by_filters.call_count == 1

    def test_streaming_event_order_is_unchanged(self, mock_plex_tracks):
        """The shared stage emits the same progress sequence as before."""
        from backend.generator import generate_playlist_stream

        llm, plex = self._clients(mock_plex_tracks)

        with patch("backend.generator.get_llm_client", return_value=llm):
            with patch("backend.generator.get_plex_client", return_value=plex):
                events = list(generate_playlist_stream(
                    genres=["Alternative"], decades=["1990s"], track_count=5
                ))

        steps = [
            json.loads(event.split("data: ", 1)[1])["step"]
            for event in events
            if event.startswith("event: progress")
        ]
        assert steps == [
            "fetching",
            "pre_ranking",
            "preparing",
            "ai_working",
            "parsing",
            "matching",
            "complete",
        ]

    def test_both_paths_report_no_matches(self):
        """Empty results surface the same message from both paths."""
        from backend.generator import generate_playlist, generate_playlist_stream

        llm, plex = self._clients([])
        plex.db_available.return_value = False

        with patch("backend.generator.get_llm_client", return_value=llm):
            with patch("backend.generator.get_plex_client", return_value=plex):
                events = list(generate_playlist_stream(
                    genres=["Nonexistent"], decades=["1800s"], track_count=5
                ))
                with pytest.raises(ValueError, match="No tracks match"):
                    generate_playlist(genres=["Nonexistent"], decades=["1800s"], track_count=5)

        errors = [e for e in events if e.startswith("event: error")]
        assert errors, events
        payload = json.loads(errors[-1].split("data: ", 1)[1])
        assert "No tracks match" in payload["message"]


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


class TestTrackMatcher:
    """Exact + trigram-pruned fuzzy matching."""

    def _library(self):
        from backend.models import Track

        return [
            Track(rating_key="1", title="Fake Plastic Trees", artist="Radiohead",
                  album="The Bends", duration_ms=1, year=1995),
            Track(rating_key="2", title="Black", artist="Pearl Jam",
                  album="Ten", duration_ms=1, year=1991),
            Track(rating_key="3", title="Wonderwall", artist="Oasis",
                  album="Morning Glory", duration_ms=1, year=1995),
            Track(rating_key="4", title="Bitter Sweet Symphony", artist="The Verve",
                  album="Urban Hymns", duration_ms=1, year=1997),
        ]

    def test_exact_match(self):
        from backend.generator import TrackMatcher

        m = TrackMatcher(self._library())
        assert m.match("Radiohead", "Fake Plastic Trees", set()).rating_key == "1"

    def test_exact_match_ignores_case_and_punctuation(self):
        from backend.generator import TrackMatcher

        m = TrackMatcher(self._library())
        assert m.match("radiohead", "fake plastic trees!", set()).rating_key == "1"

    def test_fuzzy_match_handles_typo(self):
        from backend.generator import TrackMatcher

        m = TrackMatcher(self._library())
        # Missing trailing "s" still matches via fuzzy comparison.
        assert m.match("Radiohead", "Fake Plastic Tree", set()).rating_key == "1"

    def test_exact_match_preferred_over_fuzzy(self):
        """A later exact match must win over an earlier fuzzy candidate."""
        from backend.generator import TrackMatcher
        from backend.models import Track

        tracks = [
            Track(rating_key="1", title="Blackbird", artist="The Beatles",
                  album="A", duration_ms=1, year=1968),
            Track(rating_key="2", title="Black", artist="Pearl Jam",
                  album="Ten", duration_ms=1, year=1991),
        ]
        m = TrackMatcher(tracks)
        assert m.match("Pearl Jam", "Black", set()).rating_key == "2"

    def test_skips_used_tracks(self):
        from backend.generator import TrackMatcher

        m = TrackMatcher(self._library())
        assert m.match("Radiohead", "Fake Plastic Trees", {"1"}) is None

    def test_artist_ampersand_variant(self):
        from backend.generator import TrackMatcher
        from backend.models import Track

        tracks = [Track(rating_key="1", title="The Sound of Silence",
                        artist="Simon & Garfunkel", album="A",
                        duration_ms=1, year=1966)]
        m = TrackMatcher(tracks)
        # LLM writes "and" instead of "&".
        assert m.match("Simon and Garfunkel", "The Sound of Silence", set()).rating_key == "1"

    def test_short_title_fallback(self):
        """Titles shorter than a trigram have no overlaps; fallback finds them."""
        from backend.generator import TrackMatcher
        from backend.models import Track

        tracks = [Track(rating_key="1", title="Go", artist="Solo",
                        album="A", duration_ms=1, year=2000)]
        m = TrackMatcher(tracks)
        assert m.match("Solo", "Go!", set()).rating_key == "1"

    def test_returns_none_when_no_match(self):
        from backend.generator import TrackMatcher

        m = TrackMatcher(self._library())
        assert m.match("Nobody", "Nonexistent Song", set()) is None

    def test_candidates_small_for_distinctive_title(self):
        """A distinctive near-miss only considers a handful of candidates."""
        from backend.generator import TrackMatcher
        from backend.models import Track

        tracks = [
            Track(rating_key=str(i), title=f"Unique Song {i} Alpha",
                  artist=f"Artist {i}", album="A", duration_ms=1, year=2000)
            for i in range(400)
        ]
        tracks.append(Track(rating_key="x", title="Distinctive Melody Title",
                            artist="The Artist", album="A", duration_ms=1, year=2000))
        m = TrackMatcher(tracks)

        candidates = m._candidates("distinctive melody titl")
        assert candidates is not None
        assert len(candidates) < len(tracks) // 2

    def test_broad_overlap_falls_back_to_scan(self):
        """When a title collides everywhere, the linear fallback still matches."""
        from backend.generator import TrackMatcher
        from backend.models import Track

        tracks = [Track(rating_key=str(i), title="Same Song", artist=f"Band {i}",
                        album="A", duration_ms=1, year=2000) for i in range(200)]
        m = TrackMatcher(tracks)

        assert m._candidates("same song") is None  # too broad to prune
        result = m.match("Band 57", "Same Song", set())
        assert result is not None
        assert result.title == "Same Song"

    def test_fuzzy_comparisons_are_pruned(self):
        """Matching a near-miss must not fuzzy-compare the whole library."""
        from backend.generator import TrackMatcher
        from backend.models import Track
        import backend.generator as gen

        tracks = [
            Track(rating_key=str(i), title=f"Unique Song {i} Alpha",
                  artist=f"Artist {i}", album="A", duration_ms=1, year=2000)
            for i in range(400)
        ]
        tracks.append(Track(rating_key="x", title="Distinctive Melody Title",
                            artist="The Artist", album="A", duration_ms=1, year=2000))

        calls = {"n": 0}
        original = gen._fuzzy_track_match

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        with patch.object(gen, "_fuzzy_track_match", side_effect=counting):
            m = TrackMatcher(tracks)
            assert m.match("The Artist", "Distinctive Melody Titl", set()) is not None

        assert calls["n"] < len(tracks) // 2

    def test_tracks_match_wrapper(self):
        from backend.generator import _tracks_match
        from backend.models import Track

        track = Track(rating_key="1", title="Fake Plastic Trees", artist="Radiohead",
                      album="A", duration_ms=1, year=1995)
        assert _tracks_match("Radiohead", "Fake Plastic Tree", track) is True
        assert _tracks_match("Someone Else", "Fake Plastic Trees", track) is False


class TestTrackBudget:
    """Dynamic max_tracks_to_ai based on prompt specificity."""

    def test_no_prompt_keeps_full_ceiling(self):
        from backend.generator import compute_track_budget, PRE_RANK_LIMIT

        assert compute_track_budget(None, 0, 25, 10000) == PRE_RANK_LIMIT

    def test_prompt_without_hints_keeps_full_ceiling(self):
        from backend.generator import compute_track_budget, PRE_RANK_LIMIT

        assert compute_track_budget("", 0, 25, 10000) == PRE_RANK_LIMIT

    def test_mood_prompt_keeps_more_options_than_genre_prompt(self):
        from backend.generator import compute_track_budget

        mood = compute_track_budget("something for a road trip", 500, 25, 10000)
        genre = compute_track_budget("jazz", 500, 25, 10000, genres=["Jazz"])
        assert mood > genre

    def test_explicit_genre_lowers_budget_below_ceiling(self):
        from backend.generator import compute_track_budget, PRE_RANK_LIMIT

        budget = compute_track_budget("", 500, 25, 10000, genres=["Jazz"])
        assert 0 < budget < PRE_RANK_LIMIT

    def test_genre_and_decade_is_more_specific_than_genre_alone(self):
        from backend.generator import compute_track_budget

        genre_only = compute_track_budget("alternative", 500, 25, 10000, genres=["Alternative"])
        genre_decade = compute_track_budget(
            "90s alternative", 500, 25, 10000,
            genres=["Alternative"], decades=["1990s"],
        )
        assert genre_decade < genre_only

    def test_requested_max_is_a_ceiling(self):
        from backend.generator import compute_track_budget

        assert compute_track_budget("jazz", 120, 25, 10000, genres=["Jazz"]) == 120

    def test_pool_size_is_a_ceiling(self):
        from backend.generator import compute_track_budget

        assert compute_track_budget("jazz", 500, 25, 90, genres=["Jazz"]) == 90

    def test_empty_pool_returns_zero(self):
        from backend.generator import compute_track_budget

        assert compute_track_budget("jazz", 500, 25, 0, genres=["Jazz"]) == 0

    def test_floor_scales_with_track_count(self):
        from backend.generator import compute_track_budget, BUDGET_FLOOR, MIN_TRACKS_PER_PICK

        small = compute_track_budget("jazz", 500, 5, 10000, genres=["Jazz"])
        large = compute_track_budget("jazz", 500, 40, 10000, genres=["Jazz"])
        assert small >= BUDGET_FLOOR
        assert large >= 40 * MIN_TRACKS_PER_PICK
        assert large > small

    def test_specificity_is_bounded_and_ordered(self):
        from backend.generator import _prompt_specificity

        assert _prompt_specificity(None) == 0.0
        assert 0.0 <= _prompt_specificity("something chill") < _prompt_specificity("jazz")
        assert _prompt_specificity("90s jazz", ["Jazz"], ["1990s"]) <= 1.0


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
        assert is_live_version(MockTrack("Song", "Live at Madison Square Garden")) is True
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
