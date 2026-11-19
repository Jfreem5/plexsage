"""Tests for the rule/embedding prompt classifier (backend.prompt_classifier)."""

from datetime import date

from backend.prompt_classifier import (
    CONFIDENCE_THRESHOLD,
    PromptClassification,
    classify_prompt,
    extract_compilation_exclusion,
    extract_decade_hints,
    extract_duration_window,
    extract_genre_hints,
    extract_negated_artists,
    extract_negated_decades,
    extract_play_window,
    extract_year_window,
    is_genre_keyword,
    match_genres_to_available,
)

# A small but representative library used across most tests.
LIB_GENRES = [
    "Alternative",
    "Ambient",
    "Blues",
    "Grunge",
    "Jazz",
    "Lo-Fi",
    "Post-Rock",
    "Rock",
    "Shoegaze",
]
LIB_DECADES = ["1980s", "1990s", "2000s", "2010s"]


class TestKeywordRules:
    """The keyword→genre/decade mapping."""

    def test_mood_keyword_maps_to_library_genres(self):
        """'chill' should expand to mood genres that exist in the library."""
        result = classify_prompt("something chill for a rainy day", LIB_GENRES, LIB_DECADES)

        assert result.method == "rules"
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert "Ambient" in result.genres
        assert "Lo-Fi" in result.genres
        assert result.decades == []

    def test_genre_and_decade_hints_combine(self):
        """Genre and decade words should both be picked up."""
        result = classify_prompt("melancholy 90s alternative", LIB_GENRES, LIB_DECADES)

        assert result.method == "rules"
        assert "Alternative" in result.genres
        assert result.decades == ["1990s"]

    def test_is_genre_keyword_distinguishes_genre_from_mood(self):
        """Genre names are specific; mood/activity words are not."""
        assert is_genre_keyword("jazz") is True
        assert is_genre_keyword("shoegaze") is True
        assert is_genre_keyword("road trip") is False
        assert is_genre_keyword("chill") is False
        assert is_genre_keyword("party") is False

    def test_decade_only_prompt(self):
        """Era-only requests are still useful and should not need an LLM."""
        result = classify_prompt("music from the 90s", ["Rock"], LIB_DECADES)

        assert result.method == "rules"
        assert result.decades == ["1990s"]
        assert result.genres == []

    def test_unknown_prompt_is_low_confidence(self):
        """Prompts with no recognizable hints should signal the caller to fall back."""
        result = classify_prompt("tell me a joke", LIB_GENRES, LIB_DECADES)

        assert result.method == "none"
        assert not result.confident
        assert result.confidence < CONFIDENCE_THRESHOLD

    def test_confidence_depends_on_keyword_count_not_genre_spread(self):
        """One keyword expanding to many genres must not out-score two keywords."""
        one = classify_prompt("chill", LIB_GENRES, [])
        two = classify_prompt("melancholy alternative", LIB_GENRES, [])

        assert two.confidence > one.confidence


class TestEmbeddingMatch:
    """Embedding similarity against genre descriptions."""

    def test_descriptive_phrase_matches_genre(self):
        """A phrase that rules can't parse should match via embeddings."""
        result = classify_prompt("ethereal wall of sound guitars", LIB_GENRES, [])

        assert result.method == "embeddings"
        assert result.confident
        assert "Shoegaze" in result.genres

    def test_rules_take_priority_over_embeddings(self):
        """A rule keyword should short-circuit the embedding stage."""
        result = classify_prompt("rock", LIB_GENRES, [])

        assert result.method == "rules"

    def test_nonsense_prompt_matches_nothing(self):
        """Random text shouldn't fabricate an embedding match."""
        result = classify_prompt("asdf qwerty zxcv", LIB_GENRES, [])

        assert result.method == "none"
        assert result.genres == []


class TestAvailableMapping:
    """Mapping candidate genres onto the user's actual library."""

    def test_exact_match_preserves_library_casing(self):
        assert match_genres_to_available(["lo-fi"], ["Lo-Fi", "Rock"]) == ["Lo-Fi"]

    def test_substring_match(self):
        """A candidate contained in a longer library genre should match it."""
        assert match_genres_to_available(["post rock"], ["Post-Rock", "Jazz"]) == ["Post-Rock"]

    def test_fuzzy_match_handles_typo(self):
        assert match_genres_to_available(["alternitive"], ["Alternative", "Rock"]) == ["Alternative"]

    def test_unavailable_genres_are_dropped(self):
        result = classify_prompt("jazz please", ["Ambient", "Rock"], [])

        for genre in result.genres:
            assert genre in ["Ambient", "Rock"]

    def test_results_follow_library_order(self):
        assert match_genres_to_available(
            ["jazz", "rock"], ["Rock", "Jazz"]
        ) == ["Rock", "Jazz"]


class TestHintExtraction:
    """Lower-level hint extractors."""

    def test_genre_hints_are_lowercased(self):
        hints = extract_genre_hints("chill party vibes")

        assert "ambient" in hints
        assert "edm" in hints

    def test_decade_hints(self):
        assert extract_decade_hints("90s") == {"1990s"}
        assert extract_decade_hints("classic") == {"1970s", "1980s", "1990s"}

    def test_no_hints(self):
        assert extract_genre_hints("asdf qwerty") == set()
        assert extract_decade_hints("asdf qwerty") == set()


class TestPromptClassificationModel:
    """The dataclass contract used by analyzer.analyze_prompt()."""

    def test_confident_requires_filters(self):
        """An empty classification is never 'confident' even with high score."""
        empty = PromptClassification(genres=[], decades=[], confidence=0.99)
        assert not empty.confident

    def test_confident_with_filters(self):
        filled = PromptClassification(genres=["Jazz"], decades=[], confidence=0.7)
        assert filled.confident


class TestPromptEvidence:
    """The classifier reports which prompt phrases drove each filter."""

    def test_matched_hints_name_prompt_phrases(self):
        result = classify_prompt("melancholy 90s alternative", LIB_GENRES, LIB_DECADES)

        assert result.method == "rules"
        assert "alternative" in result.matched_hints
        assert "Alternative" in result.matched_hints["alternative"]

    def test_hints_record_phrases_with_no_library_match(self):
        """A genre the library lacks is reported with an empty match list."""
        result = classify_prompt("disco road trip", LIB_GENRES, LIB_DECADES)

        assert "disco" in result.matched_hints
        assert result.matched_hints["disco"] == []

    def test_hints_are_empty_when_no_keyword_fires(self):
        result = classify_prompt("asdf qwerty", LIB_GENRES, LIB_DECADES)

        assert result.matched_hints == {}

    def test_exclusions_are_library_genres_left_out(self):
        """Anything listed as excluded is a real library genre, not a suggestion."""
        result = classify_prompt("lo-fi home recorded tapes", LIB_GENRES, LIB_DECADES)

        for excluded in result.excluded_genres:
            assert excluded in LIB_GENRES
        assert set(result.excluded_genres).isdisjoint(result.genres)


# Library artists the fixture prompts can name or ban.
LIB_ARTISTS = ["Nickelback", "Yes", "Metallica", "No Doubt", "The Beatles"]


class TestNegatedArtistExtraction:
    """A named ban becomes a hard artist exclusion."""

    def test_artist_after_a_negation_cue_is_banned(self):
        assert extract_negated_artists("grunge but not Nickelback".lower(), LIB_ARTISTS) == ["Nickelback"]

    def test_ban_wins_over_a_positive_mention(self):
        result = classify_prompt(
            "grunge but not Nickelback", LIB_GENRES, LIB_DECADES, available_artists=LIB_ARTISTS
        )
        assert "Nickelback" in result.exclude_artists
        assert "Nickelback" not in result.artists

    def test_a_plain_mention_is_not_a_ban(self):
        assert extract_negated_artists("some metallica only", LIB_ARTISTS) == []

    def test_bans_are_impossible_without_a_roster(self):
        assert extract_negated_artists("grunge but not nickelback", []) == []


class TestNegatedDecadeExtraction:
    """A decade can be ruled out as well as requested."""

    def test_decade_after_a_negation_cue_is_banned(self):
        assert extract_negated_decades("modern post-rock but no 2010s", LIB_DECADES) == ["2010s"]

    def test_requested_decade_is_not_banned(self):
        assert extract_negated_decades("90s alternative", LIB_DECADES) == []

    def test_classified_decade_ban_stays_out_of_the_suggestions(self):
        result = classify_prompt("modern post-rock but no 2010s", LIB_GENRES, LIB_DECADES)
        assert result.exclude_decades == ["2010s"]
        assert "2010s" not in result.decades


class TestPlayWindowExtraction:
    """Familiarity words become a window on Plex's play counter."""

    def test_well_loved_floor(self):
        assert extract_play_window("my most played blues") == (3, 0, False)

    def test_off_rotation_ceiling(self):
        assert extract_play_window("deep cuts, off rotation") == (0, 2, False)

    def test_ceiling_is_read_before_floor(self):
        assert extract_play_window("not the most played jazz") == (0, 2, False)

    def test_unheard_gets_its_own_flag(self):
        assert extract_play_window("hidden gems, studio albums only") == (0, 0, True)

    def test_unheard_does_not_become_a_zero_ceiling(self):
        result = classify_prompt("hidden gems, studio albums only", LIB_GENRES, LIB_DECADES)
        assert result.unheard_only is True
        assert result.max_plays == 0

    def test_lengths_are_not_play_counts(self):
        assert extract_play_window("nothing over 3 minutes") == (0, 0, False)

    def test_no_familiarity_words_is_unbounded(self):
        assert extract_play_window("quiet evening jazz") == (0, 0, False)


class TestYearWindowExtraction:
    """Release dates become a year window, one notch finer than decades."""

    def test_explicit_span(self):
        assert extract_year_window("between 1975 and 1985") == (1975, 1985)

    def test_from_to_span(self):
        assert extract_year_window("shoegaze from 1990 to 1999") == (1990, 1999)

    def test_since_bound_has_no_ceiling(self):
        assert extract_year_window("recorded after 2015") == (2015, 0)

    def test_bare_year_pins_one_year(self):
        assert extract_year_window("recorded in 1977") == (1977, 1977)

    def test_recent_window_is_anchored_to_today(self):
        assert extract_year_window("the last three years") == (date.today().year - 3, 0)

    def test_decade_words_stay_with_the_decade_filter(self):
        assert extract_year_window("shoegaze from the 2010s") == (0, 0)

    def test_numbers_that_are_not_years_are_ignored(self):
        assert extract_year_window("4000 plays") == (0, 0)


class TestCompilationExclusion:
    """A request for original albums drops the Various-Artists rows."""

    def test_no_compilations(self):
        assert extract_compilation_exclusion("no compilations please") is True

    def test_studio_albums_only(self):
        assert extract_compilation_exclusion("studio albums only") is True

    def test_naming_the_placeholders_is_not_exclusion(self):
        assert extract_compilation_exclusion("various artists best of the 80s") is False


class TestDurationNegation:
    """A negated length phrase is a ceiling, not a floor."""

    def test_nothing_over_is_a_ceiling(self):
        assert extract_duration_window("no songs over 3 minutes") == (0, 180_000)

    def test_nothing_longer_than_is_a_ceiling(self):
        assert extract_duration_window("nothing longer than 4 minutes") == (0, 240_000)

    def test_ban_in_another_clause_does_not_flip_the_window(self):
        assert extract_duration_window("no metal, over 3 minutes") == (180_000, 0)

    def test_under_is_still_a_ceiling(self):
        assert extract_duration_window("under 2 minutes") == (0, 120_000)


class TestNarrowFacetsConfidentlyClassify:
    """Facets with no genre keyword still resolve locally, at no token cost."""

    def test_year_window_alone(self):
        result = classify_prompt("between 1975 and 1985", LIB_GENRES, LIB_DECADES)
        assert result.method == "rules"
        assert result.confident
        assert (result.min_year, result.max_year) == (1975, 1985)

    def test_familiarity_alone(self):
        result = classify_prompt("hidden gems, studio albums only", LIB_GENRES, LIB_DECADES)
        assert result.method == "rules"
        assert result.confidence > CONFIDENCE_THRESHOLD
        assert result.unheard_only is True
        assert result.exclude_compilations is True

    def test_soft_cues_alone_stay_below_the_threshold(self):
        """Mood words alone must not stop the caller from consulting the LLM."""
        result = classify_prompt("something for a rainy afternoon", LIB_GENRES, LIB_DECADES)
        assert not result.confident
