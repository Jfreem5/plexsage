"""Tests for the rule/embedding prompt classifier (backend.prompt_classifier)."""

from backend.prompt_classifier import (
    CONFIDENCE_THRESHOLD,
    PromptClassification,
    classify_prompt,
    extract_decade_hints,
    extract_genre_hints,
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
