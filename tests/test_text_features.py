"""Tests for model/text_features.py — tokenization and feature hashing."""

import pytest

from legislator.model.text_features import (
    tokenize, feature_hash, extract_text_features, reverse_hash_map,
    _stable_hash, NUM_BUCKETS, TEXT_FEATURE_NAMES,
)


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Solar energy development in California")
        assert "solar" in tokens
        assert "energy" in tokens
        assert "california" in tokens

    def test_stopwords_removed(self):
        tokens = tokenize("This is a bill about the solar energy")
        # "this", "about" are common stopwords; "bill" is a legislative stopword
        assert "this" not in tokens
        assert "about" not in tokens
        assert "bill" not in tokens
        # "solar" and "energy" are NOT stopwords, so they stay
        assert "solar" in tokens
        assert "energy" in tokens

    def test_legislative_stopwords_removed(self):
        tokens = tokenize("The senate committee enacted this statute amendment")
        # "senate", "committee", "enacted", "statute" are legislative stopwords
        assert "senate" not in tokens
        assert "enacted" not in tokens

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize(None) == []

    def test_short_tokens_removed(self):
        """Tokens shorter than 3 characters are removed."""
        tokens = tokenize("An AB CD solar")
        # "an" and "ab" and "cd" are < 3 chars or won't match regex
        assert "solar" in tokens

    def test_numeric_tokens(self):
        """Tokens starting with a letter can contain digits."""
        tokens = tokenize("section42 rule3b")
        # _TOKEN_RE = r'[a-z][a-z0-9]+' — these should match
        assert "section42" in tokens
        assert "rule3b" in tokens

    def test_bigrams_generated(self):
        """Bigrams are appended when bigrams=True (default)."""
        tokens = tokenize("solar energy development")
        assert "solar_energy" in tokens
        assert "energy_development" in tokens
        # Unigrams still present
        assert "solar" in tokens
        assert "energy" in tokens
        assert "development" in tokens

    def test_bigrams_disabled(self):
        """No bigrams when bigrams=False."""
        tokens = tokenize("solar energy development", bigrams=False)
        assert "solar_energy" not in tokens
        assert "solar" in tokens

    def test_bigrams_single_token(self):
        """No bigrams when only one unigram remains after filtering."""
        tokens = tokenize("the solar")  # "the" is a stopword
        assert "solar" in tokens
        # No bigrams possible with a single unigram
        assert all("_" not in t for t in tokens)


class TestStableHash:
    def test_deterministic(self):
        """Same input always produces same hash."""
        assert _stable_hash("solar") == _stable_hash("solar")
        assert _stable_hash("energy") == _stable_hash("energy")

    def test_different_inputs_differ(self):
        assert _stable_hash("solar") != _stable_hash("wind")

    def test_returns_int(self):
        assert isinstance(_stable_hash("test"), int)

    def test_32bit_range(self):
        h = _stable_hash("some long input string here")
        assert 0 <= h <= 0xFFFFFFFF


class TestFeatureHash:
    def test_output_length(self):
        result = feature_hash(["solar", "energy"], num_buckets=50)
        assert len(result) == 50

    def test_empty_tokens(self):
        result = feature_hash([], num_buckets=10)
        assert result == [0.0] * 10

    def test_signed_hashing(self):
        """Feature hashing uses signed values (+1 or -1)."""
        result = feature_hash(["solar"], num_buckets=50)
        # Exactly one bucket should be non-zero
        nonzero = [v for v in result if v != 0.0]
        assert len(nonzero) == 1
        assert nonzero[0] in (1.0, -1.0)

    def test_collision_handling(self):
        """Multiple tokens can map to the same bucket."""
        # With enough tokens and few buckets, collisions will happen
        tokens = [f"token{i}" for i in range(100)]
        result = feature_hash(tokens, num_buckets=5)
        # All buckets should have some activity
        assert all(v != 0.0 for v in result)

    def test_deterministic(self):
        tokens = ["solar", "energy", "tax", "credit"]
        r1 = feature_hash(tokens, num_buckets=20)
        r2 = feature_hash(tokens, num_buckets=20)
        assert r1 == r2


class TestExtractTextFeatures:
    def test_full_pipeline(self):
        result = extract_text_features("Solar energy tax credit bill")
        assert len(result) == NUM_BUCKETS

    def test_empty_string(self):
        result = extract_text_features("")
        assert result == [0.0] * NUM_BUCKETS

    def test_feature_names_count(self):
        assert len(TEXT_FEATURE_NAMES) == NUM_BUCKETS
        assert TEXT_FEATURE_NAMES[0] == "text_hash_0"
        assert TEXT_FEATURE_NAMES[-1] == f"text_hash_{NUM_BUCKETS - 1}"


class TestReverseHashMap:
    def test_basic_mapping(self):
        """Tokens should map to buckets consistently."""
        mapping = reverse_hash_map("solar energy development")
        # Flatten all tokens from the mapping
        all_tokens = []
        for token_list in mapping.values():
            all_tokens.extend(t for t, _sign in token_list)
        assert "solar" in all_tokens
        assert "energy" in all_tokens
        assert "development" in all_tokens

    def test_bigrams_included(self):
        """Bigrams should appear in the mapping."""
        mapping = reverse_hash_map("solar energy development")
        all_tokens = []
        for token_list in mapping.values():
            all_tokens.extend(t for t, _sign in token_list)
        assert "solar_energy" in all_tokens
        assert "energy_development" in all_tokens

    def test_consistent_with_feature_hash(self):
        """Buckets in reverse_hash_map should match feature_hash output."""
        text = "net metering incentives for renewable energy"
        mapping = reverse_hash_map(text)
        features = extract_text_features(text)
        # Every bucket in the mapping should have a non-zero feature value
        for idx in mapping:
            assert features[idx] != 0.0

    def test_empty_input(self):
        assert reverse_hash_map("") == {}

    def test_sign_included(self):
        """Each entry should include a sign (+1.0 or -1.0)."""
        mapping = reverse_hash_map("solar energy")
        for token_list in mapping.values():
            for _token, sign in token_list:
                assert sign in (1.0, -1.0)
