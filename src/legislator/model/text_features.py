"""Pure-Python text feature extraction using the hashing trick.

Tokenizes bill text and maps tokens into a fixed-size feature vector
via signed feature hashing (no vocabulary needed, zero dependencies).

Based on the hashing trick used in Vowpal Wabbit and similar systems.
See: Weinberger et al., "Feature Hashing for Large Scale Multitask Learning" (2009).
"""

import re
from typing import List

# Number of hash buckets for text features
NUM_BUCKETS = 500

# Feature names for the hash buckets
TEXT_FEATURE_NAMES = [f"text_hash_{i}" for i in range(NUM_BUCKETS)]

# Common English stopwords (no external dependency needed)
_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "if", "in",
    "into", "is", "it", "its", "itself", "just", "may", "me", "might",
    "more", "most", "must", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "per", "same", "shall", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "upon", "us", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
    # Legislative boilerplate stopwords
    "act", "section", "chapter", "title", "code", "statute", "enacted",
    "provide", "amend", "relate", "relating", "provide", "providing",
    "whereas", "therefore", "herein", "thereof", "therein", "hereby",
    "shall", "state", "states", "bill", "resolution", "senate", "house",
    "assembly", "committee", "session", "legislature", "legislative",
})

# Regex for splitting text into tokens
_TOKEN_RE = re.compile(r'[a-z][a-z0-9]+')


def tokenize(text: str, bigrams: bool = True) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens with optional bigrams.

    Removes stopwords and tokens shorter than 3 characters.
    When bigrams=True, appends adjacent-token bigrams (e.g. "net_metering").
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    unigrams = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]
    if bigrams and len(unigrams) >= 2:
        bi = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
        return unigrams + bi
    return unigrams


def _stable_hash(s: str) -> int:
    """Deterministic string hash (Python's built-in hash varies across runs).

    Uses djb2 algorithm for cross-platform consistency.
    """
    h = 5381
    for c in s:
        h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
    return h


def feature_hash(tokens: List[str], num_buckets: int = NUM_BUCKETS) -> List[float]:
    """Map tokens into a fixed-size feature vector using signed hashing.

    Each token is hashed to a bucket index, and a second hash determines
    the sign (+1 or -1). This reduces collision bias compared to unsigned
    hashing. The result is a list of floats representing the text signal.

    Args:
        tokens: List of string tokens from tokenize().
        num_buckets: Size of the output feature vector.

    Returns:
        List of floats of length num_buckets.
    """
    buckets = [0.0] * num_buckets
    for token in tokens:
        h = _stable_hash(token)
        idx = h % num_buckets
        # Use a bit from the hash to determine sign
        sign = 1.0 if (h >> 16) & 1 == 0 else -1.0
        buckets[idx] += sign
    return buckets


def extract_text_features(text: str, num_buckets: int = NUM_BUCKETS) -> List[float]:
    """Full pipeline: tokenize text and return hashed feature vector.

    Args:
        text: Raw bill title, description, or full text.
        num_buckets: Number of hash buckets.

    Returns:
        List of floats of length num_buckets.
    """
    tokens = tokenize(text)
    return feature_hash(tokens, num_buckets)
