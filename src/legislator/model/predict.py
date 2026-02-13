"""Bill passage prediction using trained logistic regression weights.

Loads weights from weights.json and scores TrackedBill instances.
No external ML dependencies — just multiplication and sigmoid.
"""

import json
import math
from pathlib import Path
from typing import Optional

from legislator.model.features import FEATURE_NAMES, extract_from_tracked_bill

WEIGHTS_PATH = Path(__file__).parent / "weights.json"

_model_cache = None


def _load_model() -> Optional[dict]:
    """Load model weights from disk. Cached after first load."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not WEIGHTS_PATH.exists():
        return None
    with open(WEIGHTS_PATH) as f:
        _model_cache = json.load(f)
    return _model_cache


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def model_available() -> bool:
    """Check whether a trained model exists on disk."""
    return WEIGHTS_PATH.exists()


def predict_passage(bill) -> Optional[dict]:
    """Predict passage likelihood for a TrackedBill.

    Returns dict with:
      - probability: float 0.0-1.0 (raw model output)
      - score: int 0-100 (percentage)
      - label: str (Very Likely, Likely, Possible, Unlikely, Very Unlikely)
      - feature_contributions: dict mapping feature names to their
        contribution to the score (weight * normalized_value)
      - model_metrics: dict with test set performance info

    Returns None if no trained model is available.
    """
    model = _load_model()
    if model is None:
        return None

    weights = model["weights"]
    bias = model["bias"]
    means = model["means"]
    stds = model["stds"]

    # Extract raw features
    raw_features = extract_from_tracked_bill(bill)

    # Normalize using training statistics
    normalized = []
    for i, name in enumerate(FEATURE_NAMES):
        val = raw_features.get(name, 0.0)
        norm_val = (val - means[i]) / stds[i] if stds[i] != 0 else 0.0
        normalized.append(norm_val)

    # Compute logit
    z = sum(w * x for w, x in zip(weights, normalized)) + bias
    probability = _sigmoid(z)
    score = max(0, min(100, round(probability * 100)))

    # Compute per-feature contributions
    contributions = {}
    for i, name in enumerate(FEATURE_NAMES):
        contrib = weights[i] * normalized[i]
        contributions[name] = round(contrib, 4)

    # Sort by absolute contribution (most impactful first)
    sorted_contributions = dict(sorted(
        contributions.items(), key=lambda x: abs(x[1]), reverse=True
    ))

    # Label
    if score >= 75:
        label = "Very Likely"
    elif score >= 55:
        label = "Likely"
    elif score >= 35:
        label = "Possible"
    elif score >= 15:
        label = "Unlikely"
    else:
        label = "Very Unlikely"

    return {
        "probability": round(probability, 4),
        "score": score,
        "label": label,
        "feature_contributions": sorted_contributions,
        "raw_features": {name: raw_features[name] for name in FEATURE_NAMES},
        "model_metrics": model.get("metrics", {}),
        "trained_at": model.get("trained_at", "unknown"),
    }


def get_top_factors(prediction: dict, top_n: int = 5) -> list[dict]:
    """Extract the top positive and negative contributing features.

    Returns list of dicts with 'feature', 'contribution', 'direction', 'raw_value'.
    """
    if not prediction:
        return []

    contribs = prediction["feature_contributions"]
    raw = prediction.get("raw_features", {})

    # Human-readable feature names
    display_names = {
        "sponsor_count": "Total sponsors",
        "primary_sponsor_count": "Primary sponsors",
        "cosponsor_count": "Co-sponsors",
        "is_bipartisan": "Bipartisan support",
        "minority_party_sponsors": "Cross-party sponsors",
        "senate_origin": "Senate origin",
        "is_resolution": "Resolution type",
        "num_subjects": "Subject tags",
        "focused_scope": "Focused scope",
        "amends_existing_law": "Amends existing law",
        "committee_referral": "Committee referral",
        "committee_passage": "Cleared committee",
        "passed_one_chamber": "Passed a chamber",
        "num_actions": "Legislative actions",
        "days_since_introduction": "Days active",
        "session_pct_at_intro": "Introduction timing",
        "has_companion": "Has companion bill",
        "action_density_30d": "Recent activity (30d)",
    }

    factors = []
    for name, contrib in contribs.items():
        factors.append({
            "feature": display_names.get(name, name),
            "feature_key": name,
            "contribution": contrib,
            "direction": "positive" if contrib > 0 else "negative",
            "raw_value": raw.get(name, 0),
        })

    # Return top N by absolute contribution
    return factors[:top_n]
