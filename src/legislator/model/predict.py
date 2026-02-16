"""Bill passage prediction using trained logistic regression weights.

Loads weights from weights.json and scores TrackedBill instances.
No external ML dependencies — just multiplication and sigmoid.

v2 changes:
  - Supports v2 weights format (threshold, state_passage_rates, etc.)
  - Falls back gracefully to v1 format
  - Uses optimal threshold from training instead of fixed 0.5
  - Passes state_passage_rate to feature extraction
  - Updated display names for new/removed features
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
      - model_version: int (1 or 2)

    Returns None if no trained model is available.
    """
    model = _load_model()
    if model is None:
        return None

    version = model.get("version", 1)
    weights = model["weights"]
    bias = model["bias"]
    means = model["means"]
    stds = model["stds"]
    threshold = model.get("threshold", 0.5)
    feature_names = model.get("feature_names", FEATURE_NAMES)

    # Get state passage rate for this bill's state (v2 only)
    state_passage_rate = 0.0
    if version >= 2:
        state_rates = model.get("state_passage_rates", {})
        bill_state = getattr(bill, "state", "")
        if bill_state:
            state_passage_rate = state_rates.get(bill_state.upper(), 0.0)

    # Extract raw features
    raw_features = extract_from_tracked_bill(
        bill, state_passage_rate=state_passage_rate
    )

    # Normalize using training statistics
    normalized = []
    for i, name in enumerate(feature_names):
        val = raw_features.get(name, 0.0)
        norm_val = (val - means[i]) / stds[i] if stds[i] != 0 else 0.0
        normalized.append(norm_val)

    # Compute logit
    z = sum(w * x for w, x in zip(weights, normalized)) + bias
    probability = _sigmoid(z)
    score = max(0, min(100, round(probability * 100)))

    # Compute per-feature contributions
    contributions = {}
    for i, name in enumerate(feature_names):
        contrib = weights[i] * normalized[i]
        contributions[name] = round(contrib, 4)

    # Sort by absolute contribution (most impactful first)
    sorted_contributions = dict(sorted(
        contributions.items(), key=lambda x: abs(x[1]), reverse=True
    ))

    # Label based on threshold-adjusted score
    # Map probability relative to threshold into label buckets
    if probability >= threshold + 0.25:
        label = "Very Likely"
    elif probability >= threshold + 0.05:
        label = "Likely"
    elif probability >= threshold - 0.15:
        label = "Possible"
    elif probability >= threshold - 0.35:
        label = "Unlikely"
    else:
        label = "Very Unlikely"

    return {
        "probability": round(probability, 4),
        "score": score,
        "label": label,
        "feature_contributions": sorted_contributions,
        "raw_features": {name: raw_features.get(name, 0.0)
                         for name in feature_names},
        "model_metrics": model.get("metrics", {}),
        "cv_metrics": model.get("cv_metrics", {}),
        "trained_at": model.get("trained_at", "unknown"),
        "model_version": version,
        "threshold": threshold,
    }


def _describe_factor(feature_key: str, raw_value: float, direction: str) -> str:
    """Generate a plain-English contextual description for a factor."""
    v = raw_value
    pos = direction == "positive"

    descriptions = {
        "sponsor_count": (
            f"This bill has {int(v)} sponsor{'s' if v != 1 else ''} — "
            + ("a strong coalition signals support" if v >= 5 else
               "more sponsors would strengthen its chances" if v < 3 else
               "a moderate level of backing")
        ),
        "primary_sponsor_count": (
            f"{int(v)} primary sponsor{'s' if v != 1 else ''} — "
            + ("multiple primary sponsors show serious intent" if v >= 2 else
               "having just one primary sponsor is typical")
        ),
        "cosponsor_count": (
            f"{int(v)} co-sponsor{'s' if v != 1 else ''} — "
            + ("broad co-sponsorship is a strong signal" if v >= 5 else
               "limited co-sponsorship so far")
        ),
        "is_bipartisan": (
            "Has sponsors from both parties — bipartisan bills pass at higher rates"
            if v > 0 else
            "Single-party sponsorship — bipartisan bills historically do better"
        ),
        "minority_party_sponsors": (
            f"{int(v)} sponsor{'s' if v != 1 else ''} from the minority party — "
            + ("cross-aisle support is a strong indicator" if v >= 2 else
               "some cross-party buy-in" if v == 1 else
               "no minority party sponsors yet")
        ),
        "sponsor_party_majority": (
            f"{v:.0%} of sponsors are in the majority party — "
            + ("strong alignment with the controlling party" if v >= 0.7 else
               "mixed party alignment" if v >= 0.3 else
               "sponsors are mostly in the minority party")
        ),
        "senate_origin": (
            "Originated in the Senate" if v > 0 else "Originated in the House"
        ),
        "is_resolution": (
            "This is a resolution, not a standard bill" if v > 0 else
            "Standard bill type"
        ),
        "num_subjects": (
            f"Tagged with {int(v)} subject{'s' if v != 1 else ''} — "
            + ("narrowly focused" if v <= 2 else "broadly scoped")
        ),
        "focused_scope": (
            "Narrowly focused bill — tends to face less opposition" if v > 0 else
            "Broader scope bill"
        ),
        "amends_existing_law": (
            "Amends existing law rather than creating new statute" if v > 0 else
            "Creates new statutory language"
        ),
        "committee_referral": (
            "Has been referred to committee — a standard first step" if v > 0 else
            "Not yet referred to committee"
        ),
        "committee_passage": (
            "Cleared committee — a major milestone that most bills never reach" if v > 0 else
            "Has not yet cleared committee"
        ),
        "num_actions": (
            f"{int(round(v))} legislative action{'s' if round(v) != 1 else ''} on record"
        ),
        "early_action_count": (
            f"{int(v)} action{'s' if v != 1 else ''} in the first 30 days — "
            + ("strong early momentum" if v >= 3 else
               "moderate early activity" if v >= 1 else
               "slow start out of the gate")
        ),
        "days_since_introduction": (
            f"Introduced {int(v)} day{'s' if v != 1 else ''} ago"
        ),
        "session_pct_at_intro": (
            f"Introduced at the {v:.0%} mark of the session — "
            + ("early introduction gives more runway" if v <= 0.25 else
               "mid-session introduction" if v <= 0.6 else
               "late introduction leaves little time")
        ),
        "action_density_30d": (
            f"Recent activity rate: {v:.1f} actions/day over 30 days — "
            + ("very active" if v >= 0.3 else
               "some recent activity" if v > 0 else
               "no recent activity")
        ),
        "title_length": f"Title is {int(v)} characters long",
        "has_fiscal_note": (
            "Has a fiscal note attached — indicates budget impact review" if v > 0 else
            "No fiscal note"
        ),
        "solar_category_count": (
            f"Covers {int(v)} solar policy categor{'y' if v == 1 else 'ies'}"
            if v > 0 else "No specific solar policy categories detected"
        ),
        "has_solar_keywords": (
            "Contains solar energy keywords" if v > 0 else
            "No solar-specific language detected"
        ),
        "state_passage_rate": (
            f"This state passes {v:.0%} of introduced bills — "
            + ("above average" if v >= 0.25 else
               "below average" if v <= 0.15 else
               "near the national average")
        ),
    }

    if feature_key in descriptions:
        return descriptions[feature_key]
    if feature_key.startswith("text_hash_"):
        return "Language pattern in the bill text " + ("associated with passage" if pos else "associated with failure")
    return ""


def get_top_factors(prediction: dict, top_n: int = 5) -> list[dict]:
    """Extract the top positive and negative contributing features.

    Returns list of dicts with 'feature', 'contribution', 'direction',
    'raw_value', and 'description'.
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
        "sponsor_party_majority": "Majority party alignment",
        "senate_origin": "Senate origin",
        "is_resolution": "Resolution type",
        "num_subjects": "Subject tags",
        "focused_scope": "Focused scope",
        "amends_existing_law": "Amends existing law",
        "committee_referral": "Committee referral",
        "committee_passage": "Cleared committee",
        "num_actions": "Legislative actions",
        "early_action_count": "Early momentum (30d)",
        "days_since_introduction": "Days active",
        "session_pct_at_intro": "Introduction timing",
        "action_density_30d": "Recent activity (30d)",
        "title_length": "Title complexity",
        "has_fiscal_note": "Fiscal impact",
        "solar_category_count": "Solar policy categories",
        "has_solar_keywords": "Solar relevance",
        "state_passage_rate": "State passage rate",
    }
    # Text hash features get a generic name
    for i in range(50):
        display_names[f"text_hash_{i}"] = f"Text signal #{i}"

    factors = []
    for name, contrib in contribs.items():
        # Skip tiny contributions (especially text hash features)
        if abs(contrib) < 0.01:
            continue
        direction = "positive" if contrib > 0 else "negative"
        raw_val = raw.get(name, 0)
        factors.append({
            "feature": display_names.get(name, name),
            "feature_key": name,
            "contribution": contrib,
            "direction": direction,
            "raw_value": raw_val,
            "description": _describe_factor(name, raw_val, direction),
        })

    # Return top N by absolute contribution
    return factors[:top_n]
