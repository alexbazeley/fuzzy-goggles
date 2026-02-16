"""Bill passage prediction using trained logistic regression weights.

Loads weights from weights.json and scores TrackedBill instances.
No external ML dependencies — just multiplication and sigmoid.

v2 changes:
  - Supports v2 weights format (threshold, state_passage_rates, etc.)
  - Falls back gracefully to v1 format
  - Uses optimal threshold from training instead of fixed 0.5
  - Passes state_passage_rate to feature extraction
  - Updated display names for new/removed features

v3 changes:
  - Platt scaling for calibrated probabilities
  - State passage rate as post-hoc Bayesian prior (not a model feature)
  - Resolution note when model was trained excluding resolutions
  - 500 text hash features (up from 50)
  - New features: days_to_first_committee, committee_speed
  - Removed features: days_since_introduction, action_density_30d,
    state_passage_rate
"""

import json
import math
import re
from pathlib import Path
from typing import Optional

from legislator.model.features import FEATURE_NAMES, extract_from_tracked_bill
from legislator.model.text_features import TEXT_FEATURE_NAMES, reverse_hash_map

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
      - probability: float 0.0-1.0 (calibrated in v3+)
      - score: int 0-100 (percentage)
      - label: str (Very Likely, Likely, Possible, Unlikely, Very Unlikely)
      - feature_contributions: dict mapping feature names to their
        contribution to the score (weight * normalized_value)
      - model_metrics: dict with test set performance info
      - model_version: int (1, 2, or 3)
      - resolution_note: str (v3 only, if bill is a resolution)

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

    # v2: pass state_passage_rate as a model feature
    # v3: state_passage_rate removed from features, applied as post-hoc prior
    if version == 2:
        state_passage_rate = 0.0
        state_rates = model.get("state_passage_rates", {})
        bill_state = getattr(bill, "state", "")
        if bill_state:
            state_passage_rate = state_rates.get(bill_state.upper(), 0.0)
        raw_features = extract_from_tracked_bill(bill)
        # v2 models still expect state_passage_rate in feature vector
        raw_features["state_passage_rate"] = state_passage_rate
    else:
        raw_features = extract_from_tracked_bill(bill)

    # Normalize using training statistics
    normalized = []
    for i, name in enumerate(feature_names):
        val = raw_features.get(name, 0.0)
        norm_val = (val - means[i]) / stds[i] if stds[i] != 0 else 0.0
        normalized.append(norm_val)

    # Compute logit
    z = sum(w * x for w, x in zip(weights, normalized)) + bias

    # v3: apply Platt scaling for calibrated probabilities
    if version >= 3:
        cal_A = model.get("calibration_A", 1.0)
        cal_B = model.get("calibration_B", 0.0)
        z = cal_A * z + cal_B

    probability = _sigmoid(z)

    # v3: blend with state passage rate as Bayesian prior
    if version >= 3:
        state_rates = model.get("state_passage_rates", {})
        bill_state = getattr(bill, "state", "")
        state_rate = state_rates.get(bill_state.upper(), 0.0) if bill_state else 0.0
        if state_rate > 0:
            probability = 0.9 * probability + 0.1 * state_rate

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

    # Build reverse hash map so we can show which tokens drive each text bucket
    title_raw = getattr(bill, "title", "") or ""
    description_raw = getattr(bill, "description", "") or ""
    full_text = f"{title_raw} {description_raw}".strip()
    bucket_tokens = reverse_hash_map(full_text)
    text_hash_tokens = {}
    for idx, token_list in bucket_tokens.items():
        text_hash_tokens[f"text_hash_{idx}"] = [t for t, _sign in token_list]

    result = {
        "probability": round(probability, 4),
        "score": score,
        "label": label,
        "feature_contributions": sorted_contributions,
        "raw_features": {name: raw_features.get(name, 0.0)
                         for name in feature_names},
        "text_hash_tokens": text_hash_tokens,
        "model_metrics": model.get("metrics", {}),
        "cv_metrics": model.get("cv_metrics", {}),
        "trained_at": model.get("trained_at", "unknown"),
        "model_version": version,
        "threshold": threshold,
    }

    # v3: note if bill is a resolution and model excluded resolutions
    if version >= 3 and model.get("exclude_resolutions", False):
        bn = getattr(bill, "bill_number", "") or ""
        if re.search(r'\b(SJR|HJR|SR|HR|SCR|HCR)\b', bn.upper()):
            result["resolution_note"] = (
                "This is a resolution. The model was trained on substantive "
                "legislation only; prediction may be less reliable for "
                "resolutions."
            )

    return result


def _format_token(token: str) -> str:
    """Format a token for display: replace underscores with spaces for bigrams."""
    return token.replace("_", " ")


def _describe_factor(feature_key: str, raw_value: float, direction: str,
                     tokens: list[str] | None = None) -> str:
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
            f"{int(round(v))} legislative action{'s' if round(v) != 1 else ''} "
            "in the first 60 days"
        ),
        "early_action_count": (
            f"{int(v)} action{'s' if v != 1 else ''} in the first 30 days — "
            + ("strong early momentum" if v >= 3 else
               "moderate early activity" if v >= 1 else
               "slow start out of the gate")
        ),
        "days_to_first_committee": (
            f"{int(v)} day{'s' if v != 1 else ''} from introduction to "
            "committee referral"
            + (" — quick committee action" if v <= 14 else
               "" if v <= 60 else " — slow to reach committee")
        ),
        "session_pct_at_intro": (
            f"Introduced at the {v:.0%} mark of the session — "
            + ("early introduction gives more runway" if v <= 0.25 else
               "mid-session introduction" if v <= 0.6 else
               "late introduction leaves little time")
        ),
        "committee_speed": (
            f"{int(v)} day{'s' if v != 1 else ''} from committee referral "
            "to passage"
            + (" — fast committee action" if v <= 30 else
               "" if v <= 90 else " — slow committee progress")
            if v > 0 else
            "Bill has not yet cleared committee"
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
    }

    if feature_key in descriptions:
        return descriptions[feature_key]
    if feature_key.startswith("text_hash_"):
        if tokens:
            formatted = [_format_token(t) for t in tokens[:3]]
            quoted = ", ".join(f"'{t}'" for t in formatted)
            return (f"Bill text contains {quoted} — language pattern "
                    + ("associated with passage" if pos else "associated with failure"))
        return ("Language pattern in the bill text "
                + ("associated with passage" if pos else "associated with failure"))
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
    text_hash_tokens = prediction.get("text_hash_tokens", {})

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
        "num_actions": "Early legislative actions",
        "early_action_count": "Early momentum (30d)",
        "days_to_first_committee": "Time to committee",
        "session_pct_at_intro": "Introduction timing",
        "committee_speed": "Committee speed",
        "title_length": "Title complexity",
        "has_fiscal_note": "Fiscal impact",
        "solar_category_count": "Solar policy categories",
        "has_solar_keywords": "Solar relevance",
    }

    factors = []
    for name, contrib in contribs.items():
        # Skip tiny contributions (especially text hash features)
        if abs(contrib) < 0.01:
            continue
        direction = "positive" if contrib > 0 else "negative"
        raw_val = raw.get(name, 0)

        # For text hash features, build display name from actual tokens
        tokens = text_hash_tokens.get(name)
        if name.startswith("text_hash_") and tokens:
            formatted = [_format_token(t) for t in tokens[:3]]
            display = "Text: " + ", ".join(formatted)
        else:
            display = display_names.get(name, name)

        factors.append({
            "feature": display,
            "feature_key": name,
            "contribution": contrib,
            "direction": direction,
            "raw_value": raw_val,
            "description": _describe_factor(name, raw_val, direction, tokens),
        })

    # Return top N by absolute contribution
    return factors[:top_n]
