"""Passage likelihood scoring and session calendar awareness for tracked bills."""

from collections import namedtuple
from datetime import date, datetime, timedelta
from typing import Optional
import re

from legislator.checker import TrackedBill, PROGRESS_EVENTS, STATUS_CODES


# --- Passage Likelihood Model ---

DimensionResult = namedtuple("DimensionResult", ["score", "max_score", "detail"])


def _parse_date(d: str) -> Optional[date]:
    """Parse YYYY-MM-DD string to date, or None on failure."""
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _score_procedural(bill: TrackedBill) -> DimensionResult:
    """0-30 points based on legislative stage advancement."""
    MAX = 30
    events = set(bill.progress_events)

    if bill.status == 4:
        return DimensionResult(MAX, MAX, f"Bill has passed (+{MAX})")
    if bill.status in (5, 6):
        return DimensionResult(0, MAX, f"Bill is {bill.status_text}")
    if bill.status == 3:  # Enrolled
        return DimensionResult(27, MAX, "Enrolled — passed both chambers (+27)")
    if bill.status == 2:  # Engrossed
        return DimensionResult(22, MAX, "Engrossed — passed origin chamber (+22)")

    # Status 1 (Introduced) — differentiate by progress events
    if 11 in events:  # Committee Report DNP
        return DimensionResult(0, MAX, "Committee reported Do Not Pass (+0)")
    if 10 in events:  # Committee Report Pass
        return DimensionResult(16, MAX, "Committee reported favorably (+16)")
    if 9 in events:   # Committee Referral
        return DimensionResult(8, MAX, "Referred to committee (+8)")

    return DimensionResult(3, MAX, "Introduced, awaiting committee referral (+3)")


def _score_sponsors(bill: TrackedBill) -> DimensionResult:
    """0-20 points based on sponsor quality and quantity."""
    MAX = 20
    sponsors = bill.sponsors
    if not sponsors:
        return DimensionResult(0, MAX, "No sponsor data")

    points = 0
    details = []

    # Primary sponsor exists
    primaries = [s for s in sponsors if s.get("sponsor_type") in
                 ("Primary Sponsor", "Sponsor")]
    if primaries:
        points += 2
        if len(primaries) > 1:
            points += 2
            details.append(f"{len(primaries)} primary sponsors")
        else:
            details.append("Has primary sponsor")

    # Co-sponsor count
    cosponsors = [s for s in sponsors if s.get("sponsor_type") == "Co-Sponsor"]
    n_co = len(cosponsors)
    if n_co >= 10:
        points += 6
    elif n_co >= 5:
        points += 4
    elif n_co >= 2:
        points += 2
    if n_co > 0:
        details.append(f"{n_co} co-sponsor(s)")

    # Cross-chamber sponsors (bicameral)
    roles = {s.get("role", "") for s in sponsors}
    if "Rep" in roles and "Sen" in roles:
        points += 3
        details.append("bicameral")

    # Joint sponsors
    if any(s.get("sponsor_type") == "Joint Sponsor" for s in sponsors):
        points += 3
        details.append("joint sponsors")

    points = min(points, MAX)
    return DimensionResult(points, MAX, "; ".join(details) + f" (+{points})")


def _score_bipartisan(bill: TrackedBill) -> DimensionResult:
    """0-12 points based on depth of cross-party support."""
    MAX = 12
    sponsors = bill.sponsors
    if not sponsors:
        return DimensionResult(0, MAX, "No sponsor data")

    parties = {}
    for s in sponsors:
        p = s.get("party", "")
        if p:
            parties[p] = parties.get(p, 0) + 1

    num_parties = len(parties)
    if num_parties < 2:
        return DimensionResult(0, MAX, "Single-party sponsors (+0)")

    if num_parties >= 3:
        return DimensionResult(12, MAX,
                               f"Tri-partisan: {', '.join(sorted(parties))} (+12)")

    # Two parties — scale by minority party depth
    sorted_counts = sorted(parties.values())
    minority_count = sorted_counts[0]
    party_names = ", ".join(sorted(parties))

    if minority_count >= 3:
        pts = 8
    elif minority_count >= 2:
        pts = 6
    else:
        pts = 4  # Single token cosponsor from other party

    return DimensionResult(pts, MAX,
                           f"Bipartisan: {party_names} ({minority_count} minority) (+{pts})")


def _score_momentum(bill: TrackedBill) -> DimensionResult:
    """0-15 points based on activity recency and velocity."""
    MAX = 15
    points = 0
    details = []
    today = date.today()

    # Recency of last action
    if bill.last_history_date:
        last_dt = _parse_date(bill.last_history_date)
        if last_dt:
            days_ago = (today - last_dt).days
            if days_ago <= 7:
                points += 5
                details.append("active in last 7 days")
            elif days_ago <= 30:
                points += 3
                details.append("active in last 30 days")
            elif days_ago <= 60:
                points += 1
                details.append("active in last 60 days")
            elif days_ago > 90:
                points -= 3
                details.append("inactive 90+ days")

    # Velocity between milestones
    if len(bill.progress_details) >= 2:
        dates = sorted(
            [_parse_date(p["date"]) for p in bill.progress_details if p.get("date")],
            key=lambda d: d or date.min
        )
        dates = [d for d in dates if d is not None]
        if len(dates) >= 2:
            intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval < 14:
                points += 4
                details.append("fast advancement")
            elif avg_interval < 45:
                points += 2
                details.append("moderate pace")

    # Upcoming calendar events
    today_str = today.isoformat()
    future_events = [c for c in bill.calendar if c.get("date", "") >= today_str]
    if future_events:
        points += 3
        details.append(f"{len(future_events)} upcoming event(s)")

    # Action density in last 30 days
    cutoff = (today - timedelta(days=30)).isoformat()
    recent_actions = [h for h in bill.history if h.get("date", "") >= cutoff]
    if len(recent_actions) >= 3:
        points += 3
        details.append("high recent activity")
    elif len(recent_actions) >= 1:
        points += 1

    points = max(0, min(points, MAX))
    return DimensionResult(points, MAX, "; ".join(details) + f" (+{points})" if details else f"No activity data (+{points})")


def _get_session_pct_elapsed(bill: TrackedBill) -> Optional[float]:
    """Return session % elapsed (0-100), or None if no session data."""
    if not bill.session_year_end:
        return None
    try:
        session_end = date(bill.session_year_end, 12, 31)
        session_start = date(bill.session_year_start, 1, 1)
    except (ValueError, TypeError):
        return None
    total_days = (session_end - session_start).days or 1
    elapsed = (date.today() - session_start).days
    return max(0, min(100, (elapsed / total_days) * 100))


def _score_timing(bill: TrackedBill) -> DimensionResult:
    """0-10 points based on bill progress vs session timeline."""
    MAX = 10
    pct = _get_session_pct_elapsed(bill)
    if pct is None:
        return DimensionResult(5, MAX, "No session data, assuming mid-range (+5)")

    # Has the bill cleared committee?
    events = set(bill.progress_events)
    cleared_committee = bill.status >= 2 or 10 in events
    in_committee = 9 in events and not cleared_committee
    just_introduced = not in_committee and not cleared_committee and bill.status == 1

    if pct < 25:
        # Early session — most bills are fine
        if bill.status >= 2:
            pts = 10
            detail = "Early session, already advancing"
        else:
            pts = 7
            detail = "Early in session, plenty of time"
    elif pct < 50:
        if cleared_committee:
            pts = 8
            detail = "Pre-midpoint, cleared committee"
        elif in_committee:
            pts = 6
            detail = "Pre-midpoint, in committee"
        else:
            pts = 4
            detail = "Pre-midpoint, not yet in committee"
    elif pct < 75:
        if bill.status >= 2:
            pts = 8
            detail = "Past midpoint, passed a chamber"
        elif cleared_committee:
            pts = 6
            detail = "Past midpoint, cleared committee"
        elif in_committee:
            pts = 2
            detail = "Past midpoint, still in committee"
        else:
            pts = 1
            detail = "Past midpoint, no committee action"
    else:
        # Session > 75% elapsed
        if bill.status >= 3:
            pts = 7
            detail = "Late session, enrolled or beyond"
        elif bill.status == 2:
            pts = 5
            detail = "Late session, engrossed"
        elif cleared_committee:
            pts = 2
            detail = "Late session, only cleared committee"
        else:
            pts = 0
            detail = "Late session, not through committee"

    return DimensionResult(pts, MAX, f"{detail} ({pct:.0f}% elapsed) (+{pts})")


def _score_structure(bill: TrackedBill) -> DimensionResult:
    """0-5 points based on bill structural characteristics."""
    MAX = 5
    points = 0
    details = []

    bn = bill.bill_number.upper().strip()

    # Senate origin
    if bn.startswith("S") or bn.startswith("SB"):
        points += 1
        details.append("Senate origin")

    # Resolution type (higher pass rate but less legal weight)
    if re.search(r'\b(SJR|HJR|SR|HR|SCR|HCR)\b', bn):
        points += 1
        details.append("resolution")

    # Focused scope
    n_subjects = len(bill.subjects)
    if 1 <= n_subjects <= 2:
        points += 2
        details.append("focused scope")
    elif n_subjects == 0:
        pass  # no data, no penalty
    # 3+ subjects = broader bill, no bonus

    # Amendment to existing law
    desc_lower = (bill.description or "").lower()
    if any(kw in desc_lower for kw in ("amend", "relating to", "revise", "modify")):
        points += 1
        details.append("amends existing law")

    points = min(points, MAX)
    return DimensionResult(points, MAX, "; ".join(details) + f" (+{points})" if details else f"(+{points})")


def _compute_confidence(bill: TrackedBill) -> str:
    """Rate data completeness as high/medium/low."""
    completeness = 0
    if bill.sponsors:
        completeness += 1
    if bill.history:
        completeness += 1
    if bill.progress_details:
        completeness += 1
    if bill.session_year_end:
        completeness += 1
    if bill.calendar:
        completeness += 0.5

    if completeness >= 3.5:
        return "high"
    if completeness >= 2:
        return "medium"
    return "low"


def _collect_risks(bill: TrackedBill, dims: dict) -> list[str]:
    """Identify key risk factors for bill passage."""
    risks = []

    # No bipartisan support
    if dims["bipartisan"].score == 0 and bill.sponsors:
        risks.append("No bipartisan support detected")

    # Stalled in committee late in session
    pct = _get_session_pct_elapsed(bill)
    if pct is not None and pct > 50 and bill.status == 1:
        events = set(bill.progress_events)
        if 10 not in events:  # hasn't cleared committee
            risks.append(f"Still in committee with session {pct:.0f}% elapsed")

    # No recent activity
    if bill.last_history_date:
        last_dt = _parse_date(bill.last_history_date)
        if last_dt and (date.today() - last_dt).days > 90:
            risks.append("No legislative activity in 90+ days")

    # Few sponsors
    if len(bill.sponsors) <= 1:
        risks.append("Only 1 sponsor — limited coalition support")

    # Committee DNP
    if 11 in set(bill.progress_events):
        risks.append("Committee reported Do Not Pass")

    # No upcoming events and session active
    today_str = date.today().isoformat()
    future_events = [c for c in bill.calendar if c.get("date", "") >= today_str]
    if not future_events and bill.status < 4:
        if pct is not None and 0 < pct < 100:
            risks.append("No upcoming hearings or votes scheduled")

    return risks


def _compute_model_score(bill: TrackedBill) -> Optional[dict]:
    """Try to compute passage score using the trained logistic regression model.

    Returns a formatted result dict matching the heuristic output shape,
    or None if the model is not available.
    """
    try:
        from legislator.model.predict import model_available, predict_passage, get_top_factors
    except ImportError:
        return None

    if not model_available():
        return None

    prediction = predict_passage(bill)
    if prediction is None:
        return None

    # Build dimension-like breakdown from feature contributions
    # Group features into logical dimensions for the UI
    dimension_groups = {
        "sponsors": {
            "label": "Sponsor Strength",
            "features": ["sponsor_count", "primary_sponsor_count", "cosponsor_count"],
        },
        "bipartisan": {
            "label": "Bipartisan Support",
            "features": ["is_bipartisan", "minority_party_sponsors"],
        },
        "procedural": {
            "label": "Procedural Progress",
            "features": ["committee_referral", "committee_passage", "passed_one_chamber"],
        },
        "momentum": {
            "label": "Momentum",
            "features": ["num_actions", "days_since_introduction", "action_density_30d"],
        },
        "timing": {
            "label": "Session Timing",
            "features": ["session_pct_at_intro"],
        },
        "structure": {
            "label": "Bill Structure",
            "features": ["senate_origin", "is_resolution", "num_subjects",
                         "focused_scope", "amends_existing_law", "has_companion"],
        },
    }

    contribs = prediction["feature_contributions"]
    dimensions = {}
    for dim_key, group in dimension_groups.items():
        total_contrib = sum(contribs.get(f, 0) for f in group["features"])
        # Normalize contribution to a 0-10 display scale
        # Positive contributions help, negative hurt
        display_score = max(0, min(10, round((total_contrib + 5) * 1)))
        dimensions[dim_key] = {
            "score": display_score,
            "max": 10,
            "detail": f"{group['label']}: {total_contrib:+.2f} contribution",
        }

    # Build factors from top contributors
    top = get_top_factors(prediction, top_n=6)
    factors = []
    for f in top:
        direction = "helps" if f["direction"] == "positive" else "hurts"
        factors.append(f"{f['feature']}: {direction} ({f['contribution']:+.3f})")

    confidence = _compute_confidence(bill)

    # Use heuristic risk detection (still useful regardless of model)
    heuristic_dims = {}  # minimal stub for _collect_risks
    dim_bipartisan_check = _score_bipartisan(bill)
    heuristic_dims["bipartisan"] = dim_bipartisan_check
    risks = _collect_risks(bill, heuristic_dims)

    return {
        "score": prediction["score"],
        "label": prediction["label"],
        "confidence": confidence,
        "dimensions": dimensions,
        "factors": factors,
        "risks": risks,
        "model": "trained",
        "model_metrics": prediction.get("model_metrics", {}),
        "trained_at": prediction.get("trained_at", ""),
    }


def compute_passage_likelihood(bill: TrackedBill) -> dict:
    """Compute a 0-100 passage likelihood score.

    Uses a trained logistic regression model if available (trained on
    historical Open States data). Falls back to heuristic scoring if
    no trained model exists.

    Returns dict with 'score', 'label', 'confidence', 'dimensions',
    'factors', 'risks', and optionally 'model' indicating the source.
    """
    # Terminal states short-circuit
    if bill.status in (5, 6):
        return {
            "score": 0,
            "label": "Dead",
            "confidence": "high",
            "dimensions": {},
            "factors": [f"Bill is {bill.status_text}"],
            "risks": [f"Bill has been {bill.status_text.lower()}"],
        }
    if bill.status == 4:
        return {
            "score": 98,
            "label": "Passed",
            "confidence": "high",
            "dimensions": {},
            "factors": ["Bill has passed"],
            "risks": [],
        }

    # Try the trained model first
    model_result = _compute_model_score(bill)
    if model_result is not None:
        return model_result

    # Fallback: heuristic scoring
    dim_procedural = _score_procedural(bill)
    dim_sponsors = _score_sponsors(bill)
    dim_bipartisan = _score_bipartisan(bill)
    dim_momentum = _score_momentum(bill)
    dim_timing = _score_timing(bill)
    dim_structure = _score_structure(bill)

    dims = {
        "procedural": dim_procedural,
        "sponsors": dim_sponsors,
        "bipartisan": dim_bipartisan,
        "momentum": dim_momentum,
        "timing": dim_timing,
        "structure": dim_structure,
    }

    raw_score = sum(d.score for d in dims.values())

    # Session timing decay multiplier for stalled bills
    pct = _get_session_pct_elapsed(bill)
    events = set(bill.progress_events)
    stalled_late = (
        pct is not None
        and pct > 70
        and bill.status == 1
        and 10 not in events  # hasn't cleared committee
    )
    if stalled_late:
        raw_score = int(raw_score * 0.5)

    # Normalize to 0-100 scale (max raw is 92)
    score = max(0, min(100, round(raw_score * 100 / 92)))

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

    confidence = _compute_confidence(bill)
    risks = _collect_risks(bill, dims)

    factors = [d.detail for d in dims.values() if d.detail]

    dimensions_dict = {
        name: {"score": d.score, "max": d.max_score, "detail": d.detail}
        for name, d in dims.items()
    }

    return {
        "score": score,
        "label": label,
        "confidence": confidence,
        "dimensions": dimensions_dict,
        "factors": factors,
        "risks": risks,
        "model": "heuristic",
    }


# --- Session Calendar Awareness ---

def get_session_status(bill: TrackedBill) -> Optional[dict]:
    """Determine session status and warnings for a bill.

    Returns dict with:
      - 'session_name': str
      - 'year_start', 'year_end': int
      - 'days_remaining': int or None (estimated)
      - 'warning': str or None
      - 'is_ending_soon': bool
    """
    if not bill.session_year_end:
        return None

    today = date.today()

    # Estimate session end as Dec 31 of year_end (real sine_die varies by state)
    try:
        session_end = date(bill.session_year_end, 12, 31)
        session_start = date(bill.session_year_start, 1, 1)
    except (ValueError, TypeError):
        return None

    days_remaining = (session_end - today).days
    total_days = (session_end - session_start).days or 1
    pct_elapsed = ((today - session_start).days / total_days) * 100

    warning = None
    is_ending_soon = False

    if days_remaining < 0:
        warning = "Session has ended"
        is_ending_soon = False
    elif days_remaining <= 30:
        warning = f"Session ending soon — ~{days_remaining} days remaining"
        is_ending_soon = True
    elif days_remaining <= 90:
        warning = f"Session in final quarter — ~{days_remaining} days remaining"
        is_ending_soon = True

    # Warn if bill hasn't advanced and session is progressing
    if bill.status == 1 and pct_elapsed > 50 and days_remaining > 0:
        stall_msg = "Bill still at Introduced stage past session midpoint"
        warning = f"{warning}; {stall_msg}" if warning else stall_msg

    return {
        "session_name": bill.session_name,
        "year_start": bill.session_year_start,
        "year_end": bill.session_year_end,
        "days_remaining": max(days_remaining, 0) if days_remaining >= 0 else 0,
        "pct_elapsed": round(min(pct_elapsed, 100), 1),
        "warning": warning,
        "is_ending_soon": is_ending_soon,
    }
