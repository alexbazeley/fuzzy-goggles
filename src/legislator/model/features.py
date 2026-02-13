"""Feature extraction for bill passage prediction.

Extracts numeric features from either:
  - Open States bill dicts (for training on historical data)
  - TrackedBill objects (for live prediction on tracked bills)

All features are numeric (float). The feature vector is a plain dict
mapping feature names to values, suitable for logistic regression.
"""

from datetime import date, datetime
from typing import Optional

# Canonical feature order — must match between training and prediction
FEATURE_NAMES = [
    "sponsor_count",
    "primary_sponsor_count",
    "cosponsor_count",
    "is_bipartisan",
    "minority_party_sponsors",
    "senate_origin",
    "is_resolution",
    "num_subjects",
    "focused_scope",
    "amends_existing_law",
    "committee_referral",
    "committee_passage",
    "passed_one_chamber",
    "num_actions",
    "days_since_introduction",
    "session_pct_at_intro",
    "has_companion",
    "action_density_30d",
]


def _parse_date(d: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(d[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Extract from Open States bill dict (for training)
# ---------------------------------------------------------------------------

def extract_from_openstates(bill: dict, session_start: Optional[date] = None,
                            session_end: Optional[date] = None) -> dict:
    """Extract feature dict from an Open States API bill object.

    Requires bill to have been fetched with include=sponsorships,actions,votes.
    """
    features = {name: 0.0 for name in FEATURE_NAMES}

    # --- Sponsors ---
    sponsorships = bill.get("sponsorships", [])
    features["sponsor_count"] = float(len(sponsorships))
    features["primary_sponsor_count"] = float(
        sum(1 for s in sponsorships if s.get("primary"))
    )
    features["cosponsor_count"] = float(
        sum(1 for s in sponsorships if not s.get("primary"))
    )

    # Bipartisan: check sponsor parties
    parties = set()
    party_counts = {}
    for s in sponsorships:
        person = s.get("person") or {}
        party = person.get("party", "") or s.get("party", "")
        if party:
            parties.add(party)
            party_counts[party] = party_counts.get(party, 0) + 1

    features["is_bipartisan"] = 1.0 if len(parties) >= 2 else 0.0
    if len(parties) >= 2:
        sorted_counts = sorted(party_counts.values())
        features["minority_party_sponsors"] = float(sorted_counts[0])

    # --- Bill structure ---
    identifier = bill.get("identifier", "").upper()
    from_org = (bill.get("from_organization") or {}).get("name", "").lower()
    features["senate_origin"] = 1.0 if ("senate" in from_org or
                                         identifier.startswith("S")) else 0.0

    classification = bill.get("classification", [])
    features["is_resolution"] = 1.0 if any(
        "resolution" in c for c in classification) else 0.0

    subjects = bill.get("subject", [])
    features["num_subjects"] = float(len(subjects))
    features["focused_scope"] = 1.0 if 1 <= len(subjects) <= 2 else 0.0

    title = (bill.get("title") or "").lower()
    amend_keywords = ("amend", "relating to", "revise", "modify")
    features["amends_existing_law"] = 1.0 if any(
        kw in title for kw in amend_keywords) else 0.0

    # --- Actions / procedural progress ---
    actions = bill.get("actions", [])
    features["num_actions"] = float(len(actions))

    action_classes = set()
    for a in actions:
        for c in (a.get("classification") or []):
            action_classes.add(c)

    features["committee_referral"] = 1.0 if "referral-committee" in action_classes else 0.0
    features["committee_passage"] = 1.0 if "committee-passage" in action_classes else 0.0
    features["passed_one_chamber"] = 1.0 if "passage" in action_classes else 0.0

    # --- Timing ---
    first_action_str = bill.get("first_action_date") or ""
    latest_action_str = bill.get("latest_action_date") or ""
    first_dt = _parse_date(first_action_str)
    latest_dt = _parse_date(latest_action_str)

    if first_dt and latest_dt:
        features["days_since_introduction"] = float((latest_dt - first_dt).days)

    if first_dt and session_start and session_end:
        total_days = (session_end - session_start).days or 1
        elapsed = (first_dt - session_start).days
        features["session_pct_at_intro"] = max(0.0, min(100.0,
            (elapsed / total_days) * 100))

    # --- Action density (last 30 days of the bill's life) ---
    if latest_dt and actions:
        cutoff = latest_dt.isoformat()
        from datetime import timedelta
        cutoff_30 = (latest_dt - timedelta(days=30)).isoformat()
        recent = [a for a in actions if (a.get("date") or "") >= cutoff_30]
        features["action_density_30d"] = float(len(recent))

    # --- Related bills ---
    related = bill.get("related_bills", [])
    features["has_companion"] = 1.0 if any(
        r.get("relation_type") == "companion" for r in related) else 0.0

    return features


def label_from_openstates(bill: dict) -> Optional[int]:
    """Determine outcome label from an Open States bill.

    Returns:
      1 = became law (executive-signature or became-law action)
      0 = did not pass (session ended without passage)
      None = indeterminate (still active / can't determine)
    """
    actions = bill.get("actions", [])
    action_classes = set()
    for a in actions:
        for c in (a.get("classification") or []):
            action_classes.add(c)

    if "executive-signature" in action_classes or "became-law" in action_classes:
        return 1

    # Check for veto
    if "executive-veto" in action_classes:
        return 0

    # If the bill had a passage action in at least one chamber but no signature,
    # we can't definitively label it without knowing the session ended.
    # For training, we'll label bills from completed sessions that lack
    # a signature as 0.
    return 0


# ---------------------------------------------------------------------------
# Extract from TrackedBill (for live prediction)
# ---------------------------------------------------------------------------

def extract_from_tracked_bill(bill) -> dict:
    """Extract feature dict from a TrackedBill instance.

    Maps TrackedBill fields to the same features used in training.
    """
    features = {name: 0.0 for name in FEATURE_NAMES}

    # --- Sponsors ---
    sponsors = bill.sponsors or []
    features["sponsor_count"] = float(len(sponsors))
    features["primary_sponsor_count"] = float(
        sum(1 for s in sponsors if s.get("sponsor_type") in
            ("Primary Sponsor", "Sponsor"))
    )
    features["cosponsor_count"] = float(
        sum(1 for s in sponsors if s.get("sponsor_type") == "Co-Sponsor")
    )

    parties = set()
    party_counts = {}
    for s in sponsors:
        p = s.get("party", "")
        if p:
            parties.add(p)
            party_counts[p] = party_counts.get(p, 0) + 1

    features["is_bipartisan"] = 1.0 if len(parties) >= 2 else 0.0
    if len(parties) >= 2:
        sorted_counts = sorted(party_counts.values())
        features["minority_party_sponsors"] = float(sorted_counts[0])

    # --- Bill structure ---
    bn = bill.bill_number.upper().strip()
    features["senate_origin"] = 1.0 if bn.startswith("S") else 0.0
    import re
    features["is_resolution"] = 1.0 if re.search(
        r'\b(SJR|HJR|SR|HR|SCR|HCR)\b', bn) else 0.0

    features["num_subjects"] = float(len(bill.subjects))
    features["focused_scope"] = 1.0 if 1 <= len(bill.subjects) <= 2 else 0.0

    desc = (bill.description or "").lower()
    features["amends_existing_law"] = 1.0 if any(
        kw in desc for kw in ("amend", "relating to", "revise", "modify")
    ) else 0.0

    # --- Procedural progress (from progress_events) ---
    events = set(bill.progress_events)
    features["committee_referral"] = 1.0 if 9 in events else 0.0
    features["committee_passage"] = 1.0 if 10 in events else 0.0
    features["passed_one_chamber"] = 1.0 if bill.status >= 2 else 0.0

    # --- Actions ---
    features["num_actions"] = float(len(bill.history))

    # Days since introduction
    if bill.history:
        dates = sorted(h.get("date", "") for h in bill.history if h.get("date"))
        if dates:
            first_dt = _parse_date(dates[0])
            last_dt = _parse_date(dates[-1])
            if first_dt and last_dt:
                features["days_since_introduction"] = float(
                    (last_dt - first_dt).days)

    # Session timing
    if bill.session_year_start and bill.session_year_end and bill.history:
        try:
            session_start = date(bill.session_year_start, 1, 1)
            session_end = date(bill.session_year_end, 12, 31)
            dates_sorted = sorted(
                h.get("date", "") for h in bill.history if h.get("date"))
            if dates_sorted:
                first_dt = _parse_date(dates_sorted[0])
                if first_dt:
                    total_days = (session_end - session_start).days or 1
                    elapsed = (first_dt - session_start).days
                    features["session_pct_at_intro"] = max(0.0, min(100.0,
                        (elapsed / total_days) * 100))
        except (ValueError, TypeError):
            pass

    # Action density (last 30 days from today)
    from datetime import timedelta
    today = date.today()
    cutoff_30 = (today - timedelta(days=30)).isoformat()
    recent = [h for h in bill.history if (h.get("date") or "") >= cutoff_30]
    features["action_density_30d"] = float(len(recent))

    # Companion — not directly available from LegiScan, default 0
    features["has_companion"] = 0.0

    return features
