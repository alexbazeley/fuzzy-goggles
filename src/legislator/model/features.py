"""Feature extraction for bill passage prediction.

Extracts numeric features from either:
  - Open States bill dicts (for training on historical data)
  - TrackedBill objects (for live prediction on tracked bills)

All features are numeric (float). The feature vector is a plain dict
mapping feature names to values, suitable for logistic regression.

Changes from v1:
  - Removed 'passed_one_chamber' (feature leakage / tautological)
  - Removed 'has_companion' (always 0 for LegiScan bills)
  - Log-transformed 'num_actions' to reduce conflation with outcome
  - Added 'early_action_count' (actions within first 30 days)
  - Added 'sponsor_party_majority' (fraction in majority party)
  - Added 'title_length' (log word count of title)
  - Added 'has_fiscal_note' (fiscal/appropriation keywords)
  - Added 'solar_category_count' and 'has_solar_keywords'
  - Added 50 text hash features (hashing trick on title+description)
  - Fixed label_from_openstates() to check session completion
  - Fixed sponsor party extraction to try multiple JSON paths

Changes from v2 (model v3):
  - num_actions capped at 60-day window to prevent temporal leakage
  - Removed 'days_since_introduction' (leaky) → 'days_to_first_committee'
  - Removed 'action_density_30d' (leaky) → 'committee_speed'
  - Removed 'state_passage_rate' (tautological, moved to post-hoc prior)
  - Expanded text hash from 50 → 500 buckets with bigram support
"""

import math
import re
from datetime import date, datetime, timedelta
from typing import Optional

from legislator.model.text_features import (
    NUM_BUCKETS, TEXT_FEATURE_NAMES, extract_text_features,
)

# Solar keyword categories (imported lazily to avoid circular imports)
_SOLAR_KEYWORDS = None


def _get_solar_keywords():
    """Lazily import solar keyword dict."""
    global _SOLAR_KEYWORDS
    if _SOLAR_KEYWORDS is None:
        try:
            from legislator.solar import SOLAR_KEYWORDS
            _SOLAR_KEYWORDS = SOLAR_KEYWORDS
        except ImportError:
            _SOLAR_KEYWORDS = {}
    return _SOLAR_KEYWORDS


# Fiscal/appropriation keywords for has_fiscal_note feature
_FISCAL_KEYWORDS = (
    "appropriat", "fiscal", "revenue", "tax", "budget", "fund",
    "expenditure", "treasury",
)

# Canonical feature order — must match between training and prediction
FEATURE_NAMES = [
    # Sponsor features (6)
    "sponsor_count",
    "primary_sponsor_count",
    "cosponsor_count",
    "is_bipartisan",
    "minority_party_sponsors",
    "sponsor_party_majority",
    # Bill structure features (5)
    "senate_origin",
    "is_resolution",
    "num_subjects",
    "focused_scope",
    "amends_existing_law",
    # Procedural progress features (2) — no more passed_one_chamber
    "committee_referral",
    "committee_passage",
    # Action/momentum features (5) — num_actions capped at 60-day window
    "num_actions",
    "early_action_count",
    "days_to_first_committee",
    "session_pct_at_intro",
    "committee_speed",
    # Text-derived features (4)
    "title_length",
    "has_fiscal_note",
    "solar_category_count",
    "has_solar_keywords",
] + TEXT_FEATURE_NAMES  # 500 text hash features

# Total: 22 structured + 500 text hash = 522 features


def _parse_date(d: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(d[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
    return None


def _count_solar_categories(text: str) -> tuple:
    """Count solar keyword categories matched in text.

    Returns (num_categories, has_any) tuple.
    """
    if not text:
        return 0.0, 0.0
    text_lower = text.lower()
    keywords = _get_solar_keywords()
    matched_categories = set()
    for category, kw_list in keywords.items():
        for kw in kw_list:
            if len(kw) <= 3:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    matched_categories.add(category)
                    break
            else:
                if kw in text_lower:
                    matched_categories.add(category)
                    break
    count = float(len(matched_categories))
    has_any = 1.0 if count > 0 else 0.0
    return count, has_any


def _extract_party(sponsorship: dict) -> str:
    """Extract party from an Open States sponsorship, trying multiple paths.

    Open States bulk data sometimes has party at different nesting levels.
    """
    # Try person.party first (most common in API responses)
    person = sponsorship.get("person") or {}
    party = person.get("party", "")
    if party:
        return party

    # Try person.current_role.party
    role = person.get("current_role") or {}
    party = role.get("party", "")
    if party:
        return party

    # Try person.primary_party (some bulk formats)
    party = person.get("primary_party", "")
    if party:
        return party

    # Try top-level party on the sponsorship itself
    party = sponsorship.get("party", "")
    if party:
        return party

    # Try organization.name (some formats nest party as an org)
    org = sponsorship.get("organization") or {}
    org_name = org.get("name", "")
    if org_name and org_name.lower() in ("democratic", "republican",
                                          "independent", "green",
                                          "libertarian", "nonpartisan"):
        return org_name

    return ""


# Map common party name variants to canonical short form
_PARTY_NORMALIZE = {
    "democratic": "D", "democrat": "D", "dem": "D", "d": "D",
    "republican": "R", "gop": "R", "rep": "R", "r": "R",
    "independent": "I", "ind": "I", "i": "I",
    "green": "G", "g": "G",
    "libertarian": "L", "l": "L",
    "nonpartisan": "NP", "np": "NP",
}


def _normalize_party(party: str) -> str:
    """Normalize party name to a short canonical form."""
    if not party:
        return ""
    return _PARTY_NORMALIZE.get(party.lower().strip(), party.upper().strip())


# ---------------------------------------------------------------------------
# Extract from Open States bill dict (for training)
# ---------------------------------------------------------------------------

def extract_from_openstates(bill: dict, session_start: Optional[date] = None,
                            session_end: Optional[date] = None,
                            chamber_majority: Optional[str] = None) -> dict:
    """Extract feature dict from an Open States API bill object.

    Args:
        bill: Open States bill dict (with sponsorships, actions).
        session_start: Estimated session start date.
        session_end: Estimated session end date.
        chamber_majority: Canonical party code of the majority party
            in the originating chamber (e.g. "D" or "R"). Used for
            sponsor_party_majority feature.
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

    # Bipartisan: check sponsor parties (with improved extraction)
    parties = set()
    party_counts = {}
    for s in sponsorships:
        raw_party = _extract_party(s)
        party = _normalize_party(raw_party)
        if party:
            parties.add(party)
            party_counts[party] = party_counts.get(party, 0) + 1

    features["is_bipartisan"] = 1.0 if len(parties) >= 2 else 0.0
    if len(parties) >= 2:
        sorted_counts = sorted(party_counts.values())
        features["minority_party_sponsors"] = float(sorted_counts[0])

    # Sponsor party majority: fraction of sponsors in the majority party
    if chamber_majority and sponsorships:
        majority_count = party_counts.get(chamber_majority, 0)
        features["sponsor_party_majority"] = majority_count / len(sponsorships)
    elif party_counts:
        # Fallback: use the most common party as proxy for majority
        most_common = max(party_counts, key=party_counts.get)
        features["sponsor_party_majority"] = (
            party_counts[most_common] / len(sponsorships)
        )

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

    # --- Text-derived features ---
    title_raw = bill.get("title") or ""
    abstract = bill.get("abstract") or bill.get("description") or ""
    full_text = f"{title_raw} {abstract}".strip()

    # Title length (log-transformed word count)
    word_count = len(title_raw.split())
    features["title_length"] = math.log(1 + word_count)

    # Fiscal note detection
    title_lower = title  # already lowercased above
    abstract_lower = abstract.lower()
    combined_lower = f"{title_lower} {abstract_lower}"
    features["has_fiscal_note"] = 1.0 if any(
        kw in combined_lower for kw in _FISCAL_KEYWORDS) else 0.0

    # Solar keyword categories
    solar_cats, has_solar = _count_solar_categories(full_text)
    features["solar_category_count"] = solar_cats
    features["has_solar_keywords"] = has_solar

    # Text hash features
    text_hash = extract_text_features(full_text)
    for i, val in enumerate(text_hash):
        features[f"text_hash_{i}"] = val

    # --- Actions / procedural progress ---
    actions = bill.get("actions", [])

    action_classes = set()
    for a in actions:
        for c in (a.get("classification") or []):
            action_classes.add(c)

    features["committee_referral"] = 1.0 if "referral-committee" in action_classes else 0.0
    features["committee_passage"] = 1.0 if "committee-passage" in action_classes else 0.0

    # --- Timing ---
    first_action_str = bill.get("first_action_date") or ""
    first_dt = _parse_date(first_action_str)

    # num_actions: capped at 60-day window to prevent temporal leakage
    if first_dt and actions:
        cutoff_60 = (first_dt + timedelta(days=60)).isoformat()
        early_60 = [a for a in actions if (a.get("date") or "") <= cutoff_60]
        features["num_actions"] = math.log(1 + len(early_60))
    else:
        features["num_actions"] = math.log(1 + len(actions))

    if first_dt and session_start and session_end:
        total_days = (session_end - session_start).days or 1
        elapsed = (first_dt - session_start).days
        features["session_pct_at_intro"] = max(0.0, min(100.0,
            (elapsed / total_days) * 100))

    # --- Early action count (first 30 days) ---
    if first_dt and actions:
        cutoff_30 = (first_dt + timedelta(days=30)).isoformat()
        early = [a for a in actions
                 if (a.get("date") or "") <= cutoff_30]
        features["early_action_count"] = float(len(early))

    # --- days_to_first_committee: days from introduction to first committee action ---
    if first_dt and actions:
        committee_dates = []
        for a in actions:
            cls = a.get("classification") or []
            if "referral-committee" in cls or "committee-passage" in cls:
                d = _parse_date(a.get("date") or "")
                if d and d >= first_dt:
                    committee_dates.append(d)
        if committee_dates:
            features["days_to_first_committee"] = float(
                (min(committee_dates) - first_dt).days)

    # --- committee_speed: days from committee referral to committee passage ---
    if actions:
        referral_dates = []
        passage_dates = []
        for a in actions:
            cls = a.get("classification") or []
            d = _parse_date(a.get("date") or "")
            if d:
                if "referral-committee" in cls:
                    referral_dates.append(d)
                if "committee-passage" in cls:
                    passage_dates.append(d)
        if referral_dates and passage_dates:
            first_referral = min(referral_dates)
            first_passage = min(passage_dates)
            if first_passage >= first_referral:
                features["committee_speed"] = float(
                    (first_passage - first_referral).days)

    return features


def label_from_openstates(bill: dict,
                          session_end_date: Optional[date] = None) -> Optional[int]:
    """Determine outcome label from an Open States bill.

    Args:
        bill: Open States bill dict.
        session_end_date: When the session ended/ends. If in the future,
            the bill is still active and returns None (indeterminate).

    Returns:
      1 = became law (executive-signature, became-law, or veto override)
      0 = did not pass (session ended without passage)
      None = indeterminate (still active / can't determine)
    """
    actions = bill.get("actions", [])
    action_classes = set()
    for a in actions:
        for c in (a.get("classification") or []):
            action_classes.add(c)

    # Positive outcomes
    if "executive-signature" in action_classes or "became-law" in action_classes:
        return 1

    # Veto override = passed despite veto
    if "executive-veto" in action_classes and "veto-override" in action_classes:
        return 1

    # Resolutions that passed (adopted) — they don't get signed
    classification = bill.get("classification", [])
    is_resolution = any("resolution" in c for c in classification)
    if is_resolution and "passage" in action_classes:
        return 1

    # Clear veto without override
    if "executive-veto" in action_classes:
        return 0

    # Check if session has ended — only label as 0 for completed sessions
    if session_end_date is not None:
        today = date.today()
        if session_end_date > today:
            # Session still active, bill might still pass
            return None

    # Session is over (or no date provided — legacy behavior),
    # bill didn't pass
    return 0


# ---------------------------------------------------------------------------
# Extract from TrackedBill (for live prediction)
# ---------------------------------------------------------------------------

def extract_from_tracked_bill(bill) -> dict:
    """Extract feature dict from a TrackedBill instance.

    Maps TrackedBill fields to the same features used in training.

    Args:
        bill: TrackedBill instance.
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
        p = _normalize_party(s.get("party", ""))
        if p:
            parties.add(p)
            party_counts[p] = party_counts.get(p, 0) + 1

    features["is_bipartisan"] = 1.0 if len(parties) >= 2 else 0.0
    if len(parties) >= 2:
        sorted_counts = sorted(party_counts.values())
        features["minority_party_sponsors"] = float(sorted_counts[0])

    # Sponsor party majority (use most common party as proxy)
    if party_counts and sponsors:
        most_common = max(party_counts, key=party_counts.get)
        features["sponsor_party_majority"] = (
            party_counts[most_common] / len(sponsors)
        )

    # --- Bill structure ---
    bn = bill.bill_number.upper().strip()
    features["senate_origin"] = 1.0 if bn.startswith("S") else 0.0
    features["is_resolution"] = 1.0 if re.search(
        r'\b(SJR|HJR|SR|HR|SCR|HCR)\b', bn) else 0.0

    features["num_subjects"] = float(len(bill.subjects))
    features["focused_scope"] = 1.0 if 1 <= len(bill.subjects) <= 2 else 0.0

    desc = (bill.description or "").lower()
    features["amends_existing_law"] = 1.0 if any(
        kw in desc for kw in ("amend", "relating to", "revise", "modify")
    ) else 0.0

    # --- Text-derived features ---
    title_raw = bill.title or ""
    description_raw = bill.description or ""
    full_text = f"{title_raw} {description_raw}".strip()

    # Title length
    word_count = len(title_raw.split())
    features["title_length"] = math.log(1 + word_count)

    # Fiscal note detection
    combined_lower = f"{title_raw} {description_raw}".lower()
    features["has_fiscal_note"] = 1.0 if any(
        kw in combined_lower for kw in _FISCAL_KEYWORDS) else 0.0

    # Solar keyword categories — use cached data if available
    solar_keywords = getattr(bill, "solar_keywords", None) or []
    if solar_keywords:
        # solar_keywords is a list of "Category: keyword" strings
        categories = set()
        for entry in solar_keywords:
            if ":" in entry:
                categories.add(entry.split(":")[0].strip())
        features["solar_category_count"] = float(len(categories))
        features["has_solar_keywords"] = 1.0 if categories else 0.0
    else:
        solar_cats, has_solar = _count_solar_categories(full_text)
        features["solar_category_count"] = solar_cats
        features["has_solar_keywords"] = has_solar

    # Text hash features
    text_hash = extract_text_features(full_text)
    for i, val in enumerate(text_hash):
        features[f"text_hash_{i}"] = val

    # --- Procedural progress (from progress_events) ---
    events = set(bill.progress_events)
    features["committee_referral"] = 1.0 if 9 in events else 0.0
    features["committee_passage"] = 1.0 if 10 in events else 0.0
    # passed_one_chamber REMOVED — feature leakage

    # --- Actions ---
    first_dt = None
    if bill.history:
        dates = sorted(h.get("date", "") for h in bill.history if h.get("date"))
        if dates:
            first_dt = _parse_date(dates[0])

    # num_actions: capped at 60-day window to prevent temporal leakage
    if first_dt and bill.history:
        cutoff_60 = (first_dt + timedelta(days=60)).isoformat()
        early_60 = [h for h in bill.history
                    if (h.get("date") or "") <= cutoff_60]
        features["num_actions"] = math.log(1 + len(early_60))
    else:
        features["num_actions"] = math.log(1 + len(bill.history))

    # Early action count (first 30 days)
    if first_dt and bill.history:
        cutoff_30 = (first_dt + timedelta(days=30)).isoformat()
        early = [h for h in bill.history
                 if (h.get("date") or "") <= cutoff_30]
        features["early_action_count"] = float(len(early))

    # days_to_first_committee: days from introduction to committee referral
    if first_dt and bill.progress_details:
        for pd in bill.progress_details:
            if pd.get("event") == 9:  # committee referral
                ref_dt = _parse_date(pd.get("date") or "")
                if ref_dt and ref_dt >= first_dt:
                    features["days_to_first_committee"] = float(
                        (ref_dt - first_dt).days)
                    break

    # committee_speed: days from committee referral to committee passage
    if bill.progress_details:
        referral_dt = None
        passage_dt = None
        for pd in bill.progress_details:
            evt = pd.get("event")
            d = _parse_date(pd.get("date") or "")
            if d:
                if evt == 9 and referral_dt is None:  # committee referral
                    referral_dt = d
                if evt == 10 and passage_dt is None:  # committee passage
                    passage_dt = d
        if referral_dt and passage_dt and passage_dt >= referral_dt:
            features["committee_speed"] = float(
                (passage_dt - referral_dt).days)

    # Session timing — use Feb 1 / Dec 31 as more realistic session bounds
    if bill.session_year_start and bill.session_year_end and first_dt:
        try:
            session_start = date(bill.session_year_start, 2, 1)
            session_end = date(bill.session_year_end, 12, 31)
            total_days = (session_end - session_start).days or 1
            elapsed = (first_dt - session_start).days
            features["session_pct_at_intro"] = max(0.0, min(100.0,
                (elapsed / total_days) * 100))
        except (ValueError, TypeError):
            pass

    return features
