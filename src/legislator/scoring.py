"""Impact scoring and session calendar awareness for tracked bills."""

from datetime import date, datetime
from typing import Optional

from legislator.checker import TrackedBill, PROGRESS_EVENTS, STATUS_CODES


# --- Impact Scoring ---

def compute_impact_score(bill: TrackedBill) -> dict:
    """Compute a 0-100 impact/likelihood score for a bill passing.

    Returns dict with 'score', 'label', and 'factors' (list of explanation strings).
    """
    score = 0
    factors = []

    # 1. Status progression (0-40 points)
    status_points = {1: 10, 2: 25, 3: 35, 4: 40, 5: 0, 6: 0}
    sp = status_points.get(bill.status, 0)
    score += sp
    if bill.status >= 2:
        factors.append(f"Advanced to {bill.status_text} (+{sp})")
    elif bill.status == 1:
        factors.append(f"Still at {bill.status_text} stage (+{sp})")

    if bill.status in (5, 6):
        factors.append(f"Bill is {bill.status_text} — unlikely to pass")
        return {"score": 0, "label": "Dead", "factors": factors}

    # 2. Sponsor count (0-20 points)
    num_sponsors = len(bill.sponsors)
    if num_sponsors >= 10:
        sponsor_pts = 20
    elif num_sponsors >= 5:
        sponsor_pts = 15
    elif num_sponsors >= 3:
        sponsor_pts = 10
    elif num_sponsors >= 1:
        sponsor_pts = 5
    else:
        sponsor_pts = 0
    score += sponsor_pts
    if num_sponsors > 0:
        factors.append(f"{num_sponsors} sponsor(s) (+{sponsor_pts})")

    # 3. Bipartisan support (0-15 points)
    parties = {s.get("party", "") for s in bill.sponsors if s.get("party")}
    parties.discard("")
    if len(parties) >= 2:
        score += 15
        factors.append(f"Bipartisan support from {', '.join(sorted(parties))} (+15)")

    # 4. Progress milestones (0-15 points)
    milestone_count = len(bill.progress_events)
    milestone_pts = min(milestone_count * 3, 15)
    score += milestone_pts
    if milestone_count > 0:
        factors.append(f"{milestone_count} milestone(s) reached (+{milestone_pts})")

    # 5. Recent activity bonus (0-10 points)
    if bill.last_history_date:
        try:
            last_dt = datetime.strptime(bill.last_history_date, "%Y-%m-%d").date()
            days_ago = (date.today() - last_dt).days
            if days_ago <= 7:
                score += 10
                factors.append("Active in last 7 days (+10)")
            elif days_ago <= 30:
                score += 5
                factors.append("Active in last 30 days (+5)")
            elif days_ago > 90:
                score -= 5
                factors.append("No activity in 90+ days (-5)")
        except ValueError:
            pass

    score = max(0, min(100, score))

    if score >= 70:
        label = "High"
    elif score >= 40:
        label = "Moderate"
    elif score >= 20:
        label = "Low"
    else:
        label = "Very Low"

    return {"score": score, "label": label, "factors": factors}


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
    current_year = today.year

    # Estimate session end as Dec 31 of year_end (real sine_die varies by state)
    # We use year_end as a rough proxy
    try:
        # Most sessions end within their stated year_end
        # Some states have specific adjourn dates, but we use year boundaries
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
