"""Change detection logic and data models for tracked bills."""

import fcntl
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from legislator.api import LegiScanAPI

PROGRESS_EVENTS = {
    1: "Introduced",
    2: "Engrossed",
    3: "Enrolled",
    4: "Passed",
    5: "Vetoed",
    6: "Failed",
    7: "Veto Override",
    8: "Chapter/Act",
    9: "Committee Referral",
    10: "Committee Report Pass",
    11: "Committee Report DNP",
}

STATUS_CODES = {
    1: "Introduced",
    2: "Engrossed",
    3: "Enrolled",
    4: "Passed",
    5: "Vetoed",
    6: "Failed",
}


@dataclass
class Sponsor:
    people_id: int
    name: str
    party: str = ""
    role: str = ""
    sponsor_type: str = ""  # "Primary" or "Co-Sponsor"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalendarEvent:
    date: str
    description: str
    location: str = ""
    event_type: str = ""  # "hearing", "vote", etc.

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackedBill:
    bill_id: int
    state: str
    bill_number: str
    title: str
    url: str
    state_link: str
    change_hash: str
    status: int
    last_history_date: str
    last_history_action: str
    progress_events: list[int] = field(default_factory=list)
    progress_details: list[dict] = field(default_factory=list)  # [{event, date}]
    # New fields
    priority: str = "medium"  # "high", "medium", "low"
    sponsors: list[dict] = field(default_factory=list)
    calendar: list[dict] = field(default_factory=list)
    session_id: int = 0
    session_name: str = ""
    session_year_start: int = 0
    session_year_end: int = 0
    description: str = ""
    subjects: list[str] = field(default_factory=list)
    committee: str = ""
    committee_id: int = 0
    history: list[dict] = field(default_factory=list)
    solar_keywords: list[str] = field(default_factory=list)

    @property
    def status_text(self) -> str:
        return STATUS_CODES.get(self.status, f"Unknown ({self.status})")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status_text"] = self.status_text
        return d


@dataclass
class SponsorChange:
    added: list[dict] = field(default_factory=list)
    removed: list[dict] = field(default_factory=list)


@dataclass
class BillChange:
    bill: TrackedBill
    old_status: int
    new_status: int
    new_progress_events: list[dict]
    new_history_actions: list[dict]
    sponsor_changes: Optional[SponsorChange] = None
    new_calendar_events: list[dict] = field(default_factory=list)


def load_tracked_bills(path: Path) -> list[TrackedBill]:
    """Load tracked bills from JSON file. Returns empty list if file missing."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    bills = []
    for b in data:
        # Handle old data without new fields gracefully
        bills.append(TrackedBill(
            bill_id=b["bill_id"],
            state=b["state"],
            bill_number=b["bill_number"],
            title=b["title"],
            url=b["url"],
            state_link=b["state_link"],
            change_hash=b["change_hash"],
            status=b["status"],
            last_history_date=b["last_history_date"],
            last_history_action=b["last_history_action"],
            progress_events=b.get("progress_events", []),
            priority=b.get("priority", "medium"),
            sponsors=b.get("sponsors", []),
            calendar=b.get("calendar", []),
            session_id=b.get("session_id", 0),
            session_name=b.get("session_name", ""),
            session_year_start=b.get("session_year_start", 0),
            session_year_end=b.get("session_year_end", 0),
            description=b.get("description", ""),
            subjects=b.get("subjects", []),
            committee=b.get("committee", ""),
            committee_id=b.get("committee_id", 0),
            history=b.get("history", []),
            solar_keywords=b.get("solar_keywords", []),
            progress_details=b.get("progress_details", []),
        ))
    return bills


def save_tracked_bills(bills: list[TrackedBill], path: Path) -> None:
    """Save tracked bills to JSON file with file locking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file then rename for atomicity
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump([asdict(b) for b in bills], f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    tmp_path.rename(path)


def _extract_sponsors(bill_data: dict) -> list[dict]:
    """Extract sponsor list from API bill data.

    LegiScan may return party as 'party' or 'party_id', and sponsor type
    as 'sponsor_type', 'sponsor_type_id', or numeric values.
    """
    sponsors_raw = bill_data.get("sponsors", [])
    if isinstance(sponsors_raw, dict):
        # Some API responses wrap sponsors in {"sponsor": [...]} or {0: {...}, 1: {...}}
        if "sponsor" in sponsors_raw:
            sponsors_raw = sponsors_raw["sponsor"]
            if isinstance(sponsors_raw, dict):
                sponsors_raw = [sponsors_raw]  # Single sponsor wrapped in dict
        else:
            sponsors_raw = list(sponsors_raw.values())
    sponsors = []
    for s in sponsors_raw:
        if not isinstance(s, dict):
            continue
        # Build full name from parts if 'name' is missing
        name = s.get("name", "")
        if not name:
            parts = [s.get("first_name", ""), s.get("middle_name", ""), s.get("last_name", "")]
            suffix = s.get("suffix", "")
            name = " ".join(p for p in parts if p)
            if suffix:
                name = f"{name} {suffix}"

        # Party: try 'party', then 'party_id' (LegiScan uses both)
        party = s.get("party", "") or s.get("party_id", "")

        # Role: try 'role', then map 'role_id' (1=Rep, 2=Sen, 3=Joint Conference)
        role = s.get("role", "")
        if not role:
            role_id = s.get("role_id", 0)
            role = {1: "Rep", 2: "Sen", 3: "Joint Conference"}.get(role_id, "")

        # Sponsor type: try 'sponsor_type', then map 'sponsor_type_id'
        # (0=Sponsor, 1=Primary Sponsor, 2=Co-Sponsor, 3=Joint Sponsor)
        sponsor_type = s.get("sponsor_type", "")
        if not sponsor_type:
            st_id = s.get("sponsor_type_id", 0)
            sponsor_type = {
                0: "Sponsor",
                1: "Primary Sponsor",
                2: "Co-Sponsor",
                3: "Joint Sponsor",
            }.get(st_id, "Sponsor")

        sponsors.append({
            "people_id": s.get("people_id", 0),
            "name": name,
            "party": str(party),
            "role": role,
            "sponsor_type": sponsor_type,
        })
    return sponsors


def _extract_calendar(bill_data: dict) -> list[dict]:
    """Extract calendar/hearing events from API bill data."""
    calendar_raw = bill_data.get("calendar", [])
    events = []
    for c in calendar_raw:
        events.append({
            "date": c.get("date", ""),
            "description": c.get("description", c.get("desc", "")),
            "location": c.get("location", ""),
            "event_type": c.get("type", "hearing"),
        })
    return events


def _extract_subjects(bill_data: dict) -> list[str]:
    """Extract subject/topic tags from API bill data."""
    subjects_raw = bill_data.get("subjects", [])
    if isinstance(subjects_raw, list):
        return [s.get("subject_name", str(s)) if isinstance(s, dict) else str(s)
                for s in subjects_raw]
    return []


def _detect_sponsor_changes(old_sponsors: list[dict], new_sponsors: list[dict]) -> Optional[SponsorChange]:
    """Compare sponsor lists and return changes, or None if unchanged."""
    old_ids = {s["people_id"] for s in old_sponsors}
    new_ids = {s["people_id"] for s in new_sponsors}

    added_ids = new_ids - old_ids
    removed_ids = old_ids - new_ids

    if not added_ids and not removed_ids:
        return None

    added = [s for s in new_sponsors if s["people_id"] in added_ids]
    removed = [s for s in old_sponsors if s["people_id"] in removed_ids]
    return SponsorChange(added=added, removed=removed)


def check_bill(api: LegiScanAPI, tracked: TrackedBill) -> Optional[BillChange]:
    """Check a single bill for changes. Returns BillChange if changed, None otherwise."""
    bill_data = api.get_bill(tracked.bill_id)

    new_hash = bill_data["change_hash"]
    if new_hash == tracked.change_hash:
        return None

    # Find new progress events
    api_progress = bill_data.get("progress", [])
    api_event_codes = [p["event"] for p in api_progress]
    new_progress = [p for p in api_progress if p["event"] not in tracked.progress_events]

    # Find new history actions (after our last known date)
    api_history = bill_data.get("history", [])
    new_history = [h for h in api_history if h["date"] > tracked.last_history_date]

    # Detect sponsor changes
    new_sponsors = _extract_sponsors(bill_data)
    sponsor_changes = _detect_sponsor_changes(tracked.sponsors, new_sponsors)

    # Detect new calendar events
    new_calendar = _extract_calendar(bill_data)
    old_cal_dates = {c["date"] + c.get("description", "") for c in tracked.calendar}
    new_cal_events = [c for c in new_calendar
                      if c["date"] + c.get("description", "") not in old_cal_dates]

    change = BillChange(
        bill=tracked,
        old_status=tracked.status,
        new_status=bill_data["status"],
        new_progress_events=new_progress,
        new_history_actions=new_history,
        sponsor_changes=sponsor_changes,
        new_calendar_events=new_cal_events,
    )

    # Update tracked bill in place
    tracked.change_hash = new_hash
    tracked.status = bill_data["status"]
    tracked.progress_events = api_event_codes
    tracked.progress_details = [{"event": p["event"], "date": p.get("date", "")} for p in api_progress]
    tracked.url = bill_data.get("url", tracked.url)
    tracked.state_link = bill_data.get("state_link", tracked.state_link)
    tracked.sponsors = new_sponsors
    tracked.calendar = new_calendar
    tracked.subjects = _extract_subjects(bill_data)
    tracked.description = bill_data.get("description", tracked.description)

    # Update session info
    session = bill_data.get("session", {})
    if session:
        tracked.session_id = session.get("session_id", tracked.session_id)
        tracked.session_name = session.get("session_name", tracked.session_name)
        tracked.session_year_start = session.get("year_start", tracked.session_year_start)
        tracked.session_year_end = session.get("year_end", tracked.session_year_end)

    # Update committee info
    committee = bill_data.get("committee", {})
    if committee:
        tracked.committee = committee.get("name", tracked.committee)
        tracked.committee_id = committee.get("committee_id", tracked.committee_id)

    if api_history:
        latest = max(api_history, key=lambda h: h["date"])
        tracked.last_history_date = latest["date"]
        tracked.last_history_action = latest["action"]
        tracked.history = [{"date": h["date"], "action": h["action"], "chamber": h.get("chamber", ""), "chamber_id": h.get("chamber_id", 0)} for h in api_history]

    return change


def check_all_bills(api: LegiScanAPI, data_path: Path) -> list[BillChange]:
    """Check all tracked bills for changes. Saves updated data. Returns changes."""
    bills = load_tracked_bills(data_path)
    if not bills:
        return []

    changes = []
    for bill in bills:
        try:
            change = check_bill(api, bill)
            if change:
                changes.append(change)
        except Exception as e:
            print(f"Error checking {bill.state} {bill.bill_number}: {e}")

    save_tracked_bills(bills, data_path)
    return changes
