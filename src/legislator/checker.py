"""Change detection logic and data models for tracked bills."""

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

    @property
    def status_text(self) -> str:
        return STATUS_CODES.get(self.status, f"Unknown ({self.status})")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status_text"] = self.status_text
        return d


@dataclass
class BillChange:
    bill: TrackedBill
    old_status: int
    new_status: int
    new_progress_events: list[dict]
    new_history_actions: list[dict]


def load_tracked_bills(path: Path) -> list[TrackedBill]:
    """Load tracked bills from JSON file. Returns empty list if file missing."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [TrackedBill(**b) for b in data]


def save_tracked_bills(bills: list[TrackedBill], path: Path) -> None:
    """Save tracked bills to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(b) for b in bills], f, indent=2)


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

    change = BillChange(
        bill=tracked,
        old_status=tracked.status,
        new_status=bill_data["status"],
        new_progress_events=new_progress,
        new_history_actions=new_history,
    )

    # Update tracked bill in place
    tracked.change_hash = new_hash
    tracked.status = bill_data["status"]
    tracked.progress_events = api_event_codes
    tracked.url = bill_data.get("url", tracked.url)
    tracked.state_link = bill_data.get("state_link", tracked.state_link)
    if api_history:
        latest = max(api_history, key=lambda h: h["date"])
        tracked.last_history_date = latest["date"]
        tracked.last_history_action = latest["action"]

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
