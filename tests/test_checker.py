"""Tests for checker.py — data models, sponsor extraction, change detection."""

import json
import pytest
from pathlib import Path

from legislator.checker import (
    TrackedBill, BillChange, SponsorChange, Sponsor, CalendarEvent,
    PROGRESS_EVENTS, STATUS_CODES,
    _extract_sponsors, _extract_calendar, _extract_subjects,
    _detect_sponsor_changes,
    load_tracked_bills, save_tracked_bills,
)


# --- Data Models ---

class TestTrackedBill:
    def test_status_text_known(self, sample_bill):
        assert sample_bill.status_text == "Introduced"

    def test_status_text_passed(self):
        bill = TrackedBill(
            bill_id=1, state="CA", bill_number="SB1", title="T",
            url="u", state_link="s", change_hash="h", status=4,
            last_history_date="", last_history_action="",
        )
        assert bill.status_text == "Passed"

    def test_status_text_unknown(self):
        bill = TrackedBill(
            bill_id=1, state="CA", bill_number="SB1", title="T",
            url="u", state_link="s", change_hash="h", status=99,
            last_history_date="", last_history_action="",
        )
        assert bill.status_text == "Unknown (99)"

    def test_to_dict(self, sample_bill):
        d = sample_bill.to_dict()
        assert d["bill_id"] == 12345
        assert d["status_text"] == "Introduced"
        assert isinstance(d["sponsors"], list)
        assert isinstance(d["progress_events"], list)

    def test_default_values(self):
        bill = TrackedBill(
            bill_id=1, state="TX", bill_number="HB1", title="Title",
            url="u", state_link="s", change_hash="h", status=1,
            last_history_date="", last_history_action="",
        )
        assert bill.priority == "medium"
        assert bill.sponsors == []
        assert bill.calendar == []
        assert bill.solar_keywords is None


# --- Sponsor Extraction ---

class TestExtractSponsors:
    def test_basic_list(self):
        data = {
            "sponsors": [
                {"people_id": 1, "name": "Alice", "party": "D",
                 "role_id": 2, "sponsor_type_id": 1},
            ]
        }
        result = _extract_sponsors(data)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["party"] == "D"
        assert result[0]["role"] == "Sen"
        assert result[0]["sponsor_type"] == "Primary Sponsor"

    def test_sponsor_type_mapping(self):
        data = {
            "sponsors": [
                {"people_id": 1, "name": "A", "sponsor_type_id": 0},
                {"people_id": 2, "name": "B", "sponsor_type_id": 1},
                {"people_id": 3, "name": "C", "sponsor_type_id": 2},
                {"people_id": 4, "name": "D", "sponsor_type_id": 3},
            ]
        }
        result = _extract_sponsors(data)
        types = [s["sponsor_type"] for s in result]
        assert types == ["Sponsor", "Primary Sponsor", "Co-Sponsor", "Joint Sponsor"]

    def test_role_id_mapping(self):
        data = {
            "sponsors": [
                {"people_id": 1, "name": "A", "role_id": 1},
                {"people_id": 2, "name": "B", "role_id": 2},
                {"people_id": 3, "name": "C", "role_id": 3},
            ]
        }
        result = _extract_sponsors(data)
        roles = [s["role"] for s in result]
        assert roles == ["Rep", "Sen", "Joint Conference"]

    def test_dict_wrapper_with_sponsor_key(self):
        """LegiScan sometimes wraps sponsors in {"sponsor": [...]}."""
        data = {
            "sponsors": {
                "sponsor": [
                    {"people_id": 1, "name": "Alice", "party": "R"},
                ]
            }
        }
        result = _extract_sponsors(data)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    def test_dict_wrapper_single_sponsor(self):
        """Single sponsor wrapped in {"sponsor": {...}} (not a list)."""
        data = {
            "sponsors": {
                "sponsor": {"people_id": 1, "name": "Solo", "party": "D"}
            }
        }
        result = _extract_sponsors(data)
        assert len(result) == 1
        assert result[0]["name"] == "Solo"

    def test_numeric_keys_dict(self):
        """Sponsors as dict with numeric keys {0: {...}, 1: {...}}."""
        data = {
            "sponsors": {
                0: {"people_id": 1, "name": "First", "party": "D"},
                1: {"people_id": 2, "name": "Second", "party": "R"},
            }
        }
        result = _extract_sponsors(data)
        assert len(result) == 2

    def test_name_fallback_to_parts(self):
        """When 'name' is missing, build from first/middle/last/suffix."""
        data = {
            "sponsors": [
                {"people_id": 1, "first_name": "John", "middle_name": "Q",
                 "last_name": "Public", "suffix": "Jr"},
            ]
        }
        result = _extract_sponsors(data)
        assert result[0]["name"] == "John Q Public Jr"

    def test_name_fallback_no_middle(self):
        data = {
            "sponsors": [
                {"people_id": 1, "first_name": "Jane", "last_name": "Doe"},
            ]
        }
        result = _extract_sponsors(data)
        assert result[0]["name"] == "Jane Doe"

    def test_party_id_fallback(self):
        """When 'party' is empty, fall back to 'party_id'."""
        data = {
            "sponsors": [
                {"people_id": 1, "name": "A", "party": "", "party_id": 2},
            ]
        }
        result = _extract_sponsors(data)
        assert result[0]["party"] == "2"

    def test_empty_sponsors(self):
        assert _extract_sponsors({"sponsors": []}) == []
        assert _extract_sponsors({}) == []

    def test_non_dict_items_skipped(self):
        data = {"sponsors": ["bad", 123, None]}
        assert _extract_sponsors(data) == []

    def test_sponsor_type_string_passthrough(self):
        """If sponsor_type is already a string, use it directly."""
        data = {
            "sponsors": [
                {"people_id": 1, "name": "A", "sponsor_type": "Primary Sponsor"},
            ]
        }
        result = _extract_sponsors(data)
        assert result[0]["sponsor_type"] == "Primary Sponsor"


# --- Calendar Extraction ---

class TestExtractCalendar:
    def test_basic(self):
        data = {
            "calendar": [
                {"date": "2025-07-01", "description": "Hearing",
                 "location": "Room 1", "type": "hearing"},
            ]
        }
        result = _extract_calendar(data)
        assert len(result) == 1
        assert result[0]["date"] == "2025-07-01"
        assert result[0]["event_type"] == "hearing"

    def test_desc_fallback(self):
        """Falls back to 'desc' if 'description' is missing."""
        data = {"calendar": [{"date": "2025-01-01", "desc": "Floor vote"}]}
        result = _extract_calendar(data)
        assert result[0]["description"] == "Floor vote"

    def test_empty(self):
        assert _extract_calendar({}) == []
        assert _extract_calendar({"calendar": []}) == []


# --- Subject Extraction ---

class TestExtractSubjects:
    def test_dict_subjects(self):
        data = {"subjects": [{"subject_name": "Energy"}, {"subject_name": "Tax"}]}
        assert _extract_subjects(data) == ["Energy", "Tax"]

    def test_string_subjects(self):
        data = {"subjects": ["Energy", "Tax"]}
        assert _extract_subjects(data) == ["Energy", "Tax"]

    def test_empty(self):
        assert _extract_subjects({}) == []
        assert _extract_subjects({"subjects": []}) == []


# --- Sponsor Change Detection ---

class TestDetectSponsorChanges:
    def test_no_change(self):
        old = [{"people_id": 1}, {"people_id": 2}]
        new = [{"people_id": 1}, {"people_id": 2}]
        assert _detect_sponsor_changes(old, new) is None

    def test_added(self):
        old = [{"people_id": 1}]
        new = [{"people_id": 1}, {"people_id": 2, "name": "New"}]
        result = _detect_sponsor_changes(old, new)
        assert isinstance(result, SponsorChange)
        assert len(result.added) == 1
        assert result.added[0]["people_id"] == 2
        assert result.removed == []

    def test_removed(self):
        old = [{"people_id": 1, "name": "Old"}, {"people_id": 2}]
        new = [{"people_id": 2}]
        result = _detect_sponsor_changes(old, new)
        assert len(result.removed) == 1
        assert result.removed[0]["people_id"] == 1

    def test_added_and_removed(self):
        old = [{"people_id": 1}]
        new = [{"people_id": 2}]
        result = _detect_sponsor_changes(old, new)
        assert len(result.added) == 1
        assert len(result.removed) == 1


# --- Persistence (load/save) ---

class TestPersistence:
    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_tracked_bills(path) == []

    def test_round_trip(self, tmp_path, sample_bill):
        path = tmp_path / "bills.json"
        save_tracked_bills([sample_bill], path)

        loaded = load_tracked_bills(path)
        assert len(loaded) == 1
        assert loaded[0].bill_id == sample_bill.bill_id
        assert loaded[0].state == sample_bill.state
        assert loaded[0].sponsors == sample_bill.sponsors
        assert loaded[0].solar_keywords == sample_bill.solar_keywords

    def test_load_old_format_defaults(self, tmp_path):
        """Old JSON without newer fields should get defaults."""
        old_data = [{
            "bill_id": 1, "state": "TX", "bill_number": "HB1",
            "title": "Test", "url": "u", "state_link": "s",
            "change_hash": "h", "status": 1,
            "last_history_date": "", "last_history_action": "",
        }]
        path = tmp_path / "old.json"
        with open(path, "w") as f:
            json.dump(old_data, f)

        bills = load_tracked_bills(path)
        assert len(bills) == 1
        assert bills[0].priority == "medium"
        assert bills[0].sponsors == []
        assert bills[0].solar_keywords is None
        assert bills[0].session_year_start == 0

    def test_atomic_write(self, tmp_path, sample_bill):
        """Verify no .tmp file remains after save."""
        path = tmp_path / "bills.json"
        save_tracked_bills([sample_bill], path)
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()
