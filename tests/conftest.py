"""Shared test fixtures for the legislation tracker test suite."""

import pytest
from legislator.checker import TrackedBill


@pytest.fixture
def sample_bill():
    """A minimal TrackedBill for testing."""
    return TrackedBill(
        bill_id=12345,
        state="CA",
        bill_number="SB 100",
        title="Solar Energy Development Act",
        url="https://legiscan.com/CA/bill/SB100/2025",
        state_link="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml",
        change_hash="abc123",
        status=1,
        last_history_date="2025-06-15",
        last_history_action="Referred to Committee on Energy",
        progress_events=[1, 9],
        progress_details=[
            {"event": 1, "date": "2025-01-10"},
            {"event": 9, "date": "2025-02-05"},
        ],
        sponsors=[
            {"people_id": 1, "name": "Jane Smith", "party": "D",
             "role": "Sen", "sponsor_type": "Primary Sponsor"},
            {"people_id": 2, "name": "Bob Jones", "party": "R",
             "role": "Sen", "sponsor_type": "Co-Sponsor"},
        ],
        calendar=[
            {"date": "2025-07-01", "description": "Hearing in Energy Committee",
             "location": "Room 4202", "event_type": "hearing"},
        ],
        session_id=1999,
        session_name="2025-2026 Regular Session",
        session_year_start=2025,
        session_year_end=2026,
        description="An act relating to solar energy development and net metering",
        subjects=["Energy", "Solar"],
        committee="Energy, Utilities and Communications",
        committee_id=500,
        history=[
            {"date": "2025-01-10", "action": "Introduced", "chamber": "Senate", "chamber_id": 1},
            {"date": "2025-02-05", "action": "Referred to Com. on E., U. & C.",
             "chamber": "Senate", "chamber_id": 1},
            {"date": "2025-06-15", "action": "Set for hearing July 1",
             "chamber": "Senate", "chamber_id": 1},
        ],
        solar_keywords=["Net Metering & Interconnection: net metering",
                        "Solar Technology: solar energy"],
    )


@pytest.fixture
def sample_api_bill_data():
    """Sample LegiScan API response for a bill (the 'bill' key contents)."""
    return {
        "bill_id": 12345,
        "state": "CA",
        "bill_number": "SB 100",
        "title": "Solar Energy Development Act",
        "url": "https://legiscan.com/CA/bill/SB100/2025",
        "state_link": "https://leginfo.legislature.ca.gov",
        "change_hash": "xyz789",
        "status": 2,
        "description": "An act relating to solar energy development",
        "progress": [
            {"event": 1, "date": "2025-01-10"},
            {"event": 9, "date": "2025-02-05"},
            {"event": 2, "date": "2025-07-20"},
        ],
        "history": [
            {"date": "2025-01-10", "action": "Introduced", "chamber": "Senate", "chamber_id": 1},
            {"date": "2025-02-05", "action": "Referred to Com. on E., U. & C.",
             "chamber": "Senate", "chamber_id": 1},
            {"date": "2025-06-15", "action": "Set for hearing July 1",
             "chamber": "Senate", "chamber_id": 1},
            {"date": "2025-07-20", "action": "Passed Senate",
             "chamber": "Senate", "chamber_id": 1},
        ],
        "sponsors": [
            {"people_id": 1, "name": "Jane Smith", "party": "D",
             "role_id": 2, "sponsor_type_id": 1},
            {"people_id": 2, "name": "Bob Jones", "party": "R",
             "role_id": 2, "sponsor_type_id": 2},
            {"people_id": 3, "name": "New Cosponsor", "party": "D",
             "role_id": 1, "sponsor_type_id": 2},
        ],
        "calendar": [
            {"date": "2025-07-01", "description": "Hearing in Energy Committee",
             "location": "Room 4202", "type": "hearing"},
            {"date": "2025-08-15", "description": "Assembly Floor Vote",
             "location": "Assembly Chamber", "type": "vote"},
        ],
        "subjects": [
            {"subject_name": "Energy"},
            {"subject_name": "Solar"},
            {"subject_name": "Utilities"},
        ],
        "committee": {"name": "Appropriations", "committee_id": 501},
        "session": {
            "session_id": 1999,
            "session_name": "2025-2026 Regular Session",
            "year_start": 2025,
            "year_end": 2026,
        },
    }
