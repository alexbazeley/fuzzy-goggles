"""Tests for scoring.py — passage likelihood scoring and session awareness."""

import pytest
from datetime import date
from unittest.mock import patch

from legislator.checker import TrackedBill
from legislator.scoring import (
    compute_passage_likelihood,
    get_session_status,
    _score_procedural, _score_sponsors, _score_bipartisan,
    _score_momentum, _score_timing, _score_structure,
    _compute_confidence,
)


def _make_bill(**overrides):
    """Create a TrackedBill with sensible defaults, applying overrides."""
    defaults = dict(
        bill_id=1, state="CA", bill_number="SB 100",
        title="Solar Energy Act", url="u", state_link="s",
        change_hash="h", status=1,
        last_history_date="2025-06-01",
        last_history_action="Referred to committee",
        progress_events=[1, 9],
        progress_details=[
            {"event": 1, "date": "2025-01-10"},
            {"event": 9, "date": "2025-02-05"},
        ],
        sponsors=[
            {"people_id": 1, "name": "A", "party": "D",
             "role": "Sen", "sponsor_type": "Primary Sponsor"},
            {"people_id": 2, "name": "B", "party": "R",
             "role": "Sen", "sponsor_type": "Co-Sponsor"},
        ],
        calendar=[],
        session_year_start=2025, session_year_end=2026,
        session_name="2025-2026 Regular Session",
        description="An act relating to solar energy",
        subjects=["Energy", "Solar"],
        history=[
            {"date": "2025-01-10", "action": "Introduced"},
            {"date": "2025-02-05", "action": "Referred to committee"},
            {"date": "2025-06-01", "action": "Hearing scheduled"},
        ],
    )
    defaults.update(overrides)
    return TrackedBill(**defaults)


# --- Procedural Scoring ---

class TestScoreProcedural:
    def test_passed_bill(self):
        bill = _make_bill(status=4)
        result = _score_procedural(bill)
        assert result.score == 30  # max

    def test_vetoed_bill(self):
        bill = _make_bill(status=5)
        assert _score_procedural(bill).score == 0

    def test_failed_bill(self):
        bill = _make_bill(status=6)
        assert _score_procedural(bill).score == 0

    def test_enrolled(self):
        bill = _make_bill(status=3)
        assert _score_procedural(bill).score == 27

    def test_engrossed(self):
        bill = _make_bill(status=2)
        assert _score_procedural(bill).score == 22

    def test_committee_report_pass(self):
        bill = _make_bill(progress_events=[1, 9, 10])
        assert _score_procedural(bill).score == 16

    def test_committee_referral(self):
        bill = _make_bill(progress_events=[1, 9])
        assert _score_procedural(bill).score == 8

    def test_just_introduced(self):
        bill = _make_bill(progress_events=[1])
        assert _score_procedural(bill).score == 3

    def test_committee_dnp(self):
        bill = _make_bill(progress_events=[1, 9, 11])
        assert _score_procedural(bill).score == 0


# --- Sponsor Scoring ---

class TestScoreSponsors:
    def test_no_sponsors(self):
        bill = _make_bill(sponsors=[])
        assert _score_sponsors(bill).score == 0

    def test_single_primary(self):
        bill = _make_bill(sponsors=[
            {"people_id": 1, "name": "A", "party": "D",
             "role": "Sen", "sponsor_type": "Primary Sponsor"},
        ])
        result = _score_sponsors(bill)
        assert result.score >= 2

    def test_many_cosponsors(self):
        cosponsors = [
            {"people_id": i, "name": f"Cosponsor{i}", "party": "D",
             "role": "Rep", "sponsor_type": "Co-Sponsor"}
            for i in range(10)
        ]
        bill = _make_bill(sponsors=[
            {"people_id": 0, "name": "Primary", "party": "D",
             "role": "Sen", "sponsor_type": "Primary Sponsor"},
        ] + cosponsors)
        result = _score_sponsors(bill)
        assert result.score >= 8  # primary + 10 cosponsors

    def test_bicameral_bonus(self):
        bill = _make_bill(sponsors=[
            {"people_id": 1, "name": "A", "party": "D",
             "role": "Sen", "sponsor_type": "Primary Sponsor"},
            {"people_id": 2, "name": "B", "party": "D",
             "role": "Rep", "sponsor_type": "Co-Sponsor"},
        ])
        result = _score_sponsors(bill)
        assert result.score >= 5  # primary + cosponsor + bicameral


# --- Bipartisan Scoring ---

class TestScoreBipartisan:
    def test_no_sponsors(self):
        bill = _make_bill(sponsors=[])
        assert _score_bipartisan(bill).score == 0

    def test_single_party(self):
        bill = _make_bill(sponsors=[
            {"people_id": 1, "party": "D"},
            {"people_id": 2, "party": "D"},
        ])
        assert _score_bipartisan(bill).score == 0

    def test_bipartisan(self):
        bill = _make_bill(sponsors=[
            {"people_id": 1, "party": "D"},
            {"people_id": 2, "party": "R"},
        ])
        assert _score_bipartisan(bill).score >= 4

    def test_deep_bipartisan(self):
        bill = _make_bill(sponsors=[
            {"people_id": 1, "party": "D"},
            {"people_id": 2, "party": "D"},
            {"people_id": 3, "party": "D"},
            {"people_id": 4, "party": "R"},
            {"people_id": 5, "party": "R"},
            {"people_id": 6, "party": "R"},
        ])
        # minority_count=3 -> 8 points
        assert _score_bipartisan(bill).score >= 8

    def test_tripartisan(self):
        bill = _make_bill(sponsors=[
            {"people_id": 1, "party": "D"},
            {"people_id": 2, "party": "R"},
            {"people_id": 3, "party": "I"},
        ])
        assert _score_bipartisan(bill).score == 12  # max


# --- Momentum Scoring ---

class TestScoreMomentum:
    @patch("legislator.scoring.date")
    def test_recent_activity(self, mock_date):
        mock_date.today.return_value = date(2025, 6, 5)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        bill = _make_bill(last_history_date="2025-06-01")
        result = _score_momentum(bill)
        assert result.score >= 5  # active in last 7 days


# --- Timing Scoring ---

class TestScoreTiming:
    def test_no_session_data(self):
        bill = _make_bill(session_year_start=0, session_year_end=0)
        result = _score_timing(bill)
        assert result.score == 5  # mid-range default


# --- Structure Scoring ---

class TestScoreStructure:
    def test_senate_origin(self):
        bill = _make_bill(bill_number="SB 100")
        result = _score_structure(bill)
        assert result.score >= 1

    def test_resolution(self):
        bill = _make_bill(bill_number="SJR 5")
        result = _score_structure(bill)
        assert result.score >= 2  # senate + resolution

    def test_focused_scope(self):
        bill = _make_bill(subjects=["Energy"])
        result = _score_structure(bill)
        assert result.score >= 2  # focused scope


# --- Confidence ---

class TestComputeConfidence:
    def test_high(self, sample_bill):
        assert _compute_confidence(sample_bill) == "high"

    def test_low(self):
        bill = _make_bill(sponsors=[], history=[], progress_details=[],
                          session_year_end=0, calendar=[])
        assert _compute_confidence(bill) == "low"


# --- Integration: compute_passage_likelihood ---

class TestComputePassageLikelihood:
    @patch("legislator.scoring._compute_model_score", return_value=None)
    def test_terminal_vetoed(self, mock_model):
        bill = _make_bill(status=5)
        result = compute_passage_likelihood(bill)
        assert result["score"] == 0
        assert result["label"] == "Dead"

    @patch("legislator.scoring._compute_model_score", return_value=None)
    def test_terminal_passed(self, mock_model):
        bill = _make_bill(status=4)
        result = compute_passage_likelihood(bill)
        assert result["score"] == 98
        assert result["label"] == "Passed"

    @patch("legislator.scoring._compute_model_score", return_value=None)
    def test_heuristic_returns_all_keys(self, mock_model):
        bill = _make_bill()
        result = compute_passage_likelihood(bill)
        assert "score" in result
        assert "label" in result
        assert "confidence" in result
        assert "dimensions" in result
        assert "factors" in result
        assert "risks" in result
        assert 0 <= result["score"] <= 100

    @patch("legislator.scoring._compute_model_score", return_value=None)
    def test_score_range(self, mock_model):
        bill = _make_bill()
        result = compute_passage_likelihood(bill)
        assert 0 <= result["score"] <= 100


# --- Session Status ---

class TestGetSessionStatus:
    def test_no_session_data(self):
        bill = _make_bill(session_year_end=0)
        assert get_session_status(bill) is None

    @patch("legislator.scoring.date")
    def test_active_session(self, mock_date):
        mock_date.today.return_value = date(2025, 6, 15)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        bill = _make_bill(session_year_start=2025, session_year_end=2026)
        result = get_session_status(bill)
        assert result is not None
        assert result["days_remaining"] > 0
        assert result["session_name"] == "2025-2026 Regular Session"

    @patch("legislator.scoring.date")
    def test_session_ended(self, mock_date):
        mock_date.today.return_value = date(2027, 3, 1)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        bill = _make_bill(session_year_start=2025, session_year_end=2026)
        result = get_session_status(bill)
        assert result["warning"] is not None
        assert "ended" in result["warning"]
