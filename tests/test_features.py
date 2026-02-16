"""Tests for model/features.py — feature extraction for training and prediction."""

import math
import pytest
from datetime import date

from legislator.model.features import (
    FEATURE_NAMES, extract_from_openstates, extract_from_tracked_bill,
    label_from_openstates,
    _extract_party, _normalize_party, _count_solar_categories, _parse_date,
)
from legislator.model.text_features import NUM_BUCKETS


class TestFeatureNames:
    def test_total_count(self):
        """522 features: 22 structured + 500 text hash."""
        assert len(FEATURE_NAMES) == 522

    def test_no_duplicates(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_removed_features_not_present(self):
        """v2 removed passed_one_chamber and has_companion.
        v3 removed days_since_introduction, action_density_30d, state_passage_rate."""
        assert "passed_one_chamber" not in FEATURE_NAMES
        assert "has_companion" not in FEATURE_NAMES
        assert "days_since_introduction" not in FEATURE_NAMES
        assert "action_density_30d" not in FEATURE_NAMES
        assert "state_passage_rate" not in FEATURE_NAMES

    def test_v3_features_present(self):
        """v3 added days_to_first_committee and committee_speed."""
        assert "days_to_first_committee" in FEATURE_NAMES
        assert "committee_speed" in FEATURE_NAMES

    def test_text_hash_features_present(self):
        text_features = [f for f in FEATURE_NAMES if f.startswith("text_hash_")]
        assert len(text_features) == NUM_BUCKETS


class TestExtractParty:
    def test_person_party(self):
        assert _extract_party({"person": {"party": "Democratic"}}) == "Democratic"

    def test_person_current_role_party(self):
        s = {"person": {"current_role": {"party": "Republican"}}}
        assert _extract_party(s) == "Republican"

    def test_person_primary_party(self):
        s = {"person": {"primary_party": "Independent"}}
        assert _extract_party(s) == "Independent"

    def test_top_level_party(self):
        assert _extract_party({"party": "Green"}) == "Green"

    def test_organization_name(self):
        s = {"organization": {"name": "Democratic"}}
        assert _extract_party(s) == "Democratic"

    def test_empty(self):
        assert _extract_party({}) == ""
        assert _extract_party({"person": {}}) == ""


class TestNormalizeParty:
    def test_democratic_variants(self):
        assert _normalize_party("Democratic") == "D"
        assert _normalize_party("democrat") == "D"
        assert _normalize_party("Dem") == "D"
        assert _normalize_party("D") == "D"

    def test_republican_variants(self):
        assert _normalize_party("Republican") == "R"
        assert _normalize_party("GOP") == "R"
        assert _normalize_party("R") == "R"

    def test_independent(self):
        assert _normalize_party("Independent") == "I"
        assert _normalize_party("I") == "I"

    def test_empty(self):
        assert _normalize_party("") == ""

    def test_unknown_passes_through(self):
        assert _normalize_party("Unknown Party") == "UNKNOWN PARTY"


class TestParseDate:
    def test_standard_format(self):
        assert _parse_date("2025-06-15") == date(2025, 6, 15)

    def test_datetime_format(self):
        assert _parse_date("2025-06-15T10:30:00") == date(2025, 6, 15)

    def test_invalid(self):
        assert _parse_date("not-a-date") is None
        assert _parse_date("") is None
        assert _parse_date(None) is None


class TestCountSolarCategories:
    def test_no_match(self):
        count, has_any = _count_solar_categories("highway speed limits")
        assert count == 0.0
        assert has_any == 0.0

    def test_single_category(self):
        count, has_any = _count_solar_categories("solar energy development")
        assert count >= 1.0
        assert has_any == 1.0

    def test_multiple_categories(self):
        text = "solar panel net metering tax credit battery storage"
        count, has_any = _count_solar_categories(text)
        assert count >= 3.0

    def test_empty(self):
        count, has_any = _count_solar_categories("")
        assert count == 0.0
        assert has_any == 0.0


class TestExtractFromOpenstates:
    def _make_bill(self, **overrides):
        bill = {
            "identifier": "SB 100",
            "title": "Solar Energy Act",
            "abstract": "An act promoting solar energy",
            "classification": ["bill"],
            "from_organization": {"name": "Senate"},
            "subject": ["Energy", "Solar"],
            "sponsorships": [
                {"primary": True, "person": {"party": "Democratic"}},
                {"primary": False, "person": {"party": "Republican"}},
            ],
            "actions": [
                {"date": "2025-01-10", "classification": ["introduction"]},
                {"date": "2025-02-05", "classification": ["referral-committee"]},
                {"date": "2025-03-20", "classification": ["committee-passage"]},
            ],
            "first_action_date": "2025-01-10",
            "latest_action_date": "2025-03-20",
        }
        bill.update(overrides)
        return bill

    def test_sponsor_features(self):
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        assert features["sponsor_count"] == 2.0
        assert features["primary_sponsor_count"] == 1.0
        assert features["cosponsor_count"] == 1.0
        assert features["is_bipartisan"] == 1.0

    def test_senate_origin(self):
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        assert features["senate_origin"] == 1.0

    def test_house_origin(self):
        bill = self._make_bill(
            identifier="HB 200",
            from_organization={"name": "House"}
        )
        features = extract_from_openstates(bill)
        assert features["senate_origin"] == 0.0

    def test_resolution_detection(self):
        bill = self._make_bill(classification=["resolution"])
        features = extract_from_openstates(bill)
        assert features["is_resolution"] == 1.0

    def test_subject_features(self):
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        assert features["num_subjects"] == 2.0
        assert features["focused_scope"] == 1.0

    def test_amends_existing_law(self):
        bill = self._make_bill(title="An act to amend the Public Utilities Code")
        features = extract_from_openstates(bill)
        assert features["amends_existing_law"] == 1.0

    def test_committee_features(self):
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        assert features["committee_referral"] == 1.0
        assert features["committee_passage"] == 1.0

    def test_num_actions_capped_at_60_days(self):
        """num_actions only counts actions within 60 days of introduction."""
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        # first_action_date = 2025-01-10, cutoff = 2025-03-11
        # Actions: Jan 10, Feb 5 (within 60d), Mar 20 (outside 60d)
        assert features["num_actions"] == pytest.approx(math.log(3), rel=1e-4)

    def test_days_to_first_committee(self):
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        # 2025-01-10 (intro) to 2025-02-05 (referral-committee) = 26 days
        assert features["days_to_first_committee"] == 26.0

    def test_committee_speed(self):
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        # 2025-02-05 (referral-committee) to 2025-03-20 (committee-passage) = 43 days
        assert features["committee_speed"] == 43.0

    def test_title_length(self):
        bill = self._make_bill(title="Solar Energy Act")
        features = extract_from_openstates(bill)
        assert features["title_length"] == pytest.approx(math.log(4), rel=1e-4)

    def test_fiscal_note_detection(self):
        bill = self._make_bill(title="Appropriations for solar", abstract="Budget allocation")
        features = extract_from_openstates(bill)
        assert features["has_fiscal_note"] == 1.0

    def test_text_hash_features_present(self):
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        for i in range(NUM_BUCKETS):
            assert f"text_hash_{i}" in features

    def test_all_features_present(self):
        bill = self._make_bill()
        features = extract_from_openstates(bill)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_session_pct_at_intro(self):
        bill = self._make_bill()
        features = extract_from_openstates(
            bill,
            session_start=date(2025, 1, 1),
            session_end=date(2025, 12, 31),
        )
        # Intro on Jan 10 = ~2.5% into session
        assert 0 < features["session_pct_at_intro"] < 10


class TestExtractFromTrackedBill:
    def test_basic(self, sample_bill):
        features = extract_from_tracked_bill(sample_bill)
        assert features["sponsor_count"] == 2.0
        assert features["is_bipartisan"] == 1.0
        assert features["senate_origin"] == 1.0
        assert features["num_subjects"] == 2.0

    def test_all_features_present(self, sample_bill):
        features = extract_from_tracked_bill(sample_bill)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_committee_referral_from_progress(self, sample_bill):
        features = extract_from_tracked_bill(sample_bill)
        assert features["committee_referral"] == 1.0  # event 9 in progress_events
        assert features["committee_passage"] == 0.0   # event 10 not present

    def test_solar_keywords_from_cache(self, sample_bill):
        features = extract_from_tracked_bill(sample_bill)
        assert features["solar_category_count"] == 2.0  # two categories in cache
        assert features["has_solar_keywords"] == 1.0

    def test_resolution_detection(self, sample_bill):
        sample_bill.bill_number = "SJR 5"
        features = extract_from_tracked_bill(sample_bill)
        assert features["is_resolution"] == 1.0


class TestLabelFromOpenstates:
    def test_signed_into_law(self):
        bill = {"actions": [{"classification": ["executive-signature"]}]}
        assert label_from_openstates(bill) == 1

    def test_became_law(self):
        bill = {"actions": [{"classification": ["became-law"]}]}
        assert label_from_openstates(bill) == 1

    def test_veto_override(self):
        bill = {"actions": [
            {"classification": ["executive-veto"]},
            {"classification": ["veto-override"]},
        ]}
        assert label_from_openstates(bill) == 1

    def test_vetoed_no_override(self):
        bill = {"actions": [{"classification": ["executive-veto"]}]}
        assert label_from_openstates(bill) == 0

    def test_resolution_passed(self):
        bill = {
            "classification": ["resolution"],
            "actions": [{"classification": ["passage"]}],
        }
        assert label_from_openstates(bill) == 1

    def test_session_active_returns_none(self):
        bill = {"actions": [{"classification": ["introduction"]}]}
        # Session end in the future
        future = date(2099, 12, 31)
        assert label_from_openstates(bill, session_end_date=future) is None

    def test_session_ended_no_passage(self):
        bill = {"actions": [{"classification": ["introduction"]}]}
        past = date(2020, 12, 31)
        assert label_from_openstates(bill, session_end_date=past) == 0

    def test_no_session_date_defaults_to_zero(self):
        bill = {"actions": [{"classification": ["introduction"]}]}
        assert label_from_openstates(bill) == 0
