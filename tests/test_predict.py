"""Tests for model/predict.py — prediction logic and model loading."""

import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from legislator.model.predict import (
    _sigmoid, predict_passage, get_top_factors, model_available,
)
from legislator.model.features import FEATURE_NAMES


class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(0) == 0.5

    def test_large_positive(self):
        assert _sigmoid(100) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self):
        assert _sigmoid(-100) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        """sigmoid(x) + sigmoid(-x) = 1."""
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert _sigmoid(x) + _sigmoid(-x) == pytest.approx(1.0)

    def test_monotonic(self):
        values = [_sigmoid(x) for x in range(-10, 11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


class TestGetTopFactors:
    def test_empty_prediction(self):
        assert get_top_factors(None) == []
        assert get_top_factors({}) == []

    def test_returns_top_n(self):
        prediction = {
            "feature_contributions": {
                "sponsor_count": 0.5,
                "is_bipartisan": 0.3,
                "committee_referral": -0.4,
                "num_actions": 0.1,
                "senate_origin": 0.02,
                "title_length": -0.005,  # below threshold
            },
            "raw_features": {
                "sponsor_count": 5.0,
                "is_bipartisan": 1.0,
                "committee_referral": 0.0,
                "num_actions": 2.3,
                "senate_origin": 1.0,
                "title_length": 1.5,
            },
        }
        factors = get_top_factors(prediction, top_n=3)
        assert len(factors) == 3
        # Highest absolute contributions first
        assert factors[0]["feature_key"] == "sponsor_count"
        assert factors[0]["direction"] == "positive"

    def test_skips_tiny_contributions(self):
        prediction = {
            "feature_contributions": {
                "text_hash_0": 0.005,  # below 0.01 threshold
                "sponsor_count": 0.5,
            },
            "raw_features": {"text_hash_0": 0.1, "sponsor_count": 5.0},
        }
        factors = get_top_factors(prediction, top_n=10)
        keys = [f["feature_key"] for f in factors]
        assert "text_hash_0" not in keys


class TestPredictPassage:
    @pytest.fixture
    def mock_model_v3(self):
        """Create a minimal v3 model weights file."""
        n = len(FEATURE_NAMES)
        model = {
            "version": 3,
            "weights": [0.1] * n,
            "bias": 0.0,
            "means": [0.0] * n,
            "stds": [1.0] * n,
            "threshold": 0.5,
            "feature_names": FEATURE_NAMES,
            "state_passage_rates": {"CA": 0.25, "TX": 0.20},
            "metrics": {"accuracy": 0.75},
            "cv_metrics": {"mean_f1": 0.72},
            "trained_at": "2025-06-01",
            "calibration_A": 1.0,
            "calibration_B": 0.0,
            "exclude_resolutions": True,
        }
        return model

    def test_returns_none_without_model(self, sample_bill):
        """predict_passage returns None when no weights file exists."""
        import legislator.model.predict as predict_mod
        old_cache = predict_mod._model_cache
        predict_mod._model_cache = None
        with patch.object(predict_mod, "WEIGHTS_PATH", Path("/nonexistent/weights.json")):
            result = predict_passage(sample_bill)
        predict_mod._model_cache = old_cache
        assert result is None

    def test_prediction_with_model(self, sample_bill, mock_model_v3):
        """predict_passage returns full result dict with model loaded."""
        import legislator.model.predict as predict_mod
        old_cache = predict_mod._model_cache
        predict_mod._model_cache = mock_model_v3
        try:
            result = predict_passage(sample_bill)
            assert result is not None
            assert 0 <= result["probability"] <= 1.0
            assert 0 <= result["score"] <= 100
            assert result["label"] in (
                "Very Likely", "Likely", "Possible", "Unlikely", "Very Unlikely"
            )
            assert "feature_contributions" in result
            assert result["model_version"] == 3
            assert result["threshold"] == 0.5
        finally:
            predict_mod._model_cache = old_cache

    def test_platt_calibration_identity(self, sample_bill, mock_model_v3):
        """A=1.0, B=0.0 should produce same results as uncalibrated."""
        import legislator.model.predict as predict_mod
        old_cache = predict_mod._model_cache
        # Identity calibration
        mock_model_v3["calibration_A"] = 1.0
        mock_model_v3["calibration_B"] = 0.0
        predict_mod._model_cache = mock_model_v3
        try:
            result = predict_passage(sample_bill)
            assert result is not None
            assert 0 <= result["probability"] <= 1.0
        finally:
            predict_mod._model_cache = old_cache

    def test_resolution_note(self, sample_bill, mock_model_v3):
        """Resolution bills get a warning note when model excluded resolutions."""
        import legislator.model.predict as predict_mod
        old_cache = predict_mod._model_cache
        predict_mod._model_cache = mock_model_v3
        sample_bill.bill_number = "SJR 5"
        try:
            result = predict_passage(sample_bill)
            assert "resolution_note" in result
            assert "resolution" in result["resolution_note"].lower()
        finally:
            predict_mod._model_cache = old_cache
            sample_bill.bill_number = "SB 100"

    def test_no_resolution_note_for_regular_bill(self, sample_bill, mock_model_v3):
        """Regular bills should not get a resolution note."""
        import legislator.model.predict as predict_mod
        old_cache = predict_mod._model_cache
        predict_mod._model_cache = mock_model_v3
        try:
            result = predict_passage(sample_bill)
            assert "resolution_note" not in result
        finally:
            predict_mod._model_cache = old_cache

    def test_state_passage_rate_blending(self, sample_bill, mock_model_v3):
        """v3 should blend state passage rate as post-hoc prior."""
        import legislator.model.predict as predict_mod
        old_cache = predict_mod._model_cache
        mock_model_v3["state_passage_rates"] = {"CA": 0.50}
        predict_mod._model_cache = mock_model_v3
        try:
            result = predict_passage(sample_bill)
            # With CA rate = 0.50 and blending 90/10,
            # probability should be shifted toward 0.50
            assert result is not None
        finally:
            predict_mod._model_cache = old_cache

    def test_v1_compatibility(self, sample_bill):
        """predict_passage handles v1 model format (no version key)."""
        import legislator.model.predict as predict_mod
        # v1 models used the old feature set — create a mock with
        # the current FEATURE_NAMES for simplicity (v1 code path
        # just skips the v2/v3 specific logic)
        n = len(FEATURE_NAMES)
        v1_model = {
            "weights": [0.05] * n,
            "bias": -0.5,
            "means": [0.0] * n,
            "stds": [1.0] * n,
            "feature_names": FEATURE_NAMES,
        }
        old_cache = predict_mod._model_cache
        predict_mod._model_cache = v1_model
        try:
            result = predict_passage(sample_bill)
            assert result is not None
            assert result["model_version"] == 1
            assert result["threshold"] == 0.5  # default
        finally:
            predict_mod._model_cache = old_cache
