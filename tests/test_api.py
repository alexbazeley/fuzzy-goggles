"""Tests for api.py — LegiScan API client (no real API calls)."""

import json
import pytest
from unittest.mock import patch, MagicMock
from urllib.error import URLError

from legislator.api import LegiScanAPI, MAX_RETRIES, RETRY_BACKOFF


class TestLegiScanAPI:
    def setup_method(self):
        self.api = LegiScanAPI(api_key="test_key_123")

    def test_init(self):
        assert self.api.api_key == "test_key_123"

    @patch("legislator.api.urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "status": "OK",
            "bill": {"bill_id": 1, "title": "Test"}
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.api._call("getBill", id=1)
        assert result["bill"]["bill_id"] == 1

    @patch("legislator.api.urllib.request.urlopen")
    def test_api_error_not_retried(self, mock_urlopen):
        """API-level errors (bad params) should NOT be retried."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "status": "ERROR",
            "alert": {"message": "Invalid API key"}
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Invalid API key"):
            self.api._call("getBill", id=1)
        # Should only be called once (no retries for API errors)
        assert mock_urlopen.call_count == 1

    @patch("legislator.api.time.sleep")
    @patch("legislator.api.urllib.request.urlopen")
    def test_network_error_retried(self, mock_urlopen, mock_sleep):
        """Network errors should be retried with backoff."""
        mock_urlopen.side_effect = URLError("Connection refused")

        with pytest.raises(RuntimeError, match="failed after"):
            self.api._call("getBill", id=1)

        # Should try MAX_RETRIES + 1 times total
        assert mock_urlopen.call_count == MAX_RETRIES + 1
        # Should sleep between retries
        assert mock_sleep.call_count == MAX_RETRIES

    @patch("legislator.api.time.sleep")
    @patch("legislator.api.urllib.request.urlopen")
    def test_retry_then_success(self, mock_urlopen, mock_sleep):
        """Should succeed after transient failures."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "status": "OK",
            "bill": {"bill_id": 1}
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        # Fail twice, then succeed
        mock_urlopen.side_effect = [
            URLError("timeout"),
            URLError("timeout"),
            mock_resp,
        ]

        result = self.api._call("getBill", id=1)
        assert result["bill"]["bill_id"] == 1
        assert mock_urlopen.call_count == 3

    @patch("legislator.api.urllib.request.urlopen")
    def test_get_bill(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "status": "OK",
            "bill": {"bill_id": 42, "title": "Solar Act"}
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.api.get_bill(42)
        assert result["bill_id"] == 42

    @patch("legislator.api.urllib.request.urlopen")
    def test_search(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "status": "OK",
            "searchresult": {"summary": {"count": 1}, "0": {"bill_id": 1}}
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.api.search("CA", "solar")
        assert "summary" in result


class TestRetryConfig:
    def test_retry_backoff_values(self):
        assert RETRY_BACKOFF == [1, 2, 4]

    def test_max_retries(self):
        assert MAX_RETRIES == 3
