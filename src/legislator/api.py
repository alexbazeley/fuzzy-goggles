"""LegiScan API client."""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Optional

BASE_URL = "https://api.legiscan.com/"

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds between retries


class LegiScanAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _call(self, op: str, **params: Any) -> dict:
        """Make an API call with retry logic and return the parsed JSON response."""
        params["key"] = self.api_key
        params["op"] = op
        url = BASE_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                if data.get("status") == "ERROR":
                    msg = data.get("alert", {}).get("message", str(data))
                    raise RuntimeError(f"LegiScan API error: {msg}")
                return data
            except RuntimeError:
                raise  # Don't retry API-level errors (bad params, etc.)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF[attempt])

        raise RuntimeError(f"LegiScan API request failed after {MAX_RETRIES + 1} attempts: {last_error}")

    def get_bill(self, bill_id: int) -> dict:
        """Fetch full bill details including history, progress, sponsors, calendar."""
        data = self._call("getBill", id=bill_id)
        return data["bill"]

    def search(self, state: str, query: str, page: int = 1) -> dict:
        """Search for bills by state and query terms.

        Returns dict with 'summary' key and numbered result keys.
        state: two-letter code (e.g. 'CA') or 'ALL'
        """
        data = self._call("search", state=state, query=query, page=page)
        return data["searchresult"]

    def get_session_list(self, state: str = "ALL") -> list[dict]:
        """Get available legislative sessions for a state.

        Returns list of session dicts with session_id, session_name,
        session_title, year_start, year_end, special.
        """
        data = self._call("getSessionList", state=state)
        return data.get("sessions", [])

    def get_bill_text(self, doc_id: int) -> dict:
        """Fetch the text/document content for a specific bill text document.

        Returns dict with 'doc_id', 'bill_id', 'date', 'type', 'mime', 'doc'
        (base64 encoded document content).
        """
        data = self._call("getBillText", id=doc_id)
        return data.get("text", {})

    def get_session_people(self, session_id: int) -> list[dict]:
        """Get all people (legislators) active in a specific session."""
        data = self._call("getSessionPeople", id=session_id)
        return data.get("sessionpeople", {}).get("people", [])
