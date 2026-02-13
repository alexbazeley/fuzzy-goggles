"""LegiScan API client."""

import json
import urllib.request
import urllib.parse
from typing import Any, Optional

BASE_URL = "https://api.legiscan.com/"


class LegiScanAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _call(self, op: str, **params: Any) -> dict:
        """Make an API call and return the parsed JSON response."""
        params["key"] = self.api_key
        params["op"] = op
        url = BASE_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if data.get("status") == "ERROR":
            msg = data.get("alert", {}).get("message", str(data))
            raise RuntimeError(f"LegiScan API error: {msg}")
        return data

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

    def get_session_people(self, session_id: int) -> list[dict]:
        """Get all people (legislators) active in a specific session."""
        data = self._call("getSessionPeople", id=session_id)
        return data.get("sessionpeople", {}).get("people", [])
