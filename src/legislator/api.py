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
        """Fetch full bill details including history and progress."""
        data = self._call("getBill", id=bill_id)
        return data["bill"]

    def search(self, state: str, query: str, page: int = 1) -> dict:
        """Search for bills by state and query terms.

        Returns dict with 'summary' key and numbered result keys.
        state: two-letter code (e.g. 'CA') or 'ALL'
        """
        data = self._call("search", state=state, query=query, page=page)
        return data["searchresult"]
