"""Open States API v3 client for fetching historical bill data."""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Optional

BASE_URL = "https://v3.openstates.org"

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]


class OpenStatesAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _call(self, path: str, params: Optional[dict] = None) -> Any:
        """Make a GET request to the Open States v3 API with retry logic."""
        params = params or {}
        url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params, doseq=True)}"
        req = urllib.request.Request(url, headers={"X-API-KEY": self.api_key})

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF[attempt])

        raise RuntimeError(f"Open States API request failed after {MAX_RETRIES + 1} attempts: {last_error}")

    def get_bills(self, jurisdiction: str, session: str,
                  include: Optional[list[str]] = None,
                  page: int = 1, per_page: int = 20) -> dict:
        """Fetch bills for a jurisdiction and session.

        jurisdiction: state abbreviation lowercased, e.g. 'texas' or 'ca'
                      (Open States uses full name or abbreviation)
        session: session identifier, e.g. '88' for Texas 88th
        include: list of related data to include, e.g.
                 ['sponsorships', 'actions', 'votes']
        """
        params = {
            "jurisdiction": jurisdiction,
            "session": session,
            "page": page,
            "per_page": per_page,
        }
        if include:
            params["include"] = include
        return self._call("/bills", params)

    def get_bill(self, bill_id: str, include: Optional[list[str]] = None) -> dict:
        """Fetch a single bill by its Open States ID (ocd-bill/...)."""
        params = {}
        if include:
            params["include"] = include
        return self._call(f"/bills/{urllib.parse.quote(bill_id, safe='')}", params)

    def get_sessions(self, jurisdiction: str) -> list[dict]:
        """Get available sessions for a jurisdiction."""
        data = self._call(f"/jurisdictions/{jurisdiction}")
        return data.get("legislative_sessions", [])

    def fetch_all_bills(self, jurisdiction: str, session: str,
                        include: Optional[list[str]] = None,
                        max_pages: Optional[int] = None) -> list[dict]:
        """Fetch all bills for a jurisdiction+session, paginating automatically.

        Returns a flat list of bill dicts.
        """
        all_bills = []
        page = 1
        per_page = 20  # Open States v3 max

        while True:
            data = self.get_bills(jurisdiction, session, include=include,
                                  page=page, per_page=per_page)
            results = data.get("results", [])
            all_bills.extend(results)

            pagination = data.get("pagination", {})
            max_page = pagination.get("max_page", page)

            if page >= max_page:
                break
            if max_pages and page >= max_pages:
                break

            page += 1
            time.sleep(0.5)  # be polite to the API

        return all_bills
