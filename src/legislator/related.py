"""Related bills detection — finds similar legislation across states."""

from legislator.api import LegiScanAPI
from legislator.checker import TrackedBill


def _build_search_terms(bill: TrackedBill) -> str:
    """Build search terms from a bill's title and subjects."""
    # Use subjects if available, otherwise extract key words from title
    if bill.subjects:
        return " ".join(bill.subjects[:3])

    # Extract meaningful words from title (skip common filler)
    stop_words = {
        "a", "an", "the", "of", "to", "and", "or", "for", "in", "on",
        "at", "by", "with", "from", "as", "is", "was", "be", "been",
        "are", "were", "this", "that", "act", "bill", "relating",
        "concerning", "providing", "amending", "repealing", "relative",
    }
    words = bill.title.lower().split()
    keywords = [w.strip(".,;:()\"'") for w in words if w.lower().strip(".,;:()\"'") not in stop_words]
    return " ".join(keywords[:5])


def find_related_bills(api: LegiScanAPI, bill: TrackedBill, max_results: int = 10) -> list[dict]:
    """Search for bills related to the given bill across all states.

    Returns a list of dicts with bill summary info, excluding the source bill itself.
    """
    query = _build_search_terms(bill)
    if not query.strip():
        return []

    try:
        results = api.search(state="ALL", query=query)
    except Exception:
        return []

    summary = results.get("summary", {})
    related = []
    for key in sorted(results.keys()):
        if key == "summary":
            continue
        r = results[key]
        # Skip the bill itself
        if r.get("bill_id") == bill.bill_id:
            continue
        related.append({
            "bill_id": r.get("bill_id"),
            "state": r.get("state", ""),
            "bill_number": r.get("bill_number", ""),
            "title": r.get("title", ""),
            "last_action_date": r.get("last_action_date", ""),
            "last_action": r.get("last_action", ""),
            "url": r.get("url", ""),
            "relevance": r.get("relevance", 0),
        })
        if len(related) >= max_results:
            break

    return related
