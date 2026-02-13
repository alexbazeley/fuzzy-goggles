"""Solar energy keyword analysis for bill text."""

import base64
import re

# Keywords relevant to solar energy developers, grouped by category
SOLAR_KEYWORDS = {
    "Solar Technology": [
        "solar", "photovoltaic", "pv system", "solar panel", "solar array",
        "solar farm", "solar garden", "community solar", "rooftop solar",
        "solar energy", "solar power", "solar generation",
    ],
    "Net Metering & Interconnection": [
        "net metering", "net energy metering", "nem", "interconnection",
        "grid connection", "grid-tied", "power purchase agreement", "ppa",
        "feed-in tariff", "buyback rate", "avoided cost",
    ],
    "Incentives & Finance": [
        "tax credit", "tax incentive", "tax exemption", "property tax",
        "sales tax exemption", "investment tax credit", "itc",
        "rebate", "subsidy", "grant program", "srec",
        "renewable energy credit", "rec", "green bond",
    ],
    "Permitting & Siting": [
        "permitting", "building permit", "zoning", "setback",
        "land use", "siting", "decommissioning", "agricultural land",
        "dual use", "agrivoltaic",
    ],
    "Storage & Grid": [
        "battery storage", "energy storage", "grid modernization",
        "distributed generation", "distributed energy", "microgrid",
        "virtual power plant", "demand response",
    ],
    "Renewable Standards": [
        "renewable portfolio standard", "rps", "clean energy standard",
        "renewable energy standard", "carbon free", "clean electricity",
        "renewable mandate", "clean energy target",
    ],
    "Utility & Rate Design": [
        "rate design", "time of use", "fixed charge", "demand charge",
        "standby charge", "utility commission", "public utility",
        "rate case", "cost of service",
    ],
}

# Flatten for quick scanning
_ALL_KEYWORDS = []
for category, keywords in SOLAR_KEYWORDS.items():
    for kw in keywords:
        _ALL_KEYWORDS.append((kw, category))
# Sort longest first so longer phrases match before shorter substrings
_ALL_KEYWORDS.sort(key=lambda x: -len(x[0]))


def analyze_bill_text(text_content: str) -> list[str]:
    """Scan bill text for solar-relevant keywords.

    Returns list of matched keyword strings (deduplicated, sorted by category).
    """
    if not text_content:
        return []

    text_lower = text_content.lower()
    found = set()

    for keyword, category in _ALL_KEYWORDS:
        # Use word boundary matching for short keywords to avoid false positives
        if len(keyword) <= 3:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                found.add(f"{category}: {keyword}")
        else:
            if keyword in text_lower:
                found.add(f"{category}: {keyword}")

    return sorted(found)


def decode_bill_text(doc_data: dict) -> str:
    """Decode base64 bill text from LegiScan API response.

    The API returns documents as base64-encoded content. HTML and plain text
    documents can be decoded directly. PDFs cannot be easily text-analyzed.
    """
    doc_b64 = doc_data.get("doc", "")
    mime = doc_data.get("mime", "")

    if not doc_b64:
        return ""

    try:
        raw = base64.b64decode(doc_b64)
    except Exception:
        return ""

    # HTML and plain text can be decoded as UTF-8
    if "html" in mime or "text" in mime:
        try:
            text = raw.decode("utf-8", errors="replace")
            # Strip HTML tags for plain text analysis
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text
        except Exception:
            return ""

    return ""
