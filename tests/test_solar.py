"""Tests for solar.py — keyword analysis and text decoding."""

import base64
import pytest

from legislator.solar import (
    analyze_bill_text, decode_bill_text, SOLAR_KEYWORDS,
)


class TestAnalyzeBillText:
    def test_empty_text(self):
        assert analyze_bill_text("") == []
        assert analyze_bill_text(None) == []

    def test_solar_keyword(self):
        result = analyze_bill_text("This bill promotes solar energy development")
        assert any("solar energy" in r for r in result)

    def test_net_metering(self):
        result = analyze_bill_text("Establishes net metering program for utilities")
        assert any("net metering" in r for r in result)

    def test_multiple_categories(self):
        text = "Solar panel tax credit and net metering interconnection standards"
        result = analyze_bill_text(text)
        categories = {r.split(":")[0].strip() for r in result}
        assert "Solar Technology" in categories
        assert "Net Metering & Interconnection" in categories
        assert "Incentives & Finance" in categories

    def test_case_insensitive(self):
        result = analyze_bill_text("SOLAR ENERGY development and NET METERING")
        assert len(result) > 0

    def test_short_keyword_word_boundary(self):
        """Short keywords (<=3 chars) require word boundaries."""
        # "ppa" should match as a word
        result = analyze_bill_text("The state ppa program is expanding")
        assert any("ppa" in r for r in result)

    def test_short_keyword_no_false_positive(self):
        """'rec' inside 'record' should NOT match."""
        result = analyze_bill_text("This is a record of proceedings")
        rec_matches = [r for r in result if r.endswith(": rec")]
        assert len(rec_matches) == 0

    def test_no_match(self):
        result = analyze_bill_text("An act relating to highway speed limits")
        assert result == []

    def test_results_sorted(self):
        text = "Solar panel tax credit with battery storage and net metering"
        result = analyze_bill_text(text)
        assert result == sorted(result)

    def test_deduplicated(self):
        text = "solar solar solar energy energy energy"
        result = analyze_bill_text(text)
        # Each keyword should appear at most once
        assert len(result) == len(set(result))


class TestDecodeBillText:
    def test_html_decode(self):
        html = "<html><body><p>Solar energy bill</p></body></html>"
        encoded = base64.b64encode(html.encode()).decode()
        result = decode_bill_text({"doc": encoded, "mime": "text/html"})
        assert "Solar energy bill" in result
        # HTML tags should be stripped
        assert "<p>" not in result

    def test_plain_text_decode(self):
        text = "This is a plain text bill about solar panels"
        encoded = base64.b64encode(text.encode()).decode()
        result = decode_bill_text({"doc": encoded, "mime": "text/plain"})
        assert "solar panels" in result

    def test_pdf_returns_empty(self):
        """PDFs are not decoded."""
        encoded = base64.b64encode(b"fake pdf content").decode()
        result = decode_bill_text({"doc": encoded, "mime": "application/pdf"})
        assert result == ""

    def test_empty_doc(self):
        assert decode_bill_text({"doc": "", "mime": "text/html"}) == ""
        assert decode_bill_text({}) == ""

    def test_invalid_base64(self):
        result = decode_bill_text({"doc": "not-valid-base64!!!", "mime": "text/html"})
        assert result == ""


class TestSolarKeywordsStructure:
    def test_all_categories_have_keywords(self):
        for category, keywords in SOLAR_KEYWORDS.items():
            assert len(keywords) > 0, f"Category '{category}' has no keywords"

    def test_keywords_are_lowercase(self):
        for category, keywords in SOLAR_KEYWORDS.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in '{category}' should be lowercase"
