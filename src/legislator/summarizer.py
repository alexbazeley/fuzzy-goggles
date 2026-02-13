"""LLM-powered bill summarizer for solar developer context."""

import json
import os
import urllib.request
from typing import Optional

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = (
    "You are an expert policy analyst specializing in energy legislation. "
    "Your audience is solar energy developers — companies and individuals who "
    "design, install, and maintain solar photovoltaic systems, solar farms, "
    "and related infrastructure. "
    "Summarize the given bill in 2-4 concise sentences focusing on: "
    "(1) what the bill does, "
    "(2) how it directly or indirectly affects solar developers "
    "(permits, incentives, tariffs, net metering, interconnection, tax credits, "
    "zoning, RPS/clean energy standards, utility regulations, grid access, etc.), "
    "and (3) whether it is likely helpful, harmful, or neutral for solar developers. "
    "Be specific and practical. If the bill has no obvious connection to solar, "
    "note that and explain any indirect relevance."
)


def summarize_bill(
    title: str,
    description: str,
    state: str,
    status_text: str,
    subjects: list[str],
    sponsors: list[dict],
    api_key: Optional[str] = None,
) -> str:
    """Generate a solar-developer-focused summary of a bill using Claude.

    Returns the summary text, or an error message if the API call fails.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    # Build context for the LLM
    sponsor_lines = []
    for s in sponsors[:10]:  # Limit to avoid token bloat
        line = s.get("name", "")
        if s.get("party"):
            line += f" ({s['party']})"
        if s.get("sponsor_type"):
            line += f" — {s['sponsor_type']}"
        sponsor_lines.append(line)

    user_msg = (
        f"State: {state}\n"
        f"Bill Title: {title}\n"
        f"Status: {status_text}\n"
        f"Description: {description}\n"
    )
    if subjects:
        user_msg += f"Subjects: {', '.join(subjects)}\n"
    if sponsor_lines:
        user_msg += f"Sponsors: {'; '.join(sponsor_lines)}\n"

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    try:
        req = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        # Extract text from the response
        content = result.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")
        return ""
    except Exception as e:
        return f"Summary unavailable: {e}"
