# CLAUDE.md — Legislation Tracker Project Context

## Project Overview

State-level legislation tracking system for **solar energy developers**. Built with Python/Flask backend and vanilla JS frontend. Uses LegiScan API for bill data and GitHub Actions for automated daily checks with email alerts.

## Architecture

- **Backend**: Flask REST API (`src/legislator/`)
- **Frontend**: Single-page HTML/JS app (`src/legislator/static/index.html`)
- **Data**: JSON file storage (`data/tracked_bills.json`)
- **External APIs**: LegiScan (bill data)
- **CI/CD**: GitHub Actions for twice-daily bill checks

## Key Files

| File | Purpose |
|------|---------|
| `src/legislator/app.py` | Flask routes and API endpoints |
| `src/legislator/api.py` | LegiScan API client |
| `src/legislator/checker.py` | Data models (`TrackedBill`, `BillChange`), change detection, sponsor/calendar extraction |
| `src/legislator/scoring.py` | Impact scoring (0-100) and session calendar awareness |
| `src/legislator/solar.py` | Solar energy keyword analysis for bill text |
| `src/legislator/related.py` | Cross-state related bill discovery |
| `src/legislator/emailer.py` | Email alert formatting and sending |
| `src/legislator/config.py` | Environment variable configuration |
| `src/legislator/static/index.html` | Full web UI (dashboard, tracked bills, search) |
| `data/tracked_bills.json` | Persisted bill tracking data |

## Data Flow

1. User searches bills via UI → `/api/search` → LegiScan `search` API
2. User clicks Track → `/api/bills` POST → LegiScan `getBill` → extract sponsors/calendar/subjects → save to JSON
3. Bill list loads → `/api/bills` GET → reads JSON → enriches with impact score, session status, milestones
4. GitHub Actions / manual check → `/api/check` POST → `check_all_bills()` → compares `change_hash` → detects changes → sends email alerts

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `LEGISCAN_API_KEY` | Yes | LegiScan API access |
| `SMTP_HOST` | For email | SMTP server (default: smtp.gmail.com) |
| `SMTP_PORT` | For email | SMTP port (default: 587) |
| `SMTP_USER` | For email | SMTP login |
| `SMTP_PASSWORD` | For email | SMTP password |
| `EMAIL_FROM` | For email | Sender address |
| `EMAIL_TO` | For email | Recipient address(es) |

## Known Issues & Gotchas

### LegiScan API Sponsor Data
The LegiScan API returns sponsor data with specific field conventions (from the [API User Manual](https://api.legiscan.com/dl/LegiScan_API_User_Manual.pdf)):

**Field mappings (per the LegiScan API):**
- `party_id` (int): 1=Democrat, 2=Republican, 3=Independent, 4=Green, 5=Libertarian, 6=Nonpartisan
- `party` (str): Short abbreviation like "D", "R", "I"
- `sponsor_type_id` (int): 0=Sponsor, 1=Primary Sponsor, 2=Co-Sponsor, 3=Joint Sponsor
- `role_id` (int): 1=Representative, 2=Senator, 3=Joint Conference
- `name` (str): Full name; if missing fall back to `first_name`/`middle_name`/`last_name`
- Sponsors array may be a dict with numeric keys or `{"sponsor": [...]}` wrapper

The `_extract_sponsors()` function in `checker.py` handles all these variants. If sponsors still show incorrectly, use the **Refresh** button on the bill card to force re-fetch from LegiScan.

### Change Detection
- Uses `change_hash` from LegiScan — only re-fetches when hash changes
- The `/api/bills/<id>/refresh` endpoint bypasses hash checking for force updates
- `check_all_bills()` is called by GitHub Actions and the "Check Now" button

### Atomic Writes
- `save_tracked_bills()` writes to a `.tmp` file first, then atomically renames for crash safety
- No OS-specific locking (must work on both Windows and Linux/macOS)

### API Retry Logic
- `LegiScanAPI._call()` retries up to 3 times with exponential backoff (1s, 2s, 4s) on network errors
- API-level errors (bad params, auth failures) are NOT retried

### Solar Keyword Analysis
- `solar.py` contains keyword dictionaries grouped by category (Net Metering, Incentives, Permitting, etc.)
- Bill text is fetched via `getBillText` API → base64 decoded → keyword scanned
- Results are cached in `solar_keywords` field on TrackedBill (persisted to JSON)
- Only HTML/text documents can be analyzed; PDFs are not decoded

## Running Locally

```bash
export LEGISCAN_API_KEY=your_key
PYTHONPATH=src python -m legislator
```

Opens at http://127.0.0.1:5000

## Testing Changes

No test suite currently exists. Test manually by:
1. Starting the server
2. Searching for bills (e.g., state=CA, query="solar")
3. Tracking a bill and verifying sponsors appear
4. Checking the details panel for milestones
5. Using the Refresh button to re-fetch data

## Conventions

- All API routes are in `app.py` inside `create_app()`
- Data models and extraction functions are in `checker.py`
- The frontend is a single HTML file with embedded CSS and JS
- Bill data enrichment (impact score, milestones, session status) happens at the API layer, not stored in JSON
- Solar keyword analysis results and bill history are cached in JSON
- TrackedBill has `progress_details` (with dates) in addition to `progress_events` (codes only)

## When Making Changes

- **New fields on TrackedBill**: Add to the dataclass, add to `load_tracked_bills()` with a `.get()` default, and update the frontend
- **New API endpoints**: Add inside `create_app()` in `app.py`
- **Frontend changes**: Edit `src/legislator/static/index.html`
- **Always update README.md** when adding user-facing features or new env vars
