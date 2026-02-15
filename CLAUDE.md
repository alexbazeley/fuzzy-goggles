# CLAUDE.md — Legislation Tracker Project Context

## Project Overview

State-level legislation tracking system for **solar energy developers**. Built with Python/Flask backend and vanilla JS frontend. Uses LegiScan API for bill data and GitHub Actions for automated daily checks with email alerts.

## Architecture

- **Backend**: Flask REST API (`src/legislator/`)
- **Frontend**: Single-page HTML/JS app (`src/legislator/static/index.html`)
- **Data**: JSON file storage (`data/tracked_bills.json`)
- **External APIs**: LegiScan (bill data), Open States (historical training data)
- **ML Model**: Logistic regression for passage prediction (`src/legislator/model/`)
- **CI/CD**: GitHub Actions for twice-daily bill checks

## Key Files

| File | Purpose |
|------|---------|
| `src/legislator/app.py` | Flask routes and API endpoints |
| `src/legislator/api.py` | LegiScan API client |
| `src/legislator/checker.py` | Data models (`TrackedBill`, `BillChange`), change detection, sponsor/calendar extraction |
| `src/legislator/scoring.py` | Passage likelihood scoring (trained model or heuristic fallback) |
| `src/legislator/openstates.py` | Open States API v3 client |
| `src/legislator/model/features.py` | Feature extraction (73 features) for training and prediction |
| `src/legislator/model/train.py` | Training pipeline (CV, grid search, elastic net, threshold tuning) |
| `src/legislator/model/export_pg.py` | Export training data from Open States PostgreSQL dump |
| `src/legislator/model/predict.py` | Prediction using trained logistic regression weights (v1/v2) |
| `src/legislator/model/text_features.py` | Pure-Python tokenizer and feature hashing for bill text |
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
| `OPENSTATES_API_KEY` | For model training | Open States API access |
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
- Solar keyword categories are also used as model features (`solar_category_count`, `has_solar_keywords`)

### Model v2 Architecture
The prediction model was overhauled to address critical issues (feature leakage, training data gaps, weak methodology):

**Feature changes (18 → 73 features):**
- Removed `passed_one_chamber` — tautological feature leakage (encoded outcome, not predictive signal)
- Removed `has_companion` — always 0 for LegiScan bills
- `num_actions` is now log-transformed to reduce conflation with outcome progress
- Added `sponsor_party_majority` (fraction of sponsors in majority party — strongest predictor per literature)
- Added `early_action_count` (actions in first 30 days — captures momentum without leakage)
- Added `title_length`, `has_fiscal_note`, `solar_category_count`, `has_solar_keywords`, `state_passage_rate`
- Added 50 text hash features via hashing trick in `text_features.py` (pure Python, deterministic djb2 hash)

**Training methodology:**
- Numpy-vectorized gradient descent (~100x faster than v2.0 pure-Python loops)
- `--max-bills N` CLI flag to sample a subset for faster iteration
- 5-fold cross-validation stratified by state (via `_stratified_kfold()`)
- Tighter default hyperparameter grid (24 combos): lr ∈ {0.05, 0.1}, L2 ∈ {0.001, 0.01, 0.1}, L1 ∈ {0.0, 0.01}, epochs = 1000, beta ∈ {0.4, 0.5}
- Early stopping with patience=50 on validation loss (makes large epoch counts redundant)
- Elastic net (L1 + L2) regularization — L1 drives irrelevant features to exactly zero
- Threshold tuning: searches 0.10–0.90 to maximize F1 instead of fixed 0.5 cutoff
- Session dates estimated from per-session action date ranges (not Jan 1–Dec 31)
- State passage rates computed and stored in `weights.json`
- Requires `numpy` (listed in `requirements.txt`)

**weights.json v2 format** adds: `version`, `threshold`, `state_passage_rates`, `hyperparameters`, `cv_metrics`, `text_hash_buckets`. `predict.py` is backward-compatible with v1 files (checks `model.get("version", 1)`).

**Label improvements in `label_from_openstates()`:**
- Now accepts `session_end_date` parameter — returns `None` (indeterminate) if session is still active
- Handles veto overrides (veto + override → label 1)
- Handles resolution adoption (resolution + passage → label 1)

**Sponsor party extraction (`_extract_party()` in `features.py`):**
- Tries 5 JSON paths: `person.party`, `person.current_role.party`, `person.primary_party`, top-level `party`, `organization.name`
- Normalizes party names to canonical codes (D, R, I, G, L, NP) via `_normalize_party()`

## Running Locally

```bash
export LEGISCAN_API_KEY=your_key
PYTHONPATH=src python -m legislator
```

Opens at http://127.0.0.1:5000

## Training the Model (PostgreSQL Dump)

The prediction model trains on historical Open States bill data. The best data source is their monthly PostgreSQL dump, which includes sponsor party information that JSON bulk exports lack.

```bash
# 1. Download the monthly dump from https://data.openstates.org/postgres/monthly/
# 2. Restore into a local PostgreSQL database:
createdb openstates
pg_restore --no-owner --no-acl -d openstates 2026-02-public.pgdump

# 3. Export training data as JSON (all 50 states):
pip install psycopg2-binary
PYTHONPATH=src python -m legislator.model.export_pg

# Or export specific states:
PYTHONPATH=src python -m legislator.model.export_pg --states CA,NY,TX,FL,PA

# 4. Train the model:
PYTHONPATH=src python -m legislator.model.train
```

The export script writes JSON files to `src/legislator/model/data/` (one file per state-session). The key advantage over JSON bulk exports is that the PostgreSQL dump includes `person.primary_party` on sponsorships, which enables the sponsor-related model features (bipartisan, party majority, etc.).

Alternative: you can also place Open States JSON files directly in `src/legislator/model/data/` — download from https://open.pluralpolicy.com/data/session-json/.

## Testing Changes

### Automated tests

Run the test suite (168 tests, no API keys needed):

```bash
PYTHONPATH=src pytest tests/ -v
```

Tests cover: checker.py (sponsor extraction, change detection, persistence), scoring.py (heuristic dimensions, session awareness), solar.py (keyword analysis, text decoding), model/features.py (feature extraction, party normalization, labeling), model/text_features.py (tokenization, hashing), model/predict.py (prediction, model loading), api.py (retry logic).

### Manual testing

For UI/integration changes, test manually by:
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
- **Model features**: Add to `FEATURE_NAMES` in `features.py`, implement in both `extract_from_openstates()` and `extract_from_tracked_bill()`, update `display_names` in `predict.py:get_top_factors()`, and update `dimension_groups` in `scoring.py:_compute_model_score()`
- **Always update README.md and CLAUDE.md** when adding user-facing features, new env vars, or model changes
