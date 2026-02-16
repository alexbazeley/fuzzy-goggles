# Legislation Tracker

Track state-level bills relevant to solar energy development. Get passage likelihood scores, sponsor monitoring, hearing dates, and email alerts when things change.

Built with Python/Flask and vanilla JS. Uses the [LegiScan API](https://legiscan.com/legiscan) (free tier: 30K queries/month) for bill data and GitHub Actions for automated twice-daily checks.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your LegiScan API key
export LEGISCAN_API_KEY=your_key_here

# 3. Run the app
PYTHONPATH=src python -m legislator
```

Opens at http://127.0.0.1:5000 with three tabs:

- **Dashboard** — Stats overview, status/state breakdowns, session warnings, upcoming hearings
- **Tracked Bills** — Your bills with filtering, sorting, priority controls, progress timelines, and expandable details (sponsors, hearings, subjects, passage likelihood breakdown, related bills)
- **Search** — Find bills by state and keywords, set priority on track

On Windows (PowerShell):

```powershell
$env:LEGISCAN_API_KEY = "your_key_here"
$env:PYTHONPATH = "src"
python -m legislator
```

## Features

- **Priority levels** — Mark bills as high, medium, or low; high-priority changes trigger urgent email alerts
- **Passage likelihood scoring** — 0-100 score from a logistic regression model trained on historical data, with visual breakdown and risk callouts (heuristic fallback when no model is trained)
- **Sponsor tracking** — See all sponsors/co-sponsors with party and role; get alerted when they change
- **Milestone tracking** — Visual progress timeline showing Introduced, Committee Referral, Engrossed, Enrolled, Passed, etc.
- **Hearing dates** — View upcoming committee hearings and calendar events
- **Solar keyword analysis** — Scan bill text for solar energy terms (net metering, tax credits, permitting, storage, RPS, etc.)
- **Related bills** — Find similar legislation across all 50 states
- **Session awareness** — Warns when sessions are ending and bills haven't advanced
- **Bill history** — Full legislative history timeline in the details panel
- **Email alerts** — Status changes, new sponsors, hearing dates, and milestone events
- **Dashboard** — At-a-glance stats, status/state charts, session warnings, upcoming hearings
- **Filtering & sorting** — Filter by state, status, priority, or passage likelihood; sort by date, priority, state, status, or title

## How It Works

1. You search for and track bills through the web UI, setting priority levels
2. GitHub Actions checks each tracked bill twice daily via the LegiScan API
3. Each bill has a `change_hash` — if it changes, the tool fetches updated details
4. Changes are detected across: status, progress milestones, sponsors added/removed, and new hearing dates
5. Email alerts are sent with high-priority bills first (urgent subject prefix for high-priority changes)
6. Updated data is committed back to the repo automatically

Tracked bills are saved to `data/tracked_bills.json`.

## Running Tests

The project has a test suite covering the core modules (168 tests):

```bash
PYTHONPATH=src pytest tests/ -v
```

Tests cover:
- Sponsor extraction (all LegiScan API format variants)
- Change detection and data persistence
- Passage likelihood scoring (heuristic dimensions)
- Solar keyword analysis and text decoding
- Feature extraction (73 features, party normalization, labeling)
- Text tokenization and feature hashing
- Prediction logic (sigmoid, v1/v2 model compatibility)
- API client (retry logic, error handling)

No API keys or external services are needed to run tests — everything is mocked.

## Automated Checks (GitHub Actions)

Add these secrets to your GitHub repository (Settings > Secrets and variables > Actions):

| Secret | Description |
|--------|-------------|
| `LEGISCAN_API_KEY` | Your LegiScan API key |
| `SMTP_HOST` | SMTP server (default: `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (default: `587`) |
| `SMTP_USER` | SMTP login username |
| `SMTP_PASSWORD` | SMTP password (for Gmail, use an [App Password](https://myaccount.google.com/apppasswords)) |
| `EMAIL_FROM` | Sender email address |
| `EMAIL_TO` | Recipient email(s), comma-separated |

The workflow runs twice daily (8 AM and 6 PM UTC) and can be triggered manually from the Actions tab.

After adding bills via the UI, commit and push `data/tracked_bills.json` so GitHub Actions can check them:

```bash
git add data/tracked_bills.json
git commit -m "Add tracked bills"
git push
```

## Passage Likelihood Scoring

Each bill gets a 0-100 score estimating its likelihood of becoming law.

### Trained Model (primary)

When a trained model is available (`src/legislator/model/weights.json`), scores come from a **logistic regression with elastic net regularization** trained on historical bill data from [Open States](https://openstates.org/).

The v2 model uses:
- 5-fold cross-validation stratified by state
- Hyperparameter grid search (24 combinations)
- Early stopping (patience=50 on validation loss)
- Elastic net (L1 + L2) regularization — L1 drives irrelevant features to zero
- Threshold tuning to maximize F1 score
- Numpy-vectorized gradient descent

**73 features** (23 structured + 50 text hash):

| Category | Features | Examples |
|----------|----------|---------|
| Sponsor (6) | Count, party composition, majority alignment | Total sponsors, bipartisan support, majority party fraction |
| Bill structure (5) | Chamber, type, scope, amendments | Senate origin, resolution type, focused scope |
| Procedural (2) | Committee progress | Committee referral, committee passage |
| Action/momentum (5) | Activity patterns and timing | Log action count, early momentum, recent density, session timing |
| Text-derived (4) | Title and description analysis | Title length, fiscal note, solar keyword categories |
| State-level (1) | Historical context | State passage rate |
| Text hash (50) | Bill text signal | Tokenized title+description mapped via feature hashing |

See [MODEL_ASSESSMENT.md](MODEL_ASSESSMENT.md) for a detailed technical assessment.

### Training the Model

There are two ways to get training data:

#### Option A: PostgreSQL dump (recommended — includes sponsor party data)

The Open States monthly PostgreSQL dump includes `person.primary_party` on sponsorships, which enables the sponsor-related model features (bipartisan, party majority, etc.).

```bash
# 1. Download the monthly dump from https://data.openstates.org/postgres/monthly/

# 2. Create and restore the database
createdb openstates
pg_restore --no-owner --no-acl -d openstates 2026-02-public.pgdump
```

**Note:** The dump includes PostGIS extensions for geographic boundary data that we don't use. You'll see errors like `extension "postgis" is not available` and `relation "public.boundaries_boundary" does not exist` — **these are safe to ignore**. The bill, sponsorship, and action tables will load correctly. Expect the restore to take 1-2 hours for a ~9 GB dump.

```bash
# 3. Export training data as JSON
PYTHONPATH=src python -m legislator.model.export_pg

# Or export specific states:
PYTHONPATH=src python -m legislator.model.export_pg --states CA,NY,TX,FL,PA

# 4. Train the model
PYTHONPATH=src python -m legislator.model.train
```

#### Option B: Session JSON files (faster setup, no PostgreSQL needed)

Download session JSON files directly — no database required, but sponsor party data may be missing.

1. Create a free account at [open.pluralpolicy.com](https://open.pluralpolicy.com/)
2. Download session JSON files from the [bulk data page](https://open.pluralpolicy.com/data/session-json/) — grab several completed sessions (e.g., Texas 88th, California 2023-2024, New York 2023)
3. Place the `.json` or `.zip` files in `src/legislator/model/data/`
4. Train:

```bash
PYTHONPATH=src python -m legislator.model.train

# Or sample a subset for faster iteration:
PYTHONPATH=src python -m legislator.model.train --max-bills 50000
```

#### What training does

The training pipeline will:
1. Load and extract 73 features per bill (sampling with `--max-bills` if specified)
2. Estimate session dates from actual action date ranges
3. Compute per-state passage rates
4. Run 5-fold cross-validation stratified by state across 24 hyperparameter combinations
5. Train the final model with the best hyperparameters and early stopping
6. Find the optimal classification threshold (maximizing F1)
7. Save everything to `src/legislator/model/weights.json`

The output reports CV metrics (mean +/- std F1, precision, recall), best hyperparameters, and feature weights sorted by importance.

### Heuristic Fallback

If no trained model is available, the system falls back to a hand-tuned heuristic scoring across 6 dimensions (max 92 raw points, normalized to 0-100):

| Dimension | Max Pts | What it measures |
|-----------|---------|------------------|
| Procedural Progress | 30 | Legislative stage: introduced > committee > engrossed > enrolled > passed |
| Sponsor Strength | 20 | Primary sponsors, co-sponsor count, bicameral and joint sponsors |
| Bipartisan Coalition | 12 | Cross-party support depth (not just presence) |
| Momentum & Velocity | 15 | Recency of activity, speed between milestones, upcoming hearings, action density |
| Session Timing | 10 | Bill progress vs. session timeline (stalled bills penalized) |
| Bill Structure | 5 | Chamber of origin, scope focus, bill type, amendments to existing law |

**Session decay:** Bills stalled in committee with session >70% elapsed have their entire score halved.

**Labels:** Very Likely (75+), Likely (55-74), Possible (35-54), Unlikely (15-34), Very Unlikely (0-14), Dead (vetoed/failed), Passed.

**Confidence:** Based on data completeness (sponsors, history, progress details, session info, calendar). High/medium/low — low-confidence scores show a disclaimer.

**Risk callouts:** No bipartisan support, stalled past midpoint, inactive 90+ days, single sponsor, committee DNP, no upcoming events.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/bills` | List tracked bills (supports `?state=`, `?status=`, `?priority=`, `?passage=`, `?sort=`, `?order=`) |
| `POST` | `/api/bills` | Track a new bill (`{bill_id, priority?}`) |
| `DELETE` | `/api/bills/<id>` | Remove a tracked bill |
| `PATCH` | `/api/bills/<id>/priority` | Set priority (`{priority: "high"|"medium"|"low"}`) |
| `POST` | `/api/bills/<id>/refresh` | Force re-fetch bill data from LegiScan |
| `POST` | `/api/bills/<id>/analyze` | Fetch and scan bill text for solar energy keywords (cached) |
| `GET` | `/api/bills/<id>/related` | Find related bills across all states |
| `GET` | `/api/search` | Search bills (`?state=`, `?q=`, `?page=`) |
| `POST` | `/api/check` | Trigger a manual check for all tracked bills |
| `GET` | `/api/dashboard` | Dashboard stats, session warnings, upcoming hearings |

## Bill Status Codes

| Status | Meaning |
|--------|---------|
| Introduced | Bill has been introduced |
| Engrossed | Passed one chamber, sent to the other |
| Enrolled | Passed both chambers, sent to governor |
| Passed | Signed into law |
| Vetoed | Vetoed by governor |
| Failed | Did not pass |

## Project Structure

```
src/legislator/
  app.py            - Flask server and API routes
  api.py            - LegiScan API client (retry logic, backoff)
  checker.py        - Data models (TrackedBill, BillChange), change detection,
                      sponsor/calendar/subject extraction
  scoring.py        - Passage likelihood scoring (trained model + heuristic fallback),
                      session calendar awareness
  solar.py          - Solar energy keyword analysis for bill text
  related.py        - Cross-state related bill discovery
  emailer.py        - Email alert formatting and sending
  openstates.py     - Open States API v3 client
  config.py         - Environment variable configuration
  __main__.py       - CLI entry point
  model/
    features.py     - Feature extraction (73 features) for training and prediction
    train.py        - Training pipeline (CV, grid search, elastic net, threshold tuning)
    predict.py      - Prediction using trained logistic regression weights (v1/v2)
    text_features.py - Pure-Python tokenizer and feature hashing
    export_pg.py    - Export training data from Open States PostgreSQL dump
    weights.json    - Trained model weights (generated by train.py, not in git)
  static/
    index.html      - Web UI (dashboard, bill list, search)
tests/
  conftest.py       - Shared test fixtures (sample bills, API responses)
  test_checker.py   - Data models, sponsor extraction, change detection, persistence
  test_scoring.py   - Heuristic scoring dimensions, session awareness
  test_solar.py     - Keyword analysis, text decoding
  test_features.py  - Feature extraction, party normalization, labeling
  test_text_features.py - Tokenization, feature hashing
  test_predict.py   - Prediction logic, model loading
  test_api.py       - API client retry logic
data/
  tracked_bills.json - Persisted bill tracking data
```

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
