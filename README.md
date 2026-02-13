# Legislation Tracker

Track state-level bills with priority levels, sponsor monitoring, hearing dates, passage likelihood scoring, and email alerts when changes occur.

Uses the [LegiScan API](https://legiscan.com/legiscan) (free tier: 30K queries/month) for bill data and GitHub Actions for automated daily checks.

## Features

- **Priority levels** — Mark bills as high, medium, or low priority; high-priority changes trigger urgent email alerts
- **Sponsor tracking** — See all sponsors/co-sponsors and get alerted when they change
- **Milestone tracking** — See which legislative milestones each bill has reached (Introduced, Committee Referral, Engrossed, Enrolled, Passed, etc.)
- **Hearing dates** — View upcoming committee hearings and calendar events
- **Related bills** — Find similar legislation across all 50 states
- **Dashboard view** — At-a-glance stats, status/state breakdowns, session warnings, and upcoming hearings
- **Filtering & sorting** — Filter by state, status, priority, or passage likelihood; sort by date, priority, state, status, or title
- **Passage likelihood scoring** — 0–100 predictive score from a logistic regression model trained on historical Open States data, with visual breakdown, key factors, and risk callouts (falls back to heuristic if no trained model)
- **Session calendar awareness** — Warns when legislative sessions are ending and bills haven't advanced
- **Visual progress timeline** — See exactly where each bill sits in the legislative process
- **Email alerts** — Get notified of status changes, new sponsors, hearing dates, and milestone events
- **Solar keyword analysis** — Scan bill text for solar energy terms (net metering, tax credits, permitting, etc.)
- **Bill history timeline** — View the full legislative history of each bill in the details panel
- **Bill refresh** — Force re-fetch bill data from LegiScan to update sponsors, calendar, and other fields

## Setup

### 1. Get a LegiScan API key

Register for free at [legiscan.com](https://legiscan.com/legiscan).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the local UI

**Linux / macOS (Bash):**

```bash
export LEGISCAN_API_KEY=your_key_here
PYTHONPATH=src python -m legislator
```

**Windows (PowerShell):**

```powershell
$env:LEGISCAN_API_KEY = "your_key_here"
$env:PYTHONPATH = "src"
python -m legislator
```

This opens a browser to `http://127.0.0.1:5000` with three tabs:

- **Dashboard** — Overview stats, status/state charts, session warnings, and upcoming hearings
- **Tracked Bills** — Your bills with filtering, sorting, priority controls, progress timelines, and expandable details (sponsors, hearings, subjects, passage likelihood breakdown, related bills)
- **Search** — Find bills by state and keywords, set priority on track

Tracked bills are saved to `data/tracked_bills.json`.

### 4. Set up automated checks (GitHub Actions)

Add these secrets to your GitHub repository (Settings > Secrets and variables > Actions):

| Secret | Description |
|--------|-------------|
| `LEGISCAN_API_KEY` | Your LegiScan API key |
| `OPENSTATES_API_KEY` | Your Open States API key (for training the passage model) |
| `SMTP_HOST` | SMTP server (default: `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (default: `587`) |
| `SMTP_USER` | SMTP login username |
| `SMTP_PASSWORD` | SMTP password (for Gmail, use an [App Password](https://myaccount.google.com/apppasswords)) |
| `EMAIL_FROM` | Sender email address |
| `EMAIL_TO` | Recipient email(s), comma-separated |

The workflow runs twice daily (8 AM and 6 PM UTC) and can be triggered manually from the Actions tab.

### 5. Commit your tracked bills

After adding bills via the UI, commit and push `data/tracked_bills.json` so GitHub Actions can check them:

```bash
git add data/tracked_bills.json
git commit -m "Add tracked bills"
git push
```

## How it works

1. You search for and track bills through the local web UI, setting priority levels as needed
2. GitHub Actions checks each tracked bill twice daily via the LegiScan API
3. Each bill has a `change_hash` — if it changes, the tool fetches updated details
4. Changes are detected across multiple dimensions: status, progress milestones, sponsors added/removed, and new hearing dates
5. Email alerts are sent with high-priority bills first; high-priority changes get an "URGENT" subject prefix
6. Updated data (including sponsors, calendar events, and session info) is committed back to the repo automatically

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/bills` | List tracked bills (supports `?state=`, `?status=`, `?priority=`, `?passage=`, `?sort=`, `?order=`) |
| `POST` | `/api/bills` | Track a new bill (`{bill_id, priority?}`) |
| `DELETE` | `/api/bills/<id>` | Remove a tracked bill |
| `PATCH` | `/api/bills/<id>/priority` | Set priority (`{priority: "high"\|"medium"\|"low"}`) |
| `POST` | `/api/bills/<id>/refresh` | Force re-fetch bill data from LegiScan (updates sponsors, calendar, etc.) |
| `POST` | `/api/bills/<id>/analyze` | Fetch bill text and scan for solar energy keywords (cached) |
| `GET` | `/api/bills/<id>/related` | Find related bills across all states |
| `GET` | `/api/search` | Search bills (`?state=`, `?q=`, `?page=`) |
| `POST` | `/api/check` | Trigger a manual check for all tracked bills |
| `GET` | `/api/dashboard` | Dashboard stats, session warnings, upcoming hearings |

## Bill status codes

| Status | Meaning |
|--------|---------|
| Introduced | Bill has been introduced |
| Engrossed | Passed one chamber, sent to the other |
| Enrolled | Passed both chambers, sent to governor |
| Passed | Signed into law |
| Vetoed | Vetoed by governor |
| Failed | Did not pass |

## Passage likelihood scoring

Each bill gets a 0–100 score estimating its likelihood of becoming law.

### Trained model (primary)

When a trained model is available, scores come from a **logistic regression** trained on historical bill data from the [Open States API](https://openstates.org/). The model learns which features actually correlate with bill passage across thousands of real bills.

**Features used by the model:**

| Feature | What it measures |
|---------|------------------|
| Sponsor count | Total number of sponsors |
| Primary sponsor count | Number of primary/lead sponsors |
| Co-sponsor count | Number of co-sponsors |
| Bipartisan support | Whether sponsors come from multiple parties |
| Minority party sponsors | Depth of cross-party support |
| Senate origin | Whether the bill originated in the Senate |
| Resolution type | Whether the bill is a resolution |
| Subject count / focus | Number of subject tags and scope |
| Amends existing law | Whether the bill modifies existing statutes |
| Committee referral | Whether the bill has been referred to committee |
| Committee passage | Whether the bill cleared committee |
| Passed one chamber | Whether the bill passed at least one chamber |
| Number of actions | Total legislative actions taken |
| Days since introduction | Span of legislative activity |
| Introduction timing | When in the session the bill was introduced |
| Companion bill | Whether a companion bill exists in the other chamber |
| Recent action density | Number of actions in the last 30 days |

**Training the model:**

1. Create a free account at [open.pluralpolicy.com](https://open.pluralpolicy.com/)
2. Download session JSON files from the [bulk data page](https://open.pluralpolicy.com/data/session-json/) — grab several completed sessions, e.g.:
   - Texas 88th Legislature (2023)
   - California 2023-2024 Regular Session
   - New York 2023 Regular Session
   - Virginia 2024 Regular Session
   - Colorado 2024 Regular Session
3. Place the `.json` or `.zip` files in `src/legislator/model/data/`
4. Run the training script:

```bash
PYTHONPATH=src python -m legislator.model.train
```

This extracts features from the downloaded bills, trains logistic regression with class weighting for the imbalanced passed/failed ratio, and saves weights to `src/legislator/model/weights.json`. The more state-sessions you include, the better the model.

### Heuristic fallback

If no trained model is available, the system falls back to a hand-tuned heuristic scoring across 6 dimensions:

| Dimension | Max Pts | What it measures |
|-----------|---------|------------------|
| Procedural Progress | 30 | Legislative stage: introduced → committee → engrossed → enrolled → passed |
| Sponsor Strength | 20 | Primary sponsors, co-sponsor count, bicameral and joint sponsors |
| Bipartisan Coalition | 12 | Cross-party support depth (not just presence) |
| Momentum & Velocity | 15 | Recency of activity, speed between milestones, upcoming hearings, action density |
| Session Timing | 10 | Bill progress vs. session timeline (stalled bills late in session are penalized) |
| Bill Structure | 5 | Chamber of origin, scope focus, bill type, amendments to existing law |

Raw scores (max 92) are normalized to a 0–100 scale.

### Dimension scoring details

#### 1. Procedural Progress (0–30 pts)

The single strongest predictor. Rather than treating all "Introduced" bills the same, this dimension uses LegiScan progress event codes to differentiate within status=1:

| Bill Stage | Points | How it's detected |
|------------|--------|-------------------|
| Introduced, no committee action | 3 | status=1, no progress events 9/10/11 |
| Referred to committee | 8 | progress event 9 (Committee Referral) |
| Committee reported favorably | 16 | progress event 10 (Committee Report Pass) |
| Committee reported Do Not Pass | 0 | progress event 11 (Committee Report DNP) |
| Engrossed (passed origin chamber) | 22 | status=2 |
| Enrolled (passed both chambers) | 27 | status=3 |
| Passed / Signed | 30 | status=4 |

#### 2. Sponsor Strength (0–20 pts)

Measures sponsor *quality*, not just headcount:

| Signal | Points |
|--------|--------|
| Has a primary sponsor | +2 |
| Multiple primary sponsors | +2 |
| 2–4 co-sponsors | +2 |
| 5–9 co-sponsors | +4 |
| 10+ co-sponsors | +6 |
| Sponsors from both chambers (Rep + Sen) | +3 |
| Has Joint Sponsor type | +3 |

Capped at 20. A bill with a single sponsor and no co-sponsors scores 2; a bill with 12 bipartisan co-sponsors from both chambers can hit 16+.

#### 3. Bipartisan Coalition (0–12 pts)

Scales by the *depth* of minority party support, not just whether it exists:

| Pattern | Points |
|---------|--------|
| Single party only | 0 |
| Two parties, 1 minority co-sponsor | 4 |
| Two parties, 2 minority co-sponsors | 6 |
| Two parties, 3+ minority co-sponsors | 8 |
| Three or more parties represented | 12 |

This distinguishes a genuine bipartisan coalition from a token single co-sponsor from across the aisle.

#### 4. Momentum & Velocity (0–15 pts)

Measures whether the bill is actively moving, and how fast:

| Signal | Points |
|--------|--------|
| **Recency**: last action ≤7 days ago | +5 |
| **Recency**: last action 8–30 days ago | +3 |
| **Recency**: last action 31–60 days ago | +1 |
| **Recency**: no action in 90+ days | −3 |
| **Velocity**: avg <14 days between milestones | +4 |
| **Velocity**: avg 14–44 days between milestones | +2 |
| **Upcoming**: has future calendar events (hearings/votes) | +3 |
| **Density**: 3+ history actions in last 30 days | +3 |
| **Density**: 1–2 history actions in last 30 days | +1 |

A bill with a hearing scheduled next week and 4 actions in the last month scores high here; a bill with no activity since last year gets penalized.

#### 5. Session Timing (0–10 pts)

Cross-references the bill's progress stage with where the legislative session currently stands. This is a matrix, not a simple check:

| Session elapsed | Bill through committee+ | Bill in committee | Bill just introduced |
|----------------|------------------------|-------------------|---------------------|
| <25% (early) | 10 | 7 | 7 |
| 25–50% | 8 | 6 | 4 |
| 50–75% | 6–8 | 2 | 1 |
| >75% (late) | 5–7 | 2 | 0 |

Session end is estimated as Dec 31 of the session's `year_end` (real sine die dates vary by state).

**Session decay multiplier:** If a bill is still at status=1, has not cleared committee, and the session is >70% elapsed, the *entire raw score* is halved (×0.5). This prevents well-sponsored but effectively dead bills from showing misleadingly high scores.

#### 6. Bill Structure (0–5 pts)

Structural characteristics that correlate with passage rates:

| Signal | Points |
|--------|--------|
| Originates in Senate (SB, S prefix) | +1 |
| Resolution type (SJR, HJR, etc.) | +1 |
| Focused scope (1–2 subject tags) | +2 |
| Amends existing law (description contains "amend", "relating to", etc.) | +1 |

### Labels, confidence, and risk

**Labels:** Very Likely (75+), Likely (55–74), Possible (35–54), Unlikely (15–34), Very Unlikely (0–14), Dead (vetoed/failed), Passed.

**Session decay:** Bills stalled in committee with session >70% elapsed have their score halved.

**Confidence rating:** Based on data completeness — high, medium, or low. Low-confidence scores show a disclaimer.

| Data present | Completeness points |
|-------------|-------------------|
| Sponsors | +1 |
| History | +1 |
| Progress details | +1 |
| Session year end | +1 |
| Calendar events | +0.5 |

High = 3.5+, Medium = 2–3, Low = 0–1.

**Risk callouts** are generated when specific warning conditions are met:
- No bipartisan support (bipartisan dimension = 0)
- Still in committee past session midpoint (>50% elapsed)
- No legislative activity in 90+ days
- Only 1 sponsor (limited coalition)
- Committee reported Do Not Pass
- No upcoming hearings or votes scheduled (while session is active)

## Project structure

```
src/legislator/
  app.py         - Flask server and API routes
  api.py         - LegiScan API client
  openstates.py  - Open States API v3 client
  checker.py     - Change detection logic and data models
  emailer.py     - Email alert formatting/sending
  scoring.py     - Passage likelihood scoring and session calendar awareness
  solar.py       - Solar energy keyword analysis for bill text
  related.py     - Related bills detection
  config.py      - Environment variable configuration
  __main__.py    - CLI entry point
  model/
    features.py  - Feature extraction for training and prediction
    train.py     - Training pipeline (fetch data, train, export weights)
    predict.py   - Prediction using trained logistic regression weights
    weights.json - Trained model weights (generated by train.py)
  static/
    index.html   - Web UI (dashboard, bill list, search)
```
