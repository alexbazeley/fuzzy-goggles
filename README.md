# Legislation Tracker

Track state-level bills and get email alerts when their status changes.

Uses the [LegiScan API](https://legiscan.com/legiscan) (free tier: 30K queries/month) for bill data and GitHub Actions for automated daily checks.

## Setup

### 1. Get a LegiScan API key

Register for free at [legiscan.com](https://legiscan.com/legiscan).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the local UI

```bash
export LEGISCAN_API_KEY=your_key_here
PYTHONPATH=src python -m legislator
```

This opens a browser to `http://127.0.0.1:5000` where you can:
- **Search** for bills by state and keywords
- **Track** bills you're interested in
- **Remove** bills you no longer want to follow
- **Check Now** to manually trigger a status check

Tracked bills are saved to `data/tracked_bills.json`.

### 4. Set up automated checks (GitHub Actions)

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

### 5. Commit your tracked bills

After adding bills via the UI, commit and push `data/tracked_bills.json` so GitHub Actions can check them:

```bash
git add data/tracked_bills.json
git commit -m "Add tracked bills"
git push
```

## How it works

1. You search for and track bills through the local web UI
2. GitHub Actions checks each tracked bill twice daily via the LegiScan API
3. Each bill has a `change_hash` — if it changes, the tool fetches updated details
4. New progress events (committee referrals, votes, passage, etc.) and status changes trigger an email alert
5. Updated hashes are committed back to the repo automatically

## Bill status codes

| Status | Meaning |
|--------|---------|
| Introduced | Bill has been introduced |
| Engrossed | Passed one chamber, sent to the other |
| Enrolled | Passed both chambers, sent to governor |
| Passed | Signed into law |
| Vetoed | Vetoed by governor |
| Failed | Did not pass |

## Project structure

```
src/legislator/
  app.py         - Flask server and API routes
  api.py         - LegiScan API client
  checker.py     - Change detection logic
  emailer.py     - Email alert formatting/sending
  config.py      - Environment variable configuration
  __main__.py    - CLI entry point
  static/
    index.html   - Web UI
```
