"""Flask local server for managing tracked bills."""

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from legislator.api import LegiScanAPI
from legislator.checker import (
    TrackedBill,
    PROGRESS_EVENTS,
    load_tracked_bills,
    save_tracked_bills,
    check_all_bills,
    _extract_sponsors,
    _extract_calendar,
    _extract_subjects,
)
from legislator.config import get_config, require_email_config
from legislator.emailer import send_alert
from legislator.scoring import compute_passage_likelihood, get_session_status
from legislator.related import find_related_bills
from legislator.solar import analyze_bill_text, decode_bill_text

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "tracked_bills.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR))
    config = get_config()
    api = LegiScanAPI(config["LEGISCAN_API_KEY"])

    @app.route("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/api/bills", methods=["GET"])
    def list_bills():
        bills = load_tracked_bills(DATA_PATH)

        # Filtering
        state_filter = request.args.get("state")
        status_filter = request.args.get("status", type=int)
        priority_filter = request.args.get("priority")

        if state_filter:
            bills = [b for b in bills if b.state == state_filter]
        if status_filter is not None:
            bills = [b for b in bills if b.status == status_filter]
        if priority_filter:
            bills = [b for b in bills if b.priority == priority_filter]

        # Sorting
        sort_by = request.args.get("sort", "date")
        reverse = request.args.get("order", "desc") == "desc"

        if sort_by == "state":
            bills.sort(key=lambda b: b.state, reverse=reverse)
        elif sort_by == "status":
            bills.sort(key=lambda b: b.status, reverse=reverse)
        elif sort_by == "priority":
            priority_order = {"high": 0, "medium": 1, "low": 2}
            bills.sort(key=lambda b: priority_order.get(b.priority, 1), reverse=reverse)
        elif sort_by == "title":
            bills.sort(key=lambda b: b.title.lower(), reverse=reverse)
        else:  # default: date
            bills.sort(key=lambda b: b.last_history_date or "", reverse=reverse)

        # Enrich each bill with passage likelihood, session status, and milestones
        passage_filter = request.args.get("passage")

        result = []
        for b in bills:
            d = b.to_dict()
            d["passage"] = compute_passage_likelihood(b)

            # Filter by passage label after computing scores
            if passage_filter and d["passage"]["label"] != passage_filter:
                continue

            session_info = get_session_status(b)
            if session_info:
                d["session_status"] = session_info
            # Map progress event codes to human-readable milestone labels with dates
            if b.progress_details:
                d["milestones"] = [
                    {"label": PROGRESS_EVENTS.get(p["event"], f"Event {p['event']}"), "date": p.get("date", "")}
                    for p in b.progress_details
                ]
            else:
                d["milestones"] = [
                    {"label": PROGRESS_EVENTS.get(code, f"Event {code}"), "date": ""}
                    for code in b.progress_events
                ]
            result.append(d)

        return jsonify(result)

    @app.route("/api/bills", methods=["POST"])
    def add_bill():
        body = request.get_json()
        bill_id = body.get("bill_id")
        if not bill_id:
            return jsonify({"error": "bill_id is required"}), 400

        bills = load_tracked_bills(DATA_PATH)
        if any(b.bill_id == bill_id for b in bills):
            return jsonify({"error": "Bill already tracked"}), 409

        try:
            bill_data = api.get_bill(bill_id)
        except Exception as e:
            return jsonify({"error": str(e)}), 502

        history = bill_data.get("history", [])
        latest = max(history, key=lambda h: h["date"]) if history else {"date": "", "action": ""}
        progress = bill_data.get("progress", [])

        # Extract session info
        session = bill_data.get("session", {})
        # Extract committee info
        committee = bill_data.get("committee", {})

        new_bill = TrackedBill(
            bill_id=bill_data["bill_id"],
            state=bill_data["state"],
            bill_number=bill_data["bill_number"],
            title=bill_data["title"],
            url=bill_data.get("url", ""),
            state_link=bill_data.get("state_link", ""),
            change_hash=bill_data["change_hash"],
            status=bill_data["status"],
            last_history_date=latest["date"],
            last_history_action=latest["action"],
            progress_events=[p["event"] for p in progress],
            progress_details=[{"event": p["event"], "date": p.get("date", "")} for p in progress],
            priority=body.get("priority", "medium"),
            sponsors=_extract_sponsors(bill_data),
            calendar=_extract_calendar(bill_data),
            session_id=session.get("session_id", 0),
            session_name=session.get("session_name", ""),
            session_year_start=session.get("year_start", 0),
            session_year_end=session.get("year_end", 0),
            description=bill_data.get("description", ""),
            subjects=_extract_subjects(bill_data),
            committee=committee.get("name", "") if committee else "",
            committee_id=committee.get("committee_id", 0) if committee else 0,
            history=[{"date": h["date"], "action": h["action"], "chamber": h.get("chamber", ""), "chamber_id": h.get("chamber_id", 0)} for h in history],
        )

        bills.append(new_bill)
        save_tracked_bills(bills, DATA_PATH)

        d = new_bill.to_dict()
        d["passage"] = compute_passage_likelihood(new_bill)
        session_info = get_session_status(new_bill)
        if session_info:
            d["session_status"] = session_info
        return jsonify(d), 201

    @app.route("/api/bills/<int:bill_id>/refresh", methods=["POST"])
    def refresh_bill(bill_id: int):
        """Force re-fetch bill data from LegiScan, updating sponsors, calendar, etc."""
        bills = load_tracked_bills(DATA_PATH)
        bill = next((b for b in bills if b.bill_id == bill_id), None)
        if not bill:
            return jsonify({"error": "Bill not found"}), 404

        try:
            bill_data = api.get_bill(bill_id)
        except Exception as e:
            return jsonify({"error": str(e)}), 502

        # Update all fields from fresh API data
        bill.change_hash = bill_data["change_hash"]
        bill.status = bill_data["status"]
        bill.title = bill_data.get("title", bill.title)
        bill.url = bill_data.get("url", bill.url)
        bill.state_link = bill_data.get("state_link", bill.state_link)
        bill.description = bill_data.get("description", bill.description)
        bill.sponsors = _extract_sponsors(bill_data)
        bill.calendar = _extract_calendar(bill_data)
        bill.subjects = _extract_subjects(bill_data)

        progress = bill_data.get("progress", [])
        bill.progress_events = [p["event"] for p in progress]
        bill.progress_details = [{"event": p["event"], "date": p.get("date", "")} for p in progress]

        history = bill_data.get("history", [])
        if history:
            latest = max(history, key=lambda h: h["date"])
            bill.last_history_date = latest["date"]
            bill.last_history_action = latest["action"]
            bill.history = [{"date": h["date"], "action": h["action"], "chamber": h.get("chamber", ""), "chamber_id": h.get("chamber_id", 0)} for h in history]

        session = bill_data.get("session", {})
        if session:
            bill.session_id = session.get("session_id", bill.session_id)
            bill.session_name = session.get("session_name", bill.session_name)
            bill.session_year_start = session.get("year_start", bill.session_year_start)
            bill.session_year_end = session.get("year_end", bill.session_year_end)

        committee = bill_data.get("committee", {})
        if committee:
            bill.committee = committee.get("name", bill.committee)
            bill.committee_id = committee.get("committee_id", bill.committee_id)

        save_tracked_bills(bills, DATA_PATH)

        d = bill.to_dict()
        d["passage"] = compute_passage_likelihood(bill)
        if bill.progress_details:
            d["milestones"] = [
                {"label": PROGRESS_EVENTS.get(p["event"], f"Event {p['event']}"), "date": p.get("date", "")}
                for p in bill.progress_details
            ]
        else:
            d["milestones"] = [
                {"label": PROGRESS_EVENTS.get(code, f"Event {code}"), "date": ""}
                for code in bill.progress_events
            ]
        return jsonify(d)

    @app.route("/api/bills/<int:bill_id>/analyze", methods=["POST"])
    def analyze_bill(bill_id: int):
        """Fetch bill text from LegiScan and scan for solar-relevant keywords."""
        bills = load_tracked_bills(DATA_PATH)
        bill = next((b for b in bills if b.bill_id == bill_id), None)
        if not bill:
            return jsonify({"error": "Bill not found"}), 404

        # Return cached results if available
        if bill.solar_keywords:
            return jsonify({"bill_id": bill_id, "solar_keywords": bill.solar_keywords, "cached": True})

        try:
            bill_data = api.get_bill(bill_id)
        except Exception as e:
            return jsonify({"error": str(e)}), 502

        # LegiScan provides text documents in the 'texts' array
        texts = bill_data.get("texts", [])
        if not texts:
            return jsonify({"bill_id": bill_id, "solar_keywords": [], "error": "No bill text available"})

        # Get the most recent text document
        latest_text = max(texts, key=lambda t: t.get("date", ""))
        doc_id = latest_text.get("doc_id", 0)
        if not doc_id:
            return jsonify({"bill_id": bill_id, "solar_keywords": [], "error": "No doc_id found"})

        try:
            doc_data = api.get_bill_text(doc_id)
        except Exception as e:
            return jsonify({"error": f"Failed to fetch bill text: {e}"}), 502

        text_content = decode_bill_text(doc_data)
        if not text_content:
            return jsonify({"bill_id": bill_id, "solar_keywords": [], "error": "Could not decode bill text (may be PDF)"})

        keywords = analyze_bill_text(text_content)

        # Cache results
        bill.solar_keywords = keywords
        save_tracked_bills(bills, DATA_PATH)

        return jsonify({"bill_id": bill_id, "solar_keywords": keywords, "cached": False})

    @app.route("/api/bills/<int:bill_id>", methods=["DELETE"])
    def remove_bill(bill_id: int):
        bills = load_tracked_bills(DATA_PATH)
        before = len(bills)
        bills = [b for b in bills if b.bill_id != bill_id]
        if len(bills) == before:
            return jsonify({"error": "Bill not found"}), 404
        save_tracked_bills(bills, DATA_PATH)
        return jsonify({"ok": True})

    @app.route("/api/bills/<int:bill_id>/priority", methods=["PATCH"])
    def set_priority(bill_id: int):
        body = request.get_json()
        priority = body.get("priority", "medium")
        if priority not in ("high", "medium", "low"):
            return jsonify({"error": "Priority must be high, medium, or low"}), 400

        bills = load_tracked_bills(DATA_PATH)
        found = None
        for b in bills:
            if b.bill_id == bill_id:
                b.priority = priority
                found = b
                break
        if not found:
            return jsonify({"error": "Bill not found"}), 404

        save_tracked_bills(bills, DATA_PATH)
        d = found.to_dict()
        d["passage"] = compute_passage_likelihood(found)
        return jsonify(d)

    @app.route("/api/bills/<int:bill_id>/related", methods=["GET"])
    def get_related(bill_id: int):
        bills = load_tracked_bills(DATA_PATH)
        bill = next((b for b in bills if b.bill_id == bill_id), None)
        if not bill:
            return jsonify({"error": "Bill not found"}), 404

        try:
            related = find_related_bills(api, bill)
        except Exception as e:
            return jsonify({"error": str(e)}), 502

        return jsonify({"bill_id": bill_id, "related": related})

    @app.route("/api/search", methods=["GET"])
    def search_bills():
        state = request.args.get("state", "ALL")
        query = request.args.get("q", "")
        page = request.args.get("page", 1, type=int)
        if not query:
            return jsonify({"error": "q parameter is required"}), 400

        try:
            results = api.search(state=state, query=query, page=page)
        except Exception as e:
            return jsonify({"error": str(e)}), 502

        summary = results.get("summary", {})
        result_list = []
        for key in sorted(results.keys()):
            if key == "summary":
                continue
            result_list.append(results[key])

        return jsonify({"summary": summary, "results": result_list})

    @app.route("/api/check", methods=["POST"])
    def trigger_check():
        changes = check_all_bills(api, DATA_PATH)
        change_summaries = []
        for c in changes:
            summary = {
                "bill_id": c.bill.bill_id,
                "state": c.bill.state,
                "bill_number": c.bill.bill_number,
                "title": c.bill.title,
                "old_status": c.old_status,
                "new_status": c.new_status,
                "new_progress_events": c.new_progress_events,
                "new_history_actions": c.new_history_actions,
            }
            if c.sponsor_changes:
                summary["sponsor_changes"] = {
                    "added": c.sponsor_changes.added,
                    "removed": c.sponsor_changes.removed,
                }
            if c.new_calendar_events:
                summary["new_calendar_events"] = c.new_calendar_events
            change_summaries.append(summary)

        # Try to send email if configured
        email_sent = False
        if changes:
            try:
                require_email_config(config)
                send_alert(changes, config)
                email_sent = True
            except EnvironmentError:
                pass  # Email not configured, that's fine for local use

        return jsonify({
            "bills_checked": len(load_tracked_bills(DATA_PATH)),
            "changes_found": len(changes),
            "changes": change_summaries,
            "email_sent": email_sent,
        })

    @app.route("/api/dashboard", methods=["GET"])
    def dashboard():
        """Dashboard summary with stats and session warnings."""
        bills = load_tracked_bills(DATA_PATH)

        # Stats
        by_status = {}
        by_state = {}
        by_priority = {"high": 0, "medium": 0, "low": 0}
        session_warnings = []

        for b in bills:
            by_status[b.status_text] = by_status.get(b.status_text, 0) + 1
            by_state[b.state] = by_state.get(b.state, 0) + 1
            by_priority[b.priority] = by_priority.get(b.priority, 0) + 1

            sess = get_session_status(b)
            if sess and sess.get("warning"):
                session_warnings.append({
                    "bill_id": b.bill_id,
                    "state": b.state,
                    "bill_number": b.bill_number,
                    "title": b.title,
                    "warning": sess["warning"],
                    "days_remaining": sess["days_remaining"],
                    "is_ending_soon": sess["is_ending_soon"],
                })

        # Bills with upcoming hearings
        upcoming_hearings = []
        from datetime import date as date_cls
        today = date_cls.today().isoformat()
        for b in bills:
            for c in b.calendar:
                if c.get("date", "") >= today:
                    upcoming_hearings.append({
                        "bill_id": b.bill_id,
                        "state": b.state,
                        "bill_number": b.bill_number,
                        "date": c["date"],
                        "description": c.get("description", ""),
                        "location": c.get("location", ""),
                    })
        upcoming_hearings.sort(key=lambda h: h["date"])

        return jsonify({
            "total_bills": len(bills),
            "by_status": by_status,
            "by_state": by_state,
            "by_priority": by_priority,
            "session_warnings": session_warnings,
            "upcoming_hearings": upcoming_hearings[:20],
        })

    return app
