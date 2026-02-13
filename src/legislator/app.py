"""Flask local server for managing tracked bills."""

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from legislator.api import LegiScanAPI
from legislator.checker import (
    TrackedBill,
    load_tracked_bills,
    save_tracked_bills,
    check_all_bills,
)
from legislator.config import get_config, require_email_config
from legislator.emailer import send_alert

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
        bills.sort(key=lambda b: b.last_history_date or "", reverse=True)
        return jsonify([b.to_dict() for b in bills])

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
        )

        bills.append(new_bill)
        save_tracked_bills(bills, DATA_PATH)
        return jsonify(new_bill.to_dict()), 201

    @app.route("/api/bills/<int:bill_id>", methods=["DELETE"])
    def remove_bill(bill_id: int):
        bills = load_tracked_bills(DATA_PATH)
        before = len(bills)
        bills = [b for b in bills if b.bill_id != bill_id]
        if len(bills) == before:
            return jsonify({"error": "Bill not found"}), 404
        save_tracked_bills(bills, DATA_PATH)
        return jsonify({"ok": True})

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
            change_summaries.append({
                "bill_id": c.bill.bill_id,
                "state": c.bill.state,
                "bill_number": c.bill.bill_number,
                "title": c.bill.title,
                "old_status": c.old_status,
                "new_status": c.new_status,
                "new_progress_events": c.new_progress_events,
                "new_history_actions": c.new_history_actions,
            })

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

    return app
