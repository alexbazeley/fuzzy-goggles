"""Email alert formatting and sending."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from legislator.checker import BillChange, PROGRESS_EVENTS, STATUS_CODES


def format_change_summary(change: BillChange) -> str:
    """Format a single bill's changes into readable text."""
    b = change.bill
    lines = [
        f"{'=' * 60}",
        f"{b.state} {b.bill_number}: {b.title}",
        f"{'=' * 60}",
    ]

    if change.old_status != change.new_status:
        old = STATUS_CODES.get(change.old_status, str(change.old_status))
        new = STATUS_CODES.get(change.new_status, str(change.new_status))
        lines.append(f"\nStatus changed: {old} -> {new}")

    if change.new_progress_events:
        lines.append("\nNew milestones:")
        for p in change.new_progress_events:
            event_name = PROGRESS_EVENTS.get(p["event"], f"Event {p['event']}")
            lines.append(f"  - {p['date']}: {event_name}")

    if change.new_history_actions:
        lines.append("\nRecent actions:")
        for h in change.new_history_actions:
            chamber = h.get("chamber", "")
            prefix = f"[{chamber}] " if chamber else ""
            lines.append(f"  - {h['date']}: {prefix}{h['action']}")

    lines.append(f"\nLegiScan: {b.url}")
    lines.append(f"Official: {b.state_link}")
    return "\n".join(lines)


def send_alert(changes: list[BillChange], config: dict) -> None:
    """Send email alert for bill changes. Does nothing if changes is empty."""
    if not changes:
        return

    header = f"Legislation Tracker Alert - {len(changes)} bill(s) updated\n\n"
    sections = [format_change_summary(c) for c in changes]
    body = header + "\n\n".join(sections)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Bill Tracker: {len(changes)} bill(s) updated"
    msg["From"] = config["EMAIL_FROM"]
    msg["To"] = config["EMAIL_TO"]
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(config["SMTP_HOST"], config["SMTP_PORT"]) as server:
        server.starttls()
        server.login(config["SMTP_USER"], config["SMTP_PASSWORD"])
        server.sendmail(config["EMAIL_FROM"], config["EMAIL_TO"].split(","), msg.as_string())

    print(f"Alert email sent to {config['EMAIL_TO']}")
