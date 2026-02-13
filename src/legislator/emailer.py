"""Email alert formatting and sending."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from legislator.checker import BillChange, PROGRESS_EVENTS, STATUS_CODES


PRIORITY_LABELS = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}


def format_change_summary(change: BillChange) -> str:
    """Format a single bill's changes into readable text."""
    b = change.bill
    priority_tag = f" [{PRIORITY_LABELS.get(b.priority, 'MEDIUM')}]" if b.priority != "medium" else ""
    lines = [
        f"{'=' * 60}",
        f"{b.state} {b.bill_number}: {b.title}{priority_tag}",
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

    # Sponsor changes
    if change.sponsor_changes:
        if change.sponsor_changes.added:
            lines.append("\nNew sponsors:")
            for s in change.sponsor_changes.added:
                party = f" ({s['party']})" if s.get("party") else ""
                stype = f" - {s['sponsor_type']}" if s.get("sponsor_type") else ""
                lines.append(f"  + {s['name']}{party}{stype}")
        if change.sponsor_changes.removed:
            lines.append("\nRemoved sponsors:")
            for s in change.sponsor_changes.removed:
                party = f" ({s['party']})" if s.get("party") else ""
                lines.append(f"  - {s['name']}{party}")

    # New hearing/calendar events
    if change.new_calendar_events:
        lines.append("\nUpcoming hearings/events:")
        for c in change.new_calendar_events:
            loc = f" at {c['location']}" if c.get("location") else ""
            lines.append(f"  - {c['date']}: {c['description']}{loc}")

    lines.append(f"\nLegiScan: {b.url}")
    lines.append(f"Official: {b.state_link}")
    return "\n".join(lines)


def send_alert(changes: list[BillChange], config: dict) -> None:
    """Send email alert for bill changes. Does nothing if changes is empty."""
    if not changes:
        return

    # Sort: high priority first, then medium, then low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_changes = sorted(changes, key=lambda c: priority_order.get(c.bill.priority, 1))

    high_count = sum(1 for c in sorted_changes if c.bill.priority == "high")
    subject_prefix = "URGENT: " if high_count > 0 else ""

    header = f"Legislation Tracker Alert - {len(sorted_changes)} bill(s) updated\n"
    if high_count:
        header += f"  ({high_count} high-priority bill(s))\n"
    header += "\n"

    sections = [format_change_summary(c) for c in sorted_changes]
    body = header + "\n\n".join(sections)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{subject_prefix}Bill Tracker: {len(sorted_changes)} bill(s) updated"
    msg["From"] = config["EMAIL_FROM"]
    msg["To"] = config["EMAIL_TO"]
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(config["SMTP_HOST"], config["SMTP_PORT"]) as server:
        server.starttls()
        server.login(config["SMTP_USER"], config["SMTP_PASSWORD"])
        server.sendmail(config["EMAIL_FROM"], config["EMAIL_TO"].split(","), msg.as_string())

    print(f"Alert email sent to {config['EMAIL_TO']}")
