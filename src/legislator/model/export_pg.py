"""Export training data from the Open States PostgreSQL dump.

Connects to a local PostgreSQL database (restored from the Open States
monthly dump), queries bills with sponsorships, actions, and session
metadata, and writes JSON files per state-session into the model data
directory — in the exact format the training pipeline expects.

Setup:
    1. Install PostgreSQL (e.g. `sudo apt install postgresql` or
       `brew install postgresql`)
    2. Download the monthly dump from:
       https://data.openstates.org/postgres/monthly/
       (file looks like: 2026-02-public.pgdump)
    3. Create a database and restore the dump:
           createdb openstates
           pg_restore --no-owner --no-acl -d openstates 2026-02-public.pgdump
       This can take 10-30 minutes depending on your machine.
    4. Install the Python PostgreSQL driver:
           pip install psycopg2-binary
    5. Run this script:
           PYTHONPATH=src python -m legislator.model.export_pg
       Or with a custom connection string:
           PYTHONPATH=src python -m legislator.model.export_pg \
               --db "postgresql://user:pass@localhost/openstates"

    The script exports JSON files to src/legislator/model/data/ and then
    you can train normally:
        PYTHONPATH=src python -m legislator.model.train

Options:
    --db          PostgreSQL connection string (default: postgresql://localhost/openstates)
    --states      Comma-separated state abbreviations to export (default: all)
    --out         Output directory (default: src/legislator/model/data/)
    --min-actions Minimum number of actions a bill must have (default: 2)
    --limit       Max bills per state-session (default: no limit)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"


def get_connection(dsn: str):
    """Create a PostgreSQL connection."""
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 is not installed.")
        print("Install it with: pip install psycopg2-binary")
        sys.exit(1)
    return psycopg2.connect(dsn)


def fetch_states(conn) -> list[str]:
    """Get list of US state abbreviations available in the database."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT
            UPPER(SUBSTRING(j.id FROM '/state:([a-z]+)/'))
        FROM opencivicdata_jurisdiction j
        WHERE j.id LIKE '%%/state:%%'
          AND j.classification = 'state'
        ORDER BY 1
    """)
    states = [row[0] for row in cur.fetchall() if row[0]]
    cur.close()
    return states


def fetch_sessions(conn, state: str) -> list[dict]:
    """Get all legislative sessions for a state."""
    cur = conn.cursor()
    cur.execute("""
        SELECT ls.id, ls.identifier, ls.name, ls.classification,
               ls.start_date, ls.end_date, ls.active
        FROM opencivicdata_legislativesession ls
        JOIN opencivicdata_jurisdiction j ON ls.jurisdiction_id = j.id
        WHERE j.id LIKE %s
        ORDER BY ls.start_date DESC NULLS LAST
    """, (f"%/state:{state.lower()}/%",))
    sessions = []
    for row in cur.fetchall():
        sessions.append({
            "id": row[0],
            "identifier": row[1],
            "name": row[2],
            "classification": row[3],
            "start_date": row[4] or "",
            "end_date": row[5] or "",
            "active": row[6],
        })
    cur.close()
    return sessions


def fetch_bills_for_session(
    conn, session_id: str, state: str,
    min_actions: int = 2, limit: Optional[int] = None,
) -> list[dict]:
    """Fetch all bills for a session with actions and sponsorships.

    Returns list of bill dicts in Open States JSON format compatible
    with the training pipeline's extract_from_openstates().
    """
    cur = conn.cursor()

    # Fetch bills
    limit_clause = f"LIMIT {limit}" if limit else ""
    cur.execute(f"""
        SELECT
            b.id,
            b.identifier,
            b.title,
            b.classification,
            b.subject,
            b.first_action_date,
            b.latest_action_date,
            b.latest_passage_date,
            o.name AS from_org_name,
            o.classification AS from_org_class
        FROM opencivicdata_bill b
        LEFT JOIN opencivicdata_organization o ON b.from_organization_id = o.id
        WHERE b.legislative_session_id = %s
        ORDER BY b.identifier
        {limit_clause}
    """, (session_id,))

    bill_rows = cur.fetchall()
    if not bill_rows:
        cur.close()
        return []

    bill_ids = [row[0] for row in bill_rows]

    # Fetch all actions for these bills in bulk
    cur.execute("""
        SELECT
            ba.bill_id,
            ba.description,
            ba.date,
            ba.classification,
            ba.order,
            o.name AS org_name,
            o.classification AS org_class
        FROM opencivicdata_billaction ba
        LEFT JOIN opencivicdata_organization o ON ba.organization_id = o.id
        WHERE ba.bill_id = ANY(%s)
        ORDER BY ba.bill_id, ba.order
    """, (bill_ids,))

    actions_by_bill = {}
    for row in cur.fetchall():
        bid = row[0]
        if bid not in actions_by_bill:
            actions_by_bill[bid] = []
        # classification is a Postgres array, comes as a Python list
        classifications = row[3] if row[3] else []
        actions_by_bill[bid].append({
            "description": row[1] or "",
            "date": row[2] or "",
            "classification": classifications,
            "order": row[4],
            "organization": {
                "name": row[5] or "",
                "classification": row[6] or "",
            },
        })

    # Fetch all sponsorships with person party info
    cur.execute("""
        SELECT
            bs.bill_id,
            bs.name,
            bs."primary",
            bs.classification,
            bs.entity_type,
            p.id AS person_id,
            p.name AS person_name,
            p.primary_party
        FROM opencivicdata_billsponsorship bs
        LEFT JOIN opencivicdata_person p ON bs.person_id = p.id
        WHERE bs.bill_id = ANY(%s)
        ORDER BY bs.bill_id
    """, (bill_ids,))

    sponsorships_by_bill = {}
    for row in cur.fetchall():
        bid = row[0]
        if bid not in sponsorships_by_bill:
            sponsorships_by_bill[bid] = []
        sponsorship = {
            "name": row[1] or "",
            "primary": row[2] or False,
            "classification": row[3] or "",
            "entity_type": row[4] or "person",
        }
        # Attach person data with primary_party — this is the key field
        # that was missing from JSON bulk exports
        if row[5]:  # person_id not null
            sponsorship["person"] = {
                "id": row[5],
                "name": row[6] or "",
                "primary_party": row[7] or "",
            }
        sponsorships_by_bill[bid].append(sponsorship)

    cur.close()

    # Build bill dicts in the format extract_from_openstates() expects
    jurisdiction_id = f"ocd-jurisdiction/country:us/state:{state.lower()}/government"
    bills = []

    for row in bill_rows:
        bid = row[0]
        actions = actions_by_bill.get(bid, [])

        # Skip bills with too few actions
        if len(actions) < min_actions:
            continue

        bill_classification = row[3] if row[3] else []
        bill_subjects = row[4] if row[4] else []

        bill = {
            "id": bid,
            "identifier": row[1] or "",
            "title": row[2] or "",
            "classification": bill_classification,
            "subject": bill_subjects,
            "first_action_date": row[5] or "",
            "latest_action_date": row[6] or "",
            "latest_passage_date": row[7] or "",
            "from_organization": {
                "name": row[8] or "",
                "classification": row[9] or "",
            },
            "jurisdiction": {"id": jurisdiction_id},
            "jurisdiction_id": jurisdiction_id,
            "legislative_session": "",  # filled below
            "actions": actions,
            "sponsorships": sponsorships_by_bill.get(bid, []),
        }
        bills.append(bill)

    return bills


def export_state(
    conn, state: str, out_dir: Path,
    min_actions: int = 2, limit: Optional[int] = None,
) -> int:
    """Export all sessions for a state. Returns total bill count."""
    sessions = fetch_sessions(conn, state)
    if not sessions:
        return 0

    total = 0
    for sess in sessions:
        bills = fetch_bills_for_session(
            conn, sess["id"], state,
            min_actions=min_actions, limit=limit,
        )
        if not bills:
            continue

        # Set session identifier on each bill
        for b in bills:
            b["legislative_session"] = sess["identifier"]

        # Write one JSON file per state-session
        safe_name = sess["identifier"].replace("/", "-").replace(" ", "_")
        filename = f"{state.lower()}_{safe_name}.json"
        filepath = out_dir / filename

        with open(filepath, "w") as f:
            json.dump(bills, f)

        total += len(bills)
        active = " (active)" if sess.get("active") else ""
        print(f"  {sess['identifier']}: {len(bills)} bills{active} -> {filename}")

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Export Open States PostgreSQL dump to training JSON files")
    parser.add_argument(
        "--db", default="postgresql://localhost/openstates",
        help="PostgreSQL connection string (default: postgresql://localhost/openstates)")
    parser.add_argument(
        "--states", default=None,
        help="Comma-separated state abbreviations (default: all US states)")
    parser.add_argument(
        "--out", default=None,
        help=f"Output directory (default: {DATA_DIR})")
    parser.add_argument(
        "--min-actions", type=int, default=2,
        help="Skip bills with fewer than N actions (default: 2)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max bills per state-session (default: no limit)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Open States PostgreSQL -> Training JSON Export")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Output:   {out_dir}")
    print()

    conn = get_connection(args.db)

    # Determine which states to export
    if args.states:
        states = [s.strip().upper() for s in args.states.split(",")]
    else:
        print("Discovering available states...")
        states = fetch_states(conn)
        print(f"Found {len(states)} states: {', '.join(states)}\n")

    grand_total = 0
    states_with_data = 0

    for state in states:
        print(f"\n{state}:")
        count = export_state(
            conn, state, out_dir,
            min_actions=args.min_actions,
            limit=args.limit,
        )
        if count > 0:
            states_with_data += 1
        grand_total += count

    conn.close()

    print("\n" + "=" * 60)
    print(f"Export complete: {grand_total} bills from "
          f"{states_with_data} states")
    print(f"JSON files written to: {out_dir}")
    print()
    print("Next step — train the model:")
    print("  PYTHONPATH=src python -m legislator.model.train")
    print("=" * 60)


if __name__ == "__main__":
    main()
