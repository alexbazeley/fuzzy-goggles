"""Entry point: python -m legislator"""

import argparse
import sys
import webbrowser
from pathlib import Path

from legislator.app import create_app, DATA_PATH
from legislator.api import LegiScanAPI
from legislator.checker import check_all_bills
from legislator.config import get_config, require_email_config
from legislator.emailer import send_alert


def run_server(host: str, port: int) -> None:
    """Start the local Flask server and open the browser."""
    app = create_app()
    print(f"Starting legislation tracker at http://{host}:{port}")
    webbrowser.open(f"http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


def run_check() -> None:
    """Check all tracked bills and send email alerts. Used by GitHub Actions."""
    config = get_config()
    require_email_config(config)
    api = LegiScanAPI(config["LEGISCAN_API_KEY"])

    changes = check_all_bills(api, DATA_PATH)
    if changes:
        send_alert(changes, config)
        print(f"{len(changes)} bill(s) changed. Email sent.")
    else:
        print("No changes detected.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="legislator", description="State legislation tracker")
    parser.add_argument("--check", action="store_true",
                        help="Check bills and send alerts (for CI/cron use)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    args = parser.parse_args()

    if args.check:
        run_check()
    else:
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()
