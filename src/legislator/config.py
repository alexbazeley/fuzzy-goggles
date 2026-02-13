"""Configuration from environment variables."""

import os


def get_config() -> dict:
    """Return configuration dictionary from environment variables.

    Required: LEGISCAN_API_KEY
    Required for email alerts: SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO
    """
    config = {
        "LEGISCAN_API_KEY": os.environ.get("LEGISCAN_API_KEY"),
        "OPENSTATES_API_KEY": os.environ.get("OPENSTATES_API_KEY"),
        "SMTP_HOST": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "SMTP_PORT": int(os.environ.get("SMTP_PORT", "587")),
        "SMTP_USER": os.environ.get("SMTP_USER"),
        "SMTP_PASSWORD": os.environ.get("SMTP_PASSWORD"),
        "EMAIL_FROM": os.environ.get("EMAIL_FROM"),
        "EMAIL_TO": os.environ.get("EMAIL_TO"),
    }

    if not config["LEGISCAN_API_KEY"]:
        raise EnvironmentError("Missing required environment variable: LEGISCAN_API_KEY")

    return config


def require_email_config(config: dict) -> None:
    """Raise if email-related config is missing."""
    missing = [
        k for k in ("SMTP_USER", "SMTP_PASSWORD", "EMAIL_FROM", "EMAIL_TO")
        if not config.get(k)
    ]
    if missing:
        raise EnvironmentError(f"Missing required environment variables for email: {', '.join(missing)}")
