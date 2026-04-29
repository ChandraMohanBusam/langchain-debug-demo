"""
teams_alert.py
Microsoft Teams Alert Integration via Incoming Webhook

Used by the Document Review Agent to send error and loop alerts
to a Microsoft Teams channel.

Setup:
    See README.md for the full step-by-step guide to create a Teams
    Incoming Webhook and configure it in your .env file.

Message Format:
    Uses Teams Adaptive Card format (simple text version).
    For richer cards with buttons and tables, use the Adaptive Card Designer:
    https://adaptivecards.io/designer/
"""

import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")

# Severity color mapping (Teams uses hex colors for card theming)
SEVERITY_COLORS = {
    "info": "0076D7",       # Blue
    "warning": "FFC107",    # Amber
    "error": "D32F2F",      # Red
    "success": "2E7D32",    # Green
}

SEVERITY_ICONS = {
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "🔥",
    "success": "✅",
}


def send_teams_alert(
    title: str,
    message: str,
    severity: str = "info",
    details: dict = None,
) -> bool:
    """
    Sends an alert message to a Microsoft Teams channel via Incoming Webhook.

    Args:
        title:    Alert title shown in bold at the top of the card.
        message:  The main alert body text.
        severity: One of 'info', 'warning', 'error', 'success'.
                  Controls the icon and color accent.
        details:  Optional dict of key/value pairs added as a fact table.
                  Example: {"Document": "contract.pdf", "Revision": "3"}

    Returns:
        True if the message was posted successfully, False otherwise.
    """
    webhook_url = TEAMS_WEBHOOK_URL

    if not webhook_url:
        # Console fallback when webhook is not configured
        icon = SEVERITY_ICONS.get(severity, "ℹ️")
        print(
            f"\n[TeamsAlert - Console Fallback]\n"
            f"{icon} {title}\n"
            f"{message}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        return False

    icon = SEVERITY_ICONS.get(severity, "ℹ️")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build the Teams message card payload
    # Using MessageCard format for broad compatibility
    # For newer Adaptive Cards, see: https://docs.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/connectors-using
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": SEVERITY_COLORS.get(severity, "0076D7"),
        "summary": title,
        "sections": [
            {
                "activityTitle": f"{icon} **{title}**",
                "activitySubtitle": f"Document Review Agent | {timestamp}",
                "activityText": message,
                "facts": [
                    {"name": "Severity", "value": severity.upper()},
                    {"name": "Time", "value": timestamp},
                    {"name": "Source", "value": "LangGraph Document Review Agent"},
                ],
                "markdown": True,
            }
        ],
    }

    # Add optional details as additional facts
    if details:
        for key, value in details.items():
            payload["sections"][0]["facts"].append(
                {"name": key, "value": str(value)}
            )

    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if response.status_code == 200:
            return True
        else:
            print(
                f"[TeamsAlert] Failed. "
                f"Status: {response.status_code} | Response: {response.text}"
            )
            return False
    except requests.exceptions.RequestException as e:
        print(f"[TeamsAlert] Network error: {e}")
        return False


def send_loop_detected_alert(
    node_name: str, cycle_count: int, thread_id: str
) -> bool:
    """
    Convenience function for loop detection alerts.
    Called when the same node has been visited too many times.
    """
    return send_teams_alert(
        title="Loop Detected - Document Review Agent",
        message=(
            f"Node '{node_name}' has been visited {cycle_count} times "
            f"in thread '{thread_id}'. The agent may be stuck."
        ),
        severity="error",
        details={
            "Node": node_name,
            "Visit Count": str(cycle_count),
            "Thread ID": thread_id,
            "Action": "Review checkpointer state via get_state_history()",
        },
    )
