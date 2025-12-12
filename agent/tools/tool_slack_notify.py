from __future__ import annotations

from typing import Dict

from agent.integrations.slack_client import SlackClient


def tool_slack_notify(
    message: str,
    username: str = "SmartAgent",
) -> Dict[str, object]:
    """
    Send a notification to Slack using Incoming Webhook.

    Args:
        message: Message text to send
        username: Optional Slack display name
    """
    slack = SlackClient()
    return slack.send_message(
        text=message,
        username=username,
    )
