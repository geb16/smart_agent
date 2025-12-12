from __future__ import annotations

import logging
from typing import Dict

import requests

from agent.config import SLACK_WEBHOOK_URL

logger = logging.getLogger(__name__)


class SlackClient:
    """
    Slack Incoming Webhook client (REAL).
    Sends messages directly to Slack channels.
    """

    def __init__(self) -> None:
        if not SLACK_WEBHOOK_URL:
            raise RuntimeError("SLACK_WEBHOOK_URL is not set")

        self.webhook_url = SLACK_WEBHOOK_URL

    def send_message(
        self,
        text: str,
        username: str = "SmartAgent",
        icon_emoji: str = ":robot_face:",
    ) -> Dict[str, object]:
        payload = {
            "text": text,
            "username": username,
            "icon_emoji": icon_emoji,
        }

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,   # ✅ correct way
                timeout=5,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Slack webhook call failed")
            return {"ok": False, "error": str(exc)}

        # Slack returns plain text "ok"
        return {"ok": True, "response": resp.text}
