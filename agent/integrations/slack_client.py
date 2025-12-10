# # agent/integrations/slack_client.py
# from __future__ import annotations

# import os
# import json
# import logging
# from typing import Optional

# import requests

# logger = logging.getLogger(__name__)


# class SlackClient:
#     """
#     Minimal Slack webhook client.
#     You can point this to:
#       - a real Slack incoming webhook URL, or
#       - a local FastAPI mock endpoint for dev/testing.
#     """

#     def __init__(self, webhook_url: Optional[str] = None) -> None:
#         self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
#         if not self.webhook_url:
#             raise RuntimeError("SLACK_WEBHOOK_URL is not set")

#     def send_message(
#         self, 
#         text: str, 
#         username: str = "SmartAgent", 
#         icon_emoji: str = ":robot_face:"
#     ) -> dict:
#         """
#         Send a message to the configured webhook.
#         Returns a dict with status and response text.
#         """
#         payload = {
#             "text": text,
#             "username": username,
#             "icon_emoji": icon_emoji,
#         }

#         try:
#             resp = requests.post(
#                 self.webhook_url,
#                 data=json.dumps(payload),
#                 headers={"Content-Type": "application/json"},
#                 timeout=5,
#             )
#         except Exception as e:
#             logger.exception("Slack webhook request failed")
#             return {"ok": False, "error": str(e)}

#         if resp.status_code != 200:
#             return {
#                 "ok": False,
#                 "status_code": resp.status_code,
#                 "response": resp.text,
#             }

#         return {"ok": True, "status_code": resp.status_code, "response": resp.text}
