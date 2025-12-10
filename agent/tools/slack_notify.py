# # agent/tools.py (snippet)
# from typing import Optional
# from agent.integrations.slack_client import SlackClient

# # existing tools...
# # from .math_tools import tool_add, tool_multiply, ...
# # etc.

# slack_client = SlackClient()  # create once


# def tool_slack_notify(
#     message: str, 
#     username: Optional[str] = None,
# ) -> dict:
#     """
#     Send a notification message to Slack (or Slack mock).

#     Args:
#         message: The text to send.
#         username: Optional override of the display name.

#     Returns:
#         A result dict with {ok: bool, status_code, response, ...}
#     """
#     display_name = username or "SmartAgent"
#     return slack_client.send_message(text=message, username=display_name)
