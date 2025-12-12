from __future__ import annotations

from typing import Optional
from agent.integrations.slack_client import SlackClient


class SlackNotifier:
    """
    Infrastructure-level Slack notifications.
    Triggered by agent runtime behavior (NOT LLM decisions).
    """

    def __init__(self) -> None:
        self.slack = SlackClient()

    # ---------------- PLANNER ----------------
    def planner_success(self, step_count: int) -> None:
        self.slack.send_message(
            text=f"🧠 *Planner Success*\nGenerated `{step_count}` steps",
            icon_emoji=":brain:",
        )

    def planner_failure(self, error: Exception) -> None:
        self.slack.send_message(
            text=f"❌ *Planner Failure*\n```{error}```",
            icon_emoji=":rotating_light:",
        )

    # ---------------- EXECUTOR ----------------
    def executor_success(self) -> None:
        self.slack.send_message(
            text="⚙️ *Executor Success*\nWorkflow executed successfully",
            icon_emoji=":gear:",
        )

    def executor_failure(self, error: Exception) -> None:
        self.slack.send_message(
            text=f"❌ *Executor Failure*\n```{error}```",
            icon_emoji=":rotating_light:",
        )
    # ---------------- VERIFIER ----------------
    def verifier_success(self) -> None:
        self.slack.send_message(
            text="✅ *Verifier Success*\nOutput verified successfully",
            icon_emoji=":white_check_mark:",
        )
    def verifier_failure(self, error: Exception) -> None:
        self.slack.send_message(
            text=f"❌ *Verifier Failure*\n```{error}```",
            icon_emoji=":rotating_light:",
        )
    # ---------------- FINAL OUTPUT ----------------
    def final_blocked(self, reason: str) -> None:
        self.slack.send_message(
            text=f"⚠️ *Final Output Blocked*\nReason: {reason}",
            icon_emoji=":warning:",
        )

    def final_success(self) -> None:
        self.slack.send_message(
            text="✅ *Final Output Delivered Successfully*",
            icon_emoji=":white_check_mark:",
        )
