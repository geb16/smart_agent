# smart_agent/agent/observability/slack_notifier.py
from __future__ import annotations

import traceback

from agent.integrations.slack_client import SlackClient
from agent.utilities.theader import fire_and_forget


class SlackNotifier:
    """
    Infrastructure-level Slack notifications.
    Non-blocking, daemon-theaded notifications.
    Triggered by agent runtime behavior (NOT LLM decisions).
    """

    def __init__(self) -> None:
        self.slack = SlackClient()

    # --------------- INTERNAL WRAPPER ---------------
    def _safe_send(self, text: str, icon_emoji: str) -> None:
        """
        Fire-and-forget Slack send. Errors are isolated & never break flow.
        """

        def _send():
            try:
                self.slack.send_message(text=text, icon_emoji=icon_emoji)
            except Exception:
                traceback.print_exc()  # optional: redirect to file/log
                # Optionally: implement retry queue here

        fire_and_forget(_send)

    # ---------------- PLANNER ----------------
    def planner_success(self, step_count: int) -> None:
        self._safe_send(
            text=f"🧠 *Planner Success*\nGenerated `{step_count}` steps",
            icon_emoji=":brain:",
        )

    def planner_failure(self, error: Exception) -> None:
        self._safe_send(
            text=f"❌ *Planner Failure*\n```{error}```",
            icon_emoji=":rotating_light:",
        )

    # ---------------- EXECUTOR ----------------
    def executor_success(self) -> None:
        self._safe_send(
            text="⚙️ *Executor Success*\nWorkflow executed successfully",
            icon_emoji=":gear:",
        )

    def executor_failure(self, error: Exception) -> None:
        self._safe_send(
            text=f"❌ *Executor Failure*\n```{error}```",
            icon_emoji=":rotating_light:",
        )

    # ---------------- VERIFIER ----------------
    def verifier_success(self) -> None:
        self._safe_send(
            text="✅ *Verifier Success*\nOutput verified successfully",
            icon_emoji=":white_check_mark:",
        )

    def verifier_failure(self, error: Exception) -> None:
        self._safe_send(
            text=f"❌ *Verifier Failure*\n```{error}```",
            icon_emoji=":rotating_light:",
        )

    # ---------------- FINAL OUTPUT ----------------
    def final_blocked(self, reason: str) -> None:
        self._safe_send(
            text=f"⚠️ *Final Output Blocked*\nReason: {reason}",
            icon_emoji=":warning:",
        )

    def final_success(self) -> None:
        self._safe_send(
            text="✅ *Final Output Delivered Successfully*",
            icon_emoji=":white_check_mark:",
        )

    # cache hit/miss could also be notified here
    def cache_hit(self, layer: str) -> None:
        self._safe_send(
            text=f"🗃️ *Cache Hit* → {layer}",
            icon_emoji=":file_cabinet:",
        )
