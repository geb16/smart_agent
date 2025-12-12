# smart_agent/agent/safety_gardrails/safety_manager.py

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from agent.config import OPENAI_MODEL, client
from agent.safety_guardrails.pii_cleaner import scrub_output


class SafetyManager:
    """
    Unified safety gateway for all agent operations.
    Handles:
        - Input sanitization (moderation, jailbreak)
        - Output safety validation
        - PII scrubbing
        - RAG hallucination check
        - Tool safety validation
        - JSON schema enforcement
        - Trace logging for audits
    """

    # -------------------------------------------
    #  Initializer
    # -------------------------------------------
    def __init__(self, allowed_tools: Dict[str, bool]) -> None:
        self.allowed_tools = allowed_tools
        self.traces: List[Dict[str, Any]] = []  # Store safety audit events

    # ============================================================
    # 1) INPUT SANITIZATION (moderation + jailbreak detection)
    # ============================================================
    def sanitize_input(self, text: str) -> str:
        """Moderation + jailbreak detection"""

        moderation = client.moderations.create(
            model="omni-moderation-latest",
            input=text,
        ).results[0]

        cat = moderation.categories

        high_risk = (
            cat.violence
            or cat.violence_graphic
            or cat.self_harm
            or cat.self_harm_intent
            or cat.self_harm_planning
            or cat.hate
            or cat.hate_threatening
            or cat.harassment
            or cat.sexual
            or cat.sexual_minors
            or moderation.flagged
        )

        if high_risk:
            self._log("input_block", "Moderation flagged unsafe content", {"input": text})
            return "⚠️ Input rejected for safety."

        # ---- Jailbreak / Prompt injection signals ----
        signals = [
            "ignore previous",
            "override system",
            "disable safety",
            "forget instructions",
            "you are now",
            "system:",
        ]
        lowered = text.lower()

        if any(sig in lowered for sig in signals):
            self._log("input_block", "Prompt injection detected", {"input": text})
            return "⚠️ Unsafe instruction detected."

        return text

    # ============================================================
    # 2) TOOL SAFETY VALIDATION
    # ============================================================
    def validate_tool(self, tool_name: str) -> None:
        """
        Ensures the planner is not invoking unauthorized or dangerous tools.
        Called inside ExecutorAgent before execution.
        """

        allowed = self.allowed_tools.get(tool_name, False)

        if not allowed:
            self._log("tool_block", "Disallowed tool selected", {"tool": tool_name})
            raise ValueError(f"❌ Tool not allowed for safety: {tool_name}")

    # ============================================================
    # 3) RAG SAFETY / HALLUCINATION CHECK
    # ============================================================
    def validate_rag(self, query: str, docs: List[str], answer: str) -> bool:
        """
        Prevents hallucinations by checking that the answer is grounded in docs.
        Light-weight real-world check:
            - ensure answer uses keywords from docs
            - ensure docs are not empty
        """
        if not docs:
            self._log("rag_warning", "RAG returned empty docs", {"query": query})
            return False

        lowered_answer = answer.lower()
        grounded_terms = 0

        for d in docs:
            for token in d.lower().split():
                if len(token) > 4 and token in lowered_answer:
                    grounded_terms += 1

        if grounded_terms == 0:
            self._log("rag_unverified", "Possible hallucination", {"query": query, "answer": answer, "docs": docs})
            return False

        return True

    # ============================================================
    # 4) OUTPUT SAFETY (LLM-based supervisor)
    # ============================================================
    def sanitize_output(self, user_input: str, output: str) -> str:
        """LLM safety supervisor + PII scrubber."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a safety and compliance supervisor.
                    Evaluate whether the assistant's answer is safe, grounded, non-harmful,
                    non-biased, and contains no private personal data.

                    Return strict JSON:
                    {
                        "safe": true|false,
                        "reason": "string",
                        "final": "string"
                    }
                    """,
                },
                {"role": "user", "content": json.dumps({"user": user_input, "answer": output})},
            ],
        )

        result = json.loads(resp.choices[0].message.content)

        # Supervisor says unsafe
        if not result.get("safe", False):
            reason = result.get("reason", "Content not permitted.")
            safe_final = result.get("final", "The system cannot provide this answer.")
            self._log("output_block", reason, {"unsafe_output": output})
            return f"⚠️ Output blocked for safety.\nReason: {reason}\n\n{safe_final}"

        # Safe → apply PII scrubbing
        cleaned = scrub_output(result["final"])
        return cleaned

    # ============================================================
    # 5) JSON SCHEMA ENFORCEMENT
    # ============================================================
    def enforce_json(self, content: str) -> Dict[str, Any]:
        """
        Ensures that planner responses conform to strict JSON schemas.
        If invalid → fail early.
        """

        try:
            parsed = json.loads(content)
            self._log("json_valid", "Valid JSON from model", {})
            return parsed
        except Exception as e:
            self._log("json_invalid", f"Invalid JSON: {e}", {"content": content})
            raise ValueError(f"JSON schema violation: {e}")

    # ============================================================
    # INTERNAL TRACE LOGGER
    # ============================================================
    def _log(self, event: str, message: str, data: Dict[str, Any]):
        """
        Every safety-relevant event is logged with timestamp.
        Useful for auditing, debugging, or replaying agent flows.
        """
        self.traces.append({"timestamp": time.time(), "event": event, "message": message, "data": data})

    def get_traces(self) -> List[Dict[str, Any]]:
        """Return full safety logs for debugging or audit."""
        return self.traces
