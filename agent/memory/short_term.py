from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List


class ShortTermMemory:
    """
    Rolling working memory for the agent.
    Stores only the last N meaningful user–assistant exchanges,
    with optional relevance filtering and safe formatting.
    """

    def __init__(self, max_turns: int = 5):
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")

        self.max_turns = max_turns
        self.turns: List[Dict] = []

    # ============================================================
    # ADD ENTRY
    # ============================================================
    def add(self, user: str, agent: str):
        """Store a single short-term memory turn with timestamp."""
        self.turns.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user": str(user).strip(),
                "agent": str(agent).strip(),
            }
        )

        # Strict rolling buffer
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    # ============================================================
    # CLEAN HISTORY FOR PLANNER
    # ============================================================
    def _compress_text(self, text: str) -> str:
        """
        Remove extra whitespace and overly long reasoning.
        """
        text = re.sub(r"\s+", " ", text).strip()

        # Never allow model to see chain-of-thought or verbose explanations
        text = re.sub(r"(Thought:.*?)(User:|Assistant:|$)", "", text, flags=re.I | re.S)
        return text[:400]  # Hard cap for safety

    # ============================================================
    # RETRIEVE STM AS COMPACT HISTORY
    # ============================================================
    def as_text(self) -> str:
        """
        Planner receives a small, compressed history block.
        Must NEVER leak thoughts, internal reasoning, or JSON-breaking content.
        """
        if not self.turns:
            return ""

        formatted = []
        for t in self.turns:
            u = self._compress_text(t["user"])
            a = self._compress_text(t["agent"])
            formatted.append(f"User: {u}\nAssistant: {a}")

        # Two-line separation, clean and consistent
        return "\n\n".join(formatted)

    # ============================================================
    # RETRIEVE ONE TRUE MEMORY MATCH (Exact or near-exact)
    # ============================================================
    def get(self, user_query: str) -> str:
        """
        Return the most recent answer to the same or near-same user question.
        Used to skip the entire pipeline for repeated inputs.
        """
        user_query = user_query.strip().lower()
        if not user_query or not self.turns:
            return ""

        # 1️⃣ Exact match search (preferred, fastest)
        for t in reversed(self.turns):
            if t["user"].strip().lower() == user_query:
                return t["agent"]

        # 2️⃣ Soft match (syntactic similarity for minor variations)
        # e.g., "1+2-9", "1 + 2 - 9?", "calculate 1 + 2 - 9"
        for t in reversed(self.turns):
            saved = t["user"].strip().lower()
            if user_query in saved or saved in user_query:
                return t["agent"]

        return ""

    # ============================================================
    # CLEAR STM
    # ============================================================
    def clear(self):
        """Reset the STM completely."""
        self.turns.clear()
