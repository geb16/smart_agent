# agent/memory/episodic.py
from __future__ import annotations

import json
import uuid
from datetime import datetime

from agent.config import OPENAI_MODEL, client
from agent.memory.long_term import LongTermMemory


class EpisodicMemory:
    """
    Episodic memory stores traces of past completed workflows.
    """

    def __init__(self):
        # ALWAYS anchor inside memory_db/
        self.store = LongTermMemory(collection="episodic_memories")

    # ============================================================
    # STORE EPISODE
    # ============================================================

    def store_episode(self, user_input: str, episode: dict) -> dict:
        """
        Store a structured episodic memory record.

        episode may contain:
            - "steps": [...]
            - "final": "..."           # preferred
            - "draft": "..."           # fallback (ExecutorAgent)
            - "verifier_status": "approved" | "corrected" | "rejected" | "unknown"
            - "results": [...]         # raw workflow results
        """
        steps = episode.get("steps", [])

        tools = [s.get("tool_name") for s in steps if isinstance(s, dict) and s.get("action") == "tool"]

        # 🔁 UPDATED: accept either "final" or "draft"
        final_answer = episode.get("final") or episode.get("draft")

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "steps": steps,
            "results": episode.get("results", []),  # 🔧 NEW: keep raw results if present
            "final_answer": final_answer,
            "verifier_status": episode.get("verifier_status", "unknown"),
            "tools_used": tools,
        }

        summary_text = self._summarize(record)
        self.store.add(summary_text, metadata={"type": "episodic"})

        return record

    # ============================================================
    # SUMMARIZATION
    # ============================================================

    def _summarize(self, record: dict) -> str:
        prompt = f"""
            Summarize the episode below into one short factual sentence.

            Episode:
            {json.dumps(record, indent=2)}

            Rules:
            - One sentence.
            - No chain of thought.
            - No internal thoughts.
            - No mention of planner, verifier, tools, or steps.
            - Capture ONLY user intent and final factual result.
        """

        resp = client.chat.completions.create(model=OPENAI_MODEL, temperature=0.0, messages=[{"role": "system", "content": prompt}])

        return resp.choices[0].message.content.strip()

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieve_similar(self, query: str, k: int = 3):
        """Returns list[str] summaries of similar episodes"""
        return self.store.recall(query, k=k)

    def recall_recent(self, n: int = 3):
        """Reserved for future chronological memory tracking."""
        return []
