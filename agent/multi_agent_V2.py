# agent/multi_agent.py

from __future__ import annotations

import json

from typing import Any, List, Optional


from agent.config import client, OPENAI_MODEL
from agent.planners.planner_m16 import WorkflowPlanner
from agent.executors.ExecutorAgent_m19 import ExecutorAgent
from agent.verifiers.verifier_agent import VerifierAgent

from agent.memory.short_term import ShortTermMemory
from agent.memory.long_term import LongTermMemory
from agent.memory.episodic import EpisodicMemory


class PlannerAgent:
    """High-level planning agent - delegates to WorkflowPlanner."""

    def __init__(self, planner, stm, ltm, epi) -> None:
        self.planner = planner
        self.stm = stm
        self.ltm = ltm
        self.epi = epi

    def plan(self, user_input: str, memory_text: Optional[str] = None):
        
        if memory_text is None:

            # ---- 1. STM: short-term context ----
            stm_text = self.stm.as_text() or ""

            # ---- 2. Preferences ----
            prefs = self.ltm.all_prefs()
            prefs_text = json.dumps(prefs, ensure_ascii=False) if prefs else ""

            # ---- 3. LTM: semantic recall ----
            ltm_chunks = self.ltm.recall(user_input) or []
            ltm_text = "\n".join(ltm_chunks)

            # ---- 4. ETM: episodic recall ----
            epi_chunks = self.epi.retrieve_similar(user_input, k=3) or []
            epi_text = "\n".join(epi_chunks)

            # ---- Combine all memory sources ----
            memory_chunks = [
                stm_text,
                prefs_text,
                ltm_text,
                epi_text,
            ]

            memory_text = "\n".join(t for t in memory_chunks if t.strip())

        try:
            validated_steps = self.planner.plan(user_input, memory_text)
        except ValueError:
            validated_steps = [
                {
                    "action": "rag",
                    "tool_name": None,
                    "tool_args": {},
                    "rag_query": user_input,
                }
            ]

        return validated_steps



# --- Multi-Agent Orchestrator ---
class MultiAgentOrchestrator:
    """High-level orchestrator for Planner → Executor → Verifier."""

    def __init__(self) -> None:
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.epi = EpisodicMemory()

        self.planner = WorkflowPlanner()
        self.planner_agent = PlannerAgent(self.planner, self.stm, self.ltm, self.epi)
        self.executor_agent = ExecutorAgent(self.stm, self.ltm, self.epi)
        self.verifier_agent = VerifierAgent()
        
        
    def handle(self, user_input: str) -> str:

        """
        High-level orchestrator:
        1) Sanitize input
        2) Extract preferences
        3) Planner builds workflow
        4) Executor runs tools + RAG
        5) Verifier corrects draft
        6) Safety Supervisor final check
        """
        # --------------------------------------------------
        # IMPORT SAFETY AND PREFERENCE MODULES
        # --------------------------------------------------
        
        from agent.memory.preference_extractor import extract_preferences
        from agent.safety_guardrails.sanitizer import sanitize_user_input
        from agent.safety_guardrails.safety_superviser import safety_supervisor
        
        # --------------------------------------------------
        # 1️⃣ Safety gate (sanitizer)
        # --------------------------------------------------
        cleaned = sanitize_user_input(user_input)

        # If sanitizer returns a warning → stop the agent pipeline
        if cleaned.startswith("⚠️"):
            # Abort pipeline immediately on unsafe input
            return cleaned
        
        # Now the input is safe
        user_input = cleaned


        # --------------------------------------------------
        # 2️⃣ Extract global user preferences (only from safe input)
        # --------------------------------------------------
        extracted = extract_preferences(user_input)
        if extracted:
            for k, v in extracted.items():
                self.ltm.set_pref(k, v)


        # --------------------------------------------------
        # 3️⃣ Planner: builds workflow steps using all memory
        # --------------------------------------------------
        steps = self.planner_agent.plan(user_input)


        # --------------------------------------------------
        # 4️⃣ Executor: RAG + Tools execution
        # --------------------------------------------------
        workflow_results, draft = self.executor_agent.execute(user_input, steps)


        # ---------------------------------------------------------------------------
        # 5️⃣ Verifier: ensures correctness + final answer(Correct or Approcve Draft)
        # ---------------------------------------------------------------------------
        final = self.verifier_agent.verify(user_input, workflow_results, draft)

        # --------------------------------------------------------------------------
        # 6️⃣ Safety Supervisor: (POST-PROCESS SAFETY)final safety check on the final answer
        # --------------------------------------------------------------------------
        safe_report = safety_supervisor(user_input, final)
        # safe_report = {
        # "safe": true/false, 
        # "reason": "....", 
        # "final": 2final sanitizied output"
        # }  

        if not safe_report.get("safe", False):
            # Superviser blocks unsafe or ungrounded final answers
            reason = safe_report.get("reason", "Content flagged.")
            safe_final = safe_report.get("final", "The system cannot provide this answer safely.")
            return f"⚠️  Output blocked by safety supervisor:\nReason: {reason}\n{safe_final}"
        
        #Otherwise output is safe
        return safe_report["final"]

        
        # Optional: Store episodic record
        # self.epi.store(user_input, final)

