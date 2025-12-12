# agent/multi_agent.py

from __future__ import annotations

from agent.executors.ExecutorAgent_V24 import ExecutorAgent
from agent.memory.episodic import EpisodicMemory
from agent.memory.long_term import LongTermMemory
from agent.memory.short_term import ShortTermMemory
from agent.observability.slack_notifier import SlackNotifier
from agent.planners.planner_agent import PlannerAgent
from agent.planners.planner_m16 import WorkflowPlanner
from agent.verifiers.verifier_agent import VerifierAgent


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
        self.notifier = SlackNotifier()

        from agent.integrations.slack_client import SlackClient

        self.slack_client = SlackClient()

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
        from agent.safety_guardrails.safety_superviser import safety_supervisor
        from agent.safety_guardrails.sanitizer import sanitize_user_input

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
        try:
            steps = self.planner_agent.plan(user_input)
            self.notifier.planner_success(len(steps))
        except Exception as e:
            self.notifier.planner_failure(e)
            raise

        # --------------------------------------------------
        # 4️⃣ Executor: RAG + Tools execution
        # --------------------------------------------------
        try:
            workflow_results, draft = self.executor_agent.execute(user_input, steps)
            self.notifier.executor_success()
        except Exception as e:
            self.notifier.executor_failure(e)
            raise

        # ---------------------------------------------------------------------------
        # 5️⃣ Verifier: ensures correctness + final answer(Correct or Approcve Draft)
        # ---------------------------------------------------------------------------
        try:
            final = self.verifier_agent.verify(user_input, workflow_results, draft)
            self.notifier.verifier_success()
        except Exception as e:
            self.notifier.verifier_failure(e)
            raise

        # --------------------------------------------------------------------------
        # 6️⃣ Safety Supervisor: (POST-PROCESS SAFETY)final safety check on the final answer
        # --------------------------------------------------------------------------
        safe_report = safety_supervisor(user_input, final)
        if not safe_report.get("safe", False):
            reason = safe_report.get("reason", "Unknown reason")
            self.notifier.final_blocked(reason)
            return safe_report.get("final")

        self.notifier.final_success()
        return safe_report["final"]

        # safe_report = {
        # "safe": true/false,
        # "reason": "....",
        # "final": 2final sanitizied output"
        # # }

        # if not safe_report.get("safe", False):
        #     # Superviser blocks unsafe or ungrounded final answers
        #     reason = safe_report.get("reason", "Content flagged.")
        #     safe_final = safe_report.get("final", "The system cannot provide this answer safely.")
        #     return f"⚠️  Output blocked by safety supervisor:\nReason: {reason}\n{safe_final}"

        # #Otherwise output is safe
        # return safe_report["final"]

        # # Optional: Store episodic record
        # self.epi.store(user_input, safe_final)
