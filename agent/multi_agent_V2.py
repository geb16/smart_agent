# agent/multi_agent_V2.py

from __future__ import annotations

from agent.caching_tool.cache import InMemoryAnswerCache
from agent.caching_tool.semantic_cache import RedisSemanticCache  # 👈 New Redis feature
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

        # L2 cache for semantic retrieval
        self.semantic_cache = RedisSemanticCache()

        # NEW: Low-latency L1 cache for final answers
        self.answer_cache = InMemoryAnswerCache(ttl_seconds=3600)

    def handle(self, user_input: str, *, stream: bool = False, on_token=None) -> str:
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

        # 2.1 ------ STM (chepest , fastest) ------
        stm_answer = self.stm.get(user_input)
        if stm_answer:
            if stream and on_token:
                # Stream the cached answer token by token
                for token in stm_answer.split():
                    on_token(token + " ")
            self.notifier.cache_hit("STM")
            return stm_answer
        # 2.2------ L1 EXACT CACHE (In-Memory with Pref Identity) ------
        prefs_for_identity = None
        if "temperature_unit" in self.ltm.all_prefs():
            prefs_for_identity = {"temperature_unit": self.ltm.get_pref("temperature_unit")}

        cached_answer = self.answer_cache.get(user_input, prefs_for_identity)
        if cached_answer:
            if stream and on_token:
                # Stream the cached answer token by token
                for token in cached_answer.split():
                    on_token(token + " ")
            self.notifier.cache_hit("L1")
            return cached_answer

        # 2.3 ------ L2 Semantic CACHE (Redis/cosine) ------
        semantic_answer = self.semantic_cache.get_similar(user_input)
        if semantic_answer:
            if stream and on_token:
                # Stream the cached answer token by token
                for token in semantic_answer.split():
                    on_token(token + " ")
            self.notifier.cache_hit("L2")
            return semantic_answer
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
            workflow_results, draft = self.executor_agent.execute(user_input, steps, stream=stream, on_token=on_token)
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
            safe_final = safe_report.get("final", final)
            # do NOT cache blocked answers
            return safe_final
        final_answer = safe_report["final"]

        # Cache the final answer in L1 and L2 caches
        self.answer_cache._set(user_input, prefs_for_identity, final_answer)
        self.stm.add(user_input, final_answer)
        self.semantic_cache._set(user_input, final_answer)

        self.notifier.final_success()
        return final_answer
