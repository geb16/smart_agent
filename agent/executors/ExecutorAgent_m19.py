# agent/multi_agent.py

from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Optional, Tuple

from agent.config import OPENAI_MODEL, client
from agent.memory.episodic import EpisodicMemory
from agent.memory.long_term import LongTermMemory
from agent.memory.short_term import ShortTermMemory
from agent.rag import RagRetriever
from agent.tools import TOOL_REGISTRY


# --- Executor Agent ---
class ExecutorAgent:
    """
    Executes planner steps using tools + RAG, and (Module 19) adds:
    - adaptive retries
    - step repair
    - intelligent fallback
    """

    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory, epi: EpisodicMemory) -> None:
        self.stm = stm
        self.ltm = ltm
        self.epi = epi
        self.rag = RagRetriever()

    # ---------------------------------------------------------
    # Helper: Intelligent Repair for Failed Steps (Module 19)
    # ---------------------------------------------------------
    def _repair_step(
        self,
        user_input: str,
        failed_step: Dict[str, Any],
        error_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Ask LLM to repair a failed step:
        - Fix wrong tool args
        - Fix wrong tool name
        - Switch tool → RAG if needed
        - Or skip entirely
        """

        system_prompt = """
            You are a step-repair assistant for an AI agent.

            Fix ONLY the step that failed.
            Return STRICT JSON:

            {
            "action": "tool" | "rag" | "skip",
            "tool_name": string | null,
            "tool_args": {},
            "rag_query": string | null
            }

            Rules:
            - If tool args are wrong → correct them.
            - If tool name is wrong → fix it.
            - If tool cannot answer → switch to RAG.
            - If step unnecessary → "skip".
            - NEVER invent unsupported tools.
            - For RAG, rag_query must be explicit.
        """

        payload = {
            "user_input": user_input,
            "failed_step": failed_step,
            "error": error_result,
        }

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, indent=2, ensure_ascii=False)},
            ],
        )

        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("LLM response content is None and cannot be parsed as JSON.")
        repaired = json.loads(content)

        if repaired.get("action") == "skip":
            return None

        # Normalise into your internal step schema
        return {
            "action": repaired.get("action"),
            "tool_name": repaired.get("tool_name"),
            "tool_args": repaired.get("tool_args") or {},
            "rag_query": repaired.get("rag_query"),
        }

    # ---------------------------------------------------------
    # Single Step Execution
    # ---------------------------------------------------------
    def _run_single_step(self, step: Dict[str, Any], user_input: str) -> Any:
        action = step.get("action")

        # ---- RAG ----
        if action == "rag":
            query: Optional[str] = step.get("rag_query") or user_input
            if query is None:
                query = ""
            docs = self.rag.retrieve(query)
            return {
                "type": "rag",
                "query": query,
                "docs": docs,
            }

        # ---- TOOL EXECUTION ----
        if action == "tool":
            name: Optional[str] = (step.get("tool_name") or "").strip()
            args: Dict[str, Any] = step.get("tool_args") or {}

            tool_fn = TOOL_REGISTRY.get(name if name is not None else "")
            if not tool_fn:
                return {
                    "type": "tool_error",
                    "error": "UNKNOWN_TOOL",
                    "tool_name": name,
                    "available_tools": list(TOOL_REGISTRY.keys()),
                }

            # Inject LTM preferences
            for k, v in self.ltm.all_prefs().items():
                args.setdefault(k, v)

            # Enforce only valid args
            sig = inspect.signature(tool_fn)
            allowed = set(sig.parameters.keys())
            args = {k: v for k, v in args.items() if k in allowed}

            # Compound interest schema normalization
            if name == "tool_calculate_compound_interest":
                if "rate" in args and isinstance(args["rate"], (int, float)) and args["rate"] > 1:
                    args["rate"] /= 100.0
                if "time" in args and "years" not in args:
                    args["years"] = args.pop("time")
                args.setdefault("times_compounded", 1)

            try:
                out = tool_fn(**args)
                return {
                    "type": "tool_result",
                    "tool_name": name,
                    "input": args,
                    "output": out,
                    "status": "success",
                }
            except Exception as e:
                return {
                    "type": "tool_error",
                    "tool_name": name,
                    "input": args,
                    "exception": str(e),
                }

        # ---- DIRECT ANSWER ----
        if action == "direct_answer":
            return {"type": "final"}

        return {"type": "error", "message": f"Unknown action: {action}"}

    # ---------------------------------------------------------
    # Draft Synthesis from workflow results
    # ---------------------------------------------------------
    def _synthesise_draft(self, user_input: str, step_results: List[Any]) -> str:
        combined = json.dumps(step_results, indent=2, ensure_ascii=False)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": ("Generate a draft ONLY from workflow tool results and RAG." "Do NOT invent facts."),
                },
                {
                    "role": "user",
                    "content": f"User query:\n{user_input}\n\nWorkflow Results:\n{combined}",
                },
            ],
        )
        content = resp.choices[0].message.content
        if content is None:
            return ""
        return content.strip()

    # ---------------------------------------------------------
    # MAIN EXECUTION LOOP (Now Adaptive – Module 19)
    # ---------------------------------------------------------
    def execute(self, user_input: str, steps: List[Dict[str, Any]]) -> Tuple[List[Any], str]:
        step_results: List[Any] = []
        i = 0

        # Steps may grow during execution (due to repaired steps)
        while i < len(steps):
            step = steps[i]
            result = self._run_single_step(step, user_input)

            # -----------------------------------------------------
            # MODULE 19: Detect FAILED steps → Attempt Repair
            # -----------------------------------------------------
            needs_repair = False

            # TOOL FAILURE
            if result.get("type") == "tool_error":
                needs_repair = True

            # EMPTY RAG RESULT
            elif result.get("type") == "rag":
                docs = result.get("docs", {})
                matches = docs.get("matches") if isinstance(docs, dict) else None
                if not matches:
                    needs_repair = True

            # ---- Attempt repair ----
            if needs_repair:
                repaired = self._repair_step(user_input, step, result)

                if repaired:
                    # Insert repaired step AFTER current one
                    steps.insert(i + 1, repaired)

            # -----------------------------------------------------
            # Handle final answer
            # -----------------------------------------------------
            if step.get("action") == "direct_answer":
                draft = self._synthesise_draft(user_input, step_results)

                self.stm.add(user_input, draft)
                self.epi.store_episode(
                    user_input,
                    {"steps": steps, "results": step_results, "draft": draft},
                )
                return step_results, draft

            step_results.append(result)
            i += 1

        # ---------------------------------------------------------
        # Fallback if no direct_answer reached
        # ---------------------------------------------------------
        fallback = "Workflow could not complete a valid direct answer."
        self.stm.add(user_input, fallback)
        self.epi.store_episode(user_input, {"steps": steps, "results": step_results, "draft": fallback})
        return step_results, fallback
