# agent/agent_core.py

from __future__ import annotations

import json
import inspect
from typing import Any, Dict, List, Optional


from agent.config import client, OPENAI_MODEL
from agent.tools import TOOL_REGISTRY
from agent.rag import RagRetriever

from agent.memory.short_term import ShortTermMemory
from agent.memory.long_term import LongTermMemory
from agent.memory.episodic import EpisodicMemory
from agent.memory.preference_extractor import extract_preferences




class WorkflowAgent:
    def __init__(self, planner) -> None:
        self.planner = planner
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.epi = EpisodicMemory()
        self.rag = RagRetriever()

    # ----------- Single Step Executor ----------- #
    def _run_single_step(self, step: Dict[str, Any], user_input: str) -> Any:
        action = step.get("action")

        # ----------- RAG ----------- #
        if action == "rag":
            query: Optional[str] = step.get("rag_query") or user_input
            docs = self.rag.retrieve(query)
            return {
                "type": "rag",
                "query": query,
                "docs": docs,
            }

        # ----------- TOOLS ----------- #
        if action == "tool":
            name: Optional[str] = step.get("tool_name", "").strip()
            args: Dict[str, Any] = step.get("tool_args") or {}

            tool_fn = TOOL_REGISTRY.get(name)
            if not tool_fn:
                return {
                    "type": "tool_error",
                    "error": "UNKNOWN_TOOL",
                    "tool_name": name,
                    "available_tools": list(TOOL_REGISTRY.keys()),
                }

            # ✅ Inject persistent preferences
            global_prefs = self.ltm.all_prefs()
            for k, v in global_prefs.items():
                args.setdefault(k, v)

            # ✅ Tool argument filtering AFTER tool exists
            sig = inspect.signature(tool_fn)
            allowed = set(sig.parameters.keys())
            args = {k: v for k, v in args.items() if k in allowed}

            # ✅ Domain-specific sanitization
            if name == "tool_calculate_compound_interest":
                if "rate" in args and isinstance(args["rate"], (int, float)) and args["rate"] > 1:
                    args["rate"] /= 100.0

                if "time" in args and "years" not in args:
                    args["years"] = args.pop("time")

                args.setdefault("times_compounded", 1)

                required = {"principal", "rate", "times_compounded", "years"}
                missing = required - args.keys()
                if missing:
                    return {
                        "type": "tool_error",
                        "error": "TOOL_SCHEMA_ERROR",
                        "tool_name": name,
                        "missing_args": list(missing),
                        "received_args": args,
                    }

            try:
                output = tool_fn(**args)
                return {
                    "type": "tool_result",
                    "tool_name": name,
                    "input": args,
                    "output": output,
                    "status": "success",
                }

            except Exception as e:
                return {
                    "type": "tool_error",
                    "tool_name": name,
                    "exception": str(e),
                    "input": args,
                }

        # ----------- FINAL ----------- #
        if action == "direct_answer":
            return {"type": "final"}

        return {"type": "error", "message": f"Unknown action: {action}"}

    # ----------- Public Agent Entry ----------- #
    def handle(self, user_input: str) -> str:

        # ---- ✅ 0️⃣ Extract & Persist User Preferences ----
        if any(p in user_input.lower() for p in ["prefer", "default", "always", "from now on"]):
            new_prefs = extract_preferences(user_input)
            for k, v in new_prefs.items():
                self.ltm.set_pref(k, v)

        # ---- 1️⃣ Build planner memory context ----
        ltm_text = self.ltm.recall(user_input) or ""
        stm_text = self.stm.as_text() or ""


        memory_chunks = [chunk for chunk in [stm_text, ltm_text] if chunk.strip()]
        memory_text = "\n".join(memory_chunks)

        # ---- 2️⃣ Ask Planner for Structured Steps ----
        steps = self.planner.plan(user_input, memory_text)

        step_results: List[Any] = []

        # ---- 3️⃣ Execute Workflow Steps ----
        for step in steps:
            result = self._run_single_step(step, user_input)

            # ---- 4️⃣ Final Answer Synthesis ----
            if step.get("action") == "direct_answer":
                combined = json.dumps(step_results, indent=2, ensure_ascii=False)

                final = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You must generate the final user-facing answer using ONLY the "
                                "structured workflow results provided.\n\n"
                                "Rules:\n"
                                "- Do NOT invent facts.\n"
                                "- Do NOT assume missing information.\n"
                                "- If results contain errors, return the error explicitly.\n"
                                "- Be concise and user-friendly.\n"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"User question:\n{user_input}\n\n"
                                f"Workflow results:\n{combined}"
                            ),
                        },
                    ],
                    max_tokens=200,
                ).choices[0].message.content.strip()
              

                # ---- 5️⃣ Memory Writes (Aligned With Your APIs) ----
                self.stm.add(user_input, final)
                #🔖Risk 3
                self.epi.store_episode(
                    user_input, 
                    {
                        "user_input": user_input,
                        "steps": steps,
                        "results": step_results,
                        "final": final
                    }
                )

                return final

            step_results.append(result)

        # ---- 6️⃣ Fallback Safety ----
        fallback = "Workflow could not complete a valid direct answer."
        self.stm.add(user_input, fallback)
        self.epi.store_episode(user_input, step_results + [fallback])
        return fallback
