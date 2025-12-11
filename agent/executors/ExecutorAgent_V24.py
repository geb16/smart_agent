# agent/executors/ExecutorAgent_V24.py

from __future__ import annotations

import json
import inspect
import asyncio
from typing import Any, Dict, List, Tuple, Optional

from agent.config import client, OPENAI_MODEL
from agent.tools import TOOL_REGISTRY
from agent.rag import RagRetriever

from agent.memory.short_term import ShortTermMemory
from agent.memory.long_term import LongTermMemory
from agent.memory.episodic import EpisodicMemory


class ExecutorAgent:
    """Executes planner steps using tools + RAG and synthesizes a draft answer."""

    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory, epi: EpisodicMemory) -> None:
        self.stm = stm
        self.ltm = ltm
        self.epi = epi
        self.rag = RagRetriever()

    # -------------------------------------------------------
    # 🔹 Parallel Tool Execution (Async worker threads)
    # -------------------------------------------------------
    async def _run_tools_parallel(self, tool_names: List[str], base_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        global_prefs = self.ltm.all_prefs() or {}

        async def run_single_tool(name: str) -> Dict[str, Any]:
            tool_fn = TOOL_REGISTRY.get(name)
            if not tool_fn:
                return {
                    "type": "tool_error",
                    "tool_name": name,
                    "error": "UNKNOWN_TOOL",
                    "available_tools": list(TOOL_REGISTRY.keys()),
                }

            args = dict(base_args)

            for k, v in global_prefs.items():
                args.setdefault(k, v)

            sig = inspect.signature(tool_fn)
            allowed = set(sig.parameters.keys())
            args = {k: v for k, v in args.items() if k in allowed}

            # Tool-specific normalization
            if name == "tool_calculate_compound_interest":
                if "rate" in args and isinstance(args["rate"], (int, float)) and args["rate"] > 1:
                    args["rate"] /= 100.0
                if "time" in args and "years" not in args:
                    args["years"] = args.pop("time")
                args.setdefault("times_compounded", 1)

            def execute_tool():
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
                        "input": args,
                        "exception": str(e),
                    }

            return await asyncio.to_thread(execute_tool)

        tasks = [run_single_tool(n) for n in tool_names]
        return await asyncio.gather(*tasks)

    # -------------------------------------------------------
    # 🔹 Execute a single planner step (RAG / Tool / Final)
    # -------------------------------------------------------
    def _run_single_step(self, step: Dict[str, Any], user_input: str) -> Any:
        action = step.get("action")

        # RAG Retrieval
        if action == "rag":
            query: Optional[str] = step.get("rag_query") or user_input
            docs = self.rag.retrieve(query)
            return {"type": "rag", "query": query, "docs": docs}

        # TOOL execution (single or batch)
        if action == "tool":
            tool_name_field = step.get("tool_name")
            args = step.get("tool_args") or {}

            # Batch tool execution
            if isinstance(tool_name_field, list):
                try:
                    results = asyncio.run(
                        self._run_tools_parallel(tool_name_field, args)
                    )
                except RuntimeError:
                    # Fallback for already-running event loop
                    results = []
                    for name in tool_name_field:
                        sub_step = {
                            "action": "tool",
                            "tool_name": name,
                            "tool_args": dict(args),
                        }
                        results.append(self._run_single_step(sub_step, user_input))

                return {"type": "tool_batch_result", "tools": tool_name_field, "results": results}

            # Single tool call
            name = (tool_name_field or "").strip()
            tool_fn = TOOL_REGISTRY.get(name)

            if not tool_fn:
                return {
                    "type": "tool_error",
                    "tool_name": name,
                    "error": "UNKNOWN_TOOL",
                    "available_tools": list(TOOL_REGISTRY.keys()),
                }

            # Merge global prefs
            global_prefs = self.ltm.all_prefs()
            for k, v in global_prefs.items():
                args.setdefault(k, v)

            sig = inspect.signature(tool_fn)
            allowed = set(sig.parameters.keys())
            args = {k: v for k, v in args.items() if k in allowed}

            # normalize compound interest args
            if name == "tool_calculate_compound_interest":
                if "rate" in args and isinstance(args["rate"], (int, float)) and args["rate"] > 1:
                    args["rate"] /= 100.0
                if "time" in args and "years" not in args:
                    args["years"] = args.pop("time")
                args.setdefault("times_compounded", 1)

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
                    "input": args,
                    "exception": str(e),
                }

        if action == "direct_answer":
            return {"type": "final"}

        return {"type": "error", "message": f"Unknown action: {action}"}

    # -------------------------------------------------------
    # 🔹 Synthesize a Draft Answer
    # -------------------------------------------------------
    def _synthesise_draft(self, user_input: str, step_results: List[Any]) -> str:
        combined = json.dumps(step_results, indent=2, ensure_ascii=False)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate a draft answer based ONLY on workflow results. "
                        "Do not invent facts or numbers not present in the results."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User question:\n{user_input}\n\nWorkflow results:\n{combined}",
                },
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    # -------------------------------------------------------
    # 🔹 Execute full workflow
    # -------------------------------------------------------
    def execute(self, user_input: str, steps: List[Dict[str, Any]]) -> Tuple[List[Any], str]:
        step_results: List[Any] = []

        for step in steps:
            result = self._run_single_step(step, user_input)

            if step.get("action") == "direct_answer":
                draft = self._synthesise_draft(user_input, step_results)

                self.stm.add(user_input, draft)
                self.epi.store_episode(
                    user_input,
                    {"steps": steps, "results": step_results, "draft": draft},
                )
                return step_results, draft

            step_results.append(result)

        fallback = "Workflow could not complete a valid direct answer."

        self.stm.add(user_input, fallback)
        self.epi.store_episode(
            user_input,
            {"steps": steps, "results": step_results, "draft": fallback},
        )

        return step_results, fallback
