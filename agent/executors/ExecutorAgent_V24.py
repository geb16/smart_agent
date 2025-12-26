# agent/executors/ExecutorAgent_V24.py

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.config import OPENAI_MODEL, client
from agent.memory.episodic import EpisodicMemory
from agent.memory.long_term import LongTermMemory
from agent.memory.short_term import ShortTermMemory
from agent.rag import RagRetriever
from agent.runtime.context_utils import truncate_json
from agent.tools import TOOL_REGISTRY


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
            safe_query: str = query if query is not None else ""
            docs = self.rag.retrieve(safe_query)
            return {"type": "rag", "query": safe_query, "docs": docs}

        # TOOL execution (single or batch)
        if action == "tool":
            tool_name = step.get("tool_name")
            args = step.get("tool_args") or {}

            # Batch tool execution
            if isinstance(tool_name, list):
                try:
                    results = asyncio.run(self._run_tools_parallel(tool_name, args))
                except RuntimeError:
                    # Fallback for already-running event loop
                    results = []
                    for name in tool_name:
                        sub_step = {
                            "action": "tool",
                            "tool_name": name,
                            "tool_args": dict(args),
                        }
                        results.append(self._run_single_step(sub_step, user_input))

                return {"type": "tool_batch_result", "tools": tool_name, "results": results}

            # Single tool call
            name = (tool_name or "").strip()
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

    def _synthesise_draft(
        self, user_input: str, step_results: List[Any], *, stream: bool = False, on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        If stream=True:
          - use OpenAI streaming to build the full answer
          - call on_token(chunk) for every text delta, if provided
        Otherwise:
          - single non-streaming call (current behaviour).
        """

        # combined = json.dumps(step_results, indent=2, ensure_ascii=False)
        # ↑↓Adding in runtime truncation to avoid exceeding token limits

        combined = truncate_json(step_results)
        if not stream:
            # existing non-streaming path

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
            content = resp.choices[0].message.content
            return content.strip() if content is not None else ""

        # ---- streaming path: stream tokens and accumulate full answer ----
        chunks: List[str] = []
        completion_stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate a draft answer based ONLY on workflow results. " "Do not invent facts or numbers not present in the results."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User question:\n{user_input}\n\nWorkflow results:\n{combined}",
                },
            ],
            max_tokens=300,
            stream=True,  # 👈 key flag from OpenAI docs
        )
        for chunk in completion_stream:
            choice = chunk.choices[0]
            delta = getattr(choice.delta, "content", None)

            if not delta:
                continue
            chunks.append(delta)
            if on_token:
                # push each toekn/segment to call (CLI/HTTP WebSocket)
                on_token(delta)
        return "".join(chunks).strip()

    # -------------------------------------------------------
    # 🔹 Execute full workflow (thread streaming option)
    # -------------------------------------------------------
    def execute(
        self, user_input: str, steps: List[Dict[str, Any]], *, stream: bool = False, on_token: Optional[Callable[[str], None]] = None
    ) -> Tuple[List[Any], str]:
        """
        Execut planner steps and return (step_results, draft_answer).
        IF stream = True, the draft will be streamed token-by-token via the on_token(chunk) callback.
        """
        # 1) Short_term Memory instant answer check
        stm_answer = self.stm.get(user_input) if hasattr(self.stm, "get") else None
        if stm_answer:
            if stream and on_token:
                on_token(stm_answer)
            return [], stm_answer

        step_results: List[Any] = []

        for step in steps:
            result = self._run_single_step(step, user_input)

            if step.get("action") == "direct_answer":
                draft = self._synthesise_draft(user_input, step_results, stream=stream, on_token=on_token)

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
