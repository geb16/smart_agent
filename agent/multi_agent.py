# agent/multi_agent.py

from __future__ import annotations

import json
import inspect
from typing import Any, Dict, List, Tuple, Optional

from pathlib import Path

from agent.config import client, OPENAI_MODEL
from agent.tools import TOOL_REGISTRY
from agent.rag import RagRetriever
from agent.planner_m16 import WorkflowPlanner
from agent.ExecutorAgent_m19 import ExecutorAgent

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


# --- Executor Agent ---
class ExecutorAgent_1:
    """Executes planner steps using tools + RAG, and produces a draft answer."""

    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory, epi: EpisodicMemory) -> None:
        self.stm = stm
        self.ltm = ltm
        self.epi = epi
        self.rag = RagRetriever()

    # ---- internal single-step executor ----
    def _run_single_step(self, step: Dict[str, Any], user_input: str) -> Any:
        action = step.get("action")

        # ---- RAG ----
        if action == "rag":
            query: Optional[str] = step.get("rag_query") or user_input
            docs = self.rag.retrieve(query)
            return {
                "type": "rag",
                "query": query,
                "docs": docs,
            }

        # ---- TOOLS ----
        if action == "tool":
            name: Optional[str] = (step.get("tool_name") or "").strip()
            args: Dict[str, Any] = step.get("tool_args") or {}

            tool_fn = TOOL_REGISTRY.get(name)
            if not tool_fn:
                return {
                    "type": "tool_error",
                    "error": "UNKNOWN_TOOL",
                    "tool_name": name,
                    "available_tools": list(TOOL_REGISTRY.keys()),
                }

            # inject global prefs
            global_prefs = self.ltm.all_prefs()
            for k, v in global_prefs.items():
                args.setdefault(k, v)

            # filter args to function signature
            sig = inspect.signature(tool_fn)
            allowed = set(sig.parameters.keys())
            #🔖
            args = {k: v for k, v in args.items() if k in allowed}

            # tool-specific schema fixups
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

    def _synthesise_draft(self, user_input: str, step_results: List[Any]) -> str:
        combined = json.dumps(step_results, indent=2, ensure_ascii=False)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant generating a draft answer based ONLY on "
                        "the provided workflow results (tools + RAG). "
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

    def execute(self, user_input: str, steps: List[Dict[str, Any]]) -> Tuple[List[Any], str]:
        step_results: List[Any] = []

        for step in steps:
            result = self._run_single_step(step, user_input)

            if step.get("action") == "direct_answer":
                draft = self._synthesise_draft(user_input, step_results)

                # STM write
                self.stm.add(user_input, draft)

                # episodic
                self.epi.store_episode(
                    user_input,
                    {
                        "steps": steps,
                        "results": step_results,
                        "draft": draft,
                    },
                )
                return step_results, draft

            step_results.append(result)

        fallback = "Workflow could not complete a valid direct answer."
        self.stm.add(user_input, fallback)
        self.epi.store_episode(
            user_input, 
            {"steps": steps, 
             "results": step_results, 
             "draft": fallback,
             },
        )
        return step_results, fallback


# --- Verifier Agent ---
class VerifierAgent:
    """
    Robust verifier that ensures the final answer is FULLY grounded
    in tool outputs or RAG documents.
    """

    def verify(self, user_input: str, workflow_results: List[Any], draft_answer: str) -> str:

        payload = {
            "user_input": user_input,
            "workflow_results": workflow_results,
            "draft_answer": draft_answer,
        }

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict verification system for an AI agent.\n"
                        "Your response MUST be valid JSON.\n\n"

                        "You MUST evaluate whether the draft answer is completely grounded "
                        "in the provided workflow results.\n\n"

                        "GROUNDING RULES:\n"
                        "1. Any number in the final answer MUST appear in workflow_results.\n"
                        "2. Any factual statement MUST match content from RAG docs.\n"
                        "3. Any math result MUST match a tool_result.\n"
                        "4. NO new numbers, NO new facts, NO new interpretations.\n"
                        "5. If draft_answer violates grounding, you MUST correct it.\n\n"

                        "OUTPUT FORMAT:\n"
                        "{\n"
                        '  "approved": true|false,\n'
                        '  "final_answer": "string"\n'
                        "}\n"
                    ),
                },

                {
                    "role": "user",
                    "content": (
                        "Verify this agent output. Here is the verification payload (JSON):\n\n"
                        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
                    ),
                },
            ],
        )

        try:
            data = json.loads(resp.choices[0].message.content)
        except Exception:
            return draft_answer  # fallback on parse error

        approved = data.get("approved") 
        # approved can be True, False, or None
        final_answer = data.get("final_answer")

        # CASE 1: Verifier approved → return verifier's approved answer
        if approved and final_answer:
            return final_answer 
        # CASE 2: Verifier rejected but provided corrected answer → use corrected
        if (approved is False) and final_answer:
            # return corrected answer but warn user
            return (
                "Not in the tools or knowledge base. " 
                "Based on available information, here is corrected answer:\n" + final_answer
            )
        
        # CASE 3: Verifier retruned invalid payload → fallback to draft
        return "Unable to verify the answer using available workflow results." 

        

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
        from agent.memory.preference_extractor import extract_preferences
        #0️⃣ Extract preferences BEFORE planning
        extracted = extract_preferences(user_input)
        if extracted:
            for k, v in extracted.items():
                self.ltm.set_pref(k, v)

        # 1️⃣ Planner: build workflow using STM + LTM + prefs
        # 🔖 UPDATED: let PlannerAgent build memory_text internally
        steps = self.planner_agent.plan(user_input)
        # integrate LTM prefs into planning context

        # 2️⃣ Executor: run tools + RAG
        workflow_results, draft = self.executor_agent.execute(user_input, steps)
        
        # 3️⃣ Verifier: approve or correct
        final = self.verifier_agent.verify(user_input, workflow_results, draft)
        
        # Optional future extension: store final + verifier_status in EpisodicMemory here)
       # self.epi.store(user_input, final)   
        
        return final



# 🔖🈁🔴 For later use

# --- Suggestion Agent ---
class SuggestionAgent:
    """Generates next-sentence suggestions based on STM, LTM, and episodic memory."""

    def __init__(self, stm, ltm, epi):
        self.stm = stm
        self.ltm = ltm
        self.epi = epi

    def suggest(self, user_input: str) -> str:
        """
        Returns a short suggestion based on similar past queries or preferences.
        """
        past = self.stm.as_text()

        similar = self.epi.retrieve_similar(user_input, k=1)
        ltm_facts = self.ltm.recall(user_input, k=1)

        prompt = f"""
        User typed an incomplete query: {user_input}

        Recent conversation:
        {past}

        Similar queries from episodic memory:
        {similar}

        Relevant long-term memory:
        {ltm_facts}

        Suggest the next likely thing the user wants to ask.
        Keep the suggestion short, not intrusive, and relevant.
        Do NOT complete the whole query, just offer a subtle suggestion.
        """

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "system", "content": prompt}]
        )

        return resp.choices[0].message.content.strip()
