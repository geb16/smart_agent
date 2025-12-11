# agent/planner.py

from __future__ import annotations

import json
from typing import Any, Dict, List

from agent.config import client, OPENAI_MODEL

# Add long term memory 
from agent.memory.long_term import LongTermMemory


VALID_ACTIONS = {"rag", "tool", "direct_answer"}
VALID_TOOLS = {
    "tool_add",
    "tool_multiply",
    "tool_weather",
    "tool_calculate_compound_interest",
}

WORKFLOW_SYSTEM_PROMPT = """
Return a JSON plan:

{
  "steps": [
    {"action": "rag" | "tool" | "direct_answer", "tool_name": string | null, "tool_args": {}, "rag_query": string | null}
  ]
}

Rules:
- Math → tool_add or tool_multiply
- Weather → tool_weather with {"city": "..."}
- Any tool usage MUST use: {"action": "tool", "tool_name": "..."}
- Compound interest → 
  {"action": "tool", "tool_name": "tool_calculate_compound_interest",
   "tool_args": {"principal": float, "rate": decimal, "times_compounded": int, "years": float}}

- Any question about:
  returns, refunds, orders, shipping, delivery, cancellations, replacements,
  payments, invoices, customer support, complaints, warranties, policies
  → MUST use {"action": "rag"}

- RAG must ALWAYS include a meaningful rag_query rewritten from the user question.
- TOOL must ALWAYS include correct tool_name + tool_args
- Weather must NEVER use "location"
- Final step → direct_answer
- If the question could require business, product, or policy knowledge and no tool applies → use RAG first
- NO extra text, JSON only
"""


class WorkflowPlanner:
    
    def __init__(self):
        self.ltm = LongTermMemory() # For future use if needed

    def plan(self, user_input: str, memory_text: str) -> List[Dict[str, Any]]:

        # ✅ Load persistent user preferences
        prefs = self.ltm.all_prefs()
        messages = [
            {"role": "system", "content": WORKFLOW_SYSTEM_PROMPT},
            # ✅ Preferences injected as policy (not chat history)
            {
                "role": "system",
                "content": f"User persistent preferences (ALWAYS respect):\n{json.dumps(prefs, indent=2)}"
            },
            
            {
                "role": "user",
                "content": f"History:\n{memory_text}\n\nUser: {user_input}",
            },
        ]

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            data = json.loads(resp.choices[0].message.content)
            raw_steps = data.get("steps", [])

        except Exception as e:
            raise RuntimeError(f"[Planner LLM Failure] {e}")

        validated_steps = [self._validate_and_norm(step) for step in raw_steps]

        # ✅ Enforce final answer step
        if not validated_steps or validated_steps[-1]["action"] != "direct_answer":
            validated_steps.append(
                {
                    "action": "direct_answer",
                    "tool_name": None,
                    "tool_args": {},
                    "rag_query": None,
                }
            )

        # ---------- SUPPORT QUERY FAIL-SAFE (RUNTIME OVERRIDE) ----------
        support_keywords = {
            "return", "refund", "order", "delivery", "shipping",
            "cancel", "complaint", "replace", "replacement",
            "warranty", "payment", "invoice", "support"
        }

        if any(k in user_input.lower() for k in support_keywords):
            if not any(step["action"] == "rag" for step in validated_steps):
                validated_steps.insert(0, {
                    "action": "rag",
                    "tool_name": None,
                    "tool_args": {},
                    "rag_query": user_input
                })


        return validated_steps

        

    def _validate_and_norm(self, s: Dict[str, Any]) -> Dict[str, Any]:

        action = s.get("action")
        if s.get("action") in VALID_TOOLS:
            s["tool_name"] = s["action"]
            s["action"] = "tool"

        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action from planner: {action}")

        base = {
            "action": action,
            "tool_name": s.get("tool_name"),
            "tool_args": s.get("tool_args") or {},
            "rag_query": s.get("rag_query"),
        }

        # ---- TOOL ENFORCEMENT ----
        if action == "tool":
            tool_name = base["tool_name"]

            if tool_name not in VALID_TOOLS:
                raise ValueError(f"Invalid tool selected by planner: {tool_name}")

            base["rag_query"] = None

        # ---- RAG ENFORCEMENT ----
        elif action == "rag":
            if not base["rag_query"]:
                raise ValueError("RAG action requires explicit rag_query")

            base["tool_name"] = None
            base["tool_args"] = {}

        # ---- DIRECT ANSWER ----
        elif action == "direct_answer":
            base["tool_name"] = None
            base["tool_args"] = {}
            base["rag_query"] = None

        return base
