# agent/planner.py

from __future__ import annotations

import json
from typing import Any, Dict, List

from agent.config import client, OPENAI_MODEL

# Add long term memory 
from agent.memory.long_term import LongTermMemory

# ---------------------------------------------------------
#   Constants: Valid Actions + Tools + System Prompt
VALID_ACTIONS = {"rag", "tool", "direct_answer"}
VALID_TOOLS = {
    "tool_add",
    "tool_subtract",
    "tool_divide",
    "tool_square_root",
    "tool_multiply",
    "tool_weather",
    "tool_slack_notify",
    "tool_calculate_compound_interest",
}
#🦺:Safety: explicitly allow only approved tools
ALLOWED_TOOLS = {
    "tool_add": True,
    "tool_subtract": True,
    "tool_divide": True,
    "tool_multiply": True,
    "tool_square_root": True,
    "tool_calculate_compound_interest": True,
    "tool_slack_notify": True,
    "tool_weather": True,
    # dangerous tools would be False or absent
}




WORKFLOW_SYSTEM_PROMPT = """
Return a JSON plan:

{
  "steps": [
    {
      "thought": "string (why this step is required)",
      "action": "rag" | "tool" | "direct_answer",
      "tool_name": string | null,
      "tool_args": {},
      "rag_query": string | null
    }
  ]
}

Rules:
- Each step MUST include a clear "thought".
-- Any expression containing numbers + math operators MUST use a math tool.
- “divided by”, “divide”, “/” → tool_divide
- “times”, “multiply”, “*” → tool_multiply
- “plus”, “add”, “+” → tool_add
- “minus”, “subtract”, “-” → tool_subtract
- “square root”, “sqrt” → tool_square_root

- ALWAYS extract numbers into tool_args.
- Example:
  User: “what is 6 divided by 4?”
  → 
  {
    "thought": "User asked a division question",
    "action": "tool",
    "tool_name": "tool_divide",
    "tool_args": { "a": 6, "b": 4 }
  }

- Weather → tool_weather with {"city": "..."}
- Any tool usage MUST use: {"action": "tool", "tool_name": "..."}
- Compound interest → 
  {"action": "tool", "tool_name": "tool_calculate_compound_interest",
   "tool_args": {"principal": float, "rate": decimal, "times_compounded": int, "years": float}}

   - Any request to "notify", "send message", "alert", "ping someone", 
  or "post update" → use:
  {
    "action": "tool",
    "tool_name": "tool_slack_notify",
    "tool_args": {
      "message": "string (the message to send)",
      "username": "optional display name"
    }
  }

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
        self.ltm = LongTermMemory() # For future use in plan method
        # include it like this: plan(user_input, memory_text)
        # where memory_text includes LTM prefs + STM history

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

      
        # that means if action is a tool name, convert to action: tool, tool_name: <name>
        raw_action = s.get("action")
        # -----------------------------------------------------------------
        # 1️⃣ Normaize "tool_xxx" action → ("tool", tool_name)
        # -----------------------------------------------------------------
        if raw_action in VALID_TOOLS:
            s["tool_name"] = raw_action
            s["action"] = "tool"
        
        # -----------------------------------------------------------------
        # 2️⃣ Re-read normalized action + validate
        # -----------------------------------------------------------------
        action = s.get("action") # ✅ RE-READ normalized action
        
        #✅ Now validate correctly
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action from planner: {action}")
        
        # ----------------------------------------------------------------
        # 3️⃣ 🦺 Tool safety checks(only if tool step)
        # ----------------------------------------------------------------
        if action == "tool":
            tool_name = s.get("tool_name")

            # ---- Ensure the planner provided a tool_name ---
            if not tool_name:
                raise ValueError("Planner prodced a tool step with no tool_name")
            # ---- Ensure tools exists in the system -----
            if tool_name not in VALID_TOOLS:
                raise ValueError(f"Unknown tool requested: {tool_name}")
            # ---- Ensure tool is explicitly allowed for safety -----
            if not ALLOWED_TOOLS.get(tool_name, False): 
                raise ValueError(f"Disallowed tool selected by planner: {tool_name}")

        base = {
            "thought": s.get("thought"), # This preserves reasoning for logging/debugging
            "action": action,
            "tool_name": s.get("tool_name"),
            "tool_args": s.get("tool_args") or {},
            "rag_query": s.get("rag_query"),
        }

        # ---- TOOL ENFORCEMENT ----
        if action == "tool":
            tool_name = base["tool_name"]

            # app must not crash on invalid tool usage
            # put in try catch in orchestrator instead

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
