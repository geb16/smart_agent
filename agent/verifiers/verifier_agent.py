# agent/verifiers/verifier_agent.py

from __future__ import annotations
import json
from typing import Any, List

from agent.config import client, OPENAI_MODEL

from agent.runtime.context_utils import truncate_json


# --- Verifier Agent ---
class VerifierAgent:
    """
    Robust verifier that ensures the final answer is FULLY grounded
    in tool outputs or RAG documents.
    """

    def verify(self, user_input: str, workflow_results: List[Any], draft_answer: str) -> str:

        # Full payload for grounding 
        payload = {
            "user_input": user_input,
            "workflow_results": workflow_results,
            "draft_answer": draft_answer,
        }
        # → core of Module 24.D
        # --------------------------------------------------
        # 24.D:🔖  - safe context truncation BEFORE sending to LLM
        #          - Prevent overflow
        # --------------------------------------------------
        trancated_payload_str = truncate_json(payload)
        
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

                        "Verify whether the draft answer is COMPLETELY grounded in  "
                        "the workflow results and RAG documents.\n\n"

                        "GROUNDING RULES:\n"
                        "1. Any number MUST appear in workflow_results.\n"
                        "2. Any factual statement MUST come from RAG docs.\n"
                        "3. Any math MUST match tool_result outputs.\n"
                        "4. NO new numbers, NO new facts, NO new interpretations.\n"
                        "5. If draft_answer is ungrounded, correct it.\n\n"

                        "Return strictly JSON formatted as:\n"
                        "{\n"
                        '  "approved": true|false,\n'
                        '  "final_answer": "string"\n'
                        "}\n"
                    ),
                },

                {
                    "role": "user",
                    "content": (
                        "Verification payload(JSON, truncated for safety):\n\n"
                        f"{trancated_payload_str}"
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
