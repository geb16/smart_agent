
from agent.config import client, OPENAI_MODEL
from typing import Any, Dict
import json



# Safety Supervisor to check final answers
def safety_supervisor(user_input, final_answer):
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
You are a compliance and safety auditor.
Check if final_answer is safe, grounded, non-harmful, and free of PII.

Return STRICT JSON:
{
   "safe": true|false,
   "reason": "string",
   "final": "string"
}
"""
            },
            {
                "role": "user",
                "content": json.dumps({"user": user_input, "answer": final_answer})
            }
        ]
    )
    return json.loads(resp.choices[0].message.content)