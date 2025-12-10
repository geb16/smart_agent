
from agent.config import client, OPENAI_MODEL
from typing import Any, Dict

def safety_classify_input(text: str) -> Dict[str, Any]:
    resp = client.moderations.create(
        model="omni-moderation-latest", 
        input=text
    )
    return resp.results[0]


def sanitize_user_input(text: str) -> str:
    classification = safety_classify_input(text)

    if classification["violence"] or classification["self_harm"]:
        return "User input flagged: request unsafe. Cannot proceed."

    # Basic prompt injection defense
    if "ignore previous" in text.lower() or "override" in text.lower():
        return "Unsafe instruction detected."

    return text