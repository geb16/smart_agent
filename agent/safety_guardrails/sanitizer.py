# smart_agent/agent/safety_gardrails/sanitizer.py

from agent.config import client

def safety_classify_input(text: str):
    resp = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    return resp.results[0]   # ModerationResult


def sanitize_user_input(text: str) -> str:
    moderation = safety_classify_input(text)
    cat = moderation.categories

    # ----------------------------------------------------------
    #   Correct, up-to-date OpenAI moderation category fields
    # ----------------------------------------------------------
    high_risk = (
        cat.violence
        or cat.violence_graphic
        or cat.self_harm
        or cat.self_harm_intent
        or cat.self_harm_intent
        or cat.hate
        or cat.hate_threatening
        or cat.harassment
        or cat.sexual
        or cat.sexual_minors
        or moderation.flagged       # OpenAI's global indicator
    )

    if high_risk:
        return "⚠️  User input flagged as unsafe. Cannot proceed."

    # ----------------------------------------------------------
    #   Prompt-injection heuristics
    # ----------------------------------------------------------
    lowered = text.lower()
    injection_signals = [
        "ignore previous",
        "forget instructions",
        "override safety",
        "disable safety",
        "system:",
        "you are no longer",
    ]

    if any(sig in lowered for sig in injection_signals):
        return "⚠️  Unsafe instruction detected."

    return text
