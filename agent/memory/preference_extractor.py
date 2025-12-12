# preference_extractor.py

from __future__ import annotations
import json
from typing import Dict, Any, List

from agent.config import client, OPENAI_MODEL


SYSTEM_PROMPT = """
You extract ONLY long-term preferences that affect FUTURE interactions.

Correct mappings:
- "from now on, use fahrenheit" → {"key": "temperature_unit", "value": "fahrenheit"|"°F"}
- "use celsius by default" → {"key": "temperature_unit", "value": "celsius"|"°C"}
- "always answer in spanish" → {"key": "language", "value": "spanish"}
- "set my default city to london" → {"key": "default_city", "value": "london"}
- "I prefer metric units" → {"key": "measurement_system", "value": "metric"}

Return STRICT JSON:

{
  "preferences": [
    {"key": "...", "value": "...", "confidence": 0.95}
  ]
}

Rules:
- Extract ONLY preferences intended for future use.
- Ignore task-specific or temporary commands.
- Keys must be semantic identifiers: temperature_unit, default_city, currency, language, measurement_system.
- Confidence MUST be between 0 and 1.
- If none found, return: {"preferences": []}
"""



def normalize_key(k: str) -> str:
    return k.lower().strip().replace(" ", "_")


def extract_preferences(user_text: str, min_confidence: float = 0.7) -> Dict[str, str]:

    if not user_text.strip():
        return {}

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        )

        data = json.loads(resp.choices[0].message.content)

    except Exception:
        return {}

    extracted: Dict[str, str] = {}

    for pref in data.get("preferences", []):
        raw_key = pref.get("key")
        raw_value = pref.get("value")
        confidence = float(pref.get("confidence", 1.0))

        if not raw_key or raw_value is None:
            continue
        if confidence < min_confidence:
            continue

        key = normalize_key(raw_key)
        value = str(raw_value).strip()

        extracted[key] = value

    return extracted
