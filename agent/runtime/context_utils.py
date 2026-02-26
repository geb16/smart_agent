# agent/runtime/context_utils.py

import json

# This can be tuned as needed based on typical LLM input limits and
# expected memory sizes.
MAX_CONTEXT_CHARS = 12000


def truncate_json(payload, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Convert `payload` to pretty JSON (like json.dumps with indent=2 and
    ensure_ascii=False), but:
      - Never crashes on non-serializable objects
      - Always returns a string
      - Truncates to the last `max_chars` characters if too long.
    """

    # 1) Try normal JSON serialization (same as your original behaviour)
    try:
        raw_str = json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        # 2) Fallback: use default=str for non-serializable objects
        def default(o):
            try:
                return str(o)
            except Exception:
                return f"<non-serializable:{type(o).__name__}>"

        raw_str = json.dumps(payload, ensure_ascii=False, indent=2, default=default)

    # 3) Guarantee it's a string (paranoia guard)
    if not isinstance(raw_str, str):
        raw_str = str(raw_str)

    # 4) Truncate if necessary
    if len(raw_str) <= max_chars:
        return raw_str

    # Keep the tail (usually contains the most recent / relevant steps)
    return raw_str[-max_chars:]
