from evaluation.evaluator import _first_step_or_default


def test_first_step_or_default_uses_first_step():
    first = {"action": "tool", "tool_name": "tool_add", "tool_args": {"a": 1, "b": 2}, "rag_query": None}
    assert _first_step_or_default([first, {"action": "direct_answer"}]) == first


def test_first_step_or_default_fallback():
    out = _first_step_or_default([])
    assert out["action"] == "unknown"
    assert out["tool_name"] is None
