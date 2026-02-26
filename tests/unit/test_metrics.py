from evaluation.metrics import check_action, check_answer_contains, check_tool


def test_check_action():
    assert check_action("tool", "tool") == 1
    assert check_action("tool", "rag") == 0


def test_check_tool():
    assert check_tool("tool_add", {"action": "tool", "tool_name": "tool_add"}) == 1
    assert check_tool("tool_add", {"action": "tool", "tool_name": "tool_subtract"}) == 0
    assert check_tool("tool_add", {"action": "rag", "tool_name": "tool_add"}) == 0


def test_check_answer_contains():
    assert check_answer_contains("refund", "The refund policy is 30 days.") == 1
    assert check_answer_contains("warranty", "The refund policy is 30 days.") == 0
