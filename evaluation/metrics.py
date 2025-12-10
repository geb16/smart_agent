def check_action(expected, actual):
    return 1 if expected == actual else 0


def check_tool(expected, step):
    if step.get("action") != "tool":
        return 0
    return 1 if step.get("tool_name") == expected else 0


def check_answer_contains(expected_phrase, final_answer):
    if expected_phrase.lower() in final_answer.lower():
        return 1
    return 0
"""Evaluation metrics for smart agent actions and answers."""