import json
from datetime import datetime

# Use absolute package imports to work both when run as a module and as a script
from agent.planner import WorkflowPlanner
from agent.agent_core import WorkflowAgent
from evaluation.metrics import (
    check_action,
    check_tool,
    check_answer_contains,
)


def run_evaluation():
    planner = WorkflowPlanner()
    agent = WorkflowAgent(planner)

    with open("evaluation/test_cases.json", "r") as f:
        test_cases = json.load(f)

    results = []

    for test in test_cases:
        user_input = test["input"]

        # Run planner only first
        memory_text = agent.stm.as_text()
        steps = planner.plan(user_input, memory_text)
        first_step = steps[0]

        # Run full agent
        final = agent.handle(user_input)

        record = {
            "id": test["id"],
            "input": user_input,
            "expected_action": test.get("expected_action"),
            "actual_action": first_step["action"],
            "action_score": check_action(test.get("expected_action"), first_step["action"]),
            "expected_tool": test.get("expected_tool"),
            "tool_score": check_tool(test.get("expected_tool"), first_step),
            "final_answer": final,
            "final_answer_score": check_answer_contains(test.get("expected_final_contains"), final),
            "timestamp": datetime.now().isoformat() 
        }

        results.append(record)

        # append to log file
        with open("evaluation/logs/evaluation_log.jsonl", "a") as log:
            log.write(json.dumps(record) + "\n")

    return results


if __name__ == "__main__":
    results = run_evaluation()
    for r in results:
        print(
            f"Test {r['id']}: action={r['action_score']}, "
            f"tool={r['tool_score']}, answer={r['final_answer_score']}"
        )

# How to run:
# - From module root:  python -m smart_agent.evaluation.evaluator
# - Or from package folder: ensure PYTHONPATH includes level2/module12, e.g.
#   $env:PYTHONPATH = "E:\AWS\rag_finetune\level2\module12"; python smart_agent\evaluation\evaluator.py