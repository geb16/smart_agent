from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agent.executors.ExecutorAgent import ExecutorAgent
from agent.memory.episodic import EpisodicMemory
from agent.memory.long_term import LongTermMemory
from agent.memory.preference_extractor import extract_preferences
from agent.memory.short_term import ShortTermMemory
from agent.planners.planner_agent import PlannerAgent
from agent.planners.planner_m16 import WorkflowPlanner
from agent.safety_guardrails.safety_superviser import safety_supervisor
from agent.safety_guardrails.sanitizer import sanitize_user_input
from agent.verifiers.verifier_agent import VerifierAgent
from evaluation.metrics import check_action, check_answer_contains, check_tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = Path(__file__).resolve().parent
TEST_CASES_PATH = BASE_DIR / "test_cases.json"
LOG_PATH = BASE_DIR / "logs" / "evaluation_log.jsonl"


class NullEpisodicMemory(EpisodicMemory):
    """No-op episodic memory for evaluator runs to avoid side effects."""

    def retrieve_similar(self, query: str, k: int = 3) -> List[str]:
        return []

    def store_episode(self, user_input: str, episode: Dict[str, Any]) -> Dict[str, Any]:
        return {"stored": False, "user_input": user_input}


def _load_test_cases() -> List[Dict[str, Any]]:
    with TEST_CASES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_step_or_default(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if steps and isinstance(steps[0], dict):
        return steps[0]
    return {
        "action": "unknown",
        "tool_name": None,
        "tool_args": {},
        "rag_query": None,
    }


def run_evaluation() -> List[Dict[str, Any]]:
    stm = ShortTermMemory()
    ltm = LongTermMemory()
    epi = NullEpisodicMemory()

    planner = WorkflowPlanner()
    planner_agent = PlannerAgent(planner, stm, ltm, epi)
    executor = ExecutorAgent(stm, ltm, epi)
    verifier = VerifierAgent()

    test_cases = _load_test_cases()
    results: List[Dict[str, Any]] = []

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    for test in test_cases:
        user_input = str(test["input"])

        # 1) Input sanitization
        try:
            cleaned_input = sanitize_user_input(user_input)
        except Exception:
            cleaned_input = user_input

        workflow_results: List[Any] = []
        draft = ""

        # 2) Plan + execute + verify if input is safe
        if cleaned_input.startswith("⚠️"):
            steps = [
                {
                    "action": "blocked",
                    "tool_name": None,
                    "tool_args": {},
                    "rag_query": None,
                }
            ]
            final = cleaned_input
        else:
            # Extract and persist user preferences
            try:
                extracted = extract_preferences(cleaned_input)
            except Exception:
                extracted = {}
            for key, value in extracted.items():
                ltm.set_pref(key, value)

            try:
                steps = planner_agent.plan(cleaned_input, memory_text=stm.as_text())
            except Exception as exc:
                steps = [
                    {
                        "action": "rag",
                        "tool_name": None,
                        "tool_args": {},
                        "rag_query": cleaned_input,
                    }
                ]
                workflow_results.append({"type": "planning_error", "error": str(exc)})

            try:
                exec_results, draft = executor.execute(cleaned_input, steps, stream=False)
                workflow_results.extend(exec_results)
            except Exception as exc:
                draft = f"Execution error: {exc}"
                workflow_results.append({"type": "execution_error", "error": str(exc)})

            try:
                verified = verifier.verify(cleaned_input, workflow_results, draft)
            except Exception:
                verified = draft

            try:
                safe_report = safety_supervisor(cleaned_input, verified)
                if isinstance(safe_report, dict):
                    final = safe_report.get("final", verified)
                else:
                    final = verified
            except Exception:
                final = verified

            stm.add(cleaned_input, final)

        first_step = _first_step_or_default(steps)
        expected_final_contains = str(test.get("expected_final_contains") or "")

        record = {
            "id": test["id"],
            "input": user_input,
            "expected_action": test.get("expected_action"),
            "actual_action": first_step.get("action"),
            "action_score": check_action(
                test.get("expected_action"),
                first_step.get("action"),
            ),
            "expected_tool": test.get("expected_tool"),
            "tool_score": check_tool(test.get("expected_tool"), first_step),
            "final_answer": final,
            "final_answer_score": check_answer_contains(expected_final_contains, final),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        results.append(record)
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

    return results


if __name__ == "__main__":
    output = run_evaluation()
    for row in output:
        print(f"Test {row['id']}: action={row['action_score']}, " f"tool={row['tool_score']}, answer={row['final_answer_score']}")
