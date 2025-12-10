Smart Agent

Production-ready, modular workflow agent for planning and executing tasks. This package is organized for clarity and testability, with clean separation between core agent logic, planning, tools, evaluation, and configuration.

Package Structure
- smart_agent/
	- agent/
		- planner.py: Converts user requests + short-term memory into actionable workflow steps.
		- agent_core.py: Orchestrates planning, tool selection, and execution loop; exposes WorkflowAgent.
	- tools/: Concrete tool wrappers (APIs, filesystem, models).
	- memory/: Short-term and long-term memory abstractions; supports agent.stm.
	- evaluation/
		- evaluator.py: Batch evaluation runner for test cases.
		- metrics.py: Scoring utilities used by evaluator.
	- config/: Defaults and environment variable settings.

Installation
Recommended: install in editable mode inside a virtual environment.

```powershell
# From level2/module17
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Run
Use module execution so imports resolve consistently.

```powershell
# From level2/module17
python -m smart_agent.evaluation.evaluator
```

Alternatively, when running directly from smart_agent/:

```powershell
$env:PYTHONPATH = "E:\AWS\rag_finetune\level2\module17"
python smart_agent\evaluation\evaluator.py
```

Quick Start (Code)
```python
from smart_agent.agent.planner import WorkflowPlanner
from smart_agent.agent.agent_core import WorkflowAgent

planner = WorkflowPlanner()
agent = WorkflowAgent(planner)

answer = agent.handle("Summarize weekly progress and email the team")
print(answer)
```

Design Principles
- Modularity: Planner, agent loop, tools, memory, and evaluation are separate modules.
- Deterministic Planning: WorkflowPlanner.plan(input, memory) returns explicit steps with action, optional tool, and arguments.
- Tool Contracts: Each tool exposes a simple run(params) interface with clear inputs/outputs.
- Memory Abstractions: agent.stm (short-term memory) provides recent context for planning; extend with LTM for persistence.
- Evaluation-first: evaluation/test_cases.json and evaluation/evaluator.py provide reproducible scoring.

Evaluation
Place test cases in smart_agent/evaluation/test_cases.json:
```json
[
	{
		"id": 1,
		"input": "Find meeting notes and summarize",
		"expected_action": "summarize",
		"expected_tool": "notes.search",
		"expected_final_contains": "summary"
	}
]
```
Run the evaluator:
```powershell
python -m smart_agent.evaluation.evaluator
```
Results are appended to smart_agent/evaluation/logs/evaluation_log.jsonl.

Conventions
- Imports: Use absolute imports (from smart_agent.agent...) for reliability.
- Config: Read environment variables via dotenv or os.environ in config/.
- Logging: Prefer module-level logger = logging.getLogger(__name__).

Troubleshooting
- ModuleNotFoundError: Run with python -m smart_agent... or set PYTHONPATH to the module root.
- VS Code imports: Select the correct interpreter (the .venv) and ensure the workspace root matches the module root.
- Editable install: Use pip install -e . at the module root if pyproject.toml is present.




