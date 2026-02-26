# Smart Agent

Production-oriented multi-agent workflow system with:
- Planner -> Executor -> Verifier orchestration
- Hybrid memory (short-term, long-term, episodic)
- RAG via local ChromaDB
- Tool execution (math, weather, finance, Slack)
- Multi-layer caching (STM, L1 in-memory, L2 Redis semantic)
- Safety checks before and after generation

## Repository Tree

```text
smart_agent/
|-- .github/
|   `-- workflows/
|       `-- python-package.yml
|-- agent/
|   |-- orchestrator.py
|   |-- config.py
|   |-- rag.py
|   |-- autonomy/
|   |-- caching_tool/
|   |-- executors/
|   |-- integrations/
|   |-- memory/
|   |-- observability/
|   |-- planners/
|   |-- runtime/
|   |-- safety_guardrails/
|   |-- tools/
|   |-- utilities/
|   `-- verifiers/
|-- data/
|   `-- documents.txt
|-- evaluation/
|   |-- __init__.py
|   |-- evaluator.py
|   |-- metrics.py
|   |-- test_cases.json
|   `-- logs/
|-- slack_server/
|   |-- slack_server.py
|   `-- mock_slack_server.py
|-- tests/
|   |-- conftest.py
|   `-- unit/
|       |-- test_cache.py
|       |-- test_context_utils.py
|       |-- test_evaluator_helpers.py
|       |-- test_finance_tool.py
|       |-- test_math_tools.py
|       |-- test_metrics.py
|       `-- test_short_term_memory.py
|-- memory_db/
|-- rag_db/
|-- main.py
|-- rag_prep.py
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Prerequisites

- Python: `pyproject.toml` currently declares `>=3.13`
- Redis (required by current orchestrator path)
- OpenAI API key
- Slack webhook URL (required by `SlackClient`)

Note: CI workflow currently runs lint/unit tests on Python `3.10` and `3.11`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in project root:

```dotenv
OPENAI_API_KEY=...
SLACK_WEBHOOK_URL=...
REDIS_URL=redis://localhost:6379/0
OPENWEATHER_API_KEY=...  # optional
EMBED_MODEL=text-embedding-3-small
SEMANTIC_CACHE_THRESHOLD=0.90
SEMANTIC_CACHE_MAX_CANDIDATES=50
```

## Prepare RAG Data

1. Put source content in `data/documents.txt`
2. Build vectors:

```powershell
python rag_prep.py
```

## Run

Start Redis (example):

```powershell
docker run --name smart-agent-redis -p 6379:6379 -d redis:latest
```

Start CLI agent:

```powershell
python main.py
```

## Testing

Run unit tests:

```powershell
python -m pytest -q tests/unit --maxfail=1
```

Run evaluator:

```powershell
python evaluation/evaluator.py
```

## Runtime Flow

1. Input sanitization/moderation
2. Preference extraction
3. Cache lookup (STM -> L1 -> L2)
4. Planning
5. Execution (tools/RAG)
6. Verification
7. Final safety check
8. Memory and cache updates

## Deployment Notes

- Treat as a stateful worker process (CLI-first, not API-first).
- Store secrets in a secret manager, not `.env` in source control.
- Use managed Redis and set `REDIS_URL`.
- Persist/volume-mount `rag_db/` and memory data for continuity.
- Add process supervision and centralized logging.
