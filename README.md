# Smart Agent

Multi-agent workflow system for grounded responses using:

- Planner -> Executor -> Verifier orchestration
- Hybrid memory (short-term, long-term, episodic)
- RAG over local ChromaDB
- Tooling (math, weather, finance, Slack notify)
- Multi-layer caching (STM, in-memory L1, Redis semantic L2)
- Safety checks before and after generation

## Repository Tree

```text
smart_agent/
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
|   |-- evaluator.py
|   |-- metrics.py
|   |-- test_cases.json
|   `-- logs/
|-- slack_server/
|   |-- slack_server.py
|   `-- mock_slack_server.py
|-- memory_db/
|-- rag_db/
|-- main.py
|-- rag_prep.py
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Prerequisites

- Python 3.10+ (repo CI runs on 3.9-3.11; `pyproject.toml` currently says `>=3.13`)
- Redis (required by current orchestrator path)
- OpenAI API key
- Slack webhook URL (required by current `SlackClient` initialization)

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
OPENWEATHER_API_KEY=...  # optional, only for weather tool
EMBED_MODEL=text-embedding-3-small
SEMANTIC_CACHE_THRESHOLD=0.90
SEMANTIC_CACHE_MAX_CANDIDATES=50
```

## Prepare Knowledge Base (RAG)

1. Put source text in `data/documents.txt`
2. Build embeddings and Chroma DB:

```powershell
python rag_prep.py
```

This populates `rag_db/` used by `agent/rag.py`.

## Run

Start Redis first (example with Docker):

```powershell
docker run --name smart-agent-redis -p 6379:6379 -d redis:latest
```

Run the agent:

```powershell
python main.py
```

Run evaluation:

```powershell
python evaluation/evaluator.py
```

## Runtime Flow

For each request, the orchestrator performs:

1. Input sanitization/moderation
2. Preference extraction
3. Cache lookup (STM -> L1 -> L2)
4. Planning
5. Execution (tools/RAG)
6. Verification
7. Final safety check
8. Memory and cache updates

## Deployment Notes

- Treat this service as a stateful worker process (it is CLI-based, not HTTP API-first).
- Externalize secrets via your platform secret manager (never commit `.env`).
- Use managed Redis in production and set `REDIS_URL` accordingly.
- Persist or volume-mount `rag_db/` and memory store paths if you need continuity across restarts.
- Add process supervision (systemd/PM2/container restart policy) and central logging.
