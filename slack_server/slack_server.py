# file: level2/module17/smart_agent/slack_server/mock_slack_server.py
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/mock-slack")
async def mock_slack(request: Request):
    payload = await request.json()
    print("📥 [MOCK SLACK] Incoming payload:", payload)
    return {"ok": True}


