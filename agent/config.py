# file: level2/module17/smart_agent/agent/config.py
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# slack webhook url can also be set via environment variable
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Function to run steaming chat completions
# somewhere in your llm client wrapper, e.g. agent/config.py


def stream_answer(prompt: str):
    """
    Minimal streaming wrapper. Call this from Executor/Verifier
    and forward chunks to user.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    full_text_parts = []
    for chunk in response:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            full_text_parts.append(delta)
            # UI: send delta chunk to client here
            print(delta, end="", flush=True)  # CLI example

    return "".join(full_text_parts)
