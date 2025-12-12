
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
