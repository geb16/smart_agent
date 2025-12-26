from __future__ import annotations

import asyncio
import collections.abc
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import redis  # pip install redis

from agent.config import client  # your OpenAI client (already used elsewhere)

# ------------------- Config -------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SEMANTIC_CACHE_KEY = "semantic_cache:v1"  # main hash for entries
SEMANTIC_INDEX_KEY = "semantic_cache:index"  # sorted set for similarity scores
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))
MAX_CANDIDATES = int(os.getenv("SEMANTIC_CACHE_MAX_CANDIDATES", "50"))


@dataclass
class SemanticCacheEntry:
    _id: str
    question: str
    answer: str
    embedding: List[float]


class RedisSemanticCache:
    """
    Semantic L2 cache using Redis + OpenAI embeddings.
    Design:
      - Store entries in a Redis HASH: id -> JSON(entry)
      - Maintain a separate LIST of IDs for iteration (for now).
      - Do similarity search in Python with cosine similarity over candidates.

    NOTE:
      This is intentionally simple & portable.
      You can later replace this with Redis Search / vector index if available.
    """

    def __init__(self, redis_url: str = REDIS_URL):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    # ------------------ Embeddings ------------------

    def _embed(self, text: str) -> List[float]:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    # ------------------ Public API ------------------

    def get_similar(self, question: str) -> Optional[str]:
        """
        Return an answer from cache if a semantically similar question exists.
        """
        question = question.strip()
        if not question:
            return None

        try:
            q_emb = self._embed(question)
        except Exception:
            # If embeddings fail, do not break pipeline
            return None

        # Fetch all ids we have stored (simple list key)
        ids_raw = self.redis.lrange(SEMANTIC_INDEX_KEY, -MAX_CANDIDATES, -1)
        if isinstance(ids_raw, collections.abc.Awaitable):
            ids: List[str] = asyncio.get_event_loop().run_until_complete(ids_raw)
        else:
            ids: List[str] = ids_raw
        if not ids:
            return None

        best_answer: Optional[str] = None
        best_score: float = -1.0

        for entry_id in ids:
            raw = self.redis.hget(SEMANTIC_CACHE_KEY, entry_id)
            if isinstance(raw, collections.abc.Awaitable):
                raw = asyncio.get_event_loop().run_until_complete(raw)
            if not raw:
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            emb = data.get("embedding")
            if not emb:
                continue

            sim = self._cosine_similarity(q_emb, emb)
            if sim > best_score:
                best_score = sim
                best_answer = data.get("answer")

        if best_score >= SIMILARITY_THRESHOLD:
            return best_answer

        return None

    def _set(self, question: str, answer: str) -> None:
        """
        Store a new semantic cache entry.
        """
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return

        try:
            emb = self._embed(question)
        except Exception:
            return

        entry_id = f"entry:{self._hash_key(question)}"
        entry = SemanticCacheEntry(
            _id=entry_id,
            question=question,
            answer=answer,
            embedding=emb,
        )
        payload = json.dumps(entry.__dict__)

        pipe = self.redis.pipeline()
        pipe.hset(SEMANTIC_CACHE_KEY, entry_id, payload)
        pipe.rpush(SEMANTIC_INDEX_KEY, entry_id)
        pipe.execute()

    # ------------------ Helpers ------------------

    @staticmethod
    def _hash_key(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return -1.0

        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return -1.0
        return dot / (math.sqrt(na) * math.sqrt(nb))
