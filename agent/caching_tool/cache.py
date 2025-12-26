# agent/cache.py
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CacheEntry:
    value: str
    created_at: float


class InMemoryAnswerCache:
    """
    L1 exact cache: (sanitized_input + stable prefs) -> final answer.
    Backed by a simple in-memory dict. Swap to Redis/ElastiCache in prod.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl = ttl_seconds
        self._store: Dict[str, CacheEntry] = {}

    def _make_key(self, user_input: str, prefs: Optional[dict]) -> str:
        payload = {
            "q": user_input.strip(),
            "prefs": prefs or {},
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, user_input: str, prefs: Optional[dict] = None) -> Optional[str]:
        key = self._make_key(user_input, prefs)
        entry = self._store.get(key)
        if not entry:
            return None
        if time.time() - entry.created_at > self.ttl:
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, user_input: str, prefs: Optional[dict], value: str) -> None:
        key = self._make_key(user_input, prefs)
        self._store[key] = CacheEntry(value=value, created_at=time.time())
