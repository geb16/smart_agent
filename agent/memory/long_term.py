import json
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings

from agent.config import EMBEDDING_MODEL, client

# ---------------- MEMORY ROOT -----------------------
BASE_DIR = Path(__file__).resolve().parent
MEMORY_ROOT = BASE_DIR / "memory_db"
PREF_FILE = BASE_DIR / "user_prefs.json"
# ----------------------------------------------------


class LongTermMemory:
    """
    Hybrid long-term memory:
    - Vector DB for semantic recall
    - JSON store for persistent user preferences
    """

    def __init__(self, path: str = None, collection: str = "memories"):
        # Always anchor inside memory_db
        db_path = Path(path) if path else MEMORY_ROOT
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.client.get_collection(collection)
        except Exception:
            self.collection = self.client.create_collection(collection)

        self.prefs = self._load_prefs()

    # ---------------- Vector Memory -------------------

    def embed(self, text: str):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return resp.data[0].embedding

    def add(self, text: str, metadata=None):
        emb = self.embed(text)

        final_meta = {"timestamp": datetime.utcnow().isoformat(), **(metadata or {})}

        self.collection.add(ids=[str(uuid.uuid4())], documents=[text], embeddings=[emb], metadatas=[final_meta])

    def recall(self, query: str, k=3, min_relevance=0.3) -> list:
        """Return list of semantically matching strings."""
        q_emb = self.embed(query)

        res = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "distances", "metadatas"])

        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        if not docs:
            return []

        return [docs[i] for i, dist in enumerate(dists) if dist <= min_relevance]

    # ---------------- Preferences Layer ----------------

    def _load_prefs(self) -> dict:
        if PREF_FILE.exists():
            try:
                return json.loads(PREF_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_prefs(self):
        PREF_FILE.write_text(json.dumps(self.prefs, indent=2))

    def normalize_key(self, key: str) -> str:
        return key.lower().strip().replace(" ", "_")

    def set_pref(self, key: str, value):
        key = self.normalize_key(key)
        self.prefs[key] = value
        self._save_prefs()

    def get_pref(self, key: str, default=None):
        return self.prefs.get(self.normalize_key(key), default)

    def all_prefs(self):
        return self.prefs.copy()


# ------------------------------------------------------------
# Clarification
# how to use add() and recall()
# ------------------------------------------------------------
# ltm = LongTermMemory()
# ltm.add("User prefers metric units.", metadata={"type": "preference"})
# results = ltm.recall("What measurement system does the user prefer?", k=2)
# print(results)
