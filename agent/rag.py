# agent/rag_v2.py

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings

from agent.config import EMBEDDING_MODEL, client

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = (PROJECT_ROOT / "rag_db").resolve()
COLLECTION_NAME = "docs"


# ----------------------------------------------------
# Embedding (raw)
# ----------------------------------------------------
def _embed_raw(text: str) -> List[float]:
    """
    Low-level embedding call to OpenAI.
    NOTE: Do not call directly from outside, use _embed() so caching works.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,  # single string is fine with new OpenAI client
        encoding_format="float",
    )
    return resp.data[0].embedding


class RagRetriever:
    """
    Agentic RAG retriever (v2):
    - Uses persistent ChromaDB built by rag_prep.py
    - Cached embeddings to reduce latency + cost
    - Distance → similarity conversion with a relevance threshold
    - Structured output compatible with Planner / Executor / Verifier
    """

    def __init__(self, min_relevance: float = 0.4) -> None:
        """
        :param min_relevance: minimum similarity in [0,1]
                              0.4 is a good default; tune per project.
        """
        self.min_relevance = min_relevance

        if not DB_DIR.exists():
            DB_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"[RAG INFO] Created DB directory at {DB_DIR}")
        self._client = chromadb.PersistentClient(
            path=str(DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self._ensure_collection()
        logger.info(f"[RAG LOADING] persistent DB directory: {DB_DIR}")

    # ----------------------------------------------------
    # DB utilities
    # ----------------------------------------------------
    def _ensure_collection(self):
        # to prevent silent swallowing of DB type mismatches
        if COLLECTION_NAME not in [c.name for c in self._client.list_collections()]:
            self._client.create_collection(COLLECTION_NAME)
        return self._client.get_collection(COLLECTION_NAME)

        # try:
        #     return self._client.get_collection(COLLECTION_NAME)
        # except Exception:
        #     # If missing, create an empty collection; prep script will fill it.
        #     return self._client.create_collection(COLLECTION_NAME)

    # ----------------------------------------------------
    # Cached embedding
    # ----------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=4096)
    def _embed_cached(text: str) -> Tuple[float, ...]:
        """
        Cached embedding result (tuple so LRU cache can hash it).
        Called only by _embed().
        """
        text = text.replace("\x00", " ")[:2000].strip()  # Truncate + clean null chars
        emb = _embed_raw(text)  # Call module helper
        # Convert list → tuple for cache key stability / immutability
        return tuple(emb)  # immutable for caching

    def _embed(self, text: str) -> List[float]:
        """
        Public embedding method used by retrieval.
        Returns a list but caches internally as tuple.
        """
        return list(self._embed_cached(text))

    # ----------------------------------------------------
    # Retrieval (Agentic)
    # ----------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve semantically relevant chunks for a query.

        Returns:
        {
            "query": str,
            "matches": [
                {
                    "text": str,
                    "metadata": dict,
                    "distance": float,
                    "similarity": float,  # 0..→..1, higher is better(0.85)
                },
                ...
            ],
            "context": "joined text of all selected chunks"
        }
        """

        # ----- Ensure DB exists / not empty -----
        try:
            total_docs = self.collection.count()
        except Exception:
            self.collection = self._ensure_collection()
            total_docs = self.collection.count()

        if total_docs == 0:
            return {
                "query": query,
                "matches": [],
                "context": ("Knowledge base is empty. " "Run rag_prep.py to populate rag_db."),
            }

        # ----- 1. Get cached query embedding -----
        q_emb = self._embed(query)

        # ----- 2. Query ChromaDB -----
        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs_raw = result.get("documents")
        metas_raw = result.get("metadatas")
        dists_raw = result.get("distances")

        docs = docs_raw[0] if docs_raw and len(docs_raw) > 0 else []
        metas = metas_raw[0] if metas_raw and len(metas_raw) > 0 else []
        dists = dists_raw[0] if dists_raw and len(dists_raw) > 0 else []

        if not docs:
            # Chroma found nothing for some reason
            return {
                "query": query,
                "matches": [],
                "context": f"No relevant information found for: {query}",
            }

        # ----- 3. Convert distance → similarity and filter -----
        matches: List[Dict[str, Any]] = []

        for text, meta, dist in zip(docs, metas, dists):
            if meta is None:
                meta = {}

            # For cosine metric, smaller distance = more similar.
            # Approximate similarity in [0,1].
            # If Chroma uses cosine distance in [0,2], this still behaves well.
            similarity = 1.0 - float(dist)
            if similarity < 0.0:
                similarity = 0.0
            if similarity > 1.0:
                similarity = 1.0

            # Filter by minimum similarity threshold
            if similarity < self.min_relevance:
                continue

            matches.append(
                {
                    "text": text,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity": similarity,
                }
            )

        # ----- 4. Fallback: if everything got filtered out, return raw top_k -----
        if not matches:
            # Low-confidence fallback – keep everything Chroma gave us
            fallback_matches = []
            for text, meta, dist in zip(docs, metas, dists):
                if meta is None:
                    meta = {}
                similarity = 1.0 - float(dist)
                if similarity < 0.0:
                    similarity = 0.0
                if similarity > 1.0:
                    similarity = 1.0

                fallback_matches.append(
                    {
                        "text": text,
                        "metadata": meta,
                        "distance": float(dist),
                        "similarity": similarity,
                    }
                )

            context_block = "\n\n".join(m["text"] for m in fallback_matches)

            return {
                "query": query,
                "matches": fallback_matches,
                "context": context_block,
            }

        # ----- 5. Build context block from filtered matches -----
        context_block = "\n\n".join(m["text"] for m in matches)

        return {
            "query": query,
            "matches": matches,
            "context": context_block,
        }


# if __name__ == "__main__":
#     # Simple manual test
#     retriever = RagRetriever(min_relevance=0.4)
#     q = "return policy?"
#     out = retriever.retrieve(q, top_k=5)
#     print(out)
