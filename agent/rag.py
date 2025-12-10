# agent/rag.py
import chromadb
from chromadb.config import Settings
from agent.config import client, EMBEDDING_MODEL
from pathlib import Path

# Root of the project(auto-detect based on this file's location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DB_DIR = PROJECT_ROOT / "smart_agent" / "rag_db"
COLLECTION_NAME = "docs"


class RagRetriever:
    """
    Agentic RAG retriever: 
    - deterministic
    - returns structured evidence for Verifier
    - supports semantic relevance filtering
    - compatible with the agentic workflow (Planner → Executor → Verifier)
    """

    def __init__(self, min_relevance: float = 0.4):
        self.min_relevance = min_relevance
        
        self._client = chromadb.PersistentClient(
            path=DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self._ensure_collection()

    # ----------------------------------------------------
    # DB utilities
    # ----------------------------------------------------
    def _ensure_collection(self):
        try:
            return self._client.get_collection(COLLECTION_NAME)
        except Exception:
            return self._client.create_collection(COLLECTION_NAME)

    # ----------------------------------------------------
    # Embedding
    # ----------------------------------------------------
    def _embed(self, text: str) -> list[float]:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,                    # <--- upgraded (no list wrapper)
            encoding_format="float",
        )
        return resp.data[0].embedding

    # ----------------------------------------------------
    # Retrieval (Agentic)
    # ----------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> dict:
        try:
            total_docs = self.collection.count()
        except Exception:
            self.collection = self._ensure_collection()
            total_docs = self.collection.count()

        if total_docs == 0:
            return {
                "query": query,
                "matches": [],
                "context": "Knowledge base is empty. Run rag_prep.py to populate rag_db."
            }

        # Encode query
        q_emb = self._embed(query)

        # Query vector store
        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        matches = []
        for text, meta, dist in zip(docs, metas, dists):

            # ensure metadata exists
            if meta is None:
                meta = {}

            # filter irrelevant chunks
            if dist > self.min_relevance:
                continue

            matches.append({
                "text": text,
                "metadata": meta,
                "distance": float(dist),
            })

        # If filtering removed everything → fallback to raw docs
        if not matches:
            return {
                "query": query,
                "matches": [],
                "context": f"No relevant information found for: {query}"
            }

        # Build context for LLM draft synthesis
        context_block = "\n\n".join(m["text"] for m in matches)

        return {
            "query": query,
            "matches": matches,
            "context": context_block
        }
