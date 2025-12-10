# file: level2/module12/smart_agent/rag_prep.py
import os
import re
import tiktoken
import chromadb
from chromadb.config import Settings

from agent.config import client, EMBEDDING_MODEL

DATA_PATH = "data/documents.txt"
DB_DIR = "rag_db"
COLLECTION_NAME = "docs"


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    # Basic cleaning: normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50):
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap

    return chunks

#------------- Embedding and storing functions -------------


def embed_batch(texts, batch_size: int = 100):
    """
    Embed texts in batches with proper error handling.
    OpenAI embedding API has limits on batch size and input length.
    """
    if not texts:
        print("Warning: No texts to embed")
        return []
        
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Filter out empty texts and truncate very long ones
        batch = [text[:8000] for text in batch if text.strip()]
        
        if not batch:
            print(f"Skipping empty batch {i//batch_size + 1}")
            continue
            
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                encoding_format="float"
            )
            batch_embeddings = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"Embedded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {e}")
            # Add zero embeddings for failed batch to maintain index alignment
            embedding_dim = 1536  # text-embedding-3-small dimension
            batch_embeddings = [[0.0] * embedding_dim] * len(batch)
            all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found")
        return
    
    raw = load_text(DATA_PATH)
    if not raw.strip():
        print(f"Error: Data file {DATA_PATH} is empty")
        return
        
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned, max_tokens=300, overlap=50)
    
    if not chunks:
        print("Error: No chunks created from the text")
        return
    
    print(f"Created {len(chunks)} chunks")
    
    chroma_client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # recreate collection for a clean run
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(COLLECTION_NAME)

    embeddings = embed_batch(chunks)
    
    if not embeddings:
        print("Error: No embeddings created")
        return
        
    if len(embeddings) != len(chunks):
        print(f"Warning: Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
        # Truncate to match the shorter list
        min_len = min(len(chunks), len(embeddings))
        chunks = chunks[:min_len]
        embeddings = embeddings[:min_len]
    
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metadatas = [{"index": i, "source": DATA_PATH} for i in range(len(chunks))]

    try:
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"Successfully stored {len(chunks)} chunks in vector DB at {DB_DIR}")
    except Exception as e:
        print(f"Error adding to collection: {e}")
        return


if __name__ == "__main__":
    main()
