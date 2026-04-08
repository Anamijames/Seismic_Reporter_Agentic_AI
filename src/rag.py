"""RAG pipeline: local embeddings (sentence-transformers), FAISS store, retrieval and generation via Groq API."""
import os
import json
import time
from typing import List
from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Paths and config
FAISS_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

# Embedding model (local)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# MLflow setup
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_ENABLED = os.getenv("ENABLE_MLFLOW", "false").strip().lower() in ("1", "true", "yes", "on")


def _get_mlflow():
    """Import mlflow only when it is explicitly enabled.

    Streamlit Cloud can have dependency combinations where importing mlflow at
    module load time fails before the app even starts. Keeping this lazy avoids
    breaking the whole app when metrics tracking is disabled.
    """
    if not MLFLOW_ENABLED:
        return None
    try:
        import mlflow
    except Exception:
        return None
    if MLFLOW_URI:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
        except Exception:
            return None
    return mlflow

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Compute embeddings locally using sentence-transformers."""
    arr = _embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [a.tolist() for a in arr]

def build_index(docs: List[dict]):
    texts = [d["text"] for d in docs]
    start = time.time()
    embs = embed_texts(texts)
    dim = len(embs[0])
    index = faiss.IndexFlatL2(dim)
    arr = np.array(embs).astype("float32")
    index.add(arr)
    faiss.write_index(index, FAISS_PATH)
    # Save metadata
    meta_path = FAISS_PATH + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f)
    elapsed = time.time() - start
    # Log experiment only when explicitly enabled to avoid network delays.
    mlflow = _get_mlflow()
    if mlflow is not None:
        try:
            mlflow.set_experiment("rag_indexing")
            with mlflow.start_run(run_name="build_index"):
                mlflow.log_metric("num_docs", len(docs))
                mlflow.log_metric("index_build_seconds", elapsed)
                mlflow.log_artifact(meta_path, artifact_path="meta")
        except Exception:
            pass
    return FAISS_PATH

def load_index():
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError("FAISS index not found; run build_index first")
    index = faiss.read_index(FAISS_PATH)
    meta = json.load(open(FAISS_PATH + ".meta.json", "r", encoding="utf-8"))
    return index, meta


def generate_with_groq(model: str, prompt: str, max_tokens: int = 128) -> str:
    """Call Groq Chat Completions API and return text output."""
    api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing or empty in .env")

    endpoint = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1") + "/chat/completions"
    timeout = int(os.getenv("GROQ_TIMEOUT", "60"))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for earthquake and geospatial Q&A."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.2")),
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Groq API returned no choices")
    message = choices[0].get("message", {})
    return (message.get("content") or "").strip()

def query_rag(question: str, k=3, ollama_model: str = None, max_tokens: int = 128):
    idx, meta = load_index()
    start = time.time()
    q_emb = np.array(embed_texts([question])).astype("float32")
    _, I = idx.search(q_emb, k)
    hits = [meta[i] for i in I[0] if i < len(meta)]
    # Build prompt with retrieved contexts
    context = "\n---\n".join([h["text"] for h in hits])
    prompt = (
        "You are an assistant that answers questions using the provided context. "
        f"Context:\n{context}\nQuestion: {question}\nAnswer:"
    )
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    answer = generate_with_groq(model, prompt, max_tokens=max_tokens)

    qtime = time.time() - start
    mlflow = _get_mlflow()
    if mlflow is not None:
        try:
            mlflow.set_experiment("rag_queries")
            with mlflow.start_run(run_name="query"):
                mlflow.log_metric("query_seconds", qtime)
                mlflow.log_metric("k", len(hits))
                mlflow.set_tag("model", model)
        except Exception:
            pass
    return {"answer": answer, "sources": hits}


if __name__ == "__main__":
    print("This module provides build_index() and query_rag() using Groq and sentence-transformers")
