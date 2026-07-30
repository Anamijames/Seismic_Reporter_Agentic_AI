"""RAG pipeline: local embeddings (sentence-transformers), FAISS store, retrieval and generation via Groq API."""
import os
import json
import time
import threading
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
DOCS_PATH = os.getenv("DOCS_PATH", "./data/usgs_docs.jsonl")
INDEX_REFRESH_MINUTES = int(os.getenv("INDEX_REFRESH_MINUTES", "180"))
INDEX_REFRESH_DAYS = int(os.getenv("INDEX_REFRESH_DAYS", "1"))

_index_scheduler_started = False
_index_scheduler_lock = threading.Lock()

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


def _latest_index_mtime() -> float:
    paths = [FAISS_PATH, FAISS_PATH + ".meta.json", DOCS_PATH]
    mtimes = [os.path.getmtime(path) for path in paths if os.path.exists(path)]
    return max(mtimes) if mtimes else 0.0


def index_age_minutes() -> float | None:
    latest_mtime = _latest_index_mtime()
    if latest_mtime <= 0:
        return None
    return (time.time() - latest_mtime) / 60.0

def build_index(docs: List[dict]):
    if not docs:
        raise ValueError("No documents available to build the FAISS index")
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


def refresh_index(days: int = INDEX_REFRESH_DAYS):
    from src.ingest import fetch_usgs_past_days, to_documents, save_jsonl

    geojson = fetch_usgs_past_days(days)
    docs = to_documents(geojson)
    if not docs:
        raise RuntimeError("USGS feed returned no earthquake documents")
    save_jsonl(docs)
    build_index(docs)
    return {"refreshed": True, "doc_count": len(docs)}


def ensure_index_fresh(max_age_minutes: int = INDEX_REFRESH_MINUTES, days: int = INDEX_REFRESH_DAYS):
    age_minutes = index_age_minutes()
    if age_minutes is None:
        result = refresh_index(days=days)
        result.update({"reason": "missing", "age_minutes": None})
        return result
    if age_minutes >= max_age_minutes:
        result = refresh_index(days=days)
        result.update({"reason": f"stale ({age_minutes:.1f}m)", "age_minutes": age_minutes})
        return result
    return {"refreshed": False, "reason": "fresh", "age_minutes": age_minutes}


def start_index_refresh_scheduler(interval_minutes: int = INDEX_REFRESH_MINUTES, days: int = INDEX_REFRESH_DAYS):
    global _index_scheduler_started
    with _index_scheduler_lock:
        if _index_scheduler_started:
            return False
        _index_scheduler_started = True

    interval_seconds = max(60, int(interval_minutes * 60))

    def _loop():
        while True:
            try:
                ensure_index_fresh(max_age_minutes=interval_minutes, days=days)
            except Exception:
                pass
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="index-refresh-scheduler", daemon=True)
    thread.start()
    return True

def load_index():
    if not os.path.exists(FAISS_PATH):
        docs = []
        if os.path.exists(DOCS_PATH):
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
        if not docs:
            try:
                from src.ingest import fetch_usgs_past_days, to_documents, save_jsonl

                geojson = fetch_usgs_past_days(1)
                docs = to_documents(geojson)
                if docs:
                    save_jsonl(docs)
            except Exception:
                docs = []
        if docs:
            build_index(docs)
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(
                "FAISS index not found; provide data/usgs_docs.jsonl or run build_index first"
            )
    index = faiss.read_index(FAISS_PATH)
    with open(FAISS_PATH + ".meta.json", "r", encoding="utf-8") as meta_file:
        meta = json.load(meta_file)
    return index, meta


def generate_with_groq(model: str, prompt: str, max_tokens: int = 128) -> str:
    """Call Groq Chat Completions API and return text output."""
    api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing or empty (set in .env or Streamlit secrets)")

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

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc

    if resp.status_code >= 400:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Groq API returned no choices")
    message = choices[0].get("message", {})
    return (message.get("content") or "").strip()

def query_rag(question: str, k=3, ollama_model: str = None, max_tokens: int = 128):
    start = time.time()
    hits = []
    context = ""
    prompt_prefix = "You are an assistant that answers questions using the provided context. "
    error_message = None

    try:
        idx, meta = load_index()
        q_emb = np.array(embed_texts([question])).astype("float32")
        _, I = idx.search(q_emb, k)
        hits = select_hits(meta, I[0])
        context = "\n---\n".join([h["text"] for h in hits])
        prompt = prompt_prefix + f"Context:\n{context}\nQuestion: {question}\nAnswer:"
    except FileNotFoundError:
        prompt = (
            "You are an assistant that answers questions about earthquakes. "
            "No retrieval index is currently available, so answer directly and be concise. "
            f"Question: {question}\nAnswer:"
        )

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    try:
        answer = generate_with_groq(model, prompt, max_tokens=max_tokens)
    except (RuntimeError, ValueError, OSError) as exc:
        error_message = str(exc)
        if hits:
            answer = (
                "I found relevant earthquake records, but I could not generate a Groq response right now. "
                "Review the sources below for the retrieved context."
            )
        else:
            answer = (
                "I could not generate an answer right now. Please check the Groq API key or connectivity "
                "and try again."
            )

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
    result = {"answer": answer, "sources": hits}
    if error_message:
        result["error"] = error_message
    return result


def select_hits(meta: List[dict], row_indices) -> List[dict]:
    return [meta[int(i)] for i in row_indices if 0 <= int(i) < len(meta)]


if __name__ == "__main__":
    print("This module provides build_index() and query_rag() using Groq and sentence-transformers")
