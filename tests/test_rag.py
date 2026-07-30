import numpy as np

from src import rag


class DummyIndex:
    def __init__(self, rows):
        self.rows = np.array([rows], dtype=np.int64)

    def search(self, q_emb, k):
        distances = np.zeros((1, len(self.rows[0])), dtype=np.float32)
        return distances, self.rows


def test_query_rag_ignores_negative_faiss_hits(monkeypatch):
    meta = [
        {"id": "doc-0", "text": "first", "meta": {"place": "first"}},
        {"id": "doc-1", "text": "second", "meta": {"place": "second"}},
        {"id": "doc-2", "text": "third", "meta": {"place": "third"}},
    ]

    monkeypatch.setattr(rag, "load_index", lambda: (DummyIndex([-1, 1, 2]), meta))
    monkeypatch.setattr(rag, "embed_texts", lambda texts: [[0.0, 1.0]])
    monkeypatch.setattr(rag, "generate_with_groq", lambda model, prompt, max_tokens=128: "ok")

    result = rag.query_rag("test question", k=3, max_tokens=32)

    assert [hit["id"] for hit in result["sources"]] == ["doc-1", "doc-2"]
    assert "error" not in result