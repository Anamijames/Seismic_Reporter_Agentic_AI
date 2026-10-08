import os
from src import ingest


def test_ingest_runs():
    # Run a lightweight ingest for 1 day (small)
    path, n = ingest.run_ingest(days=1)
    assert os.path.exists(path)
    assert n >= 0


def test_to_documents_adds_normalized_time_fields():
    geo = {
        "features": [
            {
                "id": "abc",
                "properties": {
                    "place": "Sample place",
                    "mag": 4.2,
                    "time": 1704067200000,  # 2024-01-01T00:00:00Z
                    "url": "https://example.test",
                    "felt": None,
                    "tsunami": 0,
                },
                "geometry": {"coordinates": [1.0, 2.0, 3.0]},
            }
        ]
    }

    docs = ingest.to_documents(geo)
    assert len(docs) == 1
    doc = docs[0]
    assert doc["meta"]["event_year"] == 2024
    assert doc["meta"]["event_time_utc"] == "2024-01-01T00:00:00Z"
    assert "event_year: 2024" in doc["text"]
