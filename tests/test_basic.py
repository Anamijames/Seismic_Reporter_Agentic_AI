import os
from src import ingest


def test_ingest_runs():
    # Run a lightweight ingest for 1 day (small)
    path, n = ingest.run_ingest(days=1)
    assert os.path.exists(path)
    assert n >= 0
