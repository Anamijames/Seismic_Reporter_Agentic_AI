"""Ingest messy data from USGS Earthquake API and produce JSONL documents."""
import os
import json
import requests
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_usgs_past_days(days=7):
    # USGS feed: past N days (earthquake catalog query)
    endtime = datetime.utcnow()
    starttime = endtime - timedelta(days=days)
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        "?format=geojson"
        f"&starttime={starttime.isoformat()}"
        f"&endtime={endtime.isoformat()}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

def to_documents(geojson):
    docs = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        geometry = feat.get("geometry", {})
        text = (
            f"id: {feat.get('id')}\n"
            f"place: {props.get('place')}\n"
            f"mag: {props.get('mag')}\n"
            f"time: {props.get('time')}\n"
            f"url: {props.get('url')}\n"
            f"felt: {props.get('felt')}\n"
            f"tsunami: {props.get('tsunami')}\n"
            f"coords: {geometry.get('coordinates')}\n"
        )
        docs.append({"id": feat.get("id"), "text": text, "meta": props})
    return docs

def save_jsonl(docs, filename="usgs_docs.jsonl"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return path

def run_ingest(days=7):
    geo = fetch_usgs_past_days(days=days)
    docs = to_documents(geo)
    path = save_jsonl(docs)
    return path, len(docs)

if __name__ == "__main__":
    p, n = run_ingest(7)
    print(f"Wrote {n} docs to {p}")
