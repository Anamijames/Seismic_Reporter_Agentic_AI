# Seismic Reporter

Seismic Reporter is a Retrieval-Augmented Generation (RAG) application that answers questions about recent earthquake activity using raw USGS data.

It is designed as a practical AI project and demonstrates end-to-end skills in:
- real-world data ingestion from public APIs
- embedding and vector retrieval
- LLM integration using Groq API
- interactive UI development with Streamlit
- local testing and containerized execution

## What this project does

1. Fetches raw earthquake data from the USGS feed.
2. Converts raw records into normalized text documents.
3. Builds embeddings with sentence-transformers.
4. Stores vectors in FAISS for similarity search.
5. Retrieves relevant context for a user query.
6. Sends grounded context to Groq for final answer generation.
7. Displays answers and source snippets in a responsive chat UI.

## Tech stack

- Python 3.10+
- Streamlit
- sentence-transformers
- FAISS (faiss-cpu)
- Groq Chat Completions API
- pytest
- Docker + Docker Compose

## Project structure

```text
app/
	streamlit_app.py         # Streamlit UI
src/
	ingest.py                # USGS data ingestion + document conversion
	rag.py                   # Embeddings, retrieval, Groq generation
data/
	usgs_docs.jsonl          # Raw-to-processed docs
	faiss_index              # Vector index file
	faiss_index.meta.json    # Source metadata for retrieval hits
tests/
	test_basic.py            # Basic ingestion test
```

## Quick start (local)

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create environment file

Windows PowerShell:

```powershell
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

### 4. Set required env values

At minimum, configure:

```env
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama-3.1-8b-instant
ENABLE_MLFLOW=false
```

### 5. Ingest and index data

```bash
python -c "from src.ingest import run_ingest; run_ingest(1)"
python -c "from src import ingest as ig, rag as r; geo=ig.fetch_usgs_past_days(1); docs=ig.to_documents(geo); r.build_index(docs)"
```

### 6. Run the app

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

## Run tests

```bash
pytest -q
```

## Docker run

```bash
docker compose up --build
```

Services:
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000

Windows note:
If you get an error mentioning `//./pipe/dockerDesktopLinuxEngine`, Docker Desktop is not running. Start Docker Desktop first, then run compose again.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | - | Groq API authentication key |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model name |
| `FAISS_INDEX_PATH` | No | `./data/faiss_index` | Local FAISS index path |
| `RAG_RETRIEVAL_K` | No | `3` | Number of retrieved chunks |
| `RAG_MAX_TOKENS` | No | `180` | Max answer tokens |
| `ENABLE_MLFLOW` | No | `false` | Enables MLflow logging when true |
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5000` | Tracking server URI |
| `AUTHOR_NAME` | No | `Anami James A` | Footer author name in UI |
| `GITHUB_URL` | No | `https://github.com/` | Footer GitHub link in UI |

## Common issues

### 1. Slow answers
- Cause: MLflow tracking connection delays.
- Fix: keep `ENABLE_MLFLOW=false` unless tracking server is running.

### 2. Groq 401 invalid key
- Cause: wrong key value or placeholder key still in `.env`.
- Fix: update `GROQ_API_KEY` and restart Streamlit.

### 3. FAISS index not found
- Cause: query before ingest/index step.
- Fix: run ingestion + index commands in Quick start step 5.

## Resume/portfolio talking points

- Built a domain-specific RAG assistant over raw public data.
- Integrated vector retrieval with source-grounded responses.
- Used Groq API for low-latency generation.
- Designed a responsive Streamlit UI with persistent chat history.
- Added test coverage and containerized run support.







