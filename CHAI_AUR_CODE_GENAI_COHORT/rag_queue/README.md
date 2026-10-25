# RAG Queue — Scalable RAG with Valkey Message Queue

A scalable RAG (Retrieval-Augmented Generation) API built with FastAPI, Valkey (Redis-compatible), and Qdrant. Uses a Producer/Consumer pattern so multiple users can query simultaneously without blocking each other.

---

## Architecture

```
User → POST /chat → FastAPI (Producer) → Valkey Queue → Worker (Consumer) → Qdrant + OpenAI
                                                                   ↓
User → GET /result/{job_id} ←──────────── result stored in Valkey
```

- **Producer** — receives the request, pushes it to the queue, returns a `job_id` immediately
- **Valkey** — message broker that decouples the API from the slow RAG work
- **Consumer** — picks up jobs from the queue, runs the RAG pipeline, stores the answer
- **Qdrant** — vector database that holds the embedded PDF chunks

---

## Prerequisites

- Docker Desktop running
- Python 3.10+
- OpenAI API key

---

## Setup

### 1. Copy environment variables

Copy the `.env` file from the `rag` folder into `rag_queue`:

```powershell
Copy-Item "..\rag\.env" ".env"
```

The `.env` must contain:

```
OPENAI_API_KEY=sk-...
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start Valkey and Qdrant

```powershell
docker compose up -d
```

Verify both containers are running:

```powershell
docker ps
```

You should see `rag_valkey` (port 6379) and `rag_qdrant` (port 6333).

---

## Running

You need **two terminals** open at the same time.

### Terminal 1 — FastAPI server

```powershell
python main.py
```

Server starts at `http://localhost:8000`.

### Terminal 2 — Worker (Consumer)

```powershell
python -c "from task_queue.worker import process; process()"
```

On first run the worker loads the PDF, chunks it, embeds it, and stores vectors in Qdrant. This takes about 1 minute. Subsequent restarts are instant because Qdrant persists the vectors in a Docker volume.

Expected output:

```
[Consumer] Loading RAG pipeline (first time — this may take a minute)...
Loaded 626 pages from '...PDF-Guide-Node-Andrew-Mead-v3.pdf'
Split into 1842 chunks ...
[Consumer] RAG pipeline ready.
[Consumer] Waiting for jobs...
```

---

## Querying

### Step 1 — Send a query

```powershell
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{"message": "how do I create a Node.js server?"}'
```

Response:

```json
{"status": "queued", "job_id": "abc-123"}
```

### Step 2 — Poll for the result

```powershell
curl http://localhost:8000/result/abc-123
```

While the worker is processing:

```json
{"detail": "still processing"}   ← HTTP 202, retry after a few seconds
```

Once done:

```json
{"job_id": "abc-123", "result": "To create a Node.js server..."}
```

---

## Scaling Workers

To handle more concurrent users, run more workers — Valkey distributes jobs automatically:

```powershell
# Terminal 2
python -c "from task_queue.worker import process; process()"

# Terminal 3 (second worker)
python -c "from task_queue.worker import process; process()"

# Terminal 4 (third worker)
python -c "from task_queue.worker import process; process()"
```

Each job goes to exactly one worker — no duplicates, no race conditions.

---

## Project Structure

```
rag_queue/
├── main.py                    # Uvicorn entry point
├── server.py                  # FastAPI routes (/chat, /result/{job_id})
├── docker-compose.yml         # Valkey + Qdrant containers
├── requirements.txt
├── .env                       # OPENAI_API_KEY (not committed)
└── task_queue/
    ├── connection.py          # Valkey client
    └── worker.py              # Producer and Consumer classes
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/chat` | Submit a query, returns `job_id` |
| GET | `/result/{job_id}` | Poll for the answer (202 if still processing) |
