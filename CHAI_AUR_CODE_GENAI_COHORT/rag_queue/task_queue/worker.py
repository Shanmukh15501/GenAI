import json
import sys
from pathlib import Path
from task_queue.connection import client

# Make the rag folder importable
RAG_DIR = str(Path(__file__).parent.parent.parent / "rag")
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from rag_pipeline import build_rag_pipeline, query_pipeline, generate_answer

QUEUE_KEY = "rag:requests"
RESULT_TTL = 300


class Producer:
    """Accepts a query from the API and pushes it onto the Valkey queue."""

    def __init__(self, queue_key: str = QUEUE_KEY):
        self.queue_key = queue_key

    def enqueue(self, job_id: str, query: str) -> None:
        payload = json.dumps({"job_id": job_id, "query": query})
        client.rpush(self.queue_key, payload)
        print(f"[Producer] Enqueued job {job_id}")


class Consumer:
    """Pops jobs from the Valkey queue, runs the RAG pipeline, stores the result."""

    def __init__(self, queue_key: str = QUEUE_KEY, result_ttl: int = RESULT_TTL):
        self.queue_key = queue_key
        self.result_ttl = result_ttl
        self.vector_store = None

    def _load_rag(self) -> None:
        if self.vector_store is None:
            print("[Consumer] Loading RAG pipeline (first time — this may take a minute)...")
            self.vector_store = build_rag_pipeline()
            print("[Consumer] RAG pipeline ready.")

    def _run_rag(self, query: str) -> str:
        results = query_pipeline(self.vector_store, query)
        return generate_answer(query, results)

    def start(self) -> None:
        self._load_rag()
        print("[Consumer] Waiting for jobs...")
        while True:
            _, raw = client.blpop(self.queue_key)
            payload = json.loads(raw)
            job_id = payload["job_id"]
            query = payload["query"]
            print(f"[Consumer] Processing job {job_id}: {query}")
            result = self._run_rag(query)
            client.setex(f"result:{job_id}", self.result_ttl, result)
            print(f"[Consumer] Job {job_id} done")


# Shared instances used by server.py and the worker entry point
producer = Producer()
consumer = Consumer()


def process_query(job_id: str, query: str) -> None:
    producer.enqueue(job_id, query)


def process() -> None:
    consumer.start()
