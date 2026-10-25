import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from task_queue.worker import process_query
from task_queue.connection import client

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "server is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    job_id = str(uuid.uuid4())
    process_query(job_id, request.message)
    return {"status": "queued", "job_id": job_id}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = client.get(f"result:{job_id}")
    if result is None:
        raise HTTPException(status_code=202, detail="still processing")
    return {"job_id": job_id, "result": result}


