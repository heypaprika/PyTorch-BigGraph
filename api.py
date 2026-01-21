import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from jobs import JobStore
from worker import worker_loop

app = FastAPI()
store = JobStore()
threading.Thread(target=worker_loop, args=(store,), daemon=True).start()

class RunRequest(BaseModel):
    input: Dict[str, Any]

@app.post("/run")
def run(req: RunRequest):
    job_id = store.create(req.input)
    return {"id": job_id, "status": "IN_QUEUE"}

@app.get("/status/{job_id}")
def status(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    now = time.time()
    delay_time = 0
    if job["started_at"] is not None:
        delay_time = int(job["started_at"] - job["created_at"])
    execution_time = 0
    if job["started_at"] is not None:
        end = job["finished_at"] or now
        execution_time = int(end - job["started_at"])

    # RunPod 스타일 응답 흉내
    resp = {
        "id": job["id"],
        "status": job["status"],
        "delayTime": delay_time,
        "executionTime": execution_time,
    }
    if job["status"] == "COMPLETED":
        resp["output"] = job["output"]
    if job["status"] == "FAILED":
        resp["error"] = job["error"]

    return resp