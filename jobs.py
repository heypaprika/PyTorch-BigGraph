import threading
import uuid
import time
from typing import Dict, Optional, Any

class JobStore:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.running = False

    def create(self, payload: dict) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "payload": payload,
                "status": "IN_QUEUE",
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "output": None,
                "error": None,
            }
        return job_id

    def can_run(self) -> bool:
        with self.lock:
            return not self.running

    def start(self, job_id: str):
        with self.lock:
            self.running = True
            self.jobs[job_id]["status"] = "RUNNING"
            self.jobs[job_id]["started_at"] = time.time()

    def finish(self, job_id: str, output: Any):
        with self.lock:
            self.jobs[job_id]["status"] = "COMPLETED"
            self.jobs[job_id]["output"] = output
            self.jobs[job_id]["finished_at"] = time.time()
            self.running = False

    def fail(self, job_id: str, error: str):
        with self.lock:
            self.jobs[job_id]["status"] = "FAILED"
            self.jobs[job_id]["error"] = error
            self.jobs[job_id]["finished_at"] = time.time()
            self.running = False

    def get(self, job_id: str) -> Optional[dict]:
        return self.jobs.get(job_id)