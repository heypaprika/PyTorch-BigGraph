import time
from handler import handler
from jobs import JobStore

def worker_loop(store: JobStore):
    while True:
        time.sleep(1)

        if not store.can_run():
            continue

        # 가장 먼저 들어온 IN_QUEUE job 하나 실행
        queued = [j for j in store.jobs.values() if j["status"] == "IN_QUEUE"]
        if not queued:
            continue

        queued.sort(key=lambda x: x["created_at"])
        job = queued[0]
        job_id = job["id"]

        try:
            store.start(job_id)
            # RunPod serverless handler 형식 그대로 호출
            output = handler({"input": job["payload"]})
            store.finish(job_id, output)
        except Exception as e:
            store.fail(job_id, str(e))