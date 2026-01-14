import os
import subprocess
import runpod

def handler(event):
    inp = event.get("input", {}) or {}

    # 예: 외부에서 config path/데이터 URI/출력 URI 받기
    config_path = inp.get("config_path", "configs/fb15k_config.py")
    out_dir = inp.get("out_dir", "/workspace/out")
    os.makedirs(out_dir, exist_ok=True)

    # 학습 실행
    cmd = ["python3", "-u", "train.py", "-c", config_path, "--out", out_dir]
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        return {"ok": False, "stderr": p.stderr[-4000:], "stdout": p.stdout[-4000:]}

    # TODO: out_dir의 best checkpoint를 S3로 업로드하고 URI 반환
    return {"ok": True, "out_dir": out_dir, "stdout_tail": p.stdout[-4000:]}

runpod.serverless.start({"handler": handler})