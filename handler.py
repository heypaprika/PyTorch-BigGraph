import os
import subprocess
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import runpod


def parse_s3_uri(s3_uri: str):
    # s3://bucket/prefix
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI (missing bucket): {s3_uri}")
    # prefix는 비어도 허용(버킷 루트)
    return bucket, prefix


def get_region():
    # boto3 기본 리전 탐색은 가능하지만,
    # S3는 리전 불일치/미지정 시 이슈가 생기기 쉬워 명시 권장
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        # 버킷이 ap-southeast-1 등 특정 리전에 있다면 여기 기본값을 박아도 됨
        raise RuntimeError("Missing AWS_REGION (or AWS_DEFAULT_REGION) env var in Runpod.")
    return region


def make_s3_client():
    # Env creds는 boto3가 자동으로 사용
    region = get_region()
    return boto3.client("s3", region_name=region)


def s3_download_prefix(s3, s3_uri: str, local_dir: str):
    bucket, prefix = parse_s3_uri(s3_uri)

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            # prefix가 ""인 경우도 고려
            rel = key[len(prefix):].lstrip("/") if prefix else key
            dst = Path(local_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dst))


def s3_upload_dir(s3, local_dir: str, s3_uri: str):
    bucket, prefix = parse_s3_uri(s3_uri)

    local = Path(local_dir)
    if not local.exists():
        raise RuntimeError(f"local_dir not found: {local_dir}")

    for p in local.rglob("*"):
        if p.is_dir():
            continue

        rel = p.relative_to(local).as_posix()
        if prefix:
            key = f"{prefix.rstrip('/')}/{rel}"
        else:
            key = rel  # 버킷 루트에 업로드

        s3.upload_file(str(p), bucket, key)

def debug_tree(root: str, limit: int = 80):
    rootp = Path(root)
    paths = [str(p) for p in rootp.rglob("*")]
    paths.sort()
    print(f"[debug_tree] {root} has {len(paths)} entries. showing first {min(limit,len(paths))}:")
    for p in paths[:limit]:
        print("  ", p)

def handler(event):
    inp = event.get("input", {}) or {}

    # 입력 검증 (KeyError로 죽지 않게)
    train_s3 = inp.get("train_s3")
    out_s3 = inp.get("out_s3")
    if not train_s3 or not out_s3:
        return {
            "ok": False,
            "error": "Missing required fields: train_s3, out_s3",
            "received_input": inp,
        }

    local_in = "/workspace/input"
    local_out = "/workspace/output"
    Path(local_in).mkdir(parents=True, exist_ok=True)
    Path(local_out).mkdir(parents=True, exist_ok=True)

    # 0) S3 client 준비 (env creds 자동 사용)
    try:
        s3 = make_s3_client()
    except Exception as e:
        return {"ok": False, "error": f"Failed to init S3 client: {e}"}

    # 1) 데이터 다운로드
    try:
        s3_download_prefix(s3, train_s3, local_in)
    except NoCredentialsError:
        return {
            "ok": False,
            "error": "AWS credentials not found. Ensure Runpod Secrets are mapped to "
                     "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN if needed) "
                     "and AWS_REGION.",
        }
    except ClientError as e:
        return {
            "ok": False,
            "error": "S3 download failed (ClientError).",
            "detail": str(e),
        }
    except Exception as e:
        return {"ok": False, "error": f"S3 download failed: {e}"}

    debug_tree(local_in, limit=120)

    # 2) 학습 실행 (env로 경로 주입)
    env = os.environ.copy()
    env["PBG_INPUT_DIR"] = local_in
    env["PBG_OUTPUT_DIR"] = local_out

    config_path = inp.get("config_path", "fb15k_config_gpu.py")

    # 로그가 크면 capture_output이 메모리를 잡아먹을 수 있어 주의.
    # 일단 지금은 유지하되, 운영에서는 파일로 리다이렉트 권장.
    cmd = ["python3", "-u", "train.py", "-c", config_path]

    p = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if p.returncode != 0:
        return {
            "ok": False,
            "error": "Training failed.",
            "stderr_tail": (p.stderr or "")[-4000:],
            "stdout_tail": (p.stdout or "")[-4000:],
        }

    # 3) 체크포인트 업로드
    try:
        s3_upload_dir(s3, local_out, out_s3)
    except ClientError as e:
        return {
            "ok": False,
            "error": "S3 upload failed (ClientError).",
            "detail": str(e),
        }
    except Exception as e:
        return {"ok": False, "error": f"S3 upload failed: {e}"}

    return {
        "ok": True,
        "out_s3": out_s3,
        "stdout_tail": (p.stdout or "")[-2000:],
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

