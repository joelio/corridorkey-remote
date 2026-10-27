#!/usr/bin/env python3
"""ckremote: send a clip to your own CorridorKey RunPod endpoint and time it.

CorridorKey Remote (unofficial). Not made by or affiliated with Corridor
Digital. See README.md for setup.

    ckremote check
    ckremote run shot.mov [--hint frame1.png] [--outputs alpha fg] [--label 4090-1080p-5s]

Configuration comes from environment variables (or a .env file next to
this script):

    RUNPOD_API_KEY          your RunPod API key
    CKREMOTE_ENDPOINT_ID    your serverless endpoint ID
    CKREMOTE_S3_ENDPOINT    e.g. https://<account>.r2.cloudflarestorage.com
    CKREMOTE_S3_BUCKET      bucket name
    CKREMOTE_S3_KEY_ID      access key ID
    CKREMOTE_S3_SECRET      secret access key
    CKREMOTE_S3_REGION      optional, defaults to "auto" (right for R2)
    CKREMOTE_RATE_PER_SEC   optional, your GPU's per-second price in USD
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.runpod.ai/v2"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
# RunPod rejects /run payloads over 10 MB; base64 adds a third.
INLINE_LIMIT = 7 * 2**20
PRESIGN_SECONDS = 3 * 3600
OUTPUT_EXT = {"alpha": ".mp4", "fg": ".mp4", "checker": ".mp4", "cutout": ".mov"}

CSV_FIELDS = [
    "timestamp",
    "label",
    "gpu",
    "vram_gb",
    "clip",
    "width",
    "height",
    "frames",
    "fps",
    "codec",
    "size_mb",
    "upload_s",
    "queue_delay_s",
    "cold_start_s",
    "model_load_s",
    "worker_download_s",
    "inference_s",
    "effective_fps",
    "worker_upload_s",
    "download_s",
    "wall_s",
    "billed_s_est",
    "rate_per_sec",
    "cost_usd_est",
    "hann_chunk",
    "hann_stride",
    "low_vram",
    "vram_budget_gb",
    "peak_vram_gb",
    "outputs",
    "job_id",
]


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def env(name: str, required: bool = True) -> str | None:
    value = os.environ.get(name)
    if required and not value:
        sys.exit(f"ckremote: {name} is not set (see README.md, Setup)")
    return value


def s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=env("CKREMOTE_S3_ENDPOINT"),
        aws_access_key_id=env("CKREMOTE_S3_KEY_ID"),
        aws_secret_access_key=env("CKREMOTE_S3_SECRET"),
        region_name=os.environ.get("CKREMOTE_S3_REGION", "auto"),
        config=Config(signature_version="s3v4"),
    )


class RunPod:
    def __init__(self, api_key: str, endpoint_id: str, session=None):
        import requests

        self.base = f"{API}/{endpoint_id}"
        self.http = session or requests.Session()
        self.http.headers.update({"authorization": api_key, "content-type": "application/json"})

    def submit(self, payload: dict) -> str:
        r = self.http.post(f"{self.base}/run", data=json.dumps({"input": payload}), timeout=60)
        r.raise_for_status()
        return r.json()["id"]

    def status(self, job_id: str) -> dict:
        r = self.http.get(f"{self.base}/status/{job_id}", timeout=60)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = self.http.get(f"{self.base}/health", timeout=30)
        r.raise_for_status()
        return r.json()

    def cancel(self, job_id: str) -> None:
        self.http.post(f"{self.base}/cancel/{job_id}", timeout=30)

    def wait(self, job_id: str, poll_s: float = 2.0, sleep=time.sleep, log=print) -> dict:
        last = None
        while True:
            st = self.status(job_id)
            if st.get("status") != last:
                last = st.get("status")
                log(f"  job {job_id}: {last}")
            if last in TERMINAL:
                return st
            sleep(poll_s)


def build_payload(
    clip: Path,
    outputs: list[str],
    *,
    input_url=None,
    hint_url=None,
    hint_path: Path | None = None,
    upload_urls=None,
    hann=None,
    low_vram=False,
    hint_quality=0.95,
    despill=0.5,
) -> dict:
    payload = {
        "input_name": clip.name,
        "outputs": outputs,
        "low_vram": low_vram,
        "hint_quality": hint_quality,
        "despill_strength": despill,
    }
    if input_url:
        payload["input_url"] = input_url
    else:
        payload["input_b64"] = base64.b64encode(clip.read_bytes()).decode()
    if hint_url:
        payload["hint_url"] = hint_url
    elif hint_path:
        payload["hint_b64"] = base64.b64encode(hint_path.read_bytes()).decode()
    if upload_urls:
        payload["upload_urls"] = upload_urls
    if hann:
        payload["hann_chunk"], payload["hann_stride"] = hann
    return payload


def parse_hann(value: str | None):
    if not value:
        return None
    chunk, _, stride = value.partition("/")
    chunk = int(chunk)
    return chunk, int(stride) if stride else max(1, chunk // 2)


def summarise(status: dict, local: dict, rate_per_sec: float | None) -> dict:
    """Flatten RunPod status, worker report and client timings into one CSV row."""
    out = status.get("output") or {}
    clip = out.get("clip", {})
    t = out.get("timings", {})
    boot = out.get("boot", {})
    gpu = out.get("gpu", {})
    hann = out.get("hann", {})
    # RunPod reports delayTime and executionTime in milliseconds.
    queue_delay = (status.get("delayTime") or 0) / 1000
    execution = (status.get("executionTime") or 0) / 1000
    cold = bool(out.get("cold"))
    # RunPod bills from worker start, so a cold job also pays for boot.
    billed = execution + (queue_delay if cold else 0)
    frames = clip.get("frames") or 0
    infer = t.get("inference_s")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": local.get("label") or "",
        "gpu": gpu.get("name"),
        "vram_gb": gpu.get("vram_total_gb"),
        "clip": local["clip"],
        "width": clip.get("width"),
        "height": clip.get("height"),
        "frames": frames,
        "fps": clip.get("fps"),
        "codec": clip.get("codec"),
        "size_mb": round((clip.get("size_bytes") or local.get("size_bytes", 0)) / 2**20, 2),
        "upload_s": _r(local.get("upload_s")),
        "queue_delay_s": _r(queue_delay),
        "cold_start_s": _r(queue_delay) if cold else "warm",
        "model_load_s": _r(boot.get("model_load_s")) if cold else "",
        "worker_download_s": _r(t.get("worker_download_s")),
        "inference_s": _r(infer),
        "effective_fps": _r(frames / infer) if infer else "",
        "worker_upload_s": _r(t.get("worker_upload_s")),
        "download_s": _r(local.get("download_s")),
        "wall_s": _r(local.get("wall_s")),
        "billed_s_est": _r(billed),
        "rate_per_sec": rate_per_sec if rate_per_sec else "",
        "cost_usd_est": round(billed * rate_per_sec, 4) if rate_per_sec else "",
        "hann_chunk": hann.get("hann_chunk"),
        "hann_stride": hann.get("hann_stride"),
        "low_vram": hann.get("low_vram"),
        "vram_budget_gb": out.get("vram_budget_gb"),
        "peak_vram_gb": out.get("peak_vram_gb"),
        "outputs": " ".join(local.get("outputs", [])),
        "job_id": status.get("id"),
    }


def _r(v, nd=2):
    return round(v, nd) if isinstance(v, (int, float)) else ""


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def print_table(row: dict) -> None:
    keys = [
        "gpu",
        "width",
        "height",
        "frames",
        "codec",
        "size_mb",
        "upload_s",
        "queue_delay_s",
        "cold_start_s",
        "model_load_s",
        "worker_download_s",
        "inference_s",
        "effective_fps",
        "worker_upload_s",
        "download_s",
        "wall_s",
        "hann_chunk",
        "hann_stride",
        "low_vram",
        "peak_vram_gb",
        "billed_s_est",
        "cost_usd_est",
    ]
    width = max(map(len, keys))
    print("\nTiming breakdown")
    for k in keys:
        print(f"  {k:<{width}}  {row.get(k, '')}")


def verify(path: Path, expected: str | None) -> None:
    """Fail if the downloaded file differs from what the worker wrote."""
    if not expected:
        return
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"ckremote: {path.name} does not match the worker's checksum")


def cmd_check(args) -> int:
    """Check credentials and endpoint before spending GPU time."""
    ok = True
    s3 = s3_client()
    bucket = env("CKREMOTE_S3_BUCKET")
    key = f"ckremote/check-{uuid.uuid4().hex}.txt"
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=b"ckremote check")
        url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=300)
        import requests

        r = requests.get(url, timeout=30)
        r.raise_for_status()
        print(f"bucket: ok ({bucket}, presigned GET works)")
    except Exception as e:  # noqa: BLE001 - report any failure plainly
        ok = False
        print(f"bucket: FAILED: {e}")
    finally:
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception:  # noqa: BLE001
            pass
    try:
        rp = RunPod(env("RUNPOD_API_KEY"), env("CKREMOTE_ENDPOINT_ID"))
        print(f"endpoint: ok {json.dumps(rp.health())}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"endpoint: FAILED: {e}")
    return 0 if ok else 1


def cmd_run(args) -> int:
    clip = Path(args.clip).expanduser().resolve()
    if not clip.is_file():
        sys.exit(f"ckremote: no such file: {clip}")
    hint = Path(args.hint).expanduser().resolve() if args.hint else None
    outputs = list(dict.fromkeys(args.outputs))
    out_dir = Path(args.out or clip.parent / f"{clip.stem}_ckremote").resolve()
    rate = args.rate_per_sec or (
        float(os.environ["CKREMOTE_RATE_PER_SEC"]) if os.environ.get("CKREMOTE_RATE_PER_SEC") else None
    )
    rp = RunPod(env("RUNPOD_API_KEY"), env("CKREMOTE_ENDPOINT_ID"))
    local = {"clip": clip.name, "label": args.label, "outputs": outputs, "size_bytes": clip.stat().st_size}

    t_wall = time.time()
    s3 = bucket = prefix = None
    upload_urls = input_url = hint_url = None
    if args.inline:
        if clip.stat().st_size > INLINE_LIMIT:
            sys.exit(f"ckremote: --inline is for clips under {INLINE_LIMIT >> 20} MB")
        local["upload_s"] = 0.0
    else:
        s3 = s3_client()
        bucket = env("CKREMOTE_S3_BUCKET")
        prefix = f"ckremote/{uuid.uuid4().hex}"

        def presign(method, key):
            return s3.generate_presigned_url(
                method, Params={"Bucket": bucket, "Key": key}, ExpiresIn=PRESIGN_SECONDS
            )

        print(f"uploading {clip.name} ({clip.stat().st_size / 2**20:.1f} MB)")
        t0 = time.time()
        s3.upload_file(str(clip), bucket, f"{prefix}/in/{clip.name}")
        if hint:
            s3.upload_file(str(hint), bucket, f"{prefix}/in/{hint.name}")
        local["upload_s"] = time.time() - t0
        input_url = presign("get_object", f"{prefix}/in/{clip.name}")
        hint_url = presign("get_object", f"{prefix}/in/{hint.name}") if hint else None
        upload_urls = {o: presign("put_object", f"{prefix}/out/{o}{OUTPUT_EXT[o]}") for o in outputs}

    payload = build_payload(
        clip,
        outputs,
        input_url=input_url,
        hint_url=hint_url,
        hint_path=hint,
        upload_urls=upload_urls,
        hann=parse_hann(args.hann),
        low_vram=args.low_vram,
        hint_quality=args.hint_quality,
        despill=args.despill,
    )
    job_id = rp.submit(payload)
    try:
        status = rp.wait(job_id, poll_s=args.poll)
    except KeyboardInterrupt:
        rp.cancel(job_id)
        print(f"\ncancelled {job_id}")
        return 130
    output = status.get("output") or {}
    if status.get("status") != "COMPLETED" or "error" in output:
        print(json.dumps(status, indent=2)[:4000])
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name in outputs:
        dest = out_dir / f"{clip.stem}_{name}{OUTPUT_EXT[name]}"
        if args.inline:
            dest.write_bytes(base64.b64decode(output["outputs"][name]["b64"]))
        else:
            s3.download_file(bucket, f"{prefix}/out/{name}{OUTPUT_EXT[name]}", str(dest))
        verify(dest, output["outputs"][name].get("sha256"))
        print(f"  wrote {dest}")
    local["download_s"] = time.time() - t0
    local["wall_s"] = time.time() - t_wall

    if s3 and not args.keep_remote:
        keys = [f"{prefix}/in/{clip.name}"] + ([f"{prefix}/in/{hint.name}"] if hint else [])
        keys += [f"{prefix}/out/{o}{OUTPUT_EXT[o]}" for o in outputs]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys]})

    row = summarise(status, local, rate)
    print_table(row)
    append_csv(Path(args.csv), row)
    print(f"\nappended to {args.csv}")
    return 0


def main(argv=None) -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    ap = argparse.ArgumentParser(prog="ckremote", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="check bucket and endpoint settings").set_defaults(fn=cmd_check)
    run = sub.add_parser("run", help="key a clip on your endpoint")
    run.add_argument("clip")
    run.add_argument("--hint", help="first-frame matte PNG (white = subject)")
    run.add_argument("--outputs", nargs="+", default=["alpha", "fg"], choices=sorted(OUTPUT_EXT))
    run.add_argument("--hann", help="window override, e.g. 48/24 (default: auto from VRAM)")
    run.add_argument("--low-vram", action="store_true")
    run.add_argument("--hint-quality", type=float, default=0.95)
    run.add_argument("--despill", type=float, default=0.5)
    run.add_argument("--out", help="output folder (default: next to the clip)")
    run.add_argument("--label", help="free text saved in the CSV, e.g. 4090-4k-30s")
    run.add_argument("--csv", default="results/timings.csv")
    run.add_argument("--rate-per-sec", type=float, help="GPU price in USD per second")
    run.add_argument(
        "--inline",
        action="store_true",
        help="send the clip in the request instead of the bucket (tiny clips only)",
    )
    run.add_argument("--keep-remote", action="store_true", help="leave files in the bucket")
    run.add_argument("--poll", type=float, default=2.0)
    run.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
