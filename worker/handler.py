"""RunPod serverless handler for CorridorKey Remote (unofficial).

The model is loaded once when the worker starts, before it takes a job, so a
warm worker skips straight to inference. Every job reports where its time
went; that report is the point of this project.

Job input (all URLs are presigned by the client, so the worker never holds
bucket credentials):

    {
      "input_url": "https://...",          # or "input_b64" for small tests
      "input_name": "shot.mov",
      "hint_url": "https://...",           # optional first-frame hint PNG
      "hint_b64": "...",                   # (or this)
      "outputs": ["alpha", "fg"],          # any of alpha fg checker cutout
      "upload_urls": {"alpha": "https://...", ...},
      "hann_chunk": 48, "hann_stride": 24, # optional; omit for auto
      "low_vram": false,
      "hint_quality": 0.95,
      "despill_strength": 0.5
    }

If "upload_urls" is missing, outputs come back base64-encoded in the job
result. RunPod caps payloads at 10 MB, so that is only for tiny tests.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Same as infer.py:8-9. The image is built with every asset baked in.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROCESS_START = time.time()
CK_ROOT = Path(os.environ.get("CK_ROOT", "/opt/corridorkey"))
ALLOWED_OUTPUTS = ("alpha", "fg", "checker", "cutout")
CHUNK = 1 << 20

# Output file names written by upstream run_inference
# (corridorkey/runtime.py, the `suffix` dict and the ".mov if cutout" line).
OUTPUT_SUFFIX = {
    "alpha": "alpha.mp4",
    "fg": "fg.mp4",
    "cutout": "cutout.mov",
}


class JobError(ValueError):
    """Bad job input. Reported back to the client instead of crashing."""


def validate_job(job_input: dict) -> dict:
    """Check the job and fill in defaults. Pure, so it is tested on CPU."""
    if not isinstance(job_input, dict):
        raise JobError("job input must be an object")
    if bool(job_input.get("input_url")) == bool(job_input.get("input_b64")):
        raise JobError("give exactly one of input_url or input_b64")
    outputs = job_input.get("outputs") or ["alpha", "fg"]
    bad = sorted(set(outputs) - set(ALLOWED_OUTPUTS))
    if bad:
        raise JobError(f"unsupported outputs: {bad}")
    upload_urls = job_input.get("upload_urls") or {}
    missing = [o for o in outputs if upload_urls and o not in upload_urls]
    if missing:
        raise JobError(f"no upload_url for outputs: {missing}")
    chunk = job_input.get("hann_chunk")
    stride = job_input.get("hann_stride")
    if stride is not None and chunk is None:
        raise JobError("hann_stride needs hann_chunk")
    if chunk is not None and (int(chunk) < 1 or (stride is not None and int(stride) < 1)):
        raise JobError("hann_chunk and hann_stride must be positive")
    name = Path(job_input.get("input_name") or "input.mp4").name
    return {
        "input_url": job_input.get("input_url"),
        "input_b64": job_input.get("input_b64"),
        "input_name": name,
        "hint_url": job_input.get("hint_url"),
        "hint_b64": job_input.get("hint_b64"),
        "outputs": list(dict.fromkeys(outputs)),
        "upload_urls": upload_urls,
        "hann_chunk": None if chunk is None else int(chunk),
        "hann_stride": None if stride is None else int(stride),
        "low_vram": bool(job_input.get("low_vram", False)),
        "hint_quality": float(job_input.get("hint_quality", 0.95)),
        "despill_strength": float(job_input.get("despill_strength", 0.5)),
    }


def inference_kwargs(job: dict, temp_dir: Path) -> dict:
    """InferenceConfig kwargs, mirroring how infer.py:135-166 builds them."""
    auto_vram = job["hann_chunk"] is None and job["hann_stride"] is None and not job["low_vram"]
    chunk = job["hann_chunk"] if job["hann_chunk"] is not None else 80
    stride = job["hann_stride"] if job["hann_stride"] is not None else max(1, chunk // 2)
    return {
        "hann_chunk": chunk,
        "hann_stride": stride,
        "auto_vram": auto_vram,
        "low_vram": job["low_vram"],
        "hint_quality": job["hint_quality"],
        "despill_strength": job["despill_strength"],
        "outputs": set(job["outputs"]),
        "temp_dir": temp_dir,
    }


def parse_settings_log(text: str) -> dict:
    """Read the Hann values upstream actually used from <clip>_settings.txt."""
    wanted = {"Hann Chunk": "hann_chunk", "Hann Stride": "hann_stride", "Low VRAM": "low_vram"}
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() in wanted:
            value = value.strip()
            out[wanted[key.strip()]] = value == "True" if value in ("True", "False") else int(value)
    return out


def output_filename(clip_stem: str, name: str, despill_strength: float) -> str:
    if name == "checker":
        return f"{clip_stem}_checker_comp_despill{int(despill_strength * 100):03d}.mp4"
    return f"{clip_stem}_{OUTPUT_SUFFIX[name]}"


def probe_clip(path: Path) -> dict:
    """Resolution, frame count and codec, from ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=codec_name,width,height,nb_read_packets,r_frame_rate,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    stream = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)["streams"][0]
    num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "frames": int(stream.get("nb_read_packets") or 0),
        "fps": round(float(num) / float(den or 1), 3) if float(den or 1) else None,
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "size_bytes": path.stat().st_size,
    }


def _fetch(url: str | None, b64: str | None, dest: Path) -> None:
    if b64:
        dest.write_bytes(base64.b64decode(b64))
        return
    import requests

    with requests.get(url, stream=True, timeout=(10, 300)) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for block in r.iter_content(CHUNK):
                f.write(block)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _put(url: str, path: Path) -> None:
    import requests

    with open(path, "rb") as f:
        r = requests.put(url, data=f, timeout=(10, 600), headers={"Content-Length": str(path.stat().st_size)})
    r.raise_for_status()


class Worker:
    """Holds the loaded model across jobs on a warm worker."""

    def __init__(self):
        import torch

        t0 = time.time()
        from corridorkey.runtime import RuntimePaths, load_model

        self.torch = torch
        self.import_s = time.time() - t0
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU visible to the worker")
        t0 = time.time()
        self.model = load_model(RuntimePaths.from_package(CK_ROOT), device="cuda")
        torch.cuda.synchronize()
        self.model_load_s = time.time() - t0
        self.ready_at = time.time()
        free, total = torch.cuda.mem_get_info()
        self.gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total / 2**30, 2),
            "vram_free_at_boot_gb": round(free / 2**30, 2),
        }
        self.jobs_done = 0

    def run(self, raw_input: dict) -> dict:
        from corridorkey.runtime import InferenceConfig, run_inference
        from corridorkey.video_io import WriterConfig, raw_clip_name

        torch = self.torch
        t_job = time.time()
        timings = {}
        job = validate_job(raw_input)
        work = Path(tempfile.mkdtemp(prefix="ckjob_"))
        try:
            # Keep the client's extension; upstream sniffs the container from it.
            src = work / job["input_name"]
            t0 = time.time()
            _fetch(job["input_url"], job["input_b64"], src)
            hint = None
            if job["hint_url"] or job["hint_b64"]:
                hint = work / "hint.png"
                _fetch(job["hint_url"], job["hint_b64"], hint)
            timings["worker_download_s"] = time.time() - t0

            clip = probe_clip(src)
            out_dir = work / "out"
            cfg = InferenceConfig(**inference_kwargs(job, work / "tmp"))
            # Same budget formula as upstream apply_auto_vram (runtime.py:191-193),
            # recorded so we can confirm the worker sees the whole card.
            free, _ = torch.cuda.mem_get_info()
            vram_budget_gb = (free + torch.cuda.memory_reserved()) / 2**30
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            run_inference(
                self.model,
                str(src),
                str(out_dir),
                cfg,
                WriterConfig(),
                initial_hint_path=str(hint) if hint else None,
                hint_video_path=None,
            )
            torch.cuda.synchronize()
            timings["inference_s"] = time.time() - t0

            # Upstream names outputs after raw_clip_name(), which swaps in the
            # parent folder for files called input/frames/rgb (video_io.py:58-63).
            stem = raw_clip_name(str(src))
            settings_file = next(out_dir.glob("*_settings.txt"), None)
            used = parse_settings_log(settings_file.read_text()) if settings_file else {}

            t0 = time.time()
            results = {}
            for name in job["outputs"]:
                path = out_dir / output_filename(stem, name, job["despill_strength"])
                if not path.exists():
                    raise RuntimeError(f"upstream did not write {path.name}")
                # The client checks this hash, proving the file it receives is
                # the one upstream wrote, with no re-encode on the way.
                results[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                if job["upload_urls"]:
                    _put(job["upload_urls"][name], path)
                else:
                    results[name]["b64"] = base64.b64encode(path.read_bytes()).decode()
            timings["worker_upload_s"] = time.time() - t0
            timings["worker_total_s"] = time.time() - t_job
        finally:
            shutil.rmtree(work, ignore_errors=True)

        cold = self.jobs_done == 0
        self.jobs_done += 1
        return {
            "cold": cold,
            "boot": {
                "import_s": self.import_s,
                "model_load_s": self.model_load_s,
                "process_to_ready_s": self.ready_at - PROCESS_START,
                "idle_before_job_s": t_job - self.ready_at if cold else None,
            },
            "gpu": self.gpu,
            "vram_budget_gb": round(vram_budget_gb, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "clip": clip,
            "hann": used,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "outputs": results,
        }


def handler(job):
    try:
        return WORKER.run(job["input"])
    except JobError as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import sys

    import runpod

    sys.path.insert(0, str(CK_ROOT))
    WORKER = Worker()
    print(f"ckremote worker ready: {WORKER.gpu} model_load_s={WORKER.model_load_s:.1f}", flush=True)
    runpod.serverless.start({"handler": handler})
