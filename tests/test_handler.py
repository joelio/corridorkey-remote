"""CPU-only tests for the worker. Upstream and torch are stubbed out."""

from __future__ import annotations

import base64
import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("handler", ROOT / "worker" / "handler.py")
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)

HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")

# Shape of upstream's settings log (corridorkey/runtime.py, run_inference).
SETTINGS = """CorridorKey v2 Inference Settings Log
====================================
Input: /tmp/x/shot.mov
Resolution: 1920x1080
Hann Chunk: 48
Hann Stride: 24
Low VRAM: False
Model Scale: 1.0
"""


def test_validate_defaults():
    job = handler.validate_job({"input_url": "https://x/in.mov"})
    assert job["outputs"] == ["alpha", "fg"]
    assert job["hann_chunk"] is None and job["low_vram"] is False
    assert job["input_name"] == "input.mp4"


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"input_url": "u", "input_b64": "b"},
        {"input_url": "u", "outputs": ["matte"]},
        {"input_url": "u", "outputs": ["alpha", "fg"], "upload_urls": {"alpha": "x"}},
        {"input_url": "u", "hann_stride": 4},
        {"input_url": "u", "hann_chunk": 0},
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(handler.JobError):
        handler.validate_job(bad)


def test_input_name_cannot_escape_workdir():
    job = handler.validate_job({"input_url": "u", "input_name": "../../etc/passwd"})
    assert job["input_name"] == "passwd"


def test_inference_kwargs_match_infer_py_defaults():
    # infer.py:135-137: auto only when no chunk, stride or low_vram given.
    auto = handler.inference_kwargs(handler.validate_job({"input_url": "u"}), Path("/t"))
    assert auto["auto_vram"] is True and (auto["hann_chunk"], auto["hann_stride"]) == (80, 40)
    manual = handler.inference_kwargs(handler.validate_job({"input_url": "u", "hann_chunk": 8}), Path("/t"))
    assert manual["auto_vram"] is False and (manual["hann_chunk"], manual["hann_stride"]) == (8, 4)
    low = handler.inference_kwargs(handler.validate_job({"input_url": "u", "low_vram": True}), Path("/t"))
    assert low["auto_vram"] is False


def test_parse_settings_log():
    assert handler.parse_settings_log(SETTINGS) == {"hann_chunk": 48, "hann_stride": 24, "low_vram": False}


def test_output_filenames():
    assert handler.output_filename("shot", "cutout", 0.5) == "shot_cutout.mov"
    assert handler.output_filename("shot", "checker", 0.75) == "shot_checker_comp_despill075.mp4"


def make_clip(path: Path, codec: str = "libx264", frames: int = 12) -> Path:
    extra = (
        ["-pix_fmt", "yuv420p"] if codec == "libx264" else ["-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x180:rate=24:duration={frames / 24}",
            "-c:v",
            codec,
            *extra,
            str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_probe_clip(tmp_path):
    info = handler.probe_clip(make_clip(tmp_path / "c.mp4"))
    assert (info["width"], info["height"], info["frames"], info["codec"]) == (320, 180, 12, "h264")
    assert info["fps"] == 24.0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_probe_prores_4444(tmp_path):
    info = handler.probe_clip(make_clip(tmp_path / "c.mov", codec="prores_ks"))
    assert info["codec"] == "prores" and info["pix_fmt"].startswith("yuva444p")


class FakeCuda:
    def is_available(self):
        return True

    def synchronize(self):
        pass

    def mem_get_info(self):
        return (22 * 2**30, 24 * 2**30)

    def memory_reserved(self):
        return 1 * 2**30

    def max_memory_reserved(self):
        return 18 * 2**30

    def reset_peak_memory_stats(self):
        pass

    def get_device_name(self, i):
        return "NVIDIA GeForce RTX 4090"


@pytest.fixture
def stub_upstream(monkeypatch):
    """Fake torch and corridorkey modules that write outputs like upstream."""
    calls = {}
    torch = types.ModuleType("torch")
    torch.cuda = FakeCuda()

    ck = types.ModuleType("corridorkey")
    runtime = types.ModuleType("corridorkey.runtime")
    video_io = types.ModuleType("corridorkey.video_io")

    class InferenceConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def run_inference(model, src, out_dir, cfg, writer, *, initial_hint_path, hint_video_path):
        calls["cfg"], calls["hint"] = cfg, initial_hint_path
        out = Path(out_dir)
        out.mkdir(parents=True)
        stem = Path(src).stem
        (out / f"{stem}_settings.txt").write_text(SETTINGS)
        for name in cfg.outputs:
            (out / handler.output_filename(stem, name, cfg.despill_strength)).write_bytes(b"x" * 10)

    runtime.RuntimePaths = types.SimpleNamespace(from_package=lambda root: root)
    runtime.load_model = lambda paths, device: "model"
    runtime.InferenceConfig = InferenceConfig
    runtime.run_inference = run_inference
    video_io.WriterConfig = lambda: "writer"
    video_io.raw_clip_name = lambda p: Path(p).stem
    for name, mod in {
        "torch": torch,
        "corridorkey": ck,
        "corridorkey.runtime": runtime,
        "corridorkey.video_io": video_io,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(
        handler,
        "probe_clip",
        lambda p: {
            "width": 320,
            "height": 180,
            "frames": 12,
            "fps": 24.0,
            "codec": "h264",
            "size_bytes": p.stat().st_size,
        },
    )
    return calls


def test_worker_inline_round_trip(stub_upstream):
    worker = handler.Worker()
    hint = base64.b64encode(b"png").decode()
    res = worker.run(
        {
            "input_b64": base64.b64encode(b"clip").decode(),
            "input_name": "shot.mov",
            "hint_b64": hint,
            "outputs": ["alpha", "cutout"],
        }
    )
    assert res["cold"] is True
    assert res["gpu"]["name"] == "NVIDIA GeForce RTX 4090"
    assert res["vram_budget_gb"] == 23.0
    assert res["hann"] == {"hann_chunk": 48, "hann_stride": 24, "low_vram": False}
    assert base64.b64decode(res["outputs"]["cutout"]["b64"]) == b"x" * 10
    assert stub_upstream["hint"].endswith("hint.png")
    assert stub_upstream["cfg"].auto_vram is True
    assert set(res["timings"]) >= {"worker_download_s", "inference_s", "worker_upload_s"}
    assert worker.run({"input_b64": "Y2xpcA==", "input_name": "shot.mov"})["cold"] is False


def test_worker_uploads_to_presigned_urls(stub_upstream, monkeypatch):
    puts = {}
    monkeypatch.setattr(handler, "_fetch", lambda url, b64, dest: dest.write_bytes(b"clip"))
    monkeypatch.setattr(handler, "_put", lambda url, path: puts.__setitem__(url, path.read_bytes()))
    res = handler.Worker().run(
        {
            "input_url": "https://get",
            "input_name": "shot.mov",
            "upload_urls": {"alpha": "https://a", "fg": "https://f"},
        }
    )
    assert puts == {"https://a": b"x" * 10, "https://f": b"x" * 10}
    assert "b64" not in res["outputs"]["alpha"]
    assert len(res["outputs"]["alpha"]["sha256"]) == 64


def test_handler_reports_bad_input(stub_upstream, monkeypatch):
    monkeypatch.setattr(handler, "WORKER", handler.Worker(), raising=False)
    assert "error" in handler.handler({"input": {"outputs": ["alpha"]}})
