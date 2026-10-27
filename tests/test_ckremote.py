"""Tests for the client. No network: RunPod and the bucket are faked."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ckremote", ROOT / "ckremote.py")
ck = importlib.util.module_from_spec(spec)
sys.modules["ckremote"] = ck
spec.loader.exec_module(ck)

WORKER_OUTPUT = {
    "cold": True,
    "boot": {"model_load_s": 21.5},
    "gpu": {"name": "NVIDIA L40S", "vram_total_gb": 44.5},
    "clip": {
        "width": 1920,
        "height": 1080,
        "frames": 120,
        "fps": 24.0,
        "codec": "prores",
        "size_bytes": 50 * 2**20,
    },
    "hann": {"hann_chunk": 48, "hann_stride": 24, "low_vram": False},
    "vram_budget_gb": 43.9,
    "peak_vram_gb": 21.0,
    "timings": {"worker_download_s": 1.0, "inference_s": 30.0, "worker_upload_s": 2.0},
}


def test_parse_hann():
    assert ck.parse_hann("48/24") == (48, 24)
    assert ck.parse_hann("8") == (8, 4)
    assert ck.parse_hann(None) is None


def test_build_payload_url_mode(tmp_path):
    clip = tmp_path / "shot.mov"
    clip.write_bytes(b"x")
    p = ck.build_payload(
        clip, ["alpha"], input_url="https://in", upload_urls={"alpha": "https://a"}, hann=(8, 4)
    )
    assert "input_b64" not in p and p["input_url"] == "https://in"
    assert (p["hann_chunk"], p["hann_stride"]) == (8, 4)


def test_build_payload_inline(tmp_path):
    clip = tmp_path / "shot.mov"
    clip.write_bytes(b"clip")
    hint = tmp_path / "h.png"
    hint.write_bytes(b"png")
    p = ck.build_payload(clip, ["alpha"], hint_path=hint)
    assert base64.b64decode(p["input_b64"]) == b"clip"
    assert base64.b64decode(p["hint_b64"]) == b"png"
    assert "hann_chunk" not in p


def test_summarise_cold():
    status = {
        "id": "j1",
        "status": "COMPLETED",
        "delayTime": 45000,
        "executionTime": 34000,
        "output": WORKER_OUTPUT,
    }
    row = ck.summarise(
        status,
        {"clip": "shot.mov", "upload_s": 3.0, "download_s": 4.0, "wall_s": 90.0, "outputs": ["alpha", "fg"]},
        0.001,
    )
    assert row["cold_start_s"] == 45.0
    assert row["model_load_s"] == 21.5
    assert row["effective_fps"] == 4.0
    assert row["billed_s_est"] == 79.0
    assert row["cost_usd_est"] == 0.079
    assert set(row) == set(ck.CSV_FIELDS)


def test_summarise_warm_without_rate():
    status = {
        "id": "j2",
        "delayTime": 300,
        "executionTime": 34000,
        "output": {**WORKER_OUTPUT, "cold": False},
    }
    row = ck.summarise(status, {"clip": "shot.mov"}, None)
    assert row["cold_start_s"] == "warm" and row["model_load_s"] == ""
    assert row["billed_s_est"] == 34.0 and row["cost_usd_est"] == ""


def test_append_csv_writes_header_once(tmp_path):
    path = tmp_path / "r" / "t.csv"
    row = ck.summarise({"output": WORKER_OUTPUT}, {"clip": "a.mov"}, None)
    ck.append_csv(path, row)
    ck.append_csv(path, row)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2 and rows[0]["gpu"] == "NVIDIA L40S"


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, statuses):
        self.headers = {}
        self.statuses = list(statuses)
        self.posts = []

    def post(self, url, data=None, timeout=None):
        self.posts.append((url, data))
        return FakeResponse({"id": "job-1"})

    def get(self, url, timeout=None):
        return FakeResponse(self.statuses.pop(0))


def test_runpod_wait_polls_until_terminal():
    session = FakeSession(
        [{"status": "IN_QUEUE"}, {"status": "IN_PROGRESS"}, {"status": "COMPLETED", "output": {}}]
    )
    rp = ck.RunPod("key", "ep", session=session)
    assert session.headers["authorization"] == "key"
    assert rp.submit({"a": 1}) == "job-1"
    assert session.posts[0][0] == "https://api.runpod.ai/v2/ep/run"
    st = rp.wait("job-1", sleep=lambda s: None, log=lambda m: None)
    assert st["status"] == "COMPLETED"


def test_run_inline_end_to_end(tmp_path, monkeypatch):
    clip = tmp_path / "shot.mov"
    clip.write_bytes(b"clip")
    out = {
        **WORKER_OUTPUT,
        "outputs": {
            "alpha": {
                "b64": base64.b64encode(b"ALPHA").decode(),
                "sha256": hashlib.sha256(b"ALPHA").hexdigest(),
            },
            "fg": {"b64": base64.b64encode(b"FG").decode()},
        },
    }
    session = FakeSession(
        [{"id": "job-1", "status": "COMPLETED", "delayTime": 1000, "executionTime": 30000, "output": out}]
    )
    real = ck.RunPod
    monkeypatch.setattr(ck, "RunPod", lambda key, ep: real(key, ep, session=session))
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("CKREMOTE_ENDPOINT_ID", "ep")
    csv_path = tmp_path / "t.csv"
    rc = ck.main(["run", str(clip), "--inline", "--csv", str(csv_path), "--out", str(tmp_path / "o")])
    assert rc == 0
    assert (tmp_path / "o" / "shot_alpha.mp4").read_bytes() == b"ALPHA"
    assert list(csv.DictReader(csv_path.open()))[0]["job_id"] == "job-1"


def test_missing_config_exits(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        ck.env("RUNPOD_API_KEY")


def test_verify_rejects_changed_file(tmp_path):
    f = tmp_path / "a.mov"
    f.write_bytes(b"one")
    ck.verify(f, hashlib.sha256(b"one").hexdigest())
    with pytest.raises(SystemExit):
        ck.verify(f, hashlib.sha256(b"two").hexdigest())
