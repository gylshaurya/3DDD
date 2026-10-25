"""Hosted ("public") mode: strangers can upload, so limits and ownership must hold."""

import json
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from threeddd import api, config


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "PUBLIC", True)
    monkeypatch.setattr(api, "_recent_by_ip", api.defaultdict(api.deque))
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p.read_bytes()


def upload(client, data, **form):
    return client.post("/api/jobs", files={"video": ("clip.mp4", data, "video/mp4")},
                       data={"backend": "modal", "preset": "preview", "gpu": "L4", **form})


def wait_finished(job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if api.manager.jobs[job_id].status in ("failed", "done", "cancelled"):
            return
        time.sleep(0.1)


def test_health_reports_public_mode(client):
    h = client.get("/api/health").json()
    assert h["public"] is True and h["local_trainer"] is False and h["limits"]["presets"] == ["preview", "fast"]


@pytest.mark.parametrize("form,msg", [
    ({"backend": "local"}, "cloud GPUs only"),
    ({"preset": "quality"}, "isn't available"),
    ({"gpu": "H100"}, "isn't available"),
])
def test_expensive_options_are_refused(client, tiny_video, form, msg):
    r = upload(client, tiny_video, **form)
    assert r.status_code == 400 and msg in r.json()["detail"]


def test_non_video_is_refused(client):
    r = upload(client, b"definitely not a video")
    assert r.status_code == 400 and "video" in r.json()["detail"]


def test_owner_token_scopes_cancel_and_delete(client, tiny_video):
    job = upload(client, tiny_video).json()
    token = job.pop("owner_token")
    assert "owner_hash" not in job
    wait_finished(job["id"])
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 403
    assert client.delete(f"/api/jobs/{job['id']}", headers={"x-owner-token": "wrong"}).status_code == 403
    assert client.delete(f"/api/jobs/{job['id']}", headers={"x-owner-token": token}).json()["ok"]


def test_strangers_jobs_are_not_listed_but_featured_ones_are(client, tiny_video):
    job = upload(client, tiny_video).json()
    wait_finished(job["id"])
    assert job["id"] not in [j["id"] for j in client.get("/api/jobs").json()]
    assert [j["id"] for j in client.get(f"/api/jobs?ids={job['id']}").json()] == [job["id"]]
    api.manager.jobs[job["id"]].featured = True
    assert job["id"] in [j["id"] for j in client.get("/api/jobs").json()]
    # featured scenes can never be deleted from the public site, even by the uploader
    assert client.delete(f"/api/jobs/{job['id']}", headers={"x-owner-token": job["owner_token"]}).status_code == 403


def test_hourly_rate_limit(client, tiny_video):
    for _ in range(config.PUBLIC_LIMITS["jobs_per_ip_per_hour"]):
        j = upload(client, tiny_video).json()
        wait_finished(j["id"])
    r = upload(client, tiny_video)
    assert r.status_code == 429 and "hourly" in r.json()["detail"]


def test_event_stream_resumes_from_cursor(client, tiny_video, monkeypatch):
    monkeypatch.setattr(api, "_recent_by_ip", api.defaultdict(api.deque))
    job = upload(client, tiny_video).json()
    wait_finished(job["id"])
    def events(since):
        out = []
        with client.stream("GET", f"/api/jobs/{job['id']}/events?since={since}") as s:
            for line in s.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    if ev["type"] == "replay_done":
                        break
                    out.append(ev)
        return out
    full = events(0)
    assert len(full) > 3 and events(3) == full[3:]
