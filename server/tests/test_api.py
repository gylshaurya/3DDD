import json

import pytest
from fastapi.testclient import TestClient

from threeddd import api, config


@pytest.fixture()
def client():
    with TestClient(api.app) as c:
        yield c


def test_health_and_budget_shape(client, monkeypatch):
    monkeypatch.setattr(api.budget, "metered_this_month", lambda: 1.25)
    assert client.get("/api/health").json()["ok"] is True
    b = client.get("/api/budget").json()
    assert b["metered"] == 1.25 and b["remaining"] == pytest.approx(b["cap"] - b["spent"])


def test_create_job_validation(client):
    assert client.post("/api/jobs", data={}).status_code == 400
    assert client.post("/api/jobs", data={"sample": "nope"}).status_code == 404
    r = client.post("/api/jobs", files={"video": ("x.mp4", b"0" * 10, "video/mp4")}, data={"preset": "bogus"})
    assert r.status_code == 400 and "preset" in r.json()["detail"]
    r = client.post("/api/jobs", files={"video": ("x.mp4", b"0" * 10, "video/mp4")}, data={"gpu": "TPU"})
    assert r.status_code == 400


def test_unknown_job_404(client):
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.post("/api/jobs/missing/cancel").status_code == 404


def test_bad_video_fails_cleanly_and_streams_events(client):
    r = client.post("/api/jobs", files={"video": ("junk.mp4", b"not a video", "video/mp4")},
                    data={"backend": "modal", "preset": "preview"})
    job = r.json()
    events = []
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as s:
        for line in s.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                events.append(ev)
                if ev["type"] == "job" and ev["job"]["status"] in ("failed", "done"):
                    break
    final = events[-1]["job"]
    assert final["status"] == "failed" and final["error"]
    assert (config.JOBS_DIR / job["id"] / "events.jsonl").exists()
    # failed jobs can be deleted; running ones cannot
    assert client.delete(f"/api/jobs/{job['id']}").json()["ok"]


def test_static_files_cannot_escape_data_dirs(client):
    (config.DATA_DIR / "secret.json").write_text('{"token": "s3cret"}')
    for path in ("/files/jobs/..%2Fsecret.json", "/files/scenes/../secret.json", "/..%2F..%2Fdata%2Fsecret.json"):
        r = client.get(path)
        assert "s3cret" not in r.text, path


def test_tmp_dirs_cleaned_on_errors(client):
    before = {p.name for p in config.DATA_DIR.iterdir()}
    client.post("/api/jobs", data={"sample": "nope"})
    client.post("/api/jobs", data={})
    after = {p.name for p in config.DATA_DIR.iterdir()}
    assert after == before


def test_torn_event_log_does_not_break_startup(tmp_path):
    from threeddd.jobs import _read_events
    f = tmp_path / "events.jsonl"
    f.write_text('{"t": 1, "type": "log", "msg": "ok"}\n{"t": 2, "type": "lo')
    assert [e["msg"] for e in _read_events(f)] == ["ok"]
