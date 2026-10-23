"""HTTP API. Run: uv run uvicorn splatcast.api:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import budget, local
from .config import DATA_DIR, REPO_ROOT, SCENES_DIR
from .jobs import JobManager

MAX_UPLOAD_BYTES = 1_500_000_000
SAMPLE_VIDEOS = DATA_DIR / "test-videos"
WEB_DIST = REPO_ROOT / "apps" / "web" / "dist"

manager: JobManager


@asynccontextmanager
async def lifespan(_: FastAPI):
    global manager
    manager = JobManager()
    manager.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="Splatcast", lifespan=lifespan)


def _job(job_id: str):
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "local_trainer": local.available()}


@app.get("/api/budget")
def get_budget() -> dict:
    return budget.status()


@app.get("/api/lan")
def lan(request: Request) -> dict:
    """The address a phone on the same Wi-Fi can open to upload a capture."""
    ip = "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        pass
    port = request.headers.get("x-forwarded-port") or request.url.port or 8000
    return {"ip": ip, "url": f"http://{ip}:{port}/capture"}


@app.get("/api/samples")
def samples() -> list[dict]:
    return [{"name": p.stem, "size_mb": round(p.stat().st_size / 1e6, 1)} for p in sorted(SAMPLE_VIDEOS.glob("*.mp4"))]


@app.post("/api/jobs")
async def create_job(
    video: UploadFile | None = File(None),
    sample: str | None = Form(None),
    name: str = Form(""),
    backend: str = Form("modal"),
    preset: str = Form("fast"),
    gpu: str = Form("L4"),
) -> dict:
    tmp = Path(tempfile.mkdtemp(dir=DATA_DIR))
    if video is not None:
        suffix = Path(video.filename or "capture.mp4").suffix.lower() or ".mp4"
        dst = tmp / f"upload{suffix}"
        size = 0
        with dst.open("wb") as f:
            while chunk := await video.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise HTTPException(413, "Video is too large (max 1.5 GB)")
                f.write(chunk)
        name = name or Path(video.filename or "").stem
    elif sample:
        src = SAMPLE_VIDEOS / f"{Path(sample).name}.mp4"
        if not src.exists():
            raise HTTPException(404, "No such sample")
        dst = tmp / "upload.mp4"
        shutil.copy2(src, dst)
        name = name or sample
    else:
        raise HTTPException(400, "Upload a video or choose a sample")
    try:
        job = manager.create(dst, name, backend, preset, gpu)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return job.public()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.public() for j in sorted(manager.jobs.values(), key=lambda j: -j.created)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job(job_id).public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    _job(job_id)
    manager.cancel(job_id)
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = _job(job_id)
    if job.status in ("queued", "running"):
        raise HTTPException(409, "Cancel the job first")
    manager.delete(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-sent events: full history first, then live events."""
    _job(job_id)

    async def stream():
        past, q = manager.subscribe(job_id)
        try:
            for ev in past:
                yield f"data: {json.dumps(ev)}\n\n"
            yield 'data: {"type": "replay_done"}\n\n'
            while not await request.is_disconnected():
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            manager.unsubscribe(job_id, q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/scenes")
def scenes() -> list[dict]:
    out = []
    for d in sorted(SCENES_DIR.iterdir(), key=lambda p: -p.stat().st_mtime) if SCENES_DIR.exists() else []:
        if not (d / "scene.splat").exists():
            continue
        job = manager.jobs.get(d.name)
        meta = json.loads((d / "scene.json").read_text()) if (d / "scene.json").exists() else {}
        out.append({
            "id": d.name,
            "name": job.name if job else d.name,
            "metrics": job.metrics if job else {},
            "scene": meta,
            "ply": f"/files/scenes/{d.name}/scene.ply",
            "splat": f"/files/scenes/{d.name}/scene.splat",
            "thumb": f"/files/scenes/{d.name}/thumb.jpg",
            "ply_mb": round((d / "scene.ply").stat().st_size / 1e6, 1) if (d / "scene.ply").exists() else None,
        })
    return out


app.mount("/files/jobs", StaticFiles(directory=DATA_DIR / "jobs"), name="job-files")
app.mount("/files/scenes", StaticFiles(directory=SCENES_DIR), name="scene-files")

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = WEB_DIST / path
        if path and f.is_file() and WEB_DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(WEB_DIST / "index.html")
