"""HTTP API. Run: uv run uvicorn threeddd.api:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import socket
import tempfile
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import budget, local
from .config import CORS_ORIGINS, DATA_DIR, PUBLIC, PUBLIC_LIMITS, REPO_ROOT, SCENES_DIR
from .frames import probe
from .jobs import JobManager

MAX_UPLOAD_BYTES = (PUBLIC_LIMITS["max_upload_mb"] if PUBLIC else 1500) * 1_000_000
# Live-event streams end after this long and the browser resumes with ?since=N, which keeps
# them inside hosting proxies' request timeouts (Modal's web endpoints cap requests at 150 s).
STREAM_MAX_S = 100
SAMPLE_VIDEOS = DATA_DIR / "test-videos"
WEB_DIST = REPO_ROOT / "apps" / "web" / "dist"
WEB_PUBLIC = REPO_ROOT / "apps" / "web" / "public"

manager: JobManager


@asynccontextmanager
async def lifespan(_: FastAPI):
    global manager
    manager = JobManager()
    manager.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="3DDD", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

_recent_by_ip: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def _check_public_limits(request: Request, backend: str, preset: str, gpu: str) -> None:
    """Hosted mode: anyone can upload, so keep each visitor (and the day) inside a small budget."""
    if backend != "modal":
        raise HTTPException(400, "The hosted version trains on cloud GPUs only")
    if preset not in PUBLIC_LIMITS["presets"]:
        raise HTTPException(400, f"Preset '{preset}' isn't available on the hosted version")
    if gpu not in PUBLIC_LIMITS["gpus"]:
        raise HTTPException(400, f"GPU '{gpu}' isn't available on the hosted version")
    if manager.queued_count() >= PUBLIC_LIMITS["max_queued"]:
        raise HTTPException(429, "The queue is full right now. Please try again in a few minutes")
    now = time.time()
    day = sum(1 for j in list(manager.jobs.values()) if not j.featured and now - j.created < 86400)
    if day >= PUBLIC_LIMITS["jobs_per_day"]:
        raise HTTPException(429, "Today's free capture limit has been reached. Please try again tomorrow")
    recent = _recent_by_ip[_client_ip(request)]
    while recent and now - recent[0] > 3600:
        recent.popleft()
    if len(recent) >= PUBLIC_LIMITS["jobs_per_ip_per_hour"]:
        raise HTTPException(429, "You've hit the hourly limit for new captures. Please try again later")


def _require_owner(job, token: str | None) -> None:
    if PUBLIC and (job.featured or not job.owned_by(token)):
        raise HTTPException(403, "Only the person who uploaded this capture can do that")


def _job(job_id: str):
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "local_trainer": local.available() and not PUBLIC, "public": PUBLIC,
            "limits": PUBLIC_LIMITS if PUBLIC else None}


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
    request: Request,
    video: UploadFile | None = File(None),
    sample: str | None = Form(None),
    name: str = Form(""),
    backend: str = Form("modal"),
    preset: str = Form("fast"),
    gpu: str = Form("L4"),
) -> dict:
    if PUBLIC:
        _check_public_limits(request, backend, preset, gpu)
    tmp = Path(tempfile.mkdtemp(dir=DATA_DIR))
    try:  # the temp dir is removed on every path: success, validation errors, client disconnects
        if video is not None:
            suffix = Path(video.filename or "capture.mp4").suffix.lower() or ".mp4"
            dst = tmp / f"upload{suffix}"
            size = 0
            with dst.open("wb") as f:
                while chunk := await video.read(4 * 1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"Video is too large (max {MAX_UPLOAD_BYTES // 1_000_000} MB)")
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
        if PUBLIC:
            try:
                seconds = probe(dst).duration
            except Exception as e:  # noqa: BLE001 - ffprobe rejects anything that isn't a video
                raise HTTPException(400, "That file doesn't look like a video") from e
            if seconds > PUBLIC_LIMITS["max_video_seconds"]:
                raise HTTPException(400, f"Please keep videos under {PUBLIC_LIMITS['max_video_seconds']} seconds")
        token = secrets.token_urlsafe(24)
        try:
            job = manager.create(dst, name[:80], backend, preset, gpu, owner_token=token)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        _recent_by_ip[_client_ip(request)].append(time.time())
        # The owner token is returned exactly once; only its hash is stored.
        return {**manager.snapshot(job), "owner_token": token}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/api/jobs")
def list_jobs(ids: str | None = None) -> list[dict]:
    jobs = sorted(list(manager.jobs.values()), key=lambda j: -j.created)
    if ids is not None:
        wanted = set(ids.split(","))
        jobs = [j for j in jobs if j.id in wanted]
    elif PUBLIC:  # hosted: never list strangers' uploads
        jobs = [j for j in jobs if j.featured]
    return [manager.snapshot(j) for j in jobs]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return manager.snapshot(_job(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, x_owner_token: str | None = Header(None)) -> dict:
    _require_owner(_job(job_id), x_owner_token)
    manager.cancel(job_id)
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, x_owner_token: str | None = Header(None)) -> dict:
    job = _job(job_id)
    _require_owner(job, x_owner_token)
    if job.status in ("queued", "running"):
        raise HTTPException(409, "Cancel the job first")
    manager.delete(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, since: int = 0) -> StreamingResponse:
    """Server-sent events: history from event #`since`, then live events. Each stream ends after
    STREAM_MAX_S; the client reconnects with the number of events it already has."""
    _job(job_id)

    async def stream():
        past, q = manager.subscribe(job_id)
        started = time.monotonic()
        try:
            for ev in past[max(0, since):]:
                yield f"data: {json.dumps(ev)}\n\n"
            yield 'data: {"type": "replay_done"}\n\n'
            while not await request.is_disconnected() and time.monotonic() - started < STREAM_MAX_S:
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
def scenes(featured: bool | None = None, ids: str | None = None) -> list[dict]:
    if featured is None and ids is None:
        featured = True if PUBLIC else None
    wanted = set(ids.split(",")) if ids is not None else None
    out = []
    for d in sorted(SCENES_DIR.iterdir(), key=lambda p: -p.stat().st_mtime) if SCENES_DIR.exists() else []:
        if not (d / "scene.splat").exists():
            continue
        job = manager.jobs.get(d.name)
        if wanted is not None and d.name not in wanted:
            continue
        if featured and not (job and job.featured):
            continue
        meta = json.loads((d / "scene.json").read_text(encoding="utf-8")) if (d / "scene.json").exists() else {}
        out.append({
            "id": d.name,
            "name": job.name if job else d.name,
            "featured": bool(job and job.featured),
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
        # Built assets first, then public/ (favicon, dev sample scenes), else the SPA shell.
        for root in (WEB_DIST, WEB_PUBLIC):
            f = (root / path).resolve()
            if path and f.is_file() and root.resolve() in f.parents:
                return FileResponse(f)
        return FileResponse(WEB_DIST / "index.html")
