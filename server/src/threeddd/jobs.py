"""Job orchestration: one pipeline at a time, events fanned out to any number of viewers.

Every event is appended to the job's history (so a browser that connects late,
or reconnects, replays the whole story) and pushed to live subscribers.
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from . import budget, cloud, local
from .config import JOBS_DIR, SCENES_DIR
from .frames import extract_frames
from .poses import solve_poses_subprocess
from .sceneinfo import ply_to_splat, scene_info, sfm_preview_splat, spc_to_ply

STAGES = ["frames", "sfm", "train", "export"]
SNAPSHOTS_KEPT = 400  # keep the whole run so it can be replayed later (~6 MB each)
CONTACT_SHEET = 24


@dataclass
class Job:
    id: str
    name: str
    backend: str  # "modal" | "local"
    preset: str
    gpu: str
    created: float
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str | None = None
    stage_progress: float = 0.0
    message: str = ""
    error: str | None = None
    metrics: dict = field(default_factory=dict)
    scene: dict | None = None
    timings: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def public(self) -> dict:
        return asdict(self)


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.history: dict[str, list[dict]] = {}
        self.subscribers: dict[str, set[asyncio.Queue]] = {}
        self.cancel_flags: dict[str, threading.Event] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        # Re-entrant: guards job state, history and subscribers across the worker thread,
        # the FastAPI threadpool and the event loop.
        self._lock = threading.RLock()
        self._load_existing()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- persistence -------------------------------------------------------------------------
    def _load_existing(self) -> None:
        for jf in sorted(JOBS_DIR.glob("*/job.json")):
            try:
                job = Job(**json.loads(jf.read_text()))
            except (TypeError, json.JSONDecodeError):
                continue
            if job.status in ("queued", "running"):
                if job.status == "running" and job.backend == "modal":
                    # The GPU may still be training for a client that no longer exists: stop it.
                    threading.Thread(target=relay_cancel, args=(job.id,), daemon=True).start()
                job.status, job.error = "failed", "Server restarted while this job was running"
            self.jobs[job.id] = job
            ev = jf.parent / "events.jsonl"
            self.history[job.id] = _read_events(ev)

    def _save(self, job: Job) -> None:
        with self._lock:
            (job.dir / "job.json").write_text(json.dumps(job.public(), indent=2))

    def snapshot(self, job: Job) -> dict:
        """A consistent copy of a job for API readers (the worker mutates nested dicts)."""
        with self._lock:
            return job.public()

    # ---- events ------------------------------------------------------------------------------
    def publish(self, job: Job, event: dict) -> None:
        event = {"t": round(time.time(), 3), **event}
        with self._lock:
            self.history.setdefault(job.id, []).append(event)
            with (job.dir / "events.jsonl").open("a") as f:
                f.write(json.dumps(event) + "\n")
            subs = list(self.subscribers.get(job.id, ()))
        if self.loop:
            for q in subs:
                self.loop.call_soon_threadsafe(q.put_nowait, event)

    def subscribe(self, job_id: str) -> tuple[list[dict], asyncio.Queue]:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self.subscribers.setdefault(job_id, set()).add(q)
            past = list(self.history.get(job_id, []))
        return past, q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            self.subscribers.get(job_id, set()).discard(q)

    def _update(self, job: Job, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(job, k, v)
            self._save(job)
            self.publish(job, {"type": "job", "job": job.public()})

    # ---- public API --------------------------------------------------------------------------
    def create(self, video: Path, name: str, backend: str, preset: str, gpu: str) -> Job:
        if backend == "local" and not local.available():
            raise ValueError("Local Brush trainer is not installed")
        if preset not in cloud.PRESETS:
            raise ValueError(f"Unknown preset {preset}")
        if backend == "modal" and gpu not in budget.GPU_RATES:
            raise ValueError(f"Unknown GPU {gpu}")
        job = Job(id=time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4], name=name or "Untitled scene",
                  backend=backend, preset=preset, gpu=gpu, created=time.time())
        job.dir.mkdir(parents=True)
        shutil.move(str(video), job.dir / ("input" + (video.suffix or ".mp4")))
        self.jobs[job.id] = job
        self.history[job.id] = []
        self.cancel_flags[job.id] = threading.Event()
        self._update(job, message="Queued")
        self.queue.put(job.id)
        return job

    def cancel(self, job_id: str) -> None:
        job = self.jobs[job_id]
        flag = self.cancel_flags.setdefault(job_id, threading.Event())
        flag.set()
        if job.backend == "modal" and job.status == "running":
            threading.Thread(target=relay_cancel, args=(job_id,), daemon=True).start()
        if job.status == "queued":
            self._update(job, status="cancelled", message="Cancelled")

    def delete(self, job_id: str) -> None:
        job = self.jobs.pop(job_id)
        self.cancel_flags.get(job_id, threading.Event()).set()
        shutil.rmtree(job.dir, ignore_errors=True)
        shutil.rmtree(SCENES_DIR / job_id, ignore_errors=True)

    # ---- pipeline ----------------------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if not job or job.status != "queued":
                continue
            try:
                self._run(job)
            except Cancelled:
                self._update(job, status="cancelled", message="Cancelled", stage=None)
            except Exception as e:  # noqa: BLE001 - any failure becomes a visible job error
                traceback.print_exc()
                self.publish(job, {"type": "log", "level": "error", "msg": str(e)})
                self._update(job, status="failed", error=str(e), message="Failed")

    def _check(self, job: Job) -> None:
        if self.cancel_flags.get(job.id, threading.Event()).is_set():
            raise Cancelled()

    def _stage(self, job: Job, stage: str) -> None:
        self._check(job)
        with self._lock:
            job.timings[stage] = {"start": time.time()}
        self._update(job, stage=stage, stage_progress=0.0)
        self.publish(job, {"type": "stage", "stage": stage})

    def _end_stage(self, job: Job, stage: str) -> None:
        with self._lock:
            t = job.timings[stage]
            t["seconds"] = round(time.time() - t["start"], 1)
        self._save(job)

    def _progress_cb(self, job: Job):
        # Counters like "image 12/150" change constantly: log each *phase* once, and
        # throttle progress updates so a 150-frame solve doesn't emit 300 events.
        last = {"t": 0.0, "phase": None}

        def cb(frac: float, msg: str) -> None:
            self._check(job)
            now = time.time()
            phase = re.sub(r"\d+", "#", msg)
            if phase != last["phase"]:
                self.publish(job, {"type": "log", "msg": msg})
            if phase != last["phase"] or now - last["t"] > 0.4:
                job.stage_progress, job.message = frac, msg
                self.publish(job, {"type": "stage_progress", "stage": job.stage, "progress": frac, "msg": msg})
                last["t"] = now
            last["phase"] = phase

        return cb

    def _run(self, job: Job) -> None:
        d = job.dir
        self._update(job, status="running", message="Starting")
        video = next(d.glob("input.*"))

        # 1. frames
        self._stage(job, "frames")
        frames = extract_frames(video, d / "frames", progress=self._progress_cb(job))
        thumb = Image.open(frames[len(frames) // 2])
        thumb.thumbnail((640, 640))
        thumb.save(d / "thumb.jpg", quality=85)
        self.publish(job, {"type": "thumb", "url": self.url(job, "thumb.jpg")})
        sheet = d / "contact"
        sheet.mkdir(exist_ok=True)
        picks = frames[:: max(1, len(frames) // CONTACT_SHEET)][:CONTACT_SHEET]
        for k, f in enumerate(picks):
            im = Image.open(f)
            im.thumbnail((320, 320))
            im.save(sheet / f"{k:02d}.jpg", quality=80)
        self.publish(job, {"type": "frames", "count": len(frames),
                           "urls": [self.url(job, f"contact/{k:02d}.jpg") for k in range(len(picks))]})
        self._end_stage(job, "frames")

        # 2. camera poses
        self._stage(job, "sfm")
        flag = self.cancel_flags.setdefault(job.id, threading.Event())
        stats = solve_poses_subprocess(d / "frames", d / "sfm", self._progress_cb(job), flag.is_set)
        info = scene_info(d / "sfm" / "sfm_preview.json")
        (d / "scene.json").write_text(json.dumps(info))
        (d / "sfm_preview.splat").write_bytes(sfm_preview_splat(d / "sfm" / "sfm_preview.json"))
        with self._lock:
            job.metrics["sfm"] = asdict(stats)
        self._update(job, scene=info)
        self.publish(job, {"type": "sfm", "url": self.url(job, "sfm_preview.splat"), "scene": info,
                           "stats": asdict(stats)})
        self._end_stage(job, "sfm")

        # 3. training
        self._stage(job, "train")
        dataset = d / "sfm" / "dataset"
        ply_path = d / "scene.ply"
        if job.backend == "modal":
            self._train_modal(job, dataset, ply_path)
        else:
            self._train_local(job, dataset, ply_path)
        self._end_stage(job, "train")

        # 4. export
        self._stage(job, "export")
        self.publish(job, {"type": "log", "msg": "Packing compact web scene"})
        (d / "scene.splat").write_bytes(ply_to_splat(ply_path))
        out = SCENES_DIR / job.id
        out.mkdir(parents=True, exist_ok=True)
        for f in ("scene.ply", "scene.splat", "scene.json", "thumb.jpg"):
            shutil.copy2(d / f, out / f)
        self._end_stage(job, "export")
        total = sum(t.get("seconds", 0) for t in job.timings.values())
        with self._lock:
            job.metrics["total_seconds"] = round(total, 1)
            job.metrics["ply_mb"] = round(ply_path.stat().st_size / 1e6, 1)
        self._update(job, status="done", stage=None, message=f"Done in {total:.0f}s")
        self.publish(job, {"type": "done", "ply": self.url(job, "scene.ply"), "splat": self.url(job, "scene.splat"),
                           "metrics": job.metrics})

    def _snapshot(self, job: Job, step: int, data: bytes) -> None:
        snaps = job.dir / "snapshots"
        snaps.mkdir(exist_ok=True)
        name = f"snap_{step:06d}.splat"
        (snaps / name).write_bytes(data)
        for old in sorted(snaps.glob("snap_*.splat"))[:-SNAPSHOTS_KEPT]:
            old.unlink()
        self.publish(job, {"type": "snapshot", "step": step, "url": self.url(job, f"snapshots/{name}"),
                           "splats": len(data) // 32})

    def _train_modal(self, job: Job, dataset: Path, ply_path: Path) -> None:
        ply = bytearray()
        compare_dir = job.dir / "compare"
        compare_dir.mkdir(exist_ok=True)
        events = cloud.train_cloud(dataset, job.preset, job.gpu, {"job_id": job.id, "eval_every": 10},
                                   log=lambda m: self.publish(job, {"type": "log", "msg": m}))
        flag = self.cancel_flags.setdefault(job.id, threading.Event())
        relayed_at = 0.0
        for ev in events:
            k = ev["type"]
            if flag.is_set() and time.time() - relayed_at > 10:
                # Cancelled locally but the GPU is still streaming: (re-)send the stop flag.
                threading.Thread(target=relay_cancel, args=(job.id,), daemon=True).start()
                relayed_at = time.time()
            if k == "progress":
                job.stage_progress = ev["step"] / ev["steps"]
                self.publish(job, {**ev, "type": "train_progress"})
            elif k == "snapshot":
                self._snapshot(job, ev["step"], ev["data"])
            elif k == "compare":
                name = f"compare_{ev['view']:03d}.jpg"
                (compare_dir / name).write_bytes(ev["data"])
                self.publish(job, {"type": "compare", "url": self.url(job, f"compare/{name}"),
                                   "held_out": ev["held_out"]})
            elif k == "metrics":
                with self._lock:
                    job.metrics["train"] = {x: v for x, v in ev.items() if x != "type"}
                self.publish(job, {"type": "metrics", "metrics": job.metrics})
            elif k == "file":
                ply += ev["data"]
            elif k == "status":
                self.publish(job, {"type": "log", "msg": ev["msg"]})
            elif k == "cancelled":
                raise Cancelled()
        if not ply:
            raise RuntimeError("Cloud trainer returned no scene")
        ply_path.write_bytes(spc_to_ply(bytes(ply)))

    def _train_local(self, job: Job, dataset: Path, ply_path: Path) -> None:
        flag = self.cancel_flags.setdefault(job.id, threading.Event())
        for ev in local.train_local(dataset, job.dir / "brush", job.preset, flag):
            k = ev["type"]
            if k == "progress":
                job.stage_progress = ev["step"] / ev["steps"]
                self.publish(job, {**ev, "type": "train_progress"})
            elif k == "snapshot_ply":
                self._snapshot(job, ev["step"], ply_to_splat(ev["path"], max_splats=300_000))
            elif k == "final_ply":
                shutil.copy2(ev["path"], ply_path)
                with self._lock:
                    job.metrics["train"] = {"train_seconds": ev["seconds"], "steps": local.PRESETS[job.preset]["steps"]}
            elif k == "status":
                self.publish(job, {"type": "log", "msg": ev["msg"]})
            elif k == "cancelled":
                raise Cancelled()

    @staticmethod
    def url(job: Job, rel: str) -> str:
        return f"/files/jobs/{job.id}/{rel}"


class Cancelled(Exception):
    pass


def relay_cancel(job_id: str, attempts: int = 5) -> bool:
    """Tell the remote trainer to stop (it polls this modal.Dict while training)."""
    import modal

    for i in range(attempts):
        try:
            modal.Dict.from_name("threeddd-cancel", create_if_missing=True)[job_id] = True
            return True
        except Exception:  # noqa: BLE001 - network hiccup; retry with backoff
            time.sleep(1.5 * (i + 1))
    return False


def _read_events(path: Path) -> list[dict]:
    """Event log reader that tolerates a torn final line (e.g. the server was killed mid-write)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
