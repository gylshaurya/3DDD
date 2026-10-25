"""3DDD hosted backend on Modal: the public API plus a CPU function for the camera solve.

    cd server && uv run modal deploy modal_app/web.py

* `web`   - the FastAPI app (jobs, live events, scene files). One small container holds the job
            queue in memory and keeps jobs/scenes on the `threeddd-data` volume.
* `solve` - structure-from-motion on 8 CPU cores, started per job; streams progress back.
GPU training is the separate `threeddd-trainer` app (modal_app/trainer.py).
"""

from __future__ import annotations

import modal

app = modal.App("threeddd-web")
volume = modal.Volume.from_name("threeddd-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi==0.162.0", "starlette==1.10.0", "pydantic==2.14.1", "python-multipart==0.0.32",
        "numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0", "plyfile==1.1.5",
        "pycolmap==4.2.1", "modal==1.6.2",
    )
    .env({
        "THREEDDD_DATA": "/data",
        "THREEDDD_REPO_ROOT": "/root",
        "THREEDDD_PUBLIC": "1",
        "THREEDDD_REMOTE_SFM": "1",
        "THREEDDD_CORS_ORIGINS": "*",
        "THREEDDD_SFM_FAST": "1",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_python_source("threeddd")
)


# A single container on purpose: the job queue and live-event fan-out live in its memory.
# scaledown_window keeps it alive long enough to finish a job after the uploader closes the tab.
@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=4096,
              max_containers=1, scaledown_window=1200, timeout=24 * 3600)
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def web():
    from threeddd import api, config

    config.on_persist(volume.commit)
    return api.app


@app.function(image=image, cpu=8, memory=16384, timeout=15 * 60, max_containers=2, scaledown_window=10)
def solve(frames_tar: bytes):
    """Camera solve for one job. Yields progress events, then the solved dataset as a tar."""
    import io
    import queue
    import tarfile
    import threading
    from pathlib import Path

    from threeddd.poses import solve_poses_subprocess
    from threeddd.remote import tar_dir

    root = Path("/tmp/job")
    with tarfile.open(fileobj=io.BytesIO(frames_tar)) as t:
        t.extractall(root, filter="data")
    events: queue.Queue = queue.Queue()
    outcome: dict = {}

    def work() -> None:
        try:
            outcome["stats"] = solve_poses_subprocess(
                root / "frames", root / "sfm",
                lambda frac, msg: events.put({"type": "progress", "frac": frac, "msg": msg}),
                lambda: False,
            )
        except Exception as e:  # noqa: BLE001 - reported to the caller
            outcome["error"] = str(e)
        events.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (ev := events.get()) is not None:
        yield ev
    if "error" in outcome:
        yield {"type": "error", "msg": outcome["error"]}
        return
    sfm = root / "sfm"
    keep = [sfm / "sfm_preview.json", sfm / "sfm_stats.json"]
    keep += [p for p in (sfm / "dataset").rglob("*") if p.is_file()]
    yield {"type": "result", "data": tar_dir(sfm, keep)}
