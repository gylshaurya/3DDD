"""Stage 2 (hosted): run the camera solve on a cloud CPU function.

The hosted server is a small container, so the CPU-heavy structure-from-motion runs in a separate
8-core Modal function (`threeddd-web/solve`). Frames go up as a tar, live progress streams back, and
the undistorted dataset comes back as a tar extracted into the job's work dir, the same layout the
local solve produces.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Callable

from . import budget
from .poses import SfmStats

APP_NAME = "threeddd-web"
SFM_TIMEOUT_S = 15 * 60

Progress = Callable[[float, str], None]


def tar_dir(root: Path, members: list[Path]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for p in members:
            t.add(p, arcname=str(p.relative_to(root)))
    return buf.getvalue()


def safe_extract(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as t:
        for m in t.getmembers():
            target = (dest / m.name).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise RuntimeError(f"Unsafe path in solve result: {m.name}")
        t.extractall(dest, filter="data")


def solve_poses_remote(frames_dir: Path, work_dir: Path, progress: Progress,
                       should_cancel: Callable[[], bool]) -> SfmStats:
    import modal

    frames = sorted(frames_dir.glob("*.jpg"))
    payload = tar_dir(frames_dir.parent, frames)
    budget.check_budget("CPU8", SFM_TIMEOUT_S)
    progress(0.01, f"Sending {len(frames)} frames to a cloud CPU for the camera solve")
    fn = modal.Function.from_name(APP_NAME, "solve")
    result: bytes | None = None
    with budget.Run("CPU8", f"sfm:{work_dir.parent.name}", SFM_TIMEOUT_S):
        gen = fn.remote_gen(payload)
        try:
            for ev in gen:
                if should_cancel():
                    progress(0.0, "Cancelled")  # raises Cancelled
                if ev["type"] == "progress":
                    progress(ev["frac"], ev["msg"])
                elif ev["type"] == "error":
                    raise RuntimeError(ev["msg"])
                elif ev["type"] == "result":
                    result = ev["data"]
        finally:
            gen.close()
    if result is None:
        raise RuntimeError("Camera solve returned no result")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    safe_extract(result, work_dir)
    return SfmStats(**json.loads((work_dir / "sfm_stats.json").read_text(encoding="utf-8")))
