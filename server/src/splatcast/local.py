"""Stage 3 (offline): train on this Mac's GPU with Brush (Rust + WebGPU/Metal).

Brush is quiet on stdout, so we drive progress from the snapshots it exports
every `export_every` steps and extrapolate the step rate in between.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator

from .config import TOOLS_DIR

BRUSH = TOOLS_DIR / "brush-app-aarch64-apple-darwin" / "brush_app"

PRESETS = {
    "preview": {"steps": 3000, "max_splats": 300_000, "max_side": 1000},
    "fast": {"steps": 5000, "max_splats": 500_000, "max_side": 1200},
    "quality": {"steps": 12000, "max_splats": 800_000, "max_side": 1400},
}


def available() -> bool:
    return BRUSH.exists()


def train_local(dataset: Path, out_dir: Path, preset: str, cancel: threading.Event) -> Iterator[dict]:
    cfg = PRESETS[preset]
    steps = cfg["steps"]
    export_every = 250 if steps <= 5000 else 500
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("export_*.ply"):
        old.unlink()
    cmd = [
        str(BRUSH), str(dataset.resolve()),
        "--total-steps", str(steps), "--max-splats", str(cfg["max_splats"]),
        "--max-resolution", str(cfg["max_side"]), "--export-every", str(export_every),
        "--export-path", ".", "--export-name", "export_{iter}.ply",
    ]
    yield {"type": "status", "msg": f"Training on this Mac's GPU with Brush ({steps} steps)"}
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=out_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    seen: set[str] = set()
    last_step, last_t = 0, t0
    rate = None
    try:
        while True:
            if cancel.is_set():
                proc.terminate()
                yield {"type": "cancelled", "step": last_step}
                return
            done = proc.poll() is not None
            exports = sorted(out_dir.glob("export_*.ply"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
            for p in exports:
                if p.name in seen:
                    continue
                # Wait until Brush has finished writing the file.
                size = p.stat().st_size
                time.sleep(0.3)
                if p.stat().st_size != size and not done:
                    continue
                seen.add(p.name)
                step = int(re.findall(r"\d+", p.stem)[0])
                now = time.time()
                if step > last_step:
                    rate = (step - last_step) / max(1e-3, now - last_t)
                last_step, last_t = step, now
                yield {"type": "snapshot_ply", "step": step, "path": p}
            now = time.time()
            est = last_step + (rate or 0) * (now - last_t)
            yield {"type": "progress", "step": int(min(steps, est)), "steps": steps,
                   "it_s": round(rate, 1) if rate else None, "elapsed": round(now - t0, 1)}
            if done:
                break
            time.sleep(1.0)
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        err = proc.stderr.read()[-2000:] if proc.stderr else ""
        raise RuntimeError(f"Brush exited with code {proc.returncode}: {err}")
    final = out_dir / f"export_{steps}.ply"
    if not final.exists():
        exports = sorted(out_dir.glob("export_*.ply"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        if not exports:
            raise RuntimeError("Brush produced no output")
        final = exports[-1]
    yield {"type": "final_ply", "path": final, "seconds": round(time.time() - t0, 1)}
