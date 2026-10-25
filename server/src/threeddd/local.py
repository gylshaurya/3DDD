"""Stage 3 (offline): train on this computer's GPU with Brush.

Brush prints little, so progress comes from the snapshots it exports.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator

from .config import TOOLS_DIR

BRUSH_VERSION = "v0.3.0"
# (system, machine) to Brush release target.
TARGETS = {
    ("darwin", "arm64"): "aarch64-apple-darwin",
    ("windows", "amd64"): "x86_64-pc-windows-msvc",
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
}
_MACHINE_ALIASES = {"x86_64": {"amd64", "x86_64", "x64"}, "amd64": {"amd64", "x86_64", "x64"},
                    "arm64": {"arm64", "aarch64"}}

PRESETS = {
    "preview": {"steps": 3000, "max_splats": 300_000, "max_side": 1000},
    "fast": {"steps": 5000, "max_splats": 500_000, "max_side": 1200},
    "quality": {"steps": 12000, "max_splats": 800_000, "max_side": 1400},
}


def brush_target(system: str | None = None, machine: str | None = None) -> str | None:
    """The Brush release target for this OS/CPU, or None if Brush doesn't ship one."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    for (sys_name, arch), target in TARGETS.items():
        if system == sys_name and machine in _MACHINE_ALIASES[arch]:
            return target
    return None


def brush_path() -> Path | None:
    """Where the Brush binary is: $THREEDDD_BRUSH, then tools/brush-app-<target>/, then PATH."""
    override = os.environ.get("THREEDDD_BRUSH")
    if override:
        return Path(override) if Path(override).is_file() else None
    target = brush_target()
    if target:
        exe = "brush_app.exe" if target.endswith("windows-msvc") else "brush_app"
        candidate = TOOLS_DIR / f"brush-app-{target}" / exe
        if candidate.is_file():
            return candidate
    found = shutil.which("brush_app")
    return Path(found) if found else None


def available() -> bool:
    return brush_path() is not None


def train_local(dataset: Path, out_dir: Path, preset: str, cancel: threading.Event) -> Iterator[dict]:
    cfg = PRESETS[preset]
    steps = cfg["steps"]
    export_every = 250 if steps <= 5000 else 500
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("export_*.ply"):
        old.unlink()
    brush = brush_path()
    if brush is None:
        raise RuntimeError("Brush is not installed. Run: uv run python -m threeddd.setup_tools")
    cmd = [
        str(brush), str(dataset.resolve()),
        "--total-steps", str(steps), "--max-splats", str(cfg["max_splats"]),
        "--max-resolution", str(cfg["max_side"]), "--export-every", str(export_every),
        "--export-path", ".", "--export-name", "export_{iter}.ply",
    ]
    yield {"type": "status", "msg": f"Training on this computer's GPU with Brush ({steps} steps)"}
    t0 = time.time()
    # Write stderr to a file. An unread pipe can fill up and block Brush.
    log_path = out_dir / "brush.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=out_dir, stdout=subprocess.DEVNULL, stderr=log_file, text=True)
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
            proc.wait()
        log_file.close()
    if proc.returncode != 0:
        err = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"Brush exited with code {proc.returncode}: {err}")
    final = out_dir / f"export_{steps}.ply"
    if not final.exists():
        exports = sorted(out_dir.glob("export_*.ply"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        if not exports:
            raise RuntimeError("Brush produced no output")
        final = exports[-1]
    yield {"type": "final_ply", "path": final, "seconds": round(time.time() - t0, 1)}
