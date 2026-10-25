"""Stage 2: structure-from-motion. Recover a camera pose for every frame.

SIFT features → sequential matching (frames come from a video, so each frame
only needs to be matched against its temporal neighbours plus a quadratic
stride for loop closure) → global SfM (GLOMAP, built into pycolmap 4), falling
back to incremental SfM if the global solver registers too few images.
Finally the images are undistorted to a pinhole model, which is what the
splat trainers expect.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pycolmap

Progress = Callable[[float, str], None]


@dataclass
class SfmStats:
    num_frames: int
    registered: int
    points3d: int
    mean_reproj_error: float
    mapper: str
    seconds: float


def _largest(recs: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction | None:
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())


def make_pairs(names: list[str], window: int = 10, keyframes: int = 15) -> list[tuple[str, str]]:
    """Which frame pairs to match.

    Video frames only overlap with their neighbours, so exhaustive matching wastes most of its time.
    But matching *only* neighbours never links the end of a walk-around back to its start, and the
    reconstruction drifts into a doubled "ghost" scene. So: a sliding window for local accuracy, plus
    every frame against a set of keyframes spread over the whole capture for loop closure.
    """
    n = len(names)
    out: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, min(n, i + window + 1)):
            out.add((i, j))
    for k in {round(x) for x in np.linspace(0, n - 1, min(n, keyframes))}:
        for i in range(n):
            if abs(i - k) > window:
                out.add((min(i, k), max(i, k)))
    return [(names[a], names[b]) for a, b in sorted(out)]


def global_options() -> pycolmap.GlobalPipelineOptions:
    """Global SfM settings. With THREEDDD_SFM_FAST=1 (used on cloud CPUs, which are much slower per
    core than Apple silicon for these mostly single-threaded solvers) we cap the number of 3D tracks
    and skip retriangulation: camera poses barely change, and the points only seed training."""
    opts = pycolmap.GlobalPipelineOptions()
    if os.environ.get("THREEDDD_SFM_FAST") == "1":
        opts.mapper.keep_max_num_tracks = 30_000
        opts.mapper.ba_num_iterations = 2
        opts.mapper.skip_retriangulation = True
    return opts


def solve_poses(
    frames_dir: Path,
    work_dir: Path,
    max_features: int = 4096,
    max_image_size: int = 1024,
    progress: Progress | None = None,
) -> SfmStats:
    progress = progress or (lambda f, m: None)
    t0 = time.time()
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    db = work_dir / "database.db"
    num_frames = len(list(frames_dir.glob("*.jpg")))

    progress(0.02, f"Extracting SIFT features from {num_frames} frames")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max_image_size
    extraction.sift.max_num_features = max_features
    pycolmap.extract_features(
        db, frames_dir, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extraction,
    )

    progress(0.30, "Matching neighbouring frames + loop-closure keyframes")
    names = sorted(p.name for p in frames_dir.glob("*.jpg"))
    pairs = make_pairs(names)
    pair_file = work_dir / "pairs.txt"
    pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = str(pair_file)
    pycolmap.match_image_pairs(db, pairing_options=pairing)

    progress(0.55, "Solving camera poses (global SfM)")
    sparse = work_dir / "sparse"
    sparse.mkdir()
    rec, mapper = None, "global"
    try:
        rec = _largest(pycolmap.global_mapping(db, frames_dir, sparse, global_options()))
    except Exception as e:  # noqa: BLE001 - the fallback below handles any solver failure
        progress(0.6, f"Global SfM failed ({e}); retrying incrementally")
    if rec is None or rec.num_reg_images() < 0.6 * num_frames:
        mapper = "incremental"
        shutil.rmtree(sparse)
        sparse.mkdir()
        done = [0]

        def on_image() -> None:
            done[0] += 1
            progress(0.6 + 0.3 * min(1.0, done[0] / num_frames), f"Registered {done[0]}/{num_frames} cameras")

        opts = pycolmap.IncrementalPipelineOptions()
        opts.multiple_models = False
        rec = _largest(pycolmap.incremental_mapping(db, frames_dir, sparse, opts, next_image_callback=on_image))
    if rec is None or rec.num_reg_images() < 10:
        raise RuntimeError(
            "Could not recover camera poses. Try a slower, steadier video that circles the object "
            "with plenty of texture in view."
        )

    model_dir = work_dir / "model"
    model_dir.mkdir(exist_ok=True)
    rec.write(model_dir)

    progress(0.92, "Undistorting images")
    dataset = work_dir / "dataset"
    pycolmap.undistort_images(dataset, model_dir, frames_dir)
    # Trainers (gsplat loaders, Brush) look for sparse/0.
    sp = dataset / "sparse"
    if sp.exists() and not (sp / "0").exists():
        (sp / "0").mkdir()
        for f in list(sp.iterdir()):
            if f.is_file():
                shutil.move(str(f), sp / "0" / f.name)

    final = pycolmap.Reconstruction(dataset / "sparse" / "0")
    export_preview(final, work_dir / "sfm_preview.json")
    stats = SfmStats(
        num_frames=num_frames,
        registered=final.num_reg_images(),
        points3d=final.num_points3D(),
        mean_reproj_error=float(final.compute_mean_reprojection_error()),
        mapper=mapper,
        seconds=time.time() - t0,
    )
    (work_dir / "sfm_stats.json").write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    progress(1.0, f"Registered {stats.registered}/{num_frames} cameras, {stats.points3d} points ({mapper})")
    return stats


def export_preview(rec: pycolmap.Reconstruction, out: Path, max_points: int = 60000) -> None:
    """Sparse points + camera frusta, so the UI can show the solve while training starts."""
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    cols = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    cams = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        if not img.has_pose:
            continue
        w2c = img.cam_from_world()
        R = np.asarray(w2c.rotation.matrix())
        center = -R.T @ np.asarray(w2c.translation)
        cam = rec.cameras[img.camera_id]
        cams.append({
            "name": img.name,
            "center": center.round(5).tolist(),
            "rotation": R.T.round(6).tolist(),  # camera-to-world rotation
            "fx": float(cam.focal_length_x), "width": cam.width, "height": cam.height,
        })
    out.write_text(json.dumps({
        "points": pts.round(4).flatten().tolist(),
        "colors": cols.flatten().tolist(),
        "cameras": cams,
    }), encoding="utf-8")


# ---- subprocess runner -------------------------------------------------------------------------
# Running the solve in a child process lets us (a) turn COLMAP's own log lines into fine-grained
# live progress and (b) cancel instantly by killing the process instead of unwinding native code.

_GLOG = [
    # (regex on the log line, progress fraction or callable(match) -> fraction, message template)
    (re.compile(r"Processed file \[(\d+)/(\d+)\]"), lambda m: 0.02 + 0.28 * int(m[1]) / int(m[2]),
     "Extracting SIFT features {1}/{2}"),
    (re.compile(r"Features:\s+(\d+)"), None, None),
    (re.compile(r"Processing image \[(\d+)/(\d+)\]"), lambda m: 0.30 + 0.25 * int(m[1]) / int(m[2]),
     "Matching image {1}/{2} against its neighbours"),
    (re.compile(r"Matching block \[(\d+)/(\d+)"), lambda m: 0.30 + 0.25 * int(m[1]) / int(m[2]),
     "Matching image pairs: block {1}/{2}"),
    (re.compile(r"=== Running rotation averaging ==="), 0.58, "Rotation averaging: solving every camera's orientation at once"),
    (re.compile(r"Established (\d+) tracks"), 0.64, "Linked features into {1} 3D tracks"),
    (re.compile(r"=== Running global positioning ==="), 0.67, "Global positioning: solving all camera centres and points jointly"),
    (re.compile(r"Global bundle adjustment iteration (\d) / (\d) finished"),
     lambda m: 0.72 + 0.14 * int(m[1]) / int(m[2]), "Bundle adjustment pass {1}/{2}: refining every pose and point"),
    (re.compile(r"=== Running iterative retriangulation"), 0.88, "Retriangulating and refining"),
    (re.compile(r"=== Running incremental"), 0.6, "Falling back to incremental SfM"),
]


def solve_poses_subprocess(frames_dir: Path, work_dir: Path, progress: Progress, should_cancel: Callable[[], bool]) -> SfmStats:
    proc = subprocess.Popen(
        [sys.executable, "-m", "threeddd.poses", str(frames_dir), str(work_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            lines.put(line.rstrip("\n"))
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    frac, features, error = 0.0, [], None
    try:
        while True:
            if should_cancel():
                progress(frac, "Cancelled")  # the callback raises Cancelled; finally kills the child
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith("@@"):
                kind, _, payload = line[2:].partition(" ")
                if kind == "PROGRESS":
                    p = json.loads(payload)
                    frac = max(frac, p["frac"])
                    progress(frac, p["msg"])
                elif kind == "ERROR":
                    error = payload
                continue
            for rx, f, msg in _GLOG:
                m = rx.search(line)
                if not m:
                    continue
                if msg is None:
                    features.append(int(m[1]))
                    break
                frac = max(frac, f(m) if callable(f) else f)
                text = msg.format(*([m[0]] + list(m.groups())))
                if features and "Matching" in text and len(features) > 1:
                    text += f" (avg {sum(features) // len(features)} SIFT features/frame)"
                progress(frac, text)
                break
    finally:
        # Any exit path (cancel, a raising progress callback, errors) must not leave the solver running.
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if proc.wait() != 0:
        raise RuntimeError(error or f"Camera solve crashed (exit code {proc.returncode})")
    return SfmStats(**json.loads((work_dir / "sfm_stats.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    def _report(frac: float, msg: str) -> None:
        print("@@PROGRESS " + json.dumps({"frac": frac, "msg": msg}), flush=True)

    try:
        solve_poses(Path(sys.argv[1]), Path(sys.argv[2]), progress=_report)
    except Exception as e:  # noqa: BLE001 - reported to the parent, which raises it
        print(f"@@ERROR {e}", flush=True)
        sys.exit(1)
