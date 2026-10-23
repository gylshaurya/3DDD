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
import shutil
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


def solve_poses(
    frames_dir: Path,
    work_dir: Path,
    max_features: int = 8192,
    max_image_size: int = 1600,
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

    progress(0.30, "Matching features between neighbouring frames")
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 12
    pairing.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=pairing)

    progress(0.55, "Solving camera poses (global SfM)")
    sparse = work_dir / "sparse"
    sparse.mkdir()
    rec, mapper = None, "global"
    try:
        rec = _largest(pycolmap.global_mapping(db, frames_dir, sparse))
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
    (work_dir / "sfm_stats.json").write_text(json.dumps(asdict(stats), indent=2))
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
    }))
