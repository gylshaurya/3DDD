"""Stage 3 (cloud): package the solved scene and stream training from Modal."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pycolmap
from PIL import Image

from . import budget

APP_NAME = "splatcast-trainer"

SETUP_HEADROOM_S = 240  # cold start, image decode, init, held-out eval and export

PRESETS = {
    # steps, gaussian cap, training image long side, wall-clock limit (s)
    "preview": {"steps": 3000, "cap_max": 300_000, "max_side": 1000, "max_seconds": 6 * 60},
    "fast": {"steps": 7000, "cap_max": 600_000, "max_side": 1400, "max_seconds": 10 * 60},
    "quality": {"steps": 15000, "cap_max": 1_000_000, "max_side": 1600, "max_seconds": 18 * 60},
}


def build_payload(dataset: Path, max_side: int) -> dict:
    """Undistorted COLMAP dataset → plain-bytes payload (no numpy pickles across Python versions)."""
    rec = pycolmap.Reconstruction(dataset / "sparse" / "0")
    images, Ks, w2cs = [], [], []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        if not img.has_pose:
            continue
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = cam.focal_length_x, cam.focal_length_y, cam.principal_point_x, cam.principal_point_y
        pil = Image.open(dataset / "images" / img.name).convert("RGB")
        s = min(1.0, max_side / max(pil.size))
        if s < 1.0:
            pil = pil.resize((round(pil.width * s), round(pil.height * s)), Image.LANCZOS)
            sx, sy = pil.width / cam.width, pil.height / cam.height
            fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=93)
        images.append(buf.getvalue())
        Ks.append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        m = np.eye(4)
        m[:3, :4] = np.asarray(img.cam_from_world().matrix())
        w2cs.append(m)
    pts = np.array([p.xyz for p in rec.points3D.values()], np.float32)
    cols = np.array([p.color for p in rec.points3D.values()], np.uint8)
    return {
        "images": images,
        "K": np.asarray(Ks, np.float32).tobytes(),
        "w2c": np.asarray(w2cs, np.float32).tobytes(),
        "points": pts.tobytes(),
        "colors": cols.tobytes(),
    }


def train_cloud(
    dataset: Path, preset: str = "fast", gpu: str = "L4", overrides: dict | None = None,
    log: Callable[[str], None] = print,
) -> Iterator[dict]:
    """Yields trainer events. The budget guard runs before any GPU is requested."""
    import modal

    cfg = dict(PRESETS[preset])
    cfg.update(overrides or {})
    # The Modal hard timeout is exactly what the guard reserves: training time plus setup/export headroom.
    hard_timeout = int(cfg["max_seconds"] + SETUP_HEADROOM_S)
    budget.check_budget(gpu, hard_timeout)
    payload = build_payload(dataset, cfg.pop("max_side"))
    mb = sum(len(b) for b in payload["images"]) / 1e6
    log(f"Uploading {len(payload['images'])} views ({mb:.0f} MB) to a Modal {gpu}")
    fn = modal.Function.from_name(APP_NAME, "train").with_options(gpu=gpu, timeout=hard_timeout)
    with budget.Run(gpu, f"train:{preset}:{dataset.parent.parent.name}", hard_timeout):
        yield from fn.remote_gen(payload, cfg)
