"""Stage 1: turn a phone video into a small set of sharp, well-spaced frames.

We over-sample the video (~3x the target count), score every candidate with the
variance of the Laplacian (a cheap blur metric), then split the timeline into
`target` equal bins and keep the sharpest frame from each bin. This keeps
temporal coverage uniform while rejecting motion-blurred frames, which are the
#1 cause of failed camera solves on handheld footage.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

Progress = Callable[[float, str], None]


@dataclass
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int


def probe(video: Path) -> VideoInfo:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,duration:stream_side_data=rotation:format=duration",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    meta = json.loads(out)
    s = meta["streams"][0]
    num, den = (s.get("avg_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(s.get("duration") or meta.get("format", {}).get("duration") or 0)
    w, h = int(s["width"]), int(s["height"])
    rotation = 0
    for sd in s.get("side_data_list", []) or []:
        rotation = int(sd.get("rotation", 0) or 0)
    if abs(rotation) in (90, 270):
        w, h = h, w
    return VideoInfo(duration=duration, fps=fps, width=w, height=h)


def sharpness(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    scale = 640 / max(img.shape)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def select_sharpest(scores: list[float], target: int) -> list[int]:
    """Indices of the sharpest candidate in each of `target` equal-width bins."""
    n = len(scores)
    if n <= target:
        return list(range(n))
    edges = np.linspace(0, n, target + 1).astype(int)
    picks = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            picks.append(a + int(np.argmax(scores[a:b])))
    return picks


def extract_frames(
    video: Path,
    out_dir: Path,
    target: int = 150,
    max_side: int = 1600,
    oversample: int = 3,
    progress: Progress | None = None,
) -> list[Path]:
    progress = progress or (lambda f, m: None)
    info = probe(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = out_dir.parent / (out_dir.name + "_candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    cand_dir.mkdir(parents=True)

    want = target * oversample
    sample_fps = min(info.fps, max(0.5, want / max(info.duration, 1e-3))) if info.duration else info.fps
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    progress(0.0, f"Decoding {info.width}x{info.height} video ({info.duration:.1f}s @ {info.fps:.0f} fps)")
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video),
            "-vf", f"fps={sample_fps:.4f},{scale}",
            "-q:v", "2", "-progress", "pipe:1", str(cand_dir / "c_%05d.jpg"),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("out_time_us=") and info.duration:
            try:
                t = int(line.split("=")[1]) / 1e6
                progress(min(0.6, 0.6 * t / info.duration), "Decoding video")
            except ValueError:
                pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.read() if proc.stderr else ''}")

    cands = sorted(cand_dir.glob("c_*.jpg"))
    if len(cands) < 10:
        raise RuntimeError(f"Video produced only {len(cands)} frames; need a longer clip")
    scores = []
    for i, c in enumerate(cands):
        scores.append(sharpness(c))
        if i % 20 == 0:
            progress(0.6 + 0.35 * i / len(cands), f"Scoring sharpness {i}/{len(cands)}")

    picks = select_sharpest(scores, target)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    kept = []
    for k, idx in enumerate(picks):
        dst = out_dir / f"frame_{k:05d}.jpg"
        shutil.move(str(cands[idx]), dst)
        kept.append(dst)
    shutil.rmtree(cand_dir, ignore_errors=True)
    med = float(np.median(scores))
    kept_med = float(np.median([scores[i] for i in picks]))
    progress(1.0, f"Kept {len(kept)} of {len(cands)} frames (median sharpness {med:.0f} → {kept_med:.0f})")
    return kept
