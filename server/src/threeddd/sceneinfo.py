"""Scene metadata from the camera solve (up direction, start view) and .splat/.ply helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

SPLAT_DTYPE = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])


def encode_splat(pos, scale, rgba, quat_wxyz) -> bytes:
    """antimatter15 .splat: linear scales, quaternion (w,x,y,z) packed as q*128+128."""
    n = len(pos)
    out = np.empty(n, SPLAT_DTYPE)
    out["pos"] = pos
    out["scale"] = scale
    out["rgba"] = np.clip(rgba, 0, 255).astype(np.uint8)
    q = quat_wxyz / np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    out["rot"] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    return out.tobytes()


def ply_to_splat(ply_path: Path, max_splats: int | None = None) -> bytes:
    """Convert a 3DGS .ply to the compact .splat format (base colour only)."""
    v = PlyData.read(str(ply_path))["vertex"].data
    pos = np.stack([v["x"], v["y"], v["z"]], 1)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    opac = 1 / (1 + np.exp(-v["opacity"]))
    C0 = 0.28209479177387814
    rgb = 0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    rgba = np.concatenate([rgb, opac[:, None]], 1) * 255
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)
    order = np.argsort(-(scale.prod(1) * opac))
    if max_splats:
        order = order[:max_splats]
    return encode_splat(pos[order], scale[order], rgba[order], quat[order])


def scene_info(preview_json: Path) -> dict:
    data = json.loads(preview_json.read_text(encoding="utf-8"))
    cams = data["cameras"]
    centers = np.array([c["center"] for c in cams])
    rots = np.array([c["rotation"] for c in cams])  # camera-to-world
    up = -rots[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    fwd = rots[:, :, 2]
    pts = np.array(data["points"], np.float32).reshape(-1, 3)

    # Target: the point closest to all camera viewing rays.
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, fwd):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    try:
        target = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        target = np.median(pts, 0)
    # If the rays are nearly parallel, use the median point instead.
    if np.linalg.norm(target - centers.mean(0)) > 3 * np.linalg.norm(centers - centers.mean(0), axis=1).max():
        target = np.median(pts, 0)

    home = cams[len(cams) // 2]
    eye = np.array(home["center"])
    eye = target + (eye - target) * 1.15  # step back so the subject fits
    radius = float(np.linalg.norm(centers - target, axis=1).mean())
    return {
        "up": up.round(5).tolist(),
        "target": target.round(5).tolist(),
        "home_eye": eye.round(5).tolist(),
        "orbit_radius": round(radius, 4),
        "num_cameras": len(cams),
    }


def sfm_preview_splat(preview_json: Path) -> bytes:
    """Sparse points as tiny splats + every camera drawn as a glowing frustum made of splats."""
    data = json.loads(preview_json.read_text(encoding="utf-8"))
    pts = np.array(data["points"], np.float32).reshape(-1, 3)
    cols = np.array(data["colors"], np.uint8).reshape(-1, 3)
    cams = data["cameras"]
    centers = np.array([c["center"] for c in cams])
    extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean())

    s = extent * 0.0035
    pos = [pts]
    scale = [np.full((len(pts), 3), s, np.float32)]
    rgba = [np.concatenate([cols, np.full((len(pts), 1), 255, np.uint8)], 1)]

    fr = extent * 0.06  # frustum depth
    edge_pts = []
    for c in cams:
        C = np.array(c["center"])
        R = np.array(c["rotation"])
        hw = 0.5 * c["width"] / c["fx"]
        hh = 0.5 * c["height"] / c["fx"]
        corners = [C + R @ (np.array([x * hw, y * hh, 1.0]) * fr) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        segs = [(C, k) for k in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        for a, b in segs:
            t = np.linspace(0, 1, 12)[:, None]
            edge_pts.append(a + (b - a) * t)
    if edge_pts:
        e = np.concatenate(edge_pts).astype(np.float32)
        pos.append(e)
        scale.append(np.full((len(e), 3), s * 0.6, np.float32))
        rgba.append(np.tile(np.array([[255, 196, 64, 255]], np.uint8), (len(e), 1)))

    pos_a = np.concatenate(pos)
    quat = np.tile(np.array([[1, 0, 0, 0]], np.float32), (len(pos_a), 1))
    return encode_splat(pos_a, np.concatenate(scale), np.concatenate(rgba).astype(np.float32), quat)


def spc_to_ply(data: bytes) -> bytes:
    """Unpack the trainer's compact SPC1 transfer format into a standard 3DGS binary PLY."""
    if data[:4] != b"SPC1":
        raise ValueError("Not an SPC1 payload")
    n, k = np.frombuffer(data, "<u4", 2, 4)
    n, k = int(n), int(k)
    off = 12

    def take(dtype, count):
        nonlocal off
        arr = np.frombuffer(data, dtype, count, off)
        off += arr.nbytes
        return arr

    means = take("<f4", n * 3).reshape(n, 3)
    scales = take("<f2", n * 3).reshape(n, 3).astype(np.float32)
    quats = take("<f2", n * 4).reshape(n, 4).astype(np.float32)
    opac = take("<f2", n).astype(np.float32)
    sh0 = take("<f2", n * 3).reshape(n, 3).astype(np.float32)
    lo = take("<f4", k * 3)
    hi = take("<f4", k * 3)
    q = take("u1", n * k * 3).reshape(n, k * 3).astype(np.float32)
    shN = (lo + q / 255 * (hi - lo)).reshape(n, k, 3)
    # PLY convention stores SH rest channel-major: f_rest_[c*K + i].
    rest = shN.transpose(0, 2, 1).reshape(n, k * 3)

    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(k * 3)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = np.concatenate([means, np.zeros((n, 3), np.float32), sh0, rest, opac[:, None], scales, quats], 1)
    header = "ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    return header.encode() + np.ascontiguousarray(cols, dtype="<f4").tobytes()
