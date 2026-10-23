"""Splatcast cloud trainer: 3D Gaussian Splatting on a Modal GPU.

Our own training loop on top of gsplat's CUDA rasterizer:
  * SfM-point initialisation (k-NN scale init, SH0 colour init)
  * MCMC densification (Kheradmand et al. 2024) with a hard Gaussian cap, so
    memory, file size and runtime are bounded for a live demo
  * L1 + D-SSIM loss, opacity/scale regularisers, progressive SH degree
  * Streams progress + live .splat snapshots back to the caller as a generator,
    then the final full-SH .ply in chunks.

Deploy:  modal deploy server/modal_app/trainer.py
"""

from __future__ import annotations

import modal

APP_NAME = "splatcast-trainer"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "gsplat==1.5.3+pt24cu124",
        extra_index_url="https://docs.gsplat.studio/whl/pt24cu124",
    )
    .pip_install("numpy<2", "pillow", "jaxtyping", "rich", "ninja", "packaging")
)

# The heavy lifting happens in _train; the decorated function is a thin generator wrapper.
# max_containers=1 + timeout are part of the budget guard: one GPU at a time, never runaway.


@app.function(image=image, gpu="L4", timeout=25 * 60, max_containers=1, scaledown_window=10)
def train(payload: dict, cfg: dict):
    yield from _train(payload, cfg)


@app.function(image=image, gpu="L4", timeout=5 * 60, max_containers=1, retries=0, scaledown_window=10)
def smoke() -> dict:
    """Cheap check that CUDA + gsplat kernels work (a 4-Gaussian render)."""
    import torch
    from gsplat import rasterization

    dev = "cuda"
    means = torch.tensor([[0, 0, 2.0], [0.3, 0, 2.5], [-0.3, 0.2, 3], [0, -0.3, 2.2]], device=dev)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 4, device=dev)
    scales = torch.full((4, 3), 0.1, device=dev)
    opac = torch.full((4,), 0.9, device=dev)
    colors = torch.rand(4, 3, device=dev)
    K = torch.tensor([[[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]]], device=dev)
    img, alpha, _ = rasterization(means, quats, scales, opac, colors, torch.eye(4, device=dev)[None], K, 128, 128)
    return {"gpu": torch.cuda.get_device_name(), "alpha_max": float(alpha.max()), "shape": list(img.shape)}


def _ssim(img1, img2, window_size: int = 11):
    import torch
    import torch.nn.functional as F

    sigma = 1.5
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    window = (g @ g.t()).expand(img1.shape[1], 1, window_size, window_size).contiguous()
    pad, ch = window_size // 2, img1.shape[1]
    mu1 = F.conv2d(img1, window, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, window, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=pad, groups=ch) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=pad, groups=ch) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad, groups=ch) - mu12
    C1, C2 = 0.01**2, 0.03**2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.mean()


def _knn_scale(pts, k: int = 3, chunk: int = 2048):
    import torch

    out = torch.empty(len(pts), device=pts.device)
    for i in range(0, len(pts), chunk):
        d = torch.cdist(pts[i : i + chunk], pts)
        out[i : i + chunk] = d.topk(k + 1, largest=False).values[:, 1:].pow(2).mean(-1).sqrt()
    return out.clamp_min(1e-7)


def _pack(params) -> bytes:
    """Compact transfer format (~3x smaller than PLY); unpacked to a standard PLY by the client.

    Header: b"SPC1", N:u32, K:u32 (SH rest coeffs per channel); then means f32[N,3],
    log-scales f16[N,3], quats f16[N,4], opacity-logits f16[N], sh0 f16[N,3],
    shN min f32[K*3], shN max f32[K*3], shN u8[N,K*3] (per-coefficient min/max quantised).
    """
    import struct

    import numpy as np
    import torch

    with torch.no_grad():
        n = len(params["means"])
        shN = params["shN"].reshape(n, -1)
        k = params["shN"].shape[1]
        lo, hi = shN.min(0).values, shN.max(0).values
        q = ((shN - lo) / (hi - lo).clamp_min(1e-8) * 255).round().clamp(0, 255).to(torch.uint8)
        parts = [
            params["means"].float(), params["scales"].half(), params["quats"].half(),
            params["opacities"].half(), params["sh0"].reshape(n, 3).half(), lo.float(), hi.float(), q,
        ]
        body = b"".join(np.ascontiguousarray(t.cpu().numpy()).tobytes() for t in parts)
    return b"SPC1" + struct.pack("<II", n, k) + body


def _train(payload: dict, cfg: dict):
    import io
    import math
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from gsplat import MCMCStrategy, rasterization
    from gsplat.exporter import export_splats

    t_start = time.time()
    dev = "cuda"
    torch.manual_seed(0)
    steps = int(cfg.get("steps", 7000))
    cap = int(cfg.get("cap_max", 1_000_000))
    max_seconds = float(cfg.get("max_seconds", 15 * 60))
    snap_every_s = float(cfg.get("snapshot_interval_s", 6.0))
    snap_max = int(cfg.get("snapshot_max_splats", 300_000))
    sh_max = int(cfg.get("sh_degree", 3))
    eval_every = int(cfg.get("eval_every", 0))  # hold out every Nth view for test PSNR (0 = off)
    job_id = cfg.get("job_id")
    cancel_flags = modal.Dict.from_name("splatcast-cancel", create_if_missing=True) if job_id else None

    yield {"type": "status", "msg": f"GPU ready: {torch.cuda.get_device_name()}"}

    n = len(payload["images"])
    Ks = torch.from_numpy(np.frombuffer(payload["K"], np.float32).reshape(n, 3, 3).copy()).to(dev)
    viewmats = torch.from_numpy(np.frombuffer(payload["w2c"], np.float32).reshape(n, 4, 4).copy()).to(dev)
    imgs = torch.stack([
        torch.from_numpy(np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))) for b in payload["images"]
    ]).to(dev)  # [N,H,W,3] uint8, kept on GPU
    H, W = imgs.shape[1:3]
    test_ids = list(range(0, n, eval_every)) if eval_every else []
    train_ids = [i for i in range(n) if i not in set(test_ids)]

    pts = torch.from_numpy(np.frombuffer(payload["points"], np.float32).reshape(-1, 3).copy()).to(dev)
    cols = torch.from_numpy(np.frombuffer(payload["colors"], np.uint8).reshape(-1, 3).copy()).to(dev).float() / 255

    cam_centers = torch.linalg.inv(viewmats)[:, :3, 3]
    scene_scale = float((cam_centers - cam_centers.mean(0)).norm(dim=1).max() * 1.1)
    yield {"type": "status", "msg": f"{n} views at {W}x{H}, {len(pts)} SfM points, scene scale {scene_scale:.2f}"}

    # ---- init ------------------------------------------------------------------------------
    N = len(pts)
    C0 = 0.28209479177387814  # SH band-0 constant
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts.clone()),
        "scales": torch.nn.Parameter(torch.log(_knn_scale(pts)).unsqueeze(-1).repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.rand(N, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.5, device=dev))),
        "sh0": torch.nn.Parameter(((cols - 0.5) / C0).unsqueeze(1)),
        "shN": torch.nn.Parameter(torch.zeros(N, (sh_max + 1) ** 2 - 1, 3, device=dev)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / steps))
    strategy = MCMCStrategy(cap_max=cap, refine_stop_iter=int(steps * 0.8), refine_every=100)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    def render(ids, sh_degree):
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        return rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            colors, viewmats[ids], Ks[ids], W, H, sh_degree=sh_degree, packed=False,
        )

    def snapshot(max_splats: int, fmt: str) -> bytes:
        with torch.no_grad():
            op = torch.sigmoid(params["opacities"])
            idx = torch.arange(len(op), device=dev)
            if len(op) > max_splats:
                idx = op.topk(max_splats).indices
            return export_splats(
                params["means"][idx], params["scales"][idx], params["quats"][idx],
                params["opacities"][idx], params["sh0"][idx], params["shN"][idx], format=fmt,
            )

    def psnr_of(ids) -> float:
        vals = []
        with torch.no_grad():
            for i in ids:
                r, _, _ = render([i], sh_max)
                mse = F.mse_loss(r[0].clamp(0, 1), imgs[i].float() / 255)
                vals.append(float(-10 * torch.log10(mse)))
        return sum(vals) / max(1, len(vals))

    yield {"type": "status", "msg": f"Initialised {N} Gaussians; training {steps} steps (cap {cap:,})"}

    # ---- train -----------------------------------------------------------------------------
    order: list[int] = []
    last_snap, ema_loss, ema_psnr = 0.0, None, None
    t_train = time.time()
    step = 0
    for step in range(steps):
        if not order:
            order = [train_ids[i] for i in torch.randperm(len(train_ids)).tolist()]
        i = order.pop()
        sh_degree = min(step // 1000, sh_max)
        renders, _, info = render([i], sh_degree)
        pred = renders[0]
        gt = imgs[i].float() / 255
        l1 = (pred - gt).abs().mean()
        ssim = _ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * (1 - ssim)
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, lr=means_sched.get_last_lr()[0])

        with torch.no_grad():
            psnr = float(-10 * torch.log10(F.mse_loss(pred.detach().clamp(0, 1), gt)))
        ema_loss = float(loss) if ema_loss is None else 0.95 * ema_loss + 0.05 * float(loss)
        ema_psnr = psnr if ema_psnr is None else 0.95 * ema_psnr + 0.05 * psnr

        now = time.time()
        if step % 50 == 0 or step == steps - 1:
            yield {"type": "progress", "step": step + 1, "steps": steps, "loss": round(ema_loss, 5),
                   "psnr": round(ema_psnr, 2), "gaussians": len(params["means"]),
                   "it_s": round((step + 1) / (now - t_train), 1), "elapsed": round(now - t_start, 1)}
        if step >= 50 and now - last_snap > snap_every_s:
            data = snapshot(snap_max, "splat")
            yield {"type": "snapshot", "step": step + 1, "format": "splat", "data": data}
            last_snap = time.time()
        if cancel_flags is not None and step % 200 == 0 and cancel_flags.get(job_id):
            yield {"type": "cancelled", "step": step + 1}
            return
        if now - t_start > max_seconds:
            yield {"type": "status", "msg": f"Time budget reached at step {step + 1}; exporting"}
            break

    # ---- export ----------------------------------------------------------------------------
    metrics = {"steps": step + 1, "gaussians": len(params["means"]), "train_psnr": round(ema_psnr or 0, 2),
               "train_seconds": round(time.time() - t_train, 1)}
    if test_ids:
        metrics["test_psnr"] = round(psnr_of(test_ids), 2)
    yield {"type": "metrics", **metrics}
    # Side-by-side (photo | render) comparisons on views the model never trained on, when available.
    compare_ids = (test_ids or train_ids)[:: max(1, len(test_ids or train_ids) // 4)][:4]
    with torch.no_grad():
        for i in compare_ids:
            r, _, _ = render([i], sh_max)
            pair = torch.cat([imgs[i].float() / 255, r[0].clamp(0, 1)], dim=1)
            buf = io.BytesIO()
            Image.fromarray((pair * 255).byte().cpu().numpy()).save(buf, "JPEG", quality=90)
            yield {"type": "compare", "view": i, "held_out": i in test_ids, "data": buf.getvalue()}

    yield {"type": "status", "msg": "Exporting full-quality scene"}
    packed = _pack(params)
    chunk = 16 * 1024 * 1024
    for off in range(0, len(packed), chunk):
        yield {"type": "file", "name": "scene.spc", "data": packed[off : off + chunk],
               "last": off + chunk >= len(packed)}
    yield {"type": "done", "total_seconds": round(time.time() - t_start, 1)}
