# Splatcast

**Film anything with your phone. Walk around inside it a few minutes later.**

Splatcast turns a casual phone video into a photoreal, explorable 3D scene using 3D Gaussian Splatting.
It solves the camera pose of every frame, trains several hundred thousand 3D Gaussians on a cloud GPU while
streaming the half-trained scene to your browser, and renders the result in a WebGPU renderer we wrote ourselves.

```
phone video ──► sharp-frame selection ──► structure-from-motion ──► Gaussian training ──► WebGPU viewer
   (ffmpeg)      (Laplacian variance)       (SIFT + GLOMAP)        (gsplat CUDA, MCMC)    (custom WGSL)
                                                   │                       │
                                                   └──── live over SSE ────┘──► browser watches it train
```

## Measured results

The test capture is the 240-frame "counter" scene from Mip-NeRF 360, re-encoded as a 30 s, 1080p video.

| | |
|---|---|
| Video → finished 3D scene (Modal L40S, *fast*) | **3 min 46 s** end to end |
| Camera solve (M4 MacBook, CPU) | 150/150 cameras registered, 0.48 px mean reprojection error, ~85 s |
| Training (7,000 steps, L40S) | ~93 s, 570k Gaussians |
| Quality on held-out views (never trained on) | **26.7 dB PSNR** |
| Viewer (M4, 1440×900, 570k splats, SH degree 3) | **60 fps**, GPU radix sort 1.2 ms/frame |
| Offline path (Brush on the M4's GPU, *preview*) | ~5 min end to end, no internet |

## What's technically interesting

**The WebGPU renderer** (`apps/web/src/viewer/`, no splat or 3D libraries):
- EWA splatting. Each 3D covariance is built from scale and quaternion and projected through the perspective
  Jacobian to a 2D conic, with a 0.3 px low-pass filter.
- Spherical harmonics up to degree 3 evaluated per splat in a compute shader, so colour changes with
  viewing angle (reflections, sheen).
- A **GPU radix sort** written from scratch in WGSL: 4×8-bit LSD passes, per-tile histograms,
  a multi-level Blelloch scan, indirect dispatch, and frustum culling with atomic compaction. A self-test
  runs at startup and falls back to a CPU counting sort in a worker if it fails.
- 32-byte packed splats (f16 covariance and colour, snorm8 quaternion); progressive streaming loads; live
  scene swaps with a crossfade; adaptive resolution.

**The pipeline** (`server/`):
- *Frame selection*: the video is over-sampled 3×. Each frame is scored by variance of the Laplacian (blur),
  and the sharpest frame in each time bin is kept.
- *Camera solve*: pycolmap SIFT → sequential and quadratic matching → **GLOMAP global SfM**, with an
  incremental-SfM fallback → undistortion to a pinhole camera model.
- *Gravity and framing*: the world "up" is the mean camera up vector. The look-at target is the
  least-squares intersection of all view rays, so scenes open level and centred.
- *Training* (`server/modal_app/trainer.py`): our own loop on gsplat's CUDA rasterizer. It uses
  **MCMC densification** with a hard Gaussian cap (bounded memory, time and file size), L1 + D-SSIM loss,
  opacity and scale regularisers, and progressive SH degree. It streams progress and `.splat` snapshots
  from the GPU as a generator. The final scene comes back in a compact transfer format (f16 attributes,
  8-bit quantised SH, about 3× smaller) and is unpacked to a standard PLY.
- *Offline fallback*: Brush (Rust/WebGPU) trains on the Mac's own GPU, and its periodic exports are turned
  into the same live snapshot stream.

**Spending guard** (`server/src/splatcast/budget.py`): before any GPU is requested, the higher of Modal's real
metered cost and a local ledger is checked against a cap (default $15). Runs are limited to one GPU
at a time, with hard timeouts, a 10 s scale-down, and cancellation relayed to the GPU through a
`modal.Dict`.

## Running it

Requirements: macOS on Apple Silicon (for the local trainer), `brew install ffmpeg uv`, `pnpm`, and a Modal
account (`modal token new`). Chrome, Edge or Safari 26 for WebGPU.

```bash
./run.sh          # builds the UI and serves everything at http://localhost:8000
./run.sh --dev    # hot reload: API on :8000, UI on http://localhost:5280
```

Deploy the GPU trainer once, and again after editing `server/modal_app/trainer.py`:

```bash
cd server && uv run modal deploy modal_app/trainer.py
```

Brush, the offline trainer, lives in `tools/` (not committed). To reinstall it:

```bash
mkdir -p tools && cd tools && gh release download v0.3.0 -R ArthurBrussee/brush -p 'brush-app-aarch64-apple-darwin.tar.xz' && tar xf brush-app-*.tar.xz
```

### Tests

```bash
cd server && uv run pytest              # pipeline units: transfer format, frame picking, gravity/target, budget guard
cd apps/web && pnpm test                # renderer units: parsers, f16 packing, bounds, camera
cd apps/web && pnpm test:e2e            # Playwright on the real GPU: renders, GPU sort vs Array.sort, live swap
cd apps/web && node scripts/e2e-job.mjs /tmp/run local preview   # drives the whole UI end to end
```

## Layout

```
apps/web/            React + Vite app and the WebGPU renderer (src/viewer)
server/src/splatcast frames.py · poses.py · cloud.py · local.py · jobs.py · api.py · budget.py · sceneinfo.py
server/modal_app/    trainer.py, the Modal GPU function
data/                jobs/, scenes/, test-videos/, modal_ledger.json   (git-ignored)
```
