# 3DDD

**Film anything with your phone. Walk around inside it a few minutes later.**

**Live demo: https://3ddd-psi.vercel.app**. Open a scene, or upload your own video (phone uploads work through the QR code).

3DDD turns a casual phone video into a photoreal, explorable 3D scene using 3D Gaussian Splatting.
It solves the camera pose of every frame, trains several hundred thousand 3D Gaussians on a cloud GPU while
streaming the half-trained scene to your browser, and renders the result in a WebGPU renderer we wrote ourselves.

```
phone video ──► sharp-frame selection ──► structure-from-motion ──► Gaussian training ──► WebGPU viewer
   (ffmpeg)      (Laplacian variance)       (SIFT + GLOMAP)        (gsplat CUDA, MCMC)    (custom WGSL)
                                                   │                       │
                                                   └──── live over SSE ────┘──► browser watches it train
```

## Measured results

Test captures are Mip-NeRF 360 scenes, each re-encoded from its photos into a 1080p "phone video" and run
through the full pipeline. PSNR is measured on held-out views, i.e. every 10th frame, never trained on.

| Scene | Preset · GPU | Gaussians | Held-out PSNR | Video → 3D |
|---|---|---|---|---|
| Bonsai | quality · L40S | 1.0M | **30.7 dB** | 9m 15s |
| Lego bulldozer (kitchen) | quality · L40S | 1.0M | **29.9 dB** | 9m 39s |
| Garden table | quality · L40S | 1.0M | **27.2 dB** | 7m 23s |
| Counter | fast · L40S | 570k | **26.7 dB** | **3m 46s** |

| | |
|---|---|
| Camera solve (M4 MacBook, CPU) | 150/150 cameras registered on every scene, 0.3-0.5 px mean reprojection error, ~1.5-3 min |
| Training throughput (L40S) | 60-110 iterations/s; 7,000 steps in ~93 s |
| Viewer (M4, 1280×800) | **60 fps** with 1M splats at SH degree 3; GPU radix sort 1-2 ms/frame |
| Offline path (Brush on the M4's GPU, *preview*) | ~5 min end to end, no internet |
| Cost | ~$0.15 (fast) to ~$0.35 (quality) of GPU time per scene |

The loop-closure pairing (below) lifted the kitchen scene from 25.5 dB, where the reconstruction was
doubled and ghosted, to 29.9 dB.

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
- *Camera solve*: pycolmap SIFT → **window + keyframe loop-closure pairing** → **GLOMAP global SfM**, with an
  incremental-SfM fallback → undistortion to a pinhole camera model. Each frame is matched against its 10
  neighbours plus 15 keyframes spread across the capture. That links the end of a walk-around back to its
  start (without it, drift produces a doubled "ghost" scene) at about 3,300 pairs, compared with 11,000 for
  exhaustive matching. The solve runs in a subprocess: COLMAP's own log lines drive the live progress UI, and
  cancelling is an instant kill.
- *Gravity and framing*: the world "up" is the mean camera up vector. The look-at target is the
  least-squares intersection of all view rays, so scenes open level and centred.
- *Training* (`server/modal_app/trainer.py`): our own loop on gsplat's CUDA rasterizer. It uses
  **MCMC densification** with a hard Gaussian cap (bounded memory, time and file size), L1 + D-SSIM loss,
  opacity and scale regularisers, and progressive SH degree. It streams progress and `.splat` snapshots
  from the GPU as a generator. The final scene comes back in a compact transfer format (f16 attributes,
  8-bit quantised SH, about 3× smaller) and is unpacked to a standard PLY.
- *Offline fallback*: Brush (Rust/WebGPU) trains on the Mac's own GPU, and its periodic exports are turned
  into the same live snapshot stream.

**Spending guard** (`server/src/threeddd/budget.py`): before any GPU is requested, the higher of Modal's real
metered cost and a local ledger is checked against a cap (default $15). Each run's worst case is
*reserved* in the ledger before it starts and settled to the measured cost afterwards, so a crash
mid-run can't hide spend. The Modal hard timeout is set per call to exactly the reserved time. Runs are
limited to one GPU at a time with a 10 s scale-down, and cancellation is relayed to the GPU through a
`modal.Dict`. Orphaned runs are cancelled when the server restarts.

**Replay** (`#/replay/<job>`): every run's event log and snapshots are kept, so a finished run can be
replayed exactly as it streamed live, time-warped so each stage gets screen time. It's insurance against
venue Wi-Fi.

## Running it

Works on **macOS (Apple Silicon), Windows (x64) and Linux (x64)**. You need [uv](https://docs.astral.sh/uv/),
Node.js 22+ with `pnpm`, `ffmpeg`, and a Modal account for cloud GPU training (`cd server && uv run modal token new`).
The viewer needs a WebGPU browser: Chrome, Edge, or Safari 26+.

| | Install ffmpeg | Start |
|---|---|---|
| macOS | `brew install ffmpeg uv` | `./run.sh` |
| Linux | `sudo apt install ffmpeg` | `./run.sh` |
| Windows | `winget install --id Gyan.FFmpeg -e` | `powershell -ExecutionPolicy Bypass -File .\run.ps1` |

Then open http://localhost:8000. `--dev` / `-Dev` starts hot reload (API on :8000, UI on :5280).

The start scripts run `uv run python -m threeddd.setup_tools`, which checks prerequisites and downloads the
offline trainer, [Brush](https://github.com/ArthurBrussee/brush), for your OS, verifying its SHA-256 checksum. Offline
training uses Metal on macOS, DirectX 12 or Vulkan on Windows, and Vulkan on Linux. Brush has no build for ARM
Linux or Intel Macs; those machines use the cloud trainer.

Deploy the GPU trainer once, and again after editing `server/modal_app/trainer.py`:

```bash
cd server && uv run modal deploy modal_app/trainer.py
```

## Hosted version

The public site is a static build on Vercel talking to a backend on Modal:

```
Vercel (apps/web, static)  ──►  Modal "threeddd-web"                    ──►  Modal "threeddd-trainer"
                                  web:   FastAPI app, 2 CPUs, jobs in a        train: GPU training (L4 / L40S)
                                         Volume, live events over SSE
                                  solve: camera solve on 8 CPUs, per job
```

```bash
cd server && uv run modal deploy modal_app/web.py                 # API + camera-solve function
cd apps/web && vercel env add VITE_API_BASE production            # the web endpoint URL printed above
cd apps/web && vercel deploy --prod
```

Because anyone can upload, hosted mode (`THREEDDD_PUBLIC=1`) is locked down:
- only the *preview* and *fast* presets, on L4 or L40S GPUs;
- videos up to 120 s / 300 MB, checked with ffprobe;
- 3 captures per visitor per hour, 25 per day, at most 3 queued;
- uploads are private: the gallery lists only featured scenes, and each uploader gets an owner token that
  is needed to cancel or delete their own job;
- every run still goes through the same budget guard.

Live event streams end every 100 s and the browser resumes them from an event cursor. This keeps them
inside Modal's 150 s request limit. Big scenes open from the compact `.splat` first, and the full-colour
`.ply` streams in behind it.

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
server/src/threeddd frames.py · poses.py · remote.py · cloud.py · local.py · jobs.py · api.py · budget.py · sceneinfo.py · setup_tools.py
server/modal_app/    trainer.py (GPU training) · web.py (hosted API + cloud camera solve)
data/                jobs/, scenes/, test-videos/, modal_ledger.json   (git-ignored)
```
