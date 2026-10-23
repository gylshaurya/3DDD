# Splatcast web

Vite + React + TypeScript app with a from-scratch **WebGPU 3D Gaussian Splatting renderer** (`src/viewer/`, no splat libraries).

## Run

```bash
pnpm install
pnpm samples          # downloads ~340 MB of public sample scenes into public/samples (gitignored)
pnpm dev              # http://localhost:5173 — app shell
                      # http://localhost:5173/viewer-test.html?url=/samples/train.splat&orbit=1  — bare viewer
pnpm test             # vitest unit tests (parsers, f16 packing, bounds, camera)
pnpm test:e2e         # Playwright on the real GPU (headless Chromium, WebGPU via Metal)
```

`viewer-test.html` query params: `url`, `orbit=1`, `up=auto|y-down|y-up|z-up|z-down`, `cpu=1` (force CPU-sort fallback), `hud=0`.

## Viewer API (`src/viewer/index.ts`)

```ts
const viewer = await createViewer(canvas, { up: 'auto', autoOrbit: false });
await viewer.load('/scene.ply');                    // URL (streams progressively), File/Blob, or ArrayBuffer
await viewer.setSplatsFromBuffer(buf, 'splat');      // live training snapshot: keeps camera, crossfades
viewer.setAutoOrbit(true); viewer.fitToScene(); viewer.lookAt(eye, target);
viewer.onStats(({ fps, splatCount, visibleCount, sortMs, gpuSort }) => …);
viewer.captureFrame();                               // PNG data URL
viewer.dispose();
```

React: `<SplatViewer src=… autoOrbit onReady={(v) => …} onStats={…} />` (`src/components/SplatViewer.tsx`).

## How it renders

1. **Load** (Web Worker): `.splat` / binary 3DGS `.ply` (SH 0–3, also plain point clouds) are streamed and packed into
   32-byte GPU records (f32 position, f16 color+opacity, f16 scale, snorm8 quaternion) plus f16 SH.
2. **Cull + project** (compute): EWA projection of the 3D covariance with the perspective Jacobian (+0.3 px low-pass),
   SH evaluation, frustum/near culling, opacity-aware quad extent, atomic compaction of visible splats.
3. **Sort** (compute): stable LSD radix sort, 4 × 8-bit passes, on GPU-side counts via indirect dispatch
   (per-tile histogram → multi-level Blelloch scan → 8× one-bit-split local sort + scatter).
4. **Draw**: one indirect instanced draw of eigenvector-aligned quads, premultiplied back-to-front blending.

A GPU sort self-test runs at startup; on failure the viewer falls back to a 16-bit counting sort in a worker.
Auto-framing uses robust percentiles; `up: 'auto'` estimates the ground plane normal via PCA.
