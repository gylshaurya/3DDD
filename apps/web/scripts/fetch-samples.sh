#!/usr/bin/env bash
# Downloads public sample scenes used by the viewer tests into public/samples/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/../public/samples"
HF=https://huggingface.co
for f in train plush nike; do
  [ -f "$f.splat" ] || curl -fsSL -o "$f.splat" "$HF/cakewalk/splat-data/resolve/main/$f.splat"
done
# Real 3DGS training output (Mip-NeRF 360 bonsai, 7k iters, SH degree 3, ~1.16M splats, 287 MB)
[ -f bonsai-7k.ply ] || curl -fsSL -o bonsai-7k.ply "$HF/datasets/dylanebert/3dgs/resolve/main/bonsai/point_cloud/iteration_7000/point_cloud.ply"
cd ../..
node scripts/make-test-ply.mjs public/samples/train.splat public/samples
ls -la public/samples
