import { describe, expect, it } from 'vitest';
import { computeBounds, estimateUp } from './bounds';
import type { Vec3 } from './math';

function samples(points: Vec3[], opacity = 1): Float32Array {
  const out = new Float32Array(points.length * 4);
  points.forEach((p, i) => out.set([...p, opacity], i * 4));
  return out;
}

// deterministic PRNG
function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000);
}

describe('computeBounds', () => {
  it('is robust to far outliers', () => {
    const r = rng(1);
    const pts: Vec3[] = [];
    for (let i = 0; i < 2000; i++) pts.push([r() * 2 - 1, r() * 2 - 1, r() * 2 - 1]);
    for (let i = 0; i < 40; i++) pts.push([1000 + i, -5000, 9000]); // floaters
    const b = computeBounds(samples(pts))!;
    for (const c of b.center) expect(Math.abs(c)).toBeLessThan(0.15);
    expect(b.radius).toBeLessThan(2);
    expect(b.focusRadius).toBeGreaterThan(0);
  });
  it('ignores near-transparent splats when enough opaque ones exist', () => {
    const opaque = samples(Array.from({ length: 100 }, (_, i) => [i % 10, 0, Math.floor(i / 10)] as Vec3), 1);
    const ghost = samples(Array.from({ length: 1000 }, () => [500, 500, 500] as Vec3), 0.05);
    const all = new Float32Array(opaque.length + ghost.length);
    all.set(opaque);
    all.set(ghost, opaque.length);
    const b = computeBounds(all)!;
    expect(b.center[0]).toBeLessThan(10);
  });
  it('returns null for empty input', () => {
    expect(computeBounds(new Float32Array(0))).toBeNull();
  });
});

describe('estimateUp', () => {
  it('recovers a tilted ground plane normal, oriented toward −y', () => {
    const r = rng(7);
    const n: Vec3 = [0.3, -0.9, 0.3];
    const len = Math.hypot(...n);
    const nn = n.map((v) => v / len) as Vec3;
    // basis of the plane
    const t1 = [1, 0, 0].map((v, i) => v - nn[0] * nn[i]);
    const l1 = Math.hypot(...t1);
    const u = t1.map((v) => v / l1);
    const w = [nn[1] * u[2] - nn[2] * u[1], nn[2] * u[0] - nn[0] * u[2], nn[0] * u[1] - nn[1] * u[0]];
    const pts: Vec3[] = [];
    for (let i = 0; i < 4000; i++) {
      const a = r() * 6 - 3, b = r() * 6 - 3, h = (r() - 0.5) * 0.2;
      pts.push([0, 1, 2].map((k) => a * u[k] + b * w[k] + h * nn[k]) as Vec3);
    }
    const s = samples(pts);
    const up = estimateUp(s, computeBounds(s)!);
    const dot = up[0] * nn[0] + up[1] * nn[1] + up[2] * nn[2];
    expect(dot).toBeGreaterThan(0.99);
  });
  it('falls back to −y for isotropic blobs', () => {
    const r = rng(3);
    const pts: Vec3[] = [];
    for (let i = 0; i < 3000; i++) pts.push([r() - 0.5, r() - 0.5, r() - 0.5]);
    const s = samples(pts);
    expect(estimateUp(s, computeBounds(s)!)).toEqual([0, -1, 0]);
  });
});
