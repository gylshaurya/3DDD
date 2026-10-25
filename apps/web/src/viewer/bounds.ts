import type { Vec3 } from './math';

export interface SceneBounds {
  center: Vec3;
  /** Radius that contains the bulk (~80%) of the visible splats. */
  radius: number;
  /** Median splat distance from the center, used by auto-framing. Captures include a lot of
   *  background, so framing the median keeps the subject (not the horizon) in view. */
  focusRadius: number;
  /** Point auto-framing orbits around (the dense core for indoor scenes, else `center`). */
  focusCenter: Vec3;
  /** Robust per-axis extents (5th / 95th percentile). */
  min: Vec3;
  max: Vec3;
}

function percentile(sorted: Float32Array, p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))));
  return sorted[idx];
}

/**
 * Robust bounds from (x, y, z, opacity) samples. Splat scenes have far-away floaters and
 * background splats, so min/max is useless. Use medians and percentiles of reasonably
 * opaque splats instead.
 */
export function computeBounds(samples: Float32Array): SceneBounds | null {
  const n = samples.length / 4;
  if (n === 0) return null;
  let kept = 0;
  for (let i = 0; i < n; i++) if (samples[i * 4 + 3] > 0.25) kept++;
  const useAll = kept < 32;
  const m = useAll ? n : kept;
  const axes = [new Float32Array(m), new Float32Array(m), new Float32Array(m)];
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (!useAll && samples[i * 4 + 3] <= 0.25) continue;
    axes[0][j] = samples[i * 4];
    axes[1][j] = samples[i * 4 + 1];
    axes[2][j] = samples[i * 4 + 2];
    j++;
  }
  for (const a of axes) a.sort();
  const center: Vec3 = [percentile(axes[0], 0.5), percentile(axes[1], 0.5), percentile(axes[2], 0.5)];
  const min: Vec3 = [percentile(axes[0], 0.05), percentile(axes[1], 0.05), percentile(axes[2], 0.05)];
  const max: Vec3 = [percentile(axes[0], 0.95), percentile(axes[1], 0.95), percentile(axes[2], 0.95)];

  const dist = new Float32Array(m);
  j = 0;
  for (let i = 0; i < n; i++) {
    if (!useAll && samples[i * 4 + 3] <= 0.25) continue;
    dist[j++] = Math.hypot(samples[i * 4] - center[0], samples[i * 4 + 1] - center[1], samples[i * 4 + 2] - center[2]);
  }
  dist.sort();
  const radius = Math.max(percentile(dist, 0.8), 1e-3);
  const r50 = Math.max(percentile(dist, 0.5), 1e-3);
  const r05 = percentile(dist, 0.05);
  let focusCenter = center;
  let focusRadius: number;
  if (r05 / r50 < 0.25) {
    // Dense core inside a distant shell (indoor 360 deg capture): frame the core from inside the room.
    const core: number[][] = [[], [], []];
    const r10 = percentile(dist, 0.1);
    j = 0;
    for (let i = 0; i < n; i++) {
      if (!useAll && samples[i * 4 + 3] <= 0.25) continue;
      if (dist[j] !== undefined && Math.hypot(samples[i * 4] - center[0], samples[i * 4 + 1] - center[1], samples[i * 4 + 2] - center[2]) <= r10) {
        for (let a = 0; a < 3; a++) core[a].push(samples[i * 4 + a]);
      }
      j++;
    }
    const med = (a: number[]) => { const f = Float32Array.from(a).sort(); return percentile(f, 0.5); };
    if (core[0].length > 16) focusCenter = [med(core[0]), med(core[1]), med(core[2])];
    focusRadius = Math.max(r10, 1e-3);
  } else if (radius / r50 < 1.8) {
    // Object-only capture (background removed): compact distribution, frame most of it.
    focusRadius = radius * 1.1;
  } else {
    // Capture with surroundings: long tail, frame the median so the subject fills the view.
    focusRadius = r50;
  }
  return { center, focusCenter, radius, focusRadius, min, max };
}

/** Symmetric 3×3 eigen-decomposition (Jacobi). Returns eigenpairs sorted by ascending value. */
function eigenSym3(c: number[][]): { value: number; vector: Vec3 }[] {
  const A = c.map((r) => r.slice());
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let it = 0; it < 64; it++) {
    let p = 0, q = 1;
    for (const [a, b] of [[0, 1], [0, 2], [1, 2]]) if (Math.abs(A[a][b]) > Math.abs(A[p][q])) { p = a; q = b; }
    if (Math.abs(A[p][q]) < 1e-14) break;
    const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
    const cs = Math.cos(th), sn = Math.sin(th);
    for (let k = 0; k < 3; k++) { const kp = A[k][p], kq = A[k][q]; A[k][p] = cs * kp - sn * kq; A[k][q] = sn * kp + cs * kq; }
    for (let k = 0; k < 3; k++) { const pk = A[p][k], qk = A[q][k]; A[p][k] = cs * pk - sn * qk; A[q][k] = sn * pk + cs * qk; }
    for (let k = 0; k < 3; k++) { const kp = V[k][p], kq = V[k][q]; V[k][p] = cs * kp - sn * kq; V[k][q] = sn * kp + cs * kq; }
  }
  return [0, 1, 2]
    .map((i) => ({ value: A[i][i], vector: [V[0][i], V[1][i], V[2][i]] as Vec3 }))
    .sort((a, b) => a.value - b.value);
}

/**
 * Guess the scene's up vector. Most captures are dominated by a ground / floor / tabletop plane,
 * whose normal is the direction of least variance. COLMAP worlds are only roughly y-down, so
 * we accept the PCA axis when it is clearly the flattest and within ~45 deg of the y axis, and
 * otherwise fall back to −y.
 */
export function estimateUp(samples: Float32Array, bounds: SceneBounds): Vec3 {
  const c = bounds.center;
  const m = [0, 0, 0, 0, 0, 0];
  let n = 0;
  for (let i = 0; i < samples.length; i += 4) {
    if (samples[i + 3] <= 0.25) continue;
    const x = samples[i] - c[0], y = samples[i + 1] - c[1], z = samples[i + 2] - c[2];
    if (x * x + y * y + z * z > bounds.radius * bounds.radius) continue;
    m[0] += x * x; m[1] += x * y; m[2] += x * z; m[3] += y * y; m[4] += y * z; m[5] += z * z;
    n++;
  }
  const fallback: Vec3 = [0, -1, 0];
  if (n < 64) return fallback;
  const cov = [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]].map((r) => r.map((v) => v / n));
  const [smallest, middle] = eigenSym3(cov);
  const v = smallest.vector;
  if (smallest.value / Math.max(middle.value, 1e-12) > 0.4 || Math.abs(v[1]) < 0.7) return fallback;
  return v[1] > 0 ? [-v[0], -v[1], -v[2]] : v;
}
