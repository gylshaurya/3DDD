import type { Vec3 } from './math';

export interface SceneBounds {
  center: Vec3;
  /** Radius that contains the bulk (~80%) of the visible splats. */
  radius: number;
  /** Median splat distance from the center — what auto-framing uses. Captures include a lot of
   *  background, so framing the median keeps the subject (not the horizon) in view. */
  focusRadius: number;
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
 * background splats, so min/max is useless — use medians and percentiles of reasonably
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
  // Object-only captures (no background) have a compact distance distribution: frame most of it.
  // Captures with surroundings have a long tail: frame the median so the subject fills the view.
  const r50 = Math.max(percentile(dist, 0.5), 1e-3);
  const focusRadius = radius / r50 < 1.8 ? radius * 1.1 : r50;
  return { center, radius, focusRadius, min, max };
}
