// Minimal vector / matrix helpers. Matrices are column-major Float32Array(16), matching WGSL.

export type Vec3 = [number, number, number];

export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const length = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a: Vec3): Vec3 => {
  const l = length(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
export const lerp3 = (a: Vec3, b: Vec3, t: number): Vec3 => [
  lerp(a[0], b[0], t),
  lerp(a[1], b[1], t),
  lerp(a[2], b[2], t),
];

/** Any unit vector perpendicular to n. */
export function perpendicular(n: Vec3): Vec3 {
  const a: Vec3 = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return normalize(cross(n, a));
}

/**
 * World -> camera matrix in the COLMAP / 3DGS camera convention:
 * +x right, +y down, +z forward (into the screen).
 */
export function lookAtCV(eye: Vec3, target: Vec3, worldUp: Vec3): Float32Array {
  const f = normalize(sub(target, eye));
  let r = cross(f, worldUp);
  if (length(r) < 1e-6) r = perpendicular(f);
  r = normalize(r);
  const d = cross(f, r); // "down" axis
  const m = new Float32Array(16);
  // rows of the rotation are r, d, f -> column-major layout
  m[0] = r[0]; m[4] = r[1]; m[8] = r[2];
  m[1] = d[0]; m[5] = d[1]; m[9] = d[2];
  m[2] = f[0]; m[6] = f[1]; m[10] = f[2];
  m[12] = -dot(r, eye);
  m[13] = -dot(d, eye);
  m[14] = -dot(f, eye);
  m[15] = 1;
  return m;
}

/** Transform a point by a column-major 4x4 (ignores projective w). */
export function transformPoint(m: Float32Array, p: Vec3): Vec3 {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}
