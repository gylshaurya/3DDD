import { SORT_TILE } from './sortShaders';

// Uniform block shared by the preprocess and render stages (128 bytes).
export const UNIFORM_BYTES = 128;

const uniformsWGSL = /* wgsl */ `
struct Uniforms {
  view: mat4x4<f32>,      // world → camera (x right, y down, z forward)
  camPos: vec4<f32>,      // world-space eye
  focal: vec2<f32>,       // fx, fy in pixels
  viewport: vec2<f32>,    // width, height in pixels
  splatCount: u32,        // splats currently loaded
  shDegree: u32,
  shStride: u32,          // u32 words of SH per splat
  compact: u32,           // 1: GPU cull+compact+sort, 0: CPU-sorted order (fallback)
  alphaScale: f32,        // global opacity (crossfades)
  splatScale: f32,        // debug / effect: scales splat footprint
  nearZ: f32,             // near plane (scene-relative)
  _pad1: f32,
};
`;

/**
 * Generates an SH evaluator that loads each packed f16 pair exactly once with constant indices
 * (dynamic per-half indexing was ~10 ms/frame for 1M SH3 splats on an M4).
 * Coefficients are coefficient-major f16 triples: half h = k*3 + channel.
 */
function shEvalWGSL(): string {
  const words = (from: number, to: number) => {
    let out = '';
    for (let w = from; w <= to; w++) out += `    let p${w} = unpack2x16float(sh[base + ${w}u]);\n`;
    return out;
  };
  const half = (h: number) => `p${h >> 1}.${h & 1 ? 'y' : 'x'}`;
  const coef = (k: number) => `vec3<f32>(${half(k * 3)}, ${half(k * 3 + 1)}, ${half(k * 3 + 2)})`;
  return /* wgsl */ `
fn evalSH(i: u32, dir: vec3<f32>) -> vec3<f32> {
  let base = i * u.shStride;
  let x = dir.x; let y = dir.y; let z = dir.z;
${words(0, 4)}
  var c = -SH_C1 * y * ${coef(0)} + SH_C1 * z * ${coef(1)} - SH_C1 * x * ${coef(2)};
  if (u.shDegree > 1u) {
${words(5, 11)}
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let yz = y * z; let xz = x * z;
    c += SH_C2[0] * xy * ${coef(3)}
       + SH_C2[1] * yz * ${coef(4)}
       + SH_C2[2] * (2.0 * zz - xx - yy) * ${coef(5)}
       + SH_C2[3] * xz * ${coef(6)}
       + SH_C2[4] * (xx - yy) * ${coef(7)};
    if (u.shDegree > 2u) {
${words(12, 22)}
      c += SH_C3[0] * y * (3.0 * xx - yy) * ${coef(8)}
         + SH_C3[1] * xy * z * ${coef(9)}
         + SH_C3[2] * y * (4.0 * zz - xx - yy) * ${coef(10)}
         + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * ${coef(11)}
         + SH_C3[4] * x * (4.0 * zz - xx - yy) * ${coef(12)}
         + SH_C3[5] * z * (xx - yy) * ${coef(13)}
         + SH_C3[6] * x * (xx - 3.0 * yy) * ${coef(14)};
    }
  }
  return c;
}`;
}

/**
 * Projects every splat: frustum cull, EWA covariance projection, view-dependent SH color,
 * then appends survivors (atomic compaction) with an inverted-depth sort key so an
 * ascending sort yields back-to-front order.
 *
 * projected record (8 × u32):
 *   [0..1] center NDC (f32x2)  [2] axis1 px (f16x2)  [3] axis2 px (f16x2)
 *   [4] rgb.rg (f16x2)  [5] rgb.b, alpha (f16x2)  [6] cutoff radius in sigmas (f32)  [7] —
 */
export const preprocessWGSL = /* wgsl */ `
${uniformsWGSL}
struct Ctrl { count: atomic<u32>, rest: array<u32, 7> };

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<u32>;
@group(0) @binding(2) var<storage, read> sh: array<u32>;
@group(0) @binding(3) var<storage, read_write> projected: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals: array<u32>;
@group(0) @binding(6) var<storage, read_write> ctrl: Ctrl;

const SH_C1: f32 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396);
const SH_C3 = array<f32, 7>(-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435);

${shEvalWGSL()}

fn writeInvisible(i: u32) {
  // fallback (non-compact) mode needs every slot defined
  let o = i * 8u;
  projected[o + 2u] = 0u;
  projected[o + 3u] = 0u;
  projected[o + 5u] = 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= u.splatCount) { return; }
  let b = i * 8u;
  let pos = vec3<f32>(bitcast<f32>(splats[b]), bitcast<f32>(splats[b + 1u]), bitcast<f32>(splats[b + 2u]));
  let pc = (u.view * vec4<f32>(pos, 1.0)).xyz;

  let colA = unpack2x16float(splats[b + 3u]);
  let colB = unpack2x16float(splats[b + 4u]);
  let opacity = colB.y * u.alphaScale;
  if (pc.z < u.nearZ || opacity < (1.0 / 255.0)) {
    if (u.compact == 0u) { writeInvisible(i); }
    return;
  }

  // frustum test on the center with a generous guard band
  let invZ = 1.0 / pc.z;
  let px = u.focal.x * pc.x * invZ;   // pixels from image center
  let py = u.focal.y * pc.y * invZ;
  let halfW = 0.5 * u.viewport.x;
  let halfH = 0.5 * u.viewport.y;
  if (abs(px) > 1.3 * halfW + 64.0 || abs(py) > 1.3 * halfH + 64.0) {
    if (u.compact == 0u) { writeInvisible(i); }
    return;
  }

  // 3D covariance Σ = R S Sᵀ Rᵀ
  let s01 = unpack2x16float(splats[b + 5u]);
  let s2 = unpack2x16float(splats[b + 6u]).x;
  let s = vec3<f32>(s01.x, s01.y, s2) * u.splatScale;
  let q = normalize(unpack4x8snorm(splats[b + 7u])); // (w, x, y, z)
  let w = q.x; let x = q.y; let y = q.z; let z = q.w;
  let R = mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
  );
  let M = R * mat3x3<f32>(vec3<f32>(s.x, 0.0, 0.0), vec3<f32>(0.0, s.y, 0.0), vec3<f32>(0.0, 0.0, s.z));
  let Sigma = M * transpose(M);

  // EWA: Jacobian of the perspective projection (tx/tz clamped as in 3DGS)
  let limX = 1.3 * halfW / u.focal.x;
  let limY = 1.3 * halfH / u.focal.y;
  let tx = clamp(pc.x * invZ, -limX, limX) * pc.z;
  let ty = clamp(pc.y * invZ, -limY, limY) * pc.z;
  let J = mat3x3<f32>(
    vec3<f32>(u.focal.x * invZ, 0.0, 0.0),
    vec3<f32>(0.0, u.focal.y * invZ, 0.0),
    vec3<f32>(-u.focal.x * tx * invZ * invZ, -u.focal.y * ty * invZ * invZ, 0.0),
  );
  let W = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
  let T = J * W;
  let cov = T * Sigma * transpose(T);
  let a = cov[0][0] + 0.3;   // low-pass dilation
  let bb = cov[0][1];
  let c = cov[1][1] + 0.3;

  // eigen-decomposition → oriented quad axes
  let mid = 0.5 * (a + c);
  let rad = length(vec2<f32>(0.5 * (a - c), bb));
  let l1 = mid + rad;
  let l2 = max(mid - rad, 0.1);
  var v1 = vec2<f32>(1.0, 0.0);
  if (abs(bb) > 1e-9) { v1 = normalize(vec2<f32>(bb, l1 - a)); } else if (c > a) { v1 = vec2<f32>(0.0, 1.0); }
  // only extend the quad as far as alpha stays above 1/255
  let cutoff = sqrt(max(2.0 * log(255.0 * opacity), 0.0));
  let r1 = min(sqrt(l1) * cutoff, 1024.0);
  let r2 = min(sqrt(l2) * cutoff, 1024.0);
  if (abs(px) - r1 > halfW || abs(py) - r1 > halfH || r1 < 0.15) {
    if (u.compact == 0u) { writeInvisible(i); }
    return;
  }
  let ax1 = v1 * r1;
  let ax2 = vec2<f32>(-v1.y, v1.x) * r2;

  var rgb = vec3<f32>(colA.x, colA.y, colB.x);
  if (u.shDegree > 0u) { rgb += evalSH(i, normalize(pos - u.camPos.xyz)); }
  rgb = max(rgb, vec3<f32>(0.0));

  var slot = i;
  if (u.compact == 1u) {
    slot = atomicAdd(&ctrl.count, 1u);
    keys[slot] = ~bitcast<u32>(pc.z);   // far first after ascending sort
    vals[slot] = slot;
  }
  let o = slot * 8u;
  projected[o] = bitcast<u32>(px / halfW);
  projected[o + 1u] = bitcast<u32>(-py / halfH);
  projected[o + 2u] = pack2x16float(ax1);
  projected[o + 3u] = pack2x16float(ax2);
  projected[o + 4u] = pack2x16float(rgb.rg);
  projected[o + 5u] = pack2x16float(vec2<f32>(rgb.b, opacity));
  projected[o + 6u] = bitcast<u32>(cutoff);
}
`;

/** Turns the visible count into indirect dispatch + draw args. */
export const prepareWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> ctrl: array<u32, 8>;
@compute @workgroup_size(1)
fn main() {
  let count = ctrl[0];
  ctrl[1] = (count + ${SORT_TILE - 1}u) / ${SORT_TILE}u;
  ctrl[2] = 1u;
  ctrl[3] = 1u;
  ctrl[4] = 4u;      // vertices per quad (triangle strip)
  ctrl[5] = count;   // instances
  ctrl[6] = 0u;
  ctrl[7] = 0u;
}
`;

export const renderWGSL = /* wgsl */ `
${uniformsWGSL}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> projected: array<u32>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec2<f32>,       // quad coords scaled to sigmas
  @location(1) color: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var out: VSOut;
  let idx = order[ii];
  let o = idx * 8u;
  let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  let ax1 = unpack2x16float(projected[o + 2u]);
  let ax2 = unpack2x16float(projected[o + 3u]);
  let center = vec2<f32>(bitcast<f32>(projected[o]), bitcast<f32>(projected[o + 1u]));
  let offPx = corner.x * ax1 + corner.y * ax2;
  // pixel offsets → NDC (screen y is down, NDC y is up)
  let offNdc = vec2<f32>(offPx.x, -offPx.y) * 2.0 / u.viewport;
  out.pos = vec4<f32>(center + offNdc, 0.0, 1.0);
  let cutoff = bitcast<f32>(projected[o + 6u]);
  out.local = corner * cutoff;
  let cA = unpack2x16float(projected[o + 4u]);
  let cB = unpack2x16float(projected[o + 5u]);
  out.color = vec4<f32>(cA.x, cA.y, cB.x, cB.y);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let power = -0.5 * dot(in.local, in.local);
  let alpha = min(0.99, in.color.a * exp(power));
  if (alpha < 1.0 / 255.0) { discard; }
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;
