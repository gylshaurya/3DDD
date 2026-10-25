// Pure (DOM-free) splat file decoding + GPU packing. Runs in the loader worker and in vitest.
//
// GPU record layout (8 × u32 = 32 bytes per splat):
//   [0..2] position xyz (f32)
//   [3]    color r,g (f16x2)    base color = 0.5 + SH_C0 * f_dc, may exceed [0,1]
//   [4]    color b, opacity (f16x2)
//   [5]    scale x,y (f16x2)    linear (already exp'd)
//   [6]    scale z, 0 (f16x2)
//   [7]    rotation quat w,x,y,z (snorm8x4)
// Optional SH record: coefficient-major f16 triples (k*3 + channel), `shStride` u32 per splat.

export type SplatFormat = 'splat' | 'ply';

export const RECORD_U32 = 8;
export const RECORD_BYTES = RECORD_U32 * 4;
export const SPLAT_FILE_STRIDE = 32;
export const SH_C0 = 0.28209479177387814;

/** Number of non-DC SH coefficients (per channel) for a given degree. */
export const shCoeffCount = (degree: number): number => (degree + 1) * (degree + 1) - 1;
/** u32 words per splat holding the SH rest coefficients. */
export const shStrideU32 = (degree: number): number => Math.ceil((shCoeffCount(degree) * 3) / 2);

// ---------- f16 packing ----------

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/** IEEE754 float32 -> float16 bits, round-to-nearest, saturating to +/-65504. */
export function toHalf(v: number): number {
  f32Scratch[0] = v;
  const x = u32Scratch[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (((x >>> 23) & 0xff) === 0xff) return mant ? sign | 0x7e00 : sign | 0x7bff; // NaN / inf -> max
  if (exp >= 31) return sign | 0x7bff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    const m = (mant | 0x800000) >>> (1 - exp);
    return sign | ((m + 0x1000) >>> 13);
  }
  const h = (exp << 10) + ((mant + 0x1000) >>> 13);
  return sign | Math.min(h, 0x7bff);
}

/** float16 bits -> number (used by tests / CPU paths). */
export function fromHalf(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

export const pack2h = (a: number, b: number): number => (toHalf(a) | (toHalf(b) << 16)) >>> 0;

const snorm8 = (v: number): number => Math.round(Math.max(-1, Math.min(1, v)) * 127) & 0xff;

export function packQuat(w: number, x: number, y: number, z: number): number {
  const l = Math.hypot(w, x, y, z) || 1;
  return (
    (snorm8(w / l) | (snorm8(x / l) << 8) | (snorm8(y / l) << 16) | (snorm8(z / l) << 24)) >>> 0
  );
}

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

// ---------- decoded batch ----------

export interface DecodedBatch {
  /** Number of splats in this batch. */
  count: number;
  /** RECORD_U32 words per splat. */
  packed: Uint32Array;
  /** SH rest coefficients (shStride u32 per splat) or null. */
  sh: Uint32Array | null;
  /** xyz + opacity of a strided subsample, for bounds estimation. */
  sample: Float32Array;
}

/** Keep every Nth splat for bounds estimation (~ 64K samples per 1M splats). */
export const SAMPLE_EVERY = 16;

function writeRecord(
  out: Uint32Array,
  outF: Float32Array,
  i: number,
  x: number, y: number, z: number,
  r: number, g: number, b: number, a: number,
  sx: number, sy: number, sz: number,
  qw: number, qx: number, qy: number, qz: number,
): void {
  const o = i * RECORD_U32;
  outF[o] = x;
  outF[o + 1] = y;
  outF[o + 2] = z;
  out[o + 3] = pack2h(r, g);
  out[o + 4] = pack2h(b, a);
  out[o + 5] = pack2h(sx, sy);
  out[o + 6] = pack2h(sz, 0);
  out[o + 7] = packQuat(qw, qx, qy, qz);
}

function pushSample(samples: number[], x: number, y: number, z: number, a: number): void {
  samples.push(x, y, z, a);
}

// ---------- .splat (antimatter15) ----------

/** Decode `count` records of a .splat file starting at byte `offset` of `bytes`. */
export function decodeSplat(bytes: Uint8Array, offset: number, count: number, globalStart = 0): DecodedBatch {
  const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, count * SPLAT_FILE_STRIDE);
  const packed = new Uint32Array(count * RECORD_U32);
  const packedF = new Float32Array(packed.buffer);
  const samples: number[] = [];
  for (let i = 0; i < count; i++) {
    const o = i * SPLAT_FILE_STRIDE;
    const x = dv.getFloat32(o, true);
    const y = dv.getFloat32(o + 4, true);
    const z = dv.getFloat32(o + 8, true);
    const sx = dv.getFloat32(o + 12, true);
    const sy = dv.getFloat32(o + 16, true);
    const sz = dv.getFloat32(o + 20, true);
    const r = dv.getUint8(o + 24) / 255;
    const g = dv.getUint8(o + 25) / 255;
    const b = dv.getUint8(o + 26) / 255;
    const a = dv.getUint8(o + 27) / 255;
    const qw = (dv.getUint8(o + 28) - 128) / 128;
    const qx = (dv.getUint8(o + 29) - 128) / 128;
    const qy = (dv.getUint8(o + 30) - 128) / 128;
    const qz = (dv.getUint8(o + 31) - 128) / 128;
    writeRecord(packed, packedF, i, x, y, z, r, g, b, a, sx, sy, sz, qw, qx, qy, qz);
    if ((globalStart + i) % SAMPLE_EVERY === 0) pushSample(samples, x, y, z, a);
  }
  return { count, packed, sh: null, sample: new Float32Array(samples) };
}

// ---------- .ply (3DGS binary little endian) ----------

type PlyType = 'char' | 'uchar' | 'short' | 'ushort' | 'int' | 'uint' | 'float' | 'double';

const PLY_TYPE_SIZE: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};

const PLY_TYPE_ALIAS: Record<string, PlyType> = {
  char: 'char', int8: 'char', uchar: 'uchar', uint8: 'uchar',
  short: 'short', int16: 'short', ushort: 'ushort', uint16: 'ushort',
  int: 'int', int32: 'int', uint: 'uint', uint32: 'uint',
  float: 'float', float32: 'float', double: 'double', float64: 'double',
};

export interface PlyProperty {
  name: string;
  type: PlyType;
  offset: number;
}

export interface PlyHeader {
  /** Bytes up to and including "end_header\n". */
  headerLength: number;
  vertexCount: number;
  /** Bytes per vertex record. */
  stride: number;
  properties: Map<string, PlyProperty>;
  shDegree: number;
}

/**
 * Parse a PLY header. Returns null if the header is not yet complete in `bytes`
 * (useful while streaming). Throws on unsupported files.
 */
export function parsePlyHeader(bytes: Uint8Array): PlyHeader | null {
  const marker = 'end_header';
  const limit = Math.min(bytes.length, 64 * 1024);
  let text = '';
  for (let i = 0; i < limit; i++) text += String.fromCharCode(bytes[i]);
  if (!text.startsWith('ply')) throw new Error('Not a PLY file');
  const end = text.indexOf(marker);
  if (end < 0) {
    if (limit >= 64 * 1024) throw new Error('PLY header too long');
    return null;
  }
  let headerLength = end + marker.length;
  if (text[headerLength] === '\r') headerLength++;
  if (text[headerLength] !== '\n') return null; // newline not arrived yet
  headerLength++;

  const lines = text.slice(0, end).split(/\r?\n/);
  let format = '';
  let vertexCount = -1;
  let inVertex = false;
  let sawVertex = false;
  let stride = 0;
  const properties = new Map<string, PlyProperty>();
  for (const raw of lines) {
    const parts = raw.trim().split(/\s+/);
    if (parts[0] === 'format') format = parts[1];
    else if (parts[0] === 'element') {
      if (sawVertex && inVertex) inVertex = false;
      if (parts[1] === 'vertex') {
        if (sawVertex) throw new Error('Multiple vertex elements');
        vertexCount = parseInt(parts[2], 10);
        inVertex = true;
        sawVertex = true;
      } else if (!sawVertex) {
        throw new Error(`Unsupported PLY: element "${parts[1]}" before vertex`);
      } else inVertex = false;
    } else if (parts[0] === 'property' && inVertex) {
      if (parts[1] === 'list') throw new Error('List properties in vertex element are not supported');
      const type = PLY_TYPE_ALIAS[parts[1]];
      if (!type) throw new Error(`Unknown PLY type ${parts[1]}`);
      properties.set(parts[2], { name: parts[2], type, offset: stride });
      stride += PLY_TYPE_SIZE[parts[1]];
    }
  }
  if (format !== 'binary_little_endian') throw new Error(`Unsupported PLY format "${format}" (need binary_little_endian)`);
  if (vertexCount < 0) throw new Error('PLY has no vertex element');
  for (const req of ['x', 'y', 'z']) if (!properties.has(req)) throw new Error(`PLY missing property ${req}`);

  let restCount = 0;
  while (properties.has(`f_rest_${restCount}`)) restCount++;
  const perChannel = Math.floor(restCount / 3);
  let shDegree = 0;
  if (perChannel >= 15) shDegree = 3;
  else if (perChannel >= 8) shDegree = 2;
  else if (perChannel >= 3) shDegree = 1;
  return { headerLength, vertexCount, stride, properties, shDegree };
}

type Reader = (dv: DataView, o: number) => number;

function makeReader(prop: PlyProperty | undefined, fallback: number): Reader {
  if (!prop) return () => fallback;
  const off = prop.offset;
  switch (prop.type) {
    case 'float': return (dv, o) => dv.getFloat32(o + off, true);
    case 'double': return (dv, o) => dv.getFloat64(o + off, true);
    case 'uchar': return (dv, o) => dv.getUint8(o + off);
    case 'char': return (dv, o) => dv.getInt8(o + off);
    case 'ushort': return (dv, o) => dv.getUint16(o + off, true);
    case 'short': return (dv, o) => dv.getInt16(o + off, true);
    case 'uint': return (dv, o) => dv.getUint32(o + off, true);
    case 'int': return (dv, o) => dv.getInt32(o + off, true);
  }
}

/**
 * Decode `count` vertex records of a 3DGS PLY, starting at byte `offset` of `bytes`
 * (offset must point at a record boundary inside the body).
 * Handles both trained-3DGS fields (f_dc, logit opacity, log scale) and plain point
 * clouds (red/green/blue uchar) as a fallback.
 */
export function decodePly(
  header: PlyHeader,
  bytes: Uint8Array,
  offset: number,
  count: number,
  globalStart = 0,
): DecodedBatch {
  const P = header.properties;
  const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, count * header.stride);
  const rx = makeReader(P.get('x'), 0);
  const ry = makeReader(P.get('y'), 0);
  const rz = makeReader(P.get('z'), 0);
  const isGaussian = P.has('f_dc_0') || P.has('scale_0');
  const rdc = [0, 1, 2].map((c) => makeReader(P.get(`f_dc_${c}`), 0));
  const rgb8 = ['red', 'green', 'blue'].map((n) => makeReader(P.get(n), 255));
  const ropacity = makeReader(P.get('opacity'), 10);
  const rscale = [0, 1, 2].map((c) => makeReader(P.get(`scale_${c}`), Math.log(0.01)));
  const rrot = [0, 1, 2, 3].map((c) => makeReader(P.get(`rot_${c}`), c === 0 ? 1 : 0));

  const deg = header.shDegree;
  const perChannel = shCoeffCount(deg);
  const stride = deg > 0 ? shStrideU32(deg) : 0;
  let restTotal = 0;
  while (P.has(`f_rest_${restTotal}`)) restTotal++;
  const filePerChannel = Math.floor(restTotal / 3);
  const rrest: Reader[] = [];
  for (let k = 0; k < restTotal; k++) rrest.push(makeReader(P.get(`f_rest_${k}`), 0));

  const packed = new Uint32Array(count * RECORD_U32);
  const packedF = new Float32Array(packed.buffer);
  const sh = deg > 0 ? new Uint32Array(count * stride) : null;
  const halves = new Uint16Array(stride * 2);
  const samples: number[] = [];

  for (let i = 0; i < count; i++) {
    const o = i * header.stride;
    const x = rx(dv, o), y = ry(dv, o), z = rz(dv, o);
    let r: number, g: number, b: number, a: number, sx: number, sy: number, sz: number;
    if (isGaussian) {
      r = 0.5 + SH_C0 * rdc[0](dv, o);
      g = 0.5 + SH_C0 * rdc[1](dv, o);
      b = 0.5 + SH_C0 * rdc[2](dv, o);
      a = sigmoid(ropacity(dv, o));
      sx = Math.exp(rscale[0](dv, o));
      sy = Math.exp(rscale[1](dv, o));
      sz = Math.exp(rscale[2](dv, o));
    } else {
      r = rgb8[0](dv, o) / 255;
      g = rgb8[1](dv, o) / 255;
      b = rgb8[2](dv, o) / 255;
      a = 1;
      sx = sy = sz = 0.01;
    }
    writeRecord(
      packed, packedF, i, x, y, z, r, g, b, a, sx, sy, sz,
      rrot[0](dv, o), rrot[1](dv, o), rrot[2](dv, o), rrot[3](dv, o),
    );
    if (sh) {
      halves.fill(0);
      // file: channel-major (all R coeffs, then G, then B); GPU: coefficient-major
      for (let k = 0; k < perChannel; k++) {
        for (let c = 0; c < 3; c++) {
          halves[k * 3 + c] = toHalf(rrest[c * filePerChannel + k](dv, o));
        }
      }
      const base = i * stride;
      for (let w = 0; w < stride; w++) sh[base + w] = (halves[2 * w] | (halves[2 * w + 1] << 16)) >>> 0;
    }
    if ((globalStart + i) % SAMPLE_EVERY === 0) pushSample(samples, x, y, z, a);
  }
  return { count, packed, sh, sample: new Float32Array(samples) };
}

/** Guess the format from a URL / filename or the first bytes. */
export function detectFormat(nameOrUrl: string | null, head?: Uint8Array): SplatFormat {
  if (head && head.length >= 3 && head[0] === 0x70 && head[1] === 0x6c && head[2] === 0x79) return 'ply';
  if (nameOrUrl) {
    const clean = nameOrUrl.split(/[?#]/)[0].toLowerCase();
    if (clean.endsWith('.ply')) return 'ply';
  }
  return 'splat';
}
