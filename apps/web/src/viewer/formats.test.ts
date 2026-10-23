import { describe, expect, it } from 'vitest';
import {
  RECORD_U32, SH_C0, decodePly, decodeSplat, detectFormat, fromHalf, pack2h, packQuat, parsePlyHeader,
  shCoeffCount, shStrideU32, toHalf,
} from './formats';

const unpack2h = (w: number): [number, number] => [fromHalf(w & 0xffff), fromHalf(w >>> 16)];
const snorm = (b: number) => (b > 127 ? b - 256 : b) / 127;

describe('f16 packing', () => {
  it('round-trips representative values within f16 precision', () => {
    for (const v of [0, 1, -1, 0.5, 0.1, 3.14159, -2.5e-3, 1234.5, 6.1e-5, 1e-6]) {
      const r = fromHalf(toHalf(v));
      expect(Math.abs(r - v)).toBeLessThanOrEqual(Math.max(Math.abs(v) * 1e-3, 6e-8));
    }
  });
  it('saturates instead of producing infinities', () => {
    expect(fromHalf(toHalf(1e9))).toBe(65504);
    expect(fromHalf(toHalf(-1e9))).toBe(-65504);
    expect(fromHalf(toHalf(Infinity))).toBe(65504);
  });
  it('flushes values below the subnormal range to signed zero', () => {
    expect(fromHalf(toHalf(1e-12))).toBe(0);
    expect(toHalf(-1e-12)).toBe(0x8000);
  });
  it('packs the first value into the low 16 bits (matches WGSL unpack2x16float)', () => {
    const [a, b] = unpack2h(pack2h(0.25, -8));
    expect(a).toBe(0.25);
    expect(b).toBe(-8);
  });
  it('packs normalized quaternions as snorm8 with w in the low byte', () => {
    const q = packQuat(2, 0, 0, 0); // unnormalized identity
    expect(snorm(q & 0xff)).toBeCloseTo(1, 2);
    expect(snorm((q >>> 8) & 0xff)).toBe(0);
    const r = packQuat(0.5, 0.5, 0.5, 0.5);
    for (let i = 0; i < 4; i++) expect(snorm((r >>> (8 * i)) & 0xff)).toBeCloseTo(0.5, 2);
  });
});

function splatRecord(): Uint8Array {
  const b = new Uint8Array(32);
  const dv = new DataView(b.buffer);
  [1.5, -2, 3].forEach((v, i) => dv.setFloat32(i * 4, v, true));
  [0.1, 0.2, 0.3].forEach((v, i) => dv.setFloat32(12 + i * 4, v, true));
  b.set([255, 128, 0, 204], 24);
  b.set([128 + 127, 128, 128, 128], 28); // w≈1
  return b;
}

describe('.splat decoding', () => {
  it('decodes position, color, opacity, scale and rotation', () => {
    const out = decodeSplat(splatRecord(), 0, 1);
    expect(out.count).toBe(1);
    expect(out.packed.length).toBe(RECORD_U32);
    const f = new Float32Array(out.packed.buffer);
    expect([f[0], f[1], f[2]]).toEqual([1.5, -2, 3]);
    const [r, g] = unpack2h(out.packed[3]);
    const [b, a] = unpack2h(out.packed[4]);
    expect(r).toBeCloseTo(1, 3);
    expect(g).toBeCloseTo(128 / 255, 3);
    expect(b).toBe(0);
    expect(a).toBeCloseTo(0.8, 3);
    const [sx, sy] = unpack2h(out.packed[5]);
    expect(sx).toBeCloseTo(0.1, 3);
    expect(sy).toBeCloseTo(0.2, 3);
    expect(unpack2h(out.packed[6])[0]).toBeCloseTo(0.3, 3);
    expect(snorm(out.packed[7] & 0xff)).toBeCloseTo(1, 2);
    expect(out.sample.length).toBe(4); // splat 0 is sampled
  });
  it('honours a byte offset inside a larger buffer', () => {
    const big = new Uint8Array(96);
    big.set(splatRecord(), 64);
    const out = decodeSplat(big, 64, 1);
    expect(new Float32Array(out.packed.buffer)[0]).toBe(1.5);
  });
});

function makePly(props: string[], rows: number[][], extraHeader = ''): Uint8Array {
  const header = `ply\nformat binary_little_endian 1.0\n${extraHeader}element vertex ${rows.length}\n` +
    props.map((p) => `property float ${p}\n`).join('') + 'end_header\n';
  const h = new TextEncoder().encode(header);
  const out = new Uint8Array(h.length + rows.length * props.length * 4);
  out.set(h);
  const dv = new DataView(out.buffer, h.length);
  rows.forEach((row, i) => row.forEach((v, j) => dv.setFloat32((i * props.length + j) * 4, v, true)));
  return out;
}

describe('PLY decoding', () => {
  const base = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];

  it('returns null for an incomplete header (streaming)', () => {
    const full = makePly(base, [new Array(base.length).fill(0)]);
    expect(parsePlyHeader(full.subarray(0, 40))).toBeNull();
    expect(parsePlyHeader(full)).not.toBeNull();
  });

  it('rejects ascii PLY and non-PLY input with readable errors', () => {
    expect(() => parsePlyHeader(new TextEncoder().encode('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n'))).toThrow(/binary_little_endian/);
    expect(() => parsePlyHeader(new TextEncoder().encode('hello'))).toThrow(/Not a PLY/);
  });

  it('applies 3DGS activations (SH DC, sigmoid opacity, exp scale)', () => {
    const row = [1, 2, 3, 1, 0, -1, 0, Math.log(0.5), Math.log(0.25), 0, 1, 0, 0, 0];
    const bytes = makePly(base, [row]);
    const h = parsePlyHeader(bytes)!;
    expect(h.vertexCount).toBe(1);
    expect(h.shDegree).toBe(0);
    const out = decodePly(h, bytes, h.headerLength, 1);
    const [r, g] = unpack2h(out.packed[3]);
    const [b, a] = unpack2h(out.packed[4]);
    expect(r).toBeCloseTo(0.5 + SH_C0, 3);
    expect(g).toBeCloseTo(0.5, 3);
    expect(b).toBeCloseTo(0.5 - SH_C0, 3);
    expect(a).toBeCloseTo(0.5, 3); // sigmoid(0)
    const [sx, sy] = unpack2h(out.packed[5]);
    expect(sx).toBeCloseTo(0.5, 3);
    expect(sy).toBeCloseTo(0.25, 3);
    expect(unpack2h(out.packed[6])[0]).toBeCloseTo(1, 3);
    expect(out.sh).toBeNull();
  });

  it('reorders SH rest coefficients from channel-major to coefficient-major', () => {
    const rest = Array.from({ length: 45 }, (_, k) => `f_rest_${k}`);
    const props = [...base.slice(0, 6), ...rest, ...base.slice(6)];
    // value encodes (channel, coeff) = channel*100 + coeff in the file layout
    const restVals = Array.from({ length: 45 }, (_, k) => Math.floor(k / 15) * 100 + (k % 15));
    const row = [0, 0, 0, 0, 0, 0, ...restVals, 0, 0, 0, 0, 1, 0, 0, 0];
    const bytes = makePly(props, [row]);
    const h = parsePlyHeader(bytes)!;
    expect(h.shDegree).toBe(3);
    const out = decodePly(h, bytes, h.headerLength, 1);
    const stride = shStrideU32(3);
    expect(stride).toBe(23);
    expect(out.sh!.length).toBe(stride);
    const halves: number[] = [];
    for (const w of out.sh!) halves.push(...unpack2h(w));
    for (let k = 0; k < shCoeffCount(3); k++) {
      for (let c = 0; c < 3; c++) expect(halves[k * 3 + c]).toBe(c * 100 + k);
    }
  });

  it('decodes plain point clouds (uchar rgb) as small opaque splats', () => {
    const header = 'ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
    const h8 = new TextEncoder().encode(header);
    const bytes = new Uint8Array(h8.length + 15);
    bytes.set(h8);
    const dv = new DataView(bytes.buffer, h8.length);
    dv.setFloat32(0, 1, true);
    bytes.set([255, 0, 51], h8.length + 12);
    const h = parsePlyHeader(bytes)!;
    expect(h.stride).toBe(15);
    const out = decodePly(h, bytes, h.headerLength, 1);
    const [r, g] = unpack2h(out.packed[3]);
    const [b, a] = unpack2h(out.packed[4]);
    expect([r, g]).toEqual([1, 0]);
    expect(b).toBeCloseTo(0.2, 3);
    expect(a).toBe(1);
  });

  it('ignores trailing non-vertex elements', () => {
    const bytes = makePly(base, [new Array(base.length).fill(0)]);
    const text = new TextDecoder().decode(bytes.subarray(0, 400));
    const withFaces = text.replace('end_header', 'element face 0\nproperty list uchar int vertex_indices\nend_header');
    const h = parsePlyHeader(new TextEncoder().encode(withFaces))!;
    expect(h.vertexCount).toBe(1);
    expect(h.properties.has('vertex_indices')).toBe(false);
  });
});

describe('detectFormat', () => {
  it('detects by magic bytes, then extension', () => {
    expect(detectFormat(null, new TextEncoder().encode('ply\n'))).toBe('ply');
    expect(detectFormat('https://x/y/scene.PLY?token=1')).toBe('ply');
    expect(detectFormat('/a/b.splat')).toBe('splat');
  });
});
