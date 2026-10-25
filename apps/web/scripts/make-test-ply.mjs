// Generates PLY test fixtures in public/samples/:
//   <name>.ply           a .splat file converted to a standard 3DGS PLY (SH degree 0)
//   sh-test.ply          synthetic scene whose color depends on view direction (SH degree 3)
//
// Usage: node scripts/make-test-ply.mjs public/samples/train.splat
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';

const SH_C0 = 0.28209479177387814;
const logit = (p) => Math.log(p / (1 - p));

function writePly(path, count, props, fill) {
  const header =
    `ply\nformat binary_little_endian 1.0\nelement vertex ${count}\n` +
    props.map((p) => `property float ${p}\n`).join('') +
    'end_header\n';
  const h = Buffer.from(header, 'ascii');
  const body = Buffer.alloc(count * props.length * 4);
  const f = new Float32Array(body.buffer, body.byteOffset, count * props.length);
  for (let i = 0; i < count; i++) fill(i, f.subarray(i * props.length, (i + 1) * props.length));
  writeFileSync(path, Buffer.concat([h, body]));
  console.log(`wrote ${path} (${count} splats, ${props.length} props)`);
}

const BASE = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
const TAIL = ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];

// 1) .splat -> .ply (SH0)
const src = process.argv[2];
if (src) {
  const buf = readFileSync(src);
  const n = Math.floor(buf.length / 32);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.length);
  const props = [...BASE, ...TAIL];
  writePly(join(dirname(src), basename(src, '.splat') + '.ply'), n, props, (i, o) => {
    const b = i * 32;
    o[0] = dv.getFloat32(b, true);
    o[1] = dv.getFloat32(b + 4, true);
    o[2] = dv.getFloat32(b + 8, true);
    for (let c = 0; c < 3; c++) o[6 + c] = (dv.getUint8(b + 24 + c) / 255 - 0.5) / SH_C0;
    const a = Math.min(Math.max(dv.getUint8(b + 27) / 255, 1e-4), 1 - 1e-4);
    o[9] = logit(a);
    for (let c = 0; c < 3; c++) o[10 + c] = Math.log(Math.max(dv.getFloat32(b + 12 + 4 * c, true), 1e-8));
    for (let c = 0; c < 4; c++) o[13 + c] = (dv.getUint8(b + 28 + c) - 128) / 128;
  });
}

// 2) synthetic SH3 scene: a 24×24 wall of opaque discs in the z=0 plane (y-down world).
// SH coefficient k=2 (the −C1·x term) carries (+0.8, 0, −0.8) for (R, G, B), so the wall looks
// RED when the view direction has −x (camera on the +x side) and BLUE when viewed from −x.
// Coefficient k=8 gets a small degree-3 term to make sure higher bands are wired up.
{
  const rest = Array.from({ length: 45 }, (_, k) => `f_rest_${k}`);
  const props = [...BASE, ...rest, ...TAIL];
  const N = 24;
  writePly(join(process.argv[3] ?? 'public/samples', 'sh-test.ply'), N * N, props, (i, o) => {
    const gx = i % N, gy = Math.floor(i / N);
    o[0] = (gx - (N - 1) / 2) * 0.1;
    o[1] = (gy - (N - 1) / 2) * 0.1;
    o[2] = 0;
    // base color mid grey
    for (let c = 0; c < 3; c++) o[6 + c] = 0;
    // f_rest layout is channel-major: R coeffs 0..14, G 15..29, B 30..44
    const set = (k, r, g, b) => { o[9 + k] = r; o[9 + 15 + k] = g; o[9 + 30 + k] = b; };
    set(2, 0.8, 0, -0.8);
    set(8, 0.05, 0.05, 0.05);
    const t = 9 + 45;
    o[t] = logit(0.99);
    o[t + 1] = Math.log(0.06);
    o[t + 2] = Math.log(0.06);
    o[t + 3] = Math.log(0.005);
    o[t + 4] = 1; o[t + 5] = 0; o[t + 6] = 0; o[t + 7] = 0;
  });
}
