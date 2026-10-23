// WGSL for an LSD radix sort (8 bits/pass) over a GPU-determined element count.
//
// Per pass:
//   1. histogram: one workgroup per 1024-element tile counts digit occurrences and writes
//      them digit-major: hist[digit * numTiles + tile].
//   2. scan: exclusive prefix sum over hist (multi-level Blelloch) → global write offsets.
//   3. scatter: each tile stably sorts itself by the digit with 8 one-bit splits in
//      workgroup memory, then writes element p to hist[digit][tile] + (p − tileDigitStart).
//
// The element count lives in ctrl[0] (written on the GPU by the culling pass), so every
// kernel derives numTiles itself and the dispatches are indirect.

export const SORT_WG = 256;
export const SORT_ITEMS = 4;
export const SORT_TILE = SORT_WG * SORT_ITEMS; // 1024
export const SCAN_BLOCK = 512; // 256 threads × 2 elements

const common = /* wgsl */ `
const WG: u32 = ${SORT_WG}u;
const ITEMS: u32 = ${SORT_ITEMS}u;
const TILE: u32 = ${SORT_TILE}u;

struct PassParams { shift: u32, _p0: u32, _p1: u32, _p2: u32 };
`;

export const histogramWGSL = /* wgsl */ `
${common}
@group(0) @binding(0) var<uniform> params: PassParams;
@group(0) @binding(1) var<storage, read> ctrl: array<u32>;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> hist: array<u32>;

var<workgroup> localHist: array<atomic<u32>, 256>;

@compute @workgroup_size(${SORT_WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let count = ctrl[0];
  let numTiles = (count + TILE - 1u) / TILE;
  let tile = wid.x;
  atomicStore(&localHist[lid.x], 0u);
  workgroupBarrier();
  for (var k = 0u; k < ITEMS; k++) {
    let idx = tile * TILE + k * WG + lid.x;
    if (idx < count) {
      let digit = (keysIn[idx] >> params.shift) & 0xffu;
      atomicAdd(&localHist[digit], 1u);
    }
  }
  workgroupBarrier();
  if (tile < numTiles) {
    hist[lid.x * numTiles + tile] = atomicLoad(&localHist[lid.x]);
  }
}
`;

/** Block-wise exclusive scan; writes each block's total into blockSums. */
export const scanBlocksWGSL = /* wgsl */ `
struct ScanParams { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> sp: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;

var<workgroup> temp: array<u32, ${SCAN_BLOCK}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${SCAN_BLOCK}u;
  let ai = lid.x;
  let bi = lid.x + 256u;
  temp[ai] = select(0u, data[base + ai], base + ai < sp.n);
  temp[bi] = select(0u, data[base + bi], base + bi < sp.n);

  // up-sweep
  var offset = 1u;
  for (var d = ${SCAN_BLOCK / 2}u; d > 0u; d >>= 1u) {
    workgroupBarrier();
    if (lid.x < d) {
      let a = offset * (2u * lid.x + 1u) - 1u;
      let b = offset * (2u * lid.x + 2u) - 1u;
      temp[b] += temp[a];
    }
    offset <<= 1u;
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    blockSums[wid.x] = temp[${SCAN_BLOCK - 1}u];
    temp[${SCAN_BLOCK - 1}u] = 0u;
  }
  // down-sweep
  for (var d = 1u; d < ${SCAN_BLOCK}u; d <<= 1u) {
    offset >>= 1u;
    workgroupBarrier();
    if (lid.x < d) {
      let a = offset * (2u * lid.x + 1u) - 1u;
      let b = offset * (2u * lid.x + 2u) - 1u;
      let t = temp[a];
      temp[a] = temp[b];
      temp[b] += t;
    }
  }
  workgroupBarrier();
  if (base + ai < sp.n) { data[base + ai] = temp[ai]; }
  if (base + bi < sp.n) { data[base + bi] = temp[bi]; }
}
`;

/** Adds the scanned block offsets back into each block. */
export const scanAddWGSL = /* wgsl */ `
struct ScanParams { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> sp: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let add = blockSums[wid.x];
  let base = wid.x * ${SCAN_BLOCK}u;
  for (var k = 0u; k < 2u; k++) {
    let i = base + k * 256u + lid.x;
    if (i < sp.n) { data[i] += add; }
  }
}
`;

export const scatterWGSL = /* wgsl */ `
${common}
@group(0) @binding(0) var<uniform> params: PassParams;
@group(0) @binding(1) var<storage, read> ctrl: array<u32>;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> valsIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> valsOut: array<u32>;
@group(0) @binding(6) var<storage, read> hist: array<u32>;

var<workgroup> sKeys: array<u32, ${SORT_TILE}>;
var<workgroup> sVals: array<u32, ${SORT_TILE}>;
var<workgroup> sScan: array<u32, ${SORT_WG}>;
var<workgroup> sDigitStart: array<u32, 256>;

fn digitOf(key: u32) -> u32 { return (key >> params.shift) & 0xffu; }

@compute @workgroup_size(${SORT_WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let count = ctrl[0];
  let numTiles = (count + TILE - 1u) / TILE;
  let tile = wid.x;
  let tileStart = tile * TILE;
  let tileCount = select(0u, min(TILE, count - tileStart), tileStart < count);

  // Each thread owns ITEMS consecutive slots of the tile. Padding keys (0xffffffff)
  // carry digit 255 and start at the end, so a stable sort keeps them last.
  var k: array<u32, ${SORT_ITEMS}>;
  var v: array<u32, ${SORT_ITEMS}>;
  for (var j = 0u; j < ITEMS; j++) {
    let p = lid.x * ITEMS + j;
    if (p < tileCount) {
      k[j] = keysIn[tileStart + p];
      v[j] = valsIn[tileStart + p];
    } else {
      k[j] = 0xffffffffu;
      v[j] = 0xffffffffu;
    }
  }

  // Stable local sort by the 8-bit digit: 8 one-bit split passes.
  for (var bit = 0u; bit < 8u; bit++) {
    let sh = params.shift + bit;
    var zeros = 0u;
    for (var j = 0u; j < ITEMS; j++) { zeros += 1u - ((k[j] >> sh) & 1u); }

    // workgroup inclusive scan of zeros (Hillis–Steele)
    sScan[lid.x] = zeros;
    workgroupBarrier();
    for (var off = 1u; off < WG; off <<= 1u) {
      var t = 0u;
      if (lid.x >= off) { t = sScan[lid.x - off]; }
      workgroupBarrier();
      sScan[lid.x] += t;
      workgroupBarrier();
    }
    let zerosBefore = sScan[lid.x] - zeros;
    let totalZeros = sScan[WG - 1u];
    workgroupBarrier();

    var zi = zerosBefore;
    var oi = totalZeros + lid.x * ITEMS - zerosBefore;
    for (var j = 0u; j < ITEMS; j++) {
      var dst: u32;
      if (((k[j] >> sh) & 1u) == 0u) { dst = zi; zi += 1u; } else { dst = oi; oi += 1u; }
      sKeys[dst] = k[j];
      sVals[dst] = v[j];
    }
    workgroupBarrier();
    for (var j = 0u; j < ITEMS; j++) {
      k[j] = sKeys[lid.x * ITEMS + j];
      v[j] = sVals[lid.x * ITEMS + j];
    }
    workgroupBarrier();
  }

  // sKeys now holds the tile sorted by digit. Find where each digit run starts.
  sDigitStart[lid.x] = 0u;
  workgroupBarrier();
  for (var j = 0u; j < ITEMS; j++) {
    let p = lid.x * ITEMS + j;
    let d = digitOf(k[j]);
    if (p == 0u || digitOf(sKeys[p - 1u]) != d) { sDigitStart[d] = p; }
  }
  workgroupBarrier();

  if (tile < numTiles) {
    for (var j = 0u; j < ITEMS; j++) {
      let p = lid.x * ITEMS + j;
      if (p < tileCount) {
        let d = digitOf(k[j]);
        let dst = hist[d * numTiles + tile] + (p - sDigitStart[d]);
        keysOut[dst] = k[j];
        valsOut[dst] = v[j];
      }
    }
  }
}
`;
