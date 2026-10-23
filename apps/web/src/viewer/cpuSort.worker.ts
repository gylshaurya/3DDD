/// <reference lib="webworker" />
// Fallback depth sort used only if the GPU radix sort fails its startup self-test.
// 16-bit counting sort of view depth, back-to-front.

export type CpuSortRequest =
  | { type: 'positions'; positions: Float32Array; count: number }
  | { type: 'sort'; view: Float32Array; count: number; seq: number };

export type CpuSortResult = { type: 'sorted'; order: Uint32Array; count: number; seq: number };

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let positions: Float32Array = new Float32Array(0);

ctx.onmessage = (e: MessageEvent<CpuSortRequest>) => {
  const msg = e.data;
  if (msg.type === 'positions') {
    positions = msg.positions;
    return;
  }
  const n = Math.min(msg.count, positions.length / 3);
  const v = msg.view;
  const depth = new Float32Array(n);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    const d = v[2] * x + v[6] * y + v[10] * z + v[14];
    depth[i] = d;
    if (d < min) min = d;
    if (d > max) max = d;
  }
  const BUCKETS = 65536;
  const scale = (BUCKETS - 1) / Math.max(max - min, 1e-6);
  const counts = new Uint32Array(BUCKETS);
  const bucket = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    // far first
    const b = (BUCKETS - 1) - Math.floor((depth[i] - min) * scale);
    bucket[i] = b;
    counts[b]++;
  }
  let sum = 0;
  for (let b = 0; b < BUCKETS; b++) {
    const c = counts[b];
    counts[b] = sum;
    sum += c;
  }
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[counts[bucket[i]]++] = i;
  const res: CpuSortResult = { type: 'sorted', order, count: n, seq: msg.seq };
  ctx.postMessage(res, [order.buffer]);
};
