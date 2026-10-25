/// <reference lib="webworker" />
// Streams and decodes splat files off the main thread, posting GPU-ready batches.

import {
  SPLAT_FILE_STRIDE,
  decodePly,
  decodeSplat,
  detectFormat,
  parsePlyHeader,
  shStrideU32,
  type DecodedBatch,
  type PlyHeader,
  type SplatFormat,
} from './formats';

export type LoaderRequest =
  | { type: 'load'; id: number; url: string; format?: SplatFormat }
  | { type: 'parse'; id: number; buffer: ArrayBuffer; format?: SplatFormat; name?: string }
  | { type: 'cancel'; id: number };

export type LoaderMessage =
  | { type: 'header'; id: number; count: number; shDegree: number; shStride: number; format: SplatFormat }
  | {
      type: 'batch';
      id: number;
      start: number;
      count: number;
      packed: ArrayBuffer;
      sh: ArrayBuffer | null;
      sample: Float32Array;
      receivedBytes: number;
      totalBytes: number;
    }
  | { type: 'done'; id: number; count: number }
  | { type: 'error'; id: number; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const cancelled = new Set<number>();

/** Splats decoded per posted batch (~ 4 MB of GPU data). */
const BATCH = 131072;

function post(msg: LoaderMessage, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}

function postBatch(id: number, start: number, b: DecodedBatch, receivedBytes: number, totalBytes: number): void {
  const packed = b.packed.buffer as ArrayBuffer;
  const sh = b.sh ? (b.sh.buffer as ArrayBuffer) : null;
  const transfer: Transferable[] = [packed, b.sample.buffer as ArrayBuffer];
  if (sh) transfer.push(sh);
  post(
    { type: 'batch', id, start, count: b.count, packed, sh, sample: b.sample, receivedBytes, totalBytes },
    transfer,
  );
}

/** Incremental decoder fed with arbitrary byte chunks. */
class StreamDecoder {
  private buf = new Uint8Array(0);
  private len = 0;
  private consumed = 0; // bytes of buf already decoded (body-relative after header)
  private header: PlyHeader | null = null;
  private bodyStart = 0;
  private done = 0;
  private total = -1;
  private stride = SPLAT_FILE_STRIDE;
  private format: SplatFormat | null;
  private readonly id: number;
  private readonly name: string | null;
  private readonly totalBytes: number;
  received = 0;

  constructor(id: number, format: SplatFormat | undefined, name: string | null, totalBytes: number) {
    this.id = id;
    this.format = format ?? null;
    this.name = name;
    this.totalBytes = totalBytes;
  }

  push(chunk: Uint8Array): void {
    this.received += chunk.length;
    if (this.len + chunk.length > this.buf.length) {
      const next = new Uint8Array(Math.max(this.buf.length * 2, this.len + chunk.length, 1 << 20));
      next.set(this.buf.subarray(0, this.len));
      this.buf = next;
    }
    this.buf.set(chunk, this.len);
    this.len += chunk.length;
    this.process(false);
  }

  finish(): void {
    this.process(true);
    if (this.total < 0) this.total = this.done;
    post({ type: 'done', id: this.id, count: this.done });
  }

  private ensureHeader(final: boolean): boolean {
    if (this.total >= 0) return true;
    if (!this.format) {
      if (this.len < 4 && !final) return false;
      this.format = detectFormat(this.name, this.buf.subarray(0, Math.min(this.len, 4)));
    }
    if (this.format === 'ply') {
      const h = parsePlyHeader(this.buf.subarray(0, this.len));
      if (!h) {
        if (final) throw new Error('Truncated PLY header');
        return false;
      }
      this.header = h;
      this.bodyStart = h.headerLength;
      this.stride = h.stride;
      this.total = h.vertexCount;
      post({
        type: 'header', id: this.id, count: h.vertexCount, shDegree: h.shDegree,
        shStride: h.shDegree > 0 ? shStrideU32(h.shDegree) : 0, format: 'ply',
      });
      return true;
    }
    // .splat has no header; total comes from Content-Length (or the full buffer).
    if (this.totalBytes > 0) this.total = Math.floor(this.totalBytes / SPLAT_FILE_STRIDE);
    else if (final) this.total = Math.floor(this.len / SPLAT_FILE_STRIDE);
    else return false; // unknown length: buffer until the end
    post({ type: 'header', id: this.id, count: this.total, shDegree: 0, shStride: 0, format: 'splat' });
    return true;
  }

  private process(final: boolean): void {
    if (cancelled.has(this.id)) return;
    if (!this.ensureHeader(final)) return;
    for (;;) {
      const availableBytes = this.len - this.bodyStart - this.consumed;
      let n = Math.floor(availableBytes / this.stride);
      n = Math.min(n, this.total - this.done);
      if (n <= 0) break;
      if (n < BATCH && !final && this.done + n < this.total) break; // wait for a full batch
      n = Math.min(n, BATCH);
      const offset = this.bodyStart + this.consumed;
      const batch =
        this.format === 'ply'
          ? decodePly(this.header!, this.buf, offset, n, this.done)
          : decodeSplat(this.buf, offset, n, this.done);
      postBatch(this.id, this.done, batch, this.received, this.totalBytes);
      this.done += n;
      this.consumed += n * this.stride;
    }
    // compact: drop decoded bytes (keep header region semantics by rebasing)
    const keepFrom = this.bodyStart + this.consumed;
    if (keepFrom > 8 << 20) {
      this.buf.copyWithin(0, keepFrom, this.len);
      this.len -= keepFrom;
      this.bodyStart = 0;
      this.consumed = 0;
    }
  }
}

async function load(id: number, url: string, format?: SplatFormat): Promise<void> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  const totalBytes = Number(res.headers.get('content-length') ?? 0) || 0;
  const encoded = res.headers.get('content-encoding');
  // With content-encoding the length is the compressed size, so don't trust it for .splat.
  const dec = new StreamDecoder(id, format, url, encoded && encoded !== 'identity' ? 0 : totalBytes);
  if (!res.body) {
    dec.push(new Uint8Array(await res.arrayBuffer()));
    dec.finish();
    return;
  }
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (cancelled.has(id)) {
      await reader.cancel();
      return;
    }
    if (done) break;
    dec.push(value);
  }
  dec.finish();
}

ctx.onmessage = async (e: MessageEvent<LoaderRequest>) => {
  const msg = e.data;
  if (msg.type === 'cancel') {
    cancelled.add(msg.id);
    return;
  }
  try {
    if (msg.type === 'load') await load(msg.id, msg.url, msg.format);
    else {
      const dec = new StreamDecoder(msg.id, msg.format, msg.name ?? null, msg.buffer.byteLength);
      dec.push(new Uint8Array(msg.buffer));
      dec.finish();
    }
  } catch (err) {
    post({ type: 'error', id: msg.id, message: err instanceof Error ? err.message : String(err) });
  }
};
