import { RECORD_BYTES } from '../formats';
import { CTRL, GpuRadixSort } from './radixSort';
import { UNIFORM_BYTES } from './splatShaders';

export interface ScenePipelines {
  preprocess: GPUComputePipeline;
  prepare: GPUComputePipeline;
  render: GPURenderPipeline;
}

/** All GPU resources for one loaded splat scene. */
export class SplatScene {
  readonly capacity: number;
  readonly shDegree: number;
  readonly shStride: number;
  loaded = 0;
  complete = false;
  /** Global opacity multiplier, used for crossfades. */
  alpha = 1;

  readonly splats: GPUBuffer;
  readonly sh: GPUBuffer;
  readonly projected: GPUBuffer;
  readonly keysA: GPUBuffer;
  readonly keysB: GPUBuffer;
  readonly valsA: GPUBuffer;
  readonly valsB: GPUBuffer;
  readonly ctrl: GPUBuffer;
  readonly uniforms: GPUBuffer;
  readonly sorter: GpuRadixSort | null;
  readonly preprocessBG: GPUBindGroup;
  readonly prepareBG: GPUBindGroup;
  readonly renderBG: GPUBindGroup;
  /** xyz + opacity subsample for bounds. */
  samples: Float32Array[] = [];
  /** Full positions, only kept when the CPU-sort fallback is active. */
  positions: Float32Array | null;

  private readonly device: GPUDevice;

  constructor(
    device: GPUDevice,
    pipes: ScenePipelines,
    capacity: number,
    shDegree: number,
    shStride: number,
    useGpuSort: boolean,
  ) {
    this.device = device;
    this.capacity = Math.max(1, capacity);
    this.shDegree = shDegree;
    this.shStride = shStride;
    const n = this.capacity;
    const S = GPUBufferUsage.STORAGE;
    const D = GPUBufferUsage.COPY_DST;
    const C = GPUBufferUsage.COPY_SRC;
    const mk = (label: string, size: number, usage: number) =>
      device.createBuffer({ label, size: Math.max(16, Math.ceil(size / 4) * 4), usage });

    this.splats = mk('splats', n * RECORD_BYTES, S | D);
    this.sh = mk('sh', shDegree > 0 ? n * shStride * 4 : 16, S | D);
    this.projected = mk('projected', n * 32, S);
    this.keysA = mk('keysA', n * 4, S | C);
    this.keysB = mk('keysB', n * 4, S);
    this.valsA = mk('valsA', n * 4, S | D | C);
    this.valsB = mk('valsB', n * 4, S);
    this.ctrl = mk('ctrl', CTRL.size, S | D | C | GPUBufferUsage.INDIRECT);
    this.uniforms = mk('uniforms', UNIFORM_BYTES, GPUBufferUsage.UNIFORM | D);
    this.sorter = useGpuSort
      ? new GpuRadixSort(device, n, {
          keysA: this.keysA, keysB: this.keysB, valsA: this.valsA, valsB: this.valsB, ctrl: this.ctrl,
        })
      : null;
    this.positions = useGpuSort ? null : new Float32Array(n * 3);

    this.preprocessBG = device.createBindGroup({
      label: 'preprocess-bg',
      layout: pipes.preprocess.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer: this.splats } },
        { binding: 2, resource: { buffer: this.sh } },
        { binding: 3, resource: { buffer: this.projected } },
        { binding: 4, resource: { buffer: this.keysA } },
        { binding: 5, resource: { buffer: this.valsA } },
        { binding: 6, resource: { buffer: this.ctrl } },
      ],
    });
    this.prepareBG = device.createBindGroup({
      label: 'prepare-bg',
      layout: pipes.prepare.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ctrl } }],
    });
    this.renderBG = device.createBindGroup({
      label: 'render-bg',
      layout: pipes.render.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer: this.projected } },
        { binding: 2, resource: { buffer: this.valsA } },
      ],
    });
  }

  /** Upload a decoded batch at splat index `start`. */
  write(start: number, count: number, packed: ArrayBuffer, sh: ArrayBuffer | null, sample: Float32Array): void {
    const end = Math.min(start + count, this.capacity);
    const n = end - start;
    if (n <= 0) return;
    this.device.queue.writeBuffer(this.splats, start * RECORD_BYTES, packed, 0, n * RECORD_BYTES);
    if (sh && this.shDegree > 0) this.device.queue.writeBuffer(this.sh, start * this.shStride * 4, sh, 0, n * this.shStride * 4);
    if (this.positions) {
      const f = new Float32Array(packed);
      for (let i = 0; i < n; i++) {
        this.positions[(start + i) * 3] = f[i * 8];
        this.positions[(start + i) * 3 + 1] = f[i * 8 + 1];
        this.positions[(start + i) * 3 + 2] = f[i * 8 + 2];
      }
    }
    this.samples.push(sample);
    this.loaded = Math.max(this.loaded, end);
  }

  allSamples(): Float32Array {
    const total = this.samples.reduce((s, a) => s + a.length, 0);
    const out = new Float32Array(total);
    let o = 0;
    for (const a of this.samples) {
      out.set(a, o);
      o += a.length;
    }
    return out;
  }

  destroy(): void {
    this.sorter?.destroy();
    for (const b of [this.splats, this.sh, this.projected, this.keysA, this.keysB, this.valsA, this.valsB, this.ctrl, this.uniforms]) {
      b.destroy();
    }
  }
}
