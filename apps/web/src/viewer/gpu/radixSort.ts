import {
  SCAN_BLOCK,
  SORT_TILE,
  histogramWGSL,
  scanAddWGSL,
  scanBlocksWGSL,
  scatterWGSL,
} from './sortShaders';

/** Byte offsets inside the shared `ctrl` buffer. */
export const CTRL = {
  /** u32: number of elements to sort (written by the cull pass). */
  count: 0,
  /** 3 × u32: indirect dispatch args for per-tile kernels. */
  dispatch: 4,
  /** 4 × u32: indirect draw args (vertexCount, instanceCount, firstVertex, firstInstance). */
  draw: 16,
  size: 32,
} as const;

interface Pipelines {
  histogram: GPUComputePipeline;
  scanBlocks: GPUComputePipeline;
  scanAdd: GPUComputePipeline;
  scatter: GPUComputePipeline;
}

const pipelineCache = new WeakMap<GPUDevice, Pipelines>();

function getPipelines(device: GPUDevice): Pipelines {
  let p = pipelineCache.get(device);
  if (p) return p;
  const make = (code: string, label: string) =>
    device.createComputePipeline({
      label,
      layout: 'auto',
      compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });
  p = {
    histogram: make(histogramWGSL, 'radix-histogram'),
    scanBlocks: make(scanBlocksWGSL, 'radix-scan-blocks'),
    scanAdd: make(scanAddWGSL, 'radix-scan-add'),
    scatter: make(scatterWGSL, 'radix-scatter'),
  };
  pipelineCache.set(device, p);
  return p;
}

interface ScanLevel {
  n: number;
  blocks: number;
  data: GPUBuffer;
  sums: GPUBuffer;
  scanBG: GPUBindGroup;
  addBG: GPUBindGroup | null;
}

/**
 * GPU LSD radix sort of (u32 key, u32 value) pairs. The element count is read on the GPU
 * from ctrl[0]; ctrl must hold valid indirect dispatch args at CTRL.dispatch
 * (x = ceil(count / 1024)) before `encode` runs (see `prepareWGSL` in the renderer).
 *
 * After an even number of passes the sorted result is back in keysA / valsA.
 */
export class GpuRadixSort {
  readonly capacity: number;
  readonly passes: number;
  private readonly hist: GPUBuffer;
  private readonly levels: ScanLevel[] = [];
  private readonly passBGs: { hist: GPUBindGroup; scatter: GPUBindGroup }[] = [];
  private readonly owned: GPUBuffer[] = [];
  private readonly pipes: Pipelines;
  private readonly ctrl: GPUBuffer;

  constructor(
    device: GPUDevice,
    capacity: number,
    buffers: { keysA: GPUBuffer; keysB: GPUBuffer; valsA: GPUBuffer; valsB: GPUBuffer; ctrl: GPUBuffer },
    keyBits = 32,
  ) {
    this.capacity = Math.max(1, capacity);
    this.passes = Math.ceil(keyBits / 8);
    if (this.passes % 2 !== 0) throw new Error('keyBits must give an even number of 8-bit passes');
    this.pipes = getPipelines(device);
    this.ctrl = buffers.ctrl;

    const maxTiles = Math.ceil(this.capacity / SORT_TILE);
    const histLen = 256 * maxTiles;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.hist = device.createBuffer({ label: 'radix-hist', size: histLen * 4, usage: storage });
    this.owned.push(this.hist);

    // Multi-level scan buffers.
    let data = this.hist;
    let n = histLen;
    for (;;) {
      const blocks = Math.ceil(n / SCAN_BLOCK);
      const sums = device.createBuffer({ label: `radix-scan-sums-${this.levels.length}`, size: Math.max(blocks, 4) * 4, usage: storage });
      this.owned.push(sums);
      const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uni, 0, new Uint32Array([n, 0, 0, 0]));
      this.owned.push(uni);
      const entries = [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: data } },
        { binding: 2, resource: { buffer: sums } },
      ];
      const scanBG = device.createBindGroup({ layout: this.pipes.scanBlocks.getBindGroupLayout(0), entries });
      const addBG = blocks > 1 ? device.createBindGroup({ layout: this.pipes.scanAdd.getBindGroupLayout(0), entries }) : null;
      this.levels.push({ n, blocks, data, sums, scanBG, addBG });
      if (blocks <= 1) break;
      data = sums;
      n = blocks;
    }

    for (let pass = 0; pass < this.passes; pass++) {
      const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uni, 0, new Uint32Array([pass * 8, 0, 0, 0]));
      this.owned.push(uni);
      const even = pass % 2 === 0;
      const kIn = even ? buffers.keysA : buffers.keysB;
      const kOut = even ? buffers.keysB : buffers.keysA;
      const vIn = even ? buffers.valsA : buffers.valsB;
      const vOut = even ? buffers.valsB : buffers.valsA;
      this.passBGs.push({
        hist: device.createBindGroup({
          layout: this.pipes.histogram.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uni } },
            { binding: 1, resource: { buffer: buffers.ctrl } },
            { binding: 2, resource: { buffer: kIn } },
            { binding: 3, resource: { buffer: this.hist } },
          ],
        }),
        scatter: device.createBindGroup({
          layout: this.pipes.scatter.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uni } },
            { binding: 1, resource: { buffer: buffers.ctrl } },
            { binding: 2, resource: { buffer: kIn } },
            { binding: 3, resource: { buffer: vIn } },
            { binding: 4, resource: { buffer: kOut } },
            { binding: 5, resource: { buffer: vOut } },
            { binding: 6, resource: { buffer: this.hist } },
          ],
        }),
      });
    }
  }

  /** Records all sort passes into an open compute pass. */
  encode(pass: GPUComputePassEncoder): void {
    for (let p = 0; p < this.passes; p++) {
      const bgs = this.passBGs[p];
      pass.setPipeline(this.pipes.histogram);
      pass.setBindGroup(0, bgs.hist);
      pass.dispatchWorkgroupsIndirect(this.ctrl, CTRL.dispatch);

      // Scan the whole capacity-sized histogram (entries past 256·numTiles are stale but
      // only ever affect prefixes after the live region).
      for (const lvl of this.levels) {
        pass.setPipeline(this.pipes.scanBlocks);
        pass.setBindGroup(0, lvl.scanBG);
        pass.dispatchWorkgroups(lvl.blocks);
      }
      for (let i = this.levels.length - 1; i >= 0; i--) {
        const lvl = this.levels[i];
        if (!lvl.addBG) continue;
        pass.setPipeline(this.pipes.scanAdd);
        pass.setBindGroup(0, lvl.addBG);
        pass.dispatchWorkgroups(lvl.blocks);
      }

      pass.setPipeline(this.pipes.scatter);
      pass.setBindGroup(0, bgs.scatter);
      pass.dispatchWorkgroupsIndirect(this.ctrl, CTRL.dispatch);
    }
  }

  destroy(): void {
    for (const b of this.owned) b.destroy();
  }
}
