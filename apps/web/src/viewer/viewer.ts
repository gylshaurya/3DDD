import { computeBounds, estimateUp, type SceneBounds } from './bounds';
import { CameraControls, OrbitCamera, type UpAxis } from './camera';
import type { CpuSortRequest, CpuSortResult } from './cpuSort.worker';
import CpuSortWorker from './cpuSort.worker?worker';
import type { SplatFormat } from './formats';
import { CTRL, GpuRadixSort } from './gpu/radixSort';
import { SplatScene, type ScenePipelines } from './gpu/splatScene';
import { UNIFORM_BYTES, preprocessWGSL, prepareWGSL, renderWGSL } from './gpu/splatShaders';
import type { LoaderMessage, LoaderRequest } from './loader.worker';
import LoaderWorker from './loader.worker?worker';
import type { Vec3 } from './math';

export interface ViewerOptions {
  /** World up axis of the scene. Default 'auto' (ground-plane estimate, else y-down as in COLMAP). */
  up?: UpAxis;
  /** Clear color, linear 0..1. */
  background?: [number, number, number];
  autoOrbit?: boolean;
  /** Cap on devicePixelRatio (default 2). */
  maxDpr?: number;
  /** Vertical field of view in degrees (default 50). */
  fovDeg?: number;
  /** Force the CPU-sort fallback path (testing). */
  forceCpuSort?: boolean;
  /** Crossfade duration for live snapshot swaps (ms, default 450). */
  crossfadeMs?: number;
  /** Called on unrecoverable GPU errors (device lost, etc.). */
  onError?: (err: Error) => void;
}

export interface ViewerStats {
  fps: number;
  /** Splats loaded in the active scene. */
  splatCount: number;
  /** Splats that survived culling last frame (GPU path; sampled). */
  visibleCount: number;
  /** GPU time of the radix sort in ms (needs timestamp-query; -1 if unavailable). */
  sortMs: number;
  /** GPU time of the whole frame in ms (-1 if unavailable). */
  gpuFrameMs: number;
  /** true when the GPU radix sort is active, false on the CPU fallback. */
  gpuSort: boolean;
  /** 0..1 progress of the current load. */
  loadProgress: number;
}

export interface LoadOptions {
  format?: SplatFormat;
  /** Keep the current camera instead of fitting the new scene. */
  keepCamera?: boolean;
  /** For buffer loads: a filename hint for format detection. */
  name?: string;
  onProgress?: (loaded: number, total: number) => void;
}

export class WebGPUUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGPUUnavailableError';
  }
}

interface PendingLoad {
  id: number;
  scene: SplatScene | null;
  opts: LoadOptions;
  progressive: boolean;
  resolve: () => void;
  reject: (e: Error) => void;
  fitted: boolean;
  total: number;
}

interface FadingScene {
  scene: SplatScene;
  start: number;
}

const STATS_INTERVAL_MS = 250;

export class Viewer {
  readonly canvas: HTMLCanvasElement;
  readonly device: GPUDevice;
  readonly camera: OrbitCamera;
  readonly controls: CameraControls;
  bounds: SceneBounds | null = null;

  private readonly context: GPUCanvasContext;
  private readonly pipes: ScenePipelines;
  private readonly opts: Required<Omit<ViewerOptions, 'onError' | 'up'>> & Pick<ViewerOptions, 'onError'>;
  private readonly loader: Worker;
  private cpuSorter: Worker | null = null;
  private readonly gpuSort: boolean;
  private upMode: UpAxis;

  private active: SplatScene | null = null;
  private fading: FadingScene[] = [];
  private loads = new Map<number, PendingLoad>();
  private nextLoadId = 1;
  private userInteractedAt = 0;
  private loadStartedAt = 0;

  private raf = 0;
  private lastT = 0;
  private dirty = true;
  private disposed = false;
  private resizeObs: ResizeObserver;

  // stats
  private statsCbs = new Set<(s: ViewerStats) => void>();
  private frames = 0;
  private fpsWindowStart = 0;
  private stats: ViewerStats;
  private querySet: GPUQuerySet | null = null;
  private queryResolve: GPUBuffer | null = null;
  private readback: GPUBuffer | null = null;
  private readbackBusy = false;
  private lastStatsAt = 0;

  // CPU sort fallback state
  private cpuSortSeq = 0;
  private cpuSortInFlight = false;
  private cpuSortedCount = 0;
  private cpuSortScene: SplatScene | null = null;
  private lastSortedView = '';

  private readonly uniformData = new ArrayBuffer(UNIFORM_BYTES);
  private readonly uniformF = new Float32Array(this.uniformData);
  private readonly uniformU = new Uint32Array(this.uniformData);

  /** Use `createViewer` instead. */
  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    gpuSort: boolean,
    opts: ViewerOptions,
  ) {
    void format;
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.gpuSort = gpuSort;
    this.opts = {
      background: opts.background ?? [0.035, 0.037, 0.045],
      autoOrbit: opts.autoOrbit ?? false,
      maxDpr: opts.maxDpr ?? 2,
      fovDeg: opts.fovDeg ?? 50,
      forceCpuSort: opts.forceCpuSort ?? false,
      crossfadeMs: opts.crossfadeMs ?? 450,
      onError: opts.onError,
    };
    this.upMode = opts.up ?? 'auto';
    this.camera = new OrbitCamera(this.upMode);
    this.camera.fovY = (this.opts.fovDeg * Math.PI) / 180;
    this.camera.autoOrbit = this.opts.autoOrbit;
    this.controls = new CameraControls(canvas, this.camera);
    this.controls.onInteract = () => {
      this.userInteractedAt = performance.now();
      this.dirty = true;
    };
    this.controls.onKeyAction = (a) => {
      if (a === 'fit') this.fitToScene();
      else if (a === 'autoOrbit') this.setAutoOrbit(!this.camera.autoOrbit);
      else this.setFlyMode(!this.camera.flyMode);
    };
    this.stats = {
      fps: 0, splatCount: 0, visibleCount: 0, sortMs: -1, gpuFrameMs: -1, gpuSort: gpuSort, loadProgress: 1,
    };

    const mod = (code: string, label: string) => device.createShaderModule({ code, label });
    const preMod = mod(preprocessWGSL, 'preprocess');
    const renderMod = mod(renderWGSL, 'render');
    this.pipes = {
      preprocess: device.createComputePipeline({ label: 'preprocess', layout: 'auto', compute: { module: preMod, entryPoint: 'main' } }),
      prepare: device.createComputePipeline({ label: 'prepare', layout: 'auto', compute: { module: mod(prepareWGSL, 'prepare'), entryPoint: 'main' } }),
      render: device.createRenderPipeline({
        label: 'splat-render',
        layout: 'auto',
        vertex: { module: renderMod, entryPoint: 'vs' },
        fragment: {
          module: renderMod,
          entryPoint: 'fs',
          targets: [{
            format,
            blend: {
              // premultiplied alpha, back-to-front "over"
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-strip' },
      }),
    };

    if (device.features.has('timestamp-query')) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
      this.queryResolve = device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    this.readback = device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    this.loader = new LoaderWorker();
    this.loader.onmessage = (e: MessageEvent<LoaderMessage>) => this.onLoaderMessage(e.data);
    if (!gpuSort) {
      this.cpuSorter = new CpuSortWorker();
      this.cpuSorter.onmessage = (e: MessageEvent<CpuSortResult>) => this.onCpuSorted(e.data);
    }

    this.resizeObs = new ResizeObserver(() => {
      this.dirty = true;
    });
    this.resizeObs.observe(canvas);

    device.lost.then((info) => {
      if (this.disposed) return;
      this.opts.onError?.(new Error(`WebGPU device lost: ${info.message || info.reason}`));
    });
    device.addEventListener('uncapturederror', (ev) => {
      const err = (ev as GPUUncapturedErrorEvent).error;
      console.error('[splat-viewer] WebGPU error:', err.message);
    });

    this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  /** @internal factory */
  static async create(canvas: HTMLCanvasElement, opts: ViewerOptions = {}): Promise<Viewer> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new WebGPUUnavailableError(
        window.isSecureContext
          ? 'This browser does not support WebGPU. Use a recent Chrome, Edge, or Safari 26+.'
          : 'WebGPU requires a secure context (https:// or localhost).',
      );
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new WebGPUUnavailableError('No WebGPU adapter is available (GPU blocklisted or disabled).');
    const features: GPUFeatureName[] = [];
    if (adapter.features.has('timestamp-query')) features.push('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
      },
    });
    const context = canvas.getContext('webgpu');
    if (!context) throw new WebGPUUnavailableError('Could not create a WebGPU canvas context.');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    let gpuSort = !opts.forceCpuSort;
    if (gpuSort) {
      try {
        const r = await testGpuSort(device, 40000);
        if (!r.correct) {
          console.warn('[splat-viewer] GPU radix sort self-test failed; using CPU sort fallback');
          gpuSort = false;
        }
      } catch (err) {
        console.warn('[splat-viewer] GPU radix sort unavailable; using CPU sort fallback', err);
        gpuSort = false;
      }
    }
    return new Viewer(canvas, device, context, format, gpuSort, opts);
  }

  // ------------------------------------------------------------------ public API

  /** Load a scene from a URL / Blob / ArrayBuffer, streaming progressively for URLs. */
  load(src: string | ArrayBuffer | Blob, opts: LoadOptions = {}): Promise<void> {
    if (typeof src === 'string') return this.startLoad({ type: 'load', id: 0, url: src, format: opts.format }, opts, true);
    if (src instanceof Blob) {
      return src.arrayBuffer().then((buf) => this.load(buf, { name: (src as File).name, ...opts }));
    }
    return this.startLoad({ type: 'parse', id: 0, buffer: src, format: opts.format, name: opts.name }, opts, true, [src]);
  }

  /**
   * Swap in a new scene from an in-memory buffer (e.g. a live training snapshot).
   * The current scene keeps rendering until the new one is fully decoded, then the two
   * crossfade. The camera is kept unless `keepCamera` is false. The buffer is transferred.
   */
  setSplatsFromBuffer(buf: ArrayBuffer, format?: SplatFormat, opts: { keepCamera?: boolean } = {}): Promise<void> {
    return this.startLoad(
      { type: 'parse', id: 0, buffer: buf, format },
      { format, keepCamera: opts.keepCamera ?? true },
      false,
      [buf],
    );
  }

  setAutoOrbit(on: boolean): void {
    this.camera.autoOrbit = on;
    this.dirty = true;
  }

  get autoOrbit(): boolean {
    return this.camera.autoOrbit;
  }

  setFlyMode(on: boolean): void {
    this.camera.flyMode = on;
    this.dirty = true;
  }

  setUpAxis(up: UpAxis): void {
    this.upMode = up;
    this.fitToScene();
  }

  setBackground(rgb: [number, number, number]): void {
    this.opts.background = rgb;
    this.dirty = true;
  }

  /** Frame the robust bounds of the active scene. */
  fitToScene(snap = true): void {
    if (!this.active) return;
    const samples = this.active.allSamples();
    const b = computeBounds(samples);
    if (!b) return;
    this.bounds = b;
    this.camera.setUp(this.upMode === 'auto' ? estimateUp(samples, b) : this.upMode);
    this.camera.fit(b.focusCenter, b.focusRadius, snap, b.radius);
    this.dirty = true;
  }

  /** Place the camera explicitly (world-space eye and look-at target). */
  lookAt(eye: Vec3, target: Vec3, snap = true): void {
    this.camera.lookFrom(eye, target, snap);
    this.dirty = true;
  }

  onStats(cb: (s: ViewerStats) => void): () => void {
    this.statsCbs.add(cb);
    return () => this.statsCbs.delete(cb);
  }

  getStats(): ViewerStats {
    return { ...this.stats };
  }

  /** Render immediately and return the frame as a PNG data URL. */
  captureFrame(): string {
    this.renderFrame(performance.now());
    return this.canvas.toDataURL('image/png');
  }

  /** Splats in the active scene. */
  get splatCount(): number {
    return this.active?.loaded ?? 0;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resizeObs.disconnect();
    this.controls.dispose();
    this.loader.terminate();
    this.cpuSorter?.terminate();
    for (const l of this.loads.values()) {
      l.scene?.destroy();
      l.reject(new Error('Viewer disposed'));
    }
    this.loads.clear();
    this.active?.destroy();
    for (const f of this.fading) f.scene.destroy();
    this.device.destroy();
  }

  // ------------------------------------------------------------------ loading

  private startLoad(
    req: LoaderRequest & { type: 'load' | 'parse' },
    opts: LoadOptions,
    progressive: boolean,
    transfer: Transferable[] = [],
  ): Promise<void> {
    const id = this.nextLoadId++;
    // A new progressive load supersedes any in-flight loads.
    if (progressive) {
      for (const l of this.loads.values()) {
        this.loader.postMessage({ type: 'cancel', id: l.id } satisfies LoaderRequest);
        if (l.scene && l.scene !== this.active) l.scene.destroy();
        l.reject(new Error('Load superseded'));
      }
      this.loads.clear();
      this.loadStartedAt = performance.now();
    }
    return new Promise<void>((resolve, reject) => {
      this.loads.set(id, { id, scene: null, opts, progressive, resolve, reject, fitted: false, total: 0 });
      this.stats.loadProgress = 0;
      this.loader.postMessage({ ...req, id }, transfer);
    });
  }

  private onLoaderMessage(msg: LoaderMessage): void {
    const l = this.loads.get(msg.id);
    if (!l || this.disposed) return;
    switch (msg.type) {
      case 'header': {
        l.total = msg.count;
        l.scene = new SplatScene(this.device, this.pipes, msg.count, msg.shDegree, msg.shStride, this.gpuSort);
        if (l.progressive) this.activate(l.scene, false);
        break;
      }
      case 'batch': {
        if (!l.scene) return;
        l.scene.write(msg.start, msg.count, msg.packed, msg.sh, msg.sample);
        const progress = l.total > 0 ? l.scene.loaded / l.total : 0;
        if (l.progressive) {
          this.stats.loadProgress = progress;
          l.opts.onProgress?.(l.scene.loaded, l.total);
          if (!l.fitted && !l.opts.keepCamera) {
            this.fitToScene(true);
            l.fitted = true;
          }
          this.dirty = true;
        }
        break;
      }
      case 'done': {
        this.loads.delete(msg.id);
        const scene = l.scene;
        if (!scene) {
          l.reject(new Error('Empty scene'));
          return;
        }
        scene.complete = true;
        if (l.progressive) {
          this.stats.loadProgress = 1;
          // refit with the complete sample set unless the user already moved the camera
          if (!l.opts.keepCamera && this.userInteractedAt < this.loadStartedAt) this.fitToScene(false);
        } else {
          const first = !this.active;
          this.activate(scene, true);
          if (first || l.opts.keepCamera === false) this.fitToScene(true);
        }
        if (!this.gpuSort && scene.positions) this.sendPositionsToCpuSorter(scene);
        this.dirty = true;
        l.resolve();
        break;
      }
      case 'error': {
        this.loads.delete(msg.id);
        if (l.scene && l.scene !== this.active) l.scene.destroy();
        this.stats.loadProgress = 1;
        l.reject(new Error(msg.message));
        break;
      }
    }
  }

  /** Make `scene` the active one; the previous scene crossfades out or is dropped. */
  private activate(scene: SplatScene, crossfade: boolean): void {
    const prev = this.active;
    this.active = scene;
    scene.alpha = 1;
    if (prev) {
      if (crossfade && prev.loaded > 0) this.fading.push({ scene: prev, start: performance.now() });
      else prev.destroy();
    }
    if (!this.gpuSort) {
      this.cpuSortedCount = 0;
      this.lastSortedView = '';
    }
    this.dirty = true;
  }

  // ------------------------------------------------------------------ CPU fallback

  private sendPositionsToCpuSorter(scene: SplatScene): void {
    if (!this.cpuSorter || !scene.positions) return;
    this.cpuSortScene = scene;
    const msg: CpuSortRequest = { type: 'positions', positions: scene.positions.slice(0, scene.loaded * 3), count: scene.loaded };
    this.cpuSorter.postMessage(msg);
    this.lastSortedView = '';
  }

  private requestCpuSort(view: Float32Array): void {
    if (!this.cpuSorter || this.cpuSortInFlight || !this.active || this.cpuSortScene !== this.active) return;
    const key = Array.from(view, (v) => v.toFixed(3)).join(',');
    if (key === this.lastSortedView) return;
    this.lastSortedView = key;
    this.cpuSortInFlight = true;
    const msg: CpuSortRequest = { type: 'sort', view: view.slice(), count: this.active.loaded, seq: ++this.cpuSortSeq };
    this.cpuSorter.postMessage(msg);
  }

  private onCpuSorted(res: CpuSortResult): void {
    this.cpuSortInFlight = false;
    if (!this.active || this.disposed) return;
    this.device.queue.writeBuffer(this.active.valsA, 0, res.order.buffer as ArrayBuffer, 0, res.count * 4);
    this.cpuSortedCount = res.count;
    this.dirty = true;
  }

  // ------------------------------------------------------------------ frame loop

  private frame(t: number): void {
    if (this.disposed) return;
    this.raf = requestAnimationFrame((tt) => this.frame(tt));
    const dt = this.lastT ? Math.min(0.1, (t - this.lastT) / 1000) : 1 / 60;
    this.lastT = t;
    const keyMoved = this.controls.tick(dt);
    const camMoved = this.camera.update(dt);
    if (this.fading.length) this.dirty = true;
    if (!(this.dirty || camMoved || keyMoved)) {
      this.maybeEmitStats(t, false);
      return;
    }
    this.dirty = false;
    this.renderFrame(t);
    this.maybeEmitStats(t, true);
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, this.opts.maxDpr);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  private writeUniforms(scene: SplatScene, view: Float32Array, eye: Vec3, alpha: number, compact: boolean): void {
    const f = this.uniformF, u = this.uniformU;
    const w = this.canvas.width, h = this.canvas.height;
    const fy = h / (2 * Math.tan(this.camera.fovY / 2));
    f.set(view, 0);
    f[16] = eye[0]; f[17] = eye[1]; f[18] = eye[2]; f[19] = 1;
    f[20] = fy; f[21] = fy;
    f[22] = w; f[23] = h;
    u[24] = scene.loaded;
    u[25] = scene.shDegree;
    u[26] = scene.shStride;
    u[27] = compact ? 1 : 0;
    f[28] = alpha;
    f[29] = 1;
    f[30] = Math.max(1e-4, this.camera.sceneRadius * 0.03);
    this.device.queue.writeBuffer(scene.uniforms, 0, this.uniformData);
  }

  private renderFrame(now: number): void {
    this.resize();
    const view = this.camera.viewMatrix();
    const eye = this.camera.eye;

    // crossfade bookkeeping: the new scene draws first, old scenes fade out on top
    const draws: { scene: SplatScene; alpha: number }[] = [];
    if (this.active && this.active.loaded > 0) draws.push({ scene: this.active, alpha: 1 });
    this.fading = this.fading.filter((f) => {
      const t = (now - f.start) / this.opts.crossfadeMs;
      if (t >= 1) {
        f.scene.destroy();
        return false;
      }
      const e = t * t * (3 - 2 * t);
      draws.push({ scene: f.scene, alpha: 1 - e });
      return true;
    });

    const enc = this.device.createCommandEncoder({ label: 'splat-frame' });
    const timing = this.querySet && this.gpuSort && draws.length > 0;
    for (let di = 0; di < draws.length; di++) {
      const { scene, alpha } = draws[di];
      this.writeUniforms(scene, view, eye, alpha, this.gpuSort);
      const wg = Math.ceil(scene.loaded / 256);
      const wgX = Math.min(wg, 65535);
      const wgY = Math.ceil(wg / 65535);
      if (this.gpuSort && scene.sorter) {
        enc.clearBuffer(scene.ctrl, 0, 4);
        const first = di === 0 && timing;
        const cp = enc.beginComputePass({
          label: 'cull',
          timestampWrites: first ? { querySet: this.querySet!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined,
        });
        cp.setPipeline(this.pipes.preprocess);
        cp.setBindGroup(0, scene.preprocessBG);
        cp.dispatchWorkgroups(wgX, wgY);
        cp.setPipeline(this.pipes.prepare);
        cp.setBindGroup(0, scene.prepareBG);
        cp.dispatchWorkgroups(1);
        cp.end();
        const sp = enc.beginComputePass({
          label: 'sort',
          timestampWrites: first ? { querySet: this.querySet!, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } : undefined,
        });
        scene.sorter.encode(sp);
        sp.end();
      } else {
        const cp = enc.beginComputePass({ label: 'project' });
        cp.setPipeline(this.pipes.preprocess);
        cp.setBindGroup(0, scene.preprocessBG);
        cp.dispatchWorkgroups(wgX, wgY);
        cp.end();
      }
    }
    if (!this.gpuSort) this.requestCpuSort(view);

    const [r, g, b] = this.opts.background;
    const rp = enc.beginRenderPass({
      label: 'splats',
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r, g, b, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    rp.setPipeline(this.pipes.render);
    for (const { scene } of draws) {
      rp.setBindGroup(0, scene.renderBG);
      if (this.gpuSort) rp.drawIndirect(scene.ctrl, CTRL.draw);
      else if (scene === this.active && this.cpuSortedCount > 0) rp.draw(4, Math.min(this.cpuSortedCount, scene.loaded));
    }
    rp.end();

    // sampled stats readback (visible count + timestamps)
    const wantStats = this.readback && !this.readbackBusy && now - this.lastStatsAt > STATS_INTERVAL_MS && draws.length > 0;
    if (wantStats) {
      if (timing) {
        enc.resolveQuerySet(this.querySet!, 0, 4, this.queryResolve!, 0);
        enc.copyBufferToBuffer(this.queryResolve!, 0, this.readback!, 0, 32);
      }
      if (this.gpuSort) enc.copyBufferToBuffer(draws[0].scene.ctrl, 0, this.readback!, 32, 32);
    }
    this.device.queue.submit([enc.finish()]);
    this.frames++;

    if (wantStats) {
      this.readbackBusy = true;
      this.lastStatsAt = now;
      const rb = this.readback!;
      rb.mapAsync(GPUMapMode.READ).then(() => {
        const data = rb.getMappedRange();
        if (timing) {
          const ts = new BigUint64Array(data, 0, 4);
          const cull = Number(ts[1] - ts[0]) / 1e6;
          const sort = Number(ts[3] - ts[2]) / 1e6;
          if (sort >= 0 && sort < 1000) this.stats.sortMs = sort;
          if (cull >= 0 && sort >= 0 && sort < 1000) this.stats.gpuFrameMs = cull + sort;
        }
        if (this.gpuSort) this.stats.visibleCount = new Uint32Array(data, 32, 8)[0];
        rb.unmap();
        this.readbackBusy = false;
      }).catch(() => {
        this.readbackBusy = false;
      });
    }
    if (!this.gpuSort) this.stats.visibleCount = this.cpuSortedCount;
  }

  private maybeEmitStats(now: number, rendered: boolean): void {
    if (!this.fpsWindowStart) this.fpsWindowStart = now;
    const elapsed = now - this.fpsWindowStart;
    if (elapsed < 500) return;
    // Only update fps while frames are actually being drawn; idle (nothing moving) keeps the last value.
    if (this.frames > 2) this.stats.fps = (this.frames * 1000) / elapsed;
    this.frames = 0;
    this.fpsWindowStart = now;
    this.stats.splatCount = this.active?.loaded ?? 0;
    void rendered;
    const s = this.getStats();
    for (const cb of this.statsCbs) cb(s);
  }
}

/**
 * Sorts `n` random (key, value) pairs on the GPU and checks the result against a stable
 * CPU sort. Returns correctness and the GPU sort time (ms, from submit to completion,
 * best of `runs`).
 */
export async function testGpuSort(
  device: GPUDevice,
  n: number,
  runs = 1,
): Promise<{ correct: boolean; ms: number; firstMismatch: number }> {
  const keys = new Uint32Array(n);
  const vals = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    // many duplicates to exercise stability
    keys[i] = ((Math.random() * 0x100000000) >>> 0) & 0xfff0ff0f;
    vals[i] = i;
  }
  const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, C = GPUBufferUsage.COPY_SRC;
  const mk = (size: number, usage: number) => device.createBuffer({ size: Math.max(16, size), usage });
  const keysA = mk(n * 4, S | D | C), keysB = mk(n * 4, S), valsA = mk(n * 4, S | D | C), valsB = mk(n * 4, S);
  const ctrl = mk(CTRL.size, S | D | GPUBufferUsage.INDIRECT);
  const tiles = Math.ceil(n / 1024);
  device.queue.writeBuffer(ctrl, 0, new Uint32Array([n, tiles, 1, 1, 0, 0, 0, 0]));
  const sorter = new GpuRadixSort(device, n, { keysA, keysB, valsA, valsB, ctrl });
  const outK = mk(n * 4, GPUBufferUsage.MAP_READ | D);
  const outV = mk(n * 4, GPUBufferUsage.MAP_READ | D);

  let best = Infinity;
  for (let r = 0; r < Math.max(1, runs); r++) {
    device.queue.writeBuffer(keysA, 0, keys);
    device.queue.writeBuffer(valsA, 0, vals);
    await device.queue.onSubmittedWorkDone();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    sorter.encode(pass);
    pass.end();
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    best = Math.min(best, performance.now() - t0);
  }
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(keysA, 0, outK, 0, n * 4);
  enc.copyBufferToBuffer(valsA, 0, outV, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await Promise.all([outK.mapAsync(GPUMapMode.READ), outV.mapAsync(GPUMapMode.READ)]);
  const gk = new Uint32Array(outK.getMappedRange().slice(0));
  const gv = new Uint32Array(outV.getMappedRange().slice(0));
  outK.unmap();
  outV.unmap();

  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => keys[a] - keys[b] || a - b); // stable reference
  let firstMismatch = -1;
  for (let i = 0; i < n; i++) {
    if (gk[i] !== keys[idx[i]] || gv[i] !== idx[i]) {
      firstMismatch = i;
      break;
    }
  }
  sorter.destroy();
  for (const b of [keysA, keysB, valsA, valsB, ctrl, outK, outV]) b.destroy();
  return { correct: firstMismatch < 0, ms: best, firstMismatch };
}
