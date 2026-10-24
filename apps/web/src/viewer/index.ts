// 3DDD viewer — a from-scratch WebGPU 3D Gaussian Splatting renderer.
//
//   const viewer = await createViewer(canvas, { up: 'y-down', autoOrbit: true });
//   await viewer.load('/samples/train.splat');
//   viewer.onStats((s) => console.log(s.fps, s.sortMs));
//
// Live training preview: `viewer.setSplatsFromBuffer(snapshotBuffer, 'splat')` swaps the
// scene with a crossfade while keeping the camera.

import { Viewer, type ViewerOptions } from './viewer';

export { Viewer, WebGPUUnavailableError, testGpuSort } from './viewer';
export type { LoadOptions, ViewerOptions, ViewerStats } from './viewer';
export type { UpAxis } from './camera';
export type { SplatFormat } from './formats';
export type { SceneBounds } from './bounds';
export { showWebGPUError } from './errorOverlay';

export function createViewer(canvas: HTMLCanvasElement, opts: ViewerOptions = {}): Promise<Viewer> {
  return Viewer.create(canvas, opts);
}
