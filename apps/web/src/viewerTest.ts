// Dev / e2e harness: /viewer-test.html?url=/samples/train.splat&orbit=1&up=y-down&cpu=1&hud=0
import { createViewer, showWebGPUError, testGpuSort, type UpAxis, type Viewer, type ViewerStats } from './viewer';

declare global {
  interface Window {
    __splat: {
      viewer: Viewer | null;
      ready: boolean;
      loaded: boolean;
      error: string | null;
      stats: ViewerStats | null;
      testGpuSort: (n: number, runs?: number) => Promise<{ correct: boolean; ms: number; firstMismatch: number }>;
    };
  }
}

const params = new URLSearchParams(location.search);
const canvas = document.getElementById('c') as HTMLCanvasElement;
const hud = document.getElementById('hud')!;
if (params.get('hud') === '0') hud.style.display = 'none';

window.__splat = {
  viewer: null,
  ready: false,
  loaded: false,
  error: null,
  stats: null,
  testGpuSort: async (n, runs) => {
    const v = window.__splat.viewer;
    if (!v) throw new Error('viewer not ready');
    return testGpuSort(v.device, n, runs);
  },
};

async function main(): Promise<void> {
  let viewer: Viewer;
  try {
    viewer = await createViewer(canvas, {
      up: (params.get('up') as UpAxis) ?? 'auto',
      autoOrbit: params.get('orbit') === '1',
      forceCpuSort: params.get('cpu') === '1',
      onError: (e) => { window.__splat.error = e.message; },
    });
  } catch (err) {
    window.__splat.error = err instanceof Error ? err.message : String(err);
    hud.style.display = 'none';
    showWebGPUError(document.getElementById('wrap')!, err);
    return;
  }
  window.__splat.viewer = viewer;
  window.__splat.ready = true;
  viewer.onStats((s) => {
    window.__splat.stats = s;
    hud.textContent =
      `${s.fps.toFixed(0)} fps · ${s.splatCount.toLocaleString()} splats · ${s.visibleCount.toLocaleString()} visible\n` +
      `sort ${s.sortMs >= 0 ? s.sortMs.toFixed(2) + ' ms' : 'n/a'} (${s.gpuSort ? 'GPU radix' : 'CPU fallback'})` +
      (s.loadProgress < 1 ? ` · loading ${(s.loadProgress * 100).toFixed(0)}%` : '');
  });
  const url = params.get('url');
  if (url) {
    try {
      await viewer.load(url);
      window.__splat.loaded = true;
    } catch (err) {
      window.__splat.error = err instanceof Error ? err.message : String(err);
    }
  }
}

void main();
