// End-to-end tests of the WebGPU splat viewer in real (headless) Chromium on the local GPU.
// Requires the sample scenes: `pnpm samples`.
import { createReadStream, existsSync, mkdirSync, statSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { join } from 'node:path';
import { expect, test, type Page } from '@playwright/test';

const SAMPLES = join(import.meta.dirname, '..', 'public', 'samples');
const SHOT_DIR = process.env.SHOT_DIR ?? join(import.meta.dirname, '..', 'test-results', 'shots');
mkdirSync(SHOT_DIR, { recursive: true });

const hasSample = (f: string) => existsSync(join(SAMPLES, f));

async function openViewer(page: Page, query: string): Promise<void> {
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(`/viewer-test.html?hud=0&${query}`);
  await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error || (window.__splat.ready && !new URLSearchParams(location.search).get('url'))), null, { timeout: 150_000 });
  const err = await page.evaluate(() => window.__splat.error);
  expect(err).toBeNull();
}

interface ImageStats { coverage: number; stdDev: number; mean: [number, number, number] }

/** Renders a frame and measures how much of it differs from the background, plus contrast. */
async function frameStats(page: Page): Promise<ImageStats> {
  return page.evaluate(async () => {
    const src = window.__splat.viewer!.captureFrame();
    const img = new Image();
    await new Promise((r) => { img.onload = r; img.src = src; });
    const c = new OffscreenCanvas(img.width, img.height);
    const g = c.getContext('2d')!;
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, img.width, img.height).data;
    const bg = [0.035 * 255, 0.037 * 255, 0.045 * 255];
    let covered = 0, sum = 0, sum2 = 0;
    const mean = [0, 0, 0];
    const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) {
      const diff = Math.abs(d[i] - bg[0]) + Math.abs(d[i + 1] - bg[1]) + Math.abs(d[i + 2] - bg[2]);
      if (diff > 24) covered++;
      const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
      sum += l; sum2 += l * l;
      mean[0] += d[i]; mean[1] += d[i + 1]; mean[2] += d[i + 2];
    }
    const m = sum / n;
    return {
      coverage: covered / n,
      stdDev: Math.sqrt(Math.max(0, sum2 / n - m * m)),
      mean: mean.map((v) => v / n) as [number, number, number],
    };
  });
}

test.describe('GPU radix sort', () => {
  test('matches a stable CPU sort for many sizes', async ({ page }) => {
    await openViewer(page, '');
    for (const n of [1, 1023, 1024, 1025, 65_537, 1_000_000, 2_000_000]) {
      const r = await page.evaluate((n) => window.__splat.testGpuSort(n, 3), n);
      console.log(`sort n=${n.toLocaleString()}: correct=${r.correct} best=${r.ms.toFixed(2)} ms (submit→done)`);
      expect(r.correct, `mismatch at ${r.firstMismatch} for n=${n}`).toBe(true);
    }
  });
});

test.describe('rendering', () => {
  const scenes = [
    { file: 'train.splat', count: 1_026_508 },
    { file: 'plush.splat', count: 281_498 },
    { file: 'nike.splat', count: 270_491 },
    { file: 'bonsai-7k.ply', count: 1_157_141 },
  ];
  for (const s of scenes) {
    test(`renders ${s.file}`, async ({ page }) => {
      test.skip(!hasSample(s.file), 'run `pnpm samples` first');
      await openViewer(page, `url=/samples/${s.file}`);
      await page.waitForTimeout(800);
      const stats = await page.evaluate(() => window.__splat.viewer!.getStats());
      expect(stats.splatCount).toBe(s.count);
      expect(stats.gpuSort).toBe(true);
      expect(stats.visibleCount).toBeGreaterThan(s.count * 0.2);
      const img = await frameStats(page);
      console.log(`${s.file}: coverage=${img.coverage.toFixed(2)} contrast=${img.stdDev.toFixed(1)} visible=${stats.visibleCount}`);
      expect(img.coverage).toBeGreaterThan(0.25);
      expect(img.stdDev).toBeGreaterThan(20);
      await page.screenshot({ path: join(SHOT_DIR, `e2e-${s.file}.png`) });
    });
  }

  test('PLY decoding is pixel-identical to .splat for the same scene', async ({ page }) => {
    test.skip(!hasSample('train.ply'), 'run `pnpm samples` first');
    const grab = async (url: string) => {
      await openViewer(page, `url=${url}`);
      return page.evaluate(() => {
        const v = window.__splat.viewer!;
        v.lookAt([-3.0, -0.11, -3.75], [0, 0, 0], true);
        return v.captureFrame();
      });
    };
    const a = await grab('/samples/train.splat');
    const b = await grab('/samples/train.ply');
    const diff = await page.evaluate(async ([a, b]) => {
      const load = async (src: string) => {
        const i = new Image();
        await new Promise((r) => { i.onload = r; i.src = src; });
        const c = new OffscreenCanvas(i.width, i.height);
        const g = c.getContext('2d')!;
        g.drawImage(i, 0, 0);
        return g.getImageData(0, 0, i.width, i.height).data;
      };
      const [da, db] = await Promise.all([load(a), load(b)]);
      let s = 0;
      for (let i = 0; i < da.length; i += 4) s += Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2]);
      return s / (da.length / 4) / 3;
    }, [a, b]);
    console.log(`train.splat vs train.ply mean abs diff: ${diff.toFixed(4)}`);
    expect(diff).toBeLessThan(0.5);
  });

  test('spherical harmonics give view-dependent color', async ({ page }) => {
    test.skip(!hasSample('sh-test.ply'), 'run `pnpm samples` first');
    await openViewer(page, 'url=/samples/sh-test.ply');
    const center = async (eye: number[]) => page.evaluate(async (eye) => {
      const v = window.__splat.viewer!;
      v.lookAt(eye as [number, number, number], [0, 0, 0], true);
      const i = new Image();
      await new Promise((r) => { i.onload = r; i.src = v.captureFrame(); });
      const c = new OffscreenCanvas(i.width, i.height);
      const g = c.getContext('2d')!;
      g.drawImage(i, 0, 0);
      const d = g.getImageData(i.width / 2, i.height / 2, 1, 1).data;
      return [d[0], d[1], d[2]];
    }, eye);
    const fromPlusX = await center([2, 0, -2]);
    const fromMinusX = await center([-2, 0, -2]);
    const front = await center([0, 0, -2.8]);
    console.log('SH colors', { fromPlusX, fromMinusX, front });
    // coefficient (+0.8, 0, −0.8) on the −C1·x basis: ±0.8·C1·sin(45°)·255 ≈ ±70
    expect(fromPlusX[0] - fromPlusX[2]).toBeGreaterThan(100);
    expect(fromMinusX[2] - fromMinusX[0]).toBeGreaterThan(100);
    expect(Math.abs(front[0] - front[2])).toBeLessThan(6);
  });
});

test.describe('loading', () => {
  let server: Server;
  let port = 0;
  test.beforeAll(async () => {
    // Serves a file slowly in chunks so progressive rendering can be observed.
    server = createServer((req, res) => {
      const file = join(SAMPLES, (req.url ?? '').replace(/^\//, '').split('?')[0]);
      if (!existsSync(file)) { res.writeHead(404).end(); return; }
      res.writeHead(200, {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(statSync(file).size),
      });
      const stream = createReadStream(file, { highWaterMark: 1 << 20 });
      stream.on('data', (chunk) => {
        stream.pause();
        res.write(chunk, () => setTimeout(() => stream.resume(), 60));
      });
      stream.on('end', () => res.end());
    });
    await new Promise<void>((r) => server.listen(0, () => r()));
    port = (server.address() as { port: number }).port;
  });
  test.afterAll(() => server.close());

  test('streams progressively: renders partial scenes while downloading', async ({ page }) => {
    test.skip(!hasSample('train.splat'), 'run `pnpm samples` first');
    await page.goto(`/viewer-test.html?hud=0&url=${encodeURIComponent(`http://localhost:${port}/train.splat`)}`);
    const seen = new Set<number>();
    let renderedPartial = false;
    for (let i = 0; i < 200; i++) {
      const s = await page.evaluate(() => ({ loaded: window.__splat.loaded, st: window.__splat.viewer?.getStats() ?? null }));
      if (s.st) {
        if (s.st.loadProgress < 1 && s.st.splatCount > 0) seen.add(s.st.splatCount);
        if (s.st.loadProgress < 1 && s.st.visibleCount > 0) renderedPartial = true;
      }
      if (s.loaded) break;
      await page.waitForTimeout(100);
    }
    console.log(`partial splat counts observed: ${[...seen].sort((a, b) => a - b).join(', ')}`);
    expect(seen.size).toBeGreaterThanOrEqual(3);
    expect(renderedPartial).toBe(true);
    const final = await page.evaluate(() => window.__splat.viewer!.getStats());
    expect(final.splatCount).toBe(1_026_508);
  });

  test('live snapshot swap keeps the camera and never flashes empty', async ({ page }) => {
    test.skip(!hasSample('plush.splat'), 'run `pnpm samples` first');
    await openViewer(page, 'url=/samples/plush.splat');
    await page.waitForTimeout(1500); // let the end-of-load refit finish easing
    const before = await frameStats(page);
    const eyeBefore = await page.evaluate(() => window.__splat.viewer!.camera.eye);
    // swap to a slightly different scene (the same plush, but every splat recolored red)
    await page.evaluate(async () => {
      const buf = await (await fetch('/samples/plush.splat')).arrayBuffer();
      const u8 = new Uint8Array(buf);
      for (let i = 0; i < u8.length; i += 32) { u8[i + 24] = 230; u8[i + 25] = 30; u8[i + 26] = 30; }
      (window as unknown as { __swap: Promise<void> }).__swap = window.__splat.viewer!.setSplatsFromBuffer(buf, 'splat');
    });
    const coverages: number[] = [];
    for (let i = 0; i < 25; i++) {
      coverages.push((await frameStats(page)).coverage);
      await page.waitForTimeout(40);
    }
    await page.evaluate(() => (window as unknown as { __swap: Promise<void> }).__swap);
    await page.waitForTimeout(600);
    const after = await frameStats(page);
    const eyeAfter = await page.evaluate(() => window.__splat.viewer!.camera.eye);
    console.log(`coverage before=${before.coverage.toFixed(2)} min during swap=${Math.min(...coverages).toFixed(2)} after=${after.coverage.toFixed(2)}; mean color after=${after.mean.map((v) => v.toFixed(0))}`);
    expect(Math.min(...coverages)).toBeGreaterThan(before.coverage * 0.8);
    expect(after.mean[0]).toBeGreaterThan(after.mean[1] * 1.5); // new (red) scene is showing
    for (let k = 0; k < 3; k++) expect(eyeAfter[k]).toBeCloseTo(eyeBefore[k], 5);
    await page.screenshot({ path: join(SHOT_DIR, 'e2e-swap-after.png') });
  });

  test('loads a dropped File / Blob', async ({ page }) => {
    test.skip(!hasSample('nike.splat'), 'run `pnpm samples` first');
    await openViewer(page, '');
    const count = await page.evaluate(async () => {
      const blob = await (await fetch('/samples/nike.splat')).blob();
      const file = new File([blob], 'nike.splat');
      await window.__splat.viewer!.load(file);
      return window.__splat.viewer!.splatCount;
    });
    expect(count).toBe(270_491);
  });

  test('reports a readable error for a corrupt file', async ({ page }) => {
    await openViewer(page, '');
    const msg = await page.evaluate(async () => {
      const bad = new TextEncoder().encode('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n');
      try { await window.__splat.viewer!.load(bad.buffer as ArrayBuffer, { name: 'bad.ply' }); return 'no error'; }
      catch (e) { return (e as Error).message; }
    });
    expect(msg).toContain('binary_little_endian');
  });
});

test.describe('fallbacks', () => {
  test('CPU sort fallback renders the same scene', async ({ page }) => {
    test.skip(!hasSample('plush.splat'), 'run `pnpm samples` first');
    await openViewer(page, 'url=/samples/plush.splat&cpu=1');
    await page.waitForTimeout(1200); // CPU sort is asynchronous
    const stats = await page.evaluate(() => window.__splat.viewer!.getStats());
    expect(stats.gpuSort).toBe(false);
    const img = await frameStats(page);
    expect(img.coverage).toBeGreaterThan(0.25);
    await page.screenshot({ path: join(SHOT_DIR, 'e2e-cpu-fallback.png') });
  });

  test('shows a clear message without WebGPU', async ({ page }) => {
    await page.addInitScript(() => Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined }));
    await page.goto('/viewer-test.html?url=/samples/plush.splat');
    await expect(page.getByRole('alert')).toContainText('WebGPU is not available');
    await page.screenshot({ path: join(SHOT_DIR, 'e2e-no-webgpu.png') });
  });
});

test.describe('performance', () => {
  for (const file of ['train.splat', 'bonsai-7k.ply']) {
    test(`orbiting ${file} at 1280×800`, async ({ page }) => {
      test.skip(!hasSample(file), 'run `pnpm samples` first');
      await page.setViewportSize({ width: 1280, height: 800 });
      await openViewer(page, `url=/samples/${file}&orbit=1`);
      await page.waitForTimeout(2500);
      const samples = [];
      for (let i = 0; i < 6; i++) {
        await page.waitForTimeout(500);
        samples.push(await page.evaluate(() => window.__splat.viewer!.getStats()));
      }
      const avg = (k: 'fps' | 'sortMs' | 'gpuFrameMs') => samples.reduce((s, x) => s + x[k], 0) / samples.length;
      const scale = await page.evaluate(() => window.__splat.viewer!.renderScale);
      console.log(`${file}: ${avg('fps').toFixed(1)} fps, sort ${avg('sortMs').toFixed(2)} ms, cull+sort ${avg('gpuFrameMs').toFixed(2)} ms, ${samples[0].splatCount.toLocaleString()} splats, renderScale ${scale.toFixed(2)}`);
      expect(avg('fps')).toBeGreaterThan(45);
    });
  }
});

test.describe('app shell', () => {
  test('React app renders samples and switches scenes', async ({ page }) => {
    test.skip(!hasSample('train.splat') || !hasSample('bonsai-7k.ply'), 'run `pnpm samples` first');
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    await page.goto('/');
    await expect(page.locator('.hud')).toContainText('1,026,508 splats', { timeout: 60_000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: join(SHOT_DIR, 'e2e-app-train.png') });
    await page.getByRole('button', { name: 'Bonsai (SH3)' }).click();
    await expect(page.locator('.hud')).toContainText('1,157,141 splats', { timeout: 60_000 });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(SHOT_DIR, 'e2e-app-bonsai.png') });
    // orbit by dragging
    const box = (await page.locator('canvas').boundingBox())!;
    await page.mouse.move(box.width / 2, box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.width / 2 + 200, box.height / 2 + 30, { steps: 10 });
    await page.mouse.up();
    await page.waitForTimeout(800);
    await page.screenshot({ path: join(SHOT_DIR, 'e2e-app-bonsai-orbited.png') });
    expect(errors.filter((e) => !e.includes('[vite]'))).toEqual([]);
  });
});
