// Measures steady-state stats with auto-orbit. Usage: node scripts/perf.mjs <url> [w] [h] [evalJs]
import { chromium } from '@playwright/test';
const [url, w = '1280', h = '800', js = ''] = process.argv.slice(2);
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
await page.goto(`http://localhost:${process.env.PORT ?? 5280}/viewer-test.html?orbit=1&url=${encodeURIComponent(url)}`);
await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error), null, { timeout: 120000 });
if (js) await page.evaluate(js);
await page.waitForTimeout(1500);
const samples = [];
await page.waitForTimeout(+(process.env.WARM ?? 0));
for (let i = 0; i < 6; i++) { await page.waitForTimeout(500); samples.push(await page.evaluate(() => window.__splat.stats)); }
const avg = (k) => samples.reduce((s, x) => s + x[k], 0) / samples.length;
const rs = await page.evaluate(() => window.__splat.viewer.renderScale);
console.log(JSON.stringify({ url, renderScale: +rs.toFixed(2), fps: +avg('fps').toFixed(1), sortMs: +avg('sortMs').toFixed(2), gpuComputeMs: +avg('gpuFrameMs').toFixed(2), visible: Math.round(avg('visibleCount')), splats: samples[0].splatCount }));
await browser.close();
