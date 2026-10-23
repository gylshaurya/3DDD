import { chromium } from '@playwright/test';
const [q, outPrefix, ...radii] = process.argv.slice(2);
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const page = await browser.newPage({ viewport: { width: 960, height: 600 } });
await page.goto(`http://localhost:5199/viewer-test.html?hud=0&${q}`);
await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error), null, { timeout: 60000 });
await page.waitForTimeout(800);
for (const r of radii) {
  await page.evaluate((r) => { const v = window.__splat.viewer; v.camera.fit(v.bounds.center, +r, true); v.dirty = true; }, r);
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${outPrefix}-${r}.png` });
}
await browser.close();
