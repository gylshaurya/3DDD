import { chromium } from '@playwright/test';
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const page = await browser.newPage();
page.on('console', (m) => { if (!m.text().includes('vite')) console.log(`[console.${m.type()}]`, m.text()); });
await page.goto(`http://localhost:5199/viewer-test.html`);
await page.waitForFunction(() => window.__splat && (window.__splat.ready || window.__splat.error), null, { timeout: 60000 });
for (const n of [1, 1000, 1025, 100000, 1000000, 2000000]) {
  const r = await page.evaluate((n) => window.__splat.testGpuSort(n, 3), n);
  console.log(n, JSON.stringify(r));
}
await browser.close();
