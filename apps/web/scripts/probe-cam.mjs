import { chromium } from '@playwright/test';
const [q] = process.argv.slice(2);
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto(`http://localhost:5199/viewer-test.html?${q}`);
await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error), null, { timeout: 60000 });
console.log(await page.evaluate(() => { const v = window.__splat.viewer; return JSON.stringify({ bounds: v.bounds, eye: v.camera.eye, target: v.camera.target }); }));
await browser.close();
