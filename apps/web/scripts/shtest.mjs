// Renders sh-test.ply from +x and −x oblique views and reports the center pixel color.
import { chromium } from '@playwright/test';
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const page = await browser.newPage({ viewport: { width: 400, height: 400 } });
await page.goto(`http://localhost:5199/viewer-test.html?hud=0&url=/samples/sh-test.ply`);
await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error), null, { timeout: 60000 });
const out = {};
for (const [name, eye] of [['plusX', [2, 0, -2]], ['minusX', [-2, 0, -2]], ['front', [0, 0, -2.8]]]) {
  const url = await page.evaluate((eye) => { const v = window.__splat.viewer; v.lookAt(eye, [0, 0, 0], true); return v.captureFrame(); }, eye);
  out[name] = await page.evaluate(async (src) => {
    const i = new Image(); await new Promise((r) => { i.onload = r; i.src = src; });
    const c = new OffscreenCanvas(i.width, i.height); const g = c.getContext('2d'); g.drawImage(i, 0, 0);
    const d = g.getImageData(i.width / 2 - 2, i.height / 2 - 2, 4, 4).data;
    let r = 0, gg = 0, b = 0; for (let k = 0; k < d.length; k += 4) { r += d[k]; gg += d[k + 1]; b += d[k + 2]; }
    return [r / 16, gg / 16, b / 16].map(Math.round);
  }, url);
  await page.screenshot({ path: `${process.argv[2]}-${name}.png` });
}
console.log(JSON.stringify(out));
await browser.close();
