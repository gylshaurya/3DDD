// Renders two URLs from the same explicit camera and reports mean abs pixel difference.
// Usage: node scripts/compare.mjs <urlA> <urlB> <eye-json> <target-json> <outPrefix>
import { chromium } from '@playwright/test';
const [a, b, eye, target, outPrefix] = process.argv.slice(2);
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
async function render(url, out) {
  const page = await browser.newPage({ viewport: { width: 640, height: 400 } });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(`http://localhost:5199/viewer-test.html?hud=0&url=${encodeURIComponent(url)}`);
  await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error), null, { timeout: 120000 });
  const err = await page.evaluate(() => window.__splat.error);
  if (err) throw new Error(err);
  await page.evaluate(([e, t]) => window.__splat.viewer.lookAt(JSON.parse(e), JSON.parse(t), true), [eye, target]);
  await page.waitForTimeout(600);
  const px = await page.evaluate(() => {
    const url = window.__splat.viewer.captureFrame();
    return url;
  });
  await page.screenshot({ path: out });
  await page.close();
  return px;
}
const pa = await render(a, `${outPrefix}-A.png`);
const pb = await render(b, `${outPrefix}-B.png`);
// decode both data URLs in a page and diff
const page = await browser.newPage();
const diff = await page.evaluate(async ([pa, pb]) => {
  const load = (src) => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = src; });
  const [ia, ib] = await Promise.all([load(pa), load(pb)]);
  const c = new OffscreenCanvas(ia.width, ia.height); const g = c.getContext('2d');
  g.drawImage(ia, 0, 0); const da = g.getImageData(0, 0, ia.width, ia.height).data;
  g.drawImage(ib, 0, 0); const db = g.getImageData(0, 0, ia.width, ia.height).data;
  let sum = 0, big = 0;
  for (let i = 0; i < da.length; i += 4) {
    const d = (Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2])) / 3;
    sum += d; if (d > 16) big++;
  }
  const n = da.length / 4;
  return { meanAbs: sum / n, fracOver16: big / n, w: ia.width, h: ia.height };
}, [pa, pb]);
console.log(JSON.stringify(diff));
await browser.close();
