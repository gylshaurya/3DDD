import { chromium } from '@playwright/test';
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
for (const url of process.argv.slice(2)) {
  const page = await browser.newPage({ viewport: { width: 320, height: 200 } });
  await page.goto(`http://localhost:5199/viewer-test.html?hud=0&url=${url}`);
  await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error), null, { timeout: 120000 });
  const r = await page.evaluate(() => {
    const v = window.__splat.viewer; const s = v.active.allSamples(); const c = v.bounds.center;
    const d = []; for (let i = 0; i < s.length; i += 4) if (s[i + 3] > 0.25) d.push(Math.hypot(s[i] - c[0], s[i + 1] - c[1], s[i + 2] - c[2]));
    d.sort((a, b) => a - b); const p = (q) => d[Math.round(q * (d.length - 1))];
    return [0.05, 0.1, 0.25, 0.5, 0.8, 0.95].map((q) => +(p(q) / p(0.5)).toFixed(2)).join(' ') + '  r50=' + p(0.5).toFixed(2);
  });
  console.log(url.padEnd(26), r);
  await page.close();
}
await browser.close();
