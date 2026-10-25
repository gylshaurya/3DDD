// Drives the real UI: home → "Run sample" → live job page, screenshotting along the way.
// Usage: node scripts/e2e-job.mjs <outDir> <backend modal|local> <preset> [gpu]
import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
const [outDir, backend = 'local', preset = 'preview', gpu = 'L4'] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
await page.goto(process.env.BASE_URL ?? `http://localhost:${process.env.PORT ?? 8000}/`);
await page.getByRole('button', { name: backend === 'modal' ? 'Cloud GPU' : 'This computer' }).click();
await page.getByRole('button', { name: preset[0].toUpperCase() + preset.slice(1), exact: true }).click();
if (backend === 'modal') await page.getByRole('button', { name: gpu, exact: true }).click();
await page.screenshot({ path: `${outDir}/00-home.png` });
await page.getByRole('button', { name: `Run sample ${process.env.SAMPLE ?? 'counter'}` }).click();
await page.waitForURL(/#\/job\//, { timeout: 60000 });
console.log('job url', page.url());
const t0 = Date.now();
let i = 1, last = '';
while (Date.now() - t0 < 25 * 60 * 1000) {
  await page.waitForTimeout(i < 4 ? 8000 : 20000);
  const caption = (await page.locator('.caption-text').textContent()) ?? '';
  const status = (await page.locator('.topbar .pill').first().textContent().catch(() => '')) ?? '';
  const name = `${String(i).padStart(2, '0')}-${Math.round((Date.now() - t0) / 1000)}s.png`;
  if (caption !== last || i % 3 === 0) await page.screenshot({ path: `${outDir}/${name}` });
  console.log(`${Math.round((Date.now() - t0) / 1000)}s [${status}] ${caption}`);
  last = caption;
  i++;
  if (/done|failed|cancelled/.test(status)) {
    await page.waitForTimeout(8000);
    await page.screenshot({ path: `${outDir}/99-final.png` });
    break;
  }
}
console.log('errors:', JSON.stringify(errors));
await browser.close();
