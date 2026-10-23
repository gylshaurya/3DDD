// Usage: node scripts/shoot.mjs <query-string> <out.png> [waitMs] [w] [h]
import { chromium } from '@playwright/test';
const [q, out, waitMs = '2500', w = '1280', h = '800'] = process.argv.slice(2);
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
page.on('console', (m) => console.log(`[console.${m.type()}]`, m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${process.env.PORT ?? 5280}/viewer-test.html?${q}`);
await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error), null, { timeout: 60000 });
await page.waitForTimeout(+waitMs);
const info = await page.evaluate(() => ({ error: window.__splat.error, stats: window.__splat.stats }));
console.log(JSON.stringify(info));
await page.screenshot({ path: out });
await browser.close();
