// Usage: node scripts/shoot-app.mjs <path-with-hash> <out.png> [waitMs] [w] [h]
import { chromium } from '@playwright/test';
const [path, out, waitMs = '4000', w = '1440', h = '900'] = process.argv.slice(2);
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
page.on('console', (m) => m.type() !== 'debug' && console.log(`[console.${m.type()}]`, m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${process.env.PORT ?? 5280}${path}`);
await page.waitForTimeout(+waitMs);
await page.screenshot({ path: out });
await browser.close();
