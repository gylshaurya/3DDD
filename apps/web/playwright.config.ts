import { defineConfig } from '@playwright/test';

// WebGPU in headless Chromium on macOS needs ANGLE on Metal to present canvases.
export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:5199',
    viewport: { width: 960, height: 600 },
    launchOptions: { args: ['--enable-unsafe-webgpu', '--use-angle=metal'] },
  },
  webServer: {
    command: 'pnpm exec vite --port 5199 --strictPort',
    url: 'http://localhost:5199/viewer-test.html',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
