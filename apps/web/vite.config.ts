import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const api = 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5280,
    strictPort: true,
    proxy: { '/api': api, '/files': api },
  },
  build: {
    // public/samples holds ~400 MB of dev scenes; the server serves them separately.
    copyPublicDir: false,
    rollupOptions: {
      input: { main: 'index.html', viewerTest: 'viewer-test.html' },
    },
  },
})
