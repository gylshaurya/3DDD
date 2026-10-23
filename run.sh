#!/usr/bin/env bash
# Splatcast — build the web app and serve everything from one port (default 8000).
#   ./run.sh            production-style: http://localhost:8000  (phones on the same Wi-Fi: http://<lan-ip>:8000/capture)
#   ./run.sh --dev      hot-reload: API on :8000 + Vite on :5280
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required: brew install ffmpeg"; exit 1; }
command -v uv >/dev/null || { echo "uv is required: brew install uv"; exit 1; }
command -v pnpm >/dev/null || { echo "pnpm is required: npm i -g pnpm"; exit 1; }

(cd server && uv sync -q)
(cd apps/web && pnpm install --frozen-lockfile --silent)

if [[ "${1:-}" == "--dev" ]]; then
  (cd server && uv run uvicorn splatcast.api:app --host 0.0.0.0 --port 8000 --reload) &
  trap 'kill 0' EXIT
  cd apps/web && pnpm exec vite
else
  (cd apps/web && pnpm build >/dev/null)
  echo "Splatcast running at http://localhost:${PORT:-8000}"
  cd server && exec uv run uvicorn splatcast.api:app --host 0.0.0.0 --port "${PORT:-8000}"
fi
