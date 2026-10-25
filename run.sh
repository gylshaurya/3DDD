#!/usr/bin/env bash
# 3DDD — build the web app and serve everything from one port (default 8000). macOS and Linux.
#   ./run.sh            production-style: http://localhost:8000  (phones on the same Wi-Fi: http://<lan-ip>:8000/capture)
#   ./run.sh --dev      hot-reload: API on :8000 + Vite on :5280
# Windows: use run.ps1
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

need() { command -v "$1" >/dev/null || { echo "$1 is required. $2"; exit 1; }; }
need uv "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
need pnpm "Install: npm install -g pnpm"
need ffmpeg "Install: brew install ffmpeg (macOS) or sudo apt install ffmpeg (Linux)"

(cd server && uv sync -q)
(cd apps/web && pnpm install --frozen-lockfile --silent)
# Checks prerequisites and downloads the offline trainer (Brush) for this OS if it's missing.
(cd server && uv run python -m threeddd.setup_tools) || echo "(setup check reported a problem; see above)"

if [[ "${1:-}" == "--dev" ]]; then
  (cd server && uv run uvicorn threeddd.api:app --host 0.0.0.0 --port 8000 --reload) &
  trap 'kill 0' EXIT
  cd apps/web && pnpm exec vite
else
  (cd apps/web && pnpm build >/dev/null)
  echo "3DDD running at http://localhost:${PORT:-8000}"
  cd server && exec uv run uvicorn threeddd.api:app --host 0.0.0.0 --port "${PORT:-8000}"
fi
