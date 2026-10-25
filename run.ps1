# 3DDD: build the web app and serve everything from one port (default 8000). Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File .\run.ps1          http://localhost:8000
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 -Dev     API on :8000 + Vite on :5280
param([switch]$Dev, [int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"   # UTF-8 everywhere regardless of the console code page

function Need($cmd, $hint) {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { Write-Host "$cmd is required. $hint" -ForegroundColor Red; exit 1 }
}
Need uv     'Install: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
Need pnpm   "Install Node.js 22+ from https://nodejs.org, then: npm install -g pnpm"
Need ffmpeg "Install: winget install --id Gyan.FFmpeg -e   (then open a new terminal)"

Push-Location server; uv sync -q; if ($LASTEXITCODE) { exit 1 }; Pop-Location
Push-Location apps/web; pnpm install --frozen-lockfile --silent; if ($LASTEXITCODE) { exit 1 }; Pop-Location
# Checks prerequisites and downloads the offline trainer (Brush) for Windows if it's missing.
Push-Location server; uv run python -m threeddd.setup_tools; if ($LASTEXITCODE) { Write-Host "(setup check reported a problem; see above)" }; Pop-Location

if ($Dev) {
  $api = Start-Process -PassThru -NoNewWindow uv -ArgumentList "run","uvicorn","threeddd.api:app","--host","0.0.0.0","--port","$Port","--reload" -WorkingDirectory server
  try { Push-Location apps/web; pnpm exec vite } finally { Pop-Location; Stop-Process -Id $api.Id -ErrorAction SilentlyContinue }
} else {
  Push-Location apps/web; pnpm build | Out-Null; if ($LASTEXITCODE) { exit 1 }; Pop-Location
  Write-Host "3DDD running at http://localhost:$Port"
  Push-Location server; uv run uvicorn threeddd.api:app --host 0.0.0.0 --port $Port
}
