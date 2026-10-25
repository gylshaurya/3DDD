from pathlib import Path
import os

_here = Path(__file__).resolve()
# server/src/threeddd/config.py → repo root; inside a cloud container the package sits at /root/threeddd.
REPO_ROOT = Path(os.environ.get("THREEDDD_REPO_ROOT", _here.parents[3] if len(_here.parents) > 3 else _here.parent))
DATA_DIR = Path(os.environ.get("THREEDDD_DATA", REPO_ROOT / "data"))
JOBS_DIR = DATA_DIR / "jobs"
SCENES_DIR = DATA_DIR / "scenes"
TOOLS_DIR = REPO_ROOT / "tools"

# Modal spend guard. The user's free credit is $30; testing must stay well below it.
MODAL_BUDGET_USD = float(os.environ.get("THREEDDD_MODAL_BUDGET", "15"))
MODAL_LEDGER = DATA_DIR / "modal_ledger.json"

# Public deployment: anyone on the internet can upload, so runs are limited and owner-scoped.
PUBLIC = os.environ.get("THREEDDD_PUBLIC") == "1"
PUBLIC_LIMITS = {
    "max_upload_mb": 300,
    "max_video_seconds": 120,
    "presets": ["preview", "fast"],
    "gpus": ["L4", "L40S"],
    "jobs_per_ip_per_hour": 3,
    "jobs_per_day": 25,
    "max_queued": 3,
}
# Camera solve on a cloud CPU function instead of this machine (used by the hosted deployment).
REMOTE_SFM = os.environ.get("THREEDDD_REMOTE_SFM") == "1"
# Comma-separated origins allowed to call the API from a browser ("*" = any).
CORS_ORIGINS = [o.strip() for o in os.environ.get("THREEDDD_CORS_ORIGINS", "*").split(",") if o.strip()]

for d in (JOBS_DIR, SCENES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Called after durable state changes (job finished, deleted, ...). The cloud deployment sets this to
# commit its storage volume; locally there's nothing to do.
_persist_hooks: list = []


def on_persist(fn) -> None:
    _persist_hooks.append(fn)


def persist() -> None:
    for fn in _persist_hooks:
        try:
            fn()
        except Exception:  # noqa: BLE001 - persistence is best effort; the next commit catches up
            pass
