from pathlib import Path
import os

_here = Path(__file__).resolve()
# Repo root, or the package folder when running inside a cloud container.
REPO_ROOT = Path(os.environ.get("THREEDDD_REPO_ROOT", _here.parents[3] if len(_here.parents) > 3 else _here.parent))
DATA_DIR = Path(os.environ.get("THREEDDD_DATA", REPO_ROOT / "data"))
JOBS_DIR = DATA_DIR / "jobs"
SCENES_DIR = DATA_DIR / "scenes"
TOOLS_DIR = REPO_ROOT / "tools"

# Modal spending cap in USD.
MODAL_BUDGET_USD = float(os.environ.get("THREEDDD_MODAL_BUDGET", "15"))
MODAL_LEDGER = DATA_DIR / "modal_ledger.json"

# Hosted mode: anyone can upload, so runs are limited.
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
# Run the camera solve on a cloud CPU instead of this machine.
REMOTE_SFM = os.environ.get("THREEDDD_REMOTE_SFM") == "1"
# Allowed browser origins for the API ("*" means any).
CORS_ORIGINS = [o.strip() for o in os.environ.get("THREEDDD_CORS_ORIGINS", "*").split(",") if o.strip()]

for d in (JOBS_DIR, SCENES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Called after a job is created, finished or deleted. The cloud version uses it to save its volume.
_persist_hooks: list = []


def on_persist(fn) -> None:
    _persist_hooks.append(fn)


def persist() -> None:
    for fn in _persist_hooks:
        try:
            fn()
        except Exception:
            pass
