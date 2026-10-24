from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("THREEDDD_DATA", REPO_ROOT / "data"))
JOBS_DIR = DATA_DIR / "jobs"
SCENES_DIR = DATA_DIR / "scenes"
TOOLS_DIR = REPO_ROOT / "tools"

# Modal spend guard. The user's free credit is $30; testing must stay well below it.
MODAL_BUDGET_USD = float(os.environ.get("THREEDDD_MODAL_BUDGET", "15"))
MODAL_LEDGER = DATA_DIR / "modal_ledger.json"

for d in (JOBS_DIR, SCENES_DIR):
    d.mkdir(parents=True, exist_ok=True)
