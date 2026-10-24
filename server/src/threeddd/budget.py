"""Hard spending guard for Modal.

Every cloud run must pass `check_budget()` first. Spend is the larger of
  * Modal's own metered cost for this month (authoritative, but can lag), and
  * our local ledger of every run we launched (instant, conservative estimate).
A run is refused if spend + the run's worst-case cost would exceed the cap.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import MODAL_BUDGET_USD, MODAL_LEDGER

# $/hour from `modal billing rates` (Oct 2026). CPU+memory overhead added per run.
GPU_RATES = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}
OVERHEAD_PER_HOUR = 4 * 0.0473 + 32 * 0.008  # 4 cores + 32 GiB memory
SAFETY = 1.25

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def _modal_cli() -> str:
    return shutil.which("modal") or str(Path.home() / ".local/bin/modal")


def run_cost(gpu: str, seconds: float) -> float:
    return (GPU_RATES[gpu] + OVERHEAD_PER_HOUR) * seconds / 3600 * SAFETY


def metered_this_month() -> float | None:
    try:
        out = subprocess.run([_modal_cli(), "billing", "summary", "--json"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        return float(json.loads(out)["metered_cost"])
    except Exception:  # noqa: BLE001 - treat any failure as "unknown"
        return None


def _load() -> dict:
    if MODAL_LEDGER.exists():
        return json.loads(MODAL_LEDGER.read_text())
    return {"runs": []}


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def ledger_this_month() -> float:
    return sum(r["est_cost"] for r in _load()["runs"] if r["month"] == _month())


def status() -> dict:
    metered = metered_this_month()
    ledger = ledger_this_month()
    spent = max(ledger, metered or 0.0)
    return {"cap": MODAL_BUDGET_USD, "metered": metered, "ledger": round(ledger, 4),
            "spent": round(spent, 4), "remaining": round(MODAL_BUDGET_USD - spent, 4)}


def check_budget(gpu: str, max_seconds: float) -> dict:
    s = status()
    if s["metered"] is None:
        raise BudgetExceeded("Could not read Modal billing; refusing to launch a paid run.")
    worst = run_cost(gpu, max_seconds)
    if s["spent"] + worst > MODAL_BUDGET_USD:
        raise BudgetExceeded(
            f"Modal budget guard: spent ${s['spent']:.2f} of ${MODAL_BUDGET_USD:.2f}; "
            f"this run could cost up to ${worst:.2f}. Refusing."
        )
    return s


class Run:
    """Reserves a run's worst-case cost in the ledger *before* any GPU starts, then settles it to the
    measured wall-clock cost. If the process dies mid-run the reservation stays, so spend is never lost."""

    def __init__(self, gpu: str, label: str, max_seconds: float):
        self.gpu, self.label, self.max_seconds = gpu, label, max_seconds
        self.id = f"{time.time():.6f}"

    def __enter__(self):
        self.t0 = time.time()
        with _lock:
            data = _load()
            data["runs"].append({
                "id": self.id, "month": _month(), "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "label": self.label, "gpu": self.gpu, "status": "reserved", "seconds": round(self.max_seconds, 1),
                "est_cost": round(run_cost(self.gpu, self.max_seconds), 4),
            })
            MODAL_LEDGER.write_text(json.dumps(data, indent=2))
        return self

    def __exit__(self, *exc):
        secs = min(self.max_seconds, time.time() - self.t0 + 30)  # + container start/stop overhead
        with _lock:
            data = _load()
            for r in data["runs"]:
                if r.get("id") == self.id:
                    r.update(status="settled", seconds=round(secs, 1), est_cost=round(run_cost(self.gpu, secs), 4))
            MODAL_LEDGER.write_text(json.dumps(data, indent=2))
        return False
