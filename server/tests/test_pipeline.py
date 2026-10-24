import json
import struct

import numpy as np
import pytest
from plyfile import PlyData

from threeddd import budget
from threeddd.frames import select_sharpest
from threeddd.sceneinfo import SPLAT_DTYPE, encode_splat, ply_to_splat, scene_info, spc_to_ply


def make_spc(n=500, k=15, seed=0):
    """Mirror of modal_app.trainer._pack written with numpy."""
    rng = np.random.default_rng(seed)
    p = {
        "means": rng.normal(size=(n, 3)).astype(np.float32),
        "scales": rng.normal(-3, 0.5, size=(n, 3)).astype(np.float32),
        "quats": rng.normal(size=(n, 4)).astype(np.float32),
        "opacities": rng.normal(size=n).astype(np.float32),
        "sh0": rng.normal(size=(n, 3)).astype(np.float32),
        "shN": rng.normal(0, 0.1, size=(n, k, 3)).astype(np.float32),
    }
    shN = p["shN"].reshape(n, -1)
    lo, hi = shN.min(0), shN.max(0)
    q = np.clip(np.round((shN - lo) / np.maximum(hi - lo, 1e-8) * 255), 0, 255).astype(np.uint8)
    body = b"".join(np.ascontiguousarray(a).tobytes() for a in [
        p["means"], p["scales"].astype(np.float16), p["quats"].astype(np.float16),
        p["opacities"].astype(np.float16), p["sh0"].astype(np.float16), lo, hi, q])
    return b"SPC1" + struct.pack("<II", n, k) + body, p


def test_spc_roundtrip_to_standard_ply(tmp_path):
    data, p = make_spc()
    ply = tmp_path / "s.ply"
    ply.write_bytes(spc_to_ply(data))
    v = PlyData.read(str(ply))["vertex"].data
    assert len(v) == 500
    np.testing.assert_allclose(np.stack([v["x"], v["y"], v["z"]], 1), p["means"])
    np.testing.assert_allclose(v["scale_0"], p["scales"][:, 0], atol=2e-3)
    np.testing.assert_allclose(v["opacity"], p["opacities"], atol=5e-3)
    # channel-major f_rest: f_rest_[c*15 + i] == shN[:, i, c]
    np.testing.assert_allclose(v["f_rest_16"], p["shN"][:, 1, 1], atol=0.01)
    np.testing.assert_allclose(v["f_rest_44"], p["shN"][:, 14, 2], atol=0.01)


def test_ply_to_splat_layout(tmp_path):
    data, _ = make_spc(n=64)
    ply = tmp_path / "s.ply"
    ply.write_bytes(spc_to_ply(data))
    out = np.frombuffer(ply_to_splat(ply, max_splats=10), SPLAT_DTYPE)
    assert len(out) == 10
    assert (out["scale"] > 0).all()


def test_encode_splat_normalises_quaternions():
    out = np.frombuffer(encode_splat(np.zeros((1, 3)), np.ones((1, 3)), np.array([[255, 0, 0, 255]]),
                                     np.array([[2.0, 0, 0, 0]])), SPLAT_DTYPE)
    assert list(out["rot"][0]) == [255, 128, 128, 128]


def test_select_sharpest_keeps_one_per_bin():
    scores = [1, 9, 2, 3, 8, 1, 7, 1, 2]
    assert select_sharpest(scores, 3) == [1, 4, 6]
    assert select_sharpest([1, 2], 5) == [0, 1]


def test_scene_info_up_and_target(tmp_path):
    # Ring of 12 cameras at height 0 looking at the origin; COLMAP camera -Y should map to world +Z here.
    cams = []
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        c = np.array([3 * np.cos(a), 3 * np.sin(a), 0.0])
        fwd = -c / np.linalg.norm(c)
        up = np.array([0, 0, 1.0])
        right = np.cross(fwd, up)
        R = np.stack([right, -up, fwd], 1)  # camera-to-world: x=right, y=down, z=forward
        cams.append({"name": "x", "center": c.tolist(), "rotation": R.tolist(), "fx": 500, "width": 640, "height": 480})
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"points": [0, 0, 0, 0.1, 0, 0], "colors": [0] * 6, "cameras": cams}))
    info = scene_info(f)
    np.testing.assert_allclose(info["up"], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(info["target"], [0, 0, 0], atol=1e-6)


def test_budget_refuses_when_over_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(budget, "MODAL_LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(budget, "MODAL_BUDGET_USD", 1.0)
    monkeypatch.setattr(budget, "metered_this_month", lambda: 0.9)
    with pytest.raises(budget.BudgetExceeded):
        budget.check_budget("H100", 600)  # ~$0.8 worst case on top of $0.90
    assert budget.check_budget("L4", 60)["spent"] == 0.9


def test_budget_refuses_when_billing_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(budget, "MODAL_LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(budget, "metered_this_month", lambda: None)
    with pytest.raises(budget.BudgetExceeded):
        budget.check_budget("L4", 60)


def test_budget_ledger_records_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(budget, "MODAL_LEDGER", tmp_path / "ledger.json")
    with budget.Run("L4", "t", max_seconds=600):
        pass
    settled = budget.ledger_this_month()
    assert 0 < settled < budget.run_cost("L4", 600)


def test_budget_reservation_survives_a_crash(monkeypatch, tmp_path):
    """If the server dies mid-run, __exit__ never runs; the worst-case reservation must remain."""
    monkeypatch.setattr(budget, "MODAL_LEDGER", tmp_path / "ledger.json")
    run = budget.Run("H100", "t", max_seconds=900)
    run.__enter__()  # process "dies" here
    assert budget.ledger_this_month() == pytest.approx(budget.run_cost("H100", 900), abs=1e-3)
