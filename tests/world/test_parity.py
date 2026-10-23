"""U5: the copied baselines land the same in the copied numpy mock and in the MuJoCo world.

``tools/u5_parity.py`` flies ``vitvs_dinov3``, ``vitvs_dinov2``, ``artvs`` and the control
``sift_ransac`` on hollywood from the first three ``near60`` and the first two ``narrow20``
starts, 150 steps each, in three worlds: the copied ``MockPlaneScene`` (``mock``),
``MujocoScene(look="raw")`` (``mujoco``, Isaac's principal point) and the same MuJoCo scene
drawn with the mock's principal point (``mujoco960``, the attribution control). Flown with::

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/u5_parity.py fly --world <mock|mujoco|mujoco960> --arms <arm>
    PYTHONPATH=. $VSBENCH_PY tools/u5_parity.py compare

This module reads the committed records under ``results/bridge/u5/`` (it renders nothing;
each test skips until its records exist) and gates (docs/PORT_PLAN.md section 7, U5):

* the records: the copied frozen schema, the provenance, 150 steps, the same starts and the
  same landmark draw seeds in every world, the principal point each world drew with, every
  MuJoCo record's GL renderer is Mesa llvmpipe;
* the landing gate, per arm and start, mock vs mujoco: ``|d final_translation_mm| <= 3`` and
  ``|d final_rotation_deg| <= 0.3`` at the cap. A miss is reported with the curve evidence
  (median over the last third, max over the second half) - never a looser threshold;
* ART-VS's coarse -> tiled switch (``artvs_switched``) fires on every start in both worlds.

With ``VSBENCH_WRITE_GATES=1`` the gates write ``results/bridge/u5_parity.json``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
U5 = REPO / "results" / "bridge" / "u5"
GATE_JSON = REPO / "results" / "bridge" / "u5_parity.json"
GATES: dict = {}

from tools import u5_parity as tool  # noqa: E402

DRAWN_PP = {"mock": [960.0, 540.0], "mujoco": [959.5, 539.5], "mujoco960": [960.0, 540.0]}
EXPECTED_STARTS = {"near60/n_000", "near60/n_001", "near60/n_002",
                   "narrow20/start_00", "narrow20/start_01"}


@pytest.fixture(scope="module", autouse=True)
def _write_gates(tmp_path_factory):
    yield
    if not GATES:
        return
    out = (GATE_JSON if os.environ.get("VSBENCH_WRITE_GATES", "").strip() == "1"
           else tmp_path_factory.mktemp("gates") / GATE_JSON.name)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(GATES)
    old["unit"] = ("U5 - baseline parity: the copied numpy mock vs the MuJoCo world "
                   "(docs/PORT_PLAN.md section 7)")
    old["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out.write_text(json.dumps(old, indent=2, sort_keys=True, default=str) + "\n")
    print(f"[gates] wrote {out}")


@pytest.fixture(scope="module")
def result():
    return tool.compare(U5, "hollywood")


def _need(result, worlds, arm):
    have = [w for w in worlds if arm in tool.load_records(U5).get(w, {})]
    if len(have) < len(worlds):
        pytest.skip(f"{arm} not flown in {sorted(set(worlds) - set(have))} yet "
                    "(see the module docstring)")


# ------------------------------------------------------------------- the tool itself
def _fake(world, t, v, final_mm, final_deg, switched=None, step=None):
    return {"start_id": "n_000", "start_translation_mm": t[0], "start_rotation_deg": 1.0,
            "final_translation_mm": final_mm, "final_rotation_deg": final_deg, "steps": len(t),
            "terminated_reason": "max_iter", "stop_step": None, "first_frame_n_matches": 24,
            "artvs_switched": switched, "sec_per_step": 1.0, "render_mode": world,
            "u5": {"world": world, "start_set": "near60", "gl_renderer": None,
                   "principal_point_drawn": [960.0, 540.0], "cpu_s": 1.0,
                   "final_position": [0.0, 0.0, final_mm / 1e3], "final_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   "artvs_switch_step": step,
                   "curve": {"i": list(range(1, len(t) + 1)), "t_mm": t, "r_deg": [0.1] * len(t),
                             "v_xyz": v, "w": v, "n": [24] * len(t)}}}


def test_curve_statistics_and_the_pair_rule():
    """Last third = steps 101-150, second half = 76-150 on a 150-step curve; the landing
    gate reads the cap, the curve-median reading is reported beside it."""
    t = [float(i) for i in range(150, 0, -1)]               # 150 ... 1
    s = tool.curve_stats(_fake("mock", t, t, 1.0, 0.1))
    assert s["steps_last_third"] == [101, 150] and s["steps_second_half"] == [76, 150]
    assert s["t_mm_med_last_third"] == 25.5 and s["t_mm_max_second_half"] == 75.0
    a = _fake("mock", t, t, 1.0, 0.10)
    b = _fake("mujoco", [x + 0.5 for x in t], t, 4.5, 0.15)
    p = tool.pair(a, b)
    assert p["d_final_translation_mm"] == 3.5 and not p["landing_gate"]
    assert p["d_t_mm_med_last_third"] == 0.5 and p["curve_median_gate"]
    assert p["landing_pose_distance_mm"] == 3.5 and p["twist_norm_corr_v_xyz"] == 1.0
    assert tool.pair(a, _fake("mujoco", t, t, 3.9, 0.39))["landing_gate"]


# ------------------------------------------------------------------------ the records
@pytest.mark.parametrize("arm", tool.ARMS)
def test_records_are_paired_schema_valid_and_on_llvmpipe(arm):
    from vsbench.world.fly import vssim_commit

    recs = tool.load_records(U5)
    worlds = [w for w in tool.WORLDS if arm in recs.get(w, {})]
    if not worlds:
        pytest.skip(f"{arm} not flown yet (see the module docstring)")
    fly_smoke = importlib.util.spec_from_file_location("_u2_fly_smoke", REPO / "tests" / "world"
                                                       / "test_fly_smoke.py")
    mod = importlib.util.module_from_spec(fly_smoke)
    fly_smoke.loader.exec_module(mod)
    seeds = {}
    for w in worlds:
        by_start = recs[w][arm]
        assert set(by_start) == EXPECTED_STARTS, (w, sorted(by_start))
        for key, rec in by_start.items():
            mod.assert_schema_valid({k: v for k, v in rec.items() if k != "u5"})
            assert rec["steps"] == rec["max_iter"] == 150 and len(rec["u5"]["curve"]["t_mm"]) == 150
            assert rec["vssim_commit"] == vssim_commit()
            assert rec["vsbench_commit"] and len(rec["vsbench_commit"].split("+")[0]) == 40
            assert rec["u5"]["principal_point_drawn"] == DRAWN_PP[w]
            assert rec["u5"]["principal_point_reported"] == [960.0, 540.0]
            assert rec.get("matcher_device") in (None, "cpu")
            if w == "mock":
                assert rec["render_mode"] == tool.MOCK_RENDER_MODE
            else:
                assert rec["render_mode"] == "mujoco-llvmpipe"
                assert "llvmpipe" in str(rec["u5"]["gl_renderer"]).lower()
                assert rec["look"] == "raw"
            seeds.setdefault(key, set()).add(rec["subsample_seed"])
    assert all(len(v) == 1 for v in seeds.values()), seeds     # the same landmark draw


# --------------------------------------------------------------------------- the gates
@pytest.mark.parametrize("arm", tool.ARMS)
def test_landing_gate_mock_vs_mujoco(result, arm):
    _need(result, tool.GATE_PAIR, arm)
    rows = {k: v["mock-mujoco"] for k, v in result["pairs"][arm].items()}
    worlds = result["table"][arm]
    entry = {"gate": result["gate"]["rule"], "pass": all(r["landing_gate"] for r in rows.values()),
             "starts": {k: {"landing_mm": {w: worlds[k][w]["final_translation_mm"] for w in worlds[k]},
                            "landing_deg": {w: worlds[k][w]["final_rotation_deg"] for w in worlds[k]},
                            "t_mm_med_last_third": {w: worlds[k][w]["t_mm_med_last_third"]
                                                    for w in worlds[k]},
                            "t_mm_max_second_half": {w: worlds[k][w]["t_mm_max_second_half"]
                                                     for w in worlds[k]},
                            "r_deg_med_last_third": {w: worlds[k][w]["r_deg_med_last_third"]
                                                     for w in worlds[k]},
                            "pairs": result["pairs"][arm][k]}
                        for k in sorted(rows)}}
    GATES[f"landing_{arm}"] = entry
    bad = {k: (r["d_final_translation_mm"], r["d_final_rotation_deg"], r["curve_median_gate"])
           for k, r in rows.items() if not r["landing_gate"]}
    assert not bad, (f"{arm}: landings differ by more than 3 mm / 0.3 deg between the mock and "
                     f"MuJoCo on {bad} (d mm, d deg, curve medians agree?)")


def test_artvs_switch_fires_in_both_worlds(result):
    _need(result, tool.GATE_PAIR, "artvs")
    v = result["verdict"]["artvs_switch"]
    GATES["artvs_switch"] = v
    assert v is not None and v["pass"], v


def test_cpu_and_renderers_are_recorded(result):
    if not result["table"]:
        pytest.skip("nothing flown yet")
    GATES["cpu_s_per_world"] = result["cpu_s_per_world"]
    GATES["gl_renderers"] = result["gl_renderers"]
    assert all("llvmpipe" in r.lower() for r in result["gl_renderers"]), result["gl_renderers"]
