# vs-sim copy: tests/fast/test_dev_compare.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""`scripts/dev_compare.py` (the development set's start-by-start comparison, 2026-10-10) on a
tiny synthetic fixture: two dev pairs written the way `check_baselines.py` writes them - the
first suite's records without depth-true fields, the October re-flies' with them - and one
extra run. Every number below is worked out by hand from the fixture's positions."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOAL = [0.0, 0.0, 0.35]                 # the goal camera, looking down at the object's top
GOAL_TOP, SMALL_TOP = 0.10, 0.07        # the goal object's top; the smaller copy's


def _load():
    spec = importlib.util.spec_from_file_location("dev_compare",
                                                  REPO / "scripts" / "dev_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mm(p, q=GOAL):
    return 1000.0 * math.dist(p, q)


def _ep(sid, *, final, deg=0.5, stop=None, own=None, live_top=GOAL_TOP, telemetry=False):
    """One flight record. ``final`` / ``stop`` / ``own``: the landing position at the cap, at
    the shared scored stop (``(step, position)``) and at DINOBot's own stop."""
    dt = [GOAL[0], GOAL[1], GOAL[2] + live_top - GOAL_TOP]
    e = {"start_id": sid, "steps": 200, "final_mm": _mm(final), "final_deg": deg,
         "final_pose": {"position": final}, "stop_step": None, "stop_mm": None,
         "stop_deg": None, "stops": {}, "own_stop_rule": None, "start_mm": 7.0,
         "start_deg": 2.0,
         "perturbation": {"swap_instance": {"goal_bbox": [[-0.1, -0.1, 0.0], [0.1, 0.1, GOAL_TOP]],
                                            "live_bbox": [[-0.1, -0.1, 0.0],
                                                          [0.1, 0.1, live_top]]}}}
    if telemetry:
        e["final_translation_mm_depth_true"] = _mm(final, dt)
    if stop is not None:
        step, pos = stop
        e.update(stop_step=step, stop_mm=_mm(pos), stop_deg=deg)
        e["stops"]["cns_plateau"] = {"step": step, "translation_mm": _mm(pos),
                                     "rotation_deg": deg, "position": pos}
        e["stops"]["velocity_quiet"] = {"step": step + 5, "translation_mm": 99.0,
                                        "rotation_deg": 9.0, "position": [0, 0, 0]}
        if telemetry:
            e["stop_translation_mm_depth_true"] = _mm(pos, dt)
    if own is not None:
        e["own_stop_rule"] = "dinobot_err"
        if own != "never":
            step, pos = own
            e["stops"]["dinobot_err"] = {"step": step, "translation_mm": _mm(pos),
                                         "rotation_deg": deg, "position": pos,
                                         **({"translation_mm_depth_true": _mm(pos, dt)}
                                            if telemetry else {})}
    return e


def _write(root: Path, pair: str, arms: dict):
    """``arms = {arm: [near, s_00, s_01]}`` -> ``root/pair/check.json``."""
    (root / pair).mkdir(parents=True, exist_ok=True)
    scene = pair.rsplit("_same", 1)[0]
    doc = {"scenes": {scene: {"goal": {"cam_pos": GOAL}, "arms": {
        arm: {"label": f"{arm}-label", "near": eps[0], "far": eps[1:]}
        for arm, eps in arms.items()}}}}
    (root / pair / "check.json").write_text(json.dumps(doc))


def _at(dz_mm, dx_mm=0.0):
    """A camera position ``dx`` sideways and ``dz`` above the goal camera (mm)."""
    return [dx_mm / 1000.0, 0.0, GOAL[2] + dz_mm / 1000.0]


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "vssim_out"
    ids = ("n_000", "s_00", "s_01")
    # run 1 (no depth-true fields): ViT-VS reads at its scored stop on the near start, at the
    # cap elsewhere; DINOBot at its own stop, or the cap when it never fired
    _write(root, "inst_shoe_same", {
        "vitvs_dinov3_objmask": [_ep("n_000", final=_at(0, 50), stop=(120, _at(0, 4))),
                                 _ep("s_00", final=_at(0, 20)), _ep("s_01", final=_at(0, 40))],
        "dinobot_objmask": [_ep(i, final=_at(0, 30), own=(3, _at(0, 5)), deg=0.3) for i in ids[:2]]
        + [_ep("s_01", final=_at(0, 25), own="never", deg=0.3)]})
    # the smaller copy: its top 30 mm lower, so the depth-true pose is 30 mm below the goal pose
    _write(root, "inst_shoe_same_small", {
        "vitvs_dinov3_objmask": [_ep(i, final=_at(-30, 6), live_top=SMALL_TOP) for i in ids],
        "dinobot_objmask": [_ep(i, final=_at(0, 1), own=(2, _at(-28, 0)), deg=0.3,
                                live_top=SMALL_TOP) for i in ids]})
    # the October re-flies (with depth-true fields)
    r2 = root / "suite_r2"
    _write(r2, "inst_shoe_same", {
        "vitvs_dinov3_objmask_ema08": [_ep(i, final=_at(0, 12), telemetry=True) for i in ids],
        "dinobot_objmask": [_ep(i, final=_at(0, 30), own=(3, _at(0, 6)), deg=0.3, telemetry=True)
                            for i in ids]})
    _write(root / "suite_r2_artvs", "inst_shoe_same", {
        "artvs_roi_sim": [_ep(i, final=_at(0, 8), telemetry=True) for i in ids]})
    dev = tmp_path / "dev_artvs"
    _write(dev, "inst_shoe_same", {
        "artvs_roi_sim_dev": [_ep("n_000", final=_at(0, 2), telemetry=True),
                              _ep("s_00", final=_at(0, 9), telemetry=True),
                              _ep("s_01", final=_at(0, 25), deg=1.5, telemetry=True)]})
    return root, dev, tmp_path / "out"


def test_the_landing_is_the_protocols_reading_and_both_targets_are_scored(fixture):
    root, dev, out = fixture
    dc = _load()
    assert dc.main(["--root", str(root), "--out", str(out),
                    "--run", f"ART-VS dev={dev}:artvs_roi_sim_dev"]) == 0
    rows = json.loads((out / "rows.json").read_text())
    R = {(r["method"], r["pair"], r["start_id"]): r for r in rows}
    v = R[("ViT-VS", "inst_shoe_same", "n_000")]
    assert (v["read_at"], v["read_step"], v["mm_goal"]) == ("stop", 120, pytest.approx(4.0))
    assert v["depth_true_from"] == "computed" and v["mm_depth_true"] == pytest.approx(4.0)
    assert R[("ViT-VS", "inst_shoe_same", "s_00")]["read_at"] == "cap"
    d = R[("DINOBot run 1", "inst_shoe_same", "s_00")]
    assert (d["read_at"], d["read_step"], d["mm_goal"]) == ("own", 3, pytest.approx(5.0))
    assert R[("DINOBot run 1", "inst_shoe_same", "s_01")]["read_at"] == "cap"   # never fired
    assert R[("DINOBot run 1", "inst_shoe_same", "s_01")]["mm_goal"] == pytest.approx(25.0)
    # the smaller copy: ViT-VS 30 mm above it ends ON the depth-true height (6 mm sideways);
    # DINOBot ends 2 mm above the depth-true pose and 28 mm below the goal pose
    vs = R[("ViT-VS", "inst_shoe_same_small", "s_00")]
    assert vs["mm_goal"] == pytest.approx(math.hypot(30, 6))
    assert vs["mm_depth_true"] == pytest.approx(6.0)
    assert vs["depth_true_position"][2] == pytest.approx(GOAL[2] - 0.03)
    ds = R[("DINOBot run 1", "inst_shoe_same_small", "n_000")]
    assert ds["mm_goal"] == pytest.approx(28.0) and ds["mm_depth_true"] == pytest.approx(2.0)
    assert ds["within_1cm_1deg_depth_true"] and not ds["within_1cm_1deg_goal"]
    assert ds["within_3cm_3deg_goal"]
    # the October records carry their own depth-true distance, which is used (and agrees)
    o = R[("DINOBot run 2", "inst_shoe_same", "s_01")]
    assert o["depth_true_from"] == "record" and o["mm_depth_true"] == pytest.approx(6.0)
    summary = json.loads((out / "summary.json").read_text())
    # record and computation agree on all 12 October rows (4 columns x 3 flights)
    assert summary["meta"]["dt_compared"] == 12 and summary["meta"]["dt_max_abs_diff_mm"] < 1e-9
    assert summary["meta"]["rows_per_method"] == {
        "ViT-VS": 6, "ViT-VS + smoothing": 3, "DINOBot run 1": 6, "DINOBot run 2": 3,
        "ART-VS faithful": 3, "ART-VS dev": 3}
    assert len(rows) == 24


def test_counts_pairs_and_the_band(fixture):
    root, dev, out = fixture
    dc = _load()
    dc.main(["--root", str(root), "--out", str(out),
             "--run", f"ART-VS dev={dev}:artvs_roi_sim_dev"])
    s = json.loads((out / "summary.json").read_text())
    assert s["method"] == "ART-VS dev"                      # the first --run by default
    c = s["counts"]["goal"]["ART-VS dev"]["same"]
    # 2, 9 and 25 mm (the last at 1.5 deg): within 3 mm / 1 deg 1; within 1 cm / 1 deg 2;
    # within 3 cm / 3 deg 3
    assert (c["n"], c["3mm_1deg"], c["1cm_1deg"], c["3cm_3deg"], c["median_mm"]) == (
        3, 1, 2, 3, 9.0)
    v = s["counts"]["goal"]["ViT-VS"]
    assert (v["same"]["1cm_1deg"], v["same"]["3cm_3deg"], v["same"]["median_mm"]) == (1, 2, 20.0)
    assert v["same_small"]["3cm_3deg"] == 0 and s["counts"]["depth_true"]["ViT-VS"][
        "same_small"]["1cm_1deg"] == 3
    assert v["all"]["n"] == 6 and s["counts"]["goal"]["ViT-VS"]["mine1"]["n"] == 0
    # start by start against the faithful ART-VS (8, 8, 8 mm): -6, +1, +17 mm - closer once,
    # farther twice, median +1 mm
    p = s["paired"]["goal"]["ART-VS faithful"]["same"]
    assert (p["n"], p["closer"], p["farther"], p["median_diff_mm"]) == (3, 1, 2, pytest.approx(1.0))
    assert p["median_abs_diff_mm"] == pytest.approx(6.0)
    # against ViT-VS run 1 (4, 20, 40 mm): closer on all three
    q = s["paired"]["goal"]["ViT-VS"]["all"]
    assert (q["n"], q["closer"], q["median_diff_mm"]) == (3, 3, pytest.approx(-11.0))
    # the band: DINOBot run 2 (6, 6, 6 mm) against run 1 (5, 5, 25 mm) on the same flights
    b = s["band"]["goal"]["same"]
    assert (s["band"]["a"], s["band"]["b"]) == ("DINOBot run 2", "DINOBot run 1")
    assert (b["n"], b["closer"], b["farther"], b["median_diff_mm"]) == (3, 1, 2, pytest.approx(1.0))
    assert b["flips"] == {"3mm_1deg": 0, "1cm_1deg": 1, "3cm_3deg": 0}
    md = (out / "table.md").read_text()
    assert "## Start by start: ART-VS dev against each, to the goal pose" in md
    assert "_band: DINOBot run 2 against DINOBot run 1_" in md
    assert "| ART-VS dev | 1 / 2 / 3 of 3, 9.0 mm |" in md


def test_the_command_line_refusals(fixture, tmp_path):
    root, dev, out = fixture
    dc = _load()
    assert dc.parse_run("A b=/x/y:z:arm_1") == ("A b", Path("/x/y:z"), "arm_1")
    for bad in ("nolabel", "=d:a", "L=d", "L=:a", "L=d:"):
        with pytest.raises(SystemExit):
            dc.parse_run(bad)
    with pytest.raises(SystemExit, match="share a label"):
        dc.main(["--root", str(root), "--out", str(out), "--run", f"ViT-VS={dev}:x"])
    with pytest.raises(SystemExit, match="not a column"):
        dc.main(["--root", str(root), "--out", str(out), "--method", "nope"])
    with pytest.raises(SystemExit, match="no dev flight"):
        dc.main(["--root", str(root), "--out", str(out), "--run", f"X={tmp_path}/empty:a"])
    # the superseded column only on request; without --run the method is the faithful ART-VS
    dc.main(["--root", str(root), "--out", str(out)])
    s = json.loads((out / "summary.json").read_text())
    assert s["method"] == "ART-VS faithful"
    assert "ART-VS as ported, superseded" not in s["meta"]["rows_per_method"]
    dc.main(["--root", str(root), "--out", str(out), "--with-ported"])
    s = json.loads((out / "summary.json").read_text())
    assert s["meta"]["rows_per_method"]["ART-VS as ported, superseded"] == 0


def test_the_script_needs_numpy_and_pyyaml_only():
    """It runs anywhere the records are (the host, the laptop): no torch, OpenCV or Pillow."""
    import subprocess
    import sys

    script = str(REPO / "scripts" / "dev_compare.py")
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('dc', {script!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(sorted(k for k in ('torch', 'cv2', 'PIL', 'transformers', 'scipy') "
            "if k in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=60, check=True).stdout.strip()
    assert out == "[]"
