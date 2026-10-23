# vs-sim copy: tests/fast/test_dev_compare_object.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""`scripts/dev_compare.py`'s score at the object (owner 2026-10-16) on a synthetic fixture with
hand-worked geometry: the goal camera 250 mm straight above the object's top, landings that
pivot about the top point, slide sideways or roll, the 2F-85 tolerances from a bounding box,
the precision tier, and the scopes with and without the lying bottle."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOP = 0.10                                   # the object's top (goal and live)
GOAL = [0.0, 0.0, TOP + 0.25]                # 250 mm above it
DOWN = np.diag([1.0, -1.0, -1.0])            # straight down (the goal's cam_R_opt)
ART = (-71.0, 1.0, 163.0)


def _load():
    spec = importlib.util.spec_from_file_location("dev_compare",
                                                  REPO / "scripts" / "dev_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ry(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pivot(deg):
    """The goal pose turned by ``deg`` about world y through the object's top point."""
    R = _ry(deg) @ DOWN
    p = np.array([0.0, 0.0, TOP]) - 0.25 * R[:, 2]
    return p, R


def test_pivot_slide_and_roll_by_hand():
    dc = _load()
    offs = (("cam", (0.0, 0.0, 0.0)), ("art", ART), ("turned", (0.0, -71.0, 0.0)))
    # a pure pivot about the top point: the camera is 250 sin(5) mm off sideways, but its axis
    # still meets the top at the same point
    p, R = pivot(5.0)
    o = dc.at_object(p, R, GOAL, DOWN, TOP, offs)
    assert o["tilt_deg"] == pytest.approx(5.0) and o["roll_deg"] == pytest.approx(0.0, abs=1e-9)
    assert o["cam_side_mm"] == pytest.approx(250 * math.sin(math.radians(5)))
    assert o["pivot_mm"] == pytest.approx(250 * math.tan(math.radians(5)))
    assert o["offsets"]["cam"]["grasp_mm"] == pytest.approx(0.0, abs=1e-9)
    assert o["offsets"]["cam"]["standoff_err_mm"] == pytest.approx(0.0, abs=1e-9)
    # a gripper 71 mm to the side: its axis meets the plane at -71/cos(5) instead of -71 (worked
    # out in closed form), and its standoff grows by 71 tan(5)
    t = math.radians(5)
    assert o["offsets"]["art"]["grasp_mm"] == pytest.approx(71 * (1 / math.cos(t) - 1), abs=0.02)
    assert o["offsets"]["art"]["standoff_mm"] == pytest.approx(87 + 71 * math.tan(t), abs=0.1)
    assert o["offsets"]["art"]["standoff_err_mm"] == pytest.approx(71 * math.tan(t), abs=0.1)
    # an offset along the pivot axis (world y) does not move at all
    assert o["offsets"]["turned"]["grasp_mm"] == pytest.approx(0.0, abs=1e-9)
    # a 4 mm slide, no turn: 4 mm at every offset
    o = dc.at_object([0.004, 0.0, GOAL[2]], DOWN, GOAL, DOWN, TOP, offs)
    assert [o["offsets"][n]["grasp_mm"] for n in ("cam", "art", "turned")] == pytest.approx(
        [4.0, 4.0, 4.0])
    assert o["offsets"]["cam"]["grasp_dxy_mm"] == pytest.approx([4.0, 0.0], abs=1e-9)
    assert o["tilt_deg"] == pytest.approx(0.0, abs=1e-6) and o["pivot_mm"] == pytest.approx(0.0)
    # a 10 deg roll about the viewing axis: nothing for the camera axis, the 71 mm offset swings
    # on a circle: chord 2 x 71 x sin(5 deg)
    Rr = DOWN @ _rz(10.0)
    o = dc.at_object(GOAL, Rr, GOAL, DOWN, TOP, offs)
    assert abs(o["roll_deg"]) == pytest.approx(10.0) and o["tilt_deg"] == pytest.approx(0.0,
                                                                                       abs=1e-6)
    assert o["offsets"]["cam"]["grasp_mm"] == pytest.approx(0.0, abs=1e-9)
    assert o["offsets"]["turned"]["grasp_mm"] == pytest.approx(2 * 71 * math.sin(math.radians(5)))
    assert dc.roll_deg(DOWN, DOWN @ _rz(-10.0)) == pytest.approx(-dc.roll_deg(DOWN, Rr))
    # 1 mm too high, straight down: the standoff error is +1 mm
    o = dc.at_object([0.0, 0.0, GOAL[2] + 0.001], DOWN, GOAL, DOWN, TOP, offs)
    assert o["offsets"]["art"]["standoff_err_mm"] == pytest.approx(1.0)
    # an axis that does not point down misses the plane
    o = dc.at_object(GOAL, np.eye(3), GOAL, DOWN, TOP, offs)
    assert o["offsets"]["cam"] is None and o["pivot_mm"] is None


def test_gripper_tolerances_from_the_box():
    dc = _load()
    t = dc.gripper_tolerances([[-0.05, -0.03, 0.0], [0.05, 0.03, 0.1]], "mug")
    assert (t["width_mm"], t["length_mm"]) == pytest.approx((60.0, 100.0))
    assert t["side_mm"] == pytest.approx(12.5) and t["height_mm"] == 19.0 and t["tilt_deg"] == 5.0
    psi = math.radians(t["roll_deg"])
    assert 60 * math.cos(psi) + 100 * math.sin(psi) == pytest.approx(85.0)   # just fits
    assert t["roll_deg"] == pytest.approx(15.83, abs=0.01)
    assert dc.gripper_tolerances([[-0.05, -0.03, 0], [0.05, 0.03, 0.1]], "bottle",
                                 frozenset({"bottle"}))["roll_deg"] == math.inf
    wide = dc.gripper_tolerances([[-0.1, -0.05, 0], [0.1, 0.05, 0.1]], "shoe")
    assert wide["side_mm"] is None and wide["roll_deg"] is None           # 100 mm > 85 mm
    small = dc.gripper_tolerances([[-0.02, -0.02, 0], [0.02, 0.02, 0.1]], "cup")
    assert small["roll_deg"] == 90.0                                      # fits at any turn
    assert dc.GRIPPER_2F85["opening_mm"] == 85.0 and dc.GRIPPER_2F85["pad_length_mm"] == 38.0


def _flight(sid, p, R, *, live_bbox, deg):
    p = [float(v) for v in p]
    return {"start_id": sid, "steps": 200, "final_mm": 1000 * math.dist(p, GOAL),
            "final_deg": deg, "final_pose": {"position": p, "rotation": np.asarray(R).tolist()},
            "stop_step": None, "stop_mm": None, "stop_deg": None, "stops": {},
            "own_stop_rule": None, "start_mm": 7.0, "start_deg": 2.0,
            "perturbation": {"swap_instance": {
                "goal_bbox": [[-0.05, -0.03, 0.0], [0.05, 0.03, TOP]], "live_bbox": live_bbox}}}


def _write(root: Path, scene: str, live: str, eps):
    pair = f"{scene}_{live}"
    (root / pair).mkdir(parents=True, exist_ok=True)
    doc = {"scenes": {scene: {"goal": {"cam_pos": GOAL, "cam_R_opt": DOWN.tolist()},
                              "arms": {"m": {"label": "m", "near": eps[0], "far": eps[1:]}}}}}
    (root / pair / "check.json").write_text(json.dumps(doc))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "runs"
    box = [[-0.05, -0.03, 0.0], [0.05, 0.03, TOP]]          # w 60 mm, L 100 mm
    pv, Rv = pivot(6.0)
    _write(root, "inst_mug", "same", [
        _flight("n_000", [0.002, 0.0, GOAL[2]], DOWN, live_bbox=box, deg=0.5),    # 2 mm: tier
        _flight("s_00", [0.004, 0.0, GOAL[2]], DOWN, live_bbox=box, deg=0.5),     # 4 mm
        _flight("s_01", GOAL, DOWN @ _rz(10.0), live_bbox=box, deg=10.0)])        # roll 10
    _write(root, "inst_mug", "other_small", [
        _flight(i, pv, Rv, live_bbox=box, deg=6.0) for i in ("n_000", "s_00", "s_01")])
    _write(root, "inst_bottle", "same", [
        _flight(i, [0.002, 0.0, GOAL[2]], DOWN, live_bbox=box, deg=0.5)
        for i in ("n_000", "s_00", "s_01")])
    return root, tmp_path / "out"


def test_rows_tables_and_scopes(run):
    root, out = run
    dc = _load()
    assert dc.main(["--root", str(root / "none"), "--out", str(out), "--run", f"X={root}:m",
                    "--tcp-offset", "cam=0,0,0", "--tcp-offset", f"art={','.join(map(str, ART))}"]
                   ) == 0
    rows = json.loads((out / "rows.json").read_text())
    R = {(r["pair"], r["start_id"]): r for r in rows}
    r = R[("inst_mug_other_small", "s_00")]
    assert r["object"]["offsets"]["cam"]["grasp_mm"] == pytest.approx(0.0, abs=1e-6)
    assert r["object"]["tilt_deg"] == pytest.approx(6.0)
    # tilt 6 deg is not < 5: the pivot fails the PROPOSED set although its grasp point is exact
    assert r["gripper_pass"] == {"cam": False, "art": False}
    assert R[("inst_mug_same", "s_00")]["gripper_pass"]["cam"] is True              # 4 < 12.5
    assert R[("inst_mug_same", "s_01")]["gripper_pass"]["cam"] is True              # 10 < 15.85
    assert R[("inst_mug_same", "s_01")]["sweep_pass"]["cam"]["5"] is False          # roll > 5
    assert R[("inst_mug_same", "s_00")]["sweep_pass"]["cam"] == {
        "5": True, "10": True, "15": True, "20": True}
    s = json.loads((out / "summary.json").read_text())
    c = s["scopes"]["all kinds"]["counts"]["depth_true"]["X"]["same"]
    # 2 mm (mug n_000) and the three bottle flights at 2 mm are inside the 3 mm tier
    assert (c["n"], c["3mm_1deg"], c["1cm_1deg"]) == (6, 4, 5)
    nb = s["scopes"]["without the lying bottle"]["counts"]["depth_true"]["X"]["same"]
    assert (nb["n"], nb["3mm_1deg"], nb["1cm_1deg"]) == (3, 1, 2)
    assert s["scopes"]["the lying bottle only"]["counts"]["depth_true"]["X"]["all"]["n"] == 3
    o = s["scopes"]["without the lying bottle"]["object"]["X"]["other_small"]
    assert o["median_cam_side_mm"] == pytest.approx(250 * math.sin(math.radians(6)), abs=0.05)
    assert o["offsets"]["cam"]["median_grasp_mm"] == 0.0
    assert o["offsets"]["cam"]["graspable"] == 3 and o["offsets"]["cam"]["gripper_pass"] == 0
    md = (out / "table.md").read_text()
    for head in ("## Camera pose against the depth-true pose - without the lying bottle",
                 "### PROPOSED gripper set (2F-85) - the lying bottle only",
                 "### Width-free sweep - all kinds", "### PROPOSED tolerances per live object"):
        assert head in md
    assert s["meta"]["offsets"] == [["cam", [0.0, 0.0, 0.0]], ["art", list(ART)]]
    assert s["meta"]["tolerances"]["inst_mug_same"]["side_mm"] == pytest.approx(12.5)


def test_round_kind_and_offset_refusals(run):
    root, out = run
    dc = _load()
    dc.main(["--root", str(root / "none"), "--out", str(out), "--run", f"X={root}:m",
             "--round-kind", "mug"])
    rows = json.loads((out / "rows.json").read_text())
    r = [x for x in rows if x["pair"] == "inst_mug_same" and x["start_id"] == "s_01"][0]
    assert math.isinf(r["gripper_tolerances"]["roll_deg"])          # JSON `Infinity`
    assert r["sweep_pass"]["camera axis"]["5"] is True               # the roll is free now
    for bad in ("nooffset", "a=1,2", "=1,2,3", "a=x,y,z"):
        with pytest.raises(SystemExit):
            dc.parse_offset(bad)
    with pytest.raises(SystemExit, match="share a name"):
        dc.main(["--root", str(root / "none"), "--out", str(out), "--run", f"X={root}:m",
                 "--tcp-offset", "a=0,0,0", "--tcp-offset", "a=1,1,1"])


def _targets_file(path: Path, pairs: dict) -> Path:
    path.write_text(json.dumps({"schema": "vssim-approach-targets-1", "pairs": {
        p: {"R_target": np.asarray(R).tolist(), "moved": m, "target_tilt_deg": t}
        for p, (R, m, t) in pairs.items()}}))
    return path


def test_the_approach_target_turns_the_depth_true_pose_about_its_grasp_point(run, tmp_path):
    """Ruling 5: the 6 deg pivot about the top point IS the target when the live top is turned
    6 deg about y - distance 0, angle 0, grasp point 0, tilt 0 - while the depth-true pose
    (straight down) scores it 26 mm; the same object keeps the depth-true pose even if
    the file names it; a pair marked not moved keeps it too; near and far are split."""
    root, out = run
    dc = _load()
    _pv, Rv = pivot(6.0)
    f = _targets_file(tmp_path / "t.json", {"inst_mug_other_small": (Rv, True, 6.0),
                                            "inst_mug_same": (Rv, True, 6.0)})
    assert dc.main(["--root", str(root / "none"), "--out", str(out), "--run", f"X={root}:m",
                    "--tcp-offset", "cam=0,0,0", "--approach-targets", str(f)]) == 0
    rows = json.loads((out / "rows.json").read_text())
    R = {(r["pair"], r["start_id"]): r for r in rows}
    r = R[("inst_mug_other_small", "s_00")]
    assert r["approach_moved"] is True and r["approach_tilt_deg"] == pytest.approx(6.0)
    assert r["mm_approach"] == pytest.approx(0.0, abs=1e-9)
    assert r["deg_approach"] == pytest.approx(0.0, abs=1e-6)
    assert r["mm_depth_true"] == pytest.approx(250 * math.hypot(math.sin(math.radians(6)),
                                                                1 - math.cos(math.radians(6))))
    assert r["within_3mm_1deg_approach"] is True and r["within_3cm_3deg_depth_true"] is False
    oa = r["object_approach"]
    assert oa["tilt_deg"] == pytest.approx(0.0, abs=1e-6)
    assert oa["offsets"]["cam"]["grasp_mm"] == pytest.approx(0.0, abs=1e-9)
    s = R[("inst_mug_same", "s_00")]                    # same object: the depth-true pose
    assert s["approach_moved"] is False and s["mm_approach"] == pytest.approx(s["mm_depth_true"])
    summ = json.loads((out / "summary.json").read_text())
    assert summ["targets"] == ["goal", "depth_true", "approach"]
    bs = summ["scopes"]["all kinds"]["by_start"]["approach"]["X"]
    assert bs["near"]["other_small"]["n"] == 1 and bs["far"]["other_small"]["n"] == 2
    assert summ["meta"]["deg_compared"] == 9             # (the fixture's angles are made up)
    f2 = _targets_file(tmp_path / "t2.json", {"inst_mug_other_small": (Rv, False, 0.0)})
    assert dc.main(["--root", str(root / "none"), "--out", str(out), "--run", f"X={root}:m",
                    "--tcp-offset", "cam=0,0,0", "--approach-targets", str(f2)]) == 0
    r = {(x["pair"], x["start_id"]): x for x in json.loads((out / "rows.json").read_text())}[
        ("inst_mug_other_small", "s_00")]
    assert r["approach_moved"] is False and r["mm_approach"] == pytest.approx(r["mm_depth_true"])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "pairs": {}}))
    with pytest.raises(SystemExit):
        dc.main(["--root", str(root / "none"), "--out", str(out), "--run", f"X={root}:m",
                 "--approach-targets", str(bad)])
