"""U4b gates (docs/PORT_PLAN.md section 7, U4b): the ``isaac_fit_v1`` look.

1. config  - every scene's curve: monotone per channel, its 256-entry table's md5 is the
             config's, the parameters reproduce the table; the Isaac pictures it was fitted on
             (read only) still have the md5s the config names;
2. raw     - ``look="raw"`` draws U1/U4's picture bit for bit (hollywood's goal md5 = the M1
             flight's, industrial1_drill's = ``tools/u4_rooms.py measure``'s);
3. RGB only - raw and look draw the same depth bytes (goal and a near pose; hollywood and
             industrial1_drill), different RGB; the records name the look;
4. picture - from ``results/bridge/u4b_look.json`` (``tools/u4b_look.py measure``): per scene,
             on the fitted goal picture **and** on the held-out near view, the look is within
             6 grey levels of Isaac's mean grey and its histogram distance (mean per-channel
             earth mover's distance, grey levels) is <= 0.2 x the raw picture's; the record
             was drawn on llvmpipe; depth identical in both looks; and the live hollywood goal
             md5 in the look equals the record's (the record is this code's).

Numbers land in ``results/bridge/u4b_look_gates.json`` only with ``VSBENCH_WRITE_GATES=1``.
CPU only: ``source tools/cpu_env.sh``; the rendering tests skip without software GL.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vsbench.glenv import software_gl_requested  # noqa: E402
from vsbench.world import look as L  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "bridge" / "u4b_look_gates.json"
RECORD = REPO / "results" / "bridge" / "u4b_look.json"
SCENES = ("hollywood", "poster2", "poster3", "industrial1_drill", "cluttered2", "cluttered3")
#: U1/U4's goal pictures (raw): the M1 flight's hollywood goal, U4's measure of the drill room
RAW_GOAL_MD5 = {"hollywood": "04938e905ddffd06dd816a45b24804c7",
                "industrial1_drill": "c350e2218d9920b9b3faf94cd5df5fde"}
MEAN_GREY_TOL = 6.0
EMD_RATIO_MAX = 0.2
RESULTS: dict = {}
renders = pytest.mark.skipif(
    not software_gl_requested(),
    reason="renders: needs VSBENCH_SOFTWARE_GL=1 (source tools/cpu_env.sh) so no GPU is touched")


@pytest.fixture(scope="module", autouse=True)
def _write_results(tmp_path_factory):
    yield
    if not RESULTS:
        return
    out = (OUT if os.environ.get("VSBENCH_WRITE_GATES", "").strip() == "1"
           else tmp_path_factory.mktemp("gates") / OUT.name)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(RESULTS)
    old["unit"] = "U4b - match Isaac's look (docs/PORT_PLAN.md section 7)"
    old["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
    print(f"[gates] wrote {out}")


def _md5(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------- 1. config
def test_config_curves_are_monotone_and_pinned():
    cfg = L.load_config("isaac_fit_v1")
    assert set(cfg["scenes"]) == set(SCENES)
    rows = {}
    for s in SCENES:
        look = L.load_look("isaac_fit_v1", s)          # raises on a table md5 mismatch
        lut = look.lut.astype(int)
        assert lut.shape == (256, 3)
        assert (np.diff(lut, axis=0) >= 0).all(), s
        x = np.linspace(0, 1, 2001)
        for p in look.channel_params:
            assert np.all(np.diff(L.curve_model(p, x)) > 0), (s, p)     # strictly, before rounding
        assert look.curve_md5 == cfg["scenes"][s]["curve"]["lut_md5"]
        assert L.lut_md5(L.curve_lut(look.channel_params)) == look.curve_md5
        src = cfg["scenes"][s]["isaac_source"]
        on_disk = None
        if Path(src["goal"]).exists():
            on_disk = L.file_md5(src["goal"]) == src["goal_md5"]
            assert on_disk, f"{src['goal']} is not the picture the curve was fitted on"
        rows[s] = {"curve_md5": look.curve_md5, "params_rgb": look.channel_params,
                   "light_gain": look.gain, "lut_black": lut[0].tolist(), "lut_white": lut[255].tolist(),
                   "isaac_goal_md5_matches": on_disk}
    RESULTS["config"] = {"file": str(L.config_path("isaac_fit_v1").relative_to(REPO)),
                         "md5": L.file_md5(L.config_path("isaac_fit_v1")),
                         "ambient_share": cfg["lighting"]["ambient_share"], "scenes": rows}


def test_unknown_look_and_scene_are_refused():
    with pytest.raises(ValueError):
        L.load_look("isaac_fit_v2", "hollywood")
    with pytest.raises(KeyError):
        L.load_look("isaac_fit_v1", "no_such_scene")
    assert L.load_look("raw", "anything").lut is None


# --------------------------------------------------------------- 2-3. raw, RGB only
@renders
@pytest.mark.parametrize("scene_name", ["hollywood", "industrial1_drill"])
def test_raw_is_u4_and_the_look_changes_rgb_only(scene_name):
    from vsbench.vs import starts as startsmod
    from vsbench.vs.scene import Pose
    from vsbench.world.mujoco_scene import MujocoScene

    out = {}
    for name in ("raw", "isaac_fit_v1"):
        with MujocoScene(scene_name, look=name) as sc:
            assert sc.render_mode == "mujoco-llvmpipe" and "llvmpipe" in sc.renderer.gl_renderer
            g_rgb, g_d = sc.render_goal()
            g = sc.goal_pose
            near = startsmod.sample_near(g.position, g.rotation, sc.working_distance, n=1)[0]
            sc.reset(Pose.from_start(near))
            n_rgb, n_d, _ = sc.render()
            cap = sc.goal_capture_record()
            out[name] = {"goal_rgb": _md5(g_rgb), "goal_depth": _md5(g_d), "near_rgb": _md5(n_rgb),
                         "near_depth": _md5(n_d), "capture": {k: cap[k] for k in (
                             "look", "look_curve_id", "look_curve_md5")},
                         "look_record": cap["look_record"]}
    raw, lk = out["raw"], out["isaac_fit_v1"]
    RESULTS.setdefault("rgb_only", {})[scene_name] = {
        k: {"raw": raw[k], "isaac_fit_v1": lk[k]} for k in ("goal_rgb", "goal_depth", "near_rgb", "near_depth")}
    RESULTS["rgb_only"][scene_name]["raw_goal_is_u4"] = raw["goal_rgb"] == RAW_GOAL_MD5[scene_name]
    assert raw["goal_rgb"] == RAW_GOAL_MD5[scene_name]
    assert raw["goal_depth"] == lk["goal_depth"] and raw["near_depth"] == lk["near_depth"]
    assert raw["goal_rgb"] != lk["goal_rgb"] and raw["near_rgb"] != lk["near_rgb"]
    assert raw["capture"] == {"look": "raw", "look_curve_id": None, "look_curve_md5": None}
    want = L.load_look("isaac_fit_v1", scene_name)
    assert lk["capture"] == {"look": "isaac_fit_v1", "look_curve_id": f"isaac_fit_v1/{scene_name}",
                             "look_curve_md5": want.curve_md5}
    rec = lk["look_record"]
    assert rec["shadowsize"] == 8192 and rec["shadowclip_m"] > 0 and rec["light_gain"] == want.gain
    assert rec["look_config_md5"] == want.config_md5


# ------------------------------------------------------------------- 4. the picture
def test_look_gate_against_isaac():
    if not RECORD.exists() or "measure" not in json.loads(RECORD.read_text()):
        pytest.skip("run tools/u4b_look.py measure first")
    m = json.loads(RECORD.read_text())["measure"]
    assert m["look_config_md5"] == L.file_md5(L.config_path("isaac_fit_v1")), \
        "the measure record was made with another config: re-run tools/u4b_look.py measure"
    rows = {}
    for s in SCENES:
        r = m["scenes"][s]
        assert "llvmpipe" in r["gl_renderer"] and r["render_mode"] == "mujoco-llvmpipe"
        assert r["depth_identical_raw_vs_look"] == {"goal": True, "near": True}, s
        for view in ("goal", "near"):
            v = r[view]
            if v.get("isaac") is None and "raw" not in v:
                continue
            raw, lk = v["raw"], v["isaac_fit_v1"]
            row = {"role": "fitted" if view == "goal" else "held out",
                   "mean_grey": {"raw": raw["mean_grey"]["mujoco"], "look": lk["mean_grey"]["mujoco"],
                                 "isaac": raw["mean_grey"]["isaac"]},
                   "hist_emd_mean": {"raw": raw["hist_emd_mean"], "look": lk["hist_emd_mean"]},
                   "pixel_mae": {"raw": raw["pixel_mae"], "look": lk["pixel_mae"]}}
            rows.setdefault(s, {})[view] = row
            assert abs(lk["mean_grey_diff"]) <= MEAN_GREY_TOL, (s, view, row)
            assert lk["hist_emd_mean"] <= EMD_RATIO_MAX * raw["hist_emd_mean"], (s, view, row)
    RESULTS["picture"] = {"scenes": rows, "gate": f"goal (fitted) and near (held out): |mean grey - "
                          f"Isaac's| <= {MEAN_GREY_TOL}, histogram EMD <= {EMD_RATIO_MAX} x raw's; "
                          "depth identical raw vs look; llvmpipe"}


@renders
def test_the_record_is_this_code():
    from vsbench.world.mujoco_scene import MujocoScene

    if not RECORD.exists():
        pytest.skip("run tools/u4b_look.py measure first")
    want = json.loads(RECORD.read_text())["measure"]["scenes"]["hollywood"]["goal_frame_md5"]
    with MujocoScene("hollywood") as sc:                      # the default look
        rgb, _ = sc.render_goal()
        assert sc.look.name == "isaac_fit_v1"
    assert _md5(rgb) == want["isaac_fit_v1"]
