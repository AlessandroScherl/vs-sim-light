"""The ``isaac_fit_v1`` look (``vsbench.world.look``).

1. config  - every scene's curve: monotone per channel, its 256-entry table's md5 is the
             config's, the parameters reproduce the table; the Isaac pictures it was fitted on
             (read only, ``$VSBENCH_ISAAC_GOALS``) still have the md5s the config names - on a
             machine without them that provenance check is not made and records ``None``;
2. raw     - ``look="raw"`` draws the unfitted picture bit for bit (pinned goal md5s of
             hollywood and industrial1_drill);
3. RGB only - raw and look draw the same depth bytes (goal and a near pose; hollywood and
             industrial1_drill), different RGB; the records name the look.

The rendering tests pin pictures drawn by Mesa llvmpipe: they run under software GL (station
mode, ``source tools/cpu_env.sh``, or ``VSBENCH_SOFTWARE_GL=1``) and skip otherwise.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vsbench.glenv import software_gl_requested  # noqa: E402
from vsbench.world import look as L  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
#: the scenes whose assets this module needs (tests/conftest.py skips it without them)
NEEDS_SCENES = ("hollywood", "industrial1_drill")
OUT = Path("u4b_look_gates.json")   # written to a pytest temp dir
SCENES = ("hollywood", "poster2", "poster3", "industrial1_drill", "cluttered2", "cluttered3")
#: the raw look's goal pictures (llvmpipe)
RAW_GOAL_MD5 = {"hollywood": "04938e905ddffd06dd816a45b24804c7",
                "industrial1_drill": "c350e2218d9920b9b3faf94cd5df5fde"}
MEAN_GREY_TOL = 6.0
EMD_RATIO_MAX = 0.2
RESULTS: dict = {}
renders = pytest.mark.skipif(
    not software_gl_requested(),
    reason="pins llvmpipe pictures: needs software GL (station mode or VSBENCH_SOFTWARE_GL=1)")


@pytest.fixture(scope="module", autouse=True)
def _write_results(tmp_path_factory):
    yield
    if not RESULTS:
        return
    out = tmp_path_factory.mktemp("gates") / OUT.name
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(RESULTS)
    old["unit"] = "the isaac_fit_v1 look"
    old["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
    print(f"[gates] wrote {out}")


def _md5(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------- 1. config
def test_config_curves_are_monotone_and_pinned():
    """The curves are monotone and pinned by their table md5s. The provenance check (the Isaac
    goal picture each curve was fitted on still has the config's md5) needs those pictures
    under ``$VSBENCH_ISAAC_GOALS``; without them it is not made, and the gate JSON records
    ``isaac_goal_md5_matches: None`` for the scene (not ``True``, not a failure)."""
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
        assert src["goal"].startswith("$VSBENCH_ISAAC_GOALS/"), src     # no personal path
        goal = L.resolve_isaac_path(src["goal"])
        if goal.exists():
            on_disk = L.file_md5(goal) == src["goal_md5"]
            assert on_disk, f"{goal} is not the picture the curve was fitted on"
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
