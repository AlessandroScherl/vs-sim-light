"""``--mask``: filter the matches with a mask on the goal picture (2026-10-23).

* the helpers: ``mask_matches`` (nearest pixel, off-picture = outside), ``takes_goal_mask``
  (a ``goal_mask`` parameter, ``uses_goal_mask``), ``check_method``'s refusal, the file
  reader (grey > 127, nearest-neighbour resize, md5 of the bytes, an empty mask refused);
* the routing of a copied arm: ``objects`` -> the protocol's ``<arm>_objmask`` cell (or the
  arm plus ``matcher.goal_mask: objects``), label ``objmask``; a file -> the same seam,
  ``goal_mask_mode: file``, label ``filemask-<md5>``, record ``goal_mask_source: file:...``;
* a file mask filters a copied arm's matches (no landmark outside the mask; without the mask
  some are) and a method's (``sift_ibvs``, ``examples/my_method.py``);
* user methods with and without mask support: the latter flies unmasked as before and is
  refused a mask, before anything is built;
* the GT-blind tripwire with a mask: every truth read of the loop poisoned, the masked
  method's commands bit-identical;
* the drivers on MuJoCo (software GL only): ``--mask objects`` on the drill with
  ``--debug-gif``, and a file mask on poster2 for a copied arm and the example.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from vsbench.methods import make_method, mask_matches, takes_goal_mask  # noqa: E402
from vsbench.methods.base import check_method  # noqa: E402
from vsbench.methods.feature_ibvs import SiftIBVS  # noqa: E402
from vsbench.world import fly, masks  # noqa: E402
from vsbench.world import method_arm as ma  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "my_method.py"


def _method_arm_helpers():
    spec = importlib.util.spec_from_file_location("_u6a_method_arm", REPO / "tests" / "world"
                                                  / "test_method_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _method_arm_helpers()


def _example():
    import vsbench.__main__ as cli

    cli.load_plugin(str(EXAMPLE))
    return make_method("my_ibvs")


def left_half_png(path: Path, w: int = 1920, h: int = 1080) -> Path:
    img = np.zeros((h, w), np.uint8)
    img[:, : w // 2] = 255
    cv2.imwrite(str(path), img)
    return path


class Mocked:
    """A scene stand-in for ``masks.resolve`` (intrinsics and a name are all a file needs)."""

    def __init__(self, scene):
        self.intrinsics = scene.intrinsics
        self.name = scene.name
        self.goal_pose = scene.goal_pose


# ----------------------------------------------------------------- helpers
def test_mask_matches():
    m = np.zeros((4, 5), bool)
    m[1:3, 1:4] = True
    keep = mask_matches([[0, 0], [2, 2], [3.4, 1.6], [3.6, 1.0], [10, 10], [-1, 2], [np.nan, 1]],
                        np.zeros((7, 2)), m)
    assert keep.tolist() == [False, True, True, False, False, False, False]
    assert mask_matches([[0, 0], [9, 9]], None, None).tolist() == [True, True]
    assert mask_matches(np.zeros((0, 2)), np.zeros((0, 2)), m).shape == (0,)
    with pytest.raises(ValueError, match="rows"):
        mask_matches([[0, 0]], np.zeros((2, 2)), m)


class _Base:
    wants_depth = "none"
    privileged = False
    name = "probe"

    def configure(self, K, width, height, dt):
        pass

    def reset(self):
        pass

    def get_control_rate(self, current_rgb, current_depth=None):
        return np.zeros(6)


class WithMask(_Base):
    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None, goal_mask=None):
        self.goal_mask = goal_mask


class WithoutMask(_Base):
    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None):
        pass


def test_takes_goal_mask():
    assert takes_goal_mask(WithMask()) and takes_goal_mask(WithMask)
    assert not takes_goal_mask(WithoutMask())
    opt_out = WithMask()
    opt_out.uses_goal_mask = False
    assert not takes_goal_mask(opt_out)
    liar = WithoutMask()
    liar.uses_goal_mask = True
    with pytest.raises(TypeError, match="goal_mask"):
        takes_goal_mask(liar)
    with pytest.raises(TypeError, match="goal_mask"):
        check_method(liar)
    check_method(WithMask())
    check_method(WithoutMask())
    assert takes_goal_mask(SiftIBVS()) and takes_goal_mask(_example())


def test_file_mask_reader(tmp_path):
    p = left_half_png(tmp_path / "half.png", 96, 54)
    mask, md5, resized = masks.load_file_mask(p, 192, 108)
    assert mask.shape == (108, 192) and resized == [96, 54]
    assert mask[:, :96].all() and not mask[:, 96:].any()          # nearest neighbour: crisp
    assert md5 == __import__("hashlib").md5(p.read_bytes()).hexdigest()
    same, _, none = masks.load_file_mask(p, 96, 54)
    assert none is None and same.mean() == 0.5
    cv2.imwrite(str(tmp_path / "empty.png"), np.zeros((10, 10), np.uint8))
    with pytest.raises(SystemExit, match="keeps no pixel"):
        masks.load_file_mask(tmp_path / "empty.png", 10, 10)
    assert masks.parse(None) == ("none", None) and masks.parse("objects") == ("objects", None)
    assert masks.parse(str(p)) == ("file", str(p))
    with pytest.raises(SystemExit, match="existing image"):
        masks.parse(str(tmp_path / "nope.png"))


# ----------------------------------------------------------------- routing
def _copied_arm(arm_id, gm, intrinsics, scene_name="hollywood", max_iter=5):
    from vsbench.vs.scenes import load_scenes

    proto = fly.load_protocol()
    specs, _ = load_scenes()
    cell, over = masks.arm_cell(proto, arm_id, gm)
    cfg = fly.arm_config(proto, cell, max_iter=max_iter, device="cpu",
                         goal_mask="objects" if over else None)
    arm = fly.build(proto, cell, cfg, intrinsics=intrinsics, scene_name=scene_name,
                    scene_spec=specs[scene_name])
    return masks.attach_goal_mask(arm, gm), cfg, cell


def test_copied_arm_routing():
    from vsbench.world.camera import d435i_intrinsics

    intr = d435i_intrinsics()
    full = np.ones((intr.height, intr.width), bool)
    obj = masks.GoalMask("objects", mask=full, source=masks.OBJECTS_SOURCE)
    fil = masks.GoalMask("file", mask=full, source="file:m.png:0123456789abcdef", file="m.png",
                         file_md5="0123456789abcdef")
    plain, _, cell = _copied_arm("sift_ransac", masks.GoalMask("none"), intr)
    assert cell == "sift_ransac" and "objmask" not in plain.name
    assert plain.goal_mask_mode == "none" and "filemask" not in plain.name
    a, cfg, cell = _copied_arm("sift_ransac", obj, intr)
    assert cell == "sift_ransac_objmask"                      # the protocol's own cell
    assert a.name.split("/").count("objmask") == 1 and a.goal_mask_mode == "objects"
    assert cfg["matcher"].get("goal_mask") == "objects"
    a, cfg, cell = _copied_arm("orb_ransac", obj, intr)       # no _objmask twin: the override
    assert cell == "orb_ransac" and "goal_mask" in cfg["_source"]
    assert "/objmask/" in a.name
    a, cfg, cell = _copied_arm("orb_ransac", fil, intr)
    assert a.goal_mask_mode == "file" and "objmask" not in a.name
    assert "/filemask-01234567/" in a.name
    assert masks.GoalMask("none").tag == "" and obj.tag == "_objmask"
    assert fil.tag == "_filemask_01234567" and fil.token == "filemask-01234567"
    rec = fil.record()
    assert rec["goal_mask_source"].startswith("file:m.png:") and rec["goal_mask_share"] == 1.0
    assert masks.GoalMask("none").record() == {}


# ----------------------------------------------------------------- the filter
class _Points:
    def __init__(self):
        self.goal_pts: list[np.ndarray] = []

    def __call__(self, step_i, rgb, debug, pose):
        from vsbench.vs.bench.films import FilmWriter

        pg, _pc = FilmWriter.points(debug)
        self.goal_pts.append(np.asarray(pg, float).reshape(-1, 2))


def _file_mask(tmp_path, scene, name="half.png"):
    # the mask at the camera's full size; the mock renders at a quarter, so it is resized
    return masks.resolve(str(left_half_png(tmp_path / name)), Mocked(scene), None)


def test_a_file_mask_filters_a_copied_arms_matches(tmp_path):
    from vsbench.vs.bench.runner import run_episode

    scene = H.hollywood_mock()
    gm = _file_mask(tmp_path, scene)
    assert gm.resized_from == [1920, 1080] and gm.mask.shape == (scene.intrinsics.height,
                                                                 scene.intrinsics.width)
    out = {}
    for label, g in (("plain", masks.GoalMask("none")), ("masked", gm)):
        arm, cfg, _ = _copied_arm("sift_ransac", g, scene.intrinsics, max_iter=4)
        hook = _Points()
        rec = run_episode(scene, arm, H.near_start(0), cfg, on_step=hook)
        g.apply(rec)
        out[label] = (rec, np.concatenate(hook.goal_pts))
    rec, pts = out["masked"]
    assert len(pts) > 20 and mask_matches(pts, None, gm.mask).all(), "a landmark left the mask"
    assert not mask_matches(out["plain"][1], None, gm.mask).all(), "the unmasked arm never left it"
    assert rec["goal_mask_mode"] == "file" and rec["goal_mask_source"] == gm.source
    assert rec["goal_mask_share"] == pytest.approx(0.5) and "/filemask-" in rec["arm"]
    assert "goal_mask_source" not in out["plain"][0]
    assert out["plain"][0]["goal_mask_mode"] == "none"


@pytest.mark.parametrize("which", ["sift_ibvs", "my_ibvs"])
def test_a_file_mask_filters_a_methods_matches(tmp_path, which):
    from vsbench.vs.bench.runner import run_episode

    scene = H.hollywood_mock()
    gm = _file_mask(tmp_path, scene)
    if which == "my_ibvs":
        _example()
    arm, cfg = ma.build_method_arm(which, intrinsics=scene.intrinsics, max_iter=3,
                                   goal_mask_mode=gm.mode, mask_token=gm.token,
                                   mask_source=gm.source)
    masks.attach_goal_mask(arm, gm)
    assert arm.name.startswith(f"external:{which}@") and f"/{gm.token}/cns_plateau" in arm.name
    assert cfg["matcher"]["goal_mask"] == "file" and cfg["matcher"]["goal_mask_source"] == gm.source
    hook = _Points()
    rec = run_episode(scene, arm, H.near_start(0), cfg, on_step=hook)
    gm.apply(rec)
    pts = np.concatenate(hook.goal_pts)
    assert len(pts) > 8 and mask_matches(pts, None, gm.mask).all()
    assert rec["external_method_takes_goal_mask"] is True and rec["goal_mask_mode"] == "file"
    if which == "my_ibvs":                  # the method's copy of the mask is read-only
        assert arm.method.goal_mask is not None and not arm.method.goal_mask.flags.writeable


def test_a_method_without_mask_support(tmp_path):
    from vsbench.vs.bench.runner import run_episode

    scene = H.hollywood_mock(0.125)
    cfg = ma.external_config("probe", max_iter=2)
    arm = ma.MethodArm(WithoutMask(), cfg, intrinsics=scene.intrinsics)     # unmasked: as before
    with pytest.warns(ma.StopFallbackWarning):
        rec = run_episode(scene, arm, H.near_start(0), cfg)
    assert rec["steps"] == 2 and "external_method_takes_goal_mask" not in rec
    masked = json.loads(json.dumps(cfg))
    masked["matcher"]["goal_mask"] = "file"
    with pytest.raises(ValueError, match="takes no goal mask"):
        ma.MethodArm(WithoutMask(), masked, intrinsics=scene.intrinsics, goal_mask_mode="file",
                     mask_token="filemask-00000000")
    with pytest.raises(ValueError, match="takes no goal_mask"):
        arm.matcher.reset(np.zeros((scene.intrinsics.height, scene.intrinsics.width, 3), np.uint8),
                          None, np.ones((scene.intrinsics.height, scene.intrinsics.width), bool))
    # the driver refuses before building anything; a method that takes one passes
    from vsbench import methods

    methods.register("probe_without_mask")(type("ProbeWithoutMask", (WithoutMask,), {}))
    try:
        with pytest.raises(SystemExit, match="takes no goal_mask"):
            fly.check_mask_methods(["probe_without_mask"])
    finally:
        methods._REGISTRY.pop("probe_without_mask", None)
    fly.check_mask_methods(["sift_ibvs", "orb_ibvs"])


# ----------------------------------------------------------------- GT-blind
def test_a_masked_method_never_sees_ground_truth(tmp_path):
    from vsbench.vs.bench.perturb import Perturbation
    from vsbench.vs.bench.runner import run_episode

    class MaskedRecordingSift(H.RecordingSift):
        pass

    def flight(scene, pert):
        gm = _file_mask(tmp_path, scene)
        cfg = ma.external_config("sift_ibvs", max_iter=6)
        cfg["episode"].update(telemetry_extended=True, depth_true=True)
        cfg["matcher"].update(goal_mask="file", goal_mask_source=gm.source)
        method = MaskedRecordingSift()
        arm = ma.MethodArm(method, cfg, intrinsics=scene.intrinsics, goal_mask_mode="file",
                           mask_token=gm.token)
        masks.attach_goal_mask(arm, gm)
        rec = run_episode(scene, arm, H.near_start(0), cfg, telemetry=[], perturbation=pert)
        return rec, method

    clean_rec, clean = flight(H.hollywood_mock(), Perturbation())
    poisoned = H.Poisoned(H.hollywood_mock())
    pois_rec, pois = flight(poisoned, H._poison_perturbation())
    assert len(clean.twists) == len(pois.twists) == 6
    assert all(np.array_equal(a, b) for a, b in zip(clean.twists, pois.twists, strict=True))
    assert poisoned.gt_calls == 6 and clean.truth_calls == pois.truth_calls == 0
    assert pois_rec["final_translation_mm"] != clean_rec["final_translation_mm"]
    assert clean_rec["goal_mask_mode"] == pois_rec["goal_mask_mode"] == "file"


# ----------------------------------------------------------------- the drivers
def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", VSBENCH_SOFTWARE_GL="1",
                PYTHONPATH=f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")


soft_gl = pytest.mark.skipif(os.environ.get("VSBENCH_SOFTWARE_GL") != "1",
                             reason="flies on MuJoCo: software GL only (source tools/cpu_env.sh)")


@soft_gl
def test_driver_mask_objects_and_debug_gif_on_the_drill(tmp_path):
    cmd = [sys.executable, "-m", "vsbench", "fly", "--scene", "industrial1_drill", "--method",
           "sift_ransac", "sift_ibvs", "--start-set", "capped30", "--limit", "1",
           "--max-iter", "3", "--mask", "objects", "--debug-gif", "--out", str(tmp_path)]
    r = subprocess.run(cmd, cwd=REPO, env=_env(), capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    rec = json.loads((tmp_path / "industrial1_drill_sift_ransac_objmask.jsonl").read_text()
                     .splitlines()[0])
    assert rec["goal_mask_mode"] == "objects" and "/objmask/" in rec["arm"]
    assert rec["goal_mask_source"] == "objects_from_depth" and 0.0 < rec["goal_mask_share"] < 0.5
    gif = tmp_path / rec["debug_gif"]
    assert gif.is_file() and gif.name == "debug.gif" and gif.stat().st_size > 10_000
    ext = json.loads((tmp_path / "industrial1_drill_external_sift_ibvs_objmask.jsonl").read_text()
                     .splitlines()[0])
    assert ext["goal_mask_mode"] == "objects" and "/objmask/" in ext["arm"]
    assert ext["goal_mask_share"] == rec["goal_mask_share"]
    assert (tmp_path / ext["debug_gif"]).is_file()
    run_json = json.loads(next((tmp_path / "industrial1_drill_sift_ransac_objmask")
                               .glob("*/run.json")).read_text())
    assert run_json["bridge_goal_mask"]["protocol_cell"] == "sift_ransac_objmask"


def _poster2_texture() -> Path:
    root = os.environ.get("VSBENCH_WORKSPACE_ROOT", str(Path.home() / "genvs_ws"))
    return Path(root) / "assets" / "poster2" / "poster2.png"


@soft_gl
def test_driver_file_mask_on_a_poster(tmp_path):
    if not _poster2_texture().exists():
        pytest.skip(f"poster2's texture is not at {_poster2_texture()}")
    png = left_half_png(tmp_path / "left.png", 640, 360)
    md5 = __import__("hashlib").md5(png.read_bytes()).hexdigest()
    cmd = [sys.executable, "-m", "vsbench", "fly", "--scene", "poster2", "--method",
           "orb_ransac", "my_ibvs", "--plugin", str(EXAMPLE), "--start-set", "near60",
           "--limit", "1", "--max-iter", "3", "--mask", str(png), "--out", str(tmp_path / "o")]
    r = subprocess.run(cmd, cwd=REPO, env=_env(), capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    for stem in (f"poster2_orb_ransac_filemask_{md5[:8]}",
                 f"poster2_external_my_ibvs_filemask_{md5[:8]}"):
        rec = json.loads((tmp_path / "o" / f"{stem}.jsonl").read_text().splitlines()[0])
        assert rec["goal_mask_mode"] == "file" and f"/filemask-{md5[:8]}/" in rec["arm"]
        assert "objmask" not in rec["arm"]
        assert rec["goal_mask_source"] == f"file:left.png:{md5}"
        assert rec["goal_mask_resized_from"] == [640, 360]
        assert rec["goal_mask_share"] == pytest.approx(0.5)
    # objects on a poster: refused, the goal view is all ground plane
    q = subprocess.run([*cmd[:cmd.index("--mask")], "--mask", "objects", "--out",
                        str(tmp_path / "p")], cwd=REPO, env=_env(), capture_output=True,
                       text=True, timeout=900)
    assert q.returncode != 0 and "ground plane" in q.stderr


def test_check_baselines_takes_the_flag():
    from vsbench.world import check_baselines as cb

    args = cb.build_parser().parse_args(["--scenes", "hollywood", "--arms", "sift_ransac",
                                         "--out", "x", "--mask", "objects"])
    assert args.mask == "objects"
    bad = cb.build_parser().parse_args(["--scenes", "hollywood", "--out", "x", "--mask",
                                        "objects", "--arm-plugin", "vsbench.world.method_arm"])
    with pytest.raises(SystemExit, match="arm-plugin"):
        cb.validate_args(bad, cb.reference(), fly.load_protocol())
