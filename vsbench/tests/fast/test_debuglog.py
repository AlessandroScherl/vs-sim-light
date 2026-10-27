# vs-sim copy: tests/fast/test_debuglog.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules + patch P3  # noqa: E501
"""The debug telemetry (owner 2026-09-30): the recorder, the truth per match, the depth-true
pose, the check script's debug path, the viewer, the re-fly chain - all on the numpy mock.

The load-bearing test is :func:`test_debug_logging_is_additive`: each arm kind of the
re-fly (the mock's truth arm, the DINOBot stub arm, the ART-VS stub arm) flown twice, the
recorder off and on - the per-step poses must be bit-identical, the old telemetry
identical, and the records equal except the new keys and the timing columns.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vsbench.vs.bench import perturb, runner
from vsbench.vs.bench.debuglog import (
    TRUTH_COMMON_PX,
    TRUTH_RIGHT_CELLS,
    DebugLog,
    cell_px,
    dinobot_block,
    display_name,
    error_split,
    orientation_split,
    read_telemetry,
)
from vsbench.vs.scene import Pose, rodrigues

REPO = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    """A sibling test module or a script, by path (the fast tier has no package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load("tests/fast/test_runner.py", "_dbg_test_runner")

#: The record's columns that measure time or hardware: they differ between ANY two runs.
TIMING = {"wall_time_s", "sec_per_step", "fps", "match_fps", "match_ms_med", "control_ms_med",
          "gpu_util_mean_pct", "gpu_util_max_pct", "gpu_util_samples", "peak_vram_alloc_mb",
          "vram_used_mb"}
#: What the debug path adds to a record: its own time, and with a depth-true pose offered
#: and `episode.depth_true` on, the depth-true reads (+ final_pose when nothing added it).
DEPTH_TRUE_KEYS = {"depth_true_pose", "start_translation_mm_depth_true",
                   "final_translation_mm_depth_true", "stop_translation_mm_depth_true",
                   "success_depth_true"}


# ------------------------------------------------------------ test worlds
class _RefDepthPerturbation:
    """Offers a size-matched pose 5 mm beside the goal and a depth-true pose 3 mm below."""

    mode = "test_ref_depth"

    def __init__(self):
        self._ref = self._dt = None

    def episode(self, scene_name, start_id):
        return 1

    def begin(self, scene, seed, *, goal, working_distance, intrinsics):
        self._ref = Pose(position=goal.position + [0.005, 0.0, 0.0], rotation=goal.rotation)
        self._dt = Pose(position=goal.position + [0.0, 0.0, -0.003], rotation=goal.rotation)
        return {"mode": self.mode}

    def live(self, rgb, step):
        return rgb

    def end(self, scene):
        return None

    def reference_pose(self):
        return self._ref

    def depth_true_pose(self):
        return self._dt


class _ScaledPlate(_RefDepthPerturbation):
    """The mock plate scaled by ``s`` about ``(cx, cy)`` for the flight (test-only): the
    world a same-object, other-size pair makes, and the truth model that says so."""

    mode = "test_scaled"

    def __init__(self, s=0.8, centre=(0.02, -0.01)):
        super().__init__()
        self.s, self.centre = float(s), tuple(float(c) for c in centre)

    def begin(self, scene, seed, *, goal, working_distance, intrinsics):
        scene.live_scale = (self.s, *self.centre)
        return super().begin(scene, seed, goal=goal, working_distance=working_distance,
                             intrinsics=intrinsics)

    def end(self, scene):
        scene.live_scale = None

    def truth_model(self):
        return {"object": "scaled", "s": self.s, "centre_xy": list(self.centre),
                "floor": "same", "floor_z": None}


class _TruthOnly(_RefDepthPerturbation):
    """A world change that only states a truth model (the world itself is unchanged)."""

    def __init__(self, model):
        super().__init__()
        self.model = dict(model)

    def truth_model(self):
        return dict(self.model)


# ---------------------------------------------------------------- the arms
def _truth_arm(noise=0.0):
    def make(scene, cfg):
        return tr.make_arm(scene, cfg, pixel_noise=noise)
    return make


def _dinobot_arm():
    tdb = _load("tests/fast/test_dinobot.py", "_dbg_test_dinobot")
    from vsbench.vs.baselines import dinobot as db

    def make(scene, cfg):
        return db.build_dinobot_arm(cfg, intrinsics=scene.intrinsics,
                                    extractor=tdb.StubExtractor())
    return make, tdb.dinobot_config(dinobot={"load_size": 128})


class _MockDetector:
    """The ART-VS detector stub on the mock's picture size."""

    def __init__(self, box, hw):
        self.box, self.hw = list(box), hw

    def detect(self, rgb, prompt):
        m = np.zeros(self.hw, bool)
        x1, y1, x2, y2 = (int(v) for v in self.box)
        m[y1:y2, x1:x2] = True
        return {"box": list(self.box), "score": 0.9, "mask": m, "n_boxes": 1}


def _artvs_arm():
    tob = _load("tests/fast/test_objroi.py", "_dbg_test_objroi")
    from vsbench.vs.baselines.objroi import ObjectRoiTracker, RoiTiledMatcher
    from vsbench.vs.bench import arms
    from vsbench.vs.bench import config as cfgmod

    cfg = cfgmod.load(REPO / "configs" / "arms" / "artvs_roi.yaml")
    cfg["matcher"]["device"] = "cpu"
    box = [120.0, 60.0, 360.0, 210.0]

    def make(scene, cfg):
        hw = (scene.intrinsics.height, scene.intrinsics.width)
        m = RoiTiledMatcher(num_landmarks=24, seed=3, backbone=tob._Backbone(), prompt="plate",
                            switch_iteration=3,
                            tracker=ObjectRoiTracker(_MockDetector(box, hw),
                                                     tob._Tracker([[124, 62, 364, 212]]),
                                                     prompt="plate", padding=0.0))
        return arms.build_arm(cfg, intrinsics=scene.intrinsics, matcher=m)
    return make, cfg


def _episode_cfg(cfg, **episode):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    cfg["episode"] = {**(cfg.get("episode") or {}), **episode}
    return cfg


def _fly(make_arm, cfg, *, scale=0.125, pert=None, debug_dir=None, start=None):
    """One mock flight; returns (record, old telemetry, per-step poses, renders)."""
    from vsbench.vs.scene import MockPlaneScene

    scene = MockPlaneScene(render_scale=scale)
    poses, telem = [], []
    dlog = DebugLog(debug_dir, arm_id="test_arm") if debug_dir is not None else None
    try:
        rec = runner.run_episode(
            scene, make_arm(scene, cfg), start or tr.make_start(scene), cfg, telemetry=telem,
            on_step=lambda i, rgb, dbg, pose: poses.append(
                (pose.position.copy(), pose.rotation.copy())),
            perturbation=pert, debug_log=dlog)
    finally:
        if dlog is not None:
            dlog.close()
    return rec, telem, poses, scene.n_renders


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# ================================================================ G1
def test_the_telemetry_file_is_written_and_parses(tmp_path):
    cfg = tr.make_config(max_iter=30, depth_true=True)
    rec, _t, _p, _n = _fly(_truth_arm(2.0), cfg, pert=_RefDepthPerturbation(),
                           debug_dir=tmp_path / "ep")
    lines = read_telemetry(tmp_path / "ep" / "telemetry.jsonl.gz")
    kinds = [e["kind"] for e in lines]
    assert kinds == ["meta"] + ["step"] * rec["steps"] + ["final", "end"]
    meta = lines[0]
    for key in ("schema", "scene", "arm_id", "display", "label", "start_id", "max_iter", "dt",
                "image_wh", "intrinsics", "poses", "heights", "truth", "cell_rule", "stops",
                "perturbation", "arm_goal", "files"):
        assert key in meta, key
    assert meta["poses"]["size_matched"] is not None and meta["poses"]["depth_true"] is not None
    assert meta["truth"]["right_cells"] == TRUTH_RIGHT_CELLS
    steps = [e for e in lines if e["kind"] == "step"]
    assert [s["i"] for s in steps] == list(range(1, rec["steps"] + 1))
    for key in ("i", "pose", "to", "r_deg", "tilt_deg", "tilt_dir_deg", "roll_deg", "feat", "n",
                "pairs", "cell_px", "truth", "right", "v_raw", "v_cmd", "lam", "stop", "end",
                "ms"):
        assert key in steps[0], key
    for s in steps:
        assert np.all(np.isfinite(s["pose"]["p"])) and len(s["pose"]["R"]) == 9
        assert set(s["to"]) == {"goal", "size_matched", "depth_true"}
        assert len(s["truth"]["stat"]) == len(s["pairs"]["cu"])
    final, end = lines[-2], lines[-1]
    assert final["after_step"] == rec["steps"]
    np.testing.assert_allclose(final["pose"]["p"], rec["final_pose"]["position"], atol=1e-6)
    assert end["outcome"] == rec["outcome"] and end["steps"] == rec["steps"]
    assert end["success_depth_true"] == rec["success_depth_true"]
    # a killed run: the file cut in half still reads up to the cut
    raw = (tmp_path / "ep" / "telemetry.jsonl.gz").read_bytes()
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(raw[: len(raw) // 2])
    part = read_telemetry(cut)
    assert 1 <= len(part) < len(lines) and part[0]["kind"] == "meta"


# ================================================================ G2
@pytest.mark.parametrize("case", [
    "truth_noisy_refs_outline_off", "truth_clean_no_refs_outline_on", "dinobot_refs",
    "artvs_no_refs", "truth_stops_refs_outline_off", "truth_stops_refs_outline_on"])
def test_debug_logging_is_additive(tmp_path, case):
    if case.startswith("truth_stops"):
        # long enough for stops to FIRE (cns_plateau ~163, velocity_quiet ~233), so the
        # stop dicts, their depth-true reads and the outline-at-stop path are compared
        make, scale, pert_cls = _truth_arm(0.5), 0.125, _RefDepthPerturbation
        cfg = tr.make_config(max_iter=300, outline_iou=case.endswith("outline_on"))
    elif case == "truth_noisy_refs_outline_off":
        make, cfg, scale, pert_cls = _truth_arm(2.0), tr.make_config(max_iter=80), 0.125, \
            _RefDepthPerturbation
    elif case == "truth_clean_no_refs_outline_on":
        make, cfg, scale, pert_cls = _truth_arm(0.0), tr.make_config(max_iter=40,
                                                                     outline_iou=True), 0.125, None
    elif case == "dinobot_refs":
        pytest.importorskip("torch")
        pytest.importorskip("sklearn")
        make, base = _dinobot_arm()
        cfg, scale, pert_cls = _episode_cfg(base, max_iter=5), 0.25, _RefDepthPerturbation
    else:
        pytest.importorskip("torch")
        make, base = _artvs_arm()
        cfg, scale, pert_cls = _episode_cfg(base, max_iter=8), 0.25, None
    off_cfg = cfg
    on_cfg = _episode_cfg(cfg, depth_true=True)

    def pert():
        return pert_cls() if pert_cls is not None else None

    off = _fly(make, off_cfg, scale=scale, pert=pert())
    on = _fly(make, on_cfg, scale=scale, pert=pert(), debug_dir=tmp_path / "ep")
    rec_off, tel_off, poses_off, n_off = off
    rec_on, tel_on, poses_on, n_on = on
    # the flights: every per-step pose bit for bit, the old telemetry identical
    assert len(poses_off) == len(poses_on) == rec_off["steps"]
    for (p0, r0), (p1, r1) in zip(poses_off, poses_on, strict=True):
        assert np.array_equal(p0, p1) and np.array_equal(r0, r1)
    assert _same(tel_off, tel_on)
    # the record: the new keys and nothing else, every old value the same
    new = {"debug_log_s"}
    if pert_cls is not None:
        new |= DEPTH_TRUE_KEYS
    assert set(rec_on) - set(rec_off) == new, set(rec_on) ^ set(rec_off)
    assert set(rec_off) <= set(rec_on)
    for k in set(rec_off) - TIMING:
        if k == "stops":
            continue
        assert _same(rec_off[k], rec_on[k]), k
    assert set(rec_off["stops"]) == set(rec_on["stops"])
    if case.startswith("truth_stops"):
        assert {"cns_plateau", "velocity_quiet"} <= set(rec_off["stops"]), rec_off["stops"]
        if case.endswith("outline_on"):
            assert all("outline_iou" in st for st in rec_off["stops"].values())
        assert rec_on["stop_translation_mm_depth_true"] is not None
    for name, st in rec_off["stops"].items():
        on_st = dict(rec_on["stops"][name])
        if pert_cls is not None:
            assert "translation_mm_depth_true" in on_st
        on_st.pop("translation_mm_depth_true", None)
        assert _same(st, on_st), name
    # at most one extra render, after the flight: none when the outline score renders anyway
    outline = bool((cfg.get("episode") or {}).get("outline_iou"))
    assert n_on - n_off == (0 if outline else 1)
    lines = read_telemetry(tmp_path / "ep" / "telemetry.jsonl.gz")
    assert sum(e["kind"] == "step" for e in lines) == rec_on["steps"]
    if case == "dinobot_refs":
        d = next(e for e in lines if e["kind"] == "step")["dinobot"]
        assert d["n_selected"] is not None and "fit_residual_mm" in d
    if case == "artvs_no_refs":
        blocks = [e["artvs"] for e in lines if e["kind"] == "step"]
        assert blocks[0]["source"] == "langsam" and blocks[0]["box"] is not None
        assert blocks[0]["enc"] == [224, 14] and blocks[-1]["mode"] == "tiled"
        assert sum(blocks[-1]["tile_counts"]) == len(
            [e for e in lines if e["kind"] == "step"][-1]["pairs"]["cu"])


def test_the_debug_log_needs_no_new_setting_to_leave_old_records_alone():
    """Without `episode.depth_true` the depth-true offer changes nothing in the record."""
    cfg = tr.make_config(max_iter=30)
    rec, *_ = _fly(_truth_arm(), cfg, pert=_RefDepthPerturbation())
    assert not DEPTH_TRUE_KEYS & set(rec) and "debug_log_s" not in rec
    for st in rec["stops"].values():
        assert "translation_mm_depth_true" not in st


# ================================================================ G3 / G4
def test_truth_on_the_mock_truth_matcher_is_all_right(tmp_path):
    rec, *_ = _fly(_truth_arm(0.0), tr.make_config(max_iter=20), debug_dir=tmp_path / "ep")
    steps = [e for e in read_telemetry(tmp_path / "ep" / "telemetry.jsonl.gz")
             if e["kind"] == "step"]
    for s in steps:
        assert s["truth"]["stat"] and set(s["truth"]["stat"]) == {"R"}, s["truth"]["stat"]
        assert max(s["truth"]["err_px"]) < 0.5
        assert s["right"]["R"] == len(s["pairs"]["cu"])
        assert s["cell_px"] == pytest.approx([7.5, 135 / 32], abs=1e-3)


def test_a_biased_match_is_wrong_and_its_error_is_the_bias(tmp_path):
    from vsbench.vs.scene import MockPlaneScene

    cfg = tr.make_config(max_iter=3)
    scene = MockPlaneScene(render_scale=0.125)
    arm = tr.make_arm(scene, cfg, pixel_noise=40.0)
    with DebugLog(tmp_path / "ep") as dlog:
        runner.run_episode(scene, arm, tr.make_start(scene), cfg, debug_log=dlog)
    m = arm.matcher
    bias_by_lid = {int(gp[1] * m.side + gp[0]): np.linalg.norm(b)
                   for gp, b in zip(m._goal_patch, m._bias, strict=True)}
    s = next(e for e in read_telemetry(tmp_path / "ep" / "telemetry.jsonl.gz")
             if e["kind"] == "step")
    cell = s["cell_px"]
    n_w = 0
    for k, lid in enumerate(s["pairs"]["lid"]):
        err, letter = s["truth"]["err_px"][k], s["truth"]["stat"][k]
        if letter == "O":
            continue
        assert err == pytest.approx(bias_by_lid[lid], abs=0.5)
        du = s["pairs"]["cu"][k] - s["truth"]["tu"][k]
        dv = s["pairs"]["cv"][k] - s["truth"]["tv"][k]
        in_cells = np.hypot(du / cell[0], dv / cell[1])
        assert letter == ("R" if in_cells <= TRUTH_RIGHT_CELLS else "W")
        n_w += letter == "W"
    assert n_w >= len(s["pairs"]["lid"]) // 2          # 40 px against a 7.5 x 4.2 px cell


# ================================================================ G5 / G6
def test_the_scaled_truth_follows_the_size_matched_copy(tmp_path):
    from vsbench.vs.scene import MockPlaneScene

    s_, c = 0.8, (0.02, -0.01)
    cfg = tr.make_config(max_iter=4)
    scene = MockPlaneScene(render_scale=0.125)
    with DebugLog(tmp_path / "ep") as dlog:
        runner.run_episode(scene, tr.make_arm(scene, cfg), tr.make_start(scene), cfg,
                           perturbation=_ScaledPlate(s_, c), debug_log=dlog)
    assert scene.live_scale is None                        # the world is clean again
    lines = read_telemetry(tmp_path / "ep" / "telemetry.jsonl.gz")
    assert lines[0]["truth"]["object"] == "scaled"
    s = next(e for e in lines if e["kind"] == "step")
    live = Pose(position=s["pose"]["p"], rotation=np.reshape(s["pose"]["R"], (3, 3)))
    pg = np.column_stack([s["pairs"]["gu"], s["pairs"]["gv"]])
    X, _ok = scene.backproject(pg, scene.goal_pose)
    P = np.array([c[0], c[1], 0.0])
    want, _v = scene.project(P + s_ * (X - P), live)
    stat = s["truth"]["stat"]
    got = np.column_stack([s["truth"]["tu"], s["truth"]["tv"]]).astype(float)
    judged = np.array([ch in "RW" for ch in stat])
    assert judged.sum() >= 10
    np.testing.assert_allclose(got[judged], want[judged], atol=0.5)
    # the plain truth matcher answers the UNSCALED plate: wrong away from the centre
    assert "W" in stat
    centre_px, _ = scene.project(P[None], live)
    dist = np.linalg.norm(np.column_stack([s["pairs"]["cu"], s["pairs"]["cv"]])
                          - centre_px, axis=1)
    rights = [d for d, ch in zip(dist, stat, strict=True) if ch == "R"]
    wrongs = [d for d, ch in zip(dist, stat, strict=True) if ch == "W"]
    assert not rights or np.mean(wrongs) > np.mean(rights)


def test_another_object_has_no_truth_and_a_retextured_floor_is_grey(tmp_path):
    cfg = tr.make_config(max_iter=2)
    _fly(_truth_arm(), cfg, pert=_TruthOnly({"object": "none", "floor": "same",
                                             "floor_z": None}), debug_dir=tmp_path / "a")
    s = next(e for e in read_telemetry(tmp_path / "a" / "telemetry.jsonl.gz")
             if e["kind"] == "step")
    assert set(s["truth"]["stat"]) == {"N"} and all(t is None for t in s["truth"]["tu"])
    assert s["right"]["N"] == len(s["pairs"]["cu"])
    # the mock's plate lies ON the floor plane: with a re-textured floor every point is F,
    # its true pixel still logged
    _fly(_truth_arm(), cfg, pert=_TruthOnly({"object": "rigid", "floor": "retextured",
                                             "floor_z": 0.0}), debug_dir=tmp_path / "b")
    s = next(e for e in read_telemetry(tmp_path / "b" / "telemetry.jsonl.gz")
             if e["kind"] == "step")
    assert set(s["truth"]["stat"]) == {"F"} and all(t is not None for t in s["truth"]["tu"])


# ================================================================ G7
def test_gt_correspondence_mapped_without_a_map_is_gt_correspondence():
    from vsbench.vs.labels import gt_correspondence, gt_correspondence_mapped

    rng = np.random.default_rng(3)
    K = np.array([[200.0, 0, 120], [0, 200.0, 67], [0, 0, 1]])
    depth_a = rng.uniform(0.3, 0.6, (135, 240))
    depth_b = rng.uniform(0.3, 0.6, (135, 240))
    pts = rng.uniform([0, 0], [239, 134], (50, 2))
    Ta, Tb = np.eye(4), np.eye(4)
    Tb[:3, :3] = rodrigues([0.1, 0.9, 0.2], 0.05)
    Tb[:3, 3] = [0.01, -0.02, 0.03]
    for db_ in (None, depth_b):
        a = gt_correspondence(pts, depth_a, K, Ta, Tb, depth_b=db_)
        b = gt_correspondence_mapped(pts, depth_a, K, Ta, Tb, depth_b=db_)
        for x, y in zip(a, b[:3], strict=True):
            assert np.array_equal(x, y)
    # a map that moves nothing gives the same pixels (to rounding)
    same = gt_correspondence_mapped(pts, depth_a, K, Ta, Tb, world_map=lambda X: X)
    np.testing.assert_allclose(same[0], a[0], atol=1e-9)


# ================================================================ G8
def test_the_depth_true_pose_hand_example():
    from vsbench.vs.scenes import load_scenes

    goal = Pose(position=[0.012, -0.004, 0.36879742251143827],
                rotation=np.diag([1.0, -1.0, -1.0]))
    dt = perturb.depth_true_pose(goal, 0.1162378544769281, 0.08608741549462048)
    assert dt.position[2] == pytest.approx(0.338647, abs=5e-7)          # the owner's 0.3387
    np.testing.assert_array_equal(dt.position[:2], goal.position[:2])
    np.testing.assert_array_equal(dt.rotation, goal.rotation)
    np.testing.assert_array_equal(perturb.depth_true_pose(goal, 0.1, 0.1).position,
                                  goal.position)
    tp = _load("tests/fast/test_perturb.py", "_dbg_test_perturb")
    specs, _ = load_scenes()
    spec = specs["inst_shoe"]
    small = spec.live_item("same_small")["size_m"]
    w = tp._BoxWorld()
    p = perturb.build("swap:@same_small", seed=3, spec=spec)
    draw = tp._begin(p, w)
    assert set(draw) == {"mode", "seed", "live_name", "goal_instance", "live_instance",
                         "goal_size_m", "live_size_m", "swapped", "size_ratio", "goal_bbox",
                         "live_bbox", "floor_centre_xy", "reference_position",
                         "live_placement", "prim_paths"}                 # the draw as before
    z = tp._goal().position[2] + (small / 3.0 - 0.12)         # _BoxWorld's tops
    assert p.depth_true_pose().position[2] == pytest.approx(z)
    tm = p.truth_model()
    assert tm["object"] == "scaled" and tm["s"] == pytest.approx(small / 0.30)
    assert tm["centre_xy"] == pytest.approx([0.05, -0.02])
    p.end(w)
    chain = perturb.build("swap:@same_small+ground:concrete", seed=3, spec=spec)
    tp._begin(chain, tp._BoxWorld())
    assert chain.depth_true_pose().position[2] == pytest.approx(z)
    assert chain.truth_model()["object"] == "scaled"
    assert chain.truth_model()["floor"] == "retextured"
    same = perturb.build("swap:@same", seed=3, spec=spec)
    tp._begin(same, tp._BoxWorld())
    np.testing.assert_allclose(same.depth_true_pose().position, tp._goal().position)
    assert same.truth_model()["object"] == "rigid"
    other = perturb.build("swap:@other_small", seed=3, spec=spec)
    tp._begin(other, tp._BoxWorld())
    assert other.truth_model()["object"] == "none"
    assert perturb.build("none").depth_true_pose() is None
    assert perturb.build("none").truth_model() == perturb.DEFAULT_TRUTH


# ================================================================ G10
def _cb():
    sys.path.insert(0, str(REPO / "scripts"))
    return _load("scripts/check_baselines.py", "_dbg_check_baselines")


@pytest.fixture(scope="module")
def debug_episode(tmp_path_factory):
    """One episode through the check script's debug path, with a stop that fires."""
    pytest.importorskip("cv2")
    from vsbench.vs.scene import MockPlaneScene

    cb = _cb()
    out = tmp_path_factory.mktemp("dbg") / "films" / "mock_plane_oracle_t0"
    scene = MockPlaneScene(render_scale=0.125)
    cfg = tr.make_config(max_iter=240)
    arm = tr.make_arm(scene, cfg, pixel_noise=0.5)
    entry = cb.episode_check(scene, arm, cfg, tr.make_start(scene), 240, film_dir=out,
                             perturbation=_ScaledPlate(0.9, (0.0, 0.0)), arm_id="oracle_truth",
                             debug_dir=out, clean_film=True)
    return out, entry


def test_check_baselines_debug_path_writes_the_episode_directory(debug_episode, monkeypatch,
                                                                 tmp_path):
    import cv2

    out, entry = debug_episode
    for f in ("telemetry.jsonl.gz", "goal.jpg", "goal_mask.png", "clean.mp4", "clean_meta.json",
              "episode.mp4", "frames_meta.json"):
        assert (out / f).is_file(), f
    assert not list(out.glob("*.jpg")) or [p.name for p in out.glob("*.jpg")] == ["goal.jpg"]
    assert entry["telemetry_file"] == "films/mock_plane_oracle_t0/telemetry.jsonl.gz"
    assert {"final_translation_mm_depth_true", "success_depth_true", "depth_true_pose",
            "debug_log_s"} <= set(entry)
    cap = cv2.VideoCapture(str(out / "clean.mp4"))
    n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        shape = img.shape
        n += 1
    assert n == entry["steps"] + 1 and shape[:2] == (68, 120)
    meta = json.loads((out / "frames_meta.json").read_text())
    assert meta[-1]["kind"] == "final" and meta[-1]["step"] == entry["steps"] + 1
    assert {"file", "step", "t_mm", "r_deg", "n_matches", "kind", "pose", "t_mm_matched",
            "t_mm_depth_true", "landing"} <= set(meta[0])
    assert any(m["landing"] for m in meta)            # the scored stop fired and is marked
    goal = cv2.imread(str(out / "goal.jpg"))
    assert goal.shape[:2] == (68, 120)
    # --clean-film without --telemetry is refused before anything starts
    cb = _cb()
    monkeypatch.setattr(sys, "argv", ["check_baselines.py", "--out", str(tmp_path),
                                      "--clean-film"])
    with pytest.raises(SystemExit, match="needs --telemetry"):
        cb.main()


# ================================================================ G11
def test_the_error_split():
    g = np.diag([1.0, -1.0, -1.0])
    ang = orientation_split(g @ rodrigues([0, 0, 1.0], np.radians(30.0)), g)
    assert ang["roll_deg"] == pytest.approx(30.0) and ang["tilt_deg"] == pytest.approx(0.0)
    ang = orientation_split(g @ rodrigues([0, 1.0, 0], np.radians(10.0)), g)
    assert ang["tilt_deg"] == pytest.approx(10.0) and ang["tilt_dir_deg"] == pytest.approx(0.0)
    assert ang["roll_deg"] == pytest.approx(0.0, abs=1e-6)
    for deg in (179.0, -179.0):
        a = orientation_split(g @ rodrigues([0, 0, 1.0], np.radians(deg)), g)
        assert a["roll_deg"] == pytest.approx(deg)
    p0 = np.array([0.1, 0.2, 0.5])
    along = error_split(p0 + g @ [0, 0, 0.01], p0, g)       # 10 mm along the viewing axis
    assert along["e_mm"] == pytest.approx([0.0, 0.0, 10.0]) and along["t_mm"] == 10.0
    side = error_split(p0 + g @ [0.003, -0.004, 0], p0, g)
    assert side["e_mm"] == pytest.approx([3.0, -4.0, 0.0]) and side["t_mm"] == 5.0


# ================================================================ G12
def test_the_cell_sizes():
    ns = types.SimpleNamespace
    vit = ns(side=32, image_wh=(1920, 1080), extras={})
    assert cell_px(vit, (1920, 1080)) == pytest.approx([60.0, 33.75])
    dino = ns(side=1920, image_wh=(1920, 1080), extras={"load_hw": (224, 398)})
    assert cell_px(dino, (1920, 1080), dinobot_stride=4) == pytest.approx(
        [4 * 1920 / 398, 4 * 1080 / 224])
    assert cell_px(dino, (1920, 1080), dinobot_stride=4)[0] == pytest.approx(19.3, abs=0.01)
    art = {"enc_input": 518, "enc_patch": 14, "live_box": [800, 200, 1318, 718],
           "roi_source": "lighttrack", "grid": [3, 2]}
    a = 14 / 518
    assert cell_px(ns(side=1920, image_wh=(1920, 1080), extras={**art, "mode": "coarse"}),
                   (1920, 1080)) == pytest.approx([a * 518, a * 518])
    assert cell_px(ns(side=1920, image_wh=(1920, 1080), extras={**art, "mode": "tiled"}),
                   (1920, 1080)) == pytest.approx([a * 518 / 3, a * 518 / 2])
    assert cell_px(ns(side=1920, image_wh=(1920, 1080),
                      extras={**art, "mode": "full", "live_box": None}),
                   (1920, 1080)) == pytest.approx([a * 1920, a * 1080])
    assert cell_px(ns(side=1920, image_wh=(1920, 1080), extras={}), (1920, 1080)) is None
    assert cell_px(None, (1920, 1080)) is None


# ================================================================ G14
def test_the_dinobot_residual():
    from vsbench.vs.baselines import dinobot as db
    from vsbench.vs.scene import D435I

    rng = np.random.default_rng(5)
    X = np.column_stack([rng.uniform(-0.1, 0.1, 8), rng.uniform(-0.05, 0.05, 8),
                         rng.uniform(0.3, 0.4, 8)])
    R = rodrigues([0.2, 1.0, 0.1], 0.1)
    t = np.array([0.01, -0.02, 0.005])
    Y = (R @ X.T).T + t

    def px(P):
        return np.column_stack([D435I.fx * P[:, 0] / P[:, 2] + D435I.cx,
                                D435I.fy * P[:, 1] / P[:, 2] + D435I.cy])

    debug = {"pts_goal": px(X), "pts_cur": px(Y), "goal_depth_values": X[:, 2],
             "depth_values": Y[:, 2], "n_selected": 8}
    out = dinobot_block(debug, D435I)
    assert out["fit_residual_mm"] == pytest.approx(0.0, abs=1e-6)
    Yn = Y + rng.normal(0, 0.002, Y.shape)
    debug.update(pts_cur=px(Yn), depth_values=Yn[:, 2])
    out = dinobot_block(debug, D435I)
    Rh, th = db.kabsch(X, Yn)
    r = np.linalg.norm((Rh @ X.T).T + th - Yn, axis=1) * 1000.0
    assert out["fit_residual_mm"] == pytest.approx(np.sqrt(np.mean(r ** 2)), abs=1e-3)
    assert out["fit_residual_max_mm"] == pytest.approx(r.max(), abs=1e-3)
    assert 0.5 < out["fit_residual_mm"] < 5.0
    assert dinobot_block({"pts_goal": np.zeros((2, 2))}, D435I)["fit_residual_mm"] is None


# ================================================================ G16
def test_the_renderer_writes_a_still_and_a_video(debug_episode, tmp_path):
    import cv2

    out, _entry = debug_episode
    view = _load("scripts/debug_view.py", "_dbg_debug_view")
    assert view.main([str(out), "--still", "landing", "--every", "4"]) == 0
    stills = list(out.glob("debug_*.png"))
    assert len(stills) == 1
    img = cv2.imread(str(stills[0]))
    assert img.shape == (1000, 1600, 3) and img.shape[1] >= 1280
    assert (out / "debug.mp4").is_file() and (out / "debug.mp4").stat().st_size > 0
    assert not (out / "_debug_frames").exists()
    assert view.main([str(out), "--still", "final", "--no-video"]) == 0
    assert (out / "debug_final.png").is_file()


# ================================================================ review 2026-09-30
class _FailingLog(DebugLog):
    """A recorder whose step() raises at step 3 (a full disk, an odd shape ...)."""

    def _step(self, **kw):
        if kw["step_i"] == 3:
            raise OSError("no space left on device (test)")
        return super()._step(**kw)


def test_a_failing_recorder_cannot_change_the_flight(tmp_path):
    """The recorder's failure is caught inside it: the same flight, the same record (but
    the new keys and debug_log_error), an `error` line, and the log goes quiet."""
    cfg = tr.make_config(max_iter=60)
    rec_off, tel_off, poses_off, _ = _fly(_truth_arm(2.0), cfg, pert=_RefDepthPerturbation())
    from vsbench.vs.scene import MockPlaneScene

    scene = MockPlaneScene(render_scale=0.125)
    poses, telem = [], []
    with _FailingLog(tmp_path / "ep", arm_id="test_arm") as dlog:
        rec_on = runner.run_episode(
            scene, _truth_arm(2.0)(scene, cfg), tr.make_start(scene), cfg, telemetry=telem,
            on_step=lambda i, rgb, dbg, pose: poses.append((pose.position.copy(),
                                                           pose.rotation.copy())),
            perturbation=_RefDepthPerturbation(), debug_log=dlog)
    assert dlog.broken and "no space left" in dlog.error
    assert rec_on["debug_log_error"].startswith("step: OSError")
    for (p0, r0), (p1, r1) in zip(poses_off, poses, strict=True):
        assert np.array_equal(p0, p1) and np.array_equal(r0, r1)
    assert _same(tel_off, telem)
    assert set(rec_on) - set(rec_off) == {"debug_log_s", "debug_log_error"}
    for k in set(rec_off) - TIMING:
        assert _same(rec_off[k], rec_on[k]), k
    lines = read_telemetry(tmp_path / "ep" / "telemetry.jsonl.gz")
    assert [e["kind"] for e in lines] == ["meta", "step", "step", "error"]
    assert lines[-1]["step"] == 3 and lines[-1]["where"] == "step"


def test_episode_check_debug_path_is_additive(tmp_path):
    """At the check script's level (the station path): the old call and the debug call on
    the same flight return the same numbers but the new keys, and the film's old five
    frame keys are identical (no double rounding of t_mm / r_deg)."""
    pytest.importorskip("cv2")
    from vsbench.vs.scene import MockPlaneScene

    cb = _cb()

    def one(debug: bool):
        scene = MockPlaneScene(render_scale=0.125)
        cfg = tr.make_config(max_iter=120)
        d = tmp_path / ("on" if debug else "off")
        out = cb.episode_check(scene, tr.make_arm(scene, cfg, pixel_noise=0.5), cfg,
                               tr.make_start(scene), 120, film_dir=d,
                               perturbation=_ScaledPlate(0.9, (0.0, 0.0)), arm_id="oracle",
                               **({"debug_dir": d, "clean_film": True} if debug else {}))
        return out, json.loads((d / "frames_meta.json").read_text())

    off, meta_off = one(False)
    on, meta_on = one(True)
    new = {"start_translation_mm_depth_true", "final_translation_mm_depth_true",
           "stop_translation_mm_depth_true", "success_depth_true", "depth_true_pose",
           "debug_log_s", "telemetry_file"}
    assert set(on) - set(off) <= new | {"own_stop_translation_mm_depth_true"}
    assert set(off) <= set(on)
    for k in set(off) - TIMING:
        if k == "stops":
            for name, st in off["stops"].items():
                assert _same(st, {a: b for a, b in on["stops"][name].items()
                                  if a != "translation_mm_depth_true"}), name
            continue
        assert _same(off[k], on[k]), k
    old = ("file", "step", "t_mm", "r_deg", "n_matches")
    steps_on = [m for m in meta_on if m.get("kind") == "step"]
    assert len(steps_on) == len(meta_off) == off["steps"]
    for a, b in zip(meta_off, steps_on, strict=True):
        assert {k: a[k] for k in old} == {k: b[k] for k in old}
    # the landing key is what the film marks; every fired rule is in fired_now
    for m in steps_on:
        assert all(lab.startswith(("the scored stop", "oracle's own stop"))
                   for lab in m["landing"])
        if m["landing"]:
            assert m["hold_s"] == 1.0 and m["fired_now"]
    assert meta_on[-1]["kind"] == "final" and meta_on[-1]["step"] == on["steps"] + 1
    assert meta_on[-1]["hold_s"] == 2.0


def test_the_common_yardstick_counts_the_near_pairs(tmp_path):
    from vsbench.vs.scene import MockPlaneScene

    cfg = tr.make_config(max_iter=2)
    scene = MockPlaneScene(render_scale=0.125)
    with DebugLog(tmp_path / "ep") as dlog:
        runner.run_episode(scene, tr.make_arm(scene, cfg, pixel_noise=15.0),
                           tr.make_start(scene), cfg, debug_log=dlog)
    s = next(e for e in read_telemetry(tmp_path / "ep" / "telemetry.jsonl.gz")
             if e["kind"] == "step")
    judged = [k for k, ch in enumerate(s["truth"]["stat"]) if ch in "RW?"]
    errs = [s["truth"]["err_px"][k] for k in judged]
    assert s["near"]["px"] == TRUTH_COMMON_PX and s["near"]["n"] == len(judged) > 0
    assert s["near"]["k"] == sum(e <= TRUTH_COMMON_PX for e in errs)
    assert s["near"]["median_px"] == pytest.approx(np.median(errs), abs=0.02)
    # the miss in mm: err_px * z / f at the true point's depth
    k = judged[0]
    z = s["near"]["z_med_m"]
    assert s["truth"]["err_mm"][k] == pytest.approx(
        s["truth"]["err_px"][k] * z / scene.intrinsics.fx * 1000.0, rel=0.05)
    assert s["cell_mm"] == pytest.approx([c * z / scene.intrinsics.fx * 1000.0
                                          for c in s["cell_px"]], rel=0.02)


def test_the_film_numbers_its_final_frame_after_the_last_step(tmp_path):
    pytest.importorskip("cv2")
    from vsbench.vs.bench import films

    rgb = np.zeros((40, 64, 3), np.uint8)
    w = films.FilmWriter(tmp_path / "f", rgb, every=5)
    for i in range(1, 14):
        w.step(i, rgb, {}, 1.0, 1.0, refs={"goal": 1.0})
    w.final(rgb, refs={"goal": 1.0}, r_deg=1.0, after_step=13,
            what="final pose (no move after step 13: diverged_feature)")
    assert [m["step"] for m in w.meta] == [1, 5, 10, 14]
    c = films.CleanFilmWriter(tmp_path / "c")
    (tmp_path / "c" / "clean_00999.jpg").write_bytes(b"stale")
    c2 = films.CleanFilmWriter(tmp_path / "c")          # a re-fly into the same directory
    assert not (tmp_path / "c" / "clean_00999.jpg").exists()
    c2.step(1, rgb)
    c2.final(rgb, after_step=7)
    assert [m["step"] for m in c2.meta] == [1, 8]
    del c


# ---------------------------------------------------------- the viewer, probed
def _synthetic_episode(d: Path, *, display="ViT-VS", live="same_small", landing_step=9,
                       scale=0.74) -> Path:
    """A hand-written debug log: 9 steps, one right pair and one wrong pair at known
    pixels, the scored stop at ``landing_step``; no clean film (the live panel is grey)."""
    import gzip

    import cv2

    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "goal.jpg"), np.zeros((450, 800, 3), np.uint8))
    to = {"goal": {"t_mm": 12.0, "e_mm": [5.0, -3.0, 2.0]},
          "size_matched": {"t_mm": 9.0, "e_mm": [5.0, -3.0, 1.0]},
          "depth_true": {"t_mm": 7.0, "e_mm": [5.0, -3.0, -1.0]}}
    step = {"kind": "step", "pose": {"p": [0, 0, 0.3], "R": [1, 0, 0, 0, -1, 0, 0, 0, -1]},
            "to": to, "r_deg": 2.0, "tilt_deg": 1.5, "tilt_dir_deg": 0.0, "roll_deg": -1.0,
            "n": 2, "pairs": {"gu": [100.0, 700.0], "gv": [100.0, 700.0],
                              "cu": [200.0, 600.0], "cv": [300.0, 200.0],
                              "score": [1.0, 1.0], "w": None, "lid": None},
            "cell_px": [60.0, 33.75], "cell_mm": [15.0, 8.4],
            "truth": {"tu": [200.0, 900.0], "tv": [300.0, 200.0], "err_px": [0.0, 300.0],
                      "err_mm": [0.0, 70.0], "err_cell": [0.0, 5.0], "stat": "RW"},
            "right": {"R": 1, "W": 1, "N": 0, "F": 0, "O": 0, "?": 0},
            "near": {"px": 20.0, "k": 1, "n": 2, "median_px": 150.0, "median_mm": 35.0,
                     "z_med_m": 0.33},
            "v_cmd": [0.001, 0, 0, 0, 0, 0.01], "end": None}
    meta = {"kind": "meta", "display": display,
            "scene": "inst_shoe_with_a_very_long_scene_name", "live_name": live,
            "start_id": "simple5_s_004_with_a_long_id", "max_iter": 10, "image_wh": [1600, 900],
            "truth": {"object": "scaled", "s": scale}, "stops": {"scored": "cns_plateau",
                                                                "own": [], "recorded": []}}
    lines = [meta] + [{**step, "i": i} for i in range(1, 10)]
    lines += [{"kind": "final", "after_step": 9, "to": to, "r_deg": 2.0, "moved": True},
              {"kind": "end", "stops": {"cns_plateau": {"step": landing_step},
                                        "velocity_quiet": {"step": 7}}}]
    with gzip.open(d / "telemetry.jsonl.gz", "wt") as f:
        for e in lines:
            f.write(json.dumps(e) + "\n")
    return d


def _near(img, xy, bgr, tol=60) -> bool:
    px = img[int(xy[1]), int(xy[0])].astype(int)
    return int(np.abs(px - np.array(bgr)).max()) <= tol


def test_the_viewer_draws_every_panel_where_it_should(tmp_path):
    pytest.importorskip("cv2")
    from vsbench.vs.bench import debugview as dv

    ep = dv.Episode(_synthetic_episode(tmp_path / "ep"))
    base = dv.draw_timeline_base(ep)
    land = dv.compose(ep, ep.step_by_i(9), kind="step", timeline_base=base)
    plain = dv.compose(ep, ep.step_by_i(1), kind="step", timeline_base=base)
    assert land.shape == (dv.CANVAS_H, dv.CANVAS_W, 3)
    top = dv.HEADER_H
    # the rings: right pair live at (200, 300) px -> panel (100, 150); wrong at (600, 200)
    ring = 9 + 3 + dv.RING_THICK // 2                    # score 1.0 -> radius 9
    assert _near(plain, (dv.PANEL_W + 100 - ring, top + 150), dv.RIGHT_BGR)
    assert _near(plain, (dv.PANEL_W + 300 - ring, top + 100), dv.WRONG_BGR)
    # the wrong pair's true spot (900, 200) -> (450, 100): its x and the miss line to it
    assert _near(plain, (dv.PANEL_W + 450, top + 100), dv.WRONG_BGR)
    assert _near(plain, (dv.PANEL_W + 380, top + 100), dv.WRONG_BGR)
    # the landing: a yellow border and the LANDED badge on the landing frame only
    border = (dv.CANVAS_W - 4, top + 200)
    assert _near(land, border, dv.YELLOW) and not _near(plain, border, dv.YELLOW)
    badge = (dv.PANEL_W + 20, top + dv.PANEL_H - 20)
    assert _near(land, badge, dv.YELLOW) and not _near(plain, badge, dv.YELLOW)
    # the stop marker in the timeline at step 9 (probed on the step-1 frame: no cursor)
    x9 = dv._X(ep, 9)
    y = dv.BOTTOM_Y + (dv.STRIP_B[1] + dv.STRIP_C[0]) // 2
    assert _near(plain, (x9, y), dv.STOP_BGR["scored"])
    # the error split's first bar (+5 mm to the right) is drawn
    bar = plain[dv.BOTTOM_Y + 70, dv.TIMELINE_W + 312: dv.TIMELINE_W + 330]
    assert (np.abs(bar.astype(int) - np.array((80, 170, 255))).max(axis=1) < 40).all()
    # the pairs' colours are never the verdicts' pure red or green
    for b, g, r in dv.PALETTE:
        assert not (g > 150 and r < 110 and b < 110) and not (r > 150 and g < 110 and b < 110)
    ep.close()


def test_the_header_fits_even_with_long_names_and_two_landings(tmp_path):
    pytest.importorskip("cv2")
    from vsbench.vs.bench import debugview as dv

    d = _synthetic_episode(tmp_path / "ep", display="DINOBot (a much longer method name)",
                           live="other_small_with_a_long_name")
    # the own stop and the scored stop on the same step
    lines = read_telemetry(d / "telemetry.jsonl.gz")
    lines[0]["stops"]["own"] = ["dinobot_err"]
    lines[-1]["stops"]["dinobot_err"] = {"step": 9}
    import gzip

    with gzip.open(d / "telemetry.jsonl.gz", "wt") as f:
        for e in lines:
            f.write(json.dumps(e) + "\n")
    ep = dv.Episode(d)
    img = dv.compose(ep, ep.step_by_i(9), kind="step", timeline_base=dv.draw_timeline_base(ep))
    assert ep.landing_marks(9) == ["own stop", "scored stop"]
    # nothing is drawn in the header's last 10 columns: every line fits
    assert not img[:dv.HEADER_H, dv.CANVAS_W - 10:].any()
    title = dv.header_title(ep, 9, "step")
    assert dv._text_width(title, dv._fit(title, 0.8, 2, dv.CANVAS_W - 28)) <= dv.CANVAS_W - 28
    ep.close()


def test_the_viewer_writes_into_out_when_the_flight_directory_is_read_only(debug_episode,
                                                                         tmp_path):
    import os
    import shutil

    ep = tmp_path / "ro" / "films" / debug_episode[0].name
    shutil.copytree(debug_episode[0], ep, ignore=shutil.ignore_patterns("debug*"))
    before = sorted(p.name for p in ep.iterdir())
    os.chmod(ep, 0o555)
    try:
        view = _load("scripts/debug_view.py", "_dbg_debug_view")
        out = tmp_path / "view"
        assert view.main([str(ep), "--still", "landing", "--every", "8", "--out", str(out)]) == 0
        assert len(list(out.glob("debug_*.png"))) == 1
        assert (out / f"{ep.name}_debug.mp4").is_file()
        assert sorted(p.name for p in ep.iterdir()) == before
        # the pair overview: the landing still of every episode and an index
        assert view.main(["--pair", str(ep.parent.parent), "--out", str(tmp_path / "pair")]) == 0
        page = (tmp_path / "pair" / "index.html").read_text()
        assert "within 20 px" in page and f"{ep.name}.png" in page
        rows = json.loads((tmp_path / "pair" / "index.json").read_text())
        assert rows[0]["landing"] in ("scored stop", "own stop", "final pose")
        assert (tmp_path / "pair" / f"{ep.name}.png").is_file()
    finally:
        os.chmod(ep, 0o755)


# ================================================================ G17 / G18
def test_the_debug_refly_plan_is_24_pairs():
    plan = _load("scripts/chains/debug_refly_plan.py", "_dbg_refly_plan").plan()
    kinds = ("shoe", "bottle", "teapot", "drill", "mug", "hammer")
    want = [(f"inst_{k}", n) for k in kinds for n in ("same", "same_small", "other_small",
                                                      "mine1")]
    assert plan == want and len(set(plan)) == 24


FAKE_PY = r"""#!/bin/bash
# a stand-in interpreter for the debug chain: the pin check, the REAL plan, check_baselines
if [ "$1" = "-" ]; then
  cat > /dev/null
  echo "[fake] pins ok"; exit 0
fi
case "$1" in
  scripts/chains/debug_refly_plan.py) exec "$REAL_PY" "$@" ;;
  scripts/check_baselines.py)
    IFS= read -r _swallowed || true        # Isaac's kit reads its stdin
    echo "$*" >> "$FAKE_CALLS"
    shift
    while [ $# -gt 0 ]; do
      case "$1" in --out) out=$2; shift 2 ;; *) shift ;; esac
    done
    mkdir -p "$out"
    echo '{"arms": {"a": {"label": "ok"}}}' > "$out/check.json"
    exit 0 ;;
esac
echo "unexpected call: $*" >&2; exit 9
"""


def test_the_debug_refly_chain_flies_each_pair_once_with_the_debug_flags(tmp_path):
    import os
    import subprocess

    fake = tmp_path / "fake_py.sh"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    out = tmp_path / "out"
    calls = tmp_path / "calls.txt"
    env = {**os.environ, "VSSIM_REPO": str(REPO), "VSSIM_CHAIN_PY": str(fake), "OUT": str(out),
           "FAKE_CALLS": str(calls), "REAL_PY": sys.executable}

    def launch():
        return subprocess.run(["bash", str(REPO / "scripts" / "chains" / "debug_refly.sh")],
                              env=env, capture_output=True, text=True, timeout=120).returncode

    assert launch() == 0
    lines = calls.read_text().splitlines()
    assert len(lines) == 24
    plan = _load("scripts/chains/debug_refly_plan.py", "_dbg_refly_plan").plan()
    for (scene, live), line in zip(plan, lines, strict=True):
        for token in (f"--scenes {scene}", "--arms dinobot_objmask vitvs_dinov3_objmask "
                      "artvs_roi_scene", "--near-steps 200", "--budget dinobot_objmask=30",
                      "--far 5 --far-set simple5", "--film", "--telemetry", "--clean-film",
                      "--outline-iou", f"--perturb swap:@{live}+ground:concrete",
                      f"--out {out}/{scene}_{live}"):
            assert token in line, (token, line)
    assert all((out / f"{s}_{n}" / "DONE").is_file() for s, n in plan)
    assert "debug re-fly done" in (out / "debug_refly.log").read_text()
    assert launch() == 0                                      # resumable: nothing flies again
    assert len(calls.read_text().splitlines()) == 24


# ================================================================ G19
def test_display_names():
    assert display_name("dinobot_objmask") == "DINOBot"
    assert display_name("vitvs_dinov3_objmask") == "ViT-VS"
    assert display_name("artvs_roi_scene") == "ART-VS"
    assert display_name("cns_v1") == "CNS (v1)" and display_name("sift") == "sift"


# ------------------------------------------- the debug chain's failure paths (review)
FAKE_PY_FAULTS = r"""#!/bin/bash
# a stand-in interpreter: pins only with FAKE_PINS_OK; same_small = an arm error inside,
# other_small = a non-zero exit, the rest clean; FAKE_MOVE_HEAD moves the checkout's HEAD
if [ "$1" = "-" ]; then
  cat > /dev/null
  [ -n "${FAKE_PINS_OK:-}" ] && { echo "[fake] pins ok"; exit 0; }
  echo "[fake] simple start files not pinned"; exit 1
fi
case "$1" in
  scripts/chains/debug_refly_plan.py) exec "$REAL_PY" "$@" ;;
  scripts/check_baselines.py)
    IFS= read -r _swallowed || true
    echo "$*" >> "$FAKE_CALLS"
    if [ -n "${FAKE_MOVE_HEAD:-}" ]; then
      git -C "$VSSIM_REPO" -c user.name=t -c user.email=t@t commit -q --allow-empty -m moved
    fi
    shift
    while [ $# -gt 0 ]; do
      case "$1" in --out) out=$2; shift 2 ;; --perturb) pert=$2; shift 2 ;; *) shift ;; esac
    done
    mkdir -p "$out"
    case "$pert" in
      *@same_small+*) echo '{"arms": {"a": {"build_error": "boom"}}}' > "$out/check.json" ;;
      *@other_small+*) exit 3 ;;
      *) echo '{"arms": {"a": {"label": "ok"}}}' > "$out/check.json" ;;
    esac
    exit 0 ;;
esac
echo "unexpected call: $*" >&2; exit 9
"""


def _chain(tmp_path, *, repo=REPO, pins_ok=True, **env_extra):
    import os
    import subprocess

    fake = tmp_path / "fake_faults.sh"
    fake.write_text(FAKE_PY_FAULTS)
    fake.chmod(0o755)
    out = tmp_path / "out"
    calls = tmp_path / "calls.txt"
    env = {**os.environ, "VSSIM_REPO": str(repo), "VSSIM_CHAIN_PY": str(fake), "OUT": str(out),
           "FAKE_CALLS": str(calls), "REAL_PY": sys.executable, **env_extra}
    if pins_ok:
        env["FAKE_PINS_OK"] = "1"
    rc = subprocess.run(["bash", str(REPO / "scripts" / "chains" / "debug_refly.sh")],
                        env=env, capture_output=True, text=True, timeout=120).returncode
    log = (out / "debug_refly.log").read_text() if (out / "debug_refly.log").is_file() else ""
    return rc, out, log, (calls.read_text().splitlines() if calls.is_file() else [])


def test_the_debug_chain_refuses_without_the_pinned_start_files(tmp_path):
    rc, _out, log, calls = _chain(tmp_path, pins_ok=False)
    assert rc == 1 and "refused" in log and calls == []


def test_the_debug_chain_marks_done_only_the_clean_pairs(tmp_path):
    rc, out, log, calls = _chain(tmp_path)
    assert rc == 0 and len(calls) == 24
    for kind in ("shoe", "bottle", "teapot", "drill", "mug", "hammer"):
        assert (out / f"inst_{kind}_same" / "DONE").is_file()
        assert (out / f"inst_{kind}_mine1" / "DONE").is_file()
        assert not (out / f"inst_{kind}_same_small" / "DONE").exists()      # an arm error
        assert not (out / f"inst_{kind}_other_small" / "DONE").exists()     # rc = 3
    assert log.count("arm error - not DONE") == 6 and log.count("rc=3 - not DONE") == 6
    # provenance: the commit is in the log and in commit.txt
    import subprocess

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert f"at {head}" in log and head in (out / "commit.txt").read_text()
    # a re-launch flies only the 12 unfinished pairs
    _rc, _out, _log, calls = _chain(tmp_path)
    assert len(calls) == 24 + 12


def test_the_debug_chain_smoke_mode_is_one_pair_with_the_same_flags(tmp_path):
    rc, out, log, calls = _chain(tmp_path, SMOKE="1")
    assert rc == 0 and len(calls) == 1
    line = calls[0]
    for token in ("--scenes inst_shoe", "--near-steps 50", "--far 1 --far-set simple5",
                  "--budget dinobot_objmask=30", "--film --telemetry --clean-film --outline-iou",
                  "--perturb swap:@same_small+ground:concrete",
                  "--arms dinobot_objmask vitvs_dinov3_objmask artvs_roi_scene"):
        assert token in line, (token, line)
    assert "smoke=1" in log


def _tmp_checkout(tmp_path, *, telemetry: bool) -> Path:
    """A throwaway git checkout that looks like the repository to the chain."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / "scripts" / "chains").mkdir(parents=True)
    for name in ("vs", "configs", "benchmark"):
        (repo / name).symlink_to(REPO / name)
    for name in ("debug_refly_plan.py",):
        (repo / "scripts" / "chains" / name).symlink_to(REPO / "scripts" / "chains" / name)
    (repo / "scripts" / "check_baselines.py").write_text(
        "# stand-in\n" + ("# --telemetry\n" if telemetry else ""))
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "start"], check=True)
    return repo


def test_the_debug_chain_stops_when_the_checkout_changes(tmp_path):
    repo = _tmp_checkout(tmp_path, telemetry=True)
    rc, out, log, calls = _chain(tmp_path, repo=repo, FAKE_MOVE_HEAD="1")
    assert rc == 1 and len(calls) == 1                  # the first pair, then the stop
    assert "the checkout changed" in log and "debug re-fly stopped" in log
    assert (out / "inst_shoe_same" / "DONE").is_file()


def test_the_debug_chain_stops_when_check_baselines_has_no_telemetry(tmp_path):
    repo = _tmp_checkout(tmp_path, telemetry=False)
    rc, _out, log, calls = _chain(tmp_path, repo=repo)
    assert rc == 1 and calls == [] and "has no --telemetry" in log


def test_the_debug_chain_refuses_a_second_launch_on_the_same_output(tmp_path):
    import fcntl
    import shutil

    if shutil.which("flock") is None:
        pytest.skip("no flock on this machine")
    out = tmp_path / "out"
    out.mkdir()
    with open(out / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc, _out, log, calls = _chain(tmp_path)
    assert rc == 1 and calls == [] and "refusing" in log


def test_render_pair_survives_an_unreadable_episode(tmp_path):
    """Recheck finding: one episode without a meta line (the recorder failed in begin, or a kill
    before the first flush) must not cost the pair's index - it gets an 'unreadable' row."""
    import gzip
    import json as _json

    from vsbench.vs.bench import debugview

    bad = tmp_path / "films" / "scene_arm_s_00"
    bad.mkdir(parents=True)
    with gzip.open(bad / "telemetry.jsonl.gz", "wt") as fh:
        fh.write(_json.dumps({"kind": "error", "where": "begin", "error": "boom"}) + "\n")
    debugview.render_pair(tmp_path, out_dir=tmp_path / "view")
    rows = _json.loads((tmp_path / "view" / "index.json").read_text())
    assert len(rows) == 1 and rows[0]["landing"].startswith("unreadable")
    assert (tmp_path / "view" / "index.html").is_file()


def test_a_flight_read_at_its_limit_shows_its_last_step_with_matches(tmp_path):
    """Smoke finding (2026-09-30): ART-VS never stops by itself, so its landing still showed the
    frame after the last move, which carries no matches. The still and the index row must use
    the last logged step's matches instead (the distances stay those of the final pose)."""
    from vsbench.vs.bench import debugview

    ep_dir = tmp_path / "ep"
    _fly(_truth_arm(2.0), tr.make_config(max_iter=6, depth_true=True), pert=_RefDepthPerturbation(),
         debug_dir=ep_dir)
    ep = debugview.Episode(ep_dir)
    try:
        if ep.landing_step() is not None:
            import pytest
            pytest.skip("this mock episode stopped by itself")
        row = debugview._landing_summary(ep)
    finally:
        ep.close()
    assert row["landing"] == "final pose"
    assert row["right_own_cell"] != "-" or row["within_px"] != "-"
    png = debugview.render_still(ep_dir, "landing", out_dir=tmp_path / "v", name="x.png")
    assert png.name == "x.png" and png.is_file()
