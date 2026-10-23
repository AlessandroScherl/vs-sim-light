# vs-sim copy: tests/fast/test_cns.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The CNS (v1) arm, exercised with a stub pipeline: the x2, the dt, BGR, the stop.

Protocol v2 section 4 says CNS must run **as published** before it is quoted as the
bar, and the old arm had neither of its two published mechanisms. This file pins
both, plus the refusals, without a GPU, without the CNS checkout and without its
weights: a stub pipeline returns a scripted twist and a scripted correspondence
graph, so everything between the network's output and the record is measured.

``tests/slow/test_cns_arm.py`` runs the real pipeline where ``$CNS_ROOT`` exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from vsbench.vs.baselines import cns as cnsmod
from vsbench.vs.baselines.control import CNS_STOP, CnsPlateau, Intrinsics, StopObservation
from vsbench.vs.bench import arms, runner
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.scene import MockPlaneScene

INTR = Intrinsics(fx=1406.75, fy=1406.75, cx=960.0, cy=540.0, width=1920, height=1080)


#: CNS's own timing dict (`benchmark/pipeline.py`): frontend / midend / backend /
#: total, in seconds. There is no "detect+match" key - the first cut read one, so
#: `match_ms` was always 0.0 (review R14, C4).
CNS_TIMING = {"frontend_time": 0.004, "midend_time": 0.001,
              "backend_time": 0.002, "total_time": 0.007}


class _Graph:
    """The shape of CNS's ``GraphData``: normalised positions plus a node mask."""

    def __init__(self, pos_cur, pos_tar, node_mask):
        self.pos_cur = np.asarray(pos_cur, float)
        self.pos_tar = np.asarray(pos_tar, float)
        self.node_mask = np.asarray(node_mask, bool)
        self.num_nodes = int(len(self.node_mask))


class StubPipeline:
    """A scripted CNS pipeline: it records what it was handed and returns a script.

    ``twists`` is the per-call twist CNS's network would emit; ``graphs`` the
    per-call payload. Both cycle when exhausted, which is what a plateau needs.
    """

    def __init__(self, twists, graphs=None):
        self.twists = list(twists)
        self.graphs = list(graphs or [])
        self.calls: list[np.ndarray] = []
        self.target = None
        self.target_dist_scale = None

    def set_target(self, bgr, dist_scale=None, intrinsic=None):
        self.target = np.asarray(bgr)
        self.target_dist_scale = dist_scale

    def get_control_rate(self, bgr):
        self.calls.append(np.asarray(bgr))
        i = min(len(self.calls) - 1, len(self.twists) - 1)
        g = self.graphs[min(len(self.calls) - 1, len(self.graphs) - 1)] if self.graphs else None
        return np.asarray(self.twists[i], float), g, dict(CNS_TIMING)


def make_config(**over) -> dict:
    cfg = {
        "seed": 20260902,
        "matcher": {"variant": "cns_v1", "num_landmarks": 1},
        "controller": {"dt": 1.0 / 60.0},
        "stop": {"rule": "cns_plateau"},
        "episode": {"max_iter": 5},
        "scoring": {"success_rule": ["predicate_a"]},
        "cns": {"detector": "SIFT"},
    }
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfgmod.resolve(cfg)


def build(pipeline, **over):
    return cnsmod.build_cns_arm(make_config(**over), intrinsics=INTR, pipeline=pipeline)


# ------------------------------------------------------- the x2 and the dt
def test_the_published_step_gain_is_applied_and_rate_compensated():
    """``vel * 2`` at CNS's 50 Hz is ``vel * 2.4`` in our 60 Hz loop - and lambda*T = 24.

    ``benchmark.py:161`` + ``sim/environment.py:110-114``: the published
    per-iteration displacement is ``vel * 2 / 50``. Reproducing it at ``dt = 1/60``
    means commanding ``vel * 2.4``, which makes our iteration *i* its iteration *i*
    and gives CNS's own decay budget of 24 over its own 600-iteration cap.
    """
    arm = build(StubPipeline([[1.0, 0, 0, 0, 0, 0]]))
    assert cnsmod.PUBLISHED == {"step_gain": 2.0, "dt": 1 / 50.0, "max_steps": 600}
    assert arm.step_scale == pytest.approx(2.4)
    assert arm.decay_budget(600) == pytest.approx(24.0)
    # the same displacement per iteration as CNS's benchmark
    assert arm.step_scale * arm.dt == pytest.approx(cnsmod.PUBLISHED["step_gain"]
                                                    * cnsmod.PUBLISHED["dt"])
    arm.reset(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)))
    twist, debug, matches = arm.step(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)))
    assert matches is None                                    # CNS has no landmarks of ours
    np.testing.assert_allclose(twist.as_array(), [2.4, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(debug["velocity_raw"], [1.0, 0, 0, 0, 0, 0])


def test_the_literal_x2_is_available_and_costs_a_sixth_of_the_decay_budget():
    """``rate_compensate=False`` is the literal ``vel * 2``: lambda*T = 20, 83 % of CNS's."""
    arm = build(StubPipeline([[1.0, 0, 0, 0, 0, 0]]), cns={"rate_compensate": False})
    assert arm.step_scale == 2.0
    assert arm.decay_budget(600) == pytest.approx(20.0)
    assert arm.decay_budget(600) / 24.0 == pytest.approx(0.8333, abs=5e-5)
    arm.reset(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)))
    twist, _, _ = arm.step(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)))
    np.testing.assert_allclose(twist.as_array(), [2.0, 0, 0, 0, 0, 0])


def test_the_step_gain_and_its_compensation_are_on_the_record():
    scene = MockPlaneScene(render_scale=0.05)
    arm = cnsmod.build_cns_arm(make_config(episode={"max_iter": 3}), intrinsics=scene.intrinsics,
                               pipeline=StubPipeline([[0.0] * 6]))
    start = {"start_id": "t0", "cam_pos": scene.goal_pose.position + np.array([0.01, 0, 0]),
             "cam_R_opt": scene.goal_pose.rotation.tolist(), "starts_seed": 1}
    rec = runner.run_episode(scene, arm, start, make_config(episode={"max_iter": 3}))
    assert rec["cns_step_gain"] == 2.0 and rec["cns_rate_compensate"] is True
    assert rec["cns_step_scale"] == pytest.approx(2.4)
    assert rec["cns_published_dt"] == pytest.approx(1 / 50.0)
    assert rec["gain"] == pytest.approx(2.4)          # the multiplier actually applied
    assert rec["matcher_name"] == "cns_v1_sift" and rec["arm_kind"] == "method"
    assert rec["feature_error_first"] is None         # CNS exposes no scalar ||e||


# ---------------------------------------------------------------- BGR + scale
def test_the_pipeline_is_handed_bgr_on_both_entry_points():
    """CNS documents ``set_target`` and ``get_control_rate`` as bgr; RGB degrades silently."""
    stub = StubPipeline([[0.0] * 6])
    arm = build(stub)
    goal = np.zeros((2, 2, 3), np.uint8)
    goal[..., 0] = 10          # R
    goal[..., 2] = 30          # B
    arm.reset(goal, np.full((2, 2), 0.5))
    assert stub.target[0, 0, 0] == 30 and stub.target[0, 0, 2] == 10
    cur = goal.copy()
    cur[..., 0] = 200
    arm.step(cur, np.ones((2, 2)))
    assert stub.calls[0][0, 0, 2] == 200


def test_dist_scale_is_the_median_valid_goal_depth_and_is_overridable():
    stub = StubPipeline([[0.0] * 6])
    arm = build(stub)
    depth = np.array([[0.0, -1.0], [0.4, 0.6]])        # two invalid, two valid
    arm.reset(np.zeros((2, 2, 3), np.uint8), depth)
    assert stub.target_dist_scale == pytest.approx(0.5)
    arm2 = build(StubPipeline([[0.0] * 6]), cns={"dist_scale": 1.25})
    arm2.reset(np.zeros((2, 2, 3), np.uint8), depth)
    assert arm2.matcher.dist_scale == 1.25
    assert arm.matcher.scene_distance(None) == 1.0     # no depth -> CNS's own default


# ------------------------------------------------------------------- the stop
def _graph(err: float, n: int = 10, mask=None):
    """A graph whose observed nodes sit around ``err`` from their target.

    The errors are **spread** (0.8x to 1.2x) on purpose: CNS's trim keeps the
    weighted errors strictly below their own 90th percentile, so an all-equal
    vector trims everything and reports 0.0 - which is CNS's behaviour, and which
    would make an arming-gate test pass for the wrong reason.
    """
    tar = np.zeros((n, 2))
    cur = np.column_stack([err * np.linspace(0.8, 1.2, n), np.zeros(n)])
    return _Graph(cur, tar, np.ones(n, bool) if mask is None else mask)


def test_the_stop_sees_cns_own_per_keypoint_errors_and_node_ids():
    """The faithful reading (review R13): CNS's persistent graph, not our landmarks."""
    mask = np.array([True, False, True, True])
    g = _Graph([[0.002, 0.0], [9.0, 9.0], [0.004, 0.0], [0.0, 0.0]],
               np.zeros((4, 2)), mask)
    errs, ids = cnsmod.CNSPipeline.graph_error(g)
    np.testing.assert_allclose(errs, [0.002, 0.004, 0.0])
    np.testing.assert_array_equal(ids, [0, 2, 3])      # positions in the FULL node array
    assert cnsmod.CNSPipeline.n_matches(g) == 4


def test_the_stop_is_wired_and_fires_where_cns_plateau_fires():
    """The arm's debug reaches `CnsPlateau` unchanged, through the runner's hook."""
    err = 0.003
    stub = StubPipeline([[0.0] * 6], [_graph(err)])
    arm = build(stub, episode={"max_iter": 80})
    arm.reset(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
    rule = CnsPlateau(dt=arm.dt)
    fired = None
    for i in range(1, 81):
        _, debug, matches = arm.step(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
        assert debug["step"] == i
        ids = runner._landmark_ids(matches, debug)
        np.testing.assert_array_equal(ids, np.arange(10))
        obs = StopObservation.from_debug(debug, twist=np.zeros(6), dt=arm.dt,
                                         landmark_ids=ids)
        if rule.update(obs).would_stop and fired is None:
            fired = i
    # a settled error under the arming gate holds for `waiting_time / dt` intervals:
    # step 1 arms and sets the minimum (accruing nothing), then 49 intervals of 1/60 s
    # cross 0.8 s on the strict `>` - so 50 (protocol v2's "48 steps at 60 Hz" is the
    # nominal hold, not the count; see CnsPlateau.hold_steps).
    assert fired == 50 == round(CNS_STOP["waiting_time"] / arm.dt) + 2


def test_a_constant_error_above_the_arming_gate_never_fires():
    stub = StubPipeline([[0.0] * 6], [_graph(0.05)])   # 0.05 > conduct_thresh 0.01
    arm = build(stub, episode={"max_iter": 200})
    arm.reset(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
    rule = CnsPlateau(dt=arm.dt)
    for _ in range(200):
        _, debug, matches = arm.step(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
        obs = StopObservation.from_debug(debug, twist=np.zeros(6), dt=arm.dt,
                                         landmark_ids=runner._landmark_ids(matches, debug))
        assert not rule.update(obs).would_stop


def test_a_degenerate_payload_is_no_data_rather_than_an_invented_error():
    arm = build(StubPipeline([[0.0] * 6], [None]))
    arm.reset(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
    _, debug, _ = arm.step(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
    assert debug["normalised_errors"] is None and debug["stop_landmark_ids"] is None
    assert debug["n_matches"] == 0
    assert np.isnan(debug["feature_error_norm"])


def _episode_with_stop(stop_cfg, err=0.003, max_iter=80):
    """One mock episode through `run_episode` with a settled sub-gate CNS error."""
    scene = MockPlaneScene(render_scale=0.05)
    cfg = make_config(stop=stop_cfg, episode={"max_iter": max_iter})
    arm = cnsmod.build_cns_arm(cfg, intrinsics=scene.intrinsics,
                               pipeline=StubPipeline([[0.0] * 6], [_graph(err)]))
    g = scene.goal_pose
    start = {"start_id": "t0", "cam_pos": g.position + np.array([0.001, 0.0, 0.0]),
             "cam_R_opt": g.rotation.tolist(), "starts_seed": 1}
    return runner.run_episode(scene, arm, start, cfg)


def test_the_stop_rules_own_dt_is_the_clock_the_run_uses():
    """`stop: {dt: 0.02}` really runs at CNS's 50 Hz - it used to be a dead knob.

    `CnsPlateau` measures its 0.8 s hold against `StopObservation.t`, and the runner
    stamped that with the CONTROLLER's period, so a rule built at 1/50 still fired on
    the 60 Hz schedule while `stop_cfg` reported `dt: 0.02` (review R14, C1). A record
    that names one clock and ran another is the mislabelled cell this repo forbids.
    """
    ours = _episode_with_stop({"rule": "cns_plateau"})
    assert ours["stop_step"] == 50                      # arming step + the 49th interval
    assert ours["stop_cfg"]["dt"] == pytest.approx(1 / 60.0)
    published = _episode_with_stop({"rule": "cns_plateau", "dt": 0.02})
    assert published["stop_step"] == 42                 # arming step + the 41st interval
    assert published["stop_cfg"]["dt"] == 0.02
    assert published["cns_stop_dt"] == 0.02 and ours["cns_stop_dt"] == pytest.approx(1 / 60.0)


def test_a_recorded_only_cns_plateau_is_built_at_the_arms_period():
    """The `also` rules used to be built at DEFAULT_DT whatever the arm ran at."""
    scene = MockPlaneScene(render_scale=0.05)
    cfg = make_config(stop={"rule": "cap_only"},
                      episode={"max_iter": 3, "record_stops": ["cns_plateau"]})
    arm = cnsmod.build_cns_arm(cfg, intrinsics=scene.intrinsics,
                               pipeline=StubPipeline([[0.0] * 6], [_graph(0.003)]))
    assert arm.dt == pytest.approx(1 / 60.0)
    g = scene.goal_pose
    start = {"start_id": "t0", "cam_pos": g.position, "cam_R_opt": g.rotation.tolist(),
             "starts_seed": 1}
    rec = runner.run_episode(scene, arm, start, cfg)
    assert rec["stop_mode"] == "cap_only" and rec["stop_step"] is None
    assert set(rec["stops"]) <= {"velocity_quiet", "cns_plateau"}


def test_the_runner_hook_prefers_an_arms_own_node_set():
    """`debug["stop_landmark_ids"]` always wins over the drawn-landmark reading."""
    assert runner.CNS_LANDMARK_IDS is None                  # the shipped interim default
    got = runner._landmark_ids(None, {"stop_landmark_ids": np.array([3, 7])})
    np.testing.assert_array_equal(got, [3, 7])
    assert runner._landmark_ids(None, {}) is None


# -------------------------------------------------------------- the refusals
@pytest.mark.parametrize("block,match", [
    ({"oracle": {"mode": "a1"}}, "no matcher for the GT wrapper"),
    ({"head": {"checkpoint": "v2cm"}}, "refinement head lives in the DINOv3"),
])
def test_cns_composes_with_nothing(block, match):
    with pytest.raises(ValueError, match=match):
        cnsmod.build_cns_arm(make_config(**block), intrinsics=INTR,
                             pipeline=StubPipeline([[0.0] * 6]))


def test_cns_refuses_the_ibvs_controller_knobs_and_the_dec_schedule():
    with pytest.raises(ValueError, match="matcher/IBVS bench machinery"):
        cnsmod.build_cns_arm(make_config(controller={"gain": 2.0}), intrinsics=INTR,
                             pipeline=StubPipeline([[0.0] * 6]))
    with pytest.raises(ValueError, match="cannot run the `dec` stop"):
        cnsmod.build_cns_arm(make_config(stop={"rule": "dec_plateau"}), intrinsics=INTR,
                             pipeline=StubPipeline([[0.0] * 6]))


def test_an_unknown_detector_and_a_frame_mask_are_refused():
    with pytest.raises(ValueError, match="detector must be one of"):
        cnsmod.CNSPipeline(detector="LOFTR")
    arm = build(StubPipeline([[0.0] * 6]))
    arm.reset(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
    with pytest.raises(ValueError, match="does not take a frame mask"):
        arm.step(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="front end"):
        arm.reset(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), np.ones((2, 2), bool))


def test_a_goal_mask_reaches_the_cns_front_end_and_names_the_cell():
    """The masked cell (owner 2026-09-21): CNS's own `update_target_frame(image, mask)`
    gets the goal mask as a uint8 image; `set_target`'s three fields are set alike."""
    class _Front:
        def __init__(self):
            self.seen = []

        def update_target_frame(self, image, mask=None):
            self.seen.append((image.shape, None if mask is None else (mask.dtype, int(mask.max()))))
            return True

    pipe = StubPipeline([[0.0] * 6])
    pipe.frontend = _Front()
    arm = cnsmod.build_cns_arm(make_config(), intrinsics=INTR, pipeline=pipe,
                               goal_mask_mode="objects")
    assert arm.goal_mask_mode == "objects" and "/objmask/" in arm.name
    mask = np.zeros((2, 2), bool)
    mask[0, 0] = True
    arm.reset(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), mask)
    assert pipe.frontend.seen == [((2, 2, 3), (np.dtype("uint8"), 255))]
    assert pipe.new_scene is True and pipe.target_kwargs == {}


def test_build_arm_routes_cns_and_refuses_an_injected_matcher():
    scene = MockPlaneScene(render_scale=0.05)
    with pytest.raises(ValueError, match="whole pipeline, not a matcher"):
        arms.build_arm(make_config(), intrinsics=scene.intrinsics, matcher=object())
    with pytest.raises(ValueError, match="not a matcher"):
        __import__("vsbench.vs.baselines.match", fromlist=["x"]).MATCHER_BUILDERS["cns_v1"]("cns_v1")


def test_the_checkout_search_explains_itself(monkeypatch, tmp_path):
    monkeypatch.delenv(cnsmod.CNS_ROOT_ENV, raising=False)
    monkeypatch.setattr(cnsmod.Path, "home", staticmethod(lambda: tmp_path))
    # `resolve_cns_root` also tries the hard-coded `/workspace/CNS`, and ON THE STATION
    # that checkout really is there - which is the whole point of the fallback. Asserting
    # the not-found branch is therefore a laptop-only statement, and pretending otherwise
    # would make this test fail for the one reason it should not (station sync 2026-09-03).
    if not (cnsmod.Path("/workspace/CNS") / "cns").is_dir():
        with pytest.raises(FileNotFoundError, match="hhcaz/CNS"):
            cnsmod.resolve_cns_root(str(tmp_path / "nope"))
    (tmp_path / "cns").mkdir()
    assert cnsmod.resolve_cns_root(str(tmp_path)) == tmp_path
    # `cns_root` used to be accepted and dropped - a silent config knob (review R14, C3)
    assert cnsmod.CNSPipeline(cns_root=str(tmp_path))._root == tmp_path
    assert cnsmod.CNSPipeline()._root is None


def test_an_unknown_cns_config_key_is_refused():
    with pytest.raises(ValueError, match="unknown cns config keys"):
        cnsmod.build_cns_arm(make_config(cns={"detecter": "SIFT"}), intrinsics=INTR,
                             pipeline=StubPipeline([[0.0] * 6]))
    assert "cns_root" in cnsmod.CNS_CONFIG_KEYS


def test_the_record_names_the_resolved_checkpoint_and_the_method():
    arm = build(StubPipeline([[0.0] * 6]))
    assert arm.checkpoint == cnsmod.DEFAULT_CHECKPOINT      # never None on a CNS row
    assert vars(arm.controller.weighting) == {"learned": "cns_v1_graphvs"}
    arm2 = build(StubPipeline([[0.0] * 6]), cns={"ckpt": "checkpoints/other.pth"})
    assert arm2.checkpoint == "checkpoints/other.pth"


def test_the_support_columns_say_what_they_are():
    """`n_matches` is CNS's constant node count; `n_matched` is this frame's."""
    g = _Graph([[0.0, 0.0]] * 4, [[0.0, 0.0]] * 4, [True, False, True, True])
    arm = build(StubPipeline([[0.0] * 6], [g]))
    arm.reset(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
    _, debug, _ = arm.step(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)))
    assert debug["n_matches"] == 4 and debug["n_matched"] == 3
    # `match_ms` is CNS's frontend (detect + match) stage, in ms
    assert debug["match_ms"] == pytest.approx(4.0)



def test_intrinsics_are_never_invented():
    """And the refusal fires BEFORE anything of CNS is imported (station verify,
    2026-09-05): with `PYTHONPATH` cleared the station's `genvs` interpreter has no
    `torch_geometric`, so the CNS import raised `ModuleNotFoundError` first and this
    property went untested exactly where the arm would fly. `RuntimeError` is now the
    only outcome on every machine - with a checkout, without one, with or without CNS's
    own dependencies - which is what makes it a property of our code."""
    pipe = cnsmod.CNSPipeline(intrinsics=None)
    pipe.goal_rgb = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(RuntimeError, match="refusing to invent a placeholder"):
        pipe._load()
