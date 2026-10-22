# vs-sim copy: tests/fast/test_isaac_scene.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The Isaac-free half of the Isaac scene.

Guards `vssim/isaac_scene.py`: the optical-frame velocity integration reproduces
the original `free_flying_scene` pose update on a synthetic twist sequence - run
**read-only** out of the frozen `../genvs-sim` where that checkout exists - plus
the optical/USD quaternion convention, the segmentation-mask contract, the goal
capture rules and the empty-annotator guard. Nothing here boots Isaac; the module
must import without it, which is itself one of the assertions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vsbench.vs.isaac_scene import (
    R_INTERNAL_TOL,
    R_OPT2USD_BASE,
    IsaacScene,
    RendererMismatchError,
    check_renderer_match,
    manifest_render_mode,
    mat_to_quat_wxyz,
    optical_R_to_usd_quat,
    quat_wxyz_to_mat,
    render_mode_of,
    reorthogonalize,
    seg_to_target_mask,
    usd_quat_to_optical_R,
)
from vsbench.vs.scene import Pose, Scene, integrate_optical_velocity

OLD_REPO = Path(__file__).resolve().parents[3] / "genvs-sim"


def _twist_sequence(n: int = 40, seed: int = 20260903) -> np.ndarray:
    """A synthetic 6-DoF twist per step: metres/s and radians/s of the size a
    real episode commands, plus one pure rotation and one zero step."""
    rng = np.random.default_rng(seed)
    v = np.concatenate([rng.normal(0.0, 0.02, size=(n, 3)),
                        rng.normal(0.0, 0.05, size=(n, 3))], axis=1)
    v[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]           # the no-op step
    v[1] = [0.0, 0.0, 0.0, 0.3, -0.2, 0.1]          # pure rotation
    v[2] = [0.05, -0.02, 0.01, 0.0, 0.0, 0.0]       # pure translation
    return v


# ------------------------------------------------- the pose / velocity maths
@pytest.mark.skipif(not (OLD_REPO / "src" / "eval" / "free_flying_scene.py").exists(),
                    reason="the frozen ../genvs-sim checkout is not here")
def test_the_integration_reproduces_the_frozen_free_flying_scene():
    """Same twists, same start, same poses - bit for bit.

    The old function is separable (module level, numpy only), so it is imported
    and run read-only rather than re-derived; this proves the Pose form the Isaac
    scene applies is that function.
    """
    sys.path.insert(0, str(OLD_REPO))
    try:
        from src.eval.free_flying_scene import integrate_optical_velocity as old_integrate
        from src.eval.free_flying_scene import reorthogonalize as old_reortho
    finally:
        sys.path.remove(str(OLD_REPO))

    pos = np.array([0.047, 0.152, 1.411])            # cluttered2's goal, roughly
    R = np.diag([1.0, -1.0, -1.0])
    pose = Pose(position=pos, rotation=R)
    dt = 1.0 / 60.0
    for twist in _twist_sequence():
        pos, R = old_integrate(pos, R, twist, dt)
        pose = integrate_optical_velocity(pose, twist, dt)
        np.testing.assert_array_equal(pose.position, pos)      # bit-identical
        # the rotation differs by at most one ULP: the old Rodrigues used
        # math.sin/cos, ours uses np.sin/np.cos. Measured max |dR| = 1.7e-18
        # after 40 steps, which is 15 orders below the 1e-3 rad entries.
        np.testing.assert_allclose(pose.rotation, R, rtol=0, atol=1e-17)
    np.testing.assert_array_equal(reorthogonalize(R), old_reortho(R))


def test_reorthogonalize_snaps_a_drifted_rotation_back_onto_so3():
    R = np.diag([1.0, -1.0, -1.0]) + 1e-6 * np.random.default_rng(0).normal(size=(3, 3))
    assert abs(np.linalg.det(R) - 1.0) > 0            # drifted (det is -1-ish here)
    Rn = reorthogonalize(R)
    np.testing.assert_allclose(Rn @ Rn.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(Rn) == pytest.approx(1.0)


def test_the_optical_usd_convention_round_trips():
    """`R_usd = R_opt @ diag(1, -1, -1)`; the base map is self-inverse, so a
    straight-down optical pose is the identity USD orientation."""
    np.testing.assert_array_equal(R_OPT2USD_BASE @ R_OPT2USD_BASE, np.eye(3))
    q = optical_R_to_usd_quat(R_OPT2USD_BASE)
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    rng = np.random.default_rng(3)
    for _ in range(20):
        A = rng.normal(size=(3, 3))
        R = reorthogonalize(A)
        np.testing.assert_allclose(usd_quat_to_optical_R(optical_R_to_usd_quat(R)), R,
                                   atol=1e-12)
        np.testing.assert_allclose(quat_wxyz_to_mat(mat_to_quat_wxyz(R)), R, atol=1e-12)


def test_the_quaternion_branches_are_all_reachable():
    """`mat_to_quat_wxyz` has four branches by trace; each must stay a unit quat."""
    for R in (np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]),
              np.diag([-1.0, -1.0, 1.0])):
        q = mat_to_quat_wxyz(R)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        np.testing.assert_allclose(quat_wxyz_to_mat(q), R, atol=1e-12)


# ---------------------------------------------------------- the mask contract
def test_seg_mask_uses_the_labelled_target_ids():
    seg = {"data": np.array([[1, 2], [2, 3]]),
           "info": {"idToLabels": {"2": {"class": "/World/Target/asset"},
                                   "3": {"class": "/World/Ground"}}}}
    mask = seg_to_target_mask(seg)
    np.testing.assert_array_equal(mask, [[False, True], [True, False]])
    assert mask.dtype == bool


def test_seg_mask_refuses_the_corner_pixel_fallback_in_a_multi_object_scene():
    """The historical `arr != arr[0, 0]` fallback would call every distractor
    "target"; with other /World/ prims labelled and none under the prefix, refuse."""
    seg = {"data": np.array([[1, 2], [2, 3]]),
           "info": {"idToLabels": {"2": {"class": "/World/Obj_0"},
                                   "3": {"class": "/World/Ground"}}}}
    with pytest.raises(RuntimeError, match="refusing the everything-differs"):
        seg_to_target_mask(seg)


def test_seg_mask_keeps_the_legacy_fallback_for_a_bare_annotator():
    """No target ids and no /World/ labels at all: the historical single-object
    annotator glitch, whose corner-pixel fallback is preserved byte-for-byte."""
    seg = {"data": np.array([[0, 5], [5, 0]]), "info": {"idToLabels": {}}}
    np.testing.assert_array_equal(seg_to_target_mask(seg),
                                  [[False, True], [True, False]])
    np.testing.assert_array_equal(seg_to_target_mask(np.array([[0, 7]])),
                                  [[False, True]])


def test_a_channelled_seg_array_is_squeezed():
    seg = {"data": np.array([[[2], [4]]]),
           "info": {"idToLabels": {"4": "/World/Target"}}}
    np.testing.assert_array_equal(seg_to_target_mask(seg), [[False, True]])


# ----------------------------------------------------------- the class itself
def test_the_module_imports_and_the_class_builds_without_isaac():
    """The two-runtime rule: `omni` / `pxr` are imported inside methods only."""
    assert "pxr" not in sys.modules and "omni" not in sys.modules
    scene = IsaacScene(boot=False, goal_render_warmup=1)
    assert scene.name == "unbuilt"
    scene._goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._distance = 0.5
    assert isinstance(scene, Scene)                   # the protocol
    assert scene.intrinsics.fx == pytest.approx(1406.75, abs=5e-3)
    assert R_INTERNAL_TOL == 1e-3


def test_an_unbuilt_scene_refuses_to_name_a_goal():
    """An episode scored against an unset goal is a silent wrong number."""
    scene = IsaacScene(boot=False, goal_render_warmup=1)
    for attr in ("goal_pose", "working_distance"):
        with pytest.raises(RuntimeError, match="no goal built yet"):
            getattr(scene, attr)


def test_apply_integrates_at_the_optical_convention_without_isaac():
    scene = IsaacScene(boot=False, goal_render_warmup=1)
    scene.set_camera_pose_optical([0.0, 0.0, 0.5], np.diag([1.0, -1.0, -1.0]))
    expected = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    for twist in _twist_sequence(10):
        scene.apply(twist, 1.0 / 60.0)
        expected = integrate_optical_velocity(expected, twist, 1.0 / 60.0)
        np.testing.assert_array_equal(scene.pose.position, expected.position)
        np.testing.assert_array_equal(scene.pose.rotation, expected.rotation)


def test_gt_correspondence_refuses_a_pose_it_has_no_depth_for():
    """The protocol carries no current depth, so the scene occlusion-tests against
    the frame it last rendered - or says it cannot."""
    scene = IsaacScene(boot=False, goal_render_warmup=1)
    scene._goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._distance = 0.5
    with pytest.raises(RuntimeError, match="needs a rendered current frame"):
        scene.gt_correspondence(np.zeros((2, 2)), scene.goal_pose)
    scene._last = (Pose(position=[0.0, 0.0, 0.4], rotation=np.eye(3)),
                   np.zeros((4, 4, 3), np.uint8), np.full((4, 4), 0.4, np.float32))
    with pytest.raises(ValueError, match="not the last rendered frame"):
        scene.gt_correspondence(np.zeros((2, 2)), scene.goal_pose)


# ------------------------------------------------- the goal frame is served once
class _CountingScene(IsaacScene):
    """`IsaacScene` with the Isaac render replaced by a counter."""

    def __init__(self, **kw):
        kw.setdefault("goal_render_warmup", 1)      # required, no default
        super().__init__(boot=False, **kw)
        self.captures = 0

    def capture_rgbd(self):
        self.captures += 1
        return (np.full((4, 4, 3), self.captures, np.uint8),
                np.full((4, 4), 0.4 + 0.01 * self.captures, np.float32))

    def _write_pose_to_usd(self):
        pass


def test_the_goal_frame_is_rendered_once_and_served_from_render():
    """An episode loop opens with `reset(goal); render()` and need never call
    `render_goal`, so the cache has to live in `render` or it is dead code. RTX is
    not frame-deterministic: re-rendering the goal per episode moves the goal
    pixels as well as costing a render.
    """
    scene = _CountingScene()
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    scene.reset(goal)
    rgb_a, depth_a, _ = scene.render()
    assert scene.captures == 1
    scene.reset(Pose(position=[0.01, 0.0, 0.5], rotation=goal.rotation))
    scene.render()
    assert scene.captures == 2                       # a real frame, not the cache
    scene.reset(goal)                                # a second episode
    rgb_b, depth_b, _ = scene.render()
    assert scene.captures == 2                       # served from the cache
    np.testing.assert_array_equal(rgb_a, rgb_b)
    np.testing.assert_array_equal(depth_a, depth_b)
    assert scene.render_goal()[0] is rgb_b           # and so is render_goal


# ------------------------------------------ the goal capture's render warm-up
def test_the_goal_capture_takes_n_frames_and_keeps_the_last():
    """The first RTX capture after a pose change carries render history, and the
    goal frame was exactly that in every early campaign. So the GOAL capture takes
    `goal_render_warmup` frames and keeps the LAST one; the live frames of an
    episode are untouched, which is what the capture counts below assert.
    """
    scene = _CountingScene(goal_render_warmup=3)
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    scene.reset(goal)
    rgb, _depth, _pose = scene.render()
    assert scene.captures == 3                       # three captures for one goal
    assert int(rgb[0, 0, 0]) == 3                    # and the THIRD is the goal

    scene.reset(Pose(position=[0.01, 0.0, 0.5], rotation=goal.rotation))
    scene.render()
    assert scene.captures == 4                       # a live frame is ONE capture
    scene.reset(goal)
    assert int(scene.render()[0][0, 0, 0]) == 3      # and the goal is still the cache
    assert scene.captures == 4


def test_the_goal_warm_up_has_no_default_and_refuses_a_count_below_one():
    """A count nobody wrote down is what the warm-up rule exists to prevent, so the
    constructor states it too - `bake` passes 1 at the call site with "captures no
    goal" beside it."""
    with pytest.raises(TypeError, match="goal_render_warmup"):
        IsaacScene(boot=False)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="goal_render_warmup must be an int >= 1"):
            IsaacScene(boot=False, goal_render_warmup=bad)
    scene = _CountingScene(goal_render_warmup=1)
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    scene.reset(goal)
    scene.render()
    assert scene.captures == 1                       # 1 = no warm-up


def test_the_goal_capture_keeps_the_first_when_a_diagnostic_cell_asks():
    """Eight renders that keep the FIRST separate "the goal was unsettled" from
    "rendering eight times changed the scene". `last` stays the default; `first` is
    a diagnostic, never a default."""
    scene = _CountingScene(goal_render_warmup=3, goal_capture_keep="first")
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    scene.reset(goal)
    rgb, _d, _p = scene.render()
    assert scene.captures == 3 and int(rgb[0, 0, 0]) == 1     # three renders, the FIRST
    series = scene.goal_capture_record()
    assert series["goal_capture_keep"] == "first"
    assert series["goal_settle_series"]["kept_index"] == 0
    assert series["goal_frame_md5"] == series["goal_settle_series"]["rgb_md5"][0]
    with pytest.raises(ValueError, match="goal_capture_keep must be one of"):
        IsaacScene(boot=False, goal_render_warmup=1, goal_capture_keep="middle")


def test_the_settle_series_carries_a_checksum_for_every_capture():
    """The means told the station's settle runs apart only because the md5s did.
    Both are recorded per capture, rgb and depth."""
    scene = _CountingScene(goal_render_warmup=3)
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    scene.reset(goal)
    scene.render()
    series = scene.goal_settle
    assert len(series["rgb_md5"]) == len(series["depth_md5"]) == 3
    assert len(set(series["rgb_md5"])) == 3          # the stub changes every capture
    assert series["diff_prev"][0] is None and series["diff_prev"][1] > 0
    assert scene.goal_frame_md5 == series["rgb_md5"][-1]


def test_the_warm_up_can_be_stated_after_the_boot_but_never_after_the_capture():
    """A driver that reads its config AFTER the simulator boots cannot pass the count
    to the constructor - and a goal captured under one rule and relabelled with
    another is the silent wrong number this repository refuses."""
    scene = _CountingScene(goal_render_warmup=1)
    scene.set_goal_render_warmup(3)
    scene.set_goal_capture_keep("first")
    assert (scene.goal_render_warmup, scene.goal_capture_keep) == (3, "first")
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    scene.reset(goal)
    scene.render()
    for call in (lambda: scene.set_goal_render_warmup(5),
                 lambda: scene.set_goal_capture_keep("last")):
        with pytest.raises(RuntimeError, match="already captured"):
            call()


def test_an_injected_goal_replaces_the_render_and_says_so_on_every_record():
    """Fly against a SAVED goal, so the goal's pixels and the process that produced
    them can be separated."""
    scene = _CountingScene(goal_render_warmup=3)
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    rgb = np.full((4, 4, 3), 200, np.uint8)
    depth = np.full((4, 4), 0.43, np.float32)
    scene.inject_goal(rgb, depth, source="old_goal/goal_0.npz")
    scene.reset(goal)
    got, got_depth, _pose = scene.render()
    assert scene.captures == 0                      # nothing was rendered
    np.testing.assert_array_equal(got, rgb)
    np.testing.assert_array_equal(got_depth, depth)
    rec = scene.goal_capture_record()
    assert rec["goal_settle_series"]["injected_from"] == "old_goal/goal_0.npz"
    assert rec["goal_settle_series"]["n_captures"] == 0
    with pytest.raises(RuntimeError, match="already in hand"):
        scene.inject_goal(rgb, depth, source="again")


# ------------------------------------------------------- closing the simulator
class _Sim:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_close_closes_the_simulator_exactly_once_and_is_idempotent():
    """Every early stage exited 1 because the app was never closed - the interpreter
    tore carb down through `atexit` under a live SimulationApp and Isaac's plugins
    segfaulted after a perfectly good run. A runbook gates on the exit code, so
    closing twice (a `finally:` inside a `finally:`) must not be a crash either."""
    scene = IsaacScene(boot=False, goal_render_warmup=1)
    assert scene.sim is None
    scene.close()                                    # a never-booted scene: a no-op
    sim = _Sim()
    scene.sim = sim
    scene.close()
    scene.close()
    assert sim.closed == 1 and scene.sim is None


def test_render_goal_does_not_disturb_the_pose_or_the_reortho_counter():
    """It writes the USD pose directly: `reset` would zero `_reortho_counter`
    in the middle of an episode."""
    scene = _CountingScene()
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    here = Pose(position=[0.02, -0.01, 0.47], rotation=goal.rotation)
    scene.reset(here)
    for twist in _twist_sequence(5):
        scene.apply(twist, 1.0 / 60.0)
    pose_before, counter_before = scene.pose, scene._reortho_counter
    scene.render_goal()
    assert scene._reortho_counter == counter_before
    np.testing.assert_array_equal(scene.pose.position, pose_before.position)
    np.testing.assert_array_equal(scene.pose.rotation, pose_before.rotation)


# --------------------------------------------------------- the renderer guard
def test_the_scene_names_its_renderer():
    assert IsaacScene.render_mode == "rtx"
    assert IsaacScene(boot=False, goal_render_warmup=1).render_mode == "rtx"


def test_a_renderer_mismatch_raises_both_ways():
    with pytest.raises(RendererMismatchError, match="RENDERER MISMATCH"):
        check_renderer_match("rtx", "pt", where="banks")
    with pytest.raises(RendererMismatchError):
        check_renderer_match("pt", "rtx")


def test_a_matching_or_unknown_lineage_passes_and_unknown_warns(capsys):
    """An unrecorded lineage passes - the guard cannot verify what was never
    written down - but it must say so: silence there once hid a real hazard."""
    check_renderer_match("rtx", "rtx")
    check_renderer_match("pt", "pt")
    check_renderer_match("rtx", None, where="stored goal banks")
    out = capsys.readouterr().out
    assert "WARNING" in out and "stored goal banks" in out


def test_manifest_render_mode_parses_both_shapes_and_refuses_a_mix(tmp_path):
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps({"render_version": "banks_v1_rtx"}))
    assert manifest_render_mode(mp) == "rtx"
    mp.write_text(json.dumps([{"key": "a", "render_version": "x_pathtrace"}, {"key": "b"}]))
    assert manifest_render_mode(mp) == "pt"
    mp.write_text(json.dumps([{"key": "a"}]))
    assert manifest_render_mode(mp) is None          # a pre-guard manifest
    assert manifest_render_mode(tmp_path / "nope.json") is None
    mp.write_text(json.dumps([{"render_version": "a_rtx"}, {"render_version": "b_pathtrace"}]))
    with pytest.raises(RendererMismatchError, match="mixes renderers"):
        manifest_render_mode(mp)
    assert render_mode_of(None) is None and render_mode_of("wp1_rtx_w1") == "rtx"


# ------------------------------------------- an empty annotator is not a picture
class _FlakyAnnotator:
    """An Isaac annotator that returns nothing for its first ``empties`` reads.

    The transient a six-scene runbook hit: one Isaac process launched straight after
    the previous one exited, and ten seconds in the first read of the rgb annotator
    came back with shape ``(0,)``.
    """

    def __init__(self, frame, empties: int):
        self.frame = frame
        self.empties = int(empties)
        self.reads = 0

    def get_data(self):
        self.reads += 1
        if self.reads <= self.empties:
            return np.zeros((0,), dtype=np.uint8)
        return self.frame


class _StubOrchestrator:
    def __init__(self):
        self.steps = 0

    def step(self, rt_subframes=4):
        self.steps += 1


class _StubRep:
    def __init__(self):
        self.orchestrator = _StubOrchestrator()


def _flaky_scene(empties, *, warmup=1, retries=5):
    """A real `IsaacScene` with only the three annotators and the orchestrator stubbed."""
    scene = IsaacScene(boot=False, goal_render_warmup=warmup, capture_retries=retries,
                       verbose=False)
    scene.name = "industrial1_drill"
    scene._rep = _StubRep()
    scene._rgb_annot = _FlakyAnnotator(np.full((4, 4, 3), 7, np.uint8), empties)
    scene._depth_annot = _FlakyAnnotator(np.full((4, 4), 0.44, np.float32), 0)
    scene._seg_annot = _FlakyAnnotator(
        {"data": np.ones((4, 4), np.int32),
         "info": {"idToLabels": {"1": "/World/Obj_0"}}}, 0)
    return scene


def test_an_empty_capture_buys_another_step_and_records_it():
    """One empty read, then a frame: the capture succeeds, it cost one extra orchestrator
    step, and the count is on the scene rather than lost."""
    scene = _flaky_scene(1)
    rgb, depth = scene.capture_rgbd()
    assert rgb.shape == (4, 4, 3) and int(rgb[0, 0, 0]) == 7
    assert depth.shape == (4, 4)
    assert scene.last_capture_retries == 1
    assert scene._rep.orchestrator.steps == 2, "the retry did not render a new frame"
    # ...and a capture that came back whole costs nothing and records nothing
    scene2 = _flaky_scene(0)
    scene2.capture_rgbd()
    assert scene2.last_capture_retries == 0
    assert scene2._rep.orchestrator.steps == 1


def test_an_annotator_that_stays_empty_raises_naming_it_and_the_retries():
    scene = _flaky_scene(99, retries=5)
    with pytest.raises(RuntimeError) as exc:
        scene.capture_rgbd()
    message = str(exc.value)
    assert "rgb" in message and "industrial1_drill" in message
    assert "5 extra orchestrator step" in message
    assert scene._rep.orchestrator.steps == 6      # the first read plus five retries


def test_the_settle_series_records_what_each_capture_cost():
    scene = _flaky_scene(1, warmup=3)
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    scene._pose = goal
    scene.render_goal()
    series = scene.goal_settle
    assert series["capture_retries"] == [1, 0, 0]
    assert len(series["rgb_mean"]) == 3


def test_the_settle_difference_says_which_capture_was_empty():
    """The second wall. `capture_rgbd` raises first now, so this can only be reached by a
    capture path that does not go through it - but the message a runbook once got said
    only `operands could not be broadcast together with shapes (1080,1920,3) (0,)`,
    which names neither the capture nor the scene."""
    class _Uneven(IsaacScene):
        def __init__(self):
            super().__init__(boot=False, goal_render_warmup=2, verbose=False)
            self.name = "industrial1_drill"
            self.n = 0

        def capture_rgbd(self):
            self.n += 1
            if self.n == 1:
                return np.zeros((0,), np.uint8), np.zeros((0,), np.float32)
            return np.full((4, 4, 3), 3, np.uint8), np.full((4, 4), 0.4, np.float32)

        def _write_pose_to_usd(self):
            pass

    scene = _Uneven()
    goal = Pose(position=[0.0, 0.0, 0.5], rotation=np.diag([1.0, -1.0, -1.0]))
    scene._goal, scene._distance = goal, 0.5
    scene.reset(goal)
    with pytest.raises(RuntimeError, match="capture 1 of 2 is \\(0,\\)"):
        scene.render()


def test_an_empty_segmentation_raises_rather_than_stepping():
    """A step would make it the mask of a DIFFERENT frame from the one just returned,
    which is the one thing `capture_seg` exists to avoid."""
    scene = _flaky_scene(0)
    scene._seg_annot = _FlakyAnnotator({"data": np.ones((4, 4), np.int32),
                                        "info": {"idToLabels": {"1": "/World/Obj_0"}}}, 1)
    seg = scene.capture_seg()                      # one empty read, then the buffer
    assert np.asarray(seg["data"]).size == 16
    assert scene._rep.orchestrator.steps == 0, "capture_seg must never step"

    scene.capture_retries = 2
    scene._seg_annot = _FlakyAnnotator({"data": np.ones((4, 4), np.int32)}, 99)
    with pytest.raises(RuntimeError, match="instance_segmentation_fast"):
        scene.capture_seg()
    assert scene._rep.orchestrator.steps == 0


def test_the_retry_count_is_stated_and_recorded_not_implied():
    """A default a caller may override is stated in the module that owns it and
    recorded on the object."""
    from vsbench.vs.isaac_scene import CAPTURE_RETRIES_DEFAULT

    assert CAPTURE_RETRIES_DEFAULT == 5
    assert IsaacScene(boot=False, goal_render_warmup=1).capture_retries == 5
    assert IsaacScene(boot=False, goal_render_warmup=1,
                      capture_retries=2).capture_retries == 2
