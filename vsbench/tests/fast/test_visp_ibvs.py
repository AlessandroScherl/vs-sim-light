# vs-sim copy: tests/fast/test_visp_ibvs.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""ViSP's standard IBVS as a bench controller - the numpy twin, without ViSP installed.

Guards `vssim/baselines/visp_ibvs.py`: under the `current` interaction matrix the twin
IS the bench's own unweighted law (the same pseudo-inverse of the same matrix, up to
the singular-value rule), `desired` and `mean` are what vpServo defines, the adaptive
gain is vpAdaptiveGain's curve on the infinity norm, the refusals name what ViSP's
law does not have, the arm registry wires `controller.law: visp` and labels it, and
an episode on the mock plane converges with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vsbench.vs.baselines import visp_ibvs as vi
from vsbench.vs.baselines.control import UNIFORM_WEIGHTING, IBVSController, interaction_matrix
from vsbench.vs.bench import arms
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.bench.runner import run_episode
from vsbench.vs.bench.truth import PlaneTruthMatcher
from vsbench.vs.scene import D435I, MockPlaneScene, rodrigues

ARMS = Path(__file__).resolve().parents[2] / "configs" / "arms"


def _rows(n=24, seed=3):
    rng = np.random.default_rng(seed)
    goal = rng.uniform([200, 150], [1700, 900], size=(n, 2))
    cur = goal + rng.normal(0, 25, size=(n, 2))
    z = rng.uniform(0.4, 0.6, size=n)
    zs = rng.uniform(0.4, 0.6, size=n)
    return goal, cur, z, zs


# ------------------------------------------------------------- the twin's maths
def test_current_interaction_is_the_benchs_own_unweighted_law():
    """v = -lambda L(s, Z)^+ (s - s*): the same expression as `weighted_ls_solve` with
    uniform weights, so the two agree to round-off on a well-conditioned set."""
    goal, cur, z, _ = _rows()
    ours = IBVSController(gain=2.0, intrinsics=D435I, weighting=UNIFORM_WEIGHTING)
    v_ours, _ = ours.step({"pts_goal": goal, "pts_cur": cur}, z)
    visp = vi.VispIBVSController(gain=2.0, intrinsics=D435I, interaction="current",
                                 backend="numpy")
    v_visp, dbg = visp.step({"pts_goal": goal, "pts_cur": cur}, z)
    np.testing.assert_allclose(v_visp.as_array(), v_ours.as_array(), rtol=0, atol=1e-9)
    assert dbg["law"] == "visp" and dbg["visp_backend"] == "numpy"
    assert dbg["feature_error_norm"] == pytest.approx(
        float(np.linalg.norm(vi.normalised_coords(cur, D435I) - vi.normalised_coords(goal, D435I))))


def test_desired_and_mean_use_the_goal_depth_and_are_what_vpservo_defines():
    goal, cur, z, zs = _rows()
    s, s_star = vi.normalised_coords(cur, D435I), vi.normalised_coords(goal, D435I)
    e = (s - s_star).reshape(-1)
    want_desired = -2.0 * np.linalg.pinv(interaction_matrix(s_star, zs), rcond=1e-6) @ e
    want_mean = -2.0 * np.linalg.pinv(
        0.5 * (interaction_matrix(s, z) + interaction_matrix(s_star, zs)), rcond=1e-6) @ e
    v_d, _ = vi.visp_control_law(s, s_star, z, zs, interaction="desired", gain=2.0)
    v_m, _ = vi.visp_control_law(s, s_star, z, zs, interaction="mean", gain=2.0)
    np.testing.assert_allclose(v_d, want_desired, atol=1e-12)
    np.testing.assert_allclose(v_m, want_mean, atol=1e-12)
    assert not np.allclose(v_d, v_m)
    # at the goal every choice commands nothing
    v0, info = vi.visp_control_law(s_star, s_star, zs, zs, interaction="mean", gain=2.0)
    np.testing.assert_allclose(v0, 0.0, atol=1e-15)
    assert info["feature_error_norm"] == 0.0


def test_a_desired_cell_takes_z_star_from_the_goal_depth_map_and_refuses_without_it():
    goal, cur, z, _ = _rows(n=8)
    ctrl = vi.VispIBVSController(gain=1.0, intrinsics=D435I, interaction="desired",
                                 backend="numpy")
    with pytest.raises(RuntimeError, match="needs the goal depth"):
        ctrl.step({"pts_goal": goal, "pts_cur": cur}, z)
    depth = np.full((1080, 1920), 0.5, np.float32)
    depth[:, :960] = 0.7                       # the left half is farther away
    ctrl.set_goal_depth(depth)
    v, dbg = ctrl.step({"pts_goal": goal, "pts_cur": cur}, z)
    zs = np.where(goal[:, 0] < 960, 0.7, 0.5)
    s, s_star = vi.normalised_coords(cur, D435I), vi.normalised_coords(goal, D435I)
    want, _ = vi.visp_control_law(s, s_star, z, zs, interaction="desired", gain=1.0)
    np.testing.assert_allclose(v.as_array(), want, rtol=1e-6)   # float32 depth map
    assert dbg["visp_interaction"] == "desired"


def test_the_adaptive_gain_is_vpadaptivegain_on_the_infinity_norm():
    g = vi.AdaptiveGain(4.0, 0.4, 30.0)
    assert g(np.zeros(4)) == pytest.approx(4.0)                    # lambda(0) = gain_at_zero
    big = g(np.array([0.0, 10.0]))
    assert big == pytest.approx(0.4, abs=1e-6)                      # -> gain_at_infinity
    # the slope at zero: d lambda / dx |0 = -slope_at_zero
    eps = 1e-6
    assert (g(np.array([eps])) - g(np.zeros(1))) / eps == pytest.approx(-30.0, rel=1e-3)
    # the infinity norm, not the 2-norm: two equal components give the same lambda as one
    assert g(np.array([0.1, 0.1])) == g(np.array([0.1]))
    with pytest.raises(ValueError, match="gain_at_zero > gain_at_infinity"):
        vi.AdaptiveGain(0.4, 4.0, 30.0)
    goal, cur, z, _ = _rows()
    s, s_star = vi.normalised_coords(cur, D435I), vi.normalised_coords(goal, D435I)
    v, info = vi.visp_control_law(s, s_star, z, z, interaction="current", gain=g)
    v1, _ = vi.visp_control_law(s, s_star, z, z, interaction="current", gain=1.0)
    np.testing.assert_allclose(v, info["lambda"] * v1, atol=1e-12)
    assert 0.4 < info["lambda"] < 4.0
    # ...and it is evaluated on the TASK error e1 = L^+ e (vpServo.cpp), not on e
    assert info["lambda"] == pytest.approx(g(v1))          # v1 = -1.0 * e1, same |.|_inf


def test_the_pseudo_inverse_rule_is_visps():
    """Singular values below 1e-6 of the largest are dropped, as vpMatrix::pseudoInverse."""
    L = np.diag([1.0, 1e-7, 0.5, 2.0, 1e-8, 1.0])
    P = vi.visp_pinv(L)
    assert P[1, 1] == 0.0 and P[4, 4] == 0.0 and P[0, 0] == 1.0 and P[3, 3] == 0.5


def test_fewer_than_min_matches_is_a_zero_twist():
    ctrl = vi.VispIBVSController(gain=2.0, intrinsics=D435I, backend="numpy", min_matches=4)
    goal, cur, z, _ = _rows(n=3)
    v, dbg = ctrl.step({"pts_goal": goal, "pts_cur": cur}, z)
    assert not v.as_array().any() and dbg["degenerate"] is True
    assert np.isnan(dbg["feature_error_norm"])


# --------------------------------------------------------------- the config
def test_from_config_refuses_what_visps_law_does_not_have():
    base = {"controller": {"law": "visp", "gain": 2.0}}
    ctrl = vi.VispIBVSController.from_config(base, intrinsics=D435I)
    assert (ctrl.interaction, ctrl.gain, ctrl.ema_alpha, ctrl.solve_unrounded) == (
        "current", 2.0, 0.0, True)
    assert ctrl.weighting == UNIFORM_WEIGHTING
    for bad, match in (({"weighting": {"use_trust": True}}, "unweighted"),
                       ({"ema_alpha": 0.5}, "EMA"),
                       ({"solve_unrounded": False}, "sub-pixel"),
                       ({"interaction": "average"}, "interaction"),
                       ({"gain": 2.0, "adaptive_gain": {"gain_at_zero": 4, "gain_at_infinity": 0.4,
                                                        "slope_at_zero": 30}}, "not both"),
                       ({"trust_gamma": 3}, "unknown controller keys")):
        cfg = {"controller": {**base["controller"], **bad}}
        with pytest.raises(ValueError, match=match):
            vi.VispIBVSController.from_config(cfg, intrinsics=D435I)
    adaptive = vi.VispIBVSController.from_config(
        {"controller": {"law": "visp", "adaptive_gain": {"gain_at_zero": 4.0,
                                                         "gain_at_infinity": 0.4,
                                                         "slope_at_zero": 30.0}}},
        intrinsics=D435I)
    assert isinstance(adaptive.gain_spec, vi.AdaptiveGain) and adaptive.gain == 4.0


def test_backend_visp_is_refused_without_the_bindings_and_auto_says_which_ran():
    if vi.visp_available():
        pytest.skip("ViSP is installed here; the refusal cannot be exercised")
    with pytest.raises(RuntimeError, match="BUILD_PYTHON_BINDINGS"):
        vi.VispIBVSController(intrinsics=D435I, backend="visp")
    assert vi.VispIBVSController(intrinsics=D435I, backend="auto").backend == "numpy"


# ------------------------------------------------------------ the arm registry
class _Stub:
    name = "dinov3_vits16"
    input_size = 512
    patch_size = 16
    grid_side = 32
    dim = 384
    device = "cpu"

    def encode(self, frame):  # pragma: no cover
        raise AssertionError("builds only")


def test_the_visp_arm_file_builds_and_is_labelled_as_visps_law():
    cfg = cfgmod.load(ARMS / "visp_ibvs.yaml")
    cfg["matcher"]["device"] = "cpu"
    arm = arms.build_arm(cfg, intrinsics=D435I, backbone=_Stub())
    assert isinstance(arm.controller, vi.VispIBVSController)
    assert arm.name == ("mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/"
                        "visp-current/cns_plateau")
    mean = arms.build_arm({**cfg, "controller": {**cfg["controller"], "interaction": "mean"}},
                          intrinsics=D435I, backbone=_Stub())
    assert mean.name.endswith("/visp-mean/cns_plateau")
    with pytest.raises(ValueError, match="controller.law"):
        arms.build_controller({"controller": {"law": "lqr"}}, intrinsics=D435I)
    plain = arms.build_controller({"controller": {"gain": 2.0}}, intrinsics=D435I)
    assert isinstance(plain, IBVSController)


def test_an_episode_on_the_mock_converges_under_visps_law():
    """The mock's exact matches + ViSP's law: 27 mm / 5.7 deg -> sub-0.1 mm in 200 steps,
    for each interaction-matrix choice (`desired` and `mean` take Z* from the goal depth
    the arm hands over at reset)."""
    scene = MockPlaneScene(render_scale=0.125)
    goal = scene.goal_pose
    start = {"start_id": "t0",
             "cam_pos": goal.position + np.array([0.02, -0.015, 0.01]),
             "cam_R_opt": (goal.rotation @ rodrigues([0, 0, 1.0], 0.1)).tolist()}
    for interaction in ("current", "desired", "mean"):
        cfg = cfgmod.resolve({
            "seed": 20260902,
            "matcher": {"variant": "mutual_nn_seeded_subsample", "num_landmarks": 24,
                        "device": "cpu"},
            "controller": {"law": "visp", "interaction": interaction, "gain": 2.0},
            "stop": {"rule": "cns_plateau"}, "episode": {"max_iter": 200},
            "scoring": {"success_rule": ["predicate_a"]}, "arm": {"kind": "oracle"}})
        matcher = PlaneTruthMatcher(scene, num_landmarks=24, seed=cfg["seed"])
        arm = arms.build_arm(cfg, intrinsics=scene.intrinsics, matcher=matcher)
        rec = run_episode(scene, arm, start, cfg)
        assert rec["final_translation_mm"] < 0.1, interaction
        assert rec["final_rotation_deg"] < 0.02, interaction
        assert rec["start_translation_mm"] == pytest.approx(26.9258, abs=1e-3)
        assert rec["weighting"] == vars(UNIFORM_WEIGHTING)
        assert "visp-" + interaction in rec["arm"]
