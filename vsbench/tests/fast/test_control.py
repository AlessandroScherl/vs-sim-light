# vs-sim copy: tests/fast/test_control.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""U4 golden tests: the solve, the weights, the stop rules, the recorded-telemetry parity.

Three tiers, all CPU and all seconds:

1. **Synthetic convergence** - a plane of 3-D points, a known camera offset, the
   loop closed with the old repo's integration convention
   (``free_flying_scene.integrate_optical_velocity``: ``p += R @ v_lin * dt``,
   ``R <- R @ Rodrigues``, ``dt = 1/60``). Sub-pixel reaches 1e-15 m; the
   published rounded path stalls at the pixel-quantisation floor, ~6e-4 m, which
   is the module's first fence turned into a number.
2. **Stops on synthetic traces** - each rule fires where the constants say and
   refuses to fire where they say it must not. `CnsPlateau`, protocol v2's scored
   stop, is additionally run **against CNS's own shipped code**, vendored at
   `tests/fixtures/reference/cns_stop_policy.py`: same traces, same firing step,
   on R12's thirteen named traces and on 200 random ones.
3. **Recorded-telemetry parity** - ``tests/fixtures/u4_telemetry_hollywood_w2_057.npz``,
   300 steps of a converged ``head_w_g3_axw@v11`` episode from the N2 admission
   gate (``u4_fixture_provenance.md``). Two statements, and the first is the
   stronger one:

   - against the **old code**, given the same rows,
     ``weighted_ls_solve`` and ``ibvs_core.solve_velocity`` agree to **0.0** -
     bit-identical, not "close" (``test_the_solve_is_bit_identical_to_the_old_one``,
     skipped where ``../genvs-sim`` is not checked out);
   - against the **record**, 100 % of steps within 1e-3 relative on the recorded
     weights (p50 1.3e-5, p99 2.1e-4) and 99.7 % when the weights are recomputed
     from the recorded trust.

   The record-level gap is the record's, not the code's. It is **not** pixel
   quantisation: this arm ran the rounded path, so the stored coordinates are
   exact integers, and a +-0.005 px perturbation would move the solve by 8.8e-4
   (p50) - seventy times what is observed. It is (a) the closed-form plane depth
   standing in for the depth map the controller actually sampled, worth up to
   2.7e-4 m of Z on the far steps where the camera is 339 mm out, and the solve
   moves 9.8e-4 (p50) per millimetre of Z; and (b) the 7-dp storage of the twist
   itself on the terminal steps, a 9.1e-5 relative floor at the episode's
   smallest commanded twist.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vsbench.vs.baselines.control import (
    CNS_STOP,
    DEC,
    DEFAULT_DT,
    DEPTH_FALLBACK_M,
    DIVERGENCE_MULT,
    FIELDED_WEIGHTING,
    MIN_MATCHES,
    PLATEAU,
    STOP_RULE_BUILDERS,
    UNIFORM_WEIGHTING,
    VELOCITY_QUIET,
    CapOnly,
    CnsPlateau,
    DecPlateau,
    DivergenceGuard,
    IBVSController,
    Intrinsics,
    StopObservation,
    StopRule,
    Twist,
    VelocityQuiet,
    Weighting,
    build_stop_rule,
    clean_depth,
    effective_support,
    interaction_matrix,
    normalised_coords,
    weighted_ls_solve,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
U4_NPZ = FIXTURES / "u4_telemetry_hollywood_w2_057.npz"
#: CNS's shipped stop policy, vendored (MIT) - the reference `CnsPlateau` is run against.
CNS_REFERENCE = FIXTURES / "reference" / "cns_stop_policy.py"
#: CNS's own control period, `cns/sim/environment.py:64`. Its 0.8 s hold is 41
#: intervals there and 49 at our 1/60.
DT_CNS = 1.0 / 50.0
N_LANDMARKS = 24
#: a fixed per-landmark spread with mean 1 - see `spread()`.
SPREAD = np.linspace(0.7, 1.3, N_LANDMARKS)

#: The fielded camera (`genvs-sim/configs/camera.yaml` -> `camera_info`).
INTR = Intrinsics(fx=1406.748094934653, fy=1406.748094934653, cx=960.0, cy=540.0,
                  width=1920, height=1080)
GAIN = 2.0

#: A re-solve is "equal" to the record within this relative error, on at least
#: this share of steps. Measured: 1.000 / 0.997 (see the module docstring).
PARITY_TOL = 1e-3
PARITY_SHARE = 0.99


# ------------------------------------------------------------------ synthetic scene

def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n < 1e-15:
        return np.eye(3)
    k = axis / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def plane_points(n: int = 5, half: float = 0.12, depth: float = 0.5) -> np.ndarray:
    """A square grid of 3-D points on a fronto-parallel plane at ``depth``,
    expressed in the GOAL camera frame - so the goal pose is the identity."""
    grid = np.linspace(-half, half, n)
    X, Y = np.meshgrid(grid, grid)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, depth)])


def project(points_w: np.ndarray, p: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection into the camera at pose ``(p, R)``; also returns each
    point's depth in that camera - the per-point Z the interaction matrix needs."""
    cam = (points_w - p) @ R
    px = np.column_stack([INTR.fx * cam[:, 0] / cam[:, 2] + INTR.cx,
                          INTR.fy * cam[:, 1] / cam[:, 2] + INTR.cy])
    return px, cam[:, 2]


def servo(*, solve_unrounded: bool, weighting: Weighting, steps: int = 900,
          ema_alpha: float = 0.0, offset=(0.02, -0.015, 0.03), roll_deg: float = 3.0):
    """Close the loop on the synthetic scene; returns (|t| m, rotation deg, debug)."""
    pts = plane_points()
    goal_px, _ = project(pts, np.zeros(3), np.eye(3))
    p = np.asarray(offset, dtype=np.float64)
    R = rodrigues(np.array([1.0, 0.7, -0.4]), np.deg2rad(roll_deg))
    ctrl = IBVSController(gain=GAIN, dt=DEFAULT_DT, intrinsics=INTR,
                          solve_unrounded=solve_unrounded, weighting=weighting,
                          ema_alpha=ema_alpha)
    ctrl.reset()
    debug: dict = {}
    for _ in range(steps):
        cur_px, z = project(pts, p, R)
        rows = {"pts_goal": goal_px, "pts_cur": cur_px,
                "conf": np.full(len(z), 0.9), "margin": np.full(len(z), 0.3)}
        twist, debug = ctrl.step(rows, z)
        v = twist.as_array()
        p = p + R @ v[:3] * DEFAULT_DT
        omega = v[3:]
        n = float(np.linalg.norm(omega))
        if n > 1e-12:
            R = R @ rodrigues(omega / n, n * DEFAULT_DT)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    return float(np.linalg.norm(p)), ang, debug


@pytest.mark.parametrize("name,weighting", [
    ("uniform", UNIFORM_WEIGHTING),
    ("fielded_trust3_margin_axw", FIELDED_WEIGHTING),
    ("trust_only", Weighting(use_margin=False, per_axis=False)),
])
def test_synthetic_scene_converges_below_a_micrometre(name, weighting):
    """Sub-pixel solve, per-point depth, 900 steps at gain 2 and dt = 1/60."""
    t_m, ang_deg, _ = servo(solve_unrounded=True, weighting=weighting)
    assert t_m < 1e-6, f"{name}: |t| = {t_m:.3e} m"
    assert ang_deg < 1e-4, f"{name}: rotation {ang_deg:.3e} deg"


def test_the_rounded_path_stalls_at_the_pixel_quantisation_floor():
    """The module's first fence, measured.

    Whole-pixel rounding is a ViT-VS quirk, not classical practice. It cannot
    reach a micrometre and never could: once every current pixel rounds onto its
    goal pixel the error is exactly zero and the camera freezes wherever it is,
    anywhere inside a pixel-sized basin - half a pixel at f = 1406.75 and Z = 0.5 m
    is 1.8e-4 m. Measured 6.0e-4 m against 2.6e-15 m sub-pixel, a factor of 2e11.
    This is why protocol v2 puts sub-pixel on every arm and `solve_unrounded`
    defaults to True.
    """
    rounded, ang_r, _ = servo(solve_unrounded=False, weighting=FIELDED_WEIGHTING)
    sub_pixel, _, _ = servo(solve_unrounded=True, weighting=FIELDED_WEIGHTING)
    assert rounded < 3e-3, f"the rounded path did not settle at all: {rounded:.3e} m"
    assert ang_r < 0.1
    assert rounded > 100 * sub_pixel, "rounding stopped costing anything - check the port"


def test_per_axis_weights_change_the_twist_but_not_the_fixed_point():
    """`_axw` is a per-axis noise model, so it steers differently and lands equal.

    Only differently when the residual has something for it to re-weight: on a
    pure translation from a fronto-parallel pose the geometry is symmetric and
    the y-row factor cancels out of the pseudo-inverse. With a 3 deg rotation in
    the pose it moves the twist by 2.1 % - and both still converge.
    """
    pts = plane_points()
    goal_px, _ = project(pts, np.zeros(3), np.eye(3))
    R = rodrigues(np.array([1.0, 0.7, -0.4]), np.deg2rad(3.0))
    cur_px, z = project(pts, np.array([0.02, -0.015, 0.03]), R)
    rows = {"pts_goal": goal_px, "pts_cur": cur_px, "conf": np.full(len(z), 0.9),
            "margin": np.full(len(z), 0.3)}
    plain = IBVSController(gain=GAIN, intrinsics=INTR,
                           weighting=Weighting(per_axis=False)).step(rows, z)[0]
    axw = IBVSController(gain=GAIN, intrinsics=INTR, weighting=FIELDED_WEIGHTING).step(rows, z)[0]
    rel = np.linalg.norm(plain.as_array() - axw.as_array()) / np.linalg.norm(plain.as_array())
    assert rel > 1e-3, f"the per-axis factor did nothing (rel {rel:.2e})"
    both = [servo(solve_unrounded=True, weighting=w)[0]
            for w in (Weighting(per_axis=False), FIELDED_WEIGHTING)]
    assert max(both) < 1e-6


# ------------------------------------------------------------------ the pure solve

def test_interaction_matrix_is_the_published_row_pair():
    s = np.array([[0.1, -0.2]])
    z = np.array([0.5])
    L = interaction_matrix(s, z)
    x, y, Z = 0.1, -0.2, 0.5
    assert L.shape == (2, 6)
    np.testing.assert_allclose(L[0], [-1 / Z, 0, x / Z, x * y, -(1 + x ** 2), y], rtol=0, atol=0)
    np.testing.assert_allclose(L[1], [0, -1 / Z, y / Z, 1 + y ** 2, -x * y, -x], rtol=0, atol=0)


def test_interaction_matrix_needs_one_depth_per_point():
    with pytest.raises(ValueError, match="one Z per point"):
        interaction_matrix(np.zeros((3, 2)), np.ones(2))


def test_uniform_weights_reproduce_the_unweighted_solve():
    """The least squares is scale-invariant, so only the weight SHAPE matters."""
    pts = plane_points()
    goal_px, _ = project(pts, np.zeros(3), np.eye(3))
    cur_px, z = project(pts, np.array([0.01, 0.02, -0.01]), np.eye(3))
    v0, _ = weighted_ls_solve(goal_px, cur_px, z, INTR, gain=GAIN, weights=None)
    for scale in (1.0, 7.3):
        v1, _ = weighted_ls_solve(goal_px, cur_px, z, INTR, gain=GAIN,
                                  weights=np.full(len(z), scale))
        np.testing.assert_allclose(v1, v0, rtol=1e-9, atol=1e-12)


def test_too_few_supporting_rows_gives_a_zero_twist():
    pts = plane_points(n=2)[:3]
    goal_px, _ = project(pts, np.zeros(3), np.eye(3))
    cur_px, z = project(pts, np.array([0.01, 0.0, 0.0]), np.eye(3))
    v, info = weighted_ls_solve(goal_px, cur_px, z, INTR, gain=GAIN)
    assert info["degenerate"] and np.all(v == 0) and np.isnan(info["feature_error_norm"])
    # weights can starve the solve too: only rows of non-zero weight count as support
    pts = plane_points()
    goal_px, _ = project(pts, np.zeros(3), np.eye(3))
    cur_px, z = project(pts, np.array([0.01, 0.0, 0.0]), np.eye(3))
    w = np.zeros(len(z))
    w[:MIN_MATCHES - 1] = 1.0
    v, info = weighted_ls_solve(goal_px, cur_px, z, INTR, gain=GAIN, weights=w)
    assert info["degenerate"] and info["n_support"] == MIN_MATCHES - 1 and np.all(v == 0)
    # and the controller survives a frame the matcher gave nothing for
    empty = {"pts_goal": np.zeros((0, 2)), "pts_cur": np.zeros((0, 2)),
             "conf": np.zeros(0), "margin": np.zeros(0)}
    twist, dbg = IBVSController(gain=GAIN, intrinsics=INTR).step(empty, np.zeros(0))
    assert np.all(twist.as_array() == 0) and dbg["degenerate"] and dbg["n_matches"] == 0


def test_the_solve_refuses_non_finite_or_negative_weights():
    pts = plane_points()
    goal_px, _ = project(pts, np.zeros(3), np.eye(3))
    cur_px, z = project(pts, np.array([0.01, 0.0, 0.0]), np.eye(3))
    for bad in (np.full(len(z), np.nan), -np.ones(len(z))):
        with pytest.raises(ValueError, match="finite and non-negative"):
            weighted_ls_solve(goal_px, cur_px, z, INTR, gain=GAIN, weights=bad)


def test_an_infinite_margin_never_reaches_the_solve():
    """The 0 x inf trap (S2.b lesson): `margins_for_matches` returns +inf for an
    all-excluded row and a rejected match has trust 0, so `trust**3 * margin`
    would be NaN and `sqrt(w)` would spread it over every row."""
    n = 8
    rows = {"pts_goal": np.zeros((n, 2)), "pts_cur": np.zeros((n, 2)),
            "conf": np.linspace(0.0, 1.0, n), "margin": np.full(n, np.inf)}
    w, _ = FIELDED_WEIGHTING(rows, image_wh=(1920, 1080))
    assert np.all(np.isfinite(w)), "an infinite margin poisoned the weights"
    assert w[0, 0] == 0.0, "trust 0 must still weigh nothing"


def test_effective_support_flattens_per_axis_weights():
    """`significance.effective_support` verbatim: (K, 2) reports up to 2K."""
    assert effective_support(np.ones(24)) == pytest.approx(24.0)
    assert effective_support(np.ones((24, 2))) == pytest.approx(48.0)
    assert effective_support(np.zeros(5)) == 0.0
    w = np.array([1.0, 0.0, 0.0, 0.0])
    assert effective_support(w) == pytest.approx(1.0)


def test_clean_depth_applies_the_published_validity_rule():
    z = clean_depth([0.5, 0.0, -1.0, 11.0, np.nan, np.inf, 9.9])
    np.testing.assert_allclose(z, [0.5, DEPTH_FALLBACK_M, DEPTH_FALLBACK_M, DEPTH_FALLBACK_M,
                                   DEPTH_FALLBACK_M, DEPTH_FALLBACK_M, 9.9])


def test_normalised_coords_and_intrinsics():
    np.testing.assert_allclose(normalised_coords([[INTR.cx, INTR.cy]], INTR), [[0.0, 0.0]])
    built = Intrinsics.from_camera_info({"f_x": 1.0, "f_y": 2.0, "c_x": 3.0, "c_y": 4.0,
                                         "width": 5, "height": 6})
    assert (built.fx, built.fy, built.cx, built.cy, built.width, built.height) == (
        1.0, 2.0, 3.0, 4.0, 5, 6)
    with pytest.raises(KeyError, match="f_x"):
        Intrinsics.from_camera_info({"f_y": 2.0})
    assert Intrinsics.from_camera_info({"width": 9}, fallback=INTR).fx == INTR.fx


def test_weighting_from_config_records_what_it_will_do():
    w = Weighting.from_config({"controller": {"weighting": {"trust_gamma": 1.0}}})
    assert w.trust_gamma == 1.0 and w.use_margin and w.per_axis
    with pytest.raises(ValueError, match="unknown controller.weighting keys"):
        Weighting.from_config({"controller": {"weighting": {"gamma": 3}}})
    with pytest.raises(ValueError, match="rows\\['margin'\\]"):
        Weighting()({"pts_cur": np.zeros((3, 2)), "conf": np.ones(3)}, image_wh=(16, 9))


# --------------------------------------------------------------------------- EMA

def test_ema_zero_is_a_bit_exact_no_op_and_a_positive_alpha_is_not():
    pts = plane_points()
    goal_px, _ = project(pts, np.zeros(3), np.eye(3))
    cur_px, z = project(pts, np.array([0.02, -0.015, 0.03]), np.eye(3))
    rows = {"pts_goal": goal_px, "pts_cur": cur_px, "conf": np.full(len(z), 0.9),
            "margin": np.full(len(z), 0.3)}
    plain = IBVSController(gain=GAIN, intrinsics=INTR, ema_alpha=0.0)
    smoothed = IBVSController(gain=GAIN, intrinsics=INTR, ema_alpha=0.5)
    first_p, dbg = plain.step(rows, z)
    first_s, _ = smoothed.step(rows, z)
    # ART-VS's direction: alpha weights the NEW twist, so step 1 is unsmoothed
    np.testing.assert_array_equal(first_p.as_array(), first_s.as_array())
    np.testing.assert_array_equal(first_p.as_array(), dbg["velocity_raw"])
    cur2, z2 = project(pts, np.array([0.01, -0.005, 0.01]), np.eye(3))
    rows2 = {**rows, "pts_cur": cur2}
    second_p, _ = plain.step(rows2, z2)
    second_s, dbg_s = smoothed.step(rows2, z2)
    assert not np.allclose(second_p.as_array(), second_s.as_array())
    np.testing.assert_allclose(second_s.as_array(),
                               0.5 * dbg_s["velocity_raw"] + 0.5 * first_s.as_array())
    smoothed.reset()                      # smoothing never crosses an episode
    again, _ = smoothed.step(rows, z)
    np.testing.assert_array_equal(again.as_array(), first_s.as_array())


def test_debug_carries_step_ms_n_eff_and_the_weights():
    pts = plane_points()
    goal_px, _ = project(pts, np.zeros(3), np.eye(3))
    cur_px, z = project(pts, np.array([0.02, 0.0, 0.0]), np.eye(3))
    rows = {"pts_goal": goal_px, "pts_cur": cur_px, "conf": np.full(len(z), 0.9),
            "margin": np.full(len(z), 0.3)}
    _, dbg = IBVSController(gain=GAIN, intrinsics=INTR).step(rows, z)
    assert dbg["step_ms"] > 0.0
    assert dbg["n_eff"] == pytest.approx(effective_support(dbg["weights"]))
    assert dbg["weights"].shape == (len(z), 2) and dbg["weight_source"] == "conf"
    assert dbg["step"] == 1 and dbg["n_matches"] == len(z) and not dbg["degenerate"]
    assert dbg["normalised_errors"].shape == (len(z),)
    rows_head = {**rows, "trust": np.full(len(z), 0.5)}
    assert IBVSController(gain=GAIN, intrinsics=INTR).step(rows_head, z)[1][
        "weight_source"] == "trust"


def test_twist_is_a_six_vector_in_the_optical_frame():
    t = Twist(np.arange(6.0))
    np.testing.assert_array_equal(t.linear, [0, 1, 2])
    np.testing.assert_array_equal(t.angular, [3, 4, 5])
    np.testing.assert_array_equal(np.asarray(t), np.arange(6.0))
    assert t.norm() == pytest.approx(float(np.linalg.norm(np.arange(6.0))))
    assert np.all(Twist.zero().as_array() == 0)
    with pytest.raises(ValueError, match="6 numbers"):
        Twist(np.zeros(5))


# -------------------------------------------------------------------- stop rules

def obs(step: int, *, feat=float("nan"), vnorm=float("nan"), errs=None, ids=None, dt=DEFAULT_DT):
    return StopObservation(step=step, feature_error_norm=feat, velocity_norm=vnorm,
                           normalised_errors=errs, landmark_ids=ids, t=step * dt)


def test_cap_only_never_stops():
    rule = CapOnly()
    rule.reset()
    assert not any(rule.update(obs(i, feat=1e-9, vnorm=0.0)).would_stop for i in range(1, 700))


def test_dec_plateau_fires_where_the_constants_say_and_then_decays_lambda():
    """window 200 / 5 % / sustain 10 needs 2*window samples, then 10 in a row.

    The trigger is `detail["plateau_step"]`, not `would_stop`: under `dec` the rule's
    verdict IS the published velocity-quiet criterion evaluated on the decayed twist
    (review R13, C1), and this trace never goes quiet (its velocity is NaN).
    """
    rule = DecPlateau()
    rule.reset()
    fired = None
    for i in range(1, 700):
        verdict = rule.update(obs(i, feat=1.0))
        if verdict.detail["plateau_step"] is not None and fired is None:
            fired = verdict.detail["plateau_step"]
        assert not verdict.would_stop            # no quiet twist -> no stop, ever
    assert fired == 2 * int(PLATEAU["window"]) + int(PLATEAU["sustain"]) - 1 == 409
    # the response: lambda decays by rho per step down to the floor
    rule = DecPlateau()
    rule.reset()
    scales = [rule.update(obs(i, feat=1.0)).lambda_scale for i in range(1, 700)]
    assert scales[407] == 1.0                              # step 408, before the trigger
    assert scales[408] == pytest.approx(DEC["rho"])        # step 409, the first decay
    assert scales[-1] == pytest.approx(DEC["floor"])


def test_dec_plateau_never_fires_on_a_monotone_descent():
    rule = DecPlateau()
    rule.reset()
    assert not any(rule.update(obs(i, feat=float(np.exp(-i / 50.0)))).detail["plateau_step"]
                   for i in range(1, 700))


def test_the_dec_stop_is_the_published_quiet_criterion_under_the_decayed_lambda():
    """C1 (review R13): what a `dec`-scored `stop_step` means, and what it does not.

    `baseline_check.py` scaled the twist by the schedule (L1786-1789) and then ran the
    published quiet check on the COMMANDED twist (L1792-1797) into `stop_*`, while the
    plateau trigger went to `stop_trigger_step`. On the stored N3.7b records the two
    are 26 to 126 steps apart (median 53, n = 96), so scoring on the trigger would read
    a different pose than every stored number.

    Here: a plateaued error with a raw velocity of 1.5e-3 - above the 1e-4 stop on its
    own, and only quiet once lambda has decayed past ~1/15.
    """
    rule = DecPlateau()
    rule.reset()
    trigger = stop = None
    for i in range(1, 700):
        v = rule.update(obs(i, feat=1.0, vnorm=1.5e-3))
        if trigger is None:
            trigger = v.detail["plateau_step"]
        if v.would_stop and stop is None:
            stop = v.fired_step
    assert trigger == 409
    assert stop is not None and stop > trigger          # strictly after, as in the records
    # the decay is applied on the trigger step itself, so step `trigger + k` runs at
    # rho^(k+1); 0.9^26 * 1.5e-3 < 1e-4 first at k = 25, then 8 steps of patience
    import math
    k = math.ceil(math.log(VELOCITY_QUIET["eps"] / 1.5e-3) / math.log(DEC["rho"])) - 1
    assert stop == trigger + k + int(VELOCITY_QUIET["patience"]) - 1 == 441
    # and without the decay the same trace never goes quiet
    quiet = VelocityQuiet()
    quiet.reset()
    assert not any(quiet.update(obs(i, vnorm=1.5e-3)).would_stop for i in range(1, 700))


def test_every_stop_rule_reports_its_effective_constants():
    """C6: a record must be able to say what scored it, not only what was overridden."""
    assert CapOnly().params == {}
    assert VelocityQuiet().params == {"eps": VELOCITY_QUIET["eps"],
                                      "patience": int(VELOCITY_QUIET["patience"])}
    dec = DecPlateau(rho=0.5)
    assert dec.params == {"rho": 0.5, "floor": DEC["floor"], "window": PLATEAU["window"],
                          "rel_tol": PLATEAU["rel_tol"], "sustain": PLATEAU["sustain"],
                          "quiet_eps": VELOCITY_QUIET["eps"],
                          "quiet_patience": int(VELOCITY_QUIET["patience"])}
    cns = CnsPlateau(waiting_time=0.4)
    assert cns.params == {**CNS_STOP, "waiting_time": 0.4, "dt": DEFAULT_DT}
    assert DivergenceGuard().params == {"mult": DIVERGENCE_MULT}


def test_dec_plateau_fires_only_once_the_descent_has_left_the_window():
    """||e|| = 10 for 200 steps, then 2 for ever.

    After step t the buffer holds steps t-399..t, so the older window still
    carries k = 600 - t of the loud steps and its mean is 2 + 0.04k. The 5 %
    test passes once 0.04k < 0.05(2 + 0.04k), i.e. k <= 2, i.e. t >= 598;
    sustain 10 then fires it at 607.
    """
    rule = DecPlateau()
    rule.reset()
    fired = None
    for i in range(1, 800):
        feat = 10.0 if i <= 200 else 2.0
        step = rule.update(obs(i, feat=feat)).detail["plateau_step"]
        if step is not None and fired is None:
            fired = step
    assert fired == 607


def test_velocity_quiet_fires_after_eight_quiet_steps():
    rule = VelocityQuiet()
    assert rule.eps == VELOCITY_QUIET["eps"] and rule.patience == 8
    rule.reset()
    for i in range(1, 8):
        assert not rule.update(obs(i, vnorm=1e-9)).would_stop
    assert rule.update(obs(8, vnorm=1e-9)).would_stop
    # a single loud step resets the run
    rule.reset()
    for i in range(1, 8):
        rule.update(obs(i, vnorm=1e-9))
    rule.update(obs(8, vnorm=1.0))
    assert not rule.update(obs(9, vnorm=1e-9)).would_stop


# --- CnsPlateau against CNS's own code -------------------------------------------
#
# The first U4 implementation was written from a prose description of CNS's stop
# policy and got five things wrong (review R12): it tracked every new minimum
# instead of only 10 % improvements, accrued hold time on the new-minimum step,
# inverted the crossing interpolation, trimmed the unweighted errors, and kept
# absent landmarks in the mean. On a settling error it fired 90-185 steps early
# and on a slow creep it fired where CNS never does. So the tests below no longer
# assert a reading: they run `vsbench.vs.baselines.control.CnsPlateau` and the vendored
# `tests/fixtures/reference/cns_stop_policy.py` on the same traces and require the
# same answer.


@pytest.fixture(scope="module")
def cns_reference():
    """CNS's shipped `PixelStopPolicy`, loaded by path from the vendored file.

    Deliberately not importable as a module of ours: it is third-party MIT code
    kept for one purpose, and `tests/fixtures/reference/` is excluded from ruff.
    """
    spec = importlib.util.spec_from_file_location("cns_stop_policy_reference", CNS_REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_vendored_cns_reference_keeps_its_licence_and_its_independence():
    """Third-party code, kept as such.

    The MIT notice must travel with it, and it must not grow a dependency: it is
    loaded by path in these tests precisely so that CNS's rule is checked against
    CNS's code and not against our reading of it.
    """
    text = CNS_REFERENCE.read_text(encoding="utf-8")
    assert "MIT License" in text and "Copyright (c) 2023 hhcaz" in text
    assert "github.com/hhcaz/CNS" in text
    imports = [ln.strip() for ln in text.splitlines()
               if ln.startswith(("import ", "from ")) and "import" in ln]
    assert imports == ["import numpy as np"], imports


class _CnsData:
    """The three attributes `PixelStopPolicy.calculate_error` reads off a torch graph."""

    def __init__(self, errors, node_mask):
        n = len(errors)
        self.pos_cur = torch.tensor(np.column_stack([np.asarray(errors, float), np.zeros(n)]))
        self.pos_tar = torch.zeros(n, 2, dtype=torch.float64)
        self.node_mask = torch.tensor(np.asarray(node_mask, bool))


class _Silence:
    """CNS's policy prints a line per step; the vendored file stays verbatim, so mute it."""

    def write(self, *_args):
        return None

    def flush(self):
        return None


def run_reference(module, errors_per_step, masks=None, dt=DT_CNS):
    """CNS's benchmark loop: `stop_policy(data, env.steps * env.dt)`, 0-based clock."""
    policy = module.PixelStopPolicy(waiting_time=0.8, conduct_thresh=0.01)
    policy.reset()
    n = len(errors_per_step[0])
    with contextlib.redirect_stdout(_Silence()):
        for k, errors in enumerate(errors_per_step):
            mask = np.ones(n, bool) if masks is None else masks[k]
            if policy(_CnsData(errors, mask), k * dt):
                return k + 1
    return None


def run_port(errors_per_step, ids=None, dt=DT_CNS):
    """The same loop through `CnsPlateau`, letting it derive its own timestamps."""
    rule = CnsPlateau(dt=dt)
    rule.reset()
    for k, errors in enumerate(errors_per_step):
        obs = StopObservation(step=k + 1, normalised_errors=np.asarray(errors),
                              landmark_ids=None if ids is None else ids[k])
        if rule.update(obs).would_stop:
            return k + 1
    return None


def spread(series, jitter=0.0, seed=7):
    """A scalar error series -> per-landmark vectors with a fixed spread (mean 1).

    Spread is not cosmetic: CNS trims the WEIGHTED errors with a strict `<`, so an
    all-equal vector trims everything and reports 0.0 (its own quirk, tested below).
    """
    rng = np.random.default_rng(seed)
    return [b * SPREAD * (1 + jitter * rng.standard_normal(N_LANDMARKS)) for b in series]


#: (trace, dt, the step CNS fires at). Every number is what the **vendored** policy
#: does on the trace defined right here, and is re-derived from it on every run - so
#: a wrong expectation fails the reference assertion, not just the port one. The
#: deterministic traces reproduce review R12's numbers exactly; the two jittered
#: ones draw from a different RNG stream than R12's script did and are pinned here.
CNS_TRACES = {
    "step down 0.05 -> 0.005 at step 11, 60 Hz": (
        spread([0.05] * 10 + [0.005] * 300), DEFAULT_DT, 60),
    "the same at CNS's own 50 Hz": (
        spread([0.05] * 10 + [0.005] * 300), DT_CNS, 52),
    "slow creep, 1 %/step from 0.009": (
        spread([0.009 * 0.99 ** i for i in range(400)]), DT_CNS, None),
    "slower creep, 0.3 %/step": (
        spread([0.009 * 0.997 ** i for i in range(600)]), DT_CNS, None),
    "fast descent, 15 %/step": (
        spread([0.008 * 0.85 ** i for i in range(400)]), DT_CNS, None),
    "noisy plateau +- 20 %": (spread([0.004] * 400, jitter=0.2), DT_CNS, 42),
    "noisy plateau +- 5 %": (spread([0.004] * 400, jitter=0.05), DT_CNS, 42),
    "exponential settle, tau 30": (
        spread([0.003 + 0.006 * np.exp(-i / 30) for i in range(600)]), DT_CNS, 135),
    "exponential settle, tau 60": (
        spread([0.003 + 0.006 * np.exp(-i / 60) for i in range(800)]), DT_CNS, 165),
    "exponential settle, tau 150": (
        spread([0.003 + 0.006 * np.exp(-i / 150) for i in range(1500)]), DT_CNS, 227),
    "exponential settle + 3 % noise, tau 40": (
        spread([0.003 + 0.006 * np.exp(-i / 40) for i in range(800)], jitter=0.03), DT_CNS, 176),
    "plateau with one outlier landmark": (
        [np.r_[0.004 * SPREAD[:-1], 0.05]] * 200, DT_CNS, 42),
    "plateau with three outliers (the trim keeps one)": (
        [np.r_[0.004 * SPREAD[:-3], 0.05, 0.05, 0.05]] * 200, DT_CNS, 42),
}


@pytest.mark.parametrize("name", list(CNS_TRACES))
def test_cns_plateau_matches_the_vendored_policy(cns_reference, name):
    """Step for step against CNS's shipped code, on R12's traces.

    The two that matter most: a **slow creep** must never fire (CNS's minimum only
    moves on a 10 % improvement, so the band keeps being left behind), and an
    **exponential settle** fires at 135 / 165 / 227 for tau = 30 / 60 / 150 - deep
    into the plateau, not during the descent. The pre-correction port fired at
    step 42 on every one of those four.
    """
    series, dt, expected = CNS_TRACES[name]
    reference = run_reference(cns_reference, series, dt=dt)
    assert reference == expected, f"the vendored CNS policy fired at {reference}, not {expected}"
    assert run_port(series, dt=dt) == reference


def test_cns_plateau_matches_the_vendored_policy_when_landmarks_go_missing(cns_reference):
    """A landmark that drops out decays in the background; only the ones matched
    this frame enter the mean (`stop_policy.py:115-118`)."""
    base = [0.004 * SPREAD] * 200
    for label, gone in (("half of them for steps 31-50", lambda m, k: m[::2].fill(False)
                         if 30 <= k < 50 else None),
                        ("twelve flickering every other frame", lambda m, k: m[:12].fill(False)
                         if k >= 5 and k % 2 else None)):
        masks = []
        for k in range(len(base)):
            mask = np.ones(N_LANDMARKS, bool)
            gone(mask, k)
            masks.append(mask)
        ids = [np.arange(N_LANDMARKS)[m] for m in masks]
        seen = [e[m] for e, m in zip(base, masks, strict=True)]
        assert run_port(seen, ids) == run_reference(cns_reference, base, masks), label


def test_cns_plateau_matches_the_vendored_policy_on_random_traces(cns_reference):
    """200 random settle-and-flicker traces: the same firing step, every time."""
    rng = np.random.default_rng(20260903)
    for _ in range(200):
        length = int(rng.integers(60, 300))
        base, decay, jitter = rng.uniform(0.002, 0.02), rng.uniform(0.97, 1.0), rng.uniform(0, 0.25)
        series = [np.clip(base * decay ** i * (1 + jitter * rng.standard_normal(N_LANDMARKS))
                          * SPREAD, 1e-6, None) for i in range(length)]
        masks = []
        for _step in range(length):
            mask = rng.random(N_LANDMARKS) > rng.uniform(0.0, 0.3)
            masks.append(mask if mask.sum() >= 4 else np.ones(N_LANDMARKS, bool))
        ids = [np.arange(N_LANDMARKS)[m] for m in masks]
        seen = [e[m] for e, m in zip(series, masks, strict=True)]
        assert run_port(seen, ids) == run_reference(cns_reference, series, masks)


def test_cns_error_signal_matches_the_vendored_calculate_error(cns_reference):
    """The recency weights, in numbers.

    A landmark's first sighting weighs 20; one that missed a frame comes back at
    `0.95 * 20 + 1 = 19.05`, not at 20 - the old port reset every sighting to 20
    and kept absent landmarks in the mean, which moved this signal by 20 %.
    """
    policy = cns_reference.PixelStopPolicy(0.8)
    policy.reset()
    rule = CnsPlateau(dt=DT_CNS)
    rule.reset()
    errors = np.linspace(0.002, 0.006, N_LANDMARKS)
    ids = np.arange(N_LANDMARKS)
    present = np.ones(N_LANDMARKS, bool)
    missing = present.copy()
    missing[:8] = False                       # the eight smallest errors vanish
    with contextlib.redirect_stdout(_Silence()):
        full = policy.calculate_error(_CnsData(errors, present))
        gone = policy.calculate_error(_CnsData(errors, missing))
        back = policy.calculate_error(_CnsData(errors, present))
    assert rule.error(errors, ids) == pytest.approx(full, rel=1e-6)
    assert rule.error(errors[missing], ids[missing]) == pytest.approx(gone, rel=1e-6)
    assert rule.error(errors, ids) == pytest.approx(back, rel=1e-6)
    assert (full, gone, back) == pytest.approx((0.003739, 0.004522, 0.003760), abs=5e-7)
    assert [rule._weight[i] for i in range(8)] == pytest.approx([19.05] * 8)
    assert [rule._weight[i] for i in range(8, N_LANDMARKS)] == pytest.approx([20.0] * 16)


def test_cns_hold_accounting_matches_the_vendored_interpolation(cns_reference):
    """The crossing arithmetic on its own: a scalar error straight into the state machine.

    `[0.004, 0.004, 0.006]` crosses the band (0.0049) late in the second interval,
    so CNS credits 0.020 + 0.009 s; the pre-correction port credited the time spent
    ABOVE the band instead and got 0.031.
    """

    class ScalarReference(cns_reference.ErrorHoldingStopPolicy):
        def calculate_error(self, data):
            return float(data)

    class ScalarPort(CnsPlateau):
        def error(self, normalised_errors, landmark_ids=None):
            return float(np.asarray(normalised_errors).reshape(-1)[0])

    cases = {
        "up-crossing, previously well inside": ([0.004, 0.004, 0.0060], 0.029),
        "up-crossing, previously at the edge": ([0.004, 0.00489, 0.0090], 0.02005),
        "down-crossing back into the band": ([0.004, 0.006, 0.0040], 0.018),
        "five steps inside the band": ([0.004] + [0.0044] * 5, 0.100),
        "a new low that is not a 10 % improvement": ([0.004, 0.0035, 0.0035], 0.020),
        "a new low that is": ([0.004, 0.0030, 0.0030], 0.020),
    }
    for label, (series, expected) in cases.items():
        reference = ScalarReference(0.8, 0.01)
        reference.reset()
        port = ScalarPort(dt=DT_CNS)
        port.reset()
        with contextlib.redirect_stdout(_Silence()):
            for k, err in enumerate(series):
                reference(err, k * DT_CNS)
                port.update(StopObservation(step=k + 1, normalised_errors=np.array([err])))
        assert port._accum == pytest.approx(reference.accum_time, abs=1e-12), label
        assert port._accum == pytest.approx(expected, abs=1e-5), label


def test_cns_trims_all_equal_errors_away_entirely(cns_reference):
    """CNS's own quirk, kept: the trim is `error < percentile(error, 90)`, strictly,
    so a perfectly uniform error vector keeps nothing and the signal is 0.0."""
    equal = np.full(N_LANDMARKS, 0.05)
    with contextlib.redirect_stdout(_Silence()):
        reference = cns_reference.PixelStopPolicy(0.8)
        reference.reset()
        assert reference.calculate_error(_CnsData(equal, np.ones(N_LANDMARKS, bool))) == 0.0
    rule = CnsPlateau()
    rule.reset()
    assert rule.error(equal) == 0.0


def test_cns_plateau_stays_disarmed_above_the_gate():
    rule = CnsPlateau()
    rule.reset()
    errors = 2 * CNS_STOP["conduct_thresh"] * SPREAD
    verdicts = [rule.update(StopObservation(step=i, normalised_errors=errors))
                for i in range(1, 300)]
    assert not any(v.would_stop for v in verdicts)
    assert all(v.detail["armed"] is False for v in verdicts)


def test_cns_plateau_needs_ids_that_look_like_landmarks():
    rule = CnsPlateau()
    rule.reset()
    with pytest.raises(ValueError, match="LINEAR"):
        rule.error(np.ones(3), np.zeros((3, 2)))          # the (K, 2) goal_patch column
    with pytest.raises(ValueError, match="duplicate landmark ids"):
        rule.error(np.ones(3), np.array([4, 4, 7]))


def test_every_stop_rule_survives_a_degenerate_step():
    """A frame with fewer than four matches is routine on the classical arms.

    CNS resets and reports no stop (`stop_policy.py:26-29`); the first port raised
    `ValueError`, which would have ended any `cns_plateau`-scored episode that hit
    one. `IBVSController.step` reports exactly that state - a NaN feature error and
    no per-landmark errors.
    """
    rows = {"pts_goal": np.zeros((2, 2)), "pts_cur": np.ones((2, 2)),
            "conf": np.ones(2), "margin": np.ones(2)}
    _, dbg = IBVSController(gain=GAIN, intrinsics=INTR).step(rows, np.full(2, 0.5))
    assert dbg["degenerate"] and dbg["normalised_errors"] is None
    obs = StopObservation.from_debug(dbg, twist=Twist.zero())
    for name, builder in STOP_RULE_BUILDERS.items():
        rule = builder()
        rule.reset()
        assert rule.update(obs).would_stop is False, name
    guard = DivergenceGuard()
    guard.reset()
    assert guard.update(obs).would_stop is False


def test_divergence_guard_trips_at_three_times_the_first_finite_error():
    guard = DivergenceGuard()
    guard.reset()
    assert not guard.update(obs(1, feat=float("nan"))).would_stop
    assert not guard.update(obs(2, feat=0.0)).would_stop      # 0 is not an anchor either
    for i, feat in enumerate([1.0, 2.0, 2.999], start=3):
        assert not guard.update(obs(i, feat=feat)).would_stop
    assert guard.initial == 1.0
    verdict = guard.update(obs(6, feat=3.001))
    assert verdict.would_stop and verdict.fired_step == 6
    assert verdict.detail["reason"] == "diverged_feature"


def test_the_registry_maps_the_config_enum_and_passes_its_constants():
    from vsbench.vs.bench.config import STOP_RULES as ENUM

    for rule in ENUM:
        built = build_stop_rule({"stop": {"rule": rule}})
        assert isinstance(built, StopRule) and built.name == rule
    assert build_stop_rule({"stop": {"rule": "dec_plateau", "rho": 0.5}}).rho == 0.5
    assert build_stop_rule({"stop": {"rule": "cns_plateau", "waiting_time": 0.4}}).const[
        "waiting_time"] == 0.4
    with pytest.raises(ValueError, match="stop.rule"):
        build_stop_rule({"stop": {"rule": "whenever"}})
    with pytest.raises(ValueError, match="unknown CnsPlateau constants"):
        build_stop_rule({"stop": {"rule": "cns_plateau", "wait": 0.4}})


def test_a_stop_observation_carries_nothing_the_controller_cannot_see():
    """The GT-blind rule as a type: no pose, no distance, no scene."""
    fields = set(StopObservation.__dataclass_fields__)
    # `pixel_error_mean` (2026-10-05, the ART-VS visual-only stop): the mean pixel distance
    # of the controller's OWN points - controller-visible, like every other field here
    assert fields == {"step", "feature_error_norm", "velocity_norm", "normalised_errors",
                      "landmark_ids", "t", "pixel_error_mean"}
    dbg = {"step": 7, "feature_error_norm": 0.5, "normalised_errors": np.zeros(3)}
    built = StopObservation.from_debug(dbg, twist=Twist(np.array([3.0, 4.0, 0, 0, 0, 0])))
    assert built.step == 7 and built.velocity_norm == pytest.approx(5.0)
    # CNS's clock origin: its first call carries t = 0, so a 1-based step is (step-1)*dt
    assert built.t == pytest.approx(6 * DEFAULT_DT)
    assert np.isnan(built.pixel_error_mean)              # no points in the debug: NaN
    pts = {"pts_goal": np.array([[0.0, 0.0], [10.0, 10.0]]),
           "pts_cur": np.array([[3.0, 4.0], [10.0, 10.0]])}
    assert StopObservation.from_debug({**dbg, **pts}).pixel_error_mean == pytest.approx(2.5)


# ------------------------------------------------------- recorded-telemetry parity

@pytest.fixture(scope="module")
def telemetry():
    if not U4_NPZ.is_file():                      # pragma: no cover - fixture missing
        pytest.skip(f"{U4_NPZ.name} is not present; make it with u4_make_control_fixture.py")
    data = np.load(U4_NPZ, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data, meta


def _fixture_intrinsics(meta) -> Intrinsics:
    return Intrinsics(fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"],
                      width=meta["width"], height=meta["height"])


def test_the_fixture_is_the_episode_it_claims_to_be(telemetry):
    data, meta = telemetry
    assert (meta["scene"], meta["arm"], meta["episode"]) == (
        "hollywood", "head_w_g3_axw@v11", "w2_057")
    assert data["pts_goal"].shape == (300, 24, 2)
    assert meta["conf_gamma"] == 3.0 and meta["axis_weighting"] and not meta["margin_weighting"]
    # the two facts the parity silently leans on: the arm ran the ROUNDED path (so
    # the stored pixels are integers and lossless), and no lambda schedule was
    # armed (so `v_recorded` is the law's own twist, not a scaled one)
    assert meta["solve_unrounded"] is False
    assert meta["stop_mode"] is None and meta["lam_in_record"] is False
    # the closed-form plane depth agrees with the median depth the station logged
    assert meta["zmed_reconstruction_max_err_m"] < 1e-3
    np.testing.assert_allclose(np.median(data["depth"], axis=1), data["z_med_recorded"],
                               atol=1e-3)


def test_the_solve_is_bit_identical_to_the_old_one(telemetry):
    """The real parity statement: not "within the record's noise" but **equal**.

    Loads ``genvs-sim@legacy-pre-cleanup``'s ``ibvs_core.solve_velocity`` by path -
    the frozen old function itself, not a transcription - and runs it beside
    :func:`weighted_ls_solve` on all 300 fixture steps. Measured max |dv| = 0.0.
    Skipped where the old repo is not checked out beside this one, which is the
    only reason the record-level tests below exist at all.
    """
    old_path = REPO_ROOT.parent / "genvs-sim" / "src" / "control" / "ibvs_core.py"
    if not old_path.is_file():
        pytest.skip(f"the frozen old repo is not at {old_path.parent.parent.parent}")
    spec = importlib.util.spec_from_file_location("ibvs_core_frozen", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules["ibvs_core_frozen"] = old        # its @dataclass needs to find itself
    try:
        spec.loader.exec_module(old)
        data, meta = telemetry
        ratio = meta["width"] / meta["height"]
        old_intr = old.Intrinsics(fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"],
                                  width=meta["width"], height=meta["height"])
        old_cfg = old.IBVSConfig(gain_lambda=meta["gain"], damping=0.0,
                                 min_matches=MIN_MATCHES)
        new_intr = _fixture_intrinsics(meta)
        worst = 0.0
        for i in range(len(data["step_index"])):
            w = data["w_recorded"][i]
            weights = np.column_stack([w, w * ratio * ratio])
            args = (data["pts_goal"][i], data["pts_cur"][i], data["depth"][i])
            v_new, _ = weighted_ls_solve(*args, new_intr, gain=meta["gain"], weights=weights)
            v_old, _ = old.solve_velocity(*args, old_intr, old_cfg, weights=weights)
            worst = max(worst, float(np.max(np.abs(v_new - v_old))))
        assert worst == 0.0, f"the port is no longer bit-identical to the old law: {worst:.3e}"
    finally:
        sys.modules.pop("ibvs_core_frozen", None)


def test_the_solve_reproduces_the_recorded_twist(telemetry):
    """The parity number: the new `weighted_ls_solve` on the recorded rows and
    the recorded weights returns the twist the station commanded."""
    data, meta = telemetry
    intr = _fixture_intrinsics(meta)
    ratio = meta["width"] / meta["height"]
    rel = []
    for i in range(len(data["step_index"])):
        w = data["w_recorded"][i]
        v, _ = weighted_ls_solve(data["pts_goal"][i], data["pts_cur"][i], data["depth"][i],
                                 intr, gain=meta["gain"],
                                 weights=np.column_stack([w, w * ratio * ratio]))
        recorded = data["v_recorded"][i]
        rel.append(np.linalg.norm(v - recorded) / np.linalg.norm(recorded))
    rel = np.asarray(rel)
    assert np.mean(rel < PARITY_TOL) >= PARITY_SHARE, (
        f"only {np.mean(rel < PARITY_TOL):.3f} of steps within {PARITY_TOL} "
        f"(p99 {np.percentile(rel, 99):.2e})")
    assert np.percentile(rel, 99) < 5e-4          # measured 2.1e-4


def test_the_controller_reproduces_the_recorded_twist_from_the_recorded_trust(telemetry):
    """The whole path: rows in, weights recomputed by `Weighting`, twist out.

    Looser than re-solving with the recorded weights because the record stores the
    trust at 4 dp and the weight is its cube: 1.5e-4 of weight, on top of the depth
    reconstruction and the 7-dp twist. Measured 99.7 % of steps within 1e-3.
    """
    data, meta = telemetry
    ctrl = IBVSController(gain=meta["gain"], intrinsics=_fixture_intrinsics(meta),
                          solve_unrounded=True,
                          weighting=Weighting(trust_gamma=meta["conf_gamma"],
                                              use_margin=False, per_axis=True))
    ctrl.reset()
    rel, feat_err = [], []
    for i in range(len(data["step_index"])):
        rows = {"pts_goal": data["pts_goal"][i], "pts_cur": data["pts_cur"][i],
                "conf": data["conf"][i]}
        twist, dbg = ctrl.step(rows, data["depth"][i])
        recorded = data["v_recorded"][i]
        rel.append(np.linalg.norm(twist.as_array() - recorded) / np.linalg.norm(recorded))
        feat_err.append(abs(dbg["feature_error_norm"] - data["feature_error_recorded"][i]))
    rel = np.asarray(rel)
    assert np.mean(rel < PARITY_TOL) >= PARITY_SHARE, (
        f"only {np.mean(rel < PARITY_TOL):.3f} of steps within {PARITY_TOL}")
    # the raw ||e|| the stop rules read is reproduced to the record's own 6 dp
    assert max(feat_err) < 1e-6


def test_the_recorded_weight_is_trust_cubed(telemetry):
    """Pins the fielded weighting formula: `_w` + `_g3` and nothing else.

    The record stores the trust at 4 dp and the weight at 5 dp, so the bar is
    2e-4 absolute (measured 1.5e-4). gamma = 1, 2 or 4 miss by 0.10 to 0.38.
    """
    data, _ = telemetry
    trust = np.clip(data["conf"], 0.0, None)
    assert np.abs(trust ** 3 - data["w_recorded"]).max() < 2e-4
    for gamma in (1.0, 2.0, 4.0):
        assert np.abs(trust ** gamma - data["w_recorded"]).max() > 1e-2


def test_rounding_is_a_no_op_on_the_recorded_pixels(telemetry):
    """The fixture's arm ran the rounded path, so its stored points are already
    integers - which makes the two paths provably identical on this fixture."""
    data, meta = telemetry
    for key in ("pts_goal", "pts_cur"):
        assert np.array_equal(data[key], np.round(data[key])), key
    rows = {"pts_goal": data["pts_goal"][0], "pts_cur": data["pts_cur"][0],
            "conf": data["conf"][0]}
    weighting = Weighting(trust_gamma=3.0, use_margin=False, per_axis=True)
    intr = _fixture_intrinsics(meta)
    twists = [IBVSController(gain=meta["gain"], intrinsics=intr, solve_unrounded=ur,
                             weighting=weighting).step(rows, data["depth"][0])[0].as_array()
              for ur in (True, False)]
    np.testing.assert_array_equal(*twists)


# --------------------------------- the 2026-10-05 knobs (the faithful ART-VS arm)
ARM_FILES = Path(__file__).resolve().parents[2] / "configs" / "arms"


def test_defaults_reproduce_the_old_controller(telemetry):
    """Additivity (spec 4.9 item 3): every arm file that builds the bench's law gets the
    controller it always got - the quantisation `solve_unrounded` implies, no cap, a zero
    twist on a degenerate frame, the 0.5 m depth fallback, nothing extra on the record -
    and a stop observation of the recorded steps keeps every earlier field."""
    from vsbench.vs.bench import config as cfgmod
    from vsbench.vs.bench.arms import depth_at

    for path in sorted(ARM_FILES.glob("*.yaml")):
        cfg = cfgmod.load(path)
        ctrl = cfg.get("controller") or {}
        if cfg["matcher"]["variant"] in ("cns_v1", "dinobot") or ctrl.get("law") == "visp":
            continue                                   # not the bench's IBVS law
        if path.name == "artvs_roi_sim.yaml":
            continue                                   # the new arm states its knobs
        c = IBVSController.from_config(cfg, intrinsics=INTR)
        su = bool(ctrl.get("solve_unrounded", True))
        assert c.pixel_quantisation == ("none" if su else "round"), path.name
        assert (c.max_velocity, c.degenerate_policy, c.depth_fallback_m) == (
            None, "zero", DEPTH_FALLBACK_M), path.name
        assert c.solve_unrounded is su and c.record_fields() == {}, path.name
    z = np.array([[0.0, 0.3], [12.0, np.nan]])
    assert np.array_equal(depth_at(z, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                                                [1.0, 1.0]])), [0.5, 0.3, 0.5, 0.5])
    data, meta = telemetry
    ctrl = IBVSController(gain=meta["gain"], intrinsics=_fixture_intrinsics(meta),
                          solve_unrounded=False, weighting=UNIFORM_WEIGHTING)
    for t in range(0, len(data["pts_goal"]), 7):
        tw, dbg = ctrl.step({"pts_goal": data["pts_goal"][t], "pts_cur": data["pts_cur"][t]},
                            data["depth"][t])
        assert not ({"degenerate_hold", "velocity_capped", "quantise"} & set(dbg))
        obs = StopObservation.from_debug(dbg, twist=tw, dt=DEFAULT_DT)
        assert (obs.step, obs.feature_error_norm, obs.landmark_ids) == (
            dbg["step"], dbg["feature_error_norm"], None)
        assert obs.velocity_norm == float(np.linalg.norm(tw.as_array()))
        assert obs.normalised_errors is dbg["normalised_errors"]
        assert obs.t == (dbg["step"] - 1) * DEFAULT_DT
        assert obs.pixel_error_mean == float(np.mean(np.linalg.norm(
            dbg["pts_cur"] - dbg["pts_goal"], axis=1)))


def test_truncate_is_calculate_uv_roi_and_survives_the_rows_round_trip():
    from vsbench.vs.baselines.control import TRUNCATE_SNAP, truncate_px

    c = IBVSController(intrinsics=INTR, pixel_quantisation="truncate",
                       weighting=UNIFORM_WEIGHTING)
    assert c.solve_unrounded is False and c.record_fields() == {"pixel_quantisation": "truncate"}
    pts = np.array([[7.0 - 1e-13, 3.9999], [-0.4, 1079.7], [1925.2, 12.000000001],
                    [0.0, 6.9999999]])
    g, cur, mode = c.quantise({"pts_goal": pts, "pts_cur": pts})
    assert mode == "truncate"
    assert np.array_equal(g, [[7, 3], [0, 1079], [1919, 12], [0, 6]])   # snap, trunc, clip
    assert TRUNCATE_SNAP == 1e-9 and np.array_equal(truncate_px([6.99999]), [6.0])
    g, _, mode = c.quantise({"pts_goal": pts, "pts_cur": pts, "quantise": "round"})
    assert mode == "round" and np.array_equal(g, np.round(pts))          # a frame's own rule
    with pytest.raises(ValueError, match="disagree"):
        IBVSController.from_config({"controller": {"solve_unrounded": True,
                                                   "pixel_quantisation": "truncate"}},
                                   intrinsics=INTR)


def test_the_cap_comes_after_the_ema_and_the_ema_keeps_the_uncapped_value():
    rng = np.random.default_rng(2)
    g = rng.uniform(100, 1800, (24, 2))
    rows = [{"pts_goal": g, "pts_cur": g + rng.uniform(-80, 80, (24, 2))} for _ in range(5)]
    z = np.full(24, 0.4)
    free = IBVSController(intrinsics=INTR, weighting=UNIFORM_WEIGHTING, ema_alpha=0.8)
    capped = IBVSController(intrinsics=INTR, weighting=UNIFORM_WEIGHTING, ema_alpha=0.8,
                            max_velocity=0.01)
    for r in rows:
        a, _ = free.step(r, z)
        b, dbg = capped.step(r, z)
        assert np.array_equal(b.as_array(), np.clip(a.as_array(), -0.01, 0.01))
        assert np.array_equal(capped._ema_v, free._ema_v)                 # EMA uncapped
        assert dbg["velocity_capped"] == bool(np.any(np.abs(a.as_array()) > 0.01))
    assert capped.record_fields() == {"max_velocity": 0.01}
    assert "velocity_capped" not in free.step(rows[0], z)[1]              # absent off


def test_hold_repeats_the_last_twist_leaves_the_ema_and_the_depth_fallback_is_stated():
    from vsbench.vs.baselines.control import clean_depth

    rng = np.random.default_rng(3)
    g = rng.uniform(100, 1800, (24, 2))
    good = {"pts_goal": g, "pts_cur": g + 30.0}
    bad = {"pts_goal": g[:3], "pts_cur": g[:3] + 30.0}
    hold = IBVSController(intrinsics=INTR, weighting=UNIFORM_WEIGHTING, ema_alpha=0.8,
                          degenerate_policy="hold")
    zero = IBVSController(intrinsics=INTR, weighting=UNIFORM_WEIGHTING, ema_alpha=0.8)
    t0, d0 = hold.step(bad, np.full(3, 0.4))                  # nothing to hold yet: zero
    assert np.array_equal(t0.as_array(), np.zeros(6)) and d0["degenerate_hold"] is False
    t1, _ = hold.step(good, np.full(24, 0.4))
    ema = hold._ema_v.copy()
    t2, d2 = hold.step(bad, np.full(3, 0.4))
    assert d2["degenerate"] and d2["degenerate_hold"] is True
    assert np.array_equal(t2.as_array(), t1.as_array()) and np.array_equal(hold._ema_v, ema)
    zero.step(good, np.full(24, 0.4))
    tz, _ = zero.step(bad, np.full(3, 0.4))                   # today's rule: the EMA decays
    assert not np.array_equal(tz.as_array(), t1.as_array())
    assert hold.record_fields() == {"degenerate_policy": "hold"}
    z = np.array([0.0, -1.0, 11.0, np.nan, np.inf, 0.3])
    assert np.array_equal(clean_depth(z), [0.5, 0.5, 0.5, 0.5, 0.5, 0.3])
    assert np.array_equal(clean_depth(z, fallback_m=100.0), [100, 100, 100, 100, 100, 0.3])
    c = IBVSController(intrinsics=INTR, depth_fallback_m=100.0)
    assert c.record_fields() == {"depth_fallback_m": 100.0}
