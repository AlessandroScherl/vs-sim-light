"""Step 2 gates 2 to 5: orthonormality, composition, direction and clamp of ``FreeCamera``.

No renderer. Every pose and twist is seeded; the direction gate starts from a tilted
pose so a sign error in any axis cannot hide behind an identity rotation.
"""

import numpy as np
import pytest

from vsbench.motion import FreeCamera, MotionLimits, clamp_twist
from vsbench.poses import look_at, roll
from vsbench.se3 import exp_se3, exp_so3, log_so3, pose_error, random_twist

DT = 0.04
AXES = ["vx", "vy", "vz", "wx", "wy", "wz"]


def tilted_pose() -> np.ndarray:
    """A start pose with every axis off the world axes."""
    return roll(look_at((0.3, -0.5, 0.9), (0.05, 0.1, 0.0)), 0.4)


def test_orthonormality_over_random_clamped_steps():
    rng = np.random.default_rng(10)
    cam = FreeCamera(tilted_pose(), dt=DT)
    worst_orth = worst_det = 0.0
    for _ in range(10_000):
        # well beyond the limits most of the time, so the clamp is exercised too
        cam.step(rng.normal(scale=3.0, size=6))
        R = cam.T_wc[:3, :3]
        worst_orth = max(worst_orth, float(np.linalg.norm(R.T @ R - np.eye(3))))
        worst_det = max(worst_det, abs(float(np.linalg.det(R)) - 1.0))
        assert np.allclose(cam.T_wc[3], [0, 0, 0, 1])
    print(f"step2 orthonormality: norm of R^T R - I {worst_orth:.2e}, |det - 1| {worst_det:.2e} over {cam.k} steps")
    assert cam.k == 10_000 and abs(cam.t - 400.0) < 1e-9
    assert worst_orth < 1e-12
    assert worst_det < 1e-12


def test_composition_of_constant_twist():
    rng = np.random.default_rng(11)
    xi = random_twist(rng, max_angle=1.0, max_dist=0.5)  # inside the limits
    T0 = tilted_pose()
    cam = FreeCamera(T0, dt=DT)
    n = 1000
    for _ in range(n):
        applied = cam.step(xi)
        assert np.array_equal(applied, xi)
    T_one = T0 @ exp_se3(xi * (n * DT))
    d_t, d_r = pose_error(cam.T_wc, T_one)
    print(f"step2 composition: {d_t:.2e} m, {d_r:.2e} rad over {n} steps")
    assert d_t < 1e-9
    assert d_r < 1e-9


@pytest.mark.parametrize("axis", range(6), ids=AXES)
def test_direction_of_each_axis(axis):
    speed = 0.1 if axis < 3 else np.deg2rad(10.0)
    T0 = tilted_pose()
    R0 = T0[:3, :3]
    cam = FreeCamera(T0, dt=DT)
    xi = np.zeros(6)
    xi[axis] = speed
    steps = int(round(1.0 / DT))
    for _ in range(steps):
        cam.step(xi)
    assert abs(cam.t - 1.0) < 1e-12
    T1 = cam.T_wc
    if axis < 3:
        # translation: speed times the matching column of R, rotation untouched
        err_t = np.abs((T1[:3, 3] - T0[:3, 3]) - speed * R0[:, axis]).max()
        err_r = np.abs(T1[:3, :3] - R0).max()
    else:
        # right-hand turn by speed * 1 s about the matching camera axis, no translation
        e = np.zeros(3)
        e[axis - 3] = 1.0
        err_t = np.abs(T1[:3, 3] - T0[:3, 3]).max()
        err_r = np.abs(T1[:3, :3] - R0 @ exp_so3(speed * e)).max()
        w = log_so3(R0.T @ T1[:3, :3])
        assert np.abs(w - speed * e).max() < 1e-12
    print(f"step2 direction {AXES[axis]}: translation error {err_t:.1e} m, rotation error {err_r:.1e} over {steps} steps")
    assert err_t < 1e-12
    assert err_r < 1e-12
    # signs as in the axis table (world directions read off the start pose)
    right, down, forward = R0[:, 0], R0[:, 1], R0[:, 2]
    x1, y1, z1 = T1[:3, 0], T1[:3, 1], T1[:3, 2]
    d = T1[:3, 3] - T0[:3, 3]
    if axis == 0:
        assert d @ right > 0.0  # slides right
    elif axis == 1:
        assert d @ down > 0.0  # slides down
    elif axis == 2:
        assert d @ forward > 0.0  # moves forward
    elif axis == 3:
        assert z1 @ down < 0.0 and y1 @ forward > 0.0  # tilts up: the optical axis rises
    elif axis == 4:
        assert z1 @ right > 0.0 and x1 @ forward < 0.0  # pans right
    elif axis == 5:
        assert x1 @ down > 0.0 and y1 @ right < 0.0  # rolls clockwise seen from behind


def test_clamp_keeps_direction_and_hits_the_limit():
    lim = MotionLimits(v_max=2.0, w_max=4.0)
    rng = np.random.default_rng(12)
    worst_norm = worst_dir = 0.0
    for _ in range(100):
        xi = rng.normal(size=6)
        xi[:3] *= rng.uniform(2.5, 50.0) / np.linalg.norm(xi[:3])  # both parts above the limit
        xi[3:] *= rng.uniform(4.5, 50.0) / np.linalg.norm(xi[3:])
        out = clamp_twist(xi, lim)
        v, w = out[:3], out[3:]
        worst_norm = max(worst_norm, abs(np.linalg.norm(v) - lim.v_max), abs(np.linalg.norm(w) - lim.w_max))
        worst_dir = max(worst_dir, np.abs(v / np.linalg.norm(v) - xi[:3] / np.linalg.norm(xi[:3])).max(), np.abs(w / np.linalg.norm(w) - xi[3:] / np.linalg.norm(xi[3:])).max())
    print(f"step2 clamp: 100 twists above both limits, norm off the limit by {worst_norm:.1e}, direction off by {worst_dir:.1e}; NaN, +inf and -inf raise")
    assert worst_norm < 1e-12
    assert worst_dir < 1e-12
    # inside the limits nothing changes; the two parts are independent
    xi = np.array([0.1, -0.2, 0.3, 0.5, 0.0, -0.5])
    assert np.array_equal(clamp_twist(xi, lim), xi)
    xi = np.array([3.0, 4.0, 0.0, 0.5, 0.0, -0.5])
    out = clamp_twist(xi, lim)
    assert np.allclose(out, [1.2, 1.6, 0.0, 0.5, 0.0, -0.5])
    xi = np.array([0.1, 0.0, 0.0, 0.0, 8.0, 0.0])
    out = clamp_twist(xi, lim)
    assert np.allclose(out, [0.1, 0.0, 0.0, 0.0, 4.0, 0.0])
    # the input is never modified
    xi = np.full(6, 100.0)
    clamp_twist(xi, lim)
    assert np.all(xi == 100.0)
    for bad in (np.array([np.nan, 0, 0, 0, 0, 0]), np.array([0, 0, 0, np.inf, 0, 0]), np.array([0, -np.inf, 0, 0, 0, 0])):
        with pytest.raises(ValueError):
            clamp_twist(bad, lim)
        cam = FreeCamera(tilted_pose(), dt=DT)
        with pytest.raises(ValueError):
            cam.step(bad)
        assert cam.k == 0  # a rejected command is not a step
    with pytest.raises(ValueError):
        clamp_twist(np.zeros(5), lim)


def test_free_camera_step_returns_applied_twist_and_reset():
    T0 = tilted_pose()
    cam = FreeCamera(T0, dt=0.1, limits=MotionLimits(1.0, 1.0))
    applied = cam.step([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(applied, [1.0, 0, 0, 0, 0, 0])
    assert cam.k == 1 and abs(cam.t - 0.1) < 1e-15
    assert np.allclose(cam.T_wc, T0 @ exp_se3(applied * 0.1))
    assert not cam.T_wc.flags.writeable  # the state cannot be edited behind the camera's back
    cam.reset(T0)
    assert cam.k == 0 and cam.t == 0.0 and np.allclose(cam.T_wc, T0)
    with pytest.raises(ValueError):
        FreeCamera(np.eye(4), dt=0.0)
    with pytest.raises(ValueError):
        FreeCamera(np.diag([1.0, 1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        MotionLimits(v_max=0.0)
