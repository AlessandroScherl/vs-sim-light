"""Step 2 gate 1: exp and log round trip against SciPy's ``expm`` and ``logm``.

1000 seeded twists with rotation angles below pi, plus 1e-9, 1e-6 and 1e-3 rad: every
matrix must agree with the oracle to 1e-10 and nothing may be NaN. SciPy is an oracle
here only; ``vsbench.se3`` itself imports nothing but NumPy.
"""

import numpy as np
import pytest
from scipy.linalg import expm, logm

from vsbench.se3 import (
    exp_se3,
    exp_so3,
    hat,
    log_se3,
    log_so3,
    orthonormalise,
    pose_error,
    random_twist,
    vee,
)

TOL = 1e-10


def _twists(n: int = 1000, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = [random_twist(rng, max_angle=np.pi, max_dist=1.0) for _ in range(n)]
    for angle in (1e-9, 1e-6, 1e-3):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        out.append(np.concatenate([rng.uniform(-1.0, 1.0, 3), angle * axis]))
    return out


def test_exp_log_round_trip_against_scipy():
    worst_exp = worst_log = worst_rt = 0.0
    for xi in _twists():
        T = exp_se3(xi)
        assert np.all(np.isfinite(T))
        worst_exp = max(worst_exp, float(np.abs(T - expm(hat(xi))).max()))
        back = log_se3(T)
        assert np.all(np.isfinite(back))
        worst_rt = max(worst_rt, float(np.abs(back - xi).max()))
        L = logm(T)
        assert np.abs(np.imag(L)).max() < TOL
        worst_log = max(worst_log, float(np.abs(vee(np.real(L)) - back).max()))
    print(f"step2 exp-log round trip: exp vs expm {worst_exp:.2e}, log vs logm {worst_log:.2e}, round trip {worst_rt:.2e}")
    assert worst_exp < TOL
    assert worst_log < TOL
    assert worst_rt < TOL


def test_so3_exp_log():
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = random_twist(rng)[3:]
        R = exp_so3(w)
        assert np.abs(R - expm(hat(w))).max() < TOL
        assert np.abs(log_so3(R) - w).max() < TOL
    # rotations close to pi: the axis comes from the symmetric part
    for angle in (np.pi - 1e-3, np.pi - 1e-6, np.pi - 1e-9):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        w = angle * axis
        assert np.abs(log_so3(exp_so3(w)) - w).max() < TOL
    # exactly pi: the sign is arbitrary but the rotation is recovered
    w = np.pi * np.array([0.0, 1.0, 0.0])
    assert np.abs(exp_so3(log_so3(exp_so3(w))) - exp_so3(w)).max() < TOL


def test_hat_vee_and_series_branch():
    xi = np.array([0.1, -0.2, 0.3, 0.4, 0.5, -0.6])
    assert np.array_equal(vee(hat(xi)), xi)
    assert np.array_equal(vee(hat(xi[3:])), xi[3:])
    with pytest.raises(ValueError):
        hat(np.zeros(4))
    assert np.array_equal(exp_se3(np.zeros(6)), np.eye(4))
    assert np.array_equal(log_se3(np.eye(4)), np.zeros(6))
    # 1e-9 rad sits in the series branch: pure translation apart from a 1e-9 turn
    xi = np.array([1.0, 2.0, 3.0, 0.0, 1e-9, 0.0])
    T = exp_se3(xi)
    assert np.abs(T[:3, 3] - xi[:3]).max() < 1e-8
    assert np.abs(T - expm(hat(xi))).max() < TOL


def test_pose_error_and_orthonormalise():
    rng = np.random.default_rng(2)
    T_a = exp_se3(random_twist(rng))
    # a pure translation then a pure rotation (a mixed twist is a screw, not a shift)
    T_b = T_a @ exp_se3(np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])) @ exp_se3(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2]))
    d_t, d_r = pose_error(T_a, T_b)
    assert abs(d_t - 0.3) < TOL and abs(d_r - 0.2) < TOL
    assert pose_error(T_a, T_a) == (0.0, 0.0)
    R = exp_so3(random_twist(rng)[3:])
    R_bad = R + 1e-3 * rng.normal(size=(3, 3))
    R_fix = orthonormalise(R_bad)
    assert np.linalg.norm(R_fix.T @ R_fix - np.eye(3)) < 1e-12
    assert abs(np.linalg.det(R_fix) - 1.0) < 1e-12
    assert np.abs(R_fix - R).max() < 5e-3
    # a reflection is turned into a rotation, never returned as is
    R_fix = orthonormalise(np.diag([1.0, 1.0, -1.0]))
    assert abs(np.linalg.det(R_fix) - 1.0) < 1e-12
