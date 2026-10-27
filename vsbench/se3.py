"""SO(3) and SE(3) in closed form: hat, vee, exp, log, pose error, re-orthonormalisation.

NumPy only. A twist is ``[vx, vy, vz, wx, wy, wz]``, linear part first, and its matrix
``hat(xi)`` is the 4 x 4 element of se(3) that ``exp_se3`` exponentiates. Rotation angles
below ``SERIES_ANGLE`` (1e-6 rad) use the Taylor series of the trigonometric coefficients
so nothing is divided by a vanishing angle; everything else is the closed form
(Rodrigues for SO(3), the ``V`` matrix for the translation).
"""

from __future__ import annotations

import numpy as np

SERIES_ANGLE = 1e-6  # rad: below this the coefficients come from their series


def hat(x: np.ndarray) -> np.ndarray:
    """Skew matrix of a 3-vector, or the 4 x 4 se(3) matrix of a 6-vector twist."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape == (3,):
        return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])
    if x.shape == (6,):
        M = np.zeros((4, 4))
        M[:3, :3] = hat(x[3:])
        M[:3, 3] = x[:3]
        return M
    raise ValueError(f"hat takes a 3- or 6-vector, got shape {x.shape}")


def vee(M: np.ndarray) -> np.ndarray:
    """Inverse of ``hat``: a 3 x 3 skew matrix to a 3-vector, a 4 x 4 se(3) to a twist."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        return np.array([M[2, 1], M[0, 2], M[1, 0]])
    if M.shape == (4, 4):
        return np.concatenate([M[:3, 3], vee(M[:3, :3])])
    raise ValueError(f"vee takes a 3 x 3 or 4 x 4 matrix, got shape {M.shape}")


def _coefficients(theta: float) -> tuple[float, float, float]:
    """``A = sin t / t``, ``B = (1 - cos t) / t^2``, ``C = (t - sin t) / t^3``."""
    if theta < SERIES_ANGLE:
        t2 = theta * theta
        return 1.0 - t2 / 6.0, 0.5 - t2 / 24.0, 1.0 / 6.0 - t2 / 120.0
    s = np.sin(theta)
    one_minus_cos = 2.0 * np.sin(0.5 * theta) ** 2  # exact where 1 - cos t cancels
    A = s / theta
    B = one_minus_cos / (theta * theta)
    return A, B, (1.0 - A) / (theta * theta)


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (axis times angle, radians)."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    A, B, _ = _coefficients(theta)
    W = hat(w)
    return np.eye(3) + A * W + B * (W @ W)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix; angle in [0, pi].

    The angle comes from ``atan2`` of the antisymmetric and symmetric parts, which is
    accurate at every angle. The axis comes from the antisymmetric part except near pi,
    where it is read off the symmetric part ``(R + R^T) / 2 - cos(t) I = (1 - cos t) w w^T``
    with its sign taken from the antisymmetric part (arbitrary at exactly pi).
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    a = 0.5 * vee(R - R.T)  # sin(t) * axis
    sin_t = float(np.linalg.norm(a))
    cos_t = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    theta = float(np.arctan2(sin_t, cos_t))
    if theta < SERIES_ANGLE:
        return a * (1.0 + theta * theta / 6.0)
    if theta < 0.75 * np.pi:
        return a * (theta / sin_t)
    S = 0.5 * (R + R.T) - cos_t * np.eye(3)  # (1 - cos t) * axis axis^T
    k = int(np.argmax(np.diag(S)))
    axis = S[:, k] / np.sqrt(S[k, k] * (1.0 - cos_t))
    if a @ axis < 0.0:
        axis = -axis
    return theta * axis


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """4 x 4 rigid transform of a twist ``[v, w]``: ``[[exp(w), V v], [0, 1]]``."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    v, w = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w))
    A, B, C = _coefficients(theta)
    W = hat(w)
    W2 = W @ W
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + A * W + B * W2
    T[:3, 3] = (np.eye(3) + B * W + C * W2) @ v
    return T


def _v_inverse(w: np.ndarray, theta: float) -> np.ndarray:
    """``V^-1 = I - W/2 + c W^2`` with ``c = (1 - (t/2) cot(t/2)) / t^2``."""
    W = hat(w)
    if theta < 1e-3:
        # 1 - x cot x = x^2/3 + x^4/45 + 2 x^6/945, with x = t/2, divided by t^2 = 4 x^2
        x2 = 0.25 * theta * theta
        c = 1.0 / 12.0 + x2 / 180.0 + x2 * x2 / 1890.0
    else:
        half = 0.5 * theta
        c = (1.0 - half * np.cos(half) / np.sin(half)) / (theta * theta)
    return np.eye(3) - 0.5 * W + c * (W @ W)


def log_se3(T: np.ndarray) -> np.ndarray:
    """Twist ``[v, w]`` of a rigid transform, the inverse of ``exp_se3``."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    w = log_so3(T[:3, :3])
    theta = float(np.linalg.norm(w))
    v = _v_inverse(w, theta) @ T[:3, 3]
    return np.concatenate([v, w])


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """``(metres, radians)``: translation distance and rotation angle between two poses."""
    T_a = np.asarray(T_a, dtype=np.float64)
    T_b = np.asarray(T_b, dtype=np.float64)
    d_t = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    d_r = float(np.linalg.norm(log_so3(T_a[:3, :3].T @ T_b[:3, :3])))
    return d_t, d_r


def orthonormalise(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (Frobenius) by SVD: ``U V^T`` with the determinant forced to +1."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        U = U.copy()
        U[:, 2] = -U[:, 2]
    return U @ Vt


def orthonormalise_pose(T: np.ndarray) -> np.ndarray:
    """``T`` with its rotation block re-orthonormalised and the last row fixed."""
    out = np.array(T, dtype=np.float64).reshape(4, 4)
    out[:3, :3] = orthonormalise(out[:3, :3])
    out[3] = (0.0, 0.0, 0.0, 1.0)
    return out


def random_twist(rng: np.random.Generator, max_angle: float = np.pi, max_dist: float = 1.0) -> np.ndarray:
    """A seeded twist: uniform axis, angle uniform in ``[0, max_angle)``, uniform translation."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, max_angle)
    v = rng.uniform(-max_dist, max_dist, 3)
    return np.concatenate([v, angle * axis])
