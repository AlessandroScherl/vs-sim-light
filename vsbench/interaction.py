"""The point interaction matrix, the pixel flow it predicts, and the exact flow.

NumPy only. For a point at normalised coordinates ``x = (u - cx) / fx``,
``y = (v - cy) / fy`` and z-depth ``Z`` the interaction matrix (Chaumette and Hutchinson)
is

    L = [[-1/Z,    0,  x/Z,      x*y,  -(1 + x*x),   y],
         [   0, -1/Z,  y/Z,  1 + y*y,        -x*y,  -x]]

so that ``[x_dot, y_dot] = L @ twist`` for the camera body twist ``[v, w]`` in the
optical frame. ``predicted_flow`` is the linearised pixel flow over a short time
``tau``; ``exact_flow`` back-projects the pixel, moves the point into the new camera
frame with the inverse of ``exp_se3(twist * tau)`` and projects again, so the difference
between the two is the linearisation error.
"""

from __future__ import annotations

import numpy as np

from vsbench.camera import CameraModel
from vsbench.reference import project
from vsbench.se3 import exp_se3


def normalised(cam: CameraModel, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised image coordinates ``(x, y)`` of pixels ``(N, 2)``."""
    p = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    return (p[:, 0] - cam.cx) / cam.fx, (p[:, 1] - cam.cy) / cam.fy


def _depths(Z, n: int) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64).reshape(-1)
    if Z.shape == (1,) and n != 1:
        Z = np.full(n, Z[0])
    if Z.shape != (n,):
        raise ValueError(f"need one depth per pixel, got {Z.shape} for {n} pixels")
    if not np.all(Z > 0.0):
        raise ValueError("depths must be positive z-depths in metres")
    return Z


def interaction_matrix(cam: CameraModel, px: np.ndarray, Z) -> np.ndarray:
    """``(N, 2, 6)`` interaction matrices of pixels ``(N, 2)`` at z-depths ``Z`` (N,) or scalar."""
    x, y = normalised(cam, px)
    Z = _depths(Z, len(x))
    zero = np.zeros_like(x)
    inv = 1.0 / Z
    L = np.stack(
        [
            np.stack([-inv, zero, x * inv, x * y, -(1.0 + x * x), y], axis=-1),
            np.stack([zero, -inv, y * inv, 1.0 + y * y, -x * y, -x], axis=-1),
        ],
        axis=1,
    )
    return L


def predicted_flow(cam: CameraModel, px: np.ndarray, Z, twist: np.ndarray, tau: float) -> np.ndarray:
    """Linearised pixel flow ``(N, 2)``: ``diag(fx, fy) @ L @ twist * tau``."""
    L = interaction_matrix(cam, px, Z)
    xi = np.asarray(twist, dtype=np.float64).reshape(6)
    rate = L @ xi  # (N, 2) normalised-coordinate rates
    return rate * np.array([cam.fx, cam.fy]) * float(tau)


def exact_flow(cam: CameraModel, px: np.ndarray, Z, twist: np.ndarray, tau: float) -> np.ndarray:
    """Pixel flow ``(N, 2)`` from moving the camera by ``exp_se3(twist * tau)``, no linearisation.

    The points are back-projected with their z-depths, expressed in the new camera frame
    (the old frame is taken as the world, so ``T_wc = exp_se3(twist * tau)``) and
    re-projected with ``vsbench.reference.project``. Points that end behind the new
    camera get NaN flow.
    """
    p = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    x, y = normalised(cam, p)
    Z = _depths(Z, len(x))
    X_c0 = np.stack([x * Z, y * Z, Z], axis=-1)
    xi = np.asarray(twist, dtype=np.float64).reshape(6)
    T_c0_c1 = exp_se3(xi * float(tau))
    with np.errstate(divide="ignore", invalid="ignore"):
        p1, z1 = project(cam, T_c0_c1, X_c0)
    p1[z1 <= 0.0] = np.nan
    return p1 - p
