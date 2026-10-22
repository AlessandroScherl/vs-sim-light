"""A free-flying camera that integrates a clamped body twist on SE(3) at a fixed step.

NumPy only, no physics and no renderer. The twist is ``[vx, vy, vz, wx, wy, wz]`` in
m/s and rad/s, the body twist of the camera expressed in its optical frame (x right,
y down, z forward). One step is

    T_wc[k + 1] = T_wc[k] @ exp_se3(twist * dt)

right-multiplied because the twist lives in the camera frame, and the rotation block is
re-orthonormalised by SVD after every step. Simulated time is ``k * dt``.

``FreeCamera.step`` is the only place a command becomes motion: later motion models (a
first-order lag, noise, a dead band) slot in there and the rest of the benchmark does
not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vsbench.se3 import exp_se3, orthonormalise_pose


@dataclass(frozen=True)
class MotionLimits:
    """Norm limits of the linear and angular parts, applied separately."""

    v_max: float = 2.0  # m/s, loose on purpose, tightened in Step 7
    w_max: float = 4.0  # rad/s

    def __post_init__(self) -> None:
        if not (self.v_max > 0.0 and self.w_max > 0.0):
            raise ValueError("limits must be positive")


def clamp_twist(twist: np.ndarray, limits: MotionLimits = MotionLimits()) -> np.ndarray:
    """Scale the linear and angular parts down to their limits, keeping each direction.

    A twist with a NaN or infinite entry raises ``ValueError``.
    """
    xi = np.array(twist, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(xi)):
        raise ValueError(f"twist must be finite, got {xi}")
    v_norm = float(np.linalg.norm(xi[:3]))
    if v_norm > limits.v_max:
        xi[:3] *= limits.v_max / v_norm
    w_norm = float(np.linalg.norm(xi[3:]))
    if w_norm > limits.w_max:
        xi[3:] *= limits.w_max / w_norm
    return xi


def integrate(T_wc: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    """One step of the motion model: ``T_wc @ exp_se3(twist * dt)``, re-orthonormalised.

    ``FreeCamera.step`` calls this after clamping, and ``Trajectory.replay`` calls it with
    the logged applied twists, so a replay repeats the same floating-point operations.
    """
    return orthonormalise_pose(np.asarray(T_wc, dtype=np.float64) @ exp_se3(np.asarray(twist, dtype=np.float64) * dt))


def check_pose(T: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """A float64 copy of ``T`` after checking it is a rigid 4 x 4 (raises ``ValueError``)."""
    T = np.array(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("pose must be a finite 4 x 4 matrix")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=tol):
        raise ValueError("pose must have last row [0 0 0 1]")
    R = T[:3, :3]
    if np.abs(R.T @ R - np.eye(3)).max() > tol or np.linalg.det(R) < 0.0:
        raise ValueError("pose rotation block must be orthonormal with determinant +1")
    return T


class FreeCamera:
    """Camera pose integrator: ``step`` applies one clamped twist for ``dt`` seconds.

    ``T_wc`` is the current camera-to-world pose (read-only array, replaced each step),
    ``k`` the number of steps taken and ``t = k * dt`` the simulated time.
    """

    def __init__(self, T_wc0: np.ndarray, dt: float = 0.04, limits: MotionLimits = MotionLimits()):
        if not dt > 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.limits = limits
        self.reset(T_wc0)

    @property
    def t(self) -> float:
        return self.k * self.dt

    def reset(self, T_wc0: np.ndarray) -> None:
        """Back to step 0 at ``T_wc0`` (re-orthonormalised, so a rounded pose is fine)."""
        self._set_pose(orthonormalise_pose(check_pose(T_wc0)))
        self.k = 0

    def step(self, twist: np.ndarray) -> np.ndarray:
        """Apply ``twist`` for one step and return the twist actually applied (clamped)."""
        applied = clamp_twist(twist, self.limits)
        self._set_pose(integrate(self.T_wc, applied, self.dt))
        self.k += 1
        applied.setflags(write=False)
        return applied

    def _set_pose(self, T: np.ndarray) -> None:
        T.setflags(write=False)
        self.T_wc = T

    def __repr__(self) -> str:
        return f"FreeCamera(k={self.k}, t={self.t:.3f}, dt={self.dt}, limits={self.limits})"
