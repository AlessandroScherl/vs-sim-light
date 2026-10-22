"""IBVS on known 3D points projected with the true poses. Privileged, test instrument only.

The method receives the world coordinates of its points at construction (the harness
passes the 13 marker points; a method may not import a scene) and the true poses through
``observe_truth``. Each step it projects the points with the current and the target pose,
forms the feature error in normalised image coordinates, stacks the interaction matrix
from ``vsbench.interaction.interaction_matrix`` with the true current depths and commands

    twist = -gain * pinv(L) @ (s - s_target)

This proves the IBVS law and its signs with no image processing at all: with the error
in the range of ``L`` (true for a small rigid offset) it decays by ``1 - gain * dt`` per
step. Points behind the camera are dropped; with fewer than three points left the method
returns a zero twist and sets ``lost``. The images are ignored, so it needs no renderer.
"""

from __future__ import annotations

import numpy as np

from vsbench.camera import CameraModel
from vsbench.interaction import interaction_matrix, normalised
from vsbench.methods import register
from vsbench.reference import project

MIN_POINTS = 3


@register("true_point_ibvs")
class TruePointIBVS:
    wants_depth = "none"
    privileged = True

    def __init__(self, points_w, gain: float = 1.0):
        P = np.array(points_w, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3 or len(P) < MIN_POINTS or not np.all(np.isfinite(P)):
            raise ValueError(f"points_w must be a finite (N, 3) array with N >= {MIN_POINTS}")
        if not np.isfinite(gain):
            raise ValueError("gain must be finite")
        P.setflags(write=False)
        self.points_w = P
        self.gain = float(gain)
        self.params = {"points_w": P.tolist(), "gain": self.gain}
        self.cam: CameraModel | None = None
        self._T_wc: np.ndarray | None = None
        self._T_wc_target: np.ndarray | None = None

    def configure(self, K: np.ndarray, width: int, height: int, dt: float) -> None:
        self.cam = CameraModel(np.array(K, dtype=np.float64), int(width), int(height))
        self.dt = float(dt)

    def reset(self) -> None:
        self._T_wc = None
        self._T_wc_target = None

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None) -> None:
        pass  # the desired features come from the true target pose

    def observe_truth(self, T_wc: np.ndarray, T_wc_target: np.ndarray) -> None:
        self._T_wc = np.array(T_wc, dtype=np.float64).reshape(4, 4)
        self._T_wc_target = np.array(T_wc_target, dtype=np.float64).reshape(4, 4)

    def get_control_rate(self, current_rgb, current_depth=None):
        if self.cam is None:
            raise RuntimeError("true_point_ibvs needs configure first")
        if self._T_wc is None or self._T_wc_target is None:
            raise RuntimeError("true_point_ibvs needs observe_truth before every get_control_rate")
        cam = self.cam
        px_cur, z_cur = project(cam, self._T_wc, self.points_w)
        px_des, z_des = project(cam, self._T_wc_target, self.points_w)
        self._T_wc = None  # one observation per step, never reused
        front = (z_cur > 0.0) & (z_des > 0.0)
        info = {"kp_cur": px_cur, "kp_des": px_des, "inliers": front}
        if front.sum() < MIN_POINTS:
            info.update(lost=True, error_norm=float("nan"))
            return np.zeros(6), info
        x, y = normalised(cam, px_cur[front])
        xd, yd = normalised(cam, px_des[front])
        e = np.stack([x - xd, y - yd], axis=1).reshape(-1)  # (2N,) normalised coordinates
        L = interaction_matrix(cam, px_cur[front], z_cur[front]).reshape(-1, 6)  # (2N, 6)
        twist = -self.gain * (np.linalg.pinv(L) @ e)
        info.update(lost=False, error_norm=float(np.linalg.norm(e)))
        return twist, info
