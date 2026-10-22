"""The oracle: pose-based servoing on the true poses. Privileged, test instrument only.

``twist = gain * log_se3(inv(T_wc) @ T_wc_target)``: the twist that, applied for one
second, would move the camera exactly onto the target. After one step of ``dt`` the pose
error ``inv(T_wc) @ T_wc_target`` becomes ``exp_se3((1 - gain * dt) * xi)``, so the norm
of its logarithm shrinks by exactly ``1 - gain * dt`` per step; the oracle decay gate
checks that and thereby proves the loop, the frames and the log independently of any
image processing. The images are ignored.
"""

from __future__ import annotations

import numpy as np

from vsbench.methods import register
from vsbench.reference import invert
from vsbench.se3 import log_se3


@register("oracle_pbvs")
class OraclePBVS:
    wants_depth = "none"
    privileged = True

    def __init__(self, gain: float = 1.0):
        if not np.isfinite(gain):
            raise ValueError("gain must be finite")
        self.gain = float(gain)
        self._T_wc: np.ndarray | None = None
        self._T_wc_target: np.ndarray | None = None

    def configure(self, K: np.ndarray, width: int, height: int, dt: float) -> None:
        self.dt = float(dt)

    def reset(self) -> None:
        self._T_wc = None
        self._T_wc_target = None

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None) -> None:
        pass  # the oracle never looks at an image

    def observe_truth(self, T_wc: np.ndarray, T_wc_target: np.ndarray) -> None:
        self._T_wc = np.array(T_wc, dtype=np.float64).reshape(4, 4)
        self._T_wc_target = np.array(T_wc_target, dtype=np.float64).reshape(4, 4)

    def get_control_rate(self, current_rgb, current_depth=None):
        if self._T_wc is None or self._T_wc_target is None:
            raise RuntimeError("oracle_pbvs needs observe_truth before every get_control_rate")
        xi = log_se3(invert(self._T_wc) @ self._T_wc_target)
        self._T_wc = None  # one observation per step, never reused
        return self.gain * xi, {"error_norm": float(np.linalg.norm(xi))}
