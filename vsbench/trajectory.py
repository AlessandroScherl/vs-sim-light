"""The trajectory log: time, pose, commanded and applied twist per step, in one ``.npz``.

NumPy only. Row ``k`` holds the state at ``t[k]``: the pose ``T_wc[k]`` the camera was
at when the command ``twist_cmd[k]`` arrived, and ``twist_applied[k]``, the clamped twist
that then moved it to ``T_wc[k + 1]``. ``replay`` re-integrates the applied twists from
the first pose with the same ``integrate`` function the camera uses, so the poses it
returns are bit-identical to the log.

``save`` writes one compressed ``.npz`` with float64 arrays and a JSON ``meta`` string
(``dt``, the limits, and whatever the caller adds). Step 3's episode log adds its
errors and matches to the same file.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vsbench.motion import MotionLimits, integrate
from vsbench.se3 import pose_error

ARRAYS = ("t", "T_wc", "twist_cmd", "twist_applied")


class Trajectory:
    def __init__(self, dt: float, limits: MotionLimits = MotionLimits(), meta: dict | None = None):
        if not dt > 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.limits = limits
        self.meta: dict = dict(meta or {})
        self._t: list[float] = []
        self._T: list[np.ndarray] = []
        self._cmd: list[np.ndarray] = []
        self._app: list[np.ndarray] = []

    # -- building ----------------------------------------------------------------------
    def append(self, t: float, T_wc: np.ndarray, twist_cmd: np.ndarray, twist_applied: np.ndarray) -> None:
        T = np.array(T_wc, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("T_wc must be 4 x 4")
        self._t.append(float(t))
        self._T.append(T)
        self._cmd.append(np.array(twist_cmd, dtype=np.float64).reshape(6))
        self._app.append(np.array(twist_applied, dtype=np.float64).reshape(6))

    def __len__(self) -> int:
        return len(self._t)

    # -- arrays ------------------------------------------------------------------------
    @property
    def t(self) -> np.ndarray:
        return np.array(self._t, dtype=np.float64)

    @property
    def T_wc(self) -> np.ndarray:
        return np.array(self._T, dtype=np.float64).reshape(-1, 4, 4)

    @property
    def twist_cmd(self) -> np.ndarray:
        return np.array(self._cmd, dtype=np.float64).reshape(-1, 6)

    @property
    def twist_applied(self) -> np.ndarray:
        return np.array(self._app, dtype=np.float64).reshape(-1, 6)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ARRAYS}

    # -- file --------------------------------------------------------------------------
    def save(self, path) -> None:
        meta = {**self.meta, "dt": self.dt, "limits": asdict(self.limits), "format": "vsbench.trajectory/1"}
        np.savez_compressed(Path(path), meta=json.dumps(meta), **self.arrays())

    @classmethod
    def load(cls, path) -> Trajectory:
        with np.load(Path(path)) as f:
            meta = json.loads(str(f["meta"]))
            arrays = {k: np.array(f[k], dtype=np.float64) for k in ARRAYS}
        lim = meta.pop("limits", None)
        limits = MotionLimits(**lim) if lim else MotionLimits()
        dt = meta.pop("dt")
        meta.pop("format", None)
        out = cls(dt, limits, meta)
        for row in zip(arrays["t"], arrays["T_wc"], arrays["twist_cmd"], arrays["twist_applied"]):
            out.append(*row)
        return out

    # -- derived -----------------------------------------------------------------------
    def replay(self) -> np.ndarray:
        """Poses ``(N, 4, 4)`` re-integrated from ``T_wc[0]`` with the applied twists."""
        n = len(self)
        out = np.zeros((n, 4, 4))
        if n == 0:
            return out
        out[0] = self._T[0]
        for k in range(1, n):
            out[k] = integrate(out[k - 1], self._app[k - 1], self.dt)
        return out

    def final_pose(self) -> np.ndarray:
        """The pose after the last logged applied twist (one step past the last row)."""
        if len(self) == 0:
            raise ValueError("empty trajectory")
        return integrate(self._T[-1], self._app[-1], self.dt)

    def path_length(self) -> tuple[float, float]:
        """``(metres, radians)`` summed over consecutive logged poses."""
        d_t = d_r = 0.0
        for a, b in zip(self._T[:-1], self._T[1:]):
            dt_, dr_ = pose_error(a, b)
            d_t += dt_
            d_r += dr_
        return d_t, d_r
