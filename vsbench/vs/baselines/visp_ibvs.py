# vs-sim copy: vssim/baselines/visp_ibvs.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The standard IBVS law as the ViSP team defines it - ``vpServo`` on point features.

Purpose
-------
ViSP (the Visual Servoing Platform, Inria Rennes; Marchand, Spindler, Chaumette,
IEEE RAM 2005) is the reference implementation of image-based visual servoing.
Its ``vpServo`` class with ``vpFeaturePoint`` features is what every IBVS paper
calls "the classical law" (Chaumette & Hutchinson, IEEE RAM 2006, part I)::

    e  = s - s*                                  the feature error, normalised coords
    L  = L(s, Z) | L(s*, Z*) | (L(s,Z) + L(s*,Z*)) / 2     CURRENT | DESIRED | MEAN
    v  = -lambda * L^+ e                          eye-in-hand, camera-frame twist

with the point interaction matrix::

    [-1/Z,     0,  x/Z,      x*y,  -(1 + x^2),   y]
    [    0, -1/Z,  y/Z,  1 + y^2,       -x*y,   -x]

:class:`VispIBVSController` is that law as a controller any matcher of this
repository can feed - matched goal/current pixels in, a camera twist out - so a
front-end can be flown "with ViSP's IBVS" beside the same front-end with the
bench's own weighted law (:class:`vsbench.vs.baselines.control.IBVSController`).

Two back-ends, one law
----------------------
- ``backend: visp`` runs **ViSP itself** through its Python bindings (ViSP >= 3.6,
  built from source with ``BUILD_PYTHON_BINDINGS=ON``): a ``vpServo`` task is built
  per frame from ``vpFeaturePoint`` pairs and ``computeControlLaw()`` returns the
  twist. This is the reference and is what a table should cite.
- ``backend: numpy`` is a line-for-line twin of ``vpServo::computeControlLaw`` for
  the eye-in-hand camera case (``cVe = eJe = I``): the same error, the same
  interaction-matrix choice, the same pseudo-inverse rule (singular values below
  ``1e-6`` of the largest are dropped, ``vpMatrix::pseudoInverse``'s default), the
  same constant or adaptive gain. ``tests/slow/test_visp_parity.py`` asserts the two
  agree to 1e-9 on random point sets where ViSP is installed; ``backend: auto``
  (the default) takes ViSP when it imports and the twin otherwise, and every record
  says which ran (``visp_backend``).

What ViSP's law does NOT have, and this controller therefore refuses
--------------------------------------------------------------------
Per-row weights (the bench's trust / margin / per-axis chain), the EMA smoother,
integer-pixel rounding. A config that asks for any of them under ``law: visp``
is refused by name: the point of the cell is the published law, unchanged.

The three interaction-matrix choices (``interaction:``)
-------------------------------------------------------
``current`` - ``L(s, Z)``, the live points with their live depth; ViSP's
``tutorial-ibvs-4pts`` uses it and so does the bench's own law.
``desired`` - ``L(s*, Z*)``, the goal points with the goal depth; ``vpServo``'s
constructor default, needs no live depth at all, converges more slowly far away.
``mean`` - the average of the two, the tutorial's recommendation for large
motions (Chaumette & Hutchinson 2006, section III-C).

Gain (``gain:`` and ``adaptive_gain:``)
---------------------------------------
A constant ``lambda`` (the bench's 2.0 at 60 Hz, or ViSP's tutorial 0.5), or ViSP's
``vpAdaptiveGain(gain_at_zero, gain_at_infinity, slope_at_zero)``::

    lambda(x) = (l0 - linf) * exp(-(l0' / (l0 - linf)) * x) + linf,   x = ||e1||_inf

evaluated, as ``vpServo`` does, on the **infinity norm of the task error**
``e1 = L^+ (s - s*)`` - the 6-vector the twist is proportional to - and not on the
feature error.

Provenance
----------
New in this repository (2026-09-17). The numpy twin follows ``vpServo.cpp``
(``computeControlLaw``, ``computeInteractionMatrix``), ``vpFeaturePoint.cpp``
(``interaction``) and ``vpAdaptiveGain.cpp`` of ViSP 3.7.0; it shares
:func:`vsbench.vs.baselines.control.normalised_coords` and
:func:`vsbench.vs.baselines.control.interaction_matrix`, which were themselves ported
from the published ViT-VS controller and are the same six expressions.

Golden tests: ``tests/fast/test_visp_ibvs.py`` (the twin against the bench's own
law under uniform weights, the three interaction choices, the adaptive gain, the
refusals); ``tests/slow/test_visp_parity.py`` (the twin against ViSP itself).
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from vsbench.vs.baselines.control import (
    DEFAULT_DT,
    DEFAULT_GAIN,
    MIN_MATCHES,
    UNIFORM_WEIGHTING,
    Intrinsics,
    Twist,
    Weighting,
    clean_depth,
    interaction_matrix,
    normalised_coords,
)

#: ``vpMatrix::pseudoInverse``'s default ``svThreshold``: singular values below this
#: fraction of the largest are treated as zero.
VISP_SV_THRESHOLD = 1e-6

INTERACTION_TYPES = ("current", "desired", "mean")
BACKENDS = ("auto", "visp", "numpy")

#: The keys a ``controller:`` block may carry under ``law: visp``. ``weighting``,
#: ``ema_alpha`` and ``solve_unrounded`` are refused when they would change the law.
VISP_CONTROLLER_KEYS = frozenset({"law", "gain", "dt", "interaction", "adaptive_gain",
                                  "backend", "min_matches", "weighting", "ema_alpha",
                                  "solve_unrounded"})


def visp_available() -> bool:
    """True when ViSP's Python bindings import (``visp.vs``, ``visp.visual_features``)."""
    try:
        importlib.import_module("visp.vs")
        importlib.import_module("visp.visual_features")
    except Exception:  # noqa: BLE001 - absent, or a broken build: both mean "no"
        return False
    return True


@dataclass(frozen=True)
class AdaptiveGain:
    """``vpAdaptiveGain(gain_at_zero, gain_at_infinity, slope_at_zero)``."""

    gain_at_zero: float
    gain_at_infinity: float
    slope_at_zero: float

    def __post_init__(self) -> None:
        if self.gain_at_zero <= 0 or self.gain_at_infinity <= 0 or self.slope_at_zero <= 0:
            raise ValueError(f"adaptive gain parameters must be > 0, got {self}")
        if self.gain_at_zero <= self.gain_at_infinity:
            raise ValueError("vpAdaptiveGain needs gain_at_zero > gain_at_infinity "
                             f"(got {self.gain_at_zero} <= {self.gain_at_infinity})")

    def __call__(self, error: np.ndarray) -> float:
        """``vpAdaptiveGain::value(x)`` on ``x = ||error||_inf`` of the vector given
        (``vpServo`` passes the task error ``e1``)."""
        x = float(np.max(np.abs(np.asarray(error, dtype=np.float64)))) if len(error) else 0.0
        l0, linf, lp = self.gain_at_zero, self.gain_at_infinity, self.slope_at_zero
        return float((l0 - linf) * np.exp(-(lp / (l0 - linf)) * x) + linf)


def visp_pinv(L: np.ndarray, sv_threshold: float = VISP_SV_THRESHOLD) -> np.ndarray:
    """``vpMatrix::pseudoInverse(svThreshold)``: SVD, singular values below
    ``svThreshold * max`` dropped. ``np.linalg.pinv(rcond=)`` is exactly that rule."""
    return np.linalg.pinv(np.asarray(L, dtype=np.float64), rcond=sv_threshold)


def visp_control_law(s: np.ndarray, s_star: np.ndarray, z: np.ndarray, z_star: np.ndarray,
                     *, interaction: str, gain: float | AdaptiveGain
                     ) -> tuple[np.ndarray, dict]:
    """The numpy twin of ``vpServo::computeControlLaw`` (eye-in-hand, camera frame).

    ``s``/``s_star`` are ``(K, 2)`` normalised coordinates, ``z``/``z_star`` ``(K,)``
    depths of the current and the desired points. Returns ``(v, info)`` with the
    lambda that ran, the error norm and ``||e||_inf``.
    """
    if interaction not in INTERACTION_TYPES:
        raise ValueError(f"interaction must be one of {INTERACTION_TYPES}, got {interaction!r}")
    s = np.asarray(s, dtype=np.float64).reshape(-1, 2)
    s_star = np.asarray(s_star, dtype=np.float64).reshape(-1, 2)
    e = (s - s_star).reshape(-1)
    if interaction == "current":
        L = interaction_matrix(s, z)
    elif interaction == "desired":
        L = interaction_matrix(s_star, z_star)
    else:
        L = 0.5 * (interaction_matrix(s, z) + interaction_matrix(s_star, z_star))
    # vpServo::computeControlLaw: e1 = J1^+ error (the 6-vector TASK error), then
    # e = -lambda(e1) * e1 - so an adaptive gain is evaluated on ||e1||_inf, the task
    # error's infinity norm, not on the feature error's (vpServo.cpp L799, L140 of
    # vpAdaptiveGain.cpp: operator()(vpColVector) -> value(x.infinityNorm())).
    e1 = visp_pinv(L) @ e
    lam = gain(e1) if isinstance(gain, AdaptiveGain) else float(gain)
    v = -lam * e1
    return v, {"lambda": lam, "feature_error_norm": float(np.linalg.norm(e)),
               "feature_error_inf": float(np.max(np.abs(e))) if len(e) else 0.0,
               "normalised_errors": np.linalg.norm(s - s_star, axis=1)}


def visp_control_law_native(s: np.ndarray, s_star: np.ndarray, z: np.ndarray,
                            z_star: np.ndarray, *, interaction: str,
                            gain: float | AdaptiveGain) -> tuple[np.ndarray, dict]:
    """The same law through **ViSP itself**: a ``vpServo`` task over ``vpFeaturePoint``s.

    Built per call because the point set changes every frame (a mutual-NN front end
    re-draws its landmarks), which is also how ViSP's own examples handle a varying
    feature set: ``addFeature`` again on a fresh task.
    """
    from visp.core import ColVector  # noqa: F401 - asserts the bindings are whole
    from visp.visual_features import FeaturePoint
    from visp.vs import AdaptiveGain as VpAdaptiveGain
    from visp.vs import Servo

    s = np.asarray(s, dtype=np.float64).reshape(-1, 2)
    s_star = np.asarray(s_star, dtype=np.float64).reshape(-1, 2)
    task = Servo()
    task.setServo(Servo.EYEINHAND_CAMERA)
    task.setInteractionMatrixType({"current": Servo.CURRENT, "desired": Servo.DESIRED,
                                   "mean": Servo.MEAN}[interaction])
    if isinstance(gain, AdaptiveGain):
        task.setLambda(VpAdaptiveGain(gain.gain_at_zero, gain.gain_at_infinity,
                                      gain.slope_at_zero))
    else:
        task.setLambda(float(gain))
    feats = []                                   # keep the objects alive for the call
    for (x, y), (xs, ys), zc, zd in zip(s, s_star, z, z_star, strict=True):
        p, pd = FeaturePoint(), FeaturePoint()
        p.buildFrom(float(x), float(y), float(zc))         # vpFeaturePoint::buildFrom(x, y, Z)
        pd.buildFrom(float(xs), float(ys), float(zd))
        task.addFeature(p, pd)
        feats.append((p, pd))
    v = np.asarray(task.computeControlLaw().numpy(), dtype=np.float64).reshape(-1)
    e = (s - s_star).reshape(-1)
    e1 = np.asarray(task.getTaskJacobianPseudoInverse().numpy(), dtype=np.float64) @ e
    return v, {"lambda": (gain(e1) if isinstance(gain, AdaptiveGain) else float(gain)),
               "feature_error_norm": float(np.linalg.norm(e)),
               "feature_error_inf": float(np.max(np.abs(e))) if len(e) else 0.0,
               "normalised_errors": np.linalg.norm(s - s_star, axis=1)}


class VispIBVSController:
    """ViSP's ``vpServo`` point-feature IBVS as a bench controller.

    The same seam as :class:`vsbench.vs.baselines.control.IBVSController`: matcher rows
    and one live depth per row in, a :class:`Twist` and a debug dict out; the arm
    records ``weighting`` (always uniform here), ``ema_alpha`` (0), ``min_matches``,
    ``solve_unrounded`` (True) and ``gain`` off the same attribute names.

    Args:
        gain: constant lambda, or an :class:`AdaptiveGain`.
        dt: the loop period (carried for the stop rules, not used by the law).
        intrinsics: full-image pinhole intrinsics; no fallback camera.
        interaction: ``current`` | ``desired`` | ``mean``.
        backend: ``auto`` | ``visp`` | ``numpy``. ``visp`` raises when the bindings
            are absent; ``auto`` falls back to the twin and records it.
        min_matches: fewer supporting rows than this -> a zero twist.
    """

    def __init__(self, gain: float | AdaptiveGain = DEFAULT_GAIN, dt: float = DEFAULT_DT, *,
                 intrinsics: Intrinsics, interaction: str = "current",
                 backend: str = "auto", min_matches: int = MIN_MATCHES) -> None:
        if not isinstance(gain, AdaptiveGain) and gain <= 0:
            raise ValueError(f"gain must be > 0, got {gain}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if interaction not in INTERACTION_TYPES:
            raise ValueError(f"interaction must be one of {INTERACTION_TYPES}, got {interaction!r}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if backend == "visp" and not visp_available():
            raise RuntimeError(
                "controller.backend 'visp' asks for ViSP's Python bindings and `import "
                "visp.vs` fails. Build ViSP >= 3.6 from source with "
                "-DBUILD_PYTHON_BINDINGS=ON inside a venv/conda env (README.md, 'ViSP'), "
                "or state backend 'numpy' to run the twin and say so in the table.")
        self.gain_spec = gain
        self.gain = float(gain.gain_at_zero if isinstance(gain, AdaptiveGain) else gain)
        self.dt = float(dt)
        self.intrinsics = intrinsics
        self.interaction = str(interaction)
        self.backend = ("visp" if visp_available() else "numpy") if backend == "auto" else backend
        self.min_matches = int(min_matches)
        #: the attributes the record reads off every controller
        self.weighting: Weighting = UNIFORM_WEIGHTING
        self.ema_alpha = 0.0
        self.solve_unrounded = True
        self._goal_depth: np.ndarray | None = None
        self._step_index = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"VispIBVSController(gain={self.gain_spec}, interaction={self.interaction!r}, "
                f"backend={self.backend!r}, dt={self.dt:.6g})")

    # ------------------------------------------------------------------ config
    @classmethod
    def from_config(cls, config: dict, *, intrinsics: Intrinsics) -> VispIBVSController:
        """Build from ``config["controller"]`` with ``law: visp``.

        Refused by name: a non-uniform ``weighting``, an ``ema_alpha`` above 0 and
        ``solve_unrounded: false`` - none of them is part of ViSP's law, and an arm
        labelled "ViSP's IBVS" that ran them would be a mislabelled cell.
        """
        block = dict((config or {}).get("controller") or {})
        unknown = sorted(set(block) - VISP_CONTROLLER_KEYS)
        if unknown:
            raise ValueError(f"unknown controller keys under law 'visp': {unknown}")
        w = block.get("weighting") or {}
        if any(w.get(k, False) for k in ("use_trust", "use_margin", "per_axis")):
            raise ValueError(
                "controller.law 'visp' is ViSP's unweighted law: state "
                "weighting {use_trust: false, use_margin: false, per_axis: false} or "
                "drop the block. Per-row weights belong to the bench's own IBVSController.")
        if float(block.get("ema_alpha", 0.0)) != 0.0:
            raise ValueError("controller.law 'visp' has no EMA smoother; ema_alpha must be 0")
        if not bool(block.get("solve_unrounded", True)):
            raise ValueError("controller.law 'visp' works in sub-pixel normalised "
                             "coordinates; solve_unrounded: false is not part of it")
        ag = block.get("adaptive_gain")
        if ag is not None:
            if "gain" in block:
                raise ValueError("state controller.gain OR controller.adaptive_gain, not both")
            gain: float | AdaptiveGain = AdaptiveGain(float(ag["gain_at_zero"]),
                                                      float(ag["gain_at_infinity"]),
                                                      float(ag["slope_at_zero"]))
        else:
            gain = float(block.get("gain", DEFAULT_GAIN))
        return cls(gain=gain, dt=float(block.get("dt", DEFAULT_DT)), intrinsics=intrinsics,
                   interaction=str(block.get("interaction", "current")),
                   backend=str(block.get("backend", "auto")),
                   min_matches=int(block.get("min_matches", MIN_MATCHES)))

    # ------------------------------------------------------------------ episode
    def reset(self) -> None:
        self._step_index = 0

    def set_goal_depth(self, goal_depth: np.ndarray | None) -> None:
        """The goal view's depth map, for ``Z*`` under ``desired`` and ``mean``.

        Called by :meth:`vsbench.vs.bench.arms.Arm.reset` with the goal frame's depth. A
        ``desired``/``mean`` cell with no goal depth raises at the first step rather
        than inventing a ``Z*``.
        """
        self._goal_depth = None if goal_depth is None else np.asarray(goal_depth)

    def _z_star(self, pts_goal: np.ndarray) -> np.ndarray:
        if self._goal_depth is None:
            raise RuntimeError(
                f"interaction {self.interaction!r} needs the goal depth Z* and none was "
                "given: the arm must hand the goal depth map to set_goal_depth()")
        d = self._goal_depth
        h, w = d.shape[:2]
        u = np.clip(pts_goal[:, 0].astype(np.int64), 0, w - 1)      # truncating, as depth_at
        v = np.clip(pts_goal[:, 1].astype(np.int64), 0, h - 1)
        return clean_depth(d[v, u])

    def step(self, rows: dict, depth_per_row: np.ndarray) -> tuple[Twist, dict]:
        t0 = time.monotonic()
        self._step_index += 1
        pts_goal = np.asarray(rows["pts_goal"], dtype=np.float64).reshape(-1, 2)
        pts_cur = np.asarray(rows["pts_cur"], dtype=np.float64).reshape(-1, 2)
        n = len(pts_cur)
        z = np.asarray(depth_per_row, dtype=np.float64).reshape(-1)
        if len(z) != n:
            raise ValueError(f"{len(z)} depths for {n} rows - one Z per row")
        debug: dict[str, Any] = {
            "step": self._step_index, "pts_goal": pts_goal, "pts_cur": pts_cur, "depth": z,
            "weights": None, "weight_source": None, "weighting": self.weighting,
            "n_eff": None, "solve_unrounded": True, "ema_alpha": 0.0,
            "law": "visp", "visp_backend": self.backend, "visp_interaction": self.interaction,
            "n_matches": n, "n_support": n,
        }
        if n < self.min_matches:
            v = np.zeros(6)
            debug.update({"degenerate": True, "feature_error_norm": float("nan"),
                          "normalised_errors": None, "lambda": None})
        else:
            s_star = normalised_coords(pts_goal, self.intrinsics)
            s = normalised_coords(pts_cur, self.intrinsics)
            z_star = (self._z_star(pts_goal) if self.interaction != "current"
                      else np.full(n, np.nan))
            law = visp_control_law_native if self.backend == "visp" else visp_control_law
            v, info = law(s, s_star, z, z_star, interaction=self.interaction,
                          gain=self.gain_spec)
            debug.update({"degenerate": False, **info})
        debug["velocity_raw"] = np.asarray(v, dtype=np.float64).copy()
        debug["step_ms"] = (time.monotonic() - t0) * 1000.0
        return Twist(np.asarray(v, dtype=np.float64)), debug
