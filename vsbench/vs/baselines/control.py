# vs-sim copy: vssim/baselines/control.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Analytic IBVS: landmarks -> camera velocity, and when to stop (unit U4).

Purpose
-------
The control law - image-based visual servoing solved as a weighted least squares
over the landmark image errors, with per-point depth and the sub-pixel solve, the
head's trust cubed times the match margin as weights - plus the termination rules
(the fielded ``dec`` plateau / lambda-decay stop, the published velocity-quiet
stop, and CNS v1's plateau stop, which protocol v2 makes the scored one). This is
the line-by-line verified port of the ART-VS control law; ``../ART-VS`` stays the
read-only reference for it.

What U4 owns is the **solve on rows**: :meth:`IBVSController.step` consumes the
row dict :meth:`vsbench.vs.baselines.match.Matches.to_rows` produces (goal/current pixels in
full-image coordinates, trust, margin) plus one depth per row, and returns a
:class:`Twist`. The image side - grabbing the frame, sampling the depth map at
the current pixels, building :class:`Intrinsics` from ``camera_info`` - is U7's;
the old interface it has to satisfy is ``base_controller.VSMethod.step(rgb,
camera_info, depth, mask)`` (``baseline_check.py:1617``), so the U7 adapter is::

    def step(self, rgb, camera_info, depth, mask=None):        # U7
        rows = self._matcher.match(rgb, mask).to_rows()
        u, v = rows["pts_cur"].T                                # U7 owns the lookup:
        h, w = depth.shape[:2]                                  # TRUNCATE, do not round
        z = clean_depth(depth[np.clip(v.astype(int), 0, h - 1),
                              np.clip(u.astype(int), 0, w - 1)])
        twist, dbg = self._ibvs.step(rows, z)                  # U4 owns this line
        self.debug_data = dbg
        return twist.as_array()

The truncation is not a detail: ``ibvs_core.sample_depth`` did ``int(np.clip(u, 0,
w-1))``, so on the sub-pixel (``_ur``) chain (100.4, 200.6) and (100.6, 200.4)
both read pixel (100, 200) - rounding instead would silently change which depth
every sub-pixel landmark gets. Two more seams U7 owns: the per-step evaluation
order of the old loop (``baseline_check.py:1731-1807`` - solve, lambda scale,
telemetry, non-finite check, divergence, quiet stop, integrate), and
:class:`CnsPlateau`'s ``landmark_ids``, which is the **linear** goal-patch index
``goal_patch[:, 0] + side * goal_patch[:, 1]`` and not the ``(K, 2)`` column
``to_rows()`` carries.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
- ``src/control/ibvs_core.py`` -> :func:`normalised_coords` (L84-97),
  :func:`interaction_matrix` (L100-119, same six entries per row pair, same
  optical-frame twist order), :func:`weighted_ls_solve` (``solve_velocity``,
  L246-347: plain ``np.linalg.pinv`` - never a damped inverse - the ``(K,)``
  row-pair and ``(K, 2)`` per-axis ``sqrt(w)`` row scaling, the raw error norm
  taken *before* weighting, the ``min_matches`` support guard counting only rows
  of non-zero weight, and the evaluation order ``(-lambda * pinv(L)) @ e``).
  :func:`clean_depth` is ``sample_depth``'s validity rule (L122-144) without the
  pixel lookup.
- ``src/control/matcher_ibvs.py`` -> :class:`Weighting` and
  :class:`IBVSController`: the fielded weight chain ``trust**gamma`` (``_w`` +
  ``_g3``, L313-317) times ``margin_eps + max(margin, 0)`` (``_mw``, L318-335)
  with the per-axis ``(w, w * (W/H)**2)`` factor applied last (``_axw``,
  L368-378); ``solve_unrounded`` (``_ur``, L166-170 + L286-287); the EMA
  (L103-110 + L384-388, ART-VS ``core/controller.py::update_ema`` verbatim:
  ``alpha`` weights the NEW twist); ``debug["step_ms"]`` (L274, L427) and
  ``ls_n_eff`` -> :func:`effective_support`
  (``matchers/significance.py:158-163``).
- ``src/control/base_controller.py`` -> the ``reset`` / ``step`` shape and the
  optical-frame ``[vx, vy, vz, wx, wy, wz]`` return contract (:class:`Twist`).
- ``src/eval/baseline_check.py`` -> :class:`PlateauTrigger` (L982-1023, verbatim)
  and :class:`DecPlateau` (``StopMechanic`` L1026-1110, the dec response at
  L1090-1098, with ``STOP_MECHANICS`` L893-903: rho 0.9, floor 0.003 x lambda,
  trigger 200 / 5 % / 10 from ``scoring.PLATEAU``);
  :class:`VelocityQuiet` (L1792-1793, eps 1e-4, patience 8);
  :class:`DivergenceGuard` (``DIVERGENCE_ABORT`` L843 + L1759-1768, 3x the first
  finite feature error).
- **new**: :class:`CnsPlateau`, CNS v1's ``PixelStopPolicy`` - protocol v2's
  scored stop (``../docs/eval_protocol_v2_2026-09-02.md`` section 5). Ported
  **line by line from the source**, which is vendored (MIT) at
  ``tests/fixtures/reference/cns_stop_policy.py`` and which the golden tests run
  beside the port on every trace. The first implementation was written from a
  prose description instead and diverged in five places, firing 90-185 steps
  early on a settling error and firing at all on a slow creep, where CNS never
  does (review R12, 2026-09-03): a stop that scores every arm is not something to
  infer from a summary. Constants and their citations: :data:`CNS_STOP`.

- **2026-10-05, the faithful ART-VS arm** (``docs/results/2026-10-05_artvs_release_check/
  sim_spec.md`` section 4.4-4.5, workspace root), from the ART-VS release (``../ART-VS``,
  git ``9b1208c``, MIT): ``ros_interface/ros_controller.py`` ``calculate_uv_roi``
  (892-917) -> ``pixel_quantisation: truncate``; ``get_depth`` (407-428) ->
  ``depth_fallback_m`` (100 m there); ``run``'s skip of a frame with too few points
  (2688-2691) -> ``degenerate_policy: hold``; ``core/controller.py``
  ``store_applied_velocities`` (154-165) -> ``max_velocity``, and
  ``check_convergence_visual_only`` (326-385) -> :class:`ArtvsVisual90`, a recorded stop
  reading the new :attr:`StopObservation.pixel_error_mean`. All additive: at their
  defaults the controller and every record are what they were. The release's file is
  vendored at ``tests/fixtures/reference/artvs_release/controller.py`` and the fast tier
  runs it beside these.

Deliberately dropped (each is a measured null, a different unit's job, or dead)
------------------------------------------------------------------------------
- ``analytic_point_weights`` (``_alw``, ``ibvs_core.py:151-181``): the closed-form
  leverage weights measured a **tie** against the fielded chain (N3.7a), and they
  are a whole second weight source. Dropped; the old function is the reference if
  the null is ever re-run.
- ``depth_from_goal`` (``_cz``, constant-Z, ``matcher_ibvs.py:192-197``): measured
  **null** (N1b). No flag is needed here anyway - the caller passes
  ``depth_per_row``, so a constant-Z arm is "pass Z* instead of the live depth".
- ``irls_point_weights`` (``_irlsh`` / ``_irlst``), the consensus weights
  (``_cons`` / ``_consr``), ``avg_n`` terminal point averaging, the ``_dopt`` /
  ``_sel`` selection hooks, ``lw_checkpoint`` (U5 owns the head; its trust
  arrives in the rows), ``IBVSConfig.damping`` (never fielded; the published path
  is the plain pseudo-inverse), and the whole arm-suffix grammar
  (``baseline_check.py:275-690``) - **the config replaces it**
  (``vsbench.vs.bench.config``).
- ``rot_first`` (``matcher_ibvs.py:188-199`` + ``L433-455``) -> **U8** ``rotation.py``.
- ``_i2`` is a *head* parameter (head passes per match), not a control one, and
  the fielded chain ``head_w_g3_axw_mw_ur`` does not use it -> U5.
- ``sample_depth``'s pixel lookup and ``Intrinsics.from_camera_info``'s caller ->
  U7 owns the image side; the constants and the validity rule stay here.

Fences
------
- **Whole-pixel rounding is a ViT-VS quirk, not classical practice.** ViSP and
  every classical IBVS text solve on sub-pixel feature coordinates; the published
  ViT-VS code rounds to integers before applying the intrinsics
  (``vit_vs.py`` -> ``ibvs_core``'s module docstring). Protocol v2 puts sub-pixel
  on every arm, so ``solve_unrounded`` **defaults to True** here (the old default
  was False). The rounded path stays reachable for published-ViT-VS parity, and
  the fielded arm is ``head_w_g3_axw_mw_ur``.
- **The 0 x inf NaN trap** (S2.b live-fix lesson): margins are ``+inf`` on an
  all-excluded row (``vsbench.vs.baselines.match.margins_for_matches``) and ``trust**3`` is
  ``0.0`` for a rejected match, so the fielded product silently becomes ``NaN``,
  and ``sqrt(w)`` spreads it over the whole solve. :class:`Weighting` therefore
  clamps the margin to :data:`MARGIN_CLIP_MAX` before multiplying, and
  :func:`weighted_ls_solve` refuses non-finite weights loudly.
- **GT-blind.** Every stop rule and the divergence guard read only
  controller-visible quantities - the feature-error norm, the commanded twist,
  the per-landmark normalised errors. The true pose never enters this module, and
  :class:`StopObservation` is deliberately the only way in.
- Weights are scale-free: the least squares is invariant to a global factor, so
  only the *shape* of the weight vector matters.

Golden tests that guard it
--------------------------
``tests/fast/test_control.py`` - synthetic points converge below 1e-6 m on both
the rounded and the unrounded path, the four stop rules fire (and refuse to fire)
where they should, and one stored episode of the N2-gate telemetry re-solves to
the recorded velocity. Fixture provenance: ``tests/fixtures/u4_fixture_provenance.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------- constants

#: Campaign gain and control period: ``--gain 2.0``, ``control_dt = 1/60``
#: (protocol v2 section 5; ``baseline_check.py`` CLI defaults).
DEFAULT_GAIN: float = 2.0
DEFAULT_DT: float = 1.0 / 60.0

#: Below this many supporting rows IBVS is unstable and the twist is zero
#: (``ibvs_core.IBVSConfig.min_matches``).
MIN_MATCHES: int = 4

#: Depth validity, verbatim from ``ViTVSMethod._sample_depth`` via
#: ``ibvs_core.sample_depth``: z <= 0, z > 10 m, NaN or Inf becomes 0.5 m.
DEPTH_MAX_M: float = 10.0
DEPTH_FALLBACK_M: float = 0.5

#: ``_mw`` margin weight offset (``matchers/significance.MARGIN_WEIGHT_EPS``).
MARGIN_WEIGHT_EPS: float = 0.01

#: NEW fence (not in the old code): the largest margin that may reach the weight
#: product. ``margins_for_matches`` returns ``+inf`` for an all-excluded row and
#: ``trust ** 3`` is ``0.0`` for a rejected match, so ``0 * inf = NaN`` would
#: poison the solve through ``sqrt(w)``; the old code raised
#: ``ValueError("weights must be finite")`` instead and ended the episode.
#:
#: **OPEN ITEM for the owner / U5** (review R12, 2026-09-03): 1e6 is far above any
#: real margin - similarities live in [-1, 1], so a *finite* margin is at most 2 -
#: which means an *undefined* margin currently becomes a row that dominates the
#: whole solve whenever its trust is above zero (``trust**3 * 1e6``). Clamping at
#: 2.0, the largest margin that can exist, would make such a row merely the
#: strongest legal one. Both are defensible and the choice changes numbers, so it
#: is not an agent's to make: raise it with the head work (U5), where the first
#: arm that can produce an all-excluded row lives.
MARGIN_CLIP_MAX: float = 1.0e6

#: The in-force plateau read (``scoring.PLATEAU``): the mean of the last
#: ``window`` feature-error norms must sit within ``rel_tol`` of the mean of the
#: ``window`` before it, ``sustain`` steps running.
PLATEAU: dict[str, float] = {"window": 200, "rel_tol": 0.05, "sustain": 10}

#: The fielded ``dec`` response (``baseline_check.STOP_MECHANICS``): once the
#: plateau trigger fires, lambda decays by ``rho`` per step down to ``floor``
#: times the base gain. Calibrated over 770 committed telemetry episodes
#: (2026-08-17); ``floor`` was lowered 0.005 -> 0.003 by owner ruling 2026-08-19
#: so the published quiet stop can actually fire on the ``off`` arm.
DEC: dict[str, float] = {"rho": 0.9, "floor": 0.003}

#: The published velocity-quiet stop, verbatim (``STOP_MECHANICS``
#: ``stop_vel_eps`` / ``stop_patience``; ``baseline_check.py:1792``). Recorded on
#: every arm, never redefined by a bench.
VELOCITY_QUIET: dict[str, float] = {"eps": 1.0e-4, "patience": 8}

#: N0 divergence abort (``DIVERGENCE_ABORT``, frozen protocol section 4 and
#: protocol v2 section 5): controller-visible feature error above 3x its first
#: finite value ends the episode as diverged.
DIVERGENCE_MULT: float = 3.0

#: CNS v1's ``PixelStopPolicy`` - protocol v2's **scored** stop. Every constant is
#: cited to the line of CNS's shipped code it comes from, and that code is
#: **vendored** at ``tests/fixtures/reference/cns_stop_policy.py`` (MIT) so the
#: port is tested against the source rather than against a reading of it:
#:
#: ``conduct_thresh``  arming gate, ``benchmark.py:220``
#:                     ``PixelStopPolicy(waiting_time=0.8, conduct_thresh=0.01)``;
#:                     the policy is reset while the error is above it
#:                     (``stop_policy.py:34-36``). In normalised image units -
#:                     5.4 px at CNS's f = 540, 14.1 px at our f = 1406.75.
#: ``waiting_time``    hold duration in seconds, same call, tested with a strict
#:                     ``>`` (``stop_policy.py:66``). CNS clocks it as
#:                     ``env.steps * env.dt`` with ``dt = 1/50``
#:                     (``sim/environment.py:64``) = **41 intervals**; we run at
#:                     60 Hz, so it is **49** (protocol v2 section 5 rounds both
#:                     to "40" and "48 steps" - the nominal hold, not the count).
#: ``min_reset``       ``cur < 0.9 * min_err`` is a new minimum: it moves the
#:                     band, restarts the hold and accrues nothing this step
#:                     (``stop_policy.py:38-43``). Anything shallower leaves
#:                     ``min_err`` where it was.
#: ``hold_mult`` /     the hold band ``thresh = 1.1 * min_err + 5e-4``
#: ``hold_abs``        (``stop_policy.py:45``).
#: ``weight_decay``    ``0.95``: a landmark missing this frame decays, a
#: ``max_weight``      re-sighted one recovers as ``0.95 w + 1``, and only a
#:                     landmark's **first** sighting sets ``1/(1-0.95) = 20``
#:                     (``stop_policy.py:91-92, 109-115``).
#: ``trim_percentile`` the **weighted** errors above the 90th percentile are
#:                     dropped, with a strict ``<`` (``stop_policy.py:120-124``).
CNS_STOP: dict[str, float] = {
    "conduct_thresh": 0.01,
    "waiting_time": 0.8,
    "min_reset": 0.9,
    "hold_mult": 1.1,
    "hold_abs": 5.0e-4,
    "weight_decay": 0.95,
    "max_weight": 20.0,
    "trim_percentile": 90.0,
}


# ------------------------------------------------------------------- geometry

@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in **full-image** pixels (``ibvs_core.Intrinsics``).

    The fielded camera is the D435i at 1920x1080 with a 42 deg vertical field of
    view, so ``fx = fy = height / (2 tan 21 deg) = 1406.75`` and the principal
    point is the image centre (``genvs-sim/configs/camera.yaml`` ->
    ``free_flying_scene.camera_info``). A matcher that works at a reduced
    resolution maps its points back itself
    (:meth:`vsbench.vs.baselines.match.Matches.to_rows`); the control law never rescales.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1920
    height: int = 1080

    @classmethod
    def from_camera_info(cls, camera_info: dict, fallback: Intrinsics | None = None) -> Intrinsics:
        """Read a ``camera_info`` dict, keeping ``fallback`` for absent keys.

        Mirrors ``ibvs_core.Intrinsics.from_camera_info``, minus its D455
        hard-coded default: there is **no** fallback camera here. A caller that
        gives neither a complete dict nor a fallback gets a ``KeyError`` naming
        the field, because a silently wrong focal length is a silently wrong
        millimetre (the old repo's OPEN-2).
        """
        info = camera_info or {}
        keys = (("f_x", "fx"), ("f_y", "fy"), ("c_x", "cx"), ("c_y", "cy"),
                ("width", "width"), ("height", "height"))
        out: dict[str, Any] = {}
        for src, dst in keys:
            if src in info and info[src] is not None:
                out[dst] = info[src]
            elif fallback is not None:
                out[dst] = getattr(fallback, dst)
            else:
                raise KeyError(f"camera_info has no {src!r} and no fallback Intrinsics was given")
        return cls(fx=float(out["fx"]), fy=float(out["fy"]), cx=float(out["cx"]),
                   cy=float(out["cy"]), width=int(out["width"]), height=int(out["height"]))

    @property
    def aspect(self) -> float:
        """``width / height`` - the ``_axw`` per-axis noise ratio's square root."""
        return float(self.width) / float(self.height)


@dataclass(frozen=True)
class Twist:
    """A camera-frame velocity ``[vx, vy, vz, wx, wy, wz]``, metres/s and rad/s.

    The optical frame and the ordering of ``base_controller.VSMethod.step``'s
    return value, unchanged. The scene integrates it as
    ``pos += R @ v_lin * dt`` and ``R <- R @ Rodrigues(w_hat, |w| dt)``
    (``free_flying_scene.integrate_optical_velocity`` L134-146) - the convention
    the synthetic golden test uses.
    """

    data: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float64).reshape(-1)
        if arr.size != 6:
            raise ValueError(f"a Twist is 6 numbers [vx vy vz wx wy wz], got {arr.size}")
        object.__setattr__(self, "data", arr)

    @classmethod
    def zero(cls) -> Twist:
        return cls(np.zeros(6))

    @property
    def linear(self) -> np.ndarray:
        return self.data[:3]

    @property
    def angular(self) -> np.ndarray:
        return self.data[3:]

    def as_array(self) -> np.ndarray:
        """The plain 6-vector the harness and the scene integrator consume."""
        return self.data.copy()

    def __array__(self, dtype=None, copy=None):  # noqa: D105 - numpy interop
        arr = self.data if dtype is None else self.data.astype(dtype)
        return arr.copy() if copy else arr

    def norm(self) -> float:
        """``||v||`` over all six components - the published quiet stop's signal."""
        return float(np.linalg.norm(self.data))


def clean_depth(depth: np.ndarray, fallback_m: float = DEPTH_FALLBACK_M) -> np.ndarray:
    """Per-row depth with invalid samples replaced by ``fallback_m``.

    ``ibvs_core.sample_depth``'s validity rule (z <= 0, z > 10 m, NaN, Inf),
    lifted away from the pixel lookup: U7 samples the depth map, U4 owns what
    "unusable" means. Returns float64. ``fallback_m`` is :data:`DEPTH_FALLBACK_M`
    (0.5 m, ViT-VS's) unless an arm states another: the ART-VS release's simulation
    program reads 0 - and its depth plugin writes 0 beyond 10 m - as **100 m**
    (``ros_controller.py:423-425``; ``controller.depth_fallback_m``, 2026-10-05).
    """
    z = np.asarray(depth, dtype=np.float64).reshape(-1).copy()
    bad = ~np.isfinite(z) | (z <= 0) | (z > DEPTH_MAX_M)
    z[bad] = float(fallback_m)
    return z


def normalised_coords(pts_px: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Full-image pixels ``(u, v)`` -> ``((u - cx)/fx, (v - cy)/fy)``, ``(K, 2)``."""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([(pts[:, 0] - intr.cx) / intr.fx,
                            (pts[:, 1] - intr.cy) / intr.fy])


def interaction_matrix(s_xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Stacked point-feature interaction matrix ``L``, shape ``(2K, 6)``.

    One row pair per point, optical frame, twist order ``[vx vy vz wx wy wz]``::

        [-1/Z,     0,  x/Z,      x*y,  -(1 + x^2),   y]
        [    0, -1/Z,  y/Z,  1 + y^2,       -x*y,   -x]

    Same six expressions as ``ibvs_core.interaction_matrix``. The old version
    kept a Python loop to protect a float-summation order; there is no summation
    here - every entry is one elementwise double operation - so the vectorised
    form is bit-identical, and the recorded-telemetry parity test measures that
    it is. **Per-point depth**: ``depth`` is one Z per point, never a scalar
    stand-in (that ablation is the measured-null ``_cz``).
    """
    s = np.asarray(s_xy, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    if len(z) != len(s):
        raise ValueError(f"{len(z)} depths for {len(s)} points - one Z per point")
    x, y = s[:, 0], s[:, 1]
    zero = np.zeros_like(x)
    L = np.empty((2 * len(s), 6), dtype=np.float64)
    L[0::2] = np.column_stack([-1 / z, zero, x / z, x * y, -(1 + x ** 2), y])
    L[1::2] = np.column_stack([zero, -1 / z, y / z, 1 + y ** 2, -x * y, -x])
    return L


def effective_support(weights: np.ndarray) -> float:
    """Kish effective sample size ``n_eff = (sum w)^2 / sum w^2``.

    ``matchers/significance.effective_support`` verbatim: two-column (per-axis)
    weights are **flattened**, so a ``(K, 2)`` weight vector reports up to ``2K``
    and not ``K``; all-zero weights report 0.0. Recorded per step as
    ``debug["n_eff"]`` - the support-concentration diagnostic.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    s2 = float((w ** 2).sum())
    return 0.0 if s2 <= 0 else float(w.sum() ** 2 / s2)


# --------------------------------------------------------------------- the solve

def weighted_ls_solve(pts_goal_px: np.ndarray,
                      pts_cur_px: np.ndarray,
                      depth: np.ndarray,
                      intr: Intrinsics,
                      *,
                      gain: float,
                      weights: np.ndarray | None = None,
                      min_matches: int = MIN_MATCHES) -> tuple[np.ndarray, dict]:
    """The IBVS control law ``v = -lambda * L^+ (s - s*)``, weighted.

    ``ibvs_core.solve_velocity``, with its two documented consequences kept:
    a **plain** pseudo-inverse (never damped), and points that may be rounded
    integers or sub-pixel floats - the caller decides
    (:attr:`IBVSController.solve_unrounded`).

    ``weights`` is ``None`` (the unweighted path, bit-identical to the published
    one), ``(K,)`` - point ``i``'s **row pair** is scaled by ``sqrt(w_i)`` - or
    ``(K, 2)`` - column 0 scales the x-row and column 1 the y-row, generalised
    least squares with a per-axis diagonal noise model (the ``_axw`` lever: after
    the 16:9 squash an x-measurement carries ``(16/9)^2`` the noise variance of a
    y-measurement). Both forms solve the same weighted normal equations, and the
    solve is invariant to a global scale on ``w``.

    Returns ``(v, info)``. ``v`` is the raw 6-vector (the caller wraps it in a
    :class:`Twist`); ``info`` carries ``n_matches``, ``feature_error_norm`` (the
    **raw** ``||e||``, taken before any weighting - the signal every stop rule
    and the divergence guard read), ``normalised_errors`` (per-landmark
    ``||s - s*||``, CNS's stop signal), ``degenerate`` and ``n_support``. Fewer
    than ``min_matches`` supporting rows -> a zero twist, ``degenerate=True``,
    and a NaN error norm, exactly as before.
    """
    pts_goal = np.asarray(pts_goal_px, dtype=np.float64).reshape(-1, 2)
    pts_cur = np.asarray(pts_cur_px, dtype=np.float64).reshape(-1, 2)
    if len(pts_goal) != len(pts_cur):
        raise ValueError(f"{len(pts_goal)} goal points vs {len(pts_cur)} current points")
    n = len(pts_cur)

    per_axis = False
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim == 1 and len(weights) == n:
            per_axis = False
        elif weights.shape == (n, 2):
            per_axis = True
        else:
            raise ValueError(f"weights shape {weights.shape} for {n} points (want (K,) or (K, 2))")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(
                "weights must be finite and non-negative - a NaN here is usually the "
                "0 x inf trap: an infinite margin times a zero trust (see MARGIN_CLIP_MAX)"
            )
        n_support = int(np.count_nonzero(weights.max(axis=1) if per_axis else weights))
    else:
        n_support = n

    if n_support < min_matches:
        return np.zeros(6), {"n_matches": n, "n_support": n_support, "degenerate": True,
                             "feature_error_norm": float("nan"), "normalised_errors": None}

    s_star = normalised_coords(pts_goal, intr)
    s = normalised_coords(pts_cur, intr)
    L = interaction_matrix(s, depth)
    e = (s - s_star).reshape(-1, 1)
    err_norm = float(np.linalg.norm(e))          # raw error, before any weighting

    if weights is not None:
        row_scale = (np.sqrt(weights).reshape(-1, 1) if per_axis
                     else np.repeat(np.sqrt(weights), 2).reshape(-1, 1))
        L = L * row_scale
        e = e * row_scale

    v = -float(gain) * np.linalg.pinv(L) @ e
    v = v.flatten()
    return v, {"n_matches": n, "n_support": n_support, "degenerate": False,
               "feature_error_norm": err_norm,
               "normalised_errors": np.linalg.norm(s - s_star, axis=1)}


# ------------------------------------------------------------------- weighting

@dataclass(frozen=True)
class Weighting:
    """How a row's significance becomes its weight in the least squares.

    The **fielded recipe** (:data:`FIELDED_WEIGHTING`, the arm
    ``head_w_g3_axw_mw_ur``) is, in order:

    1. ``_w`` + ``_g<gamma>``: ``w = max(trust, 0) ** trust_gamma`` with
       ``trust_gamma = 3``;
    2. ``_mw``: ``w *= margin_eps + clip(margin, 0, MARGIN_CLIP_MAX)``;
    3. ``_axw``: ``w -> (w, w * (W/H)**2)``, applied **last**, exactly as
       ``matcher_ibvs.step`` composes them.

    Which column is "trust": the old controller weighted by ``corr.conf``, the
    provider's confidence - the head's trust on a refine arm and the raw cosine
    similarity on a bare dense arm. :meth:`vsbench.vs.baselines.match.Matches.to_rows` splits
    those into ``trust`` (the head's, present only once U5 has filled it) and
    ``conf`` (the similarity), so this reads ``trust`` when it is there and
    ``conf`` otherwise, and records which in ``debug["weight_source"]``.
    """

    use_trust: bool = True
    trust_gamma: float = 3.0
    use_margin: bool = True
    margin_eps: float = MARGIN_WEIGHT_EPS
    per_axis: bool = True

    def __post_init__(self) -> None:
        if self.trust_gamma <= 0:
            raise ValueError(f"trust_gamma must be > 0, got {self.trust_gamma}")
        if self.margin_eps < 0:
            raise ValueError(f"margin_eps must be >= 0, got {self.margin_eps}")

    @property
    def is_uniform(self) -> bool:
        """True when nothing but the per-axis factor shapes the weights."""
        return not (self.use_trust or self.use_margin or self.per_axis)

    @classmethod
    def from_config(cls, config: dict) -> Weighting:
        """Build from ``config["controller"]["weighting"]``; absent keys keep the
        fielded value, which is recorded on the object rather than implied."""
        block = ((config or {}).get("controller") or {}).get("weighting") or {}
        unknown = sorted(set(block) - {f.name for f in cls.__dataclass_fields__.values()})
        if unknown:
            raise ValueError(f"unknown controller.weighting keys: {unknown}")
        return replace(cls(), **block)

    def __call__(self, rows: dict, *, image_wh: tuple[int, int]) -> tuple[np.ndarray | None, dict]:
        """Rows -> ``(weights, debug)``; ``weights`` is ``None``, ``(K,)`` or ``(K, 2)``."""
        n = len(np.asarray(rows["pts_cur"]).reshape(-1, 2))
        w: np.ndarray | None = None
        source = None
        if self.use_trust:
            trust = rows.get("trust")
            source = "trust"
            if trust is None:
                trust = rows.get("conf")
                source = "conf"
            if trust is None:
                raise ValueError(
                    "Weighting(use_trust=True) needs rows['trust'] (the head's) or "
                    "rows['conf'] (the matcher's) - an inert weighting would fake a measurement"
                )
            w = np.clip(np.asarray(trust, dtype=np.float64).reshape(-1), 0.0, None)
            if self.trust_gamma != 1.0:
                w = w ** self.trust_gamma
        if self.use_margin:
            margin = rows.get("margin")
            if margin is None:
                raise ValueError(
                    "Weighting(use_margin=True) needs rows['margin'] - build the matcher "
                    "with margins (the _mw arm wiring)"
                )
            m = np.asarray(margin, dtype=np.float64).reshape(-1)
            if len(m) != n:
                raise ValueError(f"{len(m)} margins for {n} rows - the arrays are row-aligned")
            # the 0 x inf fence: +inf is a legal margin (an all-excluded row)
            m_w = self.margin_eps + np.clip(m, 0.0, MARGIN_CLIP_MAX)
            w = m_w if w is None else w * m_w
        if self.per_axis:
            ratio = float(image_wh[0]) / float(image_wh[1])
            base = np.ones(n, dtype=np.float64) if w is None else w
            w = np.column_stack([base, base * ratio * ratio])
        debug = {"weight_source": source, "weighting": self,
                 "n_eff": None if w is None else effective_support(w)}
        return w, debug


#: The fielded chain ``head_w_g3_axw_mw`` (protocol v2 section 4). ``_ur`` is not
#: here: sub-pixel is a controller flag, not a weight.
FIELDED_WEIGHTING = Weighting()

#: The unweighted published path (ViT-VS / ART-VS / the classical baselines).
UNIFORM_WEIGHTING = Weighting(use_trust=False, use_margin=False, per_axis=False)


# ------------------------------------------------------------------ controller

#: ``controller.pixel_quantisation``: what happens to the matcher's pixels before the
#: intrinsics. ``none`` - sub-pixel (protocol v2, every arm); ``round`` - the published
#: ViT-VS whole-pixel path; ``truncate`` - the ART-VS release's ``calculate_uv_roi``
#: (``astype(int)`` on both sides, then the clip to the frame; 2026-10-05). Without the key
#: the mode follows ``solve_unrounded`` exactly as before (``none`` / ``round``).
PIXEL_QUANTISATIONS = ("none", "round", "truncate")

#: ``controller.degenerate_policy``: what a frame with too few rows commands. ``zero`` -
#: a zero twist, and the EMA absorbs it (every arm before 2026-10-05); ``hold`` - the last
#: commanded twist again, the EMA untouched: the ART-VS release publishes nothing on such a
#: frame and Gazebo keeps the camera moving at the last twist (``ros_controller.py
#: :2688-2691``; owner decision D-E). With nothing to hold yet, ``hold`` commands zero.
DEGENERATE_POLICIES = ("zero", "hold")

#: How close to a whole pixel a coordinate must be to count as that pixel before a
#: truncation. ``Matches.from_pixel_pairs(...).to_rows()`` scales the row axis by 16/9 and
#: back by 9/16 (``Matches._split``), which moves a whole pixel by up to 1.1e-13 - and a
#: bare ``np.trunc`` then turns 7.0 into 6 (313 of 21 080 rows in a probe, 2026-10-05). The
#: ART-VS points are float32 numbers, and no float32 coordinate above 2**-6 lies within
#: 1e-9 of a whole pixel without being one, so the snap is exactly ``astype(int)`` on them.
TRUNCATE_SNAP = 1.0e-9


def truncate_px(pts: np.ndarray) -> np.ndarray:
    """``np.asarray(p).astype(int)`` (truncation toward zero) of pixel coordinates that
    may have taken the row-axis round trip of :data:`TRUNCATE_SNAP` - float64 out."""
    p = np.asarray(pts, dtype=np.float64)
    near = np.rint(p)
    return np.where(np.abs(p - near) <= TRUNCATE_SNAP, near, np.trunc(p)) + 0.0


class IBVSController:
    """The one control law: matcher rows in, a :class:`Twist` out.

    Every method in the study differs only in how it produces correspondences -
    the maths below is shared, which is the whole point of the extraction
    (``ibvs_core``'s docstring). Construction records what it will do; nothing
    here is implied.

    Args:
        gain: lambda in ``v = -lambda L^+ e``. Campaign value 2.0.
        dt: the control period the harness integrates at, 1/60 s. The law itself
            does not use it - it is carried so the stop rules can convert their
            step counts into CNS's seconds, and so U7 has one source for it.
        intrinsics: full-image pinhole intrinsics. Required - there is no
            fallback camera.
        solve_unrounded: **True by default** (protocol v2: sub-pixel on every
            arm). False reproduces the published ViT-VS integer-pixel path.
        weighting: how significance becomes weight; the fielded recipe by
            default. Pass :data:`UNIFORM_WEIGHTING` for an unweighted baseline.
        ema_alpha: ART-VS's twist smoother, ``ema = alpha * new + (1 - alpha) *
            previous``. **0.0 disables it** and is the fielded value (off since
            N0); note the direction - their ``0.8`` is *light* smoothing.
        min_matches: below this many supporting rows the twist is zero.
        pixel_quantisation: :data:`PIXEL_QUANTISATIONS`; ``None`` (default) follows
            ``solve_unrounded`` as it always did. Given, it wins and
            ``solve_unrounded`` is set to ``pixel_quantisation == "none"``. A frame's
            rows may override it for that frame (``rows["quantise"]`` - the ART-VS
            box-less fallback rounds, ``calculate_uv``).
        max_velocity: ``None`` (default, every arm before 2026-10-05) or the per-component
            cap the ART-VS release applies after its EMA (``np.clip(v, -c, c)``,
            ``core/controller.py:154-165``).
        degenerate_policy: :data:`DEGENERATE_POLICIES`, default ``zero``.
        depth_fallback_m: the metres an unusable depth sample becomes
            (:data:`DEPTH_FALLBACK_M` unless stated) - read by the arm's depth lookup
            (:func:`vsbench.vs.bench.arms.depth_at`), carried here so a config states it once.

    The four knobs of 2026-10-05 are additive: at their defaults the controller is the
    one every stored record ran, byte for byte (``tests/fast/test_control.py``), and
    :meth:`record_fields` is empty.
    """

    def __init__(self,
                 gain: float = DEFAULT_GAIN,
                 dt: float = DEFAULT_DT,
                 *,
                 intrinsics: Intrinsics,
                 solve_unrounded: bool = True,
                 weighting: Weighting = FIELDED_WEIGHTING,
                 ema_alpha: float = 0.0,
                 min_matches: int = MIN_MATCHES,
                 pixel_quantisation: str | None = None,
                 max_velocity: float | None = None,
                 degenerate_policy: str = "zero",
                 depth_fallback_m: float = DEPTH_FALLBACK_M) -> None:
        if gain <= 0:
            raise ValueError(f"gain must be > 0, got {gain}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0, 1], got {ema_alpha}")
        if pixel_quantisation is not None and pixel_quantisation not in PIXEL_QUANTISATIONS:
            raise ValueError(f"pixel_quantisation must be one of {PIXEL_QUANTISATIONS}, got "
                             f"{pixel_quantisation!r}")
        if max_velocity is not None and not float(max_velocity) > 0:
            raise ValueError(f"max_velocity must be > 0 (or None), got {max_velocity}")
        if degenerate_policy not in DEGENERATE_POLICIES:
            raise ValueError(f"degenerate_policy must be one of {DEGENERATE_POLICIES}, got "
                             f"{degenerate_policy!r}")
        if not float(depth_fallback_m) > 0:
            raise ValueError(f"depth_fallback_m must be > 0, got {depth_fallback_m}")
        self.gain = float(gain)
        self.dt = float(dt)
        self.intrinsics = intrinsics
        if pixel_quantisation is None:
            pixel_quantisation = "none" if solve_unrounded else "round"
        self.pixel_quantisation = str(pixel_quantisation)
        self.solve_unrounded = self.pixel_quantisation == "none"
        self.weighting = weighting
        self.ema_alpha = float(ema_alpha)
        self.min_matches = int(min_matches)
        self.max_velocity = None if max_velocity is None else float(max_velocity)
        self.degenerate_policy = str(degenerate_policy)
        self.depth_fallback_m = float(depth_fallback_m)
        self._ema_v: np.ndarray | None = None
        self._last_cmd: np.ndarray | None = None
        self._step_index = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"IBVSController(gain={self.gain}, dt={self.dt:.6g}, "
                f"solve_unrounded={self.solve_unrounded}, weighting={self.weighting}, "
                f"ema_alpha={self.ema_alpha}, pixel_quantisation={self.pixel_quantisation!r}, "
                f"max_velocity={self.max_velocity}, degenerate_policy="
                f"{self.degenerate_policy!r}, depth_fallback_m={self.depth_fallback_m})")

    @classmethod
    def from_config(cls, config: dict, *, intrinsics: Intrinsics) -> IBVSController:
        """Build from ``config["controller"]``; the weighting comes from
        :meth:`Weighting.from_config`. Absent keys take this module's documented
        defaults, which the returned object records. ``solve_unrounded`` and
        ``pixel_quantisation`` stated together must agree."""
        block = (config or {}).get("controller") or {}
        pq = block.get("pixel_quantisation")
        if pq is not None and "solve_unrounded" in block \
                and bool(block["solve_unrounded"]) != (pq == "none"):
            raise ValueError(
                f"controller.solve_unrounded {block['solve_unrounded']!r} and "
                f"controller.pixel_quantisation {pq!r} disagree - state one of them")
        mv = block.get("max_velocity")
        return cls(gain=float(block.get("gain", DEFAULT_GAIN)),
                   dt=float(block.get("dt", DEFAULT_DT)),
                   intrinsics=intrinsics,
                   solve_unrounded=bool(block.get("solve_unrounded", True)),
                   weighting=Weighting.from_config(config),
                   ema_alpha=float(block.get("ema_alpha", 0.0)),
                   min_matches=int(block.get("min_matches", MIN_MATCHES)),
                   pixel_quantisation=None if pq is None else str(pq),
                   max_velocity=None if mv is None else float(mv),
                   degenerate_policy=str(block.get("degenerate_policy", "zero")),
                   depth_fallback_m=float(block.get("depth_fallback_m", DEPTH_FALLBACK_M)))

    def record_fields(self) -> dict:
        """The 2026-10-05 knobs this controller runs at a non-default value - and nothing
        when it runs at the defaults, so every record before them stays byte for byte."""
        out: dict[str, Any] = {}
        if self.pixel_quantisation == "truncate":
            out["pixel_quantisation"] = self.pixel_quantisation
        if self.max_velocity is not None:
            out["max_velocity"] = self.max_velocity
        if self.degenerate_policy != "zero":
            out["degenerate_policy"] = self.degenerate_policy
        if self.depth_fallback_m != DEPTH_FALLBACK_M:
            out["depth_fallback_m"] = self.depth_fallback_m
        return out

    def reset(self) -> None:
        """Start of an episode: never carry smoothing or a step count across one."""
        self._ema_v = None
        self._last_cmd = None
        self._step_index = 0

    def quantise(self, rows: dict) -> tuple[np.ndarray, np.ndarray, str]:
        """``(pts_goal, pts_cur, mode)`` - the frame's pixels as the law will use them.

        ``mode`` is :attr:`pixel_quantisation` unless ``rows["quantise"]`` names another
        for this frame. ``truncate`` is the ART-VS release's ``calculate_uv_roi``
        (``ros_controller.py:899-914``): :func:`truncate_px` on both sides, then the clip
        to ``[0, width - 1] x [0, height - 1]``; ``round`` is ``np.round`` (no clip, the
        published ViT-VS path and the release's ``calculate_uv``); ``none`` leaves them.
        """
        pts_goal = np.asarray(rows["pts_goal"], dtype=np.float64).reshape(-1, 2)
        pts_cur = np.asarray(rows["pts_cur"], dtype=np.float64).reshape(-1, 2)
        mode = str(rows.get("quantise") or self.pixel_quantisation)
        if mode not in PIXEL_QUANTISATIONS:
            raise ValueError(f"rows['quantise'] is {mode!r}, not one of {PIXEL_QUANTISATIONS}")
        if mode == "round":
            return np.round(pts_goal), np.round(pts_cur), mode
        if mode == "truncate":
            hi = np.asarray([self.intrinsics.width - 1, self.intrinsics.height - 1], np.float64)
            return (np.clip(truncate_px(pts_goal), 0.0, hi),
                    np.clip(truncate_px(pts_cur), 0.0, hi), mode)
        return pts_goal, pts_cur, mode

    def step(self, rows: dict, depth_per_row: np.ndarray) -> tuple[Twist, dict]:
        """One control step from matcher rows and one depth per row.

        ``rows`` is :meth:`vsbench.vs.baselines.match.Matches.to_rows`' dict - ``pts_goal``
        and ``pts_cur`` as ``(K, 2)`` full-image ``(column, row)`` pixels, plus
        the significance columns the weighting asks for. ``depth_per_row`` is
        ``(K,)`` metres at the current points; run it through
        :func:`clean_depth` first if it came from a raw depth map.

        Returns ``(twist, debug)``. ``debug`` carries ``step_ms``, ``n_eff``
        (Kish), ``weights``, ``feature_error_norm`` and ``normalised_errors`` -
        everything a stop rule may read, and nothing that is not
        controller-visible.
        """
        t0 = time.monotonic()
        self._step_index += 1
        # `round`: the published ViT-VS parity path (round before applying intrinsics);
        # `truncate`: the ART-VS release's whole pixels; `none`: sub-pixel (quantise())
        pts_goal, pts_cur, q_mode = self.quantise(rows)
        n = len(pts_cur)
        z = np.asarray(depth_per_row, dtype=np.float64).reshape(-1)
        if len(z) != n:
            raise ValueError(f"{len(z)} depths for {n} rows - one Z per row (per-point depth)")

        weights, w_debug = (None, {"weight_source": None, "weighting": self.weighting,
                                   "n_eff": None})
        if n:
            weights, w_debug = self.weighting({**rows, "pts_cur": pts_cur},
                                              image_wh=(self.intrinsics.width,
                                                        self.intrinsics.height))

        if n < self.min_matches:
            v_raw = np.zeros(6)
            info = {"n_matches": n, "n_support": n, "degenerate": True,
                    "feature_error_norm": float("nan"), "normalised_errors": None}
        else:
            v_raw, info = weighted_ls_solve(pts_goal, pts_cur, z, self.intrinsics,
                                            gain=self.gain, weights=weights,
                                            min_matches=self.min_matches)

        extra: dict[str, Any] = {}
        if info["degenerate"] and self.degenerate_policy == "hold":
            # the ART-VS release publishes nothing on this frame and the camera keeps the
            # last twist; the EMA is not touched (it lives in the solve the frame skipped)
            v = np.zeros(6) if self._last_cmd is None else self._last_cmd.copy()
            extra["degenerate_hold"] = self._last_cmd is not None
        else:
            v = v_raw
            if self.ema_alpha > 0.0:
                self._ema_v = (v if self._ema_v is None
                               else self.ema_alpha * v + (1.0 - self.ema_alpha) * self._ema_v)
                v = self._ema_v.copy()
            if self.max_velocity is not None:
                # after the EMA, per component (`store_applied_velocities`); the EMA keeps
                # the uncapped value, as the release's does
                capped = np.clip(v, -self.max_velocity, self.max_velocity)
                extra["velocity_capped"] = bool(np.any(capped != v))
                v = capped
            if self.degenerate_policy == "hold":
                extra["degenerate_hold"] = False
        if self.degenerate_policy == "hold":
            self._last_cmd = np.asarray(v, dtype=np.float64).copy()
        if self.pixel_quantisation == "truncate" or "quantise" in rows:
            extra["quantise"] = q_mode

        elapsed = time.monotonic() - t0
        debug = {
            "step": self._step_index,
            "step_ms": elapsed * 1000.0,
            "pts_goal": pts_goal,
            "pts_cur": pts_cur,
            "depth": z,
            "weights": weights,
            "velocity_raw": v_raw.copy(),
            "solve_unrounded": self.solve_unrounded,
            "ema_alpha": self.ema_alpha,
            **w_debug,
            **info,
            **extra,
        }
        return Twist(v), debug


# ------------------------------------------------------------------ stop rules

@dataclass(frozen=True)
class StopObservation:
    """Everything a stop rule may look at - and deliberately nothing else.

    The GT-blind rule in one type: a stop rule sees the controller-visible
    feature-error norm, the norm of the **commanded** twist, the per-landmark
    normalised-plane errors and the step index. The true pose is not here and
    never will be; scoring reads it afterwards, from the record.
    """

    step: int
    feature_error_norm: float = float("nan")
    velocity_norm: float = float("nan")
    normalised_errors: np.ndarray | None = None
    landmark_ids: np.ndarray | None = None
    t: float | None = None
    #: the mean PIXEL distance between the controller's current and goal points (the
    #: points it solved on, quantised or not) - what the ART-VS release's visual-only stop
    #: and its tiling switch read (``ros_controller.py:2493-2494``). NaN without rows.
    #: Added 2026-10-05 for :class:`ArtvsVisual90`; no other rule reads it.
    pixel_error_mean: float = float("nan")

    @classmethod
    def from_debug(cls, debug: dict, *, twist: Twist | np.ndarray | None = None,
                   dt: float = DEFAULT_DT, landmark_ids: np.ndarray | None = None
                   ) -> StopObservation:
        """Build one from :meth:`IBVSController.step`'s debug and the commanded twist.

        ``twist`` is the twist that was actually commanded, which is not always
        the one the solve returned (a lambda schedule scales it) - pass what the
        robot got.

        ``t = (step - 1) * dt`` is CNS's clock: it passes ``env.steps * env.dt``
        where ``env.steps`` counts the actions **already applied**, so its first
        call carries t = 0, while our step index is 1-based. Only differences of
        ``t`` are ever used, but not all ways of writing the same difference are
        equal in floating point - ``(k+1)dt - k*dt`` and ``k*dt - (k-1)dt`` differ
        in the last bit, and 40 of those decide a hold that lands exactly on
        0.8 s. Measured: an exponential settle at tau = 150 fires at step 227 with
        CNS's origin and 226 with a 1-based one.
        """
        step = int(debug.get("step", 0))
        v = np.zeros(6) if twist is None else np.asarray(twist, dtype=np.float64).reshape(-1)
        return cls(step=step,
                   feature_error_norm=float(debug.get("feature_error_norm", np.nan)),
                   velocity_norm=float(np.linalg.norm(v)),
                   normalised_errors=debug.get("normalised_errors"),
                   landmark_ids=landmark_ids,
                   t=(step - 1) * float(dt),
                   pixel_error_mean=_pixel_error_mean(debug))


def _pixel_error_mean(debug: dict) -> float:
    """``mean ||pts_cur - pts_goal||`` over the controller's rows, NaN when there are none
    or they do not pair up (an arm whose debug carries other shapes is never an error)."""
    try:
        pg = debug.get("pts_goal")
        pc = debug.get("pts_cur")
        if pg is None or pc is None:
            return float("nan")
        a = np.asarray(pg, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(pc, dtype=np.float64).reshape(-1, 2)
        if not len(a) or a.shape != b.shape:
            return float("nan")
        return float(np.mean(np.linalg.norm(b - a, axis=1)))
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class StopVerdict:
    """One rule's answer for one step.

    ``would_stop`` latches: once a rule's criterion has been met it stays met,
    because protocol v2 keeps the loop running to the cap and *records* the stop
    (the final error is read at the stop, primary, and at the cap, beside it).
    ``lambda_scale`` is 1.0 for every rule but :class:`DecPlateau`.
    """

    would_stop: bool
    fired_step: int | None = None
    lambda_scale: float = 1.0
    detail: dict = field(default_factory=dict)


@runtime_checkable
class StopRule(Protocol):
    """``reset()`` per episode, ``update(obs) -> StopVerdict`` per step.

    :attr:`params` is the rule's **effective** constants - its defaults merged with
    whatever the config overrode - so a record can state what actually ran instead
    of only what was overridden. The old runner wrote the whole merged
    ``STOP_MECHANICS`` block into every record (``baseline_check.py:1057``); writing
    the overrides alone would leave a bare ``cns_plateau`` cell with ``stop_cfg: {}``
    and no way to tell from the file which waiting time scored it (review R13, C6).
    """

    name: str

    @property
    def params(self) -> dict: ...

    def reset(self) -> None: ...

    def update(self, obs: StopObservation) -> StopVerdict: ...


class CapOnly:
    """No stop at all: the episode runs to its step budget.

    ``--stop-modes off`` with the quiet stop disarmed - the cap-scored read every
    pre-2026-09-02 CNS number in this project was taken under.
    """

    name = "cap_only"

    @property
    def params(self) -> dict:
        return {}

    def reset(self) -> None:
        return None

    def update(self, obs: StopObservation) -> StopVerdict:
        return StopVerdict(False)


class PlateauTrigger:
    """The in-force ``||e||``-plateau read, online (``baseline_check.PlateauTrigger``).

    Verbatim semantics: keep the raw feature-error series with its non-finite
    entries (they poison their windows); once ``2 * window`` steps exist, compare
    the mean of the last ``window`` with the mean of the ``window`` before it and
    require ``|cur - prev| < rel_tol * prev``; ``sustain`` consecutive such steps
    latch the trigger. Fielded constants 200 / 5 % / 10 (:data:`PLATEAU`),
    N0-calibrated: 40/40 on the long-horizon rider at median step 522, and it
    refuses every mid-descent cell.
    """

    def __init__(self, window: int | None = None, rel_tol: float | None = None,
                 sustain: int | None = None) -> None:
        self.window = int(PLATEAU["window"] if window is None else window)
        self.rel_tol = float(PLATEAU["rel_tol"] if rel_tol is None else rel_tol)
        self.sustain = int(PLATEAU["sustain"] if sustain is None else sustain)
        self._hist: list[float] = []
        self._run = 0
        self.fired_step: int | None = None

    def reset(self) -> None:
        self._hist.clear()
        self._run = 0
        self.fired_step = None

    def update(self, feat_now: float, step_i: int) -> bool:
        """Feed one step; True once the rule has fired (latching)."""
        self._hist.append(float(feat_now) if np.isfinite(feat_now) else np.nan)
        if len(self._hist) > 2 * self.window:
            self._hist.pop(0)
        if self.fired_step is not None:
            return True
        if len(self._hist) == 2 * self.window:
            cur = float(np.mean(self._hist[self.window:]))
            prev = float(np.mean(self._hist[:self.window]))
            ok = (np.isfinite(cur) and np.isfinite(prev) and prev > 0
                  and abs(cur - prev) < self.rel_tol * prev)
            self._run = self._run + 1 if ok else 0
            if self._run >= self.sustain:
                self.fired_step = int(step_i)
                return True
        return False


class DecPlateau:
    """The fielded ``dec`` mechanism: plateau trigger, then a lambda decay.

    Trigger + response, the shape the local calibration forced
    (``baseline_check.STOP_MECHANICS``): while the plateau has not fired every
    arm runs the untouched law (``lambda_scale = 1``); once it fires, lambda
    decays by ``rho = 0.9`` per step down to ``floor = 0.003`` times the base
    gain. Scaling the commanded twist is exactly equivalent to running the law at
    the lower gain, because ``v = -lambda L^+ e`` is linear in lambda - which is
    what makes the mechanism identical for every arm by construction.

    Under protocol v2 this is **recorded, not scored**: it stays fielded for
    deployment, and protocol v2's scored stop is :class:`CnsPlateau`.

    **What ``would_stop`` means here** (review R13, correction C1): the *published
    velocity-quiet criterion evaluated on the decayed twist*, not the plateau
    trigger. That is what the old bench recorded: ``baseline_check.py`` computed
    ``v_cmd = v * mech.lambda_scale(feat_now)`` (L1786-1789) and then ran the quiet
    check on ``||v_cmd||`` (L1792-1797) into ``stop_step`` / ``stop_translation_mm``,
    while the plateau trigger went to ``stop_trigger_step``. The two are far apart:
    on the stored N3.7b ``dec`` records ``stop_step - stop_trigger_step`` is 26 to
    126 steps, median 53 (n = 96). Scoring a re-flown campaign on the trigger would
    compare a read ~50 steps and one lambda-decay earlier than every stored number.
    The trigger stays available as ``detail["plateau_step"]`` and on
    ``self.trigger.fired_step``.

    The quiet criterion is the published one, verbatim and not redefinable by the
    bench (:data:`VELOCITY_QUIET`); ``quiet_eps`` / ``quiet_patience`` exist only so
    a *diagnostic* cell can widen it deliberately and have that recorded.
    """

    name = "dec_plateau"

    def __init__(self, *, rho: float = DEC["rho"], floor: float = DEC["floor"],
                 window: int | None = None, rel_tol: float | None = None,
                 sustain: int | None = None,
                 quiet_eps: float = VELOCITY_QUIET["eps"],
                 quiet_patience: int = int(VELOCITY_QUIET["patience"])) -> None:
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"dec rho must be in (0, 1], got {rho}")
        if not 0.0 < floor <= 1.0:
            raise ValueError(f"dec floor must be in (0, 1], got {floor}")
        self.rho = float(rho)
        self.floor = float(floor)
        self.trigger = PlateauTrigger(window, rel_tol, sustain)
        #: the published stop, evaluated on the twist the robot was actually given
        self.quiet = VelocityQuiet(eps=quiet_eps, patience=quiet_patience)
        self.scale = 1.0

    @property
    def params(self) -> dict:
        return {"rho": self.rho, "floor": self.floor, "window": self.trigger.window,
                "rel_tol": self.trigger.rel_tol, "sustain": self.trigger.sustain,
                "quiet_eps": self.quiet.eps, "quiet_patience": self.quiet.patience}

    def reset(self) -> None:
        self.trigger.reset()
        self.quiet.reset()
        self.scale = 1.0

    def update(self, obs: StopObservation) -> StopVerdict:
        fired = self.trigger.update(obs.feature_error_norm, obs.step)
        if fired:
            self.scale = max(self.scale * self.rho, self.floor)
        # the runner hands us the RAW twist, because the schedule that scales it is
        # this object; the published criterion reads the commanded one.
        commanded = replace(obs, velocity_norm=obs.velocity_norm * self.scale)
        quiet = self.quiet.update(commanded)
        return StopVerdict(quiet.would_stop, quiet.fired_step, self.scale,
                           {"plateau_step": self.trigger.fired_step,
                            "quiet_run": quiet.detail["quiet_run"]})


class VelocityQuiet:
    """The published stop: ``||v|| < 1e-4`` for 8 consecutive steps.

    ``baseline_check.py:1792-1793`` with :data:`VELOCITY_QUIET` verbatim; the
    bench never redefines it, it only asks whether a mechanism lets it fire.
    Reads the **commanded** twist, so a lambda schedule is visible to it - which
    is the point of ``dec``.
    """

    name = "velocity_quiet"

    def __init__(self, *, eps: float = VELOCITY_QUIET["eps"],
                 patience: int = int(VELOCITY_QUIET["patience"])) -> None:
        self.eps = float(eps)
        self.patience = int(patience)
        self._quiet = 0
        self.fired_step: int | None = None

    @property
    def params(self) -> dict:
        return {"eps": self.eps, "patience": self.patience}

    def reset(self) -> None:
        self._quiet = 0
        self.fired_step = None

    def update(self, obs: StopObservation) -> StopVerdict:
        self._quiet = self._quiet + 1 if obs.velocity_norm < self.eps else 0
        if self.fired_step is None and self._quiet >= self.patience:
            self.fired_step = int(obs.step)
        return StopVerdict(self.fired_step is not None, self.fired_step, 1.0,
                           {"quiet_run": self._quiet})


class CnsPlateau:
    """CNS v1's ``PixelStopPolicy``, verbatim - protocol v2's **scored** stop.

    Ported line by line from the vendored source
    ``tests/fixtures/reference/cns_stop_policy.py`` (MIT; upstream
    ``cns/benchmark/stop_policy.py``), not from a description of it: the first
    U4 implementation was written from prose and got the minimum tracking, the
    hold accounting, the crossing interpolation, the trim and the missing-landmark
    bookkeeping wrong, firing 90-185 steps early on a settling trace (review R12,
    2026-09-03). ``tests/fast/test_control.py`` now runs both on the same traces
    and requires the same firing step.

    **The error signal** (``PixelStopPolicy.calculate_error``, upstream 100-126) is
    a recency-weighted, trimmed mean of the per-landmark position error in
    **normalised image units** - camera independent, so 0.01 is 5.4 px at CNS's
    f = 540 and 14.1 px at ours. Per frame:

    - a landmark **absent** this frame decays, ``w <- 0.95 w`` (:115 with a false
      node mask); an **absent** landmark contributes nothing else - only the
      landmarks matched this frame enter the mean at all (:117-118);
    - a landmark's **first ever** sighting sets ``w = 1/(1-0.95) = 20`` (:113-114);
      every later sighting **recovers geometrically**, ``w <- 0.95 w + 1`` (:115),
      so a landmark that missed one frame comes back at 19.05, not at 20;
    - the **weighted** errors ``w * e`` above their 90th percentile are dropped,
      strictly (``error < percentile(error, 90)``, :121), and the survivors are
      averaged with ``w / (sum(w) + 1e-7)`` (:122-124). Two consequences worth
      knowing: the trim is on ``w * e`` and not on ``e``, and an all-equal error
      vector trims **everything** and reports ``0.0``. Both are CNS's, both are
      tested.

    **The state machine** (``ErrorHoldingStopPolicy.__call__``, upstream 25-66):

    1. no data -> ``reset()``, no stop (:26-29). Here that is a degenerate step,
       ``normalised_errors is None``;
    2. **arming gate** - ``err > 0.01`` -> ``reset()``, no stop (:34-36). The
       reset clears the landmark weights and witness counts too (:94-98);
    3. **new minimum** - ``err < 0.9 * min_err`` (or the first armed sample) ->
       move the minimum, restart the hold, **accrue nothing this step**, no stop
       (:38-43). A shallower improvement leaves ``min_err`` alone, which is what
       makes a slow monotone descent eventually stop being a descent;
    4. **hold** - ``thresh = 1.1 * min_err + 5e-4`` (:45), and the interval since
       the previous sample contributes the time it spent **below** ``thresh``,
       linearly interpolated across a crossing (:52-59). Stop once the total
       exceeds ``waiting_time = 0.8 s``, strictly (:66) - 41 intervals at CNS's
       50 Hz, **49 at our 60 Hz**.

    Two deliberate differences, and only two:

    - **latching.** CNS returns a bare bool and its benchmark loop breaks on the
      first True; protocol v2 keeps the loop running to the cap and *records* the
      stop, so ``fired_step`` latches and ``would_stop`` stays true.
    - **a non-finite error resets** instead of jamming. CNS never sees one; fed a
      NaN it would set ``min_err = NaN`` and never stop again (every later
      comparison being false). Treating it like ``data is None`` - the case CNS
      does reset on - is the same intent without the trap.

    Landmark identity: pass ``landmark_ids``, the **linear** goal-patch index
    (``goal_patch[:, 0] + side * goal_patch[:, 1]``), so a landmark that drops out
    decays as CNS's graph node would. ``landmark_ids=None`` means "these rows are
    the whole graph, every one of them seen this frame" - which is what CNS's
    fixed correspondence graph does. Under our per-frame re-draw the two are
    genuinely different protocols and the choice is the owner's; both are exact
    CNS behaviour for their own node set. The id dictionary is bounded by the goal
    patch grid (32 x 32 = 1024 entries at the fielded resolution), so no
    housekeeping is needed to keep it from growing.
    """

    name = "cns_plateau"

    def __init__(self, *, dt: float = DEFAULT_DT, **overrides: float) -> None:
        unknown = sorted(set(overrides) - set(CNS_STOP))
        if unknown:
            raise ValueError(f"unknown CnsPlateau constants: {unknown} (see CNS_STOP)")
        self.dt = float(dt)
        self.const = {**CNS_STOP, **{k: float(v) for k, v in overrides.items()}}
        self.reset()

    @property
    def params(self) -> dict:
        return dict(self.const, dt=self.dt)

    @property
    def hold_steps(self) -> float:
        """``waiting_time`` in steps at this controller's dt: 48 at 60 Hz, and the
        49th interval is the one that crosses it (the test is strict ``>``)."""
        return self.const["waiting_time"] / self.dt

    def reset(self) -> None:
        """New episode: forget the landmarks, the hold and the latch."""
        self._weight: dict[int, float] = {}      # CNS `weight`        (upstream :89, :106)
        self._witness: dict[int, int] = {}       # CNS `witness_count` (upstream :90, :107)
        self._hold_reset()
        self.fired_step: int | None = None

    def _hold_reset(self) -> None:
        """``ErrorHoldingStopPolicy.reset`` (upstream :16-20)."""
        self._min_err: float | None = None
        self._prev: tuple[float, float] | None = None
        self._accum = 0.0

    def _gate_reset(self) -> None:
        """``PixelStopPolicy.reset`` (upstream :94-98): the weights **and** the hold."""
        self._weight.clear()
        self._witness.clear()
        self._hold_reset()

    def error(self, normalised_errors: np.ndarray,
              landmark_ids: np.ndarray | None = None) -> float:
        """``PixelStopPolicy.calculate_error`` (upstream :100-126). Updates the weights."""
        errs = np.asarray(normalised_errors, dtype=np.float64).reshape(-1)
        if errs.size == 0:
            return float("nan")
        ids = (np.arange(errs.size) if landmark_ids is None
               else np.asarray(landmark_ids).reshape(-1).astype(np.int64))
        if ids.size != errs.size:
            raise ValueError(
                f"{ids.size} landmark ids for {errs.size} errors - pass the LINEAR "
                "goal-patch index, one per row (Matches.to_rows()['goal_patch'] is (K, 2))"
            )
        seen = {int(i) for i in ids}
        if len(seen) != ids.size:
            raise ValueError(
                "duplicate landmark ids in one frame - CNS's node mask cannot repeat a "
                "node, and a repeat would decay its weight twice"
            )
        for key in self._weight:                          # absent: w <- 0.95 w + 0 (:115)
            if key not in seen:
                self._weight[key] *= self.const["weight_decay"]
        w = np.empty(errs.size, dtype=np.float64)
        for j, i in enumerate(int(v) for v in ids):
            if self._witness.get(i, 0) == 0:              # first witness -> 20 (:109-114)
                self._weight[i] = self.const["max_weight"]
            else:                                         # re-sighting -> 0.95 w + 1 (:115)
                self._weight[i] = self.const["weight_decay"] * self._weight[i] + 1.0
            self._witness[i] = self._witness.get(i, 0) + 1
            w[j] = self._weight[i]
        weighted = w * errs                               # (:120)
        keep = weighted < np.percentile(weighted, self.const["trim_percentile"])   # (:121)
        norm = w[keep] / (w[keep].sum() + 1e-7)           # (:122-123) - CNS's epsilon
        return float(np.sum(norm * errs[keep]))           # (:124)

    def update(self, obs: StopObservation) -> StopVerdict:
        latched = StopVerdict(self.fired_step is not None, self.fired_step, 1.0, {})
        if obs.normalised_errors is None:                 # CNS `data is None` (:26-29)
            self._gate_reset()
            return replace(latched, detail={"cns_error": float("inf"), "armed": False})

        # CNS's clock origin: its first call carries t = 0 (`env.steps * env.dt`
        # with `env.steps` = actions applied so far), ours is 1-based
        t = (obs.step - 1) * self.dt if obs.t is None else float(obs.t)
        err = self.error(obs.normalised_errors, obs.landmark_ids)
        detail = {"cns_error": err}

        if not np.isfinite(err) or err > self.const["conduct_thresh"]:      # (:34-36)
            self._gate_reset()
            return replace(latched, detail={**detail, "armed": False})

        if self._min_err is None or err < self.const["min_reset"] * self._min_err:  # (:38-43)
            self._min_err = err
            self._prev = (t, err)
            self._accum = 0.0
            return replace(latched, detail={**detail, "armed": True, "min_err": err,
                                            "accum_s": 0.0})

        thresh = self.const["hold_mult"] * self._min_err + self.const["hold_abs"]   # (:45)
        (t0, a), (t1, b) = self._prev, (t, err)                                     # (:47-48)
        self._prev = (t, err)                                                       # (:49-50)
        if a < thresh <= b:                                                         # (:52-53)
            span = (thresh - a) / (b - a) * (t1 - t0)
        elif a >= thresh > b:                                                       # (:54-55)
            span = (thresh - b) / (a - b) * (t1 - t0)
        elif a <= thresh and b <= thresh:                                           # (:56-57)
            span = t1 - t0
        else:                                                                       # (:58-59)
            span = 0.0
        self._accum += span                                                         # (:61)

        if self.fired_step is None and self._accum > self.const["waiting_time"]:    # (:66)
            self.fired_step = int(obs.step)
        return StopVerdict(self.fired_step is not None, self.fired_step, 1.0,
                           {**detail, "armed": True, "min_err": self._min_err,
                            "accum_s": self._accum, "thresh": thresh})


class DivergenceGuard:
    """The N0 divergence abort: feature error above 3x its first finite value.

    ``DIVERGENCE_ABORT`` + ``baseline_check.py:1759-1768``. The anchor is the
    first finite **positive** controller-visible feature error of the episode -
    the published reference aborts on the true position error, and that read
    stays a scoring-time classification (the GT-blind runner rule). Unlike a stop
    rule this ends the episode: the loop breaks before the pose integration, so
    the camera stays at the capture pose of the aborting step.
    """

    name = "divergence"

    def __init__(self, *, mult: float = DIVERGENCE_MULT) -> None:
        self.mult = float(mult)
        self.initial: float | None = None
        self.fired_step: int | None = None

    @property
    def params(self) -> dict:
        return {"mult": self.mult}

    def reset(self) -> None:
        self.initial = None
        self.fired_step = None

    def update(self, obs: StopObservation) -> StopVerdict:
        feat = float(obs.feature_error_norm)
        if self.initial is None and np.isfinite(feat) and feat > 0:
            self.initial = feat
        if (self.fired_step is None and self.initial is not None
                and np.isfinite(feat) and feat > self.mult * self.initial):
            self.fired_step = int(obs.step)
        return StopVerdict(self.fired_step is not None, self.fired_step, 1.0,
                           {"initial": self.initial, "reason": "diverged_feature"})


#: The ART-VS release's visual-only stop (``core/controller.py:326-385``;
#: ``real_robot_error_reduction_target`` and ``real_robot_max_iterations`` of the robot
#: config, ``config_real_robot_dinov3.yaml:138-139``). ``release_cap_iterations`` is
#: informational: the release also declares "converged" at that many iterations; the bench
#: flies to its own cap and records only the 90 % fire.
ARTVS_VISUAL: dict[str, float] = {"target": 0.90, "release_cap_iterations": 500}


class ArtvsVisual90:
    """The ART-VS release's visual-only stop: 90 % less mean pixel error than at first.

    ``check_convergence_visual_only`` (``core/controller.py:326-385``, vendored at
    ``tests/fixtures/reference/artvs_release/controller.py``): ``e0`` is the first
    ``feature_error > 0`` it sees, and it fires - latched here, as every bench stop is -
    when ``1 - e / e0 >= 0.90``. It reads :attr:`StopObservation.pixel_error_mean`, the
    PIXEL mean of the controller's (truncated) points: the release's
    ``current_feature_error`` (``ros_controller.py:2493-2494``), independent of ``fx ==
    fy``. A degenerate step (``normalised_errors is None``) is skipped and never resets
    anything: the release never calls the check on a frame it skipped. **Recorded, never
    scored** on the faithful ART-VS arm (``episode.record_stops``, owner ruling
    2026-10-05); as a ``stop.rule`` it is accepted like any other rule.
    """

    name = "artvs_visual90"

    def __init__(self, *, target: float = ARTVS_VISUAL["target"]) -> None:
        if not 0.0 < float(target) < 1.0:
            raise ValueError(f"artvs_visual90 target must be in (0, 1), got {target}")
        self.target = float(target)
        self.reset()

    @property
    def params(self) -> dict:
        return {"target": self.target,
                "release_cap_iterations": int(ARTVS_VISUAL["release_cap_iterations"])}

    def reset(self) -> None:
        self.e0: float | None = None
        self.fired_step: int | None = None

    def update(self, obs: StopObservation) -> StopVerdict:
        e = float(obs.pixel_error_mean)
        if obs.normalised_errors is None or not np.isfinite(e):
            return StopVerdict(self.fired_step is not None, self.fired_step, 1.0,
                               {"e0": self.e0, "e": None, "reduction": None, "skipped": True})
        if self.e0 is None and e > 0:                                     # :345-350
            self.e0 = e
        reduction = None
        if self.e0 is not None and self.e0 > 0:                           # :352-361
            reduction = 1.0 - (e / self.e0)
            if self.fired_step is None and reduction >= self.target:
                self.fired_step = int(obs.step)
        return StopVerdict(self.fired_step is not None, self.fired_step, 1.0,
                           {"e0": self.e0, "e": e, "reduction": reduction})


def _dinobot_err(**kwargs: Any) -> StopRule:
    """``dinobot_err`` - DINOBot's own stop, which lives with its arm
    (:class:`vsbench.vs.baselines.dinobot.DinoBotErr`); imported on use, because that module
    imports this one."""
    from vsbench.vs.baselines.dinobot import DinoBotErr  # noqa: PLC0415 - import cycle

    return DinoBotErr(**kwargs)


#: ``stop.rule`` (the config enum, ``vsbench.vs.bench.config.STOP_RULES``) -> the class (or,
#: for ``dinobot_err``, the function that imports and builds it).
STOP_RULE_BUILDERS = {
    "cap_only": CapOnly,
    "dec_plateau": DecPlateau,
    "cns_plateau": CnsPlateau,
    "velocity_quiet": VelocityQuiet,
    "dinobot_err": _dinobot_err,
    "artvs_visual90": ArtvsVisual90,
}


def build_stop_rule(config: dict, **overrides: Any) -> StopRule:
    """``config["stop"]["rule"]`` -> the rule object, with ``config["stop"]``'s
    other keys passed to its constructor.

    ``stop.rule`` is one of the five fields ``vsbench.vs.bench.config`` refuses to
    default, because a cap-scored number and a stop-scored number are different
    numbers. This resolves it; :class:`DivergenceGuard` is separate and always
    armed (it ends an episode rather than scoring one).

    Every other key under ``stop`` is forwarded to the rule's constructor, so the
    block holds **rule constants and nothing else** - an unknown key raises rather
    than being ignored. The step budget, the long-horizon cell and anything else
    about the episode belong elsewhere in the config (U7/U9).
    """
    from vsbench.vs.bench.config import STOP_RULES as _ENUM

    rule = ((config or {}).get("stop") or {}).get("rule")
    if rule not in STOP_RULE_BUILDERS:
        raise ValueError(f"stop.rule is {rule!r}, not one of: {', '.join(_ENUM)}")
    kwargs = {k: v for k, v in ((config or {}).get("stop") or {}).items() if k != "rule"}
    kwargs.update(overrides)
    return STOP_RULE_BUILDERS[rule](**kwargs)
