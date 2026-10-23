# vs-sim copy: vssim/baselines/cns.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""CNS (v1) as an arm, run **as published** (unit U7b).

Purpose
-------
The strongest published competitor and the landing bar: "land at least as close as
CNS v1 (repeated, stop-scored) on the same starts". Wrapped in process so it flies
the same starts, the same scenes and the same scoring as everything else.

**This is CNS v1**, not CNSv2. CNSv2 (arXiv:2503.00132) has no public code and no
weights; CNS v1 (github.com/hhcaz/CNS, arXiv:2309.09047) ships both. Report it as
"CNS (v1)". The checkout is cloned and read, never vendored.

Why it is not a matcher
-----------------------
Every other method here is a :class:`vsbench.vs.baselines.match.CorrespondenceMatcher` feeding
the one shared IBVS law. CNS cannot be: it is a graph-neural-network **policy**
that consumes keypoint correspondences and emits a camera twist directly,
replacing the control law rather than plugging into it. So a CNS-vs-ours contrast
is *not* a controlled comparison of correspondence quality - it varies the
controller too, and that sentence belongs under every table its numbers appear in.
:class:`CNSArm` therefore fills the :class:`vsbench.vs.bench.arms.Arm` shape with its
own front-end (:class:`CNSPipeline`, the "matcher" slot) and its own policy
(:class:`CNSPolicy`, the "controller" slot), so the runner, the record and the
scoring do not need a second code path.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
``src/control/cns_vs.py`` in full: :func:`resolve_cns_root` (L53-62, the
``$CNS_ROOT`` search and its instructions), :class:`published_torch_load` (L65-92,
``weights_only=False`` **only** while CNS loads its own pinned artifact),
:func:`shim_broken_visualize` (L95-125, the import-time matplotlib break in the
pinned commit, stubbed to refuse loudly), the lazy pipeline build on the first
step with the **real** intrinsics (L189-232, "refusing to invent a placeholder
camera model"), the BGR conversion (L282-285), :meth:`CNSPipeline.dist_scale`
(L287-299, the median valid goal depth), :data:`VALID_DETECTORS`,
:data:`DEFAULT_CHECKPOINT` (the state dict, not ``cns.pth``: the full-model pickle
holds torch-1.x ``Linear`` objects that no longer unpickle), the graph-point
de-normalisation and its :data:`FILM_POINTS_MAX` subsample (L305-334) and the
best-effort support count (L336-364).

New here - protocol v2 section 4's "run as published"
-----------------------------------------------------
The old arm had **neither** of CNS's two published mechanisms, so every CNS number
quoted before 2026-09-02 is a cap-scored read at 40 % of its step budget.

1. **The x2 step gain** (``benchmark.py:161`` into a ``dt = 1/50`` integrator,
   ``sim/environment.py:110-114``) -> :attr:`CNSArm.step_scale`, which states how the
   published per-iteration displacement is reproduced at 60 Hz and what that does to
   the decay budget.
2. **Its own stop**: ``benchmark.py:220`` runs ``PixelStopPolicy(waiting_time=0.8,
   conduct_thresh=0.01)`` and :class:`vsbench.vs.baselines.control.CnsPlateau` is U4's line-by-line
   port of it. :meth:`CNSPipeline.graph_error` feeds it **CNS's own signal** - the
   per-keypoint error of its correspondence graph keyed by node index, which is what
   its ``node_mask`` decays, not our re-drawn landmarks. That is the faithful reading
   review R13 asked for, and this is the arm where CNS's node set exists.

Refusals kept from the old wrapper (``baseline_check.py:1942-1948``, ``:2392``)
------------------------------------------------------------------------------
CNS "composes with nothing": an ``oracle`` block (no matcher for the GT wrapper to
instrument), a ``head`` block (the head lives in the DINOv3 chain), the IBVS
``controller`` knobs and the ``dec_plateau`` stop (a lambda schedule on **our**
commanded twist) are each refused loudly rather than ignored.

Fences
------
- **CNS wants BGR.** ``set_target`` / ``get_control_rate`` are documented as bgr and
  the scene hands us RGB; passing RGB degrades matching silently instead of raising.
- **``dist_scale`` is the camera-to-scene-centre distance**, taken as the median
  valid goal depth (robust to background and holes, automatic per scene). It scales
  the predicted twist, so a wrong one is a wrong speed, silently.
- **No feature error.** CNS exposes no scalar ``||e||``, so it is reported as ``NaN``
  rather than invented - which also means the N0 divergence abort (finite feature
  error above 3x) never fires on a CNS episode, exactly as before.
- **The GPU is never touched by the fast tier.** The pipeline is built lazily on the
  first step, and every fast test injects a stub.
- **Two loop differences from CNS's benchmark, both recorded rather than hidden.**
  (a) When the front-end returns no correspondences CNS's benchmark **ends** the
  episode (``benchmark.py:146-149``); here the arm commands a zero twist, the stop
  resets and the loop runs to the cap, because protocol v2 section 5 scores the stop
  from an episode that always reaches the cap. The landing is the same - the camera is
  parked and the detector is deterministic - but ``stop_step`` is not. (b) CNS's cap
  test is ``steps > 600``, so its benchmark applies 601 actions; we apply 600.

Golden tests that guard it
--------------------------
``tests/fast/test_cns.py`` - a stub pipeline returning a scripted twist verifies the
x2 and the dt handling, the RGB->BGR conversion, the ``dist_scale``, the stop wiring
(CNS's own node ids and errors reach :class:`~vsbench.vs.baselines.control.CnsPlateau`) and every
refusal. ``tests/slow/test_cns_arm.py`` - the real pipeline where ``$CNS_ROOT`` exists.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vsbench.vs.baselines.control import DEFAULT_DT, Twist, build_stop_rule

#: Environment variable pointing at a CNS checkout (must hold ``cns/`` and ``checkpoints/``).
CNS_ROOT_ENV = "CNS_ROOT"

#: The STATE-DICT artifact, not ``cns.pth``: the full-model pickle stores torch-1.x-era
#: ``Linear`` objects that no longer unpickle under torch >= 2.6. CNS's own
#: ``GraphVSController`` rebuilds ``GraphVS(2, 2, 128)`` and ``load_state_dict``s exactly
#: this file - the version-safe published path.
DEFAULT_CHECKPOINT = "checkpoints/cns_state_dict.pth"

#: Detectors CNS accepts as its "observer" front-end (SuperGlue needs its own weights).
VALID_DETECTORS = ("SIFT", "ORB", "AKAZE", "SuperGlue")

#: What CNS's own benchmark does, cited to the line. ``step_gain`` is
#: ``benchmark.py:161`` ``env.action(vel * 2)``; ``dt`` is ``sim/environment.py:64``
#: (``1/50``, and ``action`` integrates ``vel * dt``); ``max_steps`` is
#: ``benchmark/environment.py:43`` (600, raised from the sim default 200).
PUBLISHED = {"step_gain": 2.0, "dt": 1.0 / 50.0, "max_steps": 600}

#: Match-film subsample cap: 1,678 arrows are unreadable; ~64 evenly spaced over the
#: graph keep the picture honest (``cns_vs.FILM_POINTS_MAX``, N2.b films addendum).
FILM_POINTS_MAX = 64

#: Config blocks CNS composes with: none of them (see the module docstring).
_REFUSED_BLOCKS = {
    "oracle": "it has no matcher for the GT wrapper to instrument",
    "head": "the refinement head lives in the DINOv3 matcher chain",
}
#: ``controller`` keys that are IBVS machinery. ``dt`` is the loop's period and is kept.
_REFUSED_CONTROLLER_KEYS = ("gain", "weighting", "solve_unrounded", "ema_alpha",
                            "min_matches")

#: Everything the ``cns`` config block may hold. An unknown key is refused rather
#: than forwarded into a ``TypeError`` (or, worse, silently dropped as ``cns_root``
#: once was - review R14, C3).
CNS_CONFIG_KEYS = ("detector", "device", "cns_root", "ckpt", "ransac", "dist_scale",
                   "step_gain", "published_dt", "rate_compensate")


def resolve_cns_root(explicit: str | None = None) -> Path:
    """Locate the CNS checkout, or explain exactly how to provide one (verbatim)."""
    for cand in (explicit, os.environ.get(CNS_ROOT_ENV),
                 "/workspace/CNS", str(Path.home() / "CNS")):
        if cand and (Path(cand) / "cns").is_dir():
            return Path(cand)
    raise FileNotFoundError(
        "CNS checkout not found. Clone https://github.com/hhcaz/CNS and point "
        f"${CNS_ROOT_ENV} at it (the directory holding `cns/` and `checkpoints/`). "
        "It ships its own pretrained weights.")


class published_torch_load:  # noqa: N801 - a context manager, named for what it restores
    """Restore pre-torch-2.6 ``torch.load`` semantics around CNS's own loads.

    torch 2.6 flipped ``weights_only`` to True; CNS's published checkpoint pickles
    the whole ``GraphVS`` model, so their ``torch.load(ckpt)`` raises
    ``UnpicklingError`` - and inside a SimulationApp run the exception can vanish
    without a traceback (the N2.b shakedown; the smoke gate caught the empty record
    set). This passes ``weights_only=False`` **only** while CNS loads its own
    trusted, version-pinned artifact, and restores ``torch.load`` on exit.
    """

    def __enter__(self):
        import torch  # noqa: PLC0415 - torch is not imported for the fast tier

        self._torch = torch
        self._orig = torch.load

        def _load(*a, **k):
            k.setdefault("weights_only", False)
            return self._orig(*a, **k)

        torch.load = _load
        return self

    def __exit__(self, *exc):
        self._torch.load = self._orig
        return False


def shim_broken_visualize() -> None:
    """Stub ``cns.utils.visualize`` if it cannot import (the N2.b shakedown find).

    The pinned commit's ``visualize.py`` builds a colour table at **import** time
    from matplotlib internals that changed shape, and ``cns.benchmark.pipeline``
    imports it unconditionally. It is a plotting helper only - this suite always
    runs ``vis=VisOpt.NO`` - so a failed import is replaced by a stub whose
    functions refuse loudly if anything ever calls them. The CNS checkout stays
    pristine (published-reference discipline); the deviation lives here, recorded.
    """
    import importlib  # noqa: PLC0415
    import types  # noqa: PLC0415

    try:
        importlib.import_module("cns.utils.visualize")
        return                                    # imports fine - no shim
    except Exception as e:  # noqa: BLE001
        reason = f"{type(e).__name__}: {e}"

    def _refuse(*_a, **_k):
        raise RuntimeError(
            "cns.utils.visualize is stubbed (import-time failure under this "
            f"matplotlib: {reason}); run CNS with vis=VisOpt.NO")

    stub = types.ModuleType("cns.utils.visualize")
    stub.__shim_reason__ = reason
    stub.show_graph = _refuse
    stub.show_keypoints = _refuse
    sys.modules["cns.utils.visualize"] = stub


class CNSPipeline:
    """CNS's published front-end + policy behind the arm's matcher slot.

    Holds the goal view and the ``dist_scale``, builds
    ``cns.benchmark.pipeline.CorrespondenceBasedPipeline`` lazily on the first
    :meth:`control` (when the real intrinsics have arrived), and converts between
    our RGB frames and CNS's BGR ones. ``pipeline`` may be injected - that is how
    the fast tier exercises everything but the network.
    """

    def __init__(self, *, detector: str = "SIFT", device: str = "cuda:0",
                 cns_root: str | None = None, ckpt: str | None = None,
                 ransac: bool = True, dist_scale: float | None = None,
                 pipeline: Any = None, intrinsics: Any = None) -> None:
        if detector not in VALID_DETECTORS:
            raise ValueError(f"detector must be one of {VALID_DETECTORS}, got {detector!r}")
        self.detector = detector
        self.goal_mask = None
        self.device = device if ":" in device else f"{device}:0"
        self.ransac = bool(ransac)
        self.dist_scale_override = None if dist_scale is None else float(dist_scale)
        self.intrinsics = intrinsics
        self.name = f"cns_v1_{detector.lower()}"
        #: the checkout, when a config named one; `resolve_cns_root` searches otherwise.
        #: It used to be accepted and dropped - a silent config knob (review R14, C3).
        self._root: Path | None = Path(cns_root) if cns_root else None
        self._ckpt = ckpt
        self._pipeline = pipeline
        self._injected = pipeline is not None
        self.goal_rgb: np.ndarray | None = None
        self.dist_scale = 1.0
        #: what the record's `checkpoint` column says: the configured path until the
        #: pipeline resolves it against the checkout, the resolved one afterwards.
        self.checkpoint = str(ckpt or DEFAULT_CHECKPOINT)
        #: CNS steers on its whole correspondence graph, not on a drawn landmark set
        self.num_landmarks: int | None = None
        self.frozen_landmarks = False        # the record's matcher-variant columns
        self.tie_order = None

    # ------------------------------------------------------------ the checkout
    def _load(self) -> None:
        """Import CNS from its checkout and build the pipeline (once per arm).

        **The intrinsics are checked first, before anything of CNS is imported**
        (station verify, 2026-09-05). It is OUR refusal about OUR configuration and it
        needs nothing from the checkout, but standing after the import it depended on
        CNS's whole dependency tree: with `PYTHONPATH` cleared the station's `genvs`
        interpreter has no `torch_geometric`, so `cns.benchmark.pipeline` raised
        `ModuleNotFoundError` and the arm never got to say that its camera model was
        missing. A refusal that only fires where a third-party package happens to be
        importable is not a refusal.
        """
        intr = self.intrinsics
        if intr is None or not getattr(intr, "fx", None):
            raise RuntimeError(
                "CNSPipeline needs real intrinsics (fx/fy/cx/cy) before its first "
                "step - refusing to invent a placeholder camera model")
        root = resolve_cns_root(None if self._root is None else str(self._root))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        shim_broken_visualize()
        ckpt = Path(self._ckpt) if self._ckpt else Path(DEFAULT_CHECKPOINT)
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"CNS checkpoint not found: {ckpt}")
        self.checkpoint = str(ckpt)          # the RESOLVED path, for the record
        from cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt  # noqa: PLC0415
        from cns.utils.perception import CameraIntrinsic  # noqa: PLC0415

        with published_torch_load():
            self._pipeline = CorrespondenceBasedPipeline(
                ckpt_path=str(ckpt), detector=self.detector, device=self.device,
                intrinsic=CameraIntrinsic(int(intr.width), int(intr.height),
                                          float(intr.fx), float(intr.fy),
                                          float(intr.cx), float(intr.cy)),
                ransac=self.ransac, vis=VisOpt.NO)
        self._set_target()

    def _set_target(self) -> None:
        if self.goal_mask is None:
            self._pipeline.set_target(to_bgr(self.goal_rgb), dist_scale=self.dist_scale)
            return
        # CNS's own front end takes a detection mask (`Classic.update_target_frame(image,
        # mask)` -> `detectAndCompute(image, mask)`); `set_target` does not forward one, so
        # this is `set_target` with the mask, field for field (benchmark/pipeline.py:101-109)
        fe = getattr(self._pipeline, "frontend", None)
        if fe is None:
            raise ValueError("a goal mask needs a CNS pipeline with a front end "
                             "(CorrespondenceBasedPipeline.frontend) - this one has none")
        fe.update_target_frame(to_bgr(self.goal_rgb), self.goal_mask.astype(np.uint8) * 255)
        self._pipeline.new_scene = True
        self._pipeline.dist_scale = self.dist_scale
        self._pipeline.target_kwargs = {}

    # -------------------------------------------------------------- the arm API
    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        """Stash the goal view, its ``dist_scale`` and the goal mask (bool ``[H, W]``, the
        masked cell only); the pipeline builds lazily."""
        self.goal_mask = None if goal_mask is None else np.asarray(goal_mask).astype(bool)
        self.goal_rgb = np.asarray(goal_rgb).copy()
        self.dist_scale = self.scene_distance(goal_depth)
        if self._injected:
            self._set_target()
        else:
            self._pipeline = None                 # rebuilt on the next control()

    def set_episode(self, scene: str, start_id: Any) -> None:
        """Nothing to seed: CNS has no random landmark draw (the interface's hook)."""

    def scene_distance(self, depth) -> float:
        """``dist_scale``: the median **valid** goal depth, metres (``cns_vs.py:287-299``).

        Robust to background and to holes, and automatic per scene. Overridable
        through ``cns.dist_scale`` when a cell wants CNS's own convention (the norm
        of the goal translation) instead.
        """
        if self.dist_scale_override is not None:
            return float(self.dist_scale_override)
        if depth is None:
            return 1.0
        d = np.asarray(depth, float).reshape(-1)
        d = d[np.isfinite(d) & (d > 0.0) & (d < 10.0)]
        return float(np.median(d)) if d.size else 1.0

    def control(self, rgb: np.ndarray):
        """One published control call: ``(twist6, graph data, timing)``, BGR in."""
        if self.goal_rgb is None:
            raise RuntimeError("reset(goal_rgb) must be called before control()")
        if self._pipeline is None:
            self._load()
        vel, data, timing = self._pipeline.get_control_rate(to_bgr(rgb))
        v = np.zeros(6) if vel is None else np.asarray(vel, float).reshape(-1)
        if v.shape != (6,) or not np.all(np.isfinite(v)):
            v = np.zeros(6)
        return v, data, timing

    def close(self) -> None:
        self._pipeline = None if not self._injected else self._pipeline

    # ------------------------------------------------------- CNS's own signals
    @staticmethod
    def graph_error(data) -> tuple[np.ndarray | None, np.ndarray | None]:
        """CNS's own stop signal: per-keypoint error and node id (``stop_policy.py:100-118``).

        ``PixelStopPolicy.calculate_error`` takes ``data.pos_cur - data.pos_tar``
        over the nodes ``data.node_mask`` selects, in normalised image units, and
        weights them by a **persistent** per-node recency weight keyed by the node's
        index in the full graph. Returning ``(errors, ids)`` in that shape is what
        makes :class:`vsbench.vs.baselines.control.CnsPlateau` see the node set CNS decays
        rather than our re-drawn landmarks (review R13's faithful reading).
        """
        pos_cur, pos_tar = getattr(data, "pos_cur", None), getattr(data, "pos_tar", None)
        mask = getattr(data, "node_mask", None)
        if pos_cur is None or pos_tar is None or mask is None:
            return None, None
        cur, tar, m = (_to_numpy(pos_cur), _to_numpy(pos_tar), _to_numpy(mask))
        cur, tar, m = cur.reshape(-1, 2), tar.reshape(-1, 2), m.reshape(-1).astype(bool)
        ids = np.flatnonzero(m)
        if ids.size == 0:
            return None, None
        return np.linalg.norm(cur[ids] - tar[ids], axis=-1), ids

    def graph_points_px(self, data):
        """CNS correspondences -> full-image pixels for the films (``cns_vs.py:305-334``).

        The pinned commit's midend stores ``pos_tar`` / ``pos_cur`` in **normalised**
        camera-plane coordinates (measured on a real pair: x within +-cx/fx);
        de-normalised here with the intrinsics the pipeline was built with, and
        evenly subsampled to at most :data:`FILM_POINTS_MAX` rows.
        """
        pt, pc = getattr(data, "pos_tar", None), getattr(data, "pos_cur", None)
        if pt is None or pc is None or self.intrinsics is None:
            return None, None
        pt, pc = _to_numpy(pt), _to_numpy(pc)
        if pt.ndim != 2 or pt.shape[1] != 2 or len(pt) == 0:
            return None, None
        idx = np.linspace(0, len(pt) - 1, min(len(pt), FILM_POINTS_MAX)).round().astype(int)
        i = self.intrinsics

        def to_px(p):
            return np.column_stack([p[:, 0] * i.fx + i.cx, p[:, 1] * i.fy + i.cy])

        return to_px(pt[idx]), to_px(pc[idx])

    @staticmethod
    def n_matches(data) -> int:
        """Best-effort support count (``cns_vs.py:336-364``); CNS's payload shape varies.

        The pinned commit returns a torch-geometric ``GraphData`` whose ``num_nodes``
        is the correspondence-graph size (measured 1678 on a real hollywood pair).

        **Read it as what it is** (review R14, C6/§8): CNS rebuilds its graph only when
        the *target* changes, so ``num_nodes`` is the target keypoint count and is
        **constant for the whole episode**. A CNS record's ``mean_n_matches`` and
        ``min_n_matches`` are therefore not support statistics, and the number is not
        our landmark count - state that wherever the two are printed side by side. The
        per-frame matched count is ``len(graph_error(data)[0])`` and is recorded as
        ``n_matched`` in the step debug.
        """
        for attr in ("num_matches", "n_matches", "num_nodes"):
            v = getattr(data, attr, None)
            if v is not None:
                try:
                    return int(v)
                except Exception:  # noqa: BLE001, S112
                    pass
        if isinstance(data, dict):
            for key in ("num_matches", "n_matches"):
                if key in data:
                    return int(data[key])
        pos = getattr(data, "pos_cur", None)
        return 0 if pos is None else int(len(pos))


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    """RGB -> BGR. CNS documents **both** entry points as bgr (``cns_vs.py:282-285``)."""
    return np.ascontiguousarray(np.asarray(rgb)[..., :3][..., ::-1])


def _to_numpy(x) -> np.ndarray:
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


@dataclass
class CNSPolicy:
    """What sits where our IBVS controller sits - and what that costs the record.

    CNS replaces the control law, so the record's controller columns cannot be
    filled with our settings without lying. They are filled with what is true of
    CNS: no rounding of feature coordinates, no twist smoothing, no support floor,
    and a *learned* weighting rather than one of ours.
    """

    dt: float = DEFAULT_DT
    gain: float = PUBLISHED["step_gain"]
    solve_unrounded: bool = True          # CNS works on sub-pixel keypoints throughout
    ema_alpha: float = 0.0                # no twist smoothing
    min_matches: int = 0                  # no support floor of ours
    weighting: Any = field(default_factory=lambda: _LearnedWeighting())

    def reset(self) -> None:
        """Per-episode hook of the controller slot; CNS's policy is stateful inside
        its own GRU and is reset by ``set_target``."""


@dataclass(frozen=True)
class _LearnedWeighting:
    """The record's ``weighting`` column for a CNS cell: a name, not our recipe."""

    learned: str = "cns_v1_graphvs"


class CNSArm:
    """CNS v1 behind the :class:`vsbench.vs.bench.arms.Arm` shape.

    Structurally identical to an ``Arm`` (``matcher`` / ``controller`` / ``stop`` /
    ``config``), so :func:`vsbench.vs.bench.runner.run_episode`, the record and the
    scoring take it unchanged - but :meth:`step` is CNS's, not ours.
    """

    kind = "method"

    def __init__(self, pipeline: CNSPipeline, policy: CNSPolicy, stop, config: dict, *,
                 name: str | None = None, step_gain: float = PUBLISHED["step_gain"],
                 published_dt: float = PUBLISHED["dt"], rate_compensate: bool = True) -> None:
        self.matcher = pipeline
        self.controller = policy
        self.stop = stop
        self.config = config
        self.step_gain = float(step_gain)
        self.published_dt = float(published_dt)
        self.rate_compensate = bool(rate_compensate)
        self._step = 0
        self.name = name or (f"{pipeline.name}/x{self.step_scale:g}/"
                             f"{getattr(stop, 'name', 'cap_only')}")

    # ------------------------------------------------------------ the x2 and dt
    @property
    def step_scale(self) -> float:
        """The multiplier our loop applies to CNS's twist. **2.4 at 60 Hz.**

        CNS's benchmark integrates ``vel * 2`` for ``1/50`` s per iteration, i.e. a
        published per-iteration displacement of ``vel * 0.04``; our loop integrates
        for ``dt = 1/60``. ``rate_compensate=True`` (**the default**) commands
        ``vel * 2 * (dt_published / dt)`` = ``vel * 2.4``, so one of our iterations
        moves the camera exactly as far as one of CNS's - iteration *i* of our episode
        is iteration *i* of its benchmark - and the decay budget is
        ``2.4 * 600 / 60 = 24``, **CNS's own** (``2 * 600 / 50 = 24``).
        ``rate_compensate=False`` commands the literal ``vel * 2``: ``lambda * T = 20``,
        our arms' budget (protocol v2 section 5) but only **83 %** of CNS's, which
        would quote CNS short of its own budget again - the error section 4 exists to
        end. Both settings and the effective scale are recorded. The choice is
        claim-bearing: see ``CLAUDE.md``, "Carried out of U7b".
        """
        if not self.rate_compensate:
            return self.step_gain
        return self.step_gain * (self.published_dt / self.dt)

    def decay_budget(self, max_iter: int) -> float:
        """``step_scale * max_iter * dt`` - the ``lambda * T`` protocol v2 section 5 states."""
        return self.step_scale * int(max_iter) * self.dt

    # ------------------------------------------------------------- the arm API
    @property
    def matcher_name(self) -> str:
        return self.matcher.name

    @property
    def checkpoint(self) -> str:
        """The record's ``checkpoint``: CNS's **resolved** weights path, not ``None``.

        The configured value until :meth:`CNSPipeline._load` resolves it against the
        checkout, the absolute path afterwards. A CNS row whose checkpoint column was
        empty could not be told from an unconfigured one (review R14, C6).
        """
        return self.matcher.checkpoint

    @property
    def dt(self) -> float:
        return self.controller.dt

    @property
    def gain(self) -> float:
        """The record's ``gain``: the multiplier actually applied to CNS's twist."""
        return self.step_scale

    def set_episode(self, scene: str, start_id: Any) -> None:
        self.matcher.set_episode(scene, start_id)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        self.matcher.reset(goal_rgb, goal_depth, goal_mask)
        self.controller.reset()
        self.stop.reset()
        self._step = 0

    def step(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray | None = None):
        """One published CNS iteration -> ``(twist, debug, None)``.

        ``depth`` is ignored: CNS estimates its own scene scale from ``dist_scale``
        at ``set_target`` and never reads a live depth map. ``matches`` is ``None``
        because CNS has no landmark set of ours - the telemetry's point columns stay
        empty for a CNS cell, exactly as in every stored CNS record.
        """
        if mask is not None:
            raise ValueError("CNS runs its own front-end and does not take a frame mask")
        t0 = time.monotonic()
        self._step += 1
        vel, data, timing = self.matcher.control(rgb)
        elapsed = (time.monotonic() - t0) * 1000.0
        errs, ids = CNSPipeline.graph_error(data)
        pts_tar, pts_cur = self.matcher.graph_points_px(data)
        debug: dict[str, Any] = {
            "step": self._step,
            "feature_error_norm": float("nan"),   # CNS exposes no scalar ||e||
            "n_matches": CNSPipeline.n_matches(data),      # CNS's constant node count
            "n_matched": (0 if ids is None else int(len(ids))),   # matched THIS frame
            "velocity_raw": np.asarray(vel, float).copy(),
            "step_ms": elapsed,
            # CNS's timing dict is frontend/midend/backend/total; `frontend_time` is
            # its detect-and-match stage, the twin of our `match_ms`. The first cut read
            # a key CNS never emits, so it was always 0.0 (review R14, C4).
            "match_ms": (float((timing or {}).get("frontend_time", 0.0)) * 1000.0
                         if isinstance(timing, dict) else 0.0),
            "depth_values": None,
            "normalised_errors": errs,            # CNS's own per-keypoint error
            "stop_landmark_ids": ids,             # ... keyed by its own graph node
            "cns_timing": timing,
            "policy": "cns_v1",
        }
        if pts_tar is not None:
            debug["pts_desired"], debug["pts_current"] = pts_tar, pts_cur
        return Twist(np.asarray(vel, float) * self.step_scale), debug, None

    def close(self) -> None:
        self.matcher.close()

    def record_fields(self) -> dict:
        """What a CNS record adds to the common schema - the published knobs, stated."""
        return {"cns_detector": self.matcher.detector, "cns_ransac": self.matcher.ransac,
                "cns_dist_scale": self.matcher.dist_scale,
                "cns_step_gain": self.step_gain,
                "cns_rate_compensate": self.rate_compensate,
                "cns_step_scale": self.step_scale,
                "cns_published_dt": self.published_dt,
                "cns_stop_dt": float(getattr(self.stop, "dt", self.dt))}


def build_cns_arm(config: dict, *, intrinsics, scene: str | None = None,
                  start_id: Any = None, pipeline: Any = None,
                  goal_mask_mode: str | None = None) -> CNSArm:
    """Build the CNS arm a resolved config asks for (``matcher.variant: cns_v1``).

    ``config["cns"]`` holds the published knobs: ``detector`` (default SIFT),
    ``device``, ``cns_root``, ``ckpt``, ``ransac`` (CNS's own geometric verification,
    **on** by default - it is part of the published method, and the standing
    no-RANSAC ruling is about the ViT-VS/ART-VS baselines), ``dist_scale``,
    ``step_gain`` and ``rate_compensate``. ``pipeline`` injects a ready-made CNS
    pipeline (the fast tier's stub, or one shared across cells).
    """
    from vsbench.vs.bench import config as cfgmod  # noqa: PLC0415 - avoids an import cycle

    cfgmod.validate(config)
    if config["matcher"]["variant"] != "cns_v1":
        raise ValueError(
            f"build_cns_arm needs matcher.variant 'cns_v1', got "
            f"{config['matcher']['variant']!r}")
    for block, why in _REFUSED_BLOCKS.items():
        if config.get(block):
            raise ValueError(f"cns cannot take a '{block}' block - {why} "
                             "(baseline_check.py:1942-1948)")
    ctrl = dict(config.get("controller") or {})
    bad = [k for k in _REFUSED_CONTROLLER_KEYS if k in ctrl]
    if bad:
        raise ValueError(
            f"cns cannot take controller keys {bad} - they are matcher/IBVS bench "
            "machinery and CNS runs its own controller. Its published step gain is "
            "`cns.step_gain` (vssim/baselines/cns.py, PUBLISHED)")
    if config["stop"]["rule"] == "dec_plateau":
        raise ValueError(
            "cns cannot run the `dec` stop - a lambda schedule scales OUR commanded "
            "twist and CNS runs its own controller (baseline_check.py:2392). Protocol "
            "v2 scores every arm on `cns_plateau`, which is CNS's own rule")

    block = dict(config.get("cns") or {})
    unknown = sorted(set(block) - set(CNS_CONFIG_KEYS))
    if unknown:
        raise ValueError(
            f"unknown cns config keys: {unknown} - the block holds only "
            f"{', '.join(CNS_CONFIG_KEYS)}. A key that is neither refused nor built is a "
            "setting the record would not carry")
    step_gain = float(block.pop("step_gain", PUBLISHED["step_gain"]))
    published_dt = float(block.pop("published_dt", PUBLISHED["dt"]))
    rate_compensate = bool(block.pop("rate_compensate", True))
    policy = CNSPolicy(dt=float(ctrl.get("dt", DEFAULT_DT)), gain=step_gain)
    pipe = CNSPipeline(intrinsics=intrinsics, pipeline=pipeline, **block)
    stop = build_stop_rule(config, **({"dt": policy.dt}
                                      if config["stop"]["rule"] == "cns_plateau"
                                      and "dt" not in config["stop"] else {}))
    arm = CNSArm(pipe, policy, stop, config,
                 name=((config.get("arm") or {}).get("name")), step_gain=step_gain,
                 published_dt=published_dt, rate_compensate=rate_compensate)
    # `matcher.goal_mask` names the masked cell; build_arm strips the key and passes the
    # mode, a direct caller leaves the key in place - both end in the same label
    if goal_mask_mode is None:
        goal_mask_mode = str((config.get("matcher") or {}).get("goal_mask") or "none")
    arm.goal_mask_mode = str(goal_mask_mode)
    if goal_mask_mode != "none":
        # the label says so, like every masked cell (vsbench.vs.bench.arms.arm_label)
        arm.name = arm.name.replace(f"/{getattr(stop, 'name', 'cap_only')}",
                                    f"/objmask/{getattr(stop, 'name', 'cap_only')}", 1)
    if scene is not None or start_id is not None:
        arm.set_episode(scene, start_id)
    return arm
