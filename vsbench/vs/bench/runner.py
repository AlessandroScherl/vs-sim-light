# vs-sim copy: vssim/bench/runner.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The episode loop: one start, one arm, one record.

Purpose
-------
Servo from a start pose toward the goal view and write down what happened. This is
the whole bench: the per-step order of operations, the telemetry, the divergence
abort, the stop bookkeeping, the record, and the run sidecar that says which code
and which settings produced it.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
- ``src/eval/baseline_check.py::run_episode`` L1493-1878 -> :func:`run_episode`:
  the loop (capture -> ``controller.step`` L1617 -> integrate L1804-1807), the
  **evaluation order** solve -> lambda scale -> telemetry -> non-finite check ->
  divergence -> quiet stop -> integrate (L1731-1807), the N0 divergence abort on
  the controller-visible feature error (L1759-1768), and the returned record's
  measurement fields.
- ``record_step_telemetry`` L1296-1448 -> :func:`step_telemetry`, field for field
  and rounding for rounding, including the N3.8 ``--telemetry-extended`` block
  (``zv``, ``w2``, ``gt_u``/``gt_v_px``/``gt_ok``/``err_patch``, ``g_lin``/``c_lin``,
  and points at 4 dp instead of 2).
- ``TelemetryWriter`` L1247-1269 -> :class:`TelemetryWriter`, unchanged: one
  gzipped JSON line per episode, flushed at every episode boundary (the
  segfault-at-teardown lesson - never buffer results past an episode).
- ``_rotvec_deg`` / ``rotation_error_deg`` / ``translation_error_mm`` ->
  :mod:`vsbench.vs.scene`.
- ``_vram_reset`` / ``_vram_peak_mb`` / ``_nvidia_smi_used_mb`` L1146-1188 ->
  :func:`vram_reset` / :func:`vram_peak_mb` / :func:`gpu_used_mb`.
- The hand-written run headers of 41 old runbooks -> :func:`open_run`, which is
  :func:`vsbench.vs.bench.run.start_run` (resolved config + git hash + seed + MD5SUMS).

Deliberately dropped (each with its reason)
-------------------------------------------
- **The arm-suffix grammar** (~600 lines) -> :mod:`vsbench.vs.bench.arms`.
- **The film hooks** (``dump_dir``/``dump_every``/``dump_matches``): the research
  repository's film writer, driven from outside the loop, is not here.
- **The pre-aligner** is one call: :func:`vsbench.vs.baselines.prealign.apply_prealign`,
  between placing the camera at the start pose and the first control step. It is a
  **physical re-pose**, so an episode that took it did not start where its start
  file says - which is why the record carries both ``start_rotation_deg`` and
  ``start_rotation_deg_after_prealign``.
- **The oracle plumbing** (``set_geometry``/``set_current`` and the
  ``hasattr(matcher, "set_current")`` sniff): gone for good.
- **The D6 termination reads** (``term_step`` / ``term_consec_*`` /
  ``termination_cfg``, ``TERMINATION``): a reporting-only metric that the owner
  ruling of 2026-08-17 kept explicitly non-claim-bearing, and protocol v2 section 5
  replaces it with a scored stop. Recomputable from telemetry if it is ever wanted.
- **The ``inc`` and ``frz`` stop schedules** and ``StopMechanic``'s unfreeze guard:
  ``inc`` was benched with its own prediction that it cannot stop and it did not;
  ``frz`` is now a matcher variant (``frozen_landmarks``), not a stop mechanic.
- Every GEN-VS import.

Fences
------
- **GT-blind, and asserted rather than promised - with a stated reach.** The true
  pose is read exactly twice per step: into the telemetry entry and, at the end, into
  the record. The controller and the stop rules are handed a
  :class:`vsbench.vs.baselines.control.StopObservation` and nothing else - that type is the real
  fence. :func:`assert_gt_blind` is a **structural tripwire on top of it**: before
  every episode it walks the controller and the stop rule and refuses to start if
  either *is*, or holds **as a direct attribute**, the scene, a pose, or an object
  exposing goal geometry. It does not walk the matcher, and it does not chase nested
  holders. The matcher is exempt **by design**: a ground-truth (``oracle``-kind) arm's
  matcher is ground truth, that is what it is for. What defends a table from one is
  therefore not this tripwire but the record - ``matcher_name`` (what the object calls
  itself), ``arm_kind`` and the ``oracle:`` label prefix.
- **The stop is recorded, never acted on**: the loop always
  runs to the cap, so the stop-scored and cap-scored reads of an episode come from
  the *same* episode. Only the divergence abort and a non-finite twist end a run early.
- **The lambda schedule scales the commanded twist**, which is exactly equivalent
  to running the law at the lower gain (it is linear in lambda) and is therefore
  identical for every arm by construction - the D6 lesson.
- **Depth is sampled by truncation** (:func:`vsbench.vs.bench.arms.depth_at`).
- **The scored stop is clocked by the rule, not by the loop.** :class:`CnsPlateau`
  measures 0.8 s against :attr:`StopObservation.t`, so the observation is stamped with
  ``arm.stop.dt`` when the rule carries one and with the controller's period otherwise.
  A rule built with ``stop: {dt: 0.02}`` therefore really runs at 50 Hz. Recorded-only rules
  see the commanded twist on the **arm's** period, and a recorded ``cns_plateau`` is
  built with it rather than with :data:`vsbench.vs.baselines.control.DEFAULT_DT`.
- **A second reference is scored beside the goal, never instead of it.** When the world
  change offers one (:meth:`vsbench.vs.bench.perturb.Perturbation.reference_pose` - the
  instance suite's size-matched pose), every translation read is repeated against it
  (``*_matched``, ``success_matched``) and the poses are recorded; ``episode.outline_iou``
  adds the object-outline overlap with the goal picture. Without either, the record is
  exactly what it was (a test pins the key set).
- **A third reference, and a recorder that only listens (2026-09-30).** With
  ``episode.depth_true`` on and a world change that offers it
  (:meth:`vsbench.vs.bench.perturb.Perturbation.depth_true_pose`), every translation read is
  repeated against the depth-true pose too (``*_depth_true``, ``success_depth_true``).
  ``run_episode(debug_log=...)`` hands a :class:`vsbench.vs.bench.debuglog.DebugLog` a copy of
  every step - after the twist and every stop verdict exist, before the camera moves -
  and the final frame, which is the outline score's render (one render, shared). Both
  are off by default, and then the record, the telemetry and the renders are exactly
  what they were (``tests/fast/test_debuglog.py`` proves it on the mock, arm by arm).
- **Default telemetry stores points at 2 dp, extended at 4 dp.** On a sub-pixel
  arm in the terminal phase 2 dp is 0.005 px of quantisation, which moves an
  offline re-solve by ~1e-2 relative; 4 dp puts it back inside 1e-3 (measured).
- **A live mask only for an arm that asks (2026-10-11, the external-arm hook).** An arm
  with ``requires_live_mask = True`` is handed, every step, a bool ``[H, W]`` mask of the
  live frame's objects: ``run_episode(live_mask_fn=...)``, default
  :func:`default_live_mask` (the goal mask's own rule, 5 mm above the floor). The
  **runner** reads the pose for it, as it already does for the telemetry; the arm sees
  the mask and nothing else, and the tripwire :func:`_checked_live_mask` refuses anything
  that is not a bool array of the frame's size (a mask must not carry geometry). The
  record gains ``live_mask_rule`` only for such an arm; every other arm's call, record
  and telemetry are byte-identical (``tests/fast/test_runner.py``).

Golden tests: ``tests/fast/test_runner.py`` - an episode on the mock plane converges
from a small offset; the record carries every measurement field of the frozen
schema; the extended telemetry's key set equals the frozen 31 keys and the default
set is those minus the eight extended ones; points are at 2 dp by default and 4 dp
extended; and :func:`assert_gt_blind` refuses an arm whose stop rule has been handed
the scene.
"""

from __future__ import annotations

import gzip
import json
import statistics as st
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from vsbench.vs.baselines import prealign as rotmod
from vsbench.vs.baselines.control import DivergenceGuard, StopObservation, build_stop_rule
from vsbench.vs.baselines.match import episode_seed
from vsbench.vs.bench import run as runmod
from vsbench.vs.bench import scoring
from vsbench.vs.bench.arms import Arm, episode_settings
from vsbench.vs.scene import (
    Pose,
    Scene,
    rotation_error_deg,
    rotvec_deg,
    translation_error_mm,
)

#: The eight per-step columns ``telemetry_extended`` adds. With the flag off the
#: entry is the frozen default record, pinned by a test.
EXTENDED_KEYS = ("zv", "w2", "gt_u", "gt_v_px", "gt_ok", "err_patch", "g_lin", "c_lin")

#: The ninth extended column, added after the frozen telemetry files were written:
#: ``fm``, the live frame's mean grey level. Anything reading an older stored file
#: must tolerate its absence, which is why it is named apart from
#: :data:`EXTENDED_KEYS` rather than added to that tuple.
EXTENDED_KEYS_AFTER_N38 = ("fm",)

#: A match counts as "right" at or below 1.5 patch units of ground-truth error,
#: measured at 512-equivalent resolution (16-pixel patches of a 512 square).
RIGHT_PATCH_THRESHOLD = 1.5

#: Attribute names whose presence marks an object as carrying ground truth.
_GT_MARKERS = ("goal_pose", "gt_correspondence", "get_camera_pose_optical", "render")


# ------------------------------------------------------------------ plumbing
def _round_list(a, nd: int) -> list:
    return [round(float(x), nd) for x in np.asarray(a, float).reshape(-1)]


def vram_reset() -> None:
    """Reset torch's CUDA peak-allocation counter. No-op without torch or CUDA."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - a diagnostic must never end an episode
        pass


class GpuUtilSampler:
    """Mean and peak GPU utilisation (%) over an episode, from ``nvidia-smi`` polled in a
    thread every ``period`` s; ``None`` where there is no GPU or no nvidia-smi."""

    def __init__(self, period: float = 0.5) -> None:
        import shutil
        import threading

        self.period = float(period)
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._bin = shutil.which("nvidia-smi")
        self._thread = threading.Thread(target=self._run, daemon=True) if self._bin else None

    def _read(self):
        try:
            out = subprocess.run([self._bin, "--query-gpu=utilization.gpu",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=2.0, check=False).stdout
            return float(out.strip().splitlines()[0])
        except Exception:  # noqa: BLE001 - a missing reading is not an episode failure
            return None

    def _run(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.period)

    def __enter__(self):
        if self._thread:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def summary(self) -> dict:
        if not self.samples:
            return {"gpu_util_mean_pct": None, "gpu_util_max_pct": None, "gpu_util_samples": 0}
        return {"gpu_util_mean_pct": round(float(np.mean(self.samples)), 1),
                "gpu_util_max_pct": round(float(np.max(self.samples)), 1),
                "gpu_util_samples": len(self.samples)}


def vram_peak_mb() -> float | None:
    """Peak torch-allocated CUDA memory since the last reset, MiB (protocol v2 section 8)."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return round(float(torch.cuda.max_memory_allocated()) / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


_NVSMI_FAILED = {"yes": False}


def gpu_used_mb() -> float | None:
    """One ``nvidia-smi`` sample of device memory in use, MiB. Caches its failure."""
    if _NVSMI_FAILED["yes"]:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=True)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        _NVSMI_FAILED["yes"] = True
        return None


class TelemetryWriter:
    """``telemetry.jsonl.gz`` - one gzipped JSON line per episode, flushed immediately.

    ``{"meta": {...ids...}, "steps": [...]}``. One writer per invocation (it
    truncates); episodes append through its lifetime and flush at each boundary,
    because a result buffered past an episode boundary is a result lost to a
    teardown segfault.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = gzip.open(self.path, "wt")   # noqa: SIM115 - closed by close()
        self.n = 0

    def write_episode(self, meta: dict, steps: list) -> None:
        self._f.write(json.dumps({"meta": meta, "steps": steps}, separators=(",", ":")) + "\n")
        self._f.flush()
        self.n += 1

    def close(self) -> None:
        self._f.close()


def assert_gt_blind(arm: Arm, scene: Scene) -> None:
    """Refuse to run if the control path can see the world (the GT-blind rule, enforced).

    Walks the controller, the stop rule and their immediate attributes and raises
    when any of them holds the scene itself or an object exposing goal geometry.
    Cheap (once per episode) and it is the difference between a documented
    invariant and a checked one: the whole claim of every stop-scored number is
    that the rule could not have been reading the answer.
    """
    for owner, obj in (("controller", arm.controller), ("stop", arm.stop)):
        seen = [obj, *vars(obj).values()]
        for value in seen:
            if value is scene:
                raise AssertionError(
                    f"{owner} holds the scene - the control path must be GT-blind "
                    f"(vsbench.vs.baselines.control.StopObservation is the "
                    f"only way in)")
            if isinstance(value, Pose) or (
                    value is not obj and any(hasattr(value, m) for m in _GT_MARKERS)):
                raise AssertionError(
                    f"{owner} holds {type(value).__name__}, which carries ground truth "
                    f"- the control path must be GT-blind")


# ---------------------------------------------------------------- telemetry
def step_telemetry(steps_log: list, *, step_i: int, pose: Pose, goal_pose: Pose,
                   twist_cmd, debug: dict, matches, lam_eff: float | None = None,
                   frozen: bool | None = None, gt: dict | None = None,
                   extended: bool = False, frame_mean: float | None = None) -> dict:
    """Append one per-step telemetry entry and return it.

    Field for field the old ``record_step_telemetry``: ``d_cam_m`` is the goal
    direction in the CURRENT optical frame (the reference the commanded ``v[:3]``
    should point along) and ``rot_to_goal_deg`` is its angular twin for ``v[3:]``.

    ``gt``, when given, carries ``uv`` (the ground-truth correspondence pixel per
    match), ``valid`` and ``err_patch`` (512-equivalent patch units). Everything
    read from it is **logging only**: it is computed after the controller has
    produced this step's twist and it is never returned to the loop.

    Three columns are **wider** than in the oldest records, deliberately: ``lam`` and
    ``frz`` appear on every arm, and ``mg`` appears whenever the matcher carries
    margins. A stop is recorded on every arm, so the effective gain is always
    meaningful; the margins are always computed. Nothing narrowed.
    """
    pos, R = pose.position, pose.rotation
    d_world = goal_pose.position - pos
    feat = debug.get("feature_error_norm", np.nan)
    entry: dict[str, Any] = {
        "i": int(step_i),
        "t_mm": round(float(np.linalg.norm(d_world)) * 1000.0, 3),
        "r_deg": round(rotation_error_deg(R, goal_pose.rotation), 4),
        "d_cam_m": _round_list(R.T @ d_world, 6),
        "rot_to_goal_deg": rotvec_deg(R.T @ goal_pose.rotation),
        "v": _round_list(twist_cmd, 7),
        "feat": (None if not np.isfinite(feat) else round(float(feat), 6)),
        "n": int(debug.get("n_matches", 0)),
    }
    if lam_eff is not None:
        entry["lam"] = round(float(lam_eff), 5)
    if frozen is not None:
        entry["frz"] = int(bool(frozen))
    if frame_mean is not None:
        # `fm`: the mean grey level of the LIVE frame this step steered on (extended
        # telemetry only). Beside the goal's per-capture
        # brightness it is what makes "unsettled goal, settled live frames" a read.
        entry["fm"] = round(float(frame_mean), 4)

    pg = np.asarray(debug.get("pts_goal", np.zeros((0, 2))), float)
    pc = np.asarray(debug.get("pts_cur", np.zeros((0, 2))), float)
    if len(pg) and len(pc):
        # 2 dp is plenty for the rounded-point era; for a sub-pixel arm in the
        # TERMINAL phase it is not - the feature residual there is itself a few
        # hundredths of a pixel. Under `extended` the coordinates are stored at
        # 4 dp: the same numbers, not pre-quantised.
        nd = 4 if extended else 2
        entry["gu"] = _round_list(pg[:, 0], nd)
        entry["gv"] = _round_list(pg[:, 1], nd)
        entry["cu"] = _round_list(pc[:, 0], nd)
        entry["cv"] = _round_list(pc[:, 1], nd)
        z = debug.get("depth_values")
        if z is not None and len(z):
            entry["z_med"] = round(float(np.median(np.asarray(z, float))), 4)
            if extended:
                entry["zv"] = _round_list(z, 5)
    conf = debug.get("confidence")
    if conf is not None:
        entry["conf"] = _round_list(conf, 4)
    w = debug.get("weights")
    if w is not None:
        w = np.asarray(w, float)
        entry["w"] = _round_list(w[:, 0] if w.ndim == 2 else w, 5)
        if extended and w.ndim == 2 and w.shape[1] > 1:
            entry["w2"] = _round_list(w[:, 1], 5)
        if debug.get("n_eff") is not None:
            entry["ne"] = round(float(debug["n_eff"]), 2)
    mg = debug.get("margins")
    if mg is not None:
        entry["mg"] = _round_list(mg, 4)
    if gt is not None and len(pg):
        uv, valid, err_patch = gt["uv"], np.asarray(gt["valid"], bool), gt["err_patch"]
        entry["gt_v"] = int(valid.sum())
        entry["gt_r"] = int((valid & (np.asarray(err_patch) <= RIGHT_PATCH_THRESHOLD)).sum())
        if extended:
            entry["gt_u"] = _round_list(np.asarray(uv)[:, 0], 2)
            entry["gt_v_px"] = _round_list(np.asarray(uv)[:, 1], 2)
            entry["gt_ok"] = [int(b) for b in valid.reshape(-1)]
            entry["err_patch"] = _round_list(err_patch, 4)
    off = None if matches is None else matches.offsets_patch
    if off is not None:
        off = np.asarray(off, float)
        entry["off_du"] = _round_list(off[:, 0], 4)
        entry["off_dv"] = _round_list(off[:, 1], 4)
    if extended and matches is not None and len(matches):
        # The linear patch-cell index, row-major `side*row + col`. The old code
        # had to INVERT the lattice from pixels (`_cell_lin`) because the
        # controller only ever saw pixels, and returned None whenever a point had
        # left the lattice; here the matcher hands the patch grid over directly,
        # so the index is exact and always available.
        side = int(matches.side)
        entry["g_lin"] = [int(v) for v in matches.goal_patch[:, 1] * side
                          + matches.goal_patch[:, 0]]
        entry["c_lin"] = [int(v) for v in matches.cur_patch[:, 1] * side
                          + matches.cur_patch[:, 0]]
    steps_log.append(entry)
    return entry


# ------------------------------------------------------------------- episode
def _pose_error(pose: Pose, goal: Pose, step_i: int, reference: Pose | None = None,
                depth_true: Pose | None = None) -> dict:
    """Where a stop fired: the error against the goal, and - when the perturbation offers a
    second reference (the instance suite's size-matched pose) - the error against it too,
    with the pose itself so any other reference can be scored offline. The rotation is
    measured against the goal orientation for both (the size-matched pose keeps it).
    ``depth_true`` (the third reference, ``episode.depth_true``) adds its distance the same
    way."""
    out = {"step": int(step_i),
           "translation_mm": translation_error_mm(pose.position, goal.position),
           "rotation_deg": rotation_error_deg(pose.rotation, goal.rotation)}
    if reference is not None:
        out["translation_mm_matched"] = translation_error_mm(pose.position, reference.position)
    if depth_true is not None:
        out["translation_mm_depth_true"] = translation_error_mm(pose.position,
                                                                depth_true.position)
    if reference is not None or depth_true is not None:
        out["position"] = [float(v) for v in np.asarray(pose.position, float).reshape(3)]
        out["rotation"] = np.asarray(pose.rotation, float).tolist()
    return out


def _outline(depth, intrinsics, pose) -> np.ndarray:
    """The object outline of one rendered view at the outline-score threshold."""
    from vsbench.vs.labels import OUTLINE_TOL_M, object_mask_from_depth

    return object_mask_from_depth(depth, intrinsics, pose, tol=OUTLINE_TOL_M)


def _iou_at(goal_outline, depth, intrinsics, pose) -> float | None:
    from vsbench.vs.labels import outline_iou

    iou = outline_iou(goal_outline, _outline(depth, intrinsics, pose))
    return None if iou is None else round(iou, 4)


def assert_goal_matches_starts(scene: Scene, payload: dict) -> None:
    """Refuse to fly a start file against a scene whose goal has drifted.

    The paired design's whole premise is that every arm and every session replays
    the same geometry, and the goal pose is what a stored campaign was scored
    against. A silently changed asset, fill, elevation or baked layout breaks that,
    and offline tests cannot catch it because the goal of a 3-D scene comes from
    the *measured* Isaac bounding box: this call is the real check. It is
    :func:`vsbench.vs.benchmark.check_scene_goal`, so it raises a `ValueError` naming the
    drift.
    """
    from vsbench.vs.benchmark import check_scene_goal

    check_scene_goal(scene, payload)


#: The record's ``live_mask_rule`` under :func:`default_live_mask`.
DEFAULT_LIVE_MASK_RULE = "objects_from_depth_5mm"


def default_live_mask(depth, intrinsics, pose) -> np.ndarray:
    """The live frame's objects: :func:`vsbench.vs.labels.object_mask_from_depth` at its default
    5 mm above the floor - the rule the goal mask (``matcher.goal_mask: objects``), the
    tier-0 stills and the instance suite already use. Bool ``[H, W]``."""
    from vsbench.vs.labels import object_mask_from_depth

    return object_mask_from_depth(depth, intrinsics, pose)


def live_mask_rule(live_mask_fn) -> str:
    """The name the record gives a live-mask rule: :data:`DEFAULT_LIVE_MASK_RULE` for the
    default, else the function's own ``rule`` attribute, else ``custom:<its name>``."""
    if live_mask_fn is None:
        return DEFAULT_LIVE_MASK_RULE
    named = getattr(live_mask_fn, "rule", None)
    return str(named) if named else (
        f"custom:{getattr(live_mask_fn, '__name__', type(live_mask_fn).__name__)}")


def _checked_live_mask(mask, depth) -> np.ndarray:
    """The tripwire: what an arm is handed as its live mask is a bool ``[H, W]`` array of the
    frame's size and nothing else - never a depth, a point map or a pose."""
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError(f"the live mask must be a bool numpy array, got "
                        f"{type(mask).__name__} of {getattr(mask, 'dtype', None)} - a mask that "
                        "carries more than in/out could carry geometry to the arm")
    hw = np.asarray(depth).shape[:2]
    if mask.shape != hw:
        raise ValueError(f"the live mask is {mask.shape}, the frame is {hw}")
    return mask


def run_episode(scene: Scene, arm: Arm, start: dict, config: dict, *,
                telemetry: list | None = None, ground_truth: bool = True,
                starts_payload: dict | None = None, on_step=None,
                perturbation=None, debug_log=None, live_mask_fn=None) -> dict:
    """Servo from ``start`` toward the scene's goal view; return the episode record.

    ``start`` is one entry of a frozen start file (``start_id``, ``cam_pos``,
    ``cam_R_opt``, and whatever stratification fields it carries).  ``config`` is a
    validated config; :func:`vsbench.vs.bench.arms.episode_settings` resolves the step
    budget, the divergence switch, the extra stop rules to record and the extended
    telemetry flag from its ``episode`` block.

    ``starts_payload``: the whole start file the ``start`` came from. When given,
    :func:`assert_goal_matches_starts` runs first, so a scene whose geometry drifted
    since the file was built fails on episode 1 instead of producing a campaign that
    is not paired with anything. A station runbook always passes it.

    ``telemetry``: pass a list to receive one entry per control step
    (:func:`step_telemetry`). ``on_step(step_i, rgb, debug, pose)`` is called after
    the twist exists and before the camera moves - the film hook
    (:mod:`vsbench.vs.bench.films`); it sees a copy of what the loop computed and nothing
    it returns is read. ``ground_truth=False`` suppresses the GT-labelled
    telemetry columns on a scene that offers them - the pose fields of the record
    are unaffected, since they are what an episode is *for*.

    ``perturbation``: a :class:`vsbench.vs.bench.perturb.Perturbation`. Its ``begin`` runs
    after the goal capture and before the start pose (the goal picture is clean, the
    live world is not), ``live`` on every live frame before the arm sees it, ``end``
    always - the world is clean again for the next episode. Its draw is on the record.

    ``debug_log``: a :class:`vsbench.vs.bench.debuglog.DebugLog` (the debug telemetry,
    2026-09-30). It is handed copies - the goal side once, every step after its twist
    and stop verdicts exist, the final frame, the record - and nothing it returns is
    read. ``None`` (the default) is exactly the loop without it.

    ``live_mask_fn(depth, intrinsics, pose) -> bool [H, W]``: the live mask handed to an arm
    that declares ``requires_live_mask = True`` (default :func:`default_live_mask`); the
    record then names the rule (``live_mask_rule``). Ignored for every other arm, whose call
    is the two-argument ``arm.step(rgb, depth)`` exactly as before.
    """
    if starts_payload is not None:
        assert_goal_matches_starts(scene, starts_payload)
    if perturbation is None:
        from vsbench.vs.bench.perturb import Perturbation

        perturbation = Perturbation()
    settings = episode_settings(config)
    dt = arm.dt
    # The SCORED rule's clock. `StopObservation.t` is what `CnsPlateau` measures its
    # 0.8 s hold against, so a rule built with its own `stop.dt` must be stamped with it.
    stop_dt = float(getattr(arm.stop, "dt", dt))
    goal = scene.goal_pose
    start_pose = Pose.from_start(start)
    assert_gt_blind(arm, scene)

    # goal view, then the start pose. The matcher caches the goal once per episode.
    scene.reset(goal)
    goal_rgb, goal_depth, _ = scene.render()
    arm.set_episode(scene.name, start["start_id"])
    goal_mask, goal_mask_share = None, None
    if getattr(arm, "goal_mask_mode", "none") == "objects":
        # landmarks from the goal's OBJECTS only: the floor masked out from the goal
        # depth against the scene's ground plane (z = 0), no segmentation involved
        from vsbench.vs.labels import object_mask_from_depth

        goal_mask = object_mask_from_depth(goal_depth, scene.intrinsics, goal)
        goal_mask_share = float(goal_mask.mean())
        if not goal_mask.any():
            raise ValueError(
                f"{scene.name}: matcher.goal_mask 'objects' keeps no pixel - the goal "
                "view is all ground plane (a poster scene?); the mask cell is for rooms")
    arm.reset(goal_rgb, goal_depth, goal_mask)
    # the outline-overlap score (episode.outline_iou): the goal object's outline from the
    # goal depth at the 1 mm threshold, compared at every recorded stop and at the end
    goal_outline = (_outline(goal_depth, scene.intrinsics, goal)
                    if settings["outline_iou"] else None)
    p_seed = perturbation.episode(scene.name, start["start_id"])
    p_draw = perturbation.begin(scene, p_seed, goal=goal,
                                working_distance=scene.working_distance,
                                intrinsics=scene.intrinsics)
    # a second pose to score against, if the world change offers one (the named swap's
    # size-matched pose); `None` for every other perturbation, and the record is then
    # exactly what it always was
    ref_fn = getattr(perturbation, "reference_pose", None)
    reference = ref_fn() if callable(ref_fn) else None
    # the third reference (2026-09-30): the depth-true pose, a pure getter like the one
    # above; scored in the record only under `episode.depth_true`
    dt_fn = getattr(perturbation, "depth_true_pose", None)
    depth_true = dt_fn() if callable(dt_fn) else None
    try:
        if debug_log is not None:
            debug_log.begin(scene=scene, arm=arm, start=start, start_pose=start_pose,
                            goal=goal, goal_rgb=goal_rgb, goal_depth=goal_depth,
                            goal_mask=goal_mask,
                            references={"size_matched": reference, "depth_true": depth_true},
                            truth_model=_truth_model(perturbation),
                            draw=p_draw, settings=settings)
        with GpuUtilSampler() as gpu:
            rec = _fly(scene, arm, start, config, settings, goal, goal_rgb, start_pose,
                       telemetry=telemetry, ground_truth=ground_truth, on_step=on_step,
                       perturbation=perturbation, dt=dt, stop_dt=stop_dt,
                       reference=reference, goal_outline=goal_outline,
                       depth_true=depth_true if settings["depth_true"] else None,
                       debug_log=debug_log, live_mask_fn=live_mask_fn)
        rec.update(gpu.summary())
    finally:
        perturbation.end(scene)
    rec["goal_mask_mode"] = str(getattr(arm, "goal_mask_mode", "none"))
    rec["goal_mask_share"] = goal_mask_share
    rec["perturbation"] = p_draw
    if debug_log is not None:
        debug_log.end(rec)
        # the recorder never raises; a failure inside it is written down here
        if getattr(debug_log, "error", None):
            rec["debug_log_error"] = debug_log.error
    return rec


def _truth_model(perturbation) -> dict:
    """The world change's truth model for the debug log (``DEFAULT_TRUTH`` when it states
    none). A failure here must not reach the flight: it becomes "no truth", with the
    reason."""
    from vsbench.vs.bench.perturb import DEFAULT_TRUTH

    tm_fn = getattr(perturbation, "truth_model", None)
    try:
        return tm_fn() if callable(tm_fn) else dict(DEFAULT_TRUTH)
    except Exception as exc:  # noqa: BLE001 - the recorder only listens
        return {**DEFAULT_TRUTH, "object": "none", "error": repr(exc)[:500]}


def _fly(scene, arm, start, config, settings, goal, goal_rgb, start_pose, *, telemetry,
         ground_truth, on_step, perturbation, dt, stop_dt, reference=None,
         goal_outline=None, depth_true=None, debug_log=None, live_mask_fn=None) -> dict:
    """The flight itself, from the start pose to the record (split out so the
    perturbation's ``end`` can wrap it in a ``finally``)."""
    # the live mask (2026-10-11): only for an arm that asks; every other arm is called with
    # two arguments exactly as before
    wants_live_mask = bool(getattr(arm, "requires_live_mask", False))
    mask_fn = (live_mask_fn or default_live_mask) if wants_live_mask else None
    scene.reset(start_pose)
    # The published pre-servo roll search, on an arm that declares
    # `rotation: {mode: prealign_4way}` and on no other. It physically rolls the camera
    # before the loop, which is what the published methods do and what every table
    # must state - so the pose the loop starts from is no longer `start_pose`, and both
    # rotations go on the record.
    prealigned = rotmod.apply_prealign(scene, arm, config, goal_rgb=goal_rgb)
    rot_after_prealign = (None if prealigned is None
                          else rotation_error_deg(scene.pose.rotation, goal.rotation))

    guard = DivergenceGuard()
    guard.reset()
    # the recorded-only rules run on their DEFAULTS at the ARM's period - never at
    # DEFAULT_DT, which would clock a recorded `cns_plateau` at 1/60 on an arm that
    # integrates at another rate and report a hold the run did not use. A rule an arm
    # OWNS runs at the arm's own constants: the DINOBot arm's `dinobot_err` threshold and
    # error space are settings of its `dinobot:` block (DinoBotArm.recorded_stop_constants)
    owned = dict(getattr(arm, "recorded_stop_constants", dict)() or {})
    also = {name: build_stop_rule({"stop": {"rule": name,
                                            **({"dt": dt} if name == "cns_plateau" else {}),
                                            **dict(owned.get(name) or {})}})
            for name in settings["record_stops"]}
    for rule in also.values():
        rule.reset()
    fired: dict[str, dict] = {}

    gt_fn = getattr(scene, "gt_correspondence", None) if ground_truth else None
    img_w, img_h = scene.intrinsics.width, scene.intrinsics.height
    gt_scale = np.array([512.0 / img_w, 512.0 / img_h])

    t0 = time.monotonic()
    feature_errs: list[float] = []
    n_matches: list[int] = []
    match_ms: list[float] = []
    control_ms: list[float] = []
    live_means: list[float] = []          # extended telemetry only (R26 correction 8)
    lam_min = 1.0
    lam_last = 1.0
    plateau_step = None
    reason = "max_iter"
    steps = 0
    frz_n = None
    # the debug log's per-step call (2026-09-30): made at most once per step, after every
    # decision of the step and before the camera moves; it reads copies and returns nothing
    verdicts: dict = {}
    fired_before: set = set()

    def _log(end):
        debug_log.step(step_i=steps, rgb=rgb, depth=depth, pose=pose,
                       twist_raw=twist.as_array(), v_cmd=v_cmd, lam=lam_last, debug=debug,
                       matches=matches, scored=verdict, recorded=verdicts or None,
                       fired=dict(fired), fired_now=[n for n in fired if n not in fired_before],
                       end=end)

    for steps in range(1, settings["max_iter"] + 1):
        if debug_log is not None:
            fired_before = set(fired)
            verdicts = {}
        rgb, depth, pose = scene.render()
        rgb = perturbation.live(rgb, steps)          # identity unless the mode says so
        if mask_fn is not None:
            # the runner reads the pose (as it does for the telemetry); the arm gets the mask
            live_mask = _checked_live_mask(mask_fn(depth, scene.intrinsics, pose), depth)
            twist, debug, matches = arm.step(rgb, depth, live_mask)
        else:
            twist, debug, matches = arm.step(rgb, depth)
        if settings["telemetry_extended"]:
            # how bright the LIVE frame is: the goal's settledness is recorded per
            # capture, so its counterpart has to be too.
            live_means.append(float(np.asarray(rgb, np.float64).mean()))
        feature_errs.append(float(debug.get("feature_error_norm", np.nan)))
        n_matches.append(int(debug.get("n_matches", 0)))
        match_ms.append(float(debug.get("match_ms", 0.0)))
        control_ms.append(float(debug.get("step_ms", 0.0)))
        if frz_n is None and getattr(arm.matcher, "frozen_landmarks", False):
            frz_n = len(matches)

        # 1. the scored rule, which is the only one allowed to scale lambda.
        obs = StopObservation.from_debug(debug, twist=twist, dt=stop_dt,
                                         landmark_ids=_landmark_ids(matches, debug))
        verdict = arm.stop.update(obs)
        lam_last = float(verdict.lambda_scale)
        lam_min = min(lam_min, lam_last)
        plateau_step = verdict.detail.get("plateau_step", plateau_step)
        v_cmd = twist.as_array() * lam_last
        if verdict.would_stop and arm.stop.name not in fired:
            fired[arm.stop.name] = _pose_error(pose, goal, steps, reference, depth_true)
            if goal_outline is not None:
                fired[arm.stop.name]["outline_iou"] = _iou_at(goal_outline, depth,
                                                              scene.intrinsics, pose)

        # 2. telemetry, BEFORE the breaks: the terminal step must be on film too,
        #    and the scene pose is still this step's capture pose here.
        if telemetry is not None:
            step_telemetry(
                telemetry, step_i=steps, pose=pose, goal_pose=goal, twist_cmd=v_cmd,
                debug=_telemetry_debug(debug, matches), matches=matches,
                lam_eff=arm.gain * lam_last, frozen=bool(frz_n is not None),
                gt=_gt_block(gt_fn, debug, pose, gt_scale, matches=matches),
                extended=settings["telemetry_extended"],
                frame_mean=live_means[-1] if live_means else None)

        if on_step is not None:
            on_step(steps, rgb, {**debug, "n_matches": int(debug.get("n_matches", 0))}, pose)

        if not np.all(np.isfinite(v_cmd)):
            reason = "non_finite_velocity"
            if debug_log is not None:
                _log(reason)
            break

        # 3. the N0 divergence abort: controller-visible feature error above 3x its
        #    first finite value. The break sits before the integration, so the
        #    camera stays at the capture pose of the aborting step.
        if settings["divergence_abort"] and guard.update(obs).would_stop:
            reason = "diverged_feature"
            if debug_log is not None:
                _log(reason)
            break

        # 4. the recorded-only rules (the published velocity-quiet stop on every arm,
        #    plus whatever else the config asks to record). They see
        #    the COMMANDED twist, which is what makes a lambda schedule visible to
        #    the published criterion.
        obs_cmd = StopObservation.from_debug(debug, twist=v_cmd, dt=dt,
                                             landmark_ids=obs.landmark_ids)   # arm's own clock
        for name, rule in also.items():
            # the same call in the same order as ever; its verdict is kept for the debug log
            vr = rule.update(obs_cmd)
            verdicts[name] = vr
            if vr.would_stop and name not in fired:
                fired[name] = _pose_error(pose, goal, steps, reference, depth_true)
                if goal_outline is not None:
                    fired[name]["outline_iou"] = _iou_at(goal_outline, depth,
                                                         scene.intrinsics, pose)

        if debug_log is not None:
            _log(None)
        scene.apply(v_cmd, dt)

    wall = time.monotonic() - t0
    final_pose = scene.pose        # no extra frame: `Scene.pose` is the current pose
    finite = [e for e in feature_errs if np.isfinite(e)]
    scored = fired.get(arm.stop.name)
    rec = {
        # ---- identity: what ran, stated, never parsed out of a name
        "scene": scene.name,
        "arm": arm.name,
        "matcher_variant": config["matcher"]["variant"],   # what the config declared
        "matcher_name": arm.matcher_name,                  # what the object actually is
        "arm_kind": arm.kind,
        "num_landmarks": int(config["matcher"]["num_landmarks"]),
        "tie_order": getattr(arm.matcher, "tie_order", None),
        "checkpoint": arm.checkpoint,
        "solve_unrounded": bool(arm.controller.solve_unrounded),
        "weighting": vars(arm.controller.weighting),
        "ema_alpha": float(arm.controller.ema_alpha),
        "min_matches": int(arm.controller.min_matches),
        "gain": float(arm.gain),
        "dt": float(dt),
        "max_iter": int(settings["max_iter"]),
        "seed": int(config["seed"]),
        "subsample_seed": episode_seed(scene.name, start["start_id"], int(config["seed"])),
        "start_id": start["start_id"],
        "start_cell": start.get("cell"),
        "start_tier": start.get("tier"),
        "start_roll_deg": start.get("roll_deg"),
        "starts_seed": start.get("starts_seed"),
        "working_distance_m": float(scene.working_distance),
        # The goal capture and the settle evidence beside it:
        # how many captures the GOAL frame was, how the brightness moved between them,
        # and the md5 of the one that was kept. A scene that captures no goal (a mock,
        # a stub) records `None` rather than numbers nobody produced.
        **_goal_capture_fields(scene),
        # the camera and the goal, so a stored telemetry file can be re-solved offline
        # without reaching back to the run's config
        "img_wh": [int(img_w), int(img_h)],
        "intrinsics": {"fx": scene.intrinsics.fx, "fy": scene.intrinsics.fy,
                       "cx": scene.intrinsics.cx, "cy": scene.intrinsics.cy},
        "goal_pos": [round(float(x), 6) for x in goal.position],
        # ---- outcome
        "steps": steps,
        "terminated_reason": reason,
        "wall_time_s": wall,
        "sec_per_step": wall / max(1, steps),
        "match_fps": (round(1000.0 / float(np.mean(match_ms)), 3)             # matcher alone
                      if match_ms and np.mean(match_ms) > 0 else None),
        "fps": max(1, steps) / wall if wall > 0 else None,
        "match_ms_med": round(st.median(match_ms), 4) if match_ms else None,
        "control_ms_med": round(st.median(control_ms), 4) if control_ms else None,
        "start_translation_mm": translation_error_mm(start_pose.position, goal.position),
        "start_rotation_deg": rotation_error_deg(start_pose.rotation, goal.rotation),
        # `None` unless the published 4-way ran; when it did, THIS is the rotation the
        # loop actually started from.
        "start_rotation_deg_after_prealign": rot_after_prealign,
        "final_translation_mm": translation_error_mm(final_pose.position, goal.position),
        "final_rotation_deg": rotation_error_deg(final_pose.rotation, goal.rotation),
        "feature_error_first": finite[0] if finite else None,
        "feature_error_last": finite[-1] if finite else None,
        # The live frames' brightness at the park, beside the goal's per capture:
        # "settled live frames against an unsettled goal" as a column.
        # `None` unless extended telemetry ran, because it is measured per step there.
        "live_rgb_mean_last50": (round(float(np.mean(live_means[-LIVE_MEAN_WINDOW:])), 4)
                                 if live_means else None),
        "mean_n_matches": float(np.mean(n_matches)) if n_matches else 0.0,
        "min_n_matches": int(np.min(n_matches)) if n_matches else 0,
        # ---- the stop: recorded, never acted on
        "stop_mode": arm.stop.name,
        "stop_probe": True,
        "stop_step": scored["step"] if scored else None,
        "stop_translation_mm": scored["translation_mm"] if scored else None,
        "stop_rotation_deg": scored["rotation_deg"] if scored else None,
        # `plateau_step` and `stop_trigger_step` are the same number under two names
        # (the frozen schema's and the newer one); under `dec_plateau` it is the plateau
        # trigger while `stop_step` above is the published quiet fire under the decayed
        # lambda - the meaning every stored record carries.
        "plateau_step": plateau_step,
        "stop_trigger_step": plateau_step,
        "stop_cfg": arm.stop.params,               # the EFFECTIVE constants, not the overrides
        "lam_scale_final": round(lam_last, 6),
        "lam_scale_min": round(lam_min, 6),
        "stops": fired,
        "frozen_landmarks": bool(getattr(arm.matcher, "frozen_landmarks", False)),
        "frz_n": frz_n,
        # ---- hardware (name the machine that made a number)
        "peak_vram_alloc_mb": vram_peak_mb(),
        "vram_used_mb": gpu_used_mb(),
    }
    # an arm may add its own published knobs to the record (the CNS arm's x2 and the
    # rate compensation - vsbench.vs.baselines.cns.CNSArm.record_fields). Never a measurement.
    rec.update(getattr(arm, "record_fields", dict)())
    if wants_live_mask:
        # only for an arm that asked: every other record keeps its exact key set
        rec["live_mask_rule"] = live_mask_rule(live_mask_fn)
    rec.update(scoring.classify_episode(rec))
    rules = tuple(config["scoring"]["success_rule"])
    rec["success"] = scoring.success_columns(rec, rules,
                                             working_distance_m=scene.working_distance)
    if reference is not None:
        # The instance suite's second reference (the size-matched pose): every
        # translation read again against it, and the same success rules on those reads.
        # Rotations stay against the goal orientation, which the reference keeps.
        rec["reference_pose"] = {
            "position": [float(v) for v in np.asarray(reference.position, float).reshape(3)],
            "rotation": np.asarray(reference.rotation, float).tolist()}
        rec["final_pose"] = {
            "position": [float(v) for v in np.asarray(final_pose.position, float).reshape(3)],
            "rotation": np.asarray(final_pose.rotation, float).tolist()}
        rec["start_translation_mm_matched"] = translation_error_mm(start_pose.position,
                                                                   reference.position)
        rec["final_translation_mm_matched"] = translation_error_mm(final_pose.position,
                                                                   reference.position)
        rec["stop_translation_mm_matched"] = (scored["translation_mm_matched"]
                                              if scored else None)
        matched_view = {**rec, "start_translation_mm": rec["start_translation_mm_matched"],
                        "final_translation_mm": rec["final_translation_mm_matched"],
                        "stop_translation_mm": rec["stop_translation_mm_matched"]}
        rec["success_matched"] = scoring.success_columns(
            matched_view, rules, working_distance_m=scene.working_distance)
    if depth_true is not None:
        # The third reference (episode.depth_true, 2026-09-30): the depth-true pose, the
        # same reads and the same success rules as the size-matched block above.
        rec["depth_true_pose"] = {
            "position": [float(v) for v in np.asarray(depth_true.position, float).reshape(3)],
            "rotation": np.asarray(depth_true.rotation, float).tolist()}
        rec.setdefault("final_pose", {
            "position": [float(v) for v in np.asarray(final_pose.position, float).reshape(3)],
            "rotation": np.asarray(final_pose.rotation, float).tolist()})
        rec["start_translation_mm_depth_true"] = translation_error_mm(start_pose.position,
                                                                      depth_true.position)
        rec["final_translation_mm_depth_true"] = translation_error_mm(final_pose.position,
                                                                      depth_true.position)
        rec["stop_translation_mm_depth_true"] = (scored["translation_mm_depth_true"]
                                                 if scored else None)
        dt_view = {**rec, "start_translation_mm": rec["start_translation_mm_depth_true"],
                   "final_translation_mm": rec["final_translation_mm_depth_true"],
                   "stop_translation_mm": rec["stop_translation_mm_depth_true"]}
        rec["success_depth_true"] = scoring.success_columns(
            dt_view, rules, working_distance_m=scene.working_distance)
    if goal_outline is not None or debug_log is not None:
        # one more render at the final pose - the loop's last frame was taken BEFORE the
        # last move - so the final overlap is of the pose the record reports. The debug
        # log's final frame is this same render (with the outline score on, it adds none).
        rgb_f, depth_f, pose_f = scene.render()
        if goal_outline is not None:
            rec["outline_iou_final"] = _iou_at(goal_outline, depth_f, scene.intrinsics, pose_f)
            rec["outline_iou_stop"] = (scored or {}).get("outline_iou")
        if debug_log is not None:
            debug_log.final(perturbation.live(rgb_f, steps + 1), depth_f, pose_f,
                            after_step=steps, end_reason=reason)
            rec["debug_log_s"] = round(float(debug_log.seconds), 4)
    return rec


#: What a record says about the goal frame when the scene does not measure it - a mock
#: plane, a stub, anything that is not :class:`vsbench.vs.isaac_scene.IsaacScene`.
#: Three `None`s rather than three absent columns: "not measured" and "not applicable"
#: read the same way in a table, and only one of them is true here.
NO_GOAL_CAPTURE = {"goal_render_warmup": None, "goal_capture_keep": None,
                   "goal_settle_series": None, "goal_frame_md5": None}

#: How many trailing steps `live_rgb_mean_last50` averages over. The other half of the
#: settle argument: the goal frame's brightness is recorded per capture, so the LIVE
#: frames' brightness at the park has to be too.
LIVE_MEAN_WINDOW = 50


def _goal_capture_fields(scene) -> dict:
    """The goal-capture columns, from whichever scene the episode ran on."""
    fn = getattr(scene, "goal_capture_record", None)
    return dict(fn()) if callable(fn) else dict(NO_GOAL_CAPTURE)


#: Which node set :class:`vsbench.vs.baselines.control.CnsPlateau` sees under a per-frame landmark
#: re-draw. **Default ``None``**. The three readings:
#:
#: 1. ``"drawn"`` - ids are this frame's drawn goal patches. CNS's arithmetic on a
#:    signal CNS never has: under our re-draw a patch is "absent" ~96 % of frames
#:    (one N3.8 episode's draws visit 1,019 of 1,024 patches), so the weights sit
#:    bimodal for hundreds of frames and the 90th-percentile trim on ``w * e`` drops
#:    first-witness rows regardless of their error. Exact arithmetic, wrong semantics.
#: 2. ``None`` - row i is node i. With a constant landmark count every weight sits at
#:    the fixed point 20 from frame 2, so the trim is on the plain error and there is
#:    no recency term. This is CNS's own fixed-graph case for the drawn set: it adds
#:    no artefact, and it is what ships.
#: 3. the whole mutual pool per frame, keyed by goal patch - the faithful port of
#:    CNS's benchmark, where the graph is every matched node and absence means the
#:    matcher lost it. The seam exists but not the reading itself: :func:`_landmark_ids`
#:    honours ``debug["stop_landmark_ids"]`` from any arm, which is how the CNS arm feeds
#:    the stop CNS's own graph nodes (:mod:`vsbench.vs.baselines.cns`). An arm that filled
#:    ``normalised_errors`` and ``stop_landmark_ids`` over the mutual pool would select
#:    reading 3; none does.
#:
#: The choice is claim-bearing: on 40 stored episodes readings 1 and 2 fired at
#: different steps in 27/40, spread -65 to +55 steps, moving the stop-scored landing by
#: up to 0.6 mm.
CNS_LANDMARK_IDS = None


def _landmark_ids(matches, debug: dict | None = None) -> np.ndarray | None:
    """The node identity handed to the scored stop - see :data:`CNS_LANDMARK_IDS`.

    An arm that carries its **own** node set says so in ``debug["stop_landmark_ids"]``
    and that always wins: the CNS arm's stop must see CNS's persistent
    correspondence-graph nodes, keyed the way CNS keys them, and not our re-drawn
    landmarks (:mod:`vsbench.vs.baselines.cns`). Reading 3 above - the whole mutual pool keyed
    by goal patch - is the same hook.
    """
    if debug is not None and debug.get("stop_landmark_ids") is not None:
        return np.asarray(debug["stop_landmark_ids"]).reshape(-1)
    if CNS_LANDMARK_IDS != "drawn" or matches is None or not len(matches):
        return None
    return np.asarray(matches.goal_patch[:, 1] * int(matches.side)
                      + matches.goal_patch[:, 0], np.int64)


def _telemetry_debug(debug: dict, matches) -> dict:
    """The debug dict widened with the columns the telemetry names differently.

    ``confidence`` is the column the weighting actually used - the head's trust
    when it is there and the matcher's similarity otherwise, which is exactly what
    the old single ``corr.conf`` meant.
    """
    out = dict(debug)
    if matches is not None and len(matches):
        rows = matches.to_rows()
        source = debug.get("weight_source") or ("trust" if "trust" in rows else "conf")
        out["confidence"] = rows.get(source, rows.get("conf"))
        out["margins"] = rows.get("margin")
    return out


def _gt_block(gt_fn, debug: dict, pose: Pose, gt_scale, matches=None) -> dict | None:
    """The ground-truth correspondence block for the telemetry - logging only."""
    if gt_fn is None:
        return None
    pg = np.asarray(debug.get("pts_goal", np.zeros((0, 2))), float)
    pc = np.asarray(debug.get("pts_cur", np.zeros((0, 2))), float)
    if not len(pg):
        return None
    uv, valid = gt_fn(pg, pose)
    err_patch = np.linalg.norm((pc - np.asarray(uv)) * gt_scale, axis=1) / 16.0
    return {"uv": uv, "valid": valid, "err_patch": err_patch}


# ------------------------------------------------------------- the sidecar
class EpisodeSink:
    """Where a campaign's episodes go: the records file, the telemetry, the sidecar.

    Wraps :func:`vsbench.vs.bench.run.start_run`, so every run directory carries its
    resolved config, git hash, seed and MD5SUMS without a runbook writing a line of
    it. Use as a context manager; :meth:`close` stamps and checksums the directory.
    """

    def __init__(self, run: runmod.RunDir, *, telemetry: bool = False) -> None:
        self.run = run
        self.path = run.path
        self.records: list[dict] = []
        self._f = (self.path / "baseline_check.jsonl").open("w", encoding="utf-8")
        self.telemetry = TelemetryWriter(self.path / "telemetry.jsonl.gz") if telemetry else None

    def __enter__(self) -> EpisodeSink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, record: dict, steps: list | None = None) -> None:
        """Append one episode record (and its telemetry), flushed immediately."""
        self.records.append(record)
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        if self.telemetry is not None and steps is not None:
            # the meta block identifies the episode; the numbers live in the steps
            self.telemetry.write_episode(
                {k: record.get(k) for k in
                 ("scene", "arm", "arm_kind", "matcher_variant", "matcher_name",
                  "num_landmarks", "tie_order", "checkpoint", "stop_mode", "stop_cfg",
                  "gain", "dt", "max_iter", "start_id", "start_cell", "starts_seed",
                  "subsample_seed", "working_distance_m", "img_wh", "intrinsics",
                  "goal_pos", "terminated_reason", "steps")},
                steps)

    def close(self) -> None:
        self._f.close()
        if self.telemetry is not None:
            self.telemetry.close()
        runmod.finish_run(self.run)


def open_run(name: str, config: dict, out_root=None, *, telemetry: bool = False,
             allow_dirty: bool = False, allow_no_git: bool = False,
             extra: dict | None = None) -> EpisodeSink:
    """Start a run directory and return the sink episodes are written to.

    ``extra`` is merged into the sidecar's ``run.json`` (:func:`vsbench.vs.bench.run.start_run`):
    facts of the *session* that are not in the arm's config - the goal capture's
    ``goal_render_warmup`` is the first of them.
    """
    run = runmod.start_run(name, config, out_root, allow_dirty=allow_dirty,
                           allow_no_git=allow_no_git, extra=extra)
    return EpisodeSink(run, telemetry=telemetry)
