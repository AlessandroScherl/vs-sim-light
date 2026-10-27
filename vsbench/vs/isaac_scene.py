# vs-sim copy: vssim/isaac_scene.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The Isaac Sim world behind the ``Scene`` protocol.

Purpose
-------
:class:`IsaacScene` is the real implementation of :class:`vsbench.vs.scene.Scene`: a
free-flying D435i camera over a built USD stage, driven by the same four verbs the
mock offers, so an episode does not know which world it is flying in. Everything
Isaac is imported **inside** methods, so the module imports and its maths is
unit-tested on a laptop with no Isaac (the two-runtime rule); instantiating the
class needs ``$ISAAC_PY``.

A typical session::

    from vsbench.vs.isaac_scene import IsaacScene
    from vsbench.vs.scenes import goal_capture_settings, load_scenes

    specs, _ = load_scenes()
    warmup, keep = goal_capture_settings()
    with IsaacScene(goal_render_warmup=warmup, goal_capture_keep=keep) as scene:
        scene.build_eval_scene(specs["hollywood"])     # camera now at the goal
        goal_rgb, goal_depth, goal_pose = scene.render()   # captured once, cached
        scene.reset(start_pose)
        for _ in range(300):
            rgb, depth, pose = scene.render()
            twist = my_controller(goal_rgb, rgb)       # optical-frame (v, w)
            scene.apply(twist, dt=1 / 60)

Fences
------
- **Intrinsics come from ``camera.yaml``, never from a hardcoded camera.**
- **``R_INTERNAL`` is measured, not assumed.** A bare ``UsdGeom.Camera`` must
  measure the identity; above :data:`R_INTERNAL_TOL` the scene refuses to run
  rather than servo a possibly rotated command.
- **The occlusion test is this scene's own job.** The protocol's
  ``gt_correspondence(pts, cur_pose)`` carries no current depth, so the scene
  keeps the depth map it last rendered and uses it as ``depth_b``; on the
  cluttered rooms, without it the ``gt_*`` columns over-count. It raises rather
  than silently skipping the test when the cached frame is not the pose asked
  about.
- **The goal frame is rendered once and served from** :meth:`IsaacScene.render`:
  RTX is not frame-deterministic, so re-rendering the goal moves its pixels as
  well as costing a render.
- **Every referenced USD must exist.** Repo-relative library paths resolve
  against ``$VSSIM_ASSET_ROOT``, not this repo, and :meth:`place_objects` refuses
  a missing file: an absent reference makes an empty prim, a garbage bounding box
  and a wrong goal distance, in silence.
- **Objects are placed by one of two paths, never a mix.** An item with the instance
  suite's placement fields (``canonical``, ``upright_yaw_add_deg``, ``size_m``,
  ``yaw_deg: long_axis_x`` - :func:`vsbench.vs.scenes.uses_placement`) is oriented by
  :func:`vsbench.vs.scenes.item_quat` and sized to its ``size_m``; every other item takes the
  original arithmetic unchanged, so no older scene moves.
- **The scene names its renderer** (:attr:`IsaacScene.render_mode`), so anything
  stored and matched against it can call :func:`check_renderer_match` instead of
  assuming.
- **The stage is closed once, deliberately** (:meth:`IsaacScene.close`). Leaving
  it open is what made every stage exit 1: the interpreter's ``atexit`` handlers
  tear carb down under a live ``SimulationApp`` and Isaac's plugins segfault
  *after* a perfectly good run, so a runbook could not gate on the exit code at
  all. :meth:`close` is idempotent and drops the reference first.
- **A scene is held as a context manager** - ``with IsaacScene(...) as scene:``.
  :meth:`__exit__` prints the traceback and flushes **before** :meth:`close` runs
  and re-raises, so a stage that dies says why and exits non-zero.
- **The GOAL capture takes ``goal_render_warmup`` frames and keeps the last.**
  The first RTX capture after a pose change carries render history, and the goal
  frame is captured once per scene, first thing - it was the least converged
  frame of every run until this rule. The live frames of an episode are
  unchanged. ``1`` = no warm-up. The capture also records **how settled it was**
  (:meth:`IsaacScene.goal_capture_record`), so a campaign shows its goal's
  convergence instead of claiming it. No capture count is written in this module;
  ``configs/scenes.yaml`` holds it (:func:`vsbench.vs.scenes.goal_capture_settings`).

Provenance
----------
Extracted from the Refine-VS research repository (``refinevs/data/isaac_scene.py``),
ported there from ``genvs-sim`` ``src/eval/free_flying_scene.py`` (the camera rig
and the optical/USD conventions), ``src/eval/eval_scenes.py`` (the builders) and
``src/refine/isaac_adapter.py`` (lighting, scale jitter, prim clearing), and
proven equal to the old rig frame for frame on a station run. That project's
dataset-plan builder (``build(plan)``) stayed behind; :meth:`build_eval_scene` is
the one builder here.

Golden tests: ``tests/fast/test_isaac_scene.py`` (Isaac-free maths + the seg-mask
contract); ``tests/slow/test_station.py`` for the station render.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from vsbench.vs.assets import ASSET_ROOT_ENV, resolve_usd_path, target_for
from vsbench.vs.camera import CameraConfig
from vsbench.vs.labels import gt_correspondence
from vsbench.vs.scene import Pose, integrate_optical_velocity, pose_to_T
from vsbench.vs.scenes import (
    LONG_AXIS_X,
    SceneSpec,
    _quat_mul,
    axis_angle_quat,
    check_layout,
    item_quat,
    load_layout,
    long_axis_yaw,
    placed_item,
    sample_scatter,
    uses_placement,
)

#: Optical -> USD camera: a USD ``Camera`` looks down -Z with +Y up, optical is
#: +Z forward with +Y down. Self-inverse.
R_OPT2USD_BASE = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
#: Above this the measured camera-prim rotation is not the identity and the
#: scene refuses to run (the non-identity compensation order is unvalidated).
R_INTERNAL_TOL = 1e-3
TARGET_PRIM = "/World/Target"


#: Which capture of the goal is kept. ``last`` is the rule; ``first`` is a diagnostic
#: case (eight renders, keep the first).
GOAL_KEEPS = ("last", "first")
GOAL_KEEP_DEFAULT = "last"

#: How many EXTRA orchestrator steps one capture may spend waiting for the annotators to
#: come back with a frame in them. A stated default a caller may override.
#:
#: WHY IT EXISTS. A six-scene runbook launched one Isaac process straight after the
#: previous one had exited, and ten seconds into the drill scene's goal capture the
#: FIRST read of the rgb annotator came back with shape ``(0,)``. Three manual runs of
#: the same scene, each launched alone, had rendered it fine: it is a transient of the
#: renderer coming up, not a property of the scene.
CAPTURE_RETRIES_DEFAULT = 5


def _check_warmup(n) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or int(n) < 1:
        raise ValueError(f"goal_render_warmup must be an int >= 1 (1 = no warm-up), got {n!r}")
    return int(n)


def _check_keep(keep: str) -> str:
    if keep not in GOAL_KEEPS:
        raise ValueError(f"goal_capture_keep must be one of {GOAL_KEEPS}, got {keep!r}")
    return str(keep)


@contextlib.contextmanager
def nonzero_exit_on_error(stage: str = "the stage"):
    """Guarantee a **non-zero exit status** when the body dies, whatever Isaac does next.

    The other half of the silent-rc-0 hazard (:meth:`IsaacScene.__exit__` is the
    first half). That half makes a dying stage *say* why: the traceback is printed and
    both streams flushed **before** ``SimulationApp.close()`` runs, because ``close()``
    tears carb down and can take pending output with it. What it cannot do is control
    the **status**: on the station's Isaac build ``close()`` ends the process itself,
    and it ends it with **0**. A stage that fails and reports success is the worst
    thing a harness can produce, and the code must not lie.

    So the status is decided **here, before anything can exit on our behalf**::

        with IsaacScene(...) as scene:
            with nonzero_exit_on_error("capture"):
                ...the work...

    On the way out of the body this prints the traceback, flushes both streams and calls
    ``os._exit(1)`` — which no ``atexit`` handler, no ``finally`` and no
    ``SimulationApp.close()`` can override. The simulator is not closed on that path,
    deliberately: the process is ending either way, the operating system reclaims it, and
    a wrong exit status costs a station night while an unclosed simulator costs nothing.
    On the **success** path nothing happens at all, so the context manager's ordinary
    ``close()`` still runs and the stage still exits 0.

    ``KeyboardInterrupt`` and ``SystemExit`` pass through untouched: a Ctrl-C at 02:00
    already exits non-zero, and a ``SystemExit`` is a status a caller chose.
    """
    try:
        yield
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        print(f"[{stage}] FAILED — exiting non-zero before the simulator can close",
              file=sys.stderr)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


class RendererMismatchError(RuntimeError):
    """A current view and the thing it is matched against were rendered differently."""


def render_mode_of(render_version: str | None) -> str | None:
    """``'..._rtx'`` -> ``'rtx'``, ``'..._pathtrace'`` / ``'..._pt'`` -> ``'pt'``."""
    rv = str(render_version or "").lower()
    if "rtx" in rv:
        return "rtx"
    if "pathtrace" in rv or "_pt" in rv:
        return "pt"
    return None


def manifest_render_mode(manifest_path) -> str | None:
    """Renderer lineage of a stored set, from its manifest's ``render_version``.

    Handles the top-level-dict and the list-of-entries manifest shapes. Absent
    file or absent field -> ``None`` (a pre-guard manifest, which makes the guard
    warn rather than pass in silence). A manifest that **mixes** renderers raises:
    that is a corrupt stored set, not an unknown one.
    """
    mp = Path(manifest_path)
    if not mp.exists():
        return None
    doc = json.loads(mp.read_text()) or {}
    entries = doc if isinstance(doc, list) else [doc]
    modes = {render_mode_of(e.get("render_version")) for e in entries
             if isinstance(e, dict)}
    modes.discard(None)
    if len(modes) > 1:
        raise RendererMismatchError(
            f"stored render manifest {mp} mixes renderers {sorted(modes)} - a stored "
            f"set must be a single lineage; re-render it under one renderer.")
    return next(iter(modes), None)


def check_renderer_match(scene_mode: str, target_mode: str | None, *,
                         where: str = "") -> None:
    """Refuse a scene whose renderer disagrees with the stored thing it matches.

    The current view and its target must share a renderer or IBVS never drives
    the feature error to ~0 at the true goal. An **unknown** target lineage passes
    through but WARNS loudly: the guard cannot verify what was never recorded, and
    silence there once hid a real cross-renderer hazard (audit 2026-07-18).

    Any stored render matched against a live scene is the state this guard exists
    for.
    """
    loc = f" [{where}]" if where else ""
    if target_mode is None:
        print(f"[renderer-guard] WARNING{loc}: target renderer lineage UNKNOWN (no "
              f"render_version recorded) - cannot verify against the {scene_mode!r} "
              f"scene; proceeding. Stamp the manifest to restore the guard.", flush=True)
        return
    if scene_mode != target_mode:
        raise RendererMismatchError(
            f"RENDERER MISMATCH{loc}: the scene renders with {scene_mode!r} but the "
            f"targets were rendered with {target_mode!r}. Current view and target MUST "
            f"share a renderer or IBVS never reaches zero feature error at the goal.")


# ------------------------------------------------------- Isaac-free math
def reorthogonalize(R) -> np.ndarray:
    """Snap a drifted rotation back onto SO(3) via SVD (every ``reortho_every``
    integration steps, to fight float drift)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def mat_to_quat_wxyz(R) -> np.ndarray:
    """3x3 rotation -> unit quaternion ``(w, x, y, z)`` (the trace formula)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S,
             (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S,
             (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S,
             (R[1, 2] + R[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S,
             (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_wxyz_to_mat(q) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` -> 3x3 rotation. Inverse of the above."""
    w, x, y, z = (float(c) for c in np.asarray(q, dtype=np.float64))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def optical_R_to_usd_quat(R_opt, r_opt2usd=R_OPT2USD_BASE) -> np.ndarray:
    """Optical->world rotation (columns = optical axes) -> the USD orient quat."""
    return mat_to_quat_wxyz(np.asarray(R_opt, dtype=np.float64) @ r_opt2usd)


def usd_quat_to_optical_R(q_wxyz, r_opt2usd=R_OPT2USD_BASE) -> np.ndarray:
    """USD orient quat -> optical->world rotation. Inverse of the above."""
    return quat_wxyz_to_mat(q_wxyz) @ np.linalg.inv(np.asarray(r_opt2usd, np.float64))


def seg_to_target_mask(seg, target_path_prefix: str = TARGET_PRIM) -> np.ndarray:
    """Instance segmentation -> the boolean foreground of the target prim.

    Contract (``seg_mask.py``): ids labelled under the prefix give the exact
    mask; **no** target ids but other ``/World/`` prims labelled is a loud
    refusal (the historical ``arr != arr[0, 0]`` fallback would call every
    distractor "target"); neither one is the legacy corner-pixel fallback for a
    single-object scene whose annotator info glitched.
    """
    data = seg["data"] if isinstance(seg, dict) and "data" in seg else seg
    info = seg.get("info", {}) if isinstance(seg, dict) else {}
    idmap = info.get("idToLabels") or info.get("idToSemantics") or {}
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = arr[..., 0]
    ids = set()
    for k, v in idmap.items():
        if target_path_prefix in str(v):
            with contextlib.suppress(TypeError, ValueError):
                ids.add(int(k))
    if ids:
        return np.ascontiguousarray(np.isin(arr, list(ids)).astype(bool))
    others = sorted({str(v) for v in idmap.values()
                     if "/World/" in str(v) and target_path_prefix not in str(v)})
    if others:
        raise RuntimeError(
            f"instance segmentation carries {len(others)} labelled prim(s) not under "
            f"'{target_path_prefix}' (e.g. {', '.join(others[:3])}) and NONE under it - "
            f"refusing the everything-differs-from-the-corner-pixel fallback, which "
            f"would silently contaminate the GT target mask in a multi-object scene")
    return np.ascontiguousarray((arr != arr.flat[0]).astype(bool))


# ------------------------------------------------------------- the scene
class IsaacScene:
    """A booted Isaac stage with a free-flying D435i camera.

    Implements :class:`vsbench.vs.scene.Scene`. ``name``, ``goal_pose`` and
    ``working_distance`` are set by :meth:`build_eval_scene`; before one they raise,
    because an episode scored against an unset goal is a silent wrong number.

    Hold it as a context manager (``with IsaacScene(goal_render_warmup=8) as scene:``)
    and run it under Isaac Sim's own interpreter (``$ISAAC_PY``). Constructing it with
    ``boot=False`` gives an unbooted object whose Isaac-free maths is testable
    anywhere.
    """

    #: What this scene renders with. RTX is the fielded renderer and the
    #: path-traced path is deliberately gone, but stored things matched against a
    #: live scene must be able to ask (:func:`check_renderer_match`).
    render_mode = "rtx"

    def __init__(self, *, goal_render_warmup: int, cam: CameraConfig | None = None,
                 steps_per_capture: int = 1, goal_capture_keep: str = GOAL_KEEP_DEFAULT,
                 rt_subframes: int = 4, reortho_every: int = 300,
                 capture_retries: int = CAPTURE_RETRIES_DEFAULT,
                 verbose: bool = True, boot: bool = True) -> None:
        self.cam = cam or CameraConfig()
        #: orchestrator steps inside ONE capture - 1 on every campaign ever flown. Not
        #: to be confused with `goal_render_warmup`: the config key
        #: `scene.render_warmup_steps` counts whole GOAL CAPTURES; this one changes
        #: every live frame.
        self.steps_per_capture = int(steps_per_capture)
        #: `scene.render_warmup_steps` of the scenes yaml: how many whole GOAL captures
        #: are taken. **Required, no default**: a scene that captures no goal says
        #: `goal_render_warmup=1` at the call site rather than inheriting a count nobody
        #: wrote down. `1` = no warm-up. A campaign states it through
        #: :func:`vsbench.vs.scenes.goal_capture_settings`, and every record should say
        #: which ran.
        self.goal_render_warmup = _check_warmup(goal_render_warmup)
        #: which of those captures is the goal: `last` is the rule, `first` exists
        #: for one diagnostic (eight renders, keep the first, to separate "the goal is
        #: unsettled" from "eight renders changed the scene").
        self.goal_capture_keep = _check_keep(goal_capture_keep)
        #: how settled the kept goal capture was: per capture, its mean brightness, its
        #: rgb and depth md5s and the mean-abs difference from the capture before it.
        #: `None` until a goal is captured. It is EVIDENCE, not a knob.
        self.goal_settle: dict | None = None
        #: md5 of the goal frame that was actually kept - two processes whose goal
        #: captures differ can be told apart by this alone.
        self.goal_frame_md5: str | None = None
        #: how many extra orchestrator steps ONE capture may spend on an empty annotator
        #: (:data:`CAPTURE_RETRIES_DEFAULT`), recorded here rather than implied
        self.capture_retries = int(capture_retries)
        #: how many it actually spent on the LAST capture. `0` on every frame that came
        #: back whole, which is every frame the fielded chain has ever taken.
        self.last_capture_retries = 0
        #: the booted SimulationApp, or None before a boot and after :meth:`close`
        self.sim = None
        self.rt_subframes = int(rt_subframes)
        self.reortho_every = int(reortho_every)
        self._verbose = bool(verbose)
        self._reortho_counter = 0
        self.name = "unbuilt"
        self._goal: Pose | None = None
        self._distance: float | None = None
        self._goal_frame: tuple | None = None
        self._last: tuple | None = None          # (pose, rgb, depth) of the last render
        #: what the placement path resolved for each extra object it put down (by prim path)
        self.extra_placements: dict[str, dict] = {}
        self._pose = Pose(position=np.zeros(3), rotation=R_OPT2USD_BASE.copy())
        self.R_OPT2USD = R_OPT2USD_BASE.copy()
        if boot:                                          # pragma: no cover - Isaac
            self._boot()

    # ------------------------------------------------------------ the facts
    @property
    def intrinsics(self):
        return self.cam.intrinsics()

    @property
    def working_distance(self) -> float:
        if self._distance is None:
            raise RuntimeError(f"{self.name}: no goal built yet - call build_eval_scene()")
        return float(self._distance)

    @property
    def goal_pose(self) -> Pose:
        if self._goal is None:
            raise RuntimeError(f"{self.name}: no goal built yet - call build_eval_scene()")
        return self._goal

    @property
    def pose(self) -> Pose:
        return self._pose

    # ------------------------------------------------------------ the verbs
    def reset(self, pose: Pose) -> None:
        self.set_camera_pose_optical(pose.position, pose.rotation)

    def apply(self, twist, dt: float) -> None:
        self._pose = integrate_optical_velocity(self._pose, np.asarray(twist), dt)
        self._reortho_counter += 1
        if self._reortho_counter >= self.reortho_every:
            self._pose = Pose(position=self._pose.position,
                              rotation=reorthogonalize(self._pose.rotation))
            self._reortho_counter = 0
        self._write_pose_to_usd()

    def render(self) -> tuple[np.ndarray, np.ndarray, Pose]:
        """Render the current pose - or serve the cached goal frame at the goal.

        An episode loop typically opens with ``reset(goal); render()`` to capture
        the goal view. RTX is not frame-deterministic, so re-rendering it per
        episode both costs a render and moves the goal pixels; the cache is
        therefore served from **here**, which is the call a loop actually makes.
        """
        if self._at_goal():
            if self._goal_frame is None:
                self._goal_frame = self._capture_goal()
            rgb, depth = self._goal_frame
        else:
            rgb, depth = self.capture_rgbd()
        self._last = (self._pose, rgb, depth)
        return rgb, depth, self._pose

    def _at_goal(self) -> bool:
        return (self._goal is not None
                and np.array_equal(self._pose.position, self._goal.position)
                and np.array_equal(self._pose.rotation, self._goal.rotation))

    def render_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """The goal view, rendered **once** per scene and cached.

        Writes the USD pose directly rather than through :meth:`reset`, which
        would zero ``_reortho_counter`` in the middle of an episode.
        """
        if self._goal_frame is None:
            keep = self._pose
            self._pose = self.goal_pose
            self._write_pose_to_usd()
            self._goal_frame = self._capture_goal()
            self._pose = keep
            self._write_pose_to_usd()
        return self._goal_frame

    def _capture_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """The goal capture: ``goal_render_warmup`` captures, one of them kept.

        The rule keeps the **last**; ``goal_capture_keep: first`` keeps the first of
        the same N renders and exists for one diagnostic: eight renders that keep the
        first separate "the goal was unsettled" from "rendering eight times changed
        the scene".

        It also **measures how settled the goal was** and keeps that beside the frame
        (:meth:`goal_capture_record`), so a campaign can show it rather than assert it:
        the benchmark's settle experiment (2026-09-05) found that with a settled goal
        two launch orderings park at the same place - a 2 mm difference chased for
        days was an under-converged first capture - and that the third capture still
        differed from the second by 0.56-0.59 grey levels, with the brightness still
        climbing at the fourth. So the number of captures is a config field and
        **nothing here is tuned**: no count is written into this module.
        """
        means: list[float] = []
        diffs: list[float | None] = []
        rgb_md5: list[str] = []
        depth_md5: list[str] = []
        retries: list[int] = []
        frames: list[tuple[np.ndarray, np.ndarray]] = []
        prev = None
        for i in range(self.goal_render_warmup):
            rgb, depth = self.capture_rgbd()
            rgb = np.ascontiguousarray(rgb)
            depth = np.ascontiguousarray(depth)
            retries.append(int(self.last_capture_retries))
            # THE SHAPES ARE CHECKED BEFORE ANYTHING IS COMPUTED FROM THEM. An empty
            # capture used to reach the difference below and die inside numpy on
            # `operands could not be broadcast together with shapes (1080,1920,3) (0,)`,
            # which names neither the capture nor the scene. `capture_rgbd` now raises
            # before this is reached, so this is the second wall: it says which capture
            # was empty.
            if rgb.size == 0 or (prev is not None and rgb.shape != prev.shape):
                where = (f"capture {i + 1} of {self.goal_render_warmup} is {rgb.shape}"
                         if rgb.size == 0 else
                         f"capture {i + 1} is {rgb.shape} and capture {i} was "
                         f"{prev.shape}")
                raise RuntimeError(
                    f"{self.name}: goal {where} - a capture that returns nothing is not "
                    f"a picture, and the settle series cannot difference two captures of "
                    f"different sizes."
                    + (" The earlier one was empty."
                       if prev is not None and prev.size == 0 else ""))
            frames.append((rgb, depth))
            means.append(round(float(np.asarray(rgb, np.float64).mean()), 6))
            diffs.append(None if prev is None else round(float(np.abs(
                rgb.astype(np.int32) - prev.astype(np.int32)).mean()), 6))
            # per-capture checksums: they are what told four station settle runs
            # apart, and a mean alone cannot.
            rgb_md5.append(hashlib.md5(rgb.tobytes()).hexdigest())
            depth_md5.append(hashlib.md5(depth.tobytes()).hexdigest())
            prev = rgb
        kept_index = 0 if self.goal_capture_keep == "first" else len(frames) - 1
        self.goal_settle = {"n_captures": int(self.goal_render_warmup),
                            "keep": self.goal_capture_keep, "kept_index": kept_index,
                            "rgb_mean": means, "diff_prev": diffs,
                            "rgb_md5": rgb_md5, "depth_md5": depth_md5,
                            # how many extra orchestrator steps each capture spent on an
                            # empty annotator; all zeros on a frame that came back whole
                            "capture_retries": retries}
        self.goal_frame_md5 = rgb_md5[kept_index]
        if self._verbose:
            print(f"[isaac_scene] goal capture: {self.goal_settle} "
                  f"md5 {self.goal_frame_md5}", flush=True)
        return frames[kept_index]

    def set_goal_render_warmup(self, n: int) -> None:
        """The goal capture count, stated **after** the boot.

        A driver that reads its config after the simulator is up cannot pass the count
        to the constructor. It is refused once a goal exists: a goal captured under one
        rule and relabelled with another is exactly the silent wrong number this
        repository refuses.
        """
        if self._goal_frame is not None:
            raise RuntimeError(
                f"{self.name}: the goal is already captured; its warm-up cannot change")
        self.goal_render_warmup = _check_warmup(n)

    def set_goal_capture_keep(self, keep: str) -> None:
        """Which capture is kept, stated after the boot - same rule as above."""
        if self._goal_frame is not None:
            raise RuntimeError(
                f"{self.name}: the goal is already captured; what is kept cannot change")
        self.goal_capture_keep = _check_keep(keep)

    def inject_goal(self, rgb: np.ndarray, depth: np.ndarray, source: str) -> None:
        """Use a **saved** frame as the goal instead of rendering one.

        For diagnostics that separate "the goal pixels" from "the process that produced
        them". :meth:`goal_capture_record` carries ``injected_from``, so an injected
        goal can never be read as a rendered one. Refused once a goal exists.
        """
        if self._goal_frame is not None:
            raise RuntimeError(f"{self.name}: a goal frame is already in hand")
        rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        depth = np.ascontiguousarray(np.asarray(depth, dtype=np.float32))
        self._goal_frame = (rgb, depth)
        self.goal_frame_md5 = hashlib.md5(rgb.tobytes()).hexdigest()
        self.goal_settle = {"n_captures": 0, "keep": self.goal_capture_keep,
                            "injected_from": str(source),
                            "rgb_mean": [round(float(np.asarray(rgb, np.float64).mean()), 6)],
                            "diff_prev": [None], "rgb_md5": [self.goal_frame_md5],
                            "depth_md5": [hashlib.md5(depth.tobytes()).hexdigest()]}
        if self._verbose:
            print(f"[isaac_scene] goal INJECTED from {source}: md5 {self.goal_frame_md5}",
                  flush=True)

    def goal_capture_record(self) -> dict:
        """What a record should say about this scene's goal frame.

        One home for the four fields. The series is ``None`` before a goal has been
        captured.
        """
        return {"goal_render_warmup": int(self.goal_render_warmup),
                "goal_capture_keep": self.goal_capture_keep,
                "goal_settle_series": self.goal_settle,
                "goal_frame_md5": self.goal_frame_md5}

    # ------------------------------------------------- the context manager
    def __enter__(self) -> IsaacScene:
        """``with IsaacScene(...) as scene:`` — the only supported way to hold one.

        The ``except BaseException: traceback.print_exc(); flush; raise`` /
        ``finally: scene.close()`` pattern used to live in each entry point, written
        out by hand, and one scratch script did what every scratch script will: it
        wrapped ``scene.close()`` in a bare ``try/finally``, died, and **exited 0 with
        nothing on stderr** — a whole station night spent on a stage that had already
        failed. The pattern belongs to the object that owns the simulator, so that
        forgetting it is not something a caller can do.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Say why it died, flush, close the simulator, and never swallow the exception.

        The order is the whole point. ``SimulationApp.close()`` tears carb down and can
        take a pending traceback with it, so the traceback is printed **before**
        :meth:`close` runs; the flushes are why the two streams interleave in the right
        order in a log file. ``False`` is returned explicitly: the exception propagates
        and the process exits non-zero, so a runbook's ``set -e`` can see it.
        """
        if exc_type is not None:
            traceback.print_exception(exc_type, exc, tb)
        sys.stdout.flush()
        sys.stderr.flush()
        self.close()
        return False

    def close(self) -> None:
        """Close the simulator **once**; a second call is a no-op.

        :meth:`__exit__` calls this, so the interpreter never tears carb down through
        ``atexit`` under a live app - which is what segfaulted at the end of every
        stage and made the exit code unreadable.
        """
        # Flush FIRST: an explicit close can eat a pending traceback, and a station
        # stage that dies silently costs a night.
        sys.stdout.flush()
        sys.stderr.flush()
        sim, self.sim = self.sim, None       # dropped first: close() runs at most once
        if sim is not None:
            sim.close()

    def gt_correspondence(self, pts_goal_px, cur_pose: Pose,
                          goal_pose: Pose | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Where each goal pixel lands in ``cur_pose``'s image, occlusion-tested.

        Uses the cached goal depth as view A and the depth map of the **last
        rendered frame** as view B - the :class:`vsbench.vs.scene.Scene` protocol carries
        no current depth, and without one the ``gt_*`` columns over-count on
        cluttered scenes.
        """
        goal = self.goal_pose if goal_pose is None else goal_pose
        if self._last is None:
            raise RuntimeError("gt_correspondence needs a rendered current frame first")
        last_pose, _rgb_c, depth_c = self._last
        if not (np.allclose(last_pose.position, cur_pose.position)
                and np.allclose(last_pose.rotation, cur_pose.rotation)):
            raise ValueError(
                "gt_correspondence was asked about a pose that is not the last rendered "
                "frame, so no depth map can occlusion-test it; render first")
        _rgb_g, depth_g = self.render_goal()
        K = self.cam.camera_info()
        K = np.array([[K["f_x"], 0.0, K["c_x"]], [0.0, K["f_y"], K["c_y"]], [0, 0, 1.0]])
        uv, valid, _z = gt_correspondence(pts_goal_px, depth_g, K, pose_to_T(goal),
                                          pose_to_T(cur_pose), depth_b=depth_c)
        return uv, valid

    # -------------------------------------------------------- Isaac plumbing
    def _boot(self) -> None:                              # pragma: no cover - Isaac
        cam = self.cam
        from isaacsim import SimulationApp
        self.sim = SimulationApp({"headless": True, "width": cam.width,
                                  "height": cam.height, "renderer": "RaytracedLighting"})
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, UsdGeom, UsdLux
        self._Gf, self._UsdGeom, self._rep = Gf, UsdGeom, rep
        self._add_reference_to_stage = add_reference_to_stage
        # the settle in vsbench.vs.bake steps this; sim.update() alone cannot
        self.simulation_context = SimulationContext(stage_units_in_meters=1.0)
        self.stage = omni.usd.get_context().get_stage()
        self.assets_root = get_assets_root_path()

        # lighting, verbatim from the original free-flying scene
        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.CreateIntensityAttr(3000.0)
        dome.CreateColorAttr(Gf.Vec3f(0.20, 0.20, 0.20))
        sun = UsdLux.DistantLight.Define(self.stage, "/World/SunLight")
        sun.CreateIntensityAttr(3500.0)
        sun.CreateAngleAttr(0.53)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 0.0, 30.0))

        # a BARE camera prim with camera.yaml's optics - never a camera asset.
        # The clipping range is load-bearing: USD's default near plane is 1 m in
        # scene units and every scene here is flown at 0.43-1.29 m.
        self.CAMERA_PATH = "/World/RenderCamera"
        camera = UsdGeom.Camera.Define(self.stage, self.CAMERA_PATH)
        camera.CreateFocalLengthAttr(cam.focal_length)
        camera.CreateHorizontalApertureAttr(cam.horiz_aperture)
        camera.CreateVerticalApertureAttr(cam.vert_aperture)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1000.0))
        xf = UsdGeom.Xformable(camera.GetPrim())
        self._cam_translate = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        self._cam_orient = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

        self.render_product = rep.create.render_product(self.CAMERA_PATH,
                                                        (cam.width, cam.height))
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._seg_annot = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
        for a in (self._rgb_annot, self._depth_annot, self._seg_annot):
            a.attach([self.render_product])
        self._measure_r_internal()
        self.set_camera_pose_optical(np.zeros(3), R_OPT2USD_BASE.copy())
        if self._verbose:
            print("[isaac_scene] Isaac ready.", flush=True)

    def _measure_r_internal(self) -> None:                # pragma: no cover - Isaac
        """A bare camera prim must measure the identity; refuse loudly if not."""
        self._cam_orient.Set(self._Gf.Quatd(1.0, self._Gf.Vec3d(0.0, 0.0, 0.0)))
        self.sim.update()
        xform = self._UsdGeom.Xformable(
            self.stage.GetPrimAtPath(self.CAMERA_PATH)).ComputeLocalToWorldTransform(0.0)
        R = np.array([[xform[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
        dev = float(np.linalg.norm(R - np.eye(3)))
        if dev > R_INTERNAL_TOL:
            raise RuntimeError(
                f"camera prim rotation is not the identity (||R - I|| = {dev:.2e}); the "
                f"non-identity compensation order is unverified - re-derive it before "
                f"servoing a possibly rotated command")

    def set_camera_pose_optical(self, cam_pos, R_opt) -> None:
        self._pose = Pose(position=cam_pos, rotation=R_opt)
        self._reortho_counter = 0
        self._write_pose_to_usd()

    def _write_pose_to_usd(self) -> None:                 # pragma: no cover - Isaac
        if not hasattr(self, "_cam_translate"):
            return
        Gf = self._Gf
        q = optical_R_to_usd_quat(self._pose.rotation, self.R_OPT2USD)
        p = self._pose.position
        self._cam_translate.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        self._cam_orient.Set(Gf.Quatd(float(q[0]),
                                      Gf.Vec3d(float(q[1]), float(q[2]), float(q[3]))))

    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - Isaac
        """Render the current pose -> ``(rgb uint8 HxWx3, depth float32 HxW metres)``.

        No segmentation: a scene-level dataset has no single target prim, and
        :func:`seg_to_target_mask` refuses that case by design.

        **An empty annotator is "not ready", not a frame**
        (:data:`CAPTURE_RETRIES_DEFAULT`). The first read after a boot can come back with
        shape ``(0,)`` - a six-scene runbook hit it ten seconds into the drill scene,
        having launched one Isaac process straight after the previous one exited - so an
        empty rgb or depth buys one more orchestrator step and another read, up to
        :attr:`capture_retries` times, and then raises naming the annotator. How many it
        spent is on :attr:`last_capture_retries`.

        **Behaviour on a non-empty capture**: ``steps_per_capture`` orchestrator steps,
        two reads in a fixed order, the same conversions - the loop below simply does not
        go round twice.
        """
        self.last_capture_retries = 0
        for attempt in range(int(self.capture_retries) + 1):
            for _ in range(self.steps_per_capture):
                self._rep.orchestrator.step(rt_subframes=self.rt_subframes)
            rgb = np.asarray(self._rgb_annot.get_data())
            depth = np.asarray(self._depth_annot.get_data(), dtype=np.float32)
            empty = [name for name, arr in (("rgb", rgb),
                                            ("distance_to_image_plane", depth))
                     if np.asarray(arr).size == 0]
            if not empty:
                break
            self.last_capture_retries = attempt + 1
            if attempt >= int(self.capture_retries):
                raise RuntimeError(
                    f"{self.name}: the {' and '.join(empty)} annotator(s) came back "
                    f"EMPTY after {self.capture_retries} extra orchestrator step(s) "
                    f"(shapes rgb {np.asarray(rgb).shape}, depth "
                    f"{np.asarray(depth).shape}). The renderer never produced a frame "
                    f"for this pose; a capture that returns nothing is not a picture, "
                    f"and the settle series would subtract it from the next one. Raise "
                    f"`capture.retries`, or launch the process with the simulator given "
                    f"time to come up.")
            if self._verbose:
                print(f"[isaac_scene] {self.name}: {', '.join(empty)} annotator empty, "
                      f"retry {attempt + 1}/{self.capture_retries}", flush=True)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        return np.ascontiguousarray(rgb.astype(np.uint8)), np.ascontiguousarray(depth)

    # ------------------------------------------------------ stage authoring
    def build_poster(self, width_m: float, height_m: float, texture: str,
                     prim_path: str = "/World/Poster",
                     uv_tiles: float = 1.0) -> dict:      # pragma: no cover - Isaac
        """A thin textured quad in the XY plane at the world origin.

        ``uv_tiles > 1`` repeats the texture per axis (wrap "repeat") - how a
        large ground is textured without stretching one image across it; the
        default authors no wrap attributes, so eval posters are unchanged.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        if not Path(texture).exists():
            raise FileNotFoundError(
                f"texture not found: {texture} - it is deliberately not in git "
                f"(binary, third-party); stage it on the volume")
        if self.stage.GetPrimAtPath(prim_path).IsValid():
            self.stage.RemovePrim(prim_path)
            self.sim.update()
        hx, hy = float(width_m) / 2.0, float(height_m) / 2.0
        mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
        mesh.CreatePointsAttr([Gf.Vec3f(-hx, -hy, 0.0), Gf.Vec3f(hx, -hy, 0.0),
                               Gf.Vec3f(hx, hy, 0.0), Gf.Vec3f(-hx, hy, 0.0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation("faceVarying")
        mesh.CreateExtentAttr([Gf.Vec3f(-hx, -hy, 0.0), Gf.Vec3f(hx, hy, 0.0)])
        t = float(uv_tiles)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
            [Gf.Vec2f(0, 0), Gf.Vec2f(t, 0), Gf.Vec2f(t, t), Gf.Vec2f(0, t)])
        mat = UsdShade.Material.Define(self.stage, f"{prim_path}/Material")
        shader = UsdShade.Shader.Define(self.stage, f"{prim_path}/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(self.stage, f"{prim_path}/Material/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(self.stage, f"{prim_path}/Material/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        if t != 1.0:
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh).Bind(mat)
        self.sim.update()
        return {"width_m": float(width_m), "height_m": float(height_m),
                "texture": str(texture), "prim_path": prim_path}

    def build_backdrop(self, size_m: float = 4.0, gray: float = 0.5,
                       prim_path: str = "/World/Backdrop",
                       z_m: float = -0.02) -> dict:        # pragma: no cover - Isaac
        """A featureless mid-gray quad behind a poster: it replaces the void at
        wide starts and adds no matchable structure of its own."""
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        if self.stage.GetPrimAtPath(prim_path).IsValid():
            self.stage.RemovePrim(prim_path)
            self.sim.update()
        half = float(size_m) / 2.0
        mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
        mesh.CreatePointsAttr([Gf.Vec3f(-half, -half, float(z_m)),
                               Gf.Vec3f(half, -half, float(z_m)),
                               Gf.Vec3f(half, half, float(z_m)),
                               Gf.Vec3f(-half, half, float(z_m))])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mat = UsdShade.Material.Define(self.stage, f"{prim_path}/Mat")
        shader = UsdShade.Shader.Define(self.stage, f"{prim_path}/Mat/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        g = float(gray)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(g, g, g))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        self.sim.update()
        return {"prim_path": prim_path, "size_m": float(size_m), "gray": g}

    def place_objects(self, items: list, library: dict, layout: dict | None = None,
                      settle: dict | None = None) -> dict:  # pragma: no cover - Isaac
        """Reference, scale, seat and then freeze every object.

        Each object is scaled per category (metric scans - any id containing
        ``/`` - keep native scale), re-centred at the origin, lifted so its bbox
        bottom touches ``z = 0`` and offset by ``(dx, dy)``. Then, in order of
        precedence: a frozen ``layout`` is authored (after an id-drift refusal),
        else a ``settle`` config gravity-settles once
        (:func:`vsbench.vs.bake.settle_objects` - only ``bake`` passes it),
        else the analytic poses stand. The bounding box is measured **after** the
        final poses, because the goal frames what is rendered.
        Returns ``{objects, bbox_min, bbox_max, layout, handles}`` in world metres;
        ``handles`` are the ``(translate_op, orient_op, inner_prim)`` triples the
        scale jitter and the settle re-author, so a caller never has to re-find them.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        placed, handles = [], []
        for i, item in enumerate(items):
            iid = str(item["instance_id"])
            if iid not in library:
                raise KeyError(f"asset {iid!r} is not in the library")
            path = f"/World/Obj_{i}"
            if self.stage.GetPrimAtPath(path).IsValid():
                self.stage.RemovePrim(path)
                self.sim.update()
            xf = UsdGeom.Xform.Define(self.stage, path)
            inner_path = f"{path}/asset"
            usd_path, nucleus = resolve_usd_path(library[iid])
            if nucleus:
                if not self.assets_root:
                    raise RuntimeError(
                        f"{iid} is a Nucleus asset but get_assets_root_path() resolved "
                        f"None - filter it out of the pools or fix the asset root")
                usd_path = self.assets_root + usd_path
            elif not Path(usd_path).exists():
                raise FileNotFoundError(
                    f"{iid}: {usd_path} does not exist - set ${ASSET_ROOT_ENV} to the "
                    f"checkout that holds experiments/ingest/usd and assets/gso_usd "
                    f"(this repository holds the library JSON and no USD). An absent "
                    f"reference makes an empty prim, a garbage bounding box and a wrong "
                    f"goal distance, in silence.")
            self._add_reference_to_stage(usd_path=usd_path, prim_path=inner_path)
            self.sim.update()
            inner = self.stage.GetPrimAtPath(inner_path)
            for p in Usd.PrimRange(inner):      # static scene: strip physics
                for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI):
                    if p.HasAPI(api):
                        p.RemoveAPI(api)
            xf.ClearXformOpOrder()
            ops = (xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble),
                   xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat),
                   xf.AddScaleOp())
            placement = None
            if uses_placement(item):
                # the instance suite's path (2026-09-23): the library's upright canonical
                # pose, the lay-down, the in-plane yaw (`long_axis_x` resolved here) and a
                # stated size - vsbench.vs.scenes.item_quat. Nothing below it runs for an item
                # without those fields, so every older scene is placed exactly as before.
                base_quat, scale, placement = self._placement_pose(ops, inner, item,
                                                                   library[iid], iid)
            else:
                # `pre_rot_x_deg` stands a model up BEFORE placement (the google_16k
                # pitcher scan is natively LYING), so the analytic pose, the settle drop
                # and the baked layout all start upright.
                pre = item.get("pre_rot_x_deg")
                base_quat = (axis_angle_quat([1.0, 0.0, 0.0], np.deg2rad(float(pre)))
                             if pre else np.array([1.0, 0.0, 0.0, 0.0]))
                # `yaw_deg` turns the (laid-down) object about the vertical - the shoe2
                # scene lies its shoe along the image's long axis so the goal camera
                # comes closer
                yaw = item.get("yaw_deg")
                if yaw:
                    base_quat = _quat_mul(axis_angle_quat([0.0, 0.0, 1.0],
                                                          np.deg2rad(float(yaw))), base_quat)
                ops[0].Set(Gf.Vec3d(0, 0, 0))
                self._set_quat(ops[1], base_quat)
                ops[2].Set(Gf.Vec3f(1, 1, 1))
                self.sim.update()
                if "/" in iid:
                    # metric scans (YCB google_16k) are placed at native scale; the
                    # per-category size targets are for the curated library only.
                    # A per-item `scale` stays available.
                    scale = float(item.get("scale", 1.0))
                else:
                    cat = item.get("category")
                    if not cat:
                        raise KeyError(
                            f"{iid}: a curated instance needs its `category` to be scaled "
                            f"(assets.target_for); every scene spec and every plan layout "
                            f"carries one, so an absent one is a malformed item")
                    scale = self._measure_scale(ops, inner, str(cat), base_quat)
            ext = self._place_pose(ops, inner, base_quat, scale)
            cur = ops[0].Get()
            ops[0].Set(Gf.Vec3d(cur[0] + float(item.get("dx", 0.0)),
                                cur[1] + float(item.get("dy", 0.0)), cur[2] + ext[2] / 2.0))
            self.sim.update()
            entry = {"instance_id": iid, "prim_path": path, "scale": scale}
            if placement is not None:
                entry["placement"] = placement
            if item.get("settle") is False:
                entry["settle"] = False
            placed.append(entry)
            handles.append((ops[0], ops[1], inner))
        if not placed:
            raise ValueError("scene declares no objects")
        if layout is not None:
            check_layout(layout, [p["instance_id"] for p in placed])
            for (t_op, o_op, _inner), rec in zip(handles, layout["objects"],
                                                 strict=True):
                t, q = rec["pos"], rec["quat_wxyz"]
                t_op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
                o_op.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]),
                                                        float(q[3]))))
            self.sim.update()
        elif settle is not None:
            from vsbench.vs.bake import settle_objects
            n_escaped = settle_objects(self, placed, handles, settle)
            if n_escaped:
                print(f"[isaac_scene] settle: {n_escaped}/{len(placed)} object(s) "
                      f"escaped bounds - kept the analytic pose", flush=True)
        mins, maxs = [], []
        for entry, (t_op, o_op, inner) in zip(placed, handles, strict=True):
            mn, mx = self._compute_world_bbox(inner)
            mins.append(np.asarray(mn, float))
            maxs.append(np.asarray(mx, float))
            t, q = t_op.Get(), o_op.Get()
            entry["pos"] = [float(t[j]) for j in range(3)]
            entry["quat_wxyz"] = [float(q.GetReal())] + [float(q.GetImaginary()[j])
                                                         for j in range(3)]
        return {"objects": placed, "handles": handles,
                "bbox_min": [float(v) for v in np.min(np.stack(mins), axis=0)],
                "bbox_max": [float(v) for v in np.max(np.stack(maxs), axis=0)],
                "layout": {"objects": [{k: e[k] for k in ("instance_id", "prim_path",
                                                          "pos", "quat_wxyz")}
                                       for e in placed]}}

    def _compute_world_bbox(self, prim):                  # pragma: no cover - Isaac
        UsdGeom = self._UsdGeom
        cache = UsdGeom.BBoxCache(0.0, includedPurposes=[
            UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide], useExtentsHint=False)
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return (np.array([r.GetMin()[i] for i in range(3)]),
                np.array([r.GetMax()[i] for i in range(3)]))

    def _set_quat(self, o_op, q) -> None:                 # pragma: no cover - Isaac
        Gf = self._Gf
        o_op.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))

    def _measure_scale(self, ops, inner, category: str,
                       base_quat) -> float:               # pragma: no cover - Isaac
        Gf = self._Gf
        ops[0].Set(Gf.Vec3d(0, 0, 0))
        self._set_quat(ops[1], base_quat)
        ops[2].Set(Gf.Vec3f(1, 1, 1))
        self.sim.update()
        mn, mx = self._compute_world_bbox(inner)
        mode, target_m = target_for(category)
        ext = mx - mn
        ref = float(ext[2]) if mode == "height" else float(max(ext))
        return target_m / ref if ref > 1e-9 else 1.0

    def _extent_at(self, ops, inner, quat, scale: float) -> np.ndarray:  # pragma: no cover
        """The object's world-aligned extent at the origin under ``quat`` and ``scale``."""
        Gf = self._Gf
        ops[0].Set(Gf.Vec3d(0, 0, 0))
        self._set_quat(ops[1], quat)
        ops[2].Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
        self.sim.update()
        mn, mx = self._compute_world_bbox(inner)
        return np.asarray(mx - mn, float)

    def _placement_pose(self, ops, inner, item: dict, entry: dict,
                        iid: str) -> tuple:                  # pragma: no cover - Isaac
        """``(quat, scale, record)`` for an item on the placement path.

        The orientation is :func:`vsbench.vs.scenes.item_quat` (the library's upright canonical
        pose when ``canonical: library``, then the lay-down, then the yaw); ``yaw_deg:
        long_axis_x`` is resolved from the footprint measured under the orientation
        without it (:func:`vsbench.vs.scenes.long_axis_yaw`). ``size_m`` makes the longest
        extent under the final orientation exactly that many metres, for ANY id - a metric
        scan included; without it the size is the one :meth:`place_objects` always used.
        ``record`` says what was resolved, so the scene can state it.
        """
        yaw = item.get("yaw_deg")
        if str(yaw) == LONG_AXIS_X:
            ext0 = self._extent_at(ops, inner, item_quat(item, entry, yaw_deg=0.0), 1.0)
            yaw_res = long_axis_yaw(ext0)
        else:
            yaw_res = float(yaw or 0.0)
        quat = item_quat(item, entry, yaw_deg=yaw_res)
        ext = self._extent_at(ops, inner, quat, 1.0)
        if item.get("size_m") is not None:
            ref = float(np.max(ext))
            scale = float(item["size_m"]) / ref if ref > 1e-9 else 1.0
        elif "/" in iid:
            scale = float(item.get("scale", 1.0))
        else:
            cat = item.get("category")
            if not cat:
                raise KeyError(f"{iid}: a curated instance needs its `category` or a "
                               "`size_m` to be scaled")
            scale = self._measure_scale(ops, inner, str(cat), quat)
        record = {"canonical": item.get("canonical"),
                  "upright_yaw_add_deg": float(item.get("upright_yaw_add_deg") or 0.0),
                  "pre_rot_x_deg": float(item.get("pre_rot_x_deg") or 0.0),
                  "yaw_deg": yaw_res, "yaw_rule": str(yaw) if yaw is not None else None,
                  "size_m": (None if item.get("size_m") is None else float(item["size_m"])),
                  "native_extent_m": [float(e) for e in ext]}
        return quat, scale, record

    def _place_pose(self, ops, inner, quat, scale: float) -> list:  # pragma: no cover
        Gf = self._Gf
        ops[0].Set(Gf.Vec3d(0, 0, 0))
        self._set_quat(ops[1], quat)
        ops[2].Set(Gf.Vec3f(scale, scale, scale))
        self.sim.update()
        mn, mx = self._compute_world_bbox(inner)
        centre = (mn + mx) * 0.5
        ops[0].Set(Gf.Vec3d(float(-centre[0]), float(-centre[1]), float(-centre[2])))
        self.sim.update()
        mn_f, mx_f = self._compute_world_bbox(inner)
        return [float(e) for e in (mx_f - mn_f)]

    # --------------------------------------------------- the per-scene steps
    def clear_scene_prims(self) -> None:                  # pragma: no cover - Isaac
        """Remove every prim a build owns: poster, ground, collider, backdrop, objects.

        Without this a planar scene's ``/World/Poster`` (coplanar with the ground
        at z = 0) survives into the next object scene, and ``/World/Obj_k`` for
        ``k >= n`` survives from a larger scene into a smaller one. Call it before
        building a second scene in the same process.
        """
        doomed = ["/World/Poster", "/World/Ground", "/World/GroundCollider",
                  "/World/Backdrop"]
        world = self.stage.GetPrimAtPath("/World")
        if world.IsValid():
            doomed += [str(c.GetPath()) for c in world.GetChildren()
                       if c.GetName().startswith("Obj_")]
        removed = False
        for path in doomed:
            if self.stage.GetPrimAtPath(path).IsValid():
                self.stage.RemovePrim(path)
                removed = True
        if removed:
            self.sim.update()

    def set_lighting(self, randomization: dict) -> None:   # pragma: no cover - Isaac
        """Scale the boot-time dome and sun and set a colour temperature.

        ``randomization`` carries ``light_intensity_scale`` (default 1.0) and
        ``light_temperature_k`` (default 4500). Use :meth:`lighting_snapshot` /
        :meth:`restore_lighting` to undo it.
        """
        from pxr import UsdLux
        scale = float(randomization.get("light_intensity_scale", 1.0))
        temp = float(randomization.get("light_temperature_k", 4500.0))
        for path, cls, base in (("/World/DomeLight", UsdLux.DomeLight, 3000.0),
                                ("/World/SunLight", UsdLux.DistantLight, 3500.0)):
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            light = cls(prim)
            light.GetIntensityAttr().Set(base * scale)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(temp)
        self.sim.update()

    # ------------------------------------------------- the scene-editing verbs
    # Perturbation and diagnostic verbs: lighting snapshot/restore, semantic labels,
    # an occluder box, per-object reads and moves. None of them is on the flight path.

    def lighting_snapshot(self) -> dict:                  # pragma: no cover - Isaac
        """The dome and sun light attributes as they stand, so they can be put back.

        :meth:`set_lighting` turns colour temperature **on** and there is no intensity
        scale that turns it off again, so "restore the boot lighting" cannot be
        expressed as another call to it. A perturbation that cannot be undone is not a
        perturbation; it is a new scene from that point on.
        """
        from pxr import UsdLux
        snap = {}
        for path, cls in (("/World/DomeLight", UsdLux.DomeLight),
                          ("/World/SunLight", UsdLux.DistantLight)):
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            light = cls(prim)
            enable = prim.GetAttribute("inputs:enableColorTemperature")
            temp = prim.GetAttribute("inputs:colorTemperature")
            snap[path] = {
                "intensity": float(light.GetIntensityAttr().Get()),
                "enable_color_temperature": (bool(enable.Get())
                                             if enable and enable.HasAuthoredValue()
                                             else None),
                "color_temperature": (float(temp.Get())
                                      if temp and temp.HasAuthoredValue() else None),
            }
        return snap

    def restore_lighting(self, snapshot: dict) -> None:   # pragma: no cover - Isaac
        """Put a :meth:`lighting_snapshot` back on the stage."""
        from pxr import UsdLux
        for path, values in (snapshot or {}).items():
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            cls = UsdLux.DomeLight if path.endswith("DomeLight") else UsdLux.DistantLight
            light = cls(prim)
            light.GetIntensityAttr().Set(float(values["intensity"]))
            if values["enable_color_temperature"] is None:
                # it was never authored: author `false`, which is USD's own default and
                # what a bare boot light behaves as
                light.CreateEnableColorTemperatureAttr(False)
            else:
                light.CreateEnableColorTemperatureAttr(bool(
                    values["enable_color_temperature"]))
            if values["color_temperature"] is not None:
                light.CreateColorTemperatureAttr(float(values["color_temperature"]))
        self.sim.update()

    def label_prims(self, labels: dict) -> dict:           # pragma: no cover - Isaac
        """Give each prim a **semantic label**, so the instance mask can name it.

        ``labels`` is ``{prim_path: label}``; the usual value is the prim path itself,
        because that is what :func:`seg_to_target_mask` matches on. Applied to the
        object's own ``Xform`` (``/World/Obj_k``), not to the referenced
        ``/World/Obj_k/asset`` beneath it — Replicator propagates a parent's semantics
        down the subtree, and the parent is the thing the layout moves.

        WHY THIS EXISTS AT ALL. ``instance_segmentation_fast`` labels only prims that
        **carry a semantic label**, and nothing on the scene-building path applies one:
        :meth:`build_eval_scene`, :meth:`place_objects`, :meth:`build_poster` and
        :meth:`put_box` all create bare geometry. Call this after a build when you need
        per-object masks from :meth:`capture_seg`.

        The modern calls are tried **first** (``add_update_semantics``, then Isaac 5's
        ``add_labels``), with the old ``pxr.Semantics.SemanticsAPI`` as the last
        fallback: Isaac Sim 5.1 is what the station runs and ``pxr.Semantics`` is the
        deprecated half of that pair. All three are tried because the call moved twice
        across Isaac versions; the failure message names which of the three exist on
        the build at hand. A prim that could not be labelled **raises** — a warning
        here would mean a mask quietly missing an object.

        NOT CHOSEN: ``instance_id_segmentation_fast``, whose id map is prim paths and
        needs no semantics at all — its ids are per-prim rather than per-semantic-parent,
        so a referenced asset with many meshes appears as many ids that then have to be
        folded back to the object by string surgery.

        Returns ``{prim_path: label}`` for the record; raises :class:`ValueError` for a
        path that is not on the stage.
        """
        applied = {}
        for prim_path, label in labels.items():
            prim = self.stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                raise ValueError(f"cannot label {prim_path}: not a valid prim")
            self._apply_semantics(prim, str(label))
            applied[str(prim_path)] = str(label)
        self.sim.update()
        return applied

    def semantic_labels(self, prim_paths) -> dict:        # pragma: no cover - Isaac
        """``{prim_path: label or None}`` — what the STAGE says, read back.

        The proof that :meth:`label_prims` took, asked of the stage rather than of a
        picture: a prim either carries the label or it does not, whatever the camera can
        see. Without it, "the segmentation lists nothing" cannot be told from "nothing is
        in view".

        Read through the same two APIs the write goes through, newest first, so a build
        that took the write can answer the read.
        """
        out: dict[str, str | None] = {}
        for prim_path in prim_paths:
            prim = self.stage.GetPrimAtPath(str(prim_path))
            out[str(prim_path)] = self._read_semantics(prim) if prim.IsValid() else None
        return out

    @staticmethod
    def _read_semantics(prim) -> str | None:               # pragma: no cover - Isaac
        """The ``class`` label on one prim, or ``None``."""
        with contextlib.suppress(Exception):
            from isaacsim.core.utils.semantics import get_semantics
            for _key, (kind, value) in (get_semantics(prim) or {}).items():
                if str(kind) == "class":
                    return str(value)
        with contextlib.suppress(Exception):
            from pxr import Semantics
            api = Semantics.SemanticsAPI.Get(prim, "Semantics")
            if api and api.GetSemanticTypeAttr().Get() == "class":
                return str(api.GetSemanticDataAttr().Get())
        # last resort: the raw attributes the schema writes, which both APIs land on
        with contextlib.suppress(Exception):
            attr = prim.GetAttribute("semantic:Semantics:params:semanticData")
            if attr and attr.HasAuthoredValue():
                return str(attr.Get())
        return None

    @staticmethod
    def _apply_semantics(prim, label: str) -> None:        # pragma: no cover - Isaac
        """One prim, one ``class`` label. Raises when **no** API is available.

        Three attempts, because the call moved twice across Isaac versions:
        ``add_update_semantics`` first, then Isaac 5's ``add_labels``, then the
        ``pxr.Semantics`` schema. Every failure is collected and reported together, so
        a log line says which of the three exist on that build.
        """
        errors = []
        try:
            from isaacsim.core.utils.semantics import add_update_semantics
            add_update_semantics(prim, str(label), type_label="class")
            return
        except Exception as e:                             # noqa: BLE001
            errors.append(f"add_update_semantics: {e!r}")
        try:
            from isaacsim.core.utils.semantics import add_labels
            add_labels(prim, labels=[str(label)], instance_name="class")
            return
        except Exception as e:                             # noqa: BLE001
            errors.append(f"add_labels: {e!r}")
        try:
            from pxr import Semantics
            api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            api.CreateSemanticTypeAttr().Set("class")
            api.CreateSemanticDataAttr().Set(str(label))
            return
        except Exception as e:                             # noqa: BLE001
            errors.append(f"pxr.Semantics: {e!r}")
        raise RuntimeError(
            f"could not set semantics on {prim.GetPath()} (label {label!r}): "
            + "; ".join(errors)
            + ". Without a semantic label `instance_segmentation_fast` does not list the "
              "prim at all, and the dump's mask would be silently missing an object.")

    def set_prim_visible(self, prim_path: str, visible: bool) -> None:  # pragma: no cover
        """Hide or show one prim (USD visibility) without touching its transform - the
        swap-instance perturbation hides the goal object and shows it again."""
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"{prim_path} is not a valid prim")
        img = UsdGeom.Imageable(prim)
        (img.MakeVisible if visible else img.MakeInvisible)()
        self.sim.update()

    def remove_prims(self, prim_paths) -> None:           # pragma: no cover - Isaac
        for path in list(prim_paths):
            self.remove_prim(path)

    def place_extra_objects(self, items: list, library: dict | None = None,
                            prefix: str = "/World/Distractor") -> list[str]:  # pragma: no cover
        """Put extra objects into a BUILT scene without touching its own (P5 distractors).

        ``items``: ``[{instance_id, x, y, yaw_deg, scale?}]`` - world (x, y) on the floor
        and a yaw about z; each is seated on ``z = 0`` the way :meth:`place_objects` seats
        the scene's objects (native scale for metric scans, the per-category size for the
        curated library). Prims go to ``<prefix>_<k>`` - never ``/World/Obj_k``, which the
        frozen layout owns. Returns the prim paths, for :meth:`remove_prims`.

        An item with the placement fields (``canonical``, ``upright_yaw_add_deg``,
        ``size_m``, ``yaw_deg: long_axis_x`` - :func:`vsbench.vs.scenes.uses_placement`) is
        oriented and sized exactly as :meth:`place_objects` does it, and what was resolved
        is kept in :attr:`extra_placements` under its prim path; ``(x, y)`` is where its
        bounding-box centre goes, as for every item.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        from vsbench.vs.assets import load_library, merge_ycb_library

        if library is None:
            library = load_library()
            merge_ycb_library(library, [it["instance_id"] for it in items])
        paths = []
        for k, item in enumerate(items):
            iid = str(item["instance_id"])
            if iid not in library:
                raise KeyError(f"asset {iid!r} is not in the library")
            path = f"{prefix}_{k}"
            self.remove_prim(path)
            xf = UsdGeom.Xform.Define(self.stage, path)
            inner_path = f"{path}/asset"
            usd_path, nucleus = resolve_usd_path(library[iid])
            if nucleus:
                usd_path = self.assets_root + usd_path
            elif not Path(usd_path).exists():
                raise FileNotFoundError(f"{iid}: {usd_path} does not exist")
            self._add_reference_to_stage(usd_path=usd_path, prim_path=inner_path)
            self.sim.update()
            inner = self.stage.GetPrimAtPath(inner_path)
            for p in Usd.PrimRange(inner):      # static scene: strip physics
                for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI):
                    if p.HasAPI(api):
                        p.RemoveAPI(api)
            xf.ClearXformOpOrder()
            ops = (xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble),
                   xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat),
                   xf.AddScaleOp())
            if uses_placement(item):
                # the instance suite's path: canonical pose, lay-down, yaw, stated size -
                # the same composition place_objects uses for the goal object
                quat, scale, record = self._placement_pose(ops, inner, item, library[iid], iid)
                self.extra_placements[path] = record
            else:
                yaw = np.deg2rad(float(item.get("yaw_deg", 0.0)))
                quat = axis_angle_quat([0.0, 0.0, 1.0], yaw)
                pre = item.get("pre_rot_x_deg")          # lay a canonicalised scan down first
                if pre:
                    quat = _quat_mul(quat, axis_angle_quat([1.0, 0.0, 0.0],
                                                           np.deg2rad(float(pre))))
                if "/" in iid:
                    scale = float(item.get("scale", 1.0))
                else:
                    scale = self._measure_scale(ops, inner, str(item.get("category", "ycb")),
                                                quat)
            ext = self._place_pose(ops, inner, quat, scale)
            cur = ops[0].Get()
            ops[0].Set(Gf.Vec3d(cur[0] + float(item["x"]), cur[1] + float(item["y"]),
                                cur[2] + ext[2] / 2.0))
            self.sim.update()
            paths.append(path)
        return paths

    def remove_prim(self, prim_path: str) -> None:        # pragma: no cover - Isaac
        """Remove one prim if it is there. Idempotent, and narrower than
        :meth:`clear_scene_prims`, which would take the whole scene with it."""
        if self.stage.GetPrimAtPath(prim_path).IsValid():
            self.stage.RemovePrim(prim_path)
            self.sim.update()

    def put_box(self, prim_path: str, centre, size_m: float, rotation,
                gray: float = 0.5) -> dict:               # pragma: no cover - Isaac
        """A neutral matte cube of side ``size_m`` at ``centre``, oriented by ``rotation``.

        An occluder for perturbation studies. ``rotation`` is a 3x3 world rotation -
        pass the camera's optical frame and the cube's faces are square-on to the camera,
        so its image is a square of side ``f * size / distance``. Re-authored in place,
        so placing it twice at two sizes costs one prim, not two.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        self.remove_prim(prim_path)
        cube = UsdGeom.Cube.Define(self.stage, prim_path)
        cube.CreateSizeAttr(1.0)             # unit cube; the scale op carries the metres
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        t_op = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        o_op = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
        s_op = xf.AddScaleOp()
        c = np.asarray(centre, dtype=np.float64).reshape(3)
        t_op.Set(Gf.Vec3d(float(c[0]), float(c[1]), float(c[2])))
        self._set_quat(o_op, mat_to_quat_wxyz(reorthogonalize(rotation)))
        s = float(size_m)
        s_op.Set(Gf.Vec3f(s, s, s))
        mat = UsdShade.Material.Define(self.stage, f"{prim_path}/Mat")
        shader = UsdShade.Shader.Define(self.stage, f"{prim_path}/Mat/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        g = float(gray)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(g, g, g))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        self.sim.update()
        return {"prim_path": prim_path, "size_m": s, "gray": g,
                "centre": [float(v) for v in c]}

    def capture_seg(self):                                # pragma: no cover - Isaac
        """The instance segmentation of the **last rendered frame**.

        It reads the annotator without stepping the orchestrator, so it is the mask of
        the frame :meth:`render` just returned and not of a new one - which matters,
        because RTX is not frame-deterministic and a mask from a second render would be
        measuring a different picture from the one that was saved.

        An **empty** segmentation raises rather than being retried with a step: a step
        would make it the mask of a DIFFERENT frame from the one just returned, which is
        the one thing this method exists to avoid. It re-reads the buffer
        :attr:`capture_retries` times first, because a re-read costs nothing.

        **It is STALE after a render at the goal pose**: :meth:`render` serves the
        cached goal frame there without stepping, so the annotator still holds whatever
        was rendered before. A caller that wants the goal view's mask writes the pose
        and calls :meth:`capture_rgbd` itself.
        """
        for _attempt in range(int(self.capture_retries) + 1):
            seg = self._seg_annot.get_data()
            data = seg["data"] if isinstance(seg, dict) and "data" in seg else seg
            if np.asarray(data).size:
                return seg
        raise RuntimeError(
            f"{self.name}: the instance_segmentation_fast annotator came back EMPTY on "
            f"{self.capture_retries + 1} reads of the last rendered frame. It is NOT "
            f"retried with an orchestrator step, because a step would make this the mask "
            f"of a different frame from the one just returned. Re-render the pose.")

    def object_pose(self, prim_path: str):                # pragma: no cover - Isaac
        """``(pos, quat_wxyz)`` of one placed object's own transform ops.

        The pair :meth:`place_objects` writes into its returned layout, read back off
        the stage - which is what makes "the scene's objects did not move" a check
        rather than a comment.
        """
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"{prim_path} is not a valid prim")
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        t_op = next(o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        o_op = next(o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient)
        t, q = t_op.Get(), o_op.Get()
        return (np.array([float(t[i]) for i in range(3)]),
                np.array([float(q.GetReal())] + [float(q.GetImaginary()[i])
                                                 for i in range(3)]))

    def set_prim_translate(self, prim_path: str, xyz) -> None:  # pragma: no cover
        """Re-author one prim's translate op. The cheap half of moving an object.

        A distractor arrangement per start costs one of these per extra (about fifty
        milliseconds) instead of two :meth:`place_objects` passes (about ten seconds),
        which is what makes a per-start draw affordable at all.
        """
        from pxr import Gf, UsdGeom
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"{prim_path} is not a valid prim")
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        t_op = next(o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        t_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        self.sim.update()

    def object_world_bbox(self, prim_path: str):          # pragma: no cover - Isaac
        """World-space axis-aligned bounds of one placed object's referenced asset."""
        inner = self.stage.GetPrimAtPath(f"{prim_path}/asset")
        if not inner.IsValid():
            raise ValueError(f"{prim_path}/asset is not a valid prim - the scene has no "
                             f"object there")
        return self._compute_world_bbox(inner)

    def apply_scale_jitter(self, placed: list,
                           jitter: float) -> None:        # pragma: no cover - Isaac
        """Post-scale every placed object and re-seat it on the ground plane.
        A no-op at 1.0."""
        if abs(float(jitter) - 1.0) < 1e-6:
            return
        from pxr import Gf, UsdGeom
        for obj in placed:
            prim = self.stage.GetPrimAtPath(obj["prim_path"])
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            t_op = next(o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            s_op = next(o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeScale)
            sc = s_op.Get()
            s_op.Set(Gf.Vec3f(*(float(sc[k]) * float(jitter) for k in range(3))))
            self.sim.update()
            mn, _mx = self._compute_world_bbox(
                self.stage.GetPrimAtPath(obj["prim_path"] + "/asset"))
            t = t_op.Get()
            t_op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]) - float(mn[2])))
        self.sim.update()

    # ------------------------------------------------------------- builders
    def build_eval_scene(self, spec: SceneSpec, library: dict | None = None,
                         allow_settle: bool = False) -> dict:  # pragma: no cover
        """Build one scene of ``configs/scenes.yaml`` and derive its goal pose.

        Returns ``{"distance_m", "info"}`` and leaves the camera **at the goal**.
        A ``multi_object`` scene with a ``settle`` block and no baked layout is
        **refused**: settling is not bit-stable across processes, so a goal
        derived from a fresh settle would not be the goal every stored campaign
        was scored against. Bake it once with :mod:`vsbench.vs.bake`. Building a second
        scene in the same process needs :meth:`clear_scene_prims` first.
        """
        from vsbench.vs.assets import load_library, merge_ycb_library
        from vsbench.vs.scenes import poster_goal_pose, scene_view_goal_pose
        self.name = spec.name
        self._goal_frame = self._last = None
        if spec.kind == "poster":
            self.build_poster(float(spec.raw["width_m"]), float(spec.raw["height_m"]),
                              spec.raw["texture"])
            if spec.raw.get("backdrop"):
                self.build_backdrop(**spec.raw["backdrop"])
            pos, R, d = poster_goal_pose(self.cam, spec)
            info = {}
        else:
            library = dict(library or load_library())
            merge_ycb_library(library, spec.object_ids())
            g = spec.raw.get("ground")
            if g:
                self.build_poster(float(g.get("size_m", 2.4)), float(g.get("size_m", 2.4)),
                                  g["texture"], prim_path="/World/Ground",
                                  uv_tiles=float(g.get("size_m", 2.4))
                                  / float(g.get("tile_period_m", 0.6)))
            layout = load_layout(spec)
            settle = spec.raw.get("settle") if layout is None else None
            if settle and not allow_settle:
                raise RuntimeError(
                    f"{spec.name} declares a settle block and no baked layout - a "
                    f"re-settle may never define a goal; run scripts/bake_layout.py first")
            items = scatter_items(spec, with_scatter=layout is None)
            info = self.place_objects(items, library, layout=layout, settle=settle)
            goal = spec.raw.get("goal") or {}
            pos, R, d = scene_view_goal_pose(
                self.cam, info["bbox_min"], info["bbox_max"],
                frame_fill=goal.get("frame_fill"),
                elevation_deg=float(goal.get("elevation_deg", 90.0)),
                azimuth_deg=float(goal.get("azimuth_deg", -90.0)))
        self._goal = Pose(position=pos, rotation=R)
        self._distance = float(d)
        self.reset(self._goal)
        return {"distance_m": float(d), "info": info}


def scatter_items(spec: SceneSpec, with_scatter: bool = True) -> list[dict]:
    """A scene spec's object list, with its seeded scatter applied.

    The ``(dx, dy)`` offsets come from the spec's ``scatter`` block whenever no
    baked layout is going to overwrite them. A baked scene does not need it;
    :mod:`vsbench.vs.bake`, which creates the bake, does - without it cluttered3's seven
    objects would all be dropped at the origin. Isaac-free, so the placement is
    testable. Each item is merged over the scene's ``placement:`` defaults
    (:func:`vsbench.vs.scenes.placed_item`); a scene without that block gets its items back
    as written.
    """
    items = [placed_item(spec, it) for it in (spec.raw.get("objects") or [])]
    sc = spec.raw.get("scatter")
    if with_scatter and sc:
        offs = sample_scatter(len(items), seed=int(sc["seed"]),
                              half_extent_m=sc.get("half_extent_m"),
                              half_extent_x_m=sc.get("half_extent_x_m"),
                              half_extent_y_m=sc.get("half_extent_y_m"),
                              min_dist_m=float(sc["min_dist_m"]))
        for it, (dx, dy) in zip(items, offs, strict=True):
            it["dx"], it["dy"] = dx, dy
    return items

