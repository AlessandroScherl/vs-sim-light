"""``MethodArm``: a vsbench method flown by the copied benchmark loop.

The one benchmark loop is the copied ``vsbench.vs.bench.runner.run_episode`` (dt 1/60, the
stop recorded and never acted on, the divergence abort, the three success rules, the frozen
record schema, telemetry, films, debug log). A ``vsbench.methods.VSMethod`` meets it here,
through the copied ``Arm`` interface, without a line of the copy edited:

* :class:`MethodArm` subclasses the copied ``Arm`` and overrides :meth:`MethodArm.step`:
  the live frame (a read-only copy) and, if the method asked for it, the live depth go to
  ``method.get_control_rate``; its twist comes back as the copied ``Twist``, its info as the
  ``debug`` dict the loop reads (``feature_error_norm`` from ``info["error_norm"]``,
  ``step_ms``, ``match_ms``, ``n_matches``, ``pts_goal`` / ``pts_cur``, ``depth_values``,
  ``normalised_errors`` and the 1-based ``step`` the scored stop is clocked by), and its
  ``kp_cur`` / ``kp_des`` as the copied ``Matches`` (on the classical arms' one-pixel
  lattice, ``side = input_size = width``), so films, telemetry and the ``cns_plateau`` stop
  work as for a copied arm.
* :class:`MethodFrontEnd` is the arm's ``matcher``: ``set_episode``, ``reset`` (= the
  method's ``configure`` with the scene's intrinsics and the config's ``controller.dt``,
  ``reset`` and ``set_target`` with the goal picture, depth in the form ``wants_depth``
  asks for) and ``record_fields``. :class:`MethodController` is the arm's ``controller``:
  ``dt``, ``gain``, ``reset`` and the step counter; the law is the method's own.
* The config is the vsbench-owned template ``vsbench/world/configs/external_method.yaml``.
  ``matcher.variant: external`` is not in the copied enum, so :func:`validate_external`
  validates it here; the copied ``config.validate`` is not edited. The record says
  ``arm_kind: external`` and the label is ``external:<method>@<params hash>/<stop>``,
  exactly what vs-sim's external-arm hook asks of a plugin arm (``check_baselines
  --arm-plugin``: name ``external:...``, kind ``external``); this module is such a plugin
  too (:data:`ARM_IDS`, :func:`build`, :func:`describe`).

**The stop.** The template's scored stop is ``cns_plateau``, which reads the per-landmark
normalised errors. A method that reports ``kp_cur`` / ``kp_des`` gets it; a method that
does not (seen on step 1 of each episode, or declared up front with ``reports_points =
False``) falls back to ``cap_only``, **loudly**: a :class:`StopFallbackWarning` and a line
on stderr, ``stop_mode: cap_only`` and the label ``.../cap_only`` on the record, and the
record's ``stop_fallback`` block says which rule was configured and why it did not run.

**The goal mask** (``--mask objects|<file.png>``, 2026-10-23). A method whose ``set_target``
takes ``goal_mask`` (``vsbench.methods.takes_goal_mask``) can fly a masked cell: the arm is
built with ``goal_mask_mode`` ``objects`` (the copied runner computes the mask from the goal
depth, as for every ``objmask`` cell, and hands it to ``reset``) or ``file`` (the driver's
``vsbench.world.masks.attach_goal_mask`` hands it the file's mask); the front end passes it
to ``set_target(..., goal_mask=...)`` as a read-only bool array. The label gains the mask's
token before the stop (``external:<method>@<params>/objmask/<stop>`` or
``.../filemask-<md5>/<stop>``). A method without the parameter refuses a mask (a masked
label over an unmasked flight would be a lie) and flies exactly as before without one.

**GT-blind.** The method gets frames, depth (if asked), the reported intrinsics and the
control period - nothing else. A privileged method (vsbench's two truth instruments) is
refused; :func:`assert_method_gt_blind` refuses a method holding a pose or an object with
goal geometry (the copied ``assert_gt_blind`` walks the controller and the stop, which hold
no method; the method sits on the front end, which that tripwire exempts by design, so this
module walks it). ``tests/world/test_method_arm.py`` poisons every ground-truth read the
loop makes (the pose ``render`` returns and ``scene.pose``, ``gt_correspondence``, the
size-matched and depth-true reference poses) and shows the method's commands do not move.

**The run directory.** ``open_method_run`` writes the copied run sidecar (``run.json``,
``resolved_config.yaml``, ``MD5SUMS``) through the copied helpers and returns the copied
``EpisodeSink``; only the validation differs from the copied ``start_run`` (which would
refuse ``matcher.variant: external``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from vsbench.vs.baselines.control import Twist, build_stop_rule
from vsbench.vs.baselines.match import CorrespondenceMatcher, Matches
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.bench.arms import FOREIGN_BLOCKS, Arm, depth_at

#: The template every external method config starts from.
TEMPLATE = Path(__file__).resolve().parent / "configs" / "external_method.yaml"
#: ``matcher.variant`` / ``arm.kind`` / label prefix of a method cell (vs-sim's plugin rule).
EXTERNAL_VARIANT = "external"
EXTERNAL_KIND = "external"
LABEL_PREFIX = "external:"
#: The ``_schema`` stamp of a resolved external config (not the copied ``vssim-config-1``).
SCHEMA = "vsbench-external-method-1"
#: Scored stops that read the method's points (normalised errors / pixel errors).
POINT_STOPS = ("cns_plateau", "artvs_visual90")
FALLBACK_STOP = "cap_only"
#: What the vs-sim plugin pattern calls an arm id (``check_baselines.ARM_ID_PATTERN``).
_ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
#: Attribute names that mark ground truth (the copied runner's ``_GT_MARKERS``).
_GT_MARKERS = ("goal_pose", "gt_correspondence", "get_camera_pose_optical", "render",
               "observe_truth_payload")


class StopFallbackWarning(UserWarning):
    """The configured point-reading stop could not run: the cell flies ``cap_only``."""


class ExternalConfigError(ValueError):
    """An external-method config is missing a required field or holds an invalid one."""


# ------------------------------------------------------------------- the config
def _get(config: dict, dotted: str):
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node or node[part] is None:
            raise ExternalConfigError(f"external method config: required field '{dotted}' is "
                                      "missing - it has no default (external_method.yaml)")
        node = node[part]
    return node


def validate_external(config: dict) -> None:
    """The shim's own validation (the copied ``config.validate`` refuses ``external``).

    The copied five required fields, with the values an external method can carry
    (``matcher.variant`` must be ``external``, ``matcher.num_landmarks`` an int >= 0 where 0
    means "the method's own count"), plus ``arm.kind: external``, a positive
    ``controller.dt``, a named method and the copied episode block rules. The research
    repository's blocks (``refine:``, ``oracle:``, ``head:``) are refused by name, as the
    copied ``build_arm`` refuses them.
    """
    if not isinstance(config, dict):
        raise ExternalConfigError(f"config must be a mapping, got {type(config).__name__}")
    if _get(config, "matcher.variant") != EXTERNAL_VARIANT:
        raise ExternalConfigError(f"matcher.variant must be {EXTERNAL_VARIANT!r} for a method "
                                  f"cell, got {config['matcher']['variant']!r}")
    n = _get(config, "matcher.num_landmarks")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ExternalConfigError(f"matcher.num_landmarks must be an int >= 0, got {n!r}")
    seed = _get(config, "seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ExternalConfigError(f"seed must be an int in [0, 2**32), got {seed!r}")
    rule = _get(config, "stop.rule")
    if rule not in cfgmod.STOP_RULES or rule == "dinobot_err":
        raise ExternalConfigError(f"stop.rule {rule!r} is not a rule a method cell can score "
                                  f"(one of {', '.join(r for r in cfgmod.STOP_RULES if r != 'dinobot_err')})")
    rules = _get(config, "scoring.success_rule")
    if not isinstance(rules, (list, tuple)) or not rules or any(
            r not in cfgmod.SUCCESS_RULES for r in rules):
        raise ExternalConfigError(f"scoring.success_rule must be a non-empty list from "
                                  f"{', '.join(cfgmod.SUCCESS_RULES)}, got {rules!r}")
    if _get(config, "arm.kind") != EXTERNAL_KIND:
        raise ExternalConfigError(f"arm.kind must be {EXTERNAL_KIND!r}: a vsbench method is "
                                  "never a published baseline")
    dt = _get(config, "controller.dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not dt > 0:
        raise ExternalConfigError(f"controller.dt must be a positive number, got {dt!r}")
    name = _get(config, "method.name")
    if not isinstance(name, str) or not name:
        raise ExternalConfigError("method.name must name a registered vsbench method")
    gm = (config.get("matcher") or {}).get("goal_mask")
    if gm is not None and gm not in METHOD_MASK_MODES:
        raise ExternalConfigError(f"matcher.goal_mask must be one of {METHOD_MASK_MODES}, "
                                  f"got {gm!r}")
    if not isinstance((config.get("method") or {}).get("params", {}), dict):
        raise ExternalConfigError("method.params must be a mapping (constructor arguments)")
    for block, what in FOREIGN_BLOCKS.items():
        if config.get(block):
            raise ExternalConfigError(f"config block '{block}' is {what}, which this "
                                      "repository does not carry")
    from vsbench.vs.bench.arms import episode_settings

    episode_settings(config)            # the copied episode-block rules (unknown keys, dt)


def external_config(method_name: str, params: dict | None = None, *, max_iter: int | None = None,
                    telemetry_extended: bool = False, template: Path | None = None) -> dict:
    """The resolved config of one method cell: the template (copied strict loader: no
    duplicate keys), the method block, the driver's overrides; validated and stamped."""
    path = Path(template) if template is not None else TEMPLATE
    cfg = copy.deepcopy(cfgmod.load_document(path))
    cfg["method"] = {"name": str(method_name), "params": dict(params or {})}
    over = {}
    if max_iter is not None:
        cfg.setdefault("episode", {})["max_iter"] = over["max_iter"] = int(max_iter)
    if telemetry_extended:
        cfg.setdefault("episode", {})["telemetry_extended"] = over["telemetry_extended"] = True
    validate_external(cfg)
    cfg["_schema"] = SCHEMA
    cfg["_source"] = (f"{path.name} + method {method_name}"
                      + (f" + bridge {json.dumps(over, sort_keys=True)}" if over else ""))
    return cfg


def params_tag(params: dict) -> str:
    """8 hex digits of the method's constructor arguments: two cells with one label ran one
    setting (the copied ``arm_label`` rule; the full params are on the record)."""
    blob = json.dumps(params or {}, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def method_label(method, stop_rule: str, mask_token: str | None = None) -> str:
    mask = f"{mask_token}/" if mask_token else ""
    return f"{LABEL_PREFIX}{method.name}@{params_tag(_params(method))}/{mask}{stop_rule}"


#: ``goal_mask_mode`` of a method cell: none, the runner's depth outline, or the driver's file
METHOD_MASK_MODES = ("none", "objects", "file")


def _params(method) -> dict:
    p = getattr(method, "params", None)
    return dict(p) if isinstance(p, dict) else {}


def _jsonable_params(method) -> dict:
    return json.loads(json.dumps(_params(method), default=repr))


# ------------------------------------------------------------------- GT-blind
def assert_method_gt_blind(method) -> None:
    """Refuse a privileged method, and one that holds (as a direct attribute) a pose or an
    object exposing goal geometry - the copied tripwire's rule, applied to the method."""
    from vsbench.vs.scene import Pose

    if bool(getattr(method, "privileged", False)):
        raise ValueError(f"{getattr(method, 'name', method)!r} is privileged (it reads the true "
                         "pose through observe_truth): a privileged instrument never flies the "
                         "benchmark loop")
    for key, value in vars(method).items():
        if isinstance(value, Pose) or any(hasattr(value, m) for m in _GT_MARKERS):
            raise AssertionError(f"method attribute {key!r} holds {type(value).__name__}, which "
                                 "carries ground truth - a method cell must be GT-blind")


def _ro_copy(a, dtype=None) -> np.ndarray:
    """A read-only copy: the method can neither alter the loop's frame nor keep a writable
    alias of it."""
    out = np.array(a, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _distance_prior(depth) -> float | None:
    """vsbench's ``distance_prior``: the median positive depth of the goal view."""
    d = np.asarray(depth, dtype=np.float64)
    valid = d[d > 0.0]
    return float(np.median(valid)) if valid.size else None


def _K(intrinsics) -> np.ndarray:
    return np.array([[float(intrinsics.fx), 0.0, float(intrinsics.cx)],
                     [0.0, float(intrinsics.fy), float(intrinsics.cy)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# ------------------------------------------------------------------- the pieces
class MethodFrontEnd(CorrespondenceMatcher):
    """The arm's ``matcher``: owns the method, sets it up at every episode's ``reset``."""

    provides_margin = False
    provides_conf_weight = False
    tie_order = None
    frozen_landmarks = False

    def __init__(self, method, *, intrinsics, dt: float) -> None:
        from vsbench.methods.masking import takes_goal_mask

        self.method = method
        self.name = str(method.name)
        #: the method's ``set_target`` takes ``goal_mask`` (``vsbench.methods.takes_goal_mask``)
        self.honours_goal_mask = takes_goal_mask(method)
        self.intrinsics = intrinsics
        self.dt = float(dt)
        self.episode: tuple[str | None, Any] = (None, None)

    def set_episode(self, scene: str, start_id: Any) -> None:
        self.episode = (None if scene is None else str(scene), start_id)

    def reset(self, goal_rgb, goal_depth=None, goal_mask=None) -> None:
        if goal_mask is not None and not self.honours_goal_mask:
            raise ValueError(f"{self.name}: this method's set_target takes no goal_mask, so it "
                             "cannot fly a masked cell (add goal_mask=None to set_target)")
        m = self.method
        intr = self.intrinsics
        m.configure(_K(intr), int(intr.width), int(intr.height), self.dt)
        m.reset()
        mode = m.wants_depth
        dense = mode == "dense"
        prior = (_distance_prior(goal_depth) if (mode != "none" and goal_depth is not None)
                 else None)
        kw = {}
        if goal_mask is not None:
            gm = np.asarray(goal_mask).astype(bool)
            if gm.shape != (int(intr.height), int(intr.width)):
                raise ValueError(f"{self.name}: goal_mask is {gm.shape}, the picture "
                                 f"{(int(intr.height), int(intr.width))}")
            kw["goal_mask"] = _ro_copy(gm)
        m.set_target(_ro_copy(goal_rgb),
                     desired_depth=_ro_copy(goal_depth, np.float32) if dense else None,
                     distance_prior=prior if mode != "none" else None, **kw)

    def match(self, current_rgb, current_mask=None) -> Matches:
        raise RuntimeError("MethodArm.step drives the method; its front end has no match()")

    def record_fields(self) -> dict:
        m = self.method
        return {"external_method": self.name,
                "external_method_class": f"{type(m).__module__}.{type(m).__qualname__}",
                "external_method_params": _jsonable_params(m),
                "external_method_params_tag": params_tag(_params(m)),
                "external_method_wants_depth": str(m.wants_depth),
                "external_method_privileged": bool(m.privileged)}

    def close(self) -> None:
        close = getattr(self.method, "close", None)
        if callable(close):
            close()


class MethodController:
    """The arm's ``controller``: the period the loop integrates at, the method's gain for
    the record, the 1-based step counter. The law itself is the method's."""

    #: the bench law's knobs the record states for every arm; a method cell runs none of
    #: them (no weighting, no EMA, no minimum-match rule of the bench's)
    solve_unrounded = False
    ema_alpha = 0.0
    min_matches = 0

    def __init__(self, dt: float, gain: float, gain_source: str) -> None:
        self.dt = float(dt)
        self.gain = float(gain)
        self.gain_source = str(gain_source)
        self.weighting = SimpleNamespace(law="external: the method's own")
        self._step_index = 0

    def reset(self) -> None:
        self._step_index = 0

    def tick(self) -> int:
        self._step_index += 1
        return self._step_index

    def record_fields(self) -> dict:
        return {"controller_law": "external", "external_gain_source": self.gain_source}


def _method_gain(method) -> tuple[float, str]:
    g = _params(method).get("gain", getattr(method, "gain", None))
    if isinstance(g, (int, float)) and not isinstance(g, bool) and np.isfinite(g):
        return float(g), "method.params['gain']"
    return 1.0, "1.0: the method states no gain (its law applies its own)"


# ------------------------------------------------------------------- the arm
class MethodArm(Arm):
    """A vsbench method behind the copied ``Arm``: the copied loop flies it unchanged."""

    requires_live_mask = False

    def __init__(self, method, config: dict, *, intrinsics, goal_mask_mode: str = "none",
                 mask_token: str | None = None) -> None:
        from vsbench.methods.base import check_method

        validate_external(config)
        check_method(method)
        if config["method"]["name"] != method.name:
            raise ValueError(f"the config names method {config['method']['name']!r}, the object "
                             f"is {method.name!r}")
        assert_method_gt_blind(method)
        dt = float(config["controller"]["dt"])
        gain, gain_source = _method_gain(method)
        rule = str(config["stop"]["rule"])
        stop = build_stop_rule(config, **({"dt": dt} if rule == "cns_plateau"
                                          and "dt" not in config["stop"] else {}))
        if goal_mask_mode not in METHOD_MASK_MODES:
            raise ValueError(f"goal_mask_mode must be one of {METHOD_MASK_MODES}, "
                             f"got {goal_mask_mode!r}")
        cfg_mode = str((config.get("matcher") or {}).get("goal_mask") or "none")
        if cfg_mode != goal_mask_mode:
            raise ValueError(f"the config says matcher.goal_mask {cfg_mode!r}, the arm is built "
                             f"with goal_mask_mode {goal_mask_mode!r}")
        front = MethodFrontEnd(method, intrinsics=intrinsics, dt=dt)
        if goal_mask_mode != "none" and not front.honours_goal_mask:
            raise ValueError(f"{method.name!r} takes no goal mask (its set_target has no "
                             "goal_mask parameter): it cannot fly a masked cell")
        if (goal_mask_mode == "none") != (mask_token is None):
            raise ValueError("a masked method cell needs its label token, an unmasked one none")
        self.mask_token = mask_token
        super().__init__(name=method_label(method, rule, mask_token),
                         matcher=front,
                         controller=MethodController(dt, gain, gain_source), stop=stop,
                         config=config, checkpoint=None, kind=EXTERNAL_KIND,
                         goal_mask_mode=goal_mask_mode)
        self._configured_stop = stop
        self._configured_name = self.name
        self.stop_fallback: dict | None = None
        self.reports_points = getattr(method, "reports_points", None)
        if self.reports_points is False:
            self._fall_back("the method declares reports_points = False")

    @property
    def method(self):
        return self.matcher.method

    # -- the stop fallback --------------------------------------------------------------
    def _fall_back(self, reason: str) -> None:
        rule = self._configured_stop.name
        if rule not in POINT_STOPS:
            return
        self.stop = build_stop_rule({"stop": {"rule": FALLBACK_STOP}})
        self.stop.reset()
        self.name = method_label(self.method, FALLBACK_STOP, self.mask_token)
        self.stop_fallback = {"configured": rule, "ran": FALLBACK_STOP, "reason": reason}
        msg = (f"[method_arm] STOP FALLBACK: {self.method.name}: {reason}, so the scored stop "
               f"{rule!r} (it reads the method's points) cannot run - this cell flies "
               f"{FALLBACK_STOP!r}; the record says stop_mode {FALLBACK_STOP!r}, label "
               f"{self.name!r}, and stop_fallback")
        print(msg, file=sys.stderr, flush=True)
        warnings.warn(msg, StopFallbackWarning, stacklevel=3)

    def reset(self, goal_rgb, goal_depth=None, goal_mask=None) -> None:
        # every episode decides its stop afresh (on its own step 1)
        if self.reports_points is not False:
            self.stop = self._configured_stop
            self.name = self._configured_name
            self.stop_fallback = None
        super().reset(goal_rgb, goal_depth, goal_mask)

    # -- the image seam -----------------------------------------------------------------
    def step(self, rgb: np.ndarray, depth: np.ndarray,
             mask: np.ndarray | None = None) -> tuple[Twist, dict, Matches]:
        """One control step: the method's ``get_control_rate`` on the live frame."""
        from vsbench.methods.base import normalise_output

        if mask is not None:
            raise ValueError("a method cell takes no live mask (requires_live_mask is False)")
        m = self.method
        dense = m.wants_depth == "dense"
        frame = _ro_copy(rgb)
        live_depth = _ro_copy(depth, np.float32) if dense else None
        t0 = time.monotonic()
        twist_arr, info = normalise_output(m.get_control_rate(frame, live_depth))
        call_ms = (time.monotonic() - t0) * 1000.0
        step = self.controller.tick()
        kc, kd = info.get("kp_cur"), info.get("kp_des")
        has_points = kc is not None and kd is not None
        if step == 1 and not has_points and self.stop_fallback is None:
            self._fall_back("the method reported no kp_cur / kp_des on step 1")
        pc = (np.zeros((0, 2)) if kc is None else np.asarray(kc, np.float64).reshape(-1, 2))
        pg = (np.zeros((0, 2)) if kd is None else np.asarray(kd, np.float64).reshape(-1, 2))
        if len(pc) != len(pg):
            raise ValueError(f"{m.name}: kp_cur has {len(pc)} rows, kp_des {len(pg)}")
        n_reported = len(pc)
        inl = info.get("inliers")
        if inl is not None:
            inl = np.asarray(inl).reshape(-1).astype(bool)
            if len(inl) == len(pc):
                pc, pg = pc[inl], pg[inl]
        lost = bool(info.get("lost", False))
        en = info.get("error_norm")
        feat = float(en) if en is not None else float("nan")
        intr = self.matcher.intrinsics
        normalised = None
        if len(pc) and not lost:
            f = np.array([float(intr.fx), float(intr.fy)])
            c = np.array([float(intr.cx), float(intr.cy)])
            normalised = np.linalg.norm((pc - c) / f - (pg - c) / f, axis=1)
        w, h = int(intr.width), int(intr.height)
        k = len(pc)
        matches = Matches.from_pixel_pairs(
            pg, pc, side=w, input_size=w, patch_size=1, image_wh=(w, h),
            sim=np.full(k, np.nan), cyclic=np.full(k, np.nan), margin=np.full(k, np.nan),
            pool_size=n_reported, num_landmarks=k, draw_seed=None, tie_order="method")
        obs_s, ctl_s = info.get("observer_s"), info.get("controller_s")
        debug = {
            "step": step,
            "step_ms": (float(ctl_s) * 1000.0 if ctl_s is not None else call_ms),
            "match_ms": (float(obs_s) * 1000.0 if obs_s is not None else 0.0),
            "call_ms": call_ms,
            "n_matches": k,
            "feature_error_norm": feat,
            "normalised_errors": normalised,
            "degenerate": lost,
            "pts_goal": pg,
            "pts_cur": pc,
            # the bench's own read of the live depth at the method's points (truncating,
            # 0.5 m fallback): a telemetry column, not what the method used
            "depth_values": depth_at(depth, pc) if k else np.zeros(0),
            "velocity_raw": twist_arr.copy(),
            "method": {"lost": lost, "n_reported": n_reported, "n_inliers": k,
                       "error_norm": None if not np.isfinite(feat) else feat,
                       "call_ms": round(call_ms, 3)},
        }
        return Twist(twist_arr), debug, matches

    def record_fields(self) -> dict:
        out = {**super().record_fields(),
               "stop_rule_configured": self._configured_stop.name,
               "stop_fallback": self.stop_fallback}
        if self.goal_mask_mode != "none":       # an unmasked record stays key for key as before
            out["external_method_takes_goal_mask"] = bool(self.matcher.honours_goal_mask)
        return out


def build_method_arm(method_name: str, params: dict | None = None, *, intrinsics,
                     config: dict | None = None, goal_mask_mode: str = "none",
                     mask_token: str | None = None, mask_source: str | None = None,
                     **config_kw) -> tuple[MethodArm, dict]:
    """``(arm, config)`` for a registered vsbench method: ``make_method(name, **params)``,
    the template config (unless one is given) and :class:`MethodArm`. A masked cell
    (``goal_mask_mode`` ``objects`` / ``file``) states the mode and its source in the config's
    ``matcher`` block (``goal_mask``, ``goal_mask_source``)."""
    from vsbench.methods import make_method

    cfg = config if config is not None else external_config(method_name, params, **config_kw)
    if config is None and goal_mask_mode != "none":
        cfg["matcher"]["goal_mask"] = goal_mask_mode
        cfg["matcher"]["goal_mask_source"] = mask_source
        validate_external(cfg)
    method = make_method(method_name, **dict(cfg["method"].get("params") or {}))
    return MethodArm(method, cfg, intrinsics=intrinsics, goal_mask_mode=goal_mask_mode,
                     mask_token=mask_token), cfg


# ------------------------------------------------------------------- the run sidecar
def open_method_run(name: str, config: dict, out_root, *, telemetry: bool = False,
                    allow_dirty: bool = False, allow_no_git: bool = False,
                    extra: dict | None = None):
    """The copied run directory for a method cell: ``run.json``, ``resolved_config.yaml``,
    ``MD5SUMS`` written through the copied helpers, and the copied ``EpisodeSink``.

    The copied ``start_run`` line for line, except the validation (:func:`validate_external`
    instead of the copied ``config.validate``, which refuses ``external``) and two fields:
    ``config_schema`` names the external schema, ``matcher_tie_order`` is ``None``.
    """
    import platform

    import yaml

    from vsbench.vs.bench import run as runmod
    from vsbench.vs.bench.runner import EpisodeSink

    validate_external(config)
    resolved = copy.deepcopy(config)
    root = Path(out_root)
    head = runmod.git_head()
    dirty = runmod.dirty_paths(ignore_prefix=root)
    git_ok = head is not None and dirty is not None
    if not git_ok and not allow_no_git:
        raise runmod.RunError("cannot read the git state of the code (not a git checkout?); "
                              "pass allow_no_git=True to record git_commit=null")
    dirty = dirty or []
    if dirty and not allow_dirty:
        raise runmod.RunError(f"working tree is dirty ({len(dirty)} path(s)); commit first, or "
                              "pass allow_dirty=True (it is recorded in run.json)")
    now = datetime.now(timezone.utc)
    stem = f"{now:%Y-%m-%d}_{name}"
    path = root / stem
    n = 2
    while path.exists():
        path = root / f"{stem}_{n:02d}"
        n += 1
    path.mkdir(parents=True)
    meta = {
        "name": name, "run_dir": str(path), "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": head, "git_available": git_ok, "git_dirty": bool(dirty) if git_ok else None,
        "git_dirty_paths": dirty, "allow_dirty": bool(allow_dirty), "allow_no_git": bool(allow_no_git),
        "seed": resolved["seed"], "hostname": socket.gethostname(),
        "gpu": runmod._gpu_name(), "gpu_driver": runmod._gpu_driver(), "finished_utc": None,
        "python_version": platform.python_version(), "torch_version": runmod._torch_version(),
        "vssim_version": __import__("vsbench.vs", fromlist=["__version__"]).__version__,
        **runmod._torch_numeric_env(),
        "matcher_tie_order": None,
        "config_schema": SCHEMA,
    }
    clash = sorted(set(extra or {}) & set(meta))
    if clash:
        raise runmod.RunError(f"open_method_run(extra=...) may not overwrite: {', '.join(clash)}")
    meta.update(extra or {})
    with (path / runmod.CONFIG_FILE).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=True, default_flow_style=False)
    with (path / runmod.META_FILE).open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    (path / runmod.MD5_FILE).write_text("", encoding="utf-8")
    run = runmod.RunDir(path=path, name=name, config=resolved, meta=meta)
    return EpisodeSink(run, telemetry=telemetry)


# ------------------------------------------------------------------- the plugin face
def plugin_ids() -> tuple[str, ...]:
    """The registered vsbench methods a benchmark may fly as plugin arms: not privileged,
    not a copied protocol cell (``vsbench.methods.vssim_arms`` registers those; flying one
    through ``MethodArm`` would wrap a published baseline twice), ids ``[A-Za-z0-9_]+``."""
    from vsbench.methods import available_methods, method_class
    from vsbench.methods.vssim_arms import VssimArmMethod

    out = []
    for name in available_methods():
        cls = method_class(name)
        if bool(getattr(cls, "privileged", False)) or issubclass(cls, VssimArmMethod):
            continue
        if set(name) <= _ID_CHARS:
            out.append(name)
    return tuple(out)


def __getattr__(name: str):
    # ``ARM_IDS`` is read off the registry when asked (a student's method registered after
    # this module was imported is included), as vs-sim's ``--arm-plugin`` reads it
    if name == "ARM_IDS":
        return plugin_ids()
    raise AttributeError(name)


def build(arm_id: str, *, intrinsics, scene_name: str, scene_spec=None):
    """vs-sim's plugin ``build``: ``(arm, config)`` for the method ``arm_id`` at defaults."""
    if arm_id not in plugin_ids():
        raise KeyError(f"{arm_id!r} is not a plugin method here (have {', '.join(plugin_ids())})")
    return build_method_arm(arm_id, intrinsics=intrinsics)


def describe(arm_id: str) -> dict:
    """vs-sim's plugin ``describe``: where the method came from."""
    from vsbench.methods import method_class

    cls = method_class(arm_id)
    from vsbench.world.fly import vsbench_commit

    return {"module": cls.__module__, "class": cls.__qualname__, "commit": vsbench_commit(),
            "config_source": str(TEMPLATE.relative_to(TEMPLATE.parents[3])),
            "label_prefix": LABEL_PREFIX}
