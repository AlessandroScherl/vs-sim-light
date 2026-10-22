# vs-sim copy: vssim/bench/arms.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The arm registry: a config builds a method, and nothing parses a name.

Purpose
-------
An *arm* is one complete visual-servoing method: a matcher, the control law, and
a stop rule. This module builds one from a resolved config and owns the **image
seam** - match the frame, sample the depth map at the current pixels, call the
controller.

What this replaces
------------------
``genvs-sim/src/eval/baseline_check.py`` L275-690 built arms by *parsing their
names*: ``head_w_g3_axw_mw_tsel_ur@v2cm`` was split into a base, a gamma suffix,
a checkpoint tag and fourteen booleans, and the result was reassembled by
``build_controller`` (L1916-2126). Roughly 600 lines whose only job was to turn a
string into settings that a config states directly. It is the mechanism behind the
random-24 incident: a suffix that was not written down was a setting that was not
recorded. **Nothing here parses a name.** The label is an output, not an input:
:func:`arm_label` derives it from the settings so tables keep reading the way they
always did, and it is written into the record beside the settings themselves.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
- ``src/control/base_controller.py::VSMethod.step(rgb, camera_info, depth, mask)``
  and ``src/control/matcher_ibvs.py::MatcherIBVS.step`` L253-300 -> :meth:`Arm.step`.
- ``src/control/ibvs_core.py::sample_depth`` L122-144 -> :func:`depth_at`, the
  pixel lookup only (U4 owns the validity rule, :func:`vsbench.vs.baselines.control.clean_depth`).
- ``baseline_check.build_controller`` L1916-2126 -> :func:`build_arm`, minus the
  grammar: the matcher comes from :func:`vsbench.vs.baselines.match.build_matcher`, the
  controller from :meth:`vsbench.vs.baselines.control.IBVSController.from_config`, the stop
  rule from :func:`vsbench.vs.baselines.control.build_stop_rule`.

- 2026-10-05, the faithful ART-VS arm (``artvs_roi_sim``): the depth lookup's fallback
  became a stated setting (``controller.depth_fallback_m``; the release's
  ``ros_controller.py:423-425`` reads 100 m), a truncating controller reads the depth at
  the whole pixel it solves on (``get_depth(s_uv)``, ``:2515``), a matcher may name a
  frame's own pixel rule (``Matches.extras["quantise"]``), the controller's non-default
  knobs join the record, and the label gains ``tr`` / ``ema`` / ``vcap`` / ``hold`` - all
  absent on every earlier file (``tests/fast/test_arms.py`` pins the old labels).
- 2026-10-06, the LangSAM check: the faithful arm's label gains ``sam-langsam``
  (``matcher.roi_sam_path``: SAM run as the ``lang_sam`` package runs it), after
  ``lsmask``; no other arm's label moves (pinned in ``tests/fast/test_arms.py`` and
  ``tests/fast/test_protocol.py``).
- 2026-10-10, the ART-VS development cell (``artvs_roi_sim_dev``): under
  ``matcher.roi_mask: objects`` the faithful arm's label drops ``lsmask`` / ``sam-*`` and takes
  the generic ``objmask`` (the simulator's outline, ``matcher.goal_mask: objects``), and
  ``matcher.switch_max_steps`` adds ``switch-by-step<N>``; :func:`build_arm` refuses a matcher
  that ``requires_goal_mask`` on a config that asks for none. No earlier label moves.

Deliberately dropped
--------------------
The whole suffix grammar (``RefineArm``, ``parse_refine_modes``,
``parse_refine_checkpoints``, ``REFINE_SUFFIXES``, ``_GAMMA_SUFFIX`` ...), every
retired mechanism arm and the film hooks. The pre-alignment wiring is a
``rotation:`` block, not a mode string. The research repository's refinement head
and oracle matchers are not built here (:data:`FOREIGN_BLOCKS`).

Fences
------
- **The depth lookup truncates.** ``int(np.clip(u, 0, w - 1))``, not ``round``.
  On the sub-pixel chain (100.4, 200.6) and (100.6, 200.4) read the same depth,
  and rounding instead moves millimetres.
- **CNS is an arm, not a matcher.** ``matcher.variant: cns_v1`` dispatches to
  :func:`vsbench.vs.baselines.cns.build_cns_arm`, because CNS replaces the control law.
  **So is DINOBot**: ``matcher.variant: dinobot`` dispatches to
  :func:`vsbench.vs.baselines.dinobot.build_dinobot_arm` (a fitted rigid jump per step, not a
  velocity law); its front end alone is the matcher ``dinobot_corr``, which the IBVS
  law can sit behind.
- **GT-blind.** An :class:`Arm` is handed frames and depth. It has no scene, no
  pose and no goal geometry, so no stop rule and no control step can read the
  true pose even by accident.
- **Only the baselines are here.** The research repository's refinement head
  (``refine:``) and its ground-truth oracle matchers (``oracle:``) are refused by
  name rather than silently ignored.

Golden tests: ``tests/fast/test_arms.py`` - each of ``configs/arms/*.yaml`` builds
with a weight-free stub backbone and derives its label; a config missing a required
field fails in the loader; a ``refine:`` / ``oracle:`` / ``head:`` block is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vsbench.vs.baselines import prealign as rotmod
from vsbench.vs.baselines.classical import ClassicalMatcher
from vsbench.vs.baselines.control import (
    DEFAULT_DT,
    DEPTH_FALLBACK_M,
    IBVSController,
    Intrinsics,
    StopRule,
    Twist,
    build_stop_rule,
    clean_depth,
)
from vsbench.vs.baselines.features import DEFAULT_BACKBONE
from vsbench.vs.baselines.match import CorrespondenceMatcher, DenseMatcher, Matches, build_matcher
from vsbench.vs.baselines.prealign import build_rotation, rotation_record_fields
from vsbench.vs.baselines.tiling import TiledMatcher
from vsbench.vs.bench import config as cfgmod

#: What each ``matcher.variant`` is allowed to be, when a ready-made matcher is
#: injected instead of built. A cell whose record says one thing and whose object was
#: another is the random-24 failure shape, and it arrives one unit before the oracles
#: (review R13, C2), so the mismatch is refused unless the arm declares itself.
VARIANT_BUILDS = {
    "mutual_nn_seeded_subsample": DenseMatcher,
    "mutual_nn_all": DenseMatcher,
    "frozen_landmarks": DenseMatcher,
    "classical_sift": ClassicalMatcher,
    "classical_orb": ClassicalMatcher,
    "classical_akaze": ClassicalMatcher,
    "artvs": TiledMatcher,
}   # `cns_v1` is not a matcher at all (see build_arm); `lightglue` / `roma` / `romav2`
    # are registered lazily below (their modules import torch-heavy packages)


def _variant_class(variant: str):
    """The matcher class a variant names, importing the heavy ones only when asked."""
    if variant in VARIANT_BUILDS:
        return VARIANT_BUILDS[variant]
    if variant == "lightglue":
        from vsbench.vs.baselines.lightglue import LightGlueMatcher  # noqa: PLC0415

        return LightGlueMatcher
    if variant in ("roma", "romav2"):
        from vsbench.vs.baselines.roma import RoMaMatcher  # noqa: PLC0415

        return RoMaMatcher
    if variant == "artvs_roi":
        from vsbench.vs.baselines.objroi import RoiTiledMatcher  # noqa: PLC0415

        return RoiTiledMatcher
    if variant == "artvs_roi_sim":
        from vsbench.vs.baselines.objroi import RoiTiledMatcherSim  # noqa: PLC0415

        return RoiTiledMatcherSim
    if variant == "dinobot_corr":
        from vsbench.vs.baselines.dinobot import DinoBotCorrespondences  # noqa: PLC0415

        return DinoBotCorrespondences
    return None

#: ``arm.kind``. ``method`` is a real method and its matcher must match its variant.
#: ``oracle`` is a ground-truth matcher (a mock world's own truth) and ``test`` is a
#: stub; both may be injected against any variant, both are written into the record
#: and into the derived label, and neither may ever be pooled with a method cell by
#: accident.
ARM_KINDS = ("method", "oracle", "test")

#: Config blocks that belong to the research repository this was extracted from and
#: are refused here BY NAME - an arm that silently ran without a component it named
#: is the kind of mislabelled cell this bench exists to prevent.
FOREIGN_BLOCKS = {
    "refine": "the trained refinement head (Refine-VS `refinevs.refine`)",
    "oracle": "the ground-truth oracle matchers (Refine-VS `refinevs.oracle`)",
    "head": "a training config's architecture block",
}

#: ``controller.law``: which control law turns matched points into a twist. ``ibvs`` is
#: the bench's own :class:`vsbench.vs.baselines.control.IBVSController` (the published ViT-VS
#: law, with the optional weight chain); ``visp`` is ViSP's ``vpServo`` point-feature law
#: (:class:`vsbench.vs.baselines.visp_ibvs.VispIBVSController`), unweighted by definition.
CONTROL_LAWS = ("ibvs", "visp")


def build_controller(config: dict, *, intrinsics: Intrinsics):
    """The controller a config's ``controller.law`` names (default ``ibvs``)."""
    law = str(((config.get("controller") or {}).get("law")) or "ibvs")
    if law not in CONTROL_LAWS:
        raise ValueError(f"controller.law is {law!r}, not one of: {', '.join(CONTROL_LAWS)}")
    if law == "visp":
        from vsbench.vs.baselines.visp_ibvs import VispIBVSController  # noqa: PLC0415

        return VispIBVSController.from_config(config, intrinsics=intrinsics)
    block = dict(config.get("controller") or {})
    block.pop("law", None)
    return IBVSController.from_config({**config, "controller": block}, intrinsics=intrinsics)


def depth_lookup(depth_map: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """The raw depth map value at each ``(column, row)`` pixel - **truncating**.

    ``int(np.clip(u, 0, w - 1))``: a truncation, not a rounding, and it is
    load-bearing on the sub-pixel chain: (100.4, 200.6) and (100.6, 200.4) both read
    pixel (100, 200). No validity rule: :func:`depth_at` applies the IBVS one (invalid ->
    0.5 m), and :mod:`vsbench.vs.baselines.dinobot` drops invalid rows instead (a rigid fit
    must not see an invented point). One lookup, two policies.
    """
    d = np.asarray(depth_map)
    if d.ndim != 2:
        raise ValueError(f"depth map must be (H, W), got shape {d.shape}")
    h, w = d.shape
    px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    u = np.clip(px[:, 0].astype(np.int64), 0, w - 1)
    v = np.clip(px[:, 1].astype(np.int64), 0, h - 1)
    return np.asarray(d[v, u], dtype=np.float64)


def depth_at(depth_map: np.ndarray, pts_px: np.ndarray, *,
             fallback_m: float = DEPTH_FALLBACK_M) -> np.ndarray:
    """Depth in metres at each ``(column, row)`` pixel - **truncating**, as before.

    :func:`depth_lookup`, then the validity rule (z <= 0, z > 10 m, NaN, Inf -> 0.5 m),
    which lives in :func:`vsbench.vs.baselines.control.clean_depth`. ``fallback_m`` replaces the
    0.5 m where an arm states another (``controller.depth_fallback_m``: the ART-VS
    release's simulation program reads 100 m, ``ros_controller.py:423-425``).
    """
    return clean_depth(depth_lookup(depth_map, pts_px), fallback_m=fallback_m)


#: ``matcher.prompt: scene`` - the language-prompted detector (ART-VS's detector mode,
#: ``artvs_roi``) takes its word from the scene spec's ``prompt:`` (the instance suite,
#: 2026-09-23: one cell over six kinds of object). The SETTING stays ``scene``: the label
#: carries ``prompt-scene`` (one label for the cell across every kind) and the record's
#: ``roi_prompt`` carries the word the detector was given (:func:`build_arm` hands it to
#: the matcher's constructor). A cell that states its own word - the published
#: ``artvs_roi``, "shoe" - is labelled exactly as before.
SCENE_PROMPT = "scene"


def resolve_scene_prompt(config: dict, scene_spec: Any) -> str | None:
    """The word ``matcher.prompt: scene`` stands for - the scene spec's own ``prompt`` -
    or ``None`` when the cell states a word of its own.

    Refused loudly without a scene spec or when the scene states no prompt: a detector
    prompted with nothing, or with another scene's word, would be a different cell under
    this cell's name.
    """
    m = config.get("matcher") or {}
    if m.get("prompt") != SCENE_PROMPT:
        return None
    if scene_spec is None:
        raise ValueError("matcher.prompt 'scene' takes the word from the scene spec - declare "
                         "`needs: [scene_spec]` on the cell and pass the scene")
    word = getattr(scene_spec, "prompt", None)
    if not word:
        raise ValueError(f"matcher.prompt 'scene': scene {getattr(scene_spec, 'name', '?')!r} "
                         "states no `prompt:` in configs/scenes.yaml")
    return str(word)


def arm_label(config: dict) -> str:
    """A stable, readable label **derived from** the settings - never parsed back.

    ``mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/cns_plateau``: variant
    (with its encoder, for the ViT arms), landmark count, sub-pixel flag, the weight
    chain, the scored stop. Two arms with the same
    label have the same settings by construction, which is the property the old
    grammar only claimed to have. An arm that is not a ``method`` carries its
    :data:`ARM_KINDS` kind as a prefix (``oracle:...``), so a ground-truth cell can
    never be read as a method's number.
    """
    m = config["matcher"]
    w = ((config.get("controller") or {}).get("weighting") or {})
    ctrl = config.get("controller") or {}
    variant = str(m["variant"])
    if variant == "cns_v1":
        # CNS names itself from its own settings (`vsbench.vs.baselines.cns.CNSArm`): the
        # detector, the effective step scale and the stop - the IBVS weight chain and
        # landmark count above mean nothing to it. Derived here from the same fields so
        # `fly.py plan` prints the label the record will carry.
        from vsbench.vs.baselines.cns import PUBLISHED  # noqa: PLC0415 - avoids a cycle

        block = config.get("cns") or {}
        gain = float(block.get("step_gain", PUBLISHED["step_gain"]))
        if block.get("rate_compensate", True):
            gain *= float(block.get("published_dt", PUBLISHED["dt"])) / float(
                ctrl.get("dt", DEFAULT_DT))
        label = (f"cns_v1_{str(block.get('detector', 'SIFT')).lower()}/x{gain:g}/"
                 f"{'objmask/' if str(m.get('goal_mask') or 'none') == 'objects' else ''}"
                 f"{config['stop']['rule']}")
        kind = str((config.get("arm") or {}).get("kind", "method"))
        return label if kind == "method" else f"{kind}:{label}"
    if variant == "dinobot":
        # DINOBot's alignment names itself from its own `dinobot:` block (model, pair count,
        # the jump fraction, the stated error space, the reflection handling, the mask, the
        # stop) - `vsbench.vs.baselines.dinobot.dinobot_label`, the same function the arm uses.
        from vsbench.vs.baselines.dinobot import dinobot_label  # noqa: PLC0415 - avoids a cycle

        label = dinobot_label(config)
        kind = str((config.get("arm") or {}).get("kind", "method"))
        return label if kind == "method" else f"{kind}:{label}"
    if variant == "dinobot_corr":
        # the encoder is a setting of this cell as of every ViT arm: DINO v1 ViT-S/8 from
        # the `dinobot:` block, never `matcher.backbone_name`
        from vsbench.vs.baselines.dinobot import dinobot_corr_tag  # noqa: PLC0415

        variant += f"@{dinobot_corr_tag(config)}"
    if variant in cfgmod.BACKBONE_VARIANTS:
        # the encoder is a setting of every ViT arm, and the two ViT-VS cells (DINOv3,
        # the published DINOv2 chain) differ in nothing else - so it is in the label
        variant += f"@{m.get('backbone_name') or DEFAULT_BACKBONE}"
    parts = [variant, f"k{int(m['num_landmarks'])}"]
    if (str(m["variant"]) in ("artvs_roi", "artvs_roi_sim")
            and m.get("prompt") == SCENE_PROMPT):
        # the instance suite's cell takes its word from the scene: the label says so, and
        # the record's `roi_prompt` says which word it was (the published cell with its
        # stated word is labelled as it always was)
        parts.append(f"prompt-{SCENE_PROMPT}")
    if str(m["variant"]) == "artvs_roi_sim":
        roi_mask = str(m.get("roi_mask", "langsam"))
        if roi_mask != "objects":
            # LangSAM's goal mask through the release's own mask branch (owner ruling D-B (b),
            # 2026-10-05), stated in its label
            parts.append({"langsam": "lsmask"}.get(roi_mask, f"mask-{m.get('roi_mask')}"))
            # SAM run as the lang_sam package runs it (the LangSAM check, 2026-10-06), so the
            # goal mask is the release's own bit for bit; the arm flown before (the 2026-10-05
            # smoke, transformers' SAM) carries no `sam-` token - a different mask, told apart
            parts.append(f"sam-{m.get('roi_sam_path', 'langsam')}")
        # `objects` (the dev cell, 2026-10-10): the simulator's outline, which the generic
        # `objmask` token below states (the cell must say `goal_mask: objects` to get it); no
        # SAM runs, so no `sam-` token
    if str(m.get("goal_mask") or "none") == "objects":
        parts.append("objmask")           # landmarks from the goal's objects only
    if str(m["variant"]) == "artvs_roi_sim" and int(m.get("switch_max_steps") or 0) > 0:
        # the dev cell's switch (owner 2026-10-10): the tiles at the release's error rule or at
        # this bench step, whichever comes first; absent from every earlier config
        parts.append(f"switch-by-step{int(m['switch_max_steps'])}")
    pq = ctrl.get("pixel_quantisation")
    if pq == "truncate":
        parts.append("tr")                # the ART-VS release's whole pixels (2026-10-05)
    elif pq == "none" or (pq is None and bool(ctrl.get("solve_unrounded", True))):
        parts.append("ur")
    chain = []
    if str(ctrl.get("law") or "ibvs") == "visp":
        w = {"use_trust": False, "use_margin": False, "per_axis": False}   # by definition
    if w.get("use_trust", True):
        chain.append(f"w-g{int(w.get('trust_gamma', 3))}")
    if w.get("use_margin", True):
        chain.append("mw")
    if w.get("per_axis", True):
        chain.append("axw")
    parts.append("-".join(chain) if chain else "uniform")
    # the 2026-10-05 knobs (the faithful ART-VS arm); absent from every earlier file, so no
    # earlier label moves (tests/fast/test_arms.py pins them)
    if float(ctrl.get("ema_alpha", 0.0) or 0.0) > 0.0:
        parts.append(f"ema{float(ctrl['ema_alpha']):g}")
    if ctrl.get("max_velocity") is not None:
        parts.append(f"vcap{float(ctrl['max_velocity']):g}")
    if str(ctrl.get("degenerate_policy") or "zero") == "hold":
        parts.append("hold")
    law = str(ctrl.get("law") or "ibvs")
    if law != "ibvs":
        # ViSP's law beside the bench's own on the same front end is a different
        # method, and the interaction-matrix choice is its claim-bearing setting
        parts.append(f"{law}-{ctrl.get('interaction', 'current')}"
                     + ("-adaptive" if ctrl.get("adaptive_gain") else ""))
    # Rotation is claim-bearing and must never be pooled away: a baseline with and
    # without the published pre-servo roll are different methods.
    # `rotation_config` first, not `config["rotation"]` raw: a label is derived from
    # SETTINGS, and settings that have not been validated are not settings yet.
    rblock = rotmod.rotation_config(config)
    if rblock["mode"] == "prealign_4way":
        # the hypothesis count is a setting, so the label follows it: a 12-angle search
        # is not the published 4-way and must not label itself as one.
        n = len(rblock.get("prealign_angles") or rotmod.DISCRETE_ANGLES)
        parts.append(f"pre{n}way")
    if "ransac" in m:
        # Protocol v2 section 4 runs the classical arms in TWO columns, with the
        # scene's geometric model and without it, and the two are different methods -
        # so the label carries which one, and a table cannot pool them by accident.
        parts.append(f"ransac-{m['ransac']}")
    parts.append(str(config["stop"]["rule"]))
    kind = str((config.get("arm") or {}).get("kind", "method"))
    label = "/".join(parts)
    return label if kind == "method" else f"{kind}:{label}"


@dataclass
class Arm:
    """One method: matcher + control law + stop rule, and the image seam between them.

    Built by :func:`build_arm`; every field is recorded, none is implied. The
    episode runner calls :meth:`set_episode`, :meth:`reset` once, then
    :meth:`step` per frame.
    """

    name: str
    matcher: CorrespondenceMatcher
    controller: IBVSController
    stop: StopRule
    config: dict = field(repr=False)
    checkpoint: str | None = None
    kind: str = "method"
    #: `matcher.goal_mask`: `none`, or `objects` - the runner then hands `reset()` a
    #: goal mask computed from the goal depth (objects only, no floor).
    goal_mask_mode: str = "none"

    @property
    def matcher_name(self) -> str:
        """What the matcher **object** calls itself - the record's honest label.

        The config states the variant; this states what ran. They differ exactly
        when a matcher was injected, which is why an injection must declare a
        non-``method`` kind (review R13, C2).
        """
        return str(getattr(self.matcher, "name", None) or type(self.matcher).__name__)

    @property
    def dt(self) -> float:
        return self.controller.dt

    @property
    def gain(self) -> float:
        return self.controller.gain

    def set_episode(self, scene: str, start_id: Any) -> None:
        """Name the episode so the landmark draw is paired across arms."""
        self.matcher.set_episode(scene, start_id)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        """Start of an episode: cache the goal view, clear controller and stop state."""
        self.matcher.reset(goal_rgb, goal_depth, goal_mask)
        self.controller.reset()
        # a law that needs Z* (ViSP's `desired` / `mean` interaction matrix) takes the
        # goal depth map here; the bench's own law reads only the live depth
        set_goal_depth = getattr(self.controller, "set_goal_depth", None)
        if set_goal_depth is not None:
            set_goal_depth(goal_depth)
        self.stop.reset()

    def step(self, rgb: np.ndarray, depth: np.ndarray,
             mask: np.ndarray | None = None) -> tuple[Twist, dict, Matches]:
        """One control step from a frame.

        Match, look the depth map up at the *current* pixels (truncating), and
        solve. Returns ``(twist, debug, matches)``; ``debug`` additionally carries
        ``depth_values`` (the per-row metres the solve used) and ``match_ms``, the
        front-end time reported beside the controller time the solve already reports
        as ``step_ms``.
        """
        import time  # noqa: PLC0415 - one call per step, kept local to the seam

        t0 = time.monotonic()
        matches = self.matcher.match(rgb, mask)
        match_ms = (time.monotonic() - t0) * 1000.0
        rows = matches.to_rows() if len(matches) else {
            "pts_goal": np.zeros((0, 2)), "pts_cur": np.zeros((0, 2)),
            "conf": np.zeros(0), "margin": np.zeros(0),
        }
        quantise = (getattr(matches, "extras", None) or {}).get("quantise")
        if quantise is not None:
            # a frame whose matcher names its own pixel rule (the ART-VS box-less fallback
            # rounds, `calculate_uv`); the controller honours it for this frame only
            rows["quantise"] = quantise
        pts_depth = rows["pts_cur"]
        if getattr(self.controller, "pixel_quantisation", None) == "truncate" and len(pts_depth):
            # the ART-VS release reads the depth at the WHOLE pixel it solves on
            # (`get_depth(s_uv)`, ros_controller.py:2515) - the same pixel a truncating
            # lookup finds, except where the rows' 16:9 round trip left 7.0 at 6.99999...
            pts_depth = self.controller.quantise(rows)[1]
        z = depth_at(depth, pts_depth,
                     fallback_m=getattr(self.controller, "depth_fallback_m", DEPTH_FALLBACK_M))
        twist, debug = self.controller.step(rows, z)
        debug["depth_values"] = z
        debug["match_ms"] = match_ms
        debug["n_matches"] = len(matches)
        return twist, debug, matches

    def record_fields(self) -> dict:
        """The published knobs this arm ran under, for the episode record.

        The runner merges this into every record (``rec.update(arm.record_fields())``).
        An arm class may own some (:class:`vsbench.vs.baselines.cns.CNSArm` owns CNS's step gain);
        here the matcher owns them, because for the published baselines the claim-bearing
        settings - which RANSAC model ran, whether it fitted one, the tiling geometry,
        whether the coarse->tiled switch ever fired - are the matcher's. **Never a
        measurement**: everything here is a setting or a count of settings applied.
        """
        # The rotation columns go LAST on purpose: `DenseMatcher.record_fields` states
        # `prealign` on every row, and on an arm that really took the 4-way that column
        # must say which variant ran rather than the matcher's default `None`.
        # the controller adds only knobs it runs at a non-default value (2026-10-05), so an
        # earlier arm's record is unchanged key for key
        return {**dict(getattr(self.matcher, "record_fields", dict)()),
                **rotation_record_fields(self.matcher),
                "goal_mask_mode": self.goal_mask_mode,
                **dict(getattr(self.controller, "record_fields", dict)())}

    def close(self) -> None:
        close = getattr(self.matcher, "close", None)
        if close is not None:
            close()


def build_arm(config: dict, *, intrinsics: Intrinsics, scene: str | None = None,
              start_id: Any = None, matcher: CorrespondenceMatcher | None = None,
              world: Any = None, **matcher_overrides: Any) -> Arm:
    """Build the arm a resolved config asks for.

    ``config`` is validated first (:func:`vsbench.vs.bench.config.validate`), so an arm
    whose config omits ``matcher.variant``, ``matcher.num_landmarks``, ``seed``,
    ``stop.rule`` or ``scoring.success_rule`` fails here, naming the field.
    ``matcher`` injects a ready-made matcher (a test's stub, or one encoder shared
    across arms); ``matcher_overrides`` reach the matcher's constructor. An injected
    matcher must be the class ``matcher.variant`` names, unless the config declares
    ``arm.kind`` as ``oracle`` or ``test`` - otherwise the record would label a cell
    as something it did not run.

    ``world`` is accepted for signature compatibility with the research repository's
    oracle arms and must be ``None`` here.
    """
    cfgmod.validate(config)
    kind = str((config.get("arm") or {}).get("kind", "method"))
    if kind not in ARM_KINDS:
        raise ValueError(f"arm.kind is {kind!r}, not one of: {', '.join(ARM_KINDS)}")
    for block, what in FOREIGN_BLOCKS.items():
        if config.get(block):
            raise ValueError(
                f"config block '{block}' is {what}, which this repository does not carry: "
                "only the published baselines are built here, and an arm that silently "
                "ran without a component it named would be a mislabelled cell.")
    if world is not None:
        raise ValueError("`world=` serves the research repository's oracle arms; there are "
                         "none here, and a method arm must stay GT-blind")
    goal_mask_mode = str((config.get("matcher") or {}).get("goal_mask") or "none")
    if goal_mask_mode != "none":
        # the mode is the RUNNER's business (it computes the mask from the goal depth);
        # the matcher's constructor never sees the key
        config = {**config, "matcher": {k: v for k, v in config["matcher"].items()
                                        if k != "goal_mask"}}
    variant = config["matcher"]["variant"]
    if variant in ("artvs_roi", "artvs_roi_sim"):
        # the ROI matcher takes no scene spec; a cell that `needs` one reads its prompt from
        # it. The word goes to the constructor, the config keeps `scene` - so the label is
        # derived from the setting and the record's `roi_prompt` carries the word. An
        # injected matcher brings its own word (the refusals still run).
        word = resolve_scene_prompt(config, matcher_overrides.pop("scene_spec", None))
        if word is not None and matcher is None:
            matcher_overrides["prompt"] = word
    if variant != "dinobot":
        own_stop = [r for r in (config["stop"]["rule"],
                                *((config.get("episode") or {}).get("record_stops") or []))
                    if r == "dinobot_err"]
        if own_stop:
            raise ValueError(
                "stop rule 'dinobot_err' is DINOBot's own stop: it reads DINOBot's "
                "compute_error (resized-picture pixels or metres), which only the dinobot "
                f"arm produces. On matcher.variant {variant!r} it would compare this arm's "
                "feature error with a threshold written for another quantity")
    if variant == "dinobot":
        from vsbench.vs.baselines.dinobot import build_dinobot_arm  # noqa: PLC0415 - import cycle

        extractor = matcher_overrides.pop("extractor", None)
        if matcher is not None or matcher_overrides:
            raise ValueError(
                "matcher.variant 'dinobot' is a whole arm (front end + look-then-move), not a "
                "matcher. Pass `extractor=` to inject a network; its front end alone is "
                "matcher.variant 'dinobot_corr'")
        return build_dinobot_arm(config, intrinsics=intrinsics, scene=scene, start_id=start_id,
                                 extractor=extractor, goal_mask_mode=goal_mask_mode)
    if variant == "dinobot_corr" and rotmod.rotation_config(config)["mode"] != "none":
        raise ValueError(
            "rotation.mode 'prealign_4way' is a published component of ViT-VS / ART-VS; "
            "DINOBot's front end has no rotation search, and the pre-roll scores rotations "
            "with a square-grid ViT backbone this front end does not have")
    if config["matcher"]["variant"] == "cns_v1":
        from vsbench.vs.baselines.cns import build_cns_arm  # noqa: PLC0415 - avoids an import cycle

        if matcher is not None or matcher_overrides:
            raise ValueError(
                "matcher.variant 'cns_v1' is a whole pipeline, not a matcher - CNS "
                "replaces the control law. Pass `pipeline=` to vsbench.vs.baselines.cns.build_cns_arm "
                "to inject one")
        return build_cns_arm(config, intrinsics=intrinsics, scene=scene, start_id=start_id,
                             goal_mask_mode=goal_mask_mode)
    if matcher is None:
        matcher = build_matcher(config, **matcher_overrides)
    else:
        if matcher_overrides:
            raise ValueError(
                f"matcher_overrides {sorted(matcher_overrides)} were given together with a "
                "ready-made matcher - they would be silently ignored"
            )
        variant = config["matcher"]["variant"]
        want = _variant_class(variant)
        if kind == "method" and not (want is not None and isinstance(matcher, want)):
            # For a variant with no registered matcher class the class check would be
            # skipped, and a ground-truth matcher could then be injected on a `method`
            # arm. Ground truth in a method's column is exactly what this check exists
            # to stop, so an injection on a method arm is allowed ONLY against a
            # variant whose class is registered.
            named = (f"names a {want.__name__}, but a {type(matcher).__name__} was injected"
                     if want is not None else
                     f"has no registered matcher class, so no injected "
                     f"{type(matcher).__name__} can be a method arm")
            raise ValueError(
                f"matcher.variant {variant!r} {named}. Declare `arm: {{kind: oracle}}` "
                f"(a ground-truth matcher) or `arm: {{kind: test}}` (a stub) so the record "
                f"cannot label a cell as something it did not run."
            )
    # ViSP's law is unweighted by definition (`VispIBVSController.from_config` refuses a
    # weight chain by name), so the two weighting guards below are the bench law's alone.
    weighted_law = str(((config.get("controller") or {}).get("law")) or "ibvs") != "visp"
    if (weighted_law
            and ((config.get("controller") or {}).get("weighting") or {}).get("use_margin", True)
            and not getattr(matcher, "provides_margin", True)):
        # `Weighting.use_margin` defaults to True, and a matcher with no second peak
        # fills `margin` with NaN, which `np.clip` propagates into every row of the
        # solve - a silent zero twist wearing a published baseline's name. The
        # classical and tiled arms are unweighted by construction; say so in the config.
        raise ValueError(
            f"controller.weighting.use_margin is on, but {type(matcher).__name__} "
            f"reports no margin (matcher.variant {config['matcher']['variant']!r}). The "
            "published classical and ART-VS baselines are unweighted: set "
            "controller.weighting.use_margin (and use_trust/per_axis) to false."
        )
    if (weighted_law
            and ((config.get("controller") or {}).get("weighting") or {}).get("use_trust", True)
            and not getattr(matcher, "provides_conf_weight", True)):
        # The twin hole. `use_trust` also defaults to True, and it falls back to
        # `rows["conf"]` when nothing has filled `trust` - which on these two matchers
        # is a number that is not this row's: the published classical similarity is
        # row-misaligned (vsbench.vs.baselines.classical, fence 3) and the tiled cosine is
        # tile-local. Unlike a NaN margin it would not even look wrong.
        raise ValueError(
            f"controller.weighting.use_trust is on, but {type(matcher).__name__} "
            f"(matcher.variant {config['matcher']['variant']!r}) carries no weightable "
            "confidence - the published classical similarity is row-misaligned "
            "(vsbench.vs.baselines.classical, fence 3) and the tiled cosine is tile-local. Set "
            "controller.weighting.use_trust to false; these baselines are unweighted.")
    if goal_mask_mode != "none" and not getattr(matcher, "honours_goal_mask", False):
        raise ValueError(
            f"matcher.goal_mask {goal_mask_mode!r} on a {type(matcher).__name__}, which "
            "ignores reset(goal_mask=...): the label would say `objmask` over a run that "
            "never used one. Only the ViT matchers (DenseMatcher, TiledMatcher) honour it.")
    if goal_mask_mode == "none" and getattr(matcher, "requires_goal_mask", False):
        # the twin (2026-10-10): a matcher that cannot run without the bench's mask (the ART-VS
        # dev cell, `roi_mask: objects`) would be handed none by the runner
        raise ValueError(
            f"{type(matcher).__name__} takes its goal mask from the bench (roi_mask: objects) "
            "and the config does not ask for one: state matcher.goal_mask: objects, or the "
            "runner hands it no mask and the cell cannot run")
    # Rotation goes on next: `prealign_4way` attaches the published pre-servo search
    # to the matcher, and the default `none` returns it unchanged.
    matcher = build_rotation(config, matcher)
    checkpoint = None
    controller = build_controller(config, intrinsics=intrinsics)
    stop = build_stop_rule(config, **({"dt": controller.dt}
                                      if config["stop"]["rule"] == "cns_plateau"
                                      and "dt" not in config["stop"] else {}))
    arm = Arm(
        name=str((config.get("arm") or {}).get("name") or arm_label(
            {**config, "matcher": {**config["matcher"], "goal_mask": goal_mask_mode}})),
        matcher=matcher, controller=controller, stop=stop, config=config,
        checkpoint=checkpoint, kind=kind, goal_mask_mode=goal_mask_mode,
    )
    if scene is not None or start_id is not None:
        arm.set_episode(scene, start_id)
    return arm


def episode_settings(config: dict) -> dict:
    """The episode-level knobs, resolved once and recorded on the run.

    ``episode.max_iter`` (600 at gain 2.0; 1500 for long-horizon cells),
    ``episode.divergence_abort``, ``episode.record_stops`` (extra stop rules recorded
    beside the scored one - the velocity-quiet stop is always among them) and
    ``episode.telemetry_extended``, ``episode.outline_iou`` (the instance suite's
    outline-overlap score, off by default) and ``episode.depth_true`` (the depth-true
    pose as a third reference in the record, 2026-09-30, off by default). The step
    budget lives here and **not** under ``stop``, whose block holds rule constants only.
    """
    block = dict((config or {}).get("episode") or {})
    unknown = sorted(set(block) - {"max_iter", "divergence_abort", "record_stops",
                                   "telemetry_extended", "dt", "outline_iou",
                                   "depth_true"})
    if unknown:
        raise ValueError(f"unknown episode keys: {unknown}")
    # `episode.dt` is redundant: the loop integrates at the controller's period, which
    # is the only period the control law was solved for. Accepting a second value and
    # ignoring it would let a config state a
    # control rate the run did not use. State it if you like - it must agree.
    ctrl_dt = float(((config or {}).get("controller") or {}).get("dt", DEFAULT_DT))
    if "dt" in block and abs(float(block["dt"]) - ctrl_dt) > 1e-12:
        raise ValueError(
            f"episode.dt {block['dt']} != controller.dt {ctrl_dt} - the loop integrates at "
            "the controller's period, so a differing episode.dt would be silently ignored")
    record = list(block.get("record_stops") or [])
    if "velocity_quiet" not in record:
        record.append("velocity_quiet")
    scored = config["stop"]["rule"]
    return {
        "max_iter": int(block.get("max_iter", 600)),
        "dt": ctrl_dt,
        "divergence_abort": bool(block.get("divergence_abort", True)),
        "record_stops": [r for r in record if r != scored],
        "telemetry_extended": bool(block.get("telemetry_extended", False)),
        # the instance suite's outline-overlap score (2026-09-23): one extra render per
        # episode and a mask per recorded stop; off by default, so records are unchanged
        "outline_iou": bool(block.get("outline_iou", False)),
        # the depth-true pose as a third reference (2026-09-30): the `*_depth_true` reads and
        # `success_depth_true`, when the world change offers the pose; off by default
        "depth_true": bool(block.get("depth_true", False)),
    }
