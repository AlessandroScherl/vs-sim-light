# vs-sim copy: vssim/bench/config.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Config loading and validation - the rule that makes the random-24 incident impossible.

A run is defined by a YAML file. A handful of fields are **required and have no
default**, because every one of them silently changed a published number at least
once in the old workspace:

===========================  ====================================================
``matcher.variant``          which matcher ran (the random-24 incident)
``matcher.num_landmarks``    how many landmarks the controller steered on
``seed``                     the draw is a function of (scene, start, seed) only
``stop.rule``                cap-scored and stop-scored are different numbers
``scoring.success_rule``     the three success gates disagree by 5x on hard cells
===========================  ====================================================

Two more are required **conditionally**, and neither has a default either:
``matcher.device`` whenever the variant loads an encoder (:data:`BACKBONE_VARIANTS`;
a first station run once flew a whole chain on the CPU because it did not exist), and
``scene.render_warmup_steps`` on a campaign's scene block (:func:`goal_render_warmup`).

Anything else may be added freely - extra fields are allowed and carried through.
Missing or invalid required fields raise :class:`ConfigError` naming the field, and so
do two silent-override traps: a **duplicate key** in a mapping (PyYAML's default is
last-wins, without a word) and a **YAML syntax error** (which would otherwise escape as
a ``yaml.parser.ParserError`` that no caller catches).

In plain words: the config must always say out loud which matcher ran, how many
landmarks it used, what seed it drew with, when it stopped, and what counts as
success. No silent defaults for any of the five.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "vssim-config-1"

MATCHER_VARIANTS = (
    "mutual_nn_seeded_subsample",
    "mutual_nn_all",
    "frozen_landmarks",
    "classical_sift",
    "classical_orb",
    "classical_akaze",
    "artvs",
    "artvs_roi",
    "artvs_roi_sim",  # ART-VS as the release's SIMULATION program runs object mode (2026-10-05)
    "cns_v1",
    "lightglue",
    "roma",
    "romav2",
    "dinobot",        # DINOBot's alignment phase - an ARM, like cns_v1 (vsbench.vs.baselines.dinobot)
    "dinobot_corr",   # its front end alone, as a matcher behind the IBVS law (dinobot_ibvs)
)

#: ``dinobot_err`` is DINOBot's own stop (``while error > ERR_THRESHOLD``), recorded on the
#: DINOBot arm and refused on every other (``vsbench.vs.bench.arms.build_arm``).
#: ``artvs_visual90`` is the ART-VS release's visual-only stop (90 % less mean pixel error than
#: at the first frame), recorded beside the scored stop on the faithful ART-VS arm (2026-10-05).
STOP_RULES = ("cap_only", "dec_plateau", "cns_plateau", "velocity_quiet", "dinobot_err",
              "artvs_visual90")

SUCCESS_RULES = ("predicate_a", "paper_3cm_3deg", "cns_code_0p1d_10deg",
                 "abs_10mm_1deg", "abs_20mm_2deg", "abs_50mm_5deg", "abs_100mm_10deg")

REQUIRED_FIELDS = (
    "matcher.variant",
    "matcher.num_landmarks",
    "seed",
    "stop.rule",
    "scoring.success_rule",
)

#: The matcher variants that load an **encoder**, and for which ``matcher.device`` is
#: therefore required and has no default (R22 correction 3, owner ruling 2026-09-05).
#: E0 - the first station run - flew the whole learned chain on the CPU at 0.21 s/step,
#: twice its budget, because :class:`vsbench.vs.baselines.features.FrozenBackbone` defaulted to
#: ``cpu`` and nothing in the config, the driver or the arm builder said otherwise. The
#: device does not change the numbers (R22 3.7: the twist is identical to 1e-17 across
#: devices) - it changes the compute column protocol v2 section 8 publishes, so it is a
#: setting a run must state rather than inherit. The rotated-bank and staged modes
#: encode through the same backbone (``rotation.GoalBanks`` reuses
#: ``matcher.backbone``), so they are covered by the variant they are built on.
BACKBONE_VARIANTS = (
    "mutual_nn_seeded_subsample",
    "mutual_nn_all",
    "frozen_landmarks",
    "artvs",
    "artvs_roi",      # ART-VS's published object pipeline (LangSAM + LightTrack + ROI tiling)
    "artvs_roi_sim",  # the same, as the release's simulation program runs it (2026-10-05)
)

#: Every variant that loads a network and therefore must state ``matcher.device``: the
#: ViT arms above plus the learned matchers (SuperPoint + LightGlue, RoMa, RoMa v2) and
#: DINOBot's DINO v1 ViT-S/8 (the arm and its front end alone). DINOBot is not a
#: BACKBONE_VARIANT: its encoder is not a ``vsbench.vs.baselines.features`` backbone, so no
#: shared ``backbone=`` may be handed to it, and its label names ``dinobot.model_type``.
DEVICE_VARIANTS = (*BACKBONE_VARIANTS, "lightglue", "roma", "romav2", "dinobot", "dinobot_corr")


class ConfigError(ValueError):
    """A config is missing a required field or holds an invalid value for one."""


_MERGE_TAG = "tag:yaml.org,2002:merge"


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys instead of letting the last one win.

    PyYAML's default is last-wins and silent - exactly the class of silent override
    this loader exists to prevent. ``<<:`` merge keys keep working.
    """


def _mapping_without_duplicates(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        if key_node.tag == _MERGE_TAG:
            continue
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1} - "
                "the second value would silently replace the first"
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping_without_duplicates
)


def _get(config: dict, dotted: str, required: tuple[str, ...] = REQUIRED_FIELDS) -> Any:
    """Fetch ``a.b`` out of nested dicts; raise ConfigError naming the field."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(
                f"required field '{dotted}' is missing - it has no default. "
                f"Required fields: {', '.join(required)}"
            )
        node = node[part]
    if node is None:
        raise ConfigError(f"required field '{dotted}' is null - it has no default")
    return node


def _enum(value: Any, dotted: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ConfigError(
            f"field '{dotted}' is {value!r}, which is not one of: {', '.join(allowed)}"
        )
    return value


def _backbone_device(config: dict) -> str:
    """The device the encoder runs on - stated, never defaulted.

    An arm states it as ``matcher.device`` (``cuda`` on the station, ``cpu`` in the
    fast tier). The file says out loud where the encoder ran; nothing is implied.
    """
    for dotted in ("matcher.device",):
        block, _, key = dotted.partition(".")
        value = (config.get(block) or {}).get(key) if isinstance(config.get(block), dict) \
            else None
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"field '{dotted}' must be a device string ('cpu', 'cuda', 'cuda:1'), "
                f"got {value!r}")
        return value
    raise ConfigError(
        "required field 'matcher.device' is missing - it has no default for a matcher "
        f"variant that loads a network ({', '.join(DEVICE_VARIANTS)}). State 'cuda' "
        "on the station and 'cpu' in the fast tier: the first station run flew the whole "
        "chain on the CPU at twice its budget because this field did not exist.")


#: `matcher.goal_mask`: `none` (the published methods) or `objects` - landmarks drawn
#: from the goal picture's objects only, the floor masked out from the goal depth
#: (`vsbench.vs.labels.object_mask_from_depth`). A different cell with `objmask` in its label.
GOAL_MASK_MODES = ("none", "objects")


def validate(config: dict) -> None:
    """Raise :class:`ConfigError` for the first required field that is absent or invalid."""
    if not isinstance(config, dict):
        raise ConfigError(f"config must be a mapping, got {type(config).__name__}")
    required = REQUIRED_FIELDS
    matcher_block = config.get("matcher")
    gm = matcher_block.get("goal_mask") if isinstance(matcher_block, dict) else None
    if gm is not None:
        _enum(gm, "matcher.goal_mask", GOAL_MASK_MODES)

    if "matcher.variant" in required:
        _enum(_get(config, "matcher.variant", required), "matcher.variant",
              MATCHER_VARIANTS)

    if "matcher.num_landmarks" in required:
        n = _get(config, "matcher.num_landmarks", required)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ConfigError(
                f"field 'matcher.num_landmarks' must be an int >= 1, got {n!r}")

    if "seed" in required:
        seed = _get(config, "seed", required)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigError(f"field 'seed' must be an int, got {seed!r}")
        if not 0 <= seed < 2**32:
            raise ConfigError(
                "field 'seed' must be in [0, 2**32) - numpy's RandomState and OpenCV's "
                f"RNG refuse anything else, got {seed!r}"
            )

    if "matcher.variant" in required and _get(config, "matcher.variant",
                                              required) in DEVICE_VARIANTS:
        _backbone_device(config)

    if "stop.rule" in required:
        _enum(_get(config, "stop.rule", required), "stop.rule", STOP_RULES)

    if "scoring.success_rule" in required:
        rules = _get(config, "scoring.success_rule", required)
        if not isinstance(rules, (list, tuple)) or not rules:
            raise ConfigError(
                f"field 'scoring.success_rule' must be a non-empty list from "
                f"{', '.join(SUCCESS_RULES)}, got {rules!r}"
            )
        for i, rule in enumerate(rules):
            _enum(rule, f"scoring.success_rule[{i}]", SUCCESS_RULES)


def goal_render_warmup(config: dict) -> int:
    """``scene.render_warmup_steps`` - required, no default (**protocol v2.2**).

    The goal frame is the least-converged frame of a run: the first RTX capture after a
    pose change carries render history (R22 section 3.2 - 46 % of pixels differ by more
    than 2 grey levels from a settled capture), and both harnesses captured the goal
    exactly once, first thing. Protocol v2.2 (owner ruling 2026-09-05,
    ``../docs/eval_protocol_v2_2026-09-02.md`` section 12) makes the number of goal
    captures a **stated** field of the campaign's ``scene:`` block - the protocol says
    **8** (the owner's ruling on open issue 19, 2026-09-06; it said 3 when v2.2 was
    adopted on 2026-09-05), and the N-th capture is the one kept. It changes every
    stored token's provenance, so
    it is a protocol line rather than a code default: tokens captured before it (N1-N3.8,
    E0) are labelled "no warm-up", which is what a stated ``1`` means.

    Only the GOAL capture honours it (:class:`vsbench.vs.isaac_scene.IsaacScene`); the
    live frames of an episode are unchanged.
    """
    scene = config.get("scene")
    if not isinstance(scene, dict) or scene.get("render_warmup_steps") is None:
        raise ConfigError(
            "required field 'scene.render_warmup_steps' is missing - it has no default "
            "(protocol v2.2, owner ruling 2026-09-05: the goal capture is taken N times "
            "and the N-th is kept; the protocol says 8). A campaign that does not state "
            "it cannot say what provenance its goal tokens carry.")
    n = scene["render_warmup_steps"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ConfigError(
            f"field 'scene.render_warmup_steps' must be an int >= 1 (1 = no warm-up, the "
            f"pre-v2.2 provenance of every stored token), got {n!r}")
    return int(n)


#: Which of a campaign's goal captures is kept: ``last`` is protocol v2.2's rule,
#: ``first`` is the pre-registered diagnostic (R25 measurement 3 / R26 correction 6).
#: Unlike the count, this one HAS a stated default - ``last``, the protocol's rule -
#: because a campaign that does not say cannot mean anything else; the shipped campaign
#: files state it anyway, and every record carries what ran.
GOAL_CAPTURE_KEEPS = ("last", "first")


def goal_capture_keep(config: dict) -> str:
    """``scene.goal_capture_keep`` - ``last`` (protocol v2.2) unless a cell says else."""
    keep = (config.get("scene") or {}).get("goal_capture_keep", "last")
    if keep not in GOAL_CAPTURE_KEEPS:
        raise ConfigError(
            f"field 'scene.goal_capture_keep' is {keep!r}, which is not one of: "
            f"{', '.join(GOAL_CAPTURE_KEEPS)}. `last` is protocol v2.2's rule; `first` "
            "is the pre-registered eight-renders-keep-the-first measurement and is a "
            "different cell, never a default.")
    return str(keep)


def resolve(config: dict, source: str | Path | None = None) -> dict:
    """Validate and return the fully resolved config (a deep copy, stamped).

    The stamp is ``_schema`` (this module's version) and ``_source`` (the file the
    config came from, or ``None`` when it was built in memory), so a resolved config
    says what it was validated as rather than leaving a reader to guess.
    """
    validate(config)
    resolved = copy.deepcopy(config)
    resolved["_schema"] = SCHEMA_VERSION
    resolved["_source"] = str(source) if source is not None else None
    return resolved


def load_document(path: str | Path) -> dict:
    """Read a YAML mapping under the strict loader - **no** validation.

    The reading half of :func:`load`: the missing file, empty file, syntax-error and
    duplicate-key refusals, and nothing else (the protocol file, whose schema is its
    own, is read through it). :func:`load` is this plus :func:`resolve`.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            raw = yaml.load(fh, Loader=_StrictLoader)
        except ConfigError:
            raise
        except yaml.YAMLError as e:
            raise ConfigError(
                f"config is not valid YAML ({path}): {str(e).splitlines()[0]}"
            ) from e
    if raw is None:
        raise ConfigError(f"config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must hold a mapping, got {type(raw).__name__}: {path}")
    return raw


def load(path: str | Path) -> dict:
    """Read a YAML config, validate it, and return the resolved dict."""
    return resolve(load_document(path), source=Path(path))
