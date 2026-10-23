# vs-sim copy: vssim/baselines/dinobot.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""DINOBot's alignment phase (Di Palo & Johns, ICRA 2024) as an arm, run as published.

Purpose
-------
DINOBot (arXiv 2402.13181) teaches a robot a task from one demonstration in three
phases: **retrieval** (pick the stored "bottleneck" picture most like the live one by
the DINO CLS token), **alignment** (move the wrist camera until the live picture
matches that stored picture) and **replay** (re-run the recorded velocities). Only
the alignment positions a camera, so only the alignment is this baseline; our goal
picture is given, and there is nothing to replay. Spec, taxonomy and the owner's
rulings: ``../docs/dinobot_baseline_spec_2026-09-22.md`` (workspace root).

What the alignment is (and why it is not a matcher + IBVS)
----------------------------------------------------------
From the goal and the live RGB-D pictures it finds eight DINO best-buddy
correspondences, lifts both ends to 3-D with the depth of **both** pictures, fits the
one rigid transform that maps the goal points onto the live points (Kabsch / SVD),
moves the camera by that **whole** transform, looks again, and repeats while the two
point lists are more than a threshold apart. There is no velocity law, no gain, no
interaction matrix: iterated look-then-move 3-D alignment, a position-based scheme
with a unit step. So, like :mod:`vsbench.vs.baselines.cns`, it fills the
:class:`vsbench.vs.bench.arms.Arm` shape with its own front end
(:class:`DinoBotCorrespondences`, the matcher slot) and its own move rule
(:class:`DinoBotAlign`, the controller slot), and the runner, the record, the films
and the scoring take it unchanged. Because the front end *is* a real
:class:`~vsbench.vs.baselines.match.CorrespondenceMatcher` (variant ``dinobot_corr``), the
bench's IBVS law can be put behind the same eight pairs (the ``dinobot_ibvs`` cell):
the controlled comparison "is it the correspondences or the look-then-move rule?".

Ported from
-----------
- **The DINOBot gist**, ``normandipalo/fbc21f23606fbe3d407e22c363cb134e``
  ``dinobot_oneshot_il.py``, final revision ``60d5adb`` (2024-05-21), vendored at
  ``tests/fixtures/reference/dinobot_oneshot_il.py``: the hyper-parameters
  (:data:`PUBLISHED`), ``find_transformation`` -> :func:`kabsch` (verbatim),
  ``compute_error`` -> :func:`compute_error` (verbatim), ``ERR_THRESHOLD = 50`` and
  the loop ``while error > ERR_THRESHOLD`` -> :class:`DinoBotErr`, and the direction
  of the fit (revision 3 on: goal -> live, i.e. ``find_correspondences(rgb_bn,
  rgb_live, ...)``) with the metric 3-D lift of revision 4 (owner decision D2 (a)).
- **ShirAmir/dino-vit-features** @ ``1779649`` (2022-09-09, MIT), the library the gist
  "runs inside", vendored (excerpts) at ``tests/fixtures/reference/dino_vit_features/``:
  ``ViTExtractor.patch_vit_resolution`` + ``_fix_pos_enc`` -> :func:`patch_vit_resolution`
  (stride 4, bicubic position-embedding interpolation, verbatim arithmetic);
  ``_get_hook`` / ``_extract_features`` / ``extract_descriptors`` ->
  :meth:`DinoV1Extractor.encode` (the key facet of block 9, CLS dropped, heads flattened
  in the library's order, ``_log_bin`` -> :func:`vsbench.vs.baselines.features.log_bin` with
  ``hierarchy=2``); ``extract_saliency_maps`` -> the saliency half of
  :meth:`DinoV1Extractor.encode` (block 11's CLS attention, heads 0/2/4/5, min-max);
  ``preprocess`` -> :meth:`DinoV1Extractor.preprocess` (torchvision ``Resize(224,
  LANCZOS)`` on the PIL picture + ToTensor + Normalize, reproduced with Pillow and
  :func:`vsbench.vs.baselines.features.normalize_imagenet`); ``find_correspondences`` from
  "calculate similarity" to the returned point lists -> :func:`select_pairs`,
  arithmetic unchanged, including its foreground-mask quirk (below).

Filled in (the gist's placeholders and run-time environment, with the bench's rules)
-----------------------------------------------------------------------------------
``camera_get_rgbd`` -> the scene's frame; ``project_to_3d`` -> the pinhole
back-projection with the frame's intrinsics (:func:`backproject`), depth looked up by
truncation (:func:`vsbench.vs.bench.arms.depth_lookup`, the bench's one lookup rule);
``robot_move(t, R)`` -> one bench step whose twist integrates to exactly that
transform (:func:`jump_twist`); one model instance for the whole run and the goal
encoded once per episode (the library rebuilds the network and re-encodes the goal on
every call - same numbers, asserted by a test); no file round trip (revision 5 wrote
the pictures to disk in an unknown format); ``n_init=10`` stated explicitly, because
that is what ``KMeans(n_clusters, random_state=0)`` meant under the library's pinned
scikit-learn 1.0.1 (owner decision D3); the K-means pinned to one thread
(:data:`KMEANS_THREADS` - scikit-learn's float32 result moved between identical calls at
twelve threads, so the pinned thread count is what makes a re-run pick the same pairs).

Deviations, recorded (each replaces a crash or an unexecutable command)
------------------------------------------------------------------------
1. **Invalid depth is dropped, not replaced by 0.5 m** (D6): a pair whose depth is
   ``<= 0``, ``> 10 m`` or non-finite on either side leaves the fit and is counted
   (``n_dropped_depth``). The IBVS fallback would put a false 3-D point into a rigid
   fit.
2. **Fewer than** ``min_pairs = 3`` **pairs -> no motion**, ``degenerate = True`` (D6).
   The published code crashes in ``KMeans`` on zero pairs (the port returns no pairs
   there) and fits nonsense on one or two.
3. **det(R) < 0 -> no motion**, ``reflection = True`` (D6). The published formula has
   no reflection check, so it can return an improper matrix no robot can execute.
   **On a planar scene this is not rare**: eight coplanar points leave the third
   singular direction undetermined, and random coplanar sets come back improper about
   half the time (49.7 % of 2000 on the laptop; ``tests/fast/test_dinobot.py`` checks
   30-70 %). ``reflection: fix`` (the
   Sorkine-Hornung correction of the paper's reference [32]) is a stated option and a
   different cell (``reffix`` in the label).
4. **An empty K-means cluster** (duplicate points: scikit-learn then finds fewer
   distinct clusters) contributes no pair (``n_empty_clusters``); the library would
   index with ``-inf`` there.
5. **The redundant outer** ``for k in range(n_clusters)`` **loop** of the selection is one
   pass, and the float-typed index arrays are integers (``//`` instead of ``int(i / w)``):
   the same pairs, proven by the parity test.
6. **The similarity is one matrix product**, not the library's per-row
   ``CosineSimilarity`` loop (``chunk_cosine_sim``), which costs 0.4 s per row on the
   laptop CPU - 37 minutes a frame (measured) - and by its memory traffic some 6 s a frame
   on an RTX 4090 (an estimate, not measured). The two differ
   only in the float summation order (2e-6 on a real pair), and every argmax the library's
   result reads whose runner-up is within :data:`TIE_TOL` is decided on the library's own
   row, computed by the library's own call on descriptors laid out as the library lays them
   out - so the best buddies and everything after them are the library's
   (:func:`best_buddies`; a device where the premise fails falls back to the whole loop).

Not built
---------
The retrieval phase, the replay phase, the paper's 4-DOF tabletop projection, the
February variants (revision 1: ViT-B/8 and the pixel + depth fit, owner decision D2),
the D435's depth noise and holes (a benchmark-wide perturbation question, not an
arm's), the file round trip, the 4-way pre-roll and the ``dec`` stop (refused by name:
DINOBot has no rotation search and no gain for a schedule to scale).

Fences
------
- **The foreground quirk is ported, not corrected**: a goal token is kept when it is
  the nearest neighbour of *any* foreground live token (``fg_mask2_new_coors =
  nn_2[fg_mask2]``), not only of its own buddy.
- **The fit's direction.** :func:`kabsch` is ``find_transformation(X=goal, Y=live)``,
  so ``Y ~ R X + t``: (R, t) is the goal camera's pose in the live camera's frame -
  the move the camera must make in its own frame. A camera 5 cm to the right sees
  every point 5 cm further left and gets ``t = (-0.05, 0, 0)``.
- **One published iteration = one bench step**: ``v = t / dt``, ``w = axis_angle(R) /
  dt``, and the bench's integrator (``pos += R_cam v dt``, ``R_cam <- R_cam Rot(w dt)``)
  lands exactly on ``T_cam (R, t)``. ``step_fraction != 1`` is a damped jump
  ``(R^a, a t)`` and a different cell (``jump0.5``).
- **The published stop fires after the move** (:class:`DinoBotErr`): the gist computes
  the error on the observation *before* the move and exits the loop after executing
  that move, so the rule latches on the step after the first reading under the
  threshold - the pose it records is where DINOBot ends.
- **Weights are never downloaded**: the network comes from the torch.hub cache
  (``facebookresearch_dino_main`` + ``checkpoints/dino_deitsmall8_pretrain.pth``),
  checked before ``torch.hub.load`` is called; a clear error names the staging command
  otherwise. scikit-learn is imported inside :func:`select_pairs`, never at import.

Golden tests: ``tests/fast/test_dinobot.py`` (the vendored library and gist run beside
the port on every test; a stub extractor and a tiny DINO-shaped network, no weights);
``tests/slow/test_dinobot_parity.py`` (the real library and the real weights).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import math
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vsbench.vs.baselines.control import (
    DEFAULT_DT,
    DEPTH_MAX_M,
    Intrinsics,
    StopObservation,
    StopVerdict,
    Twist,
    build_stop_rule,
)
from vsbench.vs.baselines.features import log_bin, normalize_imagenet
from vsbench.vs.baselines.match import CorrespondenceMatcher, Matches

# ------------------------------------------------------------------ constants

#: The torch.hub entry of the published network and where torch.hub caches it.
HUB_REPO = "facebookresearch/dino:main"
HUB_DIR_NAME = "facebookresearch_dino_main"
#: The weight file ``hubconf.dino_vits8`` loads (``load_state_dict_from_url``).
CHECKPOINTS = {"dino_vits8": "dino_deitsmall8_pretrain.pth"}
#: ``hubconf.dino_vits8``: ``vits.__dict__["vit_small"](patch_size=8, num_classes=0)`` -
#: the constructor name and patch size the cached repository's ``vision_transformer.py``
#: is called with (:meth:`DinoV1Extractor._load` builds it directly, without torch.hub's
#: top-level import, which the Isaac application breaks).
HUB_ARCHITECTURES = {"dino_vits8": ("vit_small", 8)}
#: Owner decision D2 (a): the final gist revision's ViT-S/8. The library's saliency map
#: exists only for it (``extract_saliency_maps`` asserts ``model_type == "dino_vits8"``),
#: so the February ViT-B/8 cell is not built.
MODEL_TYPES = ("dino_vits8",)
#: Descriptor facets ``extract_descriptors`` accepts (``'attn'`` is not a descriptor).
FACETS = ("key", "query", "value", "token")
#: ``bin_hierarchy``: 0 is the library's ``bin=False``, 2 its ``bin=True`` (``_log_bin``'s
#: default hierarchy, the only one ``extract_descriptors`` reaches).
BIN_HIERARCHIES = (0, 2)
#: ``extract_saliency_maps``: block 11's attention, CLS row, these heads, averaged.
SALIENCY_BLOCK = 11
SALIENCY_HEADS = (0, 2, 4, 5)
#: ImageNet statistics - the library picks them for any ``"dino" in model_type``.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#: ``torch.nn.CosineSimilarity``'s default ``eps`` (the library's call leaves it).
COSINE_EPS = 1e-8
#: Near-tie band of the fast similarity (deviation 6 of the module docstring). The matrix
#: product and the library's per-row cosine differ only in the float summation order:
#: measured on the real u1/u2 fixture pair (laptop CPU, 2026-09-22) the largest entry
#: difference of a row is 2.0e-6 at the median and 3.5e-6 at the worst of 160 rows. Every
#: argmax the library's result depends on whose runner-up is closer than this band is
#: decided on the library's own row; half the band (2.5e-5) is 7x the worst measured
#: difference, and a frame whose recomputed rows moved by more than half the band falls
#: back to the library's whole loop.
TIE_TOL = 5e-5
#: What ordered the selection, for the record's ``tie_order`` column.
TIE_ORDER = "kmeans_saliency_first"
#: Threads the K-means runs on. scikit-learn's float32 Lloyd sums in a thread-dependent
#: order: at the laptop's 12 threads the inertia of the same fit moved in its 7th digit
#: between identical calls (2026-09-22), and the best of ``n_init`` runs is picked by
#: inertia. One thread is deterministic - and, on the pools seen, as fast or faster.
KMEANS_THREADS = 1

#: Everything the gist and the library fix, with where it comes from.
PUBLISHED: dict[str, Any] = {
    "model_type": "dino_vits8",   # gist (rev. 3+)
    "load_size": 224,             # gist; the smaller edge (torchvision Resize(int))
    "stride": 4,                  # gist
    "layer": 9,                   # gist
    "facet": "key",               # gist
    "bin_hierarchy": 2,           # gist `bin=True` -> `_log_bin(x)` with hierarchy 2
    "saliency_thresh": 0.05,      # gist `thresh`
    "num_pairs": 8,               # gist
    "kmeans_n_init": 10,          # scikit-learn 1.0.1's default (the library's env)
    "step_fraction": 1.0,         # the whole transform per iteration
    "err_threshold": 50.0,        # gist ERR_THRESHOLD
}
#: The deviation guards (D6) and the stated error space (D4), at their ruled values.
MIN_PAIRS = 3
REFLECTIONS = ("none", "fix")
ERR_SPACES = ("pixel_load", "metric")
#: D4: the constant 50 in the space it was written for (revision 1's pixel space), and
#: the stated metric alternative.
ERR_THRESHOLDS = {"pixel_load": 50.0, "metric": 0.02}

#: The ``dinobot:`` block: the front end (shared by the arm and the ``dinobot_ibvs``
#: cell) and the look-then-move rule (the arm's alone).
FRONT_END_KEYS = ("model_type", "load_size", "stride", "layer", "facet", "bin_hierarchy",
                  "saliency_thresh", "kmeans_n_init")
ALIGN_KEYS = ("step_fraction", "min_pairs", "reflection", "err_space", "err_threshold")
DINOBOT_CONFIG_KEYS = FRONT_END_KEYS + ALIGN_KEYS

#: Controller keys that are IBVS machinery, refused by name on the arm (``dt`` - the
#: loop's period - is the only controller key it keeps).
_REFUSED_CONTROLLER_KEYS = {
    "gain": "DINOBot has no gain - its published step is the whole transform "
            "(`dinobot.step_fraction`, 1.0)",
    "weighting": "the Kabsch fit is unweighted by publication",
    "solve_unrounded": "there is no image-space solve to round",
    "ema_alpha": "the gist smooths nothing",
    "min_matches": "the pair floor is `dinobot.min_pairs`",
    "law": "the move rule IS the arm - a velocity law behind these pairs is the "
           "`dinobot_ibvs` cell (matcher.variant: dinobot_corr)",
}
#: Matcher keys the arm reads; anything else would be a setting the record never carries.
_ARM_MATCHER_KEYS = ("variant", "num_landmarks", "device", "goal_mask")


# ---------------------------------------------------------- the block, resolved
def resolve_block(config: dict, *, align: bool = True) -> dict:
    """The ``dinobot:`` block with the published values filled in, validated.

    ``align=False`` is the ``dinobot_ibvs`` cell's view: the look-then-move keys belong
    to the arm and are refused there (the cell runs the bench's IBVS law). An unknown key
    is refused rather than ignored - a key that is neither built nor refused is a
    setting the record would not carry (the CNS rule).
    """
    block = dict((config or {}).get("dinobot") or {})
    unknown = sorted(set(block) - set(DINOBOT_CONFIG_KEYS))
    if unknown:
        raise ValueError(
            f"unknown dinobot config keys: {unknown} - the block holds only "
            f"{', '.join(DINOBOT_CONFIG_KEYS)}. A key that is neither refused nor built "
            "is a setting the record would not carry")
    if not align:
        stray = sorted(set(block) & set(ALIGN_KEYS))
        if stray:
            raise ValueError(
                f"dinobot keys {stray} belong to the look-then-move arm (matcher.variant "
                "dinobot); the dinobot_corr cell runs the bench's IBVS law behind the same "
                "pairs and would silently ignore them")
    out = {k: block.get(k, PUBLISHED[k]) for k in FRONT_END_KEYS}
    if out["model_type"] not in MODEL_TYPES:
        raise ValueError(
            f"dinobot.model_type {out['model_type']!r}: only {MODEL_TYPES} is built (owner "
            "decision D2 (a), the final gist revision). The library's saliency map exists "
            "for dino_vits8 alone (`extract_saliency_maps` asserts it), so the February "
            "ViT-B/8 cell would run a step the library never validated")
    ls = out["load_size"]
    if isinstance(ls, (list, tuple)):
        if len(ls) != 2 or not all(isinstance(v, int) and v > 0 for v in ls):
            raise ValueError(f"dinobot.load_size {ls!r}: an int (the smaller edge) or [h, w]")
        out["load_size"] = [int(ls[0]), int(ls[1])]
    elif isinstance(ls, bool) or not isinstance(ls, int) or ls <= 0:
        raise ValueError(f"dinobot.load_size {ls!r}: an int (the smaller edge) or [h, w]")
    if out["stride"] not in (1, 2, 4, 8):
        raise ValueError(f"dinobot.stride {out['stride']!r} must divide the patch size 8")
    if not (isinstance(out["layer"], int) and 0 <= out["layer"] <= 11):
        raise ValueError(f"dinobot.layer {out['layer']!r} is not a block of ViT-S/8 (0-11)")
    if out["facet"] not in FACETS:
        raise ValueError(f"dinobot.facet {out['facet']!r} is not one of {FACETS}")
    if out["bin_hierarchy"] not in BIN_HIERARCHIES:
        raise ValueError(
            f"dinobot.bin_hierarchy {out['bin_hierarchy']!r}: 2 (the library's bin=True) or "
            "0 (bin=False) - `extract_descriptors` reaches no other")
    out["saliency_thresh"] = float(out["saliency_thresh"])
    n_init = out["kmeans_n_init"]
    if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 1:
        raise ValueError(f"dinobot.kmeans_n_init {n_init!r} must be an int >= 1, stated")
    if not align:
        return out
    space = block.get("err_space", "pixel_load")
    if space not in ERR_SPACES:
        raise ValueError(f"dinobot.err_space {space!r} is not one of {ERR_SPACES}")
    out["err_space"] = space
    out["err_threshold"] = float(block.get("err_threshold", ERR_THRESHOLDS[space]))
    if not out["err_threshold"] > 0:
        raise ValueError("dinobot.err_threshold must be > 0")
    out["step_fraction"] = float(block.get("step_fraction", PUBLISHED["step_fraction"]))
    if not 0.0 < out["step_fraction"] <= 1.0:
        raise ValueError(f"dinobot.step_fraction {out['step_fraction']} must be in (0, 1]")
    out["min_pairs"] = block.get("min_pairs", MIN_PAIRS)
    if isinstance(out["min_pairs"], bool) or not isinstance(out["min_pairs"], int) \
            or out["min_pairs"] < MIN_PAIRS:
        raise ValueError(
            f"dinobot.min_pairs {out['min_pairs']!r} must be an int >= {MIN_PAIRS}: a rigid "
            "fit needs three non-collinear points")
    out["reflection"] = block.get("reflection", "none")
    if out["reflection"] not in REFLECTIONS:
        raise ValueError(f"dinobot.reflection {out['reflection']!r} is not one of {REFLECTIONS}")
    return out


def _front_end_tokens(fe: dict) -> list[str]:
    """Label tokens for every front-end knob off its published value (never parsed)."""
    tokens = []
    ls = fe["load_size"]
    if ls != PUBLISHED["load_size"]:
        tokens.append(f"load{ls[0]}x{ls[1]}" if isinstance(ls, list) else f"load{ls}")
    for key, short in (("stride", "stride"), ("layer", "layer"), ("facet", "facet-"),
                       ("bin_hierarchy", "bin"), ("saliency_thresh", "sal"),
                       ("kmeans_n_init", "ninit")):
        if fe[key] != PUBLISHED[key]:
            tokens.append(f"{short}{fe[key]:g}" if isinstance(fe[key], float)
                          else f"{short}{fe[key]}")
    return tokens


def dinobot_label(config: dict) -> str:
    """The arm's label, **derived** from its settings: ``dinobot_vits8/k8/jump1/cns_plateau``.

    ``step_fraction`` -> ``jump<f>``; ``reflection: fix`` -> ``reffix``; ``err_space:
    metric`` -> ``errm``; a threshold off its space's stated value -> ``thr<v>``;
    ``min_pairs`` off 3 -> ``minp<n>``; any front-end knob off the published value ->
    its own token; ``matcher.goal_mask: objects`` -> ``objmask``; then the scored stop.
    """
    b = resolve_block(config)
    m = config["matcher"]
    parts = [f"dinobot_{b['model_type'].removeprefix('dino_')}", f"k{int(m['num_landmarks'])}",
             f"jump{b['step_fraction']:g}", *_front_end_tokens(b)]
    if b["min_pairs"] != MIN_PAIRS:
        parts.append(f"minp{b['min_pairs']}")
    if b["reflection"] == "fix":
        parts.append("reffix")
    if b["err_space"] == "metric":
        parts.append("errm")
    if b["err_threshold"] != ERR_THRESHOLDS[b["err_space"]]:
        parts.append(f"thr{b['err_threshold']:g}")
    if str(m.get("goal_mask") or "none") == "objects":
        parts.append("objmask")
    parts.append(str(config["stop"]["rule"]))
    return "/".join(parts)


def dinobot_corr_tag(config: dict) -> str:
    """The encoder tag of the ``dinobot_corr`` cell's label (``dino_vits8[+tokens]``)."""
    fe = resolve_block(config, align=False)
    return "+".join([fe["model_type"], *_front_end_tokens(fe)])


# ---------------------------------------------------------------- the network
def _fix_pos_enc(patch_size: int, stride_hw: tuple[int, int]):
    """``ViTExtractor._fix_pos_enc`` (upstream ``extractor.py`` L83-116), same arithmetic.

    The two ``assert`` lines are ``RuntimeError`` here (an assert vanishes under ``-O``);
    the interpolation call and every number in it are upstream's.

    The position embeddings of the 28 x 28 training grid, interpolated (bicubic, with the
    DINO ``+ 0.1`` against a floating-point off-by-one) to the ``1 + (H - p) // s`` x
    ``1 + (W - p) // s`` grid a stride-``s`` patch embedding produces.
    """
    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        if w0 * h0 != npatch:
            raise RuntimeError(f"got wrong grid size for {h}x{w} with patch_size {patch_size} "
                               f"and stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}")
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        side = int(math.sqrt(N))
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, side, side, dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False, recompute_scale_factor=False,
        )
        if not (int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]):
            raise RuntimeError("the interpolated position grid is not the token grid")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding


def patch_vit_resolution(model: torch.nn.Module, stride: int) -> torch.nn.Module:
    """``ViTExtractor.patch_vit_resolution`` (upstream L119-138): stride ``s`` patches.

    Sets the patch-embedding convolution's stride and swaps the model's position-encoding
    interpolation for :func:`_fix_pos_enc`'s. A no-op at ``stride == patch_size``.
    """
    patch_size = model.patch_embed.patch_size
    if stride == patch_size:  # nothing to do
        return model
    stride = torch.nn.modules.utils._pair(stride)
    if not all((patch_size // s_) * s_ == patch_size for s_ in stride):
        raise ValueError(f"stride {stride} should divide patch_size {patch_size}")
    model.patch_embed.proj.stride = stride
    model.interpolate_pos_encoding = types.MethodType(_fix_pos_enc(patch_size, stride), model)
    return model


@dataclass
class DinoFeatures:
    """One picture through the library's chain: what :func:`select_pairs` reads.

    ``desc`` ``(N, D)`` float32 (contiguous; the library holds the same numbers as a
    transposed view - :func:`library_layout`), ``saliency`` ``(N,)`` float32 in
    ``[0, 1]``, ``grid`` the token lattice ``(rows, cols)``, ``load_hw`` the resized
    picture, ``p`` / ``stride`` the patch geometry (points are ``(idx - 1) * stride +
    stride + p // 2``), ``binned`` whether ``desc`` is log-binned (it decides the
    library's layout).
    """

    desc: torch.Tensor
    saliency: torch.Tensor
    grid: tuple[int, int]
    load_hw: tuple[int, int]
    p: int
    stride: tuple[int, int]
    binned: bool
    forward_ms: float = 0.0


class DinoV1Extractor:
    """DINO v1 ViT-S/8 at stride 4 - the library's ``ViTExtractor``, one instance per run.

    ``model=`` injects a built module with DINO v1's structure (``patch_embed`` with
    ``patch_size`` and ``proj``, ``pos_embed``, ``blocks[i].attn`` with ``qkv``,
    ``num_heads`` and ``attn_drop``) - the fast tier's tiny network, the library's own
    escape hatch. Otherwise the network comes from the torch.hub cache and **only** from
    it: :meth:`_load` checks the repository and the weight file first and refuses with
    the staging command, so ``torch.hub.load`` never has anything to fetch.
    """

    def __init__(self, model_type: str = "dino_vits8", stride: int = 4, *,
                 device: str | torch.device, model: torch.nn.Module | None = None) -> None:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type {model_type!r} is not one of {MODEL_TYPES}")
        self.model_type = model_type
        self.device = torch.device(device)
        #: the facts a record states about the weights - None for an injected model
        self.weights: str | None = None
        self.weights_md5: str | None = None
        self.hub_dir: str | None = None
        self.hub_commit: str | None = None
        self.hub_code_md5: str | None = None
        model = model if model is not None else self._load()
        self.model = patch_vit_resolution(model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.model.requires_grad_(False)
        self.p = int(self.model.patch_embed.patch_size)
        self.stride = tuple(int(s) for s in self.model.patch_embed.proj.stride)
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD

    # ------------------------------------------------------------- loading
    def _load(self) -> torch.nn.Module:
        """``hubconf.dino_vits8(pretrained=True)`` from the torch.hub cache, built directly.

        The two files are checked first (a clear error names the staging command). The
        network is then built from the cached repository's own ``vision_transformer.py``
        with its ``utils.py`` beside it - the two files ``torch.hub.load`` runs - and the
        cached checkpoint is loaded ``strict=True``, exactly as ``hubconf.dino_vits8``
        does (``tests/fast/test_dinobot.py`` proves the two networks equal bit for bit
        where the cache exists). **torch.hub's own import is not used**: it imports the
        repository as top-level modules (``import vision_transformer``, and inside it
        ``from utils import trunc_normal_``), and inside the Isaac Sim application a
        module named ``utils`` already exists - OpenCV's bundled ``cv2/utils`` package,
        whose directory Isaac puts on ``sys.path`` - so the hub's import failed there
        with ``cannot import name 'trunc_normal_' from 'utils'`` on every cell of the
        first station rig check (2026-09-23) while the same call in a plain interpreter
        worked. Here ``vision_transformer`` is executed with the repository's ``utils``
        bound under ``sys.modules['utils']`` for the duration of that one import, and
        whatever ``utils`` meant before is restored afterwards.
        """
        hub = Path(torch.hub.get_dir())
        repo = hub / HUB_DIR_NAME
        ckpt = hub / "checkpoints" / CHECKPOINTS[self.model_type]
        missing = [str(p) for p in (repo / "hubconf.py", ckpt) if not p.is_file()]
        if missing:
            raise RuntimeError(
                f"DINO v1 {self.model_type} is not in the torch.hub cache of this machine "
                f"(missing: {', '.join(missing)}) and this repository never downloads "
                "weights. Stage it once, by hand, with the network on: `python -c \"import "
                f"torch; torch.hub.load('{HUB_REPO}', '{self.model_type}')\"` (TORCH_HOME "
                "decides the cache), and record the code's commit beside it: `git ls-remote "
                "https://github.com/facebookresearch/dino refs/heads/main | cut -f1 > "
                f"{repo}/COMMIT`.")
        vt_path, utils_path = repo / "vision_transformer.py", repo / "utils.py"
        absent = [str(p) for p in (vt_path, utils_path) if not p.is_file()]
        if absent:
            raise RuntimeError(
                f"the cached DINO repository at {repo} lacks {', '.join(absent)} - not the "
                "facebookresearch/dino checkout torch.hub stages; re-stage it")
        arch, patch = HUB_ARCHITECTURES[self.model_type]
        try:
            utils_mod = _import_file("vssim_dino_hub_utils", utils_path)
            saved = sys.modules.get("utils")
            sys.modules["utils"] = utils_mod              # what `vision_transformer` imports
            try:
                vits = _import_file("vssim_dino_hub_vision_transformer", vt_path)
            finally:
                if saved is None:
                    sys.modules.pop("utils", None)
                else:
                    sys.modules["utils"] = saved
            model = vits.__dict__[arch](patch_size=patch, num_classes=0)   # hubconf's line
            state = torch.load(str(ckpt), map_location="cpu")
            model.load_state_dict(state, strict=True)                     # hubconf's line
        except Exception as e:  # noqa: BLE001 - the message is the point
            raise RuntimeError(
                f"could not build DINO v1 {self.model_type} from the torch.hub cache at "
                f"{repo} ({type(e).__name__}: {e})") from e
        self.weights = str(ckpt)
        self.weights_md5 = _md5(ckpt)
        self.hub_dir = str(repo)
        commit = repo / "COMMIT"
        self.hub_commit = commit.read_text().strip() if commit.is_file() else None
        self.hub_code_md5 = hashlib.md5(
            b"".join((repo / f).read_bytes()
                     for f in ("hubconf.py", "vision_transformer.py", "utils.py")
                     if (repo / f).is_file())).hexdigest()
        return model

    # ---------------------------------------------------------- preprocess
    def preprocess(self, rgb: np.ndarray, load_size) -> torch.Tensor:
        """``ViTExtractor.preprocess`` on an array: ``(1, 3, H, W)`` float32, contiguous.

        torchvision ``Resize(load_size, LANCZOS)`` on the PIL picture - an int resizes the
        smaller edge (``int(load_size * long / short)`` for the other: 1920 x 1080 ->
        **398 x 224**), a pair is ``(h, w)`` exactly - which is Pillow's own
        ``resize((w, h), LANCZOS)``; then ToTensor + Normalize(ImageNet), which is
        ``(x / 255 - mean) / std`` (:func:`vsbench.vs.baselines.features.normalize_imagenet`,
        bit-identical, asserted against torchvision by a test). Contiguous NCHW - the
        fence of :meth:`vsbench.vs.baselines.features.PublishedDinoV2Backbone.preprocess`.
        """
        from PIL import Image

        a = np.asarray(rgb)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] < 3:
            raise ValueError(f"expected uint8 HxWx3 RGB, got {a.dtype} {a.shape}")
        pil = Image.fromarray(np.ascontiguousarray(a[..., :3]))
        w, h = pil.size
        if isinstance(load_size, (list, tuple)):
            new_h, new_w = int(load_size[0]), int(load_size[1])
        else:
            short, long = (w, h) if w <= h else (h, w)
            new_short, new_long = int(load_size), int(int(load_size) * long / short)
            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        if (new_w, new_h) != (w, h):
            pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().contiguous()[None]
        return normalize_imagenet(x, self.mean, self.std).to(self.device)

    # --------------------------------------------------------------- encode
    @torch.no_grad()
    def encode(self, rgb: np.ndarray, *, load_size, layer: int, facet: str,
               bin_hierarchy: int) -> DinoFeatures:
        """One picture -> descriptors + saliency, in ONE forward pass.

        The library runs two forwards (``extract_descriptors``, then
        ``extract_saliency_maps``); hooks only read, so both captures from one pass are
        the same numbers.

        - ``facet`` ``key`` / ``query`` / ``value`` (``_get_hook`` L169-183): a hook on
          ``blocks[layer].attn`` recomputes ``qkv`` from the block's normalised input
          and keeps index 0 / 1 / 2 - ``B x heads x tokens x 64``; ``token`` hooks the
          block's output. CLS dropped (``x[:, :, 1:, :]``), heads and channels flattened
          as ``x.permute(0, 2, 3, 1).flatten(-2, -1)`` (channel-major, head-minor), then
          :func:`vsbench.vs.baselines.features.log_bin` at ``bin_hierarchy`` (2: 17 bins,
          6528-d at ViT-S/8).
        - saliency (``extract_saliency_maps`` L304-318): a hook on
          ``blocks[11].attn.attn_drop`` keeps the attention probabilities; the CLS row,
          heads 0/2/4/5, mean, min-max normalised per picture.
        """
        t0 = time.monotonic()
        batch = self.preprocess(rgb, load_size)
        _, _, H, W = batch.shape
        feats: dict[str, torch.Tensor] = {}
        blocks = self.model.blocks
        handles = []
        if facet == "token":
            handles.append(blocks[layer].register_forward_hook(
                lambda _m, _i, out: feats.__setitem__("desc", out)))
        else:
            idx = {"query": 0, "key": 1, "value": 2}[facet]

            def _qkv_hook(module, inputs, _output):
                x = inputs[0]
                b, n, c = x.shape
                qkv = module.qkv(x).reshape(b, n, 3, module.num_heads,
                                            c // module.num_heads).permute(2, 0, 3, 1, 4)
                feats["desc"] = qkv[idx]                           # B x h x t x d

            handles.append(blocks[layer].attn.register_forward_hook(_qkv_hook))
        handles.append(blocks[SALIENCY_BLOCK].attn.attn_drop.register_forward_hook(
            lambda _m, _i, out: feats.__setitem__("attn", out)))
        try:
            self.model(batch)
        finally:
            for h in handles:
                h.remove()
        grid = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        x = feats["desc"]
        if facet == "token":
            x = x.unsqueeze(1)                                     # B x 1 x t x d
        x = x[:, :, 1:, :]                                         # remove cls token
        tokens = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)   # B x (t-1) x (d h)
        desc = log_bin(tokens, grid, hierarchy=bin_hierarchy) if bin_hierarchy else tokens
        attn = feats["attn"]                                       # B x h x t x t
        cls_attn = attn[:, list(SALIENCY_HEADS), 0, 1:].mean(dim=1)
        mins, maxs = cls_attn.min(dim=1)[0], cls_attn.max(dim=1)[0]
        sal = (cls_attn - mins) / (maxs - mins)
        return DinoFeatures(desc=desc[0].float().contiguous(), saliency=sal[0].float(),
                            grid=(int(grid[0]), int(grid[1])), load_hw=(int(H), int(W)),
                            p=self.p, stride=self.stride, binned=bool(bin_hierarchy),
                            forward_ms=(time.monotonic() - t0) * 1000.0)


def _import_file(name: str, path: Path):
    """Execute one source file as the module ``name`` (registered in ``sys.modules``)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path} as a module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- matching
def library_layout(desc: torch.Tensor, grid: tuple[int, int], binned: bool) -> torch.Tensor:
    """``(N, D)`` descriptors -> ``(1, 1, N, D)`` with the strides the library's own have.

    ``_log_bin`` returns ``bin_x.flatten(-2, -1).permute(0, 2, 1).unsqueeze(1)`` - a
    transposed **view** (element ``(i, d)`` at offset ``d * N + i``) - and the unbinned
    path returns a contiguous copy. ``torch.nn.CosineSimilarity`` reduces over the last
    dimension, and how it walks that dimension follows the layout, so a row recomputed
    to decide a near-tie must see the library's layout to be the library's row.
    """
    n, d = desc.shape
    if not binned:
        return desc.contiguous().reshape(1, 1, n, d)
    return desc.t().contiguous().reshape(1, d, n).permute(0, 2, 1).unsqueeze(1)


@contextlib.contextmanager
def _full_fp32_matmul():
    """No TF32 / reduced-precision float32 product while the similarity is formed."""
    prec = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prec)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _library_rows(x: torch.Tensor, y: torch.Tensor, rows) -> tuple[dict, str]:
    """Rows of ``chunk_cosine_sim(x, y)[0, 0]``, each the library's own number.

    ``x`` / ``y`` are ``(1, 1, N, D)`` in the library's layout (:func:`library_layout`).
    ``torch.nn.CosineSimilarity`` normalises BOTH sides inside every call; the live side's
    normalisation is the same tensor on every row, so it is formed once here with the call's
    own ops (``vector_norm(...).clone().clamp_min_(eps)``, divide) and each row then runs the
    rest of the call - ``expand``, the goal-side norm on the expanded view, divide, multiply,
    ``sum(dim=3)``. Bit for bit the call (measured 0.0 on the real pair, 1.5x faster); the
    first row of every frame is ALSO run through the call itself, and if the two ever differ
    on a device the call is used for every row of that frame. Returns ``(rows, route)``.
    """
    cos = torch.nn.CosineSimilarity(dim=3)
    n2, d = int(y.shape[2]), int(y.shape[3])
    live = y / torch.linalg.vector_norm(y, 2, dim=3, keepdim=True).clone().clamp_min_(COSINE_EPS)
    out: dict[int, torch.Tensor] = {}
    route = "split"
    for k, i in enumerate(rows):
        token = x[:, :, i, :].unsqueeze(dim=2)                 # the library's own line
        if route == "call":
            out[i] = cos(token, y)[0, 0]
            continue
        xe = token.expand(1, 1, n2, d)
        n1 = torch.linalg.vector_norm(xe, 2, dim=3, keepdim=True).clone().clamp_min_(COSINE_EPS)
        row = ((xe / n1) * live).sum(3)[0, 0]
        if k == 0:
            want = cos(token, y)[0, 0]
            if not torch.equal(row, want):
                route, row = "call", want
        out[i] = row
    return out, route


def best_buddies(goal: DinoFeatures, live: DinoFeatures, fg1: torch.Tensor, fg2: torch.Tensor,
                 *, tie_tol: float = TIE_TOL) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                       dict]:
    """``chunk_cosine_sim`` + the two ``torch.max`` of ``find_correspondences``, fast and exact
    where the library's result depends on them. Returns ``(sim_1, nn_1, nn_2, info)`` -
    ``sim_1`` each goal token's best cosine, ``nn_1`` / ``nn_2`` the library's two argmaxes.

    ``torch.nn.CosineSimilarity`` normalises each vector by ``max(||x||, 1e-8)`` and sums the
    products; the product of the two normalised matrices is the same formula in another
    float summation order (2e-6 apart on a real pair; the library's per-row loop costs 0.4 s
    a row on the laptop CPU - 37 minutes a frame - and, estimated from its memory traffic,
    some 6 s a frame on an RTX 4090).
    What ``find_correspondences`` reads of the two argmaxes is: ``nn_1[i]`` for every
    foreground goal token ``i`` (the others fail ``& fg_mask1``), and ``nn_2[j]`` for every
    ``j = nn_1[i]`` of those and every foreground live token ``j`` (the quirk). Of those, any
    whose two best candidates are within ``tie_tol`` is decided on the library's own rows
    (:func:`_library_rows`): an ambiguous goal row is replaced whole; for an ambiguous live
    column, every goal row within ``tie_tol`` of its maximum is. Everywhere else the
    product's argmax cannot differ from the library's (its error is far under half the
    band), so ``nn_1`` / ``nn_2`` are the library's wherever the library reads them - first
    index on an exact tie, as ``torch.max`` documents. If a replaced row moved by more than
    half the band, the premise is broken on this device and every row is recomputed (the
    library's whole loop, recorded as ``similarity_route: loop``).
    """
    d1, d2 = goal.desc, live.desc
    with _full_fp32_matmul():
        a = d1 / torch.linalg.vector_norm(d1, dim=-1, keepdim=True).clamp_min(COSINE_EPS)
        b = d2 / torch.linalg.vector_norm(d2, dim=-1, keepdim=True).clamp_min(COSINE_EPS)
        sim = a @ b.T
    n1, n2 = sim.shape
    x = y = None
    exact: set[int] = set()
    shift = 0.0
    route = "product"

    def replace(rows) -> None:
        nonlocal x, y, shift, route
        rows = sorted(set(int(r) for r in rows) - exact)
        if not rows:
            return
        if x is None:
            x = library_layout(d1, goal.grid, goal.binned)
            y = library_layout(d2, live.grid, live.binned)
        got, how = _library_rows(x, y, rows)
        route = how if route == "product" or how == "call" else route
        for i, row in got.items():
            shift = max(shift, float((row - sim[i]).abs().max()))
            sim[i] = row
            exact.add(i)

    fg1 = fg1.to(sim.device)
    fg2 = fg2.to(sim.device)
    if n2 >= 2:                                    # 1. ambiguous foreground goal rows
        top = sim.topk(2, dim=1).values
        replace(torch.nonzero((top[:, 0] - top[:, 1] < tie_tol) & fg1).flatten().tolist())
    nn_1 = torch.max(sim, dim=-1)[1]
    if n1 >= 2:                                    # 2. the live columns it reads
        relevant = torch.zeros(n2, dtype=torch.bool, device=sim.device)
        relevant[nn_1[fg1]] = True
        relevant |= fg2
        topc = sim.topk(2, dim=0).values
        amb = torch.nonzero((topc[0] - topc[1] < tie_tol) & relevant).flatten()
        if len(amb):
            near = sim[:, amb] >= (topc[0, amb] - tie_tol)
            replace(torch.nonzero(near.any(dim=1)).flatten().tolist())
    if shift > tie_tol / 2:                        # the premise failed here: the whole loop
        replace(range(n1))
        route = "loop"
    sim_1, nn_1 = torch.max(sim, dim=-1)
    _sim_2, nn_2 = torch.max(sim, dim=-2)
    return sim_1, nn_1, nn_2, {"n_exact_rows": len(exact), "max_exact_row_shift": shift,
                               "similarity_route": route}


def select_pairs(goal: DinoFeatures, live: DinoFeatures, *, num_pairs: int, thresh: float,
                 n_init: int, goal_fg: torch.Tensor | None = None,
                 tie_tol: float = TIE_TOL) -> dict:
    """``find_correspondences`` from "calculate similarity" to the point lists.

    The arithmetic of upstream ``correspondences.py`` L46-110, in its order: the
    saliency threshold (``> thresh``); the similarity and the two argmaxes
    (:func:`best_buddies`);
    best buddies ``nn_2[nn_1] == idx``; the two foreground masks **as the library applies
    them** - the goal token must be foreground, and it must be the nearest neighbour of
    *some* foreground live token (``nn_2[fg_mask2]``, the quirk, not the buddy's own
    foreground); K-means (scikit-learn, ``random_state=0``, ``n_init`` stated) into
    ``min(num_pairs, n)`` clusters over the concatenated, L2-normalised pair
    descriptors; per cluster the pair with the highest mean saliency (strict ``>``, the
    first index wins a tie); token centres ``(idx - 1) * stride + stride + p // 2``.

    ``goal_fg`` (the masked cell) is ANDed into the goal's foreground mask. Returns the
    library's ``points1`` / ``points2`` as ``(K, 2)`` int arrays in ``(y, x)`` order (in
    the resized pictures), the token indices, the pair cosines and the counts. Empty
    pool -> ``K = 0`` (the library crashes in ``KMeans`` there - deviation 2).
    ``n_buddies_raw`` (the mutual pairs before the foreground masks) is a diagnostic: exact
    on the foreground, where the library's result reads it; elsewhere a near-tie the result
    never reads may count differently (:func:`best_buddies`).
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415 - never at import (owner rule)
    from threadpoolctl import threadpool_limits  # noqa: PLC0415 - scikit-learn's own dependency

    t0 = time.monotonic()
    n1 = goal.grid[0] * goal.grid[1]
    fg_mask1 = goal.saliency > thresh
    if goal_fg is not None:
        fg_mask1 = torch.bitwise_and(fg_mask1, goal_fg.to(fg_mask1.device))
    fg_mask2 = live.saliency > thresh
    # nn_1 - indices of block2 closest to block1; nn_2 - indices of block1 closest to block2
    sim_1, nn_1, nn_2, tie = best_buddies(goal, live, fg_mask1, fg_mask2, tie_tol=tie_tol)
    sim_ms = (time.monotonic() - t0) * 1000.0
    image_idxs = torch.arange(n1, device=nn_1.device)
    bbs_mask = nn_2[nn_1] == image_idxs
    n_buddies_raw = int(bbs_mask.sum())
    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(n1, dtype=torch.bool, device=nn_1.device)
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)
    out = {"n_buddies_raw": n_buddies_raw, "n_buddies": int(bbs_mask.sum()),
           "n_fg_goal": int(fg_mask1.sum()), "n_fg_live": int(fg_mask2.sum()),
           "similarity_ms": sim_ms, **tie}
    empty = {"points1": np.zeros((0, 2), np.int64), "points2": np.zeros((0, 2), np.int64),
             "idx1": np.zeros(0, np.int64), "idx2": np.zeros(0, np.int64),
             "sim": np.zeros(0), "n_clusters": 0, "n_empty_clusters": 0, "kmeans_ms": 0.0}
    if out["n_buddies"] == 0:
        return {**out, **empty}
    t1 = time.monotonic()
    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = goal.desc[bbs_mask, :].cpu().numpy()
    bb_descs2 = live.desc[nn_1[bbs_mask], :].cpu().numpy()
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(int(num_pairs), len(all_keys_together))
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    with threadpool_limits(limits=KMEANS_THREADS):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=int(n_init)).fit(normalized)
    # rank pairs by their mean saliency value
    ranks = ((goal.saliency[bbs_mask] + live.saliency[nn_1[bbs_mask]]) / 2).cpu().numpy()
    best = np.full(n_clusters, -np.inf)
    pick = np.full(n_clusters, -1, np.int64)
    for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks, strict=True)):
        if rank > best[label]:                 # one pass: the outer `for k` repeated it
            best[label] = rank
            pick[label] = i
    n_empty = int((pick < 0).sum())            # deviation 4: an empty cluster gives no pair
    pick = pick[pick >= 0]
    indices = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
        torch.as_tensor(pick, device=bbs_mask.device)]
    idx1 = indices.cpu().numpy().astype(np.int64)
    idx2 = nn_1[indices].cpu().numpy().astype(np.int64)
    w1, w2 = goal.grid[1], live.grid[1]
    y1, x1, y2, x2 = idx1 // w1, idx1 % w1, idx2 // w2, idx2 % w2

    def centre(v, s, p):
        return (v - 1) * s + s + p // 2

    points1 = np.column_stack([centre(y1, goal.stride[0], goal.p),
                               centre(x1, goal.stride[1], goal.p)]).astype(np.int64)
    points2 = np.column_stack([centre(y2, live.stride[0], live.p),
                               centre(x2, live.stride[1], live.p)]).astype(np.int64)
    return {**out, "points1": points1, "points2": points2, "idx1": idx1, "idx2": idx2,
            "sim": sim_1[indices].cpu().numpy().astype(np.float64),
            "n_clusters": n_clusters, "n_empty_clusters": n_empty,
            "kmeans_ms": (time.monotonic() - t1) * 1000.0}


def token_mask_from_pixels(pixel_mask: np.ndarray, *, load_hw: tuple[int, int],
                           grid: tuple[int, int], p: int, stride: tuple[int, int]) -> np.ndarray:
    """A full-image pixel mask -> bool ``[rows * cols]`` over the stride-4 token lattice.

    The rule of :func:`vsbench.vs.baselines.match.patch_mask_from_pixels` (nearest-neighbour
    resize to the encoder input, ``> 127``, a token counts when **more than 20 %** of its
    pixels are set) on DINOBot's lattice: the resize is to the anisotropic load size
    (398 x 224 for a 1920 x 1080 frame) and a token's window is its own ``p x p`` patch
    at ``stride`` - the windows overlap.
    """
    import cv2  # noqa: PLC0415

    m = np.asarray(pixel_mask)
    m_u8 = (m.astype(np.uint8) * 255) if m.dtype == bool else m.astype(np.uint8)
    h, w = int(load_hw[0]), int(load_hw[1])
    resized = (cv2.resize(m_u8, (w, h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.int64)
    ii = np.pad(resized, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)   # integral image
    y0 = (np.arange(grid[0]) * stride[0])[:, None]
    x0 = (np.arange(grid[1]) * stride[1])[None, :]
    y1, x1 = y0 + p, x0 + p
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return (s / float(p * p) > 0.2).reshape(-1)


# ----------------------------------------------------------------- the matcher
class DinoBotCorrespondences(CorrespondenceMatcher):
    """DINOBot's front end as a matcher: eight best-buddy pairs per frame.

    ``reset`` encodes the goal once (descriptors, saliency, its foreground - the
    objects-only mask ANDed in at the token lattice for the masked cell); ``match``
    encodes the live frame and runs :func:`select_pairs`. Returns
    :class:`vsbench.vs.baselines.match.Matches` in **full-image pixels** on the nominal pixel
    lattice the classical arms use (``side = input_size = image width``, ``patch_size =
    1`` - the split is exact, so ``to_rows()`` returns the pixels it was given): the
    library's resized-picture points times the two axis scales (1920/398, 1080/224).
    ``sim`` is the pair's cosine; there is no cyclic score and no margin, and the cosine
    is not a weight (``provides_margin`` / ``provides_conf_weight`` False). ``extras``
    carries the library's own resized-picture points (``goal_load_yx`` /
    ``cur_load_yx``, what ``compute_error`` reads in the ``pixel_load`` space) and the
    per-frame counts and timings.

    The extractor is built on the first :meth:`reset` (so building an arm touches no
    weights); ``extractor=`` injects one - the fast tier's stub.
    """

    honours_goal_mask = True
    provides_margin = False
    provides_conf_weight = False
    tie_order = TIE_ORDER
    frozen_landmarks = False

    def __init__(self, *, num_landmarks: int | None = None, seed: int | None = None,
                 device: str | None = None, extractor: Any = None,
                 model_type: str = PUBLISHED["model_type"], load_size=PUBLISHED["load_size"],
                 stride: int = PUBLISHED["stride"], layer: int = PUBLISHED["layer"],
                 facet: str = PUBLISHED["facet"], bin_hierarchy: int = PUBLISHED["bin_hierarchy"],
                 saliency_thresh: float = PUBLISHED["saliency_thresh"],
                 kmeans_n_init: int = PUBLISHED["kmeans_n_init"],
                 tie_tol: float = TIE_TOL) -> None:
        if num_landmarks is None:
            raise ValueError(
                "DinoBotCorrespondences needs num_landmarks (the gist's num_pairs, 8) and it "
                "has no default - the config field 'matcher.num_landmarks' states it")
        if extractor is None and device is None:
            raise ValueError("no device: 'matcher.device' is required for dinobot "
                             "('cuda' on the station, 'cpu' in the fast tier)")
        fe = resolve_block({"dinobot": dict(
            model_type=model_type, load_size=load_size, stride=stride, layer=layer,
            facet=facet, bin_hierarchy=bin_hierarchy, saliency_thresh=saliency_thresh,
            kmeans_n_init=kmeans_n_init)}, align=False)
        self.front_end = fe
        self.num_landmarks = int(num_landmarks)
        if self.num_landmarks < 1:
            raise ValueError(f"num_landmarks must be >= 1, got {num_landmarks}")
        self.seed = None if seed is None else int(seed)     # stated, unused: random_state=0
        self.device = None if device is None else str(device)
        self.tie_tol = float(tie_tol)
        self.name = f"dinobot_{fe['model_type']}"
        self._extractor = extractor
        self._injected = extractor is not None
        self.goal: DinoFeatures | None = None
        self.goal_fg: torch.Tensor | None = None
        self.goal_masked = False
        self.image_wh: tuple[int, int] | None = None

    # --------------------------------------------------------------- plumbing
    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = DinoV1Extractor(self.front_end["model_type"],
                                              self.front_end["stride"], device=self.device)
        return self._extractor

    def _encode(self, rgb: np.ndarray) -> DinoFeatures:
        fe = self.front_end
        return self.extractor.encode(rgb, load_size=fe["load_size"], layer=fe["layer"],
                                     facet=fe["facet"], bin_hierarchy=fe["bin_hierarchy"])

    def set_episode(self, scene: str, start_id: Any) -> None:
        """Nothing to seed: the published K-means is ``random_state=0``."""

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        """Encode the goal once; ``goal_mask`` (bool ``[H, W]``) ANDs into its foreground."""
        h, w = np.asarray(goal_rgb).shape[:2]
        self.image_wh = (int(w), int(h))
        self.goal = self._encode(goal_rgb)
        self.goal_fg = None
        self.goal_masked = goal_mask is not None
        if goal_mask is not None:
            g = self.goal
            self.goal_fg = torch.as_tensor(token_mask_from_pixels(
                goal_mask, load_hw=g.load_hw, grid=g.grid, p=g.p, stride=g.stride),
                device=g.saliency.device)

    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        if current_mask is not None:
            raise ValueError("DINOBot has no live-side mask: its live foreground is the "
                             "saliency map alone")
        if self.goal is None:
            raise RuntimeError("reset() must be called before match()")
        live = self._encode(current_rgb)
        fe = self.front_end
        sel = select_pairs(self.goal, live, num_pairs=self.num_landmarks,
                           thresh=fe["saliency_thresh"], n_init=fe["kmeans_n_init"],
                           goal_fg=self.goal_fg, tie_tol=self.tie_tol)
        w, h = self.image_wh
        gh, gw = self.goal.load_hw
        lh, lw = live.load_hw
        goal_px = sel["points1"][:, ::-1].astype(np.float64) * np.array([w / gw, h / gh])
        cur_px = sel["points2"][:, ::-1].astype(np.float64) * np.array([w / lw, h / lh])
        extras = {k: sel[k] for k in ("n_buddies", "n_buddies_raw", "n_fg_goal", "n_fg_live",
                                      "n_exact_rows", "max_exact_row_shift", "similarity_route",
                                      "n_clusters", "n_empty_clusters", "similarity_ms",
                                      "kmeans_ms")}
        extras.update(goal_load_yx=sel["points1"], cur_load_yx=sel["points2"],
                      goal_token=sel["idx1"], cur_token=sel["idx2"],
                      load_hw=live.load_hw, goal_load_hw=self.goal.load_hw,
                      forward_ms=live.forward_ms)
        return pairs_to_matches(goal_px, cur_px, image_wh=self.image_wh, sim=sel["sim"],
                                pool_size=sel["n_buddies"], num_landmarks=self.num_landmarks,
                                extras=extras)

    def record_fields(self) -> dict:
        """The front end's settings and what it loaded - never a measurement."""
        ex = self._extractor
        fe = self.front_end
        import sklearn  # noqa: PLC0415 - the version is a setting of the published K-means

        ran_on = getattr(ex, "device", None)            # what ran, not what was asked for
        return {"front_end": self.name,
                "matcher_device": str(ran_on) if ran_on is not None else self.device,
                "dinobot_model_type": fe["model_type"], "dinobot_load_size": fe["load_size"],
                "dinobot_stride": fe["stride"], "dinobot_layer": fe["layer"],
                "dinobot_facet": fe["facet"], "dinobot_bin_hierarchy": fe["bin_hierarchy"],
                "dinobot_saliency_thresh": fe["saliency_thresh"],
                "dinobot_num_pairs": self.num_landmarks,
                "dinobot_kmeans_n_init": fe["kmeans_n_init"],
                "dinobot_tie_tol": self.tie_tol,
                "dinobot_weights": getattr(ex, "weights", None),
                "dinobot_weights_md5": getattr(ex, "weights_md5", None),
                "dinobot_torch_hub_commit": getattr(ex, "hub_commit", None),
                "dinobot_torch_hub_code_md5": getattr(ex, "hub_code_md5", None),
                "dinobot_extractor": ("injected" if self._injected
                                      else type(ex).__name__ if ex is not None else None),
                "dinobot_sklearn_version": sklearn.__version__,
                "dinobot_kmeans_threads": KMEANS_THREADS,
                "dinobot_goal_masked": bool(self.goal_masked),
                "prealign": None}

    def close(self) -> None:
        if not self._injected:
            self._extractor = None
        self.goal = None


def pairs_to_matches(goal_px, cur_px, *, image_wh, sim, pool_size: int, num_landmarks: int,
                     extras: dict | None = None) -> Matches:
    """Full-image pixel pairs -> :class:`Matches` on the classical arms' pixel lattice."""
    w, h = int(image_wh[0]), int(image_wh[1])
    k = len(np.asarray(goal_px).reshape(-1, 2))
    m = Matches.from_pixel_pairs(
        np.asarray(goal_px, np.float64).reshape(-1, 2),
        np.asarray(cur_px, np.float64).reshape(-1, 2),
        side=w, input_size=w, patch_size=1, image_wh=(w, h),
        sim=np.asarray(sim, np.float64).reshape(-1), cyclic=np.full(k, np.nan),
        margin=np.full(k, np.nan), pool_size=int(pool_size), num_landmarks=int(num_landmarks),
        draw_seed=None, tie_order=TIE_ORDER)
    m.extras.update(extras or {})
    return m


# ------------------------------------------------------------------ the fit
def kabsch(X: np.ndarray, Y: np.ndarray, reflection: str = "none") -> tuple[np.ndarray, np.ndarray]:
    """The gist's ``find_transformation(X, Y)`` verbatim: ``Y ~ R X + t``.

    ``C = Xc^T Yc``, ``U S V^T = svd(C)``, ``R = V U^T``, ``t = cY - R cX``, with **no
    reflection check** - on a mirrored or coplanar set it returns the improper matrix
    the published formula returns (``reflection="none"``). ``reflection="fix"`` is
    Sorkine-Hornung & Rabinovich's correction (the paper's reference [32]):
    ``R = V diag(1, 1, det(V U^T)) U^T`` - a stated option, a different cell.
    """
    if reflection not in REFLECTIONS:
        raise ValueError(f"reflection {reflection!r} is not one of {REFLECTIONS}")
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    if reflection == "fix" and np.linalg.det(R) < 0:
        R = Vt.T @ np.diag([1.0, 1.0, -1.0]) @ U.T
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t


def compute_error(points1, points2) -> float:
    """The gist's ``compute_error`` verbatim: the Frobenius norm of the pair differences."""
    return float(np.linalg.norm(np.array(points1) - np.array(points2)))


def backproject(px: np.ndarray, z: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """``project_to_3d`` (a gist placeholder, *filled in*): pinhole, optical frame.

    ``X = (u - cx) / fx * z``, ``Y = (v - cy) / fy * z``, ``Z = z`` - x right, y down,
    z forward, the frame every pose and twist in vs-sim uses (Open3D's formula, which
    the gist points at).
    """
    p = np.asarray(px, np.float64).reshape(-1, 2)
    z = np.asarray(z, np.float64).reshape(-1)
    return np.column_stack([(p[:, 0] - intr.cx) / intr.fx * z,
                            (p[:, 1] - intr.cy) / intr.fy * z, z])


def valid_depth(z: np.ndarray) -> np.ndarray:
    """The bench's validity rule (:func:`vsbench.vs.baselines.control.clean_depth`) as a mask."""
    z = np.asarray(z, np.float64)
    return np.isfinite(z) & (z > 0) & (z <= DEPTH_MAX_M)


def rotvec(R: np.ndarray) -> np.ndarray:
    """Axis times angle of a proper rotation (scipy's log map, robust near pi)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    return np.asarray(Rotation.from_matrix(np.asarray(R, np.float64)).as_rotvec(), np.float64)


def jump_twist(R: np.ndarray, t: np.ndarray, *, dt: float,
               step_fraction: float = 1.0) -> np.ndarray:
    """``robot_move(t, R)`` as ONE bench step: ``v = a t / dt``, ``w = a axis_angle(R) / dt``.

    The bench integrates ``pos += R_cam v dt`` and ``R_cam <- R_cam Rot(w_hat, |w| dt)``
    (:func:`vsbench.vs.scene.integrate_optical_velocity`), so one step lands exactly on
    ``T_cam (R^a, a t)`` - at ``a = 1`` the whole published jump.
    """
    a = float(step_fraction)
    return np.concatenate([a * np.asarray(t, np.float64).reshape(3) / float(dt),
                           a * rotvec(R) / float(dt)])


@dataclass(frozen=True)
class _KabschFit:
    """The record's ``weighting`` column for a DINOBot cell: a name, not our recipe."""

    method: str = "dinobot_kabsch_unweighted"


@dataclass
class DinoBotAlign:
    """The move rule in the controller slot: lift, fit, jump - and what the record says.

    The record's controller columns hold what is true of DINOBot: no rounding
    (``solve_unrounded``), no smoothing (``ema_alpha = 0``), the pair floor as
    ``min_matches``, the fit's name as ``weighting``, ``step_fraction`` as ``gain``.
    """

    intrinsics: Intrinsics
    dt: float = DEFAULT_DT
    step_fraction: float = 1.0
    min_pairs: int = MIN_PAIRS
    reflection: str = "none"
    err_space: str = "pixel_load"
    solve_unrounded: bool = True
    ema_alpha: float = 0.0
    weighting: Any = field(default_factory=_KabschFit)
    goal_depth: np.ndarray | None = field(default=None, repr=False)
    _step: int = 0

    @property
    def gain(self) -> float:
        return self.step_fraction

    @property
    def min_matches(self) -> int:
        return self.min_pairs

    def reset(self) -> None:
        self._step = 0

    def set_goal_depth(self, goal_depth) -> None:
        """The goal picture's depth - DINOBot is the one arm that reads it."""
        self.goal_depth = None if goal_depth is None else np.asarray(goal_depth)

    def step(self, matches: Matches, live_depth: np.ndarray) -> tuple[Twist, dict, np.ndarray]:
        """One published iteration -> ``(twist, debug, keep)``; ``keep`` masks the rows used.

        Lift both ends (depth by truncation, invalid rows dropped), ``compute_error`` in
        both spaces, then the guards: fewer than ``min_pairs`` rows -> no motion,
        ``degenerate``; an improper fit under ``reflection: none`` -> no motion,
        ``reflection``. Otherwise the whole transform (times ``step_fraction``) as one
        step's twist.
        """
        from vsbench.vs.bench.arms import depth_lookup  # noqa: PLC0415 - avoids an import cycle

        t0 = time.monotonic()
        self._step += 1
        if self.goal_depth is None:
            raise RuntimeError("DINOBot lifts the goal points with the GOAL depth and none was "
                               "given - the arm's reset() must hand it over")
        rows = matches.to_rows() if len(matches) else {"pts_goal": np.zeros((0, 2)),
                                                       "pts_cur": np.zeros((0, 2))}
        pg, pc = rows["pts_goal"], rows["pts_cur"]
        zg = depth_lookup(self.goal_depth, pg)
        zc = depth_lookup(live_depth, pc)
        keep = valid_depth(zg) & valid_depth(zc)
        n_kept = int(keep.sum())
        X = backproject(pg[keep], zg[keep], self.intrinsics)
        Y = backproject(pc[keep], zc[keep], self.intrinsics)
        gy, cy = matches.extras.get("goal_load_yx"), matches.extras.get("cur_load_yx")
        # `pixel_load` is compute_error on the library's own resized-picture points; a
        # matcher that does not hand them over has no such number (NaN, never zeros)
        err_pixel_load = (compute_error(np.asarray(gy)[keep], np.asarray(cy)[keep])
                          if n_kept and gy is not None and cy is not None else float("nan"))
        err_metric = compute_error(X, Y) if n_kept else float("nan")
        intr = self.intrinsics
        d = pg[keep] - pc[keep]
        per_pair = np.hypot(d[:, 0] / intr.fx, d[:, 1] / intr.fy)
        debug: dict[str, Any] = {
            "step": self._step, "n_matches": n_kept, "n_selected": int(len(pg)),
            "n_dropped_depth": int(len(pg) - n_kept), "degenerate": False,
            "reflection": False, "err_pixel_load": err_pixel_load, "err_metric": err_metric,
            "pts_goal": pg[keep], "pts_cur": pc[keep], "depth_values": zc[keep],
            "goal_depth_values": zg[keep], "jump_t_mm": 0.0, "jump_rot_deg": 0.0,
            "fit_det": float("nan"), "fit_t_mm": float("nan"), "policy": "dinobot_align"}
        v = np.zeros(6)
        if n_kept < self.min_pairs:
            debug.update(degenerate=True, feature_error_norm=float("nan"),
                         normalised_errors=None)
        else:
            R, t = kabsch(X, Y, self.reflection)
            det = float(np.linalg.det(R))
            debug.update(fit_det=det, fit_t_mm=float(np.linalg.norm(t)) * 1000.0,
                         feature_error_norm=(err_pixel_load if self.err_space == "pixel_load"
                                             else err_metric),
                         normalised_errors=per_pair)
            if det < 0:                        # only reachable under reflection: none
                debug["reflection"] = True
            else:
                v = jump_twist(R, t, dt=self.dt, step_fraction=self.step_fraction)
                debug.update(jump_t_mm=float(np.linalg.norm(v[:3]) * self.dt * 1000.0),
                             jump_rot_deg=float(np.degrees(np.linalg.norm(v[3:]) * self.dt)))
        debug["velocity_raw"] = v.copy()
        debug["step_ms"] = (time.monotonic() - t0) * 1000.0
        return Twist(v), debug, keep


# ------------------------------------------------------------------ the stop
class DinoBotErr:
    """DINOBot's own stop: ``while error > ERR_THRESHOLD`` - recorded, latching.

    The gist computes ``error = compute_error(points1, points2)`` on the observation
    **before** the move, executes the move, and only then tests the loop condition - so
    DINOBot ends one jump after its first reading under the threshold. This rule
    therefore latches on the step **after** the first ``feature_error_norm <
    threshold`` (``detail["under_step"]`` is that reading's step), and the pose the
    runner records at its firing step is the pose the published loop ends at. The arm's
    ``feature_error_norm`` is ``compute_error`` in the stated space (``pixel_load``: the
    library's resized-picture points; ``metric``: the 3-D points in metres); a non-finite
    reading (a degenerate step) is never under the threshold.
    """

    name = "dinobot_err"

    def __init__(self, *, threshold: float = ERR_THRESHOLDS["pixel_load"],
                 space: str = "pixel_load") -> None:
        if space not in ERR_SPACES:
            raise ValueError(f"dinobot_err space {space!r} is not one of {ERR_SPACES}")
        self.threshold = float(threshold)
        self.space = space
        self.reset()

    @property
    def params(self) -> dict:
        return {"threshold": self.threshold, "space": self.space,
                "fires": "the step after the first reading under the threshold"}

    def reset(self) -> None:
        self._under_step: int | None = None
        self.fired_step: int | None = None

    def update(self, obs: StopObservation) -> StopVerdict:
        if self.fired_step is None and self._under_step is not None:
            self.fired_step = int(obs.step)
        err = float(obs.feature_error_norm)
        if self._under_step is None and np.isfinite(err) and err < self.threshold:
            self._under_step = int(obs.step)
        return StopVerdict(self.fired_step is not None, self.fired_step, 1.0,
                           {"under_step": self._under_step, "err": err})


# ------------------------------------------------------------------- the arm
class DinoBotArm:
    """DINOBot's alignment behind the :class:`vsbench.vs.bench.arms.Arm` shape.

    Structurally an ``Arm`` (``matcher`` / ``controller`` / ``stop`` / ``config``), so
    :func:`vsbench.vs.bench.runner.run_episode`, the record and the scoring take it unchanged -
    but :meth:`step` is DINOBot's look-then-move, not the IBVS law.
    """

    kind = "method"

    def __init__(self, matcher: DinoBotCorrespondences, controller: DinoBotAlign, stop,
                 config: dict, *, name: str, err_threshold: float,
                 goal_mask_mode: str = "none") -> None:
        self.matcher = matcher
        self.controller = controller
        self.stop = stop
        self.config = config
        self.name = name
        self.err_threshold = float(err_threshold)
        self.goal_mask_mode = str(goal_mask_mode)

    @property
    def matcher_name(self) -> str:
        return self.matcher.name

    @property
    def checkpoint(self) -> str | None:
        """The resolved weight file (``None`` until the extractor is built, or injected)."""
        return getattr(getattr(self.matcher, "_extractor", None), "weights", None)

    @property
    def dt(self) -> float:
        return self.controller.dt

    @property
    def gain(self) -> float:
        """The record's ``gain``: the fraction of the fitted jump executed per step."""
        return self.controller.gain

    def set_episode(self, scene: str, start_id: Any) -> None:
        self.matcher.set_episode(scene, start_id)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        if goal_depth is None:
            raise ValueError("DINOBot lifts its goal points with the goal picture's DEPTH - "
                             "reset(goal_rgb, goal_depth) needs it")
        self.matcher.reset(goal_rgb, goal_depth, goal_mask)
        self.controller.reset()
        self.controller.set_goal_depth(goal_depth)
        self.stop.reset()

    def step(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray | None = None):
        """One published iteration -> ``(twist, debug, matches the fit used)``."""
        if mask is not None:
            raise ValueError("DINOBot runs its own front end and takes no frame mask")
        t0 = time.monotonic()
        matches = self.matcher.match(rgb)
        match_ms = (time.monotonic() - t0) * 1000.0
        twist, debug, keep = self.controller.step(matches, depth)
        ex = matches.extras
        debug.update(match_ms=match_ms, **{k: ex.get(k) for k in (
            "n_buddies", "n_buddies_raw", "n_fg_goal", "n_fg_live", "n_exact_rows",
            "max_exact_row_shift", "similarity_route", "n_clusters", "n_empty_clusters",
            "forward_ms", "similarity_ms", "kmeans_ms")})
        rows = matches.to_rows() if len(matches) else None
        extras = {k: (np.asarray(v)[keep] if k in ("goal_load_yx", "cur_load_yx", "goal_token",
                                                   "cur_token") else v) for k, v in ex.items()}
        used = pairs_to_matches(
            rows["pts_goal"][keep] if rows else np.zeros((0, 2)),
            rows["pts_cur"][keep] if rows else np.zeros((0, 2)),
            image_wh=self.matcher.image_wh, sim=np.asarray(matches.sim)[keep],
            pool_size=matches.pool_size, num_landmarks=matches.num_landmarks, extras=extras)
        return twist, debug, used

    def close(self) -> None:
        self.matcher.close()

    def recorded_stop_constants(self) -> dict:
        """The constants of the recorded rules this arm owns (read by the runner):
        ``dinobot_err`` runs at the arm's own threshold, in the arm's own space."""
        return {"dinobot_err": {"threshold": self.err_threshold,
                                "space": self.controller.err_space}}

    def record_fields(self) -> dict:
        """The published knobs, stated - never a measurement."""
        c = self.controller
        return {**self.matcher.record_fields(),
                "dinobot_step_fraction": c.step_fraction, "dinobot_min_pairs": c.min_pairs,
                "dinobot_reflection": c.reflection, "dinobot_err_space": c.err_space,
                "dinobot_err_threshold": self.err_threshold,
                "rotation_mode": "none", "goal_mask_mode": self.goal_mask_mode}


def build_dinobot_arm(config: dict, *, intrinsics: Intrinsics, scene: str | None = None,
                      start_id: Any = None, extractor: Any = None,
                      goal_mask_mode: str | None = None) -> DinoBotArm:
    """Build the DINOBot arm a resolved config asks for (``matcher.variant: dinobot``).

    Refused by name (each would be a setting the record does not carry, or a component
    DINOBot does not have): the research repository's blocks, every IBVS controller key
    but ``dt``, ``stop.rule: dec_plateau`` (a lambda schedule would damp the jump - the
    CNS refusal) and ``dec_plateau`` among the recorded rules, ``rotation.mode:
    prealign_4way`` (DINOBot has no rotation search), matcher keys other than
    ``variant`` / ``num_landmarks`` / ``device`` / ``goal_mask``, and unknown
    ``dinobot:`` keys. ``extractor`` injects a ready-made extractor (the fast tier's
    stub); otherwise the network is built from the torch.hub cache at the first reset.
    """
    from vsbench.vs.baselines import prealign as rotmod  # noqa: PLC0415
    from vsbench.vs.bench import arms as armsmod  # noqa: PLC0415 - avoids an import cycle
    from vsbench.vs.bench import config as cfgmod  # noqa: PLC0415

    cfgmod.validate(config)
    m = dict(config["matcher"])
    if m["variant"] != "dinobot":
        raise ValueError(f"build_dinobot_arm needs matcher.variant 'dinobot', got {m['variant']!r}")
    for block, what in armsmod.FOREIGN_BLOCKS.items():
        if config.get(block):
            raise ValueError(f"config block '{block}' is {what}, which DINOBot does not have")
    stray = sorted(set(m) - set(_ARM_MATCHER_KEYS))
    if stray:
        raise ValueError(f"matcher keys {stray} are not read by the DINOBot arm - its front end "
                         "is set in the `dinobot:` block")
    ctrl = dict(config.get("controller") or {})
    for key in ctrl:
        if key != "dt":
            why = _REFUSED_CONTROLLER_KEYS.get(key, "it is not a setting of DINOBot's move rule")
            raise ValueError(f"dinobot cannot take controller.{key} - {why}. Only "
                             "controller.dt (the loop's period) is kept")
    if config["stop"]["rule"] == "dec_plateau":
        raise ValueError(
            "dinobot cannot run the `dec` stop - a lambda schedule would scale (damp) the "
            "published whole-transform jump; protocol v2 scores every arm on `cns_plateau`")
    recorded = list((config.get("episode") or {}).get("record_stops") or [])
    if "dec_plateau" in recorded:
        raise ValueError("episode.record_stops names dec_plateau, which DINOBot refuses: its "
                         "lambda schedule has no gain to act on")
    if rotmod.rotation_config(config)["mode"] != "none":
        raise ValueError(
            "rotation.mode 'prealign_4way' is a published component of ViT-VS / ART-VS, not of "
            "DINOBot, which has no rotation search and no rotation invariance")
    b = resolve_block(config)
    if goal_mask_mode is None:
        goal_mask_mode = str(m.get("goal_mask") or "none")
    matcher = DinoBotCorrespondences(
        num_landmarks=m["num_landmarks"], seed=config["seed"], device=m.get("device"),
        extractor=extractor, **{k: b[k] for k in FRONT_END_KEYS})
    controller = DinoBotAlign(intrinsics=intrinsics,
                              dt=float(ctrl.get("dt", DEFAULT_DT)),
                              step_fraction=b["step_fraction"], min_pairs=b["min_pairs"],
                              reflection=b["reflection"], err_space=b["err_space"])
    stop_block = dict(config["stop"])
    over: dict[str, Any] = {}
    if stop_block["rule"] == "cns_plateau" and "dt" not in stop_block:
        over["dt"] = controller.dt
    if stop_block["rule"] == "dinobot_err":
        # the scored `dinobot_err` is the arm's own stop: its constants are the block's
        for key, want in (("threshold", b["err_threshold"]), ("space", b["err_space"])):
            if key in stop_block and stop_block[key] != want:
                raise ValueError(f"stop.{key} {stop_block[key]!r} disagrees with the dinobot "
                                 f"block ({want!r}) - state it once, in `dinobot:`")
            over[key] = want
    stop = build_stop_rule(config, **over)
    name = str((config.get("arm") or {}).get("name") or dinobot_label(
        {**config, "matcher": {**m, "goal_mask": goal_mask_mode}}))
    arm = DinoBotArm(matcher, controller, stop, config, name=name,
                     err_threshold=b["err_threshold"], goal_mask_mode=goal_mask_mode)
    if scene is not None or start_id is not None:
        arm.set_episode(scene, start_id)
    return arm
