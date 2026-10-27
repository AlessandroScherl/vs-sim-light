# vs-sim copy: vssim/baselines/prealign.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The published pre-servo roll search: the 4-way of ViT-VS and ART-VS.

Purpose
-------
What ViT-VS and ART-VS ship for roll robustness: before the loop starts, try the
current view at 0 / 90 / 180 / 270 degrees against the goal, keep the best, and
**physically roll the camera** to it. A pre-servo *robot motion*, once, off during
the loop - so a baseline that took it starts the loop roll-aligned, and every table
must say so (``rotation_mode`` / ``prealign`` on every record).

:class:`Prealign4Way` is the measurement; :func:`apply_prealign` is the pose apply,
called once by :func:`vsbench.vs.bench.runner.run_episode` between placing the camera at
the start pose and the first control step, on an arm whose config says
``rotation: {mode: prealign_4way}`` and on no other. ``mode: none`` is the default
and what every published baseline number is quoted without.

Provenance
----------
The 4-way half of the Refine-VS research repository's ``refinevs/rotation.py``
(its rotated goal banks and staged ladder - that project's own method - stayed
behind), ported there line for line from ART-VS
``features/rotation_finder_fixed.py::UnifiedRotationFinder`` through ``genvs-sim``
``src/control/prealign.py``, with the frozen search reproduced by a fixture
(``tests/fixtures/u8_rotation_parity.json``).

Golden tests: ``tests/fast/test_prealign.py``.
"""


from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from vsbench.vs.baselines.match import (
    DEFAULT_TIE_ORDER,
    TIE_ORDERS,
    CorrespondenceMatcher,
    mutual_nn_match,
)

# ---------------------------------------------------------------- constants


#: Published discrete hypothesis set (``prealign.DISCRETE_ANGLES``,
#: ``find_best_rotation_discrete``). First-listed angle wins similarity ties, as the
#: reference's ``>`` does.
DISCRETE_ANGLES: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

#: The reference finder's temporary ``num_pairs`` override for the search
#: (``prealign.SEARCH_NUM_PAIRS``). Fence 6: the shipped robot program uses 64.
SEARCH_NUM_PAIRS = 48

#: Published fill for out-of-canvas pixels after a rotation (``prealign.FILL_WHITE``).
FILL_WHITE = (255, 255, 255)

#: ``rotation.mode``. ``none`` is the default and every published number to date;
#: ``prealign_4way`` is the published pre-servo search.
ROTATION_MODES = ("none", "prealign_4way")


# ----------------------------------------------------------------- geometry


def circular_diff_deg(a: float, b: float) -> float:
    """Signed ``a - b`` wrapped to ``(-180, 180]`` - ``prealign.circular_diff_deg``."""
    d = (float(a) - float(b)) % 360.0
    return d - 360.0 if d > 180.0 else d


def unwarp_points(pts, psi_deg: float, image_wh: tuple, raw_wh: tuple | None = None):
    """Warped-frame pixels -> true-frame pixels: the inverse of ``rotate_frame(rgb, psi)``.

    Verbatim from ``rot_aware.unwarp_points`` (L73-101), half-pixel and all.

    Coordinate convention (measured there, and pinned here by a parity test against
    the frozen function): Pillow rotates about the **continuous** image centre
    ``(w/2, h/2)`` while pixel index ``i`` covers ``[i, i+1)``, so in *index*
    coordinates that is a rotation about ``(w/2 - 0.5, h/2 - 0.5)``; at 180 degrees
    the map is exactly the ``[::-1, ::-1]`` index flip. Getting this half-pixel wrong
    biases every mapped-back point by up to sqrt(2) pixels - which at poster distance
    is about a millimetre, i.e. the whole quantity this project measures.

    ``raw_wh`` serves a caller holding an already-squashed frame: the rotation acted
    in the unsquashed raw frame, so index coordinates are converted to continuous,
    scaled there and back. Banks never need it - they rotate natively by construction
    (fence 1) - and it is kept because it is part of the ported function's contract.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    w, h = float(image_wh[0]), float(image_wh[1])
    a = np.deg2rad(float(psi_deg))
    c, s = np.cos(a), np.sin(a)
    R_inv = np.array([[c, s], [-s, c]])          # R2(-psi)
    if raw_wh is None:
        ctr = np.array([w / 2.0 - 0.5, h / 2.0 - 0.5])
        return (p - ctr) @ R_inv.T + ctr
    rw, rh = float(raw_wh[0]), float(raw_wh[1])
    scale = np.array([rw / w, rh / h])
    ctr = np.array([rw / 2.0, rh / 2.0])         # continuous-coord centre
    p_raw = (p + 0.5) * scale                    # index -> continuous, raw
    p_raw = (p_raw - ctr) @ R_inv.T + ctr
    return p_raw / scale - 0.5                   # back to stored index


def rotate_frame(frame_rgb_u8: np.ndarray, psi_deg: float, *,
                 resample: str = "bicubic", fill=FILL_WHITE) -> np.ndarray:
    """``prealign.rotate_hypothesis`` semantics: ``Image.rotate(-psi, fillcolor=fill)``.

    Content map, in index coordinates: the content at ``q`` moves to
    ``R2(+psi) (q - centre) + centre``, which :func:`unwarp_points` inverts.

    **One deviation from the port, stated because it changes pixels**: the published
    call takes Pillow's default resample, which is ``NEAREST``. That is defensible for
    the 4-way, whose four angles are exact multiples of 90 degrees where nearest *is*
    exact - and :class:`Prealign4Way` therefore keeps it. It is not defensible for a
    bank at 30 degrees, where nearest-neighbour aliasing at 1920x1080 survives the
    bicubic squash to 512 and moves tokens; banks use ``"bicubic"``, the default here.
    Measured (review R19): byte-equal to the frozen ``rotate_hypothesis`` at NEAREST on
    48 image/angle combinations, and 21-33 grey levels MAD away from it at off-grid
    angles - which is the aliasing, not a bug.

    **A rotated bank is therefore resampled TWICE and the live frame once** (bicubic
    rotate, then the bicubic squash to 512), so a rotated bank's tokens are slightly
    smoother than the live frame's. That is a possible small count bias *against*
    rotated banks - opposite in sign to the valid-count bias, which also favours bank 0
    - and it is one more reason the still-image probe (the reset plan's first bank
    experiment) comes before any loop claim. Only 0 and 180 degrees escape it: Pillow
    short-circuits those to a copy and a transpose (:func:`rotate_support_px`).
    """
    from PIL import Image  # noqa: PLC0415 - Pillow stays a lazy import, as elsewhere

    filters = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
               "bicubic": Image.BICUBIC}
    if resample not in filters:
        raise ValueError(f"resample must be one of {sorted(filters)}, got {resample!r}")
    img = Image.fromarray(np.ascontiguousarray(np.asarray(frame_rgb_u8, np.uint8)))
    out = img.rotate(-float(psi_deg), resample=filters[resample], expand=False,
                     fillcolor=tuple(fill))
    return np.asarray(out)


def roll_about_optical_axis(R_opt: np.ndarray, angle_deg: float) -> np.ndarray:
    """``R`` rolled by ``angle_deg`` about its own optical (z-forward) axis.

    ``prealign.roll_about_optical_axis`` verbatim. Right-multiplication = a rotation
    in the CAMERA frame, matching how the start sampler composes perturbations
    (``goal_R @ dR``).
    """
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(R_opt, float) @ Rz


def roll_component_deg(R_rel: np.ndarray) -> float:
    """z-twist (roll about the optical axis) of a relative rotation, degrees.

    ``prealign.roll_component_deg`` verbatim: the swing-twist decomposition about z,
    ``atan2(r10 - r01, r00 + r11)``, well defined through +-180 degrees.
    """
    R = np.asarray(R_rel, float)
    return float(np.degrees(np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])))


def classify_pick(true_roll_deg: float, picked_deg: float,
                  hypotheses: Sequence[float] = DISCRETE_ANGLES) -> str:
    """``aligned`` | ``flipped`` | ``other`` - ``prealign.classify_pick`` verbatim.

    ``aligned`` is the hypothesis nearest the true roll (the best a discrete search
    can do); ``flipped`` is the one nearest ``true + 180``, the symmetry failure that
    leaves the camera confidently upside down; ``other`` is any remaining miss. The
    three must never be pooled: an undiscriminating search on hard content is not
    symmetry evidence.
    """
    hyp = [float(a) for a in hypotheses]
    best = min(hyp, key=lambda a: abs(circular_diff_deg(true_roll_deg, a)))
    flip = min(hyp, key=lambda a: abs(circular_diff_deg(true_roll_deg + 180.0, a)))
    p = float(picked_deg)
    if abs(circular_diff_deg(p, best)) < 1e-9:
        return "aligned"
    if abs(circular_diff_deg(flip, best)) > 1e-9 and abs(circular_diff_deg(p, flip)) < 1e-9:
        return "flipped"
    return "other"


# ------------------------------------------------------- the published 4-way


class Prealign4Way:
    """The published pre-servo roll search, as a measurement (``prealign.py``).

    ``pick(goal, current)`` scores the four virtual rotations of the *current* frame
    against the goal and returns the winner plus the similarity at every angle;
    :func:`apply_prealign` is the pose apply, and the two are separate because one is
    a measurement and the other physically moves the camera before the loop (fence 5).

    Faithful-port pins, each mirroring a line of the reference
    (``ART-VS/.../features/rotation_finder_fixed.py::UnifiedRotationFinder``, through
    ``genvs-sim/src/control/prealign.py``):

    - hypothesis angles ``[0, 90, 180, 270]``; the first-listed angle wins similarity
      ties, as the reference's ``>`` does;
    - ``Image.rotate(-angle, expand=False, fillcolor=white)`` applied to the **raw**
      camera frame before the square encoder resize, so on a 16:9 frame the 90/270
      hypotheses crop content and white-fill, exactly as published. Pillow's default
      ``NEAREST`` is kept here (:func:`rotate_frame`'s note): at exact multiples of 90
      degrees it is exact;
    - score = **mean similarity of the selected pairs** at ``num_pairs=48`` (the
      finder's temporary override), with tiling and ROI disabled - so ART-VS's tiled
      matcher searches untiled, as published;
    - an empty selection scores ``0.0`` (the reference's ``mean_similarity = 0.0``).

    **One deviation, and it is an improvement the old port already offered**: the
    selection subsample runs on an explicit :class:`torch.Generator` seeded by
    ``seed``, not on a fork of the process RNG. The old ``pick(seed=None)`` inherited
    the published unseeded behaviour, which means the same episode re-run in a fresh
    process scored the four angles differently - the random-24 failure shape. ``seed``
    is therefore required and has no default.
    """

    def __init__(self, backbone, *, seed: int, angles: Sequence[float] = DISCRETE_ANGLES,
                 num_pairs: int = SEARCH_NUM_PAIRS, threshold: float = 1.0,
                 tie_order: str = DEFAULT_TIE_ORDER) -> None:
        if seed is None:
            raise ValueError(
                "Prealign4Way needs a seed and has no default: its score is the mean "
                "similarity of a RANDOM subsample of the mutual pool, so an unseeded "
                "search picks a different angle in a fresh process (the published "
                "behaviour, and the incident this repository exists to prevent).")
        if len(angles) < 2:
            raise ValueError("need at least two hypothesis angles")
        if tie_order not in TIE_ORDERS:
            raise ValueError(f"tie_order must be one of {TIE_ORDERS}, got {tie_order!r}")
        self.backbone = backbone
        self.seed = int(seed)
        self.angles = tuple(float(a) for a in angles)
        self.num_pairs = int(num_pairs)
        self.threshold = float(threshold)
        self.tie_order = tie_order
        #: What the record's ``prealign`` column says. It is **derived from the
        #: settings**, not a constant (review R19 item 3): a 12-angle, 64-pair search
        #: is not the published library variant and must not record itself as one.
        #: ``DenseMatcher``/``TiledMatcher``/``ClassicalMatcher`` read it off the
        #: attached object with ``getattr(..., "label", None)``, so a matcher with no
        #: pre-aligner reports ``None`` through the same expression.
        self.label = ("4way-library"
                      if (self.angles == tuple(DISCRETE_ANGLES)
                          and self.num_pairs == SEARCH_NUM_PAIRS)
                      else f"{len(self.angles)}way-library-p{self.num_pairs}")
        #: the record of the last :meth:`pick`, so the arm can report what ran
        self.last: dict | None = None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"Prealign4Way(angles={list(self.angles)}, num_pairs={self.num_pairs}, "
                f"seed={self.seed})")

    def _selected_similarity(self, goal_tokens, rot_rgb: np.ndarray,
                             gen: torch.Generator) -> tuple[float, int]:
        """Mean similarity of the selected pairs; ``(0.0, 0)`` when there are none.

        ``gen`` is threaded through the four angles rather than re-seeded per angle,
        because that is the RNG *sequence* the reference produces: the old
        ``pick(seed=...)`` seeded the process generator once and the vendored
        ``find_correspondences_batch`` then drew one ``randperm`` per call, in angle
        order, skipping the draw entirely on an empty pool.
        """
        pool = mutual_nn_match(goal_tokens, self.backbone.encode(rot_rgb),
                               threshold=self.threshold, tie_order=self.tie_order)
        if pool.size == 0:
            return 0.0, 0
        take = min(self.num_pairs, pool.size)
        perm = torch.randperm(pool.size, generator=gen)[:take]
        sel = pool.sim[perm.to(pool.sim.device)]
        return float(sel.mean().item()), int(take)

    def pick(self, goal_rgb: np.ndarray, current_rgb: np.ndarray) -> dict:
        """Run the discrete search; return the measurement record.

        The goal is encoded once - the reference re-extracts it per angle with
        identical inputs, which is deterministic and therefore behaviourally
        identical; noted, not replicated.
        """
        goal_tokens = self.backbone.encode(goal_rgb)
        gen = torch.Generator().manual_seed(self.seed % (2**31))
        sims: dict[float, float] = {}
        counts: dict[float, int] = {}
        for ang in self.angles:
            rot = rotate_frame(current_rgb, ang, resample="nearest")
            s, n = self._selected_similarity(goal_tokens, rot, gen)
            sims[ang], counts[ang] = s, n
        best = max(self.angles, key=lambda a: sims[a])   # first max wins ties
        ordered = sorted(sims.values(), reverse=True)
        self.last = {
            "mode": "discrete",
            "variant": self.label,         # fence 6: NOT the 64-pair ROI-cropped one
            "angles_deg": list(self.angles),
            "num_pairs": self.num_pairs,
            "picked_deg": float(best),
            "similarities": {f"{a:g}": round(sims[a], 6) for a in self.angles},
            "n_selected": {f"{a:g}": counts[a] for a in self.angles},
            "margin": round(float(ordered[0] - ordered[1]), 6),
            "seed": self.seed,
        }
        return self.last

    @staticmethod
    def corrected_start(start_R_opt: np.ndarray, picked_deg: float) -> np.ndarray:
        """The reference's apply step: ``R @ Rz(-picked)``.

        ``rotate_camera_x_axis(initial_quaternion, -best_angle)`` in the published
        program (Gazebo's camera x is the optical axis; ours is z in the OpenCV
        convention), which cancels the measured roll.
        """
        return roll_about_optical_axis(start_R_opt, -float(picked_deg))


def matcher_chain(matcher) -> list:
    """A matcher and everything it wraps, outermost first.

    A deployed arm may be a stack of wrappers over a
    :class:`vsbench.vs.baselines.match.DenseMatcher`, and the rotation state can sit at any
    level, so the two readers of it (the record and the pre-servo hook) walk rather
    than guess. Cycle-safe by identity, because a wrapper
    that wrapped itself would otherwise hang here rather than fail at construction.
    """
    out, seen, m = [], set(), matcher
    while m is not None and id(m) not in seen:
        out.append(m)
        seen.add(id(m))
        m = getattr(m, "inner", None)
    return out


def find_prealign(matcher) -> Prealign4Way | None:
    """The :class:`Prealign4Way` anywhere in a matcher stack, or ``None``."""
    for m in matcher_chain(matcher):
        pre = getattr(m, "prealign", None)
        if isinstance(pre, Prealign4Way):
            return pre
    return None


def apply_prealign(scene, arm, config: dict, *, goal_rgb: np.ndarray) -> dict | None:
    """The pre-servo hook: measure the roll at the start pose and **re-pose the camera**.

    Called by :func:`vsbench.vs.bench.runner.run_episode` once, after the camera is placed
    at the start pose and before the first control step. Returns the pick record (also
    left on ``arm.matcher.prealign`` for :meth:`Arm.record_fields`), or ``None`` when
    the arm declares no pre-alignment - which is every arm to date.

    This is where the published methods' rotation compensation actually lives, and it
    is a **robot motion**: the episode's start pose is no longer the one the frozen
    start file names, so the recorded ``start_rotation_deg`` and the row's
    ``prealign`` field must be read together (fence 5).
    """
    pre = find_prealign(getattr(arm, "matcher", None))
    if pre is None:
        return None
    from vsbench.vs.scene import Pose  # noqa: PLC0415 - avoids an import cycle

    rgb, _depth, pose = scene.render()
    rec = pre.pick(goal_rgb, rgb)
    rec["applied"] = True
    # The quantity the search is trying to cancel, beside the angle it picked: the
    # swing-twist roll of the start pose relative to the goal. A pick is `aligned` when
    # the two agree to within half a hypothesis step (`classify_pick`), and without this
    # a record could not say so (review R19 item 8).
    rec["start_roll_before_deg"] = roll_component_deg(
        np.asarray(scene.goal_pose.rotation).T @ np.asarray(pose.rotation))
    scene.reset(Pose(position=pose.position,
                     rotation=pre.corrected_start(pose.rotation, rec["picked_deg"])))
    return rec


# ------------------------------------------------------------ the registry


#: Every key a ``rotation:`` block may carry. A misspelt one raises rather than
#: running the default - the suffix grammar's failure mode, in miniature.
ROTATION_KEYS = frozenset({"mode", "num_pairs", "prealign_angles", "prealign_on_ours"})


def rotation_config(config: dict) -> dict:
    """Validate a ``rotation:`` block and return it with its mode resolved.

    ``mode: none`` (the default, and every published number to date) needs nothing.
    ``mode: prealign_4way`` requires nothing beyond the mode - its constants ARE the
    published ones, which is the point of it - and records what it ran.
    """
    block = dict((config or {}).get("rotation") or {})
    unknown = sorted(set(block) - ROTATION_KEYS)
    if unknown:
        raise ValueError(f"unknown rotation keys: {unknown} (known: {sorted(ROTATION_KEYS)})")
    mode = str(block.get("mode", "none"))
    if mode not in ROTATION_MODES:
        raise ValueError(f"rotation.mode is {mode!r}, not one of: {', '.join(ROTATION_MODES)}")
    block["mode"] = mode
    return block


def build_rotation(config: dict, matcher: CorrespondenceMatcher) -> CorrespondenceMatcher:
    """Apply the config's ``rotation:`` block to a built matcher.

    ``prealign_4way`` attaches a :class:`Prealign4Way` to it (the search is a
    pre-servo step, not a matcher, so it hangs off the object for :func:`apply_prealign`
    and the record); ``none`` - the default and every published number to date - returns
    the matcher unchanged.

    One refusal, deliberate: a matcher with no encoder under ``prealign_4way`` (the
    published methods force the crutch off for classical detectors, which are
    rotation-invariant by construction).
    """
    block = rotation_config(config)
    if block["mode"] == "none":
        return matcher
    backbone = getattr(matcher, "backbone", None)
    if backbone is None:
        raise ValueError(
            "rotation.mode 'prealign_4way' scores rotations with the arm's own "
            f"encoder and {type(matcher).__name__} has none. The published methods "
            "force the crutch OFF for classical detectors (rotation-invariant by "
            "construction, prealign.py) - do not wrap them.")
    matcher.prealign = Prealign4Way(
        backbone, seed=int(config["seed"]),
        angles=tuple(block.get("prealign_angles") or DISCRETE_ANGLES),
        num_pairs=int(block.get("num_pairs", SEARCH_NUM_PAIRS)),
        tie_order=getattr(matcher, "tie_order", DEFAULT_TIE_ORDER))
    return matcher


def rotation_record_fields(matcher: CorrespondenceMatcher) -> dict:
    """The rotation columns of an arm's record.

    The pre-alignment's presence is stated in every table, and an omitted field reads
    as "not applicable" rather than "off" - so a plain arm says ``rotation_mode: none``
    and a 4-way arm says which angle it rolled to, from the same place.
    """
    pre = find_prealign(matcher)
    if pre is not None:
        last = pre.last or {}
        return {"rotation_mode": "prealign_4way",
                "prealign": last.get("variant", "4way-library"),
                "prealign_deg": last.get("picked_deg"),
                "prealign_margin": last.get("margin"),
                "prealign_start_roll_deg": last.get("start_roll_before_deg"),
                "prealign_similarities": last.get("similarities"),
                "prealign_seed": pre.seed}
    return {"rotation_mode": "none"}


#: The rotation columns every table must be able to read off a record: what the
#: mechanism was and whether the published crutch fired.
RECORD_KEYS = ("rotation_mode", "prealign")
