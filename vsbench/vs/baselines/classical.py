# vs-sim copy: vssim/baselines/classical.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Classical baselines: SIFT / ORB / AKAZE, and RANSAC as a switch (unit U3).

Purpose
-------
The hand-crafted feature baselines the paper compares against, behind the same
:class:`vsbench.vs.baselines.match.CorrespondenceMatcher` interface as the dense chain, so
they feed the *same* IBVS controller and are scored by the same stop. RANSAC is a
switch, not a default: protocol v2 section 4 asks for **both columns** - the
classical arms with their published geometric verification (the scene's model:
homography on the posters, the fundamental matrix on the rooms) and without it -
and every record states which one ran. The label carries the **setting**
(``.../ransac-scene_default/...`` vs ``.../ransac-none/...``, since that is what the
config asked for and it is one cell across scenes); the **resolved** model is on the
record as ``ransac_model``, beside ``ransac_requested``.
The ViT/ART arms stay RANSAC-free by the standing owner ruling.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
- ``src/control/matchers/classical.py`` -> :class:`ClassicalMatcher`: the
  detector guard, the per-scene RANSAC model, the goal cached at ``reset`` and
  both images re-detected per ``match``.
- ``src/control/matchers/geometric.py`` -> :func:`filter_by_ransac`, verbatim
  including the fail-open rule and :data:`MIN_MATCHES_AFTER_FILTER`. The
  planarity trade-off it documents at ``geometric.py:13-17`` is reproduced under
  ":data:`RANSAC_MODELS`" below, because it is the reason there is no default.
- ``src/control/_vitvs_vendor/.../features/classical_extractor.py`` - the parts
  ``classical.py`` used, **inlined rather than vendored**: ``_init_detector``
  (the cv2 constructor arguments, :data:`DETECTORS`), ``_init_flann_matcher``,
  ``preprocess_pil`` (greyscale only - no resize, which is why these keypoints
  are already full-image pixels and need no coordinate mapping),
  ``extract_keypoints_and_descriptors``, ``match_features`` and
  ``detect_features_classical`` -> :meth:`ClassicalMatcher._detect_and_match`.

Deliberately dropped
--------------------
``detect_features_classical_roi`` (the ROI/crop branch - this suite servos on the
full image), ``get_feature_info`` / ``get_supported_methods`` /
``get_descriptor_dim`` / ``get_patch_size`` (reporting helpers with no consumer),
the ``BaseFeatureExtractor`` ABC and its ``extract_descriptors`` stub (classical
methods raised ``NotImplementedError`` for it), and the rospy stub plus the
vendored ``sys.path`` injection that ``_ensure_vendored_imports`` existed for -
nothing in this repository imports the vendored tree.

Fences kept (each one changes numbers)
--------------------------------------
1. **The OpenCV pin.** ``cv2.AKAZE_create`` is absent from the default OpenCV 5
   build, so the constructor checks for the factory and names the pin
   (``opencv-python==4.12.0.88`` on the old dev box; ``pyproject.toml`` asks for
   ``>= 4.10``, and the two environments this repository runs, 4.10.0 and 4.11.0,
   were measured to produce **identical** keypoints, matches and inliers on the
   parity pair). A drifted environment fails here with the reason instead of
   raising ``AttributeError`` from inside a detector call.
2. **RANSAC fails open.** Too few points, no model, or a filter that would leave
   IBVS under-determined returns the *unfiltered* matches with
   ``applied=False``. Returning nothing is indistinguishable from "the controller
   diverged", so the caller must be able to tell them apart.
3. **The published confidence is row-misaligned, and it is kept that way.**
   ``match_features`` normalises the distances of the matches in *detector* order
   and then takes ``normalized_distances[:num_pairs]`` while the points come from
   the *distance-sorted* list (``classical_extractor.py`` L174-204). Row i's
   "similarity" is therefore some other match's score. It is preserved because it
   is what the published baseline ran; it is safe only because these arms are
   **unweighted** (``UNIFORM_WEIGHTING``), which is why
   :attr:`ClassicalMatcher.provides_margin` is ``False`` and
   :func:`vsbench.vs.bench.arms.build_arm` refuses to weight this matcher.
4. **OpenCV's RNG is seeded per episode and per frame** (:meth:`ClassicalMatcher.set_episode`).
   FLANN's randomised KD-tree forest is drawn from ``cv::theRNG()``, so the SIFT
   ratio-test set is *not* a function of the two images alone. Detection is
   deterministic; matching is not.
5. **Keypoints are already full-image pixels.** The vendored preprocessing only
   greyscales, so no crop or scale mapping is needed - unlike the ViT path. The
   sub-pixel keypoint is carried through :meth:`vsbench.vs.baselines.match.Matches.from_pixel_pairs`
   on the degenerate one-pixel lattice described at :data:`PIXEL_LATTICE_NOTE`,
   so ``to_rows()`` hands the controller exactly the pixels OpenCV produced.

6. **Lowe's ratio is a setting; the fielded arms' 0.7 is its default.** CNS runs its
   SIFT at 0.75 (the matching study's candidate 7), so ``lowe_ratio=`` and
   ``detector_kwargs=`` are constructor arguments that reach the object from
   ``matcher.lowe_ratio`` / ``matcher.detector_kwargs`` like every other matcher key
   (:func:`vsbench.vs.baselines.match.build_matcher` forwards the whole block), both are on the
   record, and every fielded arm file leaves them alone. The **per-match ratio** is in
   ``extras["match_ratio"]``, row-aligned - unlike the published similarity of fence 3 -
   and it is present only when RANSAC did not subset the rows (see :meth:`match`).

Golden tests that guard it
--------------------------
``tests/fast/test_classical.py`` against ``tests/fixtures/u3_classical_mug1.json``
(produced once by the frozen old ``ClassicalMatcher`` - see
``u3_fixture_provenance.md``): keypoint, match and inlier counts and the matched
pixel coordinates for all three detectors x all three RANSAC models; the
``ransac: none`` column is the pre-RANSAC match set; ``scene_default`` resolves
from a scene spec; and the registry builds every variant.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vsbench.vs.baselines.match import CorrespondenceMatcher, Matches, episode_seed

__all__ = [
    "DETECTORS",
    "MIN_MATCHES_AFTER_FILTER",
    "RANSAC_CHOICES",
    "MIN_POINTS_FOR_MODEL",
    "RANSAC_MODELS",
    "SCENE_DEFAULT",
    "ClassicalMatcher",
    "filter_by_ransac",
    "resolve_ransac",
]

#: The geometric models, and **there is no safe default** (``geometric.py:11-20``):
#:
#: - ``homography`` is correct for a **planar** target (the posters) and needs >= 4
#:   points. On a 3-D scene it is wrong - valid matches at different depths are not
#:   related by one homography and get discarded.
#: - ``fundamental`` is correct for a general **3-D** scene and needs >= 8 points,
#:   but is **degenerate for planar scenes**, exactly where homography is ideal.
#: - ``essential`` is the **calibrated** twin of ``fundamental``: with a known ``K``
#:   the model has five degrees of freedom instead of seven, so it rejects far more of
#:   a half-wrong pool. It is what the matching study's plan names for the rooms and
#:   what step 0 used (``findEssentialMat(K)``); it needs ``K`` and refuses without one.
#:   The published classical baselines do **not** use it - they run the model the scene
#:   declares, and no scene declares this one.
#: - ``none`` keeps the ratio-test matches unfiltered - protocol v2's second
#:   classical column, and what published ViT-VS does.
#:
#: So the model is a per-scene property. :data:`SCENE_DEFAULT` is how a config says
#: "whatever this scene declares" without a table of scenes living in this module.
RANSAC_MODELS = ("none", "homography", "fundamental", "essential")

#: ``matcher.ransac: scene_default`` -> the scene spec's own ``ransac`` field.
SCENE_DEFAULT = "scene_default"

#: What ``matcher.ransac`` may be.
RANSAC_CHOICES = (*RANSAC_MODELS, SCENE_DEFAULT)

#: Below this many surviving points IBVS is under-determined, so the filter fails
#: open instead of returning them (``geometric.MIN_MATCHES_AFTER_FILTER``).
MIN_MATCHES_AFTER_FILTER = 4

#: Fewest correspondences each model can be fitted from.
MIN_POINTS_FOR_MODEL = {"homography": 4, "fundamental": 8, "essential": 5}

#: Lowe's ratio (``classical_extractor.py`` L159) and the single-neighbour fallback
#: distance for SIFT (L163). Both are the published constants of the **ViT-VS**
#: classical extractor, and :data:`LOWE_RATIO` is the **default** of
#: :class:`ClassicalMatcher`, not a fixed law: ``lowe_ratio=`` is a per-matcher setting
#: (CLAUDE.md rule 3) because a second published front-end runs a second value - CNS's
#: is 0.75 (`../docs/matching_study_plan_2026-09-06.md` section 3, candidate 7). Every
#: fielded arm leaves it at 0.7, ``record_fields`` states whichever ran, and the U3
#: parity fixture is measured at the default.
LOWE_RATIO = 0.7
SIFT_SINGLE_MATCH_MAX_DISTANCE = 100.0

#: Detector -> (cv2 factory name, constructor kwargs, descriptor norm).
#: Verbatim from ``ClassicalFeatureExtractor._init_detector``: SIFT is the tuned
#: 300-keypoint / contrast 0.08 configuration the published code shipped (not
#: OpenCV's defaults), ORB asks for 1000 features, AKAZE takes its defaults.
DETECTORS: dict[str, tuple[str, dict[str, Any], str]] = {
    "sift": (
        "SIFT_create",
        {"nfeatures": 300, "nOctaveLayers": 3, "contrastThreshold": 0.08,
         "edgeThreshold": 10, "sigma": 1.6},
        "l2",
    ),
    "orb": ("ORB_create", {"nfeatures": 1000}, "hamming"),
    "akaze": ("AKAZE_create", {}, "hamming"),
}

#: Why :class:`ClassicalMatcher` hands :class:`vsbench.vs.baselines.match.Matches` a
#: ``side = input_size = image width``, ``patch_size = 1`` lattice: a classical
#: detector has no patch grid, and ``Matches`` is a patch object with an exact
#: pixel escape hatch (``from_pixel_pairs``). On this lattice every keypoint's
#: integer part is a distinct cell and its sub-pixel remainder lives in the offset
#: arrays, so ``to_rows()`` returns the OpenCV coordinates unchanged (measured
#: round-trip error 0 on the parity fixture) and the linear cell index the runner
#: derives for the stop's node set stays a stable per-keypoint identity.
PIXEL_LATTICE_NOTE = "side = input_size = image width, patch_size = 1"


def resolve_ransac(model: str, scene: Any = None) -> str:
    """``scene_default`` -> the scene's own model; every other value passes through.

    ``scene`` is anything that carries a ``ransac`` attribute or key - a
    :class:`vsbench.vs.scenes.SceneSpec`, or the scene dict a runbook already
    has. Duck-typed on purpose: :mod:`vsbench.vs.baselines.classical` must not import the data
    layer to answer a one-field question, and a scene table must never live here
    (the same rule that keeps working distances out of :mod:`refinevs.eval.rescore`).
    """
    if model not in RANSAC_CHOICES:
        raise ValueError(
            f"matcher.ransac is {model!r}, not one of: {', '.join(RANSAC_CHOICES)}"
        )
    if model != SCENE_DEFAULT:
        return model
    got = None
    if scene is not None:
        got = scene.get("ransac") if isinstance(scene, dict) else getattr(scene, "ransac", None)
    if got is None:
        raise ValueError(
            "matcher.ransac is 'scene_default' but no scene spec was given to resolve it "
            "from - pass scene_spec= to the matcher (vsbench.vs.scenes.SceneSpec.ransac "
            "is the field). The model is a per-scene property and there is no safe default."
        )
    if got not in RANSAC_MODELS:
        raise ValueError(
            f"the scene declares ransac {got!r}, not one of: {', '.join(RANSAC_MODELS)}"
        )
    return str(got)


def filter_by_ransac(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    conf: np.ndarray | None,
    model: str = "none",
    thresh_px: float = 3.0,
    K: np.ndarray | None = None,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, bool]:
    """Keep only correspondences consistent with a RANSAC-fitted model.

    Returns ``(pts_a, pts_b, conf, n_inliers, model_applied)``. **Fails open**
    (fence 2): on too few points, no model, or a filter that would leave IBVS
    under-determined, the *unfiltered* matches come back with
    ``model_applied=False``.

    ``K`` is required by - and only read by - the ``essential`` model, the calibrated
    five-point twin of ``fundamental``: a wrong ``K`` there is a silently wrong inlier
    set, so it is refused rather than defaulted.
    """
    import cv2

    n = len(pts_a)
    if model == "none":
        return pts_a, pts_b, conf, n, False
    if model not in RANSAC_MODELS:
        raise ValueError(f"model must be one of {RANSAC_MODELS}, got {model!r}")

    if model == "essential" and K is None:
        raise ValueError(
            "the essential matrix is the CALIBRATED model and has no safe default "
            "intrinsics: pass K= (exact in simulation), or ask for 'fundamental', "
            "which needs none")

    min_pts = MIN_POINTS_FOR_MODEL[model]
    if n < min_pts:
        return pts_a, pts_b, conf, n, False

    src = np.asarray(pts_a, np.float32).reshape(-1, 1, 2)
    dst = np.asarray(pts_b, np.float32).reshape(-1, 1, 2)
    if model == "homography":
        matrix, inl = cv2.findHomography(src, dst, cv2.RANSAC, thresh_px)
    elif model == "essential":
        matrix, inl = cv2.findEssentialMat(src, dst, np.asarray(K, np.float64),
                                           cv2.RANSAC, float(confidence),
                                           float(thresh_px))
    else:
        matrix, inl = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, thresh_px,
                                             float(confidence))

    if matrix is None or inl is None:
        return pts_a, pts_b, conf, n, False
    keep = inl.ravel().astype(bool)
    if int(keep.sum()) < MIN_MATCHES_AFTER_FILTER:
        return pts_a, pts_b, conf, int(keep.sum()), False
    return (pts_a[keep], pts_b[keep],
            None if conf is None else conf[keep], int(keep.sum()), True)


class ClassicalMatcher(CorrespondenceMatcher):
    honours_goal_mask = True      # OpenCV's detectAndCompute mask on the goal side
    """SIFT / ORB / AKAZE keypoint matching into the shared IBVS controller.

    ``num_landmarks`` is the published ``num_pairs``: the matches are sorted by
    descriptor distance and the best ``num_landmarks`` are kept. There is **no
    random draw** - the selection is deterministic - so ``seed`` is accepted for
    the common constructor shape, recorded, and unused, and
    :meth:`set_episode` is the base class's no-op.

    The goal view is cached at :meth:`reset` but its keypoints are not
    precomputed: the published ``detect_features_classical`` re-detects both
    images on every call and splitting that would diverge from its matching logic
    for no real gain (detection is milliseconds on CPU).
    """

    #: This matcher has no second-peak margin (fence 3): there is no similarity
    #: surface to take one from. :func:`vsbench.vs.bench.arms.build_arm` reads this and
    #: refuses ``controller.weighting.use_margin``, which would otherwise multiply
    #: every row by ``NaN`` and hand the solve a silent zero.
    provides_margin = False

    #: fence 3: the published similarity is row-misaligned, so it is a recorded number
    #: and never a weight. :func:`vsbench.vs.bench.arms.build_arm` reads this and refuses
    #: ``controller.weighting.use_trust``, which would otherwise feed some *other*
    #: match's score into this row's weight - a wrong number, not a NaN, so nothing
    #: downstream would notice.
    provides_conf_weight = False

    #: What ordered the selection, for the record's ``tie_order`` column: a sort by
    #: descriptor distance, not a seeded draw over a mutual pool.
    tie_order = "distance"

    def __init__(
        self,
        method: str,
        *,
        num_landmarks: int | None = None,
        ransac: str = SCENE_DEFAULT,
        ransac_thresh_px: float = 3.0,
        scene_spec: Any = None,
        seed: int | None = None,
        device: str = "cpu",
        lowe_ratio: float = LOWE_RATIO,
        detector_kwargs: dict[str, Any] | None = None,
    ) -> None:
        method = str(method).lower()
        if method not in DETECTORS:
            raise ValueError(f"method must be one of {tuple(DETECTORS)}, got {method!r}")
        if num_landmarks is None:
            raise ValueError(
                "ClassicalMatcher needs num_landmarks and it has no default - the config "
                "field 'matcher.num_landmarks' states it (CLAUDE.md rule 3). It is the "
                "published `num_pairs`: how many distance-sorted matches steer the loop."
            )
        num_landmarks = int(num_landmarks)
        if num_landmarks < 1:
            raise ValueError(f"num_landmarks must be >= 1, got {num_landmarks}")

        lowe_ratio = float(lowe_ratio)
        if not 0.0 < lowe_ratio <= 1.0:
            raise ValueError(
                f"lowe_ratio must be in (0, 1], got {lowe_ratio} - it is the ratio of "
                "the best descriptor distance to the second best that a match must beat")

        import cv2

        factory, kwargs, norm = DETECTORS[method]
        # The table holds the PUBLISHED settings of each detector; a candidate that runs
        # another program's detector states its differences here and they are recorded.
        # Nothing merges silently: `detector_kwargs` on the record is the resolved set.
        kwargs = {**kwargs, **dict(detector_kwargs or {})}
        if not hasattr(cv2, factory):  # fence 1
            raise RuntimeError(
                f"cv2.{factory} is unavailable in OpenCV {cv2.__version__}. The old repo "
                f"pinned opencv-python==4.12.0.88 and OpenCV 5 dropped AKAZE_create from "
                f"the default build; pyproject.toml asks for >= 4.10, where all three exist."
            )
        self.method = method
        self.name = f"classical_{method}"
        self.num_landmarks = num_landmarks
        #: Lowe's ratio this matcher runs (the module constant unless a config says
        #: otherwise), and the resolved detector arguments - both on the record.
        self.lowe_ratio = lowe_ratio
        self.detector_kwargs = dict(kwargs)
        self.seed = None if seed is None else int(seed)
        self.device = device
        self.ransac_thresh_px = float(ransac_thresh_px)
        #: what the config asked for (``scene_default`` included) and what it resolved to
        self.ransac_requested = str(ransac)
        self.ransac = resolve_ransac(str(ransac), scene_spec)
        self._detector = getattr(cv2, factory)(**kwargs)
        self._norm = cv2.NORM_L2 if norm == "l2" else cv2.NORM_HAMMING
        # Cached for SIFT exactly as the published extractor cached it, so the KD-tree
        # parameters are built once per matcher and not once per frame.
        self._flann = (
            cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 32})
            if method == "sift" else None
        )
        #: set by :meth:`set_episode`; ``None`` keeps OpenCV's process RNG untouched
        self.rng_seed: int | None = None
        self._goal_gray: np.ndarray | None = None
        self.image_wh: tuple[int, int] | None = None
        self._frames = 0
        self._frames_ransac_applied = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"ClassicalMatcher({self.method!r}, num_landmarks={self.num_landmarks}, "
            f"ransac={self.ransac!r}, ransac_thresh_px={self.ransac_thresh_px}, "
            f"lowe_ratio={self.lowe_ratio})"
        )

    # ----------------------------------------------------------------- episode
    def set_episode(self, scene: str, start_id: Any) -> None:
        """Pin OpenCV's global RNG to ``(scene, start_id, seed)`` - fence 5.

        The base class's no-op would be right if a classical detector were a pure
        function of its two images. **SIFT is not**: FLANN builds a randomised KD-tree
        forest from ``cv::theRNG()``, so the ratio-test set moves with the process RNG
        (measured on the parity pair: 101 vs 103 surviving matches across seeds, and
        three distinct sets over nine seedings). Only the top-``num_landmarks`` by
        descriptor distance happened to be stable there, which is why the fixture
        reproduces - that is luck, not a property. Seeding it the way the dense draw is
        seeded makes the classical arm a function of ``(scene, start, seed)`` too, and
        pairs it against every other arm at the same start.

        ORB and AKAZE match by brute force and are unaffected; RANSAC's estimator seeds
        its own generator per call (identical inlier sets across seeds confirm it).
        """
        self.rng_seed = None if self.seed is None else episode_seed(scene, start_id, self.seed)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        """Cache the goal view; ``goal_mask`` (bool ``[H, W]``) restricts the GOAL
        keypoints to the object through OpenCV's own ``detectAndCompute`` mask.

        The published classical baseline detects over the whole frame; the masked cell
        (`matcher.goal_mask: objects`, owner 2026-09-21) is a different, labelled cell
        beside it, never an edit to it. ``goal_depth`` is unused.
        """
        self._goal_mask_u8 = (None if goal_mask is None
                              else (np.asarray(goal_mask).astype(bool).astype(np.uint8) * 255))
        self._goal_gray = self._grey(goal_rgb)
        h, w = goal_rgb.shape[:2]
        self.image_wh = (int(w), int(h))
        self._frames = 0
        self._frames_ransac_applied = 0

    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        """Detect, ratio-test, take the best ``num_landmarks``, then RANSAC."""
        if current_mask is not None:
            raise ValueError("ClassicalMatcher does not support a per-frame current_mask")
        if self._goal_gray is None:
            raise RuntimeError("reset() must be called before match()")
        if self.rng_seed is not None:
            import cv2

            # One seed per FRAME, like the dense draw's `(episode_seed + i)`: a re-run of
            # this episode rebuilds the same KD-tree forest on the same frame index.
            cv2.setRNGSeed((self.rng_seed + self._frames) % 2**31)
        self._frames += 1

        pts_goal, pts_cur, conf, counts = self._detect_and_match(self._grey(current_rgb))
        ratio = counts.pop("_ratio_per_match", None)
        if pts_goal is None:
            return self._empty(counts)

        n_raw = len(pts_goal)
        pts_goal, pts_cur, conf, n_inliers, applied = filter_by_ransac(
            pts_goal, pts_cur, conf, self.ransac, self.ransac_thresh_px)
        self._frames_ransac_applied += int(applied)

        w, h = self.image_wh
        k = len(pts_goal)
        matches = Matches.from_pixel_pairs(
            pts_goal, pts_cur,
            side=w, input_size=w, patch_size=1, image_wh=(w, h),
            sim=np.asarray(conf, np.float64) if conf is not None else np.full(k, np.nan),
            cyclic=np.full(k, np.nan),        # a one-shot detector has no cyclic score
            margin=np.full(k, np.nan),        # fence 3 / `provides_margin`
            pool_size=int(counts["n_ratio_matches"]),
            num_landmarks=self.num_landmarks,
            draw_seed=None,                   # deterministic selection, no draw
            tie_order=self.tie_order,
        )
        matches.extras.update(counts, n_raw=n_raw, n_inliers=int(n_inliers),
                              ransac_model=self.ransac, ransac_applied=bool(applied),
                              lowe_ratio=self.lowe_ratio)
        # THE PER-MATCH RATIO, and why it is only here when the filter did not fire.
        # `filter_by_ransac` returns the surviving POINTS, not their row indices, so a
        # per-match array cannot be subset by it without re-implementing its arithmetic.
        # The matching study runs this matcher with `ransac: none` and verifies the
        # MatchSet afterwards (`refinevs.eval.matching_study.published`), where the
        # subset is `MatchSet.take` and every score follows it.
        if ratio is not None and not applied:
            matches.extras["match_ratio"] = ratio
        elif ratio is not None:
            matches.extras["match_ratio_absent"] = (
                "RANSAC subset the rows; run with ransac='none' and verify the match "
                "set separately to keep a per-match score aligned")
        return matches

    def record_fields(self) -> dict:
        """The published knobs this arm ran under - never a measurement.

        ``ransac_model`` is protocol v2's "record which column ran": the resolved
        model, beside what the config asked for and how often the filter actually
        fitted one (it fails open, so "configured" and "applied" are different facts).
        """
        return {
            "classical_detector": self.method,
            # WHICH published front-end this row is. The ratio and the detector's
            # arguments are settings (CLAUDE.md rule 3), so a table can never read a
            # 0.75-ratio row as the fielded 0.7 one.
            "lowe_ratio": self.lowe_ratio,
            "detector_kwargs": dict(self.detector_kwargs),
            # Protocol v2 section 4 asks every row to state the pre-alignment, not only
            # every ViT row (review R19 item 15). The published program forces the 4-way
            # OFF for classical detectors - they are rotation-invariant by construction -
            # so this is always `None`, and it is stated rather than omitted, because an
            # absent column reads as "not applicable" rather than "off".
            "prealign": getattr(getattr(self, "prealign", None), "label", None),
            "cv2_rng_seed": self.rng_seed,
            "conf_row_aligned": False,      # fence 3, on the record rather than implied
            "ransac_model": self.ransac,
            "ransac_requested": self.ransac_requested,
            "ransac_thresh_px": self.ransac_thresh_px,
            "ransac_frames_applied": self._frames_ransac_applied,
            "ransac_frames": self._frames,
        }

    # --------------------------------------------------------------- internals
    @staticmethod
    def _grey(rgb: np.ndarray) -> np.ndarray:
        """``preprocess_pil``: greyscale, **no resize** (fence 4)."""
        import cv2

        a = np.asarray(rgb)
        return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

    def _empty(self, counts: dict) -> Matches:
        w, h = self.image_wh
        empty = np.zeros((0, 2))
        matches = Matches.from_pixel_pairs(
            empty, empty, side=w, input_size=w, patch_size=1, image_wh=(w, h),
            sim=np.zeros(0), cyclic=np.zeros(0), margin=np.zeros(0),
            pool_size=int(counts.get("n_ratio_matches", 0)),
            num_landmarks=self.num_landmarks, draw_seed=None, tie_order=self.tie_order,
        )
        matches.extras.update(counts, n_raw=0, n_inliers=0,
                              ransac_model=self.ransac, ransac_applied=False,
                              reason="no matches from detector")
        return matches

    def _detect_and_match(self, cur_gray: np.ndarray):
        """``detect_features_classical`` + ``match_features``, line for line.

        Returns ``(pts_goal, pts_cur, conf, counts)`` with ``pts_goal is None``
        wherever the published code returned ``(None, None, None)``: either
        descriptor missing, FLANN refusing the pair, or fewer than four matches
        before *or* after the ``num_pairs`` clamp.
        """
        import cv2

        kp_goal, des_goal = self._detector.detectAndCompute(self._goal_gray, self._goal_mask_u8)
        kp_cur, des_cur = self._detector.detectAndCompute(cur_gray, None)
        counts = {"n_keypoints_goal": len(kp_goal or []), "n_keypoints_cur": len(kp_cur or []),
                  "n_ratio_matches": 0}
        if des_goal is None or des_cur is None:
            return None, None, None, counts

        if self.method == "sift":
            try:
                knn = self._flann.knnMatch(des_goal, des_cur, k=2)
            except cv2.error:
                return None, None, None, counts
            matches = []
            ratios: list[float] = []
            for pair in knn:
                if len(pair) == 2:
                    best, second = pair
                    if best.distance < self.lowe_ratio * second.distance:
                        matches.append(best)
                        # the ratio itself, as a per-match ambiguity score (lower is
                        # better) - it is what the gate tested, not a second statistic
                        ratios.append(float(best.distance)
                                      / max(float(second.distance), 1e-12))
                elif len(pair) == 1 and pair[0].distance < SIFT_SINGLE_MATCH_MAX_DISTANCE:
                    matches.append(pair[0])
                    ratios.append(float("nan"))    # no second neighbour to divide by
        else:
            matches = cv2.BFMatcher(self._norm, crossCheck=True).match(des_goal, des_cur)
            ratios = [float("nan")] * len(matches)   # crossCheck has no ratio

        counts["n_ratio_matches"] = len(matches)
        if len(matches) < MIN_MATCHES_AFTER_FILTER:
            return None, None, None, counts

        # Fence 3: normalised in DETECTOR order, then sliced by position after the
        # matches have been re-sorted by distance. Kept exactly as published.
        distances = np.array([m.distance for m in matches])
        lo, hi = float(distances.min()), float(distances.max())
        normalised = (1 - (distances - lo) / (hi - lo)) if hi > lo else np.ones_like(distances)

        # `sorted(matches, key=...)` with the indices instead of the objects: Python's
        # sort is stable and the key is the same, so the order is the published one and
        # the per-match ratio can follow it.
        order = sorted(range(len(matches)), key=lambda i: matches[i].distance)
        matches = [matches[i] for i in order]
        ratios = [ratios[i] for i in order]
        take = min(self.num_landmarks, len(matches))
        if take < MIN_MATCHES_AFTER_FILTER:
            return None, None, None, counts
        matches = matches[:take]
        counts["_ratio_per_match"] = np.asarray(ratios[:take], np.float64)

        pts_goal = np.asarray([kp_goal[m.queryIdx].pt for m in matches], np.float64)
        pts_cur = np.asarray([kp_cur[m.trainIdx].pt for m in matches], np.float64)
        return pts_goal, pts_cur, np.asarray(normalised[:take], np.float64), counts
