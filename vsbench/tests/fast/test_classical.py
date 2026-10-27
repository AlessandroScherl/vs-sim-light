# vs-sim copy: tests/fast/test_classical.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""U3 - the classical baselines, against the frozen old `ClassicalMatcher`.

`tests/fixtures/u3_classical_mug1.json` holds one run of the OLD code over the
same (goal, current) pair U1 and U2 use, for all three detectors crossed with all
three RANSAC models (`u3_fixture_provenance.md`). Everything here asserts against
that: keypoint counts, match counts, inlier counts, and the matched pixel
coordinates to 1e-6.

Determinism, precisely (corrected by review R16): **detection** is deterministic -
SIFT's and AKAZE's are, and ORB's "random" BRIEF pattern is a fixed table -
but **SIFT's matching is not**. FLANN builds a randomised KD-tree forest from
`cv::theRNG()`, so its ratio-test set moves with the process RNG: 101 vs 103
surviving matches over nine seedings on this very pair. The full nine cells came
out identical anyway because only the top-24 by descriptor distance are kept and
those happened to be stable - luck, not a property. So `cv2.setRNGSeed(0)` before
each cell is what actually pins this fixture (the maker does the same), and
`ClassicalMatcher.set_episode` seeds the RNG per episode and per frame so a
fielded classical arm is a function of `(scene, start, seed)` like every other.
RANSAC's estimator seeds its own generator per call - identical inlier sets across
seeds confirm it - and ORB/AKAZE match by brute force, so both are unaffected.

The fixture was made under cv2 4.11.0 and reproduces bit-for-bit under 4.10.0, the
two versions this repository's Python environments run. It does **not** reproduce
under 4.12.0, which the station's `genvs` env carries: AKAZE's detector returns 24
keypoints there where 4.11 returned 23 (measured on the station, 2026-09-03). That is
OpenCV moving between releases, not the port drifting, so the parity test states its
version range (`conftest.CV2_FIXTURE_VERSIONS`) and **skips** outside it. The flight
interpreter is unaffected: Isaac's python carries cv2 4.11.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import CV2_FIXTURE_VERSIONS

from vsbench.vs.baselines.classical import (
    MIN_POINTS_FOR_MODEL,
    RANSAC_MODELS,
    ClassicalMatcher,
    filter_by_ransac,
    resolve_ransac,
)
from vsbench.vs.baselines.match import build_matcher
from vsbench.vs.bench import arms
from vsbench.vs.bench import config as cfgmod

pytest.importorskip("cv2", reason="the classical baselines are OpenCV")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = json.loads((FIXTURES / "u3_classical_mug1.json").read_text())
CELLS = FIXTURE["cells"]
METHODS = ("sift", "orb", "akaze")


@pytest.fixture(scope="module")
def frames():
    from PIL import Image

    goal = np.asarray(Image.open(FIXTURES / "u1_frame_mug1_goal.jpg").convert("RGB"))
    cur = np.asarray(Image.open(FIXTURES / "u2_frame_mug1_roll010.jpg").convert("RGB"))
    return goal, cur


def _run(frames, method: str, ransac: str, **kw):
    import cv2

    cv2.setRNGSeed(0)
    goal, cur = frames
    matcher = ClassicalMatcher(method, num_landmarks=FIXTURE["num_pairs"],
                               ransac=ransac, **kw)
    matcher.reset(goal)
    return matcher, matcher.match(cur)


def _intrinsics():
    from vsbench.vs.baselines.control import Intrinsics

    return Intrinsics(fx=1406.75, fy=1406.75, cx=960.0, cy=540.0)


# --------------------------------------------------------------- parity


#: The models the frozen fixture pins - the three the published classical baselines can
#: run. `essential` was added for the matching study's rooms (it needs a K, which a
#: classical arm never has) and has no column in `u3_classical_mug1.json`; it is covered
#: by its own tests below rather than by a fixture that predates it.
FIXTURE_RANSAC_MODELS = ("none", "homography", "fundamental")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("ransac", FIXTURE_RANSAC_MODELS)
def test_parity_with_the_old_classical_matcher(frames, method, ransac):
    """Counts and coordinates, cell by cell, against the frozen old matcher.

    Pinned to the cv2 versions the fixture was measured on - see the module docstring
    for why 4.12 legitimately differs and why that is a skip rather than a failure.
    """
    if not cv2.__version__.startswith(CV2_FIXTURE_VERSIONS):
        pytest.skip(f"cv2 {cv2.__version__}: the classical fixture is pinned to "
                    f"{' / '.join(CV2_FIXTURE_VERSIONS)} (AKAZE's keypoint count moved "
                    "in 4.12); the flight interpreter runs 4.11.0")
    want = CELLS[f"{method}/{ransac}"]
    matcher, matches = _run(frames, method, ransac)
    extras = matches.extras

    assert extras["n_keypoints_goal"] == want["n_keypoints_goal"]
    assert extras["n_keypoints_cur"] == want["n_keypoints_cur"]
    assert extras["n_raw"] == want["n_raw"]
    assert extras["n_inliers"] == want["n_inliers"]
    assert extras["ransac_applied"] is want["ransac_applied"]
    assert len(matches) == want["n"]

    rows = matches.to_rows()
    # 1e-6 is the stated bar; the measured round-trip through the pixel lattice is 0.
    np.testing.assert_allclose(rows["pts_goal"], want["pts_goal"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(rows["pts_cur"], want["pts_cur"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(rows["conf"], want["conf"], atol=1e-12, rtol=0)
    assert matcher.record_fields()["ransac_model"] == ransac


@pytest.mark.parametrize("method", METHODS)
def test_the_pixel_lattice_round_trip_is_exact(frames, method):
    """Fence 4: `to_rows()` hands the controller OpenCV's own sub-pixel keypoints.

    `Matches` is a patch object; a classical detector has no patch grid, so the
    coordinates go through `from_pixel_pairs` on a one-pixel lattice. If that split
    were lossy every classical landmark would move by up to half a cell, silently.

    The x axis is bit-exact (the lattice pitch is the image width, so its scale is
    exactly 1). The y axis is not, and cannot be: it goes out through `w/h` and back
    through `h/w`, which is a float round trip - measured at 3e-13 px here, i.e.
    ~13 orders of magnitude below the half-cell error the split exists to avoid, and
    seven below the 1e-6 parity bar.
    """
    want = CELLS[f"{method}/none"]
    _, matches = _run(frames, method, "none")
    rows = matches.to_rows()
    for key in ("pts_goal", "pts_cur"):
        delta = np.abs(rows[key] - np.asarray(want[key]))
        assert delta[:, 0].max() == 0.0, f"{key}: the x axis must be bit-exact"
        assert delta[:, 1].max() < 1e-9, f"{key}: y round trip {delta[:, 1].max()}"
    assert matches.goal_offsets_patch is not None and matches.offsets_patch is not None


# ------------------------------------------------ the without-RANSAC column


@pytest.mark.parametrize("method", METHODS)
def test_ransac_none_is_the_pre_ransac_match_set(frames, method):
    """Protocol v2 section 4's second classical column, and it is a real subset.

    `ransac: none` must be exactly what the filter was handed, and each filtered
    column must be a subset of it - never a re-detection with different keypoints.
    """
    _, unfiltered = _run(frames, method, "none")
    base = unfiltered.to_rows()
    assert len(unfiltered) == unfiltered.extras["n_raw"] == CELLS[f"{method}/none"]["n_raw"]
    assert unfiltered.extras["ransac_applied"] is False

    for model in ("homography", "fundamental"):
        _, filtered = _run(frames, method, model)
        assert filtered.extras["n_raw"] == unfiltered.extras["n_raw"]
        rows = filtered.to_rows()
        kept = {tuple(np.round(p, 9)) for p in rows["pts_goal"]}
        allp = {tuple(np.round(p, 9)) for p in base["pts_goal"]}
        assert kept <= allp, f"{method}/{model} returned points the unfiltered set has not"
        assert len(filtered) <= len(unfiltered)


def test_the_record_states_which_ransac_column_ran(frames):
    """`ransac_model` on the record, plus what was asked for and how often it fitted."""
    matcher, _ = _run(frames, "sift", "homography")
    fields = matcher.record_fields()
    assert fields["ransac_model"] == "homography"
    assert fields["ransac_requested"] == "homography"
    assert fields["classical_detector"] == "sift"
    assert (fields["ransac_frames"], fields["ransac_frames_applied"]) == (1, 1)

    matcher, _ = _run(frames, "sift", "none")
    fields = matcher.record_fields()
    assert fields["ransac_model"] == "none"
    assert fields["ransac_frames_applied"] == 0, "'none' never fits a model"


def test_fail_open_is_visible_rather_than_silent():
    """Too few points, or a filter that would starve IBVS: unfiltered, `applied=False`."""
    pts = np.arange(6, dtype=np.float64).reshape(3, 2)
    for model in ("homography", "fundamental"):
        a, b, conf, n, applied = filter_by_ransac(pts, pts + 1.0, None, model)
        assert applied is False and n == 3
        np.testing.assert_array_equal(a, pts)
        np.testing.assert_array_equal(b, pts + 1.0)
        assert conf is None
    with pytest.raises(ValueError, match="model must be one of"):
        filter_by_ransac(pts, pts, None, "affine")


# ------------------------------------------------------- the scene default


class _Scene:
    ransac = "fundamental"


def test_scene_default_resolves_from_the_scene_spec():
    assert resolve_ransac("scene_default", _Scene()) == "fundamental"
    assert resolve_ransac("scene_default", {"ransac": "homography"}) == "homography"
    assert resolve_ransac("none", _Scene()) == "none", "an explicit model wins"
    with pytest.raises(ValueError, match="no scene spec"):
        resolve_ransac("scene_default")
    with pytest.raises(ValueError, match="the scene declares"):
        resolve_ransac("scene_default", {"ransac": "affine"})
    with pytest.raises(ValueError, match="matcher.ransac is"):
        resolve_ransac("maybe")


def test_scene_default_resolves_from_the_real_scene_specs():
    """The planar/3-D split protocol v2 section 4 names, read off U6's own specs."""
    scenes = pytest.importorskip("vsbench.vs.scenes", reason="U6's data layer")
    try:
        specs, _ = scenes.load_scenes()
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        pytest.skip(f"configs/scenes/eval_scenes.yaml unavailable ({type(exc).__name__}: {exc})")
    assert resolve_ransac("scene_default", specs["hollywood"]) == "homography"
    assert resolve_ransac("scene_default", specs["cluttered2"]) == "fundamental"


def test_a_classical_matcher_carries_the_resolved_model(frames):
    matcher = ClassicalMatcher("orb", num_landmarks=24, ransac="scene_default",
                               scene_spec=_Scene())
    assert (matcher.ransac, matcher.ransac_requested) == ("fundamental", "scene_default")


# ------------------------------------------------------------- the registry


@pytest.mark.parametrize("method", METHODS)
def test_the_registry_builds_every_classical_variant(method):
    cfg = {"seed": 0,
           "matcher": {"variant": f"classical_{method}", "num_landmarks": 24,
                       "ransac": "scene_default"},
           "stop": {"rule": "cap_only"}, "scoring": {"success_rule": ["predicate_a"]}}
    matcher = build_matcher(cfg, scene_spec=_Scene())
    assert isinstance(matcher, ClassicalMatcher)
    assert (matcher.method, matcher.num_landmarks, matcher.ransac) == (method, 24, "fundamental")
    assert matcher.name == f"classical_{method}"


def test_a_misspelt_matcher_key_is_a_typeerror_not_a_silent_default():
    cfg = {"seed": 0,
           "matcher": {"variant": "classical_sift", "num_landmarks": 24, "ransak": "none"},
           "stop": {"rule": "cap_only"}, "scoring": {"success_rule": ["predicate_a"]}}
    with pytest.raises(TypeError, match="ransak"):
        build_matcher(cfg)


def test_the_label_carries_the_ransac_column():
    """Two columns, two methods: a table must not be able to pool them."""
    def label(model):
        return arms.arm_label({
            "seed": 0,
            "matcher": {"variant": "classical_sift", "num_landmarks": 24, "ransac": model},
            "controller": {"weighting": {"use_trust": False, "use_margin": False,
                                         "per_axis": False}},
            "stop": {"rule": "cns_plateau"}, "scoring": {"success_rule": ["predicate_a"]}})

    assert "ransac-none" in label("none")
    assert "ransac-homography" in label("homography")
    assert label("none") != label("homography")


def test_build_arm_refuses_to_weight_a_matcher_that_cannot_be_weighted():
    """Both weighting defaults are True, and both would corrupt a classical arm.

    `use_margin` on a NaN margin zeroes the solve silently. `use_trust` is worse: it
    falls back to `rows["conf"]`, which on this matcher is the published
    row-misaligned similarity (fence 3) - a wrong number rather than a NaN, so nothing
    downstream would notice. Both are refused; the configs say `false` out loud.
    """
    base = {
        "seed": 0,
        "matcher": {"variant": "classical_orb", "num_landmarks": 24, "ransac": "none"},
        "stop": {"rule": "cap_only"}, "scoring": {"success_rule": ["predicate_a"]}}
    cfg = cfgmod.resolve(base)
    with pytest.raises(ValueError, match="reports no margin"):
        arms.build_arm(cfg, intrinsics=_intrinsics())

    cfg = cfgmod.resolve(base)
    cfg["controller"] = {"weighting": {"use_trust": True, "use_margin": False,
                                       "per_axis": False}}
    with pytest.raises(ValueError, match="no weightable confidence"):
        arms.build_arm(cfg, intrinsics=_intrinsics())

    cfg = cfgmod.resolve(base)
    cfg["controller"] = {"weighting": {"use_trust": False, "use_margin": False,
                                       "per_axis": False}}
    arm = arms.build_arm(cfg, intrinsics=_intrinsics())
    assert isinstance(arm.matcher, ClassicalMatcher)
    fields = arm.record_fields()
    assert fields["ransac_model"] == "none"
    assert fields["conf_row_aligned"] is False


# ----------------------------------------------------------------- refusals


def test_num_landmarks_has_no_default():
    with pytest.raises(ValueError, match="num_landmarks"):
        ClassicalMatcher("sift", ransac="none")
    with pytest.raises(ValueError, match="num_landmarks must be"):
        ClassicalMatcher("sift", num_landmarks=0, ransac="none")


def test_an_unknown_detector_is_refused():
    with pytest.raises(ValueError, match="method must be one of"):
        ClassicalMatcher("surf", num_landmarks=24, ransac="none")


def test_a_goal_mask_is_honoured_and_a_frame_mask_is_refused(frames):
    """A goal mask restricts the GOAL keypoints through OpenCV's own detector mask (the
    masked cell, owner 2026-09-21); a per-frame mask has no published meaning and stays
    refused rather than ignored."""
    goal, cur = frames
    matcher = ClassicalMatcher("orb", num_landmarks=24, ransac="none")
    h, w = goal.shape[:2]
    mask = np.zeros((h, w), bool)
    mask[: h // 2] = True                       # the top half only
    matcher.reset(goal, goal_mask=mask)
    rows = matcher.match(cur).to_rows()
    assert len(rows["pts_goal"]) and (rows["pts_goal"][:, 1] <= h // 2 + 1).all()
    matcher.reset(goal)
    with pytest.raises(ValueError, match="current_mask"):
        matcher.match(cur, np.ones(cur.shape[:2], bool))


def test_match_before_reset_is_refused(frames):
    with pytest.raises(RuntimeError, match="reset"):
        ClassicalMatcher("orb", num_landmarks=24, ransac="none").match(frames[1])


def test_a_frame_with_no_matches_returns_empty_matches_not_an_exception():
    """A blank pair: the detector finds nothing, and the loop must keep running."""
    blank = np.zeros((240, 320, 3), np.uint8)
    matcher = ClassicalMatcher("sift", num_landmarks=24, ransac="none")
    matcher.reset(blank)
    matches = matcher.match(blank)
    assert len(matches) == 0
    assert matches.extras["reason"] == "no matches from detector"


# ----------------------------------------------------- the OpenCV RNG (R16)


def test_set_episode_pins_opencv_rng_to_scene_start_and_seed(frames):
    """FLANN's KD-tree forest comes out of `cv::theRNG()`, so it must be seeded.

    Without this, SIFT's ratio-test set is not a function of the two images and two
    arms at the same start could see different matches - the random-24 failure shape
    with a different front end.
    """
    from vsbench.vs.baselines.match import episode_seed

    matcher = ClassicalMatcher("sift", num_landmarks=24, ransac="none", seed=7)
    assert matcher.rng_seed is None, "no episode named yet"
    matcher.set_episode("hollywood", 3)
    assert matcher.rng_seed == episode_seed("hollywood", 3, 7)
    assert matcher.record_fields()["cv2_rng_seed"] == matcher.rng_seed
    # a different start is a different seed; a seedless matcher leaves the RNG alone
    matcher.set_episode("hollywood", 4)
    assert matcher.rng_seed != episode_seed("hollywood", 3, 7)
    unseeded = ClassicalMatcher("sift", num_landmarks=24, ransac="none")
    unseeded.set_episode("hollywood", 3)
    assert unseeded.rng_seed is None


def test_a_seeded_classical_episode_is_reproducible_frame_by_frame(frames):
    """Same (scene, start, seed) twice -> the same matches on every frame.

    Deliberately run with OpenCV's RNG left in a *different* state before each
    replay, which is what a re-run in a fresh process amounts to.
    """
    import cv2

    goal, cur = frames

    def episode(disturb: int):
        cv2.setRNGSeed(disturb)
        m = ClassicalMatcher("sift", num_landmarks=24, ransac="none", seed=0)
        m.set_episode("hollywood", 3)
        m.reset(goal)
        return [m.match(cur).to_rows()["pts_cur"] for _ in range(3)]

    first, second = episode(12345), episode(999)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_the_scene_spec_refuses_to_default_a_ransac_model():
    """A scene that forgets to declare one used to become the *without*-RANSAC column
    silently, while still wearing the with-RANSAC label."""
    scenes = pytest.importorskip("vsbench.vs.scenes", reason="U6's data layer")
    spec = scenes.SceneSpec(name="nameless", kind="poster", raw={})
    with pytest.raises(ValueError, match="no `ransac` model"):
        _ = spec.ransac
    with pytest.raises(ValueError, match="no `ransac` model"):
        resolve_ransac("scene_default", spec)


# ------------------------------- the essential model (the matching study's rooms)
def test_the_essential_model_refuses_to_guess_intrinsics():
    """A wrong K is a silently wrong inlier set, so there is no default (R32 f7)."""
    pts = np.random.default_rng(0).uniform(0, 100, (20, 2))
    with pytest.raises(ValueError, match="CALIBRATED model"):
        filter_by_ransac(pts, pts.copy(), None, "essential", 3.0)


def test_the_essential_model_keeps_the_consistent_matches_and_fails_open():
    """Five points minimum, and the calibrated model rejects what disagrees with it."""
    rng = np.random.default_rng(3)
    K = np.array([[200.0, 0, 128], [0, 200.0, 72], [0, 0, 1.0]])
    # a pure sideways translation: the epipolar geometry is exact for these pairs
    X = np.column_stack([rng.uniform(-0.3, 0.3, 40), rng.uniform(-0.2, 0.2, 40),
                         rng.uniform(1.0, 2.0, 40)])
    a = (K @ X.T).T
    a = a[:, :2] / a[:, 2:3]
    Xb = X - np.array([0.05, 0.0, 0.0])
    b = (K @ Xb.T).T
    b = b[:, :2] / b[:, 2:3]
    # The camera moved sideways, so the epipolar lines are horizontal and an epipolar
    # check cannot see a displacement ALONG one - only across it. The ten bad matches
    # are therefore moved in y, which is what the model is entitled to reject.
    b[-10:, 1] += rng.uniform(20, 40, 10)
    idx = np.arange(40, dtype=np.float64)
    _, _, kept, n_inl, applied = filter_by_ransac(a, b, idx, "essential", 1.0, K=K)
    assert applied is True and n_inl >= 25
    assert set(np.asarray(kept, int).tolist()) <= set(range(30))
    # below five points it fails open, like every other model
    _, _, _, _, applied_small = filter_by_ransac(a[:4], b[:4], None, "essential", 1.0, K=K)
    assert applied_small is False


def test_the_model_minimums_are_stated_once():
    assert MIN_POINTS_FOR_MODEL == {"homography": 4, "fundamental": 8, "essential": 5}
    assert set(MIN_POINTS_FOR_MODEL) | {"none"} == set(RANSAC_MODELS)
