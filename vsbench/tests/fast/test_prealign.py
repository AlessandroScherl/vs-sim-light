# vs-sim copy: tests/fast/test_prealign.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The published 4-way pre-servo roll search, proven against the frozen code.

All on the CPU with no backbone weights:

1. :func:`vsbench.vs.baselines.prealign.unwarp_points` is the frozen ``rot_aware.unwarp_points``
   (from the stored table everywhere; read-only from ``../genvs-sim`` where it is
   checked out), and ``rotate -> unwarp`` is the identity in native pixels to 1e-9.
2. :class:`vsbench.vs.baselines.prealign.Prealign4Way` reproduces the frozen ``prealign`` -
   the rotate byte for byte, the pose arithmetic exactly, and the whole four-angle
   search against the stored record and, where the old repo exists, against the real
   ``RollPrealigner`` driven through the vendored ``find_correspondences_batch``.
3. The registry: a ``rotation:`` block validates, the 4-way attaches to a ViT matcher
   and never to a classical one, the label and the record say it ran, and the runner
   re-poses the camera before the loop.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vsbench.vs.baselines import prealign as rot
from vsbench.vs.baselines.match import DenseMatcher
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.bench.arms import arm_label, build_arm
from vsbench.vs.scene import D435I

OLD_REPO = Path(__file__).resolve().parents[2].parent / "genvs-sim"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: How far a point mapped by `unwarp_points` may sit from the frozen table, in pixels.
#: The map is one `cos`/`sin` pair and two multiply-adds, so its last bit belongs to the
#: interpreter's libm. Nine orders below a pixel; a real change fails by four or more.
TRIG_ROUNDOFF_PX = 1e-9


def _legacy(module: str):
    """Import a module from the frozen old repository, or skip naming why."""
    if not OLD_REPO.is_dir():
        pytest.skip(f"the frozen old repo is not checked out at {OLD_REPO}")
    if str(OLD_REPO) not in sys.path:
        sys.path.insert(0, str(OLD_REPO))
    import importlib

    return importlib.import_module(module)


def _fixture() -> dict:
    """The stored parity fixture - what proves the port on a machine WITHOUT the old repo."""
    return json.loads((FIXTURES / "u8_rotation_parity.json").read_text())


def _legacy_vendor():
    """The vendored ViT-VS matcher module the old prealign scored through."""
    if not OLD_REPO.is_dir():
        pytest.skip(f"the frozen old repo is not checked out at {OLD_REPO}")
    vendor = OLD_REPO / "src" / "control" / "_vitvs_vendor"
    for p in (str(OLD_REPO), str(vendor), str(vendor / "visual_servoing")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib

    if "rospy" not in sys.modules:
        sys.modules["rospy"] = importlib.import_module("src.control._rospy_stub")
    return importlib.import_module("visual_servoing.features.matcher")


# ------------------------------------------------------------- the stub encoder
FRAME_WH = (960, 540)


def _synthetic_goal(seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = FRAME_WH
    blocks = rng.integers(0, 256, size=(h // 5, w // 5, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.repeat(np.repeat(blocks, 5, axis=0), 5, axis=1))


def _tokens_from_512(arr512: np.ndarray) -> torch.Tensor:
    """A deterministic, content-addressed 18-d descriptor per 16x16 cell (with signed
    first moments, so it is not 180-degree invariant - real tokens are not either)."""
    a = arr512.astype(np.float64) / 255.0
    cells = a.reshape(32, 16, 32, 16, 3)
    mean = cells.mean(axis=(1, 3))
    std = cells.std(axis=(1, 3))
    off = (np.arange(16) - 7.5) / 7.5
    mx = (cells * off[None, None, None, :, None]).mean(axis=(1, 3))
    my = (cells * off[None, :, None, None, None]).mean(axis=(1, 3))
    feat = np.concatenate([mean, std, mx, mx * 0 + my, mean * std, mx * my], axis=-1)
    t = torch.from_numpy(feat).float()
    return torch.nn.functional.normalize(t, p=2, dim=-1)


class StubBackbone:
    """A :class:`vsbench.vs.baselines.features.FrozenBackbone`-shaped encoder with no weights."""

    name = "stub_cellmean"
    input_size = 512
    patch_size = 16
    grid_side = 32
    dim = 18
    device = "cpu"

    def encode(self, frame_rgb_u8: np.ndarray) -> torch.Tensor:
        from vsbench.vs.baselines.features import resize_pil_exact

        return _tokens_from_512(resize_pil_exact(np.asarray(frame_rgb_u8)[..., :3], 512))


def _dense(**kw) -> DenseMatcher:
    m = DenseMatcher(num_landmarks=kw.pop("num_landmarks", 24), seed=kw.pop("seed", 7),
                     backbone=StubBackbone(), **kw)
    m.set_episode("synthetic", "s0")
    return m


def _cfg(**rotation) -> dict:
    return {
        "seed": 20260902,
        "matcher": {"variant": "mutual_nn_seeded_subsample", "num_landmarks": 24,
                    "device": "cpu"},
        "stop": {"rule": "cns_plateau"},
        "scoring": {"success_rule": ["predicate_a"]},
        "rotation": dict(rotation),
    }


def _arm_config(**rotation) -> dict:
    cfg = _cfg(**rotation)
    cfg["arm"] = {"kind": "method"}
    cfg["controller"] = {"gain": 2.0}
    cfg["episode"] = {"max_iter": 3}
    return cfgmod.resolve(cfg)


def _build(cfg, **kw):
    return build_arm(cfg, intrinsics=D435I, matcher=_dense(), **kw)


# ------------------------------------------------------ 1. unwarp_points parity
def test_unwarp_points_is_the_frozen_function_stored():
    fx = _fixture()["unwarp_points"]
    pts = np.asarray(fx["pts"], np.float64)
    for key in fx["angles_deg"]:
        psi = float(key)
        got = rot.unwarp_points(pts, psi, tuple(fx["image_wh"]))
        assert np.allclose(got, np.asarray(fx["plain"][f"{psi:g}"]),
                           rtol=0.0, atol=TRIG_ROUNDOFF_PX), f"psi={psi}"
        got_raw = rot.unwarp_points(pts, psi, tuple(fx["squashed_wh"]),
                                    raw_wh=tuple(fx["raw_wh"]))
        assert np.allclose(got_raw, np.asarray(fx["raw"][f"{psi:g}"]),
                           rtol=0.0, atol=TRIG_ROUNDOFF_PX), f"psi={psi} raw"


def test_unwarp_points_is_the_frozen_function():
    legacy = _legacy("src.control.matchers.rot_aware")
    rng = np.random.default_rng(0)
    pts = rng.uniform(-50, 2000, size=(64, 2))
    for psi in (0.0, 7.5, 30.0, 90.0, 180.0, -137.25, 359.9):
        assert np.array_equal(legacy.unwarp_points(pts, psi, (1920, 1080)),
                              rot.unwarp_points(pts, psi, (1920, 1080))), f"psi={psi}"
        assert np.array_equal(
            legacy.unwarp_points(pts, psi, (512, 512), raw_wh=(1920, 1080)),
            rot.unwarp_points(pts, psi, (512, 512), raw_wh=(1920, 1080))), f"psi={psi}"


def test_rotate_then_unwarp_is_the_identity_in_native_pixels():
    """``rotate_frame`` moves content by ``R2(+psi)``; ``unwarp_points(psi)`` undoes it."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, size=(200, 2)) * np.array([1920.0, 1080.0])
    for theta in (0.0, 30.0, 60.0, 90.0, 137.0, 180.0, 330.0):
        there = rot.unwarp_points(pts, theta, (1920, 1080))
        back = rot.unwarp_points(there, -theta, (1920, 1080))
        assert np.abs(back - pts).max() < 1e-9, f"theta={theta}"


# ------------------------------------------------------------- 2. the 4-way
def test_the_ported_prealign_helpers_are_the_frozen_ones():
    legacy = _legacy("src.control.prealign")
    assert rot.DISCRETE_ANGLES == legacy.DISCRETE_ANGLES
    assert rot.SEARCH_NUM_PAIRS == legacy.SEARCH_NUM_PAIRS
    assert rot.FILL_WHITE == legacy.FILL_WHITE
    rng = np.random.default_rng(4)
    for _ in range(20):
        a, b = rng.uniform(-540, 540, 2)
        assert rot.circular_diff_deg(a, b) == legacy.circular_diff_deg(a, b)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    for ang in (0.0, 37.5, -90.0, 180.0):
        assert np.array_equal(rot.roll_about_optical_axis(R, ang),
                              legacy.roll_about_optical_axis(R, ang))
        assert rot.roll_component_deg(legacy.roll_about_optical_axis(np.eye(3), ang)) == \
            pytest.approx(legacy.roll_component_deg(
                legacy.roll_about_optical_axis(np.eye(3), ang)))
    for true_roll in (0.0, 45.0, 95.0, 179.0):
        for picked in rot.DISCRETE_ANGLES:
            assert rot.classify_pick(true_roll, picked) == \
                legacy.classify_pick(true_roll, picked)


def test_the_hypothesis_rotate_matches_the_stored_checksums():
    """The published rotate, byte for byte, without the old repo on disk."""
    fx = _fixture()["rotate_hypothesis"]
    goal = _synthetic_goal()
    assert hashlib.md5(goal.tobytes()).hexdigest() == fx["frame_md5"], \
        "the synthetic goal changed - regenerate the fixture and say why"
    for key, want in fx["md5"].items():
        got = rot.rotate_frame(goal, float(key), resample="nearest")
        assert hashlib.md5(got.tobytes()).hexdigest() == want, f"angle={key}"


def test_the_four_way_pick_matches_the_stored_frozen_record():
    """The frozen ``RollPrealigner`` record, stored - the parity a fresh clone can run.

    The "current" frame is the goal as a camera rolled by +90 degrees would see it
    (``rotate_frame(goal, -90)``), which is what the fixture was produced from.
    """
    fx = _fixture()["prealign_pick"]
    goal = _synthetic_goal()
    current = rot.rotate_frame(goal, -90.0)
    got = rot.Prealign4Way(StubBackbone(), seed=fx["seed"],
                           tie_order=fx["tie_order"]).pick(goal, current)
    want = fx["record"]
    assert got["picked_deg"] == want["picked_deg"]
    assert got["n_selected"] == want["n_selected"]
    assert got["similarities"] == want["similarities"]
    assert got["margin"] == want["margin"]


def test_the_hypothesis_rotate_is_byte_identical_to_the_published_one():
    """The published call is ``Image.rotate(-angle, fillcolor=white)`` at NEAREST."""
    legacy = _legacy("src.control.prealign")
    frame = _synthetic_goal()
    for ang in rot.DISCRETE_ANGLES + (17.0,):
        assert np.array_equal(rot.rotate_frame(frame, ang, resample="nearest"),
                              legacy.rotate_hypothesis(frame, ang)), f"angle={ang}"


class _FakeViTDenseMatcher:
    """The duck-typed surface the frozen ``RollPrealigner`` borrows, over the stub."""

    def __init__(self) -> None:
        self._extractor = self
        self._find_correspondences = _legacy_vendor().find_correspondences_batch
        self._input_size = 512
        self._patch_size = 16
        self._bin = False
        self._device = "cpu"
        self._torch = torch

    def preprocess_pil(self, pil):
        return torch.from_numpy(np.asarray(pil, dtype=np.uint8)[None].copy())

    def extract_descriptors(self, tensor, layer=11, facet="token", bin=False):  # noqa: A002
        arr = tensor[0].numpy()
        return _tokens_from_512(arr).reshape(1, 1, 32 * 32, -1)

    def _full_image_patch_mask(self, n_total: int):
        return torch.ones((1, int(n_total)), dtype=torch.bool)


def test_prealign_four_way_reproduces_the_frozen_search():
    """Same picked angle, same per-angle similarity, same selection count, against the
    frozen :class:`RollPrealigner` driven through the vendored matcher. ``tie_order=
    'legacy'`` is required and is the point: the published sort is unstable over
    equal cyclic scores, and it is that order the subsample indexes."""
    legacy = _legacy("src.control.prealign")
    goal = _synthetic_goal()
    current = rot.rotate_frame(goal, -90.0)
    seed = 2026
    want = legacy.RollPrealigner(_FakeViTDenseMatcher()).pick(goal, current, seed=seed)
    got = rot.Prealign4Way(StubBackbone(), seed=seed, tie_order="legacy").pick(goal, current)
    assert got["picked_deg"] == want["picked_deg"]
    assert got["n_selected"] == want["n_selected"]
    assert got["similarities"] == want["similarities"]
    assert got["margin"] == want["margin"]
    assert got["variant"] == "4way-library"     # not the 64-pair ROI variant


def test_prealign_needs_a_seed_and_the_corrected_start_cancels_the_roll():
    with pytest.raises(ValueError, match="no default"):
        rot.Prealign4Way(StubBackbone(), seed=None)
    R = np.eye(3)
    rolled = rot.roll_about_optical_axis(R, 90.0)
    back = rot.Prealign4Way.corrected_start(rolled, 90.0)
    assert np.abs(back - R).max() < 1e-12


def test_the_prealign_label_follows_its_settings():
    """A 12-angle, 64-pair search must not record itself as the published one."""
    published = rot.Prealign4Way(StubBackbone(), seed=0)
    assert published.label == "4way-library"
    other = rot.Prealign4Way(StubBackbone(), seed=0, angles=tuple(range(0, 360, 30)),
                             num_pairs=64)
    assert other.label == "12way-library-p64"
    cfg = _cfg(mode="prealign_4way", prealign_angles=list(range(0, 360, 30)))
    assert "pre12way" in arm_label(cfg)
    assert "pre4way" in arm_label(_cfg(mode="prealign_4way"))
    assert "pre4way" not in arm_label(_cfg())


# ------------------------------------------------------------- 3. the registry
def test_the_rotation_block_is_validated_and_defaults_to_none():
    assert rot.rotation_config({})["mode"] == "none"
    assert rot.build_rotation({}, _dense()).__class__ is DenseMatcher
    with pytest.raises(ValueError, match="unknown rotation keys"):
        rot.rotation_config(_cfg(mode="prealign_4way", n_banks=12))
    with pytest.raises(ValueError, match="rotation.mode"):
        rot.rotation_config(_cfg(mode="four_way"))
    with pytest.raises(ValueError, match="rotation.mode"):
        arm_label(_cfg(mode="banks"))            # the research repo's mode is not here


def test_the_four_way_attaches_to_a_vit_matcher_and_never_to_a_classical_one():
    m = rot.build_rotation(_cfg(mode="prealign_4way"), _dense())
    assert isinstance(m.prealign, rot.Prealign4Way)
    assert rot.rotation_record_fields(m)["rotation_mode"] == "prealign_4way"
    assert rot.rotation_record_fields(_dense())["rotation_mode"] == "none"

    class _NoEncoder:
        name = "classical_stub"

    with pytest.raises(ValueError, match="has none"):
        rot.build_rotation(_cfg(mode="prealign_4way"), _NoEncoder())


def test_build_arm_wires_the_rotation_block_and_labels_it():
    plain = _build(_arm_config())
    assert isinstance(plain.matcher, DenseMatcher)
    fields = plain.record_fields()
    assert fields["backbone"] == "stub_cellmean" and fields["prealign"] is None
    assert fields["rotation_mode"] == "none"
    assert not [k for k in fields if k.startswith("prealign_")]
    assert "pre4way" not in plain.name

    arm = _build(_arm_config(mode="prealign_4way"))
    assert isinstance(arm.matcher.prealign, rot.Prealign4Way)
    assert "pre4way" in arm.name
    fields = arm.record_fields()
    assert fields["prealign"] == "4way-library"          # NOT the matcher's None
    assert fields["rotation_mode"] == "prealign_4way"
    assert fields["prealign_deg"] is None                # nothing has run yet


def test_the_runner_re_poses_the_camera_before_the_loop():
    """``apply_prealign`` is a ROBOT MOTION, and the record must show both rotations."""
    from vsbench.vs.bench.runner import run_episode
    from vsbench.vs.scene import MockPlaneScene

    scene = MockPlaneScene(render_scale=0.125)
    goal = scene.goal_pose
    roll = np.deg2rad(88.0)
    c, s = np.cos(roll), np.sin(roll)
    start = {"start_id": "rolled",
             "cam_pos": goal.position + np.array([0.004, -0.003, 0.002]),
             "cam_R_opt": goal.rotation @ np.array([[c, -s, 0.0], [s, c, 0.0],
                                                    [0.0, 0.0, 1.0]])}
    cfg = _arm_config(mode="prealign_4way")
    arm = _build(cfg)
    rec = run_episode(scene, arm, start, cfg)
    assert rec["prealign"] == "4way-library"
    assert rec["prealign_deg"] in rot.DISCRETE_ANGLES
    assert rec["start_rotation_deg"] > 80.0             # the start file's own roll
    assert rec["start_rotation_deg_after_prealign"] < rec["start_rotation_deg"]
    assert rec["prealign_seed"] == cfg["seed"]

    plain = run_episode(scene, _build(_arm_config()), start, cfg | {"rotation": {}})
    assert plain["prealign"] is None and plain["rotation_mode"] == "none"
    assert plain["start_rotation_deg_after_prealign"] is None


def test_the_record_keys_are_the_columns_a_table_reads():
    plain = rot.rotation_record_fields(_dense())
    pre = rot.rotation_record_fields(rot.build_rotation(_cfg(mode="prealign_4way"), _dense()))
    for key in rot.RECORD_KEYS:
        assert key in pre, key
    assert "rotation_mode" in plain
