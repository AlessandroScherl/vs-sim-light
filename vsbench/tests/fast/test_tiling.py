# vs-sim copy: tests/fast/test_tiling.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""U3 - ART-VS, against the frozen old `vit_tiled.ARTVSMatcher`.

`tests/fixtures/u3_artvs_dinov3.json` holds one run of the OLD code over the same
(goal, current) pair U1 and U2 use: the tile grid and its coordinates, one
**coarse** frame, and one **tiled** frame (forced with the vendored
`switch_iteration=1`). It is on DINOv3 because a patch-**16** backbone is what
exercises trap 1, the `TilingConfig.patch_size = 14` hardcode: on DINOv2 the fence
would have 14 on both sides and prove nothing. (The DINOv2 torch.hub weights are in
the laptop's cache and are never downloaded - `u3_fixture_provenance.md`.)

The parity tests need the DINOv3 weights; where they are absent (the laptop `bot`
env has transformers but the station has the cache) they skip with the reason,
exactly as `tests/fast/test_features.py` does. Everything that does **not** need a
backbone - the tiling geometry, the (y, x) flip, the latch, the fences - runs
everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vsbench.vs.baselines.match import Matches, episode_seed
from vsbench.vs.baselines.tiling import (
    DEFAULT_SWITCH_THRESHOLD,
    TiledMatcher,
    TilingConfig,
    TilingHandler,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = json.loads((FIXTURES / "u3_artvs_dinov3.json").read_text())
TILING = FIXTURE["tiling"]
SCENE, START_ID, SEED = FIXTURE["scene"], FIXTURE["start_id"], FIXTURE["seed"]

#: measured 2.3e-13 px on the tiled frame (the pixel round trip), 0.0 on the coarse one
POINT_ATOL = 1e-9
#: measured 1.3e-6: the U1 token gap (the old chain's fused rescale+normalize) seen
#: through a cosine, not a difference in this unit's arithmetic
CONF_ATOL = 1e-5


@pytest.fixture(scope="module")
def frames():
    from PIL import Image

    goal = np.asarray(Image.open(FIXTURES / "u1_frame_mug1_goal.jpg").convert("RGB"))
    cur = np.asarray(Image.open(FIXTURES / "u2_frame_mug1_roll010.jpg").convert("RGB"))
    return goal, cur


@pytest.fixture(scope="module")
def backbone():
    """DINOv3 at the ART-VS input size, or a skip naming why it is unavailable."""
    pytest.importorskip("transformers", reason="the fast tier runs without the backbone")
    from vsbench.vs.baselines.features import FrozenBackbone

    try:
        return FrozenBackbone("dinov3_vits16",
                              input_size=TILING["backbone_input_size"], device="cpu")
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        pytest.skip(
            f"DINOv3 weights are not in the local HF cache ({type(exc).__name__}: "
            f"{str(exc).splitlines()[0]}); never download them from a test."
        )


def _matcher(backbone, **kw) -> TiledMatcher:
    m = TiledMatcher(num_landmarks=FIXTURE["num_pairs"], seed=SEED, backbone=backbone,
                     tie_order="legacy", **kw)
    m.set_episode(SCENE, START_ID)
    return m


# ------------------------------------------------------- geometry, no weights


class _StubBackbone:
    """Enough of a backbone to build the tiling: name and geometry only."""

    def __init__(self, name="dinov3_vits16", patch_size=16, input_size=224):
        self.name, self.patch_size, self.input_size = name, patch_size, input_size
        self.grid_side = input_size // patch_size
        self.dim = 8

    def encode(self, frame):  # pragma: no cover - only the geometry is used here
        raise AssertionError("this stub has no weights")


def test_the_tiling_geometry_matches_the_old_config():
    """...and repairs the two derived fields the old config left stale.

    `vit_tiled.py:99` overwrote `TilingConfig.patch_size` *after* construction, so the
    old object kept `patches_per_vit_input = 224 // 14 = 16` and
    `effective_patches = 96` - DINOv2's numbers on a patch-16 backbone. Nothing read
    them (`transform_patch_to_global` uses `patch_size`, `slice_resolution` and
    `vit_input_size` only), which is why the fixture's points still reproduce
    exactly; but they were wrong, and a required constructor argument is what makes
    them right. The fixture records the stale pair, and this test states the
    difference rather than hiding it.
    """
    cfg = TilingConfig(patch_size=TILING["backbone_patch_size"],
                       slice_number=TILING["slice_number"],
                       input_resolution=TILING["input_resolution"],
                       vit_input_size=TILING["vit_input_size"])
    for field in ("slice_number", "grid_size", "input_resolution", "slice_resolution",
                  "vit_input_size", "patch_size"):
        assert getattr(cfg, field) == TILING[field], field
    assert (TILING["patches_per_vit_input"], TILING["effective_patches"]) == (16, 96)
    assert (cfg.patches_per_vit_input, cfg.effective_patches) == (14, 84)  # 224 // 16


def test_patch_size_is_required_and_comes_from_the_backbone():
    """Trap 1: `TilingConfig.patch_size = 14` was hardcoded, and a patch-16 backbone
    then put every ART-VS point in the wrong place. It has no default now."""
    with pytest.raises(TypeError, match="patch_size"):
        TilingConfig(slice_number=36, input_resolution=1344)  # noqa: PLE1125
    matcher = TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone())
    assert matcher.config.patch_size == 16 != 14
    matcher14 = TiledMatcher(num_landmarks=24, seed=0,
                             backbone=_StubBackbone("dinov2_vits14", 14, 224))
    assert matcher14.config.patch_size == 14
    assert matcher14.config.patches_per_vit_input == 16   # 224 / 14, not 224 / 16


def test_the_tile_grid_is_refused_when_it_does_not_divide():
    with pytest.raises(ValueError, match="perfect square"):
        TilingConfig(patch_size=16, slice_number=35)
    with pytest.raises(ValueError, match="divisible"):
        TilingConfig(patch_size=16, slice_number=36, input_resolution=1000)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        TilingConfig(patch_size=16, slice_number=36, input_resolution=1344,
                     vit_input_size=225)


def test_tile_coordinates_and_slicing_match_the_old_handler():
    cfg = TilingConfig(patch_size=16, slice_number=TILING["slice_number"],
                       input_resolution=TILING["input_resolution"],
                       vit_input_size=TILING["vit_input_size"])
    tiles, coords = TilingHandler(cfg).create_tiles(np.zeros((1080, 1920, 3), np.uint8))
    assert [list(c) for c in coords] == FIXTURE["tile_coords"]
    assert len(tiles) == TILING["slice_number"]
    assert {t.shape for t in tiles} == {(cfg.slice_resolution, cfg.slice_resolution, 3)}


def test_transform_patch_to_global_flips_y_and_x_and_lands_on_cell_centres():
    """Trap 2, plus the published-configuration identity the tiled `Matches` relies on.

    With `slice_resolution == vit_input_size` the helper's scale is 1, so a tile-local
    cell maps onto the cell centre of the `effective_patches` lattice - which is what
    lets the tiled points round-trip through `Matches.from_pixel_pairs` unchanged.
    """
    cfg = TilingConfig(patch_size=16, slice_number=36, input_resolution=1344,
                       vit_input_size=224)
    handler = TilingHandler(cfg)
    _, coords = handler.create_tiles(np.zeros((1080, 1920, 3), np.uint8))
    tile_idx, cell = 8, np.array([[3, 5]])          # tile row 1 col 2; cell (y=3, x=5)
    got = handler.transform_patch_to_global(cell, tile_idx, coords)
    assert tuple(got[0]) == (1 * 224 + 3 * 16 + 8, 2 * 224 + 5 * 16 + 8)
    global_cell = np.array([[1 * 14 + 3, 2 * 14 + 5]])
    assert np.allclose(got, global_cell * (cfg.input_resolution / cfg.effective_patches)
                       + cfg.patch_size / 2)


def test_the_latch_fires_at_the_published_fraction_and_never_unlatches():
    """tau = 0.20 against the FIRST frame's error, and once tiled always tiled."""
    matcher = TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone())
    assert matcher.switch_threshold == DEFAULT_SWITCH_THRESHOLD == 0.20

    def frame(err: float) -> Matches:
        # two landmarks displaced by `err` pixels along x; the proxy is their median
        goal = np.array([[100.0, 100.0], [200.0, 200.0]])
        return Matches.from_pixel_pairs(
            goal, goal + np.array([err, 0.0]), side=32, input_size=512, patch_size=16,
            image_wh=(512, 512), sim=np.ones(2), cyclic=np.ones(2), margin=np.ones(2),
            pool_size=2, num_landmarks=24, draw_seed=None, tie_order="stable")

    trace = [100.0, 60.0, 25.0, 20.1, 19.9, 40.0, 80.0]
    fired = []
    for e in trace:
        matcher._iteration += 1
        matcher._update_switch(frame(e))
        fired.append(matcher.tiled)
    # 100 defines e0; 20.1/100 = 0.201 is not below tau, 19.9/100 = 0.199 is; and the
    # error growing back to 80 does not unlatch it.
    assert fired == [False, False, False, False, True, True, True]
    assert matcher._e0 == pytest.approx(100.0, abs=1e-9)


def test_the_switch_iteration_override_latches_on_its_own_iteration():
    matcher = TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone(),
                           switch_iteration=3)
    empty = Matches.from_pixel_pairs(
        np.zeros((0, 2)), np.zeros((0, 2)), side=32, input_size=512, patch_size=16,
        image_wh=(512, 512), sim=np.zeros(0), cyclic=np.zeros(0), margin=np.zeros(0),
        pool_size=0, num_landmarks=24, draw_seed=None, tie_order="stable")
    for i in (1, 2, 3):
        matcher._iteration = i
        matcher._update_switch(empty)
        assert matcher.tiled is (i >= 3)


def test_the_error_proxy_is_the_median_displacement_and_nan_on_an_empty_frame():
    goal = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    shifted = goal + np.array([[1.0, 0.0], [5.0, 0.0], [100.0, 0.0]])
    m = Matches.from_pixel_pairs(
        goal, shifted, side=512, input_size=512, patch_size=1, image_wh=(512, 512),
        sim=np.ones(3), cyclic=np.ones(3), margin=np.ones(3), pool_size=3,
        num_landmarks=3, draw_seed=None, tie_order="stable")
    assert TiledMatcher.error_proxy(m) == pytest.approx(5.0, abs=1e-9)
    empty = Matches(goal_patch=np.zeros((0, 2), np.int64), cur_patch=np.zeros((0, 2), np.int64),
                    sim=np.zeros(0), cyclic=np.zeros(0), margin=np.zeros(0), side=32,
                    pool_size=0, num_landmarks=24, draw_seed=None, tie_order="stable")
    assert np.isnan(TiledMatcher.error_proxy(empty))


def test_the_prealign_hook_is_a_rotation_block_not_a_matcher_knob():
    """U8 landed, and the 4-way still is not a constructor argument.

    It is a PRE-SERVO physical re-pose of the camera, so it is a `rotation:` config
    block that :func:`vsbench.vs.baselines.prealign.build_rotation` attaches and the runner applies
    before the loop. The record's `prealign` column reads the attached object's label,
    so a matcher with no pre-aligner still states `None` on every row.
    """
    with pytest.raises(NotImplementedError, match="rotation:"):
        TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone(), prealign="4way")
    matcher = TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone())
    assert matcher.record_fields()["prealign"] is None

    from vsbench.vs.baselines.prealign import Prealign4Way

    matcher.prealign = Prealign4Way(_StubBackbone(), seed=0)
    assert matcher.record_fields()["prealign"] == "4way-library"


def test_the_tiled_arm_refuses_to_guess_a_device_and_records_the_one_it_ran_on():
    """R22 correction 3, on ART-VS's own chain: `matcher.device` has no default here
    either, and the record states the device of the encoder that actually ran."""
    with pytest.raises(ValueError, match="matcher.device"):
        TiledMatcher(num_landmarks=24, seed=0)
    matcher = TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone())
    assert matcher.record_fields()["matcher_device"] is None      # the stub owns none


def test_the_record_states_the_published_tiling_knobs():
    matcher = TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone())
    fields = matcher.record_fields()
    assert fields["artvs_tiles"] == 36
    assert fields["artvs_tile_resolution"] == 1344
    assert fields["artvs_vit_input_size"] == 224
    assert fields["artvs_switch_threshold"] == 0.20
    assert fields["artvs_use_tiling"] is True
    assert fields["artvs_switched"] is False


# --------------------------------------------------- parity, needs the weights


def test_coarse_frame_matches_the_old_artvs(frames, backbone):
    """Before the latch, ART-VS is ViT-VS at the tiling input size - point for point."""
    want = FIXTURE["coarse"]
    matcher = _matcher(backbone)
    matcher.reset(frames[0])
    assert [list(c) for c in matcher.tile_coords] == FIXTURE["tile_coords"]
    assert matcher.coarse.draw.episode_seed == FIXTURE["subsample_seed"] == episode_seed(
        SCENE, START_ID, SEED)

    matches = matcher.match(frames[1])
    rows = matches.to_rows()
    assert matches.extras["mode"] == want["mode"] == "coarse"
    assert len(matches) == want["n"]
    np.testing.assert_allclose(rows["pts_goal"], want["pts_goal"], atol=POINT_ATOL, rtol=0)
    np.testing.assert_allclose(rows["pts_cur"], want["pts_cur"], atol=POINT_ATOL, rtol=0)
    np.testing.assert_allclose(rows["conf"], want["conf"], atol=CONF_ATOL, rtol=0)


def test_tiled_frame_matches_the_old_artvs(frames, backbone):
    """The tiled pass: 36 tiles, one pair each, lifted to full-image pixels."""
    want = FIXTURE["tiled"]
    matcher = _matcher(backbone, switch_iteration=1)
    matcher.reset(frames[0])
    matches = matcher.match(frames[1])
    rows = matches.to_rows()

    assert matches.extras["mode"] == want["mode"] == "tiled"
    assert matches.extras["n_tiles"] == want["n_tiles"] == 36
    assert matches.extras["per_tile_pairs"] == want["per_tile_pairs"] == 1
    assert len(matches) == want["n"]
    np.testing.assert_allclose(rows["pts_goal"], want["pts_goal"], atol=POINT_ATOL, rtol=0)
    np.testing.assert_allclose(rows["pts_cur"], want["pts_cur"], atol=POINT_ATOL, rtol=0)
    np.testing.assert_allclose(rows["conf"], want["conf"], atol=CONF_ATOL, rtol=0)
    assert matcher.record_fields()["artvs_switched"] is True


def test_use_tiling_false_is_plain_vit_vs(frames, backbone):
    """The paper's own `*_no_tiling` ablation: the coarse phase, and it says so."""
    matcher = _matcher(backbone, use_tiling=False)
    matcher.reset(frames[0])
    assert matcher.tile_coords == [] and matcher._goal_tiles == []
    matches = matcher.match(frames[1])
    assert matches.extras["tiling"] == "disabled"
    np.testing.assert_allclose(matches.to_rows()["pts_goal"], FIXTURE["coarse"]["pts_goal"],
                               atol=POINT_ATOL, rtol=0)


def test_the_tiled_points_are_cell_centres_of_the_effective_lattice(frames, backbone):
    """Under the published configuration the tiled offsets are zero, not merely small.

    That is the property `_matches` relies on when it wraps the tiled pixels on the
    `effective_patches` lattice - and it is how a lifted tile point stays a *cell*,
    so the runner's linear landmark id keeps meaning something.
    """
    matcher = _matcher(backbone, switch_iteration=1)
    matcher.reset(frames[0])
    matches = matcher.match(frames[1])
    assert matches.side == matcher.config.effective_patches == 6 * 14
    assert np.abs(matches.offsets_patch).max() < 1e-9
    assert np.abs(matches.goal_offsets_patch).max() < 1e-9
    assert matches.goal_patch.max() < matches.side


def test_the_tiled_pass_advances_one_draw_per_tile(frames, backbone):
    """The old `_call_find_correspondences` incremented per tile; frame seeds must too."""
    matcher = _matcher(backbone, switch_iteration=1)
    matcher.reset(frames[0])
    assert matcher.coarse.draw.draw_index == 0
    matcher.match(frames[1])
    # one coarse draw + one per tile
    assert matcher.coarse.draw.draw_index == 1 + TILING["slice_number"]


def test_the_backbone_runs_at_the_tiling_input_size(backbone):
    """`vit_input_size` reaches the encoder: 224, not the backbone's own 512."""
    assert backbone.input_size == 224 and backbone.grid_side == 14
    matcher = TiledMatcher(num_landmarks=24, seed=0, backbone=backbone)
    assert matcher.config.vit_input_size == 224
    assert matcher.config.effective_patches == 6 * 14
    assert torch.is_tensor(backbone.encode(np.zeros((224, 224, 3), np.uint8)))


def test_a_backbone_that_disagrees_with_vit_input_size_is_refused():
    """An injected encoder would otherwise win silently over the config, and every tile
    would be read at a resolution the config did not ask for."""
    with pytest.raises(ValueError, match="ART-VS reads every tile"):
        TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone(input_size=224),
                     vit_input_size=336)
    ok = TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone(input_size=224),
                      vit_input_size=224)
    assert ok.config.vit_input_size == 224


@pytest.mark.parametrize(("k", "expected"), [(24, 36), (48, 36), (96, 108)])
def test_a_tiled_frame_carries_n_tiles_times_the_rounded_budget(k, expected):
    """`n_tiles * max(1, round(K / n_tiles))`, verbatim old behaviour - NOT K.

    The record's `num_landmarks` still says K, so a table that wants what a frame
    actually steered on must read `n_matches`, exactly as on a CNS row.
    """
    matcher = TiledMatcher(num_landmarks=k, seed=0, backbone=_StubBackbone())
    per_tile = max(1, round(k / matcher.config.slice_number))
    assert matcher.config.slice_number * per_tile == expected


def test_build_arm_refuses_to_weight_the_tiled_arm():
    """Its cosine is tile-local (one pair per tile = every row is its own tile's best),
    so a trust weight built from it would be arbitrary rather than wrong-looking."""
    from vsbench.vs.baselines.control import Intrinsics
    from vsbench.vs.bench import arms
    from vsbench.vs.bench import config as cfgmod

    base = {"seed": 0,
            "matcher": {"variant": "artvs", "num_landmarks": 24, "device": "cpu"},
            "stop": {"rule": "cap_only"}, "scoring": {"success_rule": ["predicate_a"]}}
    intrinsics = Intrinsics(fx=1406.75, fy=1406.75, cx=960.0, cy=540.0)
    matcher = TiledMatcher(num_landmarks=24, seed=0, backbone=_StubBackbone())
    cfg = cfgmod.resolve(dict(base))
    cfg["controller"] = {"weighting": {"use_trust": True, "use_margin": False,
                                       "per_axis": False}}
    with pytest.raises(ValueError, match="no weightable confidence"):
        arms.build_arm(cfg, intrinsics=intrinsics, matcher=matcher)
    cfg = cfgmod.resolve(dict(base))
    cfg["controller"] = {"weighting": {"use_trust": False, "use_margin": False,
                                       "per_axis": False}}
    arm = arms.build_arm(cfg, intrinsics=intrinsics, matcher=matcher)
    assert arm.record_fields()["prealign"] is None
