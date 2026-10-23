# vs-sim copy: tests/fast/test_features_dinov2.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""U3 - the published ViT-VS (DINOv2) descriptor chain.

`vit_dense.py:183-188` built `dinov2_extractor.ViTExtractor("dinov2_vits14",
stride=14)`: torch.hub weights, ToTensor+Normalize, a forward hook on block 11
facet `token`, CLS stripped, `bin=True` log binning to **3456**-d un-normalized
descriptors. U1 recorded that the entry standing in `BACKBONES` was the HF
`facebook/dinov2-small` raw-token branch the old deploy matcher never called, and
left replacing it as a U3 TODO; `vsbench.vs.baselines.features.PublishedDinoV2Backbone` is
that replacement.

Weights are never downloaded. U1 recorded "the DINOv2 weights are not cached",
which is true of the HF copy `facebook/dinov2-small` and **false of the published
torch.hub one**: `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` has
been on this laptop since 2026-05-07, so this arm gets a real parity fixture
(`u3_vitvs_dinov2.json`) rather than a skipped one. Everything here still runs
where the cache is absent:

- the log binning is checked against the vendored `_log_bin` itself (read-only,
  where `../genvs-sim` is checked out) and otherwise skips with the reason;
- the shape and register contracts run on a stub module;
- the parity tests skip with the reason where the hub cache is missing, exactly
  as `tests/fast/test_features.py` does for DINOv3.

**No `vitvs_dinov2` number may be quoted until the arm is re-run on the station.**
No runbook has invoked `vitvs` or `artvs` since 2026-07-27 (U1's carried caveat,
`../INVENTORY.md` on `vit_tiled.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vsbench.vs.baselines.features import (
    BACKBONES,
    FrozenBackbone,
    PublishedDinoV2Backbone,
    build_backbone,
    log_bin,
)
from vsbench.vs.baselines.match import DenseMatcher, mutual_nn_match
from vsbench.vs.bench.config import MATCHER_VARIANTS

OLD_REPO = Path(__file__).resolve().parents[2].parent / "genvs-sim"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = json.loads((FIXTURES / "u3_vitvs_dinov2.json").read_text())
SPEC = BACKBONES["dinov2_vits14"]


@pytest.fixture(scope="module")
def published() -> PublishedDinoV2Backbone:
    """The real published chain, or a skip naming why its weights are unavailable."""
    try:
        return build_backbone("dinov2_vits14", device="cpu")
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        pytest.skip(
            f"the torch.hub DINOv2 weights are not in the local cache "
            f"({type(exc).__name__}: {str(exc).splitlines()[0]}); never download them "
            "from a test - run this where the cache already is."
        )


@pytest.fixture(scope="module")
def frames():
    from PIL import Image

    goal = np.asarray(Image.open(FIXTURES / FIXTURE["goal_image"]).convert("RGB"))
    cur = np.asarray(Image.open(FIXTURES / FIXTURE["current_image"]).convert("RGB"))
    return goal, cur


# ------------------------------------------------------------- the registry


def test_the_registry_entry_is_the_published_path_not_the_hf_one():
    """The U1 TODO, closed: the entry either becomes the published path or is deleted."""
    assert SPEC.loader == "torch_hub"
    assert (SPEC.hub_repo, SPEC.hub_entry) == ("facebookresearch/dinov2", "dinov2_vits14")
    assert (SPEC.input_size, SPEC.patch_size, SPEC.n_registers) == (518, 14, 0)
    assert SPEC.input_size // SPEC.patch_size == 37
    assert (SPEC.layer, SPEC.facet, SPEC.binned) == (11, "token", True)
    assert SPEC.dim == 9 * 384 == 3456, "log-binned, not the 384-d hidden size"
    assert "facebook/dinov2-small" not in SPEC.hf_id


def test_frozen_backbone_refuses_the_hub_entry_and_says_where_to_go():
    """Loading a published chain under the HF class is how U1's unverified entry
    became quotable in the first place; it now fails before any weights are touched."""
    with pytest.raises(ValueError, match="build_backbone"):
        FrozenBackbone("dinov2_vits14", device="cpu")


def test_build_backbone_dispatches_on_the_loader(published):
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone("dinov2_vitxl99")
    assert isinstance(published, PublishedDinoV2Backbone)
    assert (published.grid_side, published.dim) == (37, 3456)


# ------------------------------------------------------------ the log binning


def _vendored_log_bin(x: torch.Tensor, grid_hw) -> torch.Tensor:
    """`ViTExtractor._log_bin`, called read-only on the frozen old tree.

    Called unbound with a stand-in `self` (it reads only `num_patches` and
    `device`), so no model is constructed and no weights are touched.
    """
    import types

    if not OLD_REPO.is_dir():
        pytest.skip(f"the frozen old repo is not checked out at {OLD_REPO}")
    vendor = OLD_REPO / "src" / "control" / "_vitvs_vendor"
    for p in (str(OLD_REPO), str(vendor), str(vendor / "visual_servoing")):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "rospy" not in sys.modules:
        import src.control._rospy_stub as stub

        sys.modules["rospy"] = stub
    try:
        from visual_servoing.features.dinov2_extractor import ViTExtractor
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        pytest.skip(f"the vendored extractor will not import ({type(exc).__name__}: {exc})")
    fake = types.SimpleNamespace(num_patches=tuple(grid_hw), device="cpu")
    return ViTExtractor._log_bin(fake, x)[0, 0]


@pytest.mark.parametrize(("h", "w", "d"), [(4, 4, 6), (5, 3, 4), (37, 37, 8)])
def test_log_bin_matches_the_vendored_function_on_random_input(h, w, d):
    """Bit-identity against `_log_bin` itself, not against a reading of it.

    The lesson U4 wrote down (review R12): a rule that scores every arm gets ported
    from source. Measured max |diff| 0.0 on all three shapes.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, h * w, d)
    want = _vendored_log_bin(x, (h, w))
    got = log_bin(x.reshape(h * w, d), (h, w))
    assert got.shape == want.shape == (h * w, 9 * d)
    assert float((got - want).abs().max()) == 0.0


def test_log_bin_is_the_clamped_3x3_neighbourhood():
    """What the 9 bins are, without the vendored file: (dy, dx) row-major, edges clamped."""
    grid = torch.arange(3 * 3, dtype=torch.float32).reshape(9, 1)
    out = log_bin(grid, (3, 3)).reshape(3, 3, 9)
    # centre cell (1, 1) sees the whole grid in reading order
    assert out[1, 1].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # corner cell (0, 0) repeats the clamped neighbours
    assert out[0, 0].tolist() == [0, 0, 1, 0, 0, 1, 3, 3, 4]


def test_log_bin_refuses_a_hierarchy_it_does_not_implement():
    """1 (ViT-VS) and 2 (DINOBot, tests/fast/test_dinobot.py) are built; nothing else."""
    for bad in (0, 3):
        with pytest.raises(ValueError, match="hierarchy 1"):
            log_bin(torch.zeros(4, 2), (2, 2), hierarchy=bad)
    with pytest.raises(ValueError, match="do not fill"):
        log_bin(torch.zeros(5, 2), (2, 2))


# ---------------------------------------------------- the chain, on a stub


class _StubDinoV2(torch.nn.Module):
    """A weight-free module with DINOv2's shape: `blocks`, `[CLS] + side**2` tokens.

    `forward` is a deterministic function of the pixels so the descriptors differ
    between two frames, and every block is applied, so the block-11 hook fires on
    the same object the published extractor hooks.
    """

    def __init__(self, side: int = 37, dim: int = 384, n_blocks: int = 12) -> None:
        super().__init__()
        self.side, self.dim = side, dim
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(n_blocks))
        self.calls: list[int] = []

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b = pixel_values.shape[0]
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixel_values, self.side)
        tokens = pooled.reshape(b, 3, self.side * self.side).permute(0, 2, 1)
        x = torch.cat([tokens.new_zeros(b, 1, 3), tokens], dim=1)
        x = torch.nn.functional.pad(x, (0, self.dim - 3))
        for i, block in enumerate(self.blocks):
            x = block(x)
            self.calls.append(i)
        return x


@pytest.fixture
def stub_backbone() -> PublishedDinoV2Backbone:
    return PublishedDinoV2Backbone("dinov2_vits14", model=_StubDinoV2(), device="cpu")


def test_the_stub_chain_has_the_published_shape(stub_backbone):
    assert (stub_backbone.patch_size, stub_backbone.grid_side) == (14, 37)
    assert (stub_backbone.input_size, stub_backbone.dim) == (518, 3456)
    assert stub_backbone.n_prefix_tokens == 1, "CLS only - DINOv2 has no register tokens"
    tokens = stub_backbone.encode(np.zeros((1080, 1920, 3), np.uint8))
    assert tuple(tokens.shape) == (37, 37, 3456)
    assert tokens.dtype == torch.float32


def test_the_descriptors_are_un_normalized_and_the_pool_normalizes_them(stub_backbone):
    """The published descriptor has no `F.normalize`; `mutual_nn_match` does its own.

    This is the contract that lets one matcher serve a 384-d normalized DINOv3 grid
    and a 3456-d un-normalized DINOv2 one.
    """
    rng = np.random.default_rng(0)
    goal = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    cur = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    g, c = stub_backbone.encode(goal), stub_backbone.encode(cur)
    norms = g.reshape(-1, 3456).norm(dim=-1)
    assert float(norms.min()) > 0 and abs(float(norms.mean()) - 1.0) > 1e-3

    pool = mutual_nn_match(g, c, threshold=1.0, tie_order="stable")
    assert pool.side == 37 and pool.sim_rows.shape[1] == 37 * 37
    assert 0 < pool.size <= 37 * 37
    assert float(pool.sim.max()) <= 1.0 + 1e-5, "cosine similarity, so the pool normalized"


def test_the_register_prefix_is_checked_not_assumed():
    """DINOv3 strips 5 tokens, DINOv2 strips 1. Off-by-four shifts the whole grid."""
    bad = PublishedDinoV2Backbone("dinov2_vits14", model=_StubDinoV2(side=32), device="cpu")
    with pytest.raises(RuntimeError, match="not 37x37"):
        bad.encode(np.zeros((64, 64, 3), np.uint8))


def test_the_hook_reads_block_11(stub_backbone):
    """The published facet is block 11's own output, before the model's final norm."""
    stub_backbone.encode(np.zeros((64, 64, 3), np.uint8))
    assert stub_backbone.model.calls == list(range(12))
    assert stub_backbone.spec.layer == 11


# ------------------------------------------------- the arm, from a config


def _dinov2_config() -> dict:
    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / "arms" / "vitvs_dinov2.yaml"
    return yaml.safe_load(path.read_text())


def test_the_published_vitvs_arms_need_no_variant_of_their_own(stub_backbone):
    """`vitvs_dinov2` and `vitvs_dinov3` are configs, not code: one matcher, two backbones.

    And both now say `prealign: null` on every row (protocol v2 section 4: the 4-way
    pre-alignment's presence is stated in every table, and the published numbers
    effectively start roll-aligned). ART-VS said so from the start; the ViT-VS rows did
    not until review R16 - a row that simply omitted the field reads as "not
    applicable", not as "off".
    """
    import yaml

    from vsbench.vs.baselines.control import Intrinsics
    from vsbench.vs.bench import arms
    from vsbench.vs.bench import config as cfgmod

    assert "vitvs_dinov2" not in MATCHER_VARIANTS
    assert "vitvs_dinov3" not in MATCHER_VARIANTS
    intrinsics = Intrinsics(fx=1406.75, fy=1406.75, cx=960.0, cy=540.0)
    for name in ("vitvs_dinov2", "vitvs_dinov3"):
        arms_dir = Path(__file__).resolve().parents[2] / "configs" / "arms"
        cfg = cfgmod.resolve(yaml.safe_load((arms_dir / f"{name}.yaml").read_text()))
        assert cfg["matcher"]["variant"] == "mutual_nn_seeded_subsample"
        matcher = DenseMatcher(num_landmarks=cfg["matcher"]["num_landmarks"],
                               seed=cfg["seed"], backbone=stub_backbone,
                               tie_order=cfg["matcher"]["tie_order"])
        arm = arms.build_arm(cfg, intrinsics=intrinsics, matcher=matcher)
        fields = arm.record_fields()
        assert fields["prealign"] is None
        assert fields["backbone"] == stub_backbone.name


def test_the_dinov2_config_takes_its_geometry_from_the_backbone(stub_backbone):
    """Patch 14 and the 37x37 grid are declared by the backbone, never by the config.

    The config names a backbone; every geometric number downstream - the patch mask,
    the patch-centre offset, the pixel mapping - is read off the object that was
    loaded. The same matcher on a patch-16 stub produces patch-16 `Matches`.
    """
    cfg = _dinov2_config()
    assert cfg["matcher"]["backbone_name"] == "dinov2_vits14"
    assert "patch_size" not in cfg["matcher"] and "input_size" not in cfg["matcher"]
    assert BACKBONES[cfg["matcher"]["backbone_name"]].patch_size == 14

    matcher = DenseMatcher(num_landmarks=8, seed=0, backbone=stub_backbone,
                           tie_order="stable")
    matcher.set_episode("mug1", 7)
    frame = np.random.default_rng(1).integers(0, 255, (270, 480, 3), dtype=np.uint8)
    matcher.reset(frame)
    matches = matcher.match(frame + 1)
    assert (matches.patch_size, matches.input_size, matches.side) == (14, 518, 37)

    from vsbench.vs.baselines.features import BackboneSpec

    other = PublishedDinoV2Backbone.__new__(PublishedDinoV2Backbone)
    other.__init__("dinov2_vits14", model=_StubDinoV2(side=37), device="cpu")
    assert isinstance(other.spec, BackboneSpec) and other.patch_size == 14


def test_the_goal_patch_mask_uses_the_backbone_patch_size(stub_backbone):
    """518 / 14 = 37 exactly; the mask helper refuses a mismatched pair rather than
    silently making a 32x32 mask for a 37x37 grid."""
    from vsbench.vs.baselines.match import patch_mask_from_pixels

    mask = patch_mask_from_pixels(np.ones((100, 200), bool), input_size=518, patch_size=14)
    assert mask.shape == (37 * 37,) and mask.all()
    with pytest.raises(ValueError, match="is not"):
        patch_mask_from_pixels(np.ones((100, 200), bool), input_size=518, patch_size=16)


# ------------------------------------------------ parity, with the real weights


def test_the_published_descriptors_match_the_old_extractor(published, frames):
    """The 3456-d log-binned goal descriptor, against the frozen old `ViTExtractor`.

    Two invariants of the whole grid, so the check does not depend on which cells
    the draw happened to pick: the exact shape, and the descriptor's sum and absolute
    maximum. Measured max |diff| against the old chain, in one process at one thread
    count: **0.0**.

    The **sum is deliberately loose** (rel 1e-5). It is a float32 reduction over 4.7 M
    values whose summation order follows the torch thread count, so it is a property of
    the machine as much as of the descriptors: measured against the fixture on this
    laptop, rel 0 at 1 thread, 8.9e-7 at 4, 1.6e-6 at 10 (the default here). A tighter
    pin would fail on a box with a different core count while the descriptors were
    bit-identical. `test_the_vitvs_dinov2_arm_matches_the_old_one` below is the exact
    check - it compares landmarks, which no reduction order can move.
    """
    desc = published.encode(frames[0])
    assert list(desc.reshape(1, 1, -1, published.dim).shape) == FIXTURE["descriptor_shape"]
    assert float(desc.double().sum()) == pytest.approx(
        FIXTURE["goal_descriptor_sum"], rel=1e-5)
    assert float(desc.abs().max()) == pytest.approx(
        FIXTURE["goal_descriptor_absmax"], rel=1e-6)


def test_the_vitvs_dinov2_arm_matches_the_old_one(published, frames):
    """The whole published arm: encode -> mutual pool -> seeded draw -> pixels.

    A landmark-for-landmark check, which is only meaningful because the descriptors
    are bit-identical: a 1.9e-4 descriptor difference (a non-contiguous model input,
    `PublishedDinoV2Backbone.preprocess`) flipped enough argmaxes to move points by
    778 px, and it is exactly that kind of silent drift a baseline must not have.
    """
    from vsbench.vs.baselines.match import episode_seed

    matcher = DenseMatcher(num_landmarks=FIXTURE["num_pairs"], seed=FIXTURE["seed"],
                           backbone=published, tie_order="legacy")
    matcher.set_episode(FIXTURE["scene"], FIXTURE["start_id"])
    matcher.reset(frames[0])
    assert matcher.draw.episode_seed == FIXTURE["subsample_seed"] == episode_seed(
        FIXTURE["scene"], FIXTURE["start_id"], FIXTURE["seed"])

    matches = matcher.match(frames[1])
    rows = matches.to_rows()
    assert len(matches) == FIXTURE["n"]
    assert (matches.side, matches.input_size, matches.patch_size) == (
        FIXTURE["num_patches_side"], FIXTURE["input_size"], FIXTURE["patch_size"])
    # The landmarks themselves are exact - the descriptors are bit-identical, so every
    # argmax is. `conf` is not: `mutual_nn_match` takes the cosine with a matmul where
    # the vendored `chunk_cosine_sim` used `torch.einsum`, and the two reduce in a
    # different order in float32. Measured 1.1e-6, and it changes no landmark.
    np.testing.assert_allclose(rows["pts_goal"], FIXTURE["pts_goal"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(rows["pts_cur"], FIXTURE["pts_cur"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(rows["conf"], FIXTURE["conf"], atol=1e-5, rtol=0)


def test_the_model_input_must_be_contiguous(published, frames):
    """The fence, measured: the same values in a different layout are not the same run.

    `transforms.ToTensor` returns a contiguous tensor and a `permute` does not; the
    patch-embedding convolution then takes a different kernel path. This test fails
    if the `.contiguous()` in `preprocess` is ever removed as redundant.
    """
    x = published.preprocess([frames[0]])
    assert x.is_contiguous()
    # the layout `preprocess` would produce without the `.contiguous()`: NHWC in
    # memory, viewed as NCHW. Identical values, different strides.
    permuted = x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    assert not permuted.is_contiguous()
    assert float((x - permuted).abs().max()) == 0.0

    def block11(t):
        got = []
        handle = published.model.blocks[11].register_forward_hook(
            lambda _m, _i, out: got.append(out))
        try:
            with torch.no_grad():
                published.model(t)
        finally:
            handle.remove()
        return log_bin(got[0][:, 1:, :].float(), (37, 37))

    delta = float((block11(x) - block11(permuted)).abs().max())
    assert delta > 1e-5, f"layout no longer changes the descriptors ({delta}) - re-measure"
