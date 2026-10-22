# vs-sim copy: tests/fast/test_features.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""U1 golden tests: the Pillow-exact resize, and tokens equal to the old extractor's.

The resize half is pure CPU and always runs. The backbone half needs the DINOv3
weights in the local HF cache; where they are missing (or `transformers` is not
installed, as in the laptop `bot` env) it skips with that reason rather than
failing. The GPU/dtype tolerance and the DINOv2 arm live in
``tests/slow/test_features_parity.py``.

Fixture: ``tests/fixtures/u1_dinov3_tokens_mug1_goal.npz``, produced once with
``genvs-sim`` @ tag ``legacy-pre-cleanup`` - see ``u1_fixture_provenance.md``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vsbench.vs.baselines.features import (
    BACKBONES,
    FrozenBackbone,
    normalize_imagenet,
    resize_pil_exact,
    resize_pil_exact_torch,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TOKENS_NPZ = FIXTURES / "u1_dinov3_tokens_mug1_goal.npz"


def _frame() -> np.ndarray:
    return np.array(Image.open(FIXTURES / "u1_frame_mug1_goal.jpg").convert("RGB"))


def _pattern() -> np.ndarray:
    return np.array(Image.open(FIXTURES / "u1_pattern_synthetic.png").convert("RGB"))


def _random(seed: int = 20260902, shape=(733, 421, 3)) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 256, shape, dtype=np.uint8)


@pytest.fixture(scope="module")
def backbone() -> FrozenBackbone:
    """The DINOv3 backbone on CPU, or a skip naming why it is unavailable."""
    pytest.importorskip("transformers", reason="the fast tier runs without the backbone")
    try:
        return FrozenBackbone("dinov3_vits16", device="cpu")
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        pytest.skip(
            "DINOv3 ViT-S/16 weights are not in the local HF cache "
            f"({type(exc).__name__}: {str(exc).splitlines()[0]}). The weights are "
            "gated: cache them under $HF_HOME, never download them here."
        )


# ------------------------------------------------------------------- resize


@pytest.mark.parametrize("name", ["frame", "pattern", "random"])
@pytest.mark.parametrize("size", [512, 518])
def test_resize_pil_exact(name, size):
    """Bit-for-bit equal to `Image.fromarray(x).resize((size, size))`."""
    img = {"frame": _frame, "pattern": _pattern, "random": _random}[name]()
    want = np.asarray(Image.fromarray(img).resize((size, size)), dtype=np.uint8)
    got = resize_pil_exact(img, size)
    assert got.dtype == np.uint8 and got.shape == (size, size, 3)
    np.testing.assert_array_equal(got, want)


def test_resize_pil_exact_is_a_copy_at_native_size():
    img = _random(shape=(512, 512, 3))
    out = resize_pil_exact(img, 512)
    np.testing.assert_array_equal(out, np.asarray(Image.fromarray(img).resize((512, 512))))
    out[0, 0, 0] ^= 0xFF
    assert img[0, 0, 0] != out[0, 0, 0], "must not alias the caller's frame"


@pytest.mark.parametrize(
    "bad",
    [np.zeros((8, 8, 3), np.float32), np.zeros((8, 8), np.uint8), np.zeros((8, 8, 4), np.uint8)],
)
def test_resize_pil_exact_refuses_anything_but_uint8_rgb(bad):
    with pytest.raises((TypeError, ValueError)):
        resize_pil_exact(bad, 512)


@pytest.mark.parametrize("name", ["frame", "pattern", "random"])
def test_resize_pil_exact_torch_matches_within_fixed_point_rounding(name):
    """The GPU twin: float32 weights vs Pillow's int32 fixed point.

    The three bounds are the ones measured in the old repo
    (``genvs-sim/src/control/tests/test_gpu_preprocess.py``) and restated in
    ``resize_pil_exact_torch``'s docstring: <= 2 LSB anywhere, < 0.02 LSB mean,
    fewer than 1e-4 of pixels off by more than 1 LSB.
    """
    img = {"frame": _frame, "pattern": _pattern, "random": _random}[name]()
    want = resize_pil_exact(img, 512).astype(np.float32)
    got = resize_pil_exact_torch(torch.from_numpy(img), 512)
    assert tuple(got.shape) == (1, 3, 512, 512) and got.dtype == torch.float32
    got = got[0].permute(1, 2, 0).numpy()
    assert np.array_equal(got, np.rint(got)), "the uint8 pipeline must stay integer-valued"
    diff = np.abs(want - got)
    assert diff.max() <= 2.0 + 1e-6
    assert diff.mean() < 0.02
    assert float((diff > 1.0).mean()) < 1e-4


def test_normalize_imagenet_matches_the_manual_form():
    t = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4) % 256
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ref = (t / 255.0 - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    assert torch.allclose(normalize_imagenet(t, mean, std), ref)


# ------------------------------------------------------------------ backbone


def test_the_fixture_still_describes_the_committed_image():
    """A silently swapped fixture image would make the parity test meaningless."""
    z = np.load(TOKENS_NPZ, allow_pickle=False)
    committed = (FIXTURES / str(z["image_file"])).read_bytes()
    assert hashlib.md5(committed).hexdigest() == str(z["image_md5"])
    assert z["tokens"].shape == (32, 32, 384) and z["tokens"].dtype == np.float32


def test_tokens_match_legacy_dinov3(backbone):
    """CPU -> max abs <= 1e-5; measured 8.8e-7 in both envs (4.57.1 and 5.8.0).

    The model side is bit-exact: the residual is the old deploy chain's fused
    rescale+normalize against this module's plain `(x/255 - mean)/std`, which is
    what the old training encoder used - so `encode` equals the tokens the head
    was trained on exactly. Other devices and dtypes get the looser bar in the
    module docstring of ``vsbench.vs.baselines.features`` (max abs <= 1e-2 and per-token
    cosine >= 0.999); that check is ``tests/slow/test_features_parity.py``.
    """
    z = np.load(TOKENS_NPZ, allow_pickle=False)
    want = torch.from_numpy(z["tokens"])
    got = backbone.encode(_frame())
    assert tuple(got.shape) == (32, 32, 384) and got.dtype == torch.float32
    assert float((got - want).abs().max()) <= 1e-5


def test_encode_geometry_and_normalisation(backbone):
    assert (backbone.patch_size, backbone.grid_side, backbone.dim) == (16, 32, 384)
    assert backbone.n_prefix_tokens == 5  # CLS + 4 registers
    tok = backbone.encode(_pattern())
    torch.testing.assert_close(tok.norm(dim=-1), torch.ones(32, 32), atol=1e-5, rtol=0)
    batch = backbone.encode_batch([_pattern(), _frame()])
    assert tuple(batch.shape) == (2, 32, 32, 384)
    torch.testing.assert_close(batch[0], tok)


def test_preprocess_torch_tokens_stay_within_the_device_bar(backbone):
    """The GPU twin is not the fielded path: 5.7e-4 from the fixture, not 1e-5.

    It clears the cross-device bar (1e-2 and cosine 0.999) and nothing tighter,
    so a runner that must match the N3.7b records calls `encode`, never this.
    """
    want = torch.from_numpy(np.load(TOKENS_NPZ, allow_pickle=False)["tokens"]).reshape(-1, 384)
    pixels = backbone.preprocess_torch(torch.from_numpy(_frame()))
    with torch.no_grad():
        got = backbone._to_grid(backbone.model(pixel_values=pixels).last_hidden_state)
    got = got[0].reshape(-1, 384)
    diff = float((got - want).abs().max())
    assert 1e-5 < diff <= 1e-2, f"expected the documented ~5.7e-4, got {diff:.2e}"
    assert float(torch.nn.functional.cosine_similarity(got, want, dim=-1).min()) >= 0.999


def test_input_size_must_tile_the_patch_grid(backbone):
    with pytest.raises(ValueError, match="multiple of patch_size"):
        FrozenBackbone("dinov3_vits16", input_size=500, device="cpu")


# ----------------------------------------------------------------- registry


def test_unknown_backbone_is_refused_before_any_weights_are_touched():
    with pytest.raises(ValueError, match="unknown backbone"):
        FrozenBackbone("dinov4_vitxl99", device="cpu")


def test_registry_geometry_is_self_consistent():
    for name, spec in BACKBONES.items():
        assert spec.name == name
        assert spec.input_size % spec.patch_size == 0, f"{name}: input does not tile"


#: The matching study's backbone column (candidate 11, build task T6), pinned as
#: settings and not as a load: none of these three checkpoints is on this laptop.
COLUMN = {
    "dinov3_vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 512, 16, 768, 4, "hf"),
    "tipsv2_b14": ("google/tipsv2-b14", 448, 14, 768, 1, "hf_vision"),
    "tipsv1_s14": ("google/tipsv1-s14", 448, 14, 384, 1, "hf_vision"),
}


@pytest.mark.parametrize("name", sorted(COLUMN))
def test_the_backbone_columns_entries_are_pinned(name):
    """What each new entry claims, so a wrong checkpoint fails loudly on the station.

    Every field here is cross-checked against the loaded model by `FrozenBackbone`
    (fence 2); this test pins the *claim*, which is what a run 500 km away is judged
    against. No weights are touched.
    """
    hf_id, input_size, patch, dim, registers, loader = COLUMN[name]
    spec = BACKBONES[name]
    assert (spec.hf_id, spec.input_size, spec.patch_size, spec.dim) == (
        hf_id, input_size, patch, dim)
    assert (spec.n_registers, spec.loader) == (registers, loader)
    # the study's grid: 32 x 32 on every entry of the column, whatever the patch size
    assert spec.input_size // spec.patch_size == 32
    # the download line is in the note, because that is what a station operator reads
    assert "hf download" in spec.note and hf_id in spec.note


def test_the_vision_tower_entries_say_why_they_execute_repository_code():
    """`trust_remote_code` is per backbone and never a global (the station is 4.57.1)."""
    assert BACKBONES["dinov3_vits16"].trust_remote_code is False
    assert BACKBONES["dinov3_vitb16"].trust_remote_code is False
    for name in ("tipsv2_b14", "tipsv1_s14"):
        spec = BACKBONES[name]
        assert spec.trust_remote_code is True
        assert spec.loader == "hf_vision"
        # 1 CLS + 1 register slot, which TIPS uses as a second class token
        assert 1 + spec.n_registers == 2


def test_a_torch_hub_spec_is_still_refused_by_the_transformers_class():
    with pytest.raises(ValueError, match="is loaded by 'torch_hub'"):
        FrozenBackbone("dinov2_vits14", device="cpu")



# ------------------------------------- the CLS-to-patch attention (matching study T3)
#
# These need the real backbone and skip with the `backbone` fixture's reason where it
# is absent (the laptop `bot` env has no `transformers`). They are fast-tier and not
# `slow`, because they are one CPU forward pass of ViT-S/16 on a small frame and the
# fielded-path guarantee below is exactly the thing a fast tier should not let drift.


# ------------------------------------------- the vision-tower loader, without weights
class _FakeVisionConfig:
    patch_size, hidden_size, num_register_tokens, num_hidden_layers = 14, 768, 1, 12


class _FakeTopConfig:
    def __init__(self, with_vision: bool = True) -> None:
        if with_vision:
            self.vision_config = _FakeVisionConfig()


class _FakeProcessor:
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)


class _CardModel(torch.nn.Module):
    """What the TIPS model card documents: `encode_image(x).patch_tokens`."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeTopConfig()
        self.seen_revision = None

    def encode_image(self, pixel_values):
        n = pixel_values.shape[0]
        from types import SimpleNamespace

        return SimpleNamespace(patch_tokens=torch.ones(n, 32 * 32, 768))


class _TowerModel(torch.nn.Module):
    """The transformers-native route: a vision tower with `last_hidden_state`."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeVisionConfig()

    def forward(self, pixel_values=None, **_):
        from types import SimpleNamespace

        n = pixel_values.shape[0]
        return SimpleNamespace(last_hidden_state=torch.ones(n, 2 + 32 * 32, 768))


def _fake_transformers(monkeypatch, *, card=None, tower=None, with_vision=True,
                       seen=None):
    import transformers

    def config_from_pretrained(model_id, **kw):
        if seen is not None:
            seen.setdefault("config", kw)
        return _FakeTopConfig(with_vision)

    def model_from_pretrained(model_id, **kw):
        if seen is not None:
            seen.setdefault("model" if "config" not in kw else "model_with_config", kw)
        if "config" in kw:
            if tower is None:
                raise OSError("no vision class for this config")
            return tower()
        if card is None:
            raise OSError("the repository ships no such class")
        return card()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        staticmethod(config_from_pretrained))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        staticmethod(model_from_pretrained))
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained",
                        staticmethod(lambda model_id, **kw: _FakeProcessor()))


def test_the_model_cards_route_is_taken_first_and_strips_no_prefix(monkeypatch):
    """TIPS ships its own code; `encode_image(x).patch_tokens` is what it documents."""
    seen = {}
    _fake_transformers(monkeypatch, card=_CardModel, tower=_TowerModel, seen=seen)
    backbone = FrozenBackbone("tipsv2_b14", device="cpu")
    assert backbone.load_route == "card"
    assert backbone.n_prefix_tokens == 0, "patch_tokens are the patch tokens already"
    assert backbone.grid_side == 32 and backbone.patch_size == 14
    tokens = backbone.encode(np.zeros((1080, 1920, 3), np.uint8))
    assert tuple(tokens.shape) == (32, 32, 768)
    # the snapshot is pinned, and the pin reaches every call that loads code
    assert seen["config"]["revision"] == BACKBONES["tipsv2_b14"].revision
    assert seen["model"]["revision"] == BACKBONES["tipsv2_b14"].revision
    assert seen["model"]["trust_remote_code"] is True
    assert seen["model"]["local_files_only"] is True


def test_the_transformers_native_route_is_the_fallback_and_strips_the_prefix(monkeypatch):
    _fake_transformers(monkeypatch, card=None, tower=_TowerModel)
    backbone = FrozenBackbone("tipsv2_b14", device="cpu")
    assert backbone.load_route == "hf"
    assert backbone.n_prefix_tokens == 2, "one CLS plus TIPS's second class token"
    assert tuple(backbone.encode(np.zeros((1080, 1920, 3), np.uint8)).shape) == \
        (32, 32, 768)


def test_neither_route_names_both_and_refuses_the_image_text_pair(monkeypatch):
    _fake_transformers(monkeypatch, card=None, tower=None)
    with pytest.raises(RuntimeError, match="neither route to the vision tower"):
        FrozenBackbone("tipsv2_b14", device="cpu")


def test_a_checkpoint_with_no_vision_config_says_so(monkeypatch):
    _fake_transformers(monkeypatch, card=None, tower=_TowerModel, with_vision=False)
    with pytest.raises(RuntimeError, match="no `vision_config` block"):
        FrozenBackbone("tipsv2_b14", device="cpu")


# ------------------------------------- the ART-VS release's chain (2026-10-05)
def test_the_artvs_release_entry_is_pinned_and_the_others_are_untouched():
    """`dinov3_vits16_artvs256`: the release's simulation descriptor - DINOv3-S/16 at 256,
    the model's own processor, 9 log bins on the raw tokens (3456-d), then L2. Every entry
    that existed before keeps the Pillow route and no binning hierarchy of its own."""
    from vsbench.vs.baselines.features import PREPROCESS_ROUTES

    spec = BACKBONES["dinov3_vits16_artvs256"]
    assert (spec.hf_id, spec.input_size, spec.patch_size, spec.dim, spec.n_registers) == (
        "facebook/dinov3-vits16-pretrain-lvd1689m", 256, 16, 3456, 4)
    assert (spec.loader, spec.binned, spec.bin_hierarchy, spec.preprocess) == (
        "hf", True, 1, "hf_processor")
    assert spec.input_size // spec.patch_size == 16 and spec.dim == 384 * 9
    assert PREPROCESS_ROUTES == ("pillow", "hf_processor")
    for name in ("dinov3_vits16", "dinov3_vitb16", "tipsv2_b14", "tipsv1_s14",
                 "dinov2_vits14"):
        assert (BACKBONES[name].preprocess, BACKBONES[name].bin_hierarchy) == ("pillow", None)


def test_the_binned_entry_bins_the_raw_tokens_then_normalises():
    """Loaded (CPU, skipped without the cached weights): `encode` == strip CLS + 4
    registers off `last_hidden_state`, `log_bin(hierarchy=1)`, then L2 - and the processor
    route is the model's own processor at 256. Shared weights with `dinov3_vits16`."""
    pytest.importorskip("transformers")
    from vsbench.vs.baselines.features import log_bin

    try:
        bb = FrozenBackbone("dinov3_vits16_artvs256", device="cpu")
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        pytest.skip(f"DINOv3 ViT-S/16 weights are not in the local HF cache ({type(exc).__name__})")
    assert (bb.grid_side, bb.dim, bb.hidden_size, bb.bin_hierarchy) == (16, 3456, 384, 1)
    img = np.asarray(Image.fromarray(_frame()[200:700, 600:1300]).resize((256, 256)))
    px = bb.preprocess([img])
    assert torch.equal(px, bb.processor(images=Image.fromarray(img),
                                        return_tensors="pt")["pixel_values"])
    with torch.no_grad():
        raw = bb.model(pixel_values=px).last_hidden_state[:, 5:, :].float()
    want = torch.nn.functional.normalize(log_bin(raw, (16, 16), 1), p=2, dim=-1)
    got = bb.encode(img).reshape(1, 256, 3456)
    assert float((got - want).abs().max()) <= 1e-6       # the layout fence's last bits
    torch.testing.assert_close(got.norm(dim=-1), torch.ones(1, 256), atol=1e-5, rtol=0)
