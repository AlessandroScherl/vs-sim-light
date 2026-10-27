# vs-sim copy: vssim/baselines/features.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Frozen backbone: image -> patch descriptors (migration unit U1).

Purpose
-------
The one place that turns an RGB frame into ViT patch tokens, plus the
Pillow-exact 512x512 preprocessing the published ViT-VS pipeline depends on
bit-for-bit. **One encoder serves training and deployment** - the old repo had
two (a deploy adapter under ``src/control`` and a training encoder under
``src/refine``) and they had to be kept in step by hand.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
------------------------------------------------------------------------------
- ``src/control/gpu_preprocess.py``
  ``_pil_bicubic_weights`` / ``pil_like_resize_uint8`` / ``normalize_imagenet``
  -> :func:`_pil_bicubic_weights`, :func:`resize_pil_exact_torch`,
  :func:`normalize_imagenet` (verbatim arithmetic).
- ``src/control/vit_vs_dinov3.py`` lines 31-93 - ``_DinoV3ExtractorAdapter``:
  DINOv3 ViT-S/16, 512 input, 32x32 grid, 384-d, CLS + 4 register tokens
  stripped, L2-normalized -> :class:`FrozenBackbone` (``dinov3_vits16``).
- ``src/control/_vitvs_vendor/visual_servoing/features/multi_backbone_extractor.py``
  ``_init_transformers_model`` / ``_setup_model_config`` / ``preprocess_pil`` /
  ``extract_descriptors`` / ``_extract_features`` - only the ``dinov3-*`` and
  ``dinov2-*`` branches -> :data:`BACKBONES` + :class:`FrozenBackbone`.
- ``src/control/_vitvs_vendor/.../base_extractor.py`` - the four-method
  interface (``preprocess_pil`` / ``extract_descriptors`` /
  ``get_descriptor_dim`` / ``get_patch_size``), folded into the class as
  ``preprocess`` / ``encode`` / ``dim`` / ``patch_size``.
- ``src/refine/encoder.py`` - ``Dinov3Encoder``: the training side of the same
  chain (``side = input_size // patch_size``, the same stripping and the same
  L2 normalization, ``float32`` out under ``no_grad``) ->
  :meth:`FrozenBackbone.encode_batch`. Verified bit-exact (max abs 0.0, both
  environments, R10 review 2026-09-02): the new ``encode`` *is* the tokens the
  refinement head was trained on.
- The exact call shape ``Image.fromarray(rgb).resize((512, 512))`` ->
  ``preprocess_pil`` -> ``extract_descriptors(..., bin=False)`` is
  ``src/control/matchers/vit_dense.py`` lines 391-395 and 456-460; that is what
  the parity fixture reproduces.

The ART-VS release's own chain (2026-10-05, ``dinov3_vits16_artvs256``)
-----------------------------------------------------------------------
Ported from the ART-VS release (``../ART-VS``, git ``9b1208c``, MIT) for the faithful
object-mode arm (``vsbench.vs.baselines.objroi.RoiTiledMatcherSim``):
``features/multi_backbone_extractor.py`` - the DINOv3 branch of
``_init_transformers_model`` (``:268-296``: HF model, the model's own processor with
``size``/``crop_size`` set to 256), ``preprocess_pil`` (``:604``), ``extract_descriptors``
(``:646-652`` the 5-token prefix, ``:713-727`` bin then L2) and ``_log_bin`` (``:852-912``,
hierarchy 1 - the ROI calls pass none). Two :class:`BackboneSpec` fields carry it
(``bin_hierarchy``, ``preprocess="hf_processor"``) and :meth:`FrozenBackbone._to_grid`
bins before it normalises, on the release's own memory layout (fence 4 there). Left
behind: every other backbone branch of that file, its torch.hub fallback (which would
download), and the print statements.

Not here
--------
The research repository's dense-description assembler (shift-and-stitch) and the
per-layer / attention readers are that project's instruments, not part of the
published baselines, and stayed behind.

Deliberately dropped
--------------------
- The ROS stub (``sys.modules["rospy"] = _rospy_stub``) and the twin
  ``sys.path`` injection of ``_vitvs_vendor/`` + ``_vitvs_vendor/visual_servoing/``:
  no ROS here, and the vendored tree is not imported at all
  (``../INVENTORY.md`` hacks 1 and 2 retire together).
- Every unused backbone of ``MultiBackboneExtractor`` (RADIO, SigLIP, CLIP,
  MAE, I-JEPA, ConvNeXt, Swin, OWL-ViT), the YOLO-World and LangSAM detectors,
  ``dinov2_legacy_wrapper.py``, and ``_log_bin`` (see the DINOv2 note below).
- ``ViTVSDinoV3Method`` / ``ViTVSMethod`` - the GEN-VS half of
  ``vit_vs_dinov3.py``: ROI crops, mask-Otsu, feature injection, the IBVS body.
  Control is U4 (``refinevs/control.py``); the ROI path is not used by the
  full-image suite.
- ``TokenGrid`` and the ``build_encoder(name, **cfg)`` factory from
  ``src/refine/encoder.py``. The training call sites (``train.py`` lines 151-182,
  ``eval_static.py`` line 120, ``instrument.CachingEncoder``,
  ``precompute_weight_features.py`` line 170, ``spair_eval.py`` line 102) read
  ``grid.tokens`` / ``.input_size`` / ``.patch_size``; U5 rewrites them against
  :meth:`FrozenBackbone.encode_batch` plus the ``input_size`` / ``patch_size`` /
  ``grid_side`` attributes, so the wrapper dataclass earns nothing. The config
  key ``encoder: {name: dinov3, device: cuda}``
  (``configs/refine_train_n33_corr_m.yaml``) maps to
  ``FrozenBackbone("dinov3_vits16", device=...)``.
- ``StubEncoder`` from ``src/refine/encoder.py`` - a weight-free test encoder.
  It belongs to whatever needs a CPU stand-in (U5's training tests), not here.

Fences kept (each one cost real debugging in the old repo)
----------------------------------------------------------
1. **Register/CLS masking.** DINOv3 emits ``[CLS] + 4 registers + patches``;
   the old code hardcoded ``features[:, 1 + 4:, :]`` and
   ``num_register_tokens: 4`` for every DINOv3 size
   (``multi_backbone_extractor.py`` lines 501-503 / 640-645; ``../INVENTORY.md``
   load-bearing hack 4). Off-by-five here shifts the whole 32x32 grid and every
   landmark with it. Here the prefix length is read from the model config
   (``1 + config.num_register_tokens``) and then **checked**: the leftover token
   count must be exactly ``grid_side**2`` or :meth:`FrozenBackbone.encode`
   raises. DINOv2 has no registers - prefix 1.
2. **DINOv2 patch-14 vs DINOv3 patch-16.** The vendored
   ``_setup_model_config`` sets ``patch_size`` only on the DINOv3 branch, so a
   ``dinov2-*`` extractor reported the default ``16`` for a patch-**14**
   backbone, and ``TilingConfig.patch_size = 14`` (``tiling.py`` line 30) is the
   mirror-image hardcode on the ART-VS side - ``vit_tiled.py`` line 99 had to
   overwrite it from the live extractor or every ART-VS point landed on the
   wrong patch (``../INVENTORY.md`` load-bearing hack 5). Here ``patch_size`` comes
   from ``model.config.patch_size``, is cross-checked against the registry, and
   ``grid_side`` is derived from it - never assumed.
3. **The Pillow-exact resize.** The deployed chain resizes the 1920x1080 frame
   with ``Image.fromarray(rgb).resize((512, 512))`` - Pillow's antialiased
   BICUBIC (a = -0.5) on a uint8 pipeline. ``torch.nn.functional.interpolate``
   uses a = -0.75 and could not clear the Gate-5 per-patch cosine bar, so the
   GPU path re-implements Pillow's own separable resample instead
   (:func:`resize_pil_exact_torch`). Feature parity with the published method
   depends on this, so :func:`resize_pil_exact` calls Pillow itself and is
   bit-identical by construction.

Tolerances (what "the same tokens" means)
-----------------------------------------
- **CPU** (the fixture's own device): ``max |diff| <= 1e-5`` against
  ``tests/fixtures/u1_dinov3_tokens_mug1_goal.npz``. Measured 2026-09-02:
  **8.8e-7**, the same in both environments (``refinevs``/transformers 4.57.1
  and ``bot``/5.8.0), per-token cosine >= 0.9999998. The **model side is
  bit-exact** - fed the old chain's own ``pixel_values`` the new model
  reproduces the fixture at 0.0 - and the 8.8e-7 is entirely the HF fast
  processor's *fused* rescale+normalize against the plain
  ``(x/255 - mean)/std`` this module uses (:func:`normalize_imagenet`). The old
  training encoder ``src/refine/encoder.py`` used the plain form too, which is
  why it was itself 8.8e-7 from the old deploy chain and why the new ``encode``
  matches **it** bit-for-bit. Enforced by ``tests/fast/test_features.py``.
  **The model input is made contiguous** (U3, review R16): a ``permute`` leaves
  NHWC strides and the patch-embedding convolution then takes a different kernel
  path - worth ~7.5e-7 in tokens here and **1.9e-4** on the log-binned DINOv2
  chain (:class:`PublishedDinoV2Backbone`, whose docstring has the measurement).
  The old DINOv3 chain's own input was contiguous NCHW, so this moves *towards*
  it: 8.78e-7 -> 7.6e-7 against the fixture here (7.93e-7 in the review's run -
  the last digit follows the torch thread count), with an identical 612-pair
  mutual pool on the U1/U2 pair either way. The fixture is unchanged.
- **Another device or dtype** (CUDA, fp16/bf16): ``max |diff| <= 1e-2`` **and**
  per-token cosine ``>= 0.999`` - the bar the old Gate-5 parity harness used.
  Enforced by ``tests/slow/test_features_parity.py`` where a GPU exists.
- **The GPU preprocessing twin** (:meth:`FrozenBackbone.preprocess_torch`):
  tokens land **5.7e-4** from the fixture (cosine >= 0.99999) - inside the
  cross-device bar, outside the CPU one. It is not the fielded path; see its
  docstring.
- **The torch resize vs Pillow**: not bit-exact - Pillow accumulates in int32
  fixed point, we accumulate in float32. Measured and asserted: ``max <= 2``
  LSB, ``mean < 0.02`` LSB, fewer than ``1e-4`` of pixels off by more than 1
  LSB (the same three numbers as the old
  ``src/control/tests/test_gpu_preprocess.py``).

Weights are never downloaded
----------------------------
``from_pretrained`` is called with ``local_files_only=True``. The DINOv3 repo is
gated: without it, a revision lookup **precedes** the cache read and 401s even
though the weights are cached, which is why every station runbook exports
``HF_HUB_OFFLINE=1`` (``../docs/results/2026-08-18_station_bringup/RESULTS.md``
"Launch requirements"). ``local_files_only=True`` is that flag per call, so the
code is correct with or without the environment variable.

Golden tests that guard it
--------------------------
``tests/fast/test_features.py`` (resize bit-exactness, token parity against the
fixture, ``encode_layers`` shapes) and ``tests/slow/test_features_parity.py``
(the GPU/dtype tolerance, and the DINOv2 arm once its weights exist).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "BACKBONES",
    "DEFAULT_BACKBONE",
    "BackboneSpec",
    "FrozenBackbone",
    "PublishedDinoV2Backbone",
    "build_backbone",
    "log_bin",
    "normalize_imagenet",
    "resize_pil_exact",
    "resize_pil_exact_torch",
]


# --------------------------------------------------------------------- resize

_PIL_WEIGHT_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _pil_bicubic_weights(in_size: int, out_size: int) -> np.ndarray:
    """Pillow's ``precompute_coeffs`` for BICUBIC (a = -0.5, support 2.0,
    antialias always - the filter support scales with the downscale ratio), as a
    dense ``(out_size, in_size)`` row-normalized weight matrix.

    Verbatim from ``genvs-sim/src/control/gpu_preprocess.py``. torch's own
    bicubic uses a = -0.75, which is why ``F.interpolate`` could not clear the
    per-patch cosine bar.
    """
    key = (int(in_size), int(out_size))
    cached = _PIL_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    def kernel(x):
        x = np.abs(x)
        a = -0.5
        return np.where(
            x < 1,
            ((a + 2) * x - (a + 3)) * x * x + 1,
            np.where(x < 2, (((x - 5) * x + 8) * x - 4) * a, 0.0),
        )

    in_size, out_size = key
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    for i in range(out_size):
        center = (i + 0.5) * scale
        xmin = max(0, int(center - support + 0.5))
        xmax = min(in_size, int(center + support + 0.5))
        j = np.arange(xmin, xmax)
        w = kernel((j - center + 0.5) / filterscale)
        s = w.sum()
        if s != 0:
            weights[i, xmin:xmax] = w / s
    weights = weights.astype(np.float32)
    if len(_PIL_WEIGHT_CACHE) > 256:  # crop sizes vary per step - bound the cache
        _PIL_WEIGHT_CACHE.clear()
    _PIL_WEIGHT_CACHE[key] = weights
    return weights


def resize_pil_exact(frame_rgb_u8: np.ndarray, size: int = 512) -> np.ndarray:
    """``Image.fromarray(frame).resize((size, size))``, returned as uint8 HxWx3.

    **Bit-identical to Pillow by construction**: it *is* Pillow (>= 9 RGB
    default: antialiased BICUBIC on a uint8 pipeline). The wrapper exists so
    that the one resize the whole system depends on has a name, a type check and
    a test, instead of being retyped at each call site - the old repo spelled it
    out in four places (``vit_dense.reset``/``match``, ``refine/encoder``,
    ``refine/crops``) and a fifth spelling with a different filter would have
    been invisible.

    An already-``size``x``size`` input is returned as a copy: Pillow's own
    ``resize`` short-circuits the identical-size case too, so this stays exact.
    """
    a = np.asarray(frame_rgb_u8)
    if a.dtype != np.uint8:
        raise TypeError(f"frame must be uint8 RGB, got dtype {a.dtype}")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3 RGB, got shape {a.shape}")
    size = int(size)
    if a.shape[0] == size and a.shape[1] == size:
        return a.copy()
    from PIL import Image

    return np.asarray(Image.fromarray(a).resize((size, size)), dtype=np.uint8)


def resize_pil_exact_torch(frame_hwc_u8: torch.Tensor, size: int = 512) -> torch.Tensor:
    """The GPU-path twin of :func:`resize_pil_exact`: Pillow's separable
    resample re-implemented in torch, so a device-resident frame never makes the
    full-HD round trip to host memory.

    Horizontal pass then vertical pass, each rounded and clipped to uint8 like
    Pillow's 8-bit intermediate buffers. Returns ``(1, 3, size, size)`` float32
    in ``[0, 255]`` (integer-valued).

    **Guarantee**: equal to :func:`resize_pil_exact` *up to fixed-point
    rounding* - Pillow accumulates its coefficients in int32 fixed point, this
    accumulates in float32, so a pixel whose exact value lands on a rounding
    coin-flip can differ. Measured on the old repo's test images and asserted by
    the tests: ``max |diff| <= 2`` LSB, ``mean |diff| < 0.02`` LSB, and fewer
    than ``1e-4`` of pixels off by more than 1 LSB. Not bit-exact; use
    :func:`resize_pil_exact` when the bytes must match.

    **It is also not the fielded resize.** Every paper-3 arm ran
    :func:`resize_pil_exact` through ``vit_dense.py``; this twin was reached only
    by the dropped GEN-VS ROI controller (``src/control/vit_vs.py`` lines
    238-248). Those 1-LSB coin flips are worth 5.7e-4 in the resulting tokens -
    see :meth:`FrozenBackbone.preprocess_torch`.
    """
    if frame_hwc_u8.ndim != 3 or frame_hwc_u8.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3 RGB, got shape {tuple(frame_hwc_u8.shape)}")
    h, w = int(frame_hwc_u8.shape[0]), int(frame_hwc_u8.shape[1])
    size = int(size)
    dev = frame_hwc_u8.device
    wx = torch.as_tensor(_pil_bicubic_weights(w, size), device=dev)  # (size, w)
    wy = torch.as_tensor(_pil_bicubic_weights(h, size), device=dev)  # (size, h)
    t = frame_hwc_u8.float()  # (h, w, 3)
    t = torch.einsum("hwc,ow->hoc", t, wx).round().clamp_(0.0, 255.0)  # horizontal
    t = torch.einsum("hoc,ph->poc", t, wy).round().clamp_(0.0, 255.0)  # vertical
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, size, size)


def normalize_imagenet(
    t_bchw_0_255: torch.Tensor,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> torch.Tensor:
    """The HF image-processor tail: rescale (1/255) then normalize(mean, std).

    Verbatim from ``genvs-sim/src/control/gpu_preprocess.py``. ``image_mean`` and
    ``image_std`` come from the model's own ``AutoImageProcessor``, never from a
    constant in this file.
    """
    mean = torch.as_tensor(image_mean, dtype=t_bchw_0_255.dtype, device=t_bchw_0_255.device)
    std = torch.as_tensor(image_std, dtype=t_bchw_0_255.dtype, device=t_bchw_0_255.device)
    return (t_bchw_0_255 / 255.0 - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


# ------------------------------------------------------------------ backbones


@dataclass(frozen=True)
class BackboneSpec:
    """What a supported backbone must look like once loaded.

    Every field is also read back from the loaded model and cross-checked, so a
    silently different checkpoint fails loudly instead of producing a shifted
    grid (fences 1 and 2 in the module docstring).
    """

    name: str
    hf_id: str
    input_size: int
    patch_size: int
    dim: int
    n_registers: int
    note: str
    #: ``"hf"`` -> :class:`FrozenBackbone` (``AutoModel.from_pretrained``);
    #: ``"hf_vision"`` -> the same class over the **vision tower only** of an
    #: image-text model (``AutoConfig(...).vision_config``), which is what the
    #: TIPS entries need: ``AutoModel`` on those repositories builds the paired
    #: text encoder too and its ``forward`` wants token ids;
    #: ``"torch_hub"`` -> :class:`PublishedDinoV2Backbone` (the published ViT-VS
    #: chain). :func:`build_backbone` dispatches on it, so a caller names a
    #: backbone and never a loader.
    loader: str = "hf"
    #: Whether the repository's own modelling code is executed on load. ``True``
    #: only where the installed ``transformers`` may be older than the model: the
    #: station's container and the `refinevs` env are 4.57.1 and the laptop's `bot`
    #: env is 5.8.0, and the TIPS model types are in neither's registry (checked
    #: 2026-09-07), so those two entries run the code shipped **inside the snapshot**
    #: (nothing is fetched at load time - ``local_files_only`` and
    #: ``HF_HUB_OFFLINE=1`` still hold). Stated per backbone, never a global.
    trust_remote_code: bool = False
    #: The snapshot this entry means: a Hugging Face commit hash. **In practice
    #: required for a ``trust_remote_code`` entry** - what runs there is the ``.py``
    #: files inside the snapshot, so an unpinned entry is code whose version no run
    #: records (review R35 finding 3). ``None`` = the repository's current ``main``,
    #: which is honest for a weights-only entry, and every row says which it got.
    revision: str | None = None
    hub_repo: str | None = None
    hub_entry: str | None = None
    #: transformer block the descriptors are read off (``torch_hub`` only).
    layer: int | None = None
    #: ``token`` facet - the block's output, not its key/query/value projections.
    facet: str | None = None
    #: log-binned descriptors (``dim`` is then ``(1 + 8 * hierarchy) x hidden_size``).
    binned: bool = False
    #: the log-binning hierarchy of a binned ``hf`` entry (:func:`log_bin`; ``None`` = 1,
    #: 9 bins). A ``torch_hub`` entry bins at its own published hierarchy (1).
    bin_hierarchy: int | None = None
    #: how a frame becomes ``pixel_values``. ``"pillow"`` (every entry before 2026-10-05):
    #: Pillow-exact resize + this module's plain ``(x/255 - mean)/std``
    #: (:meth:`FrozenBackbone.preprocess`). ``"hf_processor"``: the same Pillow resize to the
    #: input square, then the model's OWN image processor with ``size``/``crop_size`` set to
    #: the input size - what the ART-VS release does (``multi_backbone_extractor.py``
    #: ``:286-291, 604``). The two differ by 7.2e-7 in ``pixel_values`` (the processor fuses
    #: rescale and normalise); the release's chain is bit-identical only through its own.
    preprocess: str = "pillow"


#: :attr:`BackboneSpec.preprocess` values.
PREPROCESS_ROUTES = ("pillow", "hf_processor")


BACKBONES: dict[str, BackboneSpec] = {
    "dinov3_vits16": BackboneSpec(
        name="dinov3_vits16",
        hf_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        input_size=512,  # 512 / 16 = 32 exactly -> 32x32x384 (NOT 518)
        patch_size=16,
        dim=384,
        n_registers=4,
        note="the fielded backbone: configs/camera.yaml `encoder:` block, frozen "
        "2026-07-31; the head was trained against exactly these tokens",
    ),
    # ---------------------------------------------------------- the backbone column
    # Candidate 11 of the matching study (`../docs/matching_study_plan_2026-09-06.md`
    # section 3, build task T6): the SAME mutual matcher on four encoders. Every entry
    # below is registered at the input size that gives the study's 32 x 32 grid, so one
    # square is the same 60 x 33.75 NATIVE pixels of a 1920x1080 frame on all four
    # (512/16 = 32 and 448/14 = 32), the recall universe is one lattice, and the
    # study's "within 1.5 squares" is the same physical bar on every row. A 14-pixel
    # patch read at 518 (DINOv2's own) would be a 37 x 37 grid and a 51.9-pixel square,
    # and no column of the table would mean the same thing twice.
    #
    # None of these weights is on this laptop and NOTHING downloads them: the station's
    # cache is filled once, by hand, with the commands in each `note`
    # (`HF_HOME=/workspace/cache/hf`), and every load is `local_files_only=True`.
    "dinov3_vitb16": BackboneSpec(
        name="dinov3_vitb16",
        hf_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        input_size=512,  # 512 / 16 = 32 -> the study's grid, the fielded geometry
        patch_size=16,
        dim=768,
        n_registers=4,
        note="the SAME family as the fielded ViT-S/16, one size up (86 M against "
        "21 M): the backbone column's control for capacity, with the geometry, the "
        "preprocessing and the register count of the arm we ship. The trained head "
        "does NOT run on it (it was trained on 384-d ViT-S tokens), so its study rows "
        "are matcher-only. **A GATED repository**: the station holds no Hugging Face "
        "token (2026-09-07), so this snapshot is not on the volume and the row "
        "DECLINES with that reason - it never aborts a run (review R35 finding 2). "
        "With a token: HF_HOME=/workspace/cache/hf hf download "
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
    ),
    "tipsv2_b14": BackboneSpec(
        name="tipsv2_b14",
        hf_id="google/tipsv2-b14",
        input_size=448,  # 448 / 14 = 32 -> the same 32x32 grid as the fielded 512/16
        patch_size=14,
        dim=768,
        n_registers=1,  # TIPS repurposes DINOv2's register slot as a SECOND cls token
        loader="hf_vision",
        trust_remote_code=True,
        revision="ed1e4dc6b74bf3935ae099e9d5eb30fa96528454",
        note="TIPSv2 B/14 (Google DeepMind, CVPR 2026), the language-aligned backbone "
        "`../docs/paper3_questions_2026-09-06.md` section 7 says one afternoon of "
        "still pictures decides. `AutoModel` on this repository builds the image-text "
        "pair, so the spec loads the VISION tower alone "
        "(`AutoConfig(...).vision_config`) and the prefix it strips is 2 tokens: the "
        "two class tokens, one trained on alt-text and one on synthetic captions. "
        "The snapshot on the station (2026-09-07) carries its own modeling_tips.py "
        "and image_encoder.py, and THAT is the code a load runs, so the entry is "
        "pinned to it: HF_HOME=/workspace/cache/hf hf download google/tipsv2-b14 "
        "--revision ed1e4dc6b74bf3935ae099e9d5eb30fa96528454",
    ),
    "tipsv1_s14": BackboneSpec(
        name="tipsv1_s14",
        hf_id="google/tipsv1-s14",
        input_size=448,  # 448 / 14 = 32, as above
        patch_size=14,
        dim=384,
        n_registers=1,
        loader="hf_vision",
        trust_remote_code=True,
        revision="109d6693a2a0263be13309c193eb7427ce13a386",
        note="TIPS v1 S/14 (ICLR 2025), the distilled small model - the size-matched "
        "row beside the fielded DINOv3 ViT-S/16 (21.6 M against 21 M), so the column "
        "separates 'a bigger encoder' from 'a different encoder'. Same vision-tower "
        "load and same 2-token prefix as TIPSv2, pinned to the station's snapshot: "
        "HF_HOME=/workspace/cache/hf hf download google/tipsv1-s14 "
        "--revision 109d6693a2a0263be13309c193eb7427ce13a386",
    ),
    "dinov2_vits14": BackboneSpec(
        name="dinov2_vits14",
        hf_id="",  # not an HF repo: torch.hub, see `hub_repo` / `hub_entry`
        input_size=518,  # 518 / 14 = 37 -> 37x37 cells
        patch_size=14,
        dim=3456,  # 9 log bins x 384 hidden - NOT the hidden size
        n_registers=0,  # DINOv2 emits CLS only; the prefix is 1 token
        loader="torch_hub",
        hub_repo="facebookresearch/dinov2",
        hub_entry="dinov2_vits14",
        layer=11,
        facet="token",
        binned=True,
        note="the PUBLISHED ViT-VS (DINOv2) baseline arm, ported in U3: "
        "`vit_dense.py:183-188` -> `dinov2_extractor.ViTExtractor('dinov2_vits14', "
        "stride=14)`. torch.hub weights, ToTensor+Normalize(ImageNet), a forward "
        "hook on block 11 facet `token`, CLS stripped, `bin=True` log binning to "
        "3456-d **un-normalized** descriptors. The HF `facebook/dinov2-small` "
        "raw-token entry that stood here until U3 was never called by the old "
        "deploy matcher and is deleted, not kept beside this one. Weights are not "
        "in the laptop's torch.hub cache "
        "(~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth, 2026-05-07) and are "
        "never downloaded; the parity test is fast-tier and skips with the reason where "
        "the cache is absent.",
    ),
    # ------------------------------------------- the ART-VS release's own chain (2026-10-05)
    # The faithful ART-VS arm (`artvs_roi_sim`, docs/results/2026-10-05_artvs_release_check/
    # sim_spec.md section 4.1): the release's simulation config runs DINOv3-S/16 at 256
    # (S:5-6) through `MultiBackboneViTExtractor`, and every ROI call passes `bin=True` with
    # NO `hierarchy`, so the wrapper default 1 applies: 9 bins of the RAW
    # `last_hidden_state` tokens, THEN L2 (multi_backbone_extractor.py:646-652, 713-727,
    # 793-802, 866, 942-961).
    "dinov3_vits16_artvs256": BackboneSpec(
        name="dinov3_vits16_artvs256",
        hf_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        input_size=256,  # 256 / 16 = 16 -> 16x16 cells (the release's r for DINOv3-S)
        patch_size=16,
        dim=3456,  # 9 log bins x 384 hidden - NOT the hidden size
        n_registers=4,
        loader="hf",
        binned=True,
        bin_hierarchy=1,
        preprocess="hf_processor",
        note="the ART-VS release's SIMULATION descriptor (owner ruling 2026-10-05): HF "
        "DINOv3-S/16 at 256 (config_dinov3_256_6x6_tiling.yaml:5-6), the model's own fast "
        "image processor with size 256 (multi_backbone_extractor.py:286-291, 604), "
        "last_hidden_state with CLS + 4 registers stripped (:646-652, 793-802), 9-bin log "
        "binning on the RAW tokens (hierarchy 1 - the ROI calls pass none, :945, 866), "
        "then L2 (:725-727). The same weights as `dinov3_vits16`; nothing else is read.",
    ),
}

DEFAULT_BACKBONE = "dinov3_vits16"


class FrozenBackbone:
    """A frozen ViT that turns one RGB frame into an L2-normalized token grid.

    ``encode(frame) -> (grid_side, grid_side, dim)`` float32, CLS and register
    tokens stripped - the exact contract the trained refinement head, the
    matcher and the goal banks all consume, and the same one the training code
    used (``src/refine/encoder.py``).

    Parameters
    ----------
    name:
        ``"dinov3_vits16"``, the fielded backbone and the only ``hf``-loader entry.
        The published ViT-VS DINOv2 arm lives in :class:`PublishedDinoV2Backbone`
        (``loader="torch_hub"``), and :func:`build_backbone` picks between them -
        this class refuses a ``torch_hub`` spec rather than loading a different
        chain under the same name.
    input_size:
        Square encoder input. Defaults to the backbone's own (512 for DINOv3).
        Changing it changes the patch grid, so leave it alone unless you mean it -
        it must stay a multiple of ``patch_size``. The ART-VS arm does mean it
        (:mod:`vsbench.vs.baselines.tiling` runs the backbone at its tile resolution).
    device, dtype:
        Where and in what precision the weights live. ``device`` is **required and
        has no default** (R22 correction 3): it used to default to ``"cpu"``, and
        the first station run flew the whole learned chain on the CPU at twice its
        budget without a single line of config saying so. The config field is
        ``matcher.device`` (:data:`vsbench.vs.bench.config.BACKBONE_VARIANTS`). Tokens
        always come back float32 (the vendored extractor forced ``.float()``
        before ``F.normalize`` because ``F.normalize`` does not support bf16 on
        newer GPUs - kept).
    local_files_only:
        ``True`` by default and it should stay that way: the DINOv3 repo is
        gated and a revision lookup precedes the cache read (see the module
        docstring).
    """

    def __init__(
        self,
        name: str = DEFAULT_BACKBONE,
        *,
        device: str | torch.device,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = True,
        model_id: str | None = None,
    ) -> None:
        if name not in BACKBONES:
            raise ValueError(
                f"unknown backbone {name!r}; supported: {', '.join(sorted(BACKBONES))}"
            )
        spec = BACKBONES[name]
        if spec.loader not in ("hf", "hf_vision"):
            raise ValueError(
                f"backbone {name!r} is loaded by {spec.loader!r}, not by transformers - "
                f"call vsbench.vs.baselines.features.build_backbone({name!r}) instead, which picks "
                "the right class. Loading a published chain under this class would "
                "silently report a different descriptor as a baseline."
            )
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        self.name = name
        self.spec = spec
        self.model_id = model_id or spec.hf_id
        self.device = torch.device(device)
        self.dtype = dtype

        load_kwargs = {"local_files_only": local_files_only}
        if spec.trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        if spec.revision:
            # The snapshot, not the repository: for a `trust_remote_code` entry the
            # `.py` files inside it are what runs, so an unpinned load is code no run
            # can name (review R35 finding 3). Every row records what it got.
            load_kwargs["revision"] = spec.revision
        self.revision = spec.revision

        #: WHICH SURFACE THE TOKENS CAME OFF, recorded because there are two and they
        #: are not the same code: ``"hf"`` is ``model(pixel_values).last_hidden_state``
        #: with the CLS and register prefix stripped - every DINOv3 entry, and a TIPS
        #: checkpoint whose config exposes a vision tower to ``AutoModel``; ``"card"``
        #: is the TIPS model card's own ``encode_image(pixel_values).patch_tokens``,
        #: which is the surface the snapshot's `image_encoder.py` documents and which
        #: returns the patch tokens **already stripped**.
        self.load_route = "hf"
        if spec.loader == "hf_vision":
            self.model, self.load_route = self._load_vision_tower(
                AutoConfig, AutoModel, load_kwargs)
        else:
            self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        self.model = self.model.to(self.device, dtype=dtype).eval()
        self.model.requires_grad_(False)
        processor = AutoImageProcessor.from_pretrained(
            self.model_id, local_files_only=local_files_only,
            **({"trust_remote_code": True} if spec.trust_remote_code else {})
        )
        self.image_mean = tuple(float(v) for v in processor.image_mean)
        self.image_std = tuple(float(v) for v in processor.image_std)
        if spec.preprocess not in PREPROCESS_ROUTES:
            raise ValueError(f"{name}: preprocess {spec.preprocess!r} is not one of "
                             f"{PREPROCESS_ROUTES}")
        self.input_size = int(spec.input_size if input_size is None else input_size)
        #: the model's own processor, kept only on the ``hf_processor`` route, with its
        #: size overridden to the input square exactly as the ART-VS release overrides it
        #: (``multi_backbone_extractor.py:286-291``: ``size`` and ``crop_size``).
        self.processor = None
        if spec.preprocess == "hf_processor":
            processor.size = {"height": self.input_size, "width": self.input_size}
            processor.crop_size = {"height": self.input_size, "width": self.input_size}
            self.processor = processor

        # On the card route `model.config` is the image-text config and the geometry
        # sits in its `vision_config`; on every other route it is already the vision
        # config. Fence 2 below cross-checks whatever this resolves to.
        cfg = getattr(self.model.config, "vision_config", self.model.config)
        self.patch_size = int(getattr(cfg, "patch_size", spec.patch_size))
        #: the token width the model emits (384 for ViT-S); :attr:`dim` is the descriptor
        #: width, which is the same number unless the entry is log-binned.
        self.hidden_size = int(getattr(cfg, "hidden_size", spec.dim))
        self.dim = self.hidden_size
        #: the log-binning hierarchy of a binned entry (fence 4 in :meth:`_to_grid`), else None
        self.bin_hierarchy: int | None = None
        if spec.binned:
            self.bin_hierarchy = int(spec.bin_hierarchy or 1)
            if self.bin_hierarchy not in LOG_BIN_HIERARCHIES:
                raise ValueError(f"{name}: bin_hierarchy {self.bin_hierarchy} is not one of "
                                 f"{LOG_BIN_HIERARCHIES}")
            self.dim = self.hidden_size * (1 + 8 * self.bin_hierarchy)
        self.n_registers = int(getattr(cfg, "num_register_tokens", spec.n_registers))
        #: CLS + registers, dropped from the front of the token sequence - and **0 on
        #: the card route**, whose `patch_tokens` are the patch tokens already.
        self.n_prefix_tokens = 0 if self.load_route == "card" else 1 + self.n_registers

        # Fence 2: the geometry is read from the model, then cross-checked.
        if (self.patch_size, self.dim, self.n_registers) != (
            spec.patch_size,
            spec.dim,
            spec.n_registers,
        ):
            raise RuntimeError(
                f"{self.model_id} loaded as patch={self.patch_size}, dim={self.dim}, "
                f"registers={self.n_registers}, but {name} is registered as "
                f"patch={spec.patch_size}, dim={spec.dim}, registers={spec.n_registers} "
                "- a different checkpoint would silently shift the whole token grid"
            )
        if self.input_size % self.patch_size:
            raise ValueError(
                f"input_size {self.input_size} is not a multiple of patch_size "
                f"{self.patch_size}: the token grid would not be square"
            )
        self.grid_side = self.input_size // self.patch_size

        # For encode_layers: the block list, the final LayerNorm and the
        # embedding module, resolved once. None where a backbone does not expose
        # them in the expected shape - encode_layers then raises, and encode
        # (which needs none of this) still works.
        n_layers = int(getattr(cfg, "num_hidden_layers", 0))
        self._blocks = next(
            (
                m
                for _, m in self.model.named_modules()
                if isinstance(m, torch.nn.ModuleList) and n_layers and len(m) == n_layers
            ),
            None,
        )
        self._final_norm = next(
            (
                m
                for n, m in self.model.named_children()
                if isinstance(m, torch.nn.LayerNorm) and n in ("norm", "layernorm")
            ),
            None,
        )
        self._embeddings = getattr(self.model, "embeddings", None)

    def _load_vision_tower(self, AutoConfig, AutoModel, load_kwargs):  # noqa: N803
        """An image-text checkpoint -> ``(the vision tower, which route found it)``.

        ``AutoModel`` on a TIPS repository builds the paired text encoder too and its
        ``forward`` wants token ids, so this class names the tower it encodes with.
        There are two ways in and **both are the checkpoint's own**, so both are tried
        in the order of what is documented and recorded on :attr:`load_route`:

        1. ``"card"`` - the model card's route, which is the snapshot's shipped code:
           load the whole thing and call ``encode_image(pixel_values).patch_tokens``.
           Preferred because it is what the repository documents and therefore what its
           `image_encoder.py` guarantees. The tokens come back **without** a prefix.
        2. ``"hf"`` - ``AutoModel.from_pretrained(id, config=config.vision_config)``,
           the transformers-native route (its TIPSv2 page: "use Tipsv2VisionModel if
           you only need the vision features"), which needs the *vision* config to
           carry an ``auto_map`` to a vision class. Then the contract is this class's
           usual ``last_hidden_state`` with the stated prefix stripped.

        What is refused either way: encoding with the image-text pair under a
        backbone's name. If neither route is there the error names both, because the
        fix is a stated build step and never a silent fallback.
        """
        full = AutoConfig.from_pretrained(self.model_id, **load_kwargs)
        vision_config = getattr(full, "vision_config", None)
        errors = {}
        try:
            model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
            if not hasattr(model, "encode_image"):
                raise AttributeError(
                    "the loaded model has no `encode_image`, so it is not the model "
                    "card's surface")
            return model, "card"
        except Exception as exc:                   # noqa: BLE001 - both routes tried
            errors["card"] = f"{type(exc).__name__}: {exc}"
        if vision_config is not None:
            try:
                return (AutoModel.from_pretrained(self.model_id,
                                                  **{**load_kwargs,
                                                     "config": vision_config}), "hf")
            except Exception as exc:               # noqa: BLE001
                errors["hf"] = f"{type(exc).__name__}: {exc}"
        else:
            errors["hf"] = "the config has no `vision_config` block"
        import transformers

        raise RuntimeError(
            f"{self.model_id}: neither route to the vision tower worked with "
            f"transformers {transformers.__version__}. The model card's "
            f"`encode_image(pixel_values).patch_tokens`: {errors['card']}. The "
            f"transformers-native `AutoModel(config=config.vision_config)`: "
            f"{errors['hf']}. Loading the image-text pair under a backbone's name is "
            "not a third option - this class's contract is a token grid.")

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"FrozenBackbone({self.name!r}, input_size={self.input_size}, "
            f"grid={self.grid_side}x{self.grid_side}, dim={self.dim}, "
            f"device={self.device}, dtype={self.dtype})"
        )

    # ------------------------------------------------------------ preprocess
    def preprocess(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """uint8 HxWx3 RGB frames -> ``(B, 3, S, S)`` normalized model input.

        Pillow-exact resize (:func:`resize_pil_exact`) then the processor's own
        mean/std - the deploy chain of ``vit_dense.py`` lines 391-392 and the
        training chain of ``refine/encoder.py``, the same two steps; but
        arithmetically this **is** the training chain (bit-exact to
        ``refine/encoder.py``). The deploy chain's HF fast processor fuses the
        two constants into one affine map, which lands 7.15e-7 away in
        ``pixel_values`` and 8.8e-7 away in tokens - the whole of the fixture
        gap (module docstring, "Tolerances").

        **The ``hf_processor`` route** (``dinov3_vits16_artvs256``, the ART-VS release's
        chain): the same Pillow resize to the input square - an identity copy on a frame
        that is already that size, which is how every ART-VS crop arrives - then the
        model's own image processor on each frame (``processor(images=pil,
        return_tensors="pt")``, one call per frame, concatenated: the release's
        ``preprocess_pil`` per tile and ``torch.cat``, ``ros_controller.py:1745-1754``).
        """
        if self.processor is not None:
            from PIL import Image

            px = [self.processor(images=Image.fromarray(resize_pil_exact(f, self.input_size)),
                                 return_tensors="pt")["pixel_values"] for f in frames]
            return torch.cat(px, dim=0).to(self.device).to(self.dtype)
        arr = np.stack([resize_pil_exact(f, self.input_size) for f in frames])
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float().contiguous()
        return normalize_imagenet(x, self.image_mean, self.image_std).to(self.dtype)

    def preprocess_torch(self, frame_hwc_u8: torch.Tensor) -> torch.Tensor:
        """The GPU path: one device-resident uint8 frame -> ``(1, 3, S, S)``.

        **Not the fielded path.** Every paper-3 arm went through
        :meth:`preprocess` (``vit_dense.py``); this twin was reached only by the
        dropped GEN-VS ROI controller (``src/control/vit_vs.py`` lines 238-248).
        It reproduces that old GPU chain exactly (``pil_like_resize_uint8`` +
        ``preprocess_tensor``, max abs 0.0), but tokens taken through it sit
        **5.7e-4** from the parity fixture (cosine >= 0.99999) - outside the
        1e-5 CPU bar, because of :func:`resize_pil_exact_torch`'s 1-LSB
        rounding. **A runner that must match the N3.7b records calls**
        :meth:`preprocess` / :meth:`encode`; this one is for a live GPU loop
        that would rather not move a full-HD frame to host memory.
        """
        t = resize_pil_exact_torch(frame_hwc_u8.to(self.device), self.input_size)
        return normalize_imagenet(t, self.image_mean, self.image_std).to(self.dtype)

    # ---------------------------------------------------------------- encode
    def _to_grid(self, feats: torch.Tensor) -> torch.Tensor:
        """``(B, 1 + registers + N, D)`` -> ``(B, side, side, D)`` float32, L2-normalized.

        Fence 1: the prefix length is checked against the grid before anything
        is dropped, so a backbone with a different register count fails here
        instead of shifting every landmark by five patches.

        Fence 4 (a **binned** entry, 2026-10-05): the release's order is bin the RAW tokens,
        THEN normalise (``multi_backbone_extractor.py:713-727``) - binning normalised
        tokens would weight every neighbour equally and is a different descriptor. And the
        normalise runs on the release's own memory layout: its ``_log_bin`` fills a
        channels-first ``(B, C, H, W)`` map and hands back a ``permute`` view (``:911``),
        and a reduction picks its kernel by layout - on that view the normalised
        descriptors and the cosine matrix are bit-identical to the release's (measured
        0.0), on a contiguous copy they are 8.9e-8 and 1.1e-7 off. The unbinned path is
        untouched (``tests/fast/test_features.py`` pins the ``u1`` tokens).
        """
        feats = feats.float()
        n_tokens = int(feats.shape[1])
        expected = self.grid_side**2
        if n_tokens - self.n_prefix_tokens != expected:
            raise RuntimeError(
                f"{self.model_id} returned {n_tokens} tokens; stripping "
                f"{self.n_prefix_tokens} (CLS + {self.n_registers} registers) leaves "
                f"{n_tokens - self.n_prefix_tokens}, not {self.grid_side}x"
                f"{self.grid_side} = {expected} - the register/CLS masking is wrong"
            )
        feats = feats[:, self.n_prefix_tokens :, :]
        if self.bin_hierarchy is not None:
            feats = log_bin(feats, (self.grid_side, self.grid_side), self.bin_hierarchy)
            # the release's layout: a (B, C, N) map viewed as (B, N, C) - see fence 4
            feats = feats.transpose(1, 2).contiguous().transpose(1, 2)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.reshape(feats.shape[0], self.grid_side, self.grid_side, self.dim)

    @torch.no_grad()
    def encode_batch(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Frames -> ``(B, side, side, dim)`` float32 L2-normalized patch tokens."""
        return self._to_grid(self._tokens(self.preprocess(frames)))

    def _tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``(B, prefix + N, D)`` on the ``hf`` route, ``(B, N, D)`` on the card's.

        :attr:`n_prefix_tokens` is 0 on the card route, so :meth:`_to_grid` strips
        nothing there and its token-count fence still holds on both.
        """
        if self.load_route == "card":
            out = self.model.encode_image(pixel_values)
            tokens = getattr(out, "patch_tokens", None)
            if tokens is None:
                raise RuntimeError(
                    f"{self.model_id}: `encode_image` returned no `patch_tokens` "
                    f"({type(out).__name__}) - the model card's surface is what this "
                    "route was chosen for")
            return tokens
        return self.model(pixel_values=pixel_values).last_hidden_state

    @torch.no_grad()
    def encode(self, frame_rgb_u8: np.ndarray) -> torch.Tensor:
        """One frame -> ``(side, side, dim)`` float32 L2-normalized patch tokens.

        This is the fielded contract: ``last_hidden_state`` (i.e. after the
        model's final layer norm), CLS + registers stripped, L2-normalized -
        byte-for-byte the tokens the refinement head was trained on. Parity
        tolerances are in the module docstring.
        """
        return self.encode_batch([frame_rgb_u8])[0]


#: The log-binning hierarchies :func:`log_bin` implements. ``1`` is the published ViT-VS
#: (DINOv2) chain's; ``2`` is the default of ``dino-vit-features`` ``_log_bin`` and what
#: DINOBot runs (``bin=True``). Nothing publishes a third level, so none is built.
LOG_BIN_HIERARCHIES = (1, 2)


def log_bin(tokens: torch.Tensor, grid_hw: tuple[int, int], hierarchy: int = 1) -> torch.Tensor:
    """The vendored ``ViTExtractor._log_bin``, vectorised - ``(N, D)`` -> ``(N, (1+8h) D)``.

    Each cell's descriptor is the concatenation of ``1 + 8 * hierarchy`` bins, each a
    whole ``D``-block, in the vendored loop's ``part_idx`` order:

    - **level 0** - the cell's 3x3 neighbourhood at step 1, row-major ``(dy, dx)``, the
      cell itself included (9 bins). The level-0 average pool has ``win_size = 1``, the
      identity, so these bins are a gather of the raw tokens.
    - **level 1** (``hierarchy=2``) - the 3x3 ring at step 3 of a **3x3-averaged** map
      (``AvgPool2d(3, stride=1, padding=1, count_include_pad=False)`` - near the border
      the mean is over the cells that exist), row-major, the centre skipped (8 bins).

    Border cells repeat the clamped neighbour (the vendored ``temp_i = max(0, min(i,
    H - 1))``), never zero padding - at both levels. ``hierarchy=1`` is the published
    ViT-VS (DINOv2) chain: 9 bins, 384 x 9 = **3456**-d, and it is untouched by
    ``hierarchy=2`` (the same gather, bit for bit). ``hierarchy=2`` is DINOBot's
    (``dino-vit-features`` ``_log_bin``'s default): 17 bins, 384 x 17 = **6528**-d.

    **The pooled map is built exactly as the vendored code builds it**:
    ``tokens.permute(0, 2, 1).reshape(B, D, H, W)`` on the ``(B, N, D)`` tokens, which
    leaves channels-last strides, and the average pool runs on that view through the
    same ``AvgPool2d`` call. On the laptop CPU a contiguous copy pools to the same bits
    (measured 0.0, 2026-09-22), but a pool's kernel is chosen by the layout and the
    contiguity lesson of :meth:`PublishedDinoV2Backbone.preprocess` (1.9e-4 from a
    stride) says not to find out on the station - so the vendored view is kept.
    ``tests/fast/test_dinobot.py`` checks both hierarchies bit for bit against the
    vendored loop (``tests/fixtures/reference/dino_vit_features/extractor.py``; measured
    0.0 on 7x11, 55x98, 3x3, 1x5 and 4x1 grids), and
    ``tests/fast/test_features_dinov2.py`` checks ``hierarchy=1`` against the frozen
    ViT-VS extractor as before.

    ``tokens`` is ``(N, D)`` or ``(B, N, D)``, CLS already stripped, in the row-major
    cell order the model emits.
    """
    if hierarchy not in LOG_BIN_HIERARCHIES:
        raise ValueError(
            f"log_bin implements hierarchy 1 (9 bins, the ViT-VS chain) and 2 (17 bins, "
            f"the dino-vit-features default DINOBot runs); got {hierarchy}. A third level "
            "would need the 9x9 pool and no published arm uses one.")
    h, w = int(grid_hw[0]), int(grid_hw[1])
    t = torch.as_tensor(tokens)
    squeeze = t.ndim == 2
    if squeeze:
        t = t.unsqueeze(0)
    b, n, d = t.shape
    if n != h * w:
        raise ValueError(f"{n} tokens do not fill a {h}x{w} grid")
    grid = t.reshape(b, h, w, d)
    rows = torch.arange(h, device=t.device).reshape(h, 1) + torch.tensor(
        [-1, 0, 1], device=t.device)
    cols = torch.arange(w, device=t.device).reshape(w, 1) + torch.tensor(
        [-1, 0, 1], device=t.device)
    rows = rows.clamp(0, h - 1)     # border clamping, not zero padding
    cols = cols.clamp(0, w - 1)
    # [B, h, 3, w, 3, D] -> the 9 bins in (dy, dx) row-major order, as the
    # vendored double loop `for i in range(y-1, y+2): for j in range(x-1, x+2)` fills
    # `part_idx`, and each bin contributes a whole D-block to the descriptor.
    gathered = grid[:, rows][:, :, :, cols]
    out = gathered.permute(0, 1, 3, 2, 4, 5).reshape(b, h * w, 9 * d)
    if hierarchy == 1:
        return out.squeeze(0) if squeeze else out

    # level 1: the vendored `bin_x.permute(0, 2, 1).reshape(B, D, H, W)` view (NOT made
    # contiguous - see the docstring), pooled by the same module call it makes.
    chw = t.permute(0, 2, 1).reshape(b, d, h, w)
    pooled = torch.nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)(chw)
    ring_rows = (torch.arange(h, device=t.device).reshape(h, 1)
                 + torch.tensor([-3, 0, 3], device=t.device)).clamp(0, h - 1)
    ring_cols = (torch.arange(w, device=t.device).reshape(w, 1)
                 + torch.tensor([-3, 0, 3], device=t.device)).clamp(0, w - 1)
    ring = pooled.permute(0, 2, 3, 1)[:, ring_rows][:, :, :, ring_cols]   # [B, h, 3, w, 3, D]
    ring = ring.permute(0, 1, 3, 2, 4, 5).reshape(b, h * w, 9, d)
    keep = [i for i in range(9) if i != 4]          # `if i == y and j == x and k != 0: continue`
    out = torch.cat([out, ring[:, :, keep].reshape(b, h * w, 8 * d)], dim=-1)
    return out.squeeze(0) if squeeze else out


class PublishedDinoV2Backbone:
    """The published ViT-VS descriptor chain: torch.hub DINOv2, block 11, log-binned.

    Same public surface as :class:`FrozenBackbone` (``name`` / ``input_size`` /
    ``patch_size`` / ``grid_side`` / ``dim`` / ``encode``), so
    :class:`vsbench.vs.baselines.match.DenseMatcher` and :class:`vsbench.vs.baselines.tiling.TiledMatcher`
    take either without knowing which. Three deliberate differences from the DINOv3
    chain, all of them the published behaviour:

    1. **Weights come from ``torch.hub``** (``facebookresearch/dinov2``), not from
       transformers, and are **never downloaded**: ``torch.hub.load`` runs with
       ``skip_validation=True`` against the local cache and construction fails with
       the reason if it is not there. (U1 recorded "the DINOv2 weights are not
       cached"; that is true of the HF copy ``facebook/dinov2-small`` and false of
       this one - ``~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`` has
       been on the laptop since 2026-05-07, which is why this arm has a real parity
       fixture rather than a skipped one.)
    2. **CLS only, no registers.** DINOv2 emits ``[CLS] + 37*37`` tokens, so the
       prefix is 1 (``x[:, :, 1:, :]``, ``dinov2_extractor.py`` L372); DINOv3 emits
       ``[CLS] + 4 registers + 32*32`` and strips 5. Off-by-four here would shift the
       whole grid - the same fence, a different number, which is exactly why the
       prefix is read from the spec and the leftover count is checked.
    3. **Descriptors are log-binned and un-normalized**: 3456-d, no ``F.normalize``.
       :func:`vsbench.vs.baselines.match.mutual_nn_match` normalizes its inputs itself (the
       vendored ``chunk_cosine_sim`` did), so the pool is identical either way; but
       nothing downstream may assume unit norm on this backbone.
    4. **The model input must be contiguous** - see :meth:`preprocess`. The effect is
       ~250x larger here than on the DINOv3 chain (1.9e-4 against 7.3e-7) because
       these descriptors concatenate nine neighbourhoods of *un-normalized* block-11
       outputs (scale ~11) instead of one L2-normalized token.

    The descriptors are read with a **forward hook on block 11** rather than from
    ``last_hidden_state``: the published facet is the block's own output, before the
    model's final layer norm.

    ``model=`` injects an already-built module (the vendored ``ViTExtractor``'s own
    escape hatch), which is how the fast tests drive the shape and binning contract
    with a stub instead of 84 MB of weights.
    """

    def __init__(
        self,
        name: str = "dinov2_vits14",
        *,
        device: str | torch.device,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        model: torch.nn.Module | None = None,
        local_files_only: bool = True,
    ) -> None:
        spec = BACKBONES.get(name)
        if spec is None or spec.loader != "torch_hub":
            raise ValueError(
                f"{name!r} is not a torch.hub backbone; "
                f"torch.hub entries: "
                f"{', '.join(n for n, s in BACKBONES.items() if s.loader == 'torch_hub')}"
            )
        self.name = name
        self.spec = spec
        self.device = torch.device(device)
        self.dtype = dtype
        self.patch_size = int(spec.patch_size)
        self.dim = int(spec.dim)
        self.n_registers = 0
        self.n_prefix_tokens = 1  # CLS only - difference 2 in the class docstring
        self.input_size = int(spec.input_size if input_size is None else input_size)
        if self.input_size % self.patch_size:
            raise ValueError(
                f"input_size {self.input_size} is not a multiple of patch_size "
                f"{self.patch_size}: the token grid would not be square"
            )
        self.grid_side = self.input_size // self.patch_size
        #: ImageNet statistics, because the vendored extractor picks them for any
        #: ``"dino" in model_type`` (``dinov2_extractor.py`` L52-53). ``ToTensor`` +
        #: ``Normalize`` is arithmetically ``(x/255 - mean)/std``, i.e.
        #: :func:`normalize_imagenet` - no torchvision transform is needed to
        #: reproduce it.
        self.image_mean = (0.485, 0.456, 0.406)
        self.image_std = (0.229, 0.224, 0.225)
        self.model = model if model is not None else self._load(local_files_only)
        self.model = self.model.to(self.device, dtype=dtype).eval()
        self.model.requires_grad_(False)
        self._blocks = getattr(self.model, "blocks", None)
        if self._blocks is None or len(self._blocks) <= int(spec.layer):
            raise RuntimeError(
                f"{name}: the model exposes no `blocks[{spec.layer}]` to hook - the "
                "published descriptor is block 11's own output (facet 'token')"
            )

    def _load(self, local_files_only: bool) -> torch.nn.Module:
        """``torch.hub.load(...)`` from the cache only - never a download."""
        import os

        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            return torch.hub.load(
                self.spec.hub_repo, self.spec.hub_entry,
                source="github", trust_repo=True,
                **({"skip_validation": True} if local_files_only else {}),
            )
        except Exception as e:  # noqa: BLE001 - the message is the point
            raise RuntimeError(
                f"could not load torch.hub {self.spec.hub_repo}:{self.spec.hub_entry} "
                f"offline ({type(e).__name__}: {e}). The published ViT-VS DINOv2 weights "
                "are not in the torch.hub cache of this machine "
                "(~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth) and this "
                "repository never downloads weights - run where the cache already is."
            ) from e

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"PublishedDinoV2Backbone({self.name!r}, input_size={self.input_size}, "
            f"grid={self.grid_side}x{self.grid_side}, dim={self.dim}, "
            f"device={self.device})"
        )

    def preprocess(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """uint8 RGB frames -> ``(B, 3, S, S)``: Pillow resize, then ToTensor+Normalize.

        **The ``.contiguous()`` is load-bearing** (fence 4 of the class docstring):
        ``transforms.ToTensor`` returns a contiguous tensor, a ``permute`` does not,
        and the patch-embedding convolution takes a different kernel path for a
        non-contiguous input. Values identical to the last bit, descriptors
        **1.9e-4** apart - enough to flip mutual-nearest-neighbour argmaxes, which
        moved matched landmarks by up to 778 px on the parity pair. With it the
        descriptors are bit-identical to the published chain (measured 0.0).
        """
        arr = np.stack([resize_pil_exact(f, self.input_size) for f in frames])
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float().contiguous()
        return normalize_imagenet(x, self.image_mean, self.image_std).to(self.dtype)

    @torch.no_grad()
    def encode_batch(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Frames -> ``(B, side, side, 3456)`` float32 log-binned descriptors."""
        captured: list[torch.Tensor] = []
        handle = self._blocks[int(self.spec.layer)].register_forward_hook(
            lambda _m, _i, out: captured.append(out)
        )
        try:
            self.model(self.preprocess(frames))
        finally:
            handle.remove()
        feats = captured[0].float()
        n_tokens = int(feats.shape[1])
        expected = self.grid_side**2
        if n_tokens - self.n_prefix_tokens != expected:
            raise RuntimeError(
                f"{self.name} returned {n_tokens} tokens; stripping "
                f"{self.n_prefix_tokens} (CLS) leaves {n_tokens - self.n_prefix_tokens}, "
                f"not {self.grid_side}x{self.grid_side} = {expected}"
            )
        feats = feats[:, self.n_prefix_tokens :, :]
        if self.spec.binned:
            feats = log_bin(feats, (self.grid_side, self.grid_side))
        if int(feats.shape[-1]) != self.dim:
            raise RuntimeError(
                f"{self.name} produced {feats.shape[-1]}-d descriptors, not {self.dim}"
            )
        return feats.reshape(feats.shape[0], self.grid_side, self.grid_side, self.dim)

    @torch.no_grad()
    def encode(self, frame_rgb_u8: np.ndarray) -> torch.Tensor:
        """One frame -> ``(side, side, 3456)`` un-normalized log-binned descriptors."""
        return self.encode_batch([frame_rgb_u8])[0]


def build_backbone(
    name: str = DEFAULT_BACKBONE, **kwargs
) -> FrozenBackbone | PublishedDinoV2Backbone:
    """Name -> the class that loads it. The one entry point matchers call.

    :class:`FrozenBackbone` and :class:`PublishedDinoV2Backbone` share a surface but
    not a loader, and picking the wrong one silently reports a different descriptor
    under a published method's name - so the choice is made here, from
    :data:`BACKBONES`, and nowhere else.
    """
    spec = BACKBONES.get(name)
    if spec is None:
        raise ValueError(f"unknown backbone {name!r}; supported: {', '.join(sorted(BACKBONES))}")
    if spec.loader == "torch_hub":
        return PublishedDinoV2Backbone(name, **kwargs)
    return FrozenBackbone(name, **kwargs)
