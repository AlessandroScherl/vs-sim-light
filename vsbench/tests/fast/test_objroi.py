# vs-sim copy: tests/fast/test_objroi.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""ART-VS's object mode through stubs: the DETECT / TRACK / CACHE machine, the crop and
tile geometry lifted to full-image pixels, the switch, the record; and the goal mask on
SIFT, LightGlue and RoMa (the masked cells of every method)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vsbench.vs.baselines import objroi  # noqa: E402
from vsbench.vs.baselines.objroi import ObjectRoiTracker, RoiTiledMatcher, pad_box  # noqa: E402

W, H = 1920, 1080
GOAL_BOX = [800.0, 200.0, 1100.0, 800.0]          # 300 x 600 -> grid (2, 3) at vit 224


class _Backbone:
    """Content tokens: the crop resized to the token grid, plus a whisper of position."""

    name = "dinov2_vits14"
    input_size = 224
    patch_size = 14
    grid_side = 16
    dim = 5
    device = "cpu"

    def encode(self, rgb):
        import cv2

        small = cv2.resize(np.asarray(rgb, np.uint8), (16, 16), interpolation=cv2.INTER_AREA)
        t = small.astype(np.float32) / 255.0
        ys, xs = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        tok = np.concatenate([t, 0.02 * xs[..., None] / 16, 0.02 * ys[..., None] / 16], -1)
        tok = tok / np.linalg.norm(tok, axis=-1, keepdims=True)
        return torch.as_tensor(tok, dtype=torch.float32)


class _Detector:
    def __init__(self, boxes):
        self.boxes = list(boxes)         # one per call, None = no detection
        self.calls = 0

    def detect(self, rgb, prompt):
        b = self.boxes[min(self.calls, len(self.boxes) - 1)]
        self.calls += 1
        if b is None:
            return {"box": None, "score": 0.0, "mask": None, "n_boxes": 0}
        m = np.zeros((H, W), bool)
        x1, y1, x2, y2 = [int(v) for v in b]
        m[y1:y2, x1:x2] = True
        return {"box": list(b), "score": 0.9, "mask": m, "n_boxes": 1}


class _Tracker:
    def __init__(self, updates):
        self.updates = list(updates)     # per update() call: a box or None
        self.inits = []
        self.calls = 0

    def init(self, rgb, box):
        self.inits.append(list(box))

    def update(self, rgb):
        b = self.updates[min(self.calls, len(self.updates) - 1)]
        self.calls += 1
        return None if b is None else list(b)


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    return img


def _shifted(img, dx, dy):
    out = np.zeros_like(img)
    out[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = img[max(0, -dy):H - max(0, dy),
                                                                   max(0, -dx):W - max(0, dx)]
    return out


def test_padding_grows_the_box_by_its_own_size_and_clips_to_the_image():
    assert pad_box([100, 100, 300, 500], (W, H), 0.1) == [80, 60, 320, 540]
    assert pad_box([0, 0, 100, 100], (W, H), 0.5) == [0, 0, 150, 150]


def test_the_state_machine_detects_tracks_redetects_and_caches():
    det = _Detector([GOAL_BOX, GOAL_BOX, None, GOAL_BOX])
    trk = _Tracker([[810, 210, 1110, 810], None, None])
    t = ObjectRoiTracker(det, trk, prompt="shoe", padding=0.0, cache_timeout=2)
    assert t.goal(_scene(), (W, H))["box"] == GOAL_BOX and t.goal_mask.sum() == 300 * 600
    assert t.live(_scene(), (W, H)) == (GOAL_BOX, "langsam") and trk.inits == [GOAL_BOX]
    assert t.state == "TRACK"
    assert t.live(_scene(), (W, H)) == ([810, 210, 1110, 810], "lighttrack")
    # the tracker loses it and LangSAM fails: cache the last box, re-detect after timeout
    box, src = t.live(_scene(), (W, H))
    assert (box, src) == ([810, 210, 1110, 810], "cache") and t.state == "CACHE"
    assert t.live(_scene(), (W, H))[1] == "cache"
    assert t.live(_scene(), (W, H)) == (GOAL_BOX, "langsam") and t.state == "TRACK"
    assert t.counts == {"langsam": 2, "lighttrack": 1, "cache": 2, "none": 0}


def _matcher(det, trk, **kw):
    m = RoiTiledMatcher(num_landmarks=24, seed=3, backbone=_Backbone(), prompt="shoe",
                        tracker=ObjectRoiTracker(det, trk, prompt="shoe", padding=0.0), **kw)
    m.set_episode("shoe1", "t0")
    return m


def test_identical_crops_match_at_zero_displacement_inside_the_goal_box():
    img = _scene()
    m = _matcher(_Detector([GOAL_BOX, GOAL_BOX]), _Tracker([GOAL_BOX]))
    m.reset(img)
    assert m.goal_box == GOAL_BOX and m.grid == (2, 3)
    out = m.match(img)
    rows = out.to_rows()
    assert out.extras["mode"] == "coarse" and out.extras["roi_source"] == "langsam"
    assert len(rows["pts_goal"]) == 24
    assert (rows["pts_goal"][:, 0] >= 800).all() and (rows["pts_goal"][:, 0] <= 1100).all()
    assert (rows["pts_goal"][:, 1] >= 200).all() and (rows["pts_goal"][:, 1] <= 800).all()
    np.testing.assert_allclose(rows["pts_cur"], rows["pts_goal"], atol=1e-6)


def test_a_shifted_object_gives_the_shift_as_displacement_coarse_and_tiled():
    img = _scene()
    dx, dy = 60, -40
    live = _shifted(img, dx, dy)
    live_box = [GOAL_BOX[0] + dx, GOAL_BOX[1] + dy, GOAL_BOX[2] + dx, GOAL_BOX[3] + dy]
    m = _matcher(_Detector([GOAL_BOX, live_box]), _Tracker([live_box]), switch_iteration=2)
    m.reset(img)
    a = m.match(live).to_rows()
    np.testing.assert_allclose(a["pts_cur"] - a["pts_goal"], [[dx, dy]] * len(a["pts_goal"]),
                               atol=1e-6)
    out = m.match(live)                                   # frame 2: the switch fires
    assert out.extras["mode"] == "tiled" and out.extras["grid"] == [2, 3] and m.tiled
    b = out.to_rows()
    assert len(b["pts_goal"]) == 6 * max(1, 24 // 6)      # one grid of tiles, 4 pairs each
    np.testing.assert_allclose(b["pts_cur"] - b["pts_goal"], [[dx, dy]] * len(b["pts_goal"]),
                               atol=1e-6)
    f = m.record_fields()
    assert f["roi_grid"] == [2, 3] and f["roi_switched"] is True
    assert f["roi_sources"] == {"langsam": 1, "lighttrack": 1}


def test_no_box_at_all_falls_back_to_the_full_pictures_and_the_bench_mask_is_refused():
    img = _scene()
    m = _matcher(_Detector([None]), _Tracker([None]))
    m.reset(img)
    assert m.goal_box is None
    out = m.match(img)
    assert out.extras["mode"] == "full" and out.extras["roi_source"] == "none" and len(out) == 24
    with pytest.raises(ValueError, match="goal_mask is refused"):
        m.reset(img, goal_mask=np.ones((H, W), bool))
    assert RoiTiledMatcher.honours_goal_mask is False and objroi.CACHE_TIMEOUT == 10


def test_the_error_based_switch_latches_at_the_published_fraction():
    img = _scene()
    boxes = [GOAL_BOX] + [[GOAL_BOX[0] + d, GOAL_BOX[1], GOAL_BOX[2] + d, GOAL_BOX[3]]
                          for d in (100, 100, 10)]
    m = _matcher(_Detector([GOAL_BOX, boxes[1]]), _Tracker(boxes[2:]), switch_threshold=0.2)
    m.reset(img)
    m.match(_shifted(img, 100, 0))                        # e0 = 100 px
    assert not m.tiled
    m.match(_shifted(img, 100, 0))
    assert not m.tiled
    m.match(_shifted(img, 10, 0))                         # 10 / 100 < 0.2 -> latched
    assert m.tiled and m.match(_shifted(img, 10, 0)).extras["mode"] == "tiled"


def test_the_arm_file_builds_with_the_stub_and_is_labelled():
    from pathlib import Path

    from vsbench.vs.bench import arms
    from vsbench.vs.bench import config as cfgmod
    from vsbench.vs.scene import D435I

    cfg = cfgmod.load(Path(__file__).resolve().parents[2] / "configs" / "arms" / "artvs_roi.yaml")
    cfg["matcher"]["device"] = "cpu"
    m = _matcher(_Detector([GOAL_BOX]), _Tracker([GOAL_BOX]))
    arm = arms.build_arm(cfg, intrinsics=D435I, matcher=m)
    assert arm.name == "artvs_roi@dinov2_vits14/k24/ur/uniform/cns_plateau"
    assert arm.record_fields()["roi_prompt"] == "shoe" and arm.matcher_name == "artvs_roi"


# ------------------------------------------------------------ the mask on the others
def test_sift_with_a_goal_mask_detects_only_inside_it():
    pytest.importorskip("cv2")
    from vsbench.vs.baselines.classical import ClassicalMatcher
    from vsbench.vs.scene import checker_texture

    img = np.ascontiguousarray(checker_texture(2048)[:H, :W])   # non-repeating: the
    mask = np.zeros((H, W), bool)                                # ratio test needs that
    mask[300:700, 600:1200] = True
    m = ClassicalMatcher("sift", num_landmarks=24, seed=1, ransac="none")
    m.set_episode("s", 0)
    m.reset(img, goal_mask=mask)
    rows = m.match(img).to_rows()
    assert len(rows["pts_goal"]) == 24
    assert (rows["pts_goal"][:, 0] >= 600).all() and (rows["pts_goal"][:, 0] <= 1200).all()
    assert (rows["pts_goal"][:, 1] >= 300).all() and (rows["pts_goal"][:, 1] <= 700).all()
    assert ClassicalMatcher.honours_goal_mask is True


def test_lightglue_drops_the_goal_keypoints_outside_the_mask():
    from vsbench.vs.baselines import lightglue as lg

    feats = {"keypoints": torch.tensor([[[100., 100.], [500., 300.], [900., 600.], [1300., 200.]]]),
             "keypoint_scores": torch.tensor([[0.9, 0.8, 0.7, 0.6]]),
             "descriptors": torch.zeros(1, 4, 8), "image_size": torch.tensor([[W, H]])}
    mask = np.zeros((H, W), bool)
    mask[250:650, 450:950] = True
    kept, dropped = lg._keep_in_mask(feats, mask)
    assert dropped == 2 and kept["keypoints"].shape == (1, 2, 2)
    assert kept["descriptors"].shape == (1, 2, 8)
    assert kept["image_size"].shape == (1, 2)
    assert lg.LightGlueMatcher.honours_goal_mask is True


def test_roma_zeroes_the_certainty_outside_the_goal_mask():
    from vsbench.vs.baselines import roma as rm

    class _Stub:
        def match(self, a, b, device=None):
            # RoMa v1's symmetric layout: A|B side by side, (H, 2W, 4) and (H, 2W)
            hh, ww = 64, 64
            ys, xs = torch.meshgrid(torch.linspace(-1 + 1 / hh, 1 - 1 / hh, hh),
                                    torch.linspace(-1 + 1 / ww, 1 - 1 / ww, ww), indexing="ij")
            half = torch.stack([xs, ys, xs, ys], -1)
            return torch.cat([half, half], dim=1), torch.full((hh, 2 * ww), 0.9)

        def sample(self, warp, certainty, num):
            flat = warp.reshape(-1, 4)
            c = certainty.reshape(-1)
            idx = torch.multinomial(c, num, replacement=False)
            return flat[idx], c[idx]

        @staticmethod
        def to_pixel_coordinates(matches, H_A, W_A, H_B, W_B):
            a = torch.stack([(matches[:, 0] + 1) / 2 * W_A, (matches[:, 1] + 1) / 2 * H_A], -1)
            b = torch.stack([(matches[:, 2] + 1) / 2 * W_B, (matches[:, 3] + 1) / 2 * H_B], -1)
            return a, b

    mask = np.zeros((H, W), bool)
    mask[:, : W // 2] = True                       # the left half only
    m = rm.RoMaMatcher(version=1, num_landmarks=200, device="cpu", seed=1, model=_Stub())
    m.set_episode("s", 0)
    m.reset(np.zeros((H, W, 3), np.uint8), goal_mask=mask)
    rows = m.match(np.zeros((H, W, 3), np.uint8)).to_rows()
    assert (rows["pts_goal"][:, 0] < W / 2 + 1).all() and len(rows["pts_goal"]) == 200
    assert m.record_fields()["roma_goal_masked"] is True and rm.RoMaMatcher.honours_goal_mask


def test_the_scene_prompt_reaches_the_detector_and_the_label_keeps_the_setting():
    """`matcher.prompt: scene` (the instance suite's artvs_roi_scene cell): the word comes
    from the scene spec and goes to the matcher's constructor - so the detector is asked
    for it and the record's `roi_prompt` carries it - while the label is derived from the
    setting (`prompt-scene`: one label for the cell across every kind)."""
    from pathlib import Path

    from vsbench.vs.bench import arms
    from vsbench.vs.bench import config as cfgmod
    from vsbench.vs.scene import D435I
    from vsbench.vs.scenes import load_scenes

    specs, _ = load_scenes()
    cfg = cfgmod.load(Path(__file__).resolve().parents[2] / "configs" / "arms" / "artvs_roi.yaml")
    cfg["matcher"]["device"] = "cpu"
    cfg["matcher"]["prompt"] = "scene"
    # built through build_matcher (the constructor path); the detector and tracker models
    # are lazy, so nothing is loaded
    arm = arms.build_arm(cfg, intrinsics=D435I, scene_spec=specs["inst_bottle"],
                         backbone=_Backbone())
    assert arm.name == "artvs_roi@dinov2_vits14/k24/prompt-scene/ur/uniform/cns_plateau"
    assert arm.record_fields()["roi_prompt"] == "bottle"
    assert arm.matcher.tracker.prompt == "bottle"          # the word the detector is asked for
    assert arm.config["matcher"]["prompt"] == "scene"       # the setting is unchanged
    other = arms.build_arm(cfg, intrinsics=D435I, scene_spec=specs["inst_hammer"],
                           backbone=_Backbone())
    assert other.name == arm.name and other.record_fields()["roi_prompt"] == "hammer"
    with pytest.raises(ValueError, match="takes the word from the scene spec"):
        arms.build_arm(cfg, intrinsics=D435I, backbone=_Backbone())
    with pytest.raises(ValueError, match="states no `prompt:`"):
        arms.build_arm(cfg, intrinsics=D435I, scene_spec=specs["hollywood"],
                       backbone=_Backbone())
    # an injected matcher brings its own word; the refusals still run
    injected = arms.build_arm(cfg, intrinsics=D435I, scene_spec=specs["inst_mug"],
                              matcher=_matcher(_Detector([GOAL_BOX]), _Tracker([GOAL_BOX])))
    assert "prompt-scene" in injected.name
    assert arms.resolve_scene_prompt({"matcher": {"prompt": "shoe"}}, specs["inst_mug"]) is None
    assert arms.resolve_scene_prompt(cfg, specs["inst_mug"]) == "mug"


def test_the_artvs_extras_are_logged_and_nothing_else_moved():
    """The debug telemetry's logging-only extras (2026-09-30): the live box that was cropped,
    the encoder that ran, the switch's errors, the points per tile - and the rows are the
    ones the tests above already pin (the controller reads rows, never extras)."""
    img = _scene()
    dx, dy = 60, -40
    live = _shifted(img, dx, dy)
    live_box = [GOAL_BOX[0] + dx, GOAL_BOX[1] + dy, GOAL_BOX[2] + dx, GOAL_BOX[3] + dy]
    m = _matcher(_Detector([GOAL_BOX, live_box]), _Tracker([live_box]), switch_iteration=2)
    m.reset(img)
    coarse = m.match(live)
    ex = coarse.extras
    assert ex["mode"] == "coarse" and ex["live_box"] == live_box
    assert (ex["enc_input"], ex["enc_patch"]) == (_Backbone.input_size, _Backbone.patch_size)
    assert ex["switch_e0"] is None and ex["switch_e"] == pytest.approx(np.hypot(dx, dy))
    a = coarse.to_rows()
    np.testing.assert_allclose(a["pts_cur"] - a["pts_goal"], [[dx, dy]] * len(a["pts_goal"]),
                               atol=1e-6)
    tiled = m.match(live)
    ex = tiled.extras
    assert ex["mode"] == "tiled" and ex["live_box"] == live_box
    assert sum(ex["tile_counts"]) == len(tiled) and len(ex["tile_counts"]) == 6
    b = tiled.to_rows()
    np.testing.assert_allclose(b["pts_cur"] - b["pts_goal"], [[dx, dy]] * len(b["pts_goal"]),
                               atol=1e-6)
    # no box at all: the full pictures, no live box, the encoder still stated
    f = _matcher(_Detector([None]), _Tracker([None]))
    f.reset(img)
    ex = f.match(img).extras
    assert ex["mode"] == "full" and ex["live_box"] is None and ex["enc_input"] == 224


# ------------------------------------------------- the SAM path (2026-10-06, LangSAM check)
#: SAM-2's `SAM2Transforms` written as torchvision's composition - the formula the `langsam`
#: path's picture must equal (`docs/results/2026-10-06_langsam_check/`, workspace root)
def _sam2_formula(rgb):
    tv = pytest.importorskip("torchvision.transforms")
    return tv.Normalize(list(objroi.SAM2_MEAN), list(objroi.SAM2_STD))(
        tv.Resize((1024, 1024))(tv.ToTensor()(np.ascontiguousarray(rgb))))[None]


def test_grounding_sam_keeps_the_transformers_path_by_default_and_refuses_other_paths():
    """The switch is opt-in: `GroundingSam` and the old cell's matcher run SAM as before;
    `langsam` is asked for by name; anything else is refused. Nothing is loaded."""
    assert objroi.SAM_PATHS == ("transformers", "langsam")
    gs = objroi.GroundingSam(device="cpu")
    assert gs.sam_path == "transformers" and gs._sam is None and gs._gd is None
    assert objroi.GroundingSam(device="cpu", sam_path="langsam").sam_path == "langsam"
    for bad in ("lang_sam", "LangSAM", "sam2", ""):
        with pytest.raises(ValueError, match="sam_path"):
            objroi.GroundingSam(device="cpu", sam_path=bad)
    old = RoiTiledMatcher(num_landmarks=24, backbone=_Backbone(), device="cpu")
    assert old.tracker.detector.sam_path == "transformers"      # the artvs_roi* cells


def test_sam2_pixel_values_is_sam2_transforms_formula_bit_for_bit():
    """`ToTensor` -> `Resize((1024, 1024))` -> `Normalize` (torchvision), on random pictures
    of the bench's sizes, a smaller and an odd one, and one already 1024 x 1024 - equal to
    the last bit; and on the 1024 picture (no resize) the plain float32 arithmetic."""
    rng = np.random.default_rng(20261006)
    for h, w in ((1080, 1920), (540, 960), (53, 37), (777, 1301), (1024, 1024)):
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        px = objroi.sam2_pixel_values(rgb)
        assert px.shape == (1, 3, 1024, 1024) and px.dtype == torch.float32
        assert px.device.type == "cpu"
        assert torch.equal(px, _sam2_formula(rgb)), (h, w)
    mean = np.asarray(objroi.SAM2_MEAN, np.float32)
    std = np.asarray(objroi.SAM2_STD, np.float32)
    plain = (rgb.astype(np.float32) / np.float32(255) - mean) / std        # the last picture
    assert np.array_equal(px[0].permute(1, 2, 0).numpy(), plain)
    rgba = np.concatenate([rgb, np.full((1024, 1024, 1), 9, np.uint8)], -1)
    assert torch.equal(objroi.sam2_pixel_values(rgba), px)               # alpha dropped


def test_transformers_sam2_decoder_reads_the_stability_flag_the_langsam_path_sets():
    """The `langsam` path sets `mask_decoder.dynamic_multimask_via_stability = False` on the
    loaded model. That reaches the fallback only if the decoder's `forward` reads the
    instance attribute: shown on a tiny random decoder of the installed transformers - with
    the flag on (the config's value) the fallback runs and swaps the mask, set off it does
    not run."""
    pytest.importorskip("transformers")
    from transformers.models.sam2.configuration_sam2 import Sam2MaskDecoderConfig
    from transformers.models.sam2.modeling_sam2 import Sam2MaskDecoder

    torch.manual_seed(0)
    cfg = Sam2MaskDecoderConfig(hidden_size=32, mlp_dim=64, num_hidden_layers=1,
                                num_attention_heads=2, iou_head_hidden_dim=32)
    cfg._attn_implementation = "eager"          # a bare config names no attention kernel
    dec = Sam2MaskDecoder(cfg).eval()
    assert dec.dynamic_multimask_via_stability is True              # the config's default
    g = torch.Generator().manual_seed(1)
    kw = {"image_embeddings": torch.randn(1, 32, 4, 4, generator=g),
          "image_positional_embeddings": torch.randn(1, 32, 4, 4, generator=g),
          "sparse_prompt_embeddings": torch.randn(1, 1, 2, 32, generator=g),
          "dense_prompt_embeddings": torch.randn(1, 32, 4, 4, generator=g),
          "high_resolution_features": [torch.randn(1, 4, 16, 16, generator=g),
                                       torch.randn(1, 8, 8, 8, generator=g)]}
    calls, fallback = [], dec._dynamic_multimask_via_stability

    def spy(*a):
        calls.append(1)
        return fallback(*a)

    dec._dynamic_multimask_via_stability = spy
    with torch.no_grad():
        on = dec(multimask_output=False, **kw)[0]
        assert len(calls) == 1
        dec.dynamic_multimask_via_stability = False                 # what the path does
        off = dec(multimask_output=False, **kw)[0]
    assert len(calls) == 1 and on.shape == off.shape == (1, 1, 1, 16, 16)
    assert not torch.equal(on, off)        # random weights: token 0 is "unstable", swapped


# a stand-in `transformers` with the four classes `GroundingSam` loads: the wiring of the two
# paths, without a weight
class _Feat(dict):
    """A processor's BatchFeature: a dict with attribute reads and `.to(device)`, which
    records what it moved."""

    def __init__(self, log, **kw):
        super().__init__(**kw)
        self.log = log

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def to(self, device):
        self.log.append(("to", str(device), dict(self)))
        return self


PROC_PIXELS = torch.full((1, 3, 1024, 1024), 7.0)       # the processor's own picture
SAM_BOX = [10.0, 20.0, 50.0, 40.0]


def _fake_transformers(log, *, decoder_has_flag=True):
    import types

    class _Loaded:
        def to(self, device):
            log.append(("model_to", self.__class__.__name__, str(device)))
            return self

        def eval(self):
            return self

    class _GdProc:
        def __call__(self, images, text, return_tensors):
            return _Feat(log, input_ids=torch.zeros(1, 3, dtype=torch.long))

        def post_process_grounded_object_detection(self, out, input_ids, **kw):
            return [{"boxes": torch.tensor([SAM_BOX]), "scores": torch.tensor([0.9])}]

    class _Gd(_Loaded):
        def __call__(self, **inputs):
            return None

    class _SamProc:
        def __call__(self, images, input_boxes, return_tensors):
            log.append(("sam_proc", list(input_boxes[0][0]), images.size))
            w, h = images.size
            return _Feat(log, pixel_values=PROC_PIXELS.clone(),
                         original_sizes=torch.tensor([[h, w]]),
                         input_boxes=torch.tensor([input_boxes[0]]))

        def post_process_masks(self, pred_masks, original_sizes):
            h, w = (int(v) for v in original_sizes[0])
            return [torch.ones(1, 1, h, w)]

    class _Sam(_Loaded):
        def __init__(self):
            ns = types.SimpleNamespace
            self.mask_decoder = ns(dynamic_multimask_via_stability=True, training=False)
            if not decoder_has_flag:
                del self.mask_decoder.dynamic_multimask_via_stability
            self.config = ns(mask_decoder_config=ns(dynamic_multimask_via_stability=True))

        def __call__(self, **kw):
            log.append(("sam", kw))
            return types.SimpleNamespace(pred_masks=torch.zeros(1, 1, 1, 4, 4))

    def _maker(cls):
        return types.SimpleNamespace(from_pretrained=lambda name, **kw: cls())

    mod = types.ModuleType("transformers")
    mod.AutoProcessor = _maker(_GdProc)
    mod.AutoModelForZeroShotObjectDetection = _maker(_Gd)
    mod.Sam2Processor = _maker(_SamProc)
    mod.Sam2Model = _maker(_Sam)
    return mod


def test_the_langsam_path_turns_the_fallback_off_and_feeds_sam_the_sam2_picture(monkeypatch):
    """Both paths through stand-in model classes: the same box prompt, the same sizes and
    the same device move; `transformers` hands SAM the processor's picture and leaves the
    loaded decoder's flag as its config built it, `langsam` hands SAM `sam2_pixel_values`
    of the same picture (replaced before the move to the device) and turns the flag off on
    the decoder and in its config. A decoder with no such flag is refused on `langsam`."""
    import sys

    rgb = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    seen = {}
    for path in ("transformers", "langsam"):
        log: list = []
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(log))
        gs = objroi.GroundingSam(device="cpu", sam_path=path)
        d = gs.detect(rgb, "shoe", with_mask=True)
        assert d["box"] == SAM_BOX and d["mask"].shape == (48, 64) and d["mask"].all()
        flag = gs._sam.mask_decoder.dynamic_multimask_via_stability
        cfg_flag = gs._sam.config.mask_decoder_config.dynamic_multimask_via_stability
        assert (flag, cfg_flag) == ((True, True) if path == "transformers" else (False, False))
        (sam_kw,) = [e[1] for e in log if e[0] == "sam"]
        moved = [e for e in log if e[0] == "to" and "original_sizes" in e[2]]
        assert len(moved) == 1 and moved[0][1] == "cpu"
        assert moved[0][2]["pixel_values"] is sam_kw["pixel_values"]   # replaced, then moved
        assert [e[1:] for e in log if e[0] == "sam_proc"] == [(SAM_BOX, (64, 48))]
        assert torch.equal(sam_kw["original_sizes"], torch.tensor([[48, 64]]))
        assert torch.equal(sam_kw["input_boxes"], torch.tensor([[SAM_BOX]]))
        seen[path] = sam_kw["pixel_values"]
    assert torch.equal(seen["transformers"], PROC_PIXELS)
    assert torch.equal(seen["langsam"], objroi.sam2_pixel_values(rgb))
    assert torch.equal(seen["langsam"], _sam2_formula(rgb))
    # the box alone never loads SAM, on either path
    log = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(log))
    gs = objroi.GroundingSam(device="cpu", sam_path="langsam")
    assert gs.detect(rgb, "shoe", with_mask=False)["mask"] is None and gs._sam is None
    # a decoder without the flag cannot be switched: refused, nothing kept
    monkeypatch.setitem(sys.modules, "transformers",
                        _fake_transformers([], decoder_has_flag=False))
    gs = objroi.GroundingSam(device="cpu", sam_path="langsam")
    with pytest.raises(RuntimeError, match="dynamic_multimask_via_stability"):
        gs.detect(rgb, "shoe", with_mask=True)
    assert gs._sam is None
    plain = objroi.GroundingSam(device="cpu")                   # the old path does not care
    assert plain.detect(rgb, "shoe", with_mask=True)["box"] == SAM_BOX
