# vs-sim copy: tests/fast/test_objroi_sim.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The faithful ART-VS arm (``artvs_roi_sim``) against the release's own code, on stubs.

The reference is the ART-VS release (git 9b1208c, MIT) vendored under
``tests/fixtures/reference/artvs_release/`` (``PROVENANCE.md``): ``matcher.py`` and
``controller.py`` byte for byte, ``log_bin_release.py`` (``_log_bin``), and the ROS-bound
code assembled from the release's lines - ``roi_glue.py`` (the simulation controller's
methods) and ``detector_state.py`` (the LangSAM + LightTrack state machine, with LangSAM's
``predict`` and the LightTrack class injected).

One scripted world drives both sides: a weight-free content encoder (the same function
behind the port's backbone and the release's extractor interface), a scripted detector and
tracker keyed by the FRAME's bytes - so a side that asks twice gets the same answer, as the
real models would - and the release's global ``torch.randperm`` seeded, call by call, with
the seed the port's draw used (on CPU the two permutations are the same numbers). Every
comparison below is exact unless it says otherwise. The ids F1-F13 are the build spec's
(``docs/results/2026-10-05_artvs_release_check/sim_spec.md`` section 5.2, workspace root);
the M tests are the owner's evening ruling (LangSAM's goal mask, D-B (b)).
"""

from __future__ import annotations

import hashlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from PIL import Image  # noqa: E402

from vsbench.vs.baselines import objroi  # noqa: E402
from vsbench.vs.baselines.control import (  # noqa: E402
    UNIFORM_WEIGHTING,
    ArtvsVisual90,
    IBVSController,
    StopObservation,
    build_stop_rule,
)
from vsbench.vs.baselines.features import FrozenBackbone, log_bin, resize_pil_exact  # noqa: E402
from vsbench.vs.baselines.match import DenseMatcher, EpisodeDraw, mutual_nn_match  # noqa: E402
from vsbench.vs.baselines.objroi import (  # noqa: E402
    ReleaseRoiTracker,
    RoiTiledMatcherSim,
    jpeg_round_trip,
    lift32,
    release_patch_mask,
    uv_roi,
)
from vsbench.vs.bench import arms  # noqa: E402
from vsbench.vs.bench import config as cfgmod  # noqa: E402
from vsbench.vs.scene import D435I  # noqa: E402

REF = Path(__file__).resolve().parents[1] / "fixtures" / "reference" / "artvs_release"


def _ref(name: str):
    """A vendored release module, loaded by path under its own name (nothing on sys.path)."""
    import importlib.util

    key = f"artvs_release_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REF / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


roi_glue = _ref("roi_glue")                    # the simulation controller's methods
rel_detector = _ref("detector_state")          # the LangSAM + LightTrack state machine
rel_matcher = roi_glue._vendored_matcher       # features/matcher.py, byte for byte
rel_controller = roi_glue._vendored_controller  # core/controller.py, byte for byte
log_bin_release = _ref("log_bin_release").log_bin_release

W, H = 1920, 1080
VIT, PATCH, SIDE = 256, 16, 16
D = 51                                   # the stub token width (48 content + 2 position + 1)
SEED = 20260902
SCENE, START = "inst_stub", "s0"
INTR = D435I                             # the bench camera (rule 5: no camera numbers here)


# ------------------------------------------------------------------ the stub world
def _tokens(img256: np.ndarray) -> np.ndarray:
    """A content descriptor per 16 x 16 cell: the cell subsampled 4 x 4 (48 numbers, mean
    removed), a whisper of position, L2-normalised. Distinct on a random texture."""
    a = np.asarray(img256, np.float32) / 255.0
    cells = a.reshape(SIDE, PATCH, SIDE, PATCH, 3)[:, ::4, :, ::4, :]
    feat = cells.transpose(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 48)
    feat = feat - feat.mean(axis=-1, keepdims=True)
    ys, xs = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    pos = np.stack([xs, ys], -1).astype(np.float32) * 0.002
    tok = np.concatenate([feat, pos, np.full((SIDE, SIDE, 1), 0.01, np.float32)], -1)
    return tok / (np.linalg.norm(tok, axis=-1, keepdims=True) + 1e-12)


class _SimBackbone:
    """The port's side: the backbone interface the arm encodes through."""

    name = "dinov3_vits16_artvs256"
    input_size = VIT
    patch_size = PATCH
    grid_side = SIDE
    dim = D
    device = "cpu"

    def __init__(self):
        self.batches: list[int] = []         # the batch sizes asked for, in order

    def encode_batch(self, frames):
        self.batches.append(len(frames))
        return torch.from_numpy(np.stack([_tokens(resize_pil_exact(np.asarray(f), VIT))
                                          for f in frames]))

    def encode(self, frame):
        return self.encode_batch([frame])[0]


class _ReleaseExtractor:
    """The release's side: `MultiBackboneViTExtractor`'s interface over the same tokens."""

    device = "cpu"

    def __init__(self):
        self.batches: list[int] = []

    def preprocess_pil(self, pil):
        return torch.from_numpy(np.asarray(pil.convert("RGB"), np.uint8).copy())[None]

    def extract_descriptors(self, t, layer=11, facet="token", bin=False, **kw):
        imgs = t.numpy()
        self.batches.append(len(imgs))
        toks = np.stack([_tokens(resize_pil_exact(im, VIT)) for im in imgs])
        return torch.from_numpy(toks).reshape(len(imgs), 1, SIDE * SIDE, D)

    def get_patch_size(self):
        return PATCH


def _key(rgb) -> str:
    return hashlib.md5(np.ascontiguousarray(np.asarray(rgb, np.uint8)).tobytes()).hexdigest()


class _World:
    """What the detector and the tracker answer, per frame (by its bytes)."""

    def __init__(self):
        self.boxes: dict[str, list | None] = {}    # LangSAM's best box, or None
        self.masks: dict[str, np.ndarray] = {}     # SAM's mask of that box
        self.tracks: dict[str, list | None] = {}   # LightTrack's update box (xyxy ints)
        self.port_inits: list = []
        self.rel_inits: list = []

    def add(self, rgb, *, box=None, track=None, mask=None):
        k = _key(rgb)
        self.boxes[k] = None if box is None else [float(v) for v in box]
        self.tracks[k] = None if track is None else [int(v) for v in track]
        if mask is not None:
            self.masks[k] = np.asarray(mask, bool)
        return rgb


class _PortDetector:
    def __init__(self, world):
        self.world = world

    def detect(self, rgb, prompt, with_mask=True):
        k = _key(rgb)
        box = self.world.boxes.get(k)
        if box is None:
            return {"box": None, "score": 0.0, "mask": None, "n_boxes": 0}
        mask = self.world.masks.get(k, np.ones((H, W), bool)) if with_mask else None
        return {"box": list(box), "score": 0.9, "mask": mask, "n_boxes": 1}


class _PortTracker:
    def __init__(self, world):
        self.world, self.key = world, None

    def init(self, rgb, box):
        self.world.port_inits.append(list(box))
        self.key = _key(rgb)

    def update(self, rgb):
        t = self.world.tracks.get(_key(rgb))
        return None if t is None else list(t)


def _predict_for(world):
    """LangSAM's `predict([pil], [prompt])` result shape, from the world."""
    def predict(images, prompts):
        rgb = np.asarray(images[0].convert("RGB"))
        k = _key(rgb)
        box = world.boxes.get(k)
        if box is None:
            return [{"masks": np.zeros((0, H, W), bool), "boxes": np.zeros((0, 4), np.float32),
                     "scores": np.zeros(0, np.float32)}]
        mask = world.masks.get(k, np.ones((H, W), bool))
        return [{"masks": mask[None].astype(np.float32),
                 "boxes": np.asarray([box], np.float32),
                 "scores": np.asarray([0.9], np.float32)}]
    return predict


def _lighttrack_for(world):
    """The release wrapper's interface (BGR in, `(ok, (x, y, w, h))` ints out)."""
    class _LT:
        def __init__(self, weights=None, arch=None):
            pass

        def init(self, bgr, xywh):
            world.rel_inits.append(tuple(xywh))

        def update(self, bgr):
            t = world.tracks.get(_key(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            if t is None:
                return False, None
            return True, (t[0], t[1], t[2] - t[0], t[3] - t[1])
    return _LT


def _texture(seed=0):
    """A random texture, smooth at the 4-px scale so a resized crop keeps its content."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (H // 4, W // 4, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((W, H), Image.NEAREST))


def _shift(img, dx, dy):
    out = np.zeros_like(img)
    out[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = \
        img[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
    return out


def _sbox(box, dx, dy):
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def _rel_config(**over):
    """The simulation config's values (S:) + code defaults, object mode switched on as the
    owner ruled (sim_spec.md 2.1), at the bench's camera and gain."""
    c = types.SimpleNamespace(
        backbone_model="dinov3-small", vit_input_size=VIT, u_max=W, v_max=H,
        f_x=INTR.fx, f_y=INTR.fy, c_x=W / 2, c_y=H / 2, lambda_=2.0, max_velocity=10.0,
        num_pairs=24, min_iterations=300, max_iterations=1500, max_velocity_vector_history=200,
        ema_alpha=0.8, thresh_filter_keypoints=1, use_feature_binning=True, feature_hierarchy=1,
        use_tiling=True, use_hybrid_mode=True, tiling_switch_iteration=0,
        hybrid_switch_threshold=0.2, roi_tiling_switch_threshold=0.2, tile_number=36,
        auto_calculate_tiles=True, use_roi_detection=True, use_roi_tiling=True,
        roi_padding_ratio=0.1, cache_timeout=10, roi_tile_number=4,
        enable_visualization=False, mask_path=None, detection_keyword=None)
    for k, v in over.items():
        setattr(c, k, v)
    return c


class _Pair:
    """The port's matcher and the release's controller, side by side on one world."""

    def __init__(self, world, goal, *, mask_path=None, tie_order="legacy", goal_mask=None,
                 **port_kw):
        self.world = world
        self.backbone = _SimBackbone()
        tracker = ReleaseRoiTracker(_PortDetector(world), _PortTracker(world), prompt="shoe")
        self.port = RoiTiledMatcherSim(num_landmarks=24, seed=SEED, backbone=self.backbone,
                                       tie_order=tie_order, prompt="shoe", tracker=tracker,
                                       **port_kw)
        self.port.set_episode(SCENE, START)
        self.seeds: list[int] = []
        self._spy()
        if goal_mask is None:
            self.port.reset(goal)
        else:                     # the dev cell (roi_mask: objects): the bench's goal mask
            self.port.reset(goal, goal_mask=goal_mask)
        self.extractor = _ReleaseExtractor()
        self.det = rel_detector.LangSAMLightTrackDetector(
            predict=_predict_for(world), tracker_factory=_lighttrack_for(world),
            cache_timeout=10)
        self.cfg = _rel_config(mask_path=mask_path)
        self.glue = roi_glue.ROSVisualServoingController(
            self.cfg, self.extractor, self.det, Image.fromarray(goal))
        self.glue.set_yolo_keyword("shoe")
        self.det.reset_tracking()
        self.glue.begin_sample()
        self.points = None
        real = self.glue.detect_features

        def capture():
            self.points = real()
            return self.points
        self.glue.detect_features = capture
        self.first = True

    def _spy(self):
        pair = self

        def match(d, cur_tokens, **kw):
            pair.seeds.append(d.draw.frame_seed())
            return orig(d, cur_tokens, **kw)
        orig = objroi._TokenDraw.match
        self._orig_match = orig
        self._match = match

    def frame(self, rgb, depth_mm=None):
        """One frame on both sides -> (port Matches, release ((s_uv_star, s_uv), sims) or
        None, release twist or None)."""
        self.seeds = []
        mp = pytest.MonkeyPatch()
        mp.setattr(objroi._TokenDraw, "match", self._match)
        try:
            m = self.port.match(rgb)
        finally:
            mp.undo()
        seeds = list(self.seeds)
        pil = Image.fromarray(rgb)
        g = self.glue
        g.latest_pil_image, g.latest_image = pil, rgb
        g.latest_image_depth = (np.full((H, W), 500, np.uint16) if depth_mm is None
                                else depth_mm)
        if self.first:
            # the driver, before run() (run_multi_model_experiment_with_hollywood.py:741-755)
            self.first = False
            d = self.det.detect(pil, keyword=g.yolo_keyword, padding_ratio=0.1)
            if d["bbox"] is not None:
                self.det.start_tracking(np.array(pil), d["bbox"])
        it = iter(seeds)
        real = rel_matcher.find_correspondences_batch

        def seeded(*a, **k):
            torch.manual_seed(next(it))
            return real(*a, **k)
        mp = pytest.MonkeyPatch()
        mp.setattr(roi_glue, "find_correspondences_batch", seeded)
        self.points = None
        try:
            v = g.step()
        finally:
            mp.undo()
        return m, self.points, v


def _ints(m: objroi.Matches, image_wh=(W, H)):
    """The port's rows as the controller truncates them (pixel_quantisation: truncate)."""
    ctrl = IBVSController(intrinsics=INTR, pixel_quantisation="truncate",
                          weighting=UNIFORM_WEIGHTING)
    if not len(m):
        return np.zeros((0, 2)), np.zeros((0, 2))
    rows = m.to_rows()
    if "quantise" in m.extras:
        rows["quantise"] = m.extras["quantise"]
    g, c, _ = ctrl.quantise(rows)
    return g, c


def _rel_ints(points):
    if points is None or points[0] is None:
        return None
    (g, c), _ = points
    return np.asarray(g, np.float64), np.asarray(c, np.float64)


GOAL_BOX = [800.5, 200.25, 1100.75, 800.5]      # raw 300.25 x 600.25 -> grid (2, 3)


# =============================================================== F1 - the descriptor
def test_f1_log_bin_is_the_releases_and_the_backbone_bins_before_it_normalises():
    """`log_bin(hierarchy=1)` is `_log_bin` bit for bit; `FrozenBackbone._to_grid` on a binned
    entry is bin-then-normalise on the release's memory layout - the release's descriptors to
    the last bit - and NOT normalise-then-bin."""
    rng = np.random.default_rng(1)
    raw = torch.from_numpy(rng.standard_normal((2, SIDE * SIDE, 384)).astype(np.float32)) * 3
    rel = log_bin_release(raw.unsqueeze(1), (SIDE, SIDE), 1, "cpu")[:, 0]
    assert torch.equal(log_bin(raw, (SIDE, SIDE), 1), rel)              # 0.0
    ns = types.SimpleNamespace(grid_side=SIDE, n_prefix_tokens=5, n_registers=4,
                               bin_hierarchy=1, dim=9 * 384, model_id="stub")
    prefix = torch.from_numpy(rng.standard_normal((2, 5, 384)).astype(np.float32))
    got = FrozenBackbone._to_grid(ns, torch.cat([prefix, raw], 1))
    want = torch.nn.functional.normalize(
        log_bin_release(raw.unsqueeze(1), (SIDE, SIDE), 1, "cpu").float(), p=2, dim=-1)[:, 0]
    assert torch.equal(got.reshape(2, SIDE * SIDE, -1), want)           # the release, exactly
    wrong = log_bin(torch.nn.functional.normalize(raw, dim=-1), (SIDE, SIDE), 1)
    assert not torch.allclose(got.reshape(2, SIDE * SIDE, -1),
                              torch.nn.functional.normalize(wrong, dim=-1), atol=1e-3)
    # the cosine matrix the matcher sees is the release's too (chunk_cosine_sim)
    g, c = got[0], got[1]
    sims = rel_matcher.chunk_cosine_sim(want[0:1, None], want[1:2, None])[0, 0]
    gg, cc = g.reshape(-1, g.shape[-1]), c.reshape(-1, c.shape[-1])
    port = torch.nn.functional.normalize(gg, dim=-1) @ torch.nn.functional.normalize(cc,
                                                                                    dim=-1).T
    assert torch.equal(port, sims)


# ========================================================= F2 - the preprocessing
def test_f2_the_crop_and_resize_are_the_releases_pillow_calls():
    img = _texture(3)
    box = [700.4, 150.9, 1201.2, 811.0]
    pil = Image.fromarray(img)
    roi = rel_detector.LangSAMLightTrackDetector.crop_roi(None, pil, box)   # the release's
    port = objroi.resize(objroi.crop(img, box), (VIT, VIT))
    assert np.array_equal(port, np.asarray(roi.resize((VIT, VIT))))        # :1407, 1415
    tw, th = 2 * VIT, 3 * VIT
    assert np.array_equal(objroi.resize(objroi.crop(img, box), (tw, th)),
                          np.asarray(roi.resize((tw, th))))                 # :1641-1648


def test_f2_the_backbone_preprocess_is_the_models_own_processor_at_256():
    pytest.importorskip("transformers")
    try:
        bb = FrozenBackbone("dinov3_vits16_artvs256", device="cpu")
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        pytest.skip(f"DINOv3 ViT-S/16 not in the local HF cache ({type(exc).__name__})")
    from vsbench.vs.baselines.features import normalize_imagenet

    img = np.asarray(Image.fromarray(_texture(4)[:300, :400]).resize((VIT, VIT)))
    px = bb.preprocess([img])
    want = bb.processor(images=Image.fromarray(img), return_tensors="pt")["pixel_values"]
    assert torch.equal(px, want)                                         # the release's route
    plain = normalize_imagenet(torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float(),
                               bb.image_mean, bb.image_std)
    assert float((px - plain).abs().max()) <= 1e-6                       # 7.2e-7 measured
    tok = bb.encode(img)
    assert tuple(tok.shape) == (SIDE, SIDE, 3456)
    torch.testing.assert_close(tok.norm(dim=-1), torch.ones(SIDE, SIDE), atol=1e-5, rtol=0)


# ============================================================== F3 - the matcher
@pytest.mark.parametrize("masked", [False, True])
def test_f3_the_pool_and_the_seeded_draw_are_find_correspondences_batch(masked):
    """On random 16 x 16 x 3456 descriptors, CPU, `legacy` ties: the port's 24 are the
    release's 24, in order, when the release's global RNG carries the port's frame seed -
    with and without the release's goal-cell mask."""
    rng = np.random.default_rng(7)
    for trial in range(40):
        d1 = torch.nn.functional.normalize(torch.from_numpy(
            rng.standard_normal((256, 3456)).astype(np.float32)), dim=-1)
        d2 = torch.nn.functional.normalize(
            d1 + 0.8 * torch.from_numpy(rng.standard_normal((256, 3456)).astype(np.float32)),
            dim=-1)
        mask = rng.random(256) > 0.4 if masked else None
        pool = mutual_nn_match(d1, d2, tie_order="legacy", mask=mask)
        draw = EpisodeDraw(SCENE, trial, SEED)
        seed = draw.frame_seed()
        m = draw.draw(pool, 24)
        torch.manual_seed(seed)
        mt = None if mask is None else torch.from_numpy(mask.reshape(1, -1))
        p1, p2, sim = rel_matcher.find_correspondences_batch(
            d1[None, None], d2[None, None], num_pairs=24, mask_tensor=mt, vit_input_size=VIT,
            patch_size=PATCH)
        assert np.array_equal(p1.numpy(), m.goal_patch) and np.array_equal(p2.numpy(),
                                                                          m.cur_patch)
        assert np.array_equal(sim.numpy().astype(np.float64), m.sim)


def test_the_token_draw_is_the_dense_matchers_draw_on_the_same_tokens():
    """`_TokenDraw` is `DenseMatcher` with the encode call moved out (so the tiles go
    through the encoder in the release's batch shapes)."""
    bb = _SimBackbone()
    img = _texture(5)
    a = objroi.resize(objroi.crop(img, GOAL_BOX), (VIT, VIT))
    b = objroi.resize(objroi.crop(_shift(img, 30, -20), _sbox(GOAL_BOX, 30, -20)), (VIT, VIT))
    dm = DenseMatcher(num_landmarks=24, seed=SEED, backbone=bb, tie_order="legacy")
    dm.set_episode(SCENE, START)
    dm.reset(a)
    td = objroi._TokenDraw(num_landmarks=24, seed=SEED, threshold=1.0, tie_order="legacy")
    td.set_episode(SCENE, START)
    td.reset(bb.encode(a))
    for _ in range(3):
        x, y = dm.match(b), td.match(bb.encode(b), input_size=VIT, patch_size=PATCH,
                                     image_wh=(VIT, VIT))
        assert np.array_equal(x.goal_patch, y.goal_patch) and np.array_equal(x.cur_patch,
                                                                            y.cur_patch)
        assert x.draw_seed == y.draw_seed and np.array_equal(x.sim, y.sim)


# ===================================================== F4 - the goal box and the grid
@pytest.mark.parametrize("box,grid", [
    ([800.5, 200.25, 1100.75, 800.5], (2, 3)),     # raw 300 x 600
    ([900.0, 400.0, 1140.0, 640.0], (1, 1)),       # raw 240: padded 288 would give 2 x 2
    ([300.0, 0.0, 1600.0, 1080.0], (5, 5)),        # raw 1300 x 1080 -> (6 -> 5, 5)
])
def test_f4_padded_box_raw_box_and_the_grid_from_the_unpadded_box(box, grid):
    world = _World()
    goal = world.add(_texture(6), box=box)
    pair = _Pair(world, goal)
    p, g = pair.port, pair.glue
    assert p.goal_box_raw == box and p.goal_box == g.goal_roi_bbox
    assert p.grid == grid == (g.optimal_grid_width, g.optimal_grid_height)
    padded = objroi.pad_box(box, (W, H), 0.1)
    by_padded = (min(5, math.ceil((padded[2] - padded[0]) / VIT)),
                 min(5, math.ceil((padded[3] - padded[1]) / VIT)))
    if box[2] - box[0] == 240:
        assert by_padded == (2, 2) != grid             # the two rules differ here
    n = grid[0] * grid[1]
    assert [d.num_landmarks for d in p._tiles] == [max(1, 24 // n)] * n


def _scene_pair(*, live_shifts, goal_box=GOAL_BOX, seed=11, mask=None, mask_path=None,
                tie_order="legacy", **kw):
    """A goal picture and one live frame per shift; the live object is the goal's, moved.
    LangSAM finds the moved FLOAT box; LightTrack follows with its INT box."""
    world = _World()
    tex = _texture(seed)
    goal = world.add(tex, box=goal_box, mask=mask)
    frames = []
    for dx, dy in live_shifts:
        img = _shift(tex, dx, dy)
        k = _key(img)
        if k not in world.boxes:
            world.add(img, box=_sbox(goal_box, dx, dy),
                      track=[int(v) for v in _sbox(goal_box, dx, dy)])
        frames.append(img)
    return world, goal, frames


def _assert_same_frame(pair, m, pts, v, *, mode=None):
    """The port's frame == the release's frame: integer points, the live box, the error."""
    rel = _rel_ints(pts)
    g, c = _ints(m)
    if rel is None or len(rel[0]) < objroi.MIN_POINTS:
        assert len(g) < objroi.MIN_POINTS and m.extras["degenerate"] and v is None
        return
    assert np.array_equal(g, rel[0]) and np.array_equal(c, rel[1])
    assert not m.extras["degenerate"] and v is not None
    if mode is not None:
        assert m.extras["mode"] == mode
    live = m.extras["live_box"]
    if live is not None:
        assert live == [float(x) for x in pair.glue.current_roi_bbox]
    assert pair.port.e_prev == pytest.approx(pair.glue.current_feature_error, abs=0, rel=0)
    assert pair.port.e0 == pair.glue.initial_feature_error


# ============================================================ F5 - a coarse frame
def test_f5_a_coarse_frame_is_the_releases_to_the_pixel():
    world, goal, frames = _scene_pair(live_shifts=[(37, -23), (41, -19)])
    pair = _Pair(world, goal)
    for img in frames:
        m, pts, v = pair.frame(img)
        _assert_same_frame(pair, m, pts, v, mode="coarse")
        rows = m.to_rows()
        box = m.extras["live_box"]
    ex = m.extras
    assert ex["roi_source"] == "lighttrack" and ex["enc_input"] == VIT and ex["enc_bins"] == 9
    assert ex["grid"] == [2, 3] and ex["tiled"] is False and ex["switch_e0"] == pair.port.e0
    assert len(rows["pts_goal"]) == 24 and box is not None


def test_f5_lift32_is_the_releases_map_roi_to_full_image():
    """`lift32` against the transcription's `_map_roi_to_full_image` on every patch centre of
    padded LangSAM-like boxes (float32 corners, 10 % padding): bit-identical. A float64 lift
    differs in 98 % of coordinates and truncates to a different whole pixel in 5.0e-5 of
    them (512 of 10.24 M, measured 2026-10-05) - about one point in 400 frames, which is why
    the port lifts in float32 like the release."""
    glue = types.SimpleNamespace(config=_rel_config())
    cols, rows_ = np.meshgrid(np.arange(SIDE), np.arange(SIDE))
    vit = np.stack([cols.ravel(), rows_.ravel()], 1).astype(np.float32) * 16 + 8
    rng = np.random.default_rng(0)
    n = differs = 0
    for _ in range(2000):
        xy = np.float32(rng.uniform(0, 1500, 2))
        raw = [float(xy[0]), float(xy[1]), float(xy[0] + np.float32(rng.uniform(50, 400))),
               float(xy[1] + np.float32(rng.uniform(50, 500)))]
        box = objroi.pad_box(raw, (W, H), 0.1)
        rel = roi_glue.ROSVisualServoingController._map_roi_to_full_image(
            glue, torch.from_numpy(vit), box, None).numpy()
        port = lift32(vit, box, (VIT, VIT))
        assert np.array_equal(port, rel)
        f64 = objroi.RoiTiledMatcher._lift(vit.astype(np.float64), box, (VIT, VIT))
        n += f64.size
        differs += int(np.sum(f64.astype(int) != rel.astype(np.float64).astype(int)))
    assert 0 < differs < 2e-4 * n          # rare, real: a float64 lift WOULD move pixels


def test_f5_the_controller_truncates_the_rows_exactly_as_the_matcher_did():
    """The matcher truncates a copy for its switch error; the controller truncates the rows
    the Matches carry - after their 16:9 round trip. The two must give the same integers."""
    rng = np.random.default_rng(9)
    pg = np.round(rng.uniform(0, [W - 1, H - 1], (5000, 2)), 0).astype(np.float32)
    pg[::3] += np.float32(0.5)
    pc = (pg + rng.integers(-40, 40, pg.shape)).astype(np.float32)
    m = objroi.Matches.from_pixel_pairs(
        pg.astype(np.float64), pc.astype(np.float64), side=W, input_size=W, patch_size=1,
        image_wh=(W, H), sim=np.zeros(len(pg)), cyclic=np.zeros(len(pg)),
        margin=np.zeros(len(pg)), pool_size=len(pg), num_landmarks=len(pg), draw_seed=0,
        tie_order="stable")
    g, c = _ints(m)
    rg, rc = uv_roi(pg, pc, (W, H))
    assert np.array_equal(g, rg) and np.array_equal(c, rc)
    # and a bare np.trunc would not: the round trip leaves whole pixels just below them
    raw = m.to_rows()["pts_goal"]
    assert np.any(np.trunc(raw) != np.trunc(pg.astype(np.float64)))


# ============================================================ F6 - the switch
def test_f6_the_switch_needs_ten_frames_reads_the_previous_frame_and_latches():
    """e0 = 100 px; frame 9 drops below 0.2 e0 but only 9 frames have succeeded; frame 12
    drops again -> the first tiled frame is 13 (checked at its start on frame 12's error)."""
    shifts = [(100, 0)] * 8 + [(10, 0)] + [(100, 0)] * 2 + [(10, 0)] * 3
    world, goal, frames = _scene_pair(live_shifts=shifts)
    pair = _Pair(world, goal)
    modes = []
    for t, img in enumerate(frames, start=1):
        m, pts, v = pair.frame(img)
        _assert_same_frame(pair, m, pts, v)
        assert pair.port.tiled == pair.glue.roi_tiling_activated, t
        modes.append(m.extras["mode"])
    assert modes == ["coarse"] * 12 + ["tiled"] * 2
    assert pair.port.switch_frame == pair.glue.roi_tiling_switch_iteration == 12
    assert pair.port.switch_step == 13
    assert pair.port.record_fields()["roi_switch_frame"] == 12


@pytest.mark.parametrize("e", [20.0, 19.999, 20.001])
def test_f6_the_comparison_is_the_releases_expression_at_the_boundary(e):
    """`1 - e/e0 >= 1 - 0.2` exactly as written (ros_controller.py:622, 630) - e = 0.2 e0
    included - on both sides, from the same state."""
    rel = types.SimpleNamespace(
        config=_rel_config(), roi_tiling_activated=False, roi_tiling_switch_iteration=None,
        vs_controller=types.SimpleNamespace(iteration_count=12), initial_feature_error=100.0,
        current_feature_error=e)
    roi_glue.ROSVisualServoingController.check_roi_tiling_switch(rel)
    port = RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(), tracker=ReleaseRoiTracker(
        _PortDetector(_World()), _PortTracker(_World()), prompt="x"))
    port.frames_ok, port.e0, port.e_prev = 12, 100.0, e
    port._check_switch()
    assert port.tiled == rel.roi_tiling_activated == ((1.0 - e / 100.0) >= (1.0 - 0.2))


# =================================================== F7 - a tiled frame and the cap
@pytest.mark.parametrize("goal_box,grid", [(GOAL_BOX, (2, 3)), ([300.0, 0.0, 1600.0, 1080.0],
                                                                (5, 5))])
def test_f7_a_tiled_frame_is_the_releases_tile_offsets_lift_and_top24(goal_box, grid):
    world, goal, frames = _scene_pair(live_shifts=[(24, 12), (24, 12), (30, 8)],
                                      goal_box=goal_box)
    pair = _Pair(world, goal, switch_iteration=1)
    pair.cfg.tiling_switch_iteration = 1                 # the release's iteration switch
    for t, img in enumerate(frames, start=1):
        m, pts, v = pair.frame(img)
        _assert_same_frame(pair, m, pts, v, mode="coarse" if t == 1 else "tiled")
    assert pair.port.grid == grid
    n = grid[0] * grid[1]
    assert len(m.extras["tile_counts"]) == n
    total = sum(m.extras["tile_counts"])
    assert len(m) == min(24, total)
    if n == 25:
        assert total == 25 and len(m) == 24                   # the cap binds at 5 x 5 only
        assert np.all(np.diff(m.sim) <= 0)                     # torch.topk's order
    # every tile encoded as ONE batch per side, as the release does (:1751-1762)
    assert pair.backbone.batches[-1] == n and n in pair.extractor.batches


def test_f7_the_pairs_per_tile_table_for_every_reachable_grid():
    budget = {}
    for gw in range(1, 6):
        for gh in range(1, 6):
            n = gw * gh
            budget[n] = max(1, 24 // n)
    assert budget == {1: 24, 2: 12, 3: 8, 4: 6, 5: 4, 6: 4, 8: 3, 9: 2, 10: 2, 12: 2,
                      15: 1, 16: 1, 20: 1, 25: 1}
    totals = {n: min(24, n * k) for n, k in budget.items()}
    assert totals == {1: 24, 2: 24, 3: 24, 4: 24, 5: 20, 6: 24, 8: 24, 9: 18, 10: 20,
                      12: 24, 15: 15, 16: 16, 20: 20, 25: 24}


# =================================================== F8 - the first live frame
def test_f8_the_first_frame_is_langsam_then_init_then_update_with_int_boxes():
    world, goal, frames = _scene_pair(live_shifts=[(17, 9)])
    pair = _Pair(world, goal)
    m, pts, v = pair.frame(frames[0])
    _assert_same_frame(pair, m, pts, v, mode="coarse")
    lang = _sbox(GOAL_BOX, 17, 9)                                 # LangSAM's float box
    assert world.port_inits == [[int(x) for x in lang]]           # int-cast before xywh
    x1, y1, x2, y2 = world.port_inits[0]
    assert world.rel_inits == [(x1, y1, x2 - x1, y2 - y1)]
    track = world.tracks[_key(frames[0])]
    assert m.extras["roi_source"] == "lighttrack"                 # the UPDATE's box is used
    assert m.extras["live_box"] == objroi.pad_box(track, (W, H), 0.1)
    t = pair.port.tracker
    assert (t.langsam_calls, t.tracker_inits, t.cached_bbox) == (1, 1, track)
    assert t.state == pair.det.state == "TRACK"


def test_f8_lost_cache_and_the_release_timeout_counter():
    """The state machine on its own, port vs release, frame for frame: tracking lost and
    LangSAM empty -> CACHE with the lost frame counted (re-detection nine frames later), a
    failed re-detection keeps the counter (every later frame re-detects), DETECT never
    starts the tracker by itself."""
    world = _World()
    tex = _texture(13)
    goal = world.add(tex, box=GOAL_BOX)
    seq = [world.add(_shift(tex, 5, 0), box=_sbox(GOAL_BOX, 5, 0), track=[805, 200, 1105, 800])]
    for i in range(1, 15):                                       # lost, LangSAM empty
        seq.append(world.add(_shift(tex, 5 + i, 0)))
    seq.append(world.add(_shift(tex, 40, 0), box=_sbox(GOAL_BOX, 40, 0)))   # found again
    seq.append(world.add(_shift(tex, 41, 0), track=[841, 200, 1141, 800]))
    port = ReleaseRoiTracker(_PortDetector(world), _PortTracker(world), prompt="shoe")
    port.goal(goal, (W, H))
    rel = rel_detector.LangSAMLightTrackDetector(predict=_predict_for(world),
                                                 tracker_factory=_lighttrack_for(world))
    rel.detect_and_cache_goal(Image.fromarray(goal), "shoe", 0.1)
    rel.reset_tracking()
    port.reset()
    cache = None                                                 # the controller's own cache
    for t, img in enumerate(seq):
        pil = Image.fromarray(img)
        if t == 0:
            d = rel.detect(pil, keyword="shoe", padding_ratio=0.1)
            rel.start_tracking(np.array(pil), d["bbox"])
        d = rel.detect(pil, keyword="shoe", padding_ratio=0.1)
        if d["bbox"] is not None:
            cache = d["padded_bbox"]
        want = cache
        got, _raw, src = port.live(img, (W, H))
        assert got == (None if want is None else [float(x) for x in want]), t
        assert (port.state, port.cache_counter) == (rel.state, rel.cache_counter), t
    assert port.counts["langsam"] >= 1 and port.counts["controller_cache"] > 0


# ======================================== F9 - degenerate frames and the failure run
def _cell_mask(goal_box, cells, *, image_wh=(W, H)):
    """A mask covering whole coarse cells of the padded goal box (cells = [(row, col)])."""
    m = np.zeros((image_wh[1], image_wh[0]), bool)
    x1, y1, x2, y2 = objroi.pad_box(goal_box, image_wh, 0.1)
    cw, ch = (x2 - x1) / SIDE, (y2 - y1) / SIDE
    for r, c in cells:
        m[int(y1 + (r + 0.1) * ch):int(y1 + (r + 0.9) * ch) + 1,
          int(x1 + (c - 0.2) * cw):int(x1 + (c + 1.2) * cw)] = True
    return m


def test_f9_a_frame_with_three_points_is_skipped_and_an_empty_pool_counts(tmp_path):
    """Three eligible goal cells -> three pairs -> fewer than four points: the release skips
    the frame (no twist, no count, no error) and its failure counter does NOT move (the pool
    was not empty); the port marks it degenerate and updates nothing. Then an empty pool ten
    frames in a row: the release aborts the sample, the port records the run and goes on."""
    mask3 = _cell_mask(GOAL_BOX, [(7, 4), (7, 8), (7, 12)])
    world, goal, frames = _scene_pair(live_shifts=[(0, 0)] * 3, mask=mask3)
    path = tmp_path / "mask3.png"
    Image.fromarray(mask3.astype(np.uint8) * 255).save(path)
    pair = _Pair(world, goal, mask_path=str(path))
    assert pair.port.mask_cells["coarse"] == 3
    for img in frames:
        m, pts, v = pair.frame(img)
        assert len(m) == 3 and m.extras["degenerate"] and v is None
        assert np.array_equal(_rel_ints(pts)[0], np.zeros((0, 2)))
        assert pair.port.frames_ok == pair.glue.vs_controller.iteration_count == 0
        assert pair.port.failure_run == pair.glue.feature_failure_count == 0
    # an empty pool: a mask with no eligible cell
    dot = np.zeros((H, W), bool)
    dot[500:503, 950:953] = True
    world, goal, frames = _scene_pair(live_shifts=[(0, 0)] * 12, mask=dot, seed=12)
    path = tmp_path / "dot.png"
    Image.fromarray(dot.astype(np.uint8) * 255).save(path)
    pair = _Pair(world, goal, mask_path=str(path))
    assert pair.port.mask_cells["coarse"] == 0
    raised_at = None
    for t, img in enumerate(frames, start=1):
        if raised_at is None:
            try:
                m, pts, v = pair.frame(img)
            except RuntimeError as exc:
                assert str(exc) == "Persistent feature detection failure"
                raised_at = t
                continue
        else:
            m = pair.port.match(img)
        assert len(m) == 0 and m.extras["degenerate"]
    assert raised_at == 10                                    # the release aborts here
    rec = pair.port.record_fields()
    assert rec["roi_failure_run_max"] == 12 and rec["roi_failure_abort_frame"] == 10
    assert rec["roi_degenerate_frames"] == 12


# ================================================ F10 - depth and the twist, as released
def _vs_config(**over):
    return _rel_config(**over)


def test_f10_get_depth_is_depth_at_with_the_100m_fallback():
    rng = np.random.default_rng(4)
    mm = rng.integers(200, 9_999, (H, W)).astype(np.uint16)
    mm[rng.random((H, W)) < 0.05] = 0                         # the plugin's "no depth"
    pts = np.column_stack([rng.integers(0, W, 500), rng.integers(0, H, 500)])
    pts[:20] = np.argwhere(mm.T == 0)[:20]                    # some land on a zero
    rel = roi_glue.ROSVisualServoingController.get_depth(
        types.SimpleNamespace(latest_image_depth=mm), pts)[:, 0]
    port = arms.depth_at(mm.astype(np.float64) / 1000.0, pts.astype(np.float64),
                         fallback_m=100.0)
    assert np.array_equal(port, rel) and np.sum(port == 100.0) >= 20
    # the default stays ViT-VS's 0.5 m, so every earlier arm reads what it read
    assert np.all(arms.depth_at(mm / 1000.0, pts.astype(np.float64))[port == 100.0] == 0.5)


def _release_twists(frames, cfg):
    """The release's controller on scripted integer points: transform, depth, law, EMA, the
    cap - or None for a skipped frame (< 4 points), as run() would."""
    vs = rel_controller.VisualServoingController(cfg)
    out = []
    for g, c, z in frames:
        if len(g) < 4:
            out.append(None)
            continue
        s_xy, s_star = vs.transform_to_real_world(c, g)
        v = vs.compute_control_velocity(s_xy, s_star, z.reshape(-1, 1))
        vs.update_iteration_stats(v)
        out.append(vs.store_applied_velocities(v))
    return out


def test_f10_the_twist_is_the_releases_law_ema_and_cap_and_a_skip_holds():
    """Six frames (one of them with three points) through the vendored controller and
    through `IBVSController(truncate, ema 0.8, cap, hold)`: the twist on every frame the
    release publishes, the held twist on the one it skips (the camera keeps moving at the
    last twist in Gazebo), and the EMA untouched by the skip."""
    rng = np.random.default_rng(5)
    frames = []
    for t in range(6):
        k = 3 if t == 3 else 24
        g = rng.integers(100, 1800, (k, 2)).astype(np.float64)
        g[:, 1] = np.clip(g[:, 1], 0, H - 1)
        c = g + rng.integers(-60, 60, (k, 2)) * (6 - t) / 6
        c = np.clip(np.trunc(c), 0, [W - 1, H - 1])
        frames.append((g, c, rng.uniform(0.25, 0.6, k)))
    for cap in (10.0, 0.05):                                   # 0.05 makes the cap bind
        cfg = _vs_config(max_velocity=cap)
        rel = _release_twists(frames, cfg)
        ctrl = IBVSController(gain=2.0, intrinsics=INTR, weighting=UNIFORM_WEIGHTING,
                              ema_alpha=0.8, pixel_quantisation="truncate", max_velocity=cap,
                              degenerate_policy="hold")
        held = None
        for t, (g, c, z) in enumerate(frames):
            tw, dbg = ctrl.step({"pts_goal": g, "pts_cur": c}, z)
            if rel[t] is None:
                assert dbg["degenerate"] and dbg["degenerate_hold"]
                assert np.array_equal(tw.as_array(), held)    # the last applied twist
                continue
            assert np.max(np.abs(tw.as_array() - rel[t])) <= 1e-12, (cap, t)
            assert dbg["velocity_capped"] == bool(np.any(np.abs(
                ctrl._ema_v) > cap))
            held = tw.as_array()


# ================================================ F11 - the visual-only stop, recorded
def test_f11_artvs_visual90_fires_where_check_convergence_visual_only_does():
    rng = np.random.default_rng(6)
    for trial in range(30):
        n = 120
        errs = 80.0 * np.exp(-np.arange(n) / rng.uniform(10, 60)) + rng.uniform(0, 2, n)
        skip = rng.random(n) < 0.15                           # degenerate frames
        rel = rel_controller.VisualServoingController(_vs_config(real_robot_max_iterations=10**6))
        rule = ArtvsVisual90()
        rule.reset()
        rel_fire = port_fire = None
        for step in range(1, n + 1):
            e = float(errs[step - 1])
            if skip[step - 1]:                                # the release never sees it
                v = rule.update(StopObservation(step=step, normalised_errors=None,
                                                pixel_error_mean=float("nan")))
            else:
                rel.iteration_count += 1
                done, conv = rel.check_convergence_visual_only(feature_error=e)
                if done and conv and rel_fire is None:
                    rel_fire = step
                v = rule.update(StopObservation(step=step, normalised_errors=np.zeros(3),
                                                pixel_error_mean=e))
            if v.would_stop and port_fire is None:
                port_fire = step
        assert port_fire == rel_fire, trial
        assert rule.e0 == rel.initial_feature_error_visual_only
    assert rule.params == {"target": 0.9, "release_cap_iterations": 500}


# =========================================================== F12 - the fallbacks
def test_f12_no_live_box_rounds_the_full_pictures_and_no_goal_box_scales_the_goal():
    world = _World()
    tex = _texture(21)
    goal = world.add(tex, box=GOAL_BOX)
    lost = [world.add(_shift(tex, 8 * i, 4 * i)) for i in (1, 2)]       # LangSAM: nothing
    pair = _Pair(world, goal)
    for img in lost:
        m, pts, v = pair.frame(img)
        assert m.extras["mode"] == "full_fallback" and m.extras["quantise"] == "round"
        rel = _rel_ints(pts)
        g, c = _ints(m)
        assert np.array_equal(g, rel[0]) and np.array_equal(c, rel[1])   # calculate_uv
        assert np.all(g == np.round(g)) and m.extras["live_box"] is None
    # no goal box: the full goal at 256 against the live crop, goal points x (W/256, H/256)
    world = _World()
    goal = world.add(_texture(22))                                       # nothing found
    live = world.add(_shift(_texture(22), 12, 6), box=_sbox(GOAL_BOX, 12, 6),
                     track=[812, 206, 1112, 806])
    pair = _Pair(world, goal)
    assert pair.port.goal_box is None and pair.glue.goal_roi_bbox is None
    m, pts, v = pair.frame(live)
    assert m.extras["mode"] == "goalless"
    _assert_same_frame(pair, m, pts, v)
    rows = m.to_rows()
    gx = rows["pts_goal"] / np.asarray([W / VIT, H / VIT])
    assert np.allclose((gx - 8) % 16, 0, atol=1e-6)                       # patch centres


def test_f12_the_tiled_phase_without_any_box_is_the_stated_deviation():
    """Box-less from the first frame, switched by iteration: the release would run its
    full-image 6 x 6 tiling (not transcribed - it raises here); the port runs the coarse
    full-picture match and counts the frame as the deviation."""
    world = _World()
    tex = _texture(23)
    goal = world.add(tex, box=GOAL_BOX)
    frames = [world.add(_shift(tex, 3 * i, 0)) for i in (1, 2, 3)]
    pair = _Pair(world, goal, switch_iteration=1)
    pair.cfg.tiling_switch_iteration = 1
    m, pts, v = pair.frame(frames[0])
    assert m.extras["mode"] == "full_fallback"
    with pytest.raises(NotImplementedError, match="stated deviation"):
        pair.frame(frames[1])
    m = pair.port.match(frames[2])
    assert m.extras["mode"] == "full_fallback_deviation"
    rec = pair.port.record_fields()
    assert rec["roi_modes"] == {"full_fallback": 1, "full_fallback_deviation": 2}
    assert rec["roi_fallback_frames"] == 3 and rec["roi_failure_run_max"] == 0


# ============================================================ F13 - the config
def test_f13_artvs_visual90_is_a_normal_rule_recorded_on_the_arm():
    cfg = cfgmod.load(Path(__file__).resolve().parents[2] / "configs" / "arms"
                      / "artvs_roi_sim.yaml")
    assert "artvs_visual90" in cfg["episode"]["record_stops"]
    assert "artvs_visual90" in arms.episode_settings(cfg)["record_stops"]
    scored = dict(cfg, stop={"rule": "artvs_visual90"})
    cfgmod.validate(scored)                                   # accepted as a scored rule too
    rule = build_stop_rule(scored)
    assert isinstance(rule, ArtvsVisual90) and rule.name == "artvs_visual90"
    with pytest.raises(ValueError, match="target"):
        ArtvsVisual90(target=1.5)


# ===================================================== M - LangSAM's goal mask
def test_m1_the_cell_rule_is_the_releases_at_both_thresholds():
    """`release_patch_mask` == `create_patch_mask_from_pil` and `create_patch_mask` (the
    file route): a pixel counts above 128 (128 does not), a cell when MORE than half of its
    256 pixels count (128 do not, 129 do)."""
    rng = np.random.default_rng(8)
    for _ in range(20):
        a = (rng.random((VIT, VIT)) * 255).astype(np.uint8)
        pil = Image.fromarray(a)
        want = rel_matcher.create_patch_mask_from_pil(pil, VIT, PATCH).numpy().reshape(-1)
        assert np.array_equal(release_patch_mask(pil, VIT, PATCH), want)
    a = np.zeros((VIT, VIT), np.uint8)
    a[0:16, 0:16].flat[:128] = 255          # cell (0, 0): 128 pixels set -> mean 0.5 -> out
    a[0:16, 16:32].flat[:129] = 255         # cell (0, 1): 129 -> in
    a[16:32, 0:16] = 128                    # cell (1, 0): every pixel at 128 -> not > 128
    a[16:32, 16:32] = 129                   # cell (1, 1): 129 -> in
    pil = Image.fromarray(a)
    got = release_patch_mask(pil, VIT, PATCH).reshape(SIDE, SIDE)
    assert (got[0, 0], got[0, 1], got[1, 0], got[1, 1]) == (False, True, False, True)
    assert np.array_equal(got.reshape(-1),
                          rel_matcher.create_patch_mask_from_pil(pil, VIT, PATCH).numpy()[0])


def test_m1_the_jpeg_round_trip_is_the_releases_temporary_file(tmp_path):
    rng = np.random.default_rng(10)
    m = np.zeros((H, W), np.uint8)
    m[300:700, 800:1150] = 255
    m[rng.random((H, W)) < 0.01] = 255
    roi = Image.fromarray(m).crop((770.4, 140.6, 1130.5, 860.5)).resize(
        (VIT, VIT), Image.Resampling.LANCZOS)
    roi.save(tmp_path / "m.jpg", "JPEG")                        # ros_controller.py:398-400
    via_file = rel_matcher.create_patch_mask(str(tmp_path / "m.jpg"), VIT, PATCH).numpy()[0]
    assert np.array_equal(np.asarray(jpeg_round_trip(roi)),
                          np.asarray(Image.open(tmp_path / "m.jpg").convert("L")))
    assert np.array_equal(release_patch_mask(jpeg_round_trip(roi), VIT, PATCH), via_file)


def _blob_mask(goal_box, seed=0):
    """A LangSAM-like goal mask: an ellipse inside the box with a ragged edge."""
    x1, y1, x2, y2 = goal_box
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.42, (y2 - y1) * 0.45
    r = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    rng = np.random.default_rng(seed)
    return r < 1.0 + 0.15 * rng.standard_normal((H, W))


def test_m2_the_goal_mask_through_both_phases_is_the_releases(tmp_path):
    """LangSAM's mask on the goal, held as a lossless picture and fed to the release as
    `mask_path`: the coarse frames (float-box crop, LANCZOS, the JPEG), the switch, the tiled
    frames (int-box crop, BICUBIC, per-tile cells) - the same points to the pixel - and the
    eligible cells counted the same way the release's own functions count them."""
    mask = _blob_mask(GOAL_BOX)
    path = tmp_path / "goal_mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    world, goal, frames = _scene_pair(live_shifts=[(14, 6), (15, 6), (16, 7), (16, 7)],
                                      mask=mask)
    pair = _Pair(world, goal, mask_path=str(path), switch_iteration=2)
    pair.cfg.tiling_switch_iteration = 2
    p = pair.port
    # the coarse mask is the release's own: its temporary JPEG through create_patch_mask
    want = rel_matcher.create_patch_mask(pair.glue.goal_roi_mask_path, VIT, PATCH).numpy()[0]
    assert np.array_equal(p._coarse.mask, want)
    assert 0 < p.mask_cells["coarse"] < SIDE * SIDE
    assert p.mask_cells["full"] == int(rel_matcher.create_patch_mask(
        str(path), VIT, PATCH).numpy().sum())
    for t, img in enumerate(frames, start=1):
        m, pts, v = pair.frame(img)
        _assert_same_frame(pair, m, pts, v, mode="coarse" if t <= 2 else "tiled")
        # every goal point lies on an eligible goal cell of its phase's mask
        assert len(m) > 0
    rec = p.record_fields()
    assert rec["roi_mask"] == "langsam" and rec["roi_mask_box"] == GOAL_BOX
    assert rec["roi_mask_share"] == pytest.approx(float(mask.mean()))
    assert len(rec["roi_mask_cells"]["tiles"]) == 6
    # a mask over everything leaves every coarse cell eligible (the mask is what restricts)
    world2, goal2, _ = _scene_pair(live_shifts=[(14, 6)], mask=np.ones((H, W), bool))
    assert _Pair(world2, goal2).port.mask_cells["coarse"] == SIDE * SIDE


def test_m2_the_box_less_fallback_uses_the_whole_goal_mask(tmp_path):
    mask = _blob_mask(GOAL_BOX, seed=3)
    path = tmp_path / "goal_mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    world = _World()
    tex = _texture(24)
    goal = world.add(tex, box=GOAL_BOX, mask=mask)
    lost = world.add(_shift(tex, 9, 3))
    pair = _Pair(world, goal, mask_path=str(path))
    want = rel_matcher.create_patch_mask(str(path), VIT, PATCH).numpy()[0]
    assert np.array_equal(pair.port._full.mask, want)
    m, pts, v = pair.frame(lost)
    assert m.extras["mode"] == "full_fallback"
    rel = _rel_ints(pts)
    g, c = _ints(m)
    assert np.array_equal(g, rel[0]) and np.array_equal(c, rel[1])   # padding included
    drawn = pair.port._full.mask.sum()
    assert len(g) == 24 and drawn >= 4 and m.extras["padded_rows"] == 24 - int(
        np.sum(np.any(g != 0, axis=1)))
    assert np.all(g[24 - m.extras["padded_rows"]:] == 0)
    assert pair.port.record_fields()["roi_padded_frames"] == 1


def test_m2_a_box_less_draw_of_three_pairs_is_24_zero_rows_and_not_skipped(tmp_path):
    """`calculate_uv` returns its 24 zero rows for a draw of 1-3 pairs and `ibvs` does not
    skip them (it only checks for fewer than 4 ROWS): the frame commands a zero error
    through the EMA and reports a zero feature error - on both sides."""
    mask = np.zeros((H, W), bool)
    cw, ch = W / SIDE, H / SIDE             # one full-picture cell = 120 x 67.5 frame px
    for col in (2, 7, 12):                  # three whole cells of row 6
        mask[int(6 * ch) + 2:int(7 * ch) - 2, int(col * cw) + 2:int((col + 1) * cw) - 2] = True
    path = tmp_path / "goal_mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    world = _World()
    tex = _texture(26)
    goal = world.add(tex, box=GOAL_BOX, mask=mask)
    pair = _Pair(world, goal, mask_path=str(path))
    cells = int(pair.port._full.mask.sum())
    assert 1 <= cells <= 3
    m, pts, v = pair.frame(world.add(_shift(tex, 1, 0)))        # nothing detected on it
    rel = _rel_ints(pts)
    g, c = _ints(m)
    assert len(g) == 24 and np.all(g == 0) and np.all(c == 0) and np.array_equal(g, rel[0])
    assert not m.extras["degenerate"] and v is not None           # NOT skipped
    assert pair.port.e_prev == pair.glue.current_feature_error == 0.0


def test_m3_the_old_mask_rule_and_anything_but_langsam_are_refused():
    with pytest.raises(ValueError, match="old port's mask rule"):
        RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(), use_sam_mask=True)
    with pytest.raises(ValueError, match="only value is 'langsam'"):
        RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(), roi_mask="sam")
    world = _World()
    goal = world.add(_texture(25), box=GOAL_BOX)
    m = RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(),
                           tracker=ReleaseRoiTracker(_PortDetector(world), _PortTracker(world),
                                                     prompt="x"))
    m.set_episode(SCENE, START)
    with pytest.raises(ValueError, match="goal_mask is refused"):
        m.reset(goal, goal_mask=np.ones((H, W), bool))
    assert m.tracker.goal_mask_wanted is True                 # the goal asks SAM for a mask


def test_m4_sam_runs_as_lang_sam_runs_it_and_nothing_else_is_accepted():
    """2026-10-06, the LangSAM check: the arm builds its GroundingSam on the `langsam` path
    (SAM 2's stability fallback off, SAM2Transforms' picture), states it on the record, and
    refuses any other path - including an injected real detector on another path. A stub
    detector (this file's) is accepted: it makes no SAM mask of its own."""
    m = RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone())
    assert m.roi_sam_path == "langsam" and objroi.ROI_SAM_PATHS == ("langsam",)
    assert isinstance(m.tracker.detector, objroi.GroundingSam)
    assert m.tracker.detector.sam_path == "langsam" and m.tracker.detector._sam is None
    assert m.record_fields()["roi_sam_path"] == "langsam"
    for bad in ("transformers", "lang_sam", ""):
        with pytest.raises(ValueError, match=f"roi_sam_path is {bad!r}"):
            RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(), roi_sam_path=bad)
    old_detector = objroi.GroundingSam(device="cpu")                  # transformers' SAM
    with pytest.raises(ValueError, match="injected tracker's GroundingSam"):
        RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(),
                           tracker=ReleaseRoiTracker(old_detector, _PortTracker(_World()),
                                                     prompt="x"))
    ok = RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(),
                            tracker=ReleaseRoiTracker(
                                objroi.GroundingSam(device="cpu", sam_path="langsam"),
                                _PortTracker(_World()), prompt="x"))
    assert ok.record_fields()["roi_sam_path"] == "langsam"


# ======================================================= the arm in the episode loop
class _FixedDetector:
    """The mock's object never moves in its own picture as much as a real one: one box,
    and SAM's mask is the box itself."""

    def __init__(self, box, hw):
        self.box, self.hw = [float(v) for v in box], hw

    def detect(self, rgb, prompt, with_mask=True):
        m = None
        if with_mask:
            m = np.zeros(self.hw, bool)
            x1, y1, x2, y2 = (int(v) for v in self.box)
            m[y1:y2, x1:x2] = True
        return {"box": list(self.box), "score": 0.9, "mask": m, "n_boxes": 1}


class _FixedTracker:
    def __init__(self, box):
        self.box, self.inits = [int(v) for v in box], 0

    def init(self, rgb, box):
        self.inits += 1

    def update(self, rgb):
        return list(self.box)


def test_the_arm_flies_the_episode_loop_and_its_record_says_what_ran(tmp_path):
    """The arm file's settings through `runner.run_episode` on the mock plane, with the
    debug recorder listening: the loop runs, the controller solves on whole pixels, and the
    record carries the controller's knobs, the matcher's ROI fields and the recorded stops."""
    from vsbench.vs.bench import runner
    from vsbench.vs.bench.debuglog import DebugLog
    from vsbench.vs.scene import MockPlaneScene

    scene = MockPlaneScene(render_scale=0.25)
    hw = (scene.intrinsics.height, scene.intrinsics.width)
    box = [scene.intrinsics.width * 0.3, scene.intrinsics.height * 0.25,
           scene.intrinsics.width * 0.7, scene.intrinsics.height * 0.75]
    cfg = cfgmod.load(Path(__file__).resolve().parents[2] / "configs" / "arms"
                      / "artvs_roi_sim.yaml")
    cfg["matcher"] = {**cfg["matcher"], "device": "cpu", "prompt": "plate"}
    cfg["episode"] = {**cfg["episode"], "max_iter": 40}
    m = RoiTiledMatcherSim(num_landmarks=24, seed=cfg["seed"], backbone=_SimBackbone(),
                           prompt="plate", switch_iteration=20,
                           tracker=ReleaseRoiTracker(_FixedDetector(box, hw),
                                                     _FixedTracker(box), prompt="plate"))
    arm = arms.build_arm(cfg, intrinsics=scene.intrinsics, matcher=m)
    g = scene.goal_pose
    start = {"start_id": "t0", "cam_pos": g.position + np.array([0.01, -0.008, 0.005]),
             "cam_R_opt": g.rotation.tolist(), "cell": "mock", "tier": "mock",
             "roll_deg": 0.0, "starts_seed": 1}
    telem: list = []
    dlog = DebugLog(tmp_path / "ep", arm_id="artvs_roi_sim")
    try:
        rec = runner.run_episode(scene, arm, start, cfg, telemetry=telem, debug_log=dlog)
    finally:
        dlog.close()
    assert rec["steps"] == 40 and rec["terminated_reason"] == "max_iter"
    assert "debug_log_error" not in rec
    assert rec["arm"] == ("artvs_roi_sim@dinov3_vits16_artvs256/k24/lsmask/sam-langsam/tr/"
                          "uniform/ema0.8/vcap10/hold/cns_plateau")
    assert (rec["pixel_quantisation"], rec["max_velocity"], rec["degenerate_policy"],
            rec["depth_fallback_m"], rec["ema_alpha"], rec["solve_unrounded"]) == (
        "truncate", 10.0, "hold", 100.0, 0.8, False)
    assert rec["roi_mask"] == "langsam" and rec["roi_mask_share"] > 0
    assert rec["roi_sam_path"] == "langsam"                  # on every record (2026-10-06)
    assert rec["roi_switch_frame"] == 20 and rec["roi_modes"] == {"coarse": 20, "tiled": 20}
    assert rec["roi_sources"] == {"lighttrack": 40} and rec["roi_tracker_inits"] == 1
    assert {"velocity_quiet", "dec_plateau", "artvs_visual90"} <= set(
        arms.episode_settings(cfg)["record_stops"])
    for e in telem:                                          # whole pixels on the solve
        assert all(float(u).is_integer() for u in e.get("gu", []) + e.get("cv", []))


class _GoallessDetector(_FixedDetector):
    """Misses the object on its first call (the goal picture), finds it after."""

    def __init__(self, box, hw):
        super().__init__(box, hw)
        self.calls = 0

    def detect(self, rgb, prompt, with_mask=True):
        self.calls += 1
        if self.calls == 1:
            return {"box": None, "score": 0.0, "mask": None, "n_boxes": 0}
        return super().detect(rgb, prompt, with_mask)


def test_a_goal_without_a_detection_keeps_the_debug_log(tmp_path):
    """Review 2026-10-05 (C1): with no goal detection the arm keeps `grid = None`, which the
    debug recorder's `begin` must survive - the flight runs the goal-less path and its debug
    log is complete (it used to be lost with `debug_log_error: begin: TypeError`)."""
    from vsbench.vs.bench import runner
    from vsbench.vs.bench.debuglog import DebugLog
    from vsbench.vs.scene import MockPlaneScene

    scene = MockPlaneScene(render_scale=0.25)
    hw = (scene.intrinsics.height, scene.intrinsics.width)
    box = [scene.intrinsics.width * 0.3, scene.intrinsics.height * 0.25,
           scene.intrinsics.width * 0.7, scene.intrinsics.height * 0.75]
    cfg = cfgmod.load(Path(__file__).resolve().parents[2] / "configs" / "arms"
                      / "artvs_roi_sim.yaml")
    cfg["matcher"] = {**cfg["matcher"], "device": "cpu", "prompt": "plate"}
    cfg["episode"] = {**cfg["episode"], "max_iter": 12}
    m = RoiTiledMatcherSim(num_landmarks=24, seed=cfg["seed"], backbone=_SimBackbone(),
                           prompt="plate",
                           tracker=ReleaseRoiTracker(_GoallessDetector(box, hw),
                                                     _FixedTracker(box), prompt="plate"))
    arm = arms.build_arm(cfg, intrinsics=scene.intrinsics, matcher=m)
    g = scene.goal_pose
    start = {"start_id": "t0", "cam_pos": g.position + np.array([0.01, -0.008, 0.005]),
             "cam_R_opt": g.rotation.tolist(), "cell": "mock", "tier": "mock",
             "roll_deg": 0.0, "starts_seed": 1}
    dlog = DebugLog(tmp_path / "ep", arm_id="artvs_roi_sim")
    try:
        rec = runner.run_episode(scene, arm, start, cfg, telemetry=[], debug_log=dlog)
    finally:
        dlog.close()
    assert m.grid is None and m.goal_box is None
    assert "debug_log_error" not in rec
    assert set(rec["roi_modes"]) == {"goalless"} and sum(rec["roi_modes"].values()) > 0


# ============================== D - the development cell (owner rulings, 2026-10-10)
# `artvs_roi_sim_dev`: the simulator's outline as the goal mask through the release's mask
# branch (roi_mask: objects, no SAM), and the switch at the release's error rule or at step N,
# whichever comes first (switch_max_steps). docs/dev_eval_plan_2026-10-10.md section 3.
DEV = {"roi_mask": "objects", "roi_sam_path": None}


class _SpyDetector(_PortDetector):
    """Records every `with_mask` the arm asks for: the dev cell never asks SAM for a mask."""

    def __init__(self, world):
        super().__init__(world)
        self.with_mask: list[bool] = []

    def detect(self, rgb, prompt, with_mask=True):
        self.with_mask.append(bool(with_mask))
        return super().detect(rgb, prompt, with_mask)


def test_d1_the_switch_fires_at_step_n_when_the_error_rule_has_not():
    """The error never falls (the object stays 100 px away): with switch_max_steps 5 the
    first five bench steps are coarse and the tiles latch at the start of step 6 - frame for
    frame the release's own iteration switch at 5, since every frame succeeds here."""
    world, goal, frames = _scene_pair(live_shifts=[(100, 0)] * 8)
    pair = _Pair(world, goal, switch_max_steps=5)
    pair.cfg.tiling_switch_iteration = 5          # the release, switched by its iteration rule
    modes = []
    for t, img in enumerate(frames, start=1):
        m, pts, v = pair.frame(img)
        _assert_same_frame(pair, m, pts, v)
        assert pair.port.tiled == pair.glue.roi_tiling_activated, t
        modes.append(m.extras["mode"])
    assert modes == ["coarse"] * 5 + ["tiled"] * 3
    p = pair.port
    assert (p.switch_step, p.switch_frame, p.switch_reason) == (6, 5, "step")
    assert pair.glue.roi_tiling_switch_iteration == 5
    rec = p.record_fields()
    assert (rec["roi_switch_reason"], rec["roi_switch_step"], rec["roi_switch_max_steps"]) == \
        ("step", 6, 5)


def test_d1_the_error_rule_fires_first_exactly_as_in_the_faithful_cell():
    """F6's sequence (the drop at frame 9 is too early, the one at frame 12 counts) with a
    step cap of 20: the switch is the release's, to the frame - the cap never binds."""
    shifts = [(100, 0)] * 8 + [(10, 0)] + [(100, 0)] * 2 + [(10, 0)] * 3
    world, goal, frames = _scene_pair(live_shifts=shifts)
    pair = _Pair(world, goal, switch_max_steps=20)
    modes = []
    for t, img in enumerate(frames, start=1):
        m, pts, v = pair.frame(img)
        _assert_same_frame(pair, m, pts, v)
        assert pair.port.tiled == pair.glue.roi_tiling_activated, t
        modes.append(m.extras["mode"])
    assert modes == ["coarse"] * 12 + ["tiled"] * 2
    p = pair.port
    assert (p.switch_frame, p.switch_step, p.switch_reason) == (12, 13, "error")
    assert pair.glue.roi_tiling_switch_iteration == 12


def test_d1_the_error_rule_never_fires_before_ten_frames_with_the_cap_on():
    """The error falls at frame 2 already: the rule still waits for 10 successful frames
    (the first tiled frame is 11, as in the release), and a cap later than that changes
    nothing; a cap EARLIER than that switches at the cap and says so."""
    shifts = [(100, 0)] + [(5, 0)] * 13
    world, goal, frames = _scene_pair(live_shifts=shifts)
    pair = _Pair(world, goal, switch_max_steps=50)
    for t, img in enumerate(frames, start=1):
        m, pts, v = pair.frame(img)
        _assert_same_frame(pair, m, pts, v)
        assert pair.port.tiled == pair.glue.roi_tiling_activated, t
        if t <= 10:
            assert m.extras["mode"] == "coarse", t
    p = pair.port
    assert (p.switch_frame, p.switch_step, p.switch_reason) == (10, 11, "error")
    early = _Pair(world, goal, switch_max_steps=4)
    for img in frames[:6]:
        early.port.match(img)
    q = early.port
    assert (q.switch_frame, q.switch_step, q.switch_reason) == (4, 5, "step")


def test_d1_the_cap_counts_bench_steps_and_needs_no_successful_frame():
    """The cap counts `match()` calls (bench steps), not the release's successful frames: a
    skipped frame (fewer than 4 points) still counts, and the cap latches even before any
    frame has succeeded. Off (0) it never latches - the faithful cell."""
    def port(**kw):
        return RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(),
                                  tracker=ReleaseRoiTracker(_PortDetector(_World()),
                                                            _PortTracker(_World()), prompt="x"),
                                  **kw)
    p = port(switch_max_steps=5, **DEV)
    p.frames, p.frames_ok = 5, 3                     # step 5 is being started: 4 done
    p._check_switch()
    assert not p.tiled
    p.frames = 6                                     # step 6: five steps done, two skipped
    p._check_switch()
    assert p.tiled and (p.switch_step, p.switch_frame, p.switch_reason) == (6, 3, "step")
    q = port(switch_max_steps=5)
    q.frames, q.frames_ok = 6, 0
    q._check_switch()
    assert q.tiled and q.switch_reason == "step"
    off = port()
    off.frames, off.frames_ok, off.e0, off.e_prev = 600, 600, 100.0, 90.0
    off._check_switch()
    assert not off.tiled and off.switch_max_steps == 0
    assert "roi_switch_reason" not in off.record_fields()
    with pytest.raises(ValueError, match="cannot both be set"):
        port(switch_max_steps=5, switch_iteration=3)
    with pytest.raises(ValueError, match="must be >= 0"):
        port(switch_max_steps=-1)


def _box_mask(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


def test_d2_the_true_goal_mask_through_both_phases_is_the_releases(tmp_path):
    """The simulator's outline handed to `reset()` takes the branch LangSAM's mask took: the
    coarse cells, the switch and the tiled frames equal the release fed the same mask as
    `mask_path`, to the pixel. SAM is never asked for a mask (the detector sees
    `with_mask=False` on the goal and on every live frame), and a SAM mask in the world -
    which the faithful cell would take - changes nothing."""
    true = _blob_mask(GOAL_BOX, seed=5)
    sam = np.ones((H, W), bool)                     # what SAM would answer: not used
    path = tmp_path / "goal_mask.png"
    Image.fromarray(true.astype(np.uint8) * 255).save(path)
    world, goal, frames = _scene_pair(live_shifts=[(14, 6), (15, 6), (16, 7), (16, 7)],
                                      mask=sam)
    pair = _Pair(world, goal, mask_path=str(path), switch_max_steps=2, goal_mask=true, **DEV)
    pair.cfg.tiling_switch_iteration = 2
    spy = _SpyDetector(world)
    pair.port.tracker.detector = spy                # from here on: the live frames
    p = pair.port
    want = rel_matcher.create_patch_mask(pair.glue.goal_roi_mask_path, VIT, PATCH).numpy()[0]
    assert np.array_equal(p._coarse.mask, want) and 0 < p.mask_cells["coarse"] < SIDE * SIDE
    assert p.mask_cells["full"] == int(rel_matcher.create_patch_mask(
        str(path), VIT, PATCH).numpy().sum())
    for t, img in enumerate(frames, start=1):
        m, pts, v = pair.frame(img)
        _assert_same_frame(pair, m, pts, v, mode="coarse" if t <= 2 else "tiled")
        assert len(m) > 0
    assert spy.with_mask and not any(spy.with_mask)
    assert p.tracker.goal_mask is None and p.tracker.goal_mask_wanted is False
    rec = p.record_fields()
    assert rec["roi_mask"] == "objects" and rec["roi_sam_path"] is None
    assert rec["roi_mask_share"] == pytest.approx(float(true.mean()))
    assert len(rec["roi_mask_cells"]["tiles"]) == 6
    # the faithful cell on the same world takes SAM's all-ones mask instead: other cells
    faithful = _Pair(world, goal)
    assert faithful.port.mask_cells["coarse"] == SIDE * SIDE != p.mask_cells["coarse"]


def test_d2_the_true_mask_gives_the_expected_cells_on_a_synthetic_mask():
    """The release's cell rule on the simulator's outline, by hand: on the full picture at
    256 (one cell = 120 x 67.5 frame pixels) a rectangle over cell rows 2-4 that covers
    column 2 whole, column 3 to 60 % and column 6 to 40 % counts exactly columns 2 and 3 of
    those rows - a cell counts when MORE than half of its pixels are object."""
    cw, ch = W / SIDE, H / SIDE
    y1, y2 = int(round(2 * ch)), int(round(5 * ch))
    mask = _box_mask(int(2 * cw), y1, int(3 * cw + 0.6 * cw), y2)
    mask |= _box_mask(int(6 * cw), y1, int(6 * cw + 0.4 * cw), y2)
    world = _World()
    goal = world.add(_texture(31), box=GOAL_BOX)
    p = RoiTiledMatcherSim(num_landmarks=24, seed=SEED, backbone=_SimBackbone(), prompt="x",
                           tracker=ReleaseRoiTracker(_PortDetector(world), _PortTracker(world),
                                                     prompt="x"), **DEV)
    p.set_episode(SCENE, START)
    p.reset(goal, goal_mask=mask)
    cells = p._full.mask.reshape(SIDE, SIDE)
    expect = np.zeros((SIDE, SIDE), bool)
    expect[2:5, 2:4] = True
    assert np.array_equal(cells, expect), np.argwhere(cells)
    assert p.mask_cells["full"] == 6
    # a goal without a box: the outline still masks the full-picture fallbacks (the release's
    # `mask_path`); LangSAM's mask came with its box, so the faithful cell has none there
    w2 = _World()
    g2 = w2.add(_texture(32))                        # nothing detected on the goal
    q = RoiTiledMatcherSim(num_landmarks=24, seed=SEED, backbone=_SimBackbone(), prompt="x",
                           tracker=ReleaseRoiTracker(_PortDetector(w2), _PortTracker(w2),
                                                     prompt="x"), **DEV)
    q.set_episode(SCENE, START)
    q.reset(g2, goal_mask=mask)
    assert q.goal_box is None and q.mask_cells == {"full": 6}
    assert np.array_equal(q._full.mask.reshape(SIDE, SIDE), expect)


def test_d2_the_dev_cell_needs_the_mask_and_states_no_sam():
    world = _World()
    goal = world.add(_texture(33), box=GOAL_BOX)
    m = RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(), **DEV,
                           tracker=ReleaseRoiTracker(_PortDetector(world), _PortTracker(world),
                                                     prompt="x"))
    m.set_episode(SCENE, START)
    with pytest.raises(ValueError, match="needs the bench's goal mask"):
        m.reset(goal)
    assert m.requires_goal_mask and m.honours_goal_mask
    assert RoiTiledMatcherSim.requires_goal_mask is False            # the class default
    with pytest.raises(ValueError, match="roi_sam_path is 'langsam' under roi_mask 'objects'"):
        RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(), roi_mask="objects")
    # built without an injected tracker: a GroundingSam that is never asked for SAM
    built = RoiTiledMatcherSim(num_landmarks=24, backbone=_SimBackbone(), **DEV)
    assert built.tracker.goal_mask_wanted is False and built.tracker.detector._sam is None
    assert "no SAM" in built.record_fields()["roi_detector"]


class _RaisedPlate:
    """The mock plane with a 2 cm plate standing on it in the middle of every picture - a
    goal depth whose objects-only mask (the runner's, from the goal depth) is that plate."""

    def __init__(self, scene):
        self._scene = scene

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def render(self):
        rgb, depth, extra = self._scene.render()
        d = np.array(depth, dtype=np.float64, copy=True)
        h, w = d.shape[:2]
        d[h // 4:3 * h // 4, w * 3 // 10:w * 7 // 10] -= 0.02
        return rgb, d, extra


def test_d3_the_dev_cell_flies_the_episode_loop_with_the_runners_mask(tmp_path):
    """The protocol cell `artvs_roi_sim_dev` (stub encoder, detector and tracker) through
    `runner.run_episode`: the runner hands `reset()` the goal objects' outline, the record
    carries the mask, the switch and its reason, and the label is the cell's. The check
    script's identity check resets the arm with the same mask."""
    import importlib.util

    from vsbench.vs.bench import runner
    from vsbench.vs.labels import object_mask_from_depth
    from vsbench.vs.scene import MockPlaneScene

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("_fly_for_d3", repo / "scripts" / "fly.py")
    fly = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo / "scripts"))
    spec.loader.exec_module(fly)
    scene = _RaisedPlate(MockPlaneScene(render_scale=0.25))
    hw = (scene.intrinsics.height, scene.intrinsics.width)
    box = [scene.intrinsics.width * 0.3, scene.intrinsics.height * 0.25,
           scene.intrinsics.width * 0.7, scene.intrinsics.height * 0.75]
    cfg = fly.arm_config(fly.load_protocol(), "artvs_roi_sim_dev")
    # the cell's settings, at a step cap of 12 so the short mock flight reaches the tiles
    cfg["matcher"] = {**cfg["matcher"], "device": "cpu", "prompt": "plate",
                      "switch_max_steps": 12}
    cfg["episode"] = {**cfg["episode"], "max_iter": 30}
    m = RoiTiledMatcherSim(num_landmarks=24, seed=cfg["seed"], backbone=_SimBackbone(),
                           prompt="plate", switch_max_steps=12, **DEV,
                           tracker=ReleaseRoiTracker(_FixedDetector(box, hw),
                                                     _FixedTracker(box), prompt="plate"))
    arm = arms.build_arm(cfg, intrinsics=scene.intrinsics, matcher=m)
    assert arm.goal_mask_mode == "objects"
    g = scene.goal_pose
    start = {"start_id": "t0", "cam_pos": g.position + np.array([0.01, -0.008, 0.005]),
             "cam_R_opt": g.rotation.tolist(), "cell": "mock", "tier": "mock",
             "roll_deg": 0.0, "starts_seed": 1}
    rec = runner.run_episode(scene, arm, start, cfg, telemetry=[])
    assert rec["arm"] == ("artvs_roi_sim@dinov3_vits16_artvs256/k24/objmask/switch-by-step12/"
                          "tr/uniform/ema0.8/vcap10/hold/cns_plateau")
    assert rec["goal_mask_mode"] == "objects" and 0.1 < rec["goal_mask_share"] < 0.3
    assert rec["roi_mask"] == "objects" and rec["roi_sam_path"] is None
    assert rec["roi_mask_share"] == pytest.approx(rec["goal_mask_share"])
    assert rec["roi_switch_max_steps"] == 12 and rec["roi_switch_reason"] in ("error", "step")
    assert rec["roi_switch_step"] <= 13 and rec["roi_modes"]["tiled"] >= 30 - 12
    # the identity check of scripts/check_baselines.py: the same mask, no refusal
    cb_spec = importlib.util.spec_from_file_location("_cb_for_d3",
                                                     repo / "scripts" / "check_baselines.py")
    cb = importlib.util.module_from_spec(cb_spec)
    cb_spec.loader.exec_module(cb)
    idn = cb.identity_check(scene, arm)
    assert idn["n_matched"] > 0
    scene.reset(scene.goal_pose)
    _rgb, depth, _ = scene.render()
    want = object_mask_from_depth(depth, scene.intrinsics, scene.goal_pose)
    assert m.mask_share == pytest.approx(float(want.mean()))


def test_the_transcribed_release_blocks_are_the_release_lines():
    """`roi_glue.py`'s header says every block marked "release lines a-b" is those lines of
    the release's `ros_controller.py`, byte for byte: checked here wherever the read-only
    checkout sits beside this repository (the owner's workspace); skipped elsewhere."""
    import re

    here = Path(__file__).resolve().parents[2]
    release = (here.parent / "ART-VS" / "catkin_ws" / "ibvs" / "src" / "visual_servoing"
               / "ros_interface" / "ros_controller.py")
    if not release.is_file():
        pytest.skip(f"the ART-VS checkout is not beside this repository ({release})")
    rel = release.read_text(encoding="utf-8").split("\n")
    glue = (here / "tests" / "fixtures" / "reference" / "artvs_release" / "roi_glue.py"
            ).read_text(encoding="utf-8").split("\n")
    marker = re.compile(r"^\s*# ---- release lines (\d+)-(\d+)")
    blocks = []
    for i, line in enumerate(glue):
        hit = marker.match(line)
        if hit:
            a, b = int(hit.group(1)), int(hit.group(2))
            blocks.append((a, b, glue[i + 1:i + 2 + b - a] == rel[a - 1:b]))
    assert len(blocks) >= 17, blocks
    assert all(same for *_, same in blocks), [blk for blk in blocks if not blk[2]]
