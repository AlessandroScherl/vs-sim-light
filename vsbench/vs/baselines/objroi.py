# vs-sim copy: vssim/baselines/objroi.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""ART-VS's published object pipeline: LangSAM detection, LightTrack tracking, ROI tiling.

Purpose
-------
The ART-VS release (`catkin_ws/ibvs/src/visual_servoing`, "MODE 4: LangSAM + LightTrack
+ ROI tiling") does not servo on the whole picture when it is given an object to reach:
a text prompt names the object, **LangSAM** (GroundingDINO for the box, SAM for the
mask) finds it in the goal picture once and in the first live frame, **LightTrack** then
follows the box from frame to frame (re-detecting with LangSAM when it loses it, falling
back to the last box for ``cache_timeout`` frames), both boxes are padded by
``roi_padding`` and **cropped**, the crops are resized to the ViT input square, the DINO
patches are matched inside the crops, and once the feature error has fallen to
``switch_threshold`` of its first value the crops are cut into a grid of tiles - ``ceil(box
side / vit input)`` per dimension, at most 5 - and matched tile against tile (the
published "ROI tiling"). Every point is then mapped back to full-image pixels for the
control law. This module is that pipeline in the bench's matcher shape, built on the
same parity-tested DINO matcher as the ``artvs`` arm.

What is the authors' and what is ours
--------------------------------------
- **The detector**: LangSAM is GroundingDINO ``IDEA-Research/grounding-dino-base`` with
  box threshold 0.3 / text threshold 0.25 and SAM 2.1 ``facebook/sam2.1-hiera-small``
  prompted with the best box - the exact models and thresholds of ``lang_sam.LangSAM``;
  here they run through ``transformers`` (which carries both) instead of the ``lang_sam``
  package. Highest-confidence box, as the release picks it (its spatial prior is for
  the pre-servo rotation phase, which this bench's ``artvs`` cell does not run). SAM runs
  on one of two paths (``GroundingSam(sam_path=...)``, section "The SAM path" below): this
  port's cells keep ``transformers``; the faithful arm runs ``langsam``.
- **The tracker**: the release's own ``LightTrack`` wrapper (``docker/lighttrack_wrapper.py``,
  ``LightTrackM_Subnet``) over the researchmm LightTrack checkout, with the checkpoint
  that checkout ships (``snapshot/LightTrackM/LightTrackM.pth``). ``$LIGHTTRACK_ROOT``
  names the checkout; the wrapper file sits at ``tracking/lighttrack.py`` inside it, as
  the release's Dockerfile places it.
- **The state machine** (``ObjectRoiTracker``): the release's DETECT / TRACK / CACHE
  logic (``langsam_lighttrack_detector.py``) with its padding (0.1) and cache timeout
  (10 frames). The bench has no rotation phase, so the tracker starts on the first
  live frame ("no rotation needed - initialize tracker on current frame").
- **ROI matching and tiling** (``RoiTiledMatcher``): the release's
  ``detect_features_roi`` / ``detect_features_roi_tiled`` (``ros2_controller.py``): both
  crops resized to ``vit_input_size`` squares (aspect distorted, as published), the
  goal's SAM mask cropped with it and applied as the patch mask, ``num_pairs`` pairs;
  tiled: ``gw x gh = ceil(box / vit)`` per dimension from the GOAL box, clamped to
  [1, 5], crops resized to ``(vit*gw, vit*gh)``, ``max(1, num_pairs // n_tiles)`` pairs
  per tile, tile *i* against tile *i*, points lifted through the tile offset and the
  crop scale to full-image pixels. The switch is the release's error-reduction rule
  (``roi_tiling_switch_threshold`` 0.2, measured on the coarse displacement), latched.
  When no box exists at all the release matches the full pictures; so does this.
- **Ours**: the DINO encoder and the mutual-nearest-neighbour matcher are the bench's
  parity-tested port (:class:`vsbench.vs.baselines.match.DenseMatcher`); ``vit_input_size``
  follows the arm file (224, the paper's simulation setting; the real-robot ROI config
  ships 256) and is recorded.

Fences
------
- Weights are never downloaded by a run: the two HF models must be in ``$HF_HOME``
  (``HF_HUB_OFFLINE=1``) and LightTrack's checkpoint on disk; both are recorded.
- ``goal_mask`` from the bench (the depth mask) is refused on this arm: its mask is the
  published detector's, and two masks on one cell would be unlabelled.
- Per-frame bookkeeping (which source gave the box: langsam / lighttrack / cache /
  none) is recorded, never acted on.
- **Logging-only extras (2026-09-30, the debug telemetry).** Every frame's
  ``Matches.extras`` also carries the live box the crop was cut from (``live_box``), the
  encoder that actually ran (``enc_input`` / ``enc_patch`` - the DINOv2 input the crop is
  re-encoded at, logged as it is and not changed), the switch's first and current error
  (``switch_e0`` / ``switch_e``) and, when tiled, the points per tile (``tile_counts``).
  The controller never reads ``extras`` (it takes the rows), so the pixels, the rows and
  the twist are unchanged; ``tests/fast/test_objroi.py`` checks it.

Golden tests: ``tests/fast/test_objroi.py`` (stub detector and tracker through the state
machine and the crop / tile geometry with a weight-free encoder).

The faithful arm (2026-10-05): ``RoiTiledMatcherSim``
-------------------------------------------------------
``RoiTiledMatcher`` above is the 2026-09-21 port and stays as it is: the ``artvs_roi*``
cells keep their settings and their numbers ("as ported": DINOv2 re-encoded at 518, a SAM
mask, a median switch). ``RoiTiledMatcherSim`` (variant ``artvs_roi_sim``) is ART-VS as the
release's SIMULATION program runs object mode - the owner's ruling of 2026-10-05, built
from ``docs/results/2026-10-05_artvs_release_check/sim_spec.md`` (workspace root).

Ported from the ART-VS release (``../ART-VS``, git ``9b1208c``, MIT):
``ros_interface/ros_controller.py`` - ``check_roi_tiling_switch`` (594-634),
``detect_features`` (683-748, the ROI branch), ``detect_features_original`` (750-854,
the box-less fallback), ``calculate_uv`` / ``calculate_uv_roi`` (856-917),
``set_yolo_keyword`` (919-973), the second ``detect_features_roi`` (1360-1571),
``detect_features_roi_tiled`` (1573-1982), ``_map_roi_to_full_image`` (2140-2171), the
error bookkeeping of ``ibvs`` (2481-2525), ``run``'s per-sample resets (2603-2643),
``_calculate_optimal_tile_number`` (2781-2837); ``features/langsam_lighttrack_detector.py``
- the DETECT / TRACK / CACHE machine, int boxes, padding (191-703) ->
``ReleaseRoiTracker``; and the driver's first-frame detection + tracker start
(``experiments/run_multi_model_experiment_with_hollywood.py:688-759``). The descriptors are
``vsbench.vs.baselines.features`` ``dinov3_vits16_artvs256`` (the release's extractor chain);
the pool and the draw are the bench's (``mutual_nn_match`` = ``find_correspondences_batch``,
the draw seeded per frame where the release's is unseeded).

Left behind: ROS, TF, publishers and every visualisation block; the rotation search (off
by the owner's ruling); the classical and full-image paths except the two fallbacks named
in the class docstring; the dangling first ``detect_features_roi`` (975-1140, never run);
the full-image 6 x 6 tiled fallback (1142-1358, the one stated deviation); the abort after
10 empty pools (recorded instead - owner decision D-E). What is the bench's: the camera,
the gain, the step cap and the scored stop (shared by every arm), and the seeded draw.

The development cell (2026-10-10): ``roi_mask: objects`` and ``switch_max_steps``
------------------------------------------------------------------------------
Two knobs of :class:`RoiTiledMatcherSim`, both off by default so the faithful cell
``artvs_roi_sim`` runs exactly as before (vs-sim rule 7: the knobs make a different cell,
``artvs_roi_sim_dev``, with its own label). Owner rulings of 2026-10-10
(``docs/dev_eval_plan_2026-10-10.md`` section 3, workspace root):

- **The goal mask is the simulator's pixel-perfect outline** (``roi_mask: objects`` with
  ``matcher.goal_mask: objects``): the runner's mask from the goal depth against the ground
  plane (:func:`vsbench.vs.labels.object_mask_from_depth`, the ``*_objmask`` cells' mask) enters
  the release's mask branch exactly where LangSAM's did - held as the same lossless 8-bit
  picture, then the same crops, resizes, JPEG round trip and cell rule (a cell counts when the
  mean of its pixels is above 0.5). As the release treats a ``mask_path``, the full-picture
  fallbacks take the whole mask even when the goal has no box (LangSAM's mask came with its
  box, so the faithful cell has no mask then). SAM is never loaded: the detector asks for the
  box alone, on the goal as on the live frames - the box path (GroundingDINO + LightTrack, the
  state machine, the padding) is the faithful cell's, call for call.
- **The switch to tiles fires at whichever comes first**: the release's error rule (mean
  pixel error at 0.2 of its first value, after at least 10 successful frames, read on the
  previous frame - unchanged) or ``switch_max_steps`` bench steps of the coarse phase (100 in
  the cell): at the start of step ``N + 1`` the arm latches the tiles if the rule has not. The
  count is bench steps (``match()`` calls), not the release's successful frames, so the
  coarse phase is at most N steps whatever happens; ``roi_switch_reason`` says which fired.
  The release's own iteration switch (``switch_iteration``) replaces the error rule rather
  than racing it, and is refused together with ``switch_max_steps``.

The SAM path (2026-10-06): ``GroundingSam(sam_path=...)``
---------------------------------------------------------
The LangSAM check (``docs/results/2026-10-06_langsam_check/``, workspace root) ran the real
``lang_sam`` package beside :class:`GroundingSam` on 121 pictures: the GroundingDINO boxes
are identical, the SAM masks are not, for exactly two reasons. (1) The transformers
``Sam2Model`` builds its decoder with SAM 2's stability fallback ON (the cached config's
``mask_decoder_config.dynamic_multimask_via_stability: true``, copying Meta's
``build_sam2(apply_postprocessing=True)``): when the main mask's stability score is below
0.98 it swaps in one of SAM's alternative masks - on every shoe goal, one with the
insole. ``lang_sam`` builds the model with ``compose`` + ``instantiate`` and leaves it OFF.
(2) ``Sam2Processor`` resizes the 8-bit picture and rounds it to whole grey levels;
``SAM2Transforms`` resizes the float picture.
``sam_path="langsam"`` makes both of ``lang_sam``'s choices - the loaded decoder's flag off,
SAM's picture from :func:`sam2_pixel_values` - and gives its mask bit for bit
(``tests/slow/test_langsam_masks.py``, against the four masks the package made,
``tests/fixtures/langsam_masks/``). ``sam_path="transformers"``, the default, is the code
as it was: the ``artvs_roi*`` cells keep it (rule 7: their labels and numbers do not move;
the same slow test pins their masks). Only :class:`RoiTiledMatcherSim` takes ``langsam``
(``roi_sam_path``, label token ``sam-langsam``); ViT-VS and DINOBot never run SAM.

Ported from ``lang_sam`` (lang-sam 0.2.1, git ``918043e``, Apache-2.0) and SAM 2 (SAM-2
1.0, git ``c2ec8e1``, Apache-2.0), behaviour only, no code copied:
``lang_sam/models/sam.py::SAM.build_model`` (the model built without ``build_sam2``'s
post-processing overrides, so ``sam2/modeling/sam/mask_decoder.py``'s default
``dynamic_multimask_via_stability=False`` stands) and ``sam2/utils/transforms.py::
SAM2Transforms`` with ``sam2/sam2_image_predictor.py::set_image_batch`` (``ToTensor`` ->
``Resize((1024, 1024))`` -> ``Normalize`` on the CPU, then to the device; checked equal to
``SAM2Transforms`` itself on random pictures, 0.0 difference, in the check's environment).
Left behind: the package itself - never imported (seven more packages, and on CUDA its
import switches the whole thread to bf16 autocast and TF32, the check's section 6, an open
owner decision); its one SAM call over all the boxes (the best box alone here; the check
measured 0 px between the two); the hole filling it leaves off anyway. The processor still
makes the box prompt and the sizes, and its mask post-processing stays (with the two
choices made, 0 px from the package's mask on 121/121 pictures).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from vsbench.vs.baselines.match import (
    MISSING_DEVICE,
    TIE_ORDERS,
    CorrespondenceMatcher,
    DenseMatcher,
    EpisodeDraw,
    Matches,
    device_of,
    mutual_nn_match,
)

GDINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_MODEL = "facebook/sam2.1-hiera-small"
#: LangSAM's defaults (lang_sam.LangSAM.predict).
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
#: ``GroundingSam(sam_path=...)`` - how SAM 2.1 is run on the box (2026-10-06, the LangSAM
#: check). ``transformers``: the transformers ``Sam2Model`` as its config builds it (SAM 2's
#: stability fallback on) fed the ``Sam2Processor``'s own pixels (the 8-bit picture resized) -
#: the code as it was, the default, every ``artvs_roi*`` cell's. ``langsam``: SAM as the
#: ``lang_sam`` package runs it - the fallback off and :func:`sam2_pixel_values` (the float
#: picture resized); same model, weights and box, and ``lang_sam``'s mask bit for bit.
SAM_PATHS = ("transformers", "langsam")
#: ``SAM2Transforms`` (SAM-2 ``sam2/utils/transforms.py``): the square side SAM reads and
#: ImageNet's mean and spread, the values it normalises with.
SAM2_RESOLUTION = 1024
SAM2_MEAN = (0.485, 0.456, 0.406)
SAM2_STD = (0.229, 0.224, 0.225)
#: The release's ROI settings (core/config.py, langsam_lighttrack_detector.py).
ROI_PADDING = 0.1
CACHE_TIMEOUT = 10
MAX_TILES_PER_DIM = 5
LIGHTTRACK_ARCH = "LightTrackM_Subnet"
LIGHTTRACK_ROOT_ENV = "LIGHTTRACK_ROOT"
DEFAULT_LIGHTTRACK_ROOT = "/workspace/LightTrack"

def pad_box(box, image_wh, padding: float) -> list[float]:
    """The release's ``_apply_padding``: grow ``[x1, y1, x2, y2]`` by ``padding`` of its
    own size on each side, clipped to the image."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = image_wh
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0.0, x1 - px), max(0.0, y1 - py), min(float(w), x2 + px), min(float(h), y2 + py)]


def crop(rgb: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    return np.ascontiguousarray(np.asarray(rgb)[y1:y2, x1:x2])


def resize(rgb: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(np.asarray(rgb)).resize((int(wh[0]), int(wh[1]))))


# ------------------------------------------------------------------ LangSAM
def sam2_pixel_values(rgb: np.ndarray):
    """The picture SAM reads on the ``langsam`` path: ``SAM2Transforms`` (SAM-2 ``c2ec8e1``,
    ``sam2/utils/transforms.py``), which ``lang_sam``'s predictor applies on the CPU before it
    moves the picture to the device (``sam2_image_predictor.py``, ``set_image_batch``).

    torchvision's ``Normalize(SAM2_MEAN, SAM2_STD)(Resize((1024, 1024))(ToTensor()(rgb)))[None]``
    written in torch alone (the package does not import torchvision) and checked equal to it
    bit for bit (``tests/fast/test_objroi.py``): ``ToTensor`` - the 8-bit picture to float32,
    channels first, ``/ 255``; ``Resize`` on a float tensor - bilinear, ``align_corners=False``,
    ``antialias=True`` (torchvision's tensor default since 0.17), skipped when the picture is
    already 1024 x 1024 as torchvision skips it; ``Normalize`` - ``(x - mean) / std`` in
    float32. The transformers processor instead resizes the 8-bit picture and rounds it to
    whole grey levels before it normalises - at most one grey level apart, enough to move
    the outline by a pixel. ``[1, 3, 1024, 1024]`` float32 on the CPU."""
    import torch
    import torch.nn.functional as F

    a = np.array(np.asarray(rgb, np.uint8)[..., :3], copy=True)       # writable, contiguous
    x = torch.from_numpy(a.transpose(2, 0, 1)).contiguous().to(torch.float32).div(255)
    if tuple(x.shape[-2:]) != (SAM2_RESOLUTION, SAM2_RESOLUTION):
        x = F.interpolate(x[None], size=[SAM2_RESOLUTION, SAM2_RESOLUTION], mode="bilinear",
                          align_corners=False, antialias=True)[0]
    mean = torch.as_tensor(SAM2_MEAN, dtype=torch.float32).view(-1, 1, 1)
    std = torch.as_tensor(SAM2_STD, dtype=torch.float32).view(-1, 1, 1)
    return x.sub(mean).div(std)[None]


class GroundingSam:
    """GroundingDINO (box) + SAM 2.1 (mask) for one text prompt - LangSAM's two steps.

    ``sam_path`` (:data:`SAM_PATHS`) says how SAM is run; the box step is the same on both.
    ``transformers`` (the default) is the code as it was. ``langsam`` makes the two choices
    the ``lang_sam`` package makes (the LangSAM check, 2026-10-06): SAM 2's stability fallback
    off - ``lang_sam`` builds the model with hydra's ``compose`` + ``instantiate``
    (``lang_sam/models/sam.py``), never through ``build_sam2(apply_postprocessing=True)``,
    whose override turns it on and whose choice the transformers config copies - and the
    picture from :func:`sam2_pixel_values`; the processor still makes the box prompt and the
    sizes. With both, the mask equals ``lang_sam``'s bit for bit
    (``tests/slow/test_langsam_masks.py``).
    """

    def __init__(self, *, device: str, box_threshold: float = BOX_THRESHOLD,
                 text_threshold: float = TEXT_THRESHOLD, gdino: str = GDINO_MODEL,
                 sam: str = SAM_MODEL, sam_path: str = "transformers") -> None:
        if sam_path not in SAM_PATHS:
            raise ValueError(f"GroundingSam: sam_path is {sam_path!r}, not one of "
                             f"{', '.join(SAM_PATHS)}")
        self.device = str(device)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.gdino_name, self.sam_name = gdino, sam
        self.sam_path = str(sam_path)
        self._gd = self._proc = self._sam = self._sam_proc = None

    def _build(self, with_mask: bool = True):
        """Load GroundingDINO, and SAM only when a mask is asked for. The simulation arm
        asks for SAM's mask on the goal only (owner ruling 2026-10-05: LangSAM's goal
        mask) and for the box alone on live frames; the old cell asks for masks and so
        loads both, as before. Loading order has no numeric effect. On the ``langsam`` path
        the loaded decoder's stability flag - the attribute its ``forward`` reads - is set
        off, after checking that the decoder has one (a decoder without it would silently
        grow a new attribute that nothing reads)."""
        if self._gd is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._proc = AutoProcessor.from_pretrained(self.gdino_name)
            self._gd = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.gdino_name).to(self.device).eval()
        if with_mask and self._sam is None:
            from transformers import Sam2Model, Sam2Processor

            self._sam_proc = Sam2Processor.from_pretrained(self.sam_name)
            sam = Sam2Model.from_pretrained(self.sam_name).to(self.device).eval()
            if self.sam_path == "langsam":
                dec = sam.mask_decoder
                if not hasattr(dec, "dynamic_multimask_via_stability"):
                    raise RuntimeError(
                        f"GroundingSam(sam_path='langsam'): {type(dec).__name__} has no "
                        "`dynamic_multimask_via_stability` - this transformers version cannot "
                        "be switched to lang_sam's SAM; the model is not kept")
                dec.dynamic_multimask_via_stability = False          # lang_sam's value
                sam.config.mask_decoder_config.dynamic_multimask_via_stability = False
                if sam.mask_decoder.dynamic_multimask_via_stability is not False:
                    raise RuntimeError("GroundingSam(sam_path='langsam'): the stability "
                                       "fallback did not reach the loaded decoder")
            self._sam = sam

    def detect(self, rgb: np.ndarray, prompt: str, with_mask: bool = True) -> dict:
        """``{box: [x1,y1,x2,y2] | None, score, mask: bool[H,W] | None, n_boxes}`` -
        the highest-scoring box for ``prompt`` (a period-terminated phrase, as
        GroundingDINO wants it) and SAM's mask for it. ``with_mask=False`` skips SAM
        (``mask`` is then ``None``). On live frames the release's controller only checks
        that LangSAM's mask is non-empty (``langsam_lighttrack_detector.py:385-391``), so
        the simulation arm asks for the box alone there; on the goal it asks for the mask,
        which steers the matching through the release's mask branch (owner ruling
        2026-10-05: LangSAM's goal mask). On the ``langsam`` path SAM reads
        :func:`sam2_pixel_values` of the same picture in place of the processor's."""
        import torch
        from PIL import Image

        self._build(with_mask)
        arr = np.asarray(rgb, np.uint8)[..., :3]
        img = Image.fromarray(arr)
        text = prompt if prompt.strip().endswith(".") else prompt.strip() + "."
        inputs = self._proc(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._gd(**inputs)
        res = self._proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[img.size[::-1]])[0]
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()
        if len(boxes) == 0:
            return {"box": None, "score": 0.0, "mask": None, "n_boxes": 0}
        i = int(scores.argmax())
        box = [float(v) for v in boxes[i]]
        if not with_mask:
            return {"box": box, "score": float(scores[i]), "mask": None,
                    "n_boxes": int(len(boxes))}
        sam_in = self._sam_proc(images=img, input_boxes=[[box]], return_tensors="pt")
        if self.sam_path == "langsam":
            # SAM2Transforms' picture, made on the CPU as lang_sam makes it; the processor's
            # box prompt and sizes are kept
            sam_in["pixel_values"] = sam2_pixel_values(arr)
        sam_in = sam_in.to(self.device)
        with torch.no_grad():
            so = self._sam(**sam_in, multimask_output=False)
        masks = self._sam_proc.post_process_masks(so.pred_masks.cpu(),
                                                  sam_in["original_sizes"])[0]
        mask = masks[0, 0].numpy() > 0
        return {"box": box, "score": float(scores[i]), "mask": mask, "n_boxes": int(len(boxes))}


# ---------------------------------------------------------------- LightTrack
class LightTrackWrap:
    """The release's LightTrack wrapper, imported from the LightTrack checkout.

    ``int_boxes=True`` (the simulation arm, 2026-10-05) casts the boxes as the release's
    detector does: the init box's corners with ``int()`` before ``xywh``
    (``_xyxy_to_xywh``, ``langsam_lighttrack_detector.py:691-697``) and the update's
    ``xywh`` with ``int()`` before ``xyxy`` (``_xywh_to_xyxy``, ``:699-703``; the wrapper
    already returns ints, so the second cast is a no-op kept for the record). The default
    ``False`` is the old cell's float boxes, unchanged.
    """

    def __init__(self, *, root: str | None = None, weights: str | None = None,
                 int_boxes: bool = False) -> None:
        self.root = str(root or os.environ.get(LIGHTTRACK_ROOT_ENV, DEFAULT_LIGHTTRACK_ROOT))
        self.weights = str(weights
                           or Path(self.root) / "snapshot" / "LightTrackM" / "LightTrackM.pth")
        self.int_boxes = bool(int_boxes)
        self._lt = None

    def available(self) -> tuple[bool, str]:
        if not (Path(self.root) / "tracking" / "lighttrack.py").exists():
            return False, (f"no LightTrack wrapper at {self.root}/tracking/lighttrack.py "
                           f"(the release's docker/lighttrack_wrapper.py); "
                           f"set ${LIGHTTRACK_ROOT_ENV}")
        if not Path(self.weights).exists():
            return False, f"LightTrack weights not found: {self.weights}"
        return True, ""

    def init(self, rgb: np.ndarray, box) -> None:
        import cv2

        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        from tracking.lighttrack import LightTrack  # the release's wrapper

        if self.int_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]           # _xyxy_to_xywh
        else:
            x1, y1, x2, y2 = [float(v) for v in box]
        self._lt = LightTrack(weights=self.weights, arch=LIGHTTRACK_ARCH)
        bgr = cv2.cvtColor(np.asarray(rgb, np.uint8)[..., :3], cv2.COLOR_RGB2BGR)
        self._lt.init(bgr, (x1, y1, x2 - x1, y2 - y1))

    def update(self, rgb: np.ndarray):
        import cv2

        if self._lt is None:
            return None
        bgr = cv2.cvtColor(np.asarray(rgb, np.uint8)[..., :3], cv2.COLOR_RGB2BGR)
        ok, xywh = self._lt.update(bgr)
        if not ok or xywh is None:
            return None
        if self.int_boxes:
            x, y, w, h = [int(v) for v in xywh]              # _xywh_to_xyxy
        else:
            x, y, w, h = [float(v) for v in xywh]
        return [x, y, x + w, y + h]


# ------------------------------------------------------------ the state machine
class ObjectRoiTracker:
    """The release's DETECT / TRACK / CACHE logic around a detector and a tracker.

    ``detector.detect(rgb, prompt) -> {box, score, mask}``; ``tracker.init(rgb, box)``,
    ``tracker.update(rgb) -> box | None`` (both injectable: the fast tier's stubs).
    """

    def __init__(self, detector, tracker, *, prompt: str, padding: float = ROI_PADDING,
                 cache_timeout: int = CACHE_TIMEOUT) -> None:
        self.detector, self.tracker = detector, tracker
        self.prompt = str(prompt)
        self.padding = float(padding)
        self.cache_timeout = int(cache_timeout)
        self.reset()

    def reset(self) -> None:
        self.state = "DETECT"
        self.cached_box = None
        self.cache_counter = 0
        self.counts = {"langsam": 0, "lighttrack": 0, "cache": 0, "none": 0}
        self.goal_box = self.goal_mask = None
        self.goal_score = 0.0

    def goal(self, rgb: np.ndarray, image_wh) -> dict:
        """LangSAM on the goal picture, once: the padded box and the mask (or none)."""
        d = self.detector.detect(rgb, self.prompt)
        self.goal_score = float(d.get("score", 0.0))
        self.goal_mask = d.get("mask")
        self.goal_box = None if d["box"] is None else pad_box(d["box"], image_wh, self.padding)
        return {"box": self.goal_box, "score": self.goal_score, "n_boxes": d.get("n_boxes", 0)}

    def _detect_and_track(self, rgb, image_wh):
        d = self.detector.detect(rgb, self.prompt)
        if d["box"] is None:
            return None
        self.counts["langsam"] += 1
        try:
            self.tracker.init(rgb, d["box"])
            self.state = "TRACK"
        except Exception:  # noqa: BLE001 - the release stays in DETECT when the tracker fails
            self.state = "DETECT"
        self.cached_box = list(d["box"])
        return pad_box(d["box"], image_wh, self.padding)

    def live(self, rgb: np.ndarray, image_wh) -> tuple[list[float] | None, str]:
        """``(padded box | None, source)`` for one live frame."""
        if self.state == "TRACK":
            box = self.tracker.update(rgb)
            if box is not None:
                self.counts["lighttrack"] += 1
                self.cached_box = list(box)
                return pad_box(box, image_wh, self.padding), "lighttrack"
            # lost: re-detect immediately, else cache
            padded = self._detect_and_track(rgb, image_wh)
            if padded is not None:
                return padded, "langsam"
            self.state, self.cache_counter = "CACHE", 0
            return self._cached(image_wh)
        if self.state == "CACHE":
            self.cache_counter += 1
            if self.cache_counter >= self.cache_timeout:
                padded = self._detect_and_track(rgb, image_wh)
                if padded is not None:
                    return padded, "langsam"
                self.cache_counter = 0
            return self._cached(image_wh)
        padded = self._detect_and_track(rgb, image_wh)       # DETECT
        if padded is not None:
            return padded, "langsam"
        return self._cached(image_wh)

    def _cached(self, image_wh):
        if self.cached_box is not None:
            self.counts["cache"] += 1
            return pad_box(self.cached_box, image_wh, self.padding), "cache"
        self.counts["none"] += 1
        return None, "none"


# --------------------------------------------------------- the ROI matcher
class RoiTiledMatcher(CorrespondenceMatcher):
    """ART-VS ROI mode: LangSAM + LightTrack boxes, crop matching, ROI tiling."""

    name = "artvs_roi"
    provides_margin = False
    provides_conf_weight = False
    honours_goal_mask = False        # its mask is the published detector's

    def __init__(self, *, num_landmarks: int, seed: int = 0, backbone=None,
                 backbone_name: str = "dinov2_vits14", device: str | None = None,
                 tie_order: str = "stable", threshold: float = 1.0, vit_input_size: int = 224,
                 switch_threshold: float = 0.2, switch_iteration: int = 0, prompt: str = "shoe",
                 roi_padding: float = ROI_PADDING, cache_timeout: int = CACHE_TIMEOUT,
                 box_threshold: float = BOX_THRESHOLD, text_threshold: float = TEXT_THRESHOLD,
                 max_tiles_per_dim: int = MAX_TILES_PER_DIM, use_sam_mask: bool = True,
                 lighttrack_root: str | None = None, lighttrack_weights: str | None = None,
                 tracker: ObjectRoiTracker | None = None) -> None:
        self.num_landmarks = int(num_landmarks)
        self.seed = int(seed)
        self.tie_order = str(tie_order)
        self.threshold = float(threshold)
        self.vit = int(vit_input_size)
        self.switch_threshold = float(switch_threshold)
        self.switch_iteration = int(switch_iteration)
        self.prompt = str(prompt)
        self.max_tiles = int(max_tiles_per_dim)
        self.use_sam_mask = bool(use_sam_mask)
        self.device = str(device) if device is not None else None
        # the coarse matcher owns (or builds) the encoder; the tile matchers share it
        self.coarse = DenseMatcher(num_landmarks=self.num_landmarks, seed=self.seed,
                                   backbone=backbone, backbone_name=backbone_name,
                                   device=device, threshold=threshold, tie_order=tie_order)
        self.backbone = self.coarse.backbone
        self.full = DenseMatcher(num_landmarks=self.num_landmarks, seed=self.seed,
                                 backbone=self.backbone, threshold=threshold, tie_order=tie_order)
        self.tracker = tracker or ObjectRoiTracker(
            GroundingSam(device=self.device or "cuda", box_threshold=box_threshold,
                         text_threshold=text_threshold),
            LightTrackWrap(root=lighttrack_root, weights=lighttrack_weights),
            prompt=prompt, padding=roi_padding, cache_timeout=cache_timeout)
        self._episode = ("", 0)
        self._goal_rgb = None
        self.image_wh: tuple[int, int] | None = None
        self.goal_box = None
        self.grid = (1, 1)
        self._tile_matchers: list[DenseMatcher] = []
        self.tiled = False
        self._e0 = None
        self._last_e = None              # logging only: the switch's last reading
        self._frames = 0
        self.sources: dict[str, int] = {}

    # ---------------------------------------------------------------- episode
    def set_episode(self, scene: str, start_id: Any) -> None:
        self._episode = (str(scene), start_id)
        self.coarse.set_episode(scene, start_id)
        self.full.set_episode(scene, start_id)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        if goal_mask is not None:
            raise ValueError("artvs_roi carries the published detector's mask; the bench's "
                             "goal_mask is refused so a cell never has two masks")
        self._goal_rgb = np.ascontiguousarray(np.asarray(goal_rgb, np.uint8)[..., :3])
        h, w = self._goal_rgb.shape[:2]
        self.image_wh = (int(w), int(h))
        self.tracker.reset()
        self._frames, self._e0, self.tiled = 0, None, False
        self._last_e = None
        self.sources = {}
        self._tile_matchers = []
        g = self.tracker.goal(self._goal_rgb, self.image_wh)
        self.goal_box = g["box"]
        self.full.reset(self._goal_rgb)
        if self.goal_box is None:
            return
        gx1, gy1, gx2, gy2 = self.goal_box
        gw, gh = gx2 - gx1, gy2 - gy1
        self.grid = (max(1, min(self.max_tiles, math.ceil(gw / self.vit))),
                     max(1, min(self.max_tiles, math.ceil(gh / self.vit))))
        goal_crop = crop(self._goal_rgb, self.goal_box)
        mask_crop = None
        if self.use_sam_mask and self.tracker.goal_mask is not None:
            mask_crop = crop(self.tracker.goal_mask.astype(np.uint8) * 255, self.goal_box)
        self.coarse.reset(resize(goal_crop, (self.vit, self.vit)),
                          goal_mask=None if mask_crop is None
                          else resize(mask_crop, (self.vit, self.vit)) > 127)
        # the tile matchers: goal tiles from the (vit*gw, vit*gh) resize, cached once
        tw, th = self.vit * self.grid[0], self.vit * self.grid[1]
        big = resize(goal_crop, (tw, th))
        big_mask = None if mask_crop is None else resize(mask_crop, (tw, th)) > 127
        per_tile = max(1, self.num_landmarks // (self.grid[0] * self.grid[1]))
        for i in range(self.grid[1]):
            for j in range(self.grid[0]):
                dm = DenseMatcher(num_landmarks=per_tile, seed=self.seed, backbone=self.backbone,
                                  threshold=self.threshold, tie_order=self.tie_order)
                dm.set_episode(*self._episode)
                ys = slice(i * self.vit, (i + 1) * self.vit)
                xs = slice(j * self.vit, (j + 1) * self.vit)
                tm = None if big_mask is None else big_mask[ys, xs]
                dm.reset(np.ascontiguousarray(big[ys, xs]), goal_mask=tm)
                self._tile_matchers.append(dm)

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _lift(pts_vit: np.ndarray, box, total_wh) -> np.ndarray:
        """Resized-crop pixels -> full-image pixels through the crop's own scale."""
        x1, y1, x2, y2 = box
        sx, sy = (x2 - x1) / float(total_wh[0]), (y2 - y1) / float(total_wh[1])
        out = np.asarray(pts_vit, np.float64).copy()
        out[:, 0] = x1 + out[:, 0] * sx
        out[:, 1] = y1 + out[:, 1] * sy
        return out

    def _pack(self, pts_goal, pts_cur, m: Matches, pool: int, mode: str, source: str,
              live_box=None) -> Matches:
        w, h = self.image_wh
        k = len(pts_goal)
        out = Matches.from_pixel_pairs(
            pts_goal, pts_cur, side=w, input_size=w, patch_size=1, image_wh=(w, h),
            sim=np.asarray(m.sim, np.float64)[:k] if k else np.zeros(0),
            cyclic=np.asarray(m.cyclic, np.float64)[:k] if k else np.zeros(0),
            margin=np.asarray(m.margin, np.float64)[:k] if k else np.zeros(0),
            pool_size=int(pool), num_landmarks=self.num_landmarks, draw_seed=self.seed,
            tie_order=self.tie_order)
        out.extras.update(matcher=self.name, mode=mode, roi_source=source,
                          goal_box=self.goal_box, tiled=self.tiled, grid=list(self.grid))
        out.extras.update(self._log_extras(m, live_box))
        return out

    def _log_extras(self, m: Matches, live_box, tile_counts=None) -> dict:
        """Logging only (never read by the controller): the live box, the encoder that
        ran, the switch's two errors, the points per tile."""
        out = {"live_box": None if live_box is None else [float(v) for v in live_box],
               "enc_input": getattr(m, "input_size", None),
               "enc_patch": getattr(m, "patch_size", None),
               "switch_e0": self._e0, "switch_e": self._last_e}
        if tile_counts is not None:
            out["tile_counts"] = [int(c) for c in tile_counts]
        return out

    @staticmethod
    def _error_px(pts_goal, pts_cur) -> float:
        if len(pts_goal) == 0:
            return float("nan")
        return float(np.median(np.linalg.norm(np.asarray(pts_cur) - np.asarray(pts_goal), axis=1)))

    # ---------------------------------------------------------------- frame
    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        if current_mask is not None:
            raise ValueError("artvs_roi does not take a per-frame current_mask")
        if self._goal_rgb is None:
            raise RuntimeError("reset() must be called before match()")
        self._frames += 1
        cur = np.ascontiguousarray(np.asarray(current_rgb, np.uint8)[..., :3])
        box, source = (self.tracker.live(cur, self.image_wh) if self.goal_box is not None
                       else (None, "none"))
        self.sources[source] = self.sources.get(source, 0) + 1
        if box is None or self.goal_box is None:
            # the release: "no ROI detected and no cached ROI available, using full image"
            m = self.full.match(cur)
            rows = m.to_rows()
            return self._pack(rows["pts_goal"], rows["pts_cur"], m, m.pool_size, "full", source,
                              live_box=None)

        # coarse: both crops to the vit square
        cur_crop = crop(cur, box)
        m = self.coarse.match(resize(cur_crop, (self.vit, self.vit)))
        rows = m.to_rows()
        pg = self._lift(rows["pts_goal"], self.goal_box, (self.vit, self.vit))
        pc = self._lift(rows["pts_cur"], box, (self.vit, self.vit))
        self._update_switch(self._error_px(pg, pc))
        if not self.tiled or not self._tile_matchers:
            return self._pack(pg, pc, m, m.pool_size, "coarse", source, live_box=box)

        # tiled: both crops to (vit*gw, vit*gh), tile i against tile i
        tw, th = self.vit * self.grid[0], self.vit * self.grid[1]
        big = resize(cur_crop, (tw, th))
        goal_pts, cur_pts, sims, cycs, margs, pool = [], [], [], [], [], 0
        tile_counts: list[int] = []        # logging only: points per tile
        k = 0
        for i in range(self.grid[1]):
            for j in range(self.grid[0]):
                dm = self._tile_matchers[k]
                k += 1
                ys = slice(i * self.vit, (i + 1) * self.vit)
                xs = slice(j * self.vit, (j + 1) * self.vit)
                tm = dm.match(np.ascontiguousarray(big[ys, xs]))
                tile_counts.append(len(tm))
                if not len(tm):
                    continue
                r = tm.to_rows()
                off = np.array([j * self.vit, i * self.vit], np.float64)
                goal_pts.append(r["pts_goal"] + off)
                cur_pts.append(r["pts_cur"] + off)
                sims.append(np.asarray(tm.sim))
                cycs.append(np.asarray(tm.cyclic))
                margs.append(np.asarray(tm.margin))
                pool += int(tm.pool_size)
        if not goal_pts:                       # fail open to coarse, as the artvs arm does
            return self._pack(pg, pc, m, m.pool_size, "coarse_fallback", source, live_box=box)
        gp = self._lift(np.vstack(goal_pts), self.goal_box, (tw, th))
        cp = self._lift(np.vstack(cur_pts), box, (tw, th))
        w, h = self.image_wh
        out = Matches.from_pixel_pairs(
            gp, cp, side=w, input_size=w, patch_size=1, image_wh=(w, h),
            sim=np.concatenate(sims).astype(np.float64),
            cyclic=np.concatenate(cycs).astype(np.float64),
            margin=np.concatenate(margs).astype(np.float64), pool_size=pool,
            num_landmarks=self.num_landmarks, draw_seed=self.seed, tie_order=self.tie_order)
        out.extras.update(matcher=self.name, mode="tiled", roi_source=source,
                          goal_box=self.goal_box, tiled=True, grid=list(self.grid))
        out.extras.update(self._log_extras(m, box, tile_counts))
        return out

    def _update_switch(self, e: float) -> None:
        self._last_e = e                   # logging only (read by nothing else)
        if self.tiled:
            return
        if self.switch_iteration > 0:
            self.tiled = self._frames >= self.switch_iteration
            return
        if not np.isfinite(e):
            return
        if self._e0 is None:
            self._e0 = e
            return
        if self._e0 > 0 and e / self._e0 < self.switch_threshold:
            self.tiled = True

    # ---------------------------------------------------------------- record
    def record_fields(self) -> dict:
        return {"front_end": self.name, "matcher_device": self.device,
                "roi_prompt": self.prompt, "roi_padding": self.tracker.padding,
                "roi_cache_timeout": self.tracker.cache_timeout,
                "roi_vit_input_size": self.vit, "roi_switch_threshold": self.switch_threshold,
                "roi_max_tiles_per_dim": self.max_tiles, "roi_use_sam_mask": self.use_sam_mask,
                "roi_goal_box": self.goal_box, "roi_goal_score": self.tracker.goal_score,
                "roi_grid": list(self.grid), "roi_switched": self.tiled,
                "roi_sources": dict(self.sources), "roi_detector": GDINO_MODEL + " + " + SAM_MODEL,
                "roi_tracker": LIGHTTRACK_ARCH,
                "prealign": getattr(getattr(self, "prealign", None), "label", None)}

    def close(self) -> None:
        self.coarse.close()
        self._tile_matchers = []


# =====================================================================================
# The faithful arm (2026-10-05): the release's SIMULATION object mode
# =====================================================================================
# Spec: docs/results/2026-10-05_artvs_release_check/sim_spec.md (workspace root), sections
# 1-4. Reference: the release's `ros_interface/ros_controller.py` (ROS1/Gazebo) with
# `configs/hollywood_500_evaluation/config_dinov3_256_6x6_tiling.yaml` and the code defaults
# for the ROI keys it does not set; the release's detector `langsam_lighttrack_detector.py`;
# the driver `experiments/run_multi_model_experiment_with_hollywood.py`. Line numbers below
# are the release's (git 9b1208c). The fast tier runs this class against a transcription of
# that code (tests/fixtures/reference/artvs_release/, tests/fast/test_objroi_sim.py).

#: ``ibvs()`` skips a frame with fewer points than this (``ros_controller.py:2486``) and
#: ``calculate_uv_roi`` returns nothing below it (``:906-908``).
MIN_POINTS = 4
#: ``feature_failure_count >= 10`` aborts the sample (``ros_controller.py:1446``).
FAILURE_ABORT = 10
#: the release's switch minimum, ``iteration_count >= 10`` (``ros_controller.py:619``).
SWITCH_MIN_FRAMES = 10


def lift32(pts_vit, box, total_wh) -> np.ndarray:
    """Encoder-input pixels -> full-image pixels through a box, in **float32**.

    ``_map_roi_to_full_image`` (``ros_controller.py:2155-2169``) and its tiled twin
    (``:1855-1883``): ``x1 + p * (box_w / total_w)`` on a float32 torch tensor - the scale is
    a Python float rounded to float32 by the multiply, the box corner by the add, each op
    rounded once. The release truncates these float32 numbers to whole pixels, so a float64
    lift would cross an integer boundary now and then and move a point by a whole pixel;
    numpy float32 does the same two rounded operations and lands on the same numbers.
    """
    p = np.asarray(pts_vit, np.float32).reshape(-1, 2)
    x1, y1, x2, y2 = (float(v) for v in box)
    sx = np.float32((x2 - x1) / float(total_wh[0]))
    sy = np.float32((y2 - y1) / float(total_wh[1]))
    out = np.empty_like(p)
    out[:, 0] = np.float32(x1) + p[:, 0] * sx
    out[:, 1] = np.float32(y1) + p[:, 1] * sy
    return out


def uv_roi(goal_px, cur_px, image_wh) -> tuple[np.ndarray, np.ndarray]:
    """``calculate_uv_roi`` (``ros_controller.py:892-917``): ``astype(int)`` - truncation -
    on both sides, nothing below :data:`MIN_POINTS`, then the clip to the frame."""
    g = np.asarray(goal_px, np.float64).reshape(-1, 2).astype(int)
    c = np.asarray(cur_px, np.float64).reshape(-1, 2).astype(int)
    if len(g) < MIN_POINTS:
        return np.zeros((0, 2), int), np.zeros((0, 2), int)
    w, h = int(image_wh[0]), int(image_wh[1])
    for a in (g, c):
        a[:, 0] = np.clip(a[:, 0], 0, w - 1)
        a[:, 1] = np.clip(a[:, 1], 0, h - 1)
    return g, c


def uv_pad(goal_px, cur_px, num_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    """The rows ``calculate_uv`` (``ros_controller.py:856-890``) hands the law, before its
    rounding: it allocates ``num_pairs`` zero rows (``:867-868``) and fills only the pairs it
    has (``:884-888``). So a short draw comes back padded with ``(0, 0) -> (0, 0)`` rows, and
    one of fewer than :data:`MIN_POINTS` pairs comes back as ``num_pairs`` zero rows - which
    ``ibvs`` does NOT skip (it checks ``len(s_uv_star) < 4``, ``:2486``): the frame commands
    the law's zero error (through the EMA) and reports a zero feature error. Replicated, not
    "fixed" - it is what the release does on its box-less fallback."""
    g = np.asarray(goal_px, np.float64).reshape(-1, 2)
    c = np.asarray(cur_px, np.float64).reshape(-1, 2)
    if len(g) == int(num_pairs):
        return g, c
    out_g, out_c = np.zeros((int(num_pairs), 2)), np.zeros((int(num_pairs), 2))
    if len(g) >= MIN_POINTS:
        out_g[:len(g)], out_c[:len(c)] = g, c
    return out_g, out_c


def uv_round(goal_px, cur_px) -> tuple[np.ndarray, np.ndarray]:
    """``calculate_uv``'s ``round()`` (half to even, as ``np.round``), no clip, on rows
    :func:`uv_pad` already shaped."""
    return (np.round(np.asarray(goal_px, np.float64).reshape(-1, 2)),
            np.round(np.asarray(cur_px, np.float64).reshape(-1, 2)))


#: ``roi_mask`` values of the simulation arm. ``langsam`` - LangSAM's mask of the goal
#: (owner ruling D-B (b), 2026-10-05) through the simulation controller's own mask branch: the
#: faithful cell ``artvs_roi_sim``. ``objects`` - the simulator's pixel-perfect outline of the
#: goal object, handed in by the bench (``matcher.goal_mask: objects``, the same mask the
#: ``*_objmask`` cells of ViT-VS and DINOBot take), through the SAME mask branch, no SAM run:
#: the development cell ``artvs_roi_sim_dev`` (owner 2026-10-10, "pixel perfect mask for all
#: evaluation and development (!) no sam needed").
ROI_MASKS = ("langsam", "objects")
#: ``roi_sam_path`` values of the simulation arm (2026-10-06, the LangSAM check): ``langsam``
#: - SAM run as the ``lang_sam`` package runs it (:data:`SAM_PATHS`), so the goal mask is the
#: release's own bit for bit. The ``transformers`` path is the old cells' and is refused here.
ROI_SAM_PATHS = ("langsam",)


def release_patch_mask(mask_l, vit: int, patch: int) -> np.ndarray:
    """``create_patch_mask_from_pil`` (``matcher.py:53-88``) = ``create_patch_mask``
    (``:11-50``) after its ``Image.open(path).convert('L')``: a LANCZOS resize to the encoder
    square (a same-size resize is Pillow's copy), ``> 128``, and a cell counts when the MEAN
    of its pixels is above 0.5 - more than 128 of its 256. Bool ``[side * side]``, row-major
    over goal cells, the order the matcher indexes. The loops of the release, vectorised:
    the mean of a cell's booleans is its count / 256 either way."""
    from PIL import Image

    m = np.array(mask_l.resize((int(vit), int(vit)), Image.LANCZOS)) > 128
    side = int(vit) // int(patch)
    cells = m.reshape(side, int(patch), side, int(patch)).mean(axis=(1, 3))
    return (cells > 0.5).reshape(-1)


def jpeg_round_trip(img_l):
    """The release writes the coarse ROI mask to a temporary JPEG and reads it back
    (``_cache_goal_roi_features``, ``ros_controller.py:395-400``: ``save(tmp.name, 'JPEG')``
    with Pillow's defaults - quality 75 -, then ``Image.open(...).convert('L')`` in
    ``create_patch_mask``, ``matcher.py:25``). Done in memory here with the same library and
    the same defaults; the bytes the encoder writes do not depend on where they go."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    img_l.save(buf, "JPEG")
    buf.seek(0)
    return Image.open(buf).convert("L")


class ReleaseRoiTracker:
    """The release's detector state machine and the controller's box cache, line for line.

    ``LangSAMLightTrackDetector`` (``langsam_lighttrack_detector.py``) as the simulation
    program drives it, plus the two pieces of ``ROSVisualServoingController`` that decide
    which box a frame uses (``ros_controller.py:1376-1389``). It differs from
    :class:`ObjectRoiTracker` (the old cell's, kept as it is) in six places, each the
    release's:

    1. **the first frame** - the driver runs ONE LangSAM detection, starts LightTrack with
       the raw box, and the loop's first detection is then LightTrack's ``update`` on that
       frame (``run_multi_model_experiment_with_hollywood.py:741-755``; spec 1.1 g);
    2. **int boxes** - the init box's corners are cast with ``int()`` before ``xywh`` and
       the update's ``xywh`` before ``xyxy`` (``:691-703``); the cache keeps the float box
       LangSAM gave (``:273``);
    3. **the lost frame is the first cache frame** - tracking lost and LangSAM empty ->
       CACHE with the counter at 0, then ``_use_cached_bbox`` counts it to 1 at once
       (``:549-553, 582``), so re-detection comes nine frames later, not ten;
    4. **a failed timeout re-detection does not reset the counter** (``:584-600``): every
       later CACHE frame re-detects until LangSAM finds the object;
    5. **DETECT never starts the tracker by itself** (``:337-426``): only the driver, a lost
       track and a cache timeout call ``start_tracking``; a frame in DETECT is LangSAM's;
    6. **the controller keeps its own cache** of the last padded box (``current_roi_bbox``,
       reset at the start of ``run()``, ``:2642-2643``) and uses it when the detector
       returns nothing; with neither, the frame has no box (the full-picture fallback).

    ``detector.detect(rgb, prompt, with_mask=False) -> {box, score, n_boxes}`` and
    ``tracker.init(rgb, box)`` / ``tracker.update(rgb) -> [x1, y1, x2, y2] | None`` are
    injectable (the fast tier's stubs). One deliberate shortcut: when the first frame's
    detection finds nothing (or the tracker fails to start), the release's loop runs LangSAM
    a second time on the same frame; it is the same picture and the same answer, so the
    first answer is reused (``langsam_calls`` counts the calls made).
    """

    def __init__(self, detector, tracker, *, prompt: str, padding: float = ROI_PADDING,
                 cache_timeout: int = CACHE_TIMEOUT, goal_mask: bool = False) -> None:
        self.detector, self.tracker = detector, tracker
        self.prompt = str(prompt)
        self.padding = float(padding)
        self.cache_timeout = int(cache_timeout)
        #: ask the detector for SAM's mask on the GOAL picture (owner ruling D-B (b)); the
        #: live frames never need one
        self.goal_mask_wanted = bool(goal_mask)
        self.reset()

    def reset(self) -> None:
        """``reset_tracking`` (``:323-335``) and ``run()``'s cache reset (``:2642-2643``)."""
        self.state = "DETECT"
        self.is_tracking = False
        self.cached_bbox: list | None = None
        self.cache_counter = 0
        self.controller_box: list | None = None
        self._first = True
        self.counts = {"langsam": 0, "lighttrack": 0, "cache": 0, "controller_cache": 0,
                       "none": 0}
        self.langsam_calls = 0
        self.tracker_inits = 0
        self.tracker_init_failures = 0
        self.goal_box = self.goal_box_raw = None
        self.goal_score = 0.0
        self.goal_n_boxes = 0
        self.goal_mask: np.ndarray | None = None

    # ------------------------------------------------------------------ the goal
    def goal(self, rgb: np.ndarray, image_wh) -> dict:
        """``detect_and_cache_goal`` (``:294-317``): LangSAM once on the goal picture. Its
        box is NOT left in the detector's cache: ``reset_tracking`` clears it before every
        sample (``run_multi_model_experiment_with_hollywood.py:688-690``)."""
        d = self.detector.detect(rgb, self.prompt, with_mask=self.goal_mask_wanted)
        self.goal_score = float(d.get("score", 0.0))
        self.goal_n_boxes = int(d.get("n_boxes", 0) or 0)
        mask = d.get("mask") if self.goal_mask_wanted else None
        self.goal_mask = None if mask is None else np.asarray(mask, bool)
        box = d.get("box")
        self.goal_box_raw = None if box is None else [float(v) for v in box]
        self.goal_box = (None if box is None
                         else pad_box(self.goal_box_raw, image_wh, self.padding))
        return {"box": self.goal_box, "box_raw": self.goal_box_raw, "score": self.goal_score,
                "n_boxes": self.goal_n_boxes}

    # ----------------------------------------------------------- the detector
    def _langsam(self, rgb):
        """``_detect_langsam`` (``:337-426``): the highest score, a float box, cached."""
        self.langsam_calls += 1
        box = self.detector.detect(rgb, self.prompt, with_mask=False).get("box")
        if box is None:
            return None
        box = [float(v) for v in box]
        self.cached_bbox = list(box)
        return box

    def _start_tracking(self, rgb, box) -> None:
        """``start_tracking`` (``:229-292``): a fresh LightTrack on this frame with the
        int-cast box; on failure the detector stays in DETECT."""
        try:
            self.tracker.init(rgb, [int(v) for v in box])
        except Exception:  # noqa: BLE001 - the release logs the failure and stays in DETECT
            self.tracker_init_failures += 1
            self.state = "DETECT"
            return
        self.tracker_inits += 1
        self.is_tracking = True
        self.state = "TRACK"
        self.cached_bbox = list(box)

    def _track_frame(self, rgb):
        """``_track_frame`` (``:474-569``): update; on loss LangSAM at once (and a fresh
        tracker if it finds the object), else CACHE through ``_use_cached_bbox``."""
        try:
            box = self.tracker.update(rgb)
        except Exception:  # noqa: BLE001 - the release's exception path is the loss path
            box = None
        if box is not None:
            box = [int(v) for v in box]                        # _xywh_to_xyxy
            self.cached_bbox = list(box)
            return box, "lighttrack"
        found = self._langsam(rgb)
        if found is not None:
            self._start_tracking(rgb, found)
            return found, "langsam"
        self.state, self.cache_counter = "CACHE", 0
        return self._use_cached(rgb)

    def _use_cached(self, rgb):
        """``_use_cached_bbox`` (``:571-626``)."""
        self.cache_counter += 1
        if self.cache_counter >= self.cache_timeout:
            found = self._langsam(rgb)
            if found is not None:
                self._start_tracking(rgb, found)
                return found, "langsam"
            return None, "none"          # the empty LangSAM result; the counter stays put
        if self.cached_bbox is not None:
            return list(self.cached_bbox), "cache"
        return None, "none"

    def _detect(self, rgb):
        """``detect``'s dispatch (``:191-227``)."""
        if self.state == "TRACK" and self.is_tracking:
            return self._track_frame(rgb)
        if self.state == "CACHE":
            return self._use_cached(rgb)
        found = self._langsam(rgb)
        return (found, "langsam") if found is not None else (None, "none")

    # ------------------------------------------------------------ one frame
    def live(self, rgb: np.ndarray, image_wh) -> tuple[list | None, list | None, str]:
        """``(padded box | None, raw box | None, source)`` for one frame, as the controller
        sees it (``ros_controller.py:1376-1389``)."""
        if self._first:
            self._first = False
            found = self._langsam(rgb)                         # the driver's one detection
            if found is not None:
                self._start_tracking(rgb, found)
            if self.state == "TRACK" and self.is_tracking:
                raw, source = self._track_frame(rgb)           # the loop's first detect()
            else:
                raw, source = (found, "langsam") if found is not None else (None, "none")
        else:
            raw, source = self._detect(rgb)
        if raw is not None:
            padded = pad_box(raw, image_wh, self.padding)
            self.controller_box = list(padded)
            self.counts[source] += 1
            return padded, list(raw), source
        if self.controller_box is not None:
            self.counts["controller_cache"] += 1
            return list(self.controller_box), None, "controller_cache"
        self.counts["none"] += 1
        return None, None, "none"


class _TokenDraw:
    """:class:`vsbench.vs.baselines.match.DenseMatcher`'s pool and seeded draw on tokens the
    caller encoded.

    The release encodes its tiles as ONE batch per side (``ros_controller.py:1751-1762``),
    and the encoder's batch shape is part of the arithmetic: on CUDA one tile encoded
    alone lands 6.4e-4 from the same tile in a batch of six (measured 2026-10-05), which
    can flip a nearest neighbour. So the sim arm encodes exactly the batches the release
    encodes and hands the tokens here. Same :func:`mutual_nn_match`, same
    :class:`EpisodeDraw`, same :class:`Matches` - only the encode call moved out
    (``tests/fast/test_objroi_sim.py`` checks it against a :class:`DenseMatcher`).
    """

    def __init__(self, *, num_landmarks: int, seed: int, threshold: float,
                 tie_order: str) -> None:
        self.num_landmarks = int(num_landmarks)
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.tie_order = str(tie_order)
        self.draw: EpisodeDraw | None = None
        self.goal_tokens = None
        self.mask: np.ndarray | None = None

    def set_episode(self, scene: str, start_id: Any) -> None:
        self.draw = EpisodeDraw(scene, start_id, self.seed)

    def reset(self, goal_tokens, mask: np.ndarray | None = None) -> None:
        """``mask``: bool eligibility over GOAL cells (``find_correspondences_batch``'s patch
        mask, ``matcher.py:225-226``), or ``None`` for every cell."""
        if self.draw is None:
            raise RuntimeError("call set_episode(scene, start_id) before reset()")
        self.goal_tokens = goal_tokens
        self.mask = None if mask is None else np.asarray(mask, bool).reshape(-1)
        self.draw.reset()

    def match(self, cur_tokens, *, input_size: int, patch_size: int, image_wh) -> Matches:
        pool = mutual_nn_match(self.goal_tokens, cur_tokens, threshold=self.threshold,
                               mask=self.mask, tie_order=self.tie_order)
        m = self.draw.draw(pool, self.num_landmarks)
        m.input_size, m.patch_size = int(input_size), int(patch_size)
        m.image_wh = (int(image_wh[0]), int(image_wh[1]))
        return m


class RoiTiledMatcherSim(CorrespondenceMatcher):
    """ART-VS as the release's SIMULATION program runs its object mode.

    The coarse phase: the live box (:class:`ReleaseRoiTracker`), both boxes padded 10 %,
    the crops cut with int corners and resized to 256 x 256 (Pillow BICUBIC), DINOv3-S/16
    descriptors of the release's own chain (``dinov3_vits16_artvs256``: 16 x 16 cells, 9
    log bins on the raw tokens, then L2), the release's mutual-nearest-neighbour pool, 24
    pairs, patch centres ``col * 16 + 8`` lifted to the full frame through the float boxes
    in float32 (:func:`lift32`). The switch, checked at the START of a frame on the PREVIOUS
    successful frame's error: the **mean** pixel error of the truncated, clipped points,
    once at least 10 frames have succeeded, when ``1 - e/e0 >= 1 - 0.2``; latched; the
    first tiled frame is the frame after the one that met it (``ros_controller.py:594-634,
    716, 2493-2499``). The tiled phase: a ``ceil(raw goal box / 256)`` grid per side, at
    most 5, from the goal's UNPADDED box (``:965, 2811-2819``); both crops resized to
    ``(256 gw, 256 gh)``; 256-px tiles encoded as one batch per side; tile *i* against tile
    *i* with ``max(1, 24 // n)`` pairs each; ``+ (j 256, i 256)`` and the float32 lift;
    concatenated in tile order, and when more than 24, the top 24 by similarity in
    ``torch.topk`` order (``:1800-1927``). The coarse crop is not matched in the tiled
    phase. Every frame's points come back as the float32 numbers the release truncates; the
    controller truncates them (``controller.pixel_quantisation: truncate``) and this class
    truncates a copy itself for the switch error - the same rule on the same numbers
    (:func:`uv_roi`; the fast tier checks that the two agree).

    **Fallbacks** (``record_fields()['roi_modes']`` counts them): no goal box -> the full
    goal picture at 256 against the live crop, goal points scaled by ``(W/256, H/256)``, in
    both phases (``goalless``; in the tiled phase the release runs its detector a second
    time on the frame, and so does this class); no live box and nothing cached -> both full
    pictures at 256 with ROUNDED points (``full_fallback``, ``calculate_uv``; the controller
    rounds that frame because ``extras["quantise"]`` says so). **The one stated deviation**:
    in the tiled phase the release's box-less fallback is a full-picture 6 x 6 tiling at
    1536 (``detect_features_tiled``), which is not built - the frame takes the coarse
    full-picture match instead and is counted as ``full_fallback_deviation``. It needs the
    goal detected, then the live object lost with nothing cached; no instance-suite record
    shows a box-less frame. Replicated although it is odd: ``calculate_uv`` pads a box-less
    draw of fewer than 24 pairs with ``(0, 0) -> (0, 0)`` rows, and turns one of 1-3 pairs
    into 24 zero rows that the loop does not skip (:func:`uv_pad`; ``extras["padded_rows"]``,
    ``record_fields()['roi_padded_frames']``) - with LangSAM's goal mask restricting the
    full-picture pool, a short draw is no longer rare on that path.

    **Degenerate frames and failures**: a frame with fewer than 4 points updates nothing
    (no error, no count; ``extras["degenerate"]``); the controller holds the last twist
    (``controller.degenerate_policy: hold``). An empty pool in the coarse phase counts
    toward the release's 10-in-a-row abort (``:1444-1449``; the tiled phase never counts);
    the bench does not abort - ``roi_failure_run_max`` and ``roi_failure_abort_frame`` say
    whether and when the release would have.

    **The goal mask is LangSAM's** (``roi_mask: langsam``, owner ruling D-B (b), the evening
    of 2026-10-05; the spec's D-B (a) "no mask" no longer applies): SAM 2.1's mask of the best
    GroundingDINO box on the GOAL picture, computed once at :meth:`reset`, enters the
    simulation controller's own mask branch exactly as if ``mask_path`` named that mask
    image - held as a lossless 8-bit picture (0 / 255; the release's own masks are files it
    reads, this one is computed, so no extra encoding is invented). Then, as the release:
    **coarse** - the mask cropped to the padded goal box with Pillow's float-box crop (which
    ROUNDS the corners, where the picture's crop truncates them), LANCZOS to 256, the
    temporary-JPEG round trip (:func:`jpeg_round_trip`), the cell rule
    (:func:`release_patch_mask`: ``> 128``, a cell counts when the mean of its pixels is
    above 0.5); **tiled** - the mask cropped with int corners, resized to ``(256 gw, 256 gh)``
    with Pillow's default filter (BICUBIC for a grey picture), cut into the same tiles, the
    same cell rule per tile (``ros_controller.py:1652-1677, 1764-1778``); the **full-frame
    fallbacks** - the whole mask, LANCZOS to 256, the cell rule (``create_patch_mask`` on
    ``mask_path``). The mask restricts which GOAL cells may match. No goal box means no mask.
    The live frames' detections ask for no mask. **SAM runs as ``lang_sam`` runs it**
    (``roi_sam_path: langsam``, 2026-10-06, the LangSAM check: the stability fallback off,
    ``SAM2Transforms``' picture - :class:`GroundingSam`), so the goal mask is the one the
    release's package makes, bit for bit; the arm flown before (the 2026-10-05 smoke) ran
    the transformers path and its label has no ``sam-langsam``. ``use_sam_mask`` (the old
    port's > 20 % nearest-neighbour rule) is refused on this arm. ``auto_calculate_tiles`` must be
    ``True`` (owner decision D-A (a), confirmed 2026-10-05 evening: the paper's adaptive
    grid; ``False`` - the release's fixed 2 x 2 - would be another cell). The draw is the
    bench's seeded per-frame draw (the release's ``torch.randperm`` is unseeded), as on every
    ViT arm.

    **The development cell** (``artvs_roi_sim_dev``, owner 2026-10-10; the module docstring's
    section): ``roi_mask: objects`` takes the bench's goal mask - the simulator's outline -
    through the same branch instead of LangSAM's (``roi_sam_path`` must then be ``None``: no SAM
    runs), and ``switch_max_steps`` races the release's switch rule with a step cap. Both
    default off; with them off this class is the faithful cell, line for line.
    """

    name = "artvs_roi_sim"
    provides_margin = False
    provides_conf_weight = False
    honours_goal_mask = False        # its mask is LangSAM's (roi_mask); a bench one would be 2nd
    requires_goal_mask = False       # True under `roi_mask: objects` (the dev cell, 2026-10-10)

    def __init__(self, *, num_landmarks: int, seed: int = 0, backbone=None,
                 backbone_name: str = "dinov3_vits16_artvs256", device: str | None = None,
                 tie_order: str = "stable", threshold: float = 1.0, vit_input_size: int = 256,
                 switch_threshold: float = 0.2, switch_min_frames: int = SWITCH_MIN_FRAMES,
                 switch_iteration: int = 0, switch_max_steps: int = 0, prompt: str = "shoe",
                 roi_padding: float = ROI_PADDING, cache_timeout: int = CACHE_TIMEOUT,
                 box_threshold: float = BOX_THRESHOLD, text_threshold: float = TEXT_THRESHOLD,
                 max_tiles_per_dim: int = MAX_TILES_PER_DIM, use_sam_mask: bool = False,
                 roi_mask: str = "langsam", roi_sam_path: str | None = "langsam",
                 auto_calculate_tiles: bool = True,
                 lighttrack_root: str | None = None, lighttrack_weights: str | None = None,
                 tracker=None) -> None:
        if use_sam_mask:
            raise ValueError(
                "artvs_roi_sim: `use_sam_mask` is the old port's mask rule (a cell counts when "
                "more than 20 % of its pixels are set, nearest-neighbour resize) and is refused "
                "here. This arm's goal mask is `roi_mask: langsam` - LangSAM's mask through the "
                "release's own mask branch and cell rule (owner ruling D-B (b), 2026-10-05).")
        if roi_mask not in ROI_MASKS:
            raise ValueError(
                f"artvs_roi_sim: roi_mask is {roi_mask!r}; the faithful cell's only value is "
                "'langsam' (owner ruling D-B (b), 2026-10-05: LangSAM's goal mask through the "
                "simulation controller's mask branch) and the development cell's is 'objects' "
                "(owner 2026-10-10: the simulator's true outline through the same branch). The "
                "unmasked simulation path is a different cell and is not built.")
        if roi_mask == "objects" and roi_sam_path is not None:
            raise ValueError(
                f"artvs_roi_sim: roi_sam_path is {roi_sam_path!r} under roi_mask 'objects'; that "
                "cell runs no SAM (the goal mask is the simulator's outline), so the path must "
                "be null - a stated SAM path would put a mask on the record that was not made")
        if roi_mask == "langsam" and roi_sam_path not in ROI_SAM_PATHS:
            raise ValueError(
                f"artvs_roi_sim: roi_sam_path is {roi_sam_path!r}; the only value is 'langsam' "
                "(the LangSAM check, 2026-10-06): SAM 2.1 run as the lang_sam package runs it - "
                "the stability fallback off, SAM2Transforms' picture - so the goal mask is the "
                "release's own. 'transformers' is the old cells' SAM (artvs_roi*); this arm "
                "flown with it (the 2026-10-05 smoke) is told apart by its label, which has no "
                "`sam-langsam`.")
        if not auto_calculate_tiles:
            raise ValueError(
                "artvs_roi_sim: auto_calculate_tiles is True by owner decision D-A (a), "
                "2026-10-05 (the paper's ceil(box / 256) grid, at most 5 per side). The "
                "release's literal `false` - a fixed 2 x 2 grid - is a different cell and is "
                "not built.")
        if tie_order not in TIE_ORDERS:
            raise ValueError(f"tie_order must be one of {TIE_ORDERS}, got {tie_order!r}")
        self.num_landmarks = int(num_landmarks)
        if self.num_landmarks < 1:
            raise ValueError(f"num_landmarks must be >= 1, got {num_landmarks}")
        if not 0.0 < float(switch_threshold) <= 1.0:
            raise ValueError(f"switch_threshold must be in (0, 1], got {switch_threshold}")
        if int(switch_min_frames) < 0 or int(switch_iteration) < 0 or int(switch_max_steps) < 0:
            raise ValueError("switch_min_frames, switch_iteration and switch_max_steps must "
                             "be >= 0")
        if int(switch_iteration) > 0 and int(switch_max_steps) > 0:
            raise ValueError(
                "artvs_roi_sim: switch_iteration (the release's iteration switch, which replaces "
                "the error rule) and switch_max_steps (the step cap that races it, 2026-10-10) "
                "cannot both be set - say which switch the cell flies")
        if backbone is None:
            if device is None:
                raise ValueError(MISSING_DEVICE)
            from vsbench.vs.baselines.features import build_backbone  # noqa: PLC0415

            backbone = build_backbone(backbone_name, device=device)
        self.backbone = backbone
        self.vit = int(vit_input_size)
        if self.vit != int(backbone.input_size):
            raise ValueError(
                f"artvs_roi_sim: vit_input_size {self.vit} but the backbone "
                f"{getattr(backbone, 'name', '?')!r} reads {backbone.input_size} px - the crops "
                "and tiles are cut at vit_input_size and must reach the encoder unresized (the "
                "old cell's 224-crop / 518-encoder mismatch is the bug this arm exists to fix)")
        self.patch = int(backbone.patch_size)
        self.seed = int(seed)
        self.tie_order = str(tie_order)
        self.threshold = float(threshold)
        self.switch_threshold = float(switch_threshold)
        self.switch_min_frames = int(switch_min_frames)
        self.switch_iteration = int(switch_iteration)
        self.switch_max_steps = int(switch_max_steps)
        self.max_tiles = int(max_tiles_per_dim)
        self.use_sam_mask = False
        self.roi_mask = str(roi_mask)
        self.roi_sam_path = None if roi_sam_path is None else str(roi_sam_path)
        # `objects` (the dev cell): the runner hands reset() the simulator's outline, and the
        # bench refuses the cell without it (vsbench.vs.bench.arms.build_arm)
        self.honours_goal_mask = self.requires_goal_mask = self.roi_mask == "objects"
        sam_mask = self.roi_mask == "langsam"
        self.auto_calculate_tiles = True
        self.device = str(device) if device is not None else None
        self.tracker = tracker or ReleaseRoiTracker(
            # the box path is the faithful cell's either way; under `objects` SAM is never
            # asked for (with_mask=False everywhere), so it is never loaded
            GroundingSam(device=self.device or "cuda", box_threshold=box_threshold,
                         text_threshold=text_threshold, sam_path=self.roi_sam_path or "langsam"),
            LightTrackWrap(root=lighttrack_root, weights=lighttrack_weights, int_boxes=True),
            prompt=prompt, padding=roi_padding, cache_timeout=cache_timeout, goal_mask=sam_mask)
        # an injected tracker asks for the goal mask too when the mask is SAM's: it is this
        # arm's setting
        self.tracker.goal_mask_wanted = sam_mask
        det = getattr(self.tracker, "detector", None)
        if sam_mask and isinstance(det, GroundingSam) and det.sam_path != self.roi_sam_path:
            # a stub detector is the fast tier's; a real one must run this arm's SAM path,
            # or the record's `roi_sam_path` would state a mask the arm did not make
            raise ValueError(f"artvs_roi_sim: the injected tracker's GroundingSam runs "
                             f"sam_path {det.sam_path!r}; this arm's roi_sam_path is "
                             f"{self.roi_sam_path!r}")
        self.prompt = str(prompt)
        self._episode: tuple[str, Any] | None = None
        self._coarse = self._draw(self.num_landmarks)
        self._full = self._draw(self.num_landmarks)
        self._tiles: list[_TokenDraw] = []
        self.image_wh: tuple[int, int] | None = None
        self.goal_box = self.goal_box_raw = None
        self.grid: tuple[int, int] | None = None
        self._clear_mask()
        self._reset_state()

    def _clear_mask(self) -> None:
        self.mask_share: float | None = None        # the goal mask's pixel share
        self.mask_cells: dict[str, Any] = {}        # eligible goal cells per path

    def _draw(self, k: int) -> _TokenDraw:
        return _TokenDraw(num_landmarks=k, seed=self.seed, threshold=self.threshold,
                          tie_order=self.tie_order)

    def _reset_state(self) -> None:
        self.frames = 0                  # match() calls this episode (bench steps)
        self.frames_ok = 0               # the release's iteration_count: successful frames
        self.e0: float | None = None     # initial_feature_error
        self.e_prev: float | None = None  # current_feature_error, as the next check reads it
        self.tiled = False               # roi_tiling_activated (latched)
        self.switch_frame: int | None = None  # roi_tiling_switch_iteration
        self.switch_step: int | None = None   # the bench step the first tiled frame ran at
        self.switch_reason: str | None = None  # "error" / "iteration" / "step" (2026-10-10)
        self.modes: dict[str, int] = {}
        self.degenerate_frames = 0
        self.padded_frames = 0           # box-less frames calculate_uv padded with zero rows
        self.failure_run = 0
        self.failure_run_max = 0
        self.failure_abort_frame: int | None = None

    # ---------------------------------------------------------------- episode
    def set_episode(self, scene: str, start_id: Any) -> None:
        self._episode = (str(scene), start_id)
        for d in (self._coarse, self._full, *self._tiles):
            d.set_episode(scene, start_id)

    def _encode(self, frames: list) -> Any:
        """One encoder call, ONE batch - the release's batch shapes (see :class:`_TokenDraw`)."""
        return self.backbone.encode_batch([np.ascontiguousarray(f) for f in frames])

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        if self.roi_mask == "langsam" and goal_mask is not None:
            raise ValueError("artvs_roi_sim: this arm's mask is LangSAM's (`roi_mask`); the "
                             "bench's goal_mask is refused so a cell never has two masks")
        if self.roi_mask == "objects" and goal_mask is None:
            raise ValueError("artvs_roi_sim (roi_mask 'objects'): reset() needs the bench's goal "
                             "mask - the simulator's object outline (matcher.goal_mask: "
                             "objects); without it the cell would fly unmasked under its label")
        if self._episode is None:
            raise RuntimeError("call set_episode(scene, start_id) before reset() - the draw "
                               "is a function of (scene, start, seed)")
        goal = np.ascontiguousarray(np.asarray(goal_rgb, np.uint8)[..., :3])
        h, w = goal.shape[:2]
        self.image_wh = (int(w), int(h))
        self._reset_state()
        self._clear_mask()
        self.tracker.reset()
        g = self.tracker.goal(goal, self.image_wh)
        self.goal_box, self.goal_box_raw = g["box"], g["box_raw"]
        # the goal mask as the picture `mask_path` would name (`Image.open(...).convert('L')`):
        # LangSAM's comes with its box; the simulator's outline is there with or without one,
        # as a `mask_path` is (`:797, 1399, 973`)
        if self.roi_mask == "objects":
            src = self._mask_image(goal_mask)
        else:
            src = self._mask_image(self.tracker.goal_mask if self.goal_box is not None else None)
        # the full goal at 256 - the fallbacks' goal (`_cache_goal_image_processing`, :328-340)
        # with the whole mask (`detect_features_original` passes `mask_path`, :797)
        full_mask = None if src is None else release_patch_mask(src, self.vit, self.patch)
        self._full.reset(self._encode([goal])[0], mask=full_mask)
        self._tiles = []
        self.grid = None
        if self.goal_box is None:
            if src is not None:          # only the simulator's outline gets here
                self.mask_cells = {"full": int(full_mask.sum())}
            return                       # `set_yolo_keyword`'s no-detection branch (:966-973)
        rx1, ry1, rx2, ry2 = self.goal_box_raw
        # `_calculate_optimal_tile_number` on the UNPADDED box (:965, 2811-2819)
        self.grid = (max(1, min(self.max_tiles, math.ceil((rx2 - rx1) / self.vit))),
                     max(1, min(self.max_tiles, math.ceil((ry2 - ry1) / self.vit))))
        goal_crop = crop(goal, self.goal_box)
        coarse_mask = None
        if src is not None:
            # `_cache_goal_roi_features` (:382-400): Pillow's float-box crop (it rounds the
            # corners), LANCZOS to the square, the temporary JPEG; `create_patch_mask` (:25-48)
            from PIL import Image

            roi = src.crop(tuple(self.goal_box)).resize((self.vit, self.vit),
                                                        Image.Resampling.LANCZOS)
            coarse_mask = release_patch_mask(jpeg_round_trip(roi), self.vit, self.patch)
        self._coarse.reset(self._encode([resize(goal_crop, (self.vit, self.vit))])[0],
                           mask=coarse_mask)
        gw, gh = self.grid
        big = resize(goal_crop, (self.vit * gw, self.vit * gh))
        tok = self._encode(self._cut(big))                   # the goal tiles' one batch
        tile_masks = [None] * (gw * gh)
        if src is not None:
            # `detect_features_roi_tiled` (:1652-1677): int corners, Pillow's default filter
            # to the tiled size, the same tiles, `create_patch_mask_from_pil` (:1764-1778)
            x1, y1, x2, y2 = [int(c) for c in self.goal_box]
            mbig = src.crop((x1, y1, x2, y2)).resize((self.vit * gw, self.vit * gh))
            tile_masks = [release_patch_mask(
                mbig.crop((j * self.vit, i * self.vit, (j + 1) * self.vit, (i + 1) * self.vit)),
                self.vit, self.patch) for i in range(gh) for j in range(gw)]
        per_tile = max(1, self.num_landmarks // (gw * gh))   # :1800
        for k in range(gw * gh):
            d = self._draw(per_tile)
            d.set_episode(*self._episode)
            d.reset(tok[k], mask=tile_masks[k])
            self._tiles.append(d)
        if src is not None:
            self.mask_cells = {"full": int(full_mask.sum()), "coarse": int(coarse_mask.sum()),
                               "tiles": [int(m.sum()) for m in tile_masks]}

    def _mask_image(self, mask):
        """The goal mask as an 8-bit grey picture (0 / 255) - what ``Image.open(mask_path)
        .convert('L')`` hands the release's mask branch for a lossless mask file."""
        if mask is None:
            return None
        from PIL import Image

        m = np.asarray(mask, bool)
        if m.shape != (self.image_wh[1], self.image_wh[0]):
            raise ValueError(f"artvs_roi_sim: the goal mask is {m.shape}, the goal picture "
                             f"{self.image_wh[1]} x {self.image_wh[0]}")
        self.mask_share = float(m.mean())
        return Image.fromarray(m.astype(np.uint8) * 255)

    def _cut(self, big: np.ndarray) -> list:
        """256-px tiles, row-major ``(i, j)`` (``:1700-1726``)."""
        gw, gh = self.grid
        return [np.ascontiguousarray(big[i * self.vit:(i + 1) * self.vit,
                                         j * self.vit:(j + 1) * self.vit])
                for i in range(gh) for j in range(gw)]

    # ---------------------------------------------------------------- frame
    def _check_switch(self) -> None:
        """``check_roi_tiling_switch`` (``:594-634``), run before the frame's detection; then,
        when ``switch_max_steps`` is set (the dev cell, owner 2026-10-10), the step cap: the
        tiles latch at the start of bench step ``N + 1`` if the release's rule has not fired
        by then (``self.frames`` already counts the frame being started)."""
        if self.tiled:
            return
        if self.frames_ok > 0:
            if self.switch_iteration > 0:
                if self.frames_ok >= self.switch_iteration:
                    self._latch("iteration")
                return
            if (self.e0 is not None and self.e0 > 0 and self.e_prev is not None
                    and self.frames_ok >= self.switch_min_frames
                    and (1.0 - (self.e_prev / self.e0)) >= (1.0 - self.switch_threshold)):
                self._latch("error")
                return
        if self.switch_max_steps > 0 and self.frames - 1 >= self.switch_max_steps:
            self._latch("step")

    def _latch(self, reason: str) -> None:
        self.tiled = True
        self.switch_frame = self.frames_ok
        self.switch_step = self.frames
        self.switch_reason = reason

    def _count_failure(self, empty: bool) -> None:
        """``feature_failure_count`` (``:1444-1452``): the coarse phase's empty pools in a
        row. The bench records the run instead of aborting."""
        if not empty:
            self.failure_run = 0
            return
        self.failure_run += 1
        self.failure_run_max = max(self.failure_run_max, self.failure_run)
        if self.failure_run >= FAILURE_ABORT and self.failure_abort_frame is None:
            self.failure_abort_frame = self.frames

    def _full_frame(self, cur, source: str, *, mode: str, count_failure: bool) -> dict:
        """``detect_features_original`` (``:750-854``): both full pictures at 256, the
        points rounded (``calculate_uv``), a short draw padded with zero rows as the
        release pads it (:func:`uv_pad`). An empty pool is a skipped frame (``:804-810``)."""
        m = self._full.match(self._encode([cur])[0], input_size=self.vit,
                             patch_size=self.patch, image_wh=self.image_wh)
        if count_failure:
            self._count_failure(len(m) == 0)
        out = {"m": m, "mode": mode, "source": source, "live_box": None, "quantise": "round",
               "padded_rows": 0}
        if not len(m):
            out.update(pg=np.zeros((0, 2)), pc=np.zeros((0, 2)))
            return out
        w, h = self.image_wh
        scale = np.asarray([w / self.vit, h / self.vit], np.float64)
        pg, pc = uv_pad(m.goal_xy * scale, m.cur_xy * scale, self.num_landmarks)
        out.update(pg=pg, pc=pc, padded_rows=len(pg) - (len(m) if len(m) >= MIN_POINTS else 0))
        return out

    def _coarse_frame(self, cur) -> dict:
        """``detect_features_roi`` - the second definition, the one that runs
        (``:1360-1571``)."""
        box, _raw, source = self.tracker.live(cur, self.image_wh)
        if box is None:                                      # :1386-1389
            return self._full_frame(cur, source, mode="full_fallback", count_failure=True)
        tok = self._encode([resize(crop(cur, box), (self.vit, self.vit))])[0]
        if self.goal_box is None:                            # the goal-less path, :1392-1404
            m = self._full.match(tok, input_size=self.vit, patch_size=self.patch,
                                 image_wh=self.image_wh)
            mode = "goalless"
        else:
            m = self._coarse.match(tok, input_size=self.vit, patch_size=self.patch,
                                   image_wh=self.image_wh)
            mode = "coarse"
        self._count_failure(len(m) == 0)
        if not len(m):
            return {"pg": np.zeros((0, 2)), "pc": np.zeros((0, 2)), "m": m, "mode": mode,
                    "source": source, "live_box": box, "quantise": "truncate"}
        g_vit = m.goal_xy.astype(np.float32)                 # col * 16 + 8, exact
        c_vit = m.cur_xy.astype(np.float32)
        if self.goal_box is None:
            w, h = self.image_wh                             # :1558-1565, float32 in place
            pg = g_vit * np.asarray([w / self.vit, h / self.vit], np.float32)
        else:
            pg = lift32(g_vit, self.goal_box, (self.vit, self.vit))
        pc = lift32(c_vit, box, (self.vit, self.vit))
        return {"pg": pg, "pc": pc, "m": m, "mode": mode, "source": source, "live_box": box,
                "quantise": "truncate"}

    def _tiled_frame(self, cur) -> dict:
        """``detect_features_roi_tiled`` (``:1573-1982``)."""
        box, _raw, source = self.tracker.live(cur, self.image_wh)
        if box is None:
            # the release: `detect_features_tiled`, full-image 6 x 6 tiling at 1536 - NOT
            # BUILT (the stated deviation); the coarse full-picture match stands in, and the
            # release's tiled path never counts a failure
            return self._full_frame(cur, source, mode="full_fallback_deviation",
                                    count_failure=False)
        if self.goal_box is None:
            # the release falls back to `detect_features_roi()`, which runs the detector
            # AGAIN on this frame (:1608-1613) - so does this
            return self._coarse_frame(cur)
        gw, gh = self.grid
        tw, th = self.vit * gw, self.vit * gh
        tok = self._encode(self._cut(resize(crop(cur, box), (tw, th))))   # current_batch
        gp, cp, sims, cyc, marg, counts, pool = [], [], [], [], [], [], 0
        for k, d in enumerate(self._tiles):
            m = d.match(tok[k], input_size=self.vit, patch_size=self.patch,
                        image_wh=self.image_wh)
            counts.append(len(m))
            pool += int(m.pool_size)
            if not len(m):
                continue
            i, j = divmod(k, gw)
            off = np.asarray([j * self.vit, i * self.vit], np.float32)
            gp.append(m.goal_xy.astype(np.float32) + off)    # :1840-1846, exact in float32
            cp.append(m.cur_xy.astype(np.float32) + off)
            sims.append(np.asarray(m.sim, np.float32))
            cyc.append(np.asarray(m.cyclic, np.float64))
            marg.append(np.asarray(m.margin, np.float64))
        out = {"mode": "tiled", "source": source, "live_box": box, "quantise": "truncate",
               "tile_counts": counts, "pool": pool, "m": None}
        if not gp:                                           # :1979-1982: nothing at all
            out.update(pg=np.zeros((0, 2)), pc=np.zeros((0, 2)), sim=np.zeros(0),
                       cyclic=np.zeros(0), margin=np.zeros(0))
            return out
        pg = lift32(np.vstack(gp), self.goal_box, (tw, th))
        pc = lift32(np.vstack(cp), box, (tw, th))
        sim = np.concatenate(sims)
        cyclic, margin = np.concatenate(cyc), np.concatenate(marg)
        if len(sim) > self.num_landmarks:                    # :1917-1927, torch.topk order
            import torch  # noqa: PLC0415 - the release's own selection call

            idx = torch.topk(torch.from_numpy(sim), self.num_landmarks).indices.numpy()
            pg, pc, sim, cyclic, margin = pg[idx], pc[idx], sim[idx], cyclic[idx], margin[idx]
        out.update(pg=pg, pc=pc, sim=sim, cyclic=cyclic, margin=margin)
        return out

    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        if current_mask is not None:
            raise ValueError("artvs_roi_sim does not take a per-frame current_mask")
        if self.image_wh is None:
            raise RuntimeError("reset() must be called before match()")
        self.frames += 1
        cur = np.ascontiguousarray(np.asarray(current_rgb, np.uint8)[..., :3])
        self._check_switch()                                 # detect_features, :716
        r = self._tiled_frame(cur) if self.tiled else self._coarse_frame(cur)
        self.modes[r["mode"]] = self.modes.get(r["mode"], 0) + 1
        # the release's error on its integer points (ibvs, :2486-2499), from a copy
        if r["quantise"] == "round":
            q_g, q_c = uv_round(r["pg"], r["pc"])
        else:
            q_g, q_c = uv_roi(r["pg"], r["pc"], self.image_wh)
        degenerate = len(q_g) < MIN_POINTS
        e = None
        if degenerate:
            self.degenerate_frames += 1                      # skipped: nothing updates
        else:
            e = float(np.mean(np.linalg.norm(q_c - q_g, axis=1)))
            if self.e0 is None and self.frames_ok <= 1:
                self.e0 = e
            self.e_prev = e
            self.frames_ok += 1
        return self._pack(r, degenerate, e)

    def _pack(self, r: dict, degenerate: bool, e: float | None) -> Matches:
        w, h = self.image_wh
        m = r.get("m")
        k = len(r["pg"])
        if m is not None:
            # the drawn rows first; a padded frame's zero rows (`uv_pad`) carry zeros, and a
            # 1-3 pair draw padded to all-zero rows keeps none of its own
            n = len(m) if (len(m) >= MIN_POINTS or not r.get("padded_rows")) else 0
            n = min(n, k)
            sim, cyclic, margin = np.zeros(k), np.zeros(k), np.zeros(k)
            sim[:n], cyclic[:n], margin[:n] = m.sim[:n], m.cyclic[:n], m.margin[:n]
            pool = m.pool_size
        else:
            sim, cyclic, margin, pool = r["sim"], r["cyclic"], r["margin"], r["pool"]
        out = Matches.from_pixel_pairs(
            np.asarray(r["pg"], np.float64).reshape(-1, 2),
            np.asarray(r["pc"], np.float64).reshape(-1, 2),
            side=w, input_size=w, patch_size=1, image_wh=(w, h),
            sim=np.asarray(sim, np.float64), cyclic=np.asarray(cyclic, np.float64),
            margin=np.asarray(margin, np.float64), pool_size=int(pool),
            num_landmarks=self.num_landmarks, draw_seed=self.seed, tie_order=self.tie_order)
        live = r.get("live_box")
        out.extras.update(
            matcher=self.name, mode=r["mode"], roi_source=r["source"], goal_box=self.goal_box,
            goal_box_raw=self.goal_box_raw, tiled=self.tiled,
            grid=None if self.grid is None else list(self.grid),
            live_box=None if live is None else [float(v) for v in live],
            enc_input=self.vit, enc_patch=self.patch, enc_bins=9,
            switch_e0=self.e0, switch_e=e, switch_frames_ok=self.frames_ok,
            degenerate=bool(degenerate))
        if "tile_counts" in r:
            out.extras["tile_counts"] = [int(c) for c in r["tile_counts"]]
        if r["quantise"] == "round":
            out.extras["quantise"] = "round"     # the controller rounds this frame
        if r.get("padded_rows"):
            out.extras["padded_rows"] = int(r["padded_rows"])   # calculate_uv's zero rows
            self.padded_frames += 1
        return out

    # ---------------------------------------------------------------- record
    def record_fields(self) -> dict:
        t = self.tracker
        modes = dict(self.modes)
        return {"front_end": self.name, "matcher_device": device_of(self.backbone),
                "roi_prompt": t.prompt, "roi_padding": t.padding,
                "roi_cache_timeout": t.cache_timeout, "roi_vit_input_size": self.vit,
                "roi_switch_threshold": self.switch_threshold,
                "roi_max_tiles_per_dim": self.max_tiles, "roi_use_sam_mask": False,
                "roi_goal_box": self.goal_box, "roi_goal_score": t.goal_score,
                "roi_grid": None if self.grid is None else list(self.grid),
                "roi_switched": self.tiled,
                "roi_sources": {s: n for s, n in t.counts.items() if n},
                "roi_detector": (GDINO_MODEL + " + " + SAM_MODEL + " (SAM on the goal only)"
                                 if self.roi_mask == "langsam" else
                                 GDINO_MODEL + " (boxes only; the goal mask is the "
                                 "simulator's outline, no SAM)"),
                "roi_tracker": LIGHTTRACK_ARCH,
                "prealign": getattr(getattr(self, "prealign", None), "label", None),
                # the simulation arm's own (spec 2026-10-05, section 4.2)
                "roi_backbone": getattr(self.backbone, "name", None), "roi_enc_bins": 9,
                "roi_goal_box_raw": self.goal_box_raw, "roi_grid_from": "unpadded",
                "roi_grid_rule": "auto",
                "roi_switch_stat": "mean", "roi_switch_min_frames": self.switch_min_frames,
                "roi_switch_lag": 1, "roi_switch_iteration": self.switch_iteration,
                "roi_switch_frame": self.switch_frame, "roi_switch_step": self.switch_step,
                "roi_topk_cap": self.num_landmarks, "roi_first_frame": "track_after_init",
                "roi_mask": self.roi_mask, "roi_sam_path": self.roi_sam_path,
                "roi_mask_share": self.mask_share,
                "roi_mask_box": self.goal_box_raw if self.mask_share is not None else None,
                "roi_mask_cells": dict(self.mask_cells) or None,
                "roi_mask_rule": "release: >128, cell mean > 0.5; coarse via JPEG (Pillow "
                                 "defaults, quality 75)",
                "roi_tracker_boxes": "int",
                "roi_modes": modes,
                "roi_fallback_frames": sum(n for k, n in modes.items()
                                           if k not in ("coarse", "tiled")),
                "roi_degenerate_frames": self.degenerate_frames,
                "roi_padded_frames": self.padded_frames,
                "roi_failure_run_max": self.failure_run_max,
                "roi_failure_abort_frame": self.failure_abort_frame,
                "roi_langsam_calls": t.langsam_calls, "roi_tracker_inits": t.tracker_inits,
                # the dev cell's switch (2026-10-10); absent from the faithful cell's record,
                # which stays key for key as it was
                **({"roi_switch_max_steps": self.switch_max_steps,
                    "roi_switch_reason": self.switch_reason}
                   if self.switch_max_steps > 0 else {})}

    def close(self) -> None:
        self._tiles = []
