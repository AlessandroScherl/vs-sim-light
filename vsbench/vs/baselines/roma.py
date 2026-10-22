# vs-sim copy: vssim/baselines/roma.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""RoMa and RoMa v2 as matchers: the dense learned baseline (CNSv2's "RoMa + IBVS").

Purpose
-------
RoMa (Edstedt et al., CVPR 2024) and RoMa v2 (Edstedt et al., 2025, arXiv
2511.15706) are *dense* matchers: for every pixel of picture A they predict where it
lands in picture B (a warp) and how sure they are (a certainty, v2: an overlap
probability). CNSv2 (arXiv 2503.00132) compares itself against "RoMa + IBVS" - RoMa's
correspondences fed to the classical control law - and that is what
:class:`RoMaMatcher` builds: per control step, the goal picture is matched against
the live one, ``num_landmarks`` correspondences are taken out of the dense warp, and
they go to the same IBVS law (the bench's or ViSP's) as every other arm, on the
pixel lattice, with the same RANSAC switch.

How the K correspondences are chosen (``sampling:``)
-----------------------------------------------------
``sample`` (the default) - **RoMa's own sampler** (``model.sample``, certainty-weighted
and, in v1, spatially balanced) with a seeded generator per frame, so the rows
change every frame the way the ViT arms' seeded draw does and a re-run of the same
episode draws the same rows. ``pinned`` - ``num_landmarks`` goal pixels drawn once per
episode on a seeded uniform grid over the goal picture and read through the warp
every frame: the same landmarks throughout (the frozen-landmark twin). Both record
which ran. ``certainty_min`` drops correspondences below a certainty (v1) / overlap
(v2) threshold before RANSAC.

Fences
------
- **The packages are the authors' own** (``romatch``, ``romav2``; pip). Weights are
  fetched by the packages the *first* time a model is built (torch.hub, ~1 GB each
  plus the DINOv2 / DINOv3 backbones) into ``$TORCH_HOME``; a station stages them
  once and a run finds them there. The weight files' names are recorded.
- **v1 runs its pure-torch local correlation** (``use_custom_corr=False``): the
  package's optional CUDA kernel is not built on a bare interpreter, and the two
  paths compute the same thing.
- **Resolution is the models' own**: v1 matches at 560 x 560 and upsamples to 864,
  v2 at its configured low/high resolution; the warp is mapped back to the native
  1920 x 1080 pixels by the packages' ``to_pixel_coordinates``.
- **No margin, no trust** (``provides_margin`` / ``provides_conf_weight`` False):
  the certainty travels as ``sim`` for the record, the arm is unweighted.

Golden tests: ``tests/fast/test_roma.py`` (a stub model through the seam);
``tests/slow/test_station.py`` records the station runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from vsbench.vs.baselines.classical import RANSAC_CHOICES, filter_by_ransac, resolve_ransac
from vsbench.vs.baselines.match import CorrespondenceMatcher, Matches, episode_seed

SAMPLING = ("sample", "pinned")
ROMA_KEYS = frozenset({"variant", "num_landmarks", "device", "ransac", "ransac_thresh_px",
                       "tie_order", "seed", "sampling", "certainty_min", "version"})

#: The weight files the two packages fetch into ``$TORCH_HOME/hub/checkpoints``.
ROMA_WEIGHTS = {1: ("roma_outdoor.pth", "dinov2_vitl14_pretrain.pth"),
                2: ("romav2.0.1.pt",)}


def roma_available(version: int) -> tuple[bool, str]:
    try:
        if version == 1:
            import romatch  # noqa: F401
        else:
            import romav2  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - env dependent
        return False, (f"RoMa v{version}'s package is not importable ({exc}); "
                       f"`pip install {'romatch' if version == 1 else 'romav2'}`")
    return True, ""


class RoMaMatcher(CorrespondenceMatcher):
    """RoMa (v1) or RoMa v2 -> ``num_landmarks`` correspondences per frame.

    ``model`` injects a ready-made model with ``match`` / ``sample`` /
    ``to_pixel_coordinates`` (the fast tier's stub); otherwise the package's is
    built lazily at the first :meth:`reset`.
    """

    provides_margin = False
    provides_conf_weight = False
    honours_goal_mask = True      # zeroed certainty AND a full-resolution reject
    #: How many extra samples a masked frame draws before the full-resolution reject.
    MASK_OVERSAMPLE = 8

    def __init__(self, *, version: int, num_landmarks: int, device: str, seed: int = 0,
                 ransac: str = "none", ransac_thresh_px: float = 3.0, scene_spec: Any = None,
                 tie_order: str = "stable", sampling: str = "sample",
                 certainty_min: float = 0.0, model: Any = None) -> None:
        if version not in (1, 2):
            raise ValueError(f"RoMa version must be 1 or 2, got {version!r}")
        if sampling not in SAMPLING:
            raise ValueError(f"sampling must be one of {SAMPLING}, got {sampling!r}")
        if ransac not in RANSAC_CHOICES:
            raise ValueError(f"ransac must be one of {RANSAC_CHOICES}, got {ransac!r}")
        if int(num_landmarks) < 1:
            raise ValueError("RoMa needs num_landmarks >= 1 (a dense warp has no 'all')")
        self.version = int(version)
        self.name = "roma_v1" if self.version == 1 else "roma_v2"
        self.num_landmarks = int(num_landmarks)
        self.device = str(device)
        self.seed = int(seed)
        self.ransac = resolve_ransac(ransac, scene_spec)
        self.ransac_requested = str(ransac)
        self.ransac_thresh_px = float(ransac_thresh_px)
        self.tie_order = str(tie_order)
        self.sampling = str(sampling)
        self.certainty_min = float(certainty_min)
        self._model = model
        self._goal: np.ndarray | None = None
        self._pinned: np.ndarray | None = None
        self.image_wh: tuple[int, int] | None = None
        self._episode: tuple[str, Any] = ("", 0)
        self._frames = 0
        self._frames_ransac_applied = 0

    # ------------------------------------------------------------------ build
    def _build(self):
        if self._model is None:
            ok, why = roma_available(self.version)
            if not ok:
                raise RuntimeError(why)
            if self.version == 1:
                from romatch import roma_outdoor

                self._model = roma_outdoor(device=self.device, use_custom_corr=False)
            else:
                from romav2 import RoMaV2

                self._model = RoMaV2().to(self.device).eval()
        return self._model

    def weights_present(self) -> dict[str, bool]:
        root = Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch"))
        return {n: (root / "hub" / "checkpoints" / n).exists() for n in ROMA_WEIGHTS[self.version]}

    # ---------------------------------------------------------------- episode
    def set_episode(self, scene: str, start_id: Any) -> None:
        self._episode = (str(scene), start_id)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        self._goal = np.ascontiguousarray(np.asarray(goal_rgb, np.uint8)[..., :3])
        h, w = self._goal.shape[:2]
        self.image_wh = (int(w), int(h))
        self._frames = 0
        self._frames_ransac_applied = 0
        self._pinned = None
        self._goal_mask = None if goal_mask is None else np.asarray(goal_mask).astype(bool)
        if self.sampling == "pinned":
            rng = np.random.default_rng(episode_seed(*self._episode, self.seed))
            if self._goal_mask is None:
                self._pinned = np.column_stack([rng.uniform(0, w - 1, self.num_landmarks),
                                                rng.uniform(0, h - 1, self.num_landmarks)])
            else:                      # inside the mask, by rejection
                vs, us = np.nonzero(self._goal_mask)
                pick = rng.choice(len(us), size=self.num_landmarks,
                                  replace=len(us) < self.num_landmarks)
                self._pinned = np.column_stack([us[pick].astype(float), vs[pick].astype(float)])

    # --------------------------------------------------------------- the warp
    def _dense(self, cur_rgb: np.ndarray):
        """``(warp, certainty)`` goal -> current, both on the model's grid, as tensors."""
        import torch
        from PIL import Image

        model = self._build()
        a = Image.fromarray(self._goal)
        b = Image.fromarray(np.ascontiguousarray(np.asarray(cur_rgb, np.uint8)[..., :3]))
        with torch.no_grad():
            if self.version == 1:
                warp, certainty = model.match(a, b, device=self.device)
                return warp, certainty, None
            preds = model.match(a, b)
            return preds["warp_AB"][0], preds["overlap_AB"][0].reshape(-1), preds

    def _take(self, warp, certainty, preds, gen, n_request: int | None = None):
        """``(goal_px, cur_px, cert)`` for the correspondences of this frame.

        ``n_request`` oversamples (a masked cell asks for more than ``num_landmarks``
        because the mask is applied afterwards at full pixel resolution)."""
        import torch
        import torch.nn.functional as F

        model = self._build()
        w, h = self.image_wh
        if self._goal_mask is not None:
            # the goal mask on the warp's own grid: certainty outside it is 0, so the
            # sampler never draws a floor pair (a masked cell, never the published one)
            import cv2

            if self.version == 1:
                # RoMa v1's symmetric match stacks A|B side by side: the certainty is
                # (H, 2W); the left half is A -> B (the goal's pixels), the right half
                # B -> A. The goal mask covers the left half; the right half is zeroed,
                # so every sampled pair starts on the goal's object.
                H_, W2 = int(certainty.shape[-2]), int(certainty.shape[-1])
                W_ = W2 // 2 if 2 * H_ == W2 else W2
                m = cv2.resize(self._goal_mask.astype(np.uint8), (W_, H_),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                if W_ != W2:
                    m = np.concatenate([m, np.zeros((H_, W2 - W_), bool)], axis=1)
                mt = torch.as_tensor(m, device=certainty.device)
                mt = mt.reshape((1,) * (certainty.ndim - 2) + (H_, W2))
                certainty = certainty * mt.to(certainty.dtype)
            else:
                hw = tuple(int(v) for v in warp.shape[:2])
                m = cv2.resize(self._goal_mask.astype(np.uint8), (hw[1], hw[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                mt = torch.as_tensor(m, device=warp.device)
                ov = preds["overlap_AB"]
                preds["overlap_AB"] = ov * mt.to(ov.dtype).reshape(ov.shape)
                certainty = preds["overlap_AB"][0].reshape(-1)
        if self.sampling == "pinned":
            # read the warp at the pinned goal pixels: grid_sample over the warp's
            # (H, W, 4) field in normalised coordinates, as RoMa's `match_keypoints` does
            pts = torch.as_tensor(self._pinned, dtype=torch.float32, device=warp.device)
            xn = torch.stack([pts[:, 0] / w * 2 - 1, pts[:, 1] / h * 2 - 1], -1)
            field = warp[..., 2:].permute(2, 0, 1)[None]
            cert_field = certainty.reshape(1, 1, warp.shape[0], warp.shape[1]).float()
            xb = F.grid_sample(field.float(), xn[None, None], align_corners=False,
                               mode="bilinear")[0, :, 0].T
            cert = F.grid_sample(cert_field, xn[None, None], align_corners=False,
                                 mode="bilinear")[0, 0, 0]
            cur_px = torch.stack([(xb[:, 0] + 1) / 2 * w, (xb[:, 1] + 1) / 2 * h], -1)
            return (np.asarray(self._pinned, np.float64), cur_px.cpu().numpy().astype(np.float64),
                    cert.cpu().numpy().astype(np.float64))
        with torch.random.fork_rng(devices=[] if warp.device.type == "cpu" else [warp.device]):
            torch.manual_seed(int(gen))
            n = int(n_request or self.num_landmarks)
            if self.version == 1:
                matches, cert = model.sample(warp, certainty, num=n)
            else:
                out = model.sample(preds, n)
                matches, cert = out[0], out[1]
        ka, kb = model.to_pixel_coordinates(matches, h, w, h, w)
        return (ka.cpu().numpy().astype(np.float64), kb.cpu().numpy().astype(np.float64),
                np.asarray(cert.detach().cpu().numpy(), np.float64).reshape(-1))

    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        if current_mask is not None:
            raise ValueError("RoMaMatcher does not support a per-frame current_mask")
        if self._goal is None:
            raise RuntimeError("reset() must be called before match()")
        self._frames += 1
        warp, certainty, preds = self._dense(current_rgb)
        frame_seed = (episode_seed(*self._episode, self.seed) + self._frames) % 2**31
        masked = self._goal_mask is not None and self.sampling != "pinned"
        n_req = self.num_landmarks * self.MASK_OVERSAMPLE if masked else None
        pts_goal, pts_cur, cert = self._take(warp, certainty, preds, frame_seed, n_req)
        n_raw = int(len(pts_goal))
        n_off_object = 0
        if masked:
            # The certainty is zeroed on the warp's own (coarse) grid, which only biases
            # the draw: a grid cell straddling the object's edge still yields goal pixels
            # off it. The mask therefore bites HERE, at full resolution, and the best
            # `num_landmarks` by certainty survive - so a masked cell really is
            # object-only (measured 2026-09-22: 61 % -> 100 % on shoe2).
            m = self._goal_mask
            u = np.clip(np.round(pts_goal[:, 0]).astype(int), 0, m.shape[1] - 1)
            v = np.clip(np.round(pts_goal[:, 1]).astype(int), 0, m.shape[0] - 1)
            keep_m = m[v, u]
            n_off_object = int((~keep_m).sum())
            pts_goal, pts_cur, cert = pts_goal[keep_m], pts_cur[keep_m], cert[keep_m]
            if len(pts_goal) > self.num_landmarks:
                order = np.lexsort((np.arange(len(cert)), -cert))[: self.num_landmarks]
                pts_goal, pts_cur, cert = pts_goal[order], pts_cur[order], cert[order]
        keep = cert >= self.certainty_min
        pts_goal, pts_cur, cert = pts_goal[keep], pts_cur[keep], cert[keep]
        n_inliers, applied = len(pts_goal), False
        if len(pts_goal):
            pts_goal, pts_cur, cert, n_inliers, applied = filter_by_ransac(
                pts_goal, pts_cur, cert, self.ransac, self.ransac_thresh_px)
        self._frames_ransac_applied += int(applied)
        w, h = self.image_wh
        k = len(pts_goal)
        matches = Matches.from_pixel_pairs(
            pts_goal, pts_cur, side=w, input_size=w, patch_size=1, image_wh=(w, h),
            sim=np.asarray(cert, np.float64) if k else np.zeros(0),
            cyclic=np.full(k, np.nan), margin=np.full(k, np.nan),
            pool_size=n_raw, num_landmarks=self.num_landmarks, draw_seed=self.seed,
            tie_order=self.tie_order)
        matches.extras.update(n_raw=n_raw, n_off_object=n_off_object,
                              n_after_certainty=int(keep.sum()),
                              n_inliers=int(n_inliers), ransac_model=self.ransac,
                              ransac_applied=bool(applied))
        return matches

    def record_fields(self) -> dict:
        return {"front_end": self.name, "roma_version": self.version,
                "roma_sampling": self.sampling, "roma_certainty_min": self.certainty_min,
                "roma_weights_present": self.weights_present(), "matcher_device": self.device,
                "roma_goal_masked": bool(getattr(self, "_goal_mask", None) is not None),
                "roma_mask_oversample": self.MASK_OVERSAMPLE,
                "ransac_model": self.ransac, "ransac_requested": self.ransac_requested,
                "ransac_thresh_px": self.ransac_thresh_px,
                "ransac_frames_applied": int(self._frames_ransac_applied),
                "prealign": getattr(getattr(self, "prealign", None), "label", None)}

    def close(self) -> None:
        self._model = None
        self._goal = None
