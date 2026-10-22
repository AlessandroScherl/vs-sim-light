# vs-sim copy: vssim/baselines/lightglue.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""SuperPoint + LightGlue as a matcher: the learned sparse baseline (ICCV 2023).

Purpose
-------
LightGlue (Lindenberger, Sarlin, Pollefeys, ICCV 2023) is the standard learned
keypoint matcher: SuperPoint detects up to ``max_num_keypoints`` corners in each
picture, LightGlue - an attention network - pairs them and gives each pair a
confidence. This module runs the **authors' own package** (``lightglue``, the
implementation the paper's numbers were measured with) and returns its matches in
the bench's :class:`vsbench.vs.baselines.match.Matches` shape on the pixel lattice, so
the same IBVS law (the bench's, or ViSP's) and the same RANSAC switch as the
classical detectors apply. It is the sparse-learned baseline beside RoMa (dense
learned) and SIFT (classical).

Settings (all recorded)
-----------------------
``max_num_keypoints`` (2048, the paper's evaluation budget), ``detection_threshold``,
``nms_radius``, ``resize`` (the long edge SuperPoint runs at; 1024 is the package's
default and the paper's; ``null`` = native 1920), ``filter_threshold``,
``depth_confidence`` / ``width_confidence`` (LightGlue's adaptive early exits),
``ransac`` as on the classical arms, ``num_landmarks`` (the best K by confidence
that the controller steers on; ``0`` = every match) and ``device``.

Fences
------
- **Weights are never downloaded.** ``superpoint_v1.pth`` and
  ``superpoint_lightglue_v0-1_arxiv.pth`` must be in ``$TORCH_HOME/hub/checkpoints``;
  both are md5-checked once at build (:data:`LIGHTGLUE_WEIGHTS`).
- **The goal is extracted once per episode** (:meth:`reset`), the live frame once
  per step; keypoints come back in native pixels whatever ``resize`` is.
- **No margin, no trust**: ``provides_margin`` / ``provides_conf_weight`` are
  ``False`` as on every keypoint arm; the arm file says ``weighting: uniform``.

Provenance
----------
The study candidate of the Refine-VS research repository
(``refinevs/eval/matching_study/published.py::LightGlueCandidate``), re-shaped as a
matcher that flies.

Golden tests: ``tests/fast/test_lightglue.py`` (a stub extractor/matcher through the
whole seam; the real package where it is installed).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from vsbench.vs.baselines.classical import RANSAC_CHOICES, filter_by_ransac, resolve_ransac
from vsbench.vs.baselines.match import CorrespondenceMatcher, Matches

#: LightGlue's published defaults, read from the installed package, and the one number
#: the LightGlue paper's evaluation states (``max_num_keypoints``).
LIGHTGLUE_DEFAULTS: dict[str, Any] = {
    "max_num_keypoints": 2048,
    "detection_threshold": 0.0005,
    "nms_radius": 4,
    "resize": 1024,
    "filter_threshold": 0.1,
    "depth_confidence": 0.95,
    "width_confidence": 0.99,
    "features": "superpoint",
}

#: The checkpoints, the file names torch.hub gives them, their md5s.
LIGHTGLUE_WEIGHTS: dict[str, dict[str, str]] = {
    "superpoint_v1.pth": {
        "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth",
        "md5": "938af9f432d327751dcbc0d6c7a0448b"},
    "superpoint_lightglue_v0-1_arxiv.pth": {
        "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth",
        "md5": "af882c2bc45f71b68139a4631ec6cf78"},
}

LIGHTGLUE_KEYS = frozenset({*LIGHTGLUE_DEFAULTS, "variant", "num_landmarks", "device",
                            "ransac", "ransac_thresh_px", "tie_order", "seed"})


def _to_torch(rgb: np.ndarray):
    """HWC uint8 -> CHW float in [0, 1] (the package's ``numpy_image_to_torch``, local so a
    stub extractor needs no package)."""
    import torch

    arr = np.asarray(rgb)[..., :3].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def _rbd(data: dict) -> dict:
    """Drop the batch dimension of every entry of ``data`` (the package's ``rbd``): a
    tensor's first axis, or the first item of a per-image list such as ``matches``."""
    return {k: (v[0] if isinstance(v, (list, tuple)) or (hasattr(v, "shape") and len(v.shape) > 0)
                else v) for k, v in data.items()}


def _keep_in_mask(feats: dict, mask: np.ndarray) -> tuple[dict, int]:
    """Keep the keypoints (native pixels) whose pixel lies inside the bool mask; every
    per-keypoint tensor of the batched feature dict is filtered alike."""
    import torch

    m = np.asarray(mask).astype(bool)
    kp = feats["keypoints"][0].detach().cpu().numpy()
    u = np.clip(np.round(kp[:, 0]).astype(int), 0, m.shape[1] - 1)
    v = np.clip(np.round(kp[:, 1]).astype(int), 0, m.shape[0] - 1)
    keep = torch.as_tensor(m[v, u], device=feats["keypoints"].device)
    n = int(kp.shape[0])
    out = {}
    for k, val in feats.items():
        if hasattr(val, "shape") and len(val.shape) >= 2 and val.shape[1] == n:
            out[k] = val[:, keep]
        else:
            out[k] = val
    return out, n - int(keep.sum())


def torch_hub_dir() -> Path:
    return Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch")) / "hub"


def lightglue_available() -> tuple[bool, str]:
    """``(ok, reason)``: the package imports and both checkpoints are on disk."""
    try:
        import lightglue  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - env dependent
        return False, (f"the `lightglue` package is not importable ({exc}); install "
                       "github.com/cvg/LightGlue (commit eb42fee2) with --no-deps")
    missing = [n for n in LIGHTGLUE_WEIGHTS
               if not (torch_hub_dir() / "checkpoints" / n).exists()]
    if missing:
        return False, (f"LightGlue's checkpoints are not in {torch_hub_dir()}/checkpoints: "
                       f"{', '.join(missing)}; a run never downloads")
    return True, ""


def checked_weights() -> dict[str, str]:
    """md5 both checkpoints against :data:`LIGHTGLUE_WEIGHTS`; raise on a mismatch."""
    got = {}
    for name, want in LIGHTGLUE_WEIGHTS.items():
        path = torch_hub_dir() / "checkpoints" / name
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != want["md5"]:
            raise RuntimeError(f"LightGlue checkpoint {path} has md5 {digest}, not the "
                               f"recorded {want['md5']} ({want['url']})")
        got[name] = digest
    return got


class LightGlueMatcher(CorrespondenceMatcher):
    """SuperPoint + LightGlue -> the best ``num_landmarks`` matches, optional RANSAC.

    ``models`` injects a ready-made ``(extractor, matcher)`` pair (the fast tier's
    stubs); otherwise the package's are built lazily on the first :meth:`reset`.
    """

    name = "superpoint_lightglue"
    honours_goal_mask = True      # goal keypoints outside the mask are dropped at reset
    provides_margin = False
    provides_conf_weight = False

    def __init__(self, *, num_landmarks: int, device: str, seed: int = 0, ransac: str = "none",
                 ransac_thresh_px: float = 3.0, scene_spec: Any = None,
                 tie_order: str = "stable", models: tuple | None = None, **conf: Any) -> None:
        unknown = sorted(set(conf) - set(LIGHTGLUE_DEFAULTS))
        if unknown:
            raise ValueError(f"unknown LightGlue settings: {unknown}")
        if ransac not in RANSAC_CHOICES:
            raise ValueError(f"ransac must be one of {RANSAC_CHOICES}, got {ransac!r}")
        self.num_landmarks = int(num_landmarks)
        self.device = str(device)
        self.seed = int(seed)          # recorded only: LightGlue draws nothing
        self.conf = {**LIGHTGLUE_DEFAULTS, **conf}
        self.ransac = resolve_ransac(ransac, scene_spec)
        self.ransac_requested = str(ransac)
        self.ransac_thresh_px = float(ransac_thresh_px)
        self.tie_order = str(tie_order)
        self._models = models
        self.weights_md5: dict[str, str] = {}
        self._goal_feats = None
        self.image_wh: tuple[int, int] | None = None
        self._frames = 0
        self._frames_ransac_applied = 0

    # ------------------------------------------------------------------ build
    def _build(self):
        if self._models is None:
            ok, why = lightglue_available()
            if not ok:
                raise RuntimeError(why)
            from lightglue import LightGlue, SuperPoint

            self.weights_md5 = checked_weights()
            extractor = SuperPoint(
                max_num_keypoints=int(self.conf["max_num_keypoints"]),
                detection_threshold=float(self.conf["detection_threshold"]),
                nms_radius=int(self.conf["nms_radius"])).eval().to(self.device)
            matcher = LightGlue(
                features=str(self.conf["features"]),
                filter_threshold=float(self.conf["filter_threshold"]),
                depth_confidence=float(self.conf["depth_confidence"]),
                width_confidence=float(self.conf["width_confidence"])).eval().to(self.device)
            self._models = (extractor, matcher)
        return self._models

    def _extract(self, rgb: np.ndarray):
        extractor, _ = self._build()
        import torch

        resize = self.conf["resize"]
        with torch.no_grad():
            return extractor.extract(_to_torch(rgb).to(self.device),
                                     resize=None if resize is None else int(resize))

    # ---------------------------------------------------------------- episode
    def set_episode(self, scene: str, start_id: Any) -> None:
        self._episode = (scene, start_id)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        h, w = np.asarray(goal_rgb).shape[:2]
        self.image_wh = (int(w), int(h))
        self._goal_feats = self._extract(goal_rgb)
        self.n_goal_masked_out = 0
        if goal_mask is not None:
            self._goal_feats, self.n_goal_masked_out = _keep_in_mask(self._goal_feats, goal_mask)
        self._frames = 0
        self._frames_ransac_applied = 0

    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        if current_mask is not None:
            raise ValueError("LightGlueMatcher does not support a per-frame current_mask")
        if self._goal_feats is None:
            raise RuntimeError("reset() must be called before match()")
        _, matcher = self._build()
        import torch

        self._frames += 1
        f1 = self._extract(current_rgb)
        with torch.no_grad():
            out = _rbd(matcher({"image0": self._goal_feats, "image1": f1}))
        k0 = _rbd(self._goal_feats)["keypoints"].detach().cpu().numpy()
        k1 = _rbd(f1)["keypoints"].detach().cpu().numpy()
        idx = out["matches"].detach().cpu().numpy().reshape(-1, 2)
        conf = out["scores"].detach().cpu().numpy().reshape(-1).astype(np.float64)
        n_raw = int(len(idx))
        pts_goal, pts_cur = k0[idx[:, 0]].astype(np.float64), k1[idx[:, 1]].astype(np.float64)
        # the best K by LightGlue's own confidence, stable on ties
        if self.num_landmarks and n_raw > self.num_landmarks:
            order = np.lexsort((np.arange(n_raw), -conf))[: self.num_landmarks]
            pts_goal, pts_cur, conf = pts_goal[order], pts_cur[order], conf[order]
        n_inliers, applied = len(pts_goal), False
        if len(pts_goal):
            pts_goal, pts_cur, conf, n_inliers, applied = filter_by_ransac(
                pts_goal, pts_cur, conf, self.ransac, self.ransac_thresh_px)
        self._frames_ransac_applied += int(applied)
        w, h = self.image_wh
        k = len(pts_goal)
        matches = Matches.from_pixel_pairs(
            pts_goal, pts_cur, side=w, input_size=w, patch_size=1, image_wh=(w, h),
            sim=np.asarray(conf, np.float64) if k else np.zeros(0),
            cyclic=np.full(k, np.nan), margin=np.full(k, np.nan),
            pool_size=n_raw, num_landmarks=self.num_landmarks, draw_seed=None,
            tie_order=self.tie_order)
        matches.extras.update(n_keypoints_goal=int(len(k0)), n_keypoints_cur=int(len(k1)),
                              n_raw=n_raw, n_inliers=int(n_inliers),
                              ransac_model=self.ransac, ransac_applied=bool(applied))
        return matches

    def record_fields(self) -> dict:
        return {"front_end": self.name, "lg_max_keypoints": int(self.conf["max_num_keypoints"]),
                "lg_resize": self.conf["resize"],
                "lg_filter_threshold": self.conf["filter_threshold"],
                "lg_weights_md5": dict(self.weights_md5), "matcher_device": self.device,
                "lg_goal_masked_out": int(getattr(self, "n_goal_masked_out", 0)),
                "ransac_model": self.ransac, "ransac_requested": self.ransac_requested,
                "ransac_thresh_px": self.ransac_thresh_px,
                "ransac_frames_applied": int(self._frames_ransac_applied),
                "prealign": getattr(getattr(self, "prealign", None), "label", None)}

    def close(self) -> None:
        self._models = None
        self._goal_feats = None
