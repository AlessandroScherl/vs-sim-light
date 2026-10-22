"""Feature-based IBVS with SIFT or ORB: the basic real controller and the template for
student methods. Not privileged: it sees the current image, the desired image and depth
in the form it asked for, nothing else.

``set_target`` detects and describes the desired image once. Every step: detect and
describe the current image, match to the desired descriptors with a ratio test, reject
outliers with RANSAC (a homography with a 3 px threshold by default, a fundamental matrix
for non-planar scenes), read the depth at the inlier keypoints, and apply the point IBVS
law with ``L`` stacked from ``vsbench.interaction.interaction_matrix``:

    twist = -gain * pinv(L) @ (s - s_target)       s in normalised image coordinates

Depth: ``depth="dense"`` (default) reads the current depth image at each keypoint and
drops keypoints without depth; ``depth="scalar"`` uses the distance prior for all. With
fewer than ``min_inliers`` (8) inliers the method returns a zero twist and sets ``lost``.
OpenCV's RANSAC is seeded before every call, so a run is repeatable.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from vsbench.camera import CameraModel
from vsbench.interaction import interaction_matrix, normalised
from vsbench.methods import register

DETECTORS = ("sift", "orb")
RANSAC_MODELS = ("homography", "fundamental")
DEFAULT_RATIO = {"sift": 0.75, "orb": 0.8}
MAX_INFO_MATCHES = 500


class FeatureIBVS:
    privileged = False

    def __init__(
        self,
        detector: str = "sift",
        gain: float = 1.0,
        depth: str = "dense",
        ransac: str = "homography",
        ransac_px: float = 3.0,
        ratio: float | None = None,
        max_features: int = 2000,
        min_inliers: int = 8,
        seed: int = 0,
    ):
        if detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}")
        if depth not in ("dense", "scalar"):
            raise ValueError("depth must be 'dense' or 'scalar'")
        if ransac not in RANSAC_MODELS:
            raise ValueError(f"ransac must be one of {RANSAC_MODELS}")
        if not (np.isfinite(gain) and ransac_px > 0.0 and max_features >= 8 and min_inliers >= 4):
            raise ValueError("bad parameters: need finite gain, ransac_px > 0, max_features >= 8, min_inliers >= 4")
        self.detector = detector
        self.gain = float(gain)
        self.wants_depth = depth
        self.ransac = ransac
        self.ransac_px = float(ransac_px)
        self.ratio = float(DEFAULT_RATIO[detector] if ratio is None else ratio)
        self.max_features = int(max_features)
        self.min_inliers = int(min_inliers)
        self.seed = int(seed)
        self.params = {"detector": detector, "gain": self.gain, "depth": depth, "ransac": ransac, "ransac_px": self.ransac_px, "ratio": self.ratio, "max_features": self.max_features, "min_inliers": self.min_inliers, "seed": self.seed}
        if detector == "sift":
            self._det = cv2.SIFT_create(nfeatures=self.max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self._det = cv2.ORB_create(nfeatures=self.max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.cam: CameraModel | None = None
        self.dt = 0.04
        self.reset()

    # -- contract ---------------------------------------------------------------------
    def configure(self, K: np.ndarray, width: int, height: int, dt: float) -> None:
        self.cam = CameraModel(np.array(K, dtype=np.float64), int(width), int(height))
        self.dt = float(dt)

    def reset(self) -> None:
        self._kp_des: np.ndarray | None = None
        self._desc_des: np.ndarray | None = None
        self._prior: float | None = None

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None) -> None:
        kp, desc = self._describe(desired_rgb)
        if desc is None or len(kp) < self.min_inliers:
            raise ValueError(f"only {len(kp)} {self.detector} keypoints in the desired image")
        self._kp_des, self._desc_des = kp, desc
        self._prior = None if distance_prior is None else float(distance_prior)

    def get_control_rate(self, current_rgb, current_depth=None):
        if self.cam is None or self._desc_des is None:
            raise RuntimeError("feature IBVS needs configure and set_target first")
        t0 = time.perf_counter()
        kp_cur, desc_cur = self._describe(current_rgb)
        idx_cur, idx_des = self._match(desc_cur)
        p_cur, p_des = kp_cur[idx_cur], self._kp_des[idx_des]
        inl = self._ransac(p_cur, p_des)
        Z = self._depths(p_cur, current_depth)
        ok = inl & np.isfinite(Z)
        t1 = time.perf_counter()
        info = {"kp_cur": p_cur, "kp_des": p_des, "inliers": ok, "observer_s": t1 - t0}
        if ok.sum() < self.min_inliers:
            info.update(lost=True, error_norm=float("nan"), controller_s=0.0)
            return np.zeros(6), info
        twist, e_norm = self._law(p_cur[ok], p_des[ok], Z[ok])
        info.update(lost=False, error_norm=e_norm, controller_s=time.perf_counter() - t1)
        return twist, info

    # -- pieces (each is what a student would swap) -------------------------------------
    def _describe(self, rgb) -> tuple[np.ndarray, np.ndarray | None]:
        """Keypoints (N, 2) float pixels and descriptors (N, D) of an RGB image."""
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        kps, desc = self._det.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
        return pts, desc

    def _match(self, desc_cur) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the current and desired keypoints that pass the ratio test."""
        if desc_cur is None or len(desc_cur) < 2:
            return np.zeros(0, int), np.zeros(0, int)
        pairs = self._matcher.knnMatch(desc_cur, self._desc_des, k=2)
        ic, id_ = [], []
        for p in pairs:
            if len(p) < 2:
                continue
            a, b = p
            if b.distance > 0.0 and a.distance < self.ratio * b.distance:
                ic.append(a.queryIdx)
                id_.append(a.trainIdx)
        return np.array(ic, int), np.array(id_, int)

    def _ransac(self, p_cur: np.ndarray, p_des: np.ndarray) -> np.ndarray:
        """Inlier mask (M,) from a homography or a fundamental matrix between the matches."""
        n = len(p_cur)
        need = 4 if self.ransac == "homography" else 8
        if n < max(need, self.min_inliers):
            return np.zeros(n, dtype=bool)
        cv2.setRNGSeed(self.seed)
        a = p_cur.astype(np.float32).reshape(-1, 1, 2)
        b = p_des.astype(np.float32).reshape(-1, 1, 2)
        if self.ransac == "homography":
            _, mask = cv2.findHomography(a, b, cv2.RANSAC, self.ransac_px)
        else:
            _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, self.ransac_px, 0.99)
        if mask is None:
            return np.zeros(n, dtype=bool)
        return mask.reshape(-1).astype(bool)

    def _depths(self, p_cur: np.ndarray, current_depth) -> np.ndarray:
        """Z-depth (M,) per current keypoint: read from the depth image (NaN where there is
        none) or the distance prior for all."""
        n = len(p_cur)
        if self.wants_depth == "scalar":
            if self._prior is None:
                raise RuntimeError("depth='scalar' needs a distance prior from set_target")
            return np.full(n, self._prior)
        if current_depth is None:
            raise RuntimeError("depth='dense' needs the current depth image")
        d = np.asarray(current_depth)
        u = np.clip(np.rint(p_cur[:, 0]).astype(int), 0, d.shape[1] - 1)
        v = np.clip(np.rint(p_cur[:, 1]).astype(int), 0, d.shape[0] - 1)
        Z = d[v, u].astype(np.float64)
        Z[Z <= 0.0] = np.nan
        return Z

    def _law(self, p_cur: np.ndarray, p_des: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, float]:
        """The point IBVS law on matched pixels with their depths: ``(twist, error norm)``."""
        cam = self.cam
        x, y = normalised(cam, p_cur)
        xd, yd = normalised(cam, p_des)
        e = np.stack([x - xd, y - yd], axis=1).reshape(-1)
        L = interaction_matrix(cam, p_cur, Z).reshape(-1, 6)
        twist = -self.gain * (np.linalg.pinv(L) @ e)
        return twist, float(np.linalg.norm(e))


@register("sift_ibvs")
class SiftIBVS(FeatureIBVS):
    def __init__(self, **kw):
        super().__init__(detector="sift", **kw)


@register("orb_ibvs")
class OrbIBVS(FeatureIBVS):
    def __init__(self, **kw):
        super().__init__(detector="orb", **kw)
