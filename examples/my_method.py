"""A complete, minimal visual-servoing method: ORB matches + the classic point IBVS law.

Fly it (one start, a film and a record under runs/):

    python -m vsbench fly --scene poster2 --method my_ibvs --plugin examples/my_method.py \\
        --start-set near60 --limit 1 --max-iter 150 --film

Add ``--mask objects`` (the goal picture's objects) or ``--mask my_mask.png`` to filter the
matches with a mask on the goal picture: it arrives in ``set_target`` as ``goal_mask``.

Copy this file, change get_control_rate, keep the two attributes and four methods below.
For a fuller pipeline (SIFT, ratio test, fundamental-matrix RANSAC for 3-D scenes) see
``vsbench/methods/feature_ibvs.py``.
"""

import cv2
import numpy as np

from vsbench.methods import mask_matches, register


@register("my_ibvs")                      # the name you pass to --method
class MyIBVS:
    wants_depth = "dense"                 # "none", "scalar" (one distance) or "dense" (depth image)
    privileged = False                    # True = "reads the true pose": refused by the benchmark

    def __init__(self, gain=1.0, n_features=1500):
        self.gain = float(gain)
        self.params = {"gain": self.gain, "n_features": n_features}   # written into every record
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def configure(self, K, width, height, dt):
        """Once per episode: the 3x3 intrinsics in pixels, the image size, the control period (s)."""
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def reset(self):
        """Forget everything from the previous episode."""
        self.kp_goal = self.desc_goal = self.goal_mask = None

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None, goal_mask=None):
        """The goal picture (uint8, H x W x 3), plus depth in the form wants_depth asked for;
        goal_mask (bool H x W, True = match here) only with --mask, else None."""
        self.kp_goal, self.desc_goal = self._features(desired_rgb)
        self.goal_mask = goal_mask

    def get_control_rate(self, current_rgb, current_depth=None):
        """One control step: the live picture (and depth) in, a camera velocity out."""
        kp, desc = self._features(current_rgb)
        m = self.bf.match(desc, self.desc_goal) if desc is not None else []
        if len(m) < 8:
            return np.zeros(6), {"lost": True}
        cur = kp[[x.queryIdx for x in m]]
        goal = self.kp_goal[[x.trainIdx for x in m]]
        keep = mask_matches(goal, cur, self.goal_mask)              # all True without --mask
        cur, goal = cur[keep], goal[keep]
        if len(cur) < 8:
            return np.zeros(6), {"lost": True}
        _, mask = cv2.findHomography(cur, goal, cv2.RANSAC, 3.0)   # drop wrong matches (planar model)
        u, v = cur[:, 0].astype(int), cur[:, 1].astype(int)
        Z = current_depth[v, u]                                     # metres, 0 = nothing hit
        ok = (mask.ravel() > 0) & (Z > 0) if mask is not None else Z > 0
        if ok.sum() < 8:
            return np.zeros(6), {"kp_cur": cur, "kp_des": goal, "inliers": ok, "lost": True}

        # point IBVS: s = normalised image coordinates, twist = -gain * pinv(L) @ (s - s*)
        x, y = (cur[ok, 0] - self.cx) / self.fx, (cur[ok, 1] - self.cy) / self.fy
        xd, yd = (goal[ok, 0] - self.cx) / self.fx, (goal[ok, 1] - self.cy) / self.fy
        z, o, zero = Z[ok], np.ones(ok.sum()), np.zeros(ok.sum())
        L = np.stack([np.stack([-o / z, zero, x / z, x * y, -(1 + x * x), y], 1),
                      np.stack([zero, -o / z, y / z, 1 + y * y, -x * y, -x], 1)], 1).reshape(-1, 6)
        e = np.stack([x - xd, y - yd], 1).reshape(-1)
        twist = -self.gain * np.linalg.pinv(L) @ e      # [vx, vy, vz, wx, wy, wz], camera frame
        return twist, {"kp_cur": cur, "kp_des": goal, "inliers": ok,
                       "error_norm": float(np.linalg.norm(e)), "lost": False}

    def _features(self, rgb):
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        kps, desc = self.orb.detectAndCompute(gray, None)
        return np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2), desc
