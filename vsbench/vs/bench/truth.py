# vs-sim copy: vssim/bench/truth.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The mock plane's own truth matcher - an ``oracle``-kind arm for testing the bench.

:class:`PlaneTruthMatcher` answers the correspondence question analytically for
:class:`vsbench.vs.scene.MockPlaneScene`, so an episode can close the loop with zero
matching error and the bench's tests measure the bench rather than a matcher. It
is **ground truth in the control path**: an arm built on it must declare
``arm: {kind: oracle}`` (or ``test``), and its records carry the ``oracle:`` label
prefix so they can never be read as a method's numbers.

Provenance: the Refine-VS research repository's ``refinevs/eval/scene.py``
(``PlaneTruthMatcher``), moved out of the simulator core because it speaks the
matcher format of :mod:`vsbench.vs.baselines.match`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vsbench.vs.baselines.match import CorrespondenceMatcher, Matches, episode_seed
from vsbench.vs.scene import MockPlaneScene


class PlaneTruthMatcher(CorrespondenceMatcher):
    """Error-free correspondences on a :class:`MockPlaneScene` - the mock's own geometry.

    Draws ``num_landmarks`` goal patches (seeded by
    :func:`vsbench.vs.baselines.match.episode_seed`, so it is paired across arms exactly like
    the real draw) and maps each one through the plane into the current view. The
    result is a :class:`vsbench.vs.baselines.match.Matches` in the fielded shape: an integer
    ``cur_patch`` plus the sub-patch remainder in ``offsets_patch``, so the runner and
    the controller see what a real matcher gives them.

    ``pixel_noise`` gives each landmark a fixed pixel bias, drawn once per episode
    in :meth:`reset` from the same episode seed. That is the shape of the real
    error - a *stable* per-landmark localisation offset, not per-frame jitter - so
    the loop parks at a slightly wrong pose and its feature error plateaus at a
    floor instead of decaying forever. Without it the mock is too good for a
    plateau stop to have anything to fire on.
    """

    name = "plane_truth"

    def __init__(self, scene: MockPlaneScene, *, num_landmarks: int, seed: int,
                 input_size: int = 512, side: int = 32, pixel_noise: float = 0.0) -> None:
        self.scene = scene
        self.num_landmarks = int(num_landmarks)
        self.seed = int(seed)
        self.input_size = int(input_size)
        self.side = int(side)
        self.patch_size = self.input_size // self.side
        self.pixel_noise = float(pixel_noise)
        self._goal_patch: np.ndarray | None = None
        self._goal_px: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._episode: tuple[str, Any] | None = None

    def set_episode(self, scene: str, start_id: Any) -> None:
        self._episode = (scene, start_id)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        """Pick the landmark patches once per episode, on the goal view's lattice."""
        h, w = np.asarray(goal_rgb).shape[:2]
        self.image_wh = (int(w), int(h))
        scene, start_id = self._episode or (self.scene.name, 0)
        rng = np.random.default_rng(episode_seed(scene, start_id, self.seed))
        # Only patches whose centre actually sees the plane can be landmarks.
        grid = np.stack(np.meshgrid(np.arange(self.side), np.arange(self.side),
                                    indexing="xy"), axis=-1).reshape(-1, 2)
        centres = grid * (self.input_size / self.side) + self.patch_size / 2.0
        px = centres * np.asarray([w / self.input_size, h / self.input_size])
        _, ok = self.scene.backproject(px, self.scene.goal_pose)
        usable = np.flatnonzero(ok)
        if len(usable) < self.num_landmarks:
            raise ValueError(
                f"the goal view sees only {len(usable)} usable patches, "
                f"fewer than num_landmarks={self.num_landmarks}")
        pick = rng.permutation(usable)[:self.num_landmarks]
        self._goal_patch = grid[pick]
        self._goal_px = px[pick]
        self._bias = (None if not self.pixel_noise
                      else rng.normal(scale=self.pixel_noise, size=(len(pick), 2)))

    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        if current_mask is not None:
            raise ValueError("PlaneTruthMatcher does not honour a mask")
        if self._goal_patch is None:
            raise ValueError("call reset(goal_rgb) before match()")
        uv, valid = self.scene.gt_correspondence(self._goal_px, self.scene.pose)
        if self._bias is not None:
            uv = uv + self._bias
        keep = np.flatnonzero(valid)
        k = len(keep)
        # full-image pixel -> (integer patch, sub-patch offset): `Matches.from_pixels`
        return Matches.from_pixels(
            self._goal_patch[keep], uv[keep],
            side=self.side, input_size=self.input_size,
            patch_size=self.patch_size, image_wh=self.image_wh,
            sim=np.ones(k), cyclic=np.ones(k), margin=np.full(k, 0.5),
            pool_size=int(self.side ** 2), num_landmarks=self.num_landmarks,
            draw_seed=self.seed, tie_order="stable", trust=np.ones(k),
        )
