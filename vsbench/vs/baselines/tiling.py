# vs-sim copy: vssim/baselines/tiling.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""ART-VS: adaptive resolution tiling over a frozen ViT (unit U3).

Purpose
-------
The published ART-VS baseline (*Adaptive Resolution Tiling for Vision Transformer
Visual Servoing*, IROS 2026) is ViT-VS plus one idea: run at the backbone's native
resolution while the camera is far from the goal, then - once the error has fallen
enough - switch to a **tiled** pass that matches tile *i* of the goal against tile
*i* of the current frame, so the final alignment is read at high resolution
without ever feeding the backbone more pixels than it takes.

Why the switch is what makes the method *correct*, not merely faster
--------------------------------------------------------------------
Tile-to-tile matching assumes the two views are already roughly aligned, which is
exactly the regime the switch guarantees: it fires only after the error proxy has
dropped by ~80 % (:data:`DEFAULT_SWITCH_THRESHOLD`). Tiling from step 0 matches
unrelated image regions. The switch **latches**: once tiled, always tiled.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
- ``src/control/matchers/vit_tiled.py`` -> :class:`TiledMatcher`: the published
  defaults (36 tiles, 1344 tiling resolution, 224 ViT input, tau = 0.20 - taken
  from ART-VS's own ``config_dinov2_224_6x6_tiling.yaml``), the median-pixel-
  displacement error proxy, the latch, the per-tile pair budget, and the fail-open
  back to the coarse result when a tiled pass finds nothing.
- ``src/control/_vitvs_vendor/.../features/tiling.py`` -> :class:`TilingConfig`
  and :class:`TilingHandler` (``create_tiles``, ``transform_patch_to_global``).
  ``calculate_uv_tiled`` is dropped: ``vit_tiled`` never called it, and it rounds
  to integer pixels, which protocol v2 section 4 removed from every arm.

The coarse phase is **not reimplemented**: :class:`TiledMatcher` holds a
:class:`vsbench.vs.baselines.match.DenseMatcher` and calls it, so the mutual pool, the seeded
draw and the patch-to-pixel mapping are the ones U2 already proved equal.

Two coordinate traps, both live
-------------------------------
1. **``TilingConfig.patch_size`` was hardcoded to 14** (DINOv2) at
   ``tiling.py:30``, and ``vit_tiled.py:99`` had to overwrite it from the live
   extractor; without that every ART-VS point lands in the wrong place on a
   patch-16 backbone (``../INVENTORY.md`` load-bearing hack 5). Here
   :class:`TilingConfig` takes ``patch_size`` as a **required argument with no
   default**, and :class:`TiledMatcher` passes the backbone's. The hardcode cannot
   come back.
2. **``transform_patch_to_global`` takes ``(y, x)``** while the matcher works in
   ``(column, row)``. The order is flipped on the way in and back on the way out,
   so the :class:`vsbench.vs.baselines.match.Matches` contract (full-image ``(column, row)``)
   is honoured.

Deliberately dropped
--------------------
RANSAC (the old ``ARTVSMatcher`` forwarded a ``ransac`` kwarg to its dense parent;
ViT-VS/ART-VS are RANSAC-free by the standing owner ruling and protocol v2 section
4, so the knob is gone rather than defaulted to ``none``), ``freeze_landmarks``
(the N1a ``frz`` stop mode, which this class refused anyway), and the
``TilingHandler`` file-loading and mask helpers with no consumer.

Not here yet
------------
The **4-way pre-alignment** (``src/control/prealign.py``) is unit U8. Protocol v2
section 4 runs the published ViT methods with it "available in pre-registered
comparison cells and its presence stated in every table", so
:attr:`TiledMatcher.prealign` is the hook where it will attach and
:meth:`TiledMatcher.record_fields` already reports ``prealign: null``, i.e. a
table built from these records says out loud that the published crutch was off.

Golden tests that guard it
--------------------------
``tests/fast/test_tiling.py`` against ``tests/fixtures/u3_artvs_dinov3.json``
(produced once by the frozen old ``ARTVSMatcher`` on CPU - see
``u3_fixture_provenance.md``): the tile grid and its coordinates, one coarse frame
and one tiled frame reproduced point for point, the latch firing at 0.20 x on a
synthetic error trace, and the patch-size fence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from vsbench.vs.baselines.match import (
    DEFAULT_TIE_ORDER,
    MISSING_DEVICE,
    CorrespondenceMatcher,
    DenseMatcher,
    Matches,
    device_of,
    mutual_nn_match,
)

__all__ = [
    "DEFAULT_SWITCH_THRESHOLD",
    "DEFAULT_TILE_INPUT_RESOLUTION",
    "DEFAULT_TILE_NUMBER",
    "DEFAULT_VIT_INPUT_SIZE",
    "TiledMatcher",
    "TilingConfig",
    "TilingHandler",
]

#: Paper default: switch once the error proxy has fallen to this fraction of its
#: first value. The latch is what makes tile-to-tile matching sound.
DEFAULT_SWITCH_THRESHOLD = 0.20

#: ART-VS's own benchmark config (``config_dinov2_224_6x6_tiling.yaml``): 36 tiles
#: (6x6) of a 1344-pixel square, each read at the backbone's native 224 - which is
#: the whole point of the method. The 4-tile / 896 defaults an earlier version of
#: the old file carried were invented, not published.
DEFAULT_TILE_NUMBER = 36
DEFAULT_TILE_INPUT_RESOLUTION = 1344
DEFAULT_VIT_INPUT_SIZE = 224


class TilingConfig:
    """Geometry of the tiling: a perfect-square grid over a square resize.

    ``patch_size`` is **required and has no default** - trap 1 in the module
    docstring. It must be the live backbone's, because every tile-local patch
    coordinate is turned into pixels with it.
    """

    def __init__(self, *, patch_size: int, slice_number: int = DEFAULT_TILE_NUMBER,
                 input_resolution: int = DEFAULT_TILE_INPUT_RESOLUTION,
                 vit_input_size: int = DEFAULT_VIT_INPUT_SIZE) -> None:
        grid_size = int(math.isqrt(int(slice_number)))
        if grid_size * grid_size != int(slice_number):
            raise ValueError(
                f"slice_number must be a perfect square (4, 9, 16, 25...), got {slice_number}")
        if int(input_resolution) % grid_size:
            raise ValueError(
                f"input_resolution ({input_resolution}) must be divisible by "
                f"sqrt(slice_number) ({grid_size})")
        if int(vit_input_size) % int(patch_size):
            raise ValueError(
                f"vit_input_size ({vit_input_size}) must be divisible by patch_size "
                f"({patch_size}) or the tile's token grid is not square")
        self.slice_number = int(slice_number)
        self.grid_size = grid_size
        self.input_resolution = int(input_resolution)
        self.slice_resolution = int(input_resolution) // grid_size
        self.vit_input_size = int(vit_input_size)
        self.patch_size = int(patch_size)
        self.patches_per_vit_input = int(vit_input_size) // int(patch_size)
        #: cells across the whole tiled image - the effective resolution the method buys
        self.effective_patches = grid_size * self.patches_per_vit_input

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"TilingConfig(slice_number={self.slice_number}, "
            f"input_resolution={self.input_resolution}, "
            f"vit_input_size={self.vit_input_size}, patch_size={self.patch_size}, "
            f"effective_patches={self.effective_patches})"
        )


class TilingHandler:
    """Slice an image into tiles and map a tile-local cell back to the full image."""

    def __init__(self, config: TilingConfig) -> None:
        self.config = config

    def create_tiles(
        self, image_rgb_u8: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
        """Resize to the tiling resolution, then slice row-major into square tiles.

        Returns ``(tiles, tile_coords)`` where ``tile_coords[i]`` is the tile's
        top-left ``(y, x)`` **in the tiling resolution**, the order
        ``transform_patch_to_global`` indexes.
        """
        from PIL import Image

        res = self.config.input_resolution
        resized = np.asarray(Image.fromarray(np.asarray(image_rgb_u8)).resize((res, res)))
        step = self.config.slice_resolution
        tiles, coords = [], []
        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                y, x = i * step, j * step
                tiles.append(resized[y:y + step, x:x + step])
                coords.append((y, x))
        return tiles, coords

    def transform_patch_to_global(self, patch_yx: np.ndarray, tile_idx: int,
                                  tile_coords: list[tuple[int, int]]) -> np.ndarray:
        """Tile-local cell ``(y, x)`` -> ``(y, x)`` pixels in the tiling resolution.

        Verbatim, including the **integer** ``patch_size // 2`` half-cell offset (the
        dense path's is ``patch_size / 2.0``; they agree for every even patch size,
        and 14 and 16 are the only ones this repository loads).
        """
        tile_y, tile_x = tile_coords[tile_idx]
        scale = self.config.slice_resolution / self.config.vit_input_size
        p = np.asarray(patch_yx, np.float64).reshape(-1, 2)
        local = p * self.config.patch_size + self.config.patch_size // 2
        return np.column_stack([tile_y + local[:, 0] * scale, tile_x + local[:, 1] * scale])


class TiledMatcher(CorrespondenceMatcher):
    honours_goal_mask = True      # forwarded to the coarse DenseMatcher at reset
    """ART-VS: :class:`vsbench.vs.baselines.match.DenseMatcher` coarse, then a tiled pass.

    ``use_tiling=False`` reduces this to plain ViT-VS at the ART-VS input size -
    the paper's own ``*_no_tiling`` ablation - and is recorded as such.

    ``backbone_name`` defaults to the published ``dinov2_vits14`` chain
    (:class:`vsbench.vs.baselines.features.PublishedDinoV2Backbone`, torch.hub, log-binned
    3456-d). Its weights are in the laptop's torch.hub cache and are never
    downloaded. The parity fixture is nevertheless on **DINOv3**, deliberately: a
    patch-**16** backbone is the case ``TilingConfig.patch_size = 14`` broke, and on
    DINOv2 the fence would have 14 on both sides and prove nothing. The arithmetic
    under test is this module's; the backbone contract is the one U1 pins.

    **The tiled frame's landmark count is not ``num_landmarks``.** The per-tile
    budget is the published ``max(1, round(K / n_tiles))``, applied to every tile, so
    a tiled frame carries ``n_tiles * max(1, round(K / n_tiles))`` rows: at 36 tiles,
    K = 24 -> 36, K = 48 -> 36, K = 96 -> 108. Verbatim old behaviour, and the record's
    ``num_landmarks`` still says K - read ``n_matches`` for what a frame actually
    steered on, exactly as on a CNS row.
    """

    #: The tiled pass computes a real second-peak margin, but only **inside its own
    #: tile**, so a margin weight would compare rows scored over 14x14 cells with
    #: coarse rows scored over the whole frame. The published arm is unweighted;
    #: :func:`vsbench.vs.bench.arms.build_arm` enforces it.
    provides_margin = False

    #: The tiled cosine is tile-local too - one pair per tile means every row is its
    #: own tile's best, so the numbers are not comparable across rows and a trust
    #: weight built from them would be arbitrary. ``build_arm`` refuses
    #: ``controller.weighting.use_trust`` on this matcher for the same reason.
    provides_conf_weight = False

    def __init__(
        self,
        *,
        num_landmarks: int | None = None,
        seed: int | None = None,
        backbone=None,
        backbone_name: str = "dinov2_vits14",
        device: str | None = None,
        threshold: float = 1.0,
        tie_order: str = DEFAULT_TIE_ORDER,
        use_tiling: bool = True,
        tile_number: int = DEFAULT_TILE_NUMBER,
        tile_input_resolution: int = DEFAULT_TILE_INPUT_RESOLUTION,
        vit_input_size: int | None = None,
        switch_threshold: float = DEFAULT_SWITCH_THRESHOLD,
        switch_iteration: int = 0,
        prealign: Any = None,
    ) -> None:
        if prealign is not None:
            raise NotImplementedError(
                "the 4-way pre-alignment is not a matcher knob: it is a PRE-SERVO "
                "physical re-pose of the camera, so it is a `rotation:` config block "
                "(`rotation: {mode: prealign_4way}`) built by vsbench.vs.baselines.prealign."
                "build_rotation and applied by the runner before the loop. Protocol v2 "
                "section 4 runs it only in pre-registered comparison cells and states "
                "its presence in every table, so it is never a silent default."
            )
        if backbone is not None and vit_input_size is not None \
                and int(backbone.input_size) != int(vit_input_size):
            # An injected encoder wins silently over the config otherwise, and every tile
            # would then be read at a resolution the config did not ask for.
            raise ValueError(
                f"vit_input_size {vit_input_size} but the injected backbone runs at "
                f"{backbone.input_size}: ART-VS reads every tile at the backbone's input size")
        if backbone is None:
            if device is None:
                raise ValueError(MISSING_DEVICE)
            from vsbench.vs.baselines.features import build_backbone  # noqa: PLC0415 - weights stay lazy

            backbone = build_backbone(
                backbone_name, device=device,
                input_size=int(vit_input_size or DEFAULT_VIT_INPUT_SIZE))
        self.backbone = backbone
        #: the coarse phase, and the owner of the goal tokens, the pool and the draw
        self.coarse = DenseMatcher(
            num_landmarks=num_landmarks, seed=seed, backbone=backbone,
            threshold=threshold, tie_order=tie_order)
        self.num_landmarks = self.coarse.num_landmarks
        self.seed = self.coarse.seed
        self.threshold = float(threshold)
        self.tie_order = tie_order
        self.use_tiling = bool(use_tiling)
        self.switch_threshold = float(switch_threshold)
        self.switch_iteration = int(switch_iteration)
        self.prealign = None
        self.name = f"artvs_{backbone.name}"
        self.config = TilingConfig(
            patch_size=int(backbone.patch_size),   # trap 1: from the backbone, never 14
            slice_number=tile_number, input_resolution=tile_input_resolution,
            vit_input_size=int(backbone.input_size))
        self.handler = TilingHandler(self.config)
        self.tile_coords: list[tuple[int, int]] = []
        self._goal_tiles: list[torch.Tensor] = []
        self.image_wh: tuple[int, int] | None = None
        self.tiled = False
        self._e0: float | None = None
        self._iteration = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"TiledMatcher(num_landmarks={self.num_landmarks}, "
            f"tiles={self.config.slice_number}, res={self.config.input_resolution}, "
            f"vit={self.config.vit_input_size}, tau={self.switch_threshold}, "
            f"use_tiling={self.use_tiling}, backbone={self.backbone.name!r})"
        )

    # ----------------------------------------------------------------- episode
    def set_episode(self, scene: str, start_id: Any) -> None:
        self.coarse.set_episode(scene, start_id)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        """Cache the coarse goal tokens and the per-tile goal tokens; clear the latch."""
        self.coarse.reset(goal_rgb, goal_depth, goal_mask)
        h, w = goal_rgb.shape[:2]
        self.image_wh = (int(w), int(h))
        self.tiled = False
        self._e0 = None
        self._iteration = 0
        self.tile_coords = []
        self._goal_tiles = []
        if self.use_tiling:
            tiles, self.tile_coords = self.handler.create_tiles(goal_rgb)
            self._goal_tiles = [self.backbone.encode(t) for t in tiles]

    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        """Coarse match, then - once latched - the tiled pass, failing open to coarse."""
        self._iteration += 1
        coarse = self.coarse.match(current_rgb, current_mask)
        if not self.use_tiling:
            coarse.extras.update(matcher=self.name, mode="coarse", tiling="disabled")
            return coarse

        self._update_switch(coarse)
        if not self.tiled:
            coarse.extras.update(matcher=self.name, mode="coarse", switched=False,
                                 error_ratio=self._error_ratio(coarse))
            return coarse

        tiled = self._match_tiled(current_rgb)
        if not len(tiled):
            # Fail open to the coarse result rather than stalling the servo.
            coarse.extras.update(matcher=self.name, mode="coarse_fallback", switched=True,
                                 reason="tiled pass found no correspondences")
            return coarse
        tiled.extras.update(matcher=self.name, mode="tiled", switched=True)
        return tiled

    def record_fields(self) -> dict:
        """The published knobs this arm ran under - never a measurement."""
        return {
            "matcher_device": device_of(self.backbone),
            "artvs_tiles": self.config.slice_number,
            "artvs_tile_resolution": self.config.input_resolution,
            "artvs_vit_input_size": self.config.vit_input_size,
            "artvs_switch_threshold": self.switch_threshold,
            "artvs_switch_iteration": self.switch_iteration,
            "artvs_use_tiling": self.use_tiling,
            "artvs_switched": self.tiled,
            # stated so a table cannot imply the published pre-servo roll ran; the
            # label lookup is `None` for a matcher with no pre-aligner attached (U8).
            "prealign": getattr(self.prealign, "label", None),
        }

    def close(self) -> None:
        self.coarse.close()

    # --------------------------------------------------------------- the latch
    @staticmethod
    def error_proxy(matches: Matches) -> float:
        """Median pixel displacement between paired points - a stand-in for ``||e||``.

        It needs no intrinsics and no depth, so the switch policy stays inside the
        matcher instead of leaking control state into it. Median, not mean, so a few
        bad matches cannot trip the latch early.
        """
        if not len(matches):
            return float("nan")
        rows = matches.to_rows()
        return float(np.median(np.linalg.norm(rows["pts_cur"] - rows["pts_goal"], axis=1)))

    def _error_ratio(self, matches: Matches) -> float:
        e = self.error_proxy(matches)
        if self._e0 is None or not np.isfinite(e) or self._e0 <= 0:
            return float("nan")
        return e / self._e0

    def _update_switch(self, coarse: Matches) -> None:
        """Latch coarse -> tiled. Once tiled, always tiled."""
        if self.tiled:
            return
        if self.switch_iteration > 0:
            self.tiled = self._iteration >= self.switch_iteration
            return
        e = self.error_proxy(coarse)
        if not np.isfinite(e):
            return
        if self._e0 is None:
            self._e0 = e            # the first frame defines "full" error
            return
        if self._e0 > 0 and (e / self._e0) < self.switch_threshold:
            self.tiled = True

    # ---------------------------------------------------------- the tiled pass
    def _match_tiled(self, current_rgb: np.ndarray) -> Matches:
        """Match goal tile *i* against current tile *i*, then lift to full-image pixels."""
        cur_tiles, _ = self.handler.create_tiles(current_rgb)
        n_tiles = min(len(cur_tiles), len(self._goal_tiles))
        # Spread the landmark budget across the tiles, at least one pair each. Note the
        # rounding: the frame ends up with `n_tiles * per_tile` rows, not `num_landmarks`
        # (36 tiles: K = 24 -> 36, K = 48 -> 36, K = 96 -> 108). Published behaviour; the
        # class docstring states it, and the record's `num_landmarks` still says K.
        per_tile = max(1, int(round(self.num_landmarks / max(1, len(cur_tiles)))))

        goal_yx, cur_yx, sims, margins, cyclics = [], [], [], [], []
        pool_total = 0
        for idx in range(n_tiles):
            pool = mutual_nn_match(
                self._goal_tiles[idx], self.backbone.encode(cur_tiles[idx]),
                threshold=self.threshold, tie_order=self.tie_order)
            pool_total += pool.size
            # One draw per tile, empty pool included, so the frame seeds advance
            # exactly as the old per-tile `_call_find_correspondences` advanced them.
            drawn = self.coarse.draw.draw(pool, per_tile)
            if not len(drawn):
                continue
            # Trap 2: the helper wants (y, x); the matcher speaks (column, row).
            goal_yx.append(self.handler.transform_patch_to_global(
                drawn.goal_patch[:, ::-1], idx, self.tile_coords))
            cur_yx.append(self.handler.transform_patch_to_global(
                drawn.cur_patch[:, ::-1], idx, self.tile_coords))
            sims.append(drawn.sim)
            margins.append(drawn.margin)
            cyclics.append(drawn.cyclic)

        w, h = self.image_wh
        if not goal_yx:
            empty = np.zeros((0, 2))
            return self._matches(empty, empty, np.zeros(0), np.zeros(0), np.zeros(0), 0,
                                 n_tiles, per_tile)
        # Tile coordinates live in the tiling resolution; rescale to the full frame,
        # flipping (y, x) back to (column, row).
        res = self.config.input_resolution
        scale = np.asarray([w / res, h / res], np.float64)
        goal_px = np.concatenate(goal_yx)[:, ::-1] * scale
        cur_px = np.concatenate(cur_yx)[:, ::-1] * scale
        return self._matches(goal_px, cur_px, np.concatenate(sims), np.concatenate(margins),
                             np.concatenate(cyclics), pool_total, n_tiles, per_tile)

    def _matches(self, goal_px, cur_px, sim, margin, cyclic, pool_size: int,
                 n_tiles: int, per_tile: int) -> Matches:
        """Wrap the tiled pixels on the **effective** tiled lattice.

        ``side = effective_patches`` (grid x cells-per-tile) and
        ``input_size = input_resolution``: under the published configuration
        (``slice_resolution == vit_input_size``, so the helper's ``scale`` is 1) the
        tiled points are exactly the cell centres of that lattice, and
        :meth:`vsbench.vs.baselines.match.Matches.from_pixel_pairs` reproduces them either way.
        """
        w, h = self.image_wh
        matches = Matches.from_pixel_pairs(
            goal_px, cur_px,
            side=self.config.effective_patches, input_size=self.config.input_resolution,
            patch_size=self.config.patch_size, image_wh=(w, h),
            sim=np.asarray(sim, np.float64), cyclic=np.asarray(cyclic, np.float64),
            margin=np.asarray(margin, np.float64), pool_size=int(pool_size),
            num_landmarks=self.num_landmarks,
            draw_seed=None if self.coarse.draw is None else self.coarse.draw.episode_seed,
            tie_order=self.tie_order)
        matches.extras.update(n_tiles=n_tiles, per_tile_pairs=per_tile,
                              tile_input_resolution=self.config.input_resolution,
                              backbone=self.backbone.name)
        return matches
