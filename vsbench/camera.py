"""Plain-data pinhole camera in pixels, plus the OpenGL projection it implies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics in pixels. ``fx != fy`` and an off-centre principal point work.

    ``K`` follows OpenCV: an integer pixel index is the centre of that pixel, so the
    principal point of a perfectly centred W x H image is ``((W - 1) / 2, (H - 1) / 2)``.
    ``near`` and ``far`` are the clipping planes in metres.
    """

    K: np.ndarray
    width: int
    height: int
    near: float = 0.05
    far: float = 10.0

    def __post_init__(self) -> None:
        K = np.array(self.K, dtype=np.float64).reshape(3, 3)
        if not np.allclose(K[2], [0.0, 0.0, 1.0]) or K[0, 1] != 0.0 or K[1, 0] != 0.0:
            raise ValueError("K must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("fx and fy must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 0.0 < self.near < self.far:
            raise ValueError("need 0 < near < far")
        K.setflags(write=False)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    # -- accessors ---------------------------------------------------------------------
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    # -- constructors ------------------------------------------------------------------
    @classmethod
    def from_focal(
        cls,
        width: int,
        height: int,
        fx: float,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        near: float = 0.05,
        far: float = 10.0,
    ) -> CameraModel:
        fy = fx if fy is None else fy
        cx = (width - 1) / 2.0 if cx is None else cx
        cy = (height - 1) / 2.0 if cy is None else cy
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return cls(K, width, height, near, far)

    def scaled(self, s: float) -> CameraModel:
        """The same camera at ``s`` times the resolution (pixel-centre convention kept)."""
        w, h = int(round(self.width * s)), int(round(self.height * s))
        fx, fy = self.fx * s, self.fy * s
        cx = (self.cx + 0.5) * s - 0.5
        cy = (self.cy + 0.5) * s - 0.5
        return CameraModel.from_focal(w, h, fx, fy, cx, cy, self.near, self.far)

    # -- OpenGL side (for the backends) -------------------------------------------------
    def frustum(self) -> tuple[float, float, float, float]:
        """``(left, right, bottom, top)`` on the near plane, as ``glFrustum`` wants them.

        OpenGL samples pixel ``(i, j)`` at window coordinate ``(i + 0.5, j + 0.5)`` with y
        up, so the OpenCV principal point ``(cx, cy)`` sits at ``(cx + 0.5, cy + 0.5)``.
        """
        n = self.near
        left = -n * (self.cx + 0.5) / self.fx
        right = n * (self.width - self.cx - 0.5) / self.fx
        top = n * (self.cy + 0.5) / self.fy
        bottom = -n * (self.height - self.cy - 0.5) / self.fy
        return left, right, bottom, top

    def gl_projection(self) -> np.ndarray:
        """4 x 4 OpenGL projection (camera looks down -z, y up) equivalent to ``K``."""
        n, f = self.near, self.far
        W, H = self.width, self.height
        P = np.zeros((4, 4))
        P[0, 0] = 2.0 * self.fx / W
        P[0, 2] = 1.0 - 2.0 * (self.cx + 0.5) / W
        P[1, 1] = 2.0 * self.fy / H
        P[1, 2] = 2.0 * (self.cy + 0.5) / H - 1.0
        P[2, 2] = -(f + n) / (f - n)
        P[2, 3] = -2.0 * f * n / (f - n)
        P[3, 2] = -1.0
        return P

    def pixel_rays(self) -> np.ndarray:
        """(H, W, 3) ray directions through every pixel centre, camera frame, z = 1."""
        u = np.arange(self.width, dtype=np.float64)
        v = np.arange(self.height, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        x = (uu - self.cx) / self.fx
        y = (vv - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x)], axis=-1)


# The camera every gate uses: fx differs from fy and the principal point is shifted by
# (+17, -11) px from the centre, so an API that only takes a field of view fails loudly.
def gate_camera(width: int = 640, height: int = 480) -> CameraModel:
    s = width / 640.0
    return CameraModel.from_focal(
        width,
        height,
        fx=600.0 * s,
        fy=620.0 * s,
        cx=(width - 1) / 2.0 + 17.0 * s,
        cy=(height - 1) / 2.0 - 11.0 * s,
    )
