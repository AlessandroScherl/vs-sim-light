"""Plain-data scene description. Each backend builds its native scene from this.

Frames: every object carries a 4 x 4 object-to-world ``pose``. A ``Plane`` is a finite
rectangle in its local x-y plane (normal +z), spanning ``[-w/2, w/2] x [-h/2, h/2]``. A
texture on it is shown once, upright when viewed from +z with local x to the right:
texture column 0 is at local x = -w/2 and texture row 0 (the top of the picture) is at
local y = +h/2. That is the mapping ``vsbench.reference.texture_to_plane`` writes down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

Vec3 = tuple[float, float, float]


def _pose(p) -> np.ndarray:
    T = np.array(p, dtype=np.float64).reshape(4, 4)
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("pose must be a rigid 4x4 with last row [0 0 0 1]")
    T.setflags(write=False)
    return T


@dataclass(frozen=True)
class Material:
    """Flat colour, optional texture (H, W, 3) uint8 with row 0 at the top, and shading.

    ``unlit`` asks for the raw colour with no lighting at all (markers, the poster in the
    gates). A backend without an unlit shader documents its approximation.
    """

    color: Vec3 = (0.8, 0.8, 0.8)
    texture: np.ndarray | None = None
    unlit: bool = False

    def __post_init__(self) -> None:
        if self.texture is not None:
            t = np.ascontiguousarray(self.texture)
            if t.ndim != 3 or t.shape[2] != 3 or t.dtype != np.uint8:
                raise ValueError("texture must be (H, W, 3) uint8")
            t.setflags(write=False)
            object.__setattr__(self, "texture", t)
        object.__setattr__(self, "color", tuple(float(c) for c in self.color))


@dataclass(frozen=True)
class Plane:
    """A finite rectangle ``size = (w, h)`` metres in its local x-y plane, normal +z."""

    pose: np.ndarray
    size: tuple[float, float]
    material: Material = field(default_factory=Material)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _pose(self.pose))
        object.__setattr__(self, "size", (float(self.size[0]), float(self.size[1])))


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in its local frame with full extents ``size = (sx, sy, sz)``."""

    pose: np.ndarray
    size: Vec3
    material: Material = field(default_factory=Material)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _pose(self.pose))
        object.__setattr__(self, "size", tuple(float(s) for s in self.size))


@dataclass(frozen=True)
class Sphere:
    center: Vec3
    radius: float
    material: Material = field(default_factory=Material)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", tuple(float(c) for c in self.center))
        object.__setattr__(self, "radius", float(self.radius))


@dataclass(frozen=True)
class Mesh:
    """Triangle mesh: ``vertices`` (N, 3), ``faces`` (M, 3) int, optional ``uvs`` (N, 2).

    ``uvs`` use the same picture convention as ``Plane``: u to the right, v = 0 at the
    top row of the texture, v = 1 at the bottom row.
    """

    pose: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    material: Material = field(default_factory=Material)
    uvs: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _pose(self.pose))
        V = np.ascontiguousarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        F = np.ascontiguousarray(self.faces, dtype=np.int32).reshape(-1, 3)
        V.setflags(write=False)
        F.setflags(write=False)
        object.__setattr__(self, "vertices", V)
        object.__setattr__(self, "faces", F)
        if self.uvs is not None:
            U = np.ascontiguousarray(self.uvs, dtype=np.float64).reshape(-1, 2)
            if U.shape[0] != V.shape[0]:
                raise ValueError("uvs must have one row per vertex")
            U.setflags(write=False)
            object.__setattr__(self, "uvs", U)


@dataclass(frozen=True)
class Light:
    """A directional light. ``direction`` is the way the light travels (world frame)."""

    direction: Vec3 = (0.0, 0.0, -1.0)
    intensity: float = 1.0
    color: Vec3 = (1.0, 1.0, 1.0)
    cast_shadow: bool = False

    def __post_init__(self) -> None:
        d = np.array(self.direction, dtype=np.float64)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("light direction must be non-zero")
        object.__setattr__(self, "direction", tuple(float(x) for x in d / n))
        object.__setattr__(self, "color", tuple(float(c) for c in self.color))
        object.__setattr__(self, "intensity", float(self.intensity))


Object = Union[Plane, Box, Sphere, Mesh]


@dataclass(frozen=True)
class SceneSpec:
    """Objects, directional lights, a global ambient term and a background colour.

    ``ambient`` is the fraction of an object's colour shown with no light at all; an
    ``ambient`` of 1 with no lights renders every surface flat in its own colour.
    """

    objects: tuple[Object, ...] = ()
    lights: tuple[Light, ...] = ()
    ambient: float = 0.3
    background: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "lights", tuple(self.lights))
        object.__setattr__(self, "background", tuple(float(c) for c in self.background))

    def with_lights(self, *lights: Light) -> SceneSpec:
        return SceneSpec(self.objects, tuple(lights), self.ambient, self.background)


# -- small pose helpers ------------------------------------------------------------------
def rigid(R: np.ndarray | None = None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)
