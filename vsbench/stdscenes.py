"""The scenes the gates and measurements use, built from plain data.

* ``marker_scene``: 13 unlit coloured spheres, 9 on the plane z = 0 and 4 above it.
* ``plane_scene``: one large plane for the depth gate.
* ``poster_scene``: the poster quad (0.8 x 0.6 m, like the vs-sim posters).
* ``clutter_scene``: the poster on a floor with 10 primitives and 3 meshes (~50k tris).
* ``lighting_scene``: a lit sphere and box on a floor for the lighting-control check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vsbench.meshgen import bumpy_sphere, torus, uv_sphere
from vsbench.scene import Box, Light, Material, Mesh, Plane, SceneSpec, Sphere, rigid, rot_x, rot_z

ASSETS = Path(__file__).resolve().parent.parent / "assets"
POSTER_SIZE = (0.8, 0.6)  # metres, 4:3 like the poster picture
MARKER_RADIUS = 0.012

# 9 on the plane (3 x 3 grid, 0.15 m pitch) and 4 raised between grid points.
MARKER_POINTS = np.array(
    [[x, y, 0.0] for y in (-0.15, 0.0, 0.15) for x in (-0.15, 0.0, 0.15)]
    + [[-0.075, -0.075, 0.06], [0.075, -0.075, 0.10], [-0.075, 0.075, 0.14], [0.075, 0.075, 0.08]]
)
MARKER_COLORS = np.array(
    [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5], [0.5, 1, 0], [0, 0.5, 1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)


def load_poster() -> np.ndarray:
    """The CC0 poster picture as (H, W, 3) uint8."""
    from PIL import Image

    return np.asarray(Image.open(ASSETS / "poster.png").convert("RGB"))


def marker_scene() -> SceneSpec:
    """13 unlit spheres over a black backing plane (so the clear colour never matters)."""
    objs = [Plane(rigid(t=(0, 0, -0.05)), (6.0, 6.0), Material((0, 0, 0), unlit=True))]
    objs += [
        Sphere(tuple(p), MARKER_RADIUS, Material(tuple(c), unlit=True))
        for p, c in zip(MARKER_POINTS, MARKER_COLORS)
    ]
    return SceneSpec(tuple(objs), lights=(), ambient=1.0, background=(0, 0, 0))


def plane_scene(size: float = 8.0) -> SceneSpec:
    plane = Plane(np.eye(4), (size, size), Material((0.6, 0.6, 0.6)))
    return SceneSpec((plane,), lights=(Light((0.3, -0.2, -1.0), 1.0),), ambient=0.3)


def poster_scene(texture: np.ndarray | None = None, lit: bool = False) -> SceneSpec:
    tex = load_poster() if texture is None else texture
    poster = Plane(np.eye(4), POSTER_SIZE, Material((1, 1, 1), texture=tex, unlit=not lit))
    if lit:
        return SceneSpec((poster,), lights=(Light((0.2, 0.1, -1.0), 1.0),), ambient=0.3)
    return SceneSpec((poster,), lights=(), ambient=1.0)


def clutter_scene(texture: np.ndarray | None = None) -> SceneSpec:
    """Poster on a 4 x 4 m floor, 10 primitives, 3 meshes of about 50k triangles."""
    tex = load_poster() if texture is None else texture
    rng = np.random.default_rng(1)
    objs: list = [
        Plane(rigid(t=(0, 0, -0.001)), (4.0, 4.0), Material((0.45, 0.45, 0.5))),
        Plane(np.eye(4), POSTER_SIZE, Material((1, 1, 1), texture=tex)),
    ]
    for i in range(5):
        p = rng.uniform(-0.8, 0.8, 2)
        s = rng.uniform(0.05, 0.15, 3)
        objs.append(Box(rigid(rot_z(rng.uniform(0, np.pi)), (p[0], p[1] + 0.5 * np.sign(p[1]), s[2] / 2)), tuple(s), Material(tuple(rng.uniform(0.2, 0.9, 3)))))
    for i in range(5):
        p = rng.uniform(-0.8, 0.8, 2)
        r = rng.uniform(0.03, 0.08)
        objs.append(Sphere((p[0] + 0.5 * np.sign(p[0]), p[1], r), r, Material(tuple(rng.uniform(0.2, 0.9, 3)))))
    V, F = torus(0.12, 0.04)  # 16000 triangles
    objs.append(Mesh(rigid(rot_x(0.4), (-0.7, 0.55, 0.12)), V, F, Material((0.8, 0.3, 0.2))))
    V, F = uv_sphere(0.09)  # 19800 triangles
    objs.append(Mesh(rigid(t=(0.7, 0.6, 0.09)), V, F, Material((0.2, 0.5, 0.8))))
    V, F = bumpy_sphere(0.1)  # 13944 triangles
    objs.append(Mesh(rigid(t=(0.0, -0.6, 0.11)), V, F, Material((0.3, 0.8, 0.3))))
    lights = (Light((0.4, 0.3, -1.0), 1.0),)
    return SceneSpec(tuple(objs), lights, ambient=0.3, background=(0.05, 0.05, 0.08))


def triangle_count(scene: SceneSpec) -> int:
    n = 0
    for o in scene.objects:
        if isinstance(o, Mesh):
            n += len(o.faces)
        elif isinstance(o, Plane):
            n += 2
        elif isinstance(o, Box):
            n += 12
    return n


def lighting_scene(lights: tuple[Light, ...] | None = None) -> SceneSpec:
    objs = (
        Plane(np.eye(4), (2.0, 2.0), Material((0.7, 0.7, 0.7))),
        Sphere((-0.15, 0.0, 0.1), 0.1, Material((0.9, 0.9, 0.9))),
        Box(rigid(rot_z(0.5), (0.2, 0.05, 0.075)), (0.15, 0.15, 0.15), Material((0.9, 0.9, 0.9))),
    )
    if lights is None:
        lights = (Light((0.5, 0.0, -1.0), 1.0),)
    return SceneSpec(objs, lights, ambient=0.2)


STD_SCENES = {
    "marker": marker_scene,
    "plane": plane_scene,
    "poster": poster_scene,
    "poster_lit": lambda: poster_scene(lit=True),
    "clutter": clutter_scene,
    "lighting": lighting_scene,
}
