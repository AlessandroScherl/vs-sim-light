"""vs-sim's scene descriptions -> vsbench ``SceneSpec`` (docs/PORT_PLAN.md section 6).

``scene_spec_for(spec)`` takes one entry of the copied ``configs/scenes.yaml``
(``vsbench.vs.scenes.load_scenes()``) and returns the plain-data scene the MuJoCo backend
compiles, plus a ``meta`` dict saying how it was built. U1 builds the posters; the rooms
(``kind: multi_object``) are U4 and are refused here.

The poster, as ``IsaacScene.build_poster`` / ``build_backdrop`` author it:

* the plate: a quad ``[-w/2, w/2] x [-h/2, h/2]`` at ``z = 0``, the texture shown once,
  row 0 at world +Y (USD uv ``(0, 1)`` at ``(-hx, +hy)``; vsbench's ``Plane`` convention is
  the same); the texture's RGB channels as stored (Isaac's ``UsdUVTexture.rgb`` ignores
  alpha, and PIL's RGBA -> RGB drops it without compositing);
* the backdrop: a 4 x 4 m quad at ``z = -0.02``, grey 0.5, no texture;
* the goal: ``vsbench.vs.scenes.poster_goal_pose`` (the world's job, see ``mujoco_scene``).

Lighting is a stand-in, never a photometric claim: Isaac's dome light (3000, colour 0.2)
becomes MuJoCo's ambient term 0.2; its distant light (3500, ``rotateXYZ(-30, 0, 30)``)
becomes one directional light whose direction is the USD light's ``-Z`` axis under that
rotation (computed below), casting shadows; its intensity is set so that a white texel of
the plate renders white, i.e. the lit plate shows the texture's own tone, which is what the
copied mock renders (the "tone fit" of PORT_PLAN section 6; measured in
``tests/world/test_scene_posters.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from vsbench.scene import Light, Material, Plane, SceneSpec, rigid

#: Isaac's dome light colour, used as MuJoCo's ambient term.
AMBIENT = 0.2
#: ``UsdLux.DistantLight`` ``rotateXYZ`` in degrees (``isaac_scene.py`` ``_boot``).
SUN_ROTATE_XYZ_DEG = (-30.0, 0.0, 30.0)
BACKDROP_DEFAULT = {"size_m": 4.0, "gray": 0.5, "z_m": -0.02}


def workspace_path(p: str | os.PathLike) -> Path:
    """``/workspace/...`` (the station volume named in vs-sim's yaml) -> the host path.

    The root is ``$VSBENCH_WORKSPACE_ROOT`` (``tools/cpu_env.sh``: ``~/genvs_ws``).
    """
    s = str(p)
    root = os.environ.get("VSBENCH_WORKSPACE_ROOT", str(Path.home() / "genvs_ws"))
    if s == "/workspace" or s.startswith("/workspace/"):
        return Path(root + s[len("/workspace"):])
    return Path(s)


def load_texture(path: str | os.PathLike) -> np.ndarray:
    """An RGB uint8 picture (H, W, 3), row 0 at the top; alpha dropped, not composited."""
    from PIL import Image

    p = workspace_path(path)
    if not p.exists():
        raise FileNotFoundError(f"texture not found: {p} (yaml says {path}; "
                                f"VSBENCH_WORKSPACE_ROOT maps /workspace)")
    with Image.open(p) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def sun_direction(rotate_xyz_deg=SUN_ROTATE_XYZ_DEG) -> np.ndarray:
    """The way Isaac's distant light travels, world frame.

    USD ``xformOp:rotateXYZ (a, b, c)`` is ``Rx(a)`` first, then ``Ry(b)``, then ``Rz(c)``
    (row vectors: ``v . Rx . Ry . Rz``; column vectors ``Rz @ Ry @ Rx @ v``), and a
    ``UsdLux.DistantLight`` shines along its local ``-Z``.
    """
    a, b, c = rotate_xyz_deg
    R = _rot("z", c) @ _rot("y", b) @ _rot("x", a)
    return R @ np.array([0.0, 0.0, -1.0])


def sun_intensity(ambient: float = AMBIENT, direction=None) -> float:
    """The directional intensity at which a white texel of the ``z = 0`` plate is white.

    MuJoCo shades a diffuse surface as ``tex * (ambient + I * max(0, n . -d))``; with the
    plate normal ``+Z`` that is 1 when ``I = (1 - ambient) / cos(incidence)``.
    """
    d = sun_direction() if direction is None else np.asarray(direction, np.float64)
    cos_inc = float(-d[2] / np.linalg.norm(d))
    return (1.0 - float(ambient)) / cos_inc


def poster_lights() -> tuple[Light, ...]:
    return (Light(direction=tuple(sun_direction()), intensity=sun_intensity(),
                  cast_shadow=True),)


def poster_spec(spec) -> tuple[SceneSpec, dict]:
    """A ``kind: poster`` entry -> (SceneSpec, meta)."""
    raw = spec.raw
    w, h = float(raw["width_m"]), float(raw["height_m"])
    tex = load_texture(raw["texture"])
    objects = [Plane(np.eye(4), (w, h), Material((1.0, 1.0, 1.0), texture=tex))]
    backdrop = None
    if raw.get("backdrop"):
        backdrop = {**BACKDROP_DEFAULT, **dict(raw["backdrop"])}
        g = float(backdrop["gray"])
        size = float(backdrop["size_m"])
        objects.append(Plane(rigid(t=(0.0, 0.0, float(backdrop["z_m"]))), (size, size),
                             Material((g, g, g))))
    lights = poster_lights()
    scene = SceneSpec(tuple(objects), lights=lights, ambient=AMBIENT, background=(0.0, 0.0, 0.0))
    meta = {
        "kind": "poster",
        "plate_m": [w, h],
        "texture": str(raw["texture"]),
        "texture_host_path": str(workspace_path(raw["texture"])),
        "texture_wh": [int(tex.shape[1]), int(tex.shape[0])],
        "backdrop": backdrop,
        "lighting": {
            "ambient": AMBIENT,
            "sun_direction": [round(float(x), 9) for x in lights[0].direction],
            "sun_intensity": round(float(lights[0].intensity), 9),
            "sun_cast_shadow": True,
            "from": "Isaac dome 3000 x 0.2 -> ambient 0.2; distant light rotateXYZ(-30, 0, 30) "
                    "-> direction; intensity so a white texel renders white (the mock's tone); "
                    "a stand-in, not photometric parity",
        },
    }
    return scene, meta


def scene_spec_for(spec) -> tuple[SceneSpec, dict]:
    """One ``configs/scenes.yaml`` entry -> (SceneSpec, meta). Posters only (U1)."""
    if spec.kind == "poster":
        return poster_spec(spec)
    raise NotImplementedError(
        f"{spec.name}: kind {spec.kind!r} - the rooms (multi_object) are port unit U4; "
        f"U1 builds the posters (hollywood, poster2, poster3) only")
