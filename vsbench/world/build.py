"""vs-sim's scene descriptions -> vsbench ``SceneSpec`` (docs/PORT_PLAN.md section 6).

``scene_spec_for(spec)`` takes one entry of the copied ``configs/scenes.yaml``
(``vsbench.vs.scenes.load_scenes()``) and returns the plain-data scene the MuJoCo backend
compiles, plus a ``meta`` dict saying how it was built: the posters (U1) and the rooms
(``kind: multi_object``, U4). :func:`build_scene` returns the same plus the placed objects
(``vsbench.world.assets.PlacedObject``), from which ``MujocoScene`` derives a room's goal.

The poster, as ``IsaacScene.build_poster`` / ``build_backdrop`` author it:

* the plate: a quad ``[-w/2, w/2] x [-h/2, h/2]`` at ``z = 0``, the texture shown once,
  row 0 at world +Y (USD uv ``(0, 1)`` at ``(-hx, +hy)``; vsbench's ``Plane`` convention is
  the same); the texture's RGB channels as stored (Isaac's ``UsdUVTexture.rgb`` ignores
  alpha, and PIL's RGBA -> RGB drops it without compositing);
* the backdrop: a 4 x 4 m quad at ``z = -0.02``, grey 0.5, no texture;
* the goal: ``vsbench.vs.scenes.poster_goal_pose`` (the world's job, see ``mujoco_scene``).

The room, as ``IsaacScene.build_eval_scene`` builds a ``multi_object`` scene:

* the ground: ``build_poster(size, size, texture, "/World/Ground", uv_tiles = size /
  tile_period)`` - a ``size x size`` quad at ``z = 0``, the texture repeated ``uv_tiles``
  times per axis (wrap ``repeat``; one span when ``tile_period == size``, as in the three
  benchmark rooms), USD uv ``(0, 0)`` at ``(-h, -h)``. A single span is vsbench's ``Plane``;
  a repeat is a ``Mesh`` of one quad per tile, each carrying the texture once (cut at the
  far edge when ``uv_tiles`` is not whole), so no GL wrap mode is relied on;
* no backdrop (Isaac authors none for a room);
* the objects: ``scatter_items(spec, with_scatter=layout is None)`` (the copied, Isaac-free
  item list), then per item ``vsbench.world.assets.placed_object`` on the converted cache
  (the copied scale rule; the frozen layout's ``pos`` / ``quat_wxyz`` when the scene has
  one) and its ``mesh`` (pose ``T(pos) R(quat)``, vertices pre-scaled by ``s``, the
  object's texture, ``v = 0`` at the top row). A scene with a ``settle`` block and no
  baked layout is refused, as Isaac refuses it ("a re-settle may never define a goal");
* the lights: Isaac's ``_boot`` lights are the same for every scene, so a room gets the
  poster's stand-in (below) - its ground is a ``z = 0`` plane like the plate, so the same
  intensity renders a white ground texel white.

Lighting is a stand-in, never a photometric claim: Isaac's dome light (3000, colour 0.2)
becomes MuJoCo's ambient term 0.2; its distant light (3500, ``rotateXYZ(-30, 0, 30)``)
becomes one directional light whose direction is the USD light's ``-Z`` axis under that
rotation (computed below), casting shadows; its intensity is set so that a white texel of
the plate renders white, i.e. the lit plate shows the texture's own tone, which is what the
copied mock renders (the "tone fit" of PORT_PLAN section 6; measured in
``tests/world/test_scene_posters.py``).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vsbench.scene import Light, Material, Mesh, Plane, SceneSpec, rigid

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
        "lighting": _lighting_meta(lights),
    }
    return scene, meta


# --------------------------------------------------------------------------- rooms
GROUND_DEFAULT = {"size_m": 2.4, "tile_period_m": 0.6}   # build_eval_scene's .get defaults


def ground_objects(ground: dict, tex: np.ndarray | None = None) -> tuple[list, dict]:
    """Isaac's ``/World/Ground`` -> vsbench objects (one ``Plane`` for a single span, a
    ``Mesh`` of per-tile quads for a repeat) and a meta block."""
    size = float(ground.get("size_m", GROUND_DEFAULT["size_m"]))
    period = float(ground.get("tile_period_m", GROUND_DEFAULT["tile_period_m"]))
    tiles = size / period
    if tex is None:
        tex = load_texture(ground["texture"])
    mat = Material((1.0, 1.0, 1.0), texture=tex)
    meta = {"texture": str(ground["texture"]),
            "texture_host_path": str(workspace_path(ground["texture"])),
            "texture_wh": [int(tex.shape[1]), int(tex.shape[0])], "size_m": size,
            "tile_period_m": period, "uv_tiles": tiles, "z_m": 0.0}
    if abs(tiles - 1.0) <= 1e-12:
        meta["build"] = "one Plane, the texture shown once (row 0 at world +Y)"
        return [Plane(np.eye(4), (size, size), mat)], meta
    h = size / 2.0
    n = int(math.ceil(tiles - 1e-9))
    edges = [-h + k * period for k in range(n)] + [h]
    V, UV, F = [], [], []
    for j in range(n):
        y0, y1 = edges[j], edges[j + 1]
        for i in range(n):
            x0, x1 = edges[i], edges[i + 1]
            # USD st = ((x + h) / p, (y + h) / p) mod 1, v up; picture v' = 1 - v
            su, sv = (x1 - x0) / period, (y1 - y0) / period
            b = len(V)
            V += [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]]
            UV += [[0.0, 1.0], [su, 1.0], [su, 1.0 - sv], [0.0, 1.0 - sv]]
            F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    meta["build"] = f"one Mesh of {n} x {n} quads, the texture once per {period} m tile (repeat)"
    return [Mesh(np.eye(4), np.array(V), np.array(F), mat, uvs=np.array(UV))], meta


@dataclass
class Built:
    """What :func:`build_scene` returns: the plain-data scene, how it was built, and for a
    room the placed objects (``vsbench.world.assets.PlacedObject``) its goal comes from."""

    scene_spec: SceneSpec
    meta: dict
    placed: list | None = None


def room_build(spec, *, cache=None, library: dict | None = None) -> Built:
    """A ``kind: multi_object`` entry -> the ground, the converted objects, the lights."""
    from vsbench.vs.assets import load_library, merge_ycb_library
    from vsbench.vs.isaac_scene import scatter_items
    from vsbench.vs.scenes import check_layout, load_layout
    from vsbench.world import assets as A

    t0 = time.monotonic()
    raw = spec.raw
    library = dict(library or load_library())
    merge_ycb_library(library, spec.object_ids())
    layout = load_layout(spec)
    if layout is None and raw.get("settle"):
        raise RuntimeError(f"{spec.name} declares a settle block and no baked layout - a "
                           f"re-settle may never define a goal (IsaacScene.build_eval_scene); "
                           f"the MuJoCo world only loads baked layouts")
    items = scatter_items(spec, with_scatter=layout is None)
    if layout is not None:
        check_layout(layout, [str(it["instance_id"]) for it in items])
    les = layout["objects"] if layout is not None else [None] * len(items)
    if not items:
        raise ValueError(f"{spec.name}: scene declares no objects")
    rules = A.default_rules()
    placed = []
    for it, le in zip(items, les, strict=True):
        iid = str(it["instance_id"])
        if iid not in library:
            raise KeyError(f"asset {iid!r} is not in the library")
        placed.append(A.placed_object(A.load_object(iid, cache), it, library[iid], le, rules=rules))
    objects: list = []
    ground_meta = None
    if raw.get("ground"):
        g_objs, ground_meta = ground_objects(dict(raw["ground"]))
        objects += g_objs
    objects += [p.mesh for p in placed]
    lights = poster_lights()
    scene = SceneSpec(tuple(objects), lights=lights, ambient=AMBIENT, background=(0.0, 0.0, 0.0))
    lo, hi = A.scene_bbox(placed, "usd")
    meta = {
        "kind": "multi_object",
        "ground": ground_meta,
        "layout_file": spec.layout_file,
        "objects": [{"instance_id": p.record["instance_id"], "source": p.record["source"],
                     "scale": p.scale, "pos": [float(v) for v in p.pos],
                     "quat_wxyz": [float(v) for v in p.quat_wxyz],
                     "vertices": int(len(p.obj.vertices)), "faces": int(len(p.obj.faces)),
                     "texture_wh": [int(p.obj.texture.shape[1]), int(p.obj.texture.shape[0])]}
                    for p in placed],
        "bbox_usd_rule": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
        "object_cache": str(A.cache_root(cache)),
        "build_s": round(time.monotonic() - t0, 3),
        "lighting": _lighting_meta(lights),
        "lost": "PBR (roughness 0.6 ground, object materials), normal maps, MDL; diffuse texture "
                "+ MuJoCo's own shading only (PORT_PLAN section 5)",
    }
    return Built(scene, meta, placed)


def _lighting_meta(lights) -> dict:
    return {
        "ambient": AMBIENT,
        "sun_direction": [round(float(x), 9) for x in lights[0].direction],
        "sun_intensity": round(float(lights[0].intensity), 9),
        "sun_cast_shadow": True,
        "from": "Isaac dome 3000 x 0.2 -> ambient 0.2; distant light rotateXYZ(-30, 0, 30) "
                "-> direction; intensity so a white texel renders white (the mock's tone); "
                "a stand-in, not photometric parity",
    }


def build_scene(spec, **kw) -> Built:
    """One ``configs/scenes.yaml`` entry -> :class:`Built` (posters and rooms)."""
    if spec.kind == "poster":
        scene, meta = poster_spec(spec)
        return Built(scene, meta, None)
    if spec.kind == "multi_object":
        return room_build(spec, **kw)
    raise NotImplementedError(f"{spec.name}: kind {spec.kind!r} is not built by the MuJoCo world")


def scene_spec_for(spec, **kw) -> tuple[SceneSpec, dict]:
    """One ``configs/scenes.yaml`` entry -> (SceneSpec, meta): posters (U1) and rooms (U4)."""
    b = build_scene(spec, **kw)
    return b.scene_spec, b.meta
