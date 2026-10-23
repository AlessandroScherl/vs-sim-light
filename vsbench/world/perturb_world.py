"""The world verbs vs-sim's perturbations need, on ``MujocoScene`` (docs/PORT_PLAN.md U8).

The copied ``vsbench.vs.bench.perturb`` asks a scene for these verbs (``perturb._needs``)
and never edits a scene itself; ``IsaacScene`` (the frozen reference,
``vsbench/vs/isaac_scene.py``) implements them on the USD stage. Here they are implemented
on the plain-data scene MuJoCo compiles:

==========================  ===================================  ==========================
verb (IsaacScene's name)    used by the modes                    MuJoCo
==========================  ===================================  ==========================
``build_poster``            ``ground:<name|path>``               re-texture: recompile
``put_box``                 ``occluder``, ``combined``           a ``Box``: recompile
``remove_prim``             ``occluder``                         drop it: recompile
``place_extra_objects``     ``distractors``, ``swap:``           cached meshes: recompile
``remove_prims``            ``distractors``, ``swap:``           drop them: recompile
``set_prim_visible``        ``swap:``                            leave it out: recompile
``object_pose``             ``swap:<instance_id>``               numpy (the placed pose)
``object_world_bbox``       ``swap:@<name>``                     numpy (Isaac's box rule)
``set_lighting``            ``light``, ``combined``              the look's tone table
``lighting_snapshot``       ``light``                            numpy
``restore_lighting``        ``light``                            the look's tone table
``extra_placements``        ``swap:@<name>`` (read)              a dict, as Isaac's
==========================  ===================================  ==========================

How a world edit happens. The scene keeps its world as an ordered table of *prims* named
by Isaac's prim paths (``/World/Ground``, ``/World/Obj_<i>`` in layout order,
``/World/Poster``, ``/World/Backdrop``, then whatever a verb adds: ``/World/Occluder``,
``/World/Distractor_<k>``, ``/World/Swap_<k>``). A verb edits the table and marks the world
dirty; the next frame drawn recompiles the MuJoCo model from the table (the old model and
its GL contexts are freed **first**: never two rooms in one process), so a perturbation's
several edits cost one compile, and the ``end`` of one episode and the ``begin`` of the
next collapse into one. MuJoCo bakes geometry and textures into the model at compile time,
so there is no cheaper exact path for these edits; cluttered2 recompiles in ~7 s
(measured per mode in ``results/bridge/u8_perturb.json``).

What a recompile keeps fixed, so an edit changes only what it says:

* the camera, the intrinsics and the pose (the camera is not part of the table);
* the look's renderer settings of the **unedited** scene (the 8192 shadow map's fitted
  square, ``vsbench.world.look``) - not re-fitted to the edited geometry;
* MuJoCo's model statistics ``stat.extent`` / ``stat.center`` of the unedited scene: the
  clipping planes and the shadow clip are stored relative to ``stat.extent``, which the
  compiler would otherwise re-derive from the new geometry. With them pinned, a pixel
  whose ray meets the same unedited surface has the same depth, bit for bit (the depth
  gate of ``tests/world/test_perturb_world.py``);
* when the table is back to the unedited scene, the model is compiled exactly as at
  start-up (the same objects, no pinned statistics), so the goal picture redrawn after
  ``end`` has the goal's md5.

The goal frame (as in vs-sim). The runner captures the goal picture once per scene before
the first perturbation's ``begin`` and serves it from the cache whenever the camera sits
exactly at the goal pose; ``IsaacScene`` does the same, so a perturbed flight is always
steered to the **unperturbed** goal picture. ``MujocoScene`` adds one safety: a verb that
edits the world captures the goal first if nobody has yet, so a goal picture can never be
drawn from a perturbed world.

The look under a perturbation (U4b's ``isaac_fit_v1`` is fitted per scene on the
unperturbed goal). **The scene's look stays fixed**: its fill share, light gain, shadow
map and tone curve are the camera-and-renderer response of that scene, not part of the
world, so an added box, a distractor, another floor or another object is drawn through the
same response, as Isaac's fixed renderer would draw it. Nothing is re-fitted.

The light (``set_lighting``). Isaac's ``light`` mode scales the dome and the sun by the
same factor ``s`` and gives both the same colour temperature ``T``. Every light of the
scene then changes by the same per-channel factor ``f_c = s * blackbody_c(T)``, so the
renderer's linear radiance changes by exactly that factor at every pixel, shadows
included. MuJoCo cannot draw it in the model: it clamps each light's diffuse term at 1 and
the fitted look already drives the sun to ~1 (``look.py``), so a brighter light would be
lost. It is therefore applied where the look models Isaac's linear radiance - inside the
tone curve: ``y = 255 srgb(w / (1 + w))`` with ``w = f_c (floor + e^k x^gamma)`` (the
scene's own fitted curve with its radiance scaled; ``f = 1`` is the scene's curve, bit for
bit), and for the ``raw`` look ``y = clip(f_c x)`` (raw is MuJoCo's linear shading). The
blackbody colour is USD's own ``UsdLux.BlackbodyTemperatureAsRgb`` (luminance 1; Isaac's
RTX may use its own table - stated on the record). Depth is never touched. A snapshot in
which the dome and the sun differ in factor is refused: it is not a pure rescale.

Refused, loudly: a prim path the scene does not have (``ValueError``, as Isaac); a Nucleus
asset (not on disk, D-6) or an object not in the converted cache (``FileNotFoundError``
naming the converter); a non-square tiled poster; light snapshots of unequal factors.
"""

from __future__ import annotations

import copy
import gc
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: IsaacScene.set_lighting's boot intensities (``_boot``: dome 3000, sun 3500).
ISAAC_LIGHTS = {"/World/DomeLight": 3000.0, "/World/SunLight": 3500.0}
#: the colour temperature IsaacScene.set_lighting uses when the draw names none
DEFAULT_TEMPERATURE_K = 4500.0
GOAL_OBJECT_PREFIX = "/World/Obj_"

LOOK_RULE = (
    "the scene's look (fill share, light gain, shadow map and its fitted square, tone curve) is "
    "fitted once on the unperturbed goal and stays fixed under every perturbation: an edited "
    "world is drawn through the same renderer response, as in Isaac; the goal picture is the "
    "unperturbed one, captured before the first begin and served from the cache at the goal "
    "pose (vs-sim's rule)")
LIGHT_RULE = (
    "light: the dome and the sun scaled by s with colour temperature T change every pixel's "
    "linear radiance by f_c = s * blackbody_c(T) (UsdLux.BlackbodyTemperatureAsRgb, luminance "
    "1); applied inside the scene's tone curve, y = 255 srgb(w/(1+w)), w = f_c (floor + "
    "e^k x^gamma) (raw look: y = clip(f_c x)); MuJoCo's own light terms clamp at 1, so the "
    "model's lights are not edited; depth untouched")


@dataclass(frozen=True, eq=False)
class Prim:
    """One named piece of the world: the plain-data objects it contributes to the model."""

    path: str
    kind: str                   # poster | backdrop | ground | object | extra | box
    objects: tuple
    visible: bool = True
    placed: Any = None          # vsbench.world.assets.PlacedObject (object / extra)
    info: dict = field(default_factory=dict)

    def replace(self, **kw) -> Prim:
        d = {"path": self.path, "kind": self.kind, "objects": self.objects,
             "visible": self.visible, "placed": self.placed, "info": self.info}
        d.update(kw)
        return Prim(**d)


def same_prim(a: Prim | None, b: Prim | None) -> bool:
    """The same geometry, material and visibility (object identity, not value)."""
    if a is None or b is None:
        return a is b
    return (a.kind == b.kind and a.visible == b.visible and len(a.objects) == len(b.objects)
            and all(x is y for x, y in zip(a.objects, b.objects, strict=True)))


def base_prims(spec, built) -> dict[str, Prim]:
    """The unedited world of a built scene, by Isaac's prim paths, in the order the
    objects sit in ``built.scene_spec`` (``vsbench_obj<i>`` = the i-th object)."""
    objs = list(built.scene_spec.objects)
    prims: dict[str, Prim] = {}
    if spec.kind == "poster":
        raw = spec.raw
        prims["/World/Poster"] = Prim("/World/Poster", "poster", (objs[0],), info={
            "texture": str(raw["texture"]), "width_m": float(raw["width_m"]),
            "height_m": float(raw["height_m"]), "uv_tiles": 1.0})
        if len(objs) > 1:
            prims["/World/Backdrop"] = Prim("/World/Backdrop", "backdrop", tuple(objs[1:]),
                                            info=dict(built.meta.get("backdrop") or {}))
        return prims
    placed = list(built.placed or [])
    n_ground = len(objs) - len(placed)
    if n_ground:
        g = built.meta.get("ground") or {}
        prims["/World/Ground"] = Prim("/World/Ground", "ground", tuple(objs[:n_ground]), info={
            "texture": str(g.get("texture")), "width_m": float(g.get("size_m", 2.4)),
            "height_m": float(g.get("size_m", 2.4)), "uv_tiles": float(g.get("uv_tiles", 1.0))})
    for i, (p, o) in enumerate(zip(placed, objs[n_ground:], strict=True)):
        path = f"{GOAL_OBJECT_PREFIX}{i}"
        prims[path] = Prim(path, "object", (o,), placed=p,
                           info={"instance_id": p.record["instance_id"]})
    return prims


# ------------------------------------------------------------------------ the light
def blackbody_rgb(temperature_k: float) -> np.ndarray:
    """USD's blackbody colour (``UsdLux.BlackbodyTemperatureAsRgb``, luminance 1)."""
    try:
        from pxr import UsdLux
    except ImportError as e:     # pragma: no cover - usd-core is in the venv (U0)
        raise NotImplementedError("the light perturbation's colour temperature needs usd-core "
                                  "(UsdLux.BlackbodyTemperatureAsRgb)") from e
    return np.asarray(UsdLux.BlackbodyTemperatureAsRgb(float(temperature_k)), np.float64)


def boot_lighting() -> dict:
    """IsaacScene's boot lights as ``lighting_snapshot`` reports them (nothing authored)."""
    return {p: {"intensity": float(b), "enable_color_temperature": None,
                "color_temperature": None} for p, b in ISAAC_LIGHTS.items()}


def light_factor(state: dict) -> np.ndarray:
    """The per-channel factor every light changed by against the boot lights; refuses a
    state in which the dome and the sun differ (that is not a pure rescale)."""
    fs = []
    for path, base in ISAAC_LIGHTS.items():
        v = state.get(path)
        if v is None:
            continue
        f = np.full(3, float(v["intensity"]) / base)
        if v.get("enable_color_temperature") and v.get("color_temperature") is not None:
            f = f * blackbody_rgb(float(v["color_temperature"]))
        fs.append(f)
    if not fs:
        return np.ones(3)
    if any(not np.allclose(f, fs[0], rtol=0, atol=1e-12) for f in fs[1:]):
        raise NotImplementedError(
            f"lighting {state}: the dome and the sun changed by different factors; the MuJoCo "
            "world relights by one factor through the look's tone curve (exact for vs-sim's "
            "`light` mode, which scales both alike) and MuJoCo's own light terms clamp at 1")
    return fs[0]


def relit_lut(look, factor) -> np.ndarray | None:
    """The look's 256-entry table with the radiance scaled per channel (``None`` = the
    look's own table, i.e. ``factor`` is exactly 1)."""
    from vsbench.world.look import srgb_encode

    f = np.asarray(factor, np.float64).reshape(3)
    if np.all(f == 1.0):
        return None
    x = np.arange(256, dtype=np.float64) / 255.0
    cols = []
    for c in range(3):
        if look.channel_params is None:            # raw: MuJoCo's linear shading
            y = 255.0 * x * f[c]
        else:
            floor, log_e, gamma = (float(p) for p in look.channel_params[c])
            w = f[c] * (floor + math.exp(log_e) * np.power(x, gamma))
            y = 255.0 * srgb_encode(w / (1.0 + w))
        cols.append(np.clip(np.rint(y), 0, 255))
    return np.ascontiguousarray(np.stack(cols, axis=1).astype(np.uint8))


def _rss_mb() -> dict:
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                k, v = line.split(":", 1)
                out[k] = round(int(v.split()[0]) / 1024.0, 1)
    except OSError:      # pragma: no cover - not Linux
        pass
    return {"rss_mb": out.get("VmRSS"), "peak_rss_mb": out.get("VmHWM")}


# ------------------------------------------------------------------- the mixin
class WorldVerbs:
    """The perturbation verbs. Mixed into ``MujocoScene``, which provides ``_prims``,
    ``_base_prims``, ``_light_state``, ``look``, ``render_goal`` and ``_recompile``."""

    # -------------------------------------------------------------- bookkeeping
    def _init_world(self, built) -> None:
        self._prims = base_prims(self.spec, built)
        self._base_prims = dict(self._prims)
        flat = [o for p in self._prims.values() for o in p.objects]
        if len(flat) != len(built.scene_spec.objects) or any(
                a is not b for a, b in zip(flat, built.scene_spec.objects, strict=True)):
            raise RuntimeError(f"{self.name}: the prim table does not reproduce the built scene")
        self._compiled_sig = self._signature()
        self._pending: list[str] = []
        self._light_state = boot_lighting()
        self._lut = None
        self._light_factor = np.ones(3)
        self.extra_placements: dict[str, dict] = {}
        self.world_log: list[dict] = []
        self._extra_texture_cache: dict[str, np.ndarray] = {}

    def _signature(self) -> tuple:
        return tuple((k, id(p), p.visible) for k, p in self._prims.items())

    def is_unedited(self) -> bool:
        """True when the prim table is the start-up scene's, prim for prim."""
        return (set(self._prims) == set(self._base_prims)
                and all(same_prim(self._prims[k], self._base_prims[k]) for k in self._prims))

    def _edit(self, path: str, prim: Prim | None, what: str) -> None:
        """Replace / add / drop one prim (the goal is captured first if nobody has)."""
        if self._goal_frame is None:
            self.render_goal()           # the goal picture is always the unedited world's
        base = self._base_prims.get(path)
        if prim is not None and same_prim(prim, base):
            prim = base                  # back to the unedited prim: keep its identity
        if prim is None:
            self._prims.pop(path, None)
        else:
            self._prims[path] = prim     # in place if present, else appended
        if self.is_unedited():
            self._prims = dict(self._base_prims)       # the start-up prims, in their order
        self._pending.append(what)

    @property
    def dirty(self) -> bool:
        return self._signature() != self._compiled_sig

    def sync(self) -> dict | None:
        """Recompile now if the world was edited (else nothing); the compile event."""
        if not self.dirty:
            return None
        return self._recompile_world()

    def world_spec(self):
        """The plain-data scene of the current table (before the look's light)."""
        from vsbench.scene import SceneSpec

        objs = [o for p in self._prims.values() if p.visible for o in p.objects]
        b = self.built_spec
        return SceneSpec(tuple(objs), lights=b.lights, ambient=b.ambient, background=b.background)

    def _recompile_world(self) -> dict:
        from vsbench.render.base import make_renderer

        before = _rss_mb()
        t0 = time.monotonic()
        old, self.renderer = self.renderer, None
        gl_before = getattr(old, "gl_renderer", None)
        if old is not None:
            old.close()
        del old
        gc.collect()
        t_free = time.monotonic()
        unedited = self.is_unedited()
        built = self.built_spec if unedited else self.world_spec()
        spec = self.look.scene_spec(built)
        kw = dict(self.renderer_kwargs)              # the unedited scene's (look) settings
        if not unedited:
            kw["stat_extent"], kw["stat_center"] = self._stat
        self.renderer = make_renderer(f"mujoco-{self._gl}", spec, self.render_camera,
                                      samples=self._samples, **kw)
        self.scene_spec = spec
        t1 = time.monotonic()
        gl_after = getattr(self.renderer, "gl_renderer", None)
        from vsbench.world.mujoco_scene import render_mode_from

        mode = render_mode_from(self._gl, gl_after)
        if mode != self.render_mode or gl_after != gl_before:
            self.close()
            raise RuntimeError(f"{self.name}: the recompiled context is {mode!r} ({gl_after!r}), "
                               f"not the scene's {self.render_mode!r} ({gl_before!r}) - refusing")
        self._compiled_sig = self._signature()
        after = _rss_mb()
        ev = {"event": "compile", "n": len(self.world_log) + 1, "unedited": unedited,
              "edits": list(self._pending), "n_objects": len(spec.objects),
              "prims": [k for k, p in self._prims.items() if p.visible],
              "hidden": [k for k, p in self._prims.items() if not p.visible],
              "free_s": round(t_free - t0, 3), "compile_s": round(t1 - t_free, 3),
              "total_s": round(t1 - t0, 3), "rss_mb_before": before["rss_mb"],
              "rss_mb_after": after["rss_mb"], "peak_rss_mb": after["peak_rss_mb"],
              "stat_pinned": not unedited}
        self._pending = []
        self.world_log.append(ev)
        return ev

    def world_record(self) -> dict:
        """What the world is now (prims, visibility, the light), for a record."""
        return {"prims": [{"path": k, "kind": p.kind, "visible": p.visible,
                           **({"instance_id": p.info["instance_id"]} if "instance_id" in p.info else {}),
                           **({"texture": p.info["texture"]} if "texture" in p.info else {})}
                          for k, p in self._prims.items()],
                "unedited": self.is_unedited(), "lighting": copy.deepcopy(self._light_state),
                "light_factor_rgb": [float(v) for v in self._light_factor],
                "n_compiles": len(self.world_log)}

    def _prim(self, path: str) -> Prim:
        p = self._prims.get(str(path))
        if p is None:
            raise ValueError(f"{path} is not a valid prim (the MuJoCo world {self.name!r} has "
                             f"{list(self._prims)})")
        return p

    # ------------------------------------------------------------- the verbs
    def build_poster(self, width_m: float, height_m: float, texture: str,
                     prim_path: str = "/World/Poster", uv_tiles: float = 1.0) -> dict:
        """``IsaacScene.build_poster``: a textured quad at ``z = 0`` (re-authored in place):
        ``uv_tiles`` repeats per axis as the room ground does (``build.ground_objects``)."""
        from vsbench.scene import Material, Plane
        from vsbench.world import build as buildmod

        w, h, t = float(width_m), float(height_m), float(uv_tiles)
        base = self._base_prims.get(prim_path)
        info = {"texture": str(texture), "width_m": w, "height_m": h, "uv_tiles": t}
        if base is not None and base.kind in ("poster", "ground") and base.info == info:
            self._edit(prim_path, base, f"build_poster {prim_path} <- {texture} (the scene's own)")
            return {"width_m": w, "height_m": h, "texture": str(texture), "prim_path": prim_path}
        tex = self._load_texture(texture)
        if abs(w - h) <= 1e-12:
            objs, meta = buildmod.ground_objects(
                {"size_m": w, "tile_period_m": w / t, "texture": str(texture)}, tex)
            info["build"] = meta["build"]
        elif abs(t - 1.0) <= 1e-12:
            objs = [Plane(np.eye(4), (w, h), Material((1.0, 1.0, 1.0), texture=tex))]
        else:
            raise NotImplementedError(f"build_poster: a {w} x {h} m quad tiled {t} times - the "
                                      "MuJoCo world tiles square quads only")
        kind = base.kind if base is not None else ("ground" if prim_path.endswith("Ground") else "poster")
        self._edit(prim_path, Prim(prim_path, kind, tuple(objs), info=info),
                   f"build_poster {prim_path} <- {texture}")
        return {"width_m": w, "height_m": h, "texture": str(texture), "prim_path": prim_path}

    def _load_texture(self, texture: str) -> np.ndarray:
        from vsbench.world import build as buildmod

        key = str(buildmod.workspace_path(texture))
        for p in self._base_prims.values():         # the scene's own: the bytes in hand
            if p.info.get("texture") and str(buildmod.workspace_path(p.info["texture"])) == key:
                tex = getattr(p.objects[0].material, "texture", None)
                if tex is not None:
                    return tex
        if key not in self._extra_texture_cache:
            self._extra_texture_cache = {key: buildmod.load_texture(texture)}   # keep one
        return self._extra_texture_cache[key]

    def put_box(self, prim_path: str, centre, size_m: float, rotation, gray: float = 0.5) -> dict:
        """``IsaacScene.put_box``: a matte grey cube of side ``size_m`` at ``centre``,
        oriented by ``rotation`` (Isaac's float32 orient and scale ops)."""
        from vsbench.scene import Box, Material
        from vsbench.vs.isaac_scene import mat_to_quat_wxyz, reorthogonalize
        from vsbench.world.assets import quat_matrix

        c = np.asarray(centre, dtype=np.float64).reshape(3)
        q = np.asarray(mat_to_quat_wxyz(reorthogonalize(rotation)), np.float32).astype(np.float64)
        T = np.eye(4)
        T[:3, :3] = quat_matrix(q)
        T[:3, 3] = c
        s = float(np.float32(size_m))
        g = float(gray)
        self.remove_prim(prim_path)
        self._edit(prim_path, Prim(prim_path, "box", (Box(T, (s, s, s), Material((g, g, g))),),
                                   info={"centre": [float(v) for v in c], "quat_wxyz": q.tolist(),
                                         "size_m": s, "gray": g}),
                   f"put_box {prim_path} {s:.4f} m")
        return {"prim_path": prim_path, "size_m": s, "gray": g, "centre": [float(v) for v in c]}

    def remove_prim(self, prim_path: str) -> None:
        """``IsaacScene.remove_prim``: idempotent."""
        if str(prim_path) in self._prims:
            self._edit(str(prim_path), None, f"remove_prim {prim_path}")

    def remove_prims(self, prim_paths) -> None:
        for path in list(prim_paths):
            self.remove_prim(path)

    def set_prim_visible(self, prim_path: str, visible: bool) -> None:
        """``IsaacScene.set_prim_visible``: a hidden prim is left out of the model."""
        p = self._prim(prim_path)
        if p.visible != bool(visible):
            self._edit(p.path, p.replace(visible=bool(visible)),
                       f"set_prim_visible {prim_path} {bool(visible)}")

    def place_extra_objects(self, items: list, library: dict | None = None,
                            prefix: str = "/World/Distractor") -> list[str]:
        """``IsaacScene.place_extra_objects``: extra objects from the converted cache, at
        ``<prefix>_<k>``, placed by the copied rules (``vsbench.world.assets.placed_object``):
        native scale for metric scans, the category size otherwise, the placement path
        (``canonical``, ``size_m``, ``long_axis_x``) when the item asks for it; the box
        centre at ``(x, y)``, the bottom on ``z = 0``."""
        from vsbench.vs.assets import load_library, merge_ycb_library, resolve_usd_path
        from vsbench.world import assets as A

        if library is None:
            library = load_library()
            merge_ycb_library(library, [it["instance_id"] for it in items])
        rules = A.default_rules()
        paths = []
        for k, item in enumerate(items):
            iid = str(item["instance_id"])
            if iid not in library:
                raise KeyError(f"asset {iid!r} is not in the library")
            _usd, nucleus = resolve_usd_path(library[iid])
            if nucleus:
                raise FileNotFoundError(f"{iid} is a Nucleus asset: not on disk, never converted "
                                        "(PORT_PLAN D-6) - the MuJoCo world cannot place it")
            path = f"{prefix}_{k}"
            self.remove_prim(path)
            obj = A.load_object(iid, self._cache)
            it = {key: v for key, v in item.items() if key not in ("x", "y", "pixel")}
            it["dx"], it["dy"] = float(item["x"]), float(item["y"])
            if "/" not in iid and not rules.uses_placement(it):
                it.setdefault("category", "ycb")       # IsaacScene: item.get("category", "ycb")
            placed = A.placed_object(obj, it, library[iid], None, rules=rules)
            if "placement" in placed.record:
                self.extra_placements[path] = placed.record["placement"]
            self._edit(path, Prim(path, "extra", (placed.mesh,), placed=placed,
                                  info={"instance_id": iid}),
                       f"place {path} {iid}")
            paths.append(path)
        return paths

    def object_pose(self, prim_path: str):
        """``IsaacScene.object_pose``: ``(pos, quat_wxyz)`` of a placed object's translate
        and orient ops (a box: its centre and orient)."""
        p = self._prim(prim_path)
        if p.placed is not None:
            return (np.asarray(p.placed.pos, np.float64).copy(),
                    np.asarray(p.placed.quat_wxyz, np.float64).copy())
        if p.kind == "box":
            return np.asarray(p.info["centre"], np.float64), np.asarray(p.info["quat_wxyz"], np.float64)
        raise ValueError(f"{prim_path} has no transform ops (a {p.kind})")

    def object_world_bbox(self, prim_path: str):
        """``IsaacScene.object_world_bbox``: Isaac's measured box of a placed object
        (``BBoxCache`` on the authored extents, ``vsbench.world.assets.aligned_bound``)."""
        p = self._prims.get(str(prim_path))
        if p is None or p.placed is None:
            raise ValueError(f"{prim_path}/asset is not a valid prim - the scene has no object there")
        lo, hi = p.placed.bound("usd")
        return np.asarray(lo, np.float64), np.asarray(hi, np.float64)

    # --------------------------------------------------------------- the light
    def lighting_snapshot(self) -> dict:
        """``IsaacScene.lighting_snapshot``: the dome and sun attributes as they stand."""
        return copy.deepcopy(self._light_state)

    def set_lighting(self, randomization: dict) -> None:
        """``IsaacScene.set_lighting``: dome and sun scaled by ``light_intensity_scale``
        (default 1), colour temperature ``light_temperature_k`` (default 4500) turned on."""
        scale = float(randomization.get("light_intensity_scale", 1.0))
        temp = float(randomization.get("light_temperature_k", DEFAULT_TEMPERATURE_K))
        state = copy.deepcopy(self._light_state)
        for path, base in ISAAC_LIGHTS.items():
            state[path] = {"intensity": base * scale, "enable_color_temperature": True,
                           "color_temperature": temp}
        self._set_light_state(state)

    def restore_lighting(self, snapshot: dict) -> None:
        """``IsaacScene.restore_lighting``: an unauthored temperature switch comes back
        as ``False`` (USD's default), exactly as Isaac authors it."""
        state = copy.deepcopy(self._light_state)
        for path, values in (snapshot or {}).items():
            if path not in state:
                continue
            state[path]["intensity"] = float(values["intensity"])
            state[path]["enable_color_temperature"] = (
                False if values["enable_color_temperature"] is None
                else bool(values["enable_color_temperature"]))
            if values["color_temperature"] is not None:
                state[path]["color_temperature"] = float(values["color_temperature"])
        self._set_light_state(state)

    def _set_light_state(self, state: dict) -> None:
        if self._goal_frame is None:
            self.render_goal()           # the goal picture is always the unedited world's
        f = light_factor(state)                  # refuses unequal factors before any change
        self._light_state = state
        self._light_factor = f
        self._lut = relit_lut(self.look, f)

    def _tone(self, rgb: np.ndarray) -> np.ndarray:
        """The look's tone table, relit when the light changed (RGB only)."""
        if self._lut is None:
            return self.look.apply(rgb)
        from vsbench.world.look import apply_lut

        return apply_lut(rgb, self._lut)


# ---------------------------------------------------------- depth under perturbation
def ray_depth(scene, pose, stride: int = 4) -> dict:
    """Per pixel of a ``stride`` grid: the prim the pixel's ray meets first on the
    compiled model (``mujoco.mj_multiRay`` from the camera centre - independent of the GL
    depth pass), the ray's z-distance, and the rendered depth there (the pose drawn fresh,
    never from the goal cache)."""
    import mujoco

    from vsbench.vs.scene import pose_to_T

    scene.sync()
    _rgb, depth = scene.draw_fresh(pose)
    cam = scene.render_camera
    T = pose_to_T(pose)
    m, d = scene.renderer.model, scene.renderer.data
    mujoco.mj_forward(m, d)
    owner = []
    for path, p in scene._prims.items():
        if p.visible:
            owner += [path] * len(p.objects)
    rays = cam.pixel_rays()
    vv, uu = np.mgrid[stride // 2:cam.height:stride, stride // 2:cam.width:stride]
    vv, uu = vv.ravel(), uu.ravel()
    dw = rays[vv, uu] @ T[:3, :3].T
    norm = np.linalg.norm(dw, axis=1)
    unit = np.ascontiguousarray(dw / norm[:, None])
    n = len(vv)
    gid = np.zeros(n, np.int32)
    dist = np.zeros(n, np.float64)
    mujoco.mj_multiRay(m, d, np.ascontiguousarray(T[:3, 3], dtype=np.float64), unit.ravel(), None,
                       True, -1, gid, dist, None, n, float(cam.far) * 2.0)
    names = [m.geom(i).name for i in range(m.ngeom)]
    hit = np.array([owner[int(names[g].removeprefix("vsbench_obj"))]
                    if g >= 0 and names[g].startswith("vsbench_obj") and dist[j] >= 0 else ""
                    for j, g in enumerate(gid)], dtype=object)
    return {"vv": vv, "uu": uu, "hit": hit, "ray_z": np.where(dist >= 0, dist / norm, 0.0),
            "depth": depth[vv, uu].astype(np.float64), "full_depth": depth}


def depth_gate(clean: dict, pert: dict, scene, edited: set[str], ground_plane=None) -> dict:
    """The depth gate under a perturbation, on the same pixel grid of the same pose.

    * unchanged pixels - the ray meets the same prim in both worlds, that prim's geometry
      was not edited (a re-textured ground counts as unchanged geometry), and no grid
      neighbour (``stride`` px away) is touched by the edit: the rendered depth must equal
      the clean world's (max |difference| reported; gated == 0 by the tests); the share
      without the one-cell margin is reported beside it;
    * every hit pixel of the perturbed world - its depth against its own ray caster's
      (median <= 1 mm, p99 <= 3 mm, U4's gate), split into new / edited geometry and the
      rest; ``ground_plane`` (the analytic ``z = 0`` depth image) adds U4's ground check.
    """
    same = (clean["hit"] == pert["hit"]) & (pert["hit"] != "")
    hit = pert["hit"] != ""
    touched = ~same | np.array([h in edited for h in pert["hit"]])
    # one grid cell (``stride`` px) around every touched sample is left out too: at a
    # silhouette edge of new geometry the rasteriser and the ray caster can disagree about
    # the one pixel the edge passes through (2 mm on a distractor's foot); that pixel is
    # changed, not unchanged, and a ray at the pixel centre cannot tell
    gv, gu = np.unique(pert["vv"]), np.unique(pert["uu"])
    grid = touched.reshape(len(gv), len(gu))
    near = grid.copy()
    near[1:] |= grid[:-1]
    near[:-1] |= grid[1:]
    near[:, 1:] |= grid[:, :-1]
    near[:, :-1] |= grid[:, 1:]
    near[1:, 1:] |= grid[:-1, :-1]
    near[:-1, :-1] |= grid[1:, 1:]
    near[1:, :-1] |= grid[:-1, 1:]
    near[:-1, 1:] |= grid[1:, :-1]
    raw_unchanged = same & ~touched
    unchanged = raw_unchanged & ~near.ravel()
    d_unch = np.abs(pert["depth"][unchanged] - clean["depth"][unchanged])
    d_raw = np.abs(pert["depth"][raw_unchanged] - clean["depth"][raw_unchanged])
    err = np.abs(pert["depth"] - pert["ray_z"])
    new = hit & np.array([h in edited for h in pert["hit"]])

    def _st(e):
        e = np.asarray(e, float) * 1e3
        if e.size == 0:
            return None
        return {"n": int(e.size), "median_mm": float(np.median(e)),
                "p99_mm": float(np.percentile(e, 99)), "max_mm": float(e.max())}

    out = {"n_grid": int(len(pert["hit"])), "n_unchanged": int(unchanged.sum()),
           "unchanged_max_abs_m": float(d_unch.max()) if d_unch.size else 0.0,
           "unchanged_identical_share": float((d_unch == 0).mean()) if d_unch.size else 1.0,
           "n_unchanged_undilated": int(raw_unchanged.sum()),
           "undilated_identical_share": float((d_raw == 0).mean()) if d_raw.size else 1.0,
           "undilated_max_abs_m": float(d_raw.max()) if d_raw.size else 0.0,
           "n_changed_hit": int((~same & hit).sum()),
           "edited_vs_ray": _st(err[new]), "all_vs_ray": _st(err[hit]),
           "miss_with_depth_0": bool((pert["depth"][~hit] == 0).all()),
           "frame_finite": bool(np.isfinite(pert["full_depth"]).all()),
           "frame_nonneg": bool((pert["full_depth"] >= 0).all())}
    if ground_plane is not None:
        g = pert["hit"] == "/World/Ground"
        out["ground_vs_plane"] = _st(np.abs(pert["depth"][g] - ground_plane[pert["vv"][g], pert["uu"][g]]))
    return out
