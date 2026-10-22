"""The object converter: vs-sim's object USDs (and the YCB OBJ scans) -> a MuJoCo-ready
cache, and the placement of a cached object by vs-sim's own scale and layout rules.

docs/PORT_PLAN.md section 5. Two readers, one cache format:

* **USD** (the curated library, ``usd-core``): every ``UsdGeom.Mesh`` under the default
  prim, ``points`` taken into **the frame Isaac's ``/World/Obj_i`` wrapper saw**: the
  asset layer's own transforms (the root prim's ops, then the mesh's) *plus* the
  corrective ``Rx(+90 deg)`` Isaac's stage adds to a Y-up layer (:func:`units_resolve`,
  found by the bbox-goal gate - PORT_PLAN section 5 had assumed none); n-gons
  fan-triangulated, the ``st`` primvar (``faceVarying`` or ``vertex``) as one uv per
  corner, the texture the bound material's diffuse input reads.
* **OBJ** (the YCB google_16k scans of the two cluttered rooms): ``v``, ``vt``, ``f``
  with a small numpy reader; ``map_Kd`` of the ``.mtl`` is the texture. Isaac placed the
  sibling USD (``usd/<id>.usd``, Y-up): its points must equal the OBJ's (refused
  otherwise) and its frame (:func:`isaac_frame`) is applied.

Cache (outside the repository; ``$VSBENCH_OBJECT_CACHE``, default
``~/genvs_ws/vsbench_assets/objects``)::

    <instance_id>/mesh.npz     V float32 (N, 3) native frame and units, UV float32 (N, 2)
                               picture convention (v = 0 at the top row), F int32 (M, 3)
    <instance_id>/texture.png  RGB, longest side <= 4096
    <instance_id>/meta.json    source + md5 + mtime, counts, the bound box, the texture,
                               what was lost, converter + usd-core versions
    INDEX.json                 instance_id -> ok / failed:<reason> / nucleus, bytes, total

**The bound box (Isaac's rule).** Isaac measured every bounding box with
``UsdGeom.BBoxCache(...).ComputeWorldBound(prim).ComputeAlignedRange()``: each mesh's
authored ``extent`` (mesh-local; it counts points that are in no face) is carried through
the mesh's full transform *as a box*, made axis-aligned, and the per-mesh results are
unioned - so a rotated object's box is the aligned hull of its rotated extent box,
looser than the box of its rotated points (4-5 cm on the three rooms' goals). The cache
keeps those boxes (``meta["bound"]["boxes"]``: ``min``, ``max`` and the 4 x 4
``to_native`` of each mesh) and :func:`aligned_bound` applies the rule; the tests pin it
against pxr.

**Placement.** :func:`placed_object` reproduces ``vssim.isaac_scene.IsaacScene.
place_objects`` on the cached mesh: the per-category scale (``target_for``; metric scans,
ids containing ``/``, keep native scale), the analytic seat when there is no layout, the
frozen ``pos`` / ``quat_wxyz`` when there is one; ``Translate(pos) . Orient(quat) .
Scale(s)`` with Isaac's float32 orient and scale. The rules themselves (``target_for``,
``item_quat``, ...) are **not** re-implemented here: they come in through
:class:`PlacementRules`, by default from the copied ``vsbench.vs`` (``rules_from`` takes
any pair of modules with vs-sim's names). This module never imports ``vssim``.

What is lost (recorded per object in ``meta["lost"]``): PBR parameters (roughness,
metallic, specular), MDL materials beyond their diffuse texture / colour, normal maps,
UV transforms, vertex colours, alpha; per-face UV wrap-around (``repeat`` UVs are
wrapped per face into [0, 1]; a face spanning a tile seam is clamped and counted); a
multi-material object gets one texture atlas. MuJoCo then shades with diffuse colour,
one RGB texture and its own lighting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONVERTER_VERSION = "vsbench-objects-1"
CACHE_ENV = "VSBENCH_OBJECT_CACHE"
DEFAULT_CACHE = Path.home() / "genvs_ws" / "vsbench_assets" / "objects"
#: The hard cap of the whole cache (PORT_PLAN D-7): a conversion that would cross it is refused.
SIZE_CAP_BYTES = 10 * 1024**3
TEXTURE_MAX = 4096
INDEX_NAME = "INDEX.json"
_UV_EPS = 1e-6

#: Mesh purposes BBoxCache counts in Isaac's measurement (``isaac_scene._compute_world_bbox``).
ISAAC_BBOX_PURPOSES = ("default", "render", "proxy", "guide")

#: Isaac's stage: Z up, metres.
ISAAC_STAGE_UP_AXIS = "Z"
ISAAC_STAGE_METERS_PER_UNIT = 1.0


def units_resolve(up_axis: str, meters_per_unit: float = 1.0) -> np.ndarray:
    """The corrective transform Isaac's (Kit's) stage puts on a referenced layer whose
    up axis / unit differ from the stage's: ``Rx(+90 deg)`` for a Y-up layer, times
    ``metersPerUnit / 1``. It sits AFTER the layer root's own xform ops (it acts on the
    asset's geometry first): ``native = root_ops . units_resolve . child . points``.

    Measured, not read from Isaac's code (PORT_PLAN section 5 assumed no correction):
    with it the converted meshes reproduce all three rooms' frozen goals to 1e-11 m;
    without it they miss by 5-15 cm; with it before the root ops the drill misses by
    0.15 mm (``results/bridge/u3_bbox_goal_gate.json``). The unit factor is the same rule
    by assumption - no object converted in U3 has ``metersPerUnit != 1``.
    """
    up = str(up_axis).upper()
    if up not in ("Y", "Z"):
        raise ValueError(f"unknown up axis {up_axis!r}")
    F = np.eye(4)
    if up != ISAAC_STAGE_UP_AXIS:
        F[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # Rx(+90): +Y -> +Z
    s = float(meters_per_unit) / ISAAC_STAGE_METERS_PER_UNIT
    if s != 1.0:
        F[:3, :3] *= s
    return F


# --------------------------------------------------------------------------- data
@dataclass
class _Part:
    """One mesh's triangles before merging: points in the native frame, the bound frame."""

    points: np.ndarray  # (P, 3) float64, native frame
    tris: np.ndarray  # (M, 3) int, point indices
    tri_uv: np.ndarray | None  # (M, 3, 2) float64 source convention (v up) or None
    look: tuple  # ("tex", path) or ("color", r, g, b)
    name: str = ""


@dataclass(frozen=True)
class ConvertedObject:
    """A cached object: ``vertices`` native frame and units, ``uvs`` picture convention."""

    instance_id: str
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    texture: np.ndarray
    meta: dict = field(repr=False)

    @property
    def bound_boxes(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``[(min, max, to_native 4 x 4)]``: one local box per source mesh (Isaac's rule)."""
        return [(np.asarray(b["min"], float), np.asarray(b["max"], float),
                 np.asarray(b["to_native"], float).reshape(4, 4))
                for b in self.meta["bound"]["boxes"]]


# ------------------------------------------------------------------- the cache root
def cache_root(override: Path | str | None = None) -> Path:
    """``override``, else ``$VSBENCH_OBJECT_CACHE``, else ``~/genvs_ws/vsbench_assets/objects``."""
    return Path(override or os.environ.get(CACHE_ENV) or DEFAULT_CACHE).expanduser()


def object_dir(instance_id: str, root: Path | str | None = None) -> Path:
    """The cache folder of one instance (ids may hold ``/`` - ``ycb/003_cracker_box``)."""
    parts = Path(str(instance_id)).parts
    if not parts or any(p in ("", ".", "..") for p in parts) or Path(instance_id).is_absolute():
        raise ValueError(f"unusable instance id for a cache path: {instance_id!r}")
    return cache_root(root) / Path(*parts)


def ycb_obj_path(ycb_root: Path | str, instance_id: str) -> Path:
    """``<ycb_root>/<oid>/<oid>/google_16k/textured.obj`` for ``ycb/<oid>``."""
    oid = str(instance_id).split("/", 1)[-1]
    return Path(ycb_root) / oid / oid / "google_16k" / "textured.obj"


def ycb_usd_path(ycb_root: Path | str, instance_id: str) -> Path:
    """The Isaac USD of a YCB scan, ``<ycb_root>/<oid>/usd/<oid>.usd`` (what Isaac placed)."""
    oid = str(instance_id).split("/", 1)[-1]
    return Path(ycb_root) / oid / "usd" / f"{oid}.usd"


# ------------------------------------------------------------------------ helpers
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "md5": _md5(path), "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))}


def _fan(counts: np.ndarray) -> tuple[np.ndarray, int]:
    """Corner indices ``(M, 3)`` of the fan triangulation of polygons with ``counts``."""
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ntri = np.clip(counts - 2, 0, None)
    n_bad = int(np.sum(counts < 3))
    s = np.repeat(starts, ntri)
    j = np.arange(int(ntri.sum())) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
    return np.stack([s, s + j, s + j + 1], axis=1), n_bad


def _wrap_uv(tri_uv: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Bring every face's uvs into [0, 1] by a per-face integer shift (``repeat``
    wrapping), then clamp; returns ``(uv, n_faces_shifted, n_faces_clamped)``."""
    uv = np.asarray(tri_uv, float)
    out = (uv < -_UV_EPS) | (uv > 1.0 + _UV_EPS)
    if not out.any():
        return np.clip(uv, 0.0, 1.0), 0, 0
    shift = np.floor(uv.min(axis=1) + _UV_EPS)  # (M, 2)
    shifted = np.any(shift != 0.0, axis=1)
    uv = uv - shift[:, None, :]
    clamped = np.any((uv > 1.0 + _UV_EPS) | (uv < -_UV_EPS), axis=(1, 2))
    return np.clip(uv, 0.0, 1.0), int(shifted.sum()), int(clamped.sum())


def _load_rgb(path: Path, lost: list) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            a = np.asarray(im.convert("RGBA"))[..., 3]
            if (a < 255).any():
                lost.append(f"alpha channel of {path.name} dropped")
        rgb = im.convert("RGB")
        if max(rgb.size) > TEXTURE_MAX:
            s = TEXTURE_MAX / max(rgb.size)
            new = (max(1, round(rgb.size[0] * s)), max(1, round(rgb.size[1] * s)))
            lost.append(f"{path.name} {rgb.size[0]}x{rgb.size[1]} downscaled to {new[0]}x{new[1]}")
            rgb = rgb.resize(new, Image.LANCZOS)
        return np.asarray(rgb, dtype=np.uint8).copy()


# ---------------------------------------------------------------------- OBJ reader
def read_obj(path: Path | str, frame: dict | None = None) -> tuple[list[_Part], dict]:
    """``v`` / ``vt`` / ``f`` (``a``, ``a/b``, ``a/b/c``, ``a//c``, negative indices,
    n-gons) of a Wavefront OBJ; the texture is ``map_Kd`` of the ``usemtl`` material.

    ``frame`` (:func:`isaac_frame` of the USD Isaac referenced instead) puts the points in
    Isaac's frame; its points must equal the OBJ's (1e-6) or the read is refused."""
    path = Path(path)
    V, VT, faces_v, faces_t, face_mtl = [], [], [], [], []
    mtllibs, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                V.append(line.split()[1:4])
            elif line.startswith("vt "):
                VT.append(line.split()[1:3])
            elif line.startswith("f "):
                toks = line.split()[1:]
                vi, ti = [], []
                for t in toks:
                    p = t.split("/")
                    vi.append(int(p[0]))
                    ti.append(int(p[1]) if len(p) > 1 and p[1] else 0)
                faces_v.append(vi)
                faces_t.append(ti)
                face_mtl.append(cur)
            elif line.startswith("usemtl"):
                cur = line.split(maxsplit=1)[1].strip() if len(line.split()) > 1 else None
            elif line.startswith("mtllib"):
                mtllibs.append(line.split(maxsplit=1)[1].strip())
    P = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(VT, dtype=np.float64).reshape(-1, 2)
    frame_info = {}
    to_native = np.eye(4)
    if frame is not None:
        fp = frame["points"]
        diff = float(np.abs(fp - P).max()) if fp.shape == P.shape else float("inf")
        if not diff <= 1e-6:
            raise ValueError(f"{path}: the OBJ points differ from {frame['usd']}'s "
                             f"(max |d| {diff}) - convert the USD instead")
        to_native = frame["root_ops"] @ frame["units_resolve"]
        M = to_native @ frame["child"]
        P = P @ M[:3, :3].T + M[:3, 3]
        frame_info = {"frame_usd": frame["usd"], "obj_vs_usd_points_max_abs": diff,
                      "up_axis": frame["up_axis"], "meters_per_unit": frame["meters_per_unit"],
                      "root_ops": frame["root_ops"].tolist(),
                      "units_resolve": frame["units_resolve"].tolist(),
                      "usd_boxes": [frame["usd_box"]]}
    maps, colors = {}, {}
    for lib in mtllibs:
        mp = path.parent / lib
        if not mp.exists():
            continue
        name = None
        for line in mp.read_text(errors="replace").splitlines():
            s = line.strip()
            if s.startswith("newmtl"):
                name = s.split(maxsplit=1)[1].strip()
            elif s.startswith("map_Kd") and name is not None:
                maps[name] = path.parent / s.split()[-1]
            elif s.startswith("Kd ") and name is not None:
                colors[name] = tuple(float(x) for x in s.split()[1:4])
    counts = np.array([len(f) for f in faces_v], dtype=np.int64)
    flat_v = np.array([i for f in faces_v for i in f], dtype=np.int64)
    flat_t = np.array([i for f in faces_t for i in f], dtype=np.int64)
    flat_v = np.where(flat_v < 0, len(P) + flat_v, flat_v - 1)
    has_t = flat_t != 0
    flat_t = np.where(flat_t < 0, len(T) + flat_t, flat_t - 1)
    corners, n_bad = _fan(counts)
    face_of_tri = np.repeat(np.arange(len(counts)), np.clip(counts - 2, 0, None))
    tri_mtl = np.array([face_mtl[i] or "" for i in face_of_tri], dtype=object)
    parts = []
    for m in dict.fromkeys(tri_mtl.tolist()):
        sel = tri_mtl == m
        c = corners[sel]
        tris = flat_v[c]
        uv = T[flat_t[c]] if len(T) and bool(has_t[c].all()) else None
        look = (("tex", str(maps[m])) if m in maps else
                ("color",) + tuple(colors.get(m, (0.8, 0.8, 0.8))))
        parts.append(_Part(P, tris, uv, look, name=m or "default"))
    info = {"reader": "obj", "n_points": int(len(P)), "n_polygons": int(len(counts)),
            "n_degenerate_polygons": n_bad, "materials": sorted(set(tri_mtl.tolist())),
            "root_to_native": to_native.tolist(), **frame_info}
    return parts, info


# ---------------------------------------------------------------------- USD reader
def _asset_file(attr_value, attr, layer_dir: Path) -> Path | None:
    if attr_value is None:
        return None
    resolved = getattr(attr_value, "resolvedPath", "") or ""
    if resolved and Path(resolved).exists():
        return Path(resolved)
    raw = getattr(attr_value, "path", str(attr_value))
    if not raw:
        return None
    stack = attr.GetPropertyStack(0) if attr is not None else []
    for spec in stack:
        cand = Path(spec.layer.realPath).parent / raw
        if cand.exists():
            return cand
    cand = layer_dir / raw
    return cand if cand.exists() else None


def _connected_shader(inp):
    from pxr import UsdShade

    if inp is None or not inp.HasConnectedSource():
        return None
    src = inp.GetConnectedSource()
    return UsdShade.Shader(src[0].GetPrim()) if src else None


def _material_look(material, layer_dir: Path, lost: set) -> tuple[tuple, str | None]:
    """``(look, uv_primvar_name)`` of a bound material: the diffuse texture, else the
    diffuse colour. UsdPreviewSurface (``diffuseColor`` -> ``UsdUVTexture.file``) and
    Omniverse MDL (``diffuse_texture`` / ``diffuse_color_constant``)."""
    from pxr import Usd, UsdShade

    shaders, seen = [], set()

    def _add(sh) -> None:
        if sh and sh.GetPrim().IsValid() and sh.GetPrim().GetPath() not in seen:
            seen.add(sh.GetPrim().GetPath())
            shaders.append(sh)

    for ctx in ([""], ["mdl"]):
        try:
            s = material.ComputeSurfaceSource(ctx)
        except Exception:
            continue
        _add(s[0] if isinstance(s, tuple) else s)
    for prim in Usd.PrimRange(material.GetPrim()):
        if prim.IsA(UsdShade.Shader):
            _add(UsdShade.Shader(prim))
    for sh in shaders:
        dc = sh.GetInput("diffuseColor")
        tex = _connected_shader(dc)
        if tex is not None and tex.GetInput("file"):
            fi = tex.GetInput("file")
            p = _asset_file(fi.Get(), fi.GetAttr(), layer_dir)
            if p is not None:
                var = None
                rd = _connected_shader(tex.GetInput("st"))
                if rd is not None and rd.GetInput("varname"):
                    var = rd.GetInput("varname").Get()
                for other in ("normal", "metallic", "roughness", "occlusion", "opacity",
                              "emissiveColor", "specularColor", "clearcoat"):
                    oi = sh.GetInput(other)
                    if oi and oi.HasConnectedSource():
                        lost.add(f"{other} texture ignored")
                for knob, neutral in (("scale", (1, 1, 1, 1)), ("bias", (0, 0, 0, 0))):
                    k = tex.GetInput(knob)
                    if k and k.Get() is not None and tuple(k.Get()) != neutral:
                        lost.add(f"UsdUVTexture {knob} {tuple(k.Get())} ignored")
                xf = _connected_shader(tex.GetInput("st"))
                if xf is not None and xf.GetIdAttr().Get() == "UsdTransform2d":
                    vals = {n: xf.GetInput(n).Get() if xf.GetInput(n) else None
                            for n in ("rotation", "scale", "translation")}
                    if ((vals["rotation"] or 0.0) != 0.0
                            or tuple(vals["scale"] or (1, 1)) != (1, 1)
                            or tuple(vals["translation"] or (0, 0)) != (0, 0)):
                        lost.add(f"UsdTransform2d {vals} on the uvs ignored")
                return ("tex", str(p)), (str(var) if var else None)
        mdl_tex = sh.GetInput("diffuse_texture")
        if mdl_tex and mdl_tex.Get() is not None:
            p = _asset_file(mdl_tex.Get(), mdl_tex.GetAttr(), layer_dir)
            if p is not None:
                lost.add("MDL material reduced to its diffuse_texture")
                return ("tex", str(p)), None
    for sh in shaders:
        for name in ("diffuseColor", "diffuse_color_constant"):
            i = sh.GetInput(name)
            if i and i.Get() is not None and not i.HasConnectedSource():
                return ("color",) + tuple(float(c) for c in i.Get()), None
    return ("color", 0.8, 0.8, 0.8), None


def _sole_texture(layer_dir: Path) -> Path | None:
    d = layer_dir / "textures"
    if not d.is_dir():
        return None
    files = [p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return files[0] if len(files) == 1 else None


def usd_versions() -> str | None:
    try:
        from importlib.metadata import version

        return version("usd-core")
    except Exception:
        return None


def read_usd(path: Path | str) -> tuple[list[_Part], dict]:
    """Every renderable ``UsdGeom.Mesh`` under the default prim, points in the asset
    layer's world frame (``ComputeLocalToWorldTransform(0)``), plus the bound frame."""
    from pxr import Usd, UsdGeom, UsdShade

    path = Path(path)
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"usd-core cannot open {path}")
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    layer_dir = path.parent
    root_l2w = (np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0), float).T
                if root.IsA(UsdGeom.Xformable) else np.eye(4))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    to_native = root_l2w @ units_resolve(up_axis, mpu)  # the frame Isaac's wrapper saw
    xf_cache = UsdGeom.XformCache(0)
    parts, lost, meshes, boxes = [], set(), [], []
    n_bad_total, skipped = 0, []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        purpose = img.ComputePurpose()
        if img.ComputeVisibility(0) == UsdGeom.Tokens.invisible:
            skipped.append(f"{prim.GetPath()} (invisible)")
            continue
        boxes.append(_mesh_box(UsdGeom.Mesh(prim), to_native @ np.array(
            xf_cache.ComputeRelativeTransform(prim, root)[0], float).T))
        if purpose == UsdGeom.Tokens.guide:
            skipped.append(f"{prim.GetPath()} (guide purpose; in Isaac's bbox, not rendered)")
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get(0)
        counts = mesh.GetFaceVertexCountsAttr().Get(0)
        idx = mesh.GetFaceVertexIndicesAttr().Get(0)
        if not pts or not counts or not idx:
            skipped.append(f"{prim.GetPath()} (empty)")
            continue
        P = np.asarray(pts, dtype=np.float64)
        rel, _ = xf_cache.ComputeRelativeTransform(prim, root)
        M = to_native @ np.array(rel, float).T  # column-vector convention
        Pn = P @ M[:3, :3].T + M[:3, 3]
        counts = np.asarray(counts, dtype=np.int64)
        idx = np.asarray(idx, dtype=np.int64)
        corners, n_bad = _fan(counts)
        n_bad_total += n_bad
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            corners = corners[:, [0, 2, 1]]
        if np.linalg.det(M[:3, :3]) < 0:  # a mirroring transform flips the winding
            corners = corners[:, [0, 2, 1]]
        tris = idx[corners]
        # material(s): direct binding, else per GeomSubset
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        look, var = (_material_look(mat, layer_dir, lost) if mat else (None, None))
        face_look = [look] * len(counts)
        subsets = UsdGeom.Subset.GetGeomSubsets(mesh, UsdGeom.Tokens.face, "materialBind")
        for ss in subsets:
            smat, _ = UsdShade.MaterialBindingAPI(ss.GetPrim()).ComputeBoundMaterial()
            if smat:
                sl, sv = _material_look(smat, layer_dir, lost)
                var = var or sv
                for fi in np.asarray(ss.GetIndicesAttr().Get(0) or [], dtype=np.int64):
                    face_look[int(fi)] = sl
        if any(lk is None for lk in face_look):
            dcol = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
            col = None
            if dcol and dcol.HasValue() and dcol.GetInterpolation() == UsdGeom.Tokens.constant:
                v = dcol.Get(0)
                col = tuple(float(c) for c in (v[0] if len(v) else (0.8, 0.8, 0.8)))
            sole = _sole_texture(layer_dir)
            fb = ("tex", str(sole)) if sole is not None else ("color",) + (col or (0.8, 0.8, 0.8))
            if sole is not None:
                lost.add("no material bound: the single file under textures/ used")
            face_look = [lk if lk is not None else fb for lk in face_look]
        # uvs, one per corner
        pv_api = UsdGeom.PrimvarsAPI(prim)
        pv = pv_api.GetPrimvar(var) if var else None
        if not pv or not pv.HasValue():
            pv = pv_api.GetPrimvar("st")
        if not pv or not pv.HasValue():
            cands = [p for p in pv_api.GetPrimvars() if str(p.GetTypeName()) in ("texCoord2f[]", "float2[]")]
            pv = cands[0] if cands else None
        tri_uv = None
        if pv:
            uv = np.asarray(pv.ComputeFlattened(0), dtype=np.float64).reshape(-1, 2)
            interp = pv.GetInterpolation()
            if interp == UsdGeom.Tokens.faceVarying and len(uv) == len(idx):
                tri_uv = uv[corners]
            elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(uv) == len(P):
                tri_uv = uv[tris]
            else:
                lost.add(f"uv primvar of {prim.GetPath().name} has interpolation {interp} "
                         f"({len(uv)} values) - not used")
        if UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor") and \
                UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor").GetInterpolation() \
                in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.faceVarying):
            lost.add("vertex colours (displayColor) ignored")
        face_of_tri = np.repeat(np.arange(len(counts)), np.clip(counts - 2, 0, None))
        tri_look = [face_look[i] for i in face_of_tri]
        for lk in dict.fromkeys(tri_look):
            sel = np.array([t == lk for t in tri_look], dtype=bool)
            parts.append(_Part(Pn, tris[sel], None if tri_uv is None else tri_uv[sel], lk,
                               name=str(prim.GetPath())))
        meshes.append({"prim": str(prim.GetPath()), "purpose": str(purpose),
                       "n_points": int(len(P)), "n_polygons": int(len(counts)),
                       "max_polygon": int(counts.max())})
    if not parts:
        raise RuntimeError(f"{path}: no renderable mesh under {root.GetPath()}")
    info = {"reader": "usd", "default_prim": str(root.GetPath()),
            "meters_per_unit": mpu, "up_axis": up_axis, "meshes": meshes,
            "root_ops": root_l2w.tolist(), "units_resolve": units_resolve(up_axis, mpu).tolist(),
            "root_to_native": to_native.tolist(), "n_degenerate_polygons": n_bad_total,
            "skipped_prims": skipped, "lost": sorted(lost), "usd_boxes": boxes}
    return parts, info


def _mesh_box(mesh, to_native: np.ndarray) -> dict:
    """One mesh's contribution to Isaac's box: its authored ``extent`` (else the extent of
    all its points) in mesh-local space and the mesh-local -> native matrix. BBoxCache
    (``useExtentsHint=False``) carries each gprim's extent box through its full transform
    and unions the axis-aligned results (tests/world/test_world_assets.py pins it against pxr)."""
    from pxr import UsdGeom

    ext = mesh.GetExtentAttr().Get(0)
    P = np.asarray(mesh.GetPointsAttr().Get(0), np.float64)
    authored = ext is not None and len(ext) == 2
    if not authored:
        ext = UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get(0))
    lo, hi = np.asarray(ext[0], float), np.asarray(ext[1], float)
    d = float(max(np.abs(P.min(0) - lo).max(), np.abs(P.max(0) - hi).max())) if len(P) else 0.0
    return {"prim": str(mesh.GetPath()), "min": lo.tolist(), "max": hi.tolist(),
            "to_native": np.asarray(to_native, float).tolist(), "extent_authored": bool(authored),
            "extent_vs_all_points_max_abs": d}


def isaac_frame(usd_path: Path | str) -> dict:
    """The frame a USD's single mesh is placed in by Isaac (``root_ops . units_resolve .
    child``), plus its points - for an OBJ whose sibling USD is what Isaac referenced."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(f"{usd_path}: {len(meshes)} meshes; an OBJ frame needs exactly one")
    root_l2w = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0), float).T
    up_axis, mpu = str(UsdGeom.GetStageUpAxis(stage)), float(UsdGeom.GetStageMetersPerUnit(stage))
    rel, _ = UsdGeom.XformCache(0).ComputeRelativeTransform(meshes[0], root)
    return {"usd": str(usd_path), "up_axis": up_axis, "meters_per_unit": mpu,
            "root_ops": root_l2w, "units_resolve": units_resolve(up_axis, mpu),
            "usd_box": _mesh_box(UsdGeom.Mesh(meshes[0]), root_l2w @ units_resolve(up_axis, mpu)
                                 @ np.array(rel, float).T),
            "child": np.array(rel, float).T,
            "points": np.asarray(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get(0), np.float64)}


# ----------------------------------------------------------------- merge + texture
def _merge(parts: list[_Part], lost: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """One mesh, one texture: split vertices per ``(point, uv)``; several looks -> an atlas."""
    looks = list(dict.fromkeys(p.look for p in parts))
    images = {}
    for lk in looks:
        if lk[0] == "tex":
            images[lk] = _load_rgb(Path(lk[1]), lost)
    g = 1 if len(looks) == 1 else math.ceil(math.sqrt(len(looks)))
    if g == 1:
        lk = looks[0]
        tex = images[lk] if lk in images else np.full((8, 8, 3), _rgb255(lk[1:]), np.uint8)
        tile = None
    else:
        side = max([max(im.shape[:2]) for im in images.values()] + [64])
        tile = int(min(TEXTURE_MAX // g, side))
        tex = np.zeros((g * tile, g * tile, 3), np.uint8)
        from PIL import Image

        for k, lk in enumerate(looks):
            r, c = divmod(k, g)
            if lk in images:
                im = Image.fromarray(images[lk]).resize((tile, tile), Image.LANCZOS)
                tex[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = np.asarray(im)
            else:
                tex[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = _rgb255(lk[1:])
        lost.append(f"{len(looks)} materials packed into one {g}x{g} atlas of {tile} px tiles")
    allV, allUV, allF = [], [], []
    n_shift = n_clamp = n_nouv = 0
    offset = 0
    for p in parts:
        k = looks.index(p.look)
        m = len(p.tris)
        if p.tri_uv is not None and p.look[0] == "tex":
            uv, s, c = _wrap_uv(p.tri_uv)
            n_shift += s
            n_clamp += c
        else:
            if p.look[0] == "tex":
                n_nouv += m
            uv = np.full((m, 3, 2), 0.5)
        uv = uv.copy()
        uv[..., 1] = 1.0 - uv[..., 1]  # USD / OBJ: v up from the bottom -> picture rows
        if tile is not None:
            r, c = divmod(k, g)
            inset = 0.5 / tile
            uv = inset + uv * (1.0 - 2.0 * inset)
            uv = (uv + np.array([c, r], float)) / g
        key = np.concatenate([p.tris.reshape(-1, 1).astype(np.float64), uv.reshape(-1, 2)], axis=1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        allV.append(p.points[uniq[:, 0].astype(np.int64)])
        allUV.append(uniq[:, 1:])
        allF.append(inv.reshape(-1, 3) + offset)
        offset += len(uniq)
    V = np.concatenate(allV).astype(np.float32)
    UV = np.concatenate(allUV).astype(np.float32)
    F = np.concatenate(allF).astype(np.int32)
    if n_shift:
        lost.append(f"repeat uvs wrapped per face into [0, 1] ({n_shift} faces shifted)")
    if n_clamp:
        lost.append(f"{n_clamp} faces span a texture seam and were clamped to [0, 1]")
    if n_nouv:
        lost.append(f"{n_nouv} textured triangles without uvs sample the texture centre")
    stats = {"n_looks": len(looks), "atlas_grid": g, "atlas_tile_px": tile,
             "uv_faces_shifted": n_shift, "uv_faces_clamped": n_clamp,
             "looks": [list(lk) for lk in looks]}
    return V, UV, F, tex, stats


def _rgb255(c) -> np.ndarray:
    return np.clip(np.round(np.asarray(c, float)[:3] * 255.0), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------- index + cap
def read_index(root: Path | str | None = None) -> dict:
    p = cache_root(root) / INDEX_NAME
    if p.exists():
        return json.loads(p.read_text())
    return {"converter": CONVERTER_VERSION, "cap_bytes": SIZE_CAP_BYTES, "total_bytes": 0, "objects": {}}


def _write_index(index: dict, root: Path) -> None:
    index["total_bytes"] = int(sum(int(v.get("bytes", 0)) for v in index["objects"].values()))
    index["cap_bytes"] = SIZE_CAP_BYTES
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (INDEX_NAME + ".tmp")
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True))
    os.replace(tmp, root / INDEX_NAME)


def mark_index(instance_id: str, status: str, root: Path | str | None = None, **extra) -> None:
    """Record ``ok`` / ``failed:<reason>`` / ``nucleus`` for an id in ``INDEX.json``."""
    r = cache_root(root)
    index = read_index(r)
    rec = {"status": status, "bytes": 0}
    rec.update(extra)
    index["objects"][str(instance_id)] = rec
    _write_index(index, r)


def _dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


# --------------------------------------------------------------------- the converter
def convert_source(instance_id: str, source: Path | str, *, root: Path | str | None = None,
                   kind: str | None = None, extra_meta: dict | None = None,
                   cap_bytes: int = SIZE_CAP_BYTES, overwrite: bool = False,
                   frame_usd: Path | str | None = None) -> dict:
    """Convert one object (``kind`` ``usd`` or ``obj``, default by suffix) into the cache.

    Writes ``mesh.npz``, ``texture.png`` and ``meta.json`` atomically (temp folder +
    rename) and records the id in ``INDEX.json``. Refuses when the cache would cross
    ``cap_bytes``. ``frame_usd`` (OBJ only): the USD Isaac referenced in its place - its
    up axis, unit and transforms define the native frame (the YCB scans). Returns the meta.
    """
    t0 = time.perf_counter()
    source = Path(source)
    r = cache_root(root)
    dest = object_dir(instance_id, r)
    if dest.exists() and not overwrite:
        return json.loads((dest / "meta.json").read_text())
    kind = kind or ("obj" if source.suffix.lower() == ".obj" else "usd")
    lost: list = []
    if kind == "obj":
        parts, info = read_obj(source, isaac_frame(frame_usd) if frame_usd else None)
    elif kind == "usd":
        parts, info = read_usd(source)
        lost.extend(info.pop("lost"))
    else:
        raise ValueError(f"unknown source kind {kind!r}")
    V, UV, F, tex, stats = _merge(parts, lost)
    if not np.isfinite(V).all() or not np.isfinite(UV).all():
        raise ValueError(f"{instance_id}: non-finite vertices or uvs after conversion")
    # loose points (in no face) are part of the bound: USD's extent counts them and so did
    # Isaac's box - handal_drill_25 carries 833 that reach 0.24 mm (placed) below its faces
    allP = {id(p.points): p for p in parts}
    used: dict = {}
    for p in parts:
        used.setdefault(id(p.points), []).append(p.tris.ravel())
    n_points = int(sum(len(p.points) for p in allP.values()))
    n_loose = int(n_points - sum(len(np.unique(np.concatenate(t))) for t in used.values()))
    boxes = info.pop("usd_boxes", None)
    if boxes:
        rule = ("usd: per source mesh, the authored extent (mesh-local) through its "
                "transform to native, then the placement, axis-aligned, unioned - Isaac's "
                "BBoxCache(...).ComputeWorldBound(prim).ComputeAlignedRange()")
    else:  # an OBJ with no USD behind it: the box of all its points, native frame
        P_all = np.concatenate([p.points for p in allP.values()])
        boxes = [{"prim": "obj", "min": P_all.min(0).tolist(), "max": P_all.max(0).tolist(),
                  "to_native": np.eye(4).tolist(), "extent_authored": False}]
        rule = "points: the box of all the OBJ's points (no USD behind this object)"
    meta = {
        "instance_id": str(instance_id), "converter": CONVERTER_VERSION,
        "usd_core": usd_versions(), "numpy": np.__version__,
        "source": {"kind": kind, **_file_record(source)},
        "counts": {"vertices": int(len(V)), "faces": int(len(F))},
        "native_bbox": {"min": V.min(0).astype(float).tolist(), "max": V.max(0).astype(float).tolist()},
        "bound": {"rule": rule, "boxes": boxes, "n_points": n_points,
                  "n_points_not_in_a_face": n_loose},
        "reader": info, "merge": stats, "lost": sorted(set(lost)),
    }
    if stats["looks"] and stats["looks"][0][0] == "tex" and stats["n_looks"] == 1:
        meta["texture_source"] = _file_record(Path(stats["looks"][0][1]))
    if extra_meta:
        meta.update(extra_meta)
    meta["texture"] = {"height": int(tex.shape[0]), "width": int(tex.shape[1])}
    r.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".convert-", dir=r))
    try:
        np.savez_compressed(tmp / "mesh.npz", V=V, UV=UV, F=F)
        from PIL import Image

        Image.fromarray(tex).save(tmp / "texture.png", compress_level=6)
        meta["seconds"] = round(time.perf_counter() - t0, 3)
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
        nbytes = _dir_bytes(tmp)
        index = read_index(r)
        others = sum(int(v.get("bytes", 0)) for k, v in index["objects"].items() if k != str(instance_id))
        if others + nbytes > cap_bytes:
            raise RuntimeError(f"{instance_id}: the cache would hold {others + nbytes} bytes, "
                               f"over the cap of {cap_bytes} (PORT_PLAN D-7) - refused")
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    mark_index(instance_id, "ok", r, bytes=nbytes, seconds=meta["seconds"],
               source=str(source), vertices=int(len(V)), faces=int(len(F)))
    return meta


def load_object(instance_id: str, root: Path | str | None = None) -> ConvertedObject:
    """A converted object from the cache (``FileNotFoundError`` names the converter)."""
    d = object_dir(instance_id, root)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{instance_id!r} is not in the object cache {cache_root(root)} - "
                                f"run tools/convert_objects.py convert {instance_id!r}")
    meta = json.loads((d / "meta.json").read_text())
    with np.load(d / "mesh.npz") as z:
        V, UV, F = z["V"], z["UV"], z["F"]
    from PIL import Image

    with Image.open(d / "texture.png") as im:
        tex = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return ConvertedObject(str(instance_id), V, F, UV, tex, meta)


# ------------------------------------------------------------------- placement
@dataclass(frozen=True)
class PlacementRules:
    """vs-sim's placement rules, as functions (never re-implemented here)."""

    target_for: Callable  # assets.target_for(category) -> (mode, target_m)
    item_quat: Callable  # scenes.item_quat(item, entry, yaw_deg=None) -> wxyz
    uses_placement: Callable  # scenes.uses_placement(item) -> bool
    long_axis_yaw: Callable  # scenes.long_axis_yaw(extent) -> deg
    long_axis_x: str  # scenes.LONG_AXIS_X


def rules_from(assets_mod, scenes_mod) -> PlacementRules:
    """Rules from a pair of modules with vs-sim's names (``vsbench.vs`` or a ``vssim``)."""
    return PlacementRules(assets_mod.target_for, scenes_mod.item_quat, scenes_mod.uses_placement,
                          scenes_mod.long_axis_yaw, scenes_mod.LONG_AXIS_X)


def default_rules() -> PlacementRules:
    """The copied vs-sim rules (``vsbench.vs.assets`` / ``vsbench.vs.scenes``)."""
    try:
        from vsbench.vs import assets as A
        from vsbench.vs import scenes as S
    except ImportError as e:  # pragma: no cover - before the copy lands
        raise ImportError("vsbench.vs (the copied vs-sim) is not importable here; pass "
                          "rules=rules_from(<assets module>, <scenes module>)") from e
    return rules_from(A, S)


def quat_matrix(q) -> np.ndarray:
    """3 x 3 rotation of a ``(w, x, y, z)`` quaternion exactly as a USD orient op builds it
    (``GfRotation(quat)``: angle ``2 acos(w)`` about the *normalised* imaginary part).

    For a unit quaternion this is the usual matrix; for a float32 one it is not: the
    ``orient:ingestfix`` op of handal_drill_25, ``(0.99999994, ~1e-8 ...)``, is a 0.039 deg
    turn in USD (and so in Isaac's box), not the identity the plain formula gives
    (tests/world/test_world_assets.py pins this against pxr).
    """
    w, x, y, z = (float(c) for c in q)
    v = np.array([x, y, z])
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return np.eye(3)
    a = v / n
    ang = 2.0 * math.acos(max(-1.0, min(1.0, w)))
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def _f32(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).astype(np.float64)


def placement_matrix(pos, quat_wxyz, scale: float) -> np.ndarray:
    """``Translate(pos) . Orient(quat) . Scale(s)`` (Isaac's op order; orient and scale
    rounded to float32 like Isaac's ``Quatf`` / ``Vec3f`` ops, translate double)."""
    T = np.eye(4)
    T[:3, :3] = quat_matrix(_f32(quat_wxyz)) * float(np.float32(scale))
    T[:3, 3] = np.asarray(pos, float)
    return T


def aligned_bound(obj: ConvertedObject, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isaac's measured box of ``obj`` under the placement ``T``: the 8 corners of the
    cached bound box through ``T . to_native``, then axis-aligned."""
    T = np.asarray(T, float)
    los, his = [], []
    for lo, hi, to_native in obj.bound_boxes:
        M = T @ to_native
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        W = c @ M[:3, :3].T + M[:3, 3]
        los.append(W.min(0))
        his.append(W.max(0))
    return np.min(los, axis=0), np.max(his, axis=0)


def points_bound(obj: ConvertedObject, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The tight box of the placed points (a diagnostic; not Isaac's rule)."""
    T = np.asarray(T, float)
    W = obj.vertices.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
    return W.min(0), W.max(0)


@dataclass(frozen=True)
class PlacedObject:
    """A cached object at its scene pose: ``mesh`` for the renderer, the rest for checks."""

    obj: ConvertedObject
    pos: np.ndarray
    quat_wxyz: np.ndarray
    scale: float
    record: dict

    @property
    def matrix(self) -> np.ndarray:
        return placement_matrix(self.pos, self.quat_wxyz, self.scale)

    @property
    def mesh(self):
        from vsbench.scene import Material, Mesh

        pose = placement_matrix(self.pos, self.quat_wxyz, 1.0)
        return Mesh(pose, self.obj.vertices.astype(np.float64) * float(np.float32(self.scale)),
                    self.obj.faces, Material(color=(1.0, 1.0, 1.0), texture=self.obj.texture),
                    uvs=self.obj.uvs)

    def bound(self, rule: str = "usd") -> tuple[np.ndarray, np.ndarray]:
        if rule == "usd":
            return aligned_bound(self.obj, self.matrix)
        if rule == "points":
            return points_bound(self.obj, self.matrix)
        raise ValueError(f"unknown bound rule {rule!r} (usd, points)")


def _extent(obj: ConvertedObject, quat, scale: float) -> np.ndarray:
    lo, hi = aligned_bound(obj, placement_matrix((0.0, 0.0, 0.0), quat, scale))
    return hi - lo


def _category_scale(obj, rules: PlacementRules, category: str, quat) -> float:
    # IsaacScene._measure_scale: the extent at the origin under quat, scale 1
    mode, target_m = rules.target_for(category)
    ext = _extent(obj, quat, 1.0)
    ref = float(ext[2]) if mode == "height" else float(max(ext))
    return target_m / ref if ref > 1e-9 else 1.0


def placed_object(obj: ConvertedObject, item: dict, entry: dict | None = None,
                  layout_entry: dict | None = None, *,
                  rules: PlacementRules | None = None) -> PlacedObject:
    """``IsaacScene.place_objects`` for one item, on the cached mesh.

    ``item`` is the scene item (after ``placed_item`` / ``scatter_items``), ``entry`` its
    asset-library entry, ``layout_entry`` the frozen ``{pos, quat_wxyz}`` (then it wins
    over the analytic seat, the scale stays).
    """
    rules = rules or default_rules()
    iid = str(item["instance_id"])
    placement = None
    if rules.uses_placement(item):
        yaw = item.get("yaw_deg")
        if str(yaw) == rules.long_axis_x:
            yaw_res = float(rules.long_axis_yaw(_extent(obj, rules.item_quat(item, entry, yaw_deg=0.0), 1.0)))
        else:
            yaw_res = float(yaw or 0.0)
        quat = np.asarray(rules.item_quat(item, entry, yaw_deg=yaw_res), float)
        ext = _extent(obj, quat, 1.0)
        if item.get("size_m") is not None:
            ref = float(np.max(ext))
            scale = float(item["size_m"]) / ref if ref > 1e-9 else 1.0
        elif "/" in iid:
            scale = float(item.get("scale", 1.0))
        else:
            if not item.get("category"):
                raise KeyError(f"{iid}: a curated instance needs its `category` or a `size_m`")
            scale = _category_scale(obj, rules, str(item["category"]), quat)
        placement = {"canonical": item.get("canonical"),
                     "upright_yaw_add_deg": float(item.get("upright_yaw_add_deg") or 0.0),
                     "pre_rot_x_deg": float(item.get("pre_rot_x_deg") or 0.0),
                     "yaw_deg": yaw_res, "yaw_rule": str(yaw) if yaw is not None else None,
                     "size_m": None if item.get("size_m") is None else float(item["size_m"]),
                     "native_extent_m": [float(e) for e in ext]}
    else:
        quat = np.asarray(rules.item_quat(item, entry), float)
        if "/" in iid:
            scale = float(item.get("scale", 1.0))
        else:
            if not item.get("category"):
                raise KeyError(f"{iid}: a curated instance needs its `category` to be scaled")
            scale = _category_scale(obj, rules, str(item["category"]), quat)
    # IsaacScene._place_pose: centre at the origin, then seat on z = 0 at (dx, dy)
    lo, hi = aligned_bound(obj, placement_matrix((0.0, 0.0, 0.0), quat, scale))
    centre = (lo + hi) * 0.5
    t = -centre
    lo2, hi2 = aligned_bound(obj, placement_matrix(t, quat, scale))
    ext = hi2 - lo2
    pos = np.array([t[0] + float(item.get("dx", 0.0)), t[1] + float(item.get("dy", 0.0)),
                    t[2] + ext[2] / 2.0])
    record = {"instance_id": iid, "scale": float(scale), "source": "analytic"}
    if placement is not None:
        record["placement"] = placement
    if layout_entry is not None:
        if str(layout_entry.get("instance_id", iid)) != iid:
            raise ValueError(f"layout entry {layout_entry.get('instance_id')!r} != item {iid!r}")
        pos = np.asarray(layout_entry["pos"], float)
        quat = _f32(layout_entry["quat_wxyz"])
        record["source"] = "layout"
    return PlacedObject(obj, pos, _f32(quat), float(np.float32(scale)), record)


def placed_mesh(obj: ConvertedObject, item: dict, entry: dict | None = None,
                layout_entry: dict | None = None, *, rules: PlacementRules | None = None):
    """The :class:`vsbench.scene.Mesh` of :func:`placed_object` (pose ``T(pos) R(quat)``,
    vertices pre-scaled by ``s``)."""
    return placed_object(obj, item, entry, layout_entry, rules=rules).mesh


def scene_bbox(placed, rule: str = "usd") -> tuple[np.ndarray, np.ndarray]:
    """The union of the placed objects' boxes - ``place_objects``'s ``bbox_min/max``."""
    boxes = [p.bound(rule) for p in placed]
    if not boxes:
        raise ValueError("no objects")
    return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
