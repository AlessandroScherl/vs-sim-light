"""The object converter (vsbench.world.assets) and the bbox-goal gate.

Fast tests build tiny OBJ / USD files in a temp folder; ``usd-core`` (pxr) is the
reference for every USD rule: the composed points, Isaac's box rule, the orient op's
matrix and the up-axis correction. The cache tests run when the rooms' objects are converted
(``tools/convert_objects.py convert --rooms --coverage``); the gate test also needs the
source USD files (``$VSSIM_ASSET_ROOT``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from vsbench.world import assets as A

REPO = Path(__file__).resolve().parents[2]
pxr = pytest.importorskip("pxr", reason="usd-core not installed")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402


# ------------------------------------------------------------------------ helpers
def _png(path: Path, h=8, w=16, seed=0) -> np.ndarray:
    from PIL import Image

    img = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)
    return img


def _fake_rules(target=0.25):
    def target_for(cat):
        return ("height", 0.14) if cat == "glass" else ("longest", target)

    def item_quat(item, entry=None, yaw_deg=None):
        pre = item.get("pre_rot_x_deg")
        if not pre:
            return np.array([1.0, 0.0, 0.0, 0.0])
        h = np.radians(float(pre)) / 2
        return np.array([np.cos(h), np.sin(h), 0.0, 0.0])

    return A.PlacementRules(target_for, item_quat, lambda item: False, lambda e: 0.0, "long_axis_x")


def _usd_asset(tmp: Path, up="Y", left=False, repeat=True, two_materials=False) -> Path:
    """A Y-up asset: root /World with translate . orient . scale ops, a child xform, a
    quad + triangle mesh with a loose point, faceVarying st, UsdPreviewSurface texture."""
    (tmp / "textures").mkdir(parents=True, exist_ok=True)
    _png(tmp / "textures" / "a.png", seed=1)
    if two_materials:
        _png(tmp / "textures" / "b.png", 8, 8, seed=2)
    path = tmp / "asset.usda"
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y if up == "Y" else UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(0.01, -0.02, 0.03))
    root.AddOrientOp().Set(Gf.Quatf(0.9238795, Gf.Vec3f(0.0, 0.3826834, 0.0)))  # 45 deg about y
    root.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 2.0))
    child = UsdGeom.Xform.Define(st, "/World/part")
    child.AddRotateZOp().Set(30.0)
    child.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.01, 0.0))
    m = UsdGeom.Mesh.Define(st, "/World/part/mesh")
    pts = [(0, 0, 0), (0.1, 0, 0), (0.1, 0.05, 0), (0, 0.05, 0), (0.05, 0.02, 0.04), (0.3, -0.2, 0.1)]
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])  # the last point is in no face
    m.CreateFaceVertexCountsAttr([4, 3])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 0, 1, 4])
    if left:
        m.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    m.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(m.GetPointsAttr().Get()))
    st_uv = [(0, 0), (1, 0), (1, 1), (0, 1), (1.2, 0.1), (1.4, 0.3), (1.7, 0.9)] if repeat else \
        [(0, 0), (1, 0), (1, 1), (0, 1), (0.2, 0.1), (0.4, 0.3), (0.7, 0.9)]
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                 UsdGeom.Tokens.faceVarying)
    pv.Set([Gf.Vec2f(*u) for u in st_uv])

    def material(name, png):
        mat = UsdShade.Material.Define(st, f"/World/Looks/{name}")
        sh = UsdShade.Shader.Define(st, f"/World/Looks/{name}/surf")
        sh.CreateIdAttr("UsdPreviewSurface")
        tex = UsdShade.Shader.Define(st, f"/World/Looks/{name}/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{png}")
        rd = UsdShade.Shader.Define(st, f"/World/Looks/{name}/st")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    mat_a = material("A", "a.png")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat_a)
    if two_materials:
        mat_b = material("B", "b.png")
        ss = UsdGeom.Subset.CreateGeomSubset(m, "tri", UsdGeom.Tokens.face, [1], "materialBind")
        UsdShade.MaterialBindingAPI.Apply(ss.GetPrim()).Bind(mat_b)
    st.Save()
    return path


def _isaac_like_bbox(asset: Path, pos, quat, scale):
    """pxr: /World/Obj = T . Orient(float) . Scale(float) over the reference, Kit's
    rotateX:unitsResolve appended for a Y-up layer; Isaac's BBoxCache call."""
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    xf = UsdGeom.Xform.Define(st, "/World/Obj")
    inner = st.DefinePrim("/World/Obj/asset")
    inner.GetReferences().AddReference(str(asset))
    if UsdGeom.GetStageUpAxis(Usd.Stage.Open(str(asset))) == UsdGeom.Tokens.y:
        UsdGeom.Xformable(inner).AddRotateXOp(opSuffix="unitsResolve").Set(90.0)
    xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Quatf(float(quat[0]), Gf.Vec3f(*[float(c) for c in quat[1:]])))
    xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    cache = UsdGeom.BBoxCache(0.0, includedPurposes=list(A.ISAAC_BBOX_PURPOSES), useExtentsHint=False)
    r = cache.ComputeWorldBound(inner).ComputeAlignedRange()
    mesh_world = np.array(UsdGeom.Xformable(st.GetPrimAtPath("/World/Obj/asset/part/mesh"))
                          .ComputeLocalToWorldTransform(0), float).T
    return np.array(r.GetMin()), np.array(r.GetMax()), mesh_world


# ------------------------------------------------------------------------- OBJ
def _obj(tmp: Path) -> Path:
    _png(tmp / "tex.png")
    (tmp / "m.mtl").write_text("newmtl mat0\nKd 1 1 1\nmap_Kd tex.png\n")
    (tmp / "m.obj").write_text(
        "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 5 5 5\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nusemtl mat0\n"
        "f 1/1/1 2/2/1 3/3/1 4/4/1\nf -5/-4 -3/-2 -2/-1\n")
    return tmp / "m.obj"


def test_obj_reader_fans_ngons_flips_v_and_keeps_loose_points_in_the_bound(tmp_path):
    meta = A.convert_source("t/quad", _obj(tmp_path), root=tmp_path / "cache")
    o = A.load_object("t/quad", tmp_path / "cache")
    assert o.faces.shape == (3, 3) and o.faces.max() < len(o.vertices)
    # n-gon fan (0,1,2),(0,2,3) + the negative-index triangle (1,3,4)
    tris = {tuple(sorted(map(tuple, np.round(o.vertices[f], 6)))) for f in o.faces}
    assert ((0, 0, 0), (1, 0, 0), (1, 1, 0)) in tris and ((0, 0, 0), (0, 1, 0), (1, 1, 0)) in tris
    # obj v = 0 is the bottom row -> picture v' = 1 - v
    i = np.where(np.all(o.vertices == [0, 1, 0], axis=1))[0][0]
    assert np.allclose(o.uvs[i], [0.0, 0.0])
    j = np.where(np.all(o.vertices == [0, 0, 0], axis=1))[0][0]
    assert np.allclose(o.uvs[j], [0.0, 1.0])
    assert meta["bound"]["boxes"][0]["max"] == [5.0, 5.0, 5.0]  # the loose vertex counts, like USD extent
    assert meta["bound"]["n_points_not_in_a_face"] == 1
    assert o.texture.shape == (8, 16, 3)
    idx = A.read_index(tmp_path / "cache")
    assert idx["objects"]["t/quad"]["status"] == "ok" and idx["total_bytes"] > 0


def test_obj_with_a_frame_usd_refuses_different_points(tmp_path):
    asset = _usd_asset(tmp_path / "u")
    with pytest.raises(ValueError, match="differ"):
        A.convert_source("t/x", _obj(tmp_path), root=tmp_path / "cache", frame_usd=asset)


def test_the_cap_refuses_and_leaves_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="cap"):
        A.convert_source("t/q", _obj(tmp_path), root=tmp_path / "cache", cap_bytes=100)
    assert not (tmp_path / "cache" / "t" / "q").exists()
    assert not list((tmp_path / "cache").glob(".convert-*"))


def test_cache_paths_refuse_escapes(tmp_path):
    for bad in ("../x", "/abs", "a/../b", ""):
        with pytest.raises(ValueError):
            A.object_dir(bad, tmp_path)


# ------------------------------------------------------------------------- USD
def test_orient_op_matrix_matches_usd_for_float32_quats():
    st = Usd.Stage.CreateInMemory()
    xf = UsdGeom.Xform.Define(st, "/A")
    op = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
    rng = np.random.default_rng(3)
    qs = [(0.9999999403953552, 1.0176e-10, 2.0849e-09, -1.5525e-08),  # handal_drill_25's ingestfix
          (0.7071067690849304, -0.7071067690849304, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
    qs += [tuple(q / np.linalg.norm(q)) for q in rng.normal(size=(5, 4))]
    for q in qs:
        qf = Gf.Quatf(float(q[0]), Gf.Vec3f(*[float(c) for c in q[1:]]))
        op.Set(qf)
        M = np.array(xf.ComputeLocalToWorldTransform(0), float).T[:3, :3]
        assert np.abs(M - A.quat_matrix((qf.GetReal(), *qf.GetImaginary()))).max() < 1e-12


def test_units_resolve_turns_y_up_into_z_up():
    F = A.units_resolve("Y")
    assert np.allclose(F[:3, :3] @ [0, 1, 0], [0, 0, 1]) and np.allclose(F[:3, :3] @ [0, 0, 1], [0, -1, 0])
    assert np.array_equal(A.units_resolve("Z"), np.eye(4))
    assert np.allclose(A.units_resolve("Z", 0.01)[:3, :3], 0.01 * np.eye(3))


@pytest.mark.parametrize("left", [False, True])
def test_usd_points_bound_and_winding_match_isaacs_composition(tmp_path, left):
    asset = _usd_asset(tmp_path / "u", left=left)
    meta = A.convert_source("t/u", asset, root=tmp_path / "cache")
    o = A.load_object("t/u", tmp_path / "cache")
    assert meta["reader"]["up_axis"] == "Y"
    rng = np.random.default_rng(7)
    for _ in range(4):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        pos, s = rng.normal(size=3) * 0.3, float(rng.uniform(0.5, 3.0))
        lo, hi, mesh_world = _isaac_like_bbox(asset, pos, q, s)
        T = A.placement_matrix(pos, q, s)
        mlo, mhi = A.aligned_bound(o, T)
        assert np.abs(mlo - lo).max() < 1e-6 and np.abs(mhi - hi).max() < 1e-6  # Isaac's box rule
        # the converted vertices are the referenced mesh's points in the wrapper's frame
        src = Usd.Stage.Open(str(asset))
        P = np.array(UsdGeom.Mesh(src.GetPrimAtPath("/World/part/mesh")).GetPointsAttr().Get(),
                     float)[:5]
        want = P @ mesh_world[:3, :3].T + mesh_world[:3, 3]
        got = o.vertices.astype(float) @ T[:3, :3].T + T[:3, 3]
        d = np.linalg.norm(want[:, None, :] - got[None, :, :], axis=2).min(axis=1)
        assert d.max() < 1e-5


def test_left_handed_meshes_are_rewound(tmp_path):
    """rightHanded keeps the source corner order (counter-clockwise = front, as MuJoCo
    and vsbench.meshgen.quad); leftHanded is rewound."""
    o_r = A.read_usd(_usd_asset(tmp_path / "r", left=False))[0]
    o_l = A.read_usd(_usd_asset(tmp_path / "l", left=True))[0]
    assert np.array_equal(o_r[0].tris[:, [0, 2, 1]], o_l[0].tris)


def test_repeat_uvs_wrap_per_face_into_the_unit_square(tmp_path):
    meta = A.convert_source("t/w", _usd_asset(tmp_path / "u", repeat=True), root=tmp_path / "c")
    o = A.load_object("t/w", tmp_path / "c")
    assert o.uvs.min() >= 0.0 and o.uvs.max() <= 1.0
    assert meta["merge"]["uv_faces_shifted"] == 1
    assert any("wrapped" in s for s in meta["lost"])


def test_two_materials_make_one_atlas(tmp_path):
    meta = A.convert_source("t/a", _usd_asset(tmp_path / "u", repeat=False, two_materials=True),
                            root=tmp_path / "c")
    o = A.load_object("t/a", tmp_path / "c")
    assert meta["merge"]["n_looks"] == 2 and meta["merge"]["atlas_grid"] == 2
    tile = meta["merge"]["atlas_tile_px"]
    assert o.texture.shape == (2 * tile, 2 * tile, 3)
    # the quad samples tile (0, 0), the triangle tile (0, 1): u < 0.5 vs u > 0.5, v < 0.5
    uq = o.uvs[np.unique(o.faces[:2])]
    ut = o.uvs[np.unique(o.faces[2:])]
    assert uq[:, 0].max() < 0.5 and ut[:, 0].min() > 0.5 and o.uvs[:, 1].max() < 0.5


# ------------------------------------------------------------------- placement
def test_a_material_with_only_a_normal_texture_takes_its_diffuse_colour(tmp_path):
    """A surface whose diffuse is a constant and whose only texture is a normal map
    (3 GSO assets) - the UsdUVTexture has no ``diffuseColor`` input; reading it must not
    throw, the look is the constant colour and the normal texture is not mistaken for it."""
    st = Usd.Stage.CreateNew(str(tmp_path / "n.usda"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(st, "/World/mesh")
    m.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    m.CreateFaceVertexCountsAttr([3])
    m.CreateFaceVertexIndicesAttr([0, 1, 2])
    (tmp_path / "textures").mkdir()
    _png(tmp_path / "textures" / "n.png")
    mat = UsdShade.Material.Define(st, "/World/Looks/M")
    sh = UsdShade.Shader.Define(st, "/World/Looks/M/surf")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.18, 0.2, 0.3))
    tex = UsdShade.Shader.Define(st, "/World/Looks/M/normalTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./textures/n.png")
    sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    st.Save()
    parts, _ = A.read_usd(tmp_path / "n.usda")
    assert len(parts) == 1 and parts[0].look[0] == "color"
    assert np.allclose(parts[0].look[1:], (0.18, 0.2, 0.3), atol=1e-6)


def test_placement_scale_seat_and_layout(tmp_path):
    A.convert_source("t/u", _usd_asset(tmp_path / "u"), root=tmp_path / "c")
    o = A.load_object("t/u", tmp_path / "c")
    rules = _fake_rules(0.25)
    p = A.placed_object(o, {"instance_id": "drillish", "category": "drill", "dx": 0.1, "dy": -0.2},
                        {}, None, rules=rules)
    lo, hi = p.bound("usd")
    assert abs(max(hi - lo) - 0.25) < 1e-6  # longest extent = the category target
    assert abs(lo[2]) < 1e-9  # seated on z = 0
    assert np.allclose((lo + hi)[:2] / 2, [0.1, -0.2], atol=1e-9)
    # a metric scan id (with "/") keeps native scale; a layout entry wins over the seat
    le = {"instance_id": "ycb/x", "pos": [0.3, 0.2, 0.1], "quat_wxyz": [0.0, 0.0, 0.0, 1.0]}
    p2 = A.placed_object(o, {"instance_id": "ycb/x", "category": "ycb"}, None, le, rules=rules)
    assert p2.scale == 1.0 and np.allclose(p2.pos, le["pos"]) and p2.record["source"] == "layout"
    mesh = p2.mesh
    assert np.allclose(mesh.pose[:3, 3], le["pos"]) and mesh.uvs.shape[0] == mesh.vertices.shape[0]


# ------------------------------------------------------------------ the object cache
CACHE_IDS = ["handal_drill_25", "ycb/003_cracker_box", "ycb/004_sugar_box", "ycb/005_tomato_soup_can",
          "ycb/008_pudding_box", "ycb/010_potted_meat_can", "ycb/011_banana", "ycb/019_pitcher_base",
          "ycb/024_bowl", "Black and Decker Cordless Drill", "023c84c399814304b3f876103faccb98",
          "teapot-blue_floral"]
_have_cache = all((A.object_dir(i) / "meta.json").exists() for i in CACHE_IDS)
needs_cache = pytest.mark.skipif(not _have_cache, reason="the rooms' objects are not converted (tools/convert_objects.py convert --rooms --coverage)")


@needs_cache
@pytest.mark.parametrize("iid", CACHE_IDS)
def test_converted_objects_are_sound(iid):
    o = A.load_object(iid)
    assert np.isfinite(o.vertices).all() and o.faces.min() >= 0 and o.faces.max() < len(o.vertices)
    assert o.uvs.shape == (len(o.vertices), 2) and o.uvs.min() >= 0.0 and o.uvs.max() <= 1.0
    assert o.texture.ndim == 3 and o.texture.shape[2] == 3 and max(o.texture.shape[:2]) <= A.TEXTURE_MAX
    area = np.linalg.norm(np.cross(o.vertices[o.faces[:, 1]] - o.vertices[o.faces[:, 0]],
                                   o.vertices[o.faces[:, 2]] - o.vertices[o.faces[:, 0]]), axis=1)
    assert np.mean(area > 0) > 0.99
    assert o.meta["converter"] == A.CONVERTER_VERSION and o.meta["source"]["md5"]
    if iid.startswith("ycb/"):  # the OBJ is the USD Isaac placed, point for point
        assert o.meta["reader"]["obj_vs_usd_points_max_abs"] <= 1e-6
        assert o.meta["reader"]["up_axis"] == "Y"


@needs_cache
def test_the_index_is_under_the_plan_budget():
    idx = A.read_index()
    assert all(idx["objects"][i]["status"] == "ok" for i in CACHE_IDS)
    assert sum(idx["objects"][i]["bytes"] for i in CACHE_IDS) < 200e6  # < 200 MB
    assert idx["total_bytes"] < A.SIZE_CAP_BYTES  # the whole cache


def _vs():
    sys.path.insert(0, str(REPO))
    from tools.convert_objects import vs_modules

    return vs_modules()


@needs_cache
@pytest.mark.parametrize("room", ["industrial1_drill", "cluttered2", "cluttered3"])
def test_bbox_goal_gate(room):
    """The converted meshes + the frozen layout + vs-sim's scale rule reproduce the room's
    frozen goal within check_goal_match's tolerances."""
    from tools.convert_objects import gate_room

    r = gate_room(_vs(), room)
    m = r["converted_usd_rule"]
    assert m["pass"], m["refusal"]
    assert m["dpos_mm"] < 1e-3 and m["ddist_mm"] < 1e-3 and m["drot_deg"] < 1e-6
    assert r["diag_pxr_on_source_usd"]["pass"]
