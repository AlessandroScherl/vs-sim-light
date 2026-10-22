#!/usr/bin/env python
"""Convert vs-sim's objects into the MuJoCo cache, gate them, draw the contact sheet.

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py convert --rooms --coverage
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py gate      # results/bridge/u3_bbox_goal_gate.json
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py sheet     # results/bridge/u3_objects.png
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py convert <instance_id> ...

docs/PORT_PLAN.md sections 5 and 7 (U3). The converter itself is
``vsbench.world.assets`` and imports no vs-sim code; this driver needs vs-sim's library,
scene, layout, start-file and goal functions and takes them from the copy
(``vsbench.vs``) when it exists, else from the pinned clone ``external/vs-sim`` (the
module set it used is written into every result). ``--all`` (U7) is not in this unit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vsbench.world import assets as A  # noqa: E402

ROOMS = ("industrial1_drill", "cluttered2", "cluttered3")
#: reader coverage beyond the rooms (U3): one object per remaining source kind
COVERAGE = {
    "gso": "Black and Decker Cordless Drill",  # repeat uvs (u up to 2.84)
    "objaverse": "023c84c399814304b3f876103faccb98",  # 12 meshes, 10 materials, root Rx(-90) . S(0.78)
    "housecat6d": "teapot-blue_floral",  # 100 k triangles
}
OUT = REPO / "results" / "bridge"


# ----------------------------------------------------------------- vs-sim modules
def vs_modules() -> types.SimpleNamespace:
    """vs-sim's modules: the copy ``vsbench.vs`` if present, else the pinned clone.

    The clone is ``external/vs-sim`` of this checkout, else ``$VSSIM_SRC``, else the main
    tree's ``~/genvs_ws/vs_sim_light/external/vs-sim`` (a worktree has no clone).
    """
    try:
        from vsbench.vs import assets, benchmark, camera, isaac_scene, scenes, starts

        label = "vsbench.vs (the copy)"
    except ImportError:
        cands = [REPO / "external" / "vs-sim", os.environ.get("VSSIM_SRC"),
                 Path.home() / "genvs_ws" / "vs_sim_light" / "external" / "vs-sim"]
        src = next((Path(c) for c in cands if c and (Path(c) / "vssim").is_dir()), None)
        if src is None:
            raise SystemExit("neither vsbench.vs nor an external/vs-sim clone is available") from None
        if str(src) not in sys.path:
            sys.path.append(str(src))
        from vssim import assets, benchmark, camera, isaac_scene, scenes, starts

        pin = (REPO / "external" / "VSSIM_COMMIT").read_text().strip()
        label = f"vssim from {src} (pin {pin})"
    return types.SimpleNamespace(assets=assets, benchmark=benchmark, camera=camera,
                                 isaac_scene=isaac_scene, scenes=scenes, starts=starts,
                                 label=label)


def workspace_path(p: str) -> Path:
    """``/workspace/...`` (the station volume in vs-sim's yaml/json) -> the host path."""
    root = os.environ.get("VSBENCH_WORKSPACE_ROOT", str(Path.home() / "genvs_ws"))
    return Path(root + p[len("/workspace"):]) if str(p).startswith("/workspace/") else Path(p)


def full_library(vs, ids) -> dict:
    lib = dict(vs.assets.load_library())
    vs.assets.merge_ycb_library(lib, list(ids))
    return lib


def source_of(vs, iid: str, entry: dict) -> tuple[str, Path | None, Path | None]:
    """``(kind, converter source, the file Isaac referenced)``; kind ``nucleus`` -> skip."""
    if iid.startswith("ycb/"):
        root = vs.assets.ycb_root()
        isaac = workspace_path(vs.assets.resolve_usd_path(entry)[0])
        return "obj", A.ycb_obj_path(root, iid), isaac
    path, nucleus = vs.assets.resolve_usd_path(entry)
    if nucleus:
        return "nucleus", None, None
    return "usd", workspace_path(path), workspace_path(path)


def room_ids(vs) -> list[str]:
    specs, _ = vs.scenes.load_scenes()
    out = []
    for r in ROOMS:
        out += [i for i in specs[r].object_ids() if i not in out]
    return out


# ---------------------------------------------------------------------- convert
def cmd_convert(args) -> int:
    vs = vs_modules()
    ids = list(args.ids)
    if args.rooms:
        ids += [i for i in room_ids(vs) if i not in ids]
    if args.coverage:
        ids += [i for i in COVERAGE.values() if i not in ids]
    if not ids:
        raise SystemExit("nothing to convert (give ids, --rooms or --coverage)")
    lib = full_library(vs, ids)
    rows, t_all = [], time.perf_counter()
    for iid in ids:
        entry = lib[iid]
        kind, src, isaac = source_of(vs, iid, entry)
        if kind == "nucleus":
            A.mark_index(iid, "nucleus", args.cache)
            rows.append({"instance_id": iid, "status": "nucleus"})
            print(f"  {iid}: nucleus (not on disk, PORT_PLAN D-6) - skipped", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            meta = A.convert_source(iid, src, root=args.cache, kind=kind, overwrite=args.overwrite,
                                    frame_usd=isaac if kind == "obj" else None,
                                    extra_meta={"library_entry": entry, "isaac_referenced": str(isaac),
                                                "category": entry.get("category"),
                                                "library_source": entry.get("source")})
        except Exception as e:  # recorded, never hidden
            A.mark_index(iid, f"failed:{type(e).__name__}: {e}", args.cache)
            rows.append({"instance_id": iid, "status": f"failed:{e}"})
            print(f"  {iid}: FAILED {e}", flush=True)
            continue
        nbytes = A.read_index(args.cache)["objects"][iid]["bytes"]
        rows.append({"instance_id": iid, "status": "ok", "kind": kind,
                     "source_kind": entry.get("source"), "vertices": meta["counts"]["vertices"],
                     "faces": meta["counts"]["faces"], "texture": meta["texture"],
                     "bytes": nbytes, "seconds": round(time.perf_counter() - t0, 2),
                     "lost": meta["lost"]})
        print(f"  {iid}: {meta['counts']['vertices']} v, {meta['counts']['faces']} f, "
              f"texture {meta['texture']['width']}x{meta['texture']['height']}, "
              f"{nbytes / 1e6:.1f} MB, {rows[-1]['seconds']} s; lost: {meta['lost'] or '-'}",
              flush=True)
    index = A.read_index(args.cache)
    rec = {"unit": "U3 - object converter (docs/PORT_PLAN.md section 5)",
           "cache": str(A.cache_root(args.cache)), "vs_modules": vs.label,
           "converter": A.CONVERTER_VERSION, "usd_core": A.usd_versions(),
           "n_ok": sum(r["status"] == "ok" for r in rows), "n_requested": len(ids),
           "seconds_total": round(time.perf_counter() - t_all, 1),
           "cache_total_bytes": index["total_bytes"], "cap_bytes": index["cap_bytes"], "objects": rows}
    if args.record:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "u3_convert.json").write_text(json.dumps(rec, indent=1))
    print(f"{rec['n_ok']}/{len(ids)} ok, cache {index['total_bytes'] / 1e6:.1f} MB, "
          f"{rec['seconds_total']} s", flush=True)
    return 0 if rec["n_ok"] == len([r for r in rows if r["status"] != "nucleus"]) else 1


# ------------------------------------------------------------------------- gate
def _pxr_isaac_bbox(vs, items, lib, layout, rules) -> tuple[np.ndarray, np.ndarray, list]:
    """Isaac's measurement emulated with usd-core on the SOURCE USDs (no converter):
    ``/World/Obj_i`` = Translate . Orient(float) . Scale(float) over a reference to the
    file Isaac referenced, Kit's ``xformOp:rotateX:unitsResolve = 90`` appended to the
    reference prim of a Y-up layer, the category scale measured the same way, then
    ``BBoxCache.ComputeWorldBound(inner).ComputeAlignedRange()`` per object. A diagnostic:
    it separates a converter error from an Isaac-side difference."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cache = UsdGeom.BBoxCache(0.0, includedPurposes=list(A.ISAAC_BBOX_PURPOSES), useExtentsHint=False)

    def bound(prim):
        cache.Clear()
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return np.array(r.GetMin(), float), np.array(r.GetMax(), float)

    def setq(op, q):
        op.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))

    mins, maxs, scales = [], [], []
    for i, (item, le) in enumerate(zip(items, layout["objects"], strict=True)):
        iid = str(item["instance_id"])
        _, _, usd = source_of(vs, iid, lib[iid])
        xf = UsdGeom.Xform.Define(stage, f"/World/Obj_{i}")
        inner = stage.DefinePrim(f"/World/Obj_{i}/asset")
        inner.GetReferences().AddReference(str(usd))
        layer_stage = Usd.Stage.Open(str(usd))
        if str(UsdGeom.GetStageUpAxis(layer_stage)).upper() == "Y":
            UsdGeom.Xformable(inner).AddRotateXOp(opSuffix="unitsResolve").Set(90.0)
        ops = (xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble),
               xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat), xf.AddScaleOp())
        quat = rules.item_quat(item, lib[iid])
        ops[0].Set(Gf.Vec3d(0, 0, 0))
        setq(ops[1], quat)
        ops[2].Set(Gf.Vec3f(1, 1, 1))
        if "/" in iid:
            s = float(item.get("scale", 1.0))
        else:
            mn, mx = bound(inner)
            mode, target = rules.target_for(str(item["category"]))
            ext = mx - mn
            ref = float(ext[2]) if mode == "height" else float(max(ext))
            s = target / ref
        ops[2].Set(Gf.Vec3f(s, s, s))
        t, q = le["pos"], le["quat_wxyz"]
        ops[0].Set(Gf.Vec3d(*[float(c) for c in t]))
        setq(ops[1], q)
        mn, mx = bound(inner)
        mins.append(mn)
        maxs.append(mx)
        scales.append(float(np.float32(s)))
    return np.min(mins, axis=0), np.max(maxs, axis=0), scales


def _goal_metrics(vs, payload, lo, hi, goal_cfg) -> dict:
    cam = vs.camera.CameraConfig()
    pos, R, d = vs.scenes.scene_view_goal_pose(
        cam, lo, hi, frame_fill=goal_cfg.get("frame_fill"),
        elevation_deg=float(goal_cfg.get("elevation_deg", 90.0)),
        azimuth_deg=float(goal_cfg.get("azimuth_deg", -90.0)))
    g = payload["goal"]
    dp = float(np.linalg.norm(np.asarray(g["cam_pos"], float) - pos))
    dr = float(vs.starts.rotation_geodesic_deg(np.asarray(g["cam_R_opt"], float), R))
    dd = abs(float(g["distance_m"]) - float(d))
    try:
        vs.starts.check_goal_match(payload, pos, R, d)
        ok, why = True, None
    except ValueError as e:
        ok, why = False, str(e)
    return {"pass": ok, "dpos_mm": dp * 1e3, "drot_deg": dr, "ddist_mm": dd * 1e3,
            "cam_pos": pos.tolist(), "distance_m": float(d), "bbox_min": list(map(float, lo)),
            "bbox_max": list(map(float, hi)), "refusal": why}


def gate_room(vs, room: str, cache=None) -> dict:
    specs, _ = vs.scenes.load_scenes()
    spec = specs[room]
    payload = vs.benchmark.load_starts(room, "v2")  # md5-pinned
    layout = vs.scenes.load_layout(spec)
    items = vs.isaac_scene.scatter_items(spec, with_scatter=layout is None)
    vs.scenes.check_layout(layout, [it["instance_id"] for it in items])
    lib = full_library(vs, spec.object_ids())
    rules = A.rules_from(vs.assets, vs.scenes)
    placed = [A.placed_object(A.load_object(str(it["instance_id"]), cache), it,
                              lib[str(it["instance_id"])], le, rules=rules)
              for it, le in zip(items, layout["objects"], strict=True)]
    goal_cfg = spec.raw.get("goal") or {}
    out = {"room": room, "frozen_goal": payload["goal"],
           "tolerance": {"pos_m": 5e-4, "rot_deg": 0.05, "dist_m": 1e-3}}
    lo, hi = A.scene_bbox(placed, "usd")
    out["converted_usd_rule"] = _goal_metrics(vs, payload, lo, hi, goal_cfg)
    lo, hi = A.scene_bbox(placed, "points")
    out["diag_converted_tight_points"] = _goal_metrics(vs, payload, lo, hi, goal_cfg)
    lo, hi, scales = _pxr_isaac_bbox(vs, items, lib, layout, rules)
    out["diag_pxr_on_source_usd"] = _goal_metrics(vs, payload, lo, hi, goal_cfg)
    out["objects"] = [{"instance_id": p.record["instance_id"], "scale": p.scale,
                       "pxr_scale": s, "pos": p.pos.tolist(), "quat_wxyz": p.quat_wxyz.tolist(),
                       "bbox_usd_rule": [b.tolist() for b in p.bound("usd")]}
                      for p, s in zip(placed, scales, strict=True)]
    return out


def cmd_gate(args) -> int:
    vs = vs_modules()
    rows = [gate_room(vs, r, args.cache) for r in ROOMS]
    rec = {"unit": "U3 - bbox-goal gate (docs/PORT_PLAN.md section 5)", "vs_modules": vs.label,
           "rule": "converted meshes + frozen layouts + vs-sim's scale rule -> Isaac's box rule "
                   "-> scene_view_goal_pose -> check_goal_match against the md5-pinned start file",
           "owner_decision_on_a_miss": "D-4 (a): the frozen goal wins, the residual is recorded",
           "rooms": rows, "all_pass": all(r["converted_usd_rule"]["pass"] for r in rows)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u3_bbox_goal_gate.json").write_text(json.dumps(rec, indent=1))
    for r in rows:
        for k in ("converted_usd_rule", "diag_converted_tight_points", "diag_pxr_on_source_usd"):
            m = r[k]
            print(f"{r['room']:18s} {k:28s} {'PASS' if m['pass'] else 'miss'}  "
                  f"|dpos| {m['dpos_mm']:.2e} mm  drot {m['drot_deg']:.2e} deg  "
                  f"|dd| {m['ddist_mm']:.2e} mm")
    print("all pass" if rec["all_pass"] else "NOT all pass (D-4: frozen goal wins, residual recorded)")
    return 0 if rec["all_pass"] else 1


# ------------------------------------------------------------------------ sheet
def sheet_items(vs, ids, lib) -> list[dict]:
    """Each object once, upright: the library's canonical pose for a curated id (the
    instance suite's ``canonical: library`` placement, category scale), native for a YCB
    scan, with a room's ``pre_rot_x_deg`` where the room stands it up (the pitcher)."""
    specs, _ = vs.scenes.load_scenes()
    pre = {}
    for r in ROOMS:
        for it in specs[r].raw.get("objects") or []:
            if it.get("pre_rot_x_deg"):
                pre[str(it["instance_id"])] = float(it["pre_rot_x_deg"])
    out = []
    for iid in ids:
        e = lib[iid]
        if iid.startswith("ycb/"):
            it = {"instance_id": iid, "category": "ycb"}
            if iid in pre:
                it["pre_rot_x_deg"] = pre[iid]
        else:
            it = {"instance_id": iid, "category": e["category"], "canonical": "library"}
        out.append(it)
    return out


def cmd_sheet(args) -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image, ImageDraw

    from vsbench.camera import CameraModel
    from vsbench.poses import look_at
    from vsbench.render.mujoco_backend import MujocoRenderer
    from vsbench.scene import Light, Material, Plane, SceneSpec

    vs = vs_modules()
    index = A.read_index(args.cache)
    ids = [i for i in (room_ids(vs) + list(COVERAGE.values()))
           if index["objects"].get(i, {}).get("status") == "ok"]
    lib = full_library(vs, ids)
    rules = A.rules_from(vs.assets, vs.scenes)
    W, H = args.tile
    cam = CameraModel.from_focal(W, H, fx=0.5 * H / np.tan(np.radians(30.0) / 2.0))
    tiles = []
    for item in sheet_items(vs, ids, lib):
        iid = item["instance_id"]
        p = A.placed_object(A.load_object(iid, args.cache), item, lib[iid], None, rules=rules)
        lo, hi = p.bound("usd")
        c, ext = (lo + hi) / 2.0, hi - lo
        rad = 0.5 * float(np.linalg.norm(ext))
        d = 1.15 * rad / np.tan(np.radians(30.0) / 2.0)
        az, el = np.radians(-60.0), np.radians(30.0)
        eye = c + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        ground = Plane(np.eye(4), (40 * rad, 40 * rad), Material(color=(0.55, 0.55, 0.55)))
        spec = SceneSpec((ground, p.mesh), lights=(Light(direction=(0.35, 0.55, -1.0), intensity=0.7,
                                                         cast_shadow=True),),
                         ambient=0.45, background=(0.92, 0.92, 0.95))
        cam_i = CameraModel(cam.K, W, H, near=max(0.01, d / 20.0), far=d * 10.0)
        r = MujocoRenderer(spec, cam_i, gl="egl", samples=4)
        rgb, _ = r.render(look_at(eye, c))
        r.close()
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        m = p.obj.meta
        name = iid if len(iid) < 38 else iid[:35] + "..."
        lines = [name, f"{m.get('library_source') or m['source']['kind']}  "
                       f"longest {1e3 * float(max(ext)):.0f} mm  scale {p.scale:.4g}",
                 f"{m['counts']['vertices']} v  {m['counts']['faces']} f  "
                 f"tex {m['texture']['width']}px"]
        dr.rectangle([0, 0, W, 14 * len(lines) + 4], fill=(255, 255, 255))
        for k, s in enumerate(lines):
            dr.text((4, 2 + 14 * k), s, fill=(0, 0, 0))
        tiles.append(im)
        print(f"  {iid}: rendered", flush=True)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * H), (255, 255, 255))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * W, (k // cols) * H))
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "u3_objects.png")
    print(f"wrote {OUT / 'u3_objects.png'} ({len(tiles)} objects)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None, help="cache root (default $VSBENCH_OBJECT_CACHE or "
                                                  "~/genvs_ws/vsbench_assets/objects)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("ids", nargs="*")
    c.add_argument("--rooms", action="store_true", help="the objects of the three benchmark rooms")
    c.add_argument("--coverage", action="store_true", help="one gso, objaverse, housecat6d object")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--record", action="store_true", help="write results/bridge/u3_convert.json")
    sub.add_parser("gate")
    s = sub.add_parser("sheet")
    s.add_argument("--tile", type=int, nargs=2, default=(480, 360))
    args = ap.parse_args(argv)
    return {"convert": cmd_convert, "gate": cmd_gate, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
