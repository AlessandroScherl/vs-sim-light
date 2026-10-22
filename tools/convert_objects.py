#!/usr/bin/env python
"""Convert vs-sim's objects into the MuJoCo cache, gate them, draw the contact sheet.

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py convert --rooms --coverage
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py gate      # results/bridge/u3_bbox_goal_gate.json
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py sheet     # results/bridge/u3_objects.png
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py convert <instance_id> ...
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py convert --all --record   # U7: u7_convert.json
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py spot      # U7: u7_extent_spot_check.json
    PYTHONPATH=. $VSBENCH_PY tools/convert_objects.py sheet --all   # U7: u7_sheet_<k>.png

docs/PORT_PLAN.md sections 5 and 7 (U3). The converter itself is
``vsbench.world.assets`` and imports no vs-sim code; this driver needs vs-sim's library,
scene, layout, start-file and goal functions and takes them from the copy
``vsbench.vs`` (the module set it used is written into every result).

U7 (PORT_PLAN section 7): ``convert --all`` walks the whole asset library (323 entries;
the Nucleus-only ones are recorded and skipped, D-6) plus the rooms' YCB scans, one
object at a time in this one process, resumable (a cached object is not re-converted),
stopping before the disk would drop under ``--min-free-gb``. A broken asset is recorded
with its reason, never worked around.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    """vs-sim's modules, from the copy ``vsbench.vs`` (the only source since U-copy)."""
    from vsbench.vs import assets, benchmark, camera, isaac_scene, scenes, starts

    return types.SimpleNamespace(assets=assets, benchmark=benchmark, camera=camera,
                                 isaac_scene=isaac_scene, scenes=scenes, starts=starts,
                                 label="vsbench.vs (the copy)")


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
def all_ids(vs) -> list[str]:
    """U7: every asset-library entry (323), then the rooms' YCB scans."""
    ids = list(vs.assets.load_library())
    return ids + [i for i in room_ids(vs) if i not in ids]


def _free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def _units_flag(meta: dict) -> str | None:
    """U7: an object whose layer is not Y-up or not in metres (the unit factor of
    ``units_resolve`` is a rule by assumption only; U3 met only Y-up metre layers)."""
    rd = meta.get("reader", {})
    up, mpu = str(rd.get("up_axis", "?")).upper(), rd.get("meters_per_unit")
    why = []
    if up != "Y":
        why.append(f"upAxis {up}")
    if mpu is None or float(mpu) != 1.0:
        why.append(f"metersPerUnit {mpu}")
    return ", ".join(why) or None


def cmd_convert(args) -> int:
    vs = vs_modules()
    ids = list(args.ids)
    if args.all:
        ids += [i for i in all_ids(vs) if i not in ids]
    if args.rooms:
        ids += [i for i in room_ids(vs) if i not in ids]
    if args.coverage:
        ids += [i for i in COVERAGE.values() if i not in ids]
    if not ids:
        raise SystemExit("nothing to convert (give ids, --all, --rooms or --coverage)")
    lib = full_library(vs, ids)
    rows, t_all = [], time.perf_counter()
    stopped = None
    for n, iid in enumerate(ids, 1):
        entry = lib[iid]
        kind, src, isaac = source_of(vs, iid, entry)
        if kind == "nucleus":
            A.mark_index(iid, "nucleus", args.cache)
            rows.append({"instance_id": iid, "status": "nucleus",
                         "usd_path": vs.assets.resolve_usd_path(entry)[0],
                         "source_kind": entry.get("source"), "category": entry.get("category")})
            print(f"  [{n}/{len(ids)}] {iid}: nucleus (not on disk, PORT_PLAN D-6) - skipped", flush=True)
            continue
        free = _free_gb(A.cache_root(args.cache))
        if free < args.min_free_gb:
            stopped = f"free disk {free:.1f} GB < {args.min_free_gb} GB before {iid!r}"
            print(f"STOP: {stopped}", flush=True)
            break
        cached = (A.object_dir(iid, args.cache) / "meta.json").exists() and not args.overwrite
        print(f"  [{n}/{len(ids)}] {iid}: {'cached' if cached else 'converting'} {src}", flush=True)
        t0 = time.perf_counter()
        try:
            meta = A.convert_source(iid, src, root=args.cache, kind=kind, overwrite=args.overwrite,
                                    frame_usd=isaac if kind == "obj" else None,
                                    extra_meta={"library_entry": entry, "isaac_referenced": str(isaac),
                                                "category": entry.get("category"),
                                                "library_source": entry.get("source")})
        except Exception as e:  # recorded, never hidden
            A.mark_index(iid, f"failed:{type(e).__name__}: {e}", args.cache)
            rows.append({"instance_id": iid, "status": f"failed:{type(e).__name__}: {e}",
                         "kind": kind, "source": str(src), "source_kind": entry.get("source"),
                         "category": entry.get("category")})
            print(f"      FAILED {type(e).__name__}: {e}", flush=True)
            continue
        nbytes = A.read_index(args.cache)["objects"][iid]["bytes"]
        rows.append({"instance_id": iid, "status": "ok", "kind": kind, "cached_before": cached,
                     "source_kind": entry.get("source"), "category": entry.get("category"),
                     "source_bytes": meta["source"]["bytes"],
                     "up_axis": meta["reader"].get("up_axis"),
                     "meters_per_unit": meta["reader"].get("meters_per_unit"),
                     "units_flag": _units_flag(meta),
                     "vertices": meta["counts"]["vertices"], "faces": meta["counts"]["faces"],
                     "n_looks": meta["merge"]["n_looks"], "texture": meta["texture"],
                     "bytes": nbytes, "seconds": meta.get("seconds"),
                     "seconds_this_run": round(time.perf_counter() - t0, 2),
                     "lost": meta["lost"]})
        print(f"      {meta['counts']['vertices']} v, {meta['counts']['faces']} f, "
              f"texture {meta['texture']['width']}x{meta['texture']['height']}, "
              f"{nbytes / 1e6:.1f} MB, {meta.get('seconds')} s; lost: {meta['lost'] or '-'}",
              flush=True)
    index = A.read_index(args.cache)
    curated = [r for r in rows if not r["instance_id"].startswith("ycb/")]
    rec = {"unit": ("U7 - convert the full library" if args.all else "U3 - object converter")
                   + " (docs/PORT_PLAN.md sections 5 and 7)",
           "cache": str(A.cache_root(args.cache)), "vs_modules": vs.label,
           "converter": A.CONVERTER_VERSION, "usd_core": A.usd_versions(),
           "n_ok": sum(r["status"] == "ok" for r in rows), "n_requested": len(ids),
           "seconds_total": round(time.perf_counter() - t_all, 1),
           "cache_total_bytes": index["total_bytes"], "cap_bytes": index["cap_bytes"], "objects": rows}
    if args.all:
        tex = [r["texture"] for r in rows if r["status"] == "ok"]
        ok_c = [r for r in curated if r["status"] == "ok"]
        rec.update({
            "library_entries": len(vs.assets.load_library()),
            "curated": {"n_entries": len(curated),
                        "n_on_disk": sum(r["status"] != "nucleus" for r in curated),
                        "n_ok": len(ok_c),
                        "n_failed": sum(r["status"].startswith("failed") for r in curated),
                        "n_nucleus_skipped": sum(r["status"] == "nucleus" for r in curated)},
            "ycb_room_scans": {"n": len(rows) - len(curated),
                               "n_ok": sum(r["status"] == "ok" for r in rows if r not in curated)},
            "gate": {"curated_ok_at_least": 310, "cap_bytes": index["cap_bytes"]},
            "gate_pass": len(ok_c) >= 310 and index["total_bytes"] < index["cap_bytes"],
            "stopped": stopped,
            "nucleus_skipped": [{"instance_id": r["instance_id"], "usd_path": r["usd_path"]}
                                for r in rows if r["status"] == "nucleus"],
            "failures": [{"instance_id": r["instance_id"], "reason": r["status"][len("failed:"):],
                          "source": r.get("source")} for r in rows if r["status"].startswith("failed")],
            "units": {"rule": "flag every layer whose upAxis != Y or metersPerUnit != 1 "
                              "(units_resolve's unit factor is an assumption)",
                      "counts": _count(f"{r['up_axis']} / {r['meters_per_unit']}" for r in rows
                                       if r["status"] == "ok"),
                      "flagged": [{"instance_id": r["instance_id"], "why": r["units_flag"]}
                                  for r in rows if r["status"] == "ok" and r["units_flag"]]},
            "sizes": {"source_bytes_ok": int(sum(r["source_bytes"] for r in rows if r["status"] == "ok")),
                      "cache_bytes_ok": int(sum(r["bytes"] for r in rows if r["status"] == "ok")),
                      "largest_cache_mb": sorted(([round(r["bytes"] / 1e6, 1), r["instance_id"]]
                                                  for r in rows if r["status"] == "ok"), reverse=True)[:10]},
            "memory_notes": {
                "textures_by_longest_side": _count(_tex_bucket(max(t["width"], t["height"])) for t in tex),
                "texture_4096_count": sum(max(t["width"], t["height"]) == 4096 for t in tex),
                "rgb_texture_bytes_in_memory_total": int(sum(3 * t["width"] * t["height"] for t in tex)),
                "most_faces": sorted(([r["faces"], r["instance_id"]] for r in rows if r["status"] == "ok"),
                                     reverse=True)[:10],
                "atlases": [[r["instance_id"], r["n_looks"]] for r in rows
                            if r["status"] == "ok" and r["n_looks"] > 1]},
            "lost_kinds": _count(_lost_kind(s) for r in rows if r["status"] == "ok" for s in r["lost"]),
            "conversion_seconds_sum": round(sum(float(r["seconds"] or 0) for r in rows
                                                if r["status"] == "ok"), 1),
        })
    if args.record:
        OUT.mkdir(parents=True, exist_ok=True)
        name = "u7_convert.json" if args.all else "u3_convert.json"
        prev = OUT / name
        if args.all and prev.exists():  # keep the spot check / sheets written by later steps
            old = json.loads(prev.read_text())
            for k in ("spot_check", "contact_sheets", "passes"):
                if k in old:
                    rec[k] = old[k]
        (OUT / name).write_text(json.dumps(rec, indent=1))
    print(f"{rec['n_ok']}/{len(ids)} ok, cache {index['total_bytes'] / 1e6:.1f} MB, "
          f"{rec['seconds_total']} s", flush=True)
    if args.all:
        return 0 if rec["gate_pass"] and not stopped else 1
    return 0 if rec["n_ok"] == len([r for r in rows if r["status"] != "nucleus"]) else 1


def _count(it) -> dict:
    out: dict = {}
    for k in it:
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _tex_bucket(n: int) -> str:
    if n <= 16:
        return "<= 16 px (a flat colour)"
    for hi in (512, 1024, 2048, 4095):
        if n <= hi:
            return f"<= {hi} px"
    return "4096 px"


def _lost_kind(s: str) -> str:
    """A ``meta["lost"]`` line without its per-object numbers and names."""
    import re

    s = re.sub(r"\(.*?\)", "(...)", s)
    s = re.sub(r"\S+\.(png|jpg|jpeg|PNG|JPG)", "<file>", s)
    s = re.sub(r"uv primvar of \S+", "uv primvar of <mesh>", s)
    return re.sub(r"\d+", "N", s)


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


def _render_tile(p, W: int, H: int, text_lines, renderer_cls, scene_mod, cam_mod, look_at):
    """One object, upright, from 60 deg azimuth / 30 deg elevation, on a grey floor."""
    from PIL import Image, ImageDraw

    lo, hi = p.bound("usd")
    c, ext = (lo + hi) / 2.0, hi - lo
    rad = 0.5 * float(np.linalg.norm(ext))
    d = 1.15 * rad / np.tan(np.radians(30.0) / 2.0)
    az, el = np.radians(-60.0), np.radians(30.0)
    eye = c + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    S = scene_mod
    ground = S.Plane(np.eye(4), (40 * rad, 40 * rad), S.Material(color=(0.55, 0.55, 0.55)))
    spec = S.SceneSpec((ground, p.mesh), lights=(S.Light(direction=(0.35, 0.55, -1.0), intensity=0.7,
                                                         cast_shadow=True),),
                       ambient=0.45, background=(0.92, 0.92, 0.95))
    cam = cam_mod.CameraModel.from_focal(W, H, fx=0.5 * H / np.tan(np.radians(30.0) / 2.0))
    cam_i = cam_mod.CameraModel(cam.K, W, H, near=max(0.01, d / 20.0), far=d * 10.0)
    r = renderer_cls(spec, cam_i, gl="egl", samples=4)
    try:
        rgb, _ = r.render(look_at(eye, c))
    finally:
        r.close()
    im = Image.fromarray(rgb)
    lines = text_lines(ext)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, 12 * len(lines) + 3], fill=(255, 255, 255))
    for k, t in enumerate(lines):
        dr.text((3, 1 + 12 * k), t, fill=(0, 0, 0))
    return im


def _placeholder(W: int, H: int, lines, colour):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (W, H), colour)
    dr = ImageDraw.Draw(im)
    for k, t in enumerate(lines):
        dr.text((3, 1 + 12 * k), t, fill=(0, 0, 0))
    return im


def _wrap(t: str, n: int) -> list[str]:
    return [t[i:i + n] for i in range(0, len(t), n)] or [""]


def cmd_sheet(args) -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image

    from vsbench import camera as cam_mod
    from vsbench import scene as scene_mod
    from vsbench.poses import look_at
    from vsbench.render.mujoco_backend import MujocoRenderer

    vs = vs_modules()
    index = A.read_index(args.cache)
    W, H = args.tile if args.tile else ((200, 160) if args.all else (480, 360))
    if args.all:  # U7: every library entry in library order; nucleus / failed as placeholders
        ids = list(vs.assets.load_library())[:args.limit]
    else:
        ids = [i for i in (room_ids(vs) + list(COVERAGE.values()))
               if index["objects"].get(i, {}).get("status") == "ok"]
    lib = full_library(vs, ids)
    rules = A.rules_from(vs.assets, vs.scenes)
    nch = max(8, W // 6 - 1)
    tiles, ratios, t0 = [], [], time.perf_counter()
    for n, item in enumerate(sheet_items(vs, ids, lib), 1):
        iid = item["instance_id"]
        st = index["objects"].get(iid, {}).get("status", "absent")
        if st != "ok":
            colour = (200, 200, 200) if st == "nucleus" else (240, 170, 170)
            why = "nucleus only - skipped (D-6)" if st == "nucleus" else st
            tiles.append(_placeholder(W, H, _wrap(f"{n}. {iid}", nch)[:2] + _wrap(why, nch)[:8], colour))
            print(f"  [{n}/{len(ids)}] {iid}: {st}", flush=True)
            continue
        p = A.placed_object(A.load_object(iid, args.cache), item, lib[iid], None, rules=rules)
        m = p.obj.meta
        src = m.get("library_source") or m["source"]["kind"]
        lo_u, hi_u = p.bound("usd")
        lo_p, hi_p = p.bound("points")
        ratios.append([round(float(max(hi_p - lo_p)) / float(max(hi_u - lo_u)), 4), iid])
        if args.all:
            def lines(ext, iid=iid, m=m, src=src):
                return [_wrap(f"{n}. {iid}", nch)[0],
                        f"{src} {m.get('category')} {1e3 * float(max(ext)):.0f}mm "
                        f"{m['counts']['faces'] // 1000}kf {m['texture']['width']}px"]
        else:
            def lines(ext, iid=iid, m=m, src=src, p=p):
                name = iid if len(iid) < 38 else iid[:35] + "..."
                return [name, f"{src}  longest {1e3 * float(max(ext)):.0f} mm  scale {p.scale:.4g}",
                        f"{m['counts']['vertices']} v  {m['counts']['faces']} f  "
                        f"tex {m['texture']['width']}px"]
        tiles.append(_render_tile(p, W, H, lines, MujocoRenderer, scene_mod, cam_mod, look_at))
        print(f"  [{n}/{len(ids)}] {iid}: rendered", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cols, per = (8, 48) if args.all else (4, len(tiles))
    written = []
    for k0 in range(0, len(tiles), per):
        chunk = tiles[k0:k0 + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * W, rows * H), (255, 255, 255))
        for k, im in enumerate(chunk):
            sheet.paste(im, ((k % cols) * W, (k // cols) * H))
        if args.all:
            path = OUT / f"u7_sheet_{k0 // per + 1}.png"
            sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
        else:
            path = OUT / "u3_objects.png"
            sheet.save(path)
        written.append({"path": str(path.relative_to(REPO)), "first": k0 + 1,
                        "last": k0 + len(chunk), "bytes": path.stat().st_size})
        print(f"wrote {path} (tiles {k0 + 1}-{k0 + len(chunk)})", flush=True)
    if args.all:
        _merge_into_u7("contact_sheets", {
            "tiles": len(tiles), "tile_px": [W, H], "cols": cols, "per_sheet": per,
            "pose": "the library's upright canonical pose (canonical: library), category scale, "
                    "azimuth -60 deg, elevation 30 deg; grey = nucleus only, red = failed",
            "seconds": round(time.perf_counter() - t0, 1), "sheets": written,
            "box_vs_points": {
                "what": "longest side of the tight box of the converted points / longest side of "
                        "Isaac's box rule (authored extents through the transforms), at the sheet "
                        "pose; < 1 means the category scale was set by a box looser than the "
                        "visible mesh (in Isaac too), so the object looks smaller than its target",
                "n_below_0.9": sum(r < 0.9 for r, _ in ratios),
                "below_0.9": sorted(x for x in ratios if x[0] < 0.9),
                "min": min(ratios)[0] if ratios else None}})
    return 0


def _merge_into_u7(key: str, value) -> None:
    p = OUT / "u7_convert.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    rec[key] = value
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))


# ------------------------------------------------------------------- spot check (U7)
def _pxr_object_extent(usd: Path, quat, category: str, rules) -> dict:
    """The source USD measured by usd-core alone, as Isaac's ``place_objects`` did: a
    reference under an ``Orient . Scale`` wrapper (Kit's ``rotateX:unitsResolve = 90``
    on a Y-up layer), the category scale from ``BBoxCache.ComputeWorldBound`` at scale 1,
    then the scaled box; also the tight box of the composed points
    (``Boundable.ComputeExtentFromPlugins`` under each mesh's world transform)."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cache = UsdGeom.BBoxCache(0.0, includedPurposes=list(A.ISAAC_BBOX_PURPOSES), useExtentsHint=False)
    xf = UsdGeom.Xform.Define(stage, "/World/Obj")
    inner = stage.DefinePrim("/World/Obj/asset")
    inner.GetReferences().AddReference(str(usd))
    layer = Usd.Stage.Open(str(usd), load=Usd.Stage.LoadNone)
    up, mpu = str(UsdGeom.GetStageUpAxis(layer)).upper(), float(UsdGeom.GetStageMetersPerUnit(layer))
    if mpu != 1.0:
        raise ValueError(f"{usd}: metersPerUnit {mpu} - Kit's unit correction is not emulated here")
    if up == "Y":
        UsdGeom.Xformable(inner).AddRotateXOp(opSuffix="unitsResolve").Set(90.0)
    o = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
    sc = xf.AddScaleOp()
    q = np.asarray(quat, float)
    o.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))

    def box():
        cache.Clear()
        r = cache.ComputeWorldBound(inner).ComputeAlignedRange()
        return np.array(r.GetMin(), float), np.array(r.GetMax(), float)

    sc.Set(Gf.Vec3f(1, 1, 1))
    lo, hi = box()
    mode, target = rules.target_for(category)
    ext1 = hi - lo
    ref = float(ext1[2]) if mode == "height" else float(max(ext1))
    s = target / ref
    sc.Set(Gf.Vec3f(s, s, s))
    lo, hi = box()
    xc = UsdGeom.XformCache(0)
    tl, th = [], []
    for prim in Usd.PrimRange(inner, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility(0) == UsdGeom.Tokens.invisible or img.ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        e = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode(0),
                                                       xc.GetLocalToWorldTransform(prim))
        if e:
            tl.append(np.array(e[0], float))
            th.append(np.array(e[1], float))
    return {"mode": mode, "target_m": target, "scale": float(np.float32(s)),
            "isaac_rule_extent_m": (hi - lo).tolist(),
            "points_extent_m": (np.max(th, 0) - np.min(tl, 0)).tolist() if tl else None}


def cmd_spot(args) -> int:
    vs = vs_modules()
    index = A.read_index(args.cache)
    ok = sorted(i for i, v in index["objects"].items()
                if v.get("status") == "ok" and not i.startswith("ycb/"))
    rng = np.random.default_rng(args.seed)
    pick = [ok[k] for k in sorted(rng.choice(len(ok), size=min(args.n, len(ok)), replace=False))]
    lib = full_library(vs, pick)
    rules = A.rules_from(vs.assets, vs.scenes)
    tol = 1e-3
    rows = []
    for iid in pick:
        e = lib[iid]
        item = {"instance_id": iid, "category": e["category"], "canonical": "library"}
        quat = rules.item_quat(item, e, yaw_deg=0.0)
        p = A.placed_object(A.load_object(iid, args.cache), item, e, None, rules=rules)
        lo, hi = p.bound("usd")
        ext = hi - lo
        plo, phi = p.bound("points")
        mode, target = rules.target_for(str(e["category"]))
        got = float(ext[2]) if mode == "height" else float(max(ext))
        _, _, usd = source_of(vs, iid, e)
        px = _pxr_object_extent(usd, quat, str(e["category"]), rules)
        pext = np.asarray(px["isaac_rule_extent_m"])
        pgot = float(pext[2]) if mode == "height" else float(max(pext))
        d_axes = float(np.abs(ext - pext).max())
        d_pts = (float(np.abs((phi - plo) - np.asarray(px["points_extent_m"])).max())
                 if px["points_extent_m"] else None)
        r = {"instance_id": iid, "source_kind": e.get("source"), "category": e["category"],
             "mode": mode, "target_m": target, "converted_scale": p.scale, "pxr_scale": px["scale"],
             "converted_extent_m": ext.tolist(), "pxr_extent_m": pext.tolist(),
             "converted_measure_m": got, "pxr_measure_m": pgot,
             "converted_vs_target_mm": 1e3 * (got - target), "pxr_vs_target_mm": 1e3 * (pgot - target),
             "converted_vs_pxr_max_axis_mm": 1e3 * d_axes,
             "diag_points_extent_converted_vs_pxr_max_axis_mm": None if d_pts is None else 1e3 * d_pts,
             "scaled_longest_mm": 1e3 * float(max(ext))}
        r["pass"] = (abs(got - target) <= tol and abs(pgot - target) <= tol and d_axes <= tol)
        rows.append(r)
        print(f"  {iid[:40]:40s} {mode:7s} target {1e3 * target:6.1f} mm  conv {1e3 * got:8.3f}  "
              f"pxr {1e3 * pgot:8.3f}  |conv-pxr| {1e3 * d_axes:.2e} mm  pts {r['diag_points_extent_converted_vs_pxr_max_axis_mm']}"
              f"  {'PASS' if r['pass'] else 'MISS'}", flush=True)
    rec = {"unit": "U7 - extent spot check (docs/PORT_PLAN.md section 7)", "seed": args.seed,
           "n": len(rows), "population": len(ok), "tolerance_mm": 1e3 * tol,
           "pose": "canonical: library (the upright library pose), yaw 0, category scale",
           "checks": ["the converted object's scaled measure (longest, or height for the "
                      "height categories) equals target_for(category) within 1 mm",
                      "the same from the SOURCE USD measured by usd-core alone (Isaac's wrapper "
                      "+ unitsResolve + BBoxCache), within 1 mm",
                      "converted vs pxr scaled box extents, every axis, within 1 mm",
                      "diagnostic (not gated): the tight box of the converted points vs the tight "
                      "box of the composed source points (pxr; it counts loose points too)"],
           "all_pass": all(r["pass"] for r in rows), "objects": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u7_extent_spot_check.json").write_text(json.dumps(rec, indent=1))
    _merge_into_u7("spot_check", {k: rec[k] for k in ("seed", "n", "population", "tolerance_mm", "all_pass")}
                   | {"record": "results/bridge/u7_extent_spot_check.json",
                      "max_converted_vs_pxr_mm": max(r["converted_vs_pxr_max_axis_mm"] for r in rows),
                      "max_abs_vs_target_mm": max(max(abs(r["converted_vs_target_mm"]),
                                                      abs(r["pxr_vs_target_mm"])) for r in rows)})
    print("all pass" if rec["all_pass"] else "NOT all pass")
    return 0 if rec["all_pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None, help="cache root (default $VSBENCH_OBJECT_CACHE or "
                                                  "~/genvs_ws/vsbench_assets/objects)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("ids", nargs="*")
    c.add_argument("--rooms", action="store_true", help="the objects of the three benchmark rooms")
    c.add_argument("--coverage", action="store_true", help="one gso, objaverse, housecat6d object")
    c.add_argument("--all", action="store_true", help="U7: the whole asset library + the rooms' YCB scans")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--min-free-gb", type=float, default=50.0,
                   help="stop before converting when the cache's disk has less free (default 50)")
    c.add_argument("--record", action="store_true",
                   help="write results/bridge/u3_convert.json (u7_convert.json with --all)")
    sub.add_parser("gate")
    s = sub.add_parser("sheet")
    s.add_argument("--tile", type=int, nargs=2, default=None,
                   help="tile px (default 480 360; 200 160 with --all)")
    s.add_argument("--all", action="store_true",
                   help="U7: every library entry, 48 tiles per sheet -> results/bridge/u7_sheet_<k>.png")
    s.add_argument("--limit", type=int, default=None, help="with --all: only the first N entries (a trial)")
    sp = sub.add_parser("spot", help="U7: the 12-object extent spot check against pxr")
    sp.add_argument("--n", type=int, default=12)
    sp.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    return {"convert": cmd_convert, "gate": cmd_gate, "sheet": cmd_sheet, "spot": cmd_spot}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
