#!/usr/bin/env python3
"""U8 (docs/PORT_PLAN.md section 7): the perturbation modes on the MuJoCo world - the cost of
each mode's world edit, the depth gate under each mode, the preview sheet, and the smoke
flights. The contract itself is ``tests/world/test_perturb_world.py``.

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/u8_perturb.py measure     # per scene in its own process:
                                                             #   recompile s + RSS per mode, the
                                                             #   depth gate, results/bridge/u8_modes.png
    PYTHONPATH=. $VSBENCH_PY tools/u8_perturb.py smoke       # sift_ransac, 100 steps, one episode
                                                             #   per mode -> results/bridge/u8/

Scenes and modes (``vsbench.vs.bench.perturb.build`` names, seed 20260918 = vs-sim's
``fly.py --perturb-seed`` default, the draw per ``(scene, start)`` from ``episode_seed``):

* ``cluttered2`` (the ``isaac_fit_v1`` look): ``none``, ``light``, ``occluder``,
  ``distractors``, ``ground:concrete``, ``ground:laminate``, ``noise``, ``combined``,
  ``swap:ycb/006_mustard_bottle`` (the old path: ``/World/Obj_0``, the cracker box, hidden,
  the mustard bottle put at its model origin) and the chain
  ``swap:ycb/006_mustard_bottle+ground:concrete``; start: ``check_baselines``' near start
  (the benchmark's near sampler, first draw ``n_000`` - U4/U4b's smoke start; the rooms have
  no protocol near set);
* ``inst_drill`` (the instance suite; ``raw`` look - no Isaac goal picture on disk, so no
  fitted curve, U4b): ``none``, ``swap:@same``, ``swap:@same_small``, ``swap:@other_small``,
  ``swap:@other_small+ground:concrete``; start: the first of its frozen ``t1near3`` set;
* ``industrial1_drill`` (``measure`` only): ``none`` and ``ground:wood`` - with cluttered2's
  ``ground:concrete`` the two floors seen through each other's fitted curve.

``measure`` lands in ``results/bridge/u8_perturb.json``; the smoke's records under
``results/bridge/u8/`` (the copied ``open_run`` run directory plus a flat
``<scene>__<mode>.jsonl``, and ``smoke_summary.json`` - inside ``--out``, so a second smoke
process does not stamp its records ``+dirty`` because of the first one's summary). A smoke
records; it does not threshold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "bridge"
RECORD = OUT / "u8_perturb.json"
SHEET = OUT / "u8_modes.png"
SEED = 20260918            # vs-sim scripts/fly.py --perturb-seed default
SWAP_ID = "ycb/006_mustard_bottle"
PLAN = {
    "cluttered2": {"look": None, "start": "near", "modes": [
        "none", "light", "occluder", "distractors", "ground:concrete", "ground:laminate", "noise",
        "combined", f"swap:{SWAP_ID}", f"swap:{SWAP_ID}+ground:concrete"]},
    "inst_drill": {"look": "raw", "start": "t1near3", "modes": [
        "none", "swap:@same", "swap:@same_small", "swap:@other_small",
        "swap:@other_small+ground:concrete"]},
    # measure only: the two floors under each other's fitted curve (the look-fixed rule's
    # cost, see docs/PORT_NOTES.md U8) - cluttered2's wood under the drill's curve here,
    # the drill's concrete under cluttered2's curve above
    "industrial1_drill": {"look": None, "start": "near", "modes": ["none", "ground:wood"]},
}
MEASURE_SCENES = ("cluttered2", "inst_drill", "industrial1_drill")
TILE = (400, 225)


def tag_of(mode: str) -> str:
    return mode.replace(":", "-").replace("/", "_").replace("@", "at_").replace("+", "__")


def _merge_record(key: str, value, record: Path = RECORD) -> None:
    rec = json.loads(record.read_text()) if record.exists() else {}
    rec[key] = value
    rec["unit"] = "U8 - the perturbation world verbs in MuJoCo (docs/PORT_PLAN.md section 7)"
    rec["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"[u8] wrote {record} [{key}]", flush=True)


def start_for(scene, how: str) -> tuple[dict, dict | None]:
    """``(start, starts_payload)``: the near sampler's first draw (no file) or a frozen set."""
    from vsbench.vs import benchmark
    from vsbench.vs import starts as startsmod

    if how == "near":
        g = scene.goal_pose
        s = startsmod.sample_near(g.position, g.rotation, scene.working_distance, n=1)[0]
        return dict(s, starts_seed=startsmod.NEAR["seed"]), None
    picked = benchmark.select_starts(scene.name, how, limit=1)
    return picked[0], benchmark.load_starts(scene.name, benchmark.START_SETS[how]["file"])


def edited_geometry(scene) -> set[str]:
    """Prims whose geometry differs from the unedited world's (a re-textured ground or
    poster of the same size keeps its geometry)."""
    from vsbench.world.perturb_world import same_prim

    out = set()
    for k, p in scene._prims.items():
        b = scene._base_prims.get(k)
        if b is None or not p.visible:
            out.add(k)
        elif not same_prim(p, b):
            keep = ("width_m", "height_m", "uv_tiles")
            if not (p.kind == b.kind and p.kind in ("ground", "poster")
                    and all(p.info.get(x) == b.info.get(x) for x in keep)):
                out.add(k)
    for k, b in scene._base_prims.items():
        if k not in scene._prims or not scene._prims[k].visible:
            out.add(k)
    return out


def _md5(a) -> str:
    import hashlib

    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _tile(rgb, label: str) -> np.ndarray:
    import cv2

    t = cv2.resize(rgb, TILE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (TILE[0], 22), (0, 0, 0), -1)
    cv2.putText(t, label[:58], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return t


# ------------------------------------------------------------------------ measure
def cmd_one(args) -> int:
    """One scene in this process: every mode's begin / end on the world, measured."""
    from vsbench.world import fly

    fly.station_guard(2)
    from vsbench.reference import plane_zdepth
    from vsbench.vs.bench import perturb
    from vsbench.vs.scene import Pose, pose_to_T
    from vsbench.vs.scenes import load_scenes
    from vsbench.world.mujoco_scene import MujocoScene
    from vsbench.world.perturb_world import _rss_mb, depth_gate, ray_depth

    plan = PLAN[args.scene]
    specs, _ = load_scenes()
    rows, tiles = {}, []
    t_init = time.monotonic()
    with MujocoScene(args.scene, specs=specs, look=plan["look"]) as scene:
        fly.require_software_gl(scene)
        init_s = round(time.monotonic() - t_init, 3)
        goal_rgb, goal_depth = scene.render_goal()
        md5_goal, md5_goal_depth = _md5(goal_rgb), _md5(goal_depth)
        start, _payload = start_for(scene, plan["start"])
        views = {"goal": scene.goal_pose, f"start {start['start_id']}": Pose.from_start(start)}
        size = float((scene.spec.raw.get("ground") or {}).get("size_m", 2.4))
        clean = {v: ray_depth(scene, p) for v, p in views.items()}
        planes = {v: plane_zdepth(scene.render_camera, pose_to_T(p), np.eye(4), size=(size, size))
                  for v, p in views.items()}
        base = {"scene": args.scene, "look": scene.look.name, "look_curve_id": scene.look.curve_id,
                "render_mode": scene.render_mode, "gl_renderer": scene.renderer.gl_renderer,
                "init_s": init_s, "compile_s": scene.compile_s, "goal_frame_md5": md5_goal,
                "goal_depth_md5": md5_goal_depth, "start_id": start["start_id"],
                "rss_after_init": _rss_mb(), "n_objects": len(scene.scene_spec.objects)}
        print(f"[u8] {args.scene}: {scene.render_mode} ({scene.renderer.gl_renderer}), look "
              f"{scene.look.name}, init {init_s} s, goal md5 {md5_goal}", flush=True)
        tiles.append(_tile(goal_rgb, f"{args.scene} goal (clean)"))
        for mode in args.modes or plan["modes"]:
            p = perturb.build(mode, seed=SEED, spec=specs[args.scene])
            seed = p.episode(scene.name, start["start_id"])
            n0 = len(scene.world_log)
            t0 = time.monotonic()
            draw = p.begin(scene, seed, goal=scene.goal_pose, working_distance=scene.working_distance,
                           intrinsics=scene.intrinsics)
            t_begin = time.monotonic() - t0
            ev_begin = scene.sync()
            rgb, _d = scene.draw_fresh(scene.goal_pose)
            live = p.live(rgb, 1)
            edited = edited_geometry(scene)
            gates = {}
            perts = {}
            for v, pose in views.items():
                perts[v] = ray_depth(scene, pose)
                gates[v] = depth_gate(clean[v], perts[v], scene, edited, ground_plane=planes[v])
            changed_rgb = _md5(live) != md5_goal
            pg = perts["goal"]
            g_px = pg["hit"] == "/World/Ground"
            ground_rgb = ([float(v) for v in live[pg["vv"][g_px], pg["uu"][g_px]].mean(0)]
                          if g_px.any() else None)
            world = scene.world_record()
            t1 = time.monotonic()
            p.end(scene)
            t_end = time.monotonic() - t1
            ev_end = scene.sync()
            back, back_d = scene.draw_fresh(scene.goal_pose)
            cached, _cd = scene.render_goal()
            row = {"mode": mode, "draw": draw, "begin_s": round(t_begin, 3), "end_s": round(t_end, 3),
                   "compile_begin": ev_begin, "compile_end": ev_end,
                   "n_compiles": len(scene.world_log) - n0, "edited_geometry": sorted(edited),
                   "world_while_perturbed": world, "depth": gates,
                   "picture_changed_at_goal": changed_rgb,
                   "picture_mean_rgb": [float(v) for v in live.reshape(-1, 3).mean(0)],
                   "ground_mean_rgb_at_goal": ground_rgb,
                   "restored_goal_md5": _md5(back) == md5_goal,
                   "restored_goal_depth_md5": _md5(back_d) == md5_goal_depth,
                   "cached_goal_md5_unchanged": _md5(cached) == md5_goal,
                   "rss_after_end": _rss_mb(), "truth_model": p.truth_model()}
            ref = p.reference_pose()
            if ref is not None:
                row["reference_position"] = [float(v) for v in ref.position]
            rows[mode] = row
            tiles.append(_tile(live, f"{args.scene} {mode}"))
            g0 = gates["goal"]
            print(f"[u8] {args.scene} {mode}: begin {row['begin_s']} s (compile "
                  f"{(ev_begin or {}).get('compile_s')} s, rss {(ev_begin or {}).get('rss_mb_after')} MB), "
                  f"end {row['end_s']} s (compile {(ev_end or {}).get('compile_s')} s); depth unchanged "
                  f"{g0['n_unchanged']} px max |d| {g0['unchanged_max_abs_m']:.3g} m, all-vs-ray "
                  f"{g0['all_vs_ray']}; restored {row['restored_goal_md5']}/{row['restored_goal_depth_md5']}",
                  flush=True)
        base["peak_rss_mb"] = _rss_mb()["peak_rss_mb"]
    out = Path(args.tmp)
    (out / f"{args.scene}.json").write_text(json.dumps({"scene": base, "modes": rows}, default=_jd))
    np.savez_compressed(out / f"{args.scene}_tiles.npz", tiles=np.stack(tiles))
    return 0


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def cmd_measure(args) -> int:
    import cv2

    out: dict = {}
    sheets = []
    with tempfile.TemporaryDirectory(prefix="u8_") as tmp:
        for scene in args.scenes:
            cmd = [sys.executable, str(Path(__file__).resolve()), "_one", scene, "--tmp", tmp]
            print(f"[u8] {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, cwd=REPO)
            out[scene] = json.loads((Path(tmp) / f"{scene}.json").read_text())
            with np.load(Path(tmp) / f"{scene}_tiles.npz") as z:
                sheets.append(z["tiles"])
    tiles = [t for s in sheets for t in s]
    cols = 4
    rows = -(-len(tiles) // cols)
    W, H = TILE
    sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = t
    cv2.imwrite(str(SHEET), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    from vsbench.world.perturb_world import LIGHT_RULE, LOOK_RULE

    _merge_record("measure", {
        "how": "per scene in a fresh process (tools/u8_perturb.py _one): MujocoScene, the goal "
               "capture, then per mode the copied perturb.build(mode, seed=20260918).begin at the "
               "smoke start's episode seed, the edited world compiled (scene.sync), drawn fresh at "
               "the goal pose, the depth gate at the goal and the start pose (ray caster on the "
               "compiled model, stride 4), end, the unedited world compiled again and redrawn at "
               "the goal; RSS from /proc/self/status",
        "look_rule": LOOK_RULE, "light_rule": LIGHT_RULE,
        "sheet": str(SHEET.relative_to(REPO)), "scenes": out})
    print(f"[u8] wrote {SHEET}", flush=True)
    return 0


# -------------------------------------------------------------------------- smoke
def cmd_smoke(args) -> int:
    from vsbench.world import fly

    env = fly.station_guard(2)
    from vsbench.vs.bench import perturb
    from vsbench.vs.bench.runner import open_run, run_episode
    from vsbench.vs.scene import Pose, rotation_error_deg, translation_error_mm
    from vsbench.vs.scenes import load_scenes
    from vsbench.world.mujoco_scene import MujocoScene
    from vsbench.world.perturb_world import LIGHT_RULE, LOOK_RULE

    plan = PLAN[args.scene]
    protocol = fly.load_protocol()
    specs, _ = load_scenes()
    out_root = Path(args.out).resolve()
    provenance = {"vsbench_commit": fly.vsbench_commit(ignore=out_root),
                  "vssim_commit": fly.vssim_commit(), "vsbench_commit_rule": fly.VSBENCH_COMMIT_RULE}
    out_root.mkdir(parents=True, exist_ok=True)
    summary = out_root / "smoke_summary.json"    # inside --out: the dirty rule ignores it
    prev = json.loads(summary.read_text()).get("smoke", {}) if summary.exists() else {}
    rows = dict(prev.get("episodes", {}))
    with MujocoScene(args.scene, specs=specs, look=plan["look"]) as scene:
        fly.require_software_gl(scene)
        goal_rgb, _ = scene.render_goal()
        goal_capture = scene.goal_capture_record()
        start, payload = start_for(scene, plan["start"])
        sp = Pose.from_start(start)
        print(f"[u8] {args.scene}: {scene.render_mode} ({scene.renderer.gl_renderer}), look "
              f"{scene.look.name}, start {start['start_id']} "
              f"{translation_error_mm(sp.position, scene.goal_pose.position):.2f} mm / "
              f"{rotation_error_deg(sp.rotation, scene.goal_pose.rotation):.2f} deg", flush=True)
        cfg = fly.arm_config(protocol, args.arm, max_iter=args.max_iter, device="cpu")
        arm = fly.build(protocol, args.arm, cfg, intrinsics=scene.intrinsics, scene_name=args.scene,
                        scene_spec=specs[args.scene])
        for mode in args.modes or plan["modes"]:
            pert = perturb.build(mode, seed=SEED, spec=specs[args.scene])
            tag = f"{args.scene}__{tag_of(mode)}"
            extra = {**goal_capture, **provenance, "bridge_env": env, "scene_meta": scene.meta,
                     "bridge_driver": "tools/u8_perturb.py smoke", "perturb_mode": mode,
                     "perturb_seed": SEED, "perturb_look_rule": LOOK_RULE,
                     "perturb_light_rule": LIGHT_RULE}
            hook = fly._Film(None, scene, goal_rgb, title="", every=1, scale=1.0)
            n0 = len(scene.world_log)
            t0 = time.monotonic()
            with open_run(tag, cfg, out_root=out_root / tag, allow_dirty=True, extra=extra) as sink:
                rec = run_episode(scene, arm, start, cfg, on_step=hook, perturbation=pert,
                                  starts_payload=payload)
                rec["cell_id"] = tag
                rec.update(provenance)
                rec["render_mode"] = scene.render_mode
                rec["gl_renderer"] = scene.renderer.gl_renderer
                rec["first_frame_n_matches"] = hook.first_n_matches
                rec["perturb_mode"] = mode
                rec["perturb_seed"] = SEED
                rec["world_compiles"] = scene.world_log[n0:]
                rec["start_source"] = ("check_baselines near_start: starts.sample_near(goal, n=1)[0], "
                                       "seed 20260902" if plan["start"] == "near"
                                       else f"benchmark.select_starts({args.scene!r}, {plan['start']!r})[0]")
                sink.write(rec, None)
            wall = time.monotonic() - t0
            (out_root / f"{tag}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
            row = {k: rec.get(k) for k in (
                "start_id", "start_translation_mm", "start_rotation_deg", "final_translation_mm",
                "final_rotation_deg", "steps", "max_iter", "terminated_reason", "outcome",
                "stop_step", "stop_translation_mm", "mean_n_matches", "first_frame_n_matches",
                "sec_per_step", "render_mode", "gl_renderer", "look", "look_curve_id",
                "goal_frame_md5", "vsbench_commit", "vssim_commit")}
            for k in ("final_reference_translation_mm", "final_depth_true_translation_mm"):
                if k in rec:
                    row[k] = rec[k]
            row["success"] = rec.get("success")
            row["perturbation"] = rec.get("perturbation")
            row["world_compiles"] = [{k: e[k] for k in ("compile_s", "total_s", "rss_mb_after",
                                                        "peak_rss_mb", "unedited", "n_objects")}
                                     for e in rec["world_compiles"]]
            row["wall_s"] = round(wall, 1)
            row["records"] = str((out_root / f"{tag}.jsonl").relative_to(REPO))
            rows[tag] = row
            print(f"[u8] {tag}: {row['start_translation_mm']:.2f} -> {row['final_translation_mm']:.3f} mm, "
                  f"{row['final_rotation_deg']:.3f} deg, {row['terminated_reason']}, {row['steps']} steps, "
                  f"{row['sec_per_step']:.2f} s/step, frame-1 matches {row['first_frame_n_matches']}, "
                  f"compiles {row['world_compiles']}, wall {row['wall_s']} s", flush=True)
            _merge_record("smoke", {"arm": args.arm, "max_iter": args.max_iter, "seed": SEED,
                                    "note": "recorded, not thresholded; context only, never pooled "
                                            "with Isaac numbers (D-9)", "episodes": rows},
                          summary)
        close = getattr(arm, "close", None)
        if close is not None:
            close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--scenes", nargs="+", default=list(MEASURE_SCENES))
    m.set_defaults(fn=cmd_measure)
    o = sub.add_parser("_one")
    o.add_argument("scene", choices=list(PLAN))
    o.add_argument("--tmp", required=True)
    o.add_argument("--modes", nargs="*", default=None)
    o.set_defaults(fn=cmd_one)
    s = sub.add_parser("smoke")
    s.add_argument("--scene", default="cluttered2", choices=["cluttered2", "inst_drill"])
    s.add_argument("--modes", nargs="*", default=None)
    s.add_argument("--arm", default="sift_ransac")
    s.add_argument("--max-iter", type=int, default=100)
    s.add_argument("--out", default=str(OUT / "u8"))
    s.set_defaults(fn=cmd_smoke)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
