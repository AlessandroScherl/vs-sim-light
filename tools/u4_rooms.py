#!/usr/bin/env python3
"""U4 (docs/PORT_PLAN.md section 7): the rooms' costs, previews, the Isaac comparison and
the smoke flights - the parts of the unit that are not in ``tests/world/test_scene_rooms.py``.

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/u4_rooms.py measure   # per room, in its own process: build /
                                                         #   compile s, peak + steady RSS, goal md5,
                                                         #   results/bridge/u4_<room>_goal.png
    PYTHONPATH=. $VSBENCH_PY tools/u4_rooms.py isaac     # results/bridge/u4_rooms_vs_isaac.png
    PYTHONPATH=. $VSBENCH_PY tools/u4_rooms.py smoke --arms sift_ransac --max-iter 100
    PYTHONPATH=. $VSBENCH_PY tools/u4_rooms.py smoke --rooms cluttered2 --arms vitvs_dinov3

Everything lands in ``results/bridge/u4_rooms.json`` (one key per subcommand) and the
smoke's records under ``results/bridge/u4/`` (the copied ``open_run`` run directory plus a
flat ``<room>_<arm>.jsonl``, as ``vsbench.world.fly`` writes them).

The smoke start is ``scripts/check_baselines.py``'s near start - the benchmark's near
sampler (3-8 mm, <= 3 deg, seed 20260902) on the room's goal, first draw, ``n_000`` - because
the protocol has no near start set for the rooms (``near60`` covers the posters only), so
``python -m vsbench.world.fly`` (protocol start sets only) cannot fly it. The flight itself
is ``fly``'s: the station guard, the protocol cell's config, ``build_arm``, ``check_scene_goal``,
the copied ``run_episode`` and ``open_run``, the provenance fields. A smoke records; it
does not threshold.

The Isaac pictures are read, never written: ``$VSBENCH_ISAAC_GOALS`` (default
``~/genvs_ws/vssim_out/check1``, Isaac's ``check_baselines`` run of 2026-09-17, whose goals
are the frozen ones).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOMS = ("industrial1_drill", "cluttered2", "cluttered3")
OUT = REPO / "results" / "bridge"
RECORD = OUT / "u4_rooms.json"
ISAAC_GOALS = Path(os.environ.get("VSBENCH_ISAAC_GOALS",
                                  str(Path.home() / "genvs_ws" / "vssim_out" / "check1")))
PREVIEW_SCALE = 0.5


def _merge_record(key: str, value) -> None:
    rec = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    rec[key] = value
    rec["unit"] = "U4 - the three rooms in MujocoScene (docs/PORT_PLAN.md section 7)"
    rec["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"[u4] wrote {RECORD} [{key}]")


def _vm(field: str) -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


# ------------------------------------------------------------------------ measure
def _measure_one(room: str) -> dict:
    """Runs in its own process (``measure`` spawns ``_one``): peak RSS is per process."""
    import hashlib
    import resource

    import numpy as np

    from vsbench.world import fly

    fly.station_guard(2)                         # GPU hidden, software GL forced, 2 threads
    import mujoco  # noqa: F401  (counted in the baseline, as every flight loads it)
    from PIL import Image

    from vsbench.vs.scenes import load_scenes
    from vsbench.world.mujoco_scene import MujocoScene

    specs, _ = load_scenes()
    base_rss = _vm("VmRSS")
    t0 = time.monotonic()
    scene = MujocoScene(room, specs=specs)
    init_s = time.monotonic() - t0
    try:
        fly.require_software_gl(scene)
        t1 = time.monotonic()
        rgb, depth = scene.render_goal()
        goal_s = time.monotonic() - t1
        scene.reset(scene.goal_pose)
        t2 = time.monotonic()
        for _ in range(3):
            scene._draw(scene.goal_pose)
        frame_s = (time.monotonic() - t2) / 3
        md5 = hashlib.md5(rgb.tobytes()).hexdigest()
        steady = _vm("VmRSS")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        png = OUT / f"u4_{room}_goal.png"
        im = Image.fromarray(np.asarray(rgb))
        im = im.resize((int(im.width * PREVIEW_SCALE), int(im.height * PREVIEW_SCALE)), Image.LANCZOS)
        im.save(png, optimize=True)
        tex = [o["texture_wh"] for o in scene.meta["objects"]] + [scene.meta["ground"]["texture_wh"]]
        tex_mb = sum(w * h * 3 for w, h in tex) / 2**20
        return {"room": room, "build_s": scene.meta["build_s"], "compile_s": scene.compile_s,
                "init_s": round(init_s, 3), "goal_capture_s": round(goal_s, 3),
                "frame_s": round(frame_s, 3), "rss_baseline_mb": round(base_rss, 1),
                "rss_steady_mb": round(steady, 1), "rss_peak_mb": round(peak, 1),
                "textures": tex, "texture_rgb_mb": round(tex_mb, 1),
                "n_objects": len(scene.meta["objects"]),
                "n_faces": sum(o["faces"] for o in scene.meta["objects"]),
                "goal_frame_md5": md5, "goal_depth_md5": hashlib.md5(depth.tobytes()).hexdigest(),
                "goal_distance_m": scene.working_distance, "goal_used": scene.goal_check["used"],
                "goal_files": scene.goal_check["files"], "render_mode": scene.render_mode,
                "gl_renderer": scene.renderer.gl_renderer, "preview": str(png.relative_to(REPO)),
                "preview_scale": PREVIEW_SCALE, "lighting": scene.meta["lighting"],
                "ground": scene.meta["ground"]}
    finally:
        scene.close()


def cmd_one(args) -> int:
    print("U4ONE " + json.dumps(_measure_one(args.room)), flush=True)
    return 0


def cmd_measure(args) -> int:
    rows = {}
    for room in args.rooms:
        p = subprocess.run([sys.executable, __file__, "_one", room], capture_output=True, text=True,
                           timeout=900, cwd=REPO)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("U4ONE ")]
        if p.returncode or not line:
            print(p.stdout[-3000:], p.stderr[-3000:])
            raise SystemExit(f"{room}: measurement failed (exit {p.returncode})")
        rows[room] = json.loads(line[0][6:])
        r = rows[room]
        print(f"{room:18s} build {r['build_s']:.2f} s  compile {r['compile_s']:.2f} s  goal "
              f"{r['goal_capture_s']:.2f} s  frame {r['frame_s']:.2f} s  RSS peak {r['rss_peak_mb']:.0f} "
              f"MB (steady {r['rss_steady_mb']:.0f}, baseline {r['rss_baseline_mb']:.0f})  textures "
              f"{r['texture_rgb_mb']:.0f} MB RGB  [{r['gl_renderer']}]")
    _merge_record("measure", {"rooms": rows, "how": "each room in a fresh process (tools/u4_rooms.py "
                              "_one): fly.station_guard, MujocoScene, the goal capture, 3 redraws; "
                              "RSS from /proc/self/status (steady) and getrusage (peak)",
                              "texture_downscale": None,
                              "texture_note": "no texture is downscaled for render: every object "
                                              "texture as cached (<= 4096^2), every ground as stored"})
    return 0


# -------------------------------------------------------------------------- isaac
def cmd_isaac(args) -> int:
    import numpy as np
    from PIL import Image, ImageDraw

    from vsbench.vs import benchmark
    from vsbench.vs.camera import CameraConfig
    from vsbench.vs.labels import object_mask_from_depth
    from vsbench.vs.scene import Pose

    intr = CameraConfig().intrinsics()
    tile_w, tile_h = 800, 450
    sheet = Image.new("RGB", (2 * tile_w, len(args.rooms) * (tile_h + 28)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    rows = {}
    for k, room in enumerate(args.rooms):
        ip = ISAAC_GOALS / f"{room}_goal.png"
        mp = OUT / f"u4_{room}_goal.png"
        if not ip.exists() or not mp.exists():
            raise SystemExit(f"need {ip} (Isaac, read only) and {mp} (tools/u4_rooms.py measure)")
        iso = np.asarray(Image.open(ip).convert("RGB"))
        muj_small = Image.open(mp).convert("RGB")
        muj = np.asarray(muj_small.resize((iso.shape[1], iso.shape[0]), Image.BICUBIC))
        g = benchmark.load_starts(room, "v2")["goal"]
        pose = Pose(position=np.asarray(g["cam_pos"], float), rotation=np.asarray(g["cam_R_opt"], float))
        depth = np.load(ISAAC_GOALS / f"{room}_goal_depth.npy").astype(np.float64)
        obj = object_mask_from_depth(depth, intr, pose)

        def tone(a, sel):
            x = a.astype(np.float64)[sel]
            return {"mean_rgb": [round(float(v), 1) for v in x.mean(0)],
                    "mean_grey": round(float(x.mean()), 1)}

        # shadows: ground pixels clearly darker than the ground's own median (luminance)
        def shadow_share(a):
            lum = a.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
            gl = lum[~obj]
            return round(float((gl < 0.6 * np.median(gl)).mean()), 4)

        rows[room] = {"isaac_png": str(ip), "mujoco_png": str(mp.relative_to(REPO)),
                      "ground": {"mujoco": tone(muj, ~obj), "isaac": tone(iso, ~obj)},
                      "objects": {"mujoco": tone(muj, obj), "isaac": tone(iso, obj)},
                      "ground_brightness_ratio_isaac_over_mujoco":
                          round(float(iso[~obj].mean() / max(muj[~obj].mean(), 1e-9)), 3),
                      "ground_shadow_share": {"mujoco": shadow_share(muj), "isaac": shadow_share(iso)},
                      "mask_from": "Isaac's goal depth (object_mask_from_depth, 5 mm above z = 0); "
                                   "the MuJoCo goal depth gives the same mask (IoU >= 0.99, "
                                   "tests/world/test_scene_rooms.py)"}
        y = k * (tile_h + 28)
        for j, (img, label) in enumerate(((muj_small, f"{room} - MuJoCo (llvmpipe)"),
                                          (Image.open(ip).convert("RGB"), f"{room} - Isaac RTX (check1, 2026-09-17)"))):
            sheet.paste(img.resize((tile_w, tile_h), Image.LANCZOS), (j * tile_w, y + 28))
            draw.text((j * tile_w + 8, y + 8), label, fill=(0, 0, 0))
    out = OUT / "u4_rooms_vs_isaac.png"
    sheet.save(out, optimize=True)
    print(json.dumps(rows, indent=1))
    _merge_record("isaac_comparison", {"sheet": str(out.relative_to(REPO)), "rooms": rows,
                                       "isaac_source": str(ISAAC_GOALS)})
    return 0


# -------------------------------------------------------------------------- smoke
def near_start(scene) -> dict:
    """``scripts/check_baselines.py near_start``, verbatim: the near sampler, first draw."""
    from vsbench.vs import starts as startsmod

    g = scene.goal_pose
    s = startsmod.sample_near(g.position, g.rotation, scene.working_distance, n=1)[0]
    return dict(s, starts_seed=startsmod.NEAR["seed"])


def cmd_smoke(args) -> int:
    from vsbench.world import fly

    env = fly.station_guard(2)
    from vsbench.vs import benchmark
    from vsbench.vs.bench.runner import open_run, run_episode
    from vsbench.vs.scene import Pose, rotation_error_deg, translation_error_mm
    from vsbench.vs.scenes import load_scenes
    from vsbench.world.mujoco_scene import MujocoScene

    protocol = fly.load_protocol()
    specs, _ = load_scenes()
    out_root = (OUT / "u4").resolve()
    provenance = {"vsbench_commit": fly.vsbench_commit(ignore=out_root),
                  "vssim_commit": fly.vssim_commit(), "vsbench_commit_rule": fly.VSBENCH_COMMIT_RULE}
    out_root.mkdir(parents=True, exist_ok=True)
    rows = json.loads(RECORD.read_text()).get("smoke", {}) if RECORD.exists() else {}
    for room in args.rooms:
        with MujocoScene(room, specs=specs) as scene:
            fly.require_software_gl(scene)
            payload = benchmark.load_starts(room, "v2")
            benchmark.check_scene_goal(scene, payload)
            goal_rgb, _ = scene.render_goal()
            goal_capture = scene.goal_capture_record()
            start = near_start(scene)
            sp = Pose.from_start(start)
            print(f"[u4] {room}: {scene.render_mode} ({scene.renderer.gl_renderer}), goal "
                  f"{scene.working_distance:.6f} m, start {start['start_id']} "
                  f"{translation_error_mm(sp.position, scene.goal_pose.position):.2f} mm / "
                  f"{rotation_error_deg(sp.rotation, scene.goal_pose.rotation):.2f} deg", flush=True)
            for arm_id in args.arms:
                cfg = fly.arm_config(protocol, arm_id, max_iter=args.max_iter, device="cpu")
                t_b = time.monotonic()
                arm = fly.build(protocol, arm_id, cfg, intrinsics=scene.intrinsics, scene_name=room,
                                scene_spec=specs[room])
                build_s = round(time.monotonic() - t_b, 3)
                tag = f"{room}_{arm_id}"
                extra = {**goal_capture, **provenance, "bridge_env": env, "scene_meta": scene.meta,
                         "arm_build_s": build_s, "bridge_driver": "tools/u4_rooms.py smoke",
                         "start_source": "check_baselines near_start: starts.sample_near(goal, n=1)[0], "
                                         "seed 20260902 (no protocol near set for the rooms)"}
                hook = fly._Film(None, scene, goal_rgb, title="", every=1, scale=1.0)
                t0 = time.monotonic()
                with open_run(tag, cfg, out_root=out_root / tag, allow_dirty=True, extra=extra) as sink:
                    rec = run_episode(scene, arm, start, cfg, on_step=hook)
                    rec["cell_id"] = tag
                    rec.update(provenance)
                    rec["render_mode"] = scene.render_mode
                    rec["gl_renderer"] = scene.renderer.gl_renderer
                    rec["first_frame_n_matches"] = hook.first_n_matches
                    rec["arm_build_s"] = build_s
                    rec["start_source"] = extra["start_source"]
                    sink.write(rec, None)
                wall = time.monotonic() - t0
                (out_root / f"{tag}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
                row = {k: rec.get(k) for k in (
                    "start_id", "start_translation_mm", "start_rotation_deg", "final_translation_mm",
                    "final_rotation_deg", "steps", "max_iter", "terminated_reason", "outcome",
                    "stop_step", "stop_translation_mm", "stop_rotation_deg", "mean_n_matches",
                    "first_frame_n_matches", "sec_per_step", "goal_mask_share", "render_mode",
                    "gl_renderer", "vsbench_commit", "vssim_commit")}
                row["success"] = rec.get("success")
                row["wall_s"] = round(wall, 1)
                row["records"] = str((out_root / f"{tag}.jsonl").relative_to(REPO))
                rows[tag] = row
                print(f"[u4] {tag}: {row['start_translation_mm']:.2f} -> {row['final_translation_mm']:.3f} mm, "
                      f"{row['final_rotation_deg']:.3f} deg, {row['terminated_reason']}, {row['steps']} steps, "
                      f"{row['sec_per_step']:.2f} s/step, frame-1 matches {row['first_frame_n_matches']}, "
                      f"predicate_a {(row['success'] or {}).get('predicate_a')}", flush=True)
                close = getattr(arm, "close", None)
                if close is not None:
                    close()
    _merge_record("smoke", rows)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--rooms", nargs="+", default=list(ROOMS))
    m.set_defaults(fn=cmd_measure)
    o = sub.add_parser("_one")
    o.add_argument("room")
    o.set_defaults(fn=cmd_one)
    i = sub.add_parser("isaac")
    i.add_argument("--rooms", nargs="+", default=list(ROOMS))
    i.set_defaults(fn=cmd_isaac)
    s = sub.add_parser("smoke")
    s.add_argument("--rooms", nargs="+", default=list(ROOMS))
    s.add_argument("--arms", nargs="+", default=["sift_ransac"])
    s.add_argument("--max-iter", type=int, default=100)
    s.set_defaults(fn=cmd_smoke)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
