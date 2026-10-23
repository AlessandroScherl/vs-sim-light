#!/usr/bin/env python3
"""U5 (docs/PORT_PLAN.md section 7, U5): the copied baselines in the copied numpy mock world
and in the MuJoCo world, from the same starts - do they land the same?

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/u5_parity.py fly --world mock      --arms vitvs_dinov3
    PYTHONPATH=. $VSBENCH_PY tools/u5_parity.py fly --world mujoco    --arms vitvs_dinov3
    PYTHONPATH=. $VSBENCH_PY tools/u5_parity.py fly --world mujoco960 --arms vitvs_dinov3
    PYTHONPATH=. $VSBENCH_PY tools/u5_parity.py compare      # -> results/bridge/u5/parity.{json,md}

The three worlds (the scene's name, goal, starts, arm, seeds and loop are the same in all):

* ``mock``      - the copied ``vsbench.vs.scene.MockPlaneScene`` named after the scene, the
                  plate's size and goal distance from ``configs/scenes.yaml``, the texture the
                  MuJoCo plate carries (``vsbench.world.build.poster_spec``: the png's RGB);
                  an exact inverse warp, bilinear texture lookup, the texture's own tone, no
                  light; principal point on pixel 960's centre; off the plate black with
                  depth 0 (the copied ``clean_depth`` fallback 0.5 m);
* ``mujoco``    - ``MujocoScene(scene, look="raw")`` (U1/U4's picture: the closest MuJoCo look
                  to the texture's own tone), Isaac's principal point (959.5, 539.5);
* ``mujoco960`` - the same MuJoCo scene drawn with the mock's principal point (960, 540) (the
                  U1 test ``test_half_pixel_convention_against_the_copied_mock``'s control
                  renderer): what is left between it and ``mock`` is tone and filtering, what
                  is left between it and ``mujoco`` is the half pixel.

Every world reports the frozen ``D435I`` intrinsics (960, 540) to the arm, as Isaac does.

Each ``fly`` is one world x the given arms, from the first ``--near`` starts of ``near60``
and the first ``--narrow`` of ``narrow20``, ``--max-iter`` (150) steps each, through the
copied ``runner.run_episode`` (with its telemetry list, kept in memory) and the copied
``open_run`` sink. The bridge driver's station rules apply (``vsbench.world.fly``: torch on
2 threads, no GPU, ``nvidia-smi`` hidden, the MuJoCo worlds refused unless their GL context
is Mesa llvmpipe). Written per world and arm under ``results/bridge/u5/<world>/``:

* ``<scene>_<arm>/`` - the copied run directory (``run.json``, ``resolved_config.yaml``,
  ``baseline_check.jsonl``, ``MD5SUMS``), no telemetry file (it would be ~100 kB a start);
* ``<scene>_<arm>.jsonl`` - one record per start: the copied record plus the bridge's
  provenance and a ``u5`` block: the world, its principal point, the GL renderer, the final
  pose, the CPU seconds the episode cost this process, ART-VS's switch step (the first step
  whose frame was matched tiled, read from the arm's ``TiledMatcher.tiled`` latch on the
  loop's ``on_step`` hook), and the compact curve per step (``t_mm``, ``r_deg``, ``|v_xyz|``,
  ``|w|``, ``n``) taken from the copied telemetry entries.

``compare`` reads those records and writes ``results/bridge/u5/parity.json`` and
``parity.md``: per arm x start x world the landing (mm, deg at the cap), the curve statistics
(median over the last third, max over the second half), the ``cns_plateau`` step, the
first-frame match count, ART-VS's switch; per arm x start the world pairs (mock-mujoco, the
gate; mock-mujoco960 and mujoco960-mujoco, the attribution): landing differences, the
distance between the two landing poses, the twist-norm curve correlation.
``tests/world/test_parity.py`` gates the committed records (and writes
``results/bridge/u5_parity.json`` under ``VSBENCH_WRITE_GATES=1``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "bridge" / "u5"
WORLDS = ("mock", "mujoco", "mujoco960")
ARMS = ("vitvs_dinov3", "vitvs_dinov2", "artvs", "sift_ransac")
#: the gate pair and the two attribution pairs
PAIRS = (("mock", "mujoco"), ("mock", "mujoco960"), ("mujoco960", "mujoco"))
GATE_PAIR = ("mock", "mujoco")
GATE_MM, GATE_DEG = 3.0, 0.3
MOCK_RENDER_MODE = "numpy-mock"


# ------------------------------------------------------------------------- worlds
class MockWorld:
    """The copied ``MockPlaneScene`` for a poster scene, wearing the scene's name (so the
    landmark draw ``episode_seed(scene.name, start_id, seed)`` pairs with the MuJoCo world's)."""

    render_mode = MOCK_RENDER_MODE

    def __init__(self, scene_name: str, specs: dict):
        from vsbench.vs.camera import CameraConfig
        from vsbench.vs.scene import MockPlaneScene
        from vsbench.vs.scenes import poster_goal_pose
        from vsbench.world.build import poster_spec

        spec = specs[scene_name]
        if spec.kind != "poster":
            raise SystemExit(f"{scene_name}: the mock world is a plate - posters only")
        built, meta = poster_spec(spec)
        tex = np.array(built.objects[0].material.texture, dtype=np.uint8)
        _pos, _R, d = poster_goal_pose(CameraConfig(), spec)
        self.scene = MockPlaneScene(name=scene_name, width_m=float(spec.raw["width_m"]),
                                    height_m=float(spec.raw["height_m"]),
                                    working_distance=float(d), texture=tex, render_scale=1)
        self.principal_point_drawn = [float(self.scene.intrinsics.cx), float(self.scene.intrinsics.cy)]
        self.gl_renderer = None
        self.describe = {"world": "mock", "class": "vsbench.vs.scene.MockPlaneScene",
                         "texture": meta.get("texture"), "texture_wh": meta.get("texture_wh"),
                         "plate_m": [self.scene.width_m, self.scene.height_m],
                         "working_distance_m": float(d), "background": self.scene.background,
                         "principal_point_drawn": self.principal_point_drawn}

    def close(self):
        pass


class MujocoWorld:
    """``MujocoScene(look="raw")``; ``pp960=True`` redraws it with the mock's principal point."""

    def __init__(self, scene_name: str, specs: dict, *, pp960: bool, samples: int = 4):
        from vsbench.camera import CameraModel
        from vsbench.render.base import make_renderer
        from vsbench.world.camera import HALF_PIXEL, describe
        from vsbench.world.mujoco_scene import MujocoScene, render_mode_from

        self.scene = MujocoScene(scene_name, samples=samples, specs=specs, look="raw")
        if pp960:
            # U1's control renderer: the same scene, light and look, the camera's principal
            # point moved by half a pixel onto the mock's (960, 540); swapped in before the
            # first frame, so the goal capture is drawn with it too
            rc = self.scene.render_camera
            cam960 = CameraModel.from_focal(rc.width, rc.height, rc.fx, rc.fy,
                                            cx=rc.cx + HALF_PIXEL, cy=rc.cy + HALF_PIXEL,
                                            near=rc.near, far=rc.far)
            self.scene.renderer.close()
            self.scene.renderer = make_renderer("mujoco-egl", self.scene.scene_spec, cam960,
                                                samples=int(samples), **self.scene.renderer_kwargs)
            self.scene.render_camera = cam960
            self.scene.render_mode = render_mode_from(
                "egl", getattr(self.scene.renderer, "gl_renderer", None))
            self.scene.meta["render_camera"] = describe(cam960)
            self.scene.meta["u5_control"] = "principal point moved to (960, 540) by tools/u5_parity.py"
        rc = self.scene.render_camera
        self.principal_point_drawn = [float(rc.cx), float(rc.cy)]
        self.gl_renderer = getattr(self.scene.renderer, "gl_renderer", None)
        self.render_mode = self.scene.render_mode
        self.describe = {"world": "mujoco960" if pp960 else "mujoco", "class":
                         "vsbench.world.mujoco_scene.MujocoScene", "look": "raw",
                         "principal_point_drawn": self.principal_point_drawn,
                         "gl_renderer": self.gl_renderer, "samples": int(samples)}

    def close(self):
        self.scene.close()


def make_world(world: str, scene_name: str, specs: dict):
    if world == "mock":
        return MockWorld(scene_name, specs)
    if world in ("mujoco", "mujoco960"):
        return MujocoWorld(scene_name, specs, pp960=(world == "mujoco960"))
    raise SystemExit(f"unknown world {world!r}; have {WORLDS}")


# ----------------------------------------------------------------------- the hook
class _Hook:
    """The loop's ``on_step``: the first frame's match count and ART-VS's switch step."""

    def __init__(self, arm):
        self.arm = arm
        self.first_n_matches: int | None = None
        self.switch_step: int | None = None

    def __call__(self, step_i, rgb, debug, pose):
        if self.first_n_matches is None:
            self.first_n_matches = int(debug.get("n_matches", 0))
        tiled = getattr(getattr(self.arm, "matcher", None), "tiled", None)
        if tiled and self.switch_step is None:
            self.switch_step = int(step_i)


def compact_curve(steps: list[dict]) -> dict:
    """Per step: the true error (mm, deg) of the frame the step steered on, the commanded
    translational / rotational speed and the match count - from the copied telemetry."""
    v = np.array([s["v"] for s in steps], float).reshape(-1, 6)
    return {"i": [int(s["i"]) for s in steps], "t_mm": [float(s["t_mm"]) for s in steps],
            "r_deg": [float(s["r_deg"]) for s in steps],
            "v_xyz": [round(float(x), 7) for x in np.linalg.norm(v[:, :3], axis=1)],
            "w": [round(float(x), 7) for x in np.linalg.norm(v[:, 3:], axis=1)],
            "n": [int(s["n"]) for s in steps]}


def pick_starts(scene_name: str, n_near: int, n_narrow: int):
    from vsbench.vs import benchmark

    out = []
    for set_name, n in (("near60", n_near), ("narrow20", n_narrow)):
        if n <= 0:
            continue
        payload = benchmark.load_starts(scene_name, benchmark.START_SETS[set_name]["file"])
        for s in benchmark.select_starts(scene_name, set_name, limit=n):
            out.append((set_name, s, payload))
    return out


# --------------------------------------------------------------------------- fly
def cmd_fly(args) -> int:
    from vsbench.world import fly as flymod

    env = flymod.station_guard(args.threads)
    from vsbench.vs import benchmark
    from vsbench.vs.bench.runner import open_run, run_episode
    from vsbench.vs.scenes import load_scenes

    protocol = flymod.load_protocol(None)
    specs, _ = load_scenes()
    starts = pick_starts(args.scene, args.near, args.narrow)
    out_root = (Path(args.out) / args.world).resolve()
    provenance = {"vsbench_commit": flymod.vsbench_commit(ignore=Path(args.out).resolve()),
                  "vssim_commit": flymod.vssim_commit(),
                  "vsbench_commit_rule": flymod.VSBENCH_COMMIT_RULE}
    out_root.mkdir(parents=True, exist_ok=True)
    world = make_world(args.world, args.scene, specs)
    try:
        scene = world.scene
        if args.world != "mock":
            flymod.require_software_gl(scene)            # before the first frame
        for payload in {id(p): p for _s, _st, p in starts}.values():
            benchmark.check_scene_goal(scene, payload)
        print(f"[u5] {args.world} {args.scene}: {world.render_mode}, GL {world.gl_renderer!r}, "
              f"principal point drawn {world.principal_point_drawn}, reported "
              f"({scene.intrinsics.cx}, {scene.intrinsics.cy}), {len(starts)} start(s)", flush=True)
        for arm_id in args.arms:
            cfg = flymod.arm_config(protocol, arm_id, max_iter=args.max_iter, device="cpu")
            t_build = time.monotonic()
            arm = flymod.build(protocol, arm_id, cfg, intrinsics=scene.intrinsics,
                               scene_name=args.scene, scene_spec=specs[args.scene])
            build_s = time.monotonic() - t_build
            tag = f"{args.scene}_{arm_id}"
            goal_capture = dict(getattr(scene, "goal_capture_record", dict)())
            extra = {**goal_capture, **provenance, "bridge_env": env, "u5_world": world.describe,
                     "arm_build_s": round(build_s, 3), "bridge_driver": "tools/u5_parity.py fly"}
            flat = []
            with open_run(tag, cfg, out_root=out_root / tag, telemetry=False,
                          allow_dirty=True, extra=extra) as sink:
                for k, (set_name, start, payload) in enumerate(starts, 1):
                    steps: list = []
                    hook = _Hook(arm)
                    cpu0, wall0 = time.process_time(), time.monotonic()
                    rec = run_episode(scene, arm, start, cfg, telemetry=steps,
                                      starts_payload=payload, on_step=hook)
                    cpu_s = time.process_time() - cpu0
                    wall_s = time.monotonic() - wall0
                    final = scene.pose
                    rec["cell_id"] = tag
                    rec.update(provenance)
                    rec["render_mode"] = world.render_mode
                    rec["first_frame_n_matches"] = hook.first_n_matches
                    rec["arm_build_s"] = round(build_s, 3)
                    sink.write(rec)
                    rec = dict(rec)
                    rec["u5"] = {"world": args.world, "start_set": set_name,
                                 "principal_point_drawn": world.principal_point_drawn,
                                 "principal_point_reported": [float(scene.intrinsics.cx),
                                                              float(scene.intrinsics.cy)],
                                 "gl_renderer": world.gl_renderer,
                                 "final_position": [float(x) for x in final.position],
                                 "final_rotation": np.asarray(final.rotation, float).tolist(),
                                 "cpu_s": round(cpu_s, 2), "wall_s": round(wall_s, 2),
                                 "artvs_switch_step": hook.switch_step,
                                 "curve": compact_curve(steps)}
                    flat.append(rec)
                    print(f"[u5] {args.world} {tag} {k}/{len(starts)} {set_name}/{start['start_id']}: "
                          f"{rec['start_translation_mm']:.2f} -> {rec['final_translation_mm']:.3f} mm, "
                          f"{rec['final_rotation_deg']:.3f} deg, {rec['steps']} steps, "
                          f"stop {rec['stop_step']}, frame-1 matches {hook.first_n_matches}, "
                          f"switch {hook.switch_step}, {wall_s / max(1, rec['steps']):.3f} s/step, "
                          f"cpu {cpu_s:.0f} s", flush=True)
                    (out_root / f"{tag}.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in flat), encoding="utf-8")
            close = getattr(arm, "close", None)
            if close is not None:
                close()
    finally:
        world.close()
    return 0


# ----------------------------------------------------------------------- compare
def load_records(root: Path = OUT, scene: str = "hollywood") -> dict:
    """``{world: {arm: {start_key: record}}}`` for every flat file present."""
    out: dict = {}
    for world in WORLDS:
        for arm in ARMS:
            p = root / world / f"{scene}_{arm}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    key = f"{r['u5']['start_set']}/{r['start_id']}"
                    out.setdefault(world, {}).setdefault(arm, {})[key] = r
    return out


def curve_stats(rec: dict) -> dict:
    c = rec["u5"]["curve"]
    t, r = np.asarray(c["t_mm"], float), np.asarray(c["r_deg"], float)
    n = len(t)
    last_third, second_half = slice(n - n // 3, n), slice(n // 2, n)
    return {"n_steps": n,
            "steps_last_third": [n - n // 3 + 1, n], "steps_second_half": [n // 2 + 1, n],
            "t_mm_med_last_third": round(float(np.median(t[last_third])), 4),
            "t_mm_max_second_half": round(float(np.max(t[second_half])), 4),
            "t_mm_min_second_half": round(float(np.min(t[second_half])), 4),
            "r_deg_med_last_third": round(float(np.median(r[last_third])), 4),
            "r_deg_max_second_half": round(float(np.max(r[second_half])), 4)}


def row(rec: dict) -> dict:
    u5 = rec["u5"]
    return {"start_translation_mm": round(float(rec["start_translation_mm"]), 3),
            "start_rotation_deg": round(float(rec["start_rotation_deg"]), 3),
            "final_translation_mm": round(float(rec["final_translation_mm"]), 4),
            "final_rotation_deg": round(float(rec["final_rotation_deg"]), 4),
            "steps": int(rec["steps"]), "terminated_reason": rec["terminated_reason"],
            "cns_plateau_step": rec.get("stop_step"),
            "first_frame_n_matches": rec.get("first_frame_n_matches"),
            "artvs_switched": rec.get("artvs_switched"),
            "artvs_switch_step": u5.get("artvs_switch_step"),
            "render_mode": rec.get("render_mode"), "gl_renderer": u5.get("gl_renderer"),
            "principal_point_drawn": u5.get("principal_point_drawn"),
            "cpu_s": u5.get("cpu_s"), "sec_per_step": round(float(rec["sec_per_step"]), 3),
            "goal_frame_md5": rec.get("goal_frame_md5"),
            **curve_stats(rec)}


def _rot_deg(Ra, Rb) -> float:
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _corr(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 3 or a.std() == 0 or b.std() == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def pair(ra: dict, rb: dict) -> dict:
    """World a vs world b on one arm x start."""
    sa, sb = curve_stats(ra), curve_stats(rb)
    pa, pb = np.asarray(ra["u5"]["final_position"]), np.asarray(rb["u5"]["final_position"])
    d = {"d_final_translation_mm": round(abs(ra["final_translation_mm"] - rb["final_translation_mm"]), 4),
         "d_final_rotation_deg": round(abs(ra["final_rotation_deg"] - rb["final_rotation_deg"]), 4),
         "landing_pose_distance_mm": round(float(np.linalg.norm(pa - pb)) * 1e3, 4),
         "landing_pose_angle_deg": round(_rot_deg(ra["u5"]["final_rotation"],
                                                  rb["u5"]["final_rotation"]), 4),
         "d_t_mm_med_last_third": round(abs(sa["t_mm_med_last_third"] - sb["t_mm_med_last_third"]), 4),
         "d_t_mm_max_second_half": round(abs(sa["t_mm_max_second_half"] - sb["t_mm_max_second_half"]), 4),
         "d_r_deg_med_last_third": round(abs(sa["r_deg_med_last_third"] - sb["r_deg_med_last_third"]), 4),
         "twist_norm_corr_v_xyz": _corr(ra["u5"]["curve"]["v_xyz"], rb["u5"]["curve"]["v_xyz"]),
         "twist_norm_corr_w": _corr(ra["u5"]["curve"]["w"], rb["u5"]["curve"]["w"]),
         "t_curve_corr": _corr(ra["u5"]["curve"]["t_mm"], rb["u5"]["curve"]["t_mm"]),
         "first_frame_n_matches": [ra.get("first_frame_n_matches"), rb.get("first_frame_n_matches")],
         "cns_plateau_step": [ra.get("stop_step"), rb.get("stop_step")]}
    d["landing_gate"] = bool(d["d_final_translation_mm"] <= GATE_MM
                             and d["d_final_rotation_deg"] <= GATE_DEG)
    d["curve_median_gate"] = bool(d["d_t_mm_med_last_third"] <= GATE_MM
                                  and d["d_r_deg_med_last_third"] <= GATE_DEG)
    return d


def compare(root: Path = OUT, scene: str = "hollywood") -> dict:
    recs = load_records(root, scene)
    table: dict = {}
    pairs: dict = {}
    for world, arms in recs.items():
        for arm, starts in arms.items():
            for key, rec in starts.items():
                table.setdefault(arm, {}).setdefault(key, {})[world] = row(rec)
    for arm in ARMS:
        for key in sorted({k for w in recs.values() for k in w.get(arm, {})}):
            for a, b in PAIRS:
                ra = recs.get(a, {}).get(arm, {}).get(key)
                rb = recs.get(b, {}).get(arm, {}).get(key)
                if ra is not None and rb is not None:
                    pairs.setdefault(arm, {}).setdefault(key, {})[f"{a}-{b}"] = pair(ra, rb)
    gate_key = f"{GATE_PAIR[0]}-{GATE_PAIR[1]}"
    verdict: dict = {"landing": {}, "artvs_switch": None}
    for arm, keys in pairs.items():
        rows = {k: v[gate_key] for k, v in keys.items() if gate_key in v}
        if rows:
            verdict["landing"][arm] = {
                "n": len(rows), "pass": all(r["landing_gate"] for r in rows.values()),
                "failing": sorted(k for k, r in rows.items() if not r["landing_gate"]),
                "failing_but_curve_medians_agree": sorted(
                    k for k, r in rows.items() if not r["landing_gate"] and r["curve_median_gate"])}
    art = {w: recs.get(w, {}).get("artvs", {}) for w in GATE_PAIR}
    if all(art.values()):
        keys = sorted(set(art[GATE_PAIR[0]]) & set(art[GATE_PAIR[1]]))
        fired = {w: {k: bool(art[w][k].get("artvs_switched"))
                     and art[w][k]["u5"]["artvs_switch_step"] is not None for k in keys}
                 for w in GATE_PAIR}
        verdict["artvs_switch"] = {
            # the plan's gate: the switch fires on every start, in both worlds
            "pass": all(fired[w][k] for w in GATE_PAIR for k in keys),
            "not_fired": {w: [k for k in keys if not fired[w][k]] for w in GATE_PAIR},
            # the parity reading beside it: the two worlds agree start by start
            "worlds_agree_on_every_start": all(fired[GATE_PAIR[0]][k] == fired[GATE_PAIR[1]][k]
                                               for k in keys),
            "steps": {w: {k: art[w][k]["u5"]["artvs_switch_step"] for k in keys}
                      for w in GATE_PAIR}}
    spread: dict = {}
    for arm, keys in pairs.items():
        for name in sorted({n for v in keys.values() for n in v}):
            rows = [v[name] for v in keys.values() if name in v]
            spread.setdefault(arm, {})[name] = {
                "n": len(rows),
                **{f"{q}_{f}": round(float(fn([r[f] for r in rows])), 4)
                   for f in ("d_final_translation_mm", "d_final_rotation_deg",
                             "d_t_mm_med_last_third", "d_r_deg_med_last_third")
                   for q, fn in (("median", np.median), ("max", np.max))}}
    cpu = {w: round(sum(r["u5"]["cpu_s"] for a in arms.values() for r in a.values()), 1)
           for w, arms in recs.items()}
    renderers = sorted({str(r["u5"].get("gl_renderer")) for w, arms in recs.items() if w != "mock"
                        for a in arms.values() for r in a.values()})
    return {"scene": scene, "worlds": sorted(recs), "table": table, "pairs": pairs,
            "gate": {"pair": gate_key, "landing_mm": GATE_MM, "landing_deg": GATE_DEG,
                     "rule": "|final_translation_mm(mock) - final_translation_mm(mujoco)| <= 3 and "
                             "|final_rotation_deg(mock) - final_rotation_deg(mujoco)| <= 0.3 at the cap, "
                             "every start; ART-VS's artvs_switched True on every start in both worlds"},
            "verdict": verdict, "spread": spread, "cpu_s_per_world": cpu, "gl_renderers": renderers}


def markdown(res: dict) -> str:
    L = [f"# U5 parity - {res['scene']}: the copied mock vs MuJoCo (raw look)", "",
         "Landing = the error at the cap (step 150). Curve: median over the last third "
         "(steps 101-150) and max over the second half (steps 76-150) of the true error of the "
         "frame each step steered on.", "",
         "| arm | start | world | start mm/deg | landing mm | landing deg | med t last 1/3 | "
         "max t 2nd half | med r last 1/3 | plateau step | frame-1 n | switch step | cpu s |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        for key, worlds in sorted(res["table"].get(arm, {}).items()):
            for w in WORLDS:
                r = worlds.get(w)
                if r is None:
                    continue
                L.append(f"| {arm} | {key} | {w} | {r['start_translation_mm']:.2f}/"
                         f"{r['start_rotation_deg']:.2f} | {r['final_translation_mm']:.3f} | "
                         f"{r['final_rotation_deg']:.3f} | {r['t_mm_med_last_third']:.3f} | "
                         f"{r['t_mm_max_second_half']:.3f} | {r['r_deg_med_last_third']:.3f} | "
                         f"{r['cns_plateau_step']} | {r['first_frame_n_matches']} | "
                         f"{r['artvs_switch_step']} | {r['cpu_s']} |")
    L += ["", "## World pairs", "",
          "| arm | start | pair | d landing mm | d landing deg | pose distance mm | pose angle deg | "
          "d med t | d med r | corr \\|v\\| | corr t | gate |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        for key, prs in sorted(res["pairs"].get(arm, {}).items()):
            for name, p in prs.items():
                L.append(f"| {arm} | {key} | {name} | {p['d_final_translation_mm']:.3f} | "
                         f"{p['d_final_rotation_deg']:.3f} | {p['landing_pose_distance_mm']:.3f} | "
                         f"{p['landing_pose_angle_deg']:.3f} | {p['d_t_mm_med_last_third']:.3f} | "
                         f"{p['d_r_deg_med_last_third']:.3f} | {p['twist_norm_corr_v_xyz']} | "
                         f"{p['t_curve_corr']} | {'pass' if p['landing_gate'] else 'FAIL'} |")
    L += ["", "## Verdict (gate pair mock-mujoco)", "", "```", json.dumps(res["verdict"], indent=2),
          "```", "", f"CPU seconds per world: {res['cpu_s_per_world']}; GL renderers: "
          f"{res['gl_renderers']}", ""]
    return "\n".join(L)


def cmd_compare(args) -> int:
    res = compare(Path(args.out), args.scene)
    if not res["table"]:
        raise SystemExit(f"no records under {args.out}")
    out = Path(args.out)
    (out / "parity.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    (out / "parity.md").write_text(markdown(res))
    print(markdown(res))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fly", help="one world x arms from the U5 starts")
    f.add_argument("--world", required=True, choices=WORLDS)
    f.add_argument("--scene", default="hollywood")
    f.add_argument("--arms", nargs="+", required=True, choices=ARMS)
    f.add_argument("--near", type=int, default=3)
    f.add_argument("--narrow", type=int, default=2)
    f.add_argument("--max-iter", type=int, default=150)
    f.add_argument("--threads", type=int, default=2)
    f.add_argument("--out", default=str(OUT))
    f.set_defaults(fn=cmd_fly)
    c = sub.add_parser("compare", help="the parity table and the gate verdicts")
    c.add_argument("--scene", default="hollywood")
    c.add_argument("--out", default=str(OUT))
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
