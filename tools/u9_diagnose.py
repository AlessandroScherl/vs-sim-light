#!/usr/bin/env python3
"""U9 (docs/PORT_PLAN.md section 7): why did a near row of the rig check twin miss?

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/u9_diagnose.py --scene hollywood --arm artvs \\
        --worlds isaac_fit_v1 raw mock

Flies ONE cell of ``python -m vsbench.world.check_baselines`` - the same near start
(``check_baselines.near_start``: the near sampler on the scene's goal, first draw), the same
protocol cell, the same step count - in up to three worlds, with the copied per-step
telemetry, to separate the three suspects of a miss:

* ``isaac_fit_v1`` - the world and look the check flew (must reproduce the check's landing:
  MuJoCo on llvmpipe is frame-deterministic);
* ``raw`` - the same MuJoCo geometry without the Isaac look (the texture's own tone, the
  original light): a miss that goes away here is the look's;
* ``mock`` - posters only: the copied numpy ``MockPlaneScene`` with the scene's texture
  (exact inverse warp, no renderer at all; ``tools/u5_parity.py``'s mock world): a miss that
  stays here is neither MuJoCo's geometry nor its look - it is the method from this start.

Writes ``results/bridge/u9/diagnose.json`` (one key per ``scene/arm/world``): the landing,
the true error at fixed steps, its minimum, the ART-VS switch step (first step with more
than the coarse phase's matches), the scored stop. Records, not thresholds.
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

OUT = REPO / "results" / "bridge" / "u9" / "diagnose.json"
CURVE_STEPS = (1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600)


class _MockWorld:
    """``tools/u5_parity.py``'s mock world: the copied mock plate with the scene's texture."""

    render_mode = "numpy-mock"

    def __init__(self, name: str, specs: dict):
        from vsbench.vs.camera import CameraConfig
        from vsbench.vs.scene import MockPlaneScene
        from vsbench.vs.scenes import poster_goal_pose
        from vsbench.world.build import poster_spec

        spec = specs[name]
        if spec.kind != "poster":
            raise SystemExit(f"{name}: the mock world is a plate - posters only")
        built, _meta = poster_spec(spec)
        tex = np.array(built.objects[0].material.texture, dtype=np.uint8)
        _p, _R, d = poster_goal_pose(CameraConfig(), spec)
        self._m = MockPlaneScene(name=name, width_m=float(spec.raw["width_m"]),
                                 height_m=float(spec.raw["height_m"]), working_distance=float(d),
                                 texture=tex, render_scale=1)
        self.name = name

    def __getattr__(self, k):
        return getattr(self._m, k)

    def close(self):
        pass


def fly_one(scene_name: str, arm_id: str, world: str, steps: int, specs, protocol, ref) -> dict:
    from vsbench.vs import benchmark
    from vsbench.vs.bench.runner import run_episode
    from vsbench.world import fly
    from vsbench.world.mujoco_scene import MujocoScene

    if world == "mock":
        scene = _MockWorld(scene_name, specs)
    else:
        scene = MujocoScene(scene_name, specs=specs, look=world)
        fly.require_software_gl(scene)
    try:
        benchmark.check_scene_goal(scene, benchmark.load_starts(scene_name, "v2"))
        if hasattr(scene, "render_goal"):
            scene.render_goal()
        near = ref.near_start(scene)
        cfg = fly.arm_config(protocol, arm_id, max_iter=steps, device="cpu")
        arm = fly.build(protocol, arm_id, cfg, intrinsics=scene.intrinsics, scene_name=scene_name,
                        scene_spec=specs[scene_name])
        telem: list = []
        t0 = time.monotonic()
        rec = run_episode(scene, arm, near, cfg, telemetry=telem)
        wall = time.monotonic() - t0
        close = getattr(arm, "close", None)
        if close is not None:
            close()
    finally:
        scene.close()
    t = [e["t_mm"] for e in telem]
    n = [e["n"] for e in telem]
    coarse_n = n[0] if n else None
    switch = next((e["i"] for e in telem if coarse_n is not None and e["n"] > coarse_n), None)
    return {"world": world, "render_mode": getattr(scene, "render_mode", None),
            "start_id": near["start_id"], "start_mm": rec["start_translation_mm"],
            "start_deg": rec["start_rotation_deg"], "final_mm": rec["final_translation_mm"],
            "final_deg": rec["final_rotation_deg"], "steps": rec["steps"],
            "terminated": rec["terminated_reason"], "stop_step": rec["stop_step"],
            "stop_mm": rec["stop_translation_mm"], "artvs_switched": rec.get("artvs_switched"),
            "switch_step_from_matches": switch, "mean_n_matches": rec["mean_n_matches"],
            "t_mm_at": {str(s): t[s - 1] for s in CURVE_STEPS if s <= len(t)},
            "t_mm_min": min(t) if t else None,
            "t_mm_argmin": (int(np.argmin(t)) + 1) if t else None,
            "closer": bool(rec["final_translation_mm"] < rec["start_translation_mm"]),
            "sec_per_step": rec["sec_per_step"], "wall_s": round(wall, 1),
            "look": rec.get("look"), "goal_frame_md5": rec.get("goal_frame_md5")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--worlds", nargs="+", default=["isaac_fit_v1", "raw", "mock"],
                    choices=["isaac_fit_v1", "raw", "mock"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    from vsbench.world import fly

    env = fly.station_guard(2)
    from vsbench.vs.scenes import load_scenes
    from vsbench.world.check_baselines import reference

    ref = reference()
    protocol = fly.load_protocol()
    specs, _ = load_scenes()
    prov = {"vsbench_commit": fly.vsbench_commit(ignore=args.out.parent),
            "vssim_commit": fly.vssim_commit(), "bridge_env": env}
    for world in args.worlds:
        row = fly_one(args.scene, args.arm, world, args.steps, specs, protocol, ref)
        row.update(prov)
        doc = json.loads(args.out.read_text()) if args.out.exists() else {}
        doc[f"{args.scene}/{args.arm}/{world}"] = row
        doc["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"[u9-diag] {args.scene}/{args.arm}/{world}: {row['start_mm']:.2f} -> "
              f"{row['final_mm']:.3f} mm ({row['terminated']}, {row['steps']} steps), min "
              f"{row['t_mm_min']} @ {row['t_mm_argmin']}, switch {row['switch_step_from_matches']}, "
              f"{row['sec_per_step']:.2f} s/step", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
