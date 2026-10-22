# vs-sim copy: scripts/smoke_arms_plane.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Fly protocol arms on the numpy plane scene with a real picture: the Isaac-free smoke.

    python3 scripts/smoke_arms_plane.py --texture goal.png --arms lightglue_ransac roma_ibvs

Builds each arm from ``configs/protocol.yaml`` with its REAL models, puts the picture on
:class:`vsbench.vs.scene.MockPlaneScene` (the poster scene without Isaac, 2 ms a frame), and
runs the rig check's two questions: the identity step (camera at the goal, the goal
picture as the live frame - the twist must be zero) and the benchmark's near start
(7.76 mm / 2.69 deg) for ``--steps`` steps. Prints one line per question. Needs the
``baselines`` extra and the arm's package/weights; nothing here downloads.

Station record 2026-09-17 (RTX 4090, Isaac's hollywood goal as the picture, 60 steps):
lightglue 0 / 0 at the goal, 7.76 -> 1.16 mm (RANSAC) / 1.05 mm (no RANSAC), 0.4 s a step;
roma 4e-4 m/s / 0.05 deg/s at the goal, 7.76 -> 0.97 mm, 0.77 s a step; romav2 5e-5 /
0.006, 7.76 -> 0.99 mm, 0.72 s a step (first frame 5 s: the model builds).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fly  # noqa: E402

from vsbench.vs import starts as startsmod  # noqa: E402
from vsbench.vs.bench.runner import run_episode  # noqa: E402
from vsbench.vs.scene import MockPlaneScene  # noqa: E402
from vsbench.vs.scenes import load_scenes  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--texture", required=True, type=Path, help="the picture on the plane")
    ap.add_argument("--arms", nargs="+", required=True, help="protocol arm ids")
    ap.add_argument("--scene-name", default="hollywood",
                    help="the benchmark scene whose RANSAC default and name to use")
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args(argv)

    import cv2

    tex = cv2.cvtColor(cv2.imread(str(args.texture)), cv2.COLOR_BGR2RGB)
    scene = MockPlaneScene(name=args.scene_name, texture=tex)
    specs, _ = load_scenes()
    protocol = fly.load_protocol()
    g = scene.goal_pose
    near = dict(startsmod.sample_near(g.position, g.rotation, scene.working_distance, n=1)[0],
                starts_seed=startsmod.NEAR["seed"])
    failed = 0
    for arm_id in args.arms:
        t0 = time.time()
        try:
            arm, cfg = fly.build(protocol, arm_id, intrinsics=scene.intrinsics,
                                 scene_name=args.scene_name, scene_spec=specs.get(args.scene_name))
            scene.reset(g)
            rgb, depth, _ = scene.render()
            arm.set_episode(args.scene_name, "identity")
            arm.reset(rgb, depth)
            tw, dbg, _m = arm.step(rgb, depth)
            v = tw.as_array()
            print(f"[smoke] {arm_id} label={arm.name} identity n={dbg.get('n_matches')} "
                  f"|v|={np.linalg.norm(v[:3]):.2e} m/s "
                  f"|w|={np.degrees(np.linalg.norm(v[3:])):.2e} deg/s "
                  f"match_ms={dbg.get('match_ms', 0):.0f}", flush=True)
            cfg = dict(cfg, episode={**cfg["episode"], "max_iter": int(args.steps)})
            rec = run_episode(scene, arm, near, cfg)
            print(f"[smoke] {arm_id} near {rec['start_translation_mm']:.2f}->"
                  f"{rec['final_translation_mm']:.3f} mm {rec['start_rotation_deg']:.2f}->"
                  f"{rec['final_rotation_deg']:.3f} deg steps={rec['steps']} "
                  f"stop={rec['terminated_reason']} n={rec['mean_n_matches']:.1f} "
                  f"s/step={rec['sec_per_step']:.2f} wall={time.time() - t0:.0f}s", flush=True)
            close = getattr(arm, "close", None)
            if close is not None:
                close()
        except Exception:  # noqa: BLE001 - one broken arm must not end the smoke
            traceback.print_exc()
            print(f"[smoke] {arm_id} FAILED", flush=True)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
