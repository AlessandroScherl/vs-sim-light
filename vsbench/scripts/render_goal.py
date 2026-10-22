#!/usr/bin/env python3
# vs-sim copy: scripts/render_goal.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Render the goal view of one or more scenes to PNG - a glance at what a scene is.

    export VSSIM_ASSET_ROOT=/workspace/genvs-sim
    $ISAAC_PY scripts/render_goal.py --out /workspace/vssim_out/goals            # all six
    $ISAAC_PY scripts/render_goal.py --scene cluttered2 --scene poster3 --out ...

Each scene is built in turn (the previous one's prims cleared), its goal checked
against the frozen start file when one exists, and the goal frame written as
`<out>/<scene>_goal.png` with its depth as `<scene>_goal_depth.npy`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsbench.vs import benchmark  # noqa: E402
from vsbench.vs.isaac_scene import IsaacScene, nonzero_exit_on_error  # noqa: E402
from vsbench.vs.scenes import goal_capture_settings, load_scenes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", action="append", help="repeatable; default: the six")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    specs, _ = load_scenes()
    names = args.scene or list(benchmark.SCENES)
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise SystemExit(f"unknown scene(s) {unknown}; have {sorted(specs)}")
    warmup, keep = goal_capture_settings()
    args.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    with (IsaacScene(goal_render_warmup=warmup, goal_capture_keep=keep) as scene,
          nonzero_exit_on_error("render_goal")):
        for name in names:
            scene.clear_scene_prims()
            scene.build_eval_scene(specs[name])
            if name in benchmark.STARTS:
                benchmark.check_scene_goal(scene, benchmark.load_starts(name))
            rgb, depth, _ = scene.render()
            Image.fromarray(rgb).save(args.out / f"{name}_goal.png")
            np.save(args.out / f"{name}_goal_depth.npy", depth)
            print(f"[render_goal] {name}: d={scene.working_distance:.6f} m, "
                  f"md5 {scene.goal_frame_md5}, settle diffs "
                  f"{scene.goal_settle['diff_prev']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
