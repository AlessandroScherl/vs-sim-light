#!/usr/bin/env python3
# vs-sim copy: scripts/build_starts.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Build a frozen start file for a scene, with its md5 sidecar.

Posters need no simulator (their goal is closed-form):

    python3 scripts/build_starts.py --scene hollywood --out /tmp/starts

Rooms need the measured bounding box, so run under Isaac:

    export VSSIM_ASSET_ROOT=/workspace/genvs-sim
    $ISAAC_PY scripts/build_starts.py --scene cluttered2 --out /workspace/starts

`--kind v2` (default) writes the campaign file (wide 500 + narrow 40, seed
20260820 unless `--seed`); `--kind near` writes the terminal-phase stratum (60
starts, seed 20260902); `--kind simple` writes the instance suite's simple cell (10
starts, seed 20260923). The benchmark's own files are in benchmark/starts/ and
regenerate from these samplers content for content (rotation matrices to ~4e-16 -
the residual is BLAS, so a rebuilt file's md5 will differ from the pin).

`--scene` takes several names: the rooms are then built in ONE Isaac process, one after
the other (the instance suite's twelve scenes: about a minute instead of twelve boots):

    $ISAAC_PY scripts/build_starts.py --kind simple --out /workspace/starts_inst \
        --scene inst_shoe inst_shoe_rev inst_bottle inst_bottle_rev ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsbench.vs import starts as startsmod  # noqa: E402
from vsbench.vs.camera import CameraConfig  # noqa: E402
from vsbench.vs.scenes import load_scenes, poster_goal_pose  # noqa: E402


def with_goal(spec, fn):
    """Call ``fn(cam_pos, R, distance_m)`` with the scene's goal - offline for a poster,
    INSIDE the Isaac context for a room: Isaac's ``close()`` ends the process on the
    station, so anything that must happen (the write) happens before it."""
    return with_goals([spec], lambda _spec, *goal: fn(*goal))


def with_goals(specs, fn):
    """``fn(spec, cam_pos, R, distance_m)`` for every spec: posters offline, rooms in ONE
    Isaac process (cleared between scenes), every call inside the Isaac context."""
    rc = 0
    rooms = [s for s in specs if s.kind != "poster"]
    for spec in specs:
        if spec.kind == "poster":
            rc |= int(fn(spec, *poster_goal_pose(CameraConfig(), spec)) or 0)
    if not rooms:
        return rc
    from vsbench.vs.isaac_scene import IsaacScene, nonzero_exit_on_error
    with (IsaacScene(goal_render_warmup=1) as scene,
          nonzero_exit_on_error("build_starts")):        # no goal frame is needed
        for spec in rooms:
            scene.clear_scene_prims()
            scene.build_eval_scene(spec)
            goal = scene.goal_pose
            rc |= int(fn(spec, goal.position, goal.rotation, scene.working_distance) or 0)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True, nargs="+")
    ap.add_argument("--kind", choices=("v2", "near", "simple"), default="v2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hardware", default=None, help="a label for the payload's record")
    ap.add_argument("--goal-from", default=None,
                    help="take each scene's goal from its pinned start file of this kind "
                         "(e.g. simple) instead of building the scene - offline, no Isaac")
    ap.add_argument("--tag", default=None,
                    help="the file's tag (<scene>_starts_<tag>.json; default: --kind)")
    ap.add_argument("--id-prefix", default=None, help="--kind near: the start ids' prefix")
    args = ap.parse_args()
    specs, _ = load_scenes()
    unknown = [s for s in args.scene if s not in specs]
    if unknown:
        raise SystemExit(f"unknown scene(s) {unknown}; have {sorted(specs)}")

    def build_and_write(spec, pos, R, d):
        if args.kind == "v2":
            payload = startsmod.build_payload(spec.name, pos, R, d,
                                              seed=args.seed or startsmod.SEED,
                                              hardware=args.hardware)
        elif args.kind == "near":
            payload = startsmod.build_near_payload(spec.name, pos, R, d,
                                                   seed=args.seed or startsmod.NEAR["seed"],
                                                   hardware=args.hardware,
                                                   id_prefix=args.id_prefix or "n_")
        else:
            payload = startsmod.build_simple_payload(spec.name, pos, R, d,
                                                     seed=args.seed or startsmod.SIMPLE["seed"],
                                                     hardware=args.hardware)
        path = args.out / f"{spec.name}_starts_{args.tag or args.kind}.json"
        args.out.mkdir(parents=True, exist_ok=True)
        digest = startsmod.write_payload(path, payload)
        print(f"[build_starts] wrote {path} (md5 {digest}, goal distance {d:.6f} m, "
              f"{len(payload['starts'])} starts)", flush=True)
        return 0

    if args.goal_from:            # the goal as already pinned - no scene is built
        from vsbench.vs import benchmark

        rc = 0
        for s in args.scene:
            goal = benchmark.load_starts(s, args.goal_from)["goal"]
            rc |= int(build_and_write(specs[s], np.asarray(goal["cam_pos"], float),
                                      np.asarray(goal["cam_R_opt"], float),
                                      float(goal["distance_m"])) or 0)
        return rc
    return with_goals([specs[s] for s in args.scene], build_and_write)


if __name__ == "__main__":
    raise SystemExit(main())
