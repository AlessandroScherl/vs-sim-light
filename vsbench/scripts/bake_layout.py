#!/usr/bin/env python3
# vs-sim copy: scripts/bake_layout.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Settle a NEW multi-object scene once and freeze its layout file.

    export VSSIM_ASSET_ROOT=/workspace/genvs-sim
    $ISAAC_PY scripts/bake_layout.py --scene my_new_room

The scene must declare a `settle:` block and a `layout_file:` in
`configs/scenes.yaml`; an existing layout file is left untouched (delete it
explicitly to re-settle, and expect the goal pose - and every start file built
against it - to change). A re-settle may never define a goal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsbench.vs.bake import bake_layout  # noqa: E402
from vsbench.vs.isaac_scene import IsaacScene, nonzero_exit_on_error  # noqa: E402
from vsbench.vs.scenes import load_scenes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    specs, _ = load_scenes()
    if args.scene not in specs:
        raise SystemExit(f"unknown scene {args.scene!r}; have {sorted(specs)}")
    with (IsaacScene(goal_render_warmup=1) as scene,
          nonzero_exit_on_error("bake_layout")):        # a bake captures no goal
        path = bake_layout(specs[args.scene], scene=scene)
        print(f"[bake_layout] {'wrote ' + str(path) if path else 'kept the existing layout'}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
