"""Render one standard scene at one seeded pose and save it: the determinism and
cross-variant checks run this in fresh processes.

    python -m vsbench.render_once --backend mujoco-egl --scene poster --seed 7 --out a.npz
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from vsbench.camera import gate_camera
from vsbench.poses import sample_poses
from vsbench.render.base import make_renderer
from vsbench.stdscenes import STD_SCENES


def main(argv=None) -> int:
    from vsbench.glenv import ensure_software_gl

    ensure_software_gl()   # software GL by default; a GPU context only with VSBENCH_ALLOW_GPU=1
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--scene", default="poster", choices=sorted(STD_SCENES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--index", type=int, default=0, help="which of the seeded poses")
    ap.add_argument("--count", type=int, default=1, help="render poses index..index+count-1")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cam = gate_camera(a.width, a.height)
    scene = STD_SCENES[a.scene]()
    poses = sample_poses(a.seed, a.index + a.count, np.eye(4), dist=(0.5, 1.2), max_slant_deg=50.0)
    r = make_renderer(a.backend, scene, cam)
    rgbs, depths, Ts = [], [], []
    for T in poses[a.index : a.index + a.count]:
        rgb, depth = r.render(T)
        rgbs.append(rgb)
        depths.append(depth)
        Ts.append(T)
    info = getattr(r, "info", lambda: {})()
    r.close()
    np.savez_compressed(a.out, rgb=np.stack(rgbs), depth=np.stack(depths), T_wc=np.stack(Ts), info=json.dumps(info))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
