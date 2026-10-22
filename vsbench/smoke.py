"""One rendered image plus a frame rate, for students.

    python -m vsbench.smoke                      # first backend that works on this machine
    python -m vsbench.smoke --backend mujoco-glfw
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from vsbench.camera import CameraModel
from vsbench.poses import orbit_poses
from vsbench.render.base import (
    BACKENDS,
    default_order,
    make_renderer,
    package_installed,
    probe_backend,
)
from vsbench.stdscenes import clutter_scene


def pick_backend() -> str:
    tried = []
    for name in default_order():
        if not package_installed(name):
            tried.append(f"{name}: package {BACKENDS[name][3]!r} not installed")
            continue
        ok, why = probe_backend(name)
        if ok:
            return name
        tried.append(f"{name}: {why.splitlines()[-1] if why else 'failed'}")
    sys.exit("no backend works here:\n  " + "\n  ".join(tried))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default=None, help=f"one of {', '.join(BACKENDS)}")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--out", default="smoke.png")
    a = ap.parse_args(argv)
    name = a.backend or pick_backend()
    cam = CameraModel.from_focal(a.width, a.height, fx=0.9 * a.width)
    t0 = time.perf_counter()
    r = make_renderer(name, clutter_scene(), cam)
    t_build = time.perf_counter() - t0
    poses = orbit_poses(a.frames, np.eye(4), radius=1.0, height=0.9)
    rgb, depth = r.render(poses[0])
    for T in poses[:5]:
        r.render(T)
    t0 = time.perf_counter()
    for T in poses:
        r.render(T)
    dt = (time.perf_counter() - t0) / a.frames
    from PIL import Image

    d = depth.copy()
    hit = d > 0
    if hit.any():
        d = np.where(hit, (d - d[hit].min()) / max(np.ptp(d[hit]), 1e-6), 1.0)
    dimg = (255 * (1.0 - d)).astype(np.uint8)
    Image.fromarray(np.hstack([rgb, np.stack([dimg] * 3, axis=-1)])).save(a.out)
    info = getattr(r, "info", lambda: {})()
    print(f"backend {name} ({info.get('gl_renderer', '?')}) built in {t_build:.1f} s")
    print(f"{a.width}x{a.height} rgb+depth: {dt * 1e3:.1f} ms/frame = {1.0 / dt:.1f} fps over {a.frames} frames")
    print(f"wrote {a.out} (rgb | depth)")
    r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
