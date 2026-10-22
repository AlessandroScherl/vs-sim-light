#!/usr/bin/env python
"""Parallel scaling: 1, 2, 4 and 8 processes, one renderer each, poster scene at 640 x 480.

    python tools/bench_parallel.py --backend mujoco-egl --out results/parallel_mujoco-egl.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsbench.sysinfo import system_record  # noqa: E402


def worker(backend: str, frames: int, warmup: int, q) -> None:
    from vsbench.camera import gate_camera
    from vsbench.poses import orbit_poses
    from vsbench.render.base import make_renderer
    from vsbench.stdscenes import poster_scene

    r = make_renderer(backend, poster_scene(lit=True), gate_camera(640, 480))
    poses = orbit_poses(frames + warmup, np.eye(4), radius=1.0, height=0.9)
    for T in poses[:warmup]:
        r.render(T)
    t0 = time.perf_counter()
    for T in poses[warmup:]:
        r.render(T)
    dt = time.perf_counter() - t0
    r.close()
    q.put(frames / dt)


def run(backend: str, n: int, frames: int, warmup: int) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(backend, frames, warmup, q)) for _ in range(n)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    fps = [q.get(timeout=1800) for _ in procs]
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    return {"processes": n, "frames_per_process": frames, "per_process_fps": fps, "total_fps": float(sum(fps)), "mean_process_fps": float(np.mean(fps)), "wall_s_incl_startup": wall}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--procs", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for n in a.procs:
        row = run(a.backend, n, a.frames, a.warmup)
        print(f"{a.label or a.backend} x{n}: total {row['total_fps']:.1f} fps, per process {row['mean_process_fps']:.1f} fps")
        rows.append(row)
    rec = {"kind": "parallel", "backend": a.backend, "label": a.label or a.backend, "system": system_record(), "cpu_threads": os.cpu_count(), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
