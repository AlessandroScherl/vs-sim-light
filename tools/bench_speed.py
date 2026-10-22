#!/usr/bin/env python
"""Speed: median ms per frame (RGB plus depth) over 200 frames after 20 warm-up frames,
on the poster scene and the clutter scene, at 640 x 480 and 1440 x 1080.

    python tools/bench_speed.py --backend mujoco-egl --out results/speed_mujoco-egl.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsbench.camera import gate_camera  # noqa: E402
from vsbench.poses import orbit_poses  # noqa: E402
from vsbench.render.base import make_renderer  # noqa: E402
from vsbench.stdscenes import STD_SCENES, triangle_count  # noqa: E402
from vsbench.sysinfo import system_record  # noqa: E402


def run_case(backend: str, scene_name: str, res, frames: int, warmup: int, **kw) -> dict:
    scene = STD_SCENES[scene_name]()
    cam = gate_camera(*res)
    t0 = time.perf_counter()
    r = make_renderer(backend, scene, cam, **kw)
    build_s = time.perf_counter() - t0
    poses = orbit_poses(frames + warmup, np.eye(4), radius=1.0, height=0.9)
    for T in poses[:warmup]:
        r.render(T)
    times = []
    for T in poses[warmup:]:
        t0 = time.perf_counter()
        r.render(T)
        times.append(time.perf_counter() - t0)
    info = getattr(r, "info", lambda: {})()
    r.close()
    t = np.array(times) * 1e3
    return {
        "scene": scene_name,
        "width": res[0],
        "height": res[1],
        "triangles": triangle_count(scene),
        "frames": frames,
        "warmup": warmup,
        "build_s": build_s,
        "median_ms": float(np.median(t)),
        "mean_ms": float(t.mean()),
        "p90_ms": float(np.percentile(t, 90)),
        "fps": float(1e3 / np.median(t)),
        "renderer": info,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--scenes", nargs="+", default=["poster_lit", "clutter"])
    ap.add_argument("--res", nargs="+", default=["640x480", "1440x1080"])
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--samples", type=int, default=None, help="MuJoCo offsamples override")
    ap.add_argument("--label", default=None, help="variant label in the JSON (default: backend)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    kw = {} if a.samples is None else {"samples": a.samples}
    cases = []
    single = len(a.scenes) == 1 and len(a.res) == 1
    for scene in a.scenes:
        for res in a.res:
            w, h = (int(x) for x in res.lower().split("x"))
            if single:
                c = run_case(a.backend, scene, (w, h), a.frames, a.warmup, **kw)
            else:
                # one renderer per process: every case gets a fresh interpreter
                with tempfile.TemporaryDirectory() as td:
                    sub = Path(td) / "case.json"
                    cmd = [sys.executable, __file__, "--backend", a.backend, "--scenes", scene, "--res", res, "--frames", str(a.frames), "--warmup", str(a.warmup), "--out", str(sub)]
                    if a.samples is not None:
                        cmd += ["--samples", str(a.samples)]
                    p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), timeout=3600)
                    if p.returncode != 0:
                        raise RuntimeError(f"case {scene} {res} failed:\n{p.stderr[-2000:]}")
                    c = json.loads(sub.read_text())["cases"][0]
            print(f"{a.label or a.backend} {scene} {w}x{h}: median {c['median_ms']:.2f} ms = {c['fps']:.1f} fps (build {c['build_s']:.1f} s)")
            cases.append(c)
    rec = {"kind": "speed", "backend": a.backend, "label": a.label or a.backend, "system": system_record(), "cases": cases}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
