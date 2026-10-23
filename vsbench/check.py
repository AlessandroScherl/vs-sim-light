"""``python -m vsbench.check --method <name>``: the three setup checks on the poster scene.

    VSBENCH_SOFTWARE_GL=1 python -m vsbench.check --method oracle_pbvs
    python -m vsbench.check --method sift_ibvs --backend mujoco-glfw --param gain=0.8

Prints pass or fail per check and per axis; the exit code is 0 only when everything
passed.
"""

from __future__ import annotations

import argparse
import ast
import sys

from vsbench.camera import gate_camera
from vsbench.checks import format_report, run_checks
from vsbench.episode import EpisodeConfig
from vsbench.methods import available_methods, make_method
from vsbench.render.base import make_renderer, package_installed, probe_backend
from vsbench.scenes import make_scene, target_pose, world_points

MUJOCO_VARIANTS = ("mujoco-egl", "mujoco-osmesa", "mujoco-glfw")


def default_backend() -> str:
    """The first MuJoCo variant that starts in a fresh process here."""
    for name in MUJOCO_VARIANTS:
        if package_installed(name) and probe_backend(name)[0]:
            return name
    raise SystemExit("no MuJoCo variant starts here; pip install vsbench[mujoco] and see docs/PORT_NOTES.md section 1 for OSMesa/EGL")


def parse_params(items: list[str]) -> dict:
    """``key=value`` pairs; values are Python literals when they parse, strings otherwise."""
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--param wants key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main(argv=None) -> int:
    from vsbench.glenv import ensure_software_gl

    ensure_software_gl()   # software GL by default; a GPU context only with VSBENCH_ALLOW_GPU=1
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", required=True, choices=available_methods())
    ap.add_argument("--param", action="append", default=[], help="constructor argument key=value, repeatable")
    ap.add_argument("--backend", default=None, help="renderer variant (default: the first MuJoCo variant that starts)")
    ap.add_argument("--scene", default=None, help="scene name (default poster; marker for true_point_ibvs)")
    ap.add_argument("--max-steps", type=int, default=EpisodeConfig().max_steps,
                    help="step budget of each restoring-direction episode (default 3000; a smaller "
                         "budget can only fail more starts)")
    a = ap.parse_args(argv)
    params = parse_params(a.param)
    scene = a.scene or ("marker" if a.method == "true_point_ibvs" else "poster")
    if a.method == "true_point_ibvs" and "points_w" not in params:
        params["points_w"] = world_points(scene)  # the harness hands the instrument its points
    backend = a.backend or default_backend()
    cam = gate_camera()
    r = make_renderer(backend, make_scene(scene), cam)
    gl = (r.info().get("gl_renderer", "") if hasattr(r, "info") else "") or ""
    try:
        method = make_method(a.method, **params)
        results = run_checks(r, method, target_pose(scene, cam), EpisodeConfig(max_steps=a.max_steps), scene)
    finally:
        r.close()
    shown = {k: v for k, v in (getattr(method, "params", None) or {}).items() if k != "points_w"}
    print(f"backend {backend}{f' ({gl})' if gl else ''}, scene {scene}, method {a.method} {shown if shown else ''}, restoring budget {a.max_steps} steps")
    print(format_report(results, a.method))
    return 0 if all(x.passed for x in results) else 1


if __name__ == "__main__":
    sys.exit(main())
