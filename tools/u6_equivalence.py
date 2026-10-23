"""U6 equivalence gate: the copied ViT-VS arm flown directly vs through the U6b wrapper + U6a
``MethodArm`` on the copied benchmark loop - same scene, start, steps; twists step by step.

    source tools/cpu_env.sh
    $VSBENCH_PY tools/u6_equivalence.py --arm vitvs_dinov3 --steps 100 \\
        --out results/bridge/u6_equivalence.json

Path A (direct): ``vsbench.world.fly.build`` of the protocol cell, ``runner.run_episode`` - what
``fly run --arms vitvs_dinov3`` flies (the M1 flight). Path B (wrapped): the registered vsbench
method ``vitvs_dinov3`` (``vsbench.methods.vssim_arms``) with ``episode=<scene>``,
``start_id=<start>`` (so its landmark draw is seeded as the runner seeds the direct arm's),
inside ``MethodArm`` with the external template at the arm's period, on the same loop. Both on
MuJoCo / llvmpipe, hollywood (D435i, 1920 x 1080), ``near60`` ``n_000`` by default.

Written: per step the max |twist A - twist B| (exact float64), the two landings, the M1
record's landing and the stored M1 telemetry's commanded twists (7 dp) against path A, and the
intrinsics each arm's controller received.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _recording(arm, out: list):
    """Wrap ``arm.step`` to keep a float64 copy of every twist it returns."""
    inner = arm.step

    def step(*a, **k):
        twist, debug, matches = inner(*a, **k)
        out.append(np.array(twist.as_array(), dtype=np.float64, copy=True))
        return twist, debug, matches

    arm.step = step
    return arm


def _intr(obj) -> dict:
    return {k: float(getattr(obj, k)) for k in ("fx", "fy", "cx", "cy", "width", "height")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", default="vitvs_dinov3")
    ap.add_argument("--scene", default="hollywood")
    ap.add_argument("--start-set", default="near60")
    ap.add_argument("--start-id", default="n_000")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from vsbench.world import fly

    env = fly.station_guard(2)
    from vsbench.vs import benchmark
    from vsbench.vs.bench.runner import run_episode
    from vsbench.vs.scenes import load_scenes
    from vsbench.world import method_arm as ma
    from vsbench.world.mujoco_scene import MujocoScene

    protocol = fly.load_protocol()
    specs, _ = load_scenes()
    start = next(s for s in benchmark.select_starts(a.scene, a.start_set)
                 if s["start_id"] == a.start_id)
    payload = benchmark.load_starts(a.scene, benchmark.START_SETS[a.start_set]["file"])
    out: dict = {"arm": a.arm, "scene": a.scene, "start_id": a.start_id, "steps": a.steps,
                 "bridge_env": env, "vsbench_commit": fly.vsbench_commit(),
                 "vssim_commit": fly.vssim_commit()}
    with MujocoScene(a.scene, specs=specs) as scene:
        fly.require_software_gl(scene)
        out["render_mode"] = scene.render_mode
        out["gl_renderer"] = scene.renderer.gl_renderer
        # path A: the copied arm, as `fly run --arms` builds it
        cfg_a = fly.arm_config(protocol, a.arm, max_iter=a.steps, device="cpu")
        arm_a = fly.build(protocol, a.arm, cfg_a, intrinsics=scene.intrinsics,
                          scene_name=a.scene, scene_spec=specs[a.scene])
        tw_a: list = []
        t0 = time.monotonic()
        rec_a = run_episode(scene, _recording(arm_a, tw_a), start, cfg_a, starts_payload=payload)
        wall_a = time.monotonic() - t0
        intr_a = _intr(arm_a.controller.intrinsics)
        arm_a.close()
        # path B: the vsbench method (U6b) inside MethodArm (U6a), on the same loop
        from vsbench.methods import make_method

        method = make_method(a.arm, episode=a.scene, start_id=a.start_id)
        cfg_b = ma.external_config(a.arm, dict(method.params), max_iter=a.steps)
        arm_b = ma.MethodArm(method, cfg_b, intrinsics=scene.intrinsics)
        tw_b: list = []
        t0 = time.monotonic()
        rec_b = run_episode(scene, _recording(arm_b, tw_b), start, cfg_b, starts_payload=payload)
        wall_b = time.monotonic() - t0
        intr_b = _intr(method._arm.controller.intrinsics)
        label_b = method.label
        arm_b.close()

    n = min(len(tw_a), len(tw_b))
    diff = [float(np.max(np.abs(tw_a[i] - tw_b[i]))) for i in range(n)]
    out.update({
        "path_a": {"label": rec_a["arm"], "steps": rec_a["steps"], "wall_s": round(wall_a, 1),
                   "final_translation_mm": rec_a["final_translation_mm"],
                   "final_rotation_deg": rec_a["final_rotation_deg"],
                   "first_twist": tw_a[0].tolist(), "controller_intrinsics": intr_a},
        "path_b": {"label": rec_b["arm"], "inner_label": label_b, "arm_kind": rec_b["arm_kind"],
                   "steps": rec_b["steps"], "wall_s": round(wall_b, 1),
                   "final_translation_mm": rec_b["final_translation_mm"],
                   "final_rotation_deg": rec_b["final_rotation_deg"],
                   "first_twist": tw_b[0].tolist(), "controller_intrinsics": intr_b},
        "scene_reported_intrinsics": _intr(scene.intrinsics),
        "n_compared": n, "twist_max_abs_diff": max(diff) if diff else None,
        "steps_bit_identical": int(sum(d == 0.0 for d in diff)),
        "twist_max_abs_diff_per_step": diff,
    })
    # the stored M1 flight (the copied arm, flown at 409656d / 04521b1)
    m1 = REPO / "results" / "bridge" / "m1" / f"{a.scene}_{a.arm}.jsonl"
    tel = sorted((REPO / "results" / "bridge" / "m1_telemetry" / f"{a.scene}_{a.arm}")
                 .glob("*/telemetry.jsonl.gz"))
    if m1.exists():
        r = json.loads(m1.read_text().splitlines()[0])
        out["m1_record"] = {"final_translation_mm": r["final_translation_mm"],
                            "final_rotation_deg": r["final_rotation_deg"],
                            "vsbench_commit": r["vsbench_commit"]}
    if tel and a.steps <= 100:
        with gzip.open(tel[0], "rt") as f:
            steps = json.loads(f.readline())["steps"]
        v = {int(s["i"]): np.asarray(s["v"], float) for s in steps}
        dev = [float(np.max(np.abs(v[i + 1] - np.round(tw_a[i], 7)))) for i in range(n)
               if i + 1 in v]
        out["m1_telemetry_v_vs_path_a_max_abs_diff_7dp"] = max(dev) if dev else None
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "twist_max_abs_diff_per_step"},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
