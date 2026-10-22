#!/usr/bin/env python
"""How much faster do the ten demo starts run with N worker processes?

Runs ``vsbench.demo`` for one method with 1, 2, 4 and 8 jobs (no video), records the wall
time and the per-call time of each run, checks that every run gives the same table, and
writes ``results/episodes_parallel_<variant>_<label>.json``. ``tools/step3_report.py``
shows these records when present.

    python tools/bench_episodes.py --backend mujoco-egl --method sift_ibvs           # GPU if there is one
    VSBENCH_SOFTWARE_GL=1 python tools/bench_episodes.py --backend mujoco-egl --label cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vsbench.demo import run_demo_parallel  # noqa: E402
from vsbench.episode import EpisodeConfig  # noqa: E402
from vsbench.scenes import world_points  # noqa: E402
from vsbench.sysinfo import system_record  # noqa: E402

COMPARE = ("start", "status", "steps", "final_error_mm", "final_error_deg")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mujoco-egl")
    ap.add_argument("--method", default="sift_ibvs")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--jobs", default="1,2,4,8")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--label", default="", help="suffix for the record file, e.g. gpu or cpu")
    ap.add_argument("--cv-threads", default="default", help="OpenCV threads per worker: default (all cores in every worker), auto (cores // jobs) or a number")
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    scene = a.scene or ("marker" if a.method == "true_point_ibvs" else "poster")
    params = {"points_w": world_points(scene)} if a.method == "true_point_ibvs" else {}
    cfg = EpisodeConfig(max_steps=a.max_steps)
    rows, tables, info = [], [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for jobs in [int(j) for j in a.jobs.split(",")]:
            cv = None if a.cv_threads == "default" else (max(1, (os.cpu_count() or 1) // jobs) if a.cv_threads == "auto" else int(a.cv_threads))
            rec = run_demo_parallel(a.backend, a.method, params, scene, Path(tmp) / f"jobs{jobs}", cfg, jobs, video_start=None, quiet=True, cv_threads=cv)
            info = rec["renderer"]
            tables.append([{k: r[k] for k in COMPARE} for r in rec["rows"]])
            rows.append({"jobs": jobs, "cv_threads": cv, "wall_s": rec["total_wall_s"], "call_ms_mean": (rec["call_s_mean"] or 0) * 1e3, "converged": rec["converged"], "steps_total": sum(r["steps"] for r in rec["rows"]), "episode_wall_s_mean": sum(r["wall_s"] for r in rec["rows"]) / len(rec["rows"])})
            r0 = rows[0]
            print(f"{a.method} on {a.backend} ({info.get('gl_renderer', '?')}): {jobs} job(s), OpenCV threads {cv or 'default'}: {rec['total_wall_s']:.1f} s wall, speedup x{r0['wall_s'] / rec['total_wall_s']:.2f}, {rows[-1]['call_ms_mean']:.1f} ms per call, {rec['converged']} of 10 converged", flush=True)
    identical = all(t == tables[0] for t in tables)
    print("tables identical across job counts:", identical)
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    label = f"_{a.label}" if a.label else ""
    path = out / f"episodes_parallel_{a.method}_{a.backend}{label}.json"
    path.write_text(json.dumps({"kind": "episodes_parallel", "backend": a.backend, "renderer": info, "method": a.method, "scene": scene, "cv_threads": a.cv_threads, "config": cfg.to_meta(), "rows": rows, "tables_identical": identical, "system": system_record()}, indent=1))
    print(f"record: {path}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
