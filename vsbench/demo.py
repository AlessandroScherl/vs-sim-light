"""``python -m vsbench.demo [--method sift_ibvs] [--scene poster] [--jobs N]``: the ten demo starts.

Runs the ten offsets of the Step 3 brief from the scene's target pose, prints status,
final error and steps per start, saves every log under ``results/demo/<method>/`` as
``start_NN.npz``, writes ``demo.json`` with the table, and for one start (10 by default,
the largest offset) the debug video ``start_NN.mp4`` and the match sheet
``start_NN_matches.png``.

``--jobs N`` runs the starts in N worker processes, one renderer and one method instance
each (MuJoCo keeps one GL context per process). Every episode is deterministic, so the
table and the logs are identical to a sequential run; only the wall time changes, and
only where one episode does not already saturate the machine (on a CPU-only laptop it
does; a GPU renderer leaves the cores to the method).

    VSBENCH_SOFTWARE_GL=1 python -m vsbench.demo --method sift_ibvs
    python -m vsbench.demo --method orb_ibvs --param gain=0.8 --backend mujoco-glfw
    python -m vsbench.demo --method true_point_ibvs          # marker scene by default
    python -m vsbench.demo --method sift_ibvs --jobs 4       # four starts at a time
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vsbench.camera import gate_camera
from vsbench.check import default_backend, parse_params
from vsbench.episode import DEMO_STARTS, EpisodeConfig, demo_start_poses, run_episode
from vsbench.methods import available_methods, make_method
from vsbench.render.base import make_renderer
from vsbench.scenes import make_scene, target_pose, world_points
from vsbench.sysinfo import system_record
from vsbench.video import match_sheet, write_video

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "demo"


def _row(log, i: int, wall: float) -> dict:
    t_cm, r_deg = DEMO_STARTS[i - 1]
    m = log.meta
    return {
        "start": i,
        "translation_cm": list(t_cm),
        "rotation_deg": list(r_deg),
        "status": m["status"],
        "steps": len(log),
        "final_error_mm": m["final_error_m"] * 1e3,
        "final_error_deg": float(np.rad2deg(m["final_error_rad"])),
        "call_s_mean": m["call_s_mean"],
        "call_s_max": m["call_s_max"],
        "lost_steps": int(log.lost.sum()),
        "wall_s": wall,
        "error": m.get("error"),
    }


def run_start(renderer, method, scene: str, out: Path, cfg: EpisodeConfig, i: int, video: bool) -> dict:
    """One demo start (1-based): the episode, its log, and the video and match sheet if asked."""
    T_t = target_pose(scene, renderer.cam)
    t0 = time.perf_counter()
    log = run_episode(renderer, method, demo_start_poses(T_t)[i - 1], T_t, cfg, scene)
    wall = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    log.save(out / f"start_{i:02d}.npz")
    row = _row(log, i, wall)
    if video:
        res = write_video(log, renderer, out / f"start_{i:02d}.mp4")
        match_sheet(log, renderer, out / f"start_{i:02d}_matches.png")
        row["video"] = str(out / f"start_{i:02d}.mp4")
        row["video_replay"] = res.summary()
    return row


HEADER = f"{'start':>5s}  {'translation cm':>16s}  {'rotation deg':>14s}  {'status':>12s}  {'steps':>5s}  {'final mm':>9s}  {'final deg':>9s}  {'ms/call':>7s}  {'wall s':>6s}"


def _print_row(row: dict) -> None:
    print(f"{row['start']:5d}  {str(tuple(row['translation_cm'])):>16s}  {str(tuple(row['rotation_deg'])):>14s}  {row['status']:>12s}  {row['steps']:5d}  {row['final_error_mm']:9.2f}  {row['final_error_deg']:9.3f}  {(row['call_s_mean'] or 0) * 1e3:7.1f}  {row['wall_s']:6.1f}", flush=True)
    if row.get("video"):
        print(f"       video {row['video']} ({row['video_replay']}), match sheet {row['video'].replace('.mp4', '_matches.png')}", flush=True)


def _record(method_name: str, params: dict, scene: str, backend: str, renderer_info: dict, cfg: EpisodeConfig, video_start: int | None, rows: list[dict], jobs: int, total_wall: float) -> dict:
    rows = sorted(rows, key=lambda r: r["start"])
    calls = [r["call_s_mean"] for r in rows if r["call_s_mean"]]
    return {
        "kind": "step3_demo",
        "method": method_name,
        "params": {k: v for k, v in (params or {}).items() if k != "points_w"},
        "scene": scene,
        "backend": backend,
        "renderer": renderer_info,
        "config": cfg.to_meta(),
        "video_start": video_start,
        "jobs": jobs,
        "total_wall_s": total_wall,
        "rows": rows,
        "converged": sum(r["status"] == "converged" for r in rows),
        "call_s_mean": float(np.mean(calls)) if calls else None,
        "system": system_record(),
    }


def _summary(rec: dict, out: Path) -> str:
    return f"{rec['method']}: {rec['converged']} of {len(rec['rows'])} starts converged; {(rec['call_s_mean'] or 0) * 1e3:.1f} ms per get_control_rate; {rec['total_wall_s']:.1f} s wall with {rec['jobs']} job(s); logs in {out}"


def run_demo(renderer, method, scene: str, out: Path, cfg: EpisodeConfig, video_start: int | None = 10, quiet: bool = False) -> dict:
    """Ten episodes one after the other in this process; returns the record written to ``demo.json``."""
    t0 = time.perf_counter()
    if not quiet:
        print(HEADER)
    rows = []
    for i in range(1, len(DEMO_STARTS) + 1):
        row = run_start(renderer, method, scene, out, cfg, i, video=(video_start == i))
        rows.append(row)
        if not quiet:
            _print_row(row)
    rec = _record(method.name, getattr(method, "params", None) or {}, scene, getattr(renderer, "name", "?"), getattr(renderer, "info", lambda: {})(), cfg, video_start, rows, 1, time.perf_counter() - t0)
    (out / "demo.json").write_text(json.dumps(rec, indent=1))
    if not quiet:
        print(_summary(rec, out))
    return rec


# -- parallel: one renderer and one method per worker process ------------------------------
_W: dict = {}


def _worker_init(backend: str, scene: str, method_name: str, params: dict, cv_threads: int | None = None) -> None:
    if cv_threads:
        import cv2

        cv2.setNumThreads(int(cv_threads))  # share the cores between workers instead of oversubscribing
    _W["renderer"] = make_renderer(backend, make_scene(scene), gate_camera())
    _W["method"] = make_method(method_name, **params)
    _W["scene"] = scene


def _worker_run(i: int, out: str, cfg: EpisodeConfig, video: bool) -> tuple[dict, dict]:
    r = _W["renderer"]
    row = run_start(r, _W["method"], _W["scene"], Path(out), cfg, i, video)
    return row, getattr(r, "info", lambda: {})()


def run_demo_parallel(backend: str, method_name: str, params: dict, scene: str, out: Path, cfg: EpisodeConfig, jobs: int, video_start: int | None = 10, quiet: bool = False, cv_threads: int | None = None) -> dict:
    """The ten starts spread over ``jobs`` spawned worker processes; same table and logs as
    ``run_demo``, written to the same ``demo.json``. ``cv_threads`` caps OpenCV's threads in
    each worker (None: OpenCV's default, every core in every worker)."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(HEADER)
    rows, info = [], {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_worker_init, initargs=(backend, scene, method_name, params, cv_threads)) as ex:
        futures = [ex.submit(_worker_run, i, str(out), cfg, video_start == i) for i in range(1, len(DEMO_STARTS) + 1)]
        for f in futures:  # in start order, so the table prints in order as results land
            row, info = f.result()
            rows.append(row)
            if not quiet:
                _print_row(row)
    rec = _record(method_name, params, scene, backend, info, cfg, video_start, rows, jobs, time.perf_counter() - t0)
    rec["cv_threads"] = cv_threads
    (out / "demo.json").write_text(json.dumps(rec, indent=1))
    if not quiet:
        print(_summary(rec, out))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", default="sift_ibvs", choices=available_methods())
    ap.add_argument("--param", action="append", default=[], help="constructor argument key=value, repeatable")
    ap.add_argument("--scene", default=None, help="scene name (default poster; marker for true_point_ibvs)")
    ap.add_argument("--backend", default=None, help="renderer variant (default: the first MuJoCo variant that starts)")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--video-start", type=int, default=10, help="which start gets the video and match sheet (0: none)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="results/demo")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes, one renderer each (default 1: this process)")
    ap.add_argument("--cv-threads", default="auto", help="OpenCV threads per worker with --jobs: a number, or auto = cores // jobs")
    a = ap.parse_args(argv)
    params = parse_params(a.param)
    scene = a.scene or ("marker" if a.method == "true_point_ibvs" else "poster")
    if a.method == "true_point_ibvs" and "points_w" not in params:
        params["points_w"] = world_points(scene)
    backend = a.backend or default_backend()
    cfg = EpisodeConfig(max_steps=a.max_steps)
    shown = {k: v for k, v in params.items() if k != "points_w"}
    if a.jobs > 1:
        print(f"backend {backend}, scene {scene}, method {a.method} {shown}, {a.jobs} worker processes")
        cv_threads = max(1, (os.cpu_count() or 1) // a.jobs) if a.cv_threads == "auto" else (int(a.cv_threads) or None)
        rec = run_demo_parallel(backend, a.method, params, scene, Path(a.out) / a.method, cfg, a.jobs, a.video_start or None, cv_threads=cv_threads)
    else:
        r = make_renderer(backend, make_scene(scene), gate_camera())
        print(f"backend {backend} ({r.info().get('gl_renderer', '') if hasattr(r, 'info') else ''}), scene {scene}, method {a.method} {shown}")
        try:
            rec = run_demo(r, make_method(a.method, **params), scene, Path(a.out) / a.method, cfg, a.video_start or None)
        finally:
            r.close()
    return 0 if all(x["status"] != "method_error" for x in rec["rows"]) else 1


if __name__ == "__main__":
    sys.exit(main())
