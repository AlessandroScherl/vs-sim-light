"""``python -m vsbench.world.fly`` - fly the copied baselines on the MuJoCo world.

    source tools/cpu_env.sh
    $VSBENCH_PY -m vsbench.world.fly run --scene hollywood --arms vitvs_dinov3 sift_ransac \\
        --start-set near60 --limit 1 --max-iter 100 --device cpu --out results/bridge/m1 --film
    $VSBENCH_PY -m vsbench.world.fly summary --out results/bridge/m1

What ``vs-sim/scripts/fly.py run`` does on Isaac (copied for reference at
``vsbench/scripts/fly.py``; docs/PORT_PLAN.md section 2.6), on ``MujocoScene``: the arm is
the protocol's cell (``configs/protocol.yaml``: its ``configs/arms/*.yaml`` file deep-merged
with the cell's deltas), built by the copied ``build_arm`` with the scene's intrinsics; the
starts are the copied ``benchmark.select_starts`` of a frozen, md5-pinned start file; the
scene's goal is checked against that file (``check_scene_goal``); the goal frame is captured
once and served to every arm; each episode is the copied ``runner.run_episode`` with the
start file as ``starts_payload``; records go through the copied ``open_run`` sink (run
directory with ``run.json``, ``resolved_config.yaml``, ``MD5SUMS``) plus a flat
``<scene>_<arm>.jsonl`` beside it, as vs-sim's driver writes them.

What the bridge adds, and nothing else:

* every record (and ``run.json``) carries ``render_mode`` (``mujoco-llvmpipe``),
  ``vsbench_commit`` (this repository's HEAD, ``+dirty`` when the tree is) and
  ``vssim_commit`` (the pin of the copy, from ``vsbench/vs/COPY_MANIFEST.json``), so a
  MuJoCo number can never be pooled with an Isaac one (PORT_PLAN D-9);
* the station rules: torch on 2 threads; refuses to run when torch sees a GPU; the
  ``nvidia-smi`` the copied loop polls for its hardware columns is shadowed by a stub
  (on this shared machine it would report the Isaac evaluation's GPU as this run's), so
  ``vram_used_mb`` / ``gpu_util_*`` read ``None``, the truth for a CPU run;
* ``--max-iter`` and ``--device`` overrides (stated on the record as ``max_iter`` and in
  the resolved config); ``--film`` writes the copied ``FilmWriter``'s match film.

``--method`` (a vsbench method through ``MethodArm``) is port unit U6a, not here.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "vsbench" / "vs" / "COPY_MANIFEST.json"


# ----------------------------------------------------------------- the station rules
def station_guard(threads: int = 2) -> dict:
    """Two threads everywhere, no GPU, no ``nvidia-smi``. Call before torch / mujoco load."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("VSBENCH_SOFTWARE_GL", "1")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ.setdefault(k, str(threads))
    stub_dir = Path(tempfile.mkdtemp(prefix="vsbench_nosmi_"))
    stub = stub_dir / "nvidia-smi"
    stub.write_text("#!/bin/sh\necho 'vsbench.world.fly: nvidia-smi hidden (CPU run)' >&2\nexit 9\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    os.environ["PATH"] = f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    import torch

    torch.set_num_threads(int(threads))
    if torch.cuda.is_available():
        raise SystemExit("torch sees a GPU - the bridge runs on the CPU only "
                         "(source tools/cpu_env.sh: CUDA_VISIBLE_DEVICES='')")
    return {"torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "cuda_available": False, "nvidia_smi": "hidden",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LP_NUM_THREADS": os.environ.get("LP_NUM_THREADS")}


def vsbench_commit() -> str | None:
    """This repository's HEAD, with ``+dirty`` when tracked files changed."""
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return head + ("+dirty" if dirty else "")


def vssim_commit() -> str:
    """The vs-sim commit the copy was made from (every manifest entry agrees; drift-checked)."""
    m = json.loads(MANIFEST.read_text())
    pins = {e["vssim_commit"] for e in m["files"].values()} | {m["vssim_commit"]}
    if len(pins) != 1:
        raise SystemExit(f"COPY_MANIFEST.json names {len(pins)} vs-sim commits: {sorted(pins)}")
    return pins.pop()


# --------------------------------------------------------------------- the protocol
def load_protocol(path=None) -> dict:
    """``configs/protocol.yaml`` under the copied strict loader (vs-sim's ``fly.py``)."""
    from vsbench.vs.bench import config as cfgmod
    from vsbench.vs.camera import REPO_ROOT

    p = Path(path) if path else REPO_ROOT / "configs" / "protocol.yaml"
    doc = cfgmod.load_document(p)
    if doc.get("schema") != "vssim-protocol-1":
        raise SystemExit(f"{p}: not a vssim-protocol-1 file")
    return doc


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def arm_entry(protocol: dict, arm_id: str) -> dict:
    for entry in protocol["arms"]:
        if entry["id"] == arm_id:
            return entry
    raise SystemExit(f"unknown arm {arm_id!r}; the protocol has "
                     f"{', '.join(e['id'] for e in protocol['arms'])}")


def arm_config(protocol: dict, arm_id: str, *, max_iter: int | None = None,
               device: str | None = None) -> dict:
    """The cell's config: its arm file deep-merged with the protocol's deltas (as vs-sim's
    ``fly.py`` does), then the bridge's two overrides, then resolved by the copied loader."""
    from vsbench.vs.bench import config as cfgmod
    from vsbench.vs.camera import REPO_ROOT

    entry = arm_entry(protocol, arm_id)
    cfg = _merge(cfgmod.load(REPO_ROOT / "configs" / "arms" / entry["base"]),
                 entry.get("overrides") or {})
    over: dict = {}
    if max_iter is not None:
        over["episode"] = {"max_iter": int(max_iter)}
    if device is not None and cfg["matcher"]["variant"] in cfgmod.DEVICE_VARIANTS:
        over["matcher"] = {"device": device}
    cfg = _merge(cfg, over)
    src = f"arms/{entry['base']} + protocol.yaml[{arm_id}]"
    if over:
        src += f" + bridge {json.dumps(over, sort_keys=True)}"
    return cfgmod.resolve(cfg, source=src)


def build(protocol: dict, arm_id: str, cfg: dict, *, intrinsics, scene_name: str, scene_spec):
    """``build_arm`` with only what the cell's ``needs`` names (vs-sim's ``fly.py build``)."""
    from vsbench.vs.bench.arms import build_arm

    overrides = {}
    if "scene_spec" in set(arm_entry(protocol, arm_id).get("needs") or []):
        overrides["scene_spec"] = scene_spec
    return build_arm(cfg, intrinsics=intrinsics, scene=scene_name, **overrides)


# ---------------------------------------------------------------------------- run
class _Film:
    """The copied ``FilmWriter`` on the loop's ``on_step`` hook, plus the first frame's
    match count (the M1 gate's ``n_matches >= 20 on frame 1``)."""

    def __init__(self, out_dir: Path | None, scene, goal_rgb, title: str, every: int, scale: float):
        from vsbench.vs.bench.films import FilmWriter

        self.scene = scene
        self.first_n_matches: int | None = None
        self.writer = (FilmWriter(out_dir, goal_rgb, every=every, scale=scale, title=title)
                       if out_dir is not None else None)

    def __call__(self, step_i, rgb, debug, pose):
        from vsbench.vs.scene import rotation_error_deg, translation_error_mm

        if self.first_n_matches is None:
            self.first_n_matches = int(debug.get("n_matches", 0))
        if self.writer is not None:
            g = self.scene.goal_pose
            self.writer.step(step_i, rgb, debug, translation_error_mm(pose.position, g.position),
                             rotation_error_deg(pose.rotation, g.rotation))

    def close(self):
        from vsbench.vs.bench.films import encode_episode

        if self.writer is None:
            return None
        d = self.writer.close()
        return encode_episode(d, keep_frames=False)


def cmd_run(args) -> int:
    env = station_guard(args.threads)
    from vsbench.vs import benchmark
    from vsbench.vs.bench.runner import open_run, run_episode
    from vsbench.vs.scenes import load_scenes
    from vsbench.world.mujoco_scene import MujocoScene

    protocol = load_protocol(args.protocol)
    specs, _ = load_scenes()
    if args.start_set not in protocol["start_sets"] and args.start_set not in benchmark.START_SETS:
        raise SystemExit(f"start set {args.start_set!r} is unknown")
    picked = benchmark.select_starts(args.scene, args.start_set, limit=args.limit)
    if args.start_ids:
        picked = [s for s in picked if s["start_id"] in set(args.start_ids)]
        if not picked:
            raise SystemExit(f"none of {args.start_ids} is in {args.scene}/{args.start_set}")
    payload = benchmark.load_starts(args.scene, benchmark.START_SETS[args.start_set]["file"])
    provenance = {"vsbench_commit": vsbench_commit(), "vssim_commit": vssim_commit()}
    out_root = Path(args.out).resolve()   # the copied encoder runs ffmpeg with cwd = the film dir
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    with MujocoScene(args.scene, samples=args.samples, specs=specs) as scene:
        benchmark.check_scene_goal(scene, payload)
        goal_rgb, _goal_depth = scene.render_goal()        # once, served to every arm
        goal_capture = scene.goal_capture_record()
        print(f"[fly] {args.scene}: {scene.render_mode}, goal {scene.working_distance:.6f} m "
              f"(file {payload['goal']['distance_m']:.6f}), goal md5 {goal_capture['goal_frame_md5']}, "
              f"{len(picked)} start(s) of {args.start_set}", flush=True)
        for arm_id in args.arms:
            cfg = arm_config(protocol, arm_id, max_iter=args.max_iter, device=args.device)
            t_build = time.monotonic()
            arm = build(protocol, arm_id, cfg, intrinsics=scene.intrinsics,
                        scene_name=args.scene, scene_spec=specs[args.scene])
            build_s = time.monotonic() - t_build
            tag = f"{args.scene}_{arm_id}"
            extra = {**goal_capture, **provenance, "bridge_env": env,
                     "scene_meta": scene.meta, "arm_build_s": round(build_s, 3),
                     "bridge_driver": "vsbench.world.fly"}
            with open_run(tag, cfg, out_root=out_root / tag, telemetry=args.telemetry,
                          allow_dirty=True, extra=extra) as sink:
                for i, start in enumerate(picked, 1):
                    steps: list | None = [] if args.telemetry else None
                    film_dir = (sink.path / "film" / start["start_id"]) if args.film else None
                    hook = _Film(film_dir, scene, goal_rgb, title=f"{arm_id} (MuJoCo)",
                                 every=args.film_every, scale=args.film_scale)
                    rec = run_episode(scene, arm, start, cfg, telemetry=steps,
                                      starts_payload=payload, on_step=hook)
                    film = hook.close()
                    rec["cell_id"] = tag
                    rec.update(provenance)
                    rec["render_mode"] = scene.render_mode
                    rec["first_frame_n_matches"] = hook.first_n_matches
                    rec["arm_build_s"] = round(build_s, 3)
                    if film is not None:
                        rec["film"] = str(Path(film).relative_to(out_root))
                    sink.write(rec, steps)
                    print(f"[fly] {tag} {i}/{len(picked)} {start['start_id']}: "
                          f"{rec['start_translation_mm']:.2f} -> {rec['final_translation_mm']:.3f} mm, "
                          f"{rec['final_rotation_deg']:.3f} deg, {rec['terminated_reason']}, "
                          f"{rec['steps']} steps, {rec['sec_per_step']:.3f} s/step, "
                          f"frame-1 matches {hook.first_n_matches}, "
                          f"predicate_a {rec['success'].get('predicate_a')}", flush=True)
                    summary_rows.append(rec)
            (out_root / f"{tag}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in sink.records), encoding="utf-8")
            seen = {r.get("goal_frame_md5") for r in sink.records}
            if seen != {goal_capture["goal_frame_md5"]}:
                raise RuntimeError(f"{tag}: the arms do not share one goal picture: {seen}")
            close = getattr(arm, "close", None)
            if close is not None:
                close()
    return 0


# ------------------------------------------------------------------------ summary
def cmd_summary(args) -> int:
    """Per cell: reached counts (stop / cap) under each success rule, landing medians."""
    from vsbench.vs.bench import scoring

    protocol = load_protocol(args.protocol)
    rules = list(protocol["scoring"]["success_rules"])
    files = sorted(Path(args.out).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl records under {args.out}")
    modes = set()
    rows = []
    for path in files:
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not recs:
            continue
        modes |= {r.get("render_mode") for r in recs}
        if len(modes) > 1:
            raise SystemExit(f"refusing to tabulate mixed renderers {sorted(map(str, modes))} "
                             "(PORT_PLAN D-9: MuJoCo and Isaac numbers are never pooled)")
        wd = recs[0].get("working_distance_m")
        row = {"cell": path.stem, "n": len(recs), "render_mode": recs[0].get("render_mode")}
        for rule in rules:
            stop = scoring.landing_stats(recs, rule=rule, at="stop", working_distance_m=wd)
            cap = scoring.landing_stats(recs, rule=rule, at="cap", working_distance_m=wd)
            row[rule] = {"stop": stop["reached"], "cap": cap["reached"],
                         "cap_translation_mm_med": cap["translation_mm_med"],
                         "cap_rotation_deg_med": cap["rotation_deg_med"]}
        rows.append(row)
        a = row[rules[0]]
        print(f"{path.stem:36s} n={len(recs):3d} {rules[0]} reached stop {a['stop']} / cap "
              f"{a['cap']}  cap median {a['cap_translation_mm_med']} mm "
              f"{a['cap_rotation_deg_med']} deg  [{row['render_mode']}]")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--protocol", default=None, help="default: vsbench/configs/protocol.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="fly one scene x N arms on the MuJoCo world (CPU)")
    r.add_argument("--scene", required=True)
    r.add_argument("--arms", nargs="+", required=True, help="protocol arm ids")
    r.add_argument("--start-set", default="capped30")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--start-ids", nargs="*", default=None)
    r.add_argument("--max-iter", type=int, default=None, help="override episode.max_iter")
    r.add_argument("--device", default="cpu", help="matcher.device for encoder arms (default cpu)")
    r.add_argument("--out", required=True)
    r.add_argument("--telemetry", action="store_true")
    r.add_argument("--film", action="store_true", help="the copied FilmWriter's match film")
    r.add_argument("--film-every", type=int, default=2)
    r.add_argument("--film-scale", type=float, default=0.25)
    r.add_argument("--samples", type=int, default=4, help="MuJoCo multisampling (colour only)")
    r.add_argument("--threads", type=int, default=2)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("summary", help="tabulate the flat records of an --out directory")
    s.add_argument("--out", required=True)
    s.add_argument("--json", default=None)
    s.set_defaults(fn=cmd_summary)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
