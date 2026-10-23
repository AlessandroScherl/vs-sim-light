"""``python -m vsbench.world.check_baselines`` - does each baseline work on the MuJoCo world?

    source tools/cpu_env.sh
    $VSBENCH_PY -m vsbench.world.check_baselines --scenes hollywood \\
        --arms sift_ransac orb_ransac vitvs_dinov3 artvs --near-steps 200 --far 1 \\
        --out results/bridge/u9/hollywood --frames preview
    $VSBENCH_PY -m vsbench.world.check_baselines summary --runs results/bridge/u9/* \\
        --out results/bridge/u9                     # + Isaac's own check numbers, context only

The twin (port unit U9, ``docs/PORT_PLAN.md`` section 7) of vs-sim's Isaac rig check
``scripts/check_baselines.py`` (copied, frozen, at ``vsbench/scripts/check_baselines.py``).
Per scene x arm, the same three checks:

1. **Identity** - the camera at the goal, the goal picture handed in as the live frame: the
   arm must find its own points again and command (almost) no motion
   (``|v| < 1e-4 m/s``, ``|w| < 0.01 deg/s``, at least one match - vs-sim's verdict ``ok``).
2. **Near** - the benchmark's near sampler on this scene's goal (3-8 mm, <= 3 deg, seed
   20260902, first draw) flown ``--near-steps`` steps.
3. **Far** (``--far N``) - the first N starts of ``--far-set`` (default ``capped30``) flown to
   the cell's cap: a smoke of the whole chain, recorded, never thresholded.

**No third copy.** The check itself is vs-sim's: :func:`reference` loads the copied driver as a
library (``identity_check``, ``episode_check``, ``near_start``, ``parse_budgets``,
``load_arm_plugin`` / ``resolve_arm_sources`` / ``build_plugin_arm``, ``write_table``, the
thresholds), so a vs-sim fix to the check arrives with the next pin move and the drift tool
guards it. The protocol helpers (``load_protocol``, ``arm_config``, ``build``), the station
guard and the provenance are ``vsbench.world.fly``'s. This module only does what vs-sim's
``main`` does with ``IsaacScene``, on ``MujocoScene``: one scene alive at a time (each scene is
built, checked against the far set's frozen start file, flown, and closed before the next).

What the bridge adds, and nothing else:

* the station rules of ``vsbench.world.fly`` (2 threads, GPU hidden by force, ``nvidia-smi``
  stubbed, the scene refused unless it draws on Mesa llvmpipe; ``--allow-gpu`` lifts it, never
  during the Isaac evaluation); ``--device`` (default ``cpu``) for the encoder arms;
* ``--look`` (default ``isaac_fit_v1``) and ``--samples``, as ``fly run``;
* every scene block and every check entry carries ``render_mode``; ``check.json`` carries a
  ``bridge`` block (``vsbench_commit``, ``vssim_commit``, the station env, the look, CPU time);
* ``records.jsonl``: every flown episode's full copied record (the copied ``run_episode``'s,
  captured as ``episode_check`` makes it) plus ``check_row`` / ``check_arm`` and the provenance
  fields, so a landing in ``check.json`` can be traced to the record that produced it;
* the near verdict ``near_ok`` (:data:`NEAR_RULE`). vs-sim's script prints the near landing
  and states the expectation in its docstring ("it must land closer than it started"), but
  computes no verdict; the twin reads that sentence literally and writes the verdict next to
  the entry (the "by a lot" is not quantified there and is not thresholded here: the ratio is
  reported);
* ``--frames full | preview | none``: ``full`` writes vs-sim's goal frame, goal depth and
  first live frame; ``preview`` (what is committed) quarter-size PNGs and the depth's md5 only.

``--mask none|objects|<file.png>`` (2026-10-23, ``vsbench.world.masks``, as ``fly run``): every
protocol arm flies its masked cell (identity, near and far alike: the identity step's reset,
which the copied check makes without a mask, gets the same mask); each arm entry carries
``goal_mask`` (mode, source, share) and every record the mask fields. Refused with
``--arm-plugin`` (the copied plugin builder has no mask seam).

``summary`` merges one or more ``check.json`` (one per scene process), refuses a mix of
renderers (PORT_PLAN D-9), and writes ``summary.json`` + ``README.md`` with Isaac's own
``check_baselines`` numbers beside each cell, read from ``$VSBENCH_ISAAC_CHECKS`` (default
``~/genvs_ws/vssim_out``; READ ONLY) - context, never pooled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
#: the copied vs-sim driver this module flies (reference code, frozen; drift-checked)
REFERENCE = REPO / "vsbench" / "scripts" / "check_baselines.py"
_REF_NAME = "_vsbench_reference_check_baselines"
#: U9's matrix (PORT_PLAN section 7): the arms and scenes the gate names
U9_ARMS = ("sift_ransac", "orb_ransac", "vitvs_dinov3", "artvs")
U9_SCENES = ("hollywood", "poster2", "poster3", "industrial1_drill", "cluttered2", "cluttered3")
#: the near verdict the twin writes (vs-sim's script writes none; see the module docstring)
NEAR_RULE = ("near_ok = final_mm < start_mm after --near-steps steps, no error (vs-sim "
             "check_baselines docstring: 'it must land closer than it started'; vs-sim's "
             "script computes no near verdict, the twin reads that sentence literally; 'by a "
             "lot' is not quantified there and is reported here as final_over_start, not "
             "thresholded)")
#: Isaac's own check runs (read only): where the context numbers come from
ISAAC_CHECKS = Path(os.environ.get("VSBENCH_ISAAC_CHECKS",
                                   str(Path.home() / "genvs_ws" / "vssim_out")))
#: identity + near (six scenes, 200 steps, 2026-09-17); far rows: hollywood + cluttered2
ISAAC_NEAR_RUN = "check1"
ISAAC_FAR_RUNS = ("check2", "check3")
PREVIEW_SCALE = 0.25


# --------------------------------------------------------------- the copied check
def reference():
    """The copied ``vsbench/scripts/check_baselines.py`` as a module (loaded once).

    The script puts ``vsbench/`` and ``vsbench/scripts/`` on ``sys.path`` (vs-sim's repository
    root and its ``scripts/``, for its ``import fly``); that is undone after the load, so
    nothing else in this process can import a top-level ``scene`` or ``camera`` from
    ``vsbench/`` by accident. Its ``fly`` stays in ``sys.modules`` (vs-sim's copied
    ``scripts/fly.py``; only its ``main`` uses it, which is never called here).
    """
    if _REF_NAME in sys.modules:
        return sys.modules[_REF_NAME]
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(_REF_NAME, REFERENCE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_REF_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(_REF_NAME, None)
        raise
    finally:
        sys.path[:] = saved
    return mod


class _Recorder:
    """While open, the reference module's ``run_episode`` is the copied one plus a capture:
    every record ``episode_check`` makes is kept (the check entry is a digest of it)."""

    def __init__(self, ref) -> None:
        self.ref = ref
        self.records: list[dict] = []
        self._orig = None

    def __enter__(self) -> _Recorder:
        self._orig = self.ref.run_episode

        def run_episode(*args, _orig=self._orig, **kw):
            rec = _orig(*args, **kw)
            self.records.append(rec)
            return rec

        self.ref.run_episode = run_episode
        return self

    def __exit__(self, *exc) -> bool:
        self.ref.run_episode = self._orig
        return False

    def take(self) -> dict | None:
        return self.records.pop() if self.records else None


def near_verdict(near: dict | None) -> dict:
    """:data:`NEAR_RULE` on one near entry: ``{"near_ok", "final_over_start"}``."""
    if not near or near.get("final_mm") is None or near.get("start_mm") is None:
        return {"near_ok": False, "final_over_start": None}
    start, final = float(near["start_mm"]), float(near["final_mm"])
    ratio = final / start if start > 0 else None
    return {"near_ok": bool(np.isfinite(final) and final < start),
            "final_over_start": None if ratio is None else round(ratio, 4)}


def _md5(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _save_png(path: Path, rgb, scale: float = 1.0) -> None:
    from PIL import Image

    im = Image.fromarray(np.asarray(rgb, np.uint8))
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.LANCZOS)
    im.save(path, optimize=True)


def _cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return round(ru.ru_utime + ru.ru_stime, 1)


def open_scene(name: str, args, specs):
    """The MuJoCo world for one scene (a hook: the fast tests hand in a stand-in)."""
    from vsbench.world.mujoco_scene import MujocoScene

    return MujocoScene(name, samples=args.samples, specs=specs, look=args.look)


def require_software_gl(scene, *, allow_gpu: bool = False) -> None:
    from vsbench.world import fly

    fly.require_software_gl(scene, allow_gpu=allow_gpu)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


# --------------------------------------------------------------------------- run
def build_parser() -> argparse.ArgumentParser:
    """vs-sim's flags, as its ``main`` defines them, plus the bridge's (last block)."""
    ap = argparse.ArgumentParser(prog="python -m vsbench.world.check_baselines",
                                 description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--scenes", nargs="+", default=list(U9_SCENES))
    ap.add_argument("--arms", nargs="+", default=None, help="default: every protocol arm")
    ap.add_argument("--near-steps", type=int, default=200)
    ap.add_argument("--far", type=int, default=0, help="fly the first N far starts to the cap")
    ap.add_argument("--far-set", default="capped30",
                    help="which protocol start set the far starts come from (default capped30; "
                         "wide100 has every roll)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--film", action="store_true",
                    help="a match video per episode: <out>/films/<scene>_<arm>_<start>/episode.mp4")
    ap.add_argument("--film-every", type=int, default=1, help="keep every n-th step")
    ap.add_argument("--perturb", default="none",
                    help="none | light | distractors | occluder | noise | combined | "
                         "swap:<id> | swap:@<live name> | ground:<name>, or a `+` chain - "
                         "applied after the goal capture of every near/far episode")
    ap.add_argument("--perturb-seed", type=int, default=20260918)
    ap.add_argument("--budget", action="append", default=[], metavar="ARM=STEPS",
                    help="one arm's own step budget for its near AND far starts (repeatable)")
    ap.add_argument("--outline-iou", action="store_true",
                    help="record the object-outline overlap with the goal picture at every "
                         "recorded stop and at the end (episode.outline_iou)")
    ap.add_argument("--telemetry", action="store_true",
                    help="the copied debug log per episode (<out>/films/<scene>_<arm>_<start>/)")
    ap.add_argument("--clean-film", action="store_true",
                    help="a clean half-size live film per episode; needs --telemetry")
    ap.add_argument("--arm-plugin", default=None, metavar="MODULE",
                    help="an importable module stating ARM_IDS, build() and describe() "
                         "(vsbench.world.method_arm flies the registered vsbench methods)")
    # the bridge's
    ap.add_argument("--mask", default="none",
                    help="none | objects | <file.png>: filter every arm's matches with a goal "
                         "mask (vsbench.world.masks; as fly run --mask)")
    ap.add_argument("--look", default=None, choices=["isaac_fit_v1", "raw"],
                    help="the picture (vsbench.world.look; default isaac_fit_v1)")
    ap.add_argument("--device", default="cpu", help="matcher.device for encoder arms")
    ap.add_argument("--samples", type=int, default=4, help="MuJoCo multisampling (colour only)")
    ap.add_argument("--frames", default="full", choices=["full", "preview", "none"],
                    help="full: vs-sim's goal frame, goal depth and first live frame; preview: "
                         "quarter-size PNGs, depth md5 only; none")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--allow-gpu", action="store_true",
                    help="lift the GPU refusals (never while the Isaac evaluation runs)")
    return ap


def validate_args(args, ref, protocol):
    """vs-sim's refusals, before any scene is built: ``--clean-film`` without
    ``--telemetry``, a ``--budget`` for an arm not flown, unknown / plugin arm ids (the
    copied resolver), an unknown far set. Returns ``(arm_ids, plugin, sources, budgets)``."""
    from vsbench.vs import benchmark

    if args.clean_film and not args.telemetry:
        raise SystemExit("--clean-film needs --telemetry (the debug viewer needs both)")
    from vsbench.world import masks

    if masks.parse(getattr(args, "mask", "none"))[0] != "none" and args.arm_plugin:
        raise SystemExit("--mask with --arm-plugin: the copied plugin builder has no goal-mask "
                         "seam; fly the methods with `python -m vsbench fly --mask`")
    if args.far_set not in benchmark.START_SETS:
        raise SystemExit(f"--far-set {args.far_set!r} is not one of {sorted(benchmark.START_SETS)}")
    budgets = ref.parse_budgets(args.budget)
    unknown_budget = sorted(set(budgets) - set(args.arms or []))
    if args.arms is not None and unknown_budget:
        raise SystemExit(f"--budget names arm(s) not flown: {unknown_budget}")
    arm_ids = args.arms or [e["id"] for e in protocol["arms"]]
    plugin = ref.load_arm_plugin(args.arm_plugin)
    sources = ref.resolve_arm_sources(protocol, arm_ids, plugin)
    return arm_ids, plugin, sources, budgets


def run(args) -> int:
    from vsbench.world import fly

    env = fly.station_guard(args.threads, allow_gpu=args.allow_gpu)   # before torch loads
    ref = reference()
    from vsbench.vs import benchmark
    from vsbench.vs.bench import perturb as perturbmod
    from vsbench.vs.scenes import load_scenes

    protocol = fly.load_protocol()
    arm_ids, plugin, sources, budgets = validate_args(args, ref, protocol)
    specs, _ = load_scenes()
    unknown = [s for s in args.scenes if s not in specs]
    if unknown:
        raise SystemExit(f"unknown scene(s) {unknown}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"vsbench_commit": fly.vsbench_commit(ignore=out),
                  "vssim_commit": fly.vssim_commit(),
                  "vsbench_commit_rule": fly.VSBENCH_COMMIT_RULE}
    # vs-sim's header, key for key, then the bridge's block
    results: dict = {"scenes": {}, "arms": arm_ids, "near_steps": args.near_steps,
                     "far": args.far, "far_set": args.far_set,
                     "perturb": args.perturb, "perturb_seed": args.perturb_seed,
                     "budgets": budgets, "outline_iou": bool(args.outline_iou),
                     **({"mask": args.mask} if args.mask != "none" else {}),
                     "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.telemetry or args.clean_film:
        results.update(telemetry=bool(args.telemetry), clean_film=bool(args.clean_film))
    if plugin is not None:
        results["arm_plugin"] = args.arm_plugin
    results["bridge"] = {"driver": "vsbench.world.check_baselines",
                         "reference": f"{REFERENCE.relative_to(REPO)} (the copied vs-sim driver, "
                                      "loaded as a library: identity_check, episode_check, "
                                      "near_start, write_table, the plugin hook)",
                         **provenance, "bridge_env": env, "look": args.look or "isaac_fit_v1",
                         "device": args.device, "samples": int(args.samples),
                         "frames": args.frames, "near_rule": NEAR_RULE,
                         "identity_v_max_m_s": float(ref.IDENTITY_V_MAX),
                         "identity_w_max_deg_s": float(np.degrees(ref.IDENTITY_W_MAX))}
    check_json = out / "check.json"
    records_path = out / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    rows = []
    t_run = time.monotonic()
    from vsbench.world import masks

    scene_gm = [masks.GoalMask("none")]      # the current scene's --mask, resolved on its goal

    def flush():
        results["bridge"]["cpu_s"] = _cpu_s()
        results["bridge"]["wall_s"] = round(time.monotonic() - t_run, 1)
        check_json.write_text(json.dumps(results, indent=1, default=_jsonable))

    def keep(rec, scene, arm_id, row):
        if rec is None:
            return
        rec = dict(rec, check_row=row, check_arm=arm_id, check_scene=scene.name, **provenance)
        scene_gm[0].apply(rec)
        rec["render_mode"] = scene.render_mode
        with records_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=_jsonable) + "\n")

    with _Recorder(ref) as recorder:
        for name in args.scenes:
            # the goal check reads the file the far set is cut from (vs-sim's rule)
            payload = benchmark.load_starts(name, benchmark.START_SETS[args.far_set]["file"])
            with open_scene(name, args, specs) as scene:
                require_software_gl(scene, allow_gpu=args.allow_gpu)   # before the first frame
                benchmark.check_scene_goal(scene, payload)
                scene.render_goal()                       # the one goal capture
                scene.reset(scene.goal_pose)
                rgb, depth, _ = scene.render()            # the cached goal frame
                _write_goal_frames(out, name, rgb, depth, args.frames)
                scene_gm[0] = gm = masks.resolve(args.mask, scene, depth)
                args._gm = gm
                g = scene.goal_pose
                capture = scene.goal_capture_record()
                results["scenes"][name] = {
                    "goal_distance_m": scene.working_distance, "goal_md5": scene.goal_frame_md5,
                    "goal_settle_diffs": scene.goal_settle["diff_prev"],
                    "goal": g.as_dict(), "arms": {},
                    # the bridge's
                    "render_mode": scene.render_mode,
                    "gl_renderer": getattr(scene.renderer, "gl_renderer", None),
                    "goal_depth_md5": _md5(depth), "look": capture.get("look"),
                    "look_curve_id": capture.get("look_curve_id"),
                    "look_curve_md5": capture.get("look_curve_md5"),
                    "goal_check": getattr(scene, "goal_check", None)}
                if gm.mode != "none":
                    results["scenes"][name]["goal_mask"] = {
                        "mode": gm.mode, "source": gm.source, "share": gm.share,
                        "mask_md5": gm.mask_md5, "resized_from": gm.resized_from}
                print(f"[check] {name}: goal d={scene.working_distance:.6f} m, "
                      f"{scene.render_mode} ({results['scenes'][name]['gl_renderer']}), look "
                      f"{capture.get('look')}, goal md5 {scene.goal_frame_md5}", flush=True)
                near = ref.near_start(scene)
                far = benchmark.select_starts(name, args.far_set, limit=args.far) if args.far else []
                pert = perturbmod.build(args.perturb, seed=args.perturb_seed, spec=specs[name])
                for arm_id in arm_ids:
                    rows.append(_check_arm(args, ref, fly, protocol, scene, specs[name], arm_id,
                                           sources, plugin, budgets, near, far, pert, out,
                                           arm_ids, results, recorder, keep))
                    flush()
            ref.write_table(rows, out)     # after every scene: a crash keeps what was flown
    flush()
    return 0


def _write_goal_frames(out: Path, name: str, rgb, depth, frames: str) -> None:
    if frames == "full":
        _save_png(out / f"{name}_goal.png", rgb)
        np.save(out / f"{name}_goal_depth.npy", depth)
    elif frames == "preview":
        _save_png(out / f"{name}_goal_preview.png", rgb, PREVIEW_SCALE)


def _check_arm(args, ref, fly, protocol, scene, spec, arm_id, sources, plugin, budgets, near, far,
               pert, out, arm_ids, results, recorder, keep):
    """vs-sim's per-arm block of ``main``, on this world: build, identity, near, far."""
    name = scene.name
    t0 = time.time()
    described = None
    try:
        if sources[arm_id] == "plugin":
            arm, cfg, described = ref.build_plugin_arm(plugin, arm_id, intrinsics=scene.intrinsics,
                                                       scene_name=name, scene_spec=spec)
        else:
            from vsbench.world import masks

            gm = getattr(args, "_gm", None) or masks.GoalMask("none")
            cell, mask_over = masks.arm_cell(protocol, arm_id, gm)
            cfg = fly.arm_config(protocol, cell, device=args.device,
                                 goal_mask="objects" if mask_over else None)
            arm = fly.build(protocol, cell, cfg, intrinsics=scene.intrinsics, scene_name=name,
                            scene_spec=spec)
            masks.attach_goal_mask(arm, gm)
            if gm.mode != "none":
                described = {"protocol_cell": cell, "goal_mask": gm.mode,
                             "goal_mask_source": gm.source}
    except Exception as exc:  # noqa: BLE001 - one broken arm must not end the night
        print(f"[check] {name} x {arm_id}: BUILD FAILED: {exc!r}", flush=True)
        results["scenes"][name]["arms"][arm_id] = {"build_error": repr(exc),
                                                   "render_mode": scene.render_mode}
        return (name, arm_id, None, None, None)
    entry: dict = {"label": arm.name}
    if described is not None and "goal_mask" in described:
        entry["goal_mask"] = described
    elif described is not None:
        entry["plugin"] = described
    roi_prompt = (getattr(arm, "record_fields", dict)() or {}).get("roi_prompt")
    if roi_prompt is not None:
        entry["roi_prompt"] = roi_prompt
    try:
        entry["identity"] = ref.identity_check(scene, arm)

        def film_dir(start_id):
            return (out / "films" / f"{name}_{arm_id}_{start_id}") if args.film else None

        def debug_dir(start_id):
            return (out / "films" / f"{name}_{arm_id}_{start_id}") if args.telemetry else None

        budget = budgets.get(arm_id)
        first = None
        if arm_id == arm_ids[0] and args.frames == "full":
            first = out / f"{name}_{arm_id}_near_first.png"
        entry["near"] = ref.episode_check(
            scene, arm, cfg, near, budget or args.near_steps, save_first=first,
            film_dir=film_dir(near["start_id"]), film_every=args.film_every, perturbation=pert,
            outline_iou=args.outline_iou, arm_id=arm_id, debug_dir=debug_dir(near["start_id"]),
            clean_film=args.clean_film)
        keep(recorder.take(), scene, arm_id, "near")
        if arm_id == arm_ids[0] and args.frames == "preview":
            from vsbench.vs.scene import Pose

            scene.reset(Pose.from_start(near))
            live, _d, _p = scene.render()
            _save_png(out / f"{name}_{arm_id}_near_first_preview.png", live, PREVIEW_SCALE)
        entry["far"] = []
        for s in far:
            entry["far"].append(ref.episode_check(
                scene, arm, cfg, s, budget or cfg["episode"]["max_iter"],
                film_dir=film_dir(s["start_id"]), film_every=args.film_every, perturbation=pert,
                outline_iou=args.outline_iou, arm_id=arm_id, debug_dir=debug_dir(s["start_id"]),
                clean_film=args.clean_film))
            keep(recorder.take(), scene, arm_id, "far")
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        entry["error"] = repr(exc)
    finally:
        close = getattr(arm, "close", None)
        if close is not None:
            close()
    entry["wall_s"] = time.time() - t0
    # the bridge's
    entry["render_mode"] = scene.render_mode
    entry.update(near_verdict(entry.get("near")) if "near" in entry else
                 {"near_ok": False, "final_over_start": None})
    results["scenes"][name]["arms"][arm_id] = entry
    idn, nr = entry.get("identity"), entry.get("near")
    print(f"[check] {name} x {arm_id}: identity "
          f"{'OK' if idn and idn['ok'] else 'FAIL'} "
          f"(n={idn['n_matched'] if idn else '?'}, "
          f"|v|={ref.fmt(idn['v_norm_m_s'] if idn else None, 6)} m/s, "
          f"|w|={ref.fmt(idn['w_norm_deg_s'] if idn else None, 4)} deg/s); near "
          f"{ref.near_text(nr)} [{'closer' if entry['near_ok'] else 'NOT CLOSER'}]"
          f"{ref.far_text(entry.get('far'))}"
          f"{' ERROR ' + entry['error'] if entry.get('error') else ''}"
          f" [{entry['wall_s']:.0f} s]", flush=True)
    return (name, arm_id, idn, nr, entry.get("far"))


# ----------------------------------------------------------------------- summary
def load_checks(paths) -> dict:
    """Merge ``check.json`` files (a file or its directory each) into one ``{"scenes": ...,
    "runs": [...]}``; refuses a mix of renderers (D-9) and a scene x arm flown twice."""
    merged: dict = {"scenes": {}, "runs": []}
    modes = set()
    for p in paths:
        p = Path(p)
        f = p / "check.json" if p.is_dir() else p
        if not f.exists():
            continue
        doc = json.loads(f.read_text())
        merged["runs"].append({"file": str(f), **{k: doc.get(k) for k in (
            "arms", "near_steps", "far", "far_set", "perturb", "started_utc")},
            "bridge": doc.get("bridge")})
        for name, sc in doc.get("scenes", {}).items():
            modes.add(sc.get("render_mode"))
            for arm_id, e in sc.get("arms", {}).items():
                modes.add(e.get("render_mode"))
            have = merged["scenes"].setdefault(name, {**{k: v for k, v in sc.items() if k != "arms"},
                                                      "arms": {}})
            clash = set(have["arms"]) & set(sc.get("arms", {}))
            if clash:
                raise SystemExit(f"{name}: arms {sorted(clash)} flown twice ({f})")
            have["arms"].update(sc.get("arms", {}))
    if len(modes) > 1:
        raise SystemExit(f"refusing to merge mixed renderers {sorted(map(str, modes))} "
                         "(PORT_PLAN D-9: MuJoCo and Isaac numbers are never pooled)")
    merged["render_mode"] = next(iter(modes)) if modes else None
    return merged


def isaac_context(root: Path = ISAAC_CHECKS) -> dict:
    """``{(scene, arm): {...}}`` from Isaac's own check runs (read only): the identity and
    near row of :data:`ISAAC_NEAR_RUN`, the first far start of :data:`ISAAC_FAR_RUNS` (the
    first run that flew the cell). Missing files give an empty dict."""
    out: dict = {}
    nf = Path(root) / ISAAC_NEAR_RUN / "check.json"
    if nf.exists():
        doc = json.loads(nf.read_text())
        for name, sc in doc.get("scenes", {}).items():
            for arm_id, e in sc.get("arms", {}).items():
                idn, nr = e.get("identity") or {}, e.get("near") or {}
                out[(name, arm_id)] = {
                    "run": f"{ISAAC_NEAR_RUN} ({doc.get('started_utc')})",
                    "identity_ok": idn.get("ok"), "identity_n_matched": idn.get("n_matched"),
                    "identity_v_norm_m_s": idn.get("v_norm_m_s"),
                    "identity_w_norm_deg_s": idn.get("w_norm_deg_s"),
                    "near_start_mm": nr.get("start_mm"), "near_final_mm": nr.get("final_mm"),
                    "near_final_deg": nr.get("final_deg"), "near_steps": nr.get("steps"),
                    "near_terminated": nr.get("terminated"),
                    **({} if not nr else near_verdict(nr))}
    for run in ISAAC_FAR_RUNS:
        ff = Path(root) / run / "check.json"
        if not ff.exists():
            continue
        doc = json.loads(ff.read_text())
        for name, sc in doc.get("scenes", {}).items():
            for arm_id, e in sc.get("arms", {}).items():
                far = e.get("far") or []
                if not far:
                    continue
                c = out.setdefault((name, arm_id), {})
                if "far_final_mm" in c:
                    continue
                f0 = far[0]
                c.update({"far_run": f"{run} ({doc.get('started_utc')})",
                          "far_start_id": f0.get("start_id"), "far_start_mm": f0.get("start_mm"),
                          "far_final_mm": f0.get("final_mm"), "far_final_deg": f0.get("final_deg"),
                          "far_steps": f0.get("steps"), "far_terminated": f0.get("terminated")})
    return out


def summarize(merged: dict, isaac: dict, *, arms=U9_ARMS, scenes=U9_SCENES) -> dict:
    """One row per (scene, arm) of the U9 matrix: our verdicts and landings, Isaac's beside."""
    rows = []
    for name in scenes:
        sc = merged["scenes"].get(name)
        for arm_id in arms:
            e = (sc or {}).get("arms", {}).get(arm_id)
            row: dict = {"scene": name, "arm": arm_id, "flown": e is not None}
            if e is not None:
                idn, nr, far = e.get("identity") or {}, e.get("near") or {}, e.get("far") or []
                row.update({
                    "label": e.get("label"), "build_error": e.get("build_error"),
                    "error": e.get("error"), "render_mode": e.get("render_mode"),
                    "identity_ok": bool(idn.get("ok", False)),
                    "identity_n_matched": idn.get("n_matched"),
                    "identity_v_norm_m_s": idn.get("v_norm_m_s"),
                    "identity_w_norm_deg_s": idn.get("w_norm_deg_s"),
                    "identity_zero_displacement_share": idn.get("zero_displacement_share"),
                    "near_start_id": nr.get("start_id"),
                    "near_start_mm": nr.get("start_mm"), "near_start_deg": nr.get("start_deg"),
                    "near_final_mm": nr.get("final_mm"), "near_final_deg": nr.get("final_deg"),
                    "near_steps": nr.get("steps"), "near_terminated": nr.get("terminated"),
                    "near_stop_step": nr.get("stop_step"), "near_stop_mm": nr.get("stop_mm"),
                    "near_sec_per_step": nr.get("sec_per_step"),
                    "near_mean_n_matches": nr.get("mean_n_matches"),
                    "near_ok": bool(e.get("near_ok", False)),
                    "final_over_start": e.get("final_over_start"),
                    "far": [{k: f.get(k) for k in ("start_id", "start_mm", "start_deg", "final_mm",
                                                   "final_deg", "steps", "terminated", "stop_step",
                                                   "stop_mm", "sec_per_step", "success")}
                            for f in far],
                    "wall_s": e.get("wall_s")})
            row["isaac"] = isaac.get((name, arm_id))
            rows.append(row)
    flown = [r for r in rows if r["flown"]]
    complete = len(flown) == len(rows)
    return {"rows": rows, "render_mode": merged.get("render_mode"), "n_cells": len(rows),
            "n_flown": len(flown), "complete": complete,
            "identity_all_green": complete and all(r["identity_ok"] for r in flown),
            "near_all_green": complete and all(r["near_ok"] for r in flown),
            "identity_fail": [f"{r['scene']}/{r['arm']}" for r in flown if not r["identity_ok"]],
            "near_fail": [f"{r['scene']}/{r['arm']}" for r in flown if not r["near_ok"]],
            "near_rule": NEAR_RULE, "runs": merged.get("runs"),
            "isaac_source": {"root": str(ISAAC_CHECKS), "near_run": ISAAC_NEAR_RUN,
                             "far_runs": list(ISAAC_FAR_RUNS),
                             "note": "Isaac's own check_baselines runs, read only; older code than "
                                     "the copy's pin for some arms; context, never pooled (D-9)"}}


def _f(x, nd=2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def table_md(summary: dict) -> str:
    lines = ["| scene | arm | identity (n / \\|v\\| m/s / \\|w\\| deg/s) | near start -> final mm, "
             "deg (MuJoCo) | near | Isaac near final mm, deg [closer] | capped30 start -> final mm "
             "(MuJoCo) | Isaac capped30 start -> final mm |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in summary["rows"]:
        isa = r.get("isaac") or {}
        isaac_near = ("-" if isa.get("near_final_mm") is None else
                      f"{_f(isa['near_final_mm'], 3)}, {_f(isa['near_final_deg'], 3)} "
                      f"[{'yes' if isa.get('near_ok') else 'no'}]")
        isaac_far = ("-" if isa.get("far_final_mm") is None else
                     f"{isa['far_start_id']}: {_f(isa['far_start_mm'], 0)} -> "
                     f"{_f(isa['far_final_mm'], 1)}")
        if not r["flown"]:
            lines.append(f"| {r['scene']} | {r['arm']} | not flown | | | {isaac_near} | | {isaac_far} |")
            continue
        if r.get("build_error"):
            lines.append(f"| {r['scene']} | {r['arm']} | BUILD FAILED | | | {isaac_near} | | {isaac_far} |")
            continue
        idn = (f"{'OK' if r['identity_ok'] else 'FAIL'}: {r['identity_n_matched']} / "
               f"{r['identity_v_norm_m_s']:.1e} / {r['identity_w_norm_deg_s']:.1e}"
               if r.get("identity_v_norm_m_s") is not None else "ERROR")
        near = ("-" if r.get("near_final_mm") is None else
                f"{_f(r['near_start_mm'])} -> {_f(r['near_final_mm'], 3)}, "
                f"{_f(r['near_start_deg'])} -> {_f(r['near_final_deg'], 3)}")
        far = ", ".join(f"{f['start_id']}: {_f(f['start_mm'], 0)} -> {_f(f['final_mm'], 1)}"
                        + ("" if f.get("terminated") == "max_iter" else f" ({f.get('terminated')})")
                        for f in r["far"]) or "-"
        lines.append(f"| {r['scene']} | {r['arm']} | {idn} | {near} | "
                     f"{'closer' if r['near_ok'] else 'NOT closer'} | {isaac_near} | {far} | "
                     f"{isaac_far} |")
    return "\n".join(lines) + "\n"


def summary_cmd(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m vsbench.world.check_baselines summary")
    ap.add_argument("--runs", nargs="+", required=True, help="check.json files or their directories")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--isaac", type=Path, default=ISAAC_CHECKS,
                    help="Isaac's check runs, read only (default $VSBENCH_ISAAC_CHECKS)")
    args = ap.parse_args(argv)
    merged = load_checks(args.runs)
    summary = summarize(merged, isaac_context(args.isaac))
    summary["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable) + "\n")
    md = table_md(summary)
    (args.out / "summary_table.md").write_text(md)
    print(md)
    print(f"identity all green: {summary['identity_all_green']} {summary['identity_fail']}\n"
          f"near all green: {summary['near_all_green']} {summary['near_fail']}\n"
          f"cells flown {summary['n_flown']} / {summary['n_cells']}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["summary"]:
        return summary_cmd(argv[1:])
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
