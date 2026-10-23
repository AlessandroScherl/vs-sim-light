#!/usr/bin/env python3
# vs-sim copy: scripts/fly.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""`fly` - plan a baseline campaign, fly it, summarise it.

    python3   scripts/fly.py plan    --scenes hollywood cluttered2 --arms vitvs_dinov3 cns_v1
    $ISAAC_PY scripts/fly.py run     --scene hollywood --arms vitvs_dinov3 artvs \\
                                     --start-set capped30 --out /workspace/vssim_runs
    python3   scripts/fly.py summary --out /workspace/vssim_runs

`plan` prints the campaign's shape and every arm's derived label without loading a
backbone (the dry-run gate). `run` boots Isaac, builds one scene, checks its goal
against the frozen start file, captures the goal view once, and flies every arm of
the list on the chosen start set through `vsbench.vs.bench.runner` - one goal picture
served to all arms, so their contrast is paired. `summary` reads the records back and
prints, per scene x arm, how many starts reached under each success rule and how close
they landed, at the stop and at the cap.

Nothing here holds a parameter. Every number comes from `configs/protocol.yaml`, the
arm file it names, or the frozen start file `vsbench.vs.benchmark` pins by md5.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsbench.vs import benchmark  # noqa: E402
from vsbench.vs.bench import config as cfgmod  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
ARMS = CONFIGS / "arms"

#: Fallback per-step cost for the budget line, measured on an RTX 4090 for a
#: 24-landmark DINOv3 arm (0.115 s/step). A plan is a floor.
SEC_PER_STEP = 0.115


# ------------------------------------------------------------------ the protocol
def load_protocol(path=None) -> dict:
    """Load `configs/protocol.yaml` (or another campaign spec) under the strict loader."""
    doc = cfgmod.load_document(Path(path) if path else CONFIGS / "protocol.yaml")
    if doc.get("schema") != "vssim-protocol-1":
        raise SystemExit(f"{path or 'configs/protocol.yaml'}: not a vssim-protocol-1 file")
    for key in ("scene", "start_sets", "arms", "episode", "scoring"):
        if key not in doc:
            raise SystemExit(f"protocol has no `{key}:` block")
    return doc


def select_starts(protocol: dict, scene: str, set_name: str, limit: int | None = None) -> list:
    """The starts of one protocol start set, in the order the campaign flies them."""
    if set_name not in protocol["start_sets"]:
        raise SystemExit(f"start set {set_name!r} is not in the protocol "
                         f"(have {', '.join(protocol['start_sets'])})")
    return benchmark.select_starts(scene, set_name, limit=limit)


def _merge(base: dict, over: dict) -> dict:
    """Deep-merge `over` into a copy of `base` (mappings recurse, everything else wins)."""
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


def arm_config(protocol: dict, arm_id: str) -> dict:
    """The resolved config of one arm: its own file, deep-merged with its deltas.

    The protocol never assembles an arm by hand - it names the file and the deltas
    that make it that cell, so a cell and its config file can never say different
    things. The id is NOT written in as `arm.name`: the label stays DERIVED from the
    settings; the id names the run directory instead.
    """
    entry = arm_entry(protocol, arm_id)
    cfg = _merge(cfgmod.load(ARMS / entry["base"]), entry.get("overrides") or {})
    return cfgmod.resolve(cfg, source=f"arms/{entry['base']} + protocol.yaml[{arm_id}]")


def build(protocol: dict, arm_id: str, *, intrinsics, scene_name: str, scene_spec=None,
          backbone=None, **overrides):
    """Build one arm, handing it only what its config actually asks for.

    `scene_spec` goes only to a cell whose `needs` names it (`ransac: scene_default`
    refuses to guess a model); `backbone` only to a variant that has an encoder, so one
    loaded encoder can be shared across ViT cells without special-casing the rest.
    """
    from vsbench.vs.bench.arms import build_arm  # noqa: PLC0415 - keeps `plan` import-light

    cfg = arm_config(protocol, arm_id)
    needs = set(arm_entry(protocol, arm_id).get("needs") or [])
    if "scene_spec" in needs:
        if scene_spec is None:
            raise SystemExit(f"{arm_id}: `ransac: scene_default` has no safe default - "
                             "pass the scene's SceneSpec (vsbench.vs.scenes.load_scenes)")
        overrides["scene_spec"] = scene_spec
    if backbone is not None and cfg["matcher"]["variant"] in cfgmod.BACKBONE_VARIANTS:
        overrides["backbone"] = backbone
    return build_arm(cfg, intrinsics=intrinsics, scene=scene_name, **overrides), cfg


# --------------------------------------------------------------------------- plan
def cmd_plan(args) -> int:
    """Print the campaign's shape and its honest time budget. No Isaac, no GPU."""
    from vsbench.vs.bench.arms import arm_label  # noqa: PLC0415

    protocol = load_protocol(args.protocol)
    print(f"[plan] goal capture: {cfgmod.goal_render_warmup(protocol)} render(s), "
          f"keep {cfgmod.goal_capture_keep(protocol)}")
    total_eps = total_steps = 0
    for scene in args.scenes:
        picked = select_starts(protocol, scene, args.start_set, args.limit)
        for arm_id in args.arms:
            cfg = arm_config(protocol, arm_id)
            cap = int((cfg.get("episode") or {}).get("max_iter", 600))
            repeats = int(arm_entry(protocol, arm_id).get("repeats", 1))
            total_eps += len(picked) * repeats
            total_steps += len(picked) * cap * repeats
            print(f"[plan] {scene} x {arm_id}: {len(picked)} starts x cap {cap}"
                  f"{f' x {repeats} repeats' if repeats > 1 else ''} "
                  f"= {len(picked) * cap * repeats} steps  (seed {cfg['seed']})")
            print(f"[plan] label {arm_id} = {arm_label(cfg)}")
        print(f"[plan] {scene} start ids: {','.join(s['start_id'] for s in picked)}")
    hours = total_steps * args.sec_per_step / 3600.0
    print(f"[plan] {len(args.scenes)} scenes x {len(args.arms)} arms x {args.limit or 'all'} "
          f"starts = {total_eps} episodes")
    print(f"[plan] {total_steps} control steps; at {args.sec_per_step} s/step that is "
          f"{hours:.2f} h of control, plus Isaac boots")
    return 0


# ---------------------------------------------------------------------------- run
def cmd_run(args) -> int:
    """Fly one scene x N arms x the chosen starts (Isaac)."""
    from vsbench.vs.bench.runner import open_run, run_episode  # noqa: PLC0415
    from vsbench.vs.isaac_scene import IsaacScene, nonzero_exit_on_error  # noqa: PLC0415
    from vsbench.vs.scenes import load_scenes  # noqa: PLC0415

    protocol = load_protocol(args.protocol)
    specs, _ = load_scenes()
    if args.scene not in benchmark.SCENES:
        raise SystemExit(f"{args.scene!r} is not a benchmark scene ({', '.join(benchmark.SCENES)})")
    picked = select_starts(protocol, args.scene, args.start_set, args.limit)
    payload = benchmark.load_starts(args.scene, benchmark.START_SETS[args.start_set]["file"])
    warmup = cfgmod.goal_render_warmup(protocol)
    out_root = Path(args.out)
    from vsbench.vs.bench import perturb as perturbmod  # noqa: PLC0415

    pert = perturbmod.build(getattr(args, "perturb", "none") or "none",
                            seed=int(getattr(args, "perturb_seed", 20260918)),
                            spec=specs[args.scene])
    with (IsaacScene(goal_render_warmup=warmup,
                     goal_capture_keep=cfgmod.goal_capture_keep(protocol)) as scene,
          nonzero_exit_on_error("fly run")):
        scene.build_eval_scene(specs[args.scene])
        benchmark.check_scene_goal(scene, payload)
        # The goal is captured HERE, once, at a defined point of the run, and served
        # to every arm from the cache: their contrast is paired on ONE picture.
        scene.render_goal()
        goal_capture = scene.goal_capture_record()
        print(f"[run] {args.scene}: goal distance {scene.working_distance:.6f} m "
              f"(file says {payload['goal']['distance_m']:.6f}), goal render warm-up "
              f"{warmup}, settle {goal_capture['goal_settle_series']}, goal md5 "
              f"{goal_capture['goal_frame_md5']}", flush=True)
        for arm_id in args.arms:
            repeats = int(arm_entry(protocol, arm_id).get("repeats", 1))
            for rep in range(1, repeats + 1):
                arm, cfg = build(protocol, arm_id, intrinsics=scene.intrinsics,
                                 scene_name=args.scene, scene_spec=specs[args.scene])
                tag = f"{args.scene}_{arm_id}" + (f"_r{rep}" if repeats > 1 else "")
                out = out_root / tag
                out.mkdir(parents=True, exist_ok=True)
                with open_run(tag, cfg, out_root=out, telemetry=args.telemetry,
                              allow_dirty=args.allow_dirty, allow_no_git=args.allow_no_git,
                              extra=goal_capture) as sink:
                    for i, start in enumerate(picked, 1):
                        steps: list = [] if args.telemetry else None
                        rec = run_episode(scene, arm, start, cfg, telemetry=steps,
                                          starts_payload=payload, perturbation=pert)
                        rec["cell_id"] = tag
                        sink.write(rec, steps)
                        print(f"[run] {tag} {i}/{len(picked)} {start['start_id']}: "
                              f"{rec['start_translation_mm']:.1f} -> "
                              f"{rec['final_translation_mm']:.3f} mm, "
                              f"{rec['final_rotation_deg']:.3f} deg, "
                              f"{rec['terminated_reason']}, "
                              f"{rec['sec_per_step']:.3f} s/step", flush=True)
                # a flat copy of the records beside the run directory
                (out_root / f"{tag}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in sink.records), encoding="utf-8")
                seen = {r.get("goal_frame_md5") for r in sink.records}
                if seen != {goal_capture["goal_frame_md5"]}:
                    raise RuntimeError(
                        f"{tag}: the run's arms do not share one goal picture - records "
                        f"carry {sorted(m for m in seen if m)}, the run captured "
                        f"{goal_capture['goal_frame_md5']}")
                close = getattr(arm, "close", None)   # free the encoder before the next cell
                if close is not None:
                    close()
        print(f"[run] goal picture pinned across {len(args.arms)} arm(s): "
              f"{goal_capture['goal_frame_md5']}", flush=True)
    return 0


# ------------------------------------------------------------------------ summary
def cmd_summary(args) -> int:
    """Per scene x arm: reached counts and landing medians at the stop and at the cap."""
    from vsbench.vs.bench import scoring  # noqa: PLC0415

    protocol = load_protocol(args.protocol)
    rules = list(protocol["scoring"]["success_rules"])
    files = sorted(Path(args.out).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl records under {args.out}")
    print(f"{'cell':34s} {'n':>3s} " + " ".join(f"{r[:14]:>14s}" for r in rules)
          + f"  {'stop mm':>8s} {'stop deg':>8s} {'cap mm':>8s} {'cap deg':>8s}")
    for path in files:
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not recs:
            continue
        wd = recs[0].get("working_distance_m")
        cols = []
        for rule in rules:
            stop = scoring.landing_stats(recs, rule=rule, at="stop", working_distance_m=wd)
            cap = scoring.landing_stats(recs, rule=rule, at="cap", working_distance_m=wd)
            cols.append(f"{stop['reached']:>3d}/{cap['reached']:<3d}s/c   ")
        stop = scoring.landing_stats(recs, rule=rules[0], at="stop", working_distance_m=wd)
        cap = scoring.landing_stats(recs, rule=rules[0], at="cap", working_distance_m=wd)

        def fmt(v):
            return "   -    " if v is None else f"{v:8.3f}"

        print(f"{path.stem:34s} {len(recs):>3d} " + " ".join(cols)
              + f"  {fmt(stop['translation_mm_med'])} {fmt(stop['rotation_deg_med'])} "
              f"{fmt(cap['translation_mm_med'])} {fmt(cap['rotation_deg_med'])}")
    print(f"\nreached counts are stop-scored / cap-scored under each rule; the medians are "
          f"over the episodes that reached under {rules[0]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--protocol", default=None, help="default: configs/protocol.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="print the shape and the budget; run nothing")
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--start-set", default="capped30")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sec-per-step", type=float, default=SEC_PER_STEP)
    p.set_defaults(fn=cmd_plan)

    r = sub.add_parser("run", help="fly one scene x N arms ($ISAAC_PY)")
    r.add_argument("--scene", required=True)
    r.add_argument("--arms", nargs="+", required=True)
    r.add_argument("--start-set", default="capped30")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--telemetry", action="store_true")
    r.add_argument("--allow-dirty", action="store_true")
    r.add_argument("--allow-no-git", action="store_true",
                   help="a container without git: run.json records git_commit=null")
    r.add_argument("--perturb", default="none",
                   help="none | light | distractors | occluder | noise | combined "
                        "(vsbench.vs.bench.perturb), applied after every episode's goal capture")
    r.add_argument("--perturb-seed", type=int, default=20260918)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("summary", help="tabulate the records of a run directory")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_summary)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
