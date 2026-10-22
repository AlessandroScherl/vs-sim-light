#!/usr/bin/env python3
# vs-sim copy: scripts/check_baselines.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Does each baseline actually work on the rig? Three checks per scene x arm, in Isaac.

    export VSSIM_ASSET_ROOT=/workspace/genvs-sim CNS_ROOT=/workspace/CNS HF_HUB_OFFLINE=1
    $ISAAC_PY scripts/check_baselines.py --out /workspace/vssim_out/check \\
        --scenes hollywood cluttered2 --arms vitvs_dinov3 artvs cns_v1 --near-steps 200

1. **Identity** - the camera is placed AT the goal and handed the goal picture as the
   live frame. A working method must find its own points again (every match with zero
   displacement) and command (almost) no motion. A method that moves here is broken
   before any start is flown.
2. **Near** - a 3-8 mm / <= 3 deg start (the benchmark's own `near` sampler, seed
   20260902, first draw) flown for `--near-steps` steps: it must land closer than it
   started, by a lot.
3. **Far** (`--far N`) - the first N starts of `capped30` (tens of centimetres, up to
   90 deg roll) flown to the cap: the real benchmark cell, for a smoke of the whole
   chain; `fly.py run` is the campaign driver, this is the smoke. `--far-set` picks
   another pinned set (`wide100`; the instance suite's `simple5`, which reads the
   scene's own `<scene>_starts_simple.json`).

The instance suite (2026-09-23) adds three switches: `--perturb "swap:@<live
name>+ground:concrete"` flies one of the scene's named live objects on the concrete
floor, and every record then also carries the landing against the size-matched pose
(`*_matched`, `success_matched`); `--outline-iou` records the object-outline overlap with
the goal picture at every recorded stop and at the end; `--budget ARM=STEPS` gives one arm
its own step budget for the near and the far starts (DINOBot: 30 jumps).

The debug telemetry (owner, 2026-09-30) adds two switches, both off by default so an old
command writes exactly the old files: `--telemetry` writes, per episode, a step-by-step
record (`<out>/films/<scene>_<arm>_<start>/telemetry.jsonl.gz`: poses, the distance to the
goal, the size-matched and the depth-true pose, the matched pairs with a right/wrong truth
check, the commanded motion, the stop rules; vsbench.vs.bench.debuglog), the goal picture and
its mask, adds the depth-true reads to every record (`*_depth_true`, `success_depth_true`)
and, with `--film`, films with the method's name, the three distances, the landing marked
and a final frame; `--clean-film` (needs `--telemetry`) adds the un-drawn half-size live
film that `scripts/debug_view.py` draws its debug view on.

The external-arm hook (2026-10-11, owner ruling pending; `docs/controller_spec_2026-10-11.md`
section 1.2 at the workspace root): `--arm-plugin MODULE` lets an arm that lives in another
repository fly this exact check. `MODULE` is importable from `PYTHONPATH` and states
`ARM_IDS`, `build(arm_id, *, intrinsics, scene_name, scene_spec) -> (arm, config)` and
`describe(arm_id) -> {"module", "commit", "config_source", "label"}`. An id of `--arms` is a
protocol arm first, else a plugin arm, else refused before Isaac boots. A plugin arm must call
itself `external:...` and be of kind `external`; its check entry carries `plugin`
(`describe`), `check.json` names the module. Arm ids are `[A-Za-z0-9_]+` (enforced): the
chain names directories after them and `dev_compare.py --run LABEL=DIR:ARM` splits at the
last colon. An arm that declares `requires_live_mask` is handed the goal's objects mask as
its live mask in the identity check (the runner hands it one every step).

Writes `<out>/check.json` (every number), `<out>/check.md` (the table), the goal
frame of every scene (`<scene>_goal.png/.npy`) and the first live frame of every near
start. Nothing here holds a parameter: arms come from `configs/protocol.yaml` (or the plugin
named on the command line, which owns its own configs).
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fly  # noqa: E402

from vsbench.vs import benchmark  # noqa: E402
from vsbench.vs import starts as startsmod  # noqa: E402
from vsbench.vs.bench import config as cfgmod  # noqa: E402
from vsbench.vs.bench.runner import run_episode  # noqa: E402
from vsbench.vs.scene import Pose  # noqa: E402

#: A method at the goal picture must command less than this: 0.1 mm/s and 0.01 deg/s
#: at gain 2 means a residual well under one pixel of feature error.
IDENTITY_V_MAX = 1e-4
IDENTITY_W_MAX = np.deg2rad(0.01)

#: What an arm id may be: letters, digits and underscores - no colon (dev_compare's
#: `LABEL=DIR:ARM` splits at the last one) and no slash (film directories carry the id).
ARM_ID_PATTERN = re.compile(r"[A-Za-z0-9_]+")
#: What a plugin's arm must call itself and be: never mistakable for a benchmark arm.
PLUGIN_LABEL_PREFIX = "external:"
PLUGIN_ARM_KIND = "external"


def load_arm_plugin(module_name: str | None):
    """``--arm-plugin MODULE`` -> the imported module, or ``None`` without one. Refuses a
    module that lacks ``ARM_IDS`` / ``build`` / ``describe`` or states an id that is not
    :data:`ARM_ID_PATTERN`."""
    if not module_name:
        return None
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"--arm-plugin {module_name!r} is not importable from PYTHONPATH "
                         f"({exc})") from exc
    missing = [a for a in ("ARM_IDS", "build", "describe") if not hasattr(mod, a)]
    if missing:
        raise SystemExit(f"--arm-plugin {module_name!r} lacks {missing}: a plugin states "
                         "ARM_IDS, build(arm_id, ...) and describe(arm_id)")
    bad = [i for i in mod.ARM_IDS if not ARM_ID_PATTERN.fullmatch(str(i))]
    if bad:
        raise SystemExit(f"--arm-plugin {module_name!r}: arm ids {bad} are not [A-Za-z0-9_]+")
    return mod


def resolve_arm_sources(protocol: dict, arm_ids, plugin=None) -> dict[str, str]:
    """``{arm_id: "protocol" | "plugin"}`` for every id of ``--arms`` - a protocol arm first,
    else a plugin arm, else ``SystemExit`` naming both lists. Refused too: an id that is not
    :data:`ARM_ID_PATTERN`, and a plugin that would shadow a protocol arm."""
    proto = [e["id"] for e in protocol["arms"]]
    plug = [str(i) for i in (plugin.ARM_IDS if plugin is not None else ())]
    clash = sorted(set(proto) & set(plug))
    if clash:
        raise SystemExit(f"--arm-plugin states {clash}, which are protocol arms: a plugin may "
                         "not shadow a benchmark arm")
    out: dict[str, str] = {}
    for arm_id in arm_ids:
        if not ARM_ID_PATTERN.fullmatch(str(arm_id)):
            raise SystemExit(f"arm id {arm_id!r} is not [A-Za-z0-9_]+ (no colon, no slash)")
        if arm_id in proto:
            out[arm_id] = "protocol"
        elif arm_id in plug:
            out[arm_id] = "plugin"
        else:
            raise SystemExit(f"unknown arm {arm_id!r}; the protocol has {', '.join(proto)}"
                             + (f"; the plugin {plugin.__name__} has {', '.join(plug)}"
                                if plugin is not None else "; no --arm-plugin given"))
    return out


def build_plugin_arm(plugin, arm_id: str, *, intrinsics, scene_name: str, scene_spec=None):
    """One plugin arm: ``plugin.build(...)``, then the refusals - the arm must call itself
    ``external:...`` and be of kind ``external``. Returns ``(arm, config, describe)``."""
    arm, cfg = plugin.build(arm_id, intrinsics=intrinsics, scene_name=scene_name,
                            scene_spec=scene_spec)
    name, kind = str(getattr(arm, "name", "")), getattr(arm, "kind", None)
    if not name.startswith(PLUGIN_LABEL_PREFIX) or kind != PLUGIN_ARM_KIND:
        close = getattr(arm, "close", None)
        if close is not None:
            close()
        raise ValueError(f"plugin arm {arm_id!r} is named {name!r} of kind {kind!r}: a plugin "
                         f"arm must be named '{PLUGIN_LABEL_PREFIX}...' and be of kind "
                         f"'{PLUGIN_ARM_KIND}', so no table can read it as a benchmark arm")
    return arm, cfg, dict(plugin.describe(arm_id))


def save_png(path: Path, rgb) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(rgb, np.uint8)).save(path)


def identity_check(scene, arm) -> dict:
    """The camera at the goal, the goal picture as the live frame: no motion expected.

    An arm whose matcher cannot run without the bench's goal mask (``requires_goal_mask``:
    the ART-VS development cell, 2026-10-10) is reset with the mask the runner would hand it
    (the goal objects' outline from the goal depth); an arm that asks for a live mask
    (``requires_live_mask``: an external arm, 2026-10-11) is reset with it AND handed it as
    the live mask - the live frame is the goal frame; every other arm exactly as before."""
    scene.reset(scene.goal_pose)
    rgb, depth, _ = scene.render()                 # the cached goal frame
    arm.set_episode(scene.name, "identity")
    wants_live = bool(getattr(arm, "requires_live_mask", False))
    if getattr(getattr(arm, "matcher", None), "requires_goal_mask", False) or wants_live:
        from vsbench.vs.labels import object_mask_from_depth

        goal_mask = object_mask_from_depth(depth, scene.intrinsics, scene.goal_pose)
        arm.reset(rgb, depth, goal_mask)
    else:
        arm.reset(rgb, depth)
    if wants_live:
        twist, debug, matches = arm.step(rgb, depth, goal_mask)
    else:
        twist, debug, matches = arm.step(rgb, depth)
    v = twist.as_array()
    out = {
        "n_matches": int(debug.get("n_matches", 0)),
        "n_matched": int(debug.get("n_matched", debug.get("n_matches", 0))),
        "feature_error_norm": (None if not np.isfinite(debug.get("feature_error_norm", np.nan))
                               else float(debug["feature_error_norm"])),
        "v_norm_m_s": float(np.linalg.norm(v[:3])),
        "w_norm_deg_s": float(np.degrees(np.linalg.norm(v[3:]))),
        "match_ms": float(debug.get("match_ms", 0.0)),
        "step_ms": float(debug.get("step_ms", 0.0)),
    }
    pg = debug.get("pts_goal", debug.get("pts_desired"))
    pc = debug.get("pts_cur", debug.get("pts_current"))
    if pg is not None and pc is not None and len(pg):
        d = np.linalg.norm(np.asarray(pg, float) - np.asarray(pc, float), axis=1)
        out["zero_displacement_share"] = float(np.mean(d < 0.5))
        out["max_displacement_px"] = float(d.max())
    out["ok"] = bool(out["v_norm_m_s"] < IDENTITY_V_MAX
                     and np.deg2rad(out["w_norm_deg_s"]) < IDENTITY_W_MAX
                     and out["n_matched"] > 0)
    return out


def parse_budgets(items) -> dict[str, int]:
    """``--budget ARM=STEPS`` (repeatable) -> ``{arm_id: steps}``; refuses a malformed or
    non-positive entry and a repeated arm."""
    out: dict[str, int] = {}
    for item in items or []:
        arm, sep, steps = str(item).partition("=")
        if not sep or not arm.strip() or not steps.strip().isdigit() or int(steps) <= 0:
            raise SystemExit(f"--budget {item!r}: expected ARM=STEPS with STEPS a positive integer")
        if arm.strip() in out:
            raise SystemExit(f"--budget: arm {arm.strip()!r} given twice")
        out[arm.strip()] = int(steps)
    return out


def episode_check(scene, arm, cfg, start: dict, max_steps: int, save_first=None,
                  film_dir: Path | None = None, film_every: int = 1,
                  perturbation=None, outline_iou: bool = False, *, arm_id: str = "",
                  debug_dir: Path | None = None, clean_film: bool = False) -> dict:
    """One flight and its numbers. ``debug_dir`` (``--telemetry``): the debug log goes
    there (vsbench.vs.bench.debuglog), the record gets the depth-true reads, and the film - if
    any - is driven by the log (names, three distances, landings, a final frame);
    ``clean_film`` adds the clean half-size live film. Without ``debug_dir`` this is the
    old path, line for line."""
    episode = {**(cfg.get("episode") or {}), "max_iter": int(max_steps)}
    if outline_iou:
        episode["outline_iou"] = True
    if debug_dir is not None:
        episode["depth_true"] = True
    cfg = dict(cfg, episode=episode)
    telem: list = []
    on_step = None
    writer = None
    if film_dir is not None:
        from vsbench.vs.bench import films
        from vsbench.vs.scene import rotation_error_deg, translation_error_mm

        scene.reset(scene.goal_pose)
        goal_rgb, _d, _p = scene.render()
        goal = scene.goal_pose
        writer = films.FilmWriter(film_dir, goal_rgb, every=film_every, scale=0.5,
                                 title=f"{scene.name} {arm.name.split('/')[0]} {start['start_id']}"
                                       + (f" [{perturbation.mode}]" if perturbation is not None
                                          and perturbation.mode != "none" else ""))

        def on_step(i, rgb, debug, pose, _w=writer):
            _w.step(i, rgb, debug, translation_error_mm(pose.position, goal.position),
                    rotation_error_deg(pose.rotation, goal.rotation))
    if debug_dir is None:
        rec = run_episode(scene, arm, start, cfg, telemetry=telem, on_step=on_step,
                          perturbation=perturbation)
    else:
        from vsbench.vs.bench.debuglog import DebugLog

        # The recorder can never cost a flight: if it cannot even open its file, the
        # episode flies without it (the film on the old hook) and the record says why.
        try:
            dlog = DebugLog(debug_dir, arm_id=arm_id, film=writer, clean=clean_film)
        except Exception as exc:  # noqa: BLE001 - the recorder only listens
            dlog, log_error = None, f"open: {exc!r}"[:500]
            print(f"[check] debug log not written ({log_error}); flying without it",
                  flush=True)
        if dlog is None:
            rec = run_episode(scene, arm, start, cfg, telemetry=telem, on_step=on_step,
                              perturbation=perturbation)
        else:
            # the log drives the film (its captions need the three distances), so no on_step
            with dlog:
                rec = run_episode(scene, arm, start, cfg, telemetry=telem,
                                  perturbation=perturbation, debug_log=dlog)
            log_error = dlog.error          # a failure while closing lands here too
        if log_error:
            rec["debug_log_error"] = log_error
            print(f"[check] debug log error: {log_error}", flush=True)
    # The method's OWN stop, when the arm owns one (DINOBot's `dinobot_err`: the pose at
    # which the published loop exits - "as it ends on the robot", owner 2026-09-23). The
    # protocol's scored stop and the cap are the two other landings; all three are read
    # from the same episode. Generic: the arm names the rules it owns
    # (`recorded_stop_constants`), the runner records where each fired (`rec["stops"]`).
    owned = list((getattr(arm, "recorded_stop_constants", dict)() or {}).keys())
    fired = rec.get("stops") or {}
    own = next((fired[r] for r in owned if r in fired), None)
    own_stop = {"own_stop_rule": owned[0] if owned else None,
                "own_stop_step": own["step"] if own else None,
                "own_stop_mm": own["translation_mm"] if own else None,
                "own_stop_deg": own["rotation_deg"] if own else None}
    if writer is not None:
        from vsbench.vs.bench import films

        writer.close()
        mp4 = films.encode_episode(film_dir, scale=1.0, keep_frames=False,
                                   **({"holds": writer.holds} if debug_dir is not None else {}))
        print(f"[check] film {mp4}", flush=True)
    if save_first is not None and telem:
        # the first live frame is not in the telemetry; re-render the start pose
        scene.reset(Pose.from_start(start))
        rgb, _d, _p = scene.render()
        save_png(save_first, rgb)
    # the instance suite's second reference (the size-matched pose) and the outline
    # overlap - present only when the world change offered one / the switch was on
    matched = {k: rec[k] for k in ("start_translation_mm_matched", "final_translation_mm_matched",
                                    "stop_translation_mm_matched", "success_matched",
                                    "reference_pose", "final_pose", "outline_iou_final",
                                    "outline_iou_stop",
                                    # the depth-true reads (--telemetry, 2026-09-30)
                                    "start_translation_mm_depth_true",
                                    "final_translation_mm_depth_true",
                                    "stop_translation_mm_depth_true", "success_depth_true",
                                    "depth_true_pose", "debug_log_s",
                                    "debug_log_error") if k in rec}
    if own is not None:
        for key in ("translation_mm_matched", "outline_iou", "position",
                    "translation_mm_depth_true"):
            if key in own:
                matched[f"own_stop_{key}"] = own[key]
    if debug_dir is not None:
        matched["telemetry_file"] = str(Path("films") / Path(debug_dir).name / "telemetry.jsonl.gz")
    return {
        "start_id": start["start_id"], "max_iter": rec["max_iter"], **matched,
        "start_mm": rec["start_translation_mm"], "start_deg": rec["start_rotation_deg"],
        "final_mm": rec["final_translation_mm"], "final_deg": rec["final_rotation_deg"],
        "stop_step": rec["stop_step"], "stop_mm": rec["stop_translation_mm"],
        "stop_deg": rec["stop_rotation_deg"],
        # every recorded stop's landing (velocity_quiet, dec_plateau, an arm's own rule ...)
        "stops": fired, **own_stop,
        "steps": rec["steps"], "terminated": rec["terminated_reason"],
        "outcome": rec["outcome"], "success": rec["success"],
        "sec_per_step": rec["sec_per_step"], "mean_n_matches": rec["mean_n_matches"],
        "prealign_deg": rec.get("prealign_deg"),
        "start_deg_after_prealign": rec.get("start_rotation_deg_after_prealign"),
        "goal_mask_share": rec.get("goal_mask_share"),
        "perturbation": rec.get("perturbation"),
        # cost columns (owner 2026-09-22): the loop's and the matcher's rate, GPU load
        "fps": rec.get("fps"), "match_fps": rec.get("match_fps"),
        "match_ms_med": rec.get("match_ms_med"), "control_ms_med": rec.get("control_ms_med"),
        "gpu_util_mean_pct": rec.get("gpu_util_mean_pct"),
        "gpu_util_max_pct": rec.get("gpu_util_max_pct"),
        "peak_vram_alloc_mb": rec.get("peak_vram_alloc_mb"),
        "vram_used_mb": rec.get("vram_used_mb"),
    }


def near_start(scene) -> dict:
    """The benchmark's near sampler on THIS scene's goal: 3-8 mm, <= 3 deg, first draw."""
    g = scene.goal_pose
    s = startsmod.sample_near(g.position, g.rotation, scene.working_distance, n=1)[0]
    return dict(s, starts_seed=startsmod.NEAR["seed"])


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def own_text(rec) -> str:
    """The method's own stop, if the arm owns one: ``own 1.5 mm/0.3 deg @2``, or
    ``own never`` when the owned rule did not fire; empty for arms without one."""
    if not rec or not rec.get("own_stop_rule"):
        return ""
    if rec.get("own_stop_step") is None:
        return " (own never)"
    return (f" (own {rec['own_stop_mm']:.2f} mm/{rec['own_stop_deg']:.2f} deg "
            f"@{rec['own_stop_step']})")


def near_text(nr) -> str:
    if not nr:
        return "-"
    return (f"{nr['start_mm']:.2f} -> {nr['final_mm']:.3f} mm, "
            f"{nr['start_deg']:.2f} -> {nr['final_deg']:.3f} deg" + own_text(nr))


def far_text(far) -> str:
    if not far:
        return ""
    return " | far " + ", ".join(f"{f['start_mm']:.0f}->{f['final_mm']:.1f} mm{own_text(f)}"
                                for f in far)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenes", nargs="+", default=list(benchmark.SCENES))
    ap.add_argument("--arms", nargs="+", default=None, help="default: every protocol arm")
    ap.add_argument("--near-steps", type=int, default=200)
    ap.add_argument("--far", type=int, default=0, help="fly the first N far starts to the cap")
    ap.add_argument("--far-set", default="capped30", choices=sorted(benchmark.START_SETS),
                    help="which protocol start set the far starts come from (default capped30; "
                         "wide100 has every roll)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--film", action="store_true",
                    help="a match video per episode: <out>/films/<scene>_<arm>_<start>/episode.mp4")
    ap.add_argument("--film-every", type=int, default=1, help="keep every n-th step")
    ap.add_argument("--perturb", default="none",
                    help="none | light | distractors | occluder | noise | combined | "
                         "swap:<id> | swap:@<live name> | ground:<name>, or a `+` chain - "
                         "applied after the goal capture of every near/far episode "
                         "(vsbench.vs.bench.perturb)")
    ap.add_argument("--perturb-seed", type=int, default=20260918)
    ap.add_argument("--budget", action="append", default=[], metavar="ARM=STEPS",
                    help="one arm's own step budget for its near AND far starts (repeatable), "
                         "e.g. dinobot_objmask=30 - recorded as max_iter")
    ap.add_argument("--outline-iou", action="store_true",
                    help="record the object-outline overlap with the goal picture at every "
                         "recorded stop and at the end (episode.outline_iou)")
    ap.add_argument("--telemetry", action="store_true",
                    help="the debug log per episode (<out>/films/<scene>_<arm>_<start>/"
                         "telemetry.jsonl.gz + goal.jpg + goal_mask.png): poses, distances to "
                         "the goal / size-matched / depth-true pose, matched pairs with a "
                         "right/wrong truth check, commands, stop rules; the depth-true reads "
                         "in every record; with --film, films with names, three distances, "
                         "landing marks and a final frame")
    ap.add_argument("--clean-film", action="store_true",
                    help="a clean, un-drawn, half-size live film per episode (clean.mp4) for "
                         "scripts/debug_view.py; needs --telemetry")
    ap.add_argument("--arm-plugin", default=None, metavar="MODULE",
                    help="an importable module (PYTHONPATH) stating ARM_IDS, build() and "
                         "describe(): its ids may then be named in --arms; its arms must be "
                         "'external:'-labelled of kind 'external' (default: none)")
    args = ap.parse_args()
    if args.clean_film and not args.telemetry:
        raise SystemExit("--clean-film needs --telemetry (the debug viewer needs both)")
    budgets = parse_budgets(args.budget)
    unknown_budget = sorted(set(budgets) - set(args.arms or []))
    if args.arms is not None and unknown_budget:
        raise SystemExit(f"--budget names arm(s) not flown: {unknown_budget}")

    protocol = fly.load_protocol()
    arm_ids = args.arms or [e["id"] for e in protocol["arms"]]
    # every id resolved BEFORE Isaac boots: an unknown one costs a second, not a boot
    plugin = load_arm_plugin(args.arm_plugin)
    sources = resolve_arm_sources(protocol, arm_ids, plugin)

    from vsbench.vs.isaac_scene import IsaacScene, nonzero_exit_on_error
    from vsbench.vs.scenes import load_scenes

    specs, _ = load_scenes()
    args.out.mkdir(parents=True, exist_ok=True)
    results: dict = {"scenes": {}, "arms": arm_ids, "near_steps": args.near_steps,
                     "far": args.far, "far_set": args.far_set,
                     "perturb": args.perturb, "perturb_seed": args.perturb_seed,
                     "budgets": budgets, "outline_iou": bool(args.outline_iou),
                     "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.telemetry or args.clean_film:
        # only when on, so an old command's check.json keeps its exact header
        results.update(telemetry=bool(args.telemetry), clean_film=bool(args.clean_film))
    if plugin is not None:
        # only with a plugin, so every other check.json keeps its exact header
        results["arm_plugin"] = args.arm_plugin
    from vsbench.vs.bench import perturb as perturbmod
    rows = []
    with (IsaacScene(goal_render_warmup=cfgmod.goal_render_warmup(protocol),
                     goal_capture_keep=cfgmod.goal_capture_keep(protocol)) as scene,
          nonzero_exit_on_error("check_baselines")):
        for name in args.scenes:
            scene.clear_scene_prims()
            scene.build_eval_scene(specs[name])
            # the goal check reads the file the far set is cut from (the instance suite's
            # scenes have only their `simple` file)
            payload = benchmark.load_starts(name, benchmark.START_SETS[args.far_set]["file"])
            benchmark.check_scene_goal(scene, payload)
            rgb, depth, _ = scene.render()
            save_png(args.out / f"{name}_goal.png", rgb)
            np.save(args.out / f"{name}_goal_depth.npy", depth)
            g = scene.goal_pose
            results["scenes"][name] = {
                "goal_distance_m": scene.working_distance, "goal_md5": scene.goal_frame_md5,
                "goal_settle_diffs": scene.goal_settle["diff_prev"],
                "goal": g.as_dict(), "arms": {}}
            print(f"[check] {name}: goal d={scene.working_distance:.6f} m, "
                  f"settle diffs {scene.goal_settle['diff_prev']}", flush=True)
            near = near_start(scene)
            far = benchmark.select_starts(name, args.far_set, limit=args.far) if args.far else []
            pert = perturbmod.build(args.perturb, seed=args.perturb_seed, spec=specs[name])
            for arm_id in arm_ids:
                t0 = time.time()
                described = None
                try:
                    if sources[arm_id] == "plugin":
                        arm, cfg, described = build_plugin_arm(
                            plugin, arm_id, intrinsics=scene.intrinsics, scene_name=name,
                            scene_spec=specs[name])
                    else:
                        arm, cfg = fly.build(protocol, arm_id, intrinsics=scene.intrinsics,
                                             scene_name=name, scene_spec=specs[name])
                except Exception as exc:  # noqa: BLE001 - one broken arm must not end the night
                    print(f"[check] {name} x {arm_id}: BUILD FAILED: {exc!r}", flush=True)
                    results["scenes"][name]["arms"][arm_id] = {"build_error": repr(exc)}
                    rows.append((name, arm_id, None, None, None))
                    continue
                entry: dict = {"label": arm.name}
                if described is not None:
                    entry["plugin"] = described
                # the detector's word when a cell takes it from the scene (its label says
                # `prompt-scene`; the record field says which word was used)
                roi_prompt = (getattr(arm, "record_fields", dict)() or {}).get("roi_prompt")
                if roi_prompt is not None:
                    entry["roi_prompt"] = roi_prompt
                try:
                    entry["identity"] = identity_check(scene, arm)
                    def film_dir(start_id, _n=name, _a=arm_id):
                        return (args.out / "films" / f"{_n}_{_a}_{start_id}") if args.film else None

                    def debug_dir(start_id, _n=name, _a=arm_id):
                        return ((args.out / "films" / f"{_n}_{_a}_{start_id}")
                                if args.telemetry else None)
                    budget = budgets.get(arm_id)
                    entry["near"] = episode_check(
                        scene, arm, cfg, near, budget or args.near_steps,
                        save_first=args.out / f"{name}_{arm_id}_near_first.png"
                        if arm_id == arm_ids[0] else None,
                        film_dir=film_dir(near["start_id"]), film_every=args.film_every,
                        perturbation=pert, outline_iou=args.outline_iou, arm_id=arm_id,
                        debug_dir=debug_dir(near["start_id"]), clean_film=args.clean_film)
                    entry["far"] = [episode_check(scene, arm, cfg, s,
                                                  budget or cfg["episode"]["max_iter"],
                                                  film_dir=film_dir(s["start_id"]),
                                                  film_every=args.film_every, perturbation=pert,
                                                  outline_iou=args.outline_iou, arm_id=arm_id,
                                                  debug_dir=debug_dir(s["start_id"]),
                                                  clean_film=args.clean_film)
                                    for s in far]
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    entry["error"] = repr(exc)
                finally:
                    close = getattr(arm, "close", None)
                    if close is not None:
                        close()
                entry["wall_s"] = time.time() - t0
                results["scenes"][name]["arms"][arm_id] = entry
                idn, nr = entry.get("identity"), entry.get("near")
                print(f"[check] {name} x {arm_id}: identity "
                      f"{'OK' if idn and idn['ok'] else 'FAIL'} "
                      f"(n={idn['n_matched'] if idn else '?'}, "
                      f"|v|={fmt(idn['v_norm_m_s'] if idn else None, 6)} m/s, "
                      f"|w|={fmt(idn['w_norm_deg_s'] if idn else None, 4)} deg/s); near "
                      f"{near_text(nr)}"
                      f"{far_text(entry.get('far'))}"
                      f"{' ERROR ' + entry['error'] if entry.get('error') else ''}"
                      f" [{entry['wall_s']:.0f} s]", flush=True)
                rows.append((name, arm_id, idn, nr, entry.get("far")))
                (args.out / "check.json").write_text(json.dumps(results, indent=1))
        write_table(rows, args.out)     # INSIDE the `with`: Isaac's close() may end the process
    return 0


def write_table(rows, out: Path) -> None:
    lines = ["| scene | arm | identity: matched / \\|v\\| m/s / \\|w\\| deg/s / zero-disp | "
             "near: start -> final (mm, deg) | stop step | "
             "near: method's own stop (mm, deg @step) | "
             "far: start -> final mm (cap) [own stop] |",
             "| --- | --- | --- | --- | --- | --- | --- |"]

    def own_cell(rec) -> str:
        if not rec or not rec.get("own_stop_rule"):
            return "-"
        if rec.get("own_stop_step") is None:
            return f"{rec['own_stop_rule']}: never"
        return (f"{rec['own_stop_mm']:.3f}, {rec['own_stop_deg']:.3f} @{rec['own_stop_step']} "
                f"({rec['own_stop_rule']})")

    for name, arm_id, idn, nr, far in rows:
        if idn is None:
            lines.append(f"| {name} | {arm_id} | BUILD FAILED | | | | |")
            continue
        zd = idn.get("zero_displacement_share")
        near_cell = ("ERROR | -" if not nr else
                     f"{nr['start_mm']:.1f} -> {nr['final_mm']:.3f}, {nr['start_deg']:.2f} -> "
                     f"{nr['final_deg']:.3f} | {nr['stop_step']}")
        far_cell = (", ".join(
            f"{f['start_mm']:.0f} -> {f['final_mm']:.1f}"
            + (f" [{f['own_stop_mm']:.1f} @{f['own_stop_step']}]"
               if f.get("own_stop_step") is not None
               else (" [never]" if f.get("own_stop_rule") else ""))
            for f in far) if far else "-")
        lines.append(
            f"| {name} | {arm_id} | {'OK' if idn['ok'] else 'FAIL'}: {idn['n_matched']} / "
            f"{idn['v_norm_m_s']:.2e} / {idn['w_norm_deg_s']:.2e} / "
            f"{'-' if zd is None else f'{zd:.2f}'} | {near_cell} | {own_cell(nr)} | {far_cell} |")
    (out / "check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
