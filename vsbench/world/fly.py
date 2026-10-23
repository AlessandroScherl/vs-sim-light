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
* the station rules: torch on 2 threads (intra- and inter-op); ``CUDA_VISIBLE_DEVICES=""`` and
  ``VSBENCH_SOFTWARE_GL=1`` forced; refuses to run when torch sees a GPU **or** the built
  scene's GL context is not Mesa llvmpipe (``--allow-gpu`` lifts both; off by default); the
  ``nvidia-smi`` the copied loop polls for its hardware columns is shadowed by a stub
  (on this shared machine it would report the Isaac evaluation's GPU as this run's), so
  ``vram_used_mb`` / ``gpu_util_*`` read ``None``, the truth for a CPU run;
* ``--look`` (port unit U4b, ``vsbench.world.look``): the picture the scene draws -
  ``isaac_fit_v1`` (default: the fitted light, shadow map and per-scene tone curve) or
  ``raw`` (the U1/U4 picture); every record carries ``look``, ``look_curve_id``,
  ``look_curve_md5`` and ``look_record`` (from ``goal_capture_record``);
* ``--max-iter``, ``--device`` and ``--telemetry-extended`` overrides (stated in the
  resolved config; ``max_iter`` on the record); ``--film`` writes the copied
  ``FilmWriter``'s match film. Start sets: the protocol's only, as vs-sim's driver.

``--perturb <mode>`` / ``--perturb-seed`` (port unit U8; vs-sim's ``fly.py run`` flags, same
default seed 20260918): the copied ``vsbench.vs.bench.perturb.build(mode, seed, spec)`` handed
to every episode, applied after the goal capture; the world verbs are
``vsbench.world.perturb_world``'s (an edit recompiles the MuJoCo model; the scene's look and
goal picture stay the unedited ones). A perturbed record adds ``perturb_mode``,
``perturb_seed`` and ``world_compiles`` (the episode's recompiles: seconds, RSS).

``--method <name> [...]`` (port unit U6a) flies a registered vsbench method through
``vsbench.world.method_arm.MethodArm`` on the same loop, scene, starts and record path: the
config is ``vsbench/world/configs/external_method.yaml`` (``--param key=value`` for the
method's constructor), the record says ``arm_kind: external`` and the label
``external:<method>@<params>/<stop>``, the run sidecar is the copied one
(``method_arm.open_method_run``). Refused: a privileged method (it reads the true pose) and a
method that is a copied protocol cell (``vitvs_dinov3`` & co. fly with ``--arms``).

``--mask none|objects|<file.png>`` (2026-10-23, ``vsbench.world.masks``): filter the matches
with a mask on the goal picture. ``objects`` flies a copied arm's ``objmask`` cell (the
protocol's ``<arm>_objmask`` when it has one, else the arm plus ``matcher.goal_mask:
objects``); a file flies the same seam with the file's mask (``goal_mask_mode: file``, label
token ``filemask-<md5[:8]>``, ``goal_mask_source: file:<name>:<md5>``); a ``--method`` gets it
through ``set_target(..., goal_mask=...)``. The cell tag gains ``_objmask`` /
``_filemask_<md5[:8]>``; an unmasked record is unchanged.

``--debug-gif`` (2026-10-23): the copied ``DebugLog`` on every episode (``debug/<start_id>/``
in the run directory: its telemetry, clean film, goal picture and mask) and the copied debug
view animated into ``film/<start_id>/debug.gif`` (``vsbench.world.gif.debug_gif``; the
record's ``debug_gif`` field); it turns ``--telemetry`` on.
"""

from __future__ import annotations

import argparse
import atexit
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
#: the GL renderer strings of Mesa's software rasterisers (``glGetString(GL_RENDERER)``)
SOFTWARE_GL_NAMES = ("llvmpipe", "softpipe", "swrast")
_KEEP: list = []


def _nosmi_stub() -> Path:
    """A throwaway directory holding an ``nvidia-smi`` that always fails, removed at exit."""
    tmp = tempfile.TemporaryDirectory(prefix="vsbench_nosmi_")
    atexit.register(tmp.cleanup)
    stub = Path(tmp.name) / "nvidia-smi"
    stub.write_text("#!/bin/sh\necho 'vsbench.world.fly: nvidia-smi hidden (CPU run)' >&2\nexit 9\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    _KEEP.append(tmp)   # the object must outlive this call, or its finaliser cleans up at once
    return Path(tmp.name)


def station_guard(threads: int = 2, *, allow_gpu: bool = False) -> dict:
    """Two threads everywhere, no GPU, no ``nvidia-smi``. Call before torch / mujoco load.

    Without ``allow_gpu`` the GPU is hidden by force, not by default: ``CUDA_VISIBLE_DEVICES``
    is set to ``""`` and ``VSBENCH_SOFTWARE_GL`` to ``1`` whatever the shell exported (so a
    shell that forgot ``tools/cpu_env.sh``, or exported ``VSBENCH_SOFTWARE_GL=0``, still runs
    on llvmpipe), and :func:`require_software_gl` refuses the scene afterwards if MuJoCo
    nevertheless opened a hardware context. ``allow_gpu`` (``--allow-gpu``, off by default;
    never while the Isaac evaluation runs) leaves both variables as exported and lifts both
    refusals. The thread caps are ``setdefault`` (a smaller cap the shell set wins). Software
    GL is the default everywhere since 2026-10-23 (``vsbench.glenv``); ``allow_gpu`` also sets
    ``VSBENCH_ALLOW_GPU=1`` by default, so an exported ``VSBENCH_SOFTWARE_GL=0`` still reaches
    the GPU as before.
    """
    if allow_gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("VSBENCH_SOFTWARE_GL", "1")
        os.environ.setdefault("VSBENCH_ALLOW_GPU", "1")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["VSBENCH_SOFTWARE_GL"] = "1"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ.setdefault(k, str(threads))
    stub_dir = _nosmi_stub()
    os.environ["PATH"] = f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    import torch

    # before any torch op: the inter-op pool can only be sized before it first runs
    try:
        torch.set_num_interop_threads(int(threads))
    except RuntimeError as exc:   # already started in this process (e.g. a test imported torch)
        print(f"[fly] torch.set_num_interop_threads({threads}) refused: {exc}", flush=True)
    torch.set_num_threads(int(threads))
    cuda = bool(torch.cuda.is_available())
    if cuda and not allow_gpu:
        raise SystemExit("torch sees a GPU - the bridge runs on the CPU only "
                         "(source tools/cpu_env.sh: CUDA_VISIBLE_DEVICES=''; --allow-gpu lifts this)")
    return {"torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_available": cuda, "nvidia_smi": "hidden", "allow_gpu": bool(allow_gpu),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "VSBENCH_SOFTWARE_GL": os.environ.get("VSBENCH_SOFTWARE_GL"),
            "LP_NUM_THREADS": os.environ.get("LP_NUM_THREADS")}


def require_software_gl(scene, *, allow_gpu: bool = False) -> None:
    """The hard refusal: the built scene must draw on Mesa's software rasteriser.

    Checks both the scene's ``render_mode`` and the GL renderer string the context reports,
    so an unreadable string can not pass for llvmpipe. ``allow_gpu`` lifts it.
    """
    if allow_gpu:
        return
    gl_name = str(getattr(scene.renderer, "gl_renderer", "") or "").lower()
    if scene.render_mode != "mujoco-llvmpipe" or not any(n in gl_name for n in SOFTWARE_GL_NAMES):
        scene.close()
        raise SystemExit(f"refusing to fly: the MuJoCo context is {scene.render_mode!r} "
                         f"(GL renderer {gl_name!r}), not Mesa llvmpipe - the bridge never "
                         "touches the GPU while the Isaac evaluation runs (--allow-gpu lifts this)")


def vsbench_commit(ignore: Path | None = None) -> str | None:
    """This repository's HEAD, with ``+dirty`` when the working tree is dirty.

    "Dirty" is the copied ``vsbench.vs.bench.run.dirty_paths`` rule (tracked changes **and**
    untracked, not-ignored files; ignored files do not count), with the run's own output
    (paths under ``ignore`` = ``--out``) left out. The one difference from the ``git_dirty``
    the copied sidecar writes into ``run.json``: the copied rule drops only paths under the
    arm's run root *inside* ``vsbench/`` (its ``REPO_ROOT``), so for the bridge's ``--out``
    (outside ``vsbench/``) the second arm's ``run.json`` lists the first arm's fresh output
    (``../<out>/``) as dirty. ``vsbench_commit`` does not; the record states the rule
    (``vsbench_commit_rule``).
    """
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO), "-c", "color.status=false",
                                 "-c", "status.relativePaths=false", "status", "--porcelain",
                                 "--untracked-files=all"], capture_output=True, text=True,
                                check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return head + ("+dirty" if dirty_paths(status, ignore) else "")


def dirty_paths(porcelain: str, ignore: Path | None = None) -> list[str]:
    """Paths of ``git status --porcelain`` output (repository-top relative), minus those
    under ``ignore``."""
    rel = None
    if ignore is not None:
        try:
            rel = Path(ignore).resolve().relative_to(REPO).as_posix().rstrip("/")
        except ValueError:
            rel = None
    out = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        path = line[3:].split(" -> ", 1)[-1].strip().strip('"')
        if rel and (path == rel or path.startswith(rel + "/")):
            continue
        out.append(path)
    return out


VSBENCH_COMMIT_RULE = ("HEAD + '+dirty' if tracked changes or untracked (not ignored) files exist "
                       "outside --out (the copied dirty_paths rule minus the run's own output; "
                       "run.json's git_dirty_paths cannot drop --out because it lies outside "
                       "vsbench/)")


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
               device: str | None = None, telemetry_extended: bool = False,
               goal_mask: str | None = None) -> dict:
    """The cell's config: its arm file deep-merged with the protocol's deltas (as vs-sim's
    ``fly.py`` does), then the bridge's overrides (``--max-iter``, ``--device``,
    ``--telemetry-extended``), then resolved by the copied loader. None of them is part of a
    cell's derived label - except ``goal_mask`` (``--mask``: ``matcher.goal_mask: objects`` on
    a cell the protocol has no ``_objmask`` twin of), which the copied label states as
    ``objmask``."""
    from vsbench.vs.bench import config as cfgmod
    from vsbench.vs.camera import REPO_ROOT

    entry = arm_entry(protocol, arm_id)
    cfg = _merge(cfgmod.load(REPO_ROOT / "configs" / "arms" / entry["base"]),
                 entry.get("overrides") or {})
    over: dict = {}
    if max_iter is not None:
        over.setdefault("episode", {})["max_iter"] = int(max_iter)
    if telemetry_extended:
        over.setdefault("episode", {})["telemetry_extended"] = True
    if device is not None and cfg["matcher"]["variant"] in cfgmod.DEVICE_VARIANTS:
        over["matcher"] = {"device": device}
    if goal_mask is not None:
        over.setdefault("matcher", {})["goal_mask"] = str(goal_mask)
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
    env = station_guard(args.threads, allow_gpu=args.allow_gpu)
    from vsbench.vs import benchmark
    from vsbench.vs.bench.runner import open_run, run_episode
    from vsbench.vs.scenes import load_scenes
    from vsbench.world import masks
    from vsbench.world.mujoco_scene import MujocoScene

    protocol = load_protocol(args.protocol)
    if not args.arms and not args.method:
        raise SystemExit("nothing to fly: give --arms and/or --method")
    check_methods(protocol, args.method)
    mask_mode, _ = masks.parse(args.mask)
    if mask_mode != "none":
        check_mask_methods(args.method)
    specs, _ = load_scenes()
    if args.start_set not in protocol["start_sets"]:     # vs-sim's fly.py: the protocol's sets only
        raise SystemExit(f"start set {args.start_set!r} is not in the protocol "
                         f"(have {', '.join(protocol['start_sets'])})")
    picked = benchmark.select_starts(args.scene, args.start_set, limit=args.limit)
    if args.start_ids:
        picked = [s for s in picked if s["start_id"] in set(args.start_ids)]
        if not picked:
            raise SystemExit(f"none of {args.start_ids} is in {args.scene}/{args.start_set}")
    payload = benchmark.load_starts(args.scene, benchmark.START_SETS[args.start_set]["file"])
    out_root = Path(args.out).resolve()   # the copied encoder runs ffmpeg with cwd = the film dir
    provenance = {"vsbench_commit": vsbench_commit(ignore=out_root), "vssim_commit": vssim_commit(),
                  "vsbench_commit_rule": VSBENCH_COMMIT_RULE}
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    from vsbench.vs.bench import perturb as perturbmod

    pert = perturbmod.build(args.perturb or "none", seed=int(args.perturb_seed),
                            spec=specs[args.scene])
    args._pert = pert
    with MujocoScene(args.scene, samples=args.samples, specs=specs, look=args.look) as scene:
        require_software_gl(scene, allow_gpu=args.allow_gpu)   # before the first frame
        benchmark.check_scene_goal(scene, payload)
        goal_rgb, _goal_depth = scene.render_goal()        # once, served to every arm
        goal_capture = scene.goal_capture_record()
        gm = masks.resolve(args.mask, scene, _goal_depth)   # known at goal time only
        args._gm = gm
        if gm.mode != "none":
            print(f"[fly] --mask {args.mask}: {gm.source}, keeps {100 * gm.share:.1f}% of the "
                  f"goal picture" + (f" (resized from {gm.resized_from[0]}x{gm.resized_from[1]}, "
                                      "nearest neighbour)" if gm.resized_from else ""), flush=True)
        print(f"[fly] {args.scene}: {scene.render_mode}, look {scene.look.name} "
              f"({scene.look.curve_id}, {scene.look.curve_md5}), goal {scene.working_distance:.6f} m "
              f"(file {payload['goal']['distance_m']:.6f}), goal md5 {goal_capture['goal_frame_md5']}, "
              f"{len(picked)} start(s) of {args.start_set}", flush=True)
        for arm_id in args.arms:
            cell, mask_over = masks.arm_cell(protocol, arm_id, gm)
            cfg = arm_config(protocol, cell, max_iter=args.max_iter, device=args.device,
                             telemetry_extended=args.telemetry_extended,
                             goal_mask="objects" if mask_over else None)
            t_build = time.monotonic()
            arm = build(protocol, cell, cfg, intrinsics=scene.intrinsics,
                        scene_name=args.scene, scene_spec=specs[args.scene])
            masks.attach_goal_mask(arm, gm)
            build_s = time.monotonic() - t_build
            tag = f"{args.scene}_{arm_id}{gm.tag}"
            extra = {**goal_capture, **provenance, "bridge_env": env,
                     "scene_meta": scene.meta, "arm_build_s": round(build_s, 3),
                     "bridge_driver": "vsbench.world.fly", **_mask_extra(gm, cell)}
            with open_run(tag, cfg, out_root=out_root / tag, telemetry=args.telemetry,
                          allow_dirty=True, extra=extra) as sink:
                for i, start in enumerate(picked, 1):
                    steps: list | None = [] if args.telemetry else None
                    film_dir = (sink.path / "film" / start["start_id"]) if args.film else None
                    hook = _Film(film_dir, scene, goal_rgb, title=f"{arm_id}{gm.tag} (MuJoCo)",
                                 every=args.film_every, scale=args.film_scale)
                    n_ev = len(scene.world_log)
                    dlog = _debug_log(args, sink, start, arm_id)
                    try:
                        rec = run_episode(scene, arm, start, cfg, telemetry=steps,
                                          starts_payload=payload, on_step=hook,
                                          perturbation=pert, debug_log=dlog)
                    finally:
                        if dlog is not None:
                            dlog.close()
                    film = hook.close()
                    rec["cell_id"] = tag
                    gm.apply(rec)
                    _debug_gif(args, rec, dlog, sink, start, out_root)
                    _perturb_fields(rec, args, scene, n_ev)
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
        if args.method:
            fly_method_cells(args, scene, picked, payload, provenance, env, goal_rgb,
                             goal_capture, out_root)
    return 0


def _perturb_fields(rec: dict, args, scene, n_ev: int) -> None:
    """``--perturb``'s record fields (U8): the mode, the seed, the episode's recompiles -
    only when a perturbation is flown, so an unperturbed record is what it always was."""
    if (args.perturb or "none") == "none":
        return
    rec["perturb_mode"] = args.perturb or "none"
    rec["perturb_seed"] = int(args.perturb_seed)
    rec["world_compiles"] = scene.world_log[n_ev:]


# ------------------------------------------------------------- --mask, --debug-gif
def _mask_extra(gm, cell) -> dict:
    """``run.json``'s ``bridge_goal_mask`` block (masked cells only)."""
    if gm.mode == "none":
        return {}
    block = {"mode": gm.mode, "source": gm.source, "share": gm.share, "mask_md5": gm.mask_md5,
             "label_token": gm.token}
    if cell is not None:
        block["protocol_cell"] = cell
    if gm.mode == "file":
        block.update(file=gm.file, file_md5=gm.file_md5, resized_from=gm.resized_from,
                     note="config matcher.goal_mask 'objects' only switches the matcher's "
                          "goal-mask seam on; the mask handed to it is the file's "
                          "(goal_mask_mode 'file'), never the depth outline")
    return {"bridge_goal_mask": block}


def check_mask_methods(names) -> None:
    """``--mask`` on a ``--method`` that takes no goal mask is refused before anything is
    built (the label would say masked over an unmasked flight)."""
    from vsbench.methods import method_class, takes_goal_mask

    for name in names:
        cls = method_class(name)
        if not takes_goal_mask(cls):
            raise SystemExit(f"--mask on --method {name!r}: its set_target takes no goal_mask "
                             "(add `goal_mask=None` to it; vsbench.methods.mask_matches filters "
                             "the matches) - or fly it without --mask")


def _debug_log(args, sink, start, arm_id: str):
    """The copied ``DebugLog`` for one episode under ``--debug-gif`` (else ``None``)."""
    if not getattr(args, "debug_gif", False):
        return None
    from vsbench.vs.bench.debuglog import DebugLog

    return DebugLog(sink.path / "debug" / start["start_id"], arm_id=arm_id, clean=True)


def _debug_gif(args, rec: dict, dlog, sink, start, out_root: Path) -> None:
    """``film/<start_id>/debug.gif`` from the episode's debug log; the record says where."""
    if dlog is None:
        return
    from vsbench.world.gif import debug_gif

    gm = args._gm
    if gm.mode == "file":
        # the copied recorder drew the depth outline (it knows no file mask): say so, and draw
        # the file's mask, the one the method was handed
        import cv2
        import numpy as np

        cv2.imwrite(str(dlog.dir / "goal_mask.png"), np.asarray(gm.mask, np.uint8) * 255)
        (dlog.dir / "goal_mask_source.json").write_text(json.dumps(
            {"goal_mask_source": gm.source, "note": "goal_mask.png replaced by the file mask "
             "after the flight; the meta line's mask_source is the copied recorder's default"},
            indent=1) + "\n")
    out = sink.path / "film" / start["start_id"] / "debug.gif"
    try:
        info = debug_gif(dlog.dir, out, every=args.debug_gif_every,
                         max_frames=args.debug_gif_frames,
                         goal_mask=gm.mask if gm.mode == "file" else None)
    except Exception as exc:  # noqa: BLE001 - a drawing fault must not lose the record
        rec["debug_gif_error"] = repr(exc)[:500]
        print(f"[fly] debug.gif failed: {exc!r}", flush=True)
        return
    rec["debug_gif"] = str(out.relative_to(out_root))
    rec["debug_gif_every"] = info["every"]
    print(f"[fly] {rec['debug_gif']}: {info['frames']} frames (every {info['every']} of "
          f"{info['logged_steps']} steps), {info['bytes'] / 1e6:.2f} MB", flush=True)


# ------------------------------------------------------------------- --method (U6a)
def check_methods(protocol: dict, names) -> None:
    """Refuse, before anything is built: an unknown or privileged method, and one that is a
    copied protocol cell (flying it through ``MethodArm`` would wrap a published arm twice)."""
    from vsbench.methods import available_methods, method_class

    proto = {e["id"] for e in protocol["arms"]}
    for name in names:
        if name in proto:
            raise SystemExit(f"--method {name!r} is a protocol arm: fly it with --arms {name}")
        if name not in available_methods():
            raise SystemExit(f"--method {name!r} is not a registered vsbench method "
                             f"(have {', '.join(available_methods())})")
        if bool(getattr(method_class(name), "privileged", False)):
            raise SystemExit(f"--method {name!r} is privileged (it reads the true pose): a "
                             "truth instrument never flies the benchmark loop")


def fly_method_cells(args, scene, picked, payload, provenance, env, goal_rgb, goal_capture,
                     out_root) -> None:
    """``--method``: each method cell as ``cmd_run`` flies an arm cell, through ``MethodArm``."""
    from vsbench.vs.bench.runner import run_episode
    from vsbench.world.method_arm import build_method_arm, open_method_run

    params = {}
    if args.param:
        from vsbench.check import parse_params

        params = parse_params(args.param)
    from vsbench.world import masks

    gm = args._gm
    for name in args.method:
        t_build = time.monotonic()
        arm, cfg = build_method_arm(name, params, intrinsics=scene.intrinsics,
                                    max_iter=args.max_iter,
                                    telemetry_extended=args.telemetry_extended,
                                    goal_mask_mode=gm.mode, mask_token=gm.token,
                                    mask_source=gm.source)
        masks.attach_goal_mask(arm, gm)
        build_s = time.monotonic() - t_build
        tag = f"{args.scene}_external_{name}{gm.tag}"
        extra = {**goal_capture, **provenance, "bridge_env": env, "scene_meta": scene.meta,
                 "arm_build_s": round(build_s, 3), "bridge_driver": "vsbench.world.fly --method",
                 **_mask_extra(gm, None)}
        with open_method_run(tag, cfg, out_root / tag, telemetry=args.telemetry,
                             allow_dirty=True, extra=extra) as sink:
            for i, start in enumerate(picked, 1):
                steps: list | None = [] if args.telemetry else None
                film_dir = (sink.path / "film" / start["start_id"]) if args.film else None
                hook = _Film(film_dir, scene, goal_rgb,
                             title=f"external:{name}{gm.tag} (MuJoCo)",
                             every=args.film_every, scale=args.film_scale)
                n_ev = len(scene.world_log)
                dlog = _debug_log(args, sink, start, f"external_{name}")
                try:
                    rec = run_episode(scene, arm, start, cfg, telemetry=steps,
                                      starts_payload=payload, on_step=hook,
                                      perturbation=args._pert, debug_log=dlog)
                finally:
                    if dlog is not None:
                        dlog.close()
                film = hook.close()
                rec["cell_id"] = tag
                gm.apply(rec)
                _debug_gif(args, rec, dlog, sink, start, out_root)
                _perturb_fields(rec, args, scene, n_ev)
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
                      f"{rec['steps']} steps, stop {rec['stop_mode']} at {rec['stop_step']}, "
                      f"{rec['sec_per_step']:.3f} s/step, frame-1 matches {hook.first_n_matches}, "
                      f"predicate_a {rec['success'].get('predicate_a')}", flush=True)
        (out_root / f"{tag}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in sink.records), encoding="utf-8")
        seen = {r.get("goal_frame_md5") for r in sink.records}
        if seen != {goal_capture["goal_frame_md5"]}:
            raise RuntimeError(f"{tag}: the cells do not share one goal picture: {seen}")
        arm.close()


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
    r.add_argument("--arms", nargs="+", default=[], help="protocol arm ids")
    r.add_argument("--method", nargs="+", default=[],
                   help="registered vsbench methods, flown through MethodArm (U6a)")
    r.add_argument("--param", action="append", default=[],
                   help="--method constructor argument key=value, repeatable")
    r.add_argument("--start-set", default="capped30")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--start-ids", nargs="*", default=None)
    r.add_argument("--max-iter", type=int, default=None, help="override episode.max_iter")
    r.add_argument("--device", default="cpu", help="matcher.device for encoder arms (default cpu)")
    r.add_argument("--out", required=True)
    r.add_argument("--telemetry", action="store_true",
                   help="the copied per-step telemetry (telemetry.jsonl.gz in each run dir)")
    r.add_argument("--telemetry-extended", action="store_true",
                   help="episode.telemetry_extended: the per-match truth columns (gt_u, gt_v_px, "
                        "gt_ok, err_patch, ...); implies --telemetry")
    r.add_argument("--film", action="store_true", help="the copied FilmWriter's match film")
    r.add_argument("--film-every", type=int, default=2)
    r.add_argument("--film-scale", type=float, default=0.25)
    r.add_argument("--mask", default="none",
                   help="filter the matches with a mask on the goal picture: none (default), "
                        "objects (the goal's objects, from the goal depth; the copied objmask "
                        "cell) or a .png file (grey > 127 = keep; vsbench.world.masks)")
    r.add_argument("--debug-gif", action="store_true",
                   help="the copied debug view animated per episode: film/<start>/debug.gif "
                        "(implies --telemetry)")
    r.add_argument("--debug-gif-every", type=int, default=None,
                   help="one debug frame every N steps (default: at most --debug-gif-frames)")
    r.add_argument("--debug-gif-frames", type=int, default=90)
    r.add_argument("--samples", type=int, default=4, help="MuJoCo multisampling (colour only)")
    r.add_argument("--look", default=None, choices=["isaac_fit_v1", "raw"],
                   help="the picture: isaac_fit_v1 (default) or raw (vsbench.world.look)")
    r.add_argument("--perturb", default="none",
                   help="none | light | distractors | occluder | noise | combined | ground:<name> | "
                        "swap:<id> | swap:@<name> | a+b (vsbench.vs.bench.perturb; U8), applied "
                        "after every episode's goal capture")
    r.add_argument("--perturb-seed", type=int, default=20260918)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--allow-gpu", action="store_true",
                   help="lift the GPU refusals (never while the Isaac evaluation runs)")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("summary", help="tabulate the flat records of an --out directory")
    s.add_argument("--out", required=True)
    s.add_argument("--json", default=None)
    s.set_defaults(fn=cmd_summary)

    args = ap.parse_args(argv)
    if getattr(args, "telemetry_extended", False) or getattr(args, "debug_gif", False):
        args.telemetry = True
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
