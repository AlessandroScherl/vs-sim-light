#!/usr/bin/env python3
"""U4b (docs/PORT_PLAN.md section 7, U4b): fit and measure the ``isaac_fit_v1`` look.

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/u4b_look.py components   # per scene (own process): the light
                                                           #   components at the goal -> build/u4b/
    PYTHONPATH=. $VSBENCH_PY tools/u4b_look.py fit          # the fill share, the gains, the curves
                                                           #   -> vsbench/world/configs/look_isaac_fit_v1.json
    PYTHONPATH=. $VSBENCH_PY tools/u4b_look.py measure      # raw vs look vs Isaac, goal + held-out,
                                                           #   per-frame cost -> results/bridge/u4b_look.json,
                                                           #   results/bridge/u4b_look_vs_isaac.png

The Isaac pictures are read, never written (``$VSBENCH_ISAAC_GOALS``, default
``~/genvs_ws/vssim_out/check1``: Isaac's ``check_baselines`` run of 2026-09-17): per scene the
goal picture ``<scene>_goal.png`` (the fit) and ``<scene>_sift_ransac_near_first.png`` - the
first live frame at the check's near start ``n_000`` (``starts.sample_near`` on the goal,
seed 20260902; re-rendered by ``check_baselines`` after the flight) - the held-out view, never
used by the fit. ``check2`` / ``check3`` (same day) hold a second and third Isaac render of the
hollywood and cluttered2 goal and near views: Isaac against itself, the floor of any
comparison.

The fit, in order (everything it chose is written into the config):

1. components - at each goal, MuJoCo's two light terms drawn apart with the look's shadow map
   (8192, scene-fitted clip): ``AMB`` (ambient 1, no sun = the texture as seen) and ``DIR``
   (no ambient, the sun at the intensity that renders a lit flat texel at its own tone,
   shadows on);
2. the fill share ``a`` - for each ``a`` on a grid the rooms' goal pictures are synthesised as
   ``a AMB + (1 - a) DIR``, a curve is fitted per room and channel, and the error is read on
   the *dim* pixels only (the sun's term below half: shadows and turned-away faces, where
   ``a`` matters); the ``a`` with the least mean dim error over the three rooms wins (posters
   have no dim pixels);
3. the gain ``G`` per scene - the light raised as far as it goes: the largest ``G`` with at
   most ``CLIP_BUDGET`` of the goal picture clipped (Isaac clips none) and the sun's term
   ``(1 - a) G <= 1`` (MuJoCo saturates the diffuse term at 1, measured);
4. the curve per scene - the real look-lit goal picture (``MujocoScene`` with the light and
   shadow map of steps 2-3, no curve) against Isaac's, per channel: binned medians (one bin
   per grey level, >= 20 pixels) fitted by ``vsbench.world.look.curve_model`` (floor,
   log-exposure, gamma; weights sqrt(count)) with ``scipy.optimize.least_squares``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCENES = ("hollywood", "poster2", "poster3", "industrial1_drill", "cluttered2", "cluttered3")
ROOMS = ("industrial1_drill", "cluttered2", "cluttered3")
SHEET_SCENES = ("industrial1_drill", "cluttered2", "cluttered3", "hollywood", "poster3")
OUT = REPO / "results" / "bridge"
RECORD = OUT / "u4b_look.json"
CACHE = REPO / "build" / "u4b"
ISAAC = Path(os.environ.get("VSBENCH_ISAAC_GOALS", str(Path.home() / "genvs_ws" / "vssim_out" / "check1")))
ISAAC_REPEATS = [ISAAC.parent / "check2", ISAAC.parent / "check3"]
LOOK = "isaac_fit_v1"
#: the fill-share grid (step 2)
A_GRID = [round(0.05 + 0.025 * k, 3) for k in range(15)]          # 0.05 .. 0.40
#: the most of a goal picture the raised light may clip (step 3)
CLIP_BUDGET = 0.001
SHADOWSIZE = 8192
SHADOWCLIP_MARGIN = 1.02
DIM_SIGMA = 0.5


def _merge_record(key: str, value) -> None:
    rec = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    rec[key] = value
    rec["unit"] = "U4b - match Isaac's look (docs/PORT_PLAN.md section 7)"
    rec["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    RECORD.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"[u4b] wrote {RECORD} [{key}]")


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _isaac(scene: str, view: str, root: Path = ISAAC):
    import numpy as np
    from PIL import Image

    p = root / (f"{scene}_goal.png" if view == "goal" else f"{scene}_sift_ransac_near_first.png")
    if not p.exists():
        return None, p
    return np.asarray(Image.open(p).convert("RGB")), p


def _child(args: list[str], timeout: int = 1800) -> dict:
    p = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True,
                       timeout=timeout, cwd=REPO)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("U4B ")]
    if p.returncode or not line:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"{args}: failed (exit {p.returncode})")
    return json.loads(line[-1][4:])


def _goal_pose(scene: str):
    import numpy as np

    from vsbench.vs import benchmark
    from vsbench.vs.scene import Pose

    g = benchmark.load_starts(scene, "v2")["goal"]
    return Pose(position=np.asarray(g["cam_pos"], float), rotation=np.asarray(g["cam_R_opt"], float))


# ------------------------------------------------------------------- 1. components
def cmd_one_components(args) -> int:
    import numpy as np

    from vsbench.world import fly

    fly.station_guard(2)
    from vsbench.render.base import make_renderer
    from vsbench.scene import Light, SceneSpec
    from vsbench.vs.scene import pose_to_T
    from vsbench.vs.scenes import load_scenes
    from vsbench.world import build as B
    from vsbench.world.camera import d435i_camera_model
    from vsbench.world.look import shadowclip_for

    specs, _ = load_scenes()
    built = B.build_scene(specs[args.scene])
    sun = built.scene_spec.lights[0]
    cos_inc = -float(sun.direction[2])
    objs = built.scene_spec.objects
    clip = shadowclip_for(built.scene_spec, sun.direction, SHADOWCLIP_MARGIN)
    T = pose_to_T(_goal_pose(args.scene))
    out, gl = {}, None
    for tag, spec in (("AMB", SceneSpec(objs, (), 1.0, built.scene_spec.background)),
                      ("DIR", SceneSpec(objs, (Light(sun.direction, 1.0 / cos_inc, cast_shadow=True),),
                                        0.0, built.scene_spec.background))):
        r = make_renderer("mujoco-egl", spec, d435i_camera_model(), samples=4,
                          shadowsize=SHADOWSIZE, shadowclip_m=clip)
        gl = r.gl_renderer
        if "llvmpipe" not in gl.lower():
            raise SystemExit(f"refusing: GL renderer {gl!r} is not llvmpipe")
        out[tag], _ = r.render(T)
        r.close()
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"comp_{args.scene}.npz"
    np.savez_compressed(path, **out)
    print("U4B " + json.dumps({"scene": args.scene, "path": str(path), "shadowclip_m": clip,
                               "gl_renderer": gl}), flush=True)
    return 0


def cmd_components(args) -> int:
    for s in args.scenes:
        r = _child(["_components", s])
        print(f"[u4b] {s}: components -> {r['path']} (clip {r['shadowclip_m']:.3f} m, {r['gl_renderer']})")
    return 0


# --------------------------------------------------------------------------- 2-4. fit
def binned_medians(x, y, min_count: int = 20):
    import numpy as np

    xi = np.clip(np.rint(np.asarray(x, float)).astype(int), 0, 255).ravel()
    yv = np.asarray(y, float).ravel()
    order = np.argsort(xi, kind="stable")
    xs, ys = xi[order], yv[order]
    edges = np.searchsorted(xs, np.arange(257))
    cnt = np.diff(edges)
    med = np.full(256, np.nan)
    for k in range(256):
        if cnt[k] >= min_count:
            med[k] = np.median(ys[edges[k]:edges[k + 1]])
    return cnt, med


def fit_channel(x, y) -> dict:
    """One channel's curve: ``curve_model`` on the binned medians, weights sqrt(count)."""
    import numpy as np
    from scipy.optimize import least_squares

    from vsbench.world.look import curve_model

    cnt, med = binned_medians(x, y)
    k = np.where(~np.isnan(med))[0]
    w = np.sqrt(cnt[k])
    res = least_squares(lambda p: w * (curve_model(p, k / 255.0) - med[k]), [0.1, 1.0, 2.2],
                        bounds=([0.0, -5.0, 0.3], [5.0, 8.0, 6.0]))
    return {"params": [float(v) for v in res.x], "bins": int(len(k)),
            "weighted_rms_bin_residual": float(np.sqrt(np.sum((w * (curve_model(res.x, k / 255.0)
                                                                  - med[k])) ** 2) / np.sum(w ** 2)))}


def synth(comp, a: float, G: float):
    """MuJoCo's picture at fill share ``a`` and gain ``G`` from the two light terms: the
    ambient adds ``a G`` of the texture, the sun's term saturates at the texture's tone."""
    import numpy as np

    A, D = comp["AMB"].astype(np.float64), comp["DIR"].astype(np.float64)
    return a * G * A + np.minimum((1.0 - a) * G * D, A)


def cmd_one_lit(args) -> int:
    """The real look-lit goal picture (no curve), in its own process."""
    import numpy as np

    from vsbench.world import fly

    fly.station_guard(2)
    from vsbench.world.look import Look
    from vsbench.world.mujoco_scene import MujocoScene

    look = Look(LOOK, args.scene, ambient_share=args.a, gain=args.G, shadowsize=SHADOWSIZE,
                shadowclip_margin=SHADOWCLIP_MARGIN, channel_params=None)
    with MujocoScene(args.scene, look=look) as sc:
        fly.require_software_gl(sc)
        rgb, _d = sc.render_goal()
        path = CACHE / f"lit_{args.scene}.npz"
        np.savez_compressed(path, goal=rgb)
        print("U4B " + json.dumps({"scene": args.scene, "path": str(path), "look": sc.look_record}),
              flush=True)
    return 0


def cmd_fit(args) -> int:
    import numpy as np

    from vsbench.world.look import config_path, curve_lut, curve_model, lut_md5

    isaac, comps, src = {}, {}, {}
    for s in SCENES:
        img, p = _isaac(s, "goal")
        if img is None:
            raise SystemExit(f"need {p} (read only)")
        isaac[s], src[s] = img, {"goal": str(p), "goal_md5": _md5(p)}
        hp = _isaac(s, "near")[1]
        if hp.exists():
            src[s]["held_out_not_fitted"] = str(hp)
            src[s]["held_out_md5"] = _md5(hp)
        cp = CACHE / f"comp_{s}.npz"
        if not cp.exists():
            raise SystemExit(f"need {cp} (tools/u4b_look.py components)")
        z = np.load(cp)
        comps[s] = {k: z[k] for k in z.files}

    # 2. the fill share, from the rooms' dim pixels
    table = {}
    for a in A_GRID:
        row = {}
        for room in ROOMS:
            c, y = comps[room], isaac[room].astype(np.float64)
            x = synth(c, a, 1.0)
            A = c["AMB"].astype(np.float64).mean(2)
            sig = c["DIR"].astype(np.float64).mean(2) / np.maximum(A, 1.0)
            dim = (sig < DIM_SIGMA) & (A > 8)
            err_dim = err_all = 0.0
            for ch in range(3):
                p = fit_channel(x[..., ch], y[..., ch])["params"]
                pr = curve_model(p, x[..., ch] / 255.0)
                err_all += float(np.abs(pr - y[..., ch]).mean()) / 3
                err_dim += float(np.abs(pr - y[..., ch])[dim].mean()) / 3
            row[room] = {"dim_mae": round(err_dim, 3), "all_mae": round(err_all, 3),
                         "dim_share": round(float(dim.mean()), 4)}
        row["mean_dim_mae"] = round(float(np.mean([row[r]["dim_mae"] for r in ROOMS])), 3)
        table[str(a)] = row
        print(f"[u4b] a = {a:.3f}: dim MAE " + "  ".join(f"{r} {row[r]['dim_mae']:.2f}" for r in ROOMS)
              + f"  mean {row['mean_dim_mae']:.2f}", flush=True)
    a_best = min(A_GRID, key=lambda a: table[str(a)]["mean_dim_mae"])
    g_cap = 1.0 / (1.0 - a_best)
    print(f"[u4b] fill share a = {a_best} (sun term saturates at G = {g_cap:.3f})")

    # 3. the gain per scene, 4. the curve per scene on the real look-lit picture
    scenes = {}
    for s in SCENES:
        gains = np.round(np.arange(1.0, g_cap + 1e-9, 0.005), 3)
        clipped = [float((synth(comps[s], a_best, G).max(2) > 255.5).mean()) for G in gains]
        ok = [G for G, c in zip(gains, clipped, strict=True) if c <= CLIP_BUDGET]
        G = float(max(ok)) if ok else 1.0
        lit = _child(["_lit", s, "--a", str(a_best), "--G", str(G)])
        x = np.load(lit["path"])["goal"]
        y = isaac[s]
        curve = {}
        for ch, name in enumerate("RGB"):
            curve[name] = fit_channel(x[..., ch].astype(float), y[..., ch].astype(float))
        params = [curve[c]["params"] for c in "RGB"]
        lut = curve_lut(params)
        for c in "RGB":
            curve[c]["monotone"] = True        # by construction (exp(log_e) > 0, gamma > 0)
        curve["lut_md5"] = lut_md5(lut)
        clip_real = float((x.max(2) >= 255).mean())
        scenes[s] = {"light_gain": G, "goal_clipped_share_raw_lit": round(clip_real, 6),
                     "gain_limited_by": (
                         "the sun's term saturating at 1 (the grid's last step below 1 / (1 - a))"
                         if G > g_cap - 0.005 else
                         "no raise: already at gain 1 more than the clip budget of the goal picture "
                         "is the texture's own white (255)" if G == 1.0 else
                         f"the clip budget ({CLIP_BUDGET:.1%} of the goal picture)"),
                     "curve": curve, "isaac_source": src[s],
                     "look_lit_goal_md5": hashlib.md5(x.tobytes()).hexdigest(),
                     "shadowclip_m": lit["look"].get("shadowclip_m")}
        print(f"[u4b] {s}: G = {G:.3f} ({scenes[s]['gain_limited_by']}), clipped {clip_real:.4%}, "
              f"curve RGB {[[round(v, 3) for v in p] for p in params]}, lut md5 {curve['lut_md5']}",
              flush=True)

    from vsbench.world.fly import vsbench_commit

    cfg = {
        "look": LOOK, "version": 1,
        "decided": "owner, 2026-10-23: fit one brightness/tone curve per scene to the Isaac goal "
                   "pictures already on disk, raise the light, use a finer, softer shadow map",
        "fitted": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "fitted_by": "tools/u4b_look.py components + fit",
        "fitted_at_vsbench_commit": vsbench_commit(),
        "lighting": {"ambient_share": a_best, "shadowsize": SHADOWSIZE,
                     "shadowclip_margin": SHADOWCLIP_MARGIN,
                     "ambient": "ambient_share * light_gain (MuJoCo's headlight ambient)",
                     "sun": "the U1 sun direction (Isaac's distant light, rotateXYZ(-30, 0, 30)) at "
                            "(1 - ambient_share) * light_gain / cos(incidence), casting shadows",
                     "shadowclip": "half-side of the square holding all geometry seen from the "
                                   "sun, times shadowclip_margin (vsbench.world.look.shadowclip_for)",
                     "fill_share_fit": {"grid": A_GRID, "rule": "least mean dim-pixel MAE over the "
                                        "three rooms (dim: the sun's term < 0.5 of the texture, "
                                        "texture > 8), curves refitted per a", "table": table},
                     "gain_rule": f"largest gain (step 0.005) with <= {CLIP_BUDGET:.1%} of the goal "
                                  f"picture clipped (synthesised), capped at 1 / (1 - a) = "
                                  f"{g_cap:.4f} where MuJoCo saturates the sun's diffuse term"},
        "curve_model": "per channel y = 255 * srgb_encode(v / (1 + v)), v = floor + "
                       "exp(log_exposure) * x ** gamma, x = raw / 255; params = [floor, "
                       "log_exposure, gamma]; applied as a 256-entry table (lut_md5) to the RGB "
                       "only, never to depth",
        "curve_fit": "binned medians of Isaac's goal picture per grey level of the look-lit MuJoCo "
                     "goal picture (bins with >= 20 pixels), weights sqrt(count), "
                     "scipy.optimize.least_squares, bounds floor [0, 5], log_exposure [-5, 8], "
                     "gamma [0.3, 6]",
        "isaac_root": str(ISAAC),
        "scenes": scenes,
    }
    path = config_path(LOOK)
    path.write_text(json.dumps(cfg, indent=1, sort_keys=False) + "\n")
    print(f"[u4b] wrote {path}")
    return 0


# ---------------------------------------------------------------------- measure
def cmd_one_measure(args) -> int:
    import resource

    import numpy as np

    from vsbench.world import fly

    fly.station_guard(2)
    from vsbench.vs import starts as startsmod
    from vsbench.vs.scene import Pose, pose_to_T
    from vsbench.world.mujoco_scene import MujocoScene

    t0 = time.monotonic()
    with MujocoScene(args.scene, look=args.look) as sc:
        fly.require_software_gl(sc)
        init_s = time.monotonic() - t0
        goal_rgb, goal_depth = sc.render_goal()
        g = sc.goal_pose
        near = startsmod.sample_near(g.position, g.rotation, sc.working_distance, n=1)[0]
        pose = Pose.from_start(near)
        sc.reset(pose)
        near_rgb, near_depth, _ = sc.render()
        times, lut_times = [], []
        for _ in range(4):
            t = time.monotonic()
            rgb, _d = sc.renderer.render(pose_to_T(pose))
            t1 = time.monotonic()
            sc.look.apply(rgb)
            t2 = time.monotonic()
            times.append(t2 - t)
            lut_times.append(t2 - t1)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"measure_{args.scene}_{args.look}.npz"
        np.savez_compressed(path, goal=goal_rgb, near=near_rgb, goal_depth=goal_depth,
                            near_depth=near_depth)
        out = {"scene": args.scene, "look": args.look, "path": str(path),
               "near_start": near["start_id"], "init_s": round(init_s, 3),
               "frame_s": [round(t, 4) for t in times], "frame_s_median": round(float(np.median(times)), 4),
               "curve_s_median": round(float(np.median(lut_times)), 4),
               "rss_peak_mb": round(peak, 1), "goal_frame_md5": sc.goal_frame_md5,
               "goal_depth_md5": hashlib.md5(goal_depth.tobytes()).hexdigest(),
               "near_depth_md5": hashlib.md5(near_depth.tobytes()).hexdigest(),
               "look_record": sc.look_record, "render_mode": sc.render_mode,
               "gl_renderer": sc.renderer.gl_renderer, "capture": sc.goal_capture_record()["look"]}
    print("U4B " + json.dumps(out, default=str), flush=True)
    return 0


def picture_stats(m, i, mask=None) -> dict:
    """MuJoCo ``m`` against Isaac ``i`` (same pose): mean grey, per-channel means, per-channel
    histogram distance (1-D earth mover's distance between the 256-bin histograms, in grey
    levels: how far, on average, the grey levels must move to turn one histogram into the
    other) and the per-pixel mean absolute difference (grey levels)."""
    import numpy as np

    m = m.astype(np.float64)
    i = i.astype(np.float64)
    sel = np.ones(m.shape[:2], bool) if mask is None else mask
    out = {"mean_grey": {"mujoco": round(float(m[sel].mean()), 2), "isaac": round(float(i[sel].mean()), 2)},
           "mean_rgb": {"mujoco": [round(float(v), 2) for v in m[sel].mean(0)],
                        "isaac": [round(float(v), 2) for v in i[sel].mean(0)]}}
    out["mean_grey_diff"] = round(out["mean_grey"]["mujoco"] - out["mean_grey"]["isaac"], 2)
    emd = []
    for c in range(3):
        hm = np.bincount(m[..., c][sel].astype(int), minlength=256) / sel.sum()
        hi = np.bincount(i[..., c][sel].astype(int), minlength=256) / sel.sum()
        emd.append(round(float(np.abs(np.cumsum(hm) - np.cumsum(hi)).sum()), 3))
    out["hist_emd_rgb"] = emd
    out["hist_emd_mean"] = round(float(np.mean(emd)), 3)
    out["pixel_mae"] = round(float(np.abs(m[sel] - i[sel]).mean()), 3)
    return out


def cmd_measure(args) -> int:
    import numpy as np
    from PIL import Image, ImageDraw

    from vsbench.vs.camera import CameraConfig
    from vsbench.vs.labels import object_mask_from_depth
    from vsbench.vs.scene import Pose

    from vsbench.world.fly import vsbench_commit

    commit = vsbench_commit()           # before this command writes anything
    intr = CameraConfig().intrinsics()
    rows, runs = {}, {}
    for s in args.scenes:
        for look in ("raw", LOOK):
            runs[(s, look)] = r = _child(["_measure", s, look])
            print(f"[u4b] {s:18s} {look:13s} frame {r['frame_s_median']:.3f} s (curve "
                  f"{r['curve_s_median'] * 1e3:.1f} ms)  RSS peak {r['rss_peak_mb']:.0f} MB  "
                  f"goal md5 {r['goal_frame_md5']}  [{r['gl_renderer']}]", flush=True)
        a, b = runs[(s, "raw")], runs[(s, LOOK)]
        za, zb = np.load(a["path"]), np.load(b["path"])
        row = {"near_start": a["near_start"],
               "depth_identical_raw_vs_look": {
                   "goal": a["goal_depth_md5"] == b["goal_depth_md5"],
                   "near": a["near_depth_md5"] == b["near_depth_md5"]},
               "cost": {lk: {k: runs[(s, lk)][k] for k in ("frame_s_median", "frame_s", "curve_s_median",
                                                          "rss_peak_mb", "init_s")} for lk in ("raw", LOOK)},
               "goal_frame_md5": {"raw": a["goal_frame_md5"], LOOK: b["goal_frame_md5"]},
               "look_record": b["look_record"], "gl_renderer": b["gl_renderer"],
               "render_mode": b["render_mode"]}
        for view in ("goal", "near"):
            iso, ip = _isaac(s, view)
            if iso is None:
                row[view] = {"isaac": None, "note": f"no Isaac picture at {ip}"}
                continue
            v = {"isaac_png": str(ip), "isaac_md5": _md5(ip),
                 "role": ("fitted (the curve was fitted on this picture)" if view == "goal" else
                          "held out (never seen by the fit): check_baselines' first live frame at "
                          f"the near start {a['near_start']}")}
            v["raw"] = picture_stats(za[view], iso)
            v[LOOK] = picture_stats(zb[view], iso)
            if s in ROOMS:
                depth = zb[f"{view}_depth"].astype(np.float64)
                g = _goal_pose(s)
                obj = object_mask_from_depth(depth, intr, g if view == "goal" else _near_pose(s))
                v["ground"] = {lk: picture_stats(z[view], iso, ~obj)["mean_grey"]
                               for lk, z in (("raw", za), (LOOK, zb))}
                v["objects"] = {lk: picture_stats(z[view], iso, obj)["mean_grey"]
                                for lk, z in (("raw", za), (LOOK, zb))}
            reps = []
            for root in ISAAC_REPEATS:
                rep, rp = _isaac(s, view, root)
                if rep is not None:
                    st = picture_stats(rep, iso)
                    reps.append({"isaac_png": str(rp), "mean_grey_diff": st["mean_grey_diff"],
                                 "hist_emd_mean": st["hist_emd_mean"], "pixel_mae": st["pixel_mae"]})
            if reps:
                v["isaac_repeat_floor"] = reps
            row[view] = v
            print(f"[u4b]   {view:4s} mean grey raw {v['raw']['mean_grey']['mujoco']:.1f} look "
                  f"{v[LOOK]['mean_grey']['mujoco']:.1f} isaac {v['raw']['mean_grey']['isaac']:.1f} | "
                  f"hist EMD raw {v['raw']['hist_emd_mean']:.1f} look {v[LOOK]['hist_emd_mean']:.1f} | "
                  f"pixel MAE raw {v['raw']['pixel_mae']:.1f} look {v[LOOK]['pixel_mae']:.1f}"
                  + (f" | Isaac repeat EMD {[r['hist_emd_mean'] for r in reps]}" if reps else ""),
                  flush=True)
        rows[s] = row

    # the side-by-side sheet: MuJoCo raw | MuJoCo Isaac-look | Isaac
    tile_w, tile_h, head = 640, 360, 24
    sheet_scenes = [s for s in SHEET_SCENES if s in args.scenes]
    sheet = Image.new("RGB", (3 * tile_w, len(sheet_scenes) * (tile_h + head)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, s in enumerate(sheet_scenes):
        y = k * (tile_h + head)
        iso, _ip = _isaac(s, "goal")
        za, zb = np.load(runs[(s, "raw")]["path"]), np.load(runs[(s, LOOK)]["path"])
        st = rows[s]["goal"]
        for j, (img, label) in enumerate((
                (za["goal"], f"{s} - MuJoCo raw (mean grey {st['raw']['mean_grey']['mujoco']:.0f})"),
                (zb["goal"], f"{s} - MuJoCo {LOOK} (mean grey {st[LOOK]['mean_grey']['mujoco']:.0f})"),
                (iso, f"{s} - Isaac RTX, check1 (mean grey {st['raw']['mean_grey']['isaac']:.0f})"))):
            sheet.paste(Image.fromarray(np.asarray(img)).resize((tile_w, tile_h), Image.LANCZOS),
                        (j * tile_w, y + head))
            draw.text((j * tile_w + 6, y + 6), label, fill=(0, 0, 0))
    out = OUT / "u4b_look_vs_isaac.png"
    sheet.save(out, optimize=True)
    print(f"[u4b] wrote {out}")
    # a crop sheet of the drill's shadow edge (raw | look | Isaac), full resolution
    if "industrial1_drill" in args.scenes:
        s = "industrial1_drill"
        za, zb = np.load(runs[(s, "raw")]["path"]), np.load(runs[(s, LOOK)]["path"])
        iso, _ = _isaac(s, "goal")
        ys, xs = slice(300, 620), slice(940, 1360)
        crop = np.concatenate([za["goal"][ys, xs], zb["goal"][ys, xs], iso[ys, xs]], axis=1)
        Image.fromarray(crop).save(OUT / "u4b_drill_shadow_crop.png", optimize=True)
    from vsbench.world.look import config_path, file_md5

    _merge_record("measure", {
        "scenes": rows, "sheet": str(out.relative_to(REPO)), "vsbench_commit": commit,
        "shadow_crop": "results/bridge/u4b_drill_shadow_crop.png (industrial1_drill goal, full "
                       "resolution, rows 300-620, columns 940-1360: raw | look | Isaac)",
        "look_config": str(config_path(LOOK).relative_to(REPO)),
        "look_config_md5": file_md5(config_path(LOOK)),
        "how": "each scene x look in a fresh process (tools/u4b_look.py _measure): MujocoScene, the "
               "goal capture, the near start's frame, 4 redraws of that frame (renderer + curve "
               "timed; the curve alone timed too); statistics against Isaac's pictures (read only)",
        "metrics": picture_stats.__doc__.split("\n", 1)[0] + " " + " ".join(
            picture_stats.__doc__.split("\n", 1)[1].split()),
        "isaac_root": str(ISAAC), "isaac_repeats": [str(p) for p in ISAAC_REPEATS]})
    return 0


def cmd_shadows(args) -> int:
    """The rooms' cast shadows at the goal: mean grey inside them (raw / look / Isaac) and on the
    lit ground around them - what the curve cannot fix (Isaac's floor is a sun-lit term)."""
    import numpy as np
    from scipy import ndimage as ndi

    rows = {}
    for s in ROOMS:
        comp = np.load(CACHE / f"comp_{s}.npz")
        A, D = comp["AMB"].astype(float).mean(2), comp["DIR"].astype(float).mean(2)
        sh = ndi.binary_erosion((D < 0.3 * A) & (A > 8), iterations=4)   # the sun's term gone
        flat = np.abs(D - A) < 3                                          # lit like the ground
        ring = ndi.binary_dilation(sh, iterations=24) & ~ndi.binary_dilation(sh, iterations=8) & flat
        iso, _ = _isaac(s, "goal")
        row = {"shadow_share": round(float(sh.mean()), 4)}
        for lk in ("raw", LOOK):
            z = np.load(CACHE / f"measure_{s}_{lk}.npz")["goal"].astype(float).mean(2)
            row[lk] = {"in_shadow": round(float(z[sh].mean()), 1), "lit_ring": round(float(z[ring].mean()), 1)}
        i = iso.astype(float).mean(2)
        row["isaac"] = {"in_shadow": round(float(i[sh].mean()), 1), "lit_ring": round(float(i[ring].mean()), 1)}
        for k in ("raw", LOOK, "isaac"):
            row[k]["ratio"] = round(row[k]["in_shadow"] / max(row[k]["lit_ring"], 1e-9), 3)
        rows[s] = row
        print(f"[u4b] {s}: shadow/lit ratio raw {row['raw']['ratio']} look {row[LOOK]['ratio']} "
              f"isaac {row['isaac']['ratio']}  (in shadow: raw {row['raw']['in_shadow']} look "
              f"{row[LOOK]['in_shadow']} isaac {row['isaac']['in_shadow']})")
    _merge_record("shadows", {"rooms": rows, "how": "goal pictures; shadow = the sun's term below "
                              "0.3 of the texture (the components of tools/u4b_look.py "
                              "components), eroded 4 px; lit ring = flat lit ground 8-24 px "
                              "around it; mean grey of the three channels"})
    return 0


def _near_pose(scene: str):
    from vsbench.vs import starts as startsmod
    from vsbench.vs.scene import Pose

    g = _goal_pose(scene)
    from vsbench.vs import benchmark

    d = float(benchmark.load_starts(scene, "v2")["goal"]["distance_m"])
    return Pose.from_start(startsmod.sample_near(g.position, g.rotation, d, n=1)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("components")
    c.add_argument("--scenes", nargs="+", default=list(SCENES))
    c.set_defaults(fn=cmd_components)
    o = sub.add_parser("_components")
    o.add_argument("scene")
    o.set_defaults(fn=cmd_one_components)
    f = sub.add_parser("fit")
    f.set_defaults(fn=cmd_fit)
    lt = sub.add_parser("_lit")
    lt.add_argument("scene")
    lt.add_argument("--a", type=float, required=True)
    lt.add_argument("--G", type=float, required=True)
    lt.set_defaults(fn=cmd_one_lit)
    m = sub.add_parser("measure")
    m.add_argument("--scenes", nargs="+", default=list(SCENES))
    m.set_defaults(fn=cmd_measure)
    sh = sub.add_parser("shadows")
    sh.set_defaults(fn=cmd_shadows)
    om = sub.add_parser("_measure")
    om.add_argument("scene")
    om.add_argument("look")
    om.set_defaults(fn=cmd_one_measure)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
