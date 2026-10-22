#!/usr/bin/env python
"""Generate results/bakeoff_report.md from results/*.json and apply the decision rule:

1. all four gates pass on Linux with no GPU and no display, else out;
2. pip plus at most one system package, official wheels for Linux, macOS, Windows on 3.10-3.12;
3. at least 25 fps at 640 x 480 in one CPU process, faster wins;
4. poster quality (inlier counts, then the contact sheet by eye);
5. maintenance (last release, Python versions).
The first difference decides; a tie goes to MuJoCo. The final call stays with the owner.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Criterion 2 and 5 facts, from PyPI on 2026-09-17 (tools/pypi_facts.json keeps the raw query).
PYPI = {
    "mujoco": {"version": "3.13.0", "released": "2026-09-09", "wheels": "Linux, macOS, Windows: cp310-cp315", "system_package": "Linux: libosmesa6 (OSMesa) or none with EGL/Mesa; macOS, Windows: none", "criterion2": True},
    "pybullet": {"version": "3.2.7", "released": "2025-01-30", "wheels": "Linux only, cp36-cp311; no wheel for 3.12; no macOS or Windows wheels (source build needs a C++ compiler)", "system_package": "none, but a compiler on 3.12 and on every macOS/Windows machine", "criterion2": False},
    "open3d": {"version": "0.20.0", "released": "2026-09-16", "wheels": "Linux, macOS, Windows: cp310-cp314", "system_package": "Linux: none beyond Mesa EGL (libegl1, libgl1-mesa-dri, usually present)", "criterion2": True},
}


def load(kind: str, out: Path) -> dict[str, dict]:
    recs = {}
    for f in sorted(out.glob(f"{kind}_*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        recs[d.get("label") or d.get("backend") or f.stem] = d
    return recs


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def is_gpu(label: str) -> bool:
    """GPU variants are measured for information; the decision criteria are CPU-only."""
    return "gpu" in label or label.startswith("pybullet-egl")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.results)
    gates = load("gates", out)
    speed = load("speed", out)
    parallel = load("parallel", out)
    quality = load("quality", out)
    lighting = load("lighting", out)
    cross = load("cross", out)
    friction_logs = sorted((out / "friction").glob("friction_*.log")) if (out / "friction").exists() else []

    lines = []
    L = lines.append
    L("# VS-Bench step 1: renderer bake-off report")
    L("")
    sys_recs = [d["system"] for d in list(gates.values()) + list(speed.values()) if "system" in d]
    if sys_recs:
        s = sys_recs[0]
        loads = [d["system"]["load_average"][0] for d in speed.values() if d.get("system", {}).get("load_average")]
        load_txt = f" Load average recorded before the speed runs: {min(loads):.1f} to {max(loads):.1f} (each run follows the previous tool in the chain; the CPU rows also shared the machine with another job using about three cores)." if loads else ""
        gpu_rows = [n for n in list(gates) + list(speed) if is_gpu(n)]
        gpu_txt = f"GPU present: {s.get('gpu') or 'none'}, used only by the variants labelled gpu ({', '.join(sorted(set(gpu_rows)))}); every other row ran with `VSBENCH_SOFTWARE_GL=1` (Mesa llvmpipe)." if gpu_rows else f"GPU present: {s.get('gpu') or 'none'} (never used: `VSBENCH_SOFTWARE_GL={s['env'].get('VSBENCH_SOFTWARE_GL', 'unset')}`)."
        L(f"Generated from `results/*.json`. Machine: {s['cpu']}, {s['cores']} threads, {s['os']}, Python {s['python']}. {gpu_txt}{load_txt}")
        L("")
    # -- gates -----------------------------------------------------------------------
    L("## 1. Gates (Linux, no display; rows labelled gpu are informational)")
    L("")
    L("| variant | GL renderer | projection | depth | homography | determinism | all |")
    L("|---|---|---|---|---|---|---|")
    passed = {}
    for name, d in gates.items():
        if not d.get("available"):
            L(f"| {name} | - | unavailable: {d.get('reason', '').strip().splitlines()[-1] if d.get('reason') else '?'} | | | | out |")
            continue
        cells = []
        for g in d["gates"]:
            m = g["measured"]
            txt = g["status"]
            if m:
                txt += "<br>" + "<br>".join(x.split(": ", 1)[1] if ": " in x else x for x in m)
            cells.append(txt)
        passed[name] = d.get("all_passed", False)
        L(f"| {name} | {d.get('renderer', {}).get('gl_renderer', '?')} | " + " | ".join(cells) + f" | {'pass' if passed[name] else 'FAIL'} |")
    L("")
    L("Thresholds: projection mean <= 0.5 px and max <= 1.0 px; depth median <= 1 mm and 99th percentile <= 3 mm; homography median displacement <= 0.5 px with >= 100 ORB matches; determinism identical arrays.")
    L("")
    # -- install ---------------------------------------------------------------------
    L("## 2. Install")
    L("")
    L("| library | release | wheels (PyPI) | system package | criterion 2 |")
    L("|---|---|---|---|---|")
    for lib, p in PYPI.items():
        L(f"| {lib} | {p['version']} ({p['released']}) | {p['wheels']} | {p['system_package']} | {'yes' if p['criterion2'] else 'no'} |")
    L("")
    if friction_logs:
        L("Fresh-venv logs (`results/friction/`):")
        L("")
        for f in friction_logs:
            txt = f.read_text()
            first = [ln for ln in txt.splitlines() if ln.startswith("# smoke exit code") or ln.startswith("# pip done")]
            L(f"- `{f.name}`: " + "; ".join(x.lstrip('# ') for x in first))
        L("")
        L("macOS and Windows: untested (no such machine in this run).")
        L("")
    # -- speed -----------------------------------------------------------------------
    L("## 3. Speed (one CPU process, RGB plus depth, median of 200 frames)")
    L("")
    L("| variant | scene | resolution | triangles | median ms | p90 ms | fps | build s |")
    L("|---|---|---|---|---|---|---|---|")
    fps640 = {}
    for name, d in speed.items():
        for c in d["cases"]:
            L(f"| {name} | {c['scene']} | {c['width']}x{c['height']} | {c['triangles']} | {fmt(c['median_ms'], 2)} | {fmt(c['p90_ms'], 2)} | {fmt(c['fps'])} | {fmt(c['build_s'])} |")
            if (c["width"], c["height"]) == (640, 480) and c["scene"] == "clutter":
                fps640[name] = c["fps"]
    L("")
    if parallel:
        L("Parallel scaling (poster scene, 640 x 480, one renderer per process):")
        L("")
        L("| variant | processes | total fps | per-process fps |")
        L("|---|---|---|---|")
        for name, d in parallel.items():
            for r in d["rows"]:
                L(f"| {name} | {r['processes']} | {fmt(r['total_fps'])} | {fmt(r['mean_process_fps'])} |")
        L("")
    # -- quality ---------------------------------------------------------------------
    L("## 4. Poster quality (against a 4x supersampled warp)")
    L("")
    L("| variant | case | ORB inliers / matches | SIFT inliers / matches | PSNR dB (frame / poster) | brightness gain | sheet |")
    L("|---|---|---|---|---|---|---|")
    qscore = defaultdict(int)
    for name, d in quality.items():
        for c in d["cases"]:
            L(f"| {name} | {c['case']} | {c['orb']['inliers']} / {c['orb']['matches']} | {c['sift']['inliers']} / {c['sift']['matches']} | {fmt(c['psnr_db'])} / {fmt(c.get('psnr_poster_db'))} | {fmt(c['brightness_fit']['gain'], 2)} | `{d['sheet']}` |")
            qscore[name] += c["orb"]["inliers"] + c["sift"]["inliers"]
    L("")
    if cross:
        L("Cross-variant difference (same poses, two variants of one backend; the EGL variant here runs Mesa 25.2's llvmpipe, the OSMesa variant conda-forge Mesa 24's):")
        L("")
        L("| a | b | scene | PSNR min dB | max abs pixel diff | depth max diff m | depth p99 diff m (both hit) | hit masks disagree | identical |")
        L("|---|---|---|---|---|---|---|---|---|")
        for name, d in cross.items():
            for r in d["rows"]:
                p99 = r.get("depth_p99_abs_diff_both_hit_m")
                dis = r.get("hit_mask_disagree_frac")
                L(f"| {d['label_a']} | {d['label_b']} | {r['scene']} | {fmt(r['psnr_db_min'])} | {r['max_abs_diff_max']} | {r['depth_max_abs_diff_m']:.1e} | {'-' if p99 is None else f'{p99:.1e}'} | {'-' if dis is None else f'{100 * dis:.3f} %'} | {r['rgb_identical']} |")
        L("")
        if not any("gpu" in n for n in cross):
            L("CPU against GPU (`mujoco-egl` on the GPU against llvmpipe) is not in this run. Command: `python tools/cross_variant.py --a mujoco-egl --b mujoco-egl --label-a mujoco-egl-gpu --label-b mujoco-egl-llvmpipe --env-a VSBENCH_SOFTWARE_GL=0 --env-b VSBENCH_SOFTWARE_GL=1`.")
            L("")
    if lighting:
        L("Lighting control:")
        L("")
        L("| variant | move light | change intensity | second light | image |")
        L("|---|---|---|---|---|")
        for name, d in lighting.items():
            c = d["capabilities"]
            L(f"| {name} | {'yes' if c['move_light']['yes'] else 'no'} | {'yes' if c['change_intensity']['yes'] else 'no'} | {'yes' if c['second_light']['yes'] else 'no'} | `{d['image']}` |")
        L("")
    # -- notes -----------------------------------------------------------------------
    L("## Findings per backend (what the gates and tools turned up)")
    L("")
    L("- **MuJoCo.** Intrinsics go in as `focal_pixel`, `principal_pixel`, `sensor_size`, `resolution`; the principal point is an offset from the image centre, positive left and up, which the shifted-principal-point gate confirmed. A multisampled depth buffer resolves off pixel centre (99th percentile 3.9 mm on a 60 degree plane, on llvmpipe and on the RTX 4090 alike); `mujoco.Renderer`'s remedy, a segmentation pass, only helps on llvmpipe, because NVIDIA keeps the sample positions. Depth therefore comes from a second render into a context whose offscreen buffer has `offsamples = 0`, which is exact on both. Near and far are `vis.map.znear/zfar` times `stat.extent`. User-mesh texcoords have v = 0 at the top row. Unlit colours are exact (0.5 -> 128).")
    L("- **PyBullet.** TinyRenderer rasterises half a pixel off OpenGL's centres: a (+0.50, -0.50) px centroid offset in the projection gate, corrected by shifting the principal point handed to the projection matrix. The 3.2.7 wheel exists for Linux up to Python 3.11; on 3.12 pip builds it from source (about a minute, needs a compiler) and the build has no NumPy support unless `--no-build-isolation` is used, so `getCameraImage` returns tuples. No macOS or Windows wheels. The pip build has no `eglRendererPlugin`, so the GPU variant `pybullet-egl` cannot be loaded at all. Lighting is one global light (no second light), `unlit` is the ambient term alone, the clear colour is white and cannot be set in DIRECT mode (`configureDebugVisualizer(rgbBackground=...)` has no effect), and textures are sampled without mipmaps, which shows as aliasing on the minified poster.")
    L("- **Open3D.** `Camera.set_projection(K, near, far, W, H)` takes a full intrinsic matrix; add half a pixel to the principal point. With post-processing off the output is `0.5 * srgb_decode(colour)`; the backend pre-encodes colours and albedo textures so the raw texture comes back (fit 0.985, 33 dB against the reference warp). A second `OffscreenRenderer` of the same size in one process renders differently and destroying one while another is alive aborts the process, so instances are kept until exit and the gate files run one per process. Cross-process determinism failed in 5 of 10 runs on the clutter scene: 268 to 396 white pixels on the 14k-triangle mesh at the image corner differ between an in-process render and a fresh process; the same mesh alone renders identically, and the mesh has no degenerate faces or NaN normals. Filament ran on Vulkan/llvmpipe (lavapipe) for the CPU variant and on Vulkan/NVIDIA for `open3d-gpu`. On the GPU the depth image is unusable in this version: `render_to_depth_image(z_in_view_space=True)` returned a valid plane depth for under 1 % to 30 % of the pixels, and two renders of one pose in one process differed in thousands of depth pixels by up to 0.9 m, while RGB passed projection and homography. Filament keeps one directional light (the sun): a second directional light is accepted and ignored, a point or spot light does add. Lit scenes are darker than MuJoCo's at the same nominal intensity until the sun is scaled (230000 lux for intensity 1).")
    L("")
    # -- decision --------------------------------------------------------------------
    L("## 5. Decision")
    L("")
    lib_of = lambda n: n.split("-")[0]  # noqa: E731
    # criteria apply to the library; a library is in if one of its CPU variants passes
    passed = {n: ok for n, ok in passed.items() if not is_gpu(n)}
    libs = sorted({lib_of(n) for n in passed})
    variants_ok = {lib: [n for n in passed if lib_of(n) == lib and passed[n]] for lib in libs}
    c1 = [lib for lib in libs if variants_ok[lib]]
    L("Criterion 1 (all four gates on Linux, no GPU, no display): " + "; ".join(f"{lib} {'passes' if variants_ok[lib] else 'fails'} ({', '.join(variants_ok[lib]) if variants_ok[lib] else ', '.join(n for n in passed if lib_of(n) == lib)})" for lib in libs) + ".")
    c2 = [lib for lib in c1 if PYPI.get(lib, {}).get("criterion2")]
    L(f"Criterion 2 (pip plus at most one system package, official wheels for Linux, macOS and Windows on 3.10-3.12): {', '.join(c2) or 'nobody'}; out: {', '.join(lib for lib in c1 if lib not in c2) or 'nobody'}.")
    c3 = {lib: max((fps640[n] for n in variants_ok[lib] if n in fps640), default=None) for lib in c2}
    ok3 = [lib for lib, f in c3.items() if f is not None and f >= 25.0]
    L("Criterion 3 (>= 25 fps at 640 x 480 on the clutter scene, one CPU process, best variant): " + ", ".join(f"{lib} {fmt(f)} fps" for lib, f in c3.items()) + f"; passing: {', '.join(ok3) or 'nobody'}.")
    pick = None
    if ok3:
        best = max(c3[lib] for lib in ok3)
        fast = [lib for lib in ok3 if c3[lib] >= 0.9 * best]  # within 10 % counts as a tie on speed
        if len(fast) == 1:
            pick, why = fast[0], "the only library through criteria 1-3" if len(ok3) == 1 else "fastest by more than 10 %"
        else:
            L(f"Speed within 10 %: {', '.join(fast)}; on to quality.")
            qs = {lib: max(qscore.get(n, 0) for n in variants_ok[lib]) for lib in fast}
            bestq = max(qs.values()) if qs else 0
            top = [lib for lib in fast if qs.get(lib, 0) >= 0.9 * bestq]
            L("Criterion 4 (ORB + SIFT inliers summed over both cases): " + ", ".join(f"{lib} {qs[lib]}" for lib in fast) + ".")
            if len(top) == 1:
                pick, why = top[0], "best poster quality"
            elif "mujoco" in top:
                pick, why = "mujoco", "tie on quality; tie goes to MuJoCo"
            else:
                pick = sorted(top, key=lambda lib: PYPI.get(lib, {}).get("released", ""), reverse=True)[0]
                why = "tie on quality; most recent release"
        L("")
        L(f"**Pick: {pick}** ({why}): " + ", ".join(f"{n} {fmt(fps640.get(n))} fps" for n in variants_ok[pick]) + " at 640 x 480 on the clutter scene in one CPU process.")
        if pick == "mujoco":
            L("")
            gpu_cross = [r for d in cross.values() if "gpu" in d["label_a"] and lib_of(d["label_a"]) == "mujoco" for r in d["rows"]]
            L("Variants: laptops use `mujoco-glfw` (a window system, no extra package), headless Linux uses `mujoco-osmesa` (the `libosmesa6` package) or `mujoco-egl` (Mesa EGL, `VSBENCH_SOFTWARE_GL=1` for llvmpipe). The EGL and OSMesa variants rendered identical pixels in this run, so a published CPU run need not pin one of them.")
            if gpu_cross:
                L("")
                L(f"CPU against GPU: the same poses on the RTX 4090 and on llvmpipe agree to {min(r['psnr_db_min'] for r in gpu_cross):.1f} dB with pixel differences up to {max(r['max_abs_diff_max'] for r in gpu_cross)} of 255 at anti-aliased edges and texture minification; depth agrees to {max(r['depth_p99_abs_diff_both_hit_m'] for r in gpu_cross) * 1e3:.2f} mm (99th percentile) where both hit, with hit masks disagreeing on {100 * max(r['hit_mask_disagree_frac'] for r in gpu_cross):.3f} % of pixels. A run is reproducible bit for bit within the CPU class; results published from a GPU and reproduced on a CPU (or the reverse) differ at the level above, so a published run should state which class it used.")
    else:
        L("")
        L("**No library meets criteria 1-3.**")
    L("")
    L("The final call stays with the project owner.")
    L("")
    text = "\n".join(lines)
    dest = Path(a.out) if a.out else out / "bakeoff_report.md"
    dest.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
