#!/usr/bin/env python
"""Cross-variant difference: the same poses rendered by two variants of one backend in
two fresh processes (each variant owns its process), PSNR and max absolute pixel
difference. Writes results/cross_<a>_vs_<b>.json.

    python tools/cross_variant.py --a mujoco-egl --b mujoco-osmesa
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vsbench.sysinfo import system_record  # noqa: E402


def render(variant: str, scene: str, n: int, out: Path, env_extra: dict | None = None) -> dict:
    env = os.environ.copy()
    env.update(env_extra or {})
    cmd = [sys.executable, "-m", "vsbench.render_once", "--backend", variant, "--scene", scene, "--seed", "31", "--count", str(n), "--out", str(out)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"{variant} failed:\n{p.stderr[-2000:]}")
    return json.loads([ln for ln in p.stdout.splitlines() if ln.startswith("{")][-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--env-a", default="", help="KEY=VAL,KEY=VAL for variant a's process")
    ap.add_argument("--env-b", default="")
    ap.add_argument("--scenes", nargs="+", default=["poster", "clutter"])
    ap.add_argument("--poses", type=int, default=10)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    la, lb = a.label_a or a.a, a.label_b or a.b
    env_a = dict(kv.split("=", 1) for kv in a.env_a.split(",") if kv)
    env_b = dict(kv.split("=", 1) for kv in a.env_b.split(",") if kv)
    tmp = Path(a.out_dir) / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in a.scenes:
        fa, fb = tmp / f"cross_{la}_{scene}.npz", tmp / f"cross_{lb}_{scene}.npz"
        ia = render(a.a, scene, a.poses, fa, env_a)
        ib = render(a.b, scene, a.poses, fb, env_b)
        A, B = np.load(fa), np.load(fb)
        psnr = [float(cv2.PSNR(x, y)) for x, y in zip(A["rgb"], B["rgb"])]
        maxabs = [int(np.abs(x.astype(int) - y.astype(int)).max()) for x, y in zip(A["rgb"], B["rgb"])]
        dmax = [float(np.abs(x - y).max()) for x, y in zip(A["depth"], B["depth"])]
        # depth where both variants hit something: the max is dominated by silhouette
        # pixels where one variant sees the object and the other the background
        both = [(x > 0) & (y > 0) for x, y in zip(A["depth"], B["depth"])]
        dp99 = [float(np.percentile(np.abs(x - y)[m], 99)) if m.any() else 0.0 for x, y, m in zip(A["depth"], B["depth"], both)]
        hit_disagree = [float(((x > 0) != (y > 0)).mean()) for x, y in zip(A["depth"], B["depth"])]
        identical = [bool(np.array_equal(x, y)) for x, y in zip(A["rgb"], B["rgb"])]
        rows.append({"scene": scene, "poses": a.poses, "psnr_db": psnr, "psnr_db_min": min(psnr), "max_abs_diff": maxabs, "max_abs_diff_max": max(maxabs), "depth_max_abs_diff_m": max(dmax), "depth_p99_abs_diff_both_hit_m": max(dp99), "hit_mask_disagree_frac": max(hit_disagree), "rgb_identical": all(identical), "renderer_a": ia, "renderer_b": ib})
        print(f"{la} vs {lb} {scene}: PSNR min {min(psnr):.1f} dB, max |diff| {max(maxabs)}, depth max |diff| {max(dmax):.2e} m (p99 where both hit {max(dp99):.2e} m, hit masks disagree on {100 * max(hit_disagree):.3f} % of pixels), identical {all(identical)}")
    rec = {"kind": "cross", "a": a.a, "b": a.b, "label_a": la, "label_b": lb, "system": system_record(), "rows": rows}
    (Path(a.out_dir) / f"cross_{la}_vs_{lb}.json").write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
