#!/usr/bin/env python
"""Lighting control: move the light, change its intensity, add a second light. Yes or no
per capability from whether the image changes the way it should, plus an image strip.

    python tools/lighting_check.py --backend mujoco-egl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vsbench.camera import gate_camera  # noqa: E402
from vsbench.poses import look_at  # noqa: E402
from vsbench.render.base import make_renderer  # noqa: E402
from vsbench.stdscenes import lighting_scene  # noqa: E402
from vsbench.sysinfo import system_record  # noqa: E402


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    from vsbench.glenv import ensure_software_gl

    ensure_software_gl()   # software GL by default; a GPU context only with VSBENCH_ALLOW_GPU=1
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    name = a.label or a.backend
    cam = gate_camera(640, 480)
    T = look_at((0.0, -0.7, 0.6), (0.0, 0.0, 0.08), (0.0, 0.0, 1.0))
    r = make_renderer(a.backend, lighting_scene(), cam)
    base, _ = r.render(T)
    r.set_light((-0.5, 0.0, -1.0), 1.0)  # light from the other side
    moved, _ = r.render(T)
    r.set_light((0.5, 0.0, -1.0), 0.4)  # back, dimmer
    dim, _ = r.render(T)
    info = getattr(r, "info", lambda: {})()
    r.close()
    # the two-light scene is a second renderer: give it its own process
    code = (
        "import sys, numpy as np; from vsbench.camera import gate_camera; from vsbench.render.base import make_renderer; "
        "from vsbench.scene import Light; from vsbench.stdscenes import lighting_scene; "
        f"r = make_renderer({a.backend!r}, lighting_scene((Light((0.5, 0.0, -1.0), 1.0), Light((-0.5, 0.0, -1.0), 1.0))), gate_camera(640, 480)); "
        f"rgb, _ = r.render(np.array({T.tolist()!r})); np.save(sys.argv[1], rgb)"
    )
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "two.npy"
        p = subprocess.run([sys.executable, "-c", code, str(f)], capture_output=True, text=True, env=os.environ.copy(), timeout=900)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        two = np.load(f)

    def mean_lum(img):
        return float(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean())

    def diff(x, y):
        return float(np.abs(x.astype(int) - y.astype(int)).mean())

    # the sphere is left of centre: its left half should brighten when the light moves
    sphere = (slice(150, 330), slice(120, 300))
    left_half, right_half = (slice(150, 330), slice(120, 210)), (slice(150, 330), slice(210, 300))

    def lr_ratio(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
        return float(g[left_half].mean() / max(g[right_half].mean(), 1e-6))

    res = {
        "move_light": {"yes": diff(base, moved) > 2.0 and lr_ratio(moved) != lr_ratio(base), "mean_abs_diff": diff(base, moved), "left_right_ratio_before": lr_ratio(base), "left_right_ratio_after": lr_ratio(moved)},
        "change_intensity": {"yes": mean_lum(dim) < 0.9 * mean_lum(base), "mean_lum_before": mean_lum(base), "mean_lum_after": mean_lum(dim)},
        "second_light": {"yes": mean_lum(two) > 1.05 * mean_lum(base) and diff(two, base) > 2.0, "mean_lum_one": mean_lum(base), "mean_lum_two": mean_lum(two)},
    }
    for k, v in res.items():
        print(f"{name} {k}: {'yes' if v['yes'] else 'NO'}  {json.dumps({kk: round(vv, 2) for kk, vv in v.items() if kk != 'yes'})}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    strip = np.hstack([label(base, "one light"), label(moved, "moved"), label(dim, "dimmer"), label(two, "two lights")])
    cv2.imwrite(str(out / f"lighting_{name}.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    (out / f"lighting_{name}.json").write_text(json.dumps({"kind": "lighting", "backend": a.backend, "label": name, "system": system_record(), "renderer": info, "capabilities": res, "image": f"lighting_{name}.png"}, indent=1))
    del sphere
    return 0


if __name__ == "__main__":
    sys.exit(main())
