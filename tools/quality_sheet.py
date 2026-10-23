#!/usr/bin/env python
"""Poster quality: the poster at 1.5 m with a 70 degree slant, and at 3x minification,
against a 4x supersampled warp reduced with INTER_AREA. Writes a contact sheet PNG and
ORB / SIFT inlier counts to results/quality_<backend>.json.

    python tools/quality_sheet.py --backend mujoco-egl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vsbench.camera import gate_camera  # noqa: E402
from vsbench.gates import inlier_counts  # noqa: E402
from vsbench.poses import look_at, roll  # noqa: E402
from vsbench.reference import texture_homography  # noqa: E402
from vsbench.render.base import make_renderer  # noqa: E402
from vsbench.stdscenes import POSTER_SIZE, load_poster, poster_scene  # noqa: E402
from vsbench.sysinfo import system_record  # noqa: E402


def cases(cam):
    """Named poses: slanted at 1.5 m, and fronto-parallel at 3x minification."""
    # 70 degree slant about the poster's vertical axis, 1.5 m from its centre
    s = np.deg2rad(70.0)
    eye = np.array([1.5 * np.sin(s), 0.0, 1.5 * np.cos(s)])
    T_slant = look_at(eye, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    # 3x minification: the 1280 px wide picture spans 1280 / 3 px in the image
    tex_w = load_poster().shape[1]
    d = cam.fx * POSTER_SIZE[0] / (tex_w / 3.0)
    T_min = roll(look_at((0.0, 0.0, d), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), 0.0)
    return [("slant70_1.5m", T_slant), ("minify3x", T_min)]


def reference(cam, T, tex, ss=4):
    """Warp at ss x resolution, reduce with INTER_AREA: a box-filtered ideal render."""
    cam_ss = cam.scaled(ss)
    th, tw = tex.shape[:2]
    H = texture_homography(cam_ss, T, np.eye(4), POSTER_SIZE, tw, th)
    big = cv2.warpPerspective(tex, H, (cam_ss.width, cam_ss.height), flags=cv2.INTER_LINEAR)
    return cv2.resize(big, (cam.width, cam.height), interpolation=cv2.INTER_AREA)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
    tex = load_poster()
    r = make_renderer(a.backend, poster_scene(), cam)
    rows, results = [], []
    for cname, T in cases(cam):
        rgb, _ = r.render(T)
        ref = reference(cam, T, tex)
        counts = inlier_counts(rgb, ref)
        m = ref.max(2) > 0
        fit = np.polyfit(ref[m].astype(float).ravel(), rgb[m].astype(float).ravel(), 1)
        psnr = float(cv2.PSNR(rgb, ref))
        # PSNR over the poster's own pixels, so a backend's clear colour does not count
        err = (rgb[m].astype(float) - ref[m].astype(float)) ** 2
        psnr_poster = float(10 * np.log10(255.0**2 / max(err.mean(), 1e-12)))
        diff = np.clip(np.abs(rgb.astype(int) - ref.astype(int)) * 4, 0, 255).astype(np.uint8)
        rows.append(np.hstack([label(ref, f"reference {cname}"), label(rgb, name), label(diff, "|diff| x4")]))
        results.append({"case": cname, "orb": counts["orb"], "sift": counts["sift"], "psnr_db": psnr, "psnr_poster_db": psnr_poster, "brightness_fit": {"gain": float(fit[0]), "offset": float(fit[1])}})
        print(f"{name} {cname}: ORB {counts['orb']['inliers']}/{counts['orb']['matches']} SIFT {counts['sift']['inliers']}/{counts['sift']['matches']} PSNR {psnr:.1f} dB (poster {psnr_poster:.1f} dB) gain {fit[0]:.2f}")
    info = getattr(r, "info", lambda: {})()
    r.close()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / f"quality_{name}.png"
    cv2.imwrite(str(sheet), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    (out / f"quality_{name}.json").write_text(json.dumps({"kind": "quality", "backend": a.backend, "label": name, "system": system_record(), "renderer": info, "sheet": sheet.name, "cases": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
