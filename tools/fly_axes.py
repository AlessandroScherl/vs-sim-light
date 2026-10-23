#!/usr/bin/env python
"""The axis sheet: fly each twist axis alone for one second and show what the image does.

One row per axis at 0.1 m/s or 10 deg/s from a seeded start pose in the clutter scene:
the start frame, the frame after 1 s (25 steps of 0.04 s through ``FreeCamera``), and
the flow arrows, predicted by ``exact_flow`` with the rendered depth in green and measured
by tracking Shi-Tomasi corners frame to frame with Lucas-Kanade in red. Writes
``results/step2_axes.png`` and ``results/step2_axes.json`` (the numbers behind each row).

    VSBENCH_SOFTWARE_GL=1 python tools/fly_axes.py --backend mujoco-egl
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
from vsbench.gates import (  # noqa: E402
    depth_edge_mask,
    flow_agreement,
    shi_tomasi_corners,
    track_lk,
)
from vsbench.interaction import exact_flow  # noqa: E402
from vsbench.motion import FreeCamera  # noqa: E402
from vsbench.poses import sample_poses  # noqa: E402
from vsbench.render.base import make_renderer, package_installed, probe_backend  # noqa: E402
from vsbench.stdscenes import STD_SCENES  # noqa: E402
from vsbench.sysinfo import system_record  # noqa: E402
from vsbench.trajectory import Trajectory  # noqa: E402

AXES = ["vx", "vy", "vz", "wx", "wy", "wz"]
WHAT = {
    "vx": "camera slides right, image moves left",
    "vy": "camera slides down, image moves up",
    "vz": "camera moves forward, image expands",
    "wx": "camera tilts up, image moves down",
    "wy": "camera pans right, image moves left",
    "wz": "camera rolls clockwise (seen from behind), image turns anticlockwise",
}
V_SPEED = 0.1  # m/s
W_SPEED = np.deg2rad(10.0)  # rad/s
DT = 0.04
SECONDS = 1.0
MARGIN = 5
SCALE = 0.5  # the sheet is written at half resolution (about 2 MB instead of 7)


def pick_mujoco_backend() -> str:
    for name in ("mujoco-egl", "mujoco-osmesa", "mujoco-glfw"):
        if package_installed(name) and probe_backend(name)[0]:
            return name
    sys.exit("no MuJoCo variant starts here (tried mujoco-egl, mujoco-osmesa, mujoco-glfw)")


def tile(img):
    """A frame at the sheet's scale."""
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def label(img, text, y=16):
    out = img.copy()
    cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def arrows(img, pts, pred, meas, n_max=80):
    """Predicted flow as a green arrow, measured as a red arrow with a red dot at its end.

    The two agree to a fraction of a pixel, so the red arrow sits under the green one;
    the dot at the measured end is what shows the agreement. ``img`` is a full-size
    frame; the drawing happens at the sheet's scale.
    """
    out = (tile(img).astype(np.float64) * 0.55).astype(np.uint8)
    idx = np.linspace(0, len(pts) - 1, min(n_max, len(pts))).astype(int) if len(pts) else []
    for i in idx:
        p = tuple(np.rint(pts[i] * SCALE).astype(int))
        q = tuple(np.rint((pts[i] + pred[i]) * SCALE).astype(int))
        m = tuple(np.rint((pts[i] + meas[i]) * SCALE).astype(int))
        cv2.arrowedLine(out, p, q, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(out, p, m, (255, 0, 0), 1, cv2.LINE_AA, tipLength=0.2)
        cv2.circle(out, m, 2, (255, 0, 0), -1, cv2.LINE_AA)
    return out


def fly_axis(cam, r, T0, axis: int) -> tuple[dict, np.ndarray]:
    xi = np.zeros(6)
    xi[axis] = V_SPEED if axis < 3 else W_SPEED
    free = FreeCamera(T0, dt=DT)
    log = Trajectory(DT, free.limits, meta={"axis": AXES[axis]})
    rgb0, depth0 = r.render(free.T_wc)
    pts = shi_tomasi_corners(rgb0, max_corners=1500)
    bad = depth_edge_mask(depth0, margin=MARGIN)
    iu = np.clip(np.rint(pts[:, 0]).astype(int), 0, cam.width - 1)
    iv = np.clip(np.rint(pts[:, 1]).astype(int), 0, cam.height - 1)
    keep = ~bad[iv, iu] & (depth0[iv, iu] > 0)
    pts, Z = pts[keep], depth0[iv, iu][keep].astype(np.float64)
    # track frame to frame through the second (a corner lost on the way is dropped), then
    # once more from the start frame straight to the end frame with the chained position
    # as the initial guess, so the measured flow carries no accumulated drift
    cur = pts.copy()
    alive = np.ones(len(pts), dtype=bool)
    prev = rgb0
    steps = int(round(SECONDS / DT))
    for _ in range(steps):
        t, pose = free.t, free.T_wc
        applied = free.step(xi)
        log.append(t, pose, xi, applied)
        rgb, _ = r.render(free.T_wc)
        nxt, ok = track_lk(prev, rgb, cur)
        alive &= ok
        cur = np.where(alive[:, None], nxt, cur)
        prev = rgb
    rgb1 = prev
    direct, ok = track_lk(rgb0, rgb1, pts, init=cur)
    alive &= ok
    cur = np.where(alive[:, None], direct, cur)
    pred = exact_flow(cam, pts, Z, xi, SECONDS)
    inside = np.all(np.isfinite(pred), axis=1)
    p_end = pts + np.nan_to_num(pred)
    inside &= (p_end[:, 0] >= MARGIN) & (p_end[:, 0] <= cam.width - 1 - MARGIN)
    inside &= (p_end[:, 1] >= MARGIN) & (p_end[:, 1] <= cam.height - 1 - MARGIN)
    sel = alive & inside
    meas = cur[sel] - pts[sel]
    stats = flow_agreement(meas, pred[sel])
    m_len, r_len = log.path_length()
    stats.update(
        {
            "axis": AXES[axis],
            "speed": float(xi[axis]),
            "unit": "m/s" if axis < 3 else "rad/s",
            "steps": steps,
            "n_corners": int(len(pts)),
            "path_m": m_len,
            "path_rad": r_len,
            "what": WHAT[AXES[axis]],
        }
    )
    speed_txt = f"{V_SPEED:.1f} m/s" if axis < 3 else f"{np.degrees(W_SPEED):.0f} deg/s"
    row = np.hstack(
        [
            label(tile(rgb0), f"{AXES[axis]} = +{speed_txt}: t = 0"),
            label(tile(rgb1), f"t = {SECONDS:.0f} s: {WHAT[AXES[axis]].split(', ')[1]}"),
            label(
                arrows(rgb0, pts[sel], pred[sel], meas),
                f"predicted (green), tracked (red): EPE {stats['median_epe_px']:.2f} px, {stats['n']} pts",
            ),
        ]
    )
    return stats, row


def main(argv=None) -> int:
    from vsbench.glenv import ensure_software_gl

    ensure_software_gl()   # software GL by default; a GPU context only with VSBENCH_ALLOW_GPU=1
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default=None, help="a MuJoCo variant; default: the first that starts")
    ap.add_argument("--scene", default="clutter", choices=sorted(STD_SCENES))
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    backend = a.backend or pick_mujoco_backend()
    cam = gate_camera(640, 480)
    r = make_renderer(backend, STD_SCENES[a.scene](), cam)
    # nearly upright so the poster reads as a picture; the gates use rolled poses
    T0 = sample_poses(a.seed, 1, np.eye(4), dist=(0.6, 1.0), max_slant_deg=40.0, target_jitter=0.05, max_roll_deg=15.0)[0]
    rows, stats = [], []
    for axis in range(6):
        s, row = fly_axis(cam, r, T0, axis)
        stats.append(s)
        rows.append(row)
        print(
            f"{AXES[axis]:2s} +{s['speed']:.3f} {s['unit']:5s}: {s['n']} tracked of {s['n_corners']} corners, "
            f"median EPE {s['median_epe_px']:.3f} px (p90 {s['p90_epe_px']:.3f}), median cos {s['median_cos']:.4f}, "
            f"median flow {s['median_pred_px']:.1f} px, path {s['path_m']:.3f} m {np.degrees(s['path_rad']):.2f} deg"
        )
    info = getattr(r, "info", lambda: {})()
    r.close()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / "step2_axes.png"
    cv2.imwrite(str(sheet), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    rec = {
        "kind": "step2_axes",
        "backend": backend,
        "scene": a.scene,
        "seed": a.seed,
        "dt": DT,
        "seconds": SECONDS,
        "system": system_record(),
        "renderer": info,
        "sheet": sheet.name,
        "axes": stats,
    }
    (out / "step2_axes.json").write_text(json.dumps(rec, indent=1))
    print(f"wrote {sheet} and {out / 'step2_axes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
