"""Produce the U4 control-parity fixture from RECORDED telemetry. Run once; see
``u4_fixture_provenance.md`` for the exact command and the resulting md5s.

Never imports ``refinevs``. Unlike the U1/U2 fixtures this one does not *run* the
old code - it extracts one episode the old code already flew on the station, from
the committed record

    ../../../docs/results/2026-08-21_n2_admission/gates/hollywood_wide/telemetry.jsonl.gz

written by ``genvs-sim@legacy-pre-cleanup src/eval/baseline_check.py::
record_step_telemetry`` (L1296-1440). Every step of a head episode carries the
exact arrays the solve consumed - ``gu/gv/cu/cv`` (goal / current pixels, full
image), ``conf`` (the head's trust), ``w`` (the least-squares weight, column 0),
``off_du/off_dv`` (the head's sub-patch offsets) - and the commanded twist ``v``
it produced. So the fixture is the control law's own input/output pair, recorded
in the field.

The one thing the record does **not** carry is the per-match depth (only the
median ``z_med``), and the depth is an input to the interaction matrix. On a
planar poster scene it is recoverable in closed form, which is what makes this
fixture possible at all; the reconstruction below is the reviewer's
(``scratchpad/review/r7_repro.py``, which reached p99 relative error 1.1e-4 over
15,064 steps of these files) and is checked here against the recorded ``z_med``:

    goal pose:   R_goal = diag(1, -1, -1), goal_pos = (0, 0, d)   [eval_scenes.py L194-206]
    per step:    R_cur = R_goal @ R_cg^T,  p_cur = goal_pos - R_cur @ d_cam_m
                 where R_cg = Rodrigues(rot_to_goal_deg)          [record_step_telemetry]
    poster:      the world plane z = 0                            [build_poster]
    ray:         m = ((u - cx)/f, (v - cy)/f, 1)
    depth:       Z = -p_cur.z / (m @ R_cur^T).z

Intrinsics come from ``genvs-sim/configs/camera.yaml`` through
``free_flying_scene.camera_info``: fx = W / (2 tan(hFoV/2)) = fy = H / (2 tan 21
deg) = 1406.75, principal point at the image centre.

Regenerating a fixture is a decision, not a fix: it means the port changed a
number (``tests/fixtures/README.md``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
TELEMETRY = (WORKSPACE / "docs/results/2026-08-21_n2_admission/gates"
             / "hollywood_wide/telemetry.jsonl.gz")
OLD_REPO = WORKSPACE / "genvs-sim"

SCENE = "hollywood"
ARM = "head_w_g3_axw@v11"          # the fielded v11-era chain: trust^3, per-axis, rounded
EPISODE = "w2_057"                 # converged (1.651 mm at the cap), plane-clean throughout
STRIDE = 2                         # 600 recorded steps -> 300 stored, whole trajectory

W, H = 1920, 1080
FOV_V_DEG = 42.0
F = H / (2.0 * math.tan(math.radians(FOV_V_DEG / 2.0)))
CX, CY = W / 2.0, H / 2.0
R_GOAL = np.diag([1.0, -1.0, -1.0])


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def rotvec_deg_to_R(rotvec_deg) -> np.ndarray:
    """Rodrigues from the axis-angle-in-degrees vector the record stores."""
    rv = np.asarray(rotvec_deg, dtype=np.float64)
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.eye(3)
    k = rv / ang
    a = math.radians(ang)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def plane_depth(cur_px: np.ndarray, p_cur: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """z-depth of the world plane z = 0 along the ray through each current pixel."""
    m = np.column_stack([(cur_px[:, 0] - CX) / F, (cur_px[:, 1] - CY) / F,
                         np.ones(len(cur_px))])
    return -p_cur[2] / (m @ R_cur.T)[:, 2]


def load_episode(path: Path, arm: str, start_id: str) -> dict:
    with gzip.open(path, "rt") as fh:
        for line in fh:
            ep = json.loads(line)
            meta = ep["meta"]
            if meta.get("refine_mode") == arm and meta.get("start_id") == start_id:
                return ep
    raise SystemExit(f"episode {arm} / {start_id} not found in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", default=EPISODE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or HERE / f"u4_telemetry_{SCENE}_{args.episode}.npz"
    ep = load_episode(TELEMETRY, ARM, args.episode)
    meta = ep["meta"]
    goal_pos = np.asarray(meta["goal_pos"], dtype=np.float64)
    gain = float(meta["gain"])
    steps = [s for s in ep["steps"] if "cu" in s][:: args.stride]

    idx, pts_goal, pts_cur, conf, w_rec, depth, v_rec, feat, t_mm, z_med = ([] for _ in range(10))
    zmed_err = 0.0
    for s in steps:
        goal = np.column_stack([np.asarray(s["gu"], float), np.asarray(s["gv"], float)])
        cur = np.column_stack([np.asarray(s["cu"], float), np.asarray(s["cv"], float)])
        R_cg = rotvec_deg_to_R(s["rot_to_goal_deg"])
        R_cur = R_GOAL @ R_cg.T
        p_cur = goal_pos - R_cur @ np.asarray(s["d_cam_m"], dtype=np.float64)
        z = plane_depth(cur, p_cur, R_cur)
        zmed_err = max(zmed_err, abs(float(np.median(z)) - float(s["z_med"])))
        idx.append(int(s["i"]))
        pts_goal.append(goal)
        pts_cur.append(cur)
        conf.append(np.asarray(s["conf"], float))
        w_rec.append(np.asarray(s["w"], float))
        depth.append(z)
        v_rec.append(np.asarray(s["v"], float))
        feat.append(float(s["feat"]))
        t_mm.append(float(s["t_mm"]))
        z_med.append(float(s["z_med"]))

    arrays = {
        "step_index": np.asarray(idx, dtype=np.int64),
        "pts_goal": np.asarray(pts_goal, dtype=np.float64),
        "pts_cur": np.asarray(pts_cur, dtype=np.float64),
        "conf": np.asarray(conf, dtype=np.float64),
        "w_recorded": np.asarray(w_rec, dtype=np.float64),
        "depth": np.asarray(depth, dtype=np.float64),
        "v_recorded": np.asarray(v_rec, dtype=np.float64),
        "feature_error_recorded": np.asarray(feat, dtype=np.float64),
        "t_mm": np.asarray(t_mm, dtype=np.float64),
        "z_med_recorded": np.asarray(z_med, dtype=np.float64),
    }
    scalars = {
        "scene": SCENE, "arm": ARM, "episode": args.episode, "stride": int(args.stride),
        "gain": gain, "fx": F, "fy": F, "cx": CX, "cy": CY, "width": W, "height": H,
        "conf_gamma": float(meta["conf_gamma"]), "axis_weighting": bool(meta["refine_axw"]),
        "solve_unrounded": bool(meta["refine_ur"]), "margin_weighting": False,
        "n_landmarks": int(arrays["pts_goal"].shape[1]),
        "n_steps_recorded": len([s for s in ep["steps"] if "cu" in s]),
        "terminated_reason": meta["terminated_reason"],
        # the two facts the parity test leans on: no lambda schedule was armed, so
        # the recorded `v` is the control law's own twist and not a scaled one
        "stop_mode": meta["stop_mode"],
        "lam_in_record": any("lam" in s for s in ep["steps"]),
        "t_final_mm": float(ep["steps"][-1]["t_mm"]),
        "zmed_reconstruction_max_err_m": zmed_err,
    }
    np.savez_compressed(out, **arrays, meta=json.dumps(scalars))

    git = subprocess.run(["git", "-C", str(OLD_REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    side = {
        "fixture": out.name,
        "unit": "U4 (refinevs/control.py)",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "md5": md5_file(out),
        "bytes": out.stat().st_size,
        "command": f"python tests/fixtures/{Path(__file__).name} --episode {args.episode}",
        "source_repo_commit": f"genvs-sim {git} (tag legacy-pre-cleanup = 905d42b)",
        "source_record": str(TELEMETRY.relative_to(WORKSPACE)),
        "source_record_md5": md5_file(TELEMETRY),
        "depth_reconstruction": "plane z=0 through the current pixel; see the module docstring",
        **scalars,
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
    }
    side_path = out.with_suffix(".json")
    side_path.write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, md5 {side['md5']})")
    print(f"wrote {side_path}")
    print(f"steps {len(steps)} x {scalars['n_landmarks']} landmarks; "
          f"z_med reconstruction max error {zmed_err:.2e} m")


if __name__ == "__main__":
    main()
