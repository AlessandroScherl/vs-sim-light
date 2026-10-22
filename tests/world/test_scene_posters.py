"""U1 gates (docs/PORT_PLAN.md section 4): the D435i camera in MuJoCo, the posters behind
vs-sim's ``Scene`` contract.

1. projection - 13 unlit markers through the rendered D435i camera at 1920 x 1080;
2. depth      - the poster plate and backdrop, analytic z, from the three goals, six
                ``narrow20`` starts and a 1.29 m view (cluttered3's goal distance);
3. convention - the hollywood plate drawn by the copied mock and by ``MujocoScene`` from
                the goal and three ``near60`` starts: the predicted half-pixel shift;
4. contract   - ``isinstance(scene, Scene)`` and ``check_scene_goal`` on all three posters,
                plus the goal cache, determinism and the truth seam.

Every number lands in a gate JSON: ``results/bridge/u1_camera_gates.json`` (the committed
record) only when ``VSBENCH_WRITE_GATES=1``, else a pytest temp file, so a plain run never
rewrites a tracked file. CPU only: ``source tools/cpu_env.sh`` first
(``VSBENCH_SOFTWARE_GL=1``, two threads, nice 19); without software GL requested the whole
module is skipped, so it can never open a GPU context.

    source tools/cpu_env.sh
    VSBENCH_WRITE_GATES=1 $VSBENCH_PY -m pytest tests/world/test_scene_posters.py -q
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vsbench.glenv import software_gl_requested  # noqa: E402
from vsbench.world.camera import HALF_PIXEL, d435i_camera_model, d435i_intrinsics  # noqa: E402

pytestmark = pytest.mark.skipif(
    not software_gl_requested(),
    reason="renders: needs VSBENCH_SOFTWARE_GL=1 (source tools/cpu_env.sh) so no GPU is touched")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "bridge" / "u1_camera_gates.json"
POSTERS = ("hollywood", "poster2", "poster3")
RESULTS: dict = {}


def write_gates() -> bool:
    return os.environ.get("VSBENCH_WRITE_GATES", "").strip() == "1"


@pytest.fixture(scope="module", autouse=True)
def _write_results(tmp_path_factory):
    yield
    if not RESULTS:
        return
    out = OUT if write_gates() else tmp_path_factory.mktemp("gates") / OUT.name
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(RESULTS)
    old["unit"] = "U1 - MujocoScene for the posters + the D435i gates (docs/PORT_PLAN.md section 4)"
    old["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
    print(f"[gates] wrote {out}")


def _scene(name):
    from vsbench.world.mujoco_scene import MujocoScene

    if not software_gl_requested():      # belt and braces beside the module mark
        pytest.skip("MujocoScene renders: needs VSBENCH_SOFTWARE_GL=1")
    scene = MujocoScene(name)
    if scene.render_mode != "mujoco-llvmpipe":
        scene.close()
        pytest.fail(f"MuJoCo opened {scene.render_mode!r}, not llvmpipe - refusing to render")
    return scene


# --------------------------------------------------------------- 1. projection
def sphere_image_centre(cam, T_wc, centre_w, radius):
    """The exact centroid of a sphere's image: the centre of the ellipse cut by the cone
    tangent to the sphere (``d^T (C C^T - (|C|^2 - r^2) I) d = 0``), in pixels.

    A sphere's image is not centred on its centre's projection: the ellipse is pushed
    outward by ``~ f tan(theta) (r / d)^2 / cos^2(theta)``, ~0.8 px at the edge of the
    D435i's 69 deg field for a 12 mm marker at 0.5 m.
    """
    from vsbench.reference import invert

    T_cw = invert(T_wc)
    C = T_cw[:3, :3] @ np.asarray(centre_w, float) + T_cw[:3, 3]
    M = np.outer(C, C) - (C @ C - radius**2) * np.eye(3)
    Ki = np.linalg.inv(cam.K)
    Q = Ki.T @ M @ Ki
    A, b = Q[:2, :2], Q[:2, 2]
    return -np.linalg.solve(A, b)


def test_projection_gate():
    from vsbench.gates import detect_markers, projection_poses
    from vsbench.reference import project
    from vsbench.render.base import make_renderer
    from vsbench.stdscenes import MARKER_POINTS, MARKER_RADIUS, marker_scene

    cam = d435i_camera_model()
    r = make_renderer("mujoco-egl", marker_scene(), cam)
    plain, exact, signed_exact, signed_960 = [], [], [], []
    try:
        for T in projection_poses(cam, 6, seed=11):
            rgb, _ = r.render(T)
            cents, _cols = detect_markers(rgb)
            assert len(cents) >= len(MARKER_POINTS)
            px, _z = project(cam, T, MARKER_POINTS)
            ell = np.array([sphere_image_centre(cam, T, p, MARKER_RADIUS) for p in MARKER_POINTS])
            for e_plain, e_exact in zip(px, ell, strict=True):
                j = int(np.argmin(np.linalg.norm(cents - e_exact, axis=1)))
                c = cents[j]
                assert np.linalg.norm(c - e_exact) < 6.0
                plain.append(float(np.linalg.norm(c - e_plain)))
                exact.append(float(np.linalg.norm(c - e_exact)))
                signed_exact.append(c - e_exact)
                signed_960.append(c - (e_exact + HALF_PIXEL))   # the mock's (960, 540) point
    finally:
        r.close()
    plain, exact = np.array(plain), np.array(exact)
    se, s960 = np.mean(signed_exact, axis=0), np.mean(signed_960, axis=0)
    RESULTS["projection"] = {
        "camera": {"fx": cam.fx, "cx": cam.cx, "cy": cam.cy, "wh": [cam.width, cam.height]},
        "n_markers": int(len(exact)), "n_poses": 6,
        "vs_exact_sphere_centroid_px": {"median": float(np.median(exact)), "max": float(exact.max()),
                                        "mean_signed_uv": se.round(4).tolist()},
        "vs_centre_projection_px": {"median": float(np.median(plain)), "max": float(plain.max())},
        "mean_signed_uv_if_principal_point_were_960_540": s960.round(4).tolist(),
        "gate": "median <= 0.15 px, max <= 0.5 px against vsbench.reference.project of the "
                "marker centres (the plan's reference) and against the exact centroid of each "
                "sphere's image",
    }
    print(json.dumps(RESULTS["projection"], indent=1))
    for e in (plain, exact):
        assert np.median(e) <= 0.15
        assert e.max() <= 0.5


# -------------------------------------------------------------------- 2. depth
def poster_depth_truth(cam, T_wc, spec_raw):
    """Analytic z of the nearest of the plate (z = 0) and the backdrop (z = -0.02)."""
    from vsbench.reference import plane_zdepth
    from vsbench.scene import rigid

    plate = plane_zdepth(cam, T_wc, np.eye(4), size=(float(spec_raw["width_m"]),
                                                      float(spec_raw["height_m"])))
    bd = spec_raw["backdrop"]
    back = plane_zdepth(cam, T_wc, rigid(t=(0, 0, -0.02)), size=(bd["size_m"], bd["size_m"]))
    return np.where(np.isfinite(plate), plate, back)


def test_depth_gate():
    from vsbench.vs import benchmark
    from vsbench.vs.scene import Pose, pose_to_T

    cam = d435i_camera_model()
    per_view, all_err = [], []
    far_pose = Pose(position=[0.0, 0.0, benchmark.GOAL_DISTANCE_M["cluttered3"]],
                    rotation=np.diag([1.0, -1.0, -1.0]))
    for name in POSTERS:
        with _scene(name) as scene:
            views = [("goal", scene.goal_pose)]
            views += [(s["start_id"], Pose.from_start(s))
                      for s in benchmark.select_starts(name, "narrow20", limit=2)]
            if name == "hollywood":
                views.append(("far_1.2853m", far_pose))
            for tag, pose in views:
                scene.reset(pose)
                _rgb, depth, _p = scene.render()
                truth = poster_depth_truth(cam, pose_to_T(pose), scene.spec.raw)
                ok = np.isfinite(truth) & (depth > 0)
                e = np.abs(depth[ok].astype(np.float64) - truth[ok])
                all_err.append(e)
                per_view.append({"scene": name, "view": tag,
                                 "distance_m": float(np.median(truth[np.isfinite(truth)])),
                                 "median_mm": float(np.median(e) * 1e3),
                                 "p99_mm": float(np.percentile(e, 99) * 1e3),
                                 "max_mm": float(e.max() * 1e3),
                                 "valid_share": float(ok.mean()),
                                 "missed_px": int((~(depth > 0)).sum())})
    e = np.concatenate(all_err)
    RESULTS["depth"] = {"views": per_view, "median_mm": float(np.median(e) * 1e3),
                        "p99_mm": float(np.percentile(e, 99) * 1e3), "n_px": int(e.size),
                        "gate": "median <= 1 mm, p99 <= 3 mm (all views pooled and each view)"}
    print(json.dumps(RESULTS["depth"], indent=1))
    assert np.median(e) <= 1e-3 and np.percentile(e, 99) <= 3e-3
    for v in per_view:
        assert v["median_mm"] <= 1.0 and v["p99_mm"] <= 3.0, v


# --------------------------------------------------------------- 3. convention
def _grad_corr(a, b) -> float:
    import cv2

    def g(img):
        x = img[..., 1].astype(np.float32)
        return np.concatenate([cv2.Sobel(x, cv2.CV_32F, 1, 0).ravel(),
                               cv2.Sobel(x, cv2.CV_32F, 0, 1).ravel()])

    return float(np.corrcoef(g(a), g(b))[0, 1])


def _phase(a, b, win) -> tuple[float, float]:
    """``cv2.phaseCorrelate`` on copies: with a window it multiplies its float inputs in
    place."""
    import cv2

    (dx, dy), _resp = cv2.phaseCorrelate(a.copy(), b.copy(), win)
    return float(dx), float(dy)


def _ecc(a, b) -> tuple[float, float]:
    """The translation that maps ``a`` onto ``b`` (``b(x + d) = a(x)``), by OpenCV's ECC
    (gradient-based, sub-pixel, no bias toward the integer grid); a 16 px border is left out."""
    import cv2

    H, W = a.shape
    mask = np.zeros((H, W), np.uint8)
    mask[16:-16, 16:-16] = 1
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    _cc, warp = cv2.findTransformECC(a.astype(np.float32), b.astype(np.float32), warp,
                                     cv2.MOTION_TRANSLATION, crit, mask, 1)
    # findTransformECC returns W with template(x) ~ input(W x): input = b, template = a
    return float(warp[0, 2]), float(warp[1, 2])


def _shift(img, dx, dy):
    import cv2

    H, W = img.shape[:2]
    return cv2.warpAffine(img, np.float64([[1, 0, dx], [0, 1, dy]]), (W, H),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


#: known exact image shifts (px) for the ECC calibration row
CALIBRATION_SHIFTS = ((-0.5, -0.5), (-0.3, -0.3), (-0.5, 0.0), (-1.5, -1.5))


def test_half_pixel_convention_against_the_copied_mock():
    """MuJoCo renders with Isaac's principal point (959.5, 539.5); the copied mock puts it
    on pixel 960's centre. So the whole MuJoCo picture sits half a pixel left of and above
    the mock's, at every pose (the two pinholes differ by a pure image translation).

    Measured at each view:

    * ``raw``         - ``cv2.phaseCorrelate(mock, mujoco)``, the plan's reading (predicted
                        (-0.5, -0.5)). Reported, not gated: an exact half pixel is
                        phaseCorrelate's degenerate case - the correlation peak splits evenly
                        between two integer bins and its centroid reads anything from -0.17
                        to -0.83 (``calibration``: the mock picture itself moved by exactly
                        (-0.5, -0.5) reads (-0.17, -0.17) or (-0.17, -0.83));
    * ``ecc``         - the same shift by OpenCV's ECC (gradient based, unbiased): gated at
                        (-0.5, -0.5) +- 0.1 px;
    * ``residual``    - phaseCorrelate of MuJoCo moved back by the predicted half pixel: ~0,
                        where phaseCorrelate has no bias; gated at 0.1 px;
    * ``control_960`` - phaseCorrelate of MuJoCo rendered with the mock's principal point
                        (960, 540): ~0; gated at 0.1 px.

    ``ecc_calibration`` - the instrument itself: the mock's goal picture of the hollywood
    texture moved by known exact shifts (``CALIBRATION_SHIFTS``, bicubic) and read back by
    ECC and by phaseCorrelate; ECC's bias is gated at 0.1 px per axis, phaseCorrelate's is
    reported (it is what makes the half pixel its degenerate case).
    """
    import cv2

    from vsbench.camera import CameraModel
    from vsbench.render.base import make_renderer
    from vsbench.vs import benchmark
    from vsbench.vs.scene import MockPlaneScene, Pose, pose_to_T

    rows = []
    with _scene("hollywood") as scene:
        tex = scene.scene_spec.objects[0].material.texture
        mock = MockPlaneScene(texture=np.array(tex), render_scale=1)
        rc = scene.render_camera
        cam960 = CameraModel.from_focal(rc.width, rc.height, rc.fx, rc.fy, cx=rc.cx + HALF_PIXEL,
                                        cy=rc.cy + HALF_PIXEL, near=rc.near, far=rc.far)
        r960 = make_renderer("mujoco-egl", scene.scene_spec, cam960)
        views = [("goal", scene.goal_pose)]
        views += [(s["start_id"], Pose.from_start(s))
                  for s in benchmark.select_starts("hollywood", "near60", limit=3)]
        H, W = scene.intrinsics.height, scene.intrinsics.width
        win = cv2.createHanningWindow((W, H), cv2.CV_64F)
        try:
            for tag, pose in views:
                scene.reset(pose)
                mock.reset(pose)
                mj, _d, _p = scene.render()
                mk, _dm, _pm = mock.render()
                m960, _ = r960.render(pose_to_T(pose))
                a = cv2.cvtColor(mk, cv2.COLOR_RGB2GRAY).astype(np.float64)
                b = cv2.cvtColor(mj, cv2.COLOR_RGB2GRAY).astype(np.float64)
                c = cv2.cvtColor(m960, cv2.COLOR_RGB2GRAY).astype(np.float64)
                raw = _phase(a, b, win)
                calib = _phase(a, _shift(a, -HALF_PIXEL, -HALF_PIXEL), win)
                resid = _phase(a, _shift(b, HALF_PIXEL, HALF_PIXEL), win)
                ctrl = _phase(a, c, win)
                ecc = _ecc(a, b)
                rows.append({"view": tag,
                             "raw_px": [round(v, 4) for v in raw],
                             "calibration_px": [round(v, 4) for v in calib],
                             "raw_minus_calibration_px": [round(raw[i] - calib[i], 4) for i in (0, 1)],
                             "ecc_px": [round(v, 4) for v in ecc],
                             "residual_after_half_pixel_px": [round(v, 4) for v in resid],
                             "control_960_px": [round(v, 4) for v in ctrl],
                             "plan_literal_raw_within_0.1_of_-0.5":
                                 bool(max(abs(raw[0] + HALF_PIXEL), abs(raw[1] + HALF_PIXEL)) <= 0.1),
                             "grad_corr_green": round(_grad_corr(mk, mj), 4),
                             "mean_grey_mujoco": round(float(b.mean()), 3),
                             "mean_grey_mock": round(float(a.mean()), 3)})
            # the calibration row: known shifts of the mock's goal picture, read back
            mock.reset(scene.goal_pose)
            mk, _dm, _pm = mock.render()
            a = cv2.cvtColor(mk, cv2.COLOR_RGB2GRAY).astype(np.float64)
            calib_rows = []
            for dx, dy in CALIBRATION_SHIFTS:
                b = _shift(a, dx, dy)
                ecc, ph = _ecc(a, b), _phase(a, b, win)
                calib_rows.append({"true_px": [dx, dy],
                                   "ecc_px": [round(v, 4) for v in ecc],
                                   "ecc_bias_px": [round(ecc[0] - dx, 4), round(ecc[1] - dy, 4)],
                                   "phase_px": [round(v, 4) for v in ph],
                                   "phase_bias_px": [round(ph[0] - dx, 4), round(ph[1] - dy, 4)]})
        finally:
            r960.close()
    RESULTS["convention"] = {
        "views": rows, "predicted_shift_px": [-HALF_PIXEL, -HALF_PIXEL],
        "ecc_calibration": {"image": "the copied mock's goal picture of the hollywood texture "
                                     "(grey, 1920 x 1080), moved by cv2.warpAffine INTER_CUBIC",
                            "rows": calib_rows, "gate": "|ecc_bias| <= 0.1 px per axis"},
        "gate": "per view and axis: |ecc - (-0.5)| <= 0.1, |residual| <= 0.1, |control_960| <= 0.1, "
                "raw < -0.25 (told apart from 0); green gradient corr >= 0.8",
        "note": "an exact half-pixel shift is cv2.phaseCorrelate's degenerate case (the peak splits "
                "between two bins; a synthetic exact -0.5 shift of the mock reads -0.17 or -0.83), "
                "so the plan's literal 'raw within 0.1 of -0.5' is reported per view and the shift "
                "is gated by ECC plus the two phaseCorrelate null tests"}
    print(json.dumps(RESULTS["convention"], indent=1))
    for c in calib_rows:
        assert max(abs(v) for v in c["ecc_bias_px"]) <= 0.1, c
    for r in rows:
        for i in (0, 1):
            assert abs(r["ecc_px"][i] + HALF_PIXEL) <= 0.1, r
            assert abs(r["residual_after_half_pixel_px"][i]) <= 0.1, r
            assert abs(r["control_960_px"][i]) <= 0.1, r
            assert r["raw_px"][i] < -0.25, r
        assert r["grad_corr_green"] >= 0.8, r


# ----------------------------------------------------------------- 4. contract
def test_contract_and_check_scene_goal_on_every_poster():
    from vsbench.vs import benchmark
    from vsbench.vs.scene import Scene

    rows = []
    for name in POSTERS:
        with _scene(name) as scene:
            assert isinstance(scene, Scene)
            assert scene.intrinsics == d435i_intrinsics()
            for which in ("v2", "near"):
                payload = benchmark.load_starts(name, which)
                benchmark.check_scene_goal(scene, payload)      # raises on a drift
                g = payload["goal"]
                rows.append({"scene": name, "file": which, "pass": True,
                             "distance_m": scene.working_distance,
                             "file_distance_m": float(g["distance_m"]),
                             "d_distance_mm": abs(scene.working_distance - float(g["distance_m"])) * 1e3,
                             "d_position_mm": float(np.linalg.norm(scene.goal_pose.position
                                                                   - np.asarray(g["cam_pos"], float)) * 1e3)})
            assert scene.render_mode == "mujoco-llvmpipe"
    RESULTS["contract"] = {"check_scene_goal": rows,
                           "render_mode": "mujoco-llvmpipe",
                           "gate": "isinstance(scene, vsbench.vs.scene.Scene); check_scene_goal on "
                                   "both pinned start files of all three posters"}
    print(json.dumps(rows, indent=1))


def test_goal_cache_determinism_reortho_and_truth():
    """The goal is drawn once and served from the cache; two draws of one pose are the
    same bytes; ``apply`` is the copied integration (+ Isaac's re-orthogonalisation);
    the truth seam maps a goal pixel to where the plane geometry puts it."""
    from vsbench.vs import benchmark
    from vsbench.vs.scene import MockPlaneScene, Pose, integrate_optical_velocity

    with _scene("hollywood") as scene:
        scene.reset(scene.goal_pose)
        g1, d1, _ = scene.render()
        g2, _, _ = scene.render()
        assert scene.n_renders == 1 and np.array_equal(g1, g2)
        rec = scene.goal_capture_record()
        assert rec["goal_frame_md5"] and rec["goal_render_warmup"] == 1
        assert rec["goal_render_warmup_protocol"] == 8 and rec["goal_render_warmup_reason"]
        assert rec["goal_settle_series"]["n_captures"] == 1
        assert rec["render_mode"] == "mujoco-llvmpipe"
        start = benchmark.select_starts("hollywood", "near60", limit=1)[0]
        pose = Pose.from_start(start)
        scene.reset(pose)
        a, da, _ = scene.render()
        b, db, _ = scene.render()
        assert np.array_equal(a, b) and np.array_equal(da, db)
        # the copied integration rule, and the re-orthogonalisation at step 300
        tw = np.array([0.01, -0.02, 0.005, 0.1, -0.05, 0.2])
        scene.reset(pose)
        scene.apply(tw, 1 / 60)
        ref = integrate_optical_velocity(pose, tw, 1 / 60)
        assert np.array_equal(scene.pose.position, ref.position)
        assert np.array_equal(scene.pose.rotation, ref.rotation)
        for _ in range(299):
            scene.apply(tw, 1 / 60)
        R = scene.pose.rotation
        assert np.abs(R.T @ R - np.eye(3)).max() < 1e-12 and scene._reortho_counter == 0
        # the truth seam against the mock's closed form (same plate, same poses)
        scene.reset(pose)
        scene.render()
        pts = np.array([[400.0, 300.0], [960.0, 540.0], [1500.0, 800.0]])
        uv, valid = scene.gt_correspondence(pts, pose)
        mock = MockPlaneScene(intrinsics=d435i_intrinsics())
        uv_m, valid_m = mock.gt_correspondence(pts, pose)
        assert valid.all() and valid_m.all()
        err = float(np.abs(uv - uv_m).max())
        RESULTS["truth_seam_vs_mock_px"] = err
        assert err < 0.02   # the goal depth is float32 at 0.434 m
    RESULTS["goal_cache"] = {"n_renders_for_two_goal_calls": 1, "goal_frame_md5": rec["goal_frame_md5"],
                             "deterministic_redraw": True}
