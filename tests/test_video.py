"""Step 3 gate 13: video. A 300-step ``sift_ibvs`` episode on the poster scene becomes an
mp4 that decodes to 150 frames of 960 x 720 under 5 MB. Also: the frame composer for a
method with and without keypoints, the OpenCV fallback writer, and the match sheet.
"""

import cv2
import numpy as np
import pytest

from vsbench.camera import gate_camera
from vsbench.episode import EpisodeConfig, demo_start_poses, run_episode
from vsbench.methods import make_method
from vsbench.replay import render_target
from vsbench.scenes import poster_target_pose
from vsbench.video import (
    VIDEO_SIZE,
    _OpenCVWriter,
    compose_frame,
    match_sheet,
    open_writer,
    write_video,
)


def decode(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cannot open {path}"
    n, shape, first = 0, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first is None:
            first = frame.copy()
        shape = frame.shape
        n += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, shape, fps, first


def test_video_gate(backend, renderer_factory, tmp_path):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    log = run_episode(r, make_method("sift_ibvs"), demo_start_poses(T_t)[9], T_t, EpisodeConfig(max_steps=300, stop_tol=None), "poster")
    assert len(log) == 300
    path = tmp_path / "episode.mp4"
    res = write_video(log, r, path)
    n, shape, fps, first = decode(path)
    size = path.stat().st_size
    print(f"{backend} step3 video: 300-step sift_ibvs episode -> {n} frames of {shape[1]} x {shape[0]} at {fps:.2f} fps, {size / 1e6:.2f} MB, encoder {res.encoder}, {res.summary()}")
    assert n == 150 and shape == (VIDEO_SIZE[1], VIDEO_SIZE[0], 3)
    assert size < 5_000_000
    assert abs(fps - 12.5) < 0.01
    assert res.identical and res.steps == 300
    # the first frame is not blank and carries the match colours
    assert first.std() > 20 and (first[..., 1].astype(int) - first[..., 2].astype(int) > 100).sum() > 200
    # the match sheet for a human: four rows of current | desired at full resolution
    sheet = tmp_path / "matches.png"
    match_sheet(log, r, sheet)
    img = cv2.imread(str(sheet))
    assert img is not None and img.shape == (4 * 480, 2 * 640, 3)


def test_compose_frame_with_and_without_keypoints(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    cfg = EpisodeConfig(max_steps=4, stop_tol=None)
    rgb_des, _, _ = render_target(run_episode(r, make_method("oracle_pbvs"), T_t, T_t, cfg, "poster"), r)
    for name in ("sift_ibvs", "oracle_pbvs"):
        log = run_episode(r, make_method(name), demo_start_poses(T_t)[6], T_t, cfg, "poster")
        rgb, _ = r.render(log.T_wc[3])
        frame = compose_frame(log, 3, rgb, rgb_des, cam, T_t, identical=False)
        assert frame.shape == (720, 960, 3) and frame.dtype == np.uint8
        has_kp = name == "sift_ibvs"
        assert (log.match_count[3] > 0) == has_kp
    with pytest.raises(ValueError):
        write_video(run_episode(r, make_method("oracle_pbvs"), T_t, T_t, EpisodeConfig(max_steps=1, hold_steps=1), "poster"), r, "/nonexistent/x.mp4")


def test_opencv_fallback_writer(tmp_path):
    path = tmp_path / "fallback.mp4"
    w = _OpenCVWriter(path, 12.5)
    rng = np.random.default_rng(0)
    for _ in range(5):
        w.write(rng.integers(0, 255, (720, 960, 3), dtype=np.uint8))
    w.close()
    n, shape, _, _ = decode(path)
    assert n == 5 and shape == (720, 960, 3)
    assert "ffmpeg" in open_writer(tmp_path / "default.mp4", 12.5).name or "OpenCV" in open_writer(tmp_path / "default2.mp4", 12.5).name
