"""Step 3 gate 11: log round trip. Save an episode log, load it, arrays and meta identical.

The episode is ``sift_ibvs`` on the poster scene from demo start 7 for 60 steps, so the
log holds real matches (more than 500 per step, thinned to 500), real frame hashes and
call times. The match encoding (``uint16`` pixels times 32) is checked against the raw
keypoints the method reported: every stored match within 1/64 px, inlier flags identical.
"""

import json

import numpy as np
import pytest
from conftest import BlankRenderer

from vsbench.camera import gate_camera
from vsbench.episode import (
    MATCH_ARRAYS,
    MAX_MATCHES,
    STEP_ARRAYS,
    EpisodeConfig,
    EpisodeLog,
    decode_matches,
    demo_start_poses,
    encode_matches,
    frame_hash,
    run_episode,
)
from vsbench.methods import make_method
from vsbench.methods.feature_ibvs import SiftIBVS
from vsbench.scenes import poster_target_pose
from vsbench.se3 import pose_error
from vsbench.trajectory import ARRAYS, Trajectory


class RecordingSift(SiftIBVS):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.infos = []

    def reset(self):
        super().reset()
        self.infos = []

    def get_control_rate(self, current_rgb, current_depth=None):
        twist, info = super().get_control_rate(current_rgb, current_depth)
        self.infos.append({k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in info.items()})
        return twist, info


def test_round_trip(backend, renderer_factory, tmp_path):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    m = RecordingSift()
    log = run_episode(r, m, demo_start_poses(T_t)[6], T_t, EpisodeConfig(max_steps=60, stop_tol=None), "poster")
    assert isinstance(log, EpisodeLog) and len(log) == 60
    path = tmp_path / "episode.npz"
    log.save(path)
    back = EpisodeLog.load(path)
    a, b = log.arrays(), back.arrays()
    assert set(a) == set(ARRAYS) | set(STEP_ARRAYS) | set(MATCH_ARRAYS) == set(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    assert back.meta == log.meta and back.dt == log.dt and back.limits == log.limits
    assert back.meta["status"] == "timeout" and back.meta["method"] == "sift_ibvs" and back.meta["scene"] == "poster"
    # dtypes and shapes of the new columns
    assert a["frame_hash"].dtype == np.uint64 and a["match_px"].dtype == np.uint16 and a["match_inlier"].dtype == bool
    assert a["match_count"].dtype == np.int32 and a["lost"].dtype == bool
    assert a["match_px"].shape == (int(a["match_count"].sum()), 4) and a["match_inlier"].shape == (int(a["match_count"].sum()),)
    assert np.all(a["match_count"] == MAX_MATCHES), "SIFT gives more than 500 matches per step on the poster, thinned to 500"
    assert np.all(a["call_s"] > 0.0) and np.all(np.isfinite(a["error_norm"])) and not a["lost"].any()
    # the true errors are those of the logged poses
    errs = np.array([pose_error(T, T_t) for T in back.T_wc])
    assert np.allclose(errs[:, 0], back.err_t) and np.allclose(errs[:, 1], back.err_r)
    assert back.err_t[0] > back.err_t[-1] and back.err_r[0] > back.err_r[-1]
    # frame hashes: 64-bit, one per step, all distinct (the camera moves every step)
    assert len(set(back.frame_hash.tolist())) == 60 and int(back.meta["target_hash"]) not in set(back.frame_hash.tolist())
    # match encoding against the raw keypoints the method reported
    worst = 0.0
    for k in range(60):
        kc, kd, inl = back.matches(k)
        raw = m.infos[k]
        px, inl_raw = encode_matches(raw)
        c2, d2 = decode_matches(px)
        assert np.array_equal(c2, kc) and np.array_equal(d2, kd) and np.array_equal(inl_raw, inl)
        idx = np.unique(np.rint(np.linspace(0, len(raw["kp_cur"]) - 1, MAX_MATCHES)).astype(int))
        worst = max(worst, float(np.abs(kc - raw["kp_cur"][idx]).max()), float(np.abs(kd - raw["kp_des"][idx]).max()))
        assert np.array_equal(inl, raw["inliers"][idx])
        assert np.all(kc >= 0) and np.all(kc[:, 0] < 640) and np.all(kc[:, 1] < 480)
    with np.load(path) as f:
        files = set(f.files)
        meta = json.loads(str(f["meta"]))
    assert files == {"meta", *ARRAYS, *STEP_ARRAYS, *MATCH_ARRAYS} and meta["format"] == "vsbench.episode/1"
    size = path.stat().st_size
    print(f"{backend} step3 log round trip: {len(back)} rows, {sum(v.nbytes for v in a.values())} bytes of arrays in {len(a)} arrays, file {size} bytes, {int(a['match_count'].sum())} matches stored, worst match quantisation {worst:.4f} px")
    assert worst <= 1.0 / 64.0 + 1e-9
    assert back.meta["call_s_mean"] > 0


def test_encode_matches_edge_cases():
    px, inl = encode_matches({})
    assert px.shape == (0, 4) and inl.shape == (0,)
    px, inl = encode_matches({"kp_cur": np.zeros((3, 2)), "kp_des": np.zeros((2, 2))})
    assert len(px) == 0
    kc = np.array([[10.5, 20.25], [3000.0, -5.0], [np.nan, 1.0]])
    px, inl = encode_matches({"kp_cur": kc, "kp_des": kc[::-1]})
    assert px.dtype == np.uint16 and inl.all() and len(px) == 3
    c, d = decode_matches(px)
    assert np.allclose(c[0], [10.5, 20.25]) and c[1, 0] == 65535 / 32.0 and c[1, 1] == 0.0 and c[2, 0] == 0.0
    kc = np.random.default_rng(0).uniform(0, 600, (1200, 2))
    px, inl = encode_matches({"kp_cur": kc, "kp_des": kc, "inliers": np.arange(1200) % 3 == 0})
    assert len(px) == MAX_MATCHES and 0.3 < inl.mean() < 0.37
    px, inl = encode_matches({"kp_cur": kc, "kp_des": kc, "inliers": np.zeros(5, bool)})
    assert inl.all(), "an inlier array of the wrong length is ignored"


def test_log_without_matches_and_wrong_format(tmp_path):
    r = BlankRenderer(gate_camera())
    T = poster_target_pose(r.cam)
    log = run_episode(r, make_method("oracle_pbvs"), T, T, EpisodeConfig(max_steps=5, stop_tol=None), "poster")
    assert len(log) == 5 and log.match_count.tolist() == [0] * 5 and log.match_px.shape == (0, 4)
    assert np.isfinite(log.error_norm).all() and len(set(log.frame_hash.tolist())) == 1  # a constant image
    assert log.frame_hash[0] == np.uint64(frame_hash(r.render(T)[0]))
    p = tmp_path / "e.npz"
    log.save(p)
    back = EpisodeLog.load(p)
    assert np.array_equal(back.match_px, log.match_px) and back.matches(0)[0].shape == (0, 2)
    # a plain trajectory file is refused, and Trajectory.load still reads an episode's core
    Trajectory(0.04).save(tmp_path / "t.npz")
    with pytest.raises(ValueError):
        EpisodeLog.load(tmp_path / "t.npz")
    core = Trajectory.load(p)
    assert np.array_equal(core.T_wc, log.T_wc)
