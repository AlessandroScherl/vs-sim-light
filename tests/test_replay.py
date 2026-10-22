"""Step 3 gate 12: replay. Re-render a logged episode in the same GL variant: every frame
hash identical, and the desired view too. The check must also be able to fail: a log with
one edited hash reports exactly that frame. ``python -m vsbench.replay`` runs on a saved
file and exits 0.
"""

import numpy as np

from vsbench import replay as replay_mod
from vsbench.camera import gate_camera
from vsbench.episode import EpisodeConfig, EpisodeLog, demo_start_poses, run_episode
from vsbench.methods import make_method
from vsbench.replay import build_renderer, camera_of, iter_frames, replay_check
from vsbench.scenes import poster_target_pose


def same_camera(a, b) -> bool:
    return np.array_equal(a.K, b.K) and (a.width, a.height, a.near, a.far) == (b.width, b.height, b.near, b.far)


def test_replay_identical(backend, renderer_factory, tmp_path):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    log = run_episode(r, make_method("sift_ibvs"), demo_start_poses(T_t)[7], T_t, EpisodeConfig(max_steps=100, stop_tol=None), "poster")
    path = tmp_path / "episode.npz"
    log.save(path)
    back = EpisodeLog.load(path)
    assert same_camera(camera_of(back), cam)
    seen = []
    res = replay_check(back, r, on_frame=lambda k, rgb, depth, ok: seen.append((k, rgb.shape, depth.shape, ok)))
    print(f"{backend} step3 replay: {res.summary()}; {len(res.mismatches)} of {res.steps} frame hashes differ")
    assert res.identical and res.steps == 100 and res.target_identical and res.mismatches == []
    assert [s[0] for s in seen] == list(range(100)) and all(s[1] == (480, 640, 3) and s[2] == (480, 640) and s[3] for s in seen)
    assert res.backend == backend == res.logged_backend
    # the check can fail: edit one hash
    back._hash[42] ^= 1
    res = replay_check(back, r)
    assert not res.identical and res.mismatches == [42] and res.target_identical
    assert "NOT identical" in res.summary() and "step 42" in res.summary()
    back.meta["target_hash"] = int(back.meta["target_hash"]) ^ 1
    assert not replay_check(back, r).target_identical
    # iter_frames stands on its own (the video writer uses it)
    frames = list(iter_frames(EpisodeLog.load(path), r))
    assert len(frames) == 100 and all(f[3] for f in frames)


def test_replay_command(backend, tmp_path, capsys):
    cam = gate_camera()
    T_t = poster_target_pose(cam)
    # a fresh renderer built from the log's meta, exactly as the command does it
    from conftest import BlankRenderer

    blank = BlankRenderer(cam)
    log = run_episode(blank, make_method("oracle_pbvs"), demo_start_poses(T_t)[0], T_t, EpisodeConfig(max_steps=3, stop_tol=None), "poster")
    log.meta["renderer"] = {"variant": backend}
    log.save(tmp_path / "blank.npz")
    r = build_renderer(EpisodeLog.load(tmp_path / "blank.npz"))
    try:
        assert r.name == backend and same_camera(r.cam, cam)
        # the blank frames were never rendered by MuJoCo, so the replay must say so
        assert replay_mod.main([str(tmp_path / "blank.npz")]) == 1
    finally:
        r.close()
    out = capsys.readouterr().out
    assert "NOT identical" in out and "3 of 3 frames differ" in out and "desired view differs" in out
    # and a real log exits 0
    r = build_renderer(log, backend)
    try:
        real = run_episode(r, make_method("oracle_pbvs"), demo_start_poses(T_t)[0], T_t, EpisodeConfig(max_steps=3, stop_tol=None), "poster")
        real.save(tmp_path / "real.npz")
    finally:
        r.close()
    assert replay_mod.main([str(tmp_path / "real.npz")]) == 0
    out = capsys.readouterr().out
    assert "replay identical: 3 frames" in out
    assert not np.array_equal(real.frame_hash, log.frame_hash)
