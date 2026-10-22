"""The parallel demo runner gives the same table and bit-identical logs as the sequential
one. ``oracle_pbvs`` on the poster scene, ten starts, sequential in this process against
three spawned worker processes with their own renderers; the video of start 10 is written
by whichever worker ran it.
"""

import json

import numpy as np

from vsbench.demo import run_demo, run_demo_parallel
from vsbench.episode import EpisodeConfig, EpisodeLog
from vsbench.methods import make_method

COMPARE = ("start", "translation_cm", "rotation_deg", "status", "steps", "final_error_mm", "final_error_deg", "lost_steps", "error")


def test_parallel_matches_sequential(backend, renderer_factory, tmp_path):
    r = renderer_factory(backend, "poster", (640, 480))
    cfg = EpisodeConfig(max_steps=300)
    seq = run_demo(r, make_method("oracle_pbvs"), "poster", tmp_path / "seq", cfg, video_start=None, quiet=True)
    par = run_demo_parallel(backend, "oracle_pbvs", {}, "poster", tmp_path / "par", cfg, jobs=3, video_start=10, quiet=True)
    assert par["jobs"] == 3 and seq["jobs"] == 1 and par["backend"] == backend
    assert [{k: x[k] for k in COMPARE} for x in par["rows"]] == [{k: x[k] for k in COMPARE} for x in seq["rows"]]
    assert par["converged"] == seq["converged"] == 10
    identical = 0
    for i in range(1, 11):
        a = EpisodeLog.load(tmp_path / "seq" / f"start_{i:02d}.npz")
        b = EpisodeLog.load(tmp_path / "par" / f"start_{i:02d}.npz")
        assert np.array_equal(a.T_wc, b.T_wc) and np.array_equal(a.twist_cmd, b.twist_cmd)
        assert np.array_equal(a.frame_hash, b.frame_hash), f"start {i}: frames differ across processes"
        assert a.meta["target_hash"] == b.meta["target_hash"]
        identical += 1
    assert (tmp_path / "par" / "start_10.mp4").exists() and (tmp_path / "par" / "start_10_matches.png").exists()
    assert not (tmp_path / "seq" / "start_10.mp4").exists()
    rec = json.loads((tmp_path / "par" / "demo.json").read_text())
    assert rec["rows"][9]["video"].endswith("start_10.mp4") and "identical" in rec["rows"][9]["video_replay"]
    print(f"{backend} step3 parallel demo: {identical} of 10 logs bit-identical between 1 and 3 processes (poses, twists, frame hashes); wall {seq['total_wall_s']:.1f} s sequential, {par['total_wall_s']:.1f} s with 3 jobs; renderer {rec['renderer'].get('gl_renderer', '?')}")
