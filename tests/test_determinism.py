"""Gate 4: the same pose twice in one process, and once in each of two fresh processes,
gives identical arrays."""

import os
import subprocess
import sys

import numpy as np

from vsbench.poses import sample_poses


def _same(a, b, what):
    """Assertion message with the number of differing pixels and the largest difference.

    With ``VSBENCH_DUMP_DIR`` set, both arrays are saved there for inspection.
    """
    dump = os.environ.get("VSBENCH_DUMP_DIR")
    if dump:
        os.makedirs(dump, exist_ok=True)
        np.savez(os.path.join(dump, what.replace(" ", "_").replace(",", "") + ".npz"), a=a, b=b)
    if np.array_equal(a, b):
        return True
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    n = int((d.reshape(d.shape[0], d.shape[1], -1).max(-1) > 0).sum())
    raise AssertionError(f"{what}: {n} pixels differ, max |diff| {d.max():.3g}")


def test_same_process(backend, renderer_factory):
    r = renderer_factory(backend, "clutter", (640, 480))
    T = sample_poses(7, 1, np.eye(4), dist=(0.5, 1.2), max_slant_deg=50.0)[0]
    rgb1, d1 = r.render(T)
    # render something else in between so cached buffers cannot hide a difference
    r.render(sample_poses(8, 1, np.eye(4))[0])
    rgb2, d2 = r.render(T)
    assert _same(rgb1, rgb2, "rgb, same pose twice in one process")
    assert _same(d1, d2, "depth, same pose twice in one process")


def test_fresh_processes(backend, renderer_factory, tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"run{k}.npz"
        cmd = [sys.executable, "-m", "vsbench.render_once", "--backend", backend, "--scene", "clutter", "--seed", "7", "--out", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), timeout=600)
        assert p.returncode == 0, p.stderr[-2000:]
        outs.append(np.load(out))
    assert _same(outs[0]["rgb"][0], outs[1]["rgb"][0], "rgb, two fresh processes")
    assert _same(outs[0]["depth"][0], outs[1]["depth"][0], "depth, two fresh processes")
    # and the in-process render agrees with the fresh processes
    r = renderer_factory(backend, "clutter", (640, 480))
    rgb, d = r.render(outs[0]["T_wc"][0])
    assert _same(rgb, outs[0]["rgb"][0], "rgb, in-process vs fresh process")
    assert _same(d, outs[0]["depth"][0], "depth, in-process vs fresh process")
