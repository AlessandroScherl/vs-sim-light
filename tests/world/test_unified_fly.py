"""``python -m vsbench fly``: one command for a published baseline and for your own method.

* routing: a protocol cell goes to ``--arms``, any other name to ``--method``, the other
  flags pass through to ``vsbench.world.fly run`` unchanged, ``--out`` defaults to ``runs``;
* ``--plugin``: a ``.py`` file registers its method (the README's ``examples/my_method.py``),
  which fills the contract and commands a zero twist at the goal;
* the README command itself, flown for three steps on poster2 (software GL only).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("yaml")

import vsbench.__main__ as cli  # noqa: E402
from vsbench.methods import available_methods, make_method  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
#: the scenes whose assets this module needs (tests/conftest.py skips it without them)
NEEDS_SCENES = ("poster2",)
EXAMPLE = REPO / "examples" / "my_method.py"


def test_split_names_routes_protocol_cells_to_arms():
    ids = cli.protocol_ids()
    assert {"vitvs_dinov3", "vitvs_dinov2", "artvs", "sift_ransac", "orb_ransac",
            "akaze_ransac"} <= set(ids)
    arms, methods = cli.split_names(["vitvs_dinov3", "my_ibvs", "sift_ransac", "sift_ibvs"], ids)
    assert arms == ["vitvs_dinov3", "sift_ransac"]
    assert methods == ["my_ibvs", "sift_ibvs"]


def test_fly_argv_passes_flags_through():
    rest = ["--scene", "poster2", "--start-set", "near60", "--limit", "1", "--film"]
    argv = cli.fly_argv(["sift_ransac", "sift_ibvs"], rest, out="runs")
    assert argv == ["run", *rest, "--out", "runs", "--arms", "sift_ransac", "--method", "sift_ibvs"]
    assert "--method" not in cli.fly_argv(["artvs"], rest, out="x")
    assert "--arms" not in cli.fly_argv(["sift_ibvs"], rest, out="x")


def test_main_calls_the_fly_driver(monkeypatch):
    from vsbench.world import fly

    seen = {}
    monkeypatch.setattr(fly, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    rc = cli.main(["fly", "--scene", "poster2", "--method", "vitvs_dinov3", "my_ibvs",
                   "--plugin", str(EXAMPLE), "--start-set", "near60", "--limit", "1",
                   "--param", "gain=0.5"])
    assert rc == 0
    argv = seen["argv"]
    assert argv[0] == "run"
    assert argv[argv.index("--out") + 1] == "runs"
    assert argv[argv.index("--arms") + 1] == "vitvs_dinov3"
    assert argv[argv.index("--method") + 1] == "my_ibvs"
    for flag, value in (("--scene", "poster2"), ("--start-set", "near60"), ("--limit", "1"),
                        ("--param", "gain=0.5")):
        assert argv[argv.index(flag) + 1] == value


def test_plugin_registers_the_example_and_it_fills_the_contract():
    cli.load_plugin(str(EXAMPLE))
    assert "my_ibvs" in available_methods()
    m = make_method("my_ibvs")
    assert m.wants_depth == "dense" and m.privileged is False
    rng = np.random.default_rng(0)
    img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    img = np.ascontiguousarray(np.repeat(np.repeat(img[::4, ::4], 4, 0), 4, 1))   # blocky texture
    depth = np.full((240, 320), 0.5, np.float32)
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]])
    m.configure(K, 320, 240, 1 / 60)
    m.reset()
    m.set_target(img, depth, 0.5)
    twist, info = m.get_control_rate(img, depth)
    assert info["lost"] is False and len(info["kp_cur"]) >= 8
    assert np.allclose(twist, 0.0, atol=1e-9)


def test_unknown_plugin_file_is_refused():
    with pytest.raises(SystemExit):
        cli.load_plugin("no/such/file.py")


def _poster2_texture() -> Path:
    from vsbench.world.build import workspace_root

    return workspace_root() / "assets" / "poster2" / "poster2.png"


@pytest.mark.skipif(os.environ.get("VSBENCH_SOFTWARE_GL") != "1",
                    reason="flies on MuJoCo: software GL only (source tools/cpu_env.sh)")
def test_readme_command_flies(tmp_path):
    if not _poster2_texture().exists():
        pytest.skip(f"poster2's texture is not at {_poster2_texture()}")
    env = dict(os.environ, PYTHONPATH=f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    cmd = [sys.executable, "-m", "vsbench", "fly", "--scene", "poster2", "--method", "my_ibvs",
           "--plugin", str(EXAMPLE), "--start-set", "near60", "--limit", "1", "--max-iter", "3",
           "--film", "--out", str(tmp_path)]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads((tmp_path / "poster2_external_my_ibvs.jsonl").read_text().splitlines()[0])
    assert rec["arm_kind"] == "external" and rec["steps"] == 3
    assert rec["external_method"] == "my_ibvs"
    assert (tmp_path / rec["film"]).is_file()
