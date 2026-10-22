"""Step 3 gate 2: no leak of the truth into a method.

A spy method with ``privileged = False`` runs an episode on the poster scene and its
``observe_truth`` must never be called. The import graph of ``vsbench/methods/`` must
not reach the episode loop, a renderer or a scene: every import statement in the package
is checked statically, and a fresh interpreter that imports ``vsbench.methods`` must not
have loaded any of those modules transitively.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from vsbench.camera import gate_camera
from vsbench.episode import EpisodeConfig, run_episode
from vsbench.scenes import poster_target_pose
from vsbench.se3 import exp_se3

METHODS_DIR = Path(__file__).resolve().parents[1] / "vsbench" / "methods"

# a method may import none of these, directly or through another module
BANNED_PREFIXES = (
    "vsbench.episode",
    "vsbench.render",
    "vsbench.scene",  # SceneSpec and its objects
    "vsbench.scenes",
    "vsbench.stdscenes",
    "vsbench.gates",  # pulls in the scenes
    "vsbench.poses",
    "vsbench.checks",
    "vsbench.check",
    "vsbench.replay",
    "vsbench.video",
    "vsbench.demo",
    "vsbench.smoke",
    "vsbench.render_once",
    "mujoco",
    "pybullet",
    "open3d",
)


def banned(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in BANNED_PREFIXES)


def imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # relative import: resolve against the package
                base = "vsbench.methods" + ("." + base if base else "")
            out.append(base)
            out += [f"{base}.{a.name}" for a in node.names]
    return out


def test_import_graph_static():
    files = sorted(METHODS_DIR.glob("*.py"))
    assert files, "no method modules found"
    offenders = []
    for f in files:
        for mod in imports_of(f):
            if banned(mod):
                offenders.append(f"{f.name}: {mod}")
    print(f"step3 leak import graph: {len(files)} modules under vsbench/methods, {len(offenders)} banned imports")
    assert not offenders, offenders


def test_import_graph_transitive():
    """What ``import vsbench.methods`` loads beyond ``import vsbench`` itself (the package
    ``__init__`` exports SceneSpec, FreeCamera and Trajectory) must contain nothing banned."""
    code = (
        "import sys, json\n"
        "def mods():\n"
        "    return sorted(m for m in sys.modules if m.startswith('vsbench') or m.split('.')[0] in ('mujoco', 'pybullet', 'open3d'))\n"
        "import vsbench\n"
        "base = mods()\n"
        "import vsbench.methods\n"
        "print('MODS ' + json.dumps({'base': base, 'after': mods()}))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=METHODS_DIR.parents[1])
    assert p.returncode == 0, p.stderr[-2000:]
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("MODS ")][-1]
    d = json.loads(line[len("MODS "):])
    added = sorted(set(d["after"]) - set(d["base"]))
    leaked = [m for m in added if banned(m)]
    heavy = [m for m in d["after"] if m.split(".")[0] in ("mujoco", "pybullet", "open3d") or m.startswith(("vsbench.render", "vsbench.episode", "vsbench.stdscenes", "vsbench.scenes"))]
    print(f"step3 leak transitive: import vsbench.methods adds {len(added)} modules to the {len(d['base'])} of import vsbench; {len(leaked)} banned: {leaked}; renderers, loop and scenes loaded: {heavy}")
    assert not leaked
    assert not heavy


class Spy:
    """Not privileged: any call to observe_truth is recorded and is a failure."""

    name = "spy"
    wants_depth = "dense"
    privileged = False

    def __init__(self):
        self.truth_calls = 0
        self.steps = 0

    def configure(self, K, width, height, dt):
        pass

    def reset(self):
        pass

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None):
        pass

    def observe_truth(self, T_wc, T_wc_target):
        self.truth_calls += 1

    def get_control_rate(self, current_rgb, current_depth=None):
        self.steps += 1
        return np.zeros(6), {"lost": False}


def test_spy_never_sees_the_truth(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    T_0 = T_t @ exp_se3([0.05, 0.0, -0.1, 0.0, 0.0, np.deg2rad(15.0)])
    spy = Spy()
    log = run_episode(r, spy, T_0, T_t, EpisodeConfig(max_steps=20), "poster")
    print(f"{backend} step3 leak spy: {spy.steps} control calls, {spy.truth_calls} observe_truth calls, status {log.meta['status']}")
    assert spy.steps == 20 and len(log) == 20
    assert spy.truth_calls == 0
    assert log.meta["status"] == "timeout" and log.meta["privileged"] is False
    assert np.allclose(log.final_pose(), T_0)  # a zero twist moves nothing
