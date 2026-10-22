"""Gate tests are parametrised over backend variants.

Selection: ``VSBENCH_BACKENDS=mujoco-egl,pybullet-tiny pytest`` or ``pytest --backends ...``.
With neither, every variant whose package is installed is tried, and a variant that
cannot open a context in a fresh process is skipped with the reason. Run one variant per
pytest process when comparing MuJoCo and Open3D: both own a GL context.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from vsbench.camera import gate_camera
from vsbench.render.base import (
    BACKENDS,
    default_order,
    make_renderer,
    package_installed,
    probe_backend,
)
from vsbench.stdscenes import STD_SCENES


def pytest_addoption(parser):
    parser.addoption("--backends", default=os.environ.get("VSBENCH_BACKENDS", ""), help="comma-separated variants")


def _selected(config) -> list[str]:
    raw = config.getoption("--backends")
    if raw.strip():
        names = [n.strip() for n in raw.split(",") if n.strip()]
        unknown = [n for n in names if n not in BACKENDS]
        if unknown:
            raise pytest.UsageError(f"unknown backends {unknown}; known {sorted(BACKENDS)}")
        return names
    # CPU variants for this platform, plus nothing that needs a GPU
    return [n for n in default_order() if package_installed(n)]


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        names = _selected(metafunc.config)
        params = []
        for n in names:
            if not package_installed(n):
                params.append(pytest.param(n, marks=pytest.mark.skip(reason=f"{BACKENDS[n][3]} not installed")))
                continue
            ok, why = probe_backend(n)
            if not ok:
                params.append(pytest.param(n, marks=pytest.mark.skip(reason=f"{n} cannot start: {why}")))
            else:
                params.append(pytest.param(n))
        metafunc.parametrize("backend", params, ids=names)


@pytest.fixture(scope="session")
def renderer_factory():
    """``get(backend, scene_name, (w, h))`` builds each renderer once per session."""
    cache = {}

    def get(backend: str, scene_name: str, res=(640, 480)):
        key = (backend, scene_name, tuple(res))
        if key not in cache:
            cache[key] = make_renderer(backend, STD_SCENES[scene_name](), gate_camera(*res))
        return cache[key]

    yield get
    for r in cache.values():
        try:
            r.close()
        except Exception:
            pass


RESOLUTIONS = [(640, 480), (1440, 1080)]


def poses_for(res):
    """20 poses at 640 x 480 and 3 at 1440 x 1080."""
    return 20 if tuple(res) == (640, 480) else 3


# -- step 3 -------------------------------------------------------------------------------
class BlankRenderer:
    """A stand-in ``Renderer`` for loop tests whose method never looks at the image:
    constant grey RGB and a constant 1 m depth, at the gate camera's resolution."""

    name = "blank"

    def __init__(self, cam=None):
        self.cam = cam or gate_camera()
        self.frames = 0

    def render(self, T_wc):
        self.frames += 1
        rgb = np.full((self.cam.height, self.cam.width, 3), 128, dtype=np.uint8)
        depth = np.ones((self.cam.height, self.cam.width), dtype=np.float32)
        return rgb, depth

    def set_light(self, direction, intensity, color=(1.0, 1.0, 1.0)):
        pass

    def close(self):
        pass

    def info(self):
        return {"variant": self.name}
