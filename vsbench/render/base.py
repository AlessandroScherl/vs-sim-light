"""The Renderer protocol and the factory over the three backends.

Backend variants are named ``<library>-<context>``. Every variant is optional at import
time: ``make_renderer`` imports the backend module only when asked for it, so a missing
package only breaks that variant.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from typing import Protocol, runtime_checkable

import numpy as np

from vsbench.camera import CameraModel
from vsbench.scene import SceneSpec


@runtime_checkable
class Renderer(Protocol):
    name: str

    def render(self, T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rgb uint8 (H, W, 3) and depth float32 (H, W) z-depth in metres, 0 where nothing is hit."""

    def set_light(self, direction, intensity: float, color=(1.0, 1.0, 1.0)) -> None: ...

    def close(self) -> None: ...


# name -> (module, class, constructor kwargs, pip package that must be importable)
BACKENDS: dict[str, tuple[str, str, dict, str]] = {
    "mujoco-egl": ("vsbench.render.mujoco_backend", "MujocoRenderer", {"gl": "egl"}, "mujoco"),
    "mujoco-osmesa": ("vsbench.render.mujoco_backend", "MujocoRenderer", {"gl": "osmesa"}, "mujoco"),
    "mujoco-glfw": ("vsbench.render.mujoco_backend", "MujocoRenderer", {"gl": "glfw"}, "mujoco"),
    "pybullet-tiny": ("vsbench.render.pybullet_backend", "PyBulletRenderer", {"hardware": False}, "pybullet"),
    "pybullet-egl": ("vsbench.render.pybullet_backend", "PyBulletRenderer", {"hardware": True}, "pybullet"),
    "open3d-cpu": ("vsbench.render.open3d_backend", "Open3DRenderer", {"cpu": True}, "open3d"),
    "open3d-gpu": ("vsbench.render.open3d_backend", "Open3DRenderer", {"cpu": False}, "open3d"),
}

# variants that never need a GPU, in the order the smoke test tries them
CPU_VARIANTS = {
    "linux": ["mujoco-egl", "mujoco-osmesa", "pybullet-tiny", "open3d-cpu", "mujoco-glfw"],
    "darwin": ["mujoco-glfw", "pybullet-tiny", "open3d-gpu"],
    "win32": ["mujoco-glfw", "pybullet-tiny", "open3d-gpu"],
}


#: the variants that draw on the CPU once ``VSBENCH_SOFTWARE_GL=1`` (Mesa llvmpipe through EGL
#: or OSMesa, PyBullet's TinyRenderer, Open3D's Filament on llvmpipe with Vulkan pinned to
#: lavapipe by ``vsbench.glenv.force_software_gl``). Every other variant can open a hardware
#: context (``open3d-gpu``, ``pybullet-egl``, ``mujoco-glfw`` through the display's GLX
#: vendor) and is refused under software GL unless ``VSBENCH_ALLOW_GPU=1``.
SOFTWARE_VARIANTS = ("mujoco-egl", "mujoco-osmesa", "pybullet-tiny", "open3d-cpu")


def gpu_allowed() -> bool:
    """``VSBENCH_ALLOW_GPU=1``: the one explicit way past the GPU refusal (never during the
    station's Isaac evaluation)."""
    return os.environ.get("VSBENCH_ALLOW_GPU", "").strip().lower() in ("1", "true", "yes")


def gpu_refusal(name: str) -> str | None:
    """Why ``name`` may not run here, or None: under software GL only
    :data:`SOFTWARE_VARIANTS` run, unless ``VSBENCH_ALLOW_GPU=1``."""
    from vsbench.glenv import software_gl_requested

    if name in SOFTWARE_VARIANTS or not software_gl_requested() or gpu_allowed():
        return None
    return (f"backend {name!r} can open a GPU context and VSBENCH_SOFTWARE_GL=1 is set: refused "
            f"(software variants: {', '.join(SOFTWARE_VARIANTS)}; VSBENCH_ALLOW_GPU=1 lifts this, "
            f"never while the Isaac evaluation runs)")


def select_backends(raw: str = "") -> list[str]:
    """The variants a gate run is parametrised over.

    ``raw`` (``--backends`` / ``VSBENCH_BACKENDS``, comma separated) names them explicitly;
    an unknown name is a ``KeyError``, a GPU-capable one under software GL a ``PermissionError``
    (:func:`gpu_refusal`). Empty: this platform's CPU order, installed packages only, and under
    software GL only :data:`SOFTWARE_VARIANTS`.
    """
    from vsbench.glenv import software_gl_requested

    if raw.strip():
        names = [n.strip() for n in raw.split(",") if n.strip()]
        unknown = [n for n in names if n not in BACKENDS]
        if unknown:
            raise KeyError(f"unknown backends {unknown}; known {sorted(BACKENDS)}")
        refused = [gpu_refusal(n) for n in names if gpu_refusal(n)]
        if refused:
            raise PermissionError("; ".join(refused))
        return names
    names = [n for n in default_order() if package_installed(n)]
    if software_gl_requested() and not gpu_allowed():
        names = [n for n in names if n in SOFTWARE_VARIANTS]
    return names


def default_order() -> list[str]:
    for k, v in CPU_VARIANTS.items():
        if sys.platform.startswith(k):
            return v
    return CPU_VARIANTS["linux"]


def package_installed(name: str) -> bool:
    return importlib.util.find_spec(BACKENDS[name][3]) is not None


def available_backends() -> list[str]:
    """Variants whose package imports (they may still fail to open a GL context)."""
    return [n for n in BACKENDS if package_installed(n)]


def make_renderer(name: str, scene: SceneSpec, cam: CameraModel, **extra) -> Renderer:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")
    refusal = gpu_refusal(name)
    if refusal:
        raise PermissionError(refusal)
    from vsbench.glenv import force_software_gl, software_gl_requested

    if software_gl_requested():
        force_software_gl()
    module, cls, kwargs, pkg = BACKENDS[name]
    if importlib.util.find_spec(pkg) is None:
        raise ImportError(f"backend {name!r} needs the {pkg!r} package: pip install vsbench[{pkg}]")
    mod = importlib.import_module(module)
    return getattr(mod, cls)(scene, cam, **{**kwargs, **extra})


def _probe_main(name: str) -> None:  # runs in a subprocess
    from vsbench.camera import CameraModel
    from vsbench.scene import Material, SceneSpec, Sphere

    cam = CameraModel.from_focal(64, 48, 60.0)
    scene = SceneSpec((Sphere((0, 0, 0), 0.1, Material((1, 0, 0), unlit=True)),), ambient=1.0)
    r = make_renderer(name, scene, cam)
    T = np.eye(4)
    T[:3, 3] = (0, 0, -0.5)
    rgb, depth = r.render(T)
    assert rgb.shape == (48, 64, 3) and depth.shape == (48, 64)
    info = getattr(r, "info", lambda: {})()
    r.close()
    print("PROBE_OK " + json.dumps(info))


_probe_cache: dict[str, tuple[bool, str]] = {}


def probe_backend(name: str, timeout: float = 180.0) -> tuple[bool, str]:
    """Try the variant in a fresh process: (ok, reason-or-info)."""
    if name in _probe_cache:
        return _probe_cache[name]
    refusal = gpu_refusal(name)
    if refusal:
        return False, refusal          # never spawned, never cached (the env may change)
    if not package_installed(name):
        res = (False, f"package {BACKENDS[name][3]!r} not installed")
    else:
        code = f"from vsbench.render.base import _probe_main; _probe_main({name!r})"
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
        except subprocess.TimeoutExpired:
            res = (False, "timed out")
        else:
            tail = (p.stdout + p.stderr).strip().splitlines()
            ok_line = [ln for ln in tail if ln.startswith("PROBE_OK")]
            if p.returncode == 0 and ok_line:
                res = (True, ok_line[-1][len("PROBE_OK "):])
            else:
                res = (False, "\n".join(tail[-6:]))
    _probe_cache[name] = res
    return res
