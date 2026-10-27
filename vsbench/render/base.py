"""The Renderer protocol and its one backend, MuJoCo.

A variant is named ``mujoco-<context>``: ``mujoco-egl`` (headless EGL: the GPU when there is
one, Mesa's software rasteriser otherwise), ``mujoco-osmesa`` (software, needs libOSMesa) or
``mujoco-glfw`` (a window-system context, needs a display). ``make_renderer`` imports MuJoCo
only when asked for a renderer.

In station mode (``VSBENCH_STATION=1``, :mod:`vsbench.runtime`) only the software variants
run (:data:`SOFTWARE_VARIANTS`) unless ``VSBENCH_ALLOW_GPU=1``; outside station mode nothing
is refused.
"""

from __future__ import annotations

import importlib
import importlib.util
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


#: name -> MuJoCo's ``MUJOCO_GL`` value
BACKENDS: dict[str, str] = {"mujoco-egl": "egl", "mujoco-osmesa": "osmesa", "mujoco-glfw": "glfw"}

#: the variants that can draw on Mesa's software rasteriser (EGL is pinned to Mesa's software
#: device under software GL, :func:`vsbench.glenv.force_software_gl`); ``mujoco-glfw`` goes
#: through the display's GL vendor and can open a GPU context
SOFTWARE_VARIANTS = ("mujoco-egl", "mujoco-osmesa")


def gpu_refusal(name: str) -> str | None:
    """Why ``name`` may not run here, or None. Only station mode refuses anything: there,
    only :data:`SOFTWARE_VARIANTS` run unless ``VSBENCH_ALLOW_GPU=1``."""
    from vsbench.glenv import gpu_allowed

    if name in SOFTWARE_VARIANTS or gpu_allowed():
        return None
    return (f"backend {name!r} can open a GPU context and station mode (VSBENCH_STATION=1) "
            f"keeps the GPU closed: refused (software variants: {', '.join(SOFTWARE_VARIANTS)}; "
            "VSBENCH_ALLOW_GPU=1 lifts this)")


def make_renderer(name: str, scene: SceneSpec, cam: CameraModel, **extra) -> Renderer:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")
    refusal = gpu_refusal(name)
    if refusal:
        raise PermissionError(refusal)
    if importlib.util.find_spec("mujoco") is None:
        raise ImportError(f"backend {name!r} needs the 'mujoco' package: pip install mujoco")
    mod = importlib.import_module("vsbench.render.mujoco_backend")
    return mod.MujocoRenderer(scene, cam, gl=BACKENDS[name], **extra)
