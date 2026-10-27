"""Which OpenGL MuJoCo draws with: the GPU by default, Mesa's software rasteriser on request.

**Default** (Linux, ``MUJOCO_GL=egl``): MuJoCo's EGL takes the first device that starts, the
GPU when there is one. When no hardware context can be made, the renderer
(``vsbench.render.mujoco_backend``) falls back to Mesa's software device once and says so
(``render_mode`` ``mujoco-llvmpipe``, ``gl_fallback`` in the renderer's info).

**Software GL** (``software_gl_requested()``): in station mode (``VSBENCH_STATION=1``,
:mod:`vsbench.runtime`) unless ``VSBENCH_ALLOW_GPU=1``, or whenever ``VSBENCH_SOFTWARE_GL=1``.
``ensure_software_gl()`` then runs before any GL library is imported: it restricts glvnd to
the Mesa EGL vendor library, asks Mesa for the software rasteriser, points MuJoCo's EGL
backend at the ``EGL_MESA_device_software`` device and the Vulkan loader at Mesa's lavapipe.
These values are **forced** (an exported ``__EGL_VENDOR_LIBRARY_FILENAMES`` naming NVIDIA's
vendor library, or a ``MUJOCO_EGL_DEVICE_ID`` naming a GPU device, would otherwise open a
hardware context); ``force_software_gl`` returns what it set and what it replaced.
"""

from __future__ import annotations

import ctypes
import os
import sys

from vsbench.runtime import _truthy, gpu_allowed, station_mode

MESA_VENDOR_JSON = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
#: Mesa's software Vulkan driver (lavapipe); the only ICD the loader may see under software GL
LAVAPIPE_ICD_JSON = "/usr/share/vulkan/icd.d/lvp_icd.json"
#: what the Vulkan loader is pointed at when lavapipe is missing: no file, so no device
NO_VULKAN_ICD = "/nonexistent/vsbench-software-gl-no-vulkan-device.json"


def software_gl_requested() -> bool:
    """``VSBENCH_SOFTWARE_GL=1``, or station mode without ``VSBENCH_ALLOW_GPU=1``."""
    return _truthy("VSBENCH_SOFTWARE_GL") or (station_mode() and not gpu_allowed())


def ensure_software_gl() -> dict[str, str]:
    """``force_software_gl()`` when software GL is requested; what it set (or ``{}``)."""
    return force_software_gl() if software_gl_requested() else {}


def gl_plan() -> dict:
    """The GL decision, made without loading a GL library or opening a context."""
    soft = software_gl_requested()
    if soft:
        why = ("VSBENCH_SOFTWARE_GL=1" if _truthy("VSBENCH_SOFTWARE_GL")
               else "station mode (VSBENCH_STATION=1)")
    else:
        why = "default: EGL on the first device that starts (the GPU when present), software fallback"
    return {"software_gl": soft, "station": station_mode(), "gpu_allowed": gpu_allowed(),
            "reason": why}


def software_egl_device_index() -> int | None:
    """Index of the Mesa software device in ``eglQueryDevicesEXT``, or None."""
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except OSError:
        return None
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    qd_addr = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    qs_addr = egl.eglGetProcAddress(b"eglQueryDeviceStringEXT")
    if not qd_addr or not qs_addr:
        return None
    qd = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(qd_addr)
    qs = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(qs_addr)
    devs = (ctypes.c_void_p * 32)()
    n = ctypes.c_int(0)
    if not qd(32, devs, ctypes.byref(n)):
        return None
    EGL_EXTENSIONS = 0x3055
    for i in range(n.value):
        ext = qs(devs[i], EGL_EXTENSIONS) or b""
        if b"EGL_MESA_device_software" in ext:
            return i
    return None


def force_software_gl() -> dict[str, str]:
    """Apply the environment; returns what was set (for the JSON records)."""
    if not sys.platform.startswith("linux"):
        return {}
    applied = {}

    def setdef(k, v):
        prev = os.environ.get(k)
        if prev != v:
            os.environ[k] = v
            applied[k] = v if prev is None else f"{v} (was {prev})"

    setdef("LIBGL_ALWAYS_SOFTWARE", "1")
    setdef("EGL_PLATFORM", "surfaceless")
    if os.path.exists(MESA_VENDOR_JSON):
        setdef("__EGL_VENDOR_LIBRARY_FILENAMES", MESA_VENDOR_JSON)
    icd = LAVAPIPE_ICD_JSON if os.path.exists(LAVAPIPE_ICD_JSON) else NO_VULKAN_ICD
    setdef("VK_DRIVER_FILES", icd)      # Vulkan loader >= 1.3.234
    setdef("VK_ICD_FILENAMES", icd)     # older loaders
    idx = software_egl_device_index()
    if idx is not None:
        setdef("MUJOCO_EGL_DEVICE_ID", str(idx))
    return applied
