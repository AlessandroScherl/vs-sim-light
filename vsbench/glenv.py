"""Mesa software OpenGL (llvmpipe) on Linux by default, so no backend touches a GPU.

**Software GL is the default** (2026-10-23): ``software_gl_requested()`` is true unless the
caller opted into the GPU with ``VSBENCH_ALLOW_GPU=1`` (and did not also ask for software GL
with ``VSBENCH_SOFTWARE_GL=1``). ``VSBENCH_SOFTWARE_GL=0`` alone does not open the GPU any
more. ``vsbench.render.make_renderer``, ``MujocoScene`` and the Step 1-3 entry points call
``ensure_software_gl()`` before any GL library is imported. It restricts glvnd to the Mesa
EGL vendor library, asks Mesa for the software rasteriser, and points MuJoCo's EGL
backend at the ``EGL_MESA_device_software`` device. Once software GL is in force these
values are **forced** (an exported ``__EGL_VENDOR_LIBRARY_FILENAMES`` naming NVIDIA's vendor
library, or a ``MUJOCO_EGL_DEVICE_ID`` naming a GPU device, would otherwise open a hardware
context on the shared station); ``force_software_gl`` returns what it set and what it
replaced.
"""

from __future__ import annotations

import ctypes
import os
import sys

MESA_VENDOR_JSON = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
#: Mesa's software Vulkan driver (lavapipe); the only ICD the loader may see under software GL
LAVAPIPE_ICD_JSON = "/usr/share/vulkan/icd.d/lvp_icd.json"
#: what the Vulkan loader is pointed at when lavapipe is missing: no file, so no device
NO_VULKAN_ICD = "/nonexistent/vsbench-software-gl-no-vulkan-device.json"


def _truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def gpu_allowed() -> bool:
    """``VSBENCH_ALLOW_GPU=1``: the one explicit way to a GPU context (never during the
    station's Isaac evaluation)."""
    return _truthy("VSBENCH_ALLOW_GPU")


def software_gl_requested() -> bool:
    """True unless the GPU was opted into: ``VSBENCH_SOFTWARE_GL=1`` always means software;
    otherwise software unless ``VSBENCH_ALLOW_GPU=1``."""
    return _truthy("VSBENCH_SOFTWARE_GL") or not gpu_allowed()


def ensure_software_gl() -> dict[str, str]:
    """``force_software_gl()`` unless the GPU was opted into; what it set (or ``{}``)."""
    return force_software_gl() if software_gl_requested() else {}


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
