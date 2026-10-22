"""Force Mesa software OpenGL (llvmpipe) on Linux so no backend touches a GPU.

Set ``VSBENCH_SOFTWARE_GL=1`` and ``vsbench.render.make_renderer`` calls
``force_software_gl()`` before any GL library is imported. It restricts glvnd to the Mesa
EGL vendor library, asks Mesa for the software rasteriser, and points MuJoCo's EGL
backend at the ``EGL_MESA_device_software`` device. Everything is ``setdefault``, so
values the user exported win.
"""

from __future__ import annotations

import ctypes
import os
import sys

MESA_VENDOR_JSON = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


def software_gl_requested() -> bool:
    return os.environ.get("VSBENCH_SOFTWARE_GL", "").strip().lower() not in ("", "0", "false", "no")


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
        if k not in os.environ:
            os.environ[k] = v
            applied[k] = v

    setdef("LIBGL_ALWAYS_SOFTWARE", "1")
    setdef("EGL_PLATFORM", "surfaceless")
    if os.path.exists(MESA_VENDOR_JSON):
        setdef("__EGL_VENDOR_LIBRARY_FILENAMES", MESA_VENDOR_JSON)
    idx = software_egl_device_index()
    if idx is not None:
        setdef("MUJOCO_EGL_DEVICE_ID", str(idx))
    return applied
