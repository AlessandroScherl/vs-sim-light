"""The GPU routes, closed (2026-10-23).

``pytest tests/world`` run without ``--backends`` once parametrised vsbench's gates over
every installed variant, and Open3D 0.20's Filament engine (``open3d-cpu`` included) opened a
Vulkan context on the station's RTX 4090. Then the last route: a world or a renderer built in
a shell that had not sourced ``tools/cpu_env.sh`` picked the GPU. Now **software GL is the
default** (``vsbench.glenv.software_gl_requested`` is true unless ``VSBENCH_ALLOW_GPU=1``):

* the default backend set holds only ``SOFTWARE_VARIANTS``; naming a GPU-capable variant is
  refused (``select_backends`` -> ``pytest.UsageError`` in ``tests/conftest.py``), and so is
  building or probing one (``make_renderer`` / ``probe_backend``, ``MujocoScene(gl=...)``),
  unless ``VSBENCH_ALLOW_GPU=1`` - with or without ``VSBENCH_SOFTWARE_GL``;
* ``force_software_gl`` pins the Vulkan loader to Mesa's lavapipe, so a process that loads
  Vulkan sees only a CPU device and never maps an NVIDIA library (checked in a child
  process with a ctypes ``vkEnumeratePhysicalDevices``; no device is created);
* in a child process whose environment names none of the switches, ``MujocoScene("poster2")``
  and ``make_renderer("mujoco-egl", ...)`` come up on llvmpipe and every GPU-capable variant is
  refused (the only test here that renders; its safety net is described at the test).
"""

from __future__ import annotations

import ctypes.util
import os
import subprocess
import sys

import pytest

from vsbench import glenv
from vsbench.render import base

GPU_CAPABLE = sorted(set(base.BACKENDS) - set(base.SOFTWARE_VARIANTS))


@pytest.fixture
def software(monkeypatch):
    monkeypatch.setenv("VSBENCH_SOFTWARE_GL", "1")
    monkeypatch.delenv("VSBENCH_ALLOW_GPU", raising=False)
    return monkeypatch


def test_the_gpu_capable_variants_are_named():
    assert {"open3d-gpu", "pybullet-egl", "mujoco-glfw"} <= set(GPU_CAPABLE)
    assert set(base.SOFTWARE_VARIANTS) <= set(base.BACKENDS)


def test_default_set_is_software_only(software, monkeypatch):
    monkeypatch.setattr(base, "package_installed", lambda n: True)     # everything "installed"
    names = base.select_backends("")
    assert names and set(names) <= set(base.SOFTWARE_VARIANTS), names
    assert not set(names) & set(GPU_CAPABLE)


@pytest.mark.parametrize("name", GPU_CAPABLE)
def test_naming_a_gpu_variant_is_refused(software, name):
    with pytest.raises(PermissionError, match="VSBENCH_ALLOW_GPU"):
        base.select_backends(f"mujoco-egl,{name}")
    with pytest.raises(PermissionError, match="VSBENCH_ALLOW_GPU"):
        base.make_renderer(name, None, None)                  # refused before any import
    ok, why = base.probe_backend(name)                         # never spawned
    assert not ok and "VSBENCH_ALLOW_GPU" in why


def test_conftest_turns_the_refusal_into_a_usage_error(software):
    import conftest

    class Cfg:
        def getoption(self, _name):
            return "open3d-gpu"

    with pytest.raises(pytest.UsageError, match="VSBENCH_ALLOW_GPU"):
        conftest._selected(Cfg())


def test_allow_gpu_is_the_one_way_past(software):
    software.setenv("VSBENCH_ALLOW_GPU", "1")
    assert base.select_backends("open3d-gpu") == ["open3d-gpu"]
    assert base.gpu_refusal("open3d-gpu") is None


@pytest.mark.parametrize("software_gl", [None, "0", "1"])
def test_without_allow_gpu_the_refusal_holds(monkeypatch, software_gl):
    """Software GL is the default: neither an unset nor a ``0`` ``VSBENCH_SOFTWARE_GL`` opens
    the GPU; only ``VSBENCH_ALLOW_GPU=1`` does."""
    monkeypatch.delenv("VSBENCH_ALLOW_GPU", raising=False)
    if software_gl is None:
        monkeypatch.delenv("VSBENCH_SOFTWARE_GL", raising=False)
    else:
        monkeypatch.setenv("VSBENCH_SOFTWARE_GL", software_gl)
    assert glenv.software_gl_requested()
    for name in GPU_CAPABLE:
        assert base.gpu_refusal(name) and "VSBENCH_ALLOW_GPU" in base.gpu_refusal(name)
    for name in base.SOFTWARE_VARIANTS:
        assert base.gpu_refusal(name) is None


def test_allow_gpu_alone_turns_software_gl_off(monkeypatch):
    monkeypatch.delenv("VSBENCH_SOFTWARE_GL", raising=False)
    monkeypatch.setenv("VSBENCH_ALLOW_GPU", "1")
    assert not glenv.software_gl_requested() and glenv.ensure_software_gl() == {}
    assert base.gpu_refusal("open3d-gpu") is None
    monkeypatch.setenv("VSBENCH_SOFTWARE_GL", "1")          # both: software GL, nothing refused
    assert glenv.software_gl_requested() and base.gpu_refusal("open3d-gpu") is None


def test_vulkan_is_pinned_to_lavapipe(software):
    software.setenv("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    software.setenv("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    glenv.force_software_gl()
    want = (glenv.LAVAPIPE_ICD_JSON if os.path.exists(glenv.LAVAPIPE_ICD_JSON)
            else glenv.NO_VULKAN_ICD)
    assert os.environ["VK_DRIVER_FILES"] == want and os.environ["VK_ICD_FILENAMES"] == want


ENUMERATE = r'''
import ctypes
vk = ctypes.CDLL("libvulkan.so.1")
class App(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("a", ctypes.c_char_p),
                ("av", ctypes.c_uint32), ("e", ctypes.c_char_p), ("ev", ctypes.c_uint32),
                ("api", ctypes.c_uint32)]
class Info(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("app", ctypes.POINTER(App)), ("nl", ctypes.c_uint32), ("l", ctypes.c_void_p),
                ("ne", ctypes.c_uint32), ("e", ctypes.c_void_p)]
class Props(ctypes.Structure):
    _fields_ = [("api", ctypes.c_uint32), ("drv", ctypes.c_uint32), ("vendor", ctypes.c_uint32),
                ("device", ctypes.c_uint32), ("type", ctypes.c_int), ("name", ctypes.c_char * 256),
                ("rest", ctypes.c_byte * 4096)]
app = App(0, None, b"vsbench-guard", 0, b"vsbench", 0, 1 << 22)
info = Info(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
inst = ctypes.c_void_p()
rc = vk.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(inst))
if rc != 0:
    print("NODEV instance", rc)
else:
    n = ctypes.c_uint32(0)
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
    devs = (ctypes.c_void_p * max(n.value, 1))()
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)
    for i in range(n.value):
        p = Props()
        vk.vkGetPhysicalDeviceProperties(ctypes.c_void_p(devs[i]), ctypes.byref(p))
        print("DEV", p.type, p.name.decode())
    vk.vkDestroyInstance(inst, None)
print("NVIDIA_MAPPED", any("nvidia" in ln.lower() for ln in open("/proc/self/maps")))
'''


def test_a_vulkan_process_sees_only_a_cpu_device(software):
    if not ctypes.util.find_library("vulkan"):
        pytest.skip("no Vulkan loader on this machine")
    glenv.force_software_gl()
    p = subprocess.run([sys.executable, "-c", ENUMERATE], capture_output=True, text=True,
                       timeout=120, env=os.environ.copy())
    out = p.stdout.splitlines()
    assert p.returncode == 0, p.stderr[-2000:]
    devs = [ln for ln in out if ln.startswith("DEV ")]
    assert all(ln.split()[1] == "4" for ln in devs), out        # VK_PHYSICAL_DEVICE_TYPE_CPU
    assert "NVIDIA_MAPPED False" in out, out


#: the switches a shell may carry; the child below starts with none of them
SWITCHES = ("VSBENCH_SOFTWARE_GL", "VSBENCH_ALLOW_GPU", "LIBGL_ALWAYS_SOFTWARE", "EGL_PLATFORM",
            "MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID",
            "__EGL_VENDOR_LIBRARY_FILENAMES", "VK_DRIVER_FILES", "VK_ICD_FILENAMES")

DEFAULT_CHILD = r"""
import json, os, sys
import numpy as np
from vsbench import glenv
from vsbench.render import base
from vsbench.world.mujoco_scene import MujocoScene

out = {"env_in": {k: os.environ.get(k) for k in ("VSBENCH_SOFTWARE_GL", "VSBENCH_ALLOW_GPU")}}
# 1. the decisions, before any GL library is loaded or any context exists
out["requested"] = glenv.software_gl_requested()
gpu = sorted(set(base.BACKENDS) - set(base.SOFTWARE_VARIANTS))
refused = {}
for name in gpu:
    try:
        base.make_renderer(name, None, None)
        refused[name] = False
    except PermissionError:
        refused[name] = True
for gl in ("glfw",):
    try:
        MujocoScene("poster2", gl=gl)
        refused["MujocoScene(gl=%s)" % gl] = False
    except PermissionError:
        refused["MujocoScene(gl=%s)" % gl] = True
out["refused"] = refused
out["gl_loaded_before"] = [m for m in ("mujoco", "OpenGL", "open3d", "pybullet") if m in sys.modules]
if not (out["requested"] and all(refused.values()) and not out["gl_loaded_before"]):
    print("RESULT " + json.dumps(out)); sys.exit(3)       # the guard is broken: no context at all
# 2. the world and the factory, in the environment the code chose itself
with MujocoScene("poster2") as s:
    out["scene_gl"] = s.renderer.gl_renderer
    out["scene_mode"] = s.render_mode
out["forced"] = {k: os.environ.get(k) for k in ("LIBGL_ALWAYS_SOFTWARE", "EGL_PLATFORM")}
from vsbench.camera import CameraModel
from vsbench.scene import Material, SceneSpec, Sphere
cam = CameraModel.from_focal(64, 48, 60.0)
r = base.make_renderer("mujoco-egl", SceneSpec((Sphere((0, 0, 0), 0.1, Material((1, 0, 0), unlit=True)),), ambient=1.0), cam)
T = np.eye(4); T[:3, 3] = (0, 0, -0.5)
rgb, _ = r.render(T)
out["factory_gl"] = r.gl_renderer
out["factory_rgb_max"] = int(rgb.max())
r.close()
out["nvidia_mapped"] = any("nvidia" in ln.lower() for ln in open("/proc/self/maps"))
print("RESULT " + json.dumps(out))
"""


def test_the_default_is_software_gl_in_a_clean_process(tmp_path):
    """A child whose environment names none of the GL switches (no ``VSBENCH_SOFTWARE_GL``, no
    ``tools/cpu_env.sh``) builds ``MujocoScene("poster2")`` and a ``mujoco-egl`` renderer on
    llvmpipe, and every GPU-capable variant is refused.

    The child first checks the code's decisions (software GL requested, every GPU-capable
    variant refused, no GL library loaded yet) and exits before creating any context if one
    fails. As a safety net in case the guard were broken, the parent still hides CUDA and
    hands the child glvnd's Mesa vendor file and MuJoCo's software EGL device (those values are
    the ones ``force_software_gl`` sets itself); ``LIBGL_ALWAYS_SOFTWARE`` and ``EGL_PLATFORM``
    are left out, so the child's records of them show what the code set on its own.
    """
    pytest.importorskip("mujoco")
    root = os.environ.get("VSBENCH_WORKSPACE_ROOT", os.path.expanduser("~/genvs_ws"))
    if not os.path.exists(os.path.join(root, "assets", "poster2", "poster2.png")):
        pytest.skip("poster2's picture is not installed ($VSBENCH_WORKSPACE_ROOT/assets/poster2)")
    if not os.path.exists(glenv.MESA_VENDOR_JSON):
        pytest.skip("no Mesa EGL vendor file: the safety net cannot be laid")
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    env["CUDA_VISIBLE_DEVICES"] = ""                                  # the safety net
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = glenv.MESA_VENDOR_JSON
    icd = glenv.LAVAPIPE_ICD_JSON if os.path.exists(glenv.LAVAPIPE_ICD_JSON) else glenv.NO_VULKAN_ICD
    env["VK_DRIVER_FILES"] = env["VK_ICD_FILENAMES"] = icd
    idx = glenv.software_egl_device_index()     # this process is already pinned to Mesa
    if idx is None:
        pytest.skip("no EGL_MESA_device_software device")
    env["MUJOCO_EGL_DEVICE_ID"] = str(idx)
    env.setdefault("LP_NUM_THREADS", "2")
    p = subprocess.run([sys.executable, "-c", DEFAULT_CHILD], capture_output=True, text=True,
                       timeout=600, env=env, cwd=str(tmp_path))
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, (p.returncode, p.stderr[-3000:])
    import json

    out = json.loads(lines[-1][len("RESULT "):])
    assert p.returncode == 0, (out, p.stderr[-2000:])
    assert out["env_in"] == {"VSBENCH_SOFTWARE_GL": None, "VSBENCH_ALLOW_GPU": None}, out
    assert out["requested"] and all(out["refused"].values()) and set(out["refused"]) >= set(GPU_CAPABLE), out
    assert "llvmpipe" in out["scene_gl"].lower() and out["scene_mode"] == "mujoco-llvmpipe", out
    assert "llvmpipe" in out["factory_gl"].lower() and out["factory_rgb_max"] > 0, out
    assert out["forced"] == {"LIBGL_ALWAYS_SOFTWARE": "1", "EGL_PLATFORM": "surfaceless"}, out
    assert out["nvidia_mapped"] is False, out
