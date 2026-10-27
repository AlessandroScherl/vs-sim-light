"""Where vsbench computes: the GPU by default, the CPU in station mode (``vsbench.runtime``).

**Default** (no ``VSBENCH_STATION``): nothing is refused, ``--device auto`` picks CUDA when
torch sees it, MuJoCo's EGL takes the first device and falls back to Mesa's software device
when that device cannot make a context. These tests check the *decisions* (and the fallback
with a stand-in failure), never by opening a GPU context.

**Station mode** (``VSBENCH_STATION=1``): software GL forced, the GPU-capable renderer
(``mujoco-glfw``) refused before any import, ``--device cuda`` refused, ``auto`` -> ``cpu``,
Vulkan pinned to lavapipe, and a clean child process comes up on llvmpipe by itself.
``VSBENCH_ALLOW_GPU=1`` is the one way past the refusals.
"""

from __future__ import annotations

import ctypes.util
import json
import os
import subprocess
import sys

import pytest

from vsbench import glenv, runtime
from vsbench.render import base

GPU_CAPABLE = sorted(set(base.BACKENDS) - set(base.SOFTWARE_VARIANTS))


@pytest.fixture
def station(monkeypatch):
    monkeypatch.setenv("VSBENCH_STATION", "1")
    monkeypatch.delenv("VSBENCH_ALLOW_GPU", raising=False)
    monkeypatch.delenv("VSBENCH_SOFTWARE_GL", raising=False)
    return monkeypatch


@pytest.fixture
def default(monkeypatch):
    for k in ("VSBENCH_STATION", "VSBENCH_ALLOW_GPU", "VSBENCH_SOFTWARE_GL"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ----------------------------------------------------------------------- station mode
def test_the_gpu_capable_variant_is_named():
    assert GPU_CAPABLE == ["mujoco-glfw"]
    assert set(base.SOFTWARE_VARIANTS) <= set(base.BACKENDS)


def test_station_mode_asks_for_software_gl(station):
    assert runtime.station_mode() and not runtime.gpu_allowed()
    assert glenv.software_gl_requested()
    plan = glenv.gl_plan()
    assert plan["software_gl"] and plan["station"] and "station" in plan["reason"]


@pytest.mark.parametrize("name", GPU_CAPABLE)
def test_station_mode_refuses_a_gpu_variant(station, name):
    assert "VSBENCH_ALLOW_GPU" in base.gpu_refusal(name)
    with pytest.raises(PermissionError, match="VSBENCH_ALLOW_GPU"):
        base.make_renderer(name, None, None)                  # refused before any import
    from vsbench.world.mujoco_scene import MujocoScene

    with pytest.raises(PermissionError, match="VSBENCH_ALLOW_GPU"):
        MujocoScene("poster2", gl=name.split("-", 1)[1])      # before the scene is built
    for soft in base.SOFTWARE_VARIANTS:
        assert base.gpu_refusal(soft) is None


def test_station_mode_keeps_torch_on_the_cpu(station):
    assert runtime.resolve_device("auto", cuda_available=True) == "cpu"
    assert runtime.resolve_device("cpu", cuda_available=True) == "cpu"
    with pytest.raises(SystemExit, match="station mode"):
        runtime.resolve_device("cuda", cuda_available=True)
    assert runtime.thread_cap() == 2 and runtime.thread_cap(1) == 1


def test_allow_gpu_is_the_one_way_past(station):
    station.setenv("VSBENCH_ALLOW_GPU", "1")
    assert runtime.gpu_allowed() and not glenv.software_gl_requested()
    assert base.gpu_refusal("mujoco-glfw") is None
    assert runtime.resolve_device("auto", cuda_available=True) == "cuda"
    station.setenv("VSBENCH_SOFTWARE_GL", "1")          # both: software GL, nothing refused
    assert glenv.software_gl_requested() and base.gpu_refusal("mujoco-glfw") is None


def test_station_mode_refuses_a_scene_on_a_hardware_context(station):
    from vsbench.world.fly import require_software_gl

    class Scene:
        def __init__(self, mode, gl):
            self.render_mode, self.renderer, self.closed = mode, type("R", (), {"gl_renderer": gl}), False

        def close(self):
            self.closed = True

    ok = Scene("mujoco-llvmpipe", "llvmpipe (LLVM 19.1.1, 256 bits)")
    require_software_gl(ok)
    for mode, gl in (("mujoco-egl-gpu", "NVIDIA GeForce RTX 4090/PCIe/SSE2"),
                     ("mujoco-llvmpipe", "unknown")):
        s = Scene(mode, gl)
        with pytest.raises(SystemExit, match="station mode"):
            require_software_gl(s)
        assert s.closed
    require_software_gl(Scene("mujoco-egl-gpu", "NVIDIA"), allow_gpu=True)   # --allow-gpu


def test_vulkan_is_pinned_to_lavapipe(station):
    station.setenv("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    station.setenv("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    glenv.force_software_gl()
    want = (glenv.LAVAPIPE_ICD_JSON if os.path.exists(glenv.LAVAPIPE_ICD_JSON)
            else glenv.NO_VULKAN_ICD)
    assert os.environ["VK_DRIVER_FILES"] == want and os.environ["VK_ICD_FILENAMES"] == want


# ---------------------------------------------------------------------- default mode
def test_default_mode_refuses_nothing_and_asks_for_no_software_gl(default):
    assert not runtime.station_mode() and runtime.gpu_allowed()
    assert not glenv.software_gl_requested() and glenv.ensure_software_gl() == {}
    plan = glenv.gl_plan()
    assert not plan["software_gl"] and not plan["station"] and "default" in plan["reason"]
    for name in base.BACKENDS:
        assert base.gpu_refusal(name) is None
    default.setenv("VSBENCH_SOFTWARE_GL", "1")          # software GL on request, still no refusal
    assert glenv.software_gl_requested() and base.gpu_refusal("mujoco-glfw") is None


def test_default_mode_puts_torch_on_cuda_when_it_is_there(default):
    assert runtime.resolve_device("auto", cuda_available=True) == "cuda"
    assert runtime.resolve_device("auto", cuda_available=False) == "cpu"
    assert runtime.resolve_device("cuda", cuda_available=True) == "cuda"
    assert runtime.resolve_device("cpu", cuda_available=True) == "cpu"
    with pytest.raises(SystemExit, match="no GPU"):
        runtime.resolve_device("cuda", cuda_available=False)
    with pytest.raises(ValueError):
        runtime.resolve_device("tpu", cuda_available=False)
    assert runtime.thread_cap() is None and runtime.thread_cap(4) == 4


def test_default_mode_does_not_refuse_a_hardware_scene(default):
    from vsbench.world.fly import require_software_gl

    s = type("S", (), {"render_mode": "mujoco-egl-gpu",
                       "renderer": type("R", (), {"gl_renderer": "NVIDIA"})})()
    require_software_gl(s)                              # no refusal outside station mode


def test_the_render_mode_names_the_context():
    from vsbench.world.mujoco_scene import render_mode_from

    assert render_mode_from("egl", "llvmpipe (LLVM 19.1.1, 256 bits)") == "mujoco-llvmpipe"
    assert render_mode_from("egl", "NVIDIA GeForce RTX 4090/PCIe/SSE2") == "mujoco-egl-gpu"


def test_egl_falls_back_to_software_when_the_first_device_fails(default):
    """A stand-in failure of the first context: the renderer retries once on Mesa's software
    device and says so. Runs only in a process that is already pinned to Mesa (station mode's
    conftest), so the retry - the one real context here - can only be llvmpipe."""
    mujoco = pytest.importorskip("mujoco")
    if not (os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1" and os.environ.get("MUJOCO_EGL_DEVICE_ID")
            and os.environ.get("MUJOCO_GL", "egl") == "egl"):
        pytest.skip("needs a process pinned to Mesa's software EGL device (station mode)")
    from vsbench.camera import CameraModel
    from vsbench.render.mujoco_backend import MujocoRenderer
    from vsbench.scene import Material, SceneSpec, Sphere

    real, calls = mujoco.GLContext, []

    def flaky(w, h):
        calls.append((w, h))
        if len(calls) == 1:
            raise RuntimeError("stand-in: no EGL context on the first device")
        return real(w, h)

    default.setattr(mujoco, "GLContext", flaky)
    scene = SceneSpec((Sphere((0, 0, 0), 0.1, Material((1, 0, 0), unlit=True)),), ambient=1.0)
    r = MujocoRenderer(scene, CameraModel.from_focal(64, 48, 60.0), gl="egl", samples=0)
    try:
        assert len(calls) == 2 and r.gl_fallback and "stand-in" in r.gl_fallback
        assert "llvmpipe" in r.gl_renderer.lower() and r.info()["gl_fallback"] == r.gl_fallback
    finally:
        r.close()


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


def test_a_vulkan_process_sees_only_a_cpu_device(station):
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


#: the switches a shell may carry; the child below starts with none of them but station mode
SWITCHES = ("VSBENCH_SOFTWARE_GL", "VSBENCH_ALLOW_GPU", "VSBENCH_STATION", "LIBGL_ALWAYS_SOFTWARE",
            "EGL_PLATFORM", "MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID",
            "__EGL_VENDOR_LIBRARY_FILENAMES", "VK_DRIVER_FILES", "VK_ICD_FILENAMES")

STATION_CHILD = r"""
import json, os, sys
from vsbench import glenv
from vsbench.render import base
from vsbench.world.mujoco_scene import MujocoScene

out = {"env_in": {k: os.environ.get(k) for k in ("VSBENCH_SOFTWARE_GL", "VSBENCH_ALLOW_GPU")}}
# 1. the decisions, before any GL library is loaded or any context exists
out["requested"] = glenv.software_gl_requested()
refused = {}
try:
    MujocoScene("poster2", gl="glfw")
    refused["MujocoScene(gl=glfw)"] = False
except PermissionError:
    refused["MujocoScene(gl=glfw)"] = True
out["refused"] = refused
out["gl_loaded_before"] = [m for m in ("mujoco", "OpenGL") if m in sys.modules]
if not (out["requested"] and all(refused.values()) and not out["gl_loaded_before"]):
    print("RESULT " + json.dumps(out)); sys.exit(3)       # the guard is broken: no context at all
# 2. the world, in the environment the code chose itself
with MujocoScene("poster2") as s:
    out["scene_gl"] = s.renderer.gl_renderer
    out["scene_mode"] = s.render_mode
out["forced"] = {k: os.environ.get(k) for k in ("LIBGL_ALWAYS_SOFTWARE", "EGL_PLATFORM")}
out["nvidia_mapped"] = any("nvidia" in ln.lower() for ln in open("/proc/self/maps"))
print("RESULT " + json.dumps(out))
"""


def test_station_mode_is_software_gl_in_a_clean_process(tmp_path):
    """A child whose environment names only ``VSBENCH_STATION=1`` (none of the GL switches)
    builds ``MujocoScene("poster2")`` on llvmpipe, and the GPU-capable context is refused.

    The child first checks the code's decisions and exits before creating any context if one
    fails. As a safety net in case the guard were broken, the parent still hides CUDA and
    hands the child glvnd's Mesa vendor file and MuJoCo's software EGL device (the values
    ``force_software_gl`` sets itself); ``LIBGL_ALWAYS_SOFTWARE`` and ``EGL_PLATFORM`` are
    left out, so the child's records of them show what the code set on its own.
    """
    pytest.importorskip("mujoco")
    from vsbench.world.build import workspace_root

    if not (workspace_root() / "assets" / "poster2" / "poster2.png").exists():
        pytest.skip("poster2's picture is not installed ($VSBENCH_WORKSPACE_ROOT/assets/poster2)")
    if not os.path.exists(glenv.MESA_VENDOR_JSON):
        pytest.skip("no Mesa EGL vendor file: the safety net cannot be laid")
    if not (os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1"):
        pytest.skip("this process is not pinned to Mesa (run in station mode)")
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    env["VSBENCH_STATION"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""                                  # the safety net
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = glenv.MESA_VENDOR_JSON
    icd = glenv.LAVAPIPE_ICD_JSON if os.path.exists(glenv.LAVAPIPE_ICD_JSON) else glenv.NO_VULKAN_ICD
    env["VK_DRIVER_FILES"] = env["VK_ICD_FILENAMES"] = icd
    idx = glenv.software_egl_device_index()     # this process is already pinned to Mesa
    if idx is None:
        pytest.skip("no EGL_MESA_device_software device")
    env["MUJOCO_EGL_DEVICE_ID"] = str(idx)
    env.setdefault("LP_NUM_THREADS", "2")
    p = subprocess.run([sys.executable, "-c", STATION_CHILD], capture_output=True, text=True,
                       timeout=600, env=env, cwd=str(tmp_path))
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, (p.returncode, p.stderr[-3000:])
    out = json.loads(lines[-1][len("RESULT "):])
    assert p.returncode == 0, (out, p.stderr[-2000:])
    assert out["env_in"] == {"VSBENCH_SOFTWARE_GL": None, "VSBENCH_ALLOW_GPU": None}, out
    assert out["requested"] and all(out["refused"].values()), out
    assert "llvmpipe" in out["scene_gl"].lower() and out["scene_mode"] == "mujoco-llvmpipe", out
    assert out["forced"] == {"LIBGL_ALWAYS_SOFTWARE": "1", "EGL_PLATFORM": "surfaceless"}, out
    assert out["nvidia_mapped"] is False, out
