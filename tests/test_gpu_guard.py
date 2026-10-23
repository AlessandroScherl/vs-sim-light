"""The third GPU route, closed (2026-10-23): the gate harness's backend selection.

``pytest tests/world`` run without ``--backends`` once parametrised vsbench's gates over
every installed variant, and Open3D 0.20's Filament engine (``open3d-cpu`` included) opened a
Vulkan context on the station's RTX 4090. Under ``VSBENCH_SOFTWARE_GL=1`` now:

* the default backend set holds only ``SOFTWARE_VARIANTS``; naming a GPU-capable variant is
  refused (``select_backends`` -> ``pytest.UsageError`` in ``tests/conftest.py``), and so is
  building or probing one (``make_renderer`` / ``probe_backend``), unless
  ``VSBENCH_ALLOW_GPU=1``;
* ``force_software_gl`` pins the Vulkan loader to Mesa's lavapipe, so a process that loads
  Vulkan sees only a CPU device and never maps an NVIDIA library (checked in a child
  process with a ctypes ``vkEnumeratePhysicalDevices``; no device is created).

No test here renders.
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


def test_without_software_gl_nothing_changes(monkeypatch):
    monkeypatch.delenv("VSBENCH_SOFTWARE_GL", raising=False)
    assert base.gpu_refusal("open3d-gpu") is None


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
