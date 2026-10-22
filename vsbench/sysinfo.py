"""What every results JSON records: CPU, cores, OS, Python, package versions, GPU."""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys


def cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def gpu_name() -> str | None:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10)
            return out.strip().splitlines()[0]
        except Exception:
            return None
    return None


def package_versions() -> dict[str, str | None]:
    out = {}
    for pkg in ("numpy", "opencv-python-headless", "opencv-python", "Pillow", "mujoco", "pybullet", "open3d", "PyOpenGL", "glfw"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def system_record() -> dict:
    try:
        load = os.getloadavg()
    except (AttributeError, OSError):
        load = None
    return {
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "gpu": gpu_name(),
        "load_average": load,
        "env": {k: os.environ.get(k) for k in ("VSBENCH_SOFTWARE_GL", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "EGL_PLATFORM", "LIBGL_ALWAYS_SOFTWARE", "__EGL_VENDOR_LIBRARY_FILENAMES", "LD_LIBRARY_PATH") if os.environ.get(k)},
    }
