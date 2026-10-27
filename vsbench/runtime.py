"""Where a flight computes: station mode, the torch device, the thread caps.

**Default** (nothing set): torch uses CUDA when it is available (``--device auto``), MuJoCo
draws with EGL on the GPU when there is one and falls back to Mesa's software rasteriser
(:mod:`vsbench.glenv`), and no thread count is capped.

**Station mode** (``VSBENCH_STATION=1``; ``source tools/cpu_env.sh`` sets it) is for a
machine whose GPU belongs to someone else's job: torch on the CPU with two threads, MuJoCo on
Mesa's software rasteriser, ``nvidia-smi`` hidden from the benchmark loop, and every route to
the GPU refused (a GPU-capable renderer, ``--device cuda``, a scene that came up on a hardware
context). ``VSBENCH_ALLOW_GPU=1`` (``--allow-gpu``) is the one way past those refusals.

Every record states what was used: ``device`` (the torch device the encoder arms ran on) and
``render_mode`` (``mujoco-llvmpipe`` for software GL, ``mujoco-<gl>-gpu`` otherwise).
"""

from __future__ import annotations

import os

STATION_ENV = "VSBENCH_STATION"
ALLOW_GPU_ENV = "VSBENCH_ALLOW_GPU"
#: the thread cap station mode applies (torch intra/inter-op, BLAS, Mesa's rasteriser)
STATION_THREADS = 2
DEVICES = ("auto", "cuda", "cpu")


def _truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def station_mode() -> bool:
    """``VSBENCH_STATION=1``: keep the GPU closed (see the module docstring)."""
    return _truthy(STATION_ENV)


def gpu_allowed() -> bool:
    """True outside station mode; inside it only with ``VSBENCH_ALLOW_GPU=1``."""
    return (not station_mode()) or _truthy(ALLOW_GPU_ENV)


def resolve_device(requested: str = "auto", *, cuda_available: bool | None = None) -> str:
    """The torch device for the encoder arms: ``auto`` -> ``cuda`` when torch sees a GPU and
    the GPU is allowed, else ``cpu``; ``cuda`` is refused in station mode (unless
    ``VSBENCH_ALLOW_GPU=1``) and when torch sees no GPU. ``cuda_available`` lets a caller
    (or a test) state the answer instead of asking torch."""
    req = str(requested or "auto").lower()
    if req not in DEVICES:
        raise ValueError(f"device {requested!r} is not one of {DEVICES}")
    if req == "cpu":
        return "cpu"
    if req == "cuda" and not gpu_allowed():
        raise SystemExit("--device cuda in station mode (VSBENCH_STATION=1): refused "
                         "(--allow-gpu / VSBENCH_ALLOW_GPU=1 lifts this)")
    if req == "auto" and not gpu_allowed():
        return "cpu"
    if cuda_available is None:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    if req == "cuda" and not cuda_available:
        raise SystemExit("--device cuda: torch sees no GPU here (use --device auto or cpu)")
    return "cuda" if cuda_available else "cpu"


def thread_cap(requested: int | None = None) -> int | None:
    """The thread cap to apply: ``requested`` if given, else 2 in station mode, else none."""
    if requested is not None:
        return int(requested)
    return STATION_THREADS if station_mode() else None
