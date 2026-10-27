# tools/cpu_env.sh - station mode: keep this process off the GPU.
#
#   source tools/cpu_env.sh            # from the repository root, in every shell that flies or tests
#
# For a machine whose GPU belongs to someone else's job. It turns on VSBENCH_STATION=1
# (vsbench/runtime.py): torch on the CPU with two threads, MuJoCo on Mesa's software
# rasteriser (llvmpipe), nvidia-smi hidden from the benchmark loop, and every route to the GPU
# refused (--allow-gpu / VSBENCH_ALLOW_GPU=1 is the one way past). It also runs Python at the
# lowest CPU priority and never downloads weights. Without this file vsbench uses the GPU when
# there is one.
#
# Set your own paths BEFORE sourcing if they are not the defaults: VSBENCH_WORKSPACE_ROOT
# (pictures and objects, default ~/vsbench_data), HF_HOME / TORCH_HOME (the weights).

export VSBENCH_STATION=1
export CUDA_VISIBLE_DEVICES=""            # torch sees no GPU
export VSBENCH_SOFTWARE_GL=1              # Mesa llvmpipe for MuJoCo, through EGL
export MUJOCO_GL=egl
# Vulkan: Mesa's lavapipe only (vsbench.glenv.force_software_gl forces the same in-process)
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
unset VSBENCH_ALLOW_GPU
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TORCH_NUM_THREADS=2                # the driver calls torch.set_num_threads(2) too
export LP_NUM_THREADS=2                   # Mesa llvmpipe's rasteriser threads (default: one per CPU)
export PXR_WORK_THREAD_LIMIT=2            # usd-core's thread pool (tools/convert_objects.py)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights from the cache only, never a download
# pip: never compile in parallel
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1

# The interpreter at nice 19: "$VSBENCH_PY" in scripts, `py` in an interactive shell. The
# repository's .venv when it has one, else the python on PATH.
_vsbench_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
if [ -x "$_vsbench_root/.venv/bin/python" ]; then _vsbench_python="$_vsbench_root/.venv/bin/python"
else _vsbench_python="$(command -v python3 || command -v python)"; fi
export VSBENCH_PY="nice -n 19 $_vsbench_python"
alias py="$VSBENCH_PY"
unset _vsbench_root _vsbench_python
