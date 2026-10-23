#!/usr/bin/env bash
# Step 1-3 measurement chains (2026-09-18); the GPU ones only when the GPU is free.
# Since 2026-10-23 software GL is the default (vsbench.glenv); every GPU row says
# VSBENCH_ALLOW_GPU=1 itself - never while the Isaac evaluation runs.
set -x
cd "$(dirname "$0")/../.."   # the repository root
unset VSBENCH_SOFTWARE_GL MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_FILENAMES LIBGL_ALWAYS_SOFTWARE EGL_PLATFORM
PY=$PWD/.venv/bin/python
F='pci id|dri2|CircularBuffer|feature level|GLFW'
echo "### STEP gpu state"
pgrep -af check_baselines | grep -v pgrep | head -2
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
echo "### STEP probes"
for b in mujoco-egl open3d-gpu pybullet-egl; do echo "-- $b"; VSBENCH_ALLOW_GPU=1 $PY -c "from vsbench.render.base import _probe_main; _probe_main('$b')" 2>&1 | grep -Ev "$F" | tail -3; done
echo "### STEP cross variant mujoco gpu vs llvmpipe"
$PY tools/cross_variant.py --a mujoco-egl --b mujoco-egl --label-a mujoco-egl-gpu --label-b mujoco-egl-llvmpipe --env-a VSBENCH_ALLOW_GPU=1 --env-b VSBENCH_SOFTWARE_GL=1 2>&1 | grep -Ev "$F"
echo "### STEP gates gpu"
VSBENCH_ALLOW_GPU=1 $PY tools/run_gates.py --backends mujoco-egl --label mujoco-egl-gpu 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/run_gates.py --backends open3d-gpu 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/run_gates.py --backends pybullet-egl 2>&1 | grep -Ev "$F"
echo "### STEP open3d-gpu determinism statistics"
for i in 1 2 3 4 5 6 7 8 9 10; do VSBENCH_ALLOW_GPU=1 VSBENCH_BACKENDS=open3d-gpu $PY -m pytest tests/test_determinism.py -q 2>&1 | grep -E "^E |passed|failed" | head -2 | tr "\n" " "; echo; done
echo "### STEP speed gpu"
VSBENCH_ALLOW_GPU=1 $PY tools/bench_speed.py --backend mujoco-egl --label mujoco-egl-gpu --out results/speed_mujoco-egl-gpu.json 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/bench_speed.py --backend mujoco-egl --samples 0 --label mujoco-egl-gpu-noaa --out results/speed_mujoco-egl-gpu-noaa.json 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/bench_speed.py --backend open3d-gpu --out results/speed_open3d-gpu.json 2>&1 | grep -Ev "$F"
echo "### STEP parallel gpu"
VSBENCH_ALLOW_GPU=1 $PY tools/bench_parallel.py --backend mujoco-egl --label mujoco-egl-gpu --out results/parallel_mujoco-egl-gpu.json 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/bench_parallel.py --backend open3d-gpu --out results/parallel_open3d-gpu.json 2>&1 | grep -Ev "$F"
echo "### STEP quality + lighting gpu"
VSBENCH_ALLOW_GPU=1 $PY tools/quality_sheet.py --backend mujoco-egl --label mujoco-egl-gpu 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/lighting_check.py --backend mujoco-egl --label mujoco-egl-gpu 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/quality_sheet.py --backend open3d-gpu 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/lighting_check.py --backend open3d-gpu 2>&1 | grep -Ev "$F"
echo "### STEP cross variant open3d gpu vs cpu"
$PY tools/cross_variant.py --a open3d-gpu --b open3d-cpu --env-a VSBENCH_ALLOW_GPU=1 --env-b VSBENCH_SOFTWARE_GL=1 2>&1 | grep -Ev "$F"
echo "### STEP report"
$PY tools/bakeoff_report.py > /dev/null && echo report written
echo "### GPU_CHAIN_DONE"
