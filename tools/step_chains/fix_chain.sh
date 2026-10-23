#!/usr/bin/env bash
# Step 1-3 measurement chains (2026-09-18); the GPU ones only when the GPU is free.
# Since 2026-10-23 software GL is the default (vsbench.glenv); every GPU row says
# VSBENCH_ALLOW_GPU=1 itself - never while the Isaac evaluation runs.
# rerun every MuJoCo row after the depth-context change: CPU (llvmpipe) and GPU
set -x
cd "$(dirname "$0")/../.."   # the repository root
unset VSBENCH_SOFTWARE_GL MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_FILENAMES LIBGL_ALWAYS_SOFTWARE EGL_PLATFORM
PY=$PWD/.venv/bin/python
OSMESA=${VSBENCH_OSMESA_LIB:-$HOME/genvs_ws/mm/osmesa24/lib}   # a Mesa build with libOSMesa
F='pci id|dri2|CircularBuffer|feature level|GLFW'
echo "### STEP gates"
VSBENCH_SOFTWARE_GL=1 $PY tools/run_gates.py --backends mujoco-egl 2>&1 | grep -Ev "$F"
VSBENCH_SOFTWARE_GL=1 LD_LIBRARY_PATH=$OSMESA $PY tools/run_gates.py --backends mujoco-osmesa 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/run_gates.py --backends mujoco-egl --label mujoco-egl-gpu 2>&1 | grep -Ev "$F"
echo "### STEP cross variants"
$PY tools/cross_variant.py --a mujoco-egl --b mujoco-egl --label-a mujoco-egl-gpu --label-b mujoco-egl-llvmpipe --env-a VSBENCH_ALLOW_GPU=1 --env-b VSBENCH_SOFTWARE_GL=1 2>&1 | grep -Ev "$F"
VSBENCH_SOFTWARE_GL=1 $PY tools/cross_variant.py --a mujoco-egl --b mujoco-osmesa --env-b LD_LIBRARY_PATH=$OSMESA 2>&1 | grep -Ev "$F"
$PY tools/cross_variant.py --a open3d-gpu --b open3d-cpu --env-a VSBENCH_ALLOW_GPU=1 --env-b VSBENCH_SOFTWARE_GL=1 2>&1 | grep -Ev "$F"
echo "### STEP speed"
VSBENCH_SOFTWARE_GL=1 $PY tools/bench_speed.py --backend mujoco-egl --out results/speed_mujoco-egl.json 2>&1 | grep -Ev "$F"
VSBENCH_SOFTWARE_GL=1 LD_LIBRARY_PATH=$OSMESA $PY tools/bench_speed.py --backend mujoco-osmesa --out results/speed_mujoco-osmesa.json 2>&1 | grep -Ev "$F"
VSBENCH_SOFTWARE_GL=1 $PY tools/bench_speed.py --backend mujoco-egl --samples 0 --label mujoco-egl-noaa --out results/speed_mujoco-egl-noaa.json 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/bench_speed.py --backend mujoco-egl --label mujoco-egl-gpu --out results/speed_mujoco-egl-gpu.json 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/bench_speed.py --backend mujoco-egl --samples 0 --label mujoco-egl-gpu-noaa --out results/speed_mujoco-egl-gpu-noaa.json 2>&1 | grep -Ev "$F"
echo "### STEP parallel"
VSBENCH_SOFTWARE_GL=1 $PY tools/bench_parallel.py --backend mujoco-egl --out results/parallel_mujoco-egl.json 2>&1 | grep -Ev "$F"
VSBENCH_SOFTWARE_GL=1 LD_LIBRARY_PATH=$OSMESA $PY tools/bench_parallel.py --backend mujoco-osmesa --out results/parallel_mujoco-osmesa.json 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/bench_parallel.py --backend mujoco-egl --label mujoco-egl-gpu --out results/parallel_mujoco-egl-gpu.json 2>&1 | grep -Ev "$F"
echo "### STEP quality + lighting"
VSBENCH_SOFTWARE_GL=1 $PY tools/quality_sheet.py --backend mujoco-egl 2>&1 | grep -Ev "$F"
VSBENCH_SOFTWARE_GL=1 $PY tools/lighting_check.py --backend mujoco-egl 2>&1 | grep -Ev "$F"
VSBENCH_SOFTWARE_GL=1 LD_LIBRARY_PATH=$OSMESA $PY tools/quality_sheet.py --backend mujoco-osmesa 2>&1 | grep -Ev "$F"
VSBENCH_SOFTWARE_GL=1 LD_LIBRARY_PATH=$OSMESA $PY tools/lighting_check.py --backend mujoco-osmesa 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/quality_sheet.py --backend mujoco-egl --label mujoco-egl-gpu 2>&1 | grep -Ev "$F"
VSBENCH_ALLOW_GPU=1 $PY tools/lighting_check.py --backend mujoco-egl --label mujoco-egl-gpu 2>&1 | grep -Ev "$F"
echo "### STEP report"
$PY tools/bakeoff_report.py > /dev/null && echo report written
echo "### FIX_CHAIN_DONE"
