#!/usr/bin/env bash
# Step 1-3 measurement chains (2026-09-18); the GPU ones only when the GPU is free.
# the measurement chain on the station; CPU only
set -x
cd "$(dirname "$0")/../.."   # the repository root
export VSBENCH_SOFTWARE_GL=1
PY=$PWD/.venv/bin/python
OSMESA=${VSBENCH_OSMESA_LIB:-$HOME/genvs_ws/mm/osmesa24/lib}   # a Mesa build with libOSMesa
F='pci id|dri2|FEngine|Vulkan|CircularBuffer|physical device|feature level|GLFW'
mkdir -p results/_tmp results/friction
echo "### STEP determinism statistics open3d"
for k in 1 2 3 4 5 6; do $PY -m vsbench.render_once --backend open3d-cpu --scene clutter --seed 7 --out results/_tmp/detstat_$k.npz 2>&1 | grep -Ev "$F" | grep -v '^{'; done
$PY - <<'PYEOF'
import itertools, numpy as np
A = [np.load(f"results/_tmp/detstat_{k}.npz") for k in range(1, 7)]
for i, j in itertools.combinations(range(6), 2):
    d = np.abs(A[i]["rgb"].astype(int) - A[j]["rgb"].astype(int)); dd = np.abs(A[i]["depth"] - A[j]["depth"])
    print(f"open3d fresh {i+1} vs {j+1}: rgb differing px {(d.max(-1) > 0).sum()} max {d.max()} depth differing {(dd > 0).sum()} max {dd.max():.2e}")
PYEOF
for i in 1 2 3; do VSBENCH_BACKENDS=open3d-cpu $PY -m pytest tests/test_determinism.py -q 2>&1 | grep -E "passed|failed" | tail -1; done
echo "### STEP gates"
$PY tools/run_gates.py --backends mujoco-egl pybullet-tiny open3d-cpu 2>&1 | grep -Ev "$F"
LD_LIBRARY_PATH=$OSMESA $PY tools/run_gates.py --backends mujoco-osmesa 2>&1 | grep -Ev "$F"
echo "### STEP speed"
for b in mujoco-egl pybullet-tiny open3d-cpu; do $PY tools/bench_speed.py --backend $b --out results/speed_$b.json 2>&1 | grep -Ev "$F"; done
LD_LIBRARY_PATH=$OSMESA $PY tools/bench_speed.py --backend mujoco-osmesa --out results/speed_mujoco-osmesa.json 2>&1 | grep -Ev "$F"
$PY tools/bench_speed.py --backend mujoco-egl --samples 0 --label mujoco-egl-noaa --out results/speed_mujoco-egl-noaa.json 2>&1 | grep -Ev "$F"
echo "### STEP parallel"
for b in mujoco-egl pybullet-tiny open3d-cpu; do $PY tools/bench_parallel.py --backend $b --out results/parallel_$b.json 2>&1 | grep -Ev "$F"; done
LD_LIBRARY_PATH=$OSMESA $PY tools/bench_parallel.py --backend mujoco-osmesa --out results/parallel_mujoco-osmesa.json 2>&1 | grep -Ev "$F"
echo "### STEP quality + lighting"
for b in mujoco-egl pybullet-tiny open3d-cpu; do $PY tools/quality_sheet.py --backend $b 2>&1 | grep -Ev "$F"; $PY tools/lighting_check.py --backend $b 2>&1 | grep -Ev "$F"; done
LD_LIBRARY_PATH=$OSMESA $PY tools/quality_sheet.py --backend mujoco-osmesa 2>&1 | grep -Ev "$F"
LD_LIBRARY_PATH=$OSMESA $PY tools/lighting_check.py --backend mujoco-osmesa 2>&1 | grep -Ev "$F"
echo "### STEP cross variant"
$PY tools/cross_variant.py --a mujoco-egl --b mujoco-osmesa --env-b LD_LIBRARY_PATH=$OSMESA 2>&1 | grep -Ev "$F"
echo "### STEP friction"
UV=${VSBENCH_UV_PYTHONS:-$HOME/genvs_ws/uvhome/pythons}   # uv-managed interpreters
tools/install_friction.sh mujoco-egl python3 2>&1 | tail -4
tools/install_friction.sh pybullet-tiny python3 2>&1 | tail -4
tools/install_friction.sh open3d-cpu python3 2>&1 | tail -4
LD_LIBRARY_PATH=$OSMESA tools/install_friction.sh mujoco-osmesa python3 2>&1 | tail -4
tools/install_friction.sh mujoco-egl $UV/cpython-3.11-linux-x86_64-gnu/bin/python3.11 2>&1 | tail -4
tools/install_friction.sh pybullet-tiny $UV/cpython-3.11-linux-x86_64-gnu/bin/python3.11 2>&1 | tail -4
tools/install_friction.sh open3d-cpu $UV/cpython-3.11-linux-x86_64-gnu/bin/python3.11 2>&1 | tail -4
tools/install_friction.sh mujoco-egl $UV/cpython-3.10-linux-x86_64-gnu/bin/python3.10 2>&1 | tail -4
echo "### STEP report"
$PY tools/bakeoff_report.py > /dev/null && echo report written
echo "### CHAIN_DONE"
