#!/usr/bin/env bash
# Install friction: a fresh venv per backend, exact commands, wall time to the first
# image. Usage: tools/install_friction.sh <backend> [python] [outdir]
#   backend: mujoco-egl | mujoco-osmesa | mujoco-glfw | pybullet-tiny | open3d-cpu
set -u
BACKEND=${1:?backend}
PY=${2:-python3}
OUT=${3:-results/friction}
PKG=${BACKEND%%-*}
mkdir -p "$OUT"
LOG="$OUT/friction_${BACKEND}_$($PY -c 'import sys;print("py%d.%d"%sys.version_info[:2])').log"
VENV="$OUT/venv_${BACKEND}_$($PY -c 'import sys;print("py%d.%d"%sys.version_info[:2])')"
rm -rf "$VENV"
{
  echo "# install friction: $BACKEND with $($PY --version 2>&1) on $(uname -srm)"
  echo "# $(date -Is)"
  echo "# system packages already present: $(ldconfig -p 2>/dev/null | grep -cE 'libOSMesa|libEGL_mesa') Mesa libs (see below)"
  ldconfig -p 2>/dev/null | grep -E 'libOSMesa|libEGL_mesa|libGL\.so' | sed 's/^/#   /'
  T0=$(date +%s)
  echo "\$ $PY -m venv $VENV"
  $PY -m venv "$VENV" 2>&1
  echo "\$ $VENV/bin/pip install --no-cache-dir -e .[$PKG]"
  "$VENV/bin/pip" install --no-cache-dir -e ".[$PKG]" 2>&1 | grep -vE '^\s*$' | tail -n 25
  T1=$(date +%s)
  echo "# pip done in $((T1 - T0)) s"
  echo "\$ $VENV/bin/python -m vsbench.smoke --backend $BACKEND --frames 20 --out $OUT/smoke_${BACKEND}.png"
  "$VENV/bin/python" -m vsbench.smoke --backend "$BACKEND" --frames 20 --out "$OUT/smoke_${BACKEND}.png" 2>&1 | grep -vE 'pci id|dri2 screen|FEngine|Vulkan|CircularBuffer|physical device|feature level|GLFW'
  RC=${PIPESTATUS[0]}
  T2=$(date +%s)
  echo "# smoke exit code $RC; first image after $((T2 - T0)) s from venv creation"
  echo "# numpy-enabled pybullet: $("$VENV/bin/python" -c 'import pybullet as p; print(p.isNumpyEnabled())' 2>/dev/null || echo n/a)"
} 2>&1 | tee "$LOG"
echo "log: $LOG"
