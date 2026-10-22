#!/usr/bin/env bash
# Reproduce Step 2 from a fresh clone: the axis sheet, the nine gates and the report.
#   VSBENCH_SOFTWARE_GL=1 tools/run_step2.sh                # Linux, CPU only, first MuJoCo variant that starts
#   tools/run_step2.sh mujoco-glfw                          # a laptop with a display
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
if [ $# -gt 0 ]; then B="$1"; else
  B=$($PY -c 'from vsbench.render.base import package_installed, probe_backend
for n in ("mujoco-egl", "mujoco-osmesa", "mujoco-glfw"):
    if package_installed(n) and probe_backend(n)[0]:
        print(n); break
else:
    raise SystemExit("no MuJoCo variant starts here")')
fi
echo "backend: $B"
mkdir -p results
$PY tools/fly_axes.py --backend "$B"
$PY tools/step2_report.py --backend "$B"
echo "sheet: results/step2_axes.png  report: results/STEP2_REPORT.md"
