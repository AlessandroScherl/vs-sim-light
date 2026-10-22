#!/usr/bin/env bash
# Reproduce Step 3 from a fresh clone: the demo for every built-in method (logs, one video
# and one match sheet each), the thirteen gates and results/STEP3_REPORT.md.
#   VSBENCH_SOFTWARE_GL=1 tools/run_step3.sh                # Linux, CPU only, first MuJoCo variant that starts
#   tools/run_step3.sh mujoco-glfw                          # a laptop with a display
#   JOBS=4 tools/run_step3.sh mujoco-egl                    # demos over 4 processes (a GPU machine)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
JOBS=${JOBS:-1}
if [ $# -gt 0 ]; then B="$1"; else
  B=$($PY -c 'from vsbench.check import default_backend; print(default_backend())')
fi
echo "backend: $B"
mkdir -p results/demo
for M in sift_ibvs orb_ibvs oracle_pbvs true_point_ibvs; do
  $PY -m vsbench.demo --method "$M" --backend "$B" --jobs "$JOBS"
done
$PY tools/step3_report.py --backend "$B"
echo "report: results/STEP3_REPORT.md  demo: results/demo/<method>/"
