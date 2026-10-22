#!/usr/bin/env bash
# Reproduce the bake-off from a fresh clone: gates, measurements, report.
#   VSBENCH_SOFTWARE_GL=1 tools/run_all.sh            # Linux, CPU only
#   tools/run_all.sh mujoco-glfw pybullet-tiny         # pick the variants
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
if [ $# -gt 0 ]; then BACKENDS=("$@"); else
  BACKENDS=($($PY -c 'from vsbench.render.base import default_order, package_installed; print(" ".join(n for n in default_order() if package_installed(n)))'))
fi
echo "backends: ${BACKENDS[*]}"
mkdir -p results
$PY tools/run_gates.py --backends "${BACKENDS[@]}"
for b in "${BACKENDS[@]}"; do
  if $PY -c "import json,sys; sys.exit(0 if json.load(open('results/gates_$b.json'))['all_passed'] else 1)"; then
    $PY tools/bench_speed.py --backend "$b" --out "results/speed_$b.json"
    $PY tools/bench_parallel.py --backend "$b" --out "results/parallel_$b.json"
    $PY tools/quality_sheet.py --backend "$b"
    $PY tools/lighting_check.py --backend "$b"
  else
    echo "$b: gates not all passed, measurements skipped"
  fi
done
$PY tools/bakeoff_report.py
echo "report: results/bakeoff_report.md"
