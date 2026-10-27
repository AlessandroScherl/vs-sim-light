#!/usr/bin/env bash
# Run the copied vs-sim test tier (vsbench/tests/fast) in its own pytest process.
#
#   tools/test_copied.sh [extra pytest args]
#
# Its own process because vsbench's tests/conftest.py and the copy's vsbench/tests/conftest.py
# are both imported as the module `conftest`. The asset-root variables are unset because
# three copied tests assert their built-in defaults. Tests whose subject is not in this
# repository are deselected by tests/copied_deselect.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${VSBENCH_PY:-python}
args=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  args+=(--deselect "$line")
done < tests/copied_deselect.txt
exec env -u VSSIM_ASSET_ROOT -u VSSIM_YCB_ROOT $PY -m pytest vsbench/tests/fast -q -p no:cacheprovider "${args[@]}" "$@"
