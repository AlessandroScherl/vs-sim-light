#!/bin/bash
# vs-sim copy: scripts/chains/debug_refly.sh @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
# The debug re-fly (owner 2026-09-30) - the station chain. WRITTEN BY THE BUILD, NOT RUN BY
# IT. Runbook: docs/results/2026-09-30_debug_telemetry_build/report.md.
#
# The instance suite's three arms, budgets, starts and world change, re-flown on 24 pairs
# with the debug telemetry on (scripts/chains/debug_refly_plan.py: for each of the six kinds
# the forward scene's same, same_small, other_small and mine1), one check_baselines
# invocation each:
#   arms   dinobot_objmask (30 jumps), vitvs_dinov3_objmask, artvs_roi_scene
#   starts the near start (DINOBot 30 jumps; the others 200 steps) + the first 5 simple
#          starts to the arm's cap (600 / DINOBot 30) = 18 flights a pair, 432 in all
#   world  swap:@<live>+ground:concrete
#   extra  --film --telemetry --clean-film --outline-iou (the debug log, the depth-true
#          reads, the richer films, the clean film for scripts/debug_view.py)
# Outputs: $OUT/<scene>_<live>/ (check.json, check.md, films/<scene>_<arm>_<start>/ with
# telemetry.jsonl.gz, goal.jpg, goal_mask.png, episode.mp4, frames_meta.json, clean.mp4,
# clean_meta.json), $OUT/<scene>_<live>.log, the chain log $OUT/debug_refly.log.
# Resumable: a finished pair leaves DONE in its output dir and is skipped on a re-launch; a
# pair whose check.json records an arm error (`build_error` / `error`) is NOT marked DONE.
# Refuses to start unless every instance scene's simple start file is pinned.
# Guards (review 2026-09-30): the checkout's commit is written to the chain log and appended
# to $OUT/commit.txt at every launch; before each pair the chain stops (not DONE, logged) if
# the checkout moved to another commit or its check_baselines.py lacks --telemetry - so a
# `git checkout main` during the night stops the chain cleanly instead of failing 20 pairs.
# A lock ($OUT/.lock, when flock exists) refuses a second launch on the same $OUT.
# SMOKE=1: the 1-pair smoke with the same environment and flags - inst_shoe same_small, the
# near start at 50 steps and 1 simple start, into /workspace/vssim_out/debug_smoke (~12 min):
#   docker exec isaac env SMOKE=1 bash /workspace/vs-sim/scripts/chains/debug_refly.sh
# Launch (host):
#   docker exec -d isaac bash /workspace/vs-sim/scripts/chains/debug_refly.sh
# Expected: about 29-33 min a pair (the instance suite's), about 12 h for the 24.
set -u
# VSSIM_REPO / VSSIM_CHAIN_PY are overridable only so the fast tier can drive this script with
# a stand-in interpreter (tests/fast/test_debuglog.py); on the station leave them unset.
REPO=${VSSIM_REPO:-/workspace/vs-sim}
cd "$REPO" || exit 1
export HF_HOME=/workspace/cache/hf HF_HUB_OFFLINE=1 TORCH_HOME=/workspace/torch_home \
  CNS_ROOT=/workspace/CNS VSSIM_ASSET_ROOT=/workspace/genvs-sim VSSIM_YCB_ROOT=/workspace/assets/ycb \
  PYTHONPATH=/workspace/isaac_site \
  LD_LIBRARY_PATH=/workspace/visp_build/lib:/workspace/miniconda3/envs/vispbuild/lib \
  LIGHTTRACK_ROOT=/workspace/LightTrack ACCEPT_EULA=Y
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
  OUT=${OUT:-/workspace/vssim_out/debug_smoke}
  NEAR_STEPS=50 FAR=1
else
  OUT=${OUT:-/workspace/vssim_out/debug_refly}
  NEAR_STEPS=200 FAR=5
fi
LOG=$OUT/debug_refly.log
ARMS="dinobot_objmask vitvs_dinov3_objmask artvs_roi_scene"
PY=${VSSIM_CHAIN_PY:-/isaac-sim/python.sh}

mkdir -p "$OUT" || exit 1

# one chain per $OUT at a time
if command -v flock > /dev/null 2>&1; then
  exec 9> "$OUT/.lock"
  if ! flock -n 9; then
    echo "[chain] another debug re-fly holds $OUT/.lock - refusing $(date -u)" >> "$LOG"
    exit 1
  fi
fi

rev() {   # the checkout's commit; without git in the container, the ref .git/HEAD names
  git -C "$REPO" rev-parse HEAD 2> /dev/null && return 0
  local g=$REPO/.git
  [ -f "$g" ] && g=$(sed -n 's/^gitdir: //p' "$g")
  cat "$g/HEAD" 2> /dev/null || echo unknown
}
REV=$(rev)

guard() {   # $1 scene, $2 live name: may this pair still fly with this checkout?
  local now
  now=$(rev)
  if [ "$now" != "$REV" ]; then
    echo "[chain] the checkout changed ($REV -> $now) - stopping before $1 $2 $(date -u)" >> "$LOG"
    return 1
  fi
  if ! grep -q -e '--telemetry' scripts/check_baselines.py; then
    echo "[chain] scripts/check_baselines.py has no --telemetry - stopping before $1 $2 $(date -u)" >> "$LOG"
    return 1
  fi
}

run() {   # $1 scene, $2 live name
  local dir=$OUT/$1_$2
  if [ -f "$dir/DONE" ]; then
    echo "[chain] $1 $2 skipped (DONE) $(date -u)" >> "$LOG"
    return 0
  fi
  echo "[chain] $1 $2 start $(date -u)" >> "$LOG"
  $PY scripts/check_baselines.py --out "$dir" --scenes "$1" --arms $ARMS \
    --near-steps "$NEAR_STEPS" --budget dinobot_objmask=30 --far "$FAR" --far-set simple5 \
    --film --telemetry --clean-film --outline-iou \
    --perturb "swap:@$2+ground:concrete" > "$dir.log" 2>&1 < /dev/null
  local rc=$?
  if [ "$rc" -ne 0 ] || [ ! -f "$dir/check.json" ]; then
    echo "[chain] $1 $2 rc=$rc - not DONE $(date -u)" >> "$LOG"
    return 1
  fi
  # check_baselines catches a broken arm, records it and exits 0 - such a pair is not done
  if grep -q -e '"build_error"' -e '"error":' "$dir/check.json"; then
    echo "[chain] $1 $2 arm error - not DONE $(date -u)" >> "$LOG"
    return 1
  fi
  # the telemetry is this chain's deliverable: a flight whose recorder failed still counts as
  # flown (the recorder never costs a flight), but it is named in the chain log
  local nlog
  nlog=$(grep -o '"debug_log_error"' "$dir/check.json" | wc -l)
  if [ "$nlog" -gt 0 ]; then
    echo "[chain] $1 $2 WARNING: $nlog episode(s) with debug_log_error - telemetry incomplete $(date -u)" >> "$LOG"
  fi
  touch "$dir/DONE"
  echo "[chain] $1 $2 rc=$rc DONE $(date -u)" >> "$LOG"
}

echo "[chain] debug re-fly start (smoke=$SMOKE) $(date -u)" >> "$LOG"
echo "[chain] checkout $REPO at $REV" >> "$LOG"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $REV" >> "$OUT/commit.txt"
# the pins first: every instance scene's simple start file must load (md5-checked)
if ! $PY - >> "$LOG" 2>&1 <<'PYEOF'
import sys
sys.path.insert(0, ".")          # the chain has cd'ed into the checkout
from vsbench.vs import benchmark
missing = []
for scene in benchmark.INSTANCE_SCENES:
    try:
        benchmark.load_starts(scene, "simple")
    except Exception as exc:  # noqa: BLE001 - listed, then the chain refuses
        missing.append(f"{scene}: {exc}")
if missing:
    print("[chain] simple start files not pinned / not loadable - refusing:\n  "
          + "\n  ".join(missing), flush=True)
    sys.exit(1)
print(f"[chain] all {len(benchmark.INSTANCE_SCENES)} simple start files load", flush=True)
PYEOF
then
  echo "[chain] refused: build and pin the simple start files first $(date -u)" >> "$LOG"
  exit 1
fi
if [ "$SMOKE" = 1 ]; then
  echo "inst_shoe same_small" > "$OUT/debug_refly_plan.txt"
else
  $PY scripts/chains/debug_refly_plan.py > "$OUT/debug_refly_plan.txt" || exit 1
fi
# The plan is read on fd 3 and every flight gets /dev/null as stdin: Isaac's kit interpreter
# reads its stdin, and on the instance suite's first launch (2026-09-23 21:56 UTC) it ate the
# plan's next line. Empty lines are skipped.
while read -r scene live <&3; do
  [ -n "$scene" ] && [ -n "$live" ] || continue
  if ! guard "$scene" "$live"; then
    echo "[chain] debug re-fly stopped $(date -u)" >> "$LOG"
    exit 1
  fi
  run "$scene" "$live"
done 3< "$OUT/debug_refly_plan.txt"
echo "[chain] debug re-fly done $(date -u)" >> "$LOG"
