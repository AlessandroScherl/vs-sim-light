#!/bin/bash
# vs-sim copy: scripts/chains/instance_suite.sh @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
# The instance suite (owner 2026-09-23) - the station chain. WRITTEN BY THE BUILD, NOT RUN
# BY IT. Runbook: docs/results/2026-09-23_instance_suite_build/report.md.
#
# Every (scene, live object) pair of configs/scenes.yaml's instance suite, core block
# first (scripts/chains/instance_suite_plan.py), one check_baselines invocation each:
#   arms   DINOBot on the object mask (its own stop is its number; a 30-jump budget),
#          ViT-VS DINOv3 on the object mask, ART-VS detector mode prompted with the
#          scene's word - no rotation compensation on any of them (owner ruling)
#   starts the near start (DINOBot: 30 jumps; the others 200 steps) + the scene's first 5
#          simple starts (|roll| < 60 deg), to the arm's cap (600 / DINOBot 30)
#   world  swap:@<live>+ground:concrete - the named live object on the concrete floor
#          (`@same` is a recorded no-op swap, so its records carry the same reference
#          fields as every other live object: the size-matched pose equals the goal)
#   extra  --outline-iou, --film
# Outputs: $OUT/<scene>_<live>/ (check.json, check.md, films/), $OUT/<scene>_<live>.log,
# the chain log $OUT/instance_suite.log. Resumable: a finished pair leaves DONE in its
# output dir and is skipped on a re-launch; a pair whose check.json records an arm error
# (`build_error` / `error`) is NOT marked DONE and is flown again on a re-launch.
# Refuses to start unless every instance scene's simple start file is pinned
# (vssim.benchmark.INSTANCE_STARTS) - otherwise each of the 82 pairs would boot Isaac
# only to fail on the missing file. Launch (host):
#   docker exec -d isaac bash /workspace/vs-sim/scripts/chains/instance_suite.sh
# or only one block: BLOCK=core (or gen) in the environment.
# FAR (default 5) is the number of simple starts per pair, a prefix of the stored order; the
# development set (owner 2026-10-10) flies BLOCK=dev with FAR=2 (or 0, the near start only)
# through scripts/chains/dev_suite.sh. A launch stamps $OUT/chain_settings.txt with FAR, ARMS
# and EXTRA and refuses an $OUT stamped with other settings: a pair's DONE says "flown", not
# "flown with these starts and arms", so a quick near-only run must never make a later full
# run skip its pairs. PAIRS (optional) restricts the block to the named pairs (a smoke).
# ARM_PLUGIN (optional, 2026-10-11, the external-arm hook): a module importable from
# REFINEVS_ROOT - REQUIRED with it, no default (the station's /workspace/refine-vs is main and
# carries no plugin; the chain refuses a root without the module's file) - appended to
# PYTHONPATH, handed to every check_baselines call as --arm-plugin; both are written into the
# settings stamp (plugin=... root=..., only when ARM_PLUGIN is set). ARMS may then name its
# ids. Every arm id must be [A-Za-z0-9_]+ (refused before anything flies). The chain runs
# in the checkout it lives in: launched as /workspace/vs-sim-plugin/scripts/chains/..., it
# flies /workspace/vs-sim-plugin.
set -u
# VSSIM_REPO / VSSIM_CHAIN_PY are overridable only so the fast tier can drive this script with a stand-in
# interpreter (tests/fast/test_instance_suite.py); on the station leave them unset - the
# checkout is the one this script lives in (2026-10-11; it was /workspace/vs-sim, which is
# where the station's copy lives, so a launch from there flies exactly what it always did).
REPO=${VSSIM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO" || exit 1
export HF_HOME=/workspace/cache/hf HF_HUB_OFFLINE=1 TORCH_HOME=/workspace/torch_home \
  CNS_ROOT=/workspace/CNS VSSIM_ASSET_ROOT=/workspace/genvs-sim VSSIM_YCB_ROOT=/workspace/assets/ycb \
  PYTHONPATH=/workspace/isaac_site \
  LD_LIBRARY_PATH=/workspace/visp_build/lib:/workspace/miniconda3/envs/vispbuild/lib \
  LIGHTTRACK_ROOT=/workspace/LightTrack ACCEPT_EULA=Y
ARM_PLUGIN=${ARM_PLUGIN:-}
REFINEVS_ROOT=${REFINEVS_ROOT:-}
PLUGIN_ARG=""
OUT=${OUT:-/workspace/vssim_out}
BLOCK=${BLOCK:-all}
LOG=$OUT/instance_suite.log
# ARMS and EXTRA may be set in the environment (2026-10-05: the re-fly flies the faithful
# ART-VS `artvs_roi_sim`, the smoothed ViT-VS `vitvs_dinov3_objmask_ema08` and DINOBot's
# second run, with EXTRA="--telemetry --clean-film" for the debug data); unset, the chain
# flies the ruled cells of 2026-09-23 exactly as before.
ARMS=${ARMS:-"dinobot_objmask vitvs_dinov3_objmask artvs_roi_scene"}
EXTRA=${EXTRA:-}
FAR=${FAR:-5}
case "$FAR" in ''|*[!0-9]*) echo "[chain] FAR is '$FAR': a whole number of simple starts" >&2; exit 2 ;; esac
# DINOBot's 30-jump budget only when DINOBot flies: check_baselines refuses a --budget for an
# arm it is not given (2026-10-06, the faithful ART-VS chain flies ART-VS alone)
BUDGET=""
case " $ARMS " in *" dinobot_objmask "*) BUDGET="--budget dinobot_objmask=30" ;; esac
PY=${VSSIM_CHAIN_PY:-/isaac-sim/python.sh}

mkdir -p "$OUT" || exit 1
# arm ids name directories and are split at the last colon by dev_compare.py: [A-Za-z0-9_]+ only
for arm in $ARMS; do
  if ! [[ "$arm" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "[chain] refused: arm id '$arm' is not [A-Za-z0-9_]+ (no colon, no slash) $(date -u)" >> "$LOG"
    exit 1
  fi
done
# the external arm's checkout: stated, and carrying the module (checked before anything flies)
if [ -n "$ARM_PLUGIN" ]; then
  if [ -z "$REFINEVS_ROOT" ]; then
    echo "[chain] refused: ARM_PLUGIN=$ARM_PLUGIN needs REFINEVS_ROOT, the checkout that carries it (no default) $(date -u)" >> "$LOG"
    exit 1
  fi
  MOD_PATH="$REFINEVS_ROOT/$(echo "$ARM_PLUGIN" | tr . /)"
  if [ ! -f "$MOD_PATH.py" ] && [ ! -f "$MOD_PATH/__init__.py" ]; then
    echo "[chain] refused: $ARM_PLUGIN is not in REFINEVS_ROOT=$REFINEVS_ROOT (no $MOD_PATH.py) $(date -u)" >> "$LOG"
    exit 1
  fi
  export PYTHONPATH="$PYTHONPATH:$REFINEVS_ROOT"
  PLUGIN_ARG="--arm-plugin $ARM_PLUGIN"
fi

# VSSIM_MIN_FREE_MB (optional, 2026-10-22; T1 Stage 2 on the shared station GPU): before each
# pair, wait until the GPU has at least this much free memory (MiB), logging every 10 minutes;
# unset or 0 = no wait. VSSIM_GPU_FREE_CMD (tests only) replaces the nvidia-smi query.
MIN_FREE_MB=${VSSIM_MIN_FREE_MB:-0}
GPU_FREE_CMD=${VSSIM_GPU_FREE_CMD:-"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0"}
wait_for_gpu() {   # $1 scene, $2 live name
  [ "$MIN_FREE_MB" -gt 0 ] 2>/dev/null || return 0
  local waited=0 free
  while :; do
    free=$($GPU_FREE_CMD 2>/dev/null | head -1 | tr -dc '0-9')
    if [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MB" ]; then
      [ "$waited" -gt 0 ] && echo "[chain] $1 $2 GPU free ${free} MiB after ${waited} s $(date -u)" >> "$LOG"
      return 0
    fi
    [ $((waited % 600)) -eq 0 ] && echo "[chain] $1 $2 waiting: GPU free ${free:-?} MiB < $MIN_FREE_MB $(date -u)" >> "$LOG"
    sleep 30; waited=$((waited + 30))
  done
}

run() {   # $1 scene, $2 live name
  local dir=$OUT/$1_$2
  if [ -f "$dir/DONE" ]; then
    echo "[chain] $1 $2 skipped (DONE) $(date -u)" >> "$LOG"
    return 0
  fi
  wait_for_gpu "$1" "$2"
  echo "[chain] $1 $2 start $(date -u)" >> "$LOG"
  $PY scripts/check_baselines.py --out "$dir" --scenes "$1" --arms $ARMS \
    --near-steps 200 $BUDGET --far "$FAR" --far-set simple5 \
    --film --outline-iou $EXTRA --perturb "swap:@$2+ground:concrete" $PLUGIN_ARG > "$dir.log" 2>&1 < /dev/null
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
  # a render that ran out of GPU memory (the station's GPU is shared) leaves corrupted frames
  # and the flight goes on: such a pair is not done (2026-10-22)
  if grep -q -e 'ERROR_OUT_OF_DEVICE_MEMORY' -e 'Out of GPU memory' "$dir.log"; then
    echo "[chain] $1 $2 GPU out of memory - not DONE $(date -u)" >> "$LOG"
    return 1
  fi
  touch "$dir/DONE"
  echo "[chain] $1 $2 rc=$rc DONE $(date -u)" >> "$LOG"
}

echo "[chain] instance suite ($BLOCK) start $(date -u)" >> "$LOG"
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
# the settings stamp (2026-10-10): one $OUT, one set of starts, arms and flags
STAMP="far=$FAR arms=$ARMS extra=$EXTRA"
# the plugin and its checkout join the stamp only when given, so every earlier OUT keeps resuming
[ -n "$ARM_PLUGIN" ] && STAMP="$STAMP plugin=$ARM_PLUGIN root=$REFINEVS_ROOT"
if [ -f "$OUT/chain_settings.txt" ]; then
  if [ "$(cat "$OUT/chain_settings.txt")" != "$STAMP" ]; then
    echo "[chain] refused: $OUT was flown with '$(cat "$OUT/chain_settings.txt")', this launch is '$STAMP' - use another OUT $(date -u)" >> "$LOG"
    exit 1
  fi
else
  echo "$STAMP" > "$OUT/chain_settings.txt" || exit 1
fi
$PY scripts/chains/instance_suite_plan.py --block "$BLOCK" > "$OUT/instance_suite_plan.txt" || exit 1
# PAIRS (optional, 2026-10-10): fly only these pairs of the block, in the plan's order,
# `scene:live` separated by spaces (a smoke: PAIRS="inst_shoe:same"); a pair not in the block
# is refused and nothing is flown
if [ -n "${PAIRS:-}" ]; then
  for pair in $PAIRS; do
    if ! grep -q -F -x -- "${pair/:/ }" "$OUT/instance_suite_plan.txt"; then
      echo "[chain] refused: PAIRS names '$pair', not a pair of block $BLOCK $(date -u)" >> "$LOG"
      exit 1
    fi
  done
  : > "$OUT/instance_suite_plan.only"
  while read -r scene live; do
    case " $PAIRS " in *" $scene:$live "*) echo "$scene $live" >> "$OUT/instance_suite_plan.only" ;; esac
  done < "$OUT/instance_suite_plan.txt"
  mv "$OUT/instance_suite_plan.only" "$OUT/instance_suite_plan.txt" || exit 1
fi
# The plan is read on fd 3 and every flight gets /dev/null as stdin: Isaac's kit interpreter
# reads its stdin, and on the first station launch (2026-09-23 21:56 UTC) it ate the plan's next
# line (an empty pair was "flown" and `inst_shoe same_small` skipped).
while read -r scene live <&3; do
  [ -n "$scene" ] && [ -n "$live" ] || continue
  run "$scene" "$live"
done 3< "$OUT/instance_suite_plan.txt"
echo "[chain] instance suite ($BLOCK) done $(date -u)" >> "$LOG"
