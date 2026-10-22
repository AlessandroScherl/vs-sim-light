#!/bin/bash
# vs-sim copy: scripts/chains/dev_suite.sh @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
# The development set's chain (owner 2026-10-10; docs/dev_eval_plan_2026-10-10.md section 2 at
# the workspace root). WRITTEN BY THE BUILD, NOT RUN BY IT. A thin front for
# scripts/chains/instance_suite.sh: the same per-pair invocation, logs, DONE files, arm-error
# check and resume, only fewer pairs and starts.
#
#   pairs  BLOCK=dev - the six kinds' FORWARD scenes x same / same_small / other_small / mine1
#          (vssim.benchmark.DEV_SCENES x DEV_LIVE_NAMES) = 24 pairs
#   starts TIER=1 (default): the near start + the scene's first two simple starts (s_00, s_01)
#          = 3 per pair, 72 flights per arm (FAR=2)
#          TIER=1a: the near start only = 24 flights per arm (FAR=0), the quick check
#   arms   ARMS - REQUIRED, no default: the dev chain flies the method under test (the stored
#          baselines are read from their records by scripts/dev_compare.py, not re-flown)
#   out    OUT - REQUIRED (a container path, e.g. /workspace/vssim_out/dev_artvs); one OUT per
#          tier and arm set - instance_suite.sh refuses an OUT stamped with other settings
#   extra  EXTRA - default "--telemetry --clean-film" (the debug data, as the 2026-10-05 re-fly);
#          EXTRA="" flies without it
#   pairs  PAIRS (optional) - only these dev pairs, `scene:live` separated by spaces, e.g. a
#          one-pair smoke: PAIRS="inst_shoe:same"
#   plugin ARM_PLUGIN (optional, 2026-10-11) - an external arm's module (e.g.
#          refinevs.bench_plugin), importable from REFINEVS_ROOT, which is then REQUIRED (no
#          default: the station's /workspace/refine-vs is main, without the plugin); ARMS may
#          then name its ids (instance_suite.sh checks the module is there, passes --arm-plugin
#          and stamps plugin= and root=)
# DINOBot's 30-jump budget is passed by instance_suite.sh when ARMS holds dinobot_objmask.
# Launch (host), e.g. the ART-VS dev cell:
#   docker exec -d -e ARMS=artvs_roi_sim_dev -e OUT=/workspace/vssim_out/dev_artvs \
#     isaac bash /workspace/vs-sim/scripts/chains/dev_suite.sh
# the quick variant: add -e TIER=1a and another OUT. An external arm (the plugin hook), flown
# from the checkout that carries the hook, e.g.:
#   docker exec -d -e ARMS=clvs_v1 -e ARM_PLUGIN=refinevs.bench_plugin \
#     -e REFINEVS_ROOT=/workspace/refine-vs-clvs -e TIER=1a \
#     -e OUT=/workspace/vssim_out/dev_clvs_v1_1a \
#     isaac bash /workspace/vs-sim-plugin/scripts/chains/dev_suite.sh
# (/workspace/vs-sim-plugin and /workspace/refine-vs-clvs: clones of vs-sim feat/arm-plugin and
# refine-vs feat/clvs-v1, copied in through the container so they belong to its user)
set -u
TIER=${TIER:-1}
case "$TIER" in
  1) FAR=2 ;;
  1a) FAR=0 ;;
  *) echo "[dev] TIER is '$TIER': 1 (near + s_00 + s_01) or 1a (near only)" >&2; exit 2 ;;
esac
if [ -z "${ARMS:-}" ] || [ -z "${OUT:-}" ]; then
  echo "[dev] set ARMS (the arm ids to fly) and OUT (a fresh output directory)" >&2
  exit 2
fi
EXTRA=${EXTRA---telemetry --clean-film}
BLOCK=dev
ARM_PLUGIN=${ARM_PLUGIN:-}
REFINEVS_ROOT=${REFINEVS_ROOT:-}
export TIER FAR ARMS OUT EXTRA BLOCK ARM_PLUGIN REFINEVS_ROOT
exec bash "$(dirname "$0")/instance_suite.sh"
