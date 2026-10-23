"""Why ``cns_plateau`` never fired on the U6a flight: replay its telemetry through the copied rule.

    source tools/cpu_env.sh
    $VSBENCH_PY tools/u6a_cns_replay.py --out results/bridge/u6a_cns_replay.json

Reads the U6a cell's ``telemetry.jsonl.gz`` (16 MB, git-ignored payload), rebuilds each step's
per-landmark normalised errors from the stored points (2 dp) and the record's intrinsics, and
feeds them to the copied ``CnsPlateau`` at the arm's 1/60 s. Writes the CNS error signal at a
few steps, the step it first armed (error < 0.01), how many times a new minimum (< 0.9 x the
old one) restarted the hold, the longest hold accrued, and whether the replay fires.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CELL = REPO / "results" / "bridge" / "u6a" / "hollywood_external_sift_ibvs"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from vsbench.vs.baselines.control import CnsPlateau, StopObservation

    tel = sorted(CELL.glob("*/telemetry.jsonl.gz"))[0]
    with gzip.open(tel, "rt") as f:
        steps = json.loads(f.readline())["steps"]
    rec = json.loads((CELL.parent / f"{CELL.name}.jsonl").read_text().splitlines()[0])
    k = rec["intrinsics"]
    f_ = np.array([k["fx"], k["fy"]])
    c_ = np.array([k["cx"], k["cy"]])
    rule = CnsPlateau(dt=rec["dt"])
    rule.reset()
    signal, restarts, armed, fired, best_hold = {}, 0, None, None, 0.0
    last_min = None
    for s in steps:
        pc, pg = np.c_[s["cu"], s["cv"]], np.c_[s["gu"], s["gv"]]
        ne = np.linalg.norm((pc - c_) / f_ - (pg - c_) / f_, axis=1)
        i = int(s["i"])
        verdict = rule.update(StopObservation(step=i, normalised_errors=ne,
                                              t=(i - 1) * rec["dt"]))
        err = float(verdict.detail["cns_error"])        # the rule's own read of this step
        if armed is None and err < rule.const["conduct_thresh"]:
            armed = i
        m = rule._min_err
        if m is not None and m != last_min:
            restarts += last_min is not None
            last_min = m
        best_hold = max(best_hold, float(rule._accum or 0.0))
        if verdict.would_stop and fired is None:
            fired = i
        if i in (1, 50, 100, 200, 300, 400, 500, 600):
            signal[i] = err
    out = {"telemetry": str(tel.relative_to(REPO)), "steps": len(steps),
           "record_stop_step": rec["stop_step"], "replay_fired_step": fired,
           "cns_error_at_step": signal, "armed_step (error < 0.01)": armed,
           "new_minimum_restarts_after_arming": restarts,
           "longest_hold_s": round(best_hold, 4), "waiting_time_s": rule.const["waiting_time"],
           "min_reset_ratio": rule.const["min_reset"],
           "t_mm_at_step": {int(s["i"]): s["t_mm"] for s in steps if int(s["i"]) in signal}}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
