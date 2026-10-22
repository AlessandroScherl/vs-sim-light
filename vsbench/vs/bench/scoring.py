# vs-sim copy: vssim/bench/scoring.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""What counts as success, and the reads that go under a table (unit U7).

Purpose
-------
Every number in a results table comes from here: the **three** success columns
protocol v2 requires, the failure taxonomy, and the two reads of the same episode
(at the recorded stop, and at the step cap). Pure post-hoc functions over stored
records - nothing here ever touches a control path, and every function takes the
true pose only from a record that has already been written.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
- ``src/eval/scoring.py:97`` ``convergence_a`` -> :func:`predicate_a`, verbatim:
  the ART-VS operational predicate
  (``ART-VS/.../core/controller.py::check_convergence_simplified``), both errors
  under a tenth of their start OR within 10 mm and 1 deg.
- ``src/eval/scoring.py::classify_episode`` -> :func:`classify_episode`, with its
  priority order and its ``converged_wrong_pose`` clause (the controller's own
  feature error settled while the true pose did not - the objects1/M4
  wrong-equilibrium signature).
- ``DIVERGENCE`` (position error > 3x initial) and ``WRONG_POSE`` verbatim.

New in protocol v2 (``../docs/eval_protocol_v2_2026-09-02.md`` section 6)
-------------------------------------------------------------------------
Two further gates, in the same function family, because the papers we compare
against disagree with their own code:

- :func:`paper_gate` - translation < 3 cm and rotation < 3 deg (the wording in the
  published tables);
- :func:`cns_code_gate` - translation < 10 % of the working distance and rotation
  < 10 deg (what CNS's *shipped scorer* computes, and therefore what its published
  numbers were produced by).

They disagree by a factor of five on a hard cell: the N3.6 cluttered2 fielded arm
scores 5/20 under predicate A, 1/20 under the paper gate and 7/20 under CNS's
code gate. One sentence under every table says which produced the quoted rivals.

Deliberately dropped
--------------------
The secondary B/C convergence columns (ViT-VS wording and DMLVS, both resting on
an unpublished "velocities near zero" that we had to invent), the ``||e||``-plateau
*scoring* function (protocol v2 promotes CNS's plateau to the scored stop and it
runs **online** in :mod:`vsbench.vs.baselines.control`, so a second offline copy would be the
second implementation this repository forbids), the D6 winmean/consec termination
reads (reporting-only, never claim-bearing, superseded by the scored stop),
``length_ratio`` / ``jerk_stats`` / ``positions_rel_goal`` (trace statistics,
:mod:`refinevs.eval.traces`), and ``sparsification`` / ``ause`` (a head diagnostic, U5).

Fences
------
- **A gate is a function of a record, not of a session.** Everything below reads
  fields every stored episode since WP7 carries, which is what makes protocol v2
  section 10 possible: the three columns are a *re-score*, not a re-run.
- **The CNS gate needs the scene's working distance** and refuses to guess one. A
  distance silently defaulted is a success rate silently wrong.
- Rotation error is reported the same way as translation (median over reached
  episodes, mean +- sd beside it), never as a pass/fail only.

Golden tests that guard it
--------------------------
``tests/fast/test_eval_rescore.py`` - the stored N3.7b records reproduce the
published table of ``../docs/n37b_outcome_2026-08-31.md`` exactly, and the three
gates give 5/20, 1/20 and 7/20 on the N3.6 cluttered2 fielded cell.
"""

from __future__ import annotations

import math
import statistics as st

#: Predicate A, the primary column (ART-VS operational predicate, mm/deg).
CONVERGENCE_A = {"rel": 0.1, "abs_pos_mm": 10.0, "abs_rot_deg": 1.0}

#: The paper gate: what the published tables' wording says (protocol v2 6b).
PAPER_GATE = {"abs_pos_mm": 30.0, "abs_rot_deg": 3.0}

#: CNS's shipped-code gate: relative to the working distance (protocol v2 6c).
CNS_CODE_GATE = {"pos_frac_of_distance": 0.1, "abs_rot_deg": 10.0}

#: Divergence at scoring time (ART-VS: position error > 3x initial). The runtime
#: abort is a different, GT-blind read of the same 3x (:mod:`vsbench.vs.baselines.control`).
DIVERGENCE = {"mult": 3.0}

#: converged_wrong_pose: the controller's own error signal says converged while
#: the true pose does not - predicate A's structure on the feature error.
WRONG_POSE = {"feat_ratio": 0.1, "feat_abs": 0.1}

#: Outcome labels, in priority order. ``aborted`` is new: the old runner ended an
#: episode on a non-finite twist and the taxonomy then called it ``stalled``,
#: which hid a numerical failure among ordinary near-misses. It fires on no stored
#: record (measured over N3.7b and N3.7d: 0 of 2947), so no published number moves.
OUTCOMES = ("converged", "converged_wrong_pose", "diverged", "aborted", "stalled")


def predicate_a(start_t_mm: float, start_r_deg: float,
                final_t_mm: float, final_r_deg: float) -> bool:
    """The ART-VS operational predicate, verbatim (``scoring.convergence_a``)."""
    c = CONVERGENCE_A
    rel_ok = (start_t_mm > 0 and start_r_deg > 0
              and final_t_mm / start_t_mm < c["rel"]
              and final_r_deg / start_r_deg < c["rel"])
    abs_ok = (final_t_mm < c["abs_pos_mm"] and final_r_deg < c["abs_rot_deg"])
    return bool(rel_ok or abs_ok)


def paper_gate(final_t_mm: float, final_r_deg: float) -> bool:
    """Translation < 3 cm and rotation < 3 deg (protocol v2 section 6b)."""
    return bool(final_t_mm < PAPER_GATE["abs_pos_mm"]
                and final_r_deg < PAPER_GATE["abs_rot_deg"])


def cns_code_gate(final_t_mm: float, final_r_deg: float,
                  working_distance_m: float) -> bool:
    """Translation < 10 % of the working distance and rotation < 10 deg.

    ``working_distance_m`` is required and has no default: CNS's shipped scorer is
    relative to the distance, so the same landing is a pass on cluttered3
    (1.285 m) and a fail on hollywood (0.434 m). Guessing it would be a silently
    wrong success rate.
    """
    if working_distance_m is None or not (working_distance_m > 0):
        raise ValueError(
            "cns_code_gate needs the scene's working distance in metres - CNS's "
            "shipped gate is a fraction of it (protocol v2 section 6c)"
        )
    return bool(final_t_mm / 1000.0 < CNS_CODE_GATE["pos_frac_of_distance"] * working_distance_m
                and final_r_deg < CNS_CODE_GATE["abs_rot_deg"])


#: ``scoring.success_rule`` (the enum of :mod:`vsbench.vs.bench.config`) -> the gate.
#: The absolute ladder the owner asked for (2026-09-22): 10 mm / 1 deg, 20 mm / 2 deg,
#: 30 mm / 3 deg (the last is `paper_3cm_3deg`). Pure thresholds, no relative clause.
#: Two looser rungs for the instance suite (owner 2026-09-23: "more convergence tiers
#: ... since we are generalizing here"): 50 mm / 5 deg and 100 mm / 10 deg.
ABS_LADDER = {"abs_10mm_1deg": (10.0, 1.0), "abs_20mm_2deg": (20.0, 2.0),
              "abs_50mm_5deg": (50.0, 5.0), "abs_100mm_10deg": (100.0, 10.0)}


def abs_gate(final_t_mm: float, final_r_deg: float, t_mm: float, r_deg: float) -> bool:
    return bool(final_t_mm < t_mm and final_r_deg < r_deg)


SUCCESS_RULES = {
    "predicate_a": predicate_a,
    "paper_3cm_3deg": paper_gate,
    "cns_code_0p1d_10deg": cns_code_gate,
    "abs_10mm_1deg": abs_gate,
    "abs_20mm_2deg": abs_gate,
    "abs_50mm_5deg": abs_gate,
    "abs_100mm_10deg": abs_gate,
}


def read(rec: dict, at: str = "cap") -> dict:
    """The pose error of one episode at ``"cap"`` or at ``"stop"``.

    Protocol v2 section 5 keeps the loop running to the cap and *records* the
    stop, so both reads exist for every episode; the stop read is ``None`` when
    the rule never fired. Returns ``{"translation_mm", "rotation_deg", "step"}``.
    """
    if at == "cap":
        return {"translation_mm": rec.get("final_translation_mm"),
                "rotation_deg": rec.get("final_rotation_deg"),
                "step": rec.get("steps")}
    if at == "stop":
        if rec.get("stop_step") is None:
            return {"translation_mm": None, "rotation_deg": None, "step": None}
        return {"translation_mm": rec.get("stop_translation_mm"),
                "rotation_deg": rec.get("stop_rotation_deg"),
                "step": rec.get("stop_step")}
    raise ValueError(f"at must be 'cap' or 'stop', got {at!r}")


def success(rec: dict, rule: str = "predicate_a", *, at: str = "cap",
            working_distance_m: float | None = None) -> bool:
    """One success column for one episode. ``rule`` is a ``scoring.success_rule``."""
    if rule not in SUCCESS_RULES:
        raise ValueError(f"unknown success rule {rule!r}; have {sorted(SUCCESS_RULES)}")
    r = read(rec, at)
    t, d = r["translation_mm"], r["rotation_deg"]
    if t is None or d is None:
        return False
    if rule == "predicate_a":
        return predicate_a(rec["start_translation_mm"], rec["start_rotation_deg"], t, d)
    if rule in ABS_LADDER:
        return abs_gate(t, d, *ABS_LADDER[rule])
    if rule == "paper_3cm_3deg":
        return paper_gate(t, d)
    return cns_code_gate(t, d, working_distance_m)


def success_columns(rec: dict, rules=("predicate_a",), *, at: str = "cap",
                    working_distance_m: float | None = None) -> dict:
    """``{rule: bool}`` for the rules a config asked for (protocol v2: all three)."""
    return {r: success(rec, r, at=at, working_distance_m=working_distance_m) for r in rules}


def classify_episode(rec: dict) -> dict:
    """Record -> ``{'converged_a', 'diverged', 'outcome'}`` (``scoring.classify_episode``).

    Priority: converged -> converged_wrong_pose -> diverged -> aborted -> stalled.
    An episode that diverged in pose while its own feature error settled IS the
    wrong-pose class: that signature is more informative than "diverged".
    """
    st_mm, sr = rec["start_translation_mm"], rec["start_rotation_deg"]
    ft, fr = rec["final_translation_mm"], rec["final_rotation_deg"]
    conv = predicate_a(st_mm, sr, ft, fr)
    diverged = bool(ft > DIVERGENCE["mult"] * st_mm
                    or rec.get("terminated_reason") == "diverged_feature")
    fe0, fe1 = rec.get("feature_error_first"), rec.get("feature_error_last")
    feat_conv = (fe0 is not None and fe1 is not None and math.isfinite(fe0)
                 and math.isfinite(fe1) and fe0 > 0
                 and (fe1 / fe0 < WRONG_POSE["feat_ratio"] or fe1 < WRONG_POSE["feat_abs"]))
    if conv:
        outcome = "converged"
    elif feat_conv:
        outcome = "converged_wrong_pose"
    elif diverged:
        outcome = "diverged"
    elif rec.get("terminated_reason") == "non_finite_velocity":
        outcome = "aborted"
    else:
        outcome = "stalled"
    return {"converged_a": conv, "diverged": diverged, "outcome": outcome}


def taxonomy(records) -> dict:
    """Counts per outcome plus the converged-wrong-pose error median (protocol v2 6)."""
    recs = list(records)
    out = {k: 0 for k in OUTCOMES}
    wrong = []
    for r in recs:
        o = r.get("outcome") or classify_episode(r)["outcome"]
        out[o] = out.get(o, 0) + 1
        if o == "converged_wrong_pose" and r.get("final_translation_mm") is not None:
            wrong.append(float(r["final_translation_mm"]))
    out["n"] = len(recs)
    out["converged_wrong_pose_med_mm"] = round(st.median(wrong), 4) if wrong else None
    return out


def med(values):
    """Median over the non-``None`` values, 4 dp, or ``None`` (the tables' primary)."""
    v = [float(x) for x in values if x is not None]
    return round(st.median(v), 4) if v else None


def mean_sd(values):
    """``(mean, sd)`` over the non-``None`` values, 4 dp - the papers' convention.

    A single value reports the population sd (0.0) rather than raising. Public, and
    the **only** copy: :mod:`refinevs.eval.rescore` imports these two rather than
    keeping its own (CLAUDE.md rule 2). :func:`vsbench.vs.bench.stats._med` stays
    separate on purpose - it is inside the frozen paired-read path that the N3.7b
    parity fixture pins, and moving it would be a change to a proven number for a
    tidiness reason.
    """
    v = [float(x) for x in values if x is not None]
    if not v:
        return (None, None)
    return (round(st.mean(v), 4), round(st.pstdev(v) if len(v) == 1 else st.stdev(v), 4))


_med, _mean_sd = med, mean_sd          # the private names this module already used


def landing_stats(records, *, rule: str = "predicate_a", at: str = "cap",
                  working_distance_m: float | None = None) -> dict:
    """One table cell: how often an arm reached, and how close it landed.

    Protocol v2 section 6 in one dict - reached count, the median over reached
    episodes (primary) **and** mean +- standard deviation beside it (the papers'
    convention), the same pair for rotation, the median error as a fraction of the
    working distance (CNS's scorer is relative), the median steps to stop over the
    episodes whose stop fired, and the failure taxonomy.
    """
    recs = list(records)
    reached = [r for r in recs
               if success(r, rule, at=at, working_distance_m=working_distance_m)]
    reads = [read(r, at) for r in reached]
    t = [x["translation_mm"] for x in reads]
    d = [x["rotation_deg"] for x in reads]
    t_mean, t_sd = _mean_sd(t)
    d_mean, d_sd = _mean_sd(d)
    med = _med(t)
    fired = [r for r in recs if r.get("stop_step") is not None]
    out = {
        "n": len(recs), "rule": rule, "at": at, "reached": len(reached),
        "reached_share": round(len(reached) / len(recs), 4) if recs else None,
        "translation_mm_med": med, "translation_mm_mean": t_mean, "translation_mm_sd": t_sd,
        "rotation_deg_med": _med(d), "rotation_deg_mean": d_mean, "rotation_deg_sd": d_sd,
        "steps_to_stop_med": _med([r.get("stop_step") for r in fired]),
        "stop_fires": len(fired),
        "error_frac_of_distance_med": (
            None if (med is None or not working_distance_m)
            else round(med / 1000.0 / float(working_distance_m), 6)),
        "taxonomy": taxonomy(recs),
    }
    return out
