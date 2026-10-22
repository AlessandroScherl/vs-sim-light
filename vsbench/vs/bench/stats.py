# vs-sim copy: vssim/bench/stats.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""One paired bootstrap, one record reader, one selector (unit U7).

Purpose
-------
Protocol v2 section 7: contrasts are **paired and in-run**, never absolute numbers
across sessions - session-to-session drift is 0.3-0.7 mm, which is the size of the
effects we argue about. This module is the whole statistics surface: read records,
select a cell, pair two cells on their shared starts, and put a seeded bootstrap
interval on the median difference.

Why it exists at all
--------------------
In ``../genvs-sim`` this code was copy-pasted into every stage module:
``bootstrap_ci_median`` lived in ``n1_verdicts.py:100`` and was imported by
``n36_verdicts``, ``n37a_verdicts``, ``n37b_verdicts``, ``n37d_reads`` ...  while
``load_records`` / ``select`` / ``paired`` / ``stats`` were re-typed in each, with
small silent differences (``n36_verdicts.paired`` defaults ``both_converged=False``,
``n37b_verdicts.paired`` defaults it ``True``). One implementation, with the
defaults written down, is CLAUDE.md rule 2.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
- ``src/eval/n1_verdicts.py:100`` ``bootstrap_ci_median`` -> :func:`bootstrap_ci_median`,
  verbatim, with ``B`` and the seed as explicit parameters. The frozen campaign
  defaults are :data:`BOOTSTRAP_B` = 10000 and :data:`BOOTSTRAP_SEED` = 20260817;
  every published interval in ``RESULTS.md`` was produced with those two numbers.
- ``src/eval/n36_verdicts.py:80-133`` ``load_records`` / ``load_stage`` / ``select`` /
  ``conv_stats`` / ``stop_stats`` and ``:143`` ``paired`` -> here, merged with
  ``n37b_verdicts.py:70-120``'s ``select`` (which added the ``sub60`` roll filter)
  and ``stats``.
- ``src/eval/n37a_verdicts.py:73`` ``capped_population`` -> :func:`capped_population`.
- **new**: :func:`roll_thirds`, protocol v2 section 3's CNS-style partition of the
  wide-100 band into three thirds by initial rotation error.

Deliberately dropped
--------------------
The 23 stage-bound verdict modules themselves: their pick rules, verdict tags,
HTML writers and per-stage arm constants (``ARM_TSEL``, ``MECHANISMS``,
``BAR_MIN_FIRE_SHARE`` ...) were about one session's question and are answered.
The general parts are here; the questions live in ``../PROJECT.md``.

Fences
------
- **Legacy and new records read alike.** A stored record names its arm in
  ``refine_mode``; a record this repository writes names it in ``arm``.
  :func:`arm_of` reads either, so the re-score parity test and a future campaign
  use the same selector.
- **A paired read below :data:`MIN_PAIRED` shared starts returns ``None``**, not a
  number. Eight was the frozen floor of every N3.x session.
- The bootstrap is seeded and its seed is a parameter; a "95 % interval" whose
  seed is not recorded is not reproducible.

Golden tests that guard it
--------------------------
``tests/fast/test_eval_rescore.py`` - the paired v2cm-v11 and v2cm-v2nm deltas of
``../docs/n37b_outcome_2026-08-31.md`` and the pooled and per-scene k24-k48 /
k24-k96 deltas of ``../docs/results/2026-09-02_n37d_ksweep/n37d_reads.json``
reproduce exactly, intervals included.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

import numpy as np

#: The frozen campaign bootstrap: 10 000 resamples, seed 20260817. Every interval
#: in the project's published tables was produced with exactly these.
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260817

#: Fewest shared starts a paired read may be computed on (frozen, all N3.x).
MIN_PAIRED = 8

#: Fields a stored record may name its arm in: the new one first, then the legacy.
ARM_KEYS = ("arm", "refine_mode")


def load_records(path) -> list[dict]:
    """One ``baseline_check.jsonl`` -> a list of records."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_session(session_dir, filename: str = "baseline_check.jsonl") -> list[dict]:
    """Every ``<session>/**/<filename>`` under a directory, sorted by path.

    ``n36_verdicts.load_stage`` walked one level; a session tree is
    ``<session>/<stage>/<cell>/baseline_check.jsonl``, so this walks all of them
    and the caller filters with :func:`select`.
    """
    root = Path(session_dir)
    recs: list[dict] = []
    for f in sorted(root.rglob(filename)):
        recs.extend(load_records(f))
    return recs


def arm_of(rec: dict) -> Any:
    """The arm label of a record, from whichever field carries it."""
    for key in ARM_KEYS:
        if rec.get(key) is not None:
            return rec[key]
    return None


def cell_key(rec: dict) -> str:
    """``"<scene>|<arm>"`` - the one cell identity every grouped read uses (U7b).

    The arm comes from :func:`arm_of`, so a stored ``refine_mode`` and a new ``arm``
    group alike. A stored record's ``base_method`` is prefixed because the SIFT and
    CNS bars were recorded under the **same** ``off`` arm label and would otherwise
    pool with the ViT baseline - the one grouping mistake that silently merges two
    methods' numbers. Used by :mod:`refinevs.eval.rescore` and
    :mod:`refinevs.eval.traces`; a telemetry ``meta`` block works too, it carries the
    same fields.
    """
    arm = arm_of(rec)
    method = rec.get("base_method") or rec.get("method")
    if method is not None and str(method) != str(arm):
        return f"{rec.get('scene')}|{method}/{arm}"
    return f"{rec.get('scene')}|{arm}"


def group_by_cell(records) -> dict[str, list]:
    """``{cell_key: [records]}``, sorted - the shared grouping of a table read."""
    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(cell_key(r), []).append(r)
    return dict(sorted(groups.items()))


def _key(rec: dict) -> tuple:
    return (rec.get("scene"), rec.get("start_id"))


def select(records, *, arm: str | None = None, scene: str | None = None,
           method: str | None = None, stop: str | None = None,
           cell: str | None = None, roll_cap: float | None = None,
           start_ids=None) -> dict[tuple, dict]:
    """Filter records into ``{(scene, start_id): record}`` - one cell of a table.

    ``arm`` matches :func:`arm_of`; ``method`` matches ``base_method`` on a stored record
    (which is what separates the SIFT and CNS bars from the ViT arms, since both are
    recorded with the same ``off`` arm label) or ``matcher_variant`` on a new one;
    ``roll_cap`` keeps starts with ``|start_roll_deg| <= cap`` (``n37b_verdicts``'s
    ``sub60``); ``cell`` matches
    ``start_cell``. The mapping is by ``(scene, start_id)``, which is what makes
    two cells pairable.
    """
    ids = None if start_ids is None else set(start_ids)
    out: dict[tuple, dict] = {}
    for r in records:
        if arm is not None and arm_of(r) != arm:
            continue
        if scene is not None and r.get("scene") != scene:
            continue
        if method is not None and method not in (r.get("base_method"),
                                                 r.get("matcher_variant")):
            continue
        if stop is not None and r.get("stop_mode") != stop:
            continue
        if cell is not None and r.get("start_cell") != cell:
            continue
        if roll_cap is not None and abs(float(r.get("start_roll_deg") or 0.0)) > roll_cap:
            continue
        if ids is not None and r.get("start_id") not in ids:
            continue
        out[_key(r)] = r
    return out


def _med(values):
    v = [x for x in values if x is not None]
    return round(st.median(v), 4) if v else None


def conv_stats(rows: dict) -> dict:
    """``{n, conv, conv_med, conv_rot}`` - the cap-scored converged read
    (``n36_verdicts.conv_stats`` widened by ``n37b_verdicts.stats``'s rotation column)."""
    conv = [r for r in rows.values() if r.get("converged_a")]
    return {"n": len(rows), "conv": len(conv),
            "conv_med": _med([r.get("final_translation_mm") for r in conv]),
            "conv_rot": _med([r.get("final_rotation_deg") for r in conv])}


def stop_stats(rows: dict) -> dict:
    """``{n, fires, stop_med, stop_step_med, cap_med}`` (``n36_verdicts.stop_stats``)."""
    fired = [r for r in rows.values() if r.get("stop_step") is not None]
    return {"n": len(rows), "fires": len(fired),
            "stop_med": _med([r.get("stop_translation_mm") for r in fired]),
            "stop_step_med": _med([r.get("stop_step") for r in fired]),
            "cap_med": _med([r.get("final_translation_mm") for r in rows.values()])}


def bootstrap_ci_median(diffs, b: int = BOOTSTRAP_B,
                        seed: int = BOOTSTRAP_SEED, stat=None) -> tuple[float, float]:
    """95 % percentile interval of the median of paired differences.

    ``n1_verdicts.bootstrap_ci_median`` verbatim: ``default_rng(seed)`` draws
    ``b`` index rows with replacement, the median of each resample is taken, and
    the 2.5th and 97.5th percentiles are returned. Fewer than two differences
    gives ``(nan, nan)``.

    ``stat`` swaps the statistic taken of each resample, leaving the resampling - the
    draw, the count and the seed - exactly as it is. The one caller that needs it is a
    difference of **shares**: the median of a mostly-zero binary vector is 0 whatever
    happens to it, which is the reason `compare`'s `reached` verdict already carries a
    count beside its interval, and a share is the statistic that moves.
    """
    d = np.asarray(diffs, dtype=np.float64)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(int(b), len(d)))
    values = (np.median if stat is None else stat)(d[idx], axis=1)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def paired_diffs(rows_a: dict, rows_b: dict, field: str, *,
                 both_converged: bool = True, both_fired: bool = False) -> list[float]:
    """Per-start ``(a - b)`` over the starts both cells ran; positive => **b** is closer."""
    diffs = []
    for k, ra in rows_a.items():
        rb = rows_b.get(k)
        if rb is None:
            continue
        if both_converged and not (ra.get("converged_a") and rb.get("converged_a")):
            continue
        if both_fired and (ra.get("stop_step") is None or rb.get("stop_step") is None):
            continue
        va, vb = ra.get(field), rb.get(field)
        if va is None or vb is None:
            continue
        diffs.append(float(va) - float(vb))
    return diffs


def paired(rows_a: dict, rows_b: dict, field: str = "final_translation_mm", *,
           both_converged: bool = True, both_fired: bool = False,
           min_paired: int = MIN_PAIRED, b: int = BOOTSTRAP_B,
           seed: int = BOOTSTRAP_SEED) -> dict | None:
    """The paired contrast: ``{n, median, ci95, wins}``, or ``None`` below ``min_paired``.

    ``both_converged`` defaults to **True** (``n37b_verdicts``'s convention, which
    every N3.7 read used: comparing landing precision only makes sense where both
    arms landed). ``n36_verdicts`` defaulted it False for its reaching reads; pass
    it explicitly when that is what you mean.
    """
    d = paired_diffs(rows_a, rows_b, field,
                     both_converged=both_converged, both_fired=both_fired)
    if len(d) < min_paired:
        return None
    lo, hi = bootstrap_ci_median(d, b=b, seed=seed)
    return {"n": len(d), "median": round(st.median(d), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "wins": sum(1 for x in d if x > 0)}


def ci_sign(result: dict | None) -> str:
    """Three-way read of a paired interval: ``pos`` (b better) / ``neg`` / ``zero``."""
    if result is None:
        return "insufficient"
    lo, hi = result["ci95"]
    if lo > 0:
        return "pos"
    if hi < 0:
        return "neg"
    return "zero"


# ------------------------------------------------------------------ strata

#: Names of the three rotation-difficulty thirds, in order (protocol v2 section 3).


# The start-set readers `roll_thirds` and `capped_population` live in `vsbench.vs.starts`.
from vsbench.vs.starts import THIRDS, capped_population, roll_thirds  # noqa: E402,F401
