# vs-sim copy: vssim/bench/debuglog.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The debug log: a recorder that listens to one flight, step by step.

Purpose
-------
The owner's request of 2026-09-30: debug the matching properly. For every control step
the recorder writes down where the camera was, how far it was from three references
(the goal pose, the size-matched pose, the depth-true pose), the error split in the
goal camera's frame (sideways, along the viewing axis, tilt, roll), which goal point the
method paired with which live point, whether each pairing was **right** (where a truth
exists), what the method commanded and what every stop rule said. One gzipped JSON line
per event goes into ``telemetry.jsonl.gz`` beside the episode's film, together with the
goal picture and the goal mask, so ``scripts/debug_view.py`` can draw the whole flight
later without re-flying it.

It only listens. :func:`vsbench.vs.bench.runner.run_episode` hands it copies after the step's
twist and stop verdicts exist and before the camera moves; nothing it returns is read,
and with no recorder the loop is exactly what it was (``tests/fast/test_debuglog.py``
flies each arm kind with and without it and compares the records and the poses bit for
bit).

It can never cost a flight (review 2026-09-30). Each of its four calls (:meth:`DebugLog.begin`,
``step``, ``final``, ``end``) and :meth:`DebugLog.close` catches its own failure (a full
disk, a gzip error, an odd shape in a method's debug dict): the first one writes one
``{"kind": "error"}`` line (best effort), sets :attr:`DebugLog.error`, and every later
call does nothing. The flight goes on untouched; the record then carries
``debug_log_error``.

The truth per matched pair (:func:`truth_block`)
------------------------------------------------
A goal pixel is a 3-D point (the goal depth and the goal pose). Where that point is in
the live picture depends on what happened to the world, which the perturbation states
(:meth:`vsbench.vs.bench.perturb.Perturbation.truth_model`): the same object (``rigid``), the
same object at another size (``scaled``: ``X -> P + s (X - P)`` about the floor centre)
or another object (``none``: no truth). A point more than 5 mm above the floor plane is
an object point; a floor point never moves, but on a re-textured floor its truth is
marked apart. (Caveat: in the scaled case an object point 5 mm or less above the floor -
the rim of a sole or a base, seen from above - counts as floor and is not moved by the
scaling; its true spot is then off by up to ``(1 - s)`` times its sideways distance from
the centre. On the re-fly's re-textured floor such points get the letter ``F``, not a
verdict, so only a handful of rim points are affected.) Each pair gets one letter:

- ``R`` right: the live pixel is within one matching cell of the true pixel;
- ``W`` wrong: further than that;
- ``N`` no truth (a point of another object);
- ``F`` a floor point on a re-textured floor (geometry known, "right" is luck);
- ``O`` the true point is outside the live picture or hidden (occlusion test of
  :mod:`vsbench.vs.labels`);
- ``?`` the method has no cell size here (an arm kind this log does not size).

The raw pixel error is always kept, so any other threshold can be applied later, and its
size on the object (``err_mm = err_px * z / f``, ``z`` the true point's depth in the live
camera). Because the cell differs about seven-fold between the methods, every step also
carries one yardstick that is the same for all of them (``near``): the share of the pairs
with a truth whose true spot is within :data:`TRUTH_COMMON_PX` full-image pixels, and the
median pixel error.

One matching cell (:func:`cell_px`), in full-image pixels
---------------------------------------------------------
The smallest piece of picture the method can tell apart, from what actually ran:
ViT-VS (DINOv3 ViT-S/16 on the picture squashed to 512 x 512) ``W / 32 x H / 32`` = 60 x
33.75 px; DINOBot (DINO ViT-S/8 at stride 4 on the picture resized to 224 x 398) ``stride
* (W / load_w, H / load_h)`` = 19.3 x 19.3 px; ART-VS's object mode ``enc_patch /
enc_input`` of its live box (coarse), of one tile (tiled) or of the whole picture (no box)
- with the DINOv2 re-encode at 518 px that is 14 / 518 of the box, a few pixels. The cell
differs per method, so shares of right matches are only comparable with that said.

Provenance
----------
New in vs-sim (2026-09-30, design ``DESIGN.md`` on ``feat/debug-telemetry``); nothing
ported. The truth is :func:`vsbench.vs.labels.gt_correspondence_mapped`; the DINOBot residual
uses the arm's own :func:`vsbench.vs.baselines.dinobot.kabsch` and
:func:`~vsbench.vs.baselines.dinobot.backproject`. The old telemetry columns ``gt_v`` /
``gt_r`` (:func:`vsbench.vs.bench.runner.step_telemetry`) assume a world that did not change
and are left as they are; for a swapped or scaled object read this log's ``truth``.

2026-10-11 (the external-arm hook): an arm that fills ``debug["method"]`` with a dict - its
own per-step readings, already rounded by the arm - gets it copied into the step line as
``method`` (JSON-safe, non-finite to ``null``). No other arm fills it, so no other line moves.

Golden tests: ``tests/fast/test_debuglog.py``.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from vsbench.vs.scene import Pose, pose_to_T, rotation_error_deg

#: The telemetry file's schema tag (``meta.schema``).
SCHEMA = "vssim-debug-telemetry/1"

#: The method names a person reads, by arm-id prefix (the first match wins).
DISPLAY_NAMES = (("dinobot", "DINOBot"), ("vitvs", "ViT-VS"), ("artvs", "ART-VS"),
                 ("cns", "CNS (v1)"), ("visp", "ViSP IBVS"))

#: A match is RIGHT when its error is at most this many matching cells of its method.
TRUTH_RIGHT_CELLS = 1.0
#: The yardstick common to every method (``near``): a pair with a truth is NEAR when its
#: true spot is within this many full-image pixels (about 5 mm on the object at the
#: instance scenes' ~0.34 m, fx ~1400 px). The per-method cell is not comparable across
#: methods (ViT-VS 60 x 34 px, DINOBot 19 px, ART-VS a few px); this is.
TRUTH_COMMON_PX = 20.0
#: A goal point more than this above the floor plane is an OBJECT point (the objects
#: mask's own 5 mm, :func:`vsbench.vs.labels.plane_membership`).
OBJECT_TOL_M = 0.005
#: Flush the gzip stream every this many steps (a killed run loses at most these).
FLUSH_EVERY = 25
#: The runner's end reasons that break the loop BEFORE the step's move (no move follows).
BREAK_REASONS = ("non_finite_velocity", "diverged_feature")

#: One sentence per arm kind: how its matching cell is computed (``meta.cell_rule``).
CELL_RULES = {
    "dense": "one encoder patch in full-image pixels: (W / side, H / side) of the token grid",
    "dinobot": "one token step: stride * (W / load_w, H / load_h) of the resized picture",
    "artvs": "enc_patch / enc_input of the live box (coarse), of one tile (tiled), or of "
             "the whole picture (no box) - the encoder that actually ran",
    "none": "no cell size for this matcher: letters are '?', the raw pixel error is kept",
}


def display_name(arm_id: str) -> str:
    """``dinobot_objmask`` -> ``DINOBot``, ``vitvs_dinov3_objmask`` -> ``ViT-VS``,
    ``artvs_roi_scene`` -> ``ART-VS``; anything else is its own id."""
    a = str(arm_id)
    for prefix, name in DISPLAY_NAMES:
        if a.startswith(prefix):
            return name
    return a


# ---------------------------------------------------------------- numbers
def _num(x, nd: int):
    """A rounded float, or ``None`` for a missing or non-finite value (never 0)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _nums(a, nd: int) -> list | None:
    if a is None:
        return None
    return [_num(v, nd) for v in np.asarray(a, np.float64).reshape(-1)]


def _jsonable(x):
    """A JSON-safe copy: numpy to lists and floats, non-finite to ``None``."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    if x is None or isinstance(x, str):
        return x
    return str(x)


def pose_dict(pose: Pose) -> dict:
    """``{"p": [3], "R": [9 row-major]}`` at 6 dp."""
    return {"p": _nums(pose.position, 6), "R": _nums(pose.rotation, 6)}


# -------------------------------------------------------------- error split
def error_split(position, ref_position, goal_rotation) -> dict:
    """Distance and the position error in the GOAL camera's frame, mm.

    ``e_mm = 1000 R_goal^T (p - p_ref)``: ``x`` toward image right, ``y`` toward image
    bottom, ``z`` along the goal's viewing direction (+ = the camera is further forward,
    closer to the object). A reader shows height = -z.
    """
    d = np.asarray(position, np.float64).reshape(3) - np.asarray(ref_position,
                                                                 np.float64).reshape(3)
    e = 1000.0 * (np.asarray(goal_rotation, np.float64).T @ d)
    return {"t_mm": _num(np.linalg.norm(d) * 1000.0, 3), "e_mm": _nums(e, 3)}


def orientation_split(rotation, goal_rotation) -> dict:
    """The angle to the goal orientation, split into tilt and roll (degrees).

    ``R_rel = R_goal^T R``. **Tilt**: the angle between the two viewing axes,
    ``arccos(R_rel[2, 2])``; **tilt direction**: where in the goal picture the current
    axis leans, ``atan2(R_rel[1, 2], R_rel[0, 2])``; **roll**: the turn about the viewing
    axis, the swing-twist twist ``2 atan2(q_z, q_w)`` of ``R_rel``'s quaternion
    (``tan(roll / 2) = (R10 - R01) / (1 + trace)``), wrapped to (-180, 180].
    """
    R = np.asarray(rotation, np.float64).reshape(3, 3)
    Rg = np.asarray(goal_rotation, np.float64).reshape(3, 3)
    rel = Rg.T @ R
    tilt = math.degrees(math.acos(float(np.clip(rel[2, 2], -1.0, 1.0))))
    tilt_dir = math.degrees(math.atan2(rel[1, 2], rel[0, 2])) if tilt > 1e-9 else 0.0
    roll = math.degrees(2.0 * math.atan2(rel[1, 0] - rel[0, 1], 1.0 + np.trace(rel)))
    roll = (roll + 180.0) % 360.0 - 180.0
    if roll == -180.0:
        roll = 180.0
    return {"r_deg": _num(rotation_error_deg(R, Rg), 4), "tilt_deg": _num(tilt, 4),
            "tilt_dir_deg": _num(tilt_dir, 4), "roll_deg": _num(roll, 4)}


# ----------------------------------------------------------- matching cell
def arm_kind_of(matches, debug: dict | None = None) -> str:
    """``artvs`` / ``dinobot`` / ``dense`` / ``none`` - which cell rule applies."""
    ex = getattr(matches, "extras", None) or {}
    if "enc_input" in ex or "roi_source" in ex:
        return "artvs"
    if "load_hw" in ex or (debug or {}).get("policy") == "dinobot_align":
        return "dinobot"
    side, wh = getattr(matches, "side", None), getattr(matches, "image_wh", None)
    if side and wh and int(side) != int(wh[0]):
        return "dense"
    return "none"


def cell_px(matches, image_wh, *, dinobot_stride: int | None = None) -> list | None:
    """The method's matching cell this step, ``[cu, cv]`` full-image pixels, or ``None``."""
    if matches is None:
        return None
    W, H = float(image_wh[0]), float(image_wh[1])
    ex = getattr(matches, "extras", None) or {}
    kind = arm_kind_of(matches)
    if kind == "artvs":
        if not ex.get("enc_input") or not ex.get("enc_patch"):
            return None
        a = float(ex["enc_patch"]) / float(ex["enc_input"])
        box = ex.get("live_box")
        if box is None or ex.get("mode") == "full":
            return [a * W, a * H]
        bw, bh = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
        if ex.get("mode") == "tiled":
            gw, gh = (int(g) for g in ex.get("grid") or (1, 1))
            return [a * bw / gw, a * bh / gh]
        return [a * bw, a * bh]
    if kind == "dinobot":
        lhw = ex.get("load_hw")
        if lhw is None or not dinobot_stride:
            return None
        return [float(dinobot_stride) * W / float(lhw[1]),
                float(dinobot_stride) * H / float(lhw[0])]
    if kind == "dense":
        side = float(matches.side)
        return [W / side, H / side]
    return None


# ------------------------------------------------------------------ truth
def truth_block(pts_goal, pts_cur, *, goal_depth, K, goal_pose: Pose, live_pose: Pose,
                live_depth, model: dict, cell) -> dict:
    """The true live pixel of every goal point and the letter of every pair.

    ``model`` is a :meth:`vsbench.vs.bench.perturb.Perturbation.truth_model` dict. Returns
    ``{"tu", "tv", "err_px", "err_mm", "err_cell", "stat"}`` (one entry per pair,
    ``stat`` one letter per pair - see the module docstring), ``"right"`` (the letter
    counts) and ``"near"``: the common yardstick over the pairs that have a true spot in
    view (letters ``R``, ``W``, ``?``) - ``{"px", "k", "n", "median_px", "median_mm",
    "z_med_m"}`` (``k`` of ``n`` within :data:`TRUTH_COMMON_PX`).
    """
    from vsbench.vs.labels import backproject, gt_correspondence_mapped, transform

    pg = np.asarray(pts_goal, np.float64).reshape(-1, 2)
    pc = np.asarray(pts_cur, np.float64).reshape(-1, 2)
    n = len(pg)
    counts = {k: 0 for k in "RWNFO?"}
    near = {"px": TRUTH_COMMON_PX, "k": 0, "n": 0, "median_px": None, "median_mm": None,
            "z_med_m": None}
    if n == 0:
        return {"tu": [], "tv": [], "err_px": [], "err_mm": [], "err_cell": [], "stat": "",
                "right": counts, "near": near}
    T_g, T_l = pose_to_T(goal_pose), pose_to_T(live_pose)
    X_a, ok_a = backproject(pg, goal_depth, K)
    X_w = transform(T_g, X_a)
    floor_z = model.get("floor_z", 0.0)
    is_obj = (np.ones(n, bool) if floor_z is None
              else X_w[:, 2] > float(floor_z) + OBJECT_TOL_M)
    obj_kind = model.get("object", "rigid")
    world_map = None
    if obj_kind == "scaled":
        s = float(model["s"])
        P = np.array([float(model["centre_xy"][0]), float(model["centre_xy"][1]),
                      0.0 if floor_z is None else float(floor_z)])

        def world_map(X, _s=s, _P=P, _obj=is_obj):
            X = np.asarray(X, np.float64)
            return np.where(_obj[:, None], _P + _s * (X - _P), X)

    uv, valid, z_b, _X = gt_correspondence_mapped(pg, goal_depth, K, T_g, T_l,
                                                  depth_b=live_depth, world_map=world_map)
    d = pc - uv
    err_px = np.hypot(d[:, 0], d[:, 1])
    # the same error on the object: one pixel is z / f metres at the true point's depth
    f = 0.5 * (float(K[0][0]) + float(K[1][1]))
    err_mm = err_px * np.asarray(z_b, np.float64) / f * 1000.0
    if cell is not None:
        err_cell = np.hypot(d[:, 0] / float(cell[0]), d[:, 1] / float(cell[1]))
    else:
        err_cell = np.full(n, np.nan)
    retex = model.get("floor") == "retextured"
    stat = []
    for i in range(n):
        if is_obj[i] and obj_kind == "none":
            letter = "N"
        elif not ok_a[i] or not valid[i]:
            letter = "O"
        elif not is_obj[i] and retex:
            letter = "F"
        elif cell is None:
            letter = "?"
        else:
            letter = "R" if err_cell[i] <= TRUTH_RIGHT_CELLS else "W"
        stat.append(letter)
        counts[letter] += 1
    has_truth = np.array([s_ != "N" for s_ in stat])
    tu = [(_num(uv[i, 0], 2) if has_truth[i] else None) for i in range(n)]
    tv = [(_num(uv[i, 1], 2) if has_truth[i] else None) for i in range(n)]
    judged = np.array([s_ in "RW?" for s_ in stat])
    if judged.any():
        near = {"px": TRUTH_COMMON_PX, "k": int(np.sum(err_px[judged] <= TRUTH_COMMON_PX)),
                "n": int(judged.sum()), "median_px": _num(np.median(err_px[judged]), 2),
                "median_mm": _num(np.median(err_mm[judged]), 2),
                "z_med_m": _num(np.median(np.asarray(z_b, np.float64)[judged]), 4)}
    return {"tu": tu, "tv": tv,
            "err_px": [(_num(err_px[i], 2) if has_truth[i] else None) for i in range(n)],
            "err_mm": [(_num(err_mm[i], 2) if has_truth[i] else None) for i in range(n)],
            "err_cell": [(_num(err_cell[i], 3) if has_truth[i] else None) for i in range(n)],
            "stat": "".join(stat), "right": counts, "near": near}


# --------------------------------------------------------- the arms' blocks
def dinobot_block(debug: dict, intrinsics, reflection: str = "none") -> dict:
    """DINOBot's own numbers this step, and the rigid fit's residual recomputed with the
    arm's own ``kabsch`` / ``backproject`` on the pairs the fit used."""
    from vsbench.vs.baselines import dinobot as db

    out = {k: _jsonable(debug.get(k)) for k in (
        "n_buddies", "n_buddies_raw", "n_fg_goal", "n_fg_live", "n_selected",
        "n_dropped_depth", "degenerate", "reflection")}
    for k, nd in (("err_pixel_load", 4), ("err_metric", 6), ("fit_t_mm", 3), ("fit_det", 6),
                  ("jump_t_mm", 3), ("jump_rot_deg", 4)):
        out[k] = _num(debug.get(k), nd)
    zg, zc = debug.get("goal_depth_values"), debug.get("depth_values")
    out["zg"], out["zc"] = _nums(zg, 5), _nums(zc, 5)
    out["fit_residual_mm"] = out["fit_residual_max_mm"] = None
    pg = np.asarray(debug.get("pts_goal", np.zeros((0, 2))), np.float64).reshape(-1, 2)
    pc = np.asarray(debug.get("pts_cur", np.zeros((0, 2))), np.float64).reshape(-1, 2)
    if zg is not None and zc is not None and len(pg) >= 3 and len(pg) == len(np.ravel(zg)):
        X = db.backproject(pg, zg, intrinsics)
        Y = db.backproject(pc, zc, intrinsics)
        R, t = db.kabsch(X, Y, reflection)
        r = np.linalg.norm((R @ X.T).T + t - Y, axis=1) * 1000.0
        out["fit_residual_mm"] = _num(np.sqrt(np.mean(r ** 2)), 3)
        out["fit_residual_max_mm"] = _num(r.max(), 3)
    return out


def artvs_block(matches) -> dict:
    """ART-VS's object mode this step: the live box, where it came from, the mode, the
    grid, points per tile, the switch's errors and the encoder that ran."""
    ex = getattr(matches, "extras", None) or {}
    e0, e = ex.get("switch_e0"), ex.get("switch_e")
    ratio = (_num(float(e) / float(e0), 4)
             if e is not None and e0 not in (None, 0) and math.isfinite(float(e0)) else None)
    return {"box": _jsonable(ex.get("live_box")), "source": ex.get("roi_source"),
            "mode": ex.get("mode"), "grid": _jsonable(ex.get("grid")),
            "tiled": bool(ex.get("tiled", False)),
            "tile_counts": _jsonable(ex.get("tile_counts")),
            "switch": {"e0_px": _num(e0, 3), "e_px": _num(e, 3), "ratio": ratio},
            "enc": [ex.get("enc_input"), ex.get("enc_patch")]}


def _pairs(debug: dict, matches) -> dict:
    """The matched pairs this step: the matcher's own pixels (unrounded) when its rows are
    the rows the controller used, else the controller's; score, weight, landmark id."""
    zero = np.zeros((0, 2))
    pg = np.asarray(debug.get("pts_goal", debug.get("pts_desired", zero)), np.float64)
    pc = np.asarray(debug.get("pts_cur", debug.get("pts_current", zero)), np.float64)
    pg, pc = pg.reshape(-1, 2), pc.reshape(-1, 2)
    n = len(pc)
    score = lid = None
    if matches is not None and len(matches) == n and n:
        rows = matches.to_rows()
        pg, pc = np.asarray(rows["pts_goal"], np.float64), np.asarray(rows["pts_cur"], np.float64)
        score = np.asarray(matches.sim, np.float64)
        ex = matches.extras or {}
        if ex.get("goal_token") is not None and len(np.ravel(ex["goal_token"])) == n:
            lid = [int(v) for v in np.ravel(ex["goal_token"])]
        elif arm_kind_of(matches, debug) == "dense":
            side = int(matches.side)
            lid = [int(v) for v in matches.goal_patch[:, 1] * side + matches.goal_patch[:, 0]]
    w = debug.get("weights")
    if w is not None:
        w = np.asarray(w, np.float64)
        w = w[:, 0] if w.ndim == 2 else w
        if len(w) != n:
            w = None
    return {"pg": pg, "pc": pc,
            "block": {"gu": _nums(pg[:, 0], 2), "gv": _nums(pg[:, 1], 2),
                      "cu": _nums(pc[:, 0], 2), "cv": _nums(pc[:, 1], 2),
                      "score": _nums(score, 4), "w": _nums(w, 5), "lid": lid}}


def _write_png_mask(path: Path, mask) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(mask, bool)).convert("1").save(path, optimize=True)


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


# ------------------------------------------------------------ the recorder
class DebugLog:
    """The per-episode recorder: ``telemetry.jsonl.gz``, ``goal.jpg``, ``goal_mask.png``.

    Owned by the caller (``scripts/check_baselines.py``) as a context manager; the runner
    calls :meth:`begin`, :meth:`step` (once per control step), :meth:`final` and
    :meth:`end`. ``film``: a :class:`vsbench.vs.bench.films.FilmWriter` the recorder drives
    with the richer captions (method name, three distances, landing marks, a final
    frame). ``clean``: also write the clean half-size live film (``clean.mp4``) for the
    offline viewer. :attr:`seconds` is the recorder's own time.

    None of the five calls raises: the first failure is written down (:attr:`error`, one
    ``error`` line in the file) and the recorder goes quiet for the rest of the episode
    (:attr:`broken`), so a logging fault can never change what is flown.
    """

    def __init__(self, out_dir, *, arm_id: str = "", display: str | None = None,
                 film=None, clean: bool = False) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.arm_id = str(arm_id)
        self.display = display or display_name(arm_id)
        self.film = film
        self.clean = bool(clean)
        self.clean_writer = None
        self.path = self.dir / "telemetry.jsonl.gz"
        self._f = gzip.open(self.path, "wt", encoding="utf-8")   # noqa: SIM115 - close()
        self.seconds = 0.0
        self._n = 0
        self._step_i = 0
        self._closed = False
        #: the first failure inside the recorder (``"<call>: <exception>"``), else ``None``
        self.error: str | None = None
        #: set by the first failure: every later call does nothing
        self.broken = False

    # -------------------------------------------------------------- plumbing
    def __enter__(self) -> DebugLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def _fail(self, where: str, exc: BaseException) -> None:
        """The first failure: remember it, write one ``error`` line if the file still
        takes one, and go quiet."""
        self.broken = True
        if self.error is None:
            self.error = f"{where}: {exc!r}"[:500]
        try:
            self._write({"kind": "error", "where": where, "step": int(self._step_i),
                         "exc": repr(exc)[:2000]})
            self._f.flush()
        except Exception:  # noqa: BLE001 - best effort; the flight must not see it
            pass

    def _guarded(self, where: str, fn, *args, **kwargs) -> None:
        """Run one of the recorder's calls so that nothing it does can reach the flight."""
        if self.broken or self._closed:
            return
        t0 = time.monotonic()
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the recorder only listens
            self._fail(where, exc)
        finally:
            self.seconds += time.monotonic() - t0

    def close(self) -> None:
        """Flush and close the file; encode the clean film. Safe to call twice; never
        raises (a failure lands in :attr:`error`)."""
        if self._closed:
            return
        self._closed = True
        t0 = time.monotonic()
        try:
            self._f.close()
        except Exception as exc:  # noqa: BLE001
            self.broken = True
            self.error = self.error or f"close: {exc!r}"[:500]
        if self.clean_writer is not None:
            try:
                self.clean_writer.close()
            except Exception as exc:  # noqa: BLE001
                self.error = self.error or f"clean film: {exc!r}"[:500]
        self.seconds += time.monotonic() - t0

    # ---------------------------------------------------- the four calls (guarded)
    def begin(self, **kwargs) -> None:
        """The goal side, once (:meth:`_begin`)."""
        self._guarded("begin", self._begin, **kwargs)

    def step(self, **kwargs) -> None:
        """One control step, after its decisions and before the camera moves (:meth:`_step`)."""
        self._step_i = int(kwargs.get("step_i", self._step_i))
        self._guarded("step", self._step, **kwargs)

    def final(self, rgb, depth, pose: Pose, **kwargs) -> None:
        """The pose after the last move (:meth:`_final`)."""
        self._guarded("final", self._final, rgb, depth, pose, **kwargs)

    def end(self, rec: dict) -> None:
        """The outcome, from the finished record (:meth:`_end`)."""
        self._guarded("end", self._end, rec)

    # ------------------------------------------------------------------ begin
    def _begin(self, *, scene, arm, start: dict, start_pose: Pose, goal: Pose, goal_rgb,
               goal_depth, goal_mask, references: dict, truth_model: dict, draw: dict,
               settings: dict) -> None:
        """The goal side, once: the ``meta`` line, ``goal.jpg``, ``goal_mask.png``."""
        import cv2  # noqa: PLC0415 - bench layer, visualisation only

        intr = scene.intrinsics
        self.K = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]])
        self.intrinsics = intr
        self.image_wh = (int(intr.width), int(intr.height))
        self.goal = goal
        self.goal_depth = np.asarray(goal_depth, np.float64)
        self.refs = {"goal": goal, "size_matched": references.get("size_matched"),
                     "depth_true": references.get("depth_true")}
        self.truth = dict(truth_model)
        self.scored_rule = str(getattr(arm.stop, "name", ""))
        self.own_rules = list((getattr(arm, "recorded_stop_constants", dict)() or {}).keys())
        matcher = getattr(arm, "matcher", None)
        fe = getattr(matcher, "front_end", None) or {}
        self.dinobot_stride = int(fe["stride"]) if "stride" in fe else None
        self.reflection = str(getattr(getattr(arm, "controller", None), "reflection", "none"))
        self.scene_name, self.start_id = scene.name, start["start_id"]
        swap = draw.get("swap_instance", draw) if isinstance(draw, dict) else {}
        self.live_name = swap.get("live_name") if isinstance(swap, dict) else None

        # the arm's goal side, read-only
        arm_goal: dict[str, Any] = {}
        tracker = getattr(matcher, "tracker", None)
        if tracker is not None and hasattr(matcher, "goal_box"):
            arm_goal["artvs"] = {
                "goal_box": _jsonable(matcher.goal_box),
                "goal_score": _jsonable(getattr(tracker, "goal_score", None)),
                "grid": _jsonable(list(getattr(matcher, "grid", None) or (1, 1))),
                "vit_input": _jsonable(getattr(matcher, "vit", None)),
                "switch_threshold": _jsonable(getattr(matcher, "switch_threshold", None)),
                "prompt": _jsonable(getattr(matcher, "prompt", None))}
        elif self.dinobot_stride is not None:
            g = getattr(matcher, "goal", None)
            arm_goal["dinobot"] = {
                "goal_load_hw": _jsonable(list(getattr(g, "load_hw", ()) or ())) or None,
                "stride": self.dinobot_stride,
                "err_threshold": _jsonable(getattr(arm, "err_threshold", None)),
                "err_space": _jsonable(getattr(arm.controller, "err_space", None))}

        # the goal picture (half size) and the mask the method was given
        gw, gh = _even(self.image_wh[0] * 0.5), _even(self.image_wh[1] * 0.5)
        gbgr = cv2.cvtColor(np.ascontiguousarray(np.asarray(goal_rgb, np.uint8)[..., :3]),
                            cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.dir / "goal.jpg"),
                    cv2.resize(gbgr, (gw, gh), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        sam = getattr(tracker, "goal_mask", None) if tracker is not None else None
        if sam is not None:
            mask, source, given = np.asarray(sam, bool), "sam", True
        elif goal_mask is not None:
            mask, source, given = np.asarray(goal_mask, bool), "objects_from_depth", True
        else:
            from vsbench.vs.labels import object_mask_from_depth

            mask = object_mask_from_depth(goal_depth, intr, goal)
            source, given = "objects_from_depth", False
        _write_png_mask(self.dir / "goal_mask.png", mask)

        heights = None
        if isinstance(swap, dict) and swap.get("goal_bbox") and swap.get("live_bbox"):
            heights = {"goal_top_z": _num(swap["goal_bbox"][1][2], 6),
                       "live_top_z": _num(swap["live_bbox"][1][2], 6)}
        kind = ("artvs" if "artvs" in arm_goal else "dinobot" if "dinobot" in arm_goal
                else "dense" if (getattr(matcher, "side", None)
                                 or getattr(getattr(matcher, "backbone", None), "grid_side",
                                            None)) else "none")
        title = f"{self.display}  |  {self.scene_name}"
        if self.live_name:
            title += f" / {self.live_name}"
        title += f"  |  start {self.start_id}"
        # the world change's name, as the old film title carried it ("[chain]")
        mode = draw.get("mode") if isinstance(draw, dict) else None
        if mode and mode != "none":
            title += f"  |  {mode}"
        self.title = title
        if self.film is not None:
            self.film.title = title
            # a crashed earlier run's frames in the same directory would be globbed into
            # this film: start clean (the debug path only; the old path is untouched)
            for f in Path(self.film.dir).glob("step_*.jpg"):
                f.unlink()
        if self.clean:
            from vsbench.vs.bench.films import CleanFilmWriter

            self.clean_writer = CleanFilmWriter(self.dir)
        tm = {**self.truth, "right_cells": TRUTH_RIGHT_CELLS, "common_px": TRUTH_COMMON_PX,
              "object_tol_m": OBJECT_TOL_M}
        from vsbench.vs.labels import OCCLUSION_ABS_TOL_M, OCCLUSION_REL_TOL

        tm["occlusion"] = {"abs_m": OCCLUSION_ABS_TOL_M, "rel": OCCLUSION_REL_TOL}
        self._write({
            "kind": "meta", "schema": SCHEMA, "scene": self.scene_name, "arm_id": self.arm_id,
            "display": self.display, "label": str(getattr(arm, "name", "")),
            "start_id": self.start_id, "live_name": self.live_name, "title": title,
            "max_iter": int(settings["max_iter"]), "dt": _num(getattr(arm, "dt", None), 6),
            "image_wh": list(self.image_wh),
            "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
            "poses": {"start": pose_dict(start_pose), "goal": pose_dict(goal),
                      "size_matched": (pose_dict(self.refs["size_matched"])
                                       if self.refs["size_matched"] is not None else None),
                      "depth_true": (pose_dict(self.refs["depth_true"])
                                     if self.refs["depth_true"] is not None else None)},
            "goal_tilt_deg": orientation_split(goal.rotation, np.diag([1.0, -1.0, -1.0]))[
                "tilt_deg"],
            "heights": heights, "truth": _jsonable(tm), "cell_rule": CELL_RULES[kind],
            "stops": {"scored": self.scored_rule, "own": self.own_rules,
                      "recorded": list(settings.get("record_stops") or [])},
            "perturbation": _jsonable(draw), "arm_goal": arm_goal,
            "files": {"goal": "goal.jpg", "goal_mask": "goal_mask.png",
                      "mask_source": source, "mask_given_to_method": given,
                      "film": "episode.mp4" if self.film is not None else None,
                      "clean_film": "clean.mp4" if self.clean else None}})
        self._f.flush()

    # ------------------------------------------------------------- the reads
    def _to(self, pose: Pose) -> dict:
        return {name: (error_split(pose.position, ref.position, self.goal.rotation)
                       if ref is not None else None) for name, ref in self.refs.items()}

    def _film_refs(self, pose: Pose) -> tuple[dict, float]:
        """The film's distances (mm) and angle (deg), UNROUNDED: the film rounds them once
        itself, so its old ``t_mm`` / ``r_deg`` keys are exactly the old path's."""
        from vsbench.vs.scene import translation_error_mm

        refs = {name: (translation_error_mm(pose.position, ref.position)
                       if ref is not None else None) for name, ref in self.refs.items()}
        return refs, rotation_error_deg(pose.rotation, self.goal.rotation)

    def _landing_labels(self, names) -> list[str]:
        out = []
        for n in names:
            if n in self.own_rules:
                out.append(f"{self.display}'s own stop ({n})")
            elif n == self.scored_rule:
                out.append(f"the scored stop ({n})")
        return out

    # ------------------------------------------------------------------- step
    def _step(self, *, step_i: int, rgb, depth, pose: Pose, twist_raw, v_cmd, lam: float,
              debug: dict, matches, scored, recorded: dict | None, fired: dict,
              fired_now: list, end: str | None) -> None:
        """One control step, after its decisions and before the camera moves."""
        to = self._to(pose)
        ang = orientation_split(pose.rotation, self.goal.rotation)
        pr = _pairs(debug, matches)
        cell = cell_px(matches, self.image_wh, dinobot_stride=self.dinobot_stride)
        truth = truth_block(pr["pg"], pr["pc"], goal_depth=self.goal_depth, K=self.K,
                            goal_pose=self.goal, live_pose=pose, live_depth=depth,
                            model=self.truth, cell=cell)
        right = truth.pop("right")
        near = truth.pop("near")
        # the cell's size on the object, at the median depth of the judged true points
        f = 0.5 * (float(self.K[0, 0]) + float(self.K[1, 1]))
        cell_mm = ([c * near["z_med_m"] / f * 1000.0 for c in cell]
                   if cell is not None and near["z_med_m"] else None)
        feat = debug.get("feature_error_norm")
        entry = {
            "kind": "step", "i": int(step_i), "pose": pose_dict(pose), "to": to, **ang,
            "feat": _num(feat, 6), "n": int(debug.get("n_matches", len(pr["pc"]))),
            "pairs": pr["block"], "cell_px": _nums(cell, 3), "cell_mm": _nums(cell_mm, 2),
            "truth": truth, "right": right, "near": near,
            "v_raw": _nums(twist_raw, 7), "v_cmd": _nums(v_cmd, 7), "lam": _num(lam, 6),
            "stop": {"scored": {"rule": self.scored_rule,
                                "would_stop": bool(getattr(scored, "would_stop", False)),
                                "lam": _num(getattr(scored, "lambda_scale", None), 6),
                                "detail": _jsonable(getattr(scored, "detail", None))},
                     "recorded": ({k: bool(v.would_stop) for k, v in recorded.items()}
                                  if recorded else None),
                     "fired_now": list(fired_now), "fired": sorted(fired)},
            "end": end,
            "ms": {"match": _num(debug.get("match_ms"), 3),
                   "control": _num(debug.get("step_ms"), 3)}}
        kind = arm_kind_of(matches, debug)
        if kind == "dinobot":
            entry["dinobot"] = dinobot_block(debug, self.intrinsics, self.reflection)
        elif kind == "artvs":
            entry["artvs"] = artvs_block(matches)
        if isinstance(debug.get("method"), dict):
            # an external arm's own per-step readings (2026-10-11), rounded by the arm; absent
            # for every arm that does not fill `debug["method"]`
            entry["method"] = _jsonable(debug["method"])
        self._write(entry)
        self._n += 1
        if self._n % FLUSH_EVERY == 0:
            self._f.flush()

        landing = self._landing_labels(fired_now)
        if self.film is not None:
            refs, r_deg = self._film_refs(pose)
            self.film.step(step_i, rgb, {**debug, "n_matches": int(debug.get("n_matches", 0))},
                           refs["goal"], r_deg, refs=refs, landing=landing or None,
                           meta_extra={"kind": "step", "pose": entry["pose"],
                                       "t_mm_matched": _num(refs["size_matched"], 3),
                                       "t_mm_depth_true": _num(refs["depth_true"], 3),
                                       # the landing as the film marks it (the method's
                                       # own stop, the scored stop); every rule that
                                       # fired on this step, recorded-only ones included
                                       "landing": landing, "fired_now": list(fired_now)})
        if self.clean_writer is not None:
            self.clean_writer.step(step_i, rgb)

    # ------------------------------------------------------------------ final
    def _final(self, rgb, depth, pose: Pose, *, after_step: int,
               end_reason: str | None = None) -> None:
        """The pose after the last move (the record's ``final``), and its picture.

        ``end_reason``: the loop's end. After a break (``non_finite_velocity``,
        ``diverged_feature``) no move followed the last step - the frame says so."""
        moved = end_reason not in BREAK_REASONS
        to = self._to(pose)
        ang = orientation_split(pose.rotation, self.goal.rotation)
        self._write({"kind": "final", "after_step": int(after_step), "pose": pose_dict(pose),
                     "to": to, **ang, "frame": True, "moved": moved, "end_reason": end_reason})
        self._f.flush()
        if self.film is not None:
            refs, r_deg = self._film_refs(pose)
            what = ("after the last move (final pose)" if moved else
                    f"final pose (no move after step {int(after_step)}: {end_reason})")
            self.film.final(rgb, refs=refs, r_deg=r_deg, after_step=int(after_step), what=what,
                            meta_extra={"kind": "final", "pose": pose_dict(pose),
                                        "t_mm_matched": _num(refs["size_matched"], 3),
                                        "t_mm_depth_true": _num(refs["depth_true"], 3),
                                        "landing": [], "fired_now": [], "moved": moved})
        if self.clean_writer is not None:
            self.clean_writer.final(rgb, after_step=int(after_step))

    # -------------------------------------------------------------------- end
    def _end(self, rec: dict) -> None:
        """The outcome, from the finished record: the timeline's markers."""
        stops = rec.get("stops") or {}
        own = next(((r, stops[r]) for r in self.own_rules if r in stops), None)
        self._write({
            "kind": "end", "terminated": rec.get("terminated_reason"), "steps": rec.get("steps"),
            "stop_step": rec.get("stop_step"),
            "own_stop": {"rule": self.own_rules[0] if self.own_rules else None,
                         "step": own[1]["step"] if own else None},
            "stops": _jsonable(stops), "success": _jsonable(rec.get("success")),
            "success_matched": _jsonable(rec.get("success_matched")),
            "success_depth_true": _jsonable(rec.get("success_depth_true")),
            "outcome": rec.get("outcome"), "debug_log_s": rec.get("debug_log_s")})
        self._f.flush()


# ------------------------------------------------------------------ reading
def read_telemetry(path) -> list[dict]:
    """Every complete line of a ``telemetry.jsonl.gz``, tolerating a truncated tail (a
    killed run): the lines up to the cut are returned, a half-written last line is not."""
    import zlib

    raw = Path(path).read_bytes()
    # one gzip member, sync-flushed every FLUSH_EVERY steps: a raw decompressor returns
    # everything up to the cut instead of raising at the missing trailer
    buf = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    out: list[dict] = []
    for line in buf.split(b"\n"):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out
