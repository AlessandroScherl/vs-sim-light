#!/usr/bin/env python3
# vs-sim copy: scripts/dev_compare.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Start by start on the development set: every stored run's landing on the 72 dev flights.

    python3 scripts/dev_compare.py --root ~/genvs_ws/vssim_out --out <dir> [--with-ported] \\
        [--run "ART-VS dev=~/genvs_ws/vssim_out/dev_artvs:artvs_roi_sim_dev" ...] \\
        [--method "ART-VS dev"]

The development set (owner 2026-10-10, ``docs/dev_eval_plan_2026-10-10.md`` section 2 at the
workspace root; ``vsbench.vs.benchmark.DEV_*``): the six forward instance scenes x ``same`` /
``same_small`` / ``mine1`` / ``other_small`` x the near start + ``s_00`` + ``s_01`` = 72
flights. The baselines are not flown again: their landings on exactly these flights are read
from the stored check records (``<root>/<scene>_<live>/check.json``, one per pair, as
``scripts/chains/instance_suite.sh`` writes them):

==================================== =================== ===============================
column                               folder under root   arm
==================================== =================== ===============================
ViT-VS                               (root itself)       ``vitvs_dinov3_objmask`` (run 1)
ViT-VS + smoothing                   ``suite_r2``        ``vitvs_dinov3_objmask_ema08``
DINOBot run 1                        (root itself)       ``dinobot_objmask``
DINOBot run 2                        ``suite_r2``        ``dinobot_objmask``
ART-VS faithful                      ``suite_r2_artvs``  ``artvs_roi_sim``
ART-VS as ported, superseded         (root itself)       ``artvs_roi_scene`` (--with-ported)
==================================== =================== ===============================

plus any number of ``--run LABEL=DIR:ARM`` (the ART-VS dev cell, a new method), read the same way.

**The landing** is the protocol's scored landing as the 2026-09-28 report and
``docs/results/2026-10-09_faithful_artvs/compare.py`` read it: an arm that owns a stop of its
own (the record's ``own_stop_rule``: DINOBot's ``dinobot_err``) at that stop if it fired, else
at the step cap; every other arm at the shared scored stop (``stop_step``, ``cns_plateau``) if
it fired, else at the cap. **Two targets**: (a) the goal pose (the record's own distances) and
(b) the owner's depth-true pose (2026-09-30 / 2026-10-05: the goal pose moved straight up or
down so the camera is as far above the LIVE object's top as the goal camera was above the goal
object's top; the grasp point = the object's top): from the record's ``*_depth_true`` fields
when it has them (``--telemetry`` runs), else computed from the landing position and the two
tops the swap recorded (``perturbation.swap_instance.goal_bbox`` / ``live_bbox``) exactly as
``vsbench.vs.bench.perturb.depth_true_pose`` does; where both exist they are compared and the
largest difference is reported. The angle is to the goal orientation for both targets.
"Within 1 cm / 1 deg" and "within 3 cm / 3 deg" are the benchmark's strict gates
(``vsbench.vs.bench.scoring.abs_gate``).

**The precision tier** (owner 2026-10-16): within 3 mm & 1 deg, for the same object and the
smaller copy (computed for every case). The camera pose stays the primary score.

**At the object** (owner 2026-10-16, :func:`at_object`): the landing pose scored at the
horizontal plane of the LIVE object's top (the swap's ``live_bbox`` maximum z) against the
depth-true pose: where the approach axis meets that plane (the "grasp point", sideways error in
the plane), the standoff along the axis, the approach tilt (the angle between the landing's and
the goal's viewing axes) and the roll (the turn about it), for each camera-to-gripper offset
(``--tcp-offset NAME=x,y,z``, the gripper's TCP in the camera's optical frame, mm, its axis
parallel to the viewing axis; default :data:`DEFAULT_OFFSETS` - offset 0 and the ART-VS
release's rig; the owner's UR5 number is not known). With it a **PROPOSED** Robotiq 2F-85
tolerance set (:func:`gripper_tolerances`, owner ruling (d): the thresholds are the owner's,
set after this check) and a width-free sweep of the sideways tolerance. **Scopes**: every table
for all six kinds, without the lying bottle (owner ruling (a), out of scope for now) and for the
bottle alone; ``--round-kind`` gives a kind an infinite roll tolerance (none by default: the
dev set's bottle lies on its side).

**The approach-normal target** (owner ruling 5, 2026-10-16; ``--approach-targets FILE``): for
the two other-object cases (``mine1``, ``other_small``) the depth-true pose's tilt is redefined
as the approach normal to the object's top - the goal camera turned by the smallest rotation
taking the goal object's top normal onto the live object's (the tops: a robust plane through
the highest 10 % of each object's points, kept only when flat - ``docs/m1_spec_2026-10-16.md``
section 1 at the workspace root). The file gives that rotation per pair (computed outside this
script, from the tier-0 stills, by refine-vs ``tier0_top``); its position keeps the depth-true
pose's grasp point (where its axis meets the plane of the live object's top) and standoff, so
only the approach direction turns; where the file says ``moved: false`` (a top that is not
flat) and for the same object and the smaller copy the target is the depth-true pose. The
angle to this target is the full rotation between the landing's orientation and the target's
(checked against the record's own angle on the unturned targets). The score at the object is
given against this target too.

**Near and far** (owner ruling 4, 2026-10-16): every camera-pose table also by start group -
the near start and the two far starts separately.

**Start by start**: one method (``--method``; default the first ``--run``, else the faithful
ART-VS) against every other column on the flights both have: in how many it lands closer
(translation to the target), the median paired difference (method minus other; negative =
the method is closer), per case and overall. **The run-to-run band**: the same numbers for
DINOBot run 2 against run 1 - two identical runs - on the same flights; a lead counts only when
it is clearly bigger than that band (the judgement is the reader's: both are printed side by
side).

Writes ``<out>/rows.json`` (one row per method and flight), ``<out>/summary.json`` (every
number in the tables) and ``<out>/table.md``. numpy + PyYAML only (through ``vssim``); reads
records, never writes them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsbench.vs.bench.scoring import abs_gate  # noqa: E402
from vsbench.vs.benchmark import DEV_LIVE_NAMES, DEV_SCENES, DEV_START_IDS  # noqa: E402

#: ``(column label, folder under --root, arm id)`` - the stored baselines, in table order.
BASELINES = (
    ("ViT-VS", "", "vitvs_dinov3_objmask"),
    ("ViT-VS + smoothing", "suite_r2", "vitvs_dinov3_objmask_ema08"),
    ("DINOBot run 1", "", "dinobot_objmask"),
    ("DINOBot run 2", "suite_r2", "dinobot_objmask"),
    ("ART-VS faithful", "suite_r2_artvs", "artvs_roi_sim"),
)
PORTED = ("ART-VS as ported, superseded", "", "artvs_roi_scene")
#: the run-to-run band: two identical runs of one method on the same flights
BAND = ("DINOBot run 2", "DINOBot run 1")
#: the gates of the tables, ``(mm, deg)``: the owner's precision tier (2026-10-16, for the same
#: object and the smaller copy; computed for every case) and the two camera-pose bins
GATES = {"3mm_1deg": (3.0, 1.0), "1cm_1deg": (10.0, 1.0), "3cm_3deg": (30.0, 3.0)}
TARGETS = ("goal", "depth_true")
TARGETS_DEFAULT = TARGETS
#: the third target (ruling 5), present when ``--approach-targets`` is given
APPROACH = "approach"
APPROACH_CASES = ("mine1", "other_small")
APPROACH_SCHEMA = "vssim-approach-targets-1"
#: the start groups of ruling 4: the near start and the far starts
START_GROUPS = ("near", "far")
CASES = DEV_LIVE_NAMES

# --- the score at the object (owner 2026-10-16) ------------------------------------------------

#: The Robotiq 2F-85 (Robotiq, "2F-85 & 2F-140 for Universal Robots - Instruction Manual",
#: 2026-03-17, section 5: Fig. 5-1 / Table 5-1 / Fig. 5-7 / section 5.2.3,
#: https://blog.robotiq.com/hubfs/support-files/2F-85_2F-140_UR_PDF_2026-03-17.pdf): stroke
#: (opening between the pads) 85 mm; the flat NBR fingertip pad 38 mm long (along the approach
#: axis) and 22 mm wide; the TCP 174.0 mm from the robot flange along its axis (with coupling
#: GRP-CPL-062).
GRIPPER_2F85 = {"opening_mm": 85.0, "pad_length_mm": 38.0, "pad_width_mm": 22.0,
                "tcp_from_flange_mm": 174.0}
#: The camera-to-gripper offsets scored: the gripper's TCP in the camera's optical frame (x image
#: right, y image down, z viewing direction; mm), its approach axis parallel to the viewing
#: axis. The owner's UR5 number is NOT known: "camera axis" is offset 0; "ART-VS rig" is the
#: ART-VS release's stated tool0 -> camera offset (``catkin_ws/ibvs/src/robot/
#: sample_evaluation_poses.py``: ``CAMERA_OFFSET = [0.071, -0.001, 0.011]`` m in tool0, the
#: frames taken as aligned, as that file does) with the 2F-85's TCP 174 mm along tool0's z:
#: TCP - camera = (-71, 1, 174 - 11) mm; "turned 90 deg" is the same mount turned a quarter
#: turn about the axis (the direction of the offset matters, its length is the same).
DEFAULT_OFFSETS = (("camera axis", (0.0, 0.0, 0.0)),
                   ("ART-VS rig", (-71.0, 1.0, 163.0)),
                   ("ART-VS rig turned 90 deg", (-1.0, -71.0, 163.0)))
#: Kinds whose roll does not matter (owner 2026-10-16: round objects get an infinite roll
#: tolerance). Empty for the dev set: the only round kind, the bottle, LIES on its side there
#: (round about a horizontal axis, not about the approach axis), and the mug and the teapot have
#: handles. ``--round-kind`` adds one.
ROUND_KINDS: frozenset = frozenset()
#: the lying bottle (owner 2026-10-16: out of scope for now; every table with and without it)
BOTTLE_KINDS = ("bottle",)
SCOPES = ("all kinds", "without the lying bottle", "the lying bottle only")
#: PROPOSED gripper tolerances (owner ruling (d) 2026-10-16: the thresholds are the owner's,
#: set after this check): height = half the pad length, tilt loose, sideways and roll from the
#: opening and the object's width across the grasp (:func:`gripper_tolerances`)
PROPOSED_TILT_DEG = 5.0
PROPOSED_HEIGHT_MM = GRIPPER_2F85["pad_length_mm"] / 2.0
#: a width-free sweep of the sideways tolerance (height and tilt as proposed, roll < 5 deg)
SWEEP_SIDE_MM = (5.0, 10.0, 15.0, 20.0)
SWEEP_ROLL_DEG = 5.0


def in_scope(row: dict, scope: str) -> bool:
    bottle = row["kind"] in BOTTLE_KINDS
    return {"all kinds": True, "without the lying bottle": not bottle,
            "the lying bottle only": bottle}[scope]


def parse_offset(text: str) -> tuple[str, tuple[float, float, float]]:
    """``NAME=x,y,z`` (mm, the camera's optical frame) -> ``(name, (x, y, z))``."""
    name, sep, rest = str(text).partition("=")
    try:
        xyz = tuple(float(v) for v in rest.split(","))
    except ValueError:
        xyz = ()
    if not sep or not name.strip() or len(xyz) != 3:
        raise SystemExit(f"--tcp-offset {text!r}: expected NAME=x,y,z (mm)")
    return name.strip(), xyz


def _axis_hit(p, R, offset_m, top_z: float):
    """Where the approach axis through the TCP ``p + R o`` (direction ``R e_z``) meets the
    horizontal plane ``z = top_z``: ``(point, standoff along the axis)``; ``None`` when the
    axis does not point down."""
    t = p + R @ offset_m
    a = R[:, 2]
    if a[2] > -1e-9:
        return None
    s = (top_z - t[2]) / a[2]
    return t + s * a, float(s)


def roll_deg(R_goal, R) -> float:
    """The turn about the viewing axis (the twist of ``R_goal^T R`` about z, degrees; signed,
    in (-180, 180])."""
    rel = np.asarray(R_goal, np.float64).T @ np.asarray(R, np.float64)
    return float(np.degrees(np.arctan2(rel[1, 0] - rel[0, 1], rel[0, 0] + rel[1, 1])))


def tilt_deg(R_goal, R) -> float:
    """The angle between the two viewing (= approach) axes, degrees."""
    c = float(np.dot(np.asarray(R_goal, np.float64)[:, 2], np.asarray(R, np.float64)[:, 2]))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotation_angle_deg(Ra, Rb) -> float:
    """The full rotation between two orientations (the geodesic angle of ``Ra^T Rb``)."""
    rel = np.asarray(Ra, np.float64).T @ np.asarray(Rb, np.float64)
    c = (float(np.trace(rel)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def load_approach_targets(path: Path) -> dict:
    """``{pair: {"R_target": 3x3, "moved": bool, ...}}`` from an approach-targets file (schema
    :data:`APPROACH_SCHEMA`); refuses another schema or a malformed rotation."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != APPROACH_SCHEMA:
        raise SystemExit(f"--approach-targets {path}: schema {doc.get('schema')!r}, "
                         f"expected {APPROACH_SCHEMA!r}")
    out = {}
    for pair, t in (doc.get("pairs") or {}).items():
        R = np.asarray(t["R_target"], np.float64)
        if R.shape != (3, 3) or not np.allclose(R.T @ R, np.eye(3), atol=1e-9):
            raise SystemExit(f"--approach-targets {path}: {pair}'s R_target is not a rotation")
        out[pair] = {**t, "R_target": R, "moved": bool(t["moved"])}
    return out


def approach_target(dt_position, goal_rotation, top_z: float, target: dict | None):
    """The approach-normal target pose ``(position, rotation, moved)``: the depth-true pose
    turned about its grasp point (where its viewing axis meets the plane ``z = top_z``) to the
    file's rotation, at the same standoff; the depth-true pose itself when ``target`` is
    ``None`` or not moved (or its axis misses the plane)."""
    pg = np.asarray(dt_position, np.float64)
    Rg = np.asarray(goal_rotation, np.float64)
    if target is None or not target["moved"]:
        return pg, Rg, False
    hit = _axis_hit(pg, Rg, np.zeros(3), top_z)
    if hit is None:
        return pg, Rg, False
    q, standoff = hit
    Rt = np.asarray(target["R_target"], np.float64)
    return q - standoff * Rt[:, 2], Rt, True


def at_object(position, rotation, dt_position, goal_rotation, top_z: float,
              offsets) -> dict:
    """The landing scored at the LIVE object's top (owner 2026-10-16), against the depth-true
    pose: per offset, where the approach axis meets the plane of the object's top (the "grasp
    point"; ``grasp_mm`` sideways in that plane, ``grasp_dxy_mm`` world x, y) and the standoff
    along the axis (``standoff_mm`` at the landing, ``standoff_err_mm`` = landing minus the
    depth-true pose's: positive = too far); and, offset-free, the approach tilt, the roll and
    the camera's own sideways / vertical error and the pivot term ``standoff x tan(tilt)``
    (the depth-true camera's standoff; ``None`` past 89 deg of tilt)."""
    p, R = np.asarray(position, np.float64), np.asarray(rotation, np.float64)
    pg, Rg = np.asarray(dt_position, np.float64), np.asarray(goal_rotation, np.float64)
    tl = tilt_deg(Rg, R)
    per = {}
    for name, off in offsets:
        o = np.asarray(off, np.float64) / 1000.0
        a, b = _axis_hit(p, R, o, top_z), _axis_hit(pg, Rg, o, top_z)
        if a is None or b is None:
            per[name] = None
            continue
        d = 1000.0 * (a[0] - b[0])
        per[name] = {"grasp_mm": float(math.hypot(d[0], d[1])),
                     "grasp_dxy_mm": [float(d[0]), float(d[1])],
                     "standoff_mm": 1000.0 * a[1], "standoff_err_mm": 1000.0 * (a[1] - b[1])}
    cam0 = _axis_hit(pg, Rg, np.zeros(3), top_z)
    return {"tilt_deg": tl, "roll_deg": roll_deg(Rg, R),
            "cam_side_mm": 1000.0 * float(math.hypot(p[0] - pg[0], p[1] - pg[1])),
            "cam_height_mm": 1000.0 * float(p[2] - pg[2]),
            "pivot_mm": (None if cam0 is None or tl >= 89.0
                         else 1000.0 * cam0[1] * math.tan(math.radians(tl))),
            "offsets": per}


def gripper_tolerances(live_bbox, kind: str, round_kinds=ROUND_KINDS,
                       g=GRIPPER_2F85) -> dict:
    """PROPOSED 2F-85 tolerances for one live object from its world bounding box (the swap's
    ``live_bbox``): the width across the grasp ``w`` = the smaller horizontal extent (every
    dev object lies with its long axis along world x), ``L`` the larger; sideways
    ``(opening - w) / 2``; roll the largest turn with ``w cos(psi) + L sin(psi) <= opening``
    (the object's extent across the closing direction still fits; infinite for a round kind);
    height half the pad length; tilt :data:`PROPOSED_TILT_DEG`. ``side_mm`` / ``roll_deg``
    ``None``: no clearance, the whole object does not fit between the pads."""
    lo, hi = np.asarray(live_bbox[0], np.float64), np.asarray(live_bbox[1], np.float64)
    ext = 1000.0 * (hi - lo)
    w, L = float(min(ext[0], ext[1])), float(max(ext[0], ext[1]))
    op = g["opening_mm"]
    side = (op - w) / 2.0 if w < op else None
    if kind in round_kinds:
        roll = math.inf
    elif w >= op:
        roll = None
    elif math.hypot(w, L) <= op:
        roll = 90.0
    else:
        phi = math.atan2(L, w)
        roll = math.degrees(phi - math.acos(op / math.hypot(w, L)))
    return {"width_mm": w, "length_mm": L, "side_mm": side, "roll_deg": roll,
            "height_mm": PROPOSED_HEIGHT_MM, "tilt_deg": PROPOSED_TILT_DEG}


def gripper_pass(obj: dict, tol: dict, name: str):
    """Whether the landing (offset ``name``) is inside the PROPOSED set; ``None`` when the
    object has no clearance (or the axis misses the plane)."""
    o = obj["offsets"].get(name)
    if o is None or tol["side_mm"] is None or tol["roll_deg"] is None:
        return None
    return bool(o["grasp_mm"] < tol["side_mm"] and abs(o["standoff_err_mm"]) < tol["height_mm"]
                and obj["tilt_deg"] < tol["tilt_deg"] and abs(obj["roll_deg"]) < tol["roll_deg"])


def sweep_pass(obj: dict, name: str, side_mm: float, round_kind: bool) -> bool | None:
    o = obj["offsets"].get(name)
    if o is None:
        return None
    return bool(o["grasp_mm"] < side_mm and abs(o["standoff_err_mm"]) < PROPOSED_HEIGHT_MM
                and obj["tilt_deg"] < PROPOSED_TILT_DEG
                and (round_kind or abs(obj["roll_deg"]) < SWEEP_ROLL_DEG))


def parse_run(text: str) -> tuple[str, Path, str]:
    """``LABEL=DIR:ARM`` -> ``(label, dir, arm)``; the arm is after the LAST colon."""
    label, sep, rest = str(text).partition("=")
    folder, sep2, arm = rest.rpartition(":")
    if not sep or not sep2 or not label.strip() or not folder.strip() or not arm.strip():
        raise SystemExit(f"--run {text!r}: expected LABEL=DIR:ARM")
    return label.strip(), Path(folder.strip()).expanduser(), arm.strip()


def landing(e: dict) -> dict:
    """The protocol's scored landing of one flight record: ``{how, step, mm, deg, position,
    rotation, mm_dt_record}`` (``rotation``: the camera's optical axes as columns, world frame,
    as :class:`vsbench.vs.scene.Pose`). ``how``: ``own`` (the arm's own stop), ``stop`` (the shared
    scored stop) or ``cap``."""
    stops = e.get("stops") or {}
    own = e.get("own_stop_rule")
    final = {"how": "cap", "step": e.get("steps"), "mm": e["final_mm"], "deg": e["final_deg"],
             "position": (e.get("final_pose") or {}).get("position"),
             "rotation": (e.get("final_pose") or {}).get("rotation"),
             "mm_dt_record": e.get("final_translation_mm_depth_true")}
    if own:
        s = stops.get(own)
        if s and s.get("step") is not None:
            return {"how": "own", "step": s["step"], "mm": s["translation_mm"],
                    "deg": s["rotation_deg"], "position": s.get("position"),
                    "rotation": s.get("rotation"),
                    "mm_dt_record": s.get("translation_mm_depth_true")}
        return final
    if e.get("stop_step") is not None:
        # the scored stop's own entry carries its position (the record's `stops`, the same
        # numbers as `stop_mm` / `stop_deg`)
        hit = [s for s in stops.values() if s.get("step") == e["stop_step"]
               and s.get("translation_mm") == e["stop_mm"]]
        return {"how": "stop", "step": e["stop_step"], "mm": e["stop_mm"], "deg": e["stop_deg"],
                "position": hit[0].get("position") if hit else None,
                "rotation": hit[0].get("rotation") if hit else None,
                "mm_dt_record": e.get("stop_translation_mm_depth_true")}
    return final


def depth_true_position(e: dict, goal_cam_pos) -> list[float] | None:
    """The depth-true pose's position from the swap's two tops: ``p_goal + (0, 0, live top -
    goal top)`` (``vsbench.vs.bench.perturb.depth_true_pose``; the tops are the bounding boxes'
    maximum z the swap recorded). ``None`` when the record has no swap."""
    sw = (e.get("perturbation") or {}).get("swap_instance")
    if not isinstance(sw, dict) or sw.get("goal_bbox") is None or sw.get("live_bbox") is None:
        return None
    gb, lb = sw["goal_bbox"], sw["live_bbox"]
    p = [float(v) for v in goal_cam_pos]
    p[2] += float(lb[1][2]) - float(gb[1][2])
    return p


def _dist_mm(a, b) -> float:
    return 1000.0 * math.dist([float(v) for v in a], [float(v) for v in b])


def _gates(mm_goal: float, mm_dt: float | None, deg: float) -> dict:
    """``within_<gate>_<target>`` for both gates and both targets (``None``: no distance)."""
    out = {}
    for g, (t_mm, r_deg) in GATES.items():
        out[f"within_{g}_goal"] = abs_gate(mm_goal, deg, t_mm, r_deg)
        out[f"within_{g}_depth_true"] = (None if mm_dt is None
                                         else abs_gate(mm_dt, deg, t_mm, r_deg))
    return out


def read_flights(check_json: Path, arm: str) -> dict[str, dict]:
    """``{start_id: record}`` of one arm in one pair's check.json (``{}`` when absent), each
    with the scene's goal camera position under ``_goal_cam_pos``."""
    if not check_json.is_file():
        return {}
    doc = json.loads(check_json.read_text())
    out = {}
    for sc in (doc.get("scenes") or {}).values():
        a = (sc.get("arms") or {}).get(arm)
        if not a or a.get("build_error") or a.get("error"):
            continue
        far = a.get("far") or []
        if isinstance(far, dict):
            far = [far]
        for e in ([a["near"]] if a.get("near") else []) + list(far):
            out[e["start_id"]] = {**e, "_goal_cam_pos": sc["goal"]["cam_pos"],
                                  "_goal_cam_R": sc["goal"].get("cam_R_opt"),
                                  "_label": a.get("label")}
    return out


def collect(sources, offsets=DEFAULT_OFFSETS,
            round_kinds=ROUND_KINDS, approach: dict | None = None) -> tuple[list[dict], dict]:
    """One row per (method, dev flight) for ``sources = [(label, folder, arm)]``; each row also
    scored at the object (:func:`at_object`, every offset) with the PROPOSED gripper set, and
    - with ``approach`` (:func:`load_approach_targets`) - against the approach-normal target
    (:func:`_approach_fields`)."""
    rows, check = [], {"dt_compared": 0, "dt_max_abs_diff_mm": 0.0, "dt_missing": 0,
                       "deg_compared": 0, "deg_max_abs_diff": 0.0}
    for label, folder, arm in sources:
        for scene in DEV_SCENES:
            for live in DEV_LIVE_NAMES:
                pair = f"{scene}_{live}"
                flights = read_flights(Path(folder) / pair / "check.json", arm)
                for sid in DEV_START_IDS:
                    e = flights.get(sid)
                    if e is None:
                        continue
                    ld = landing(e)
                    dt_pos = depth_true_position(e, e["_goal_cam_pos"])
                    computed = (_dist_mm(ld["position"], dt_pos)
                                if dt_pos is not None and ld["position"] is not None else None)
                    if ld["mm_dt_record"] is not None:
                        mm_dt, src = float(ld["mm_dt_record"]), "record"
                        if computed is not None:
                            check["dt_compared"] += 1
                            check["dt_max_abs_diff_mm"] = max(check["dt_max_abs_diff_mm"],
                                                              abs(computed - mm_dt))
                    elif computed is not None:
                        mm_dt, src = computed, "computed"
                    else:
                        mm_dt, src = None, None
                        check["dt_missing"] += 1
                    rows.append({
                        "method": label, "folder": str(folder), "arm": arm,
                        "label": e.get("_label"), "pair": pair, "scene": scene,
                        "kind": scene.removeprefix("inst_"), "case": live, "start_id": sid,
                        "start": "near" if sid == DEV_START_IDS[0] else sid,
                        "read_at": ld["how"], "read_step": ld["step"],
                        "mm_goal": float(ld["mm"]), "deg": float(ld["deg"]),
                        "mm_depth_true": mm_dt, "depth_true_from": src,
                        "position": ld["position"], "depth_true_position": dt_pos,
                        "start_mm": e.get("start_mm"), "start_deg": e.get("start_deg"),
                        **_gates(float(ld["mm"]), mm_dt, float(ld["deg"])),
                        **_object_fields(e, ld, dt_pos, scene.removeprefix("inst_"), offsets,
                                         round_kinds),
                    })
                    if approach is not None:
                        rows[-1].update(_approach_fields(e, ld, dt_pos, live, pair, approach,
                                                         offsets, check))
    return rows, check


def _approach_fields(e: dict, ld: dict, dt_pos, live: str, pair: str, approach: dict,
                     offsets, check: dict) -> dict:
    """The row against the approach-normal target: ``mm_approach``, ``deg_approach`` (the full
    rotation to the target), its gates, ``approach_moved`` and ``object_approach``
    (:func:`at_object` against it). ``check`` collects the largest gap between the record's
    own angle and the angle recomputed to the goal orientation (the unturned target)."""
    sw = (e.get("perturbation") or {}).get("swap_instance") or {}
    Rg = e.get("_goal_cam_R")
    none = {"mm_approach": None, "deg_approach": None, "approach_moved": None,
            "object_approach": None, **{f"within_{g}_{APPROACH}": None for g in GATES}}
    if (dt_pos is None or ld.get("position") is None or ld.get("rotation") is None
            or Rg is None or sw.get("live_bbox") is None):
        return none
    top = float(sw["live_bbox"][1][2])
    tgt = approach.get(pair) if live in APPROACH_CASES else None
    p_t, R_t, moved = approach_target(dt_pos, Rg, top, tgt)
    R_land = np.asarray(ld["rotation"], np.float64)
    deg_goal = rotation_angle_deg(Rg, R_land)
    check["deg_compared"] += 1
    check["deg_max_abs_diff"] = max(check["deg_max_abs_diff"], abs(deg_goal - float(ld["deg"])))
    mm = _dist_mm(ld["position"], p_t)
    deg = rotation_angle_deg(R_t, R_land)
    return {"mm_approach": mm, "deg_approach": deg, "approach_moved": moved,
            "approach_tilt_deg": tilt_deg(Rg, R_t),
            **{f"within_{g}_{APPROACH}": abs_gate(mm, deg, t_mm, r_deg)
               for g, (t_mm, r_deg) in GATES.items()},
            "object_approach": {"live_top_z": top,
                                **at_object(ld["position"], R_land, p_t, R_t, top, offsets)}}


def _object_fields(e: dict, ld: dict, dt_pos, kind: str, offsets, round_kinds) -> dict:
    """The row's ``object`` score, ``gripper_tolerances`` and ``gripper_pass`` (per offset)
    and ``sweep_pass`` (per offset and sideways tolerance); all ``None`` without a swap block,
    a landing rotation or a goal rotation."""
    sw = (e.get("perturbation") or {}).get("swap_instance") or {}
    if (dt_pos is None or ld.get("rotation") is None or ld.get("position") is None
            or e.get("_goal_cam_R") is None or sw.get("live_bbox") is None):
        return {"object": None, "gripper_tolerances": None, "gripper_pass": None,
                "sweep_pass": None}
    top = float(sw["live_bbox"][1][2])
    obj = at_object(ld["position"], ld["rotation"], dt_pos, e["_goal_cam_R"], top, offsets)
    tol = gripper_tolerances(sw["live_bbox"], kind, round_kinds)
    names = [n for n, _ in offsets]
    return {"object": {"live_top_z": top, **obj}, "gripper_tolerances": tol,
            "gripper_pass": {n: gripper_pass(obj, tol, n) for n in names},
            "sweep_pass": {n: {f"{x:g}": sweep_pass(obj, n, x, kind in round_kinds)
                               for x in SWEEP_SIDE_MM} for n in names}}


def _mm(r: dict, target: str):
    return {"goal": r["mm_goal"], "depth_true": r["mm_depth_true"]}.get(target) \
        if target != APPROACH else r.get("mm_approach")


def _deg(r: dict, target: str):
    return r.get("deg_approach") if target == APPROACH else r["deg"]


def _start_ok(r: dict, start: str | None) -> bool:
    return start is None or (r["start"] == "near") == (start == "near")


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def _q(xs, q: float):
    xs = [x for x in xs if x is not None]
    return round(float(np.quantile(xs, q)), 1) if xs else None


def counts(rows, method: str, case: str | None, target: str, scope: str = "all kinds",
           start: str | None = None) -> dict:
    rs = [r for r in rows if r["method"] == method and (case is None or r["case"] == case)
          and in_scope(r, scope) and _start_ok(r, start) and _mm(r, target) is not None]
    return {"n": len(rs),
            **{g: sum(bool(r[f"within_{g}_{target}"]) for r in rs) for g in GATES},
            "median_mm": _median([_mm(r, target) for r in rs]),
            "median_deg": _median([_deg(r, target) for r in rs])}


def paired(rows, a: str, b: str, case: str | None, target: str,
           scope: str = "all kinds") -> dict:
    """``a`` against ``b`` flight by flight on the flights both have."""
    def pick(m):
        return {(r["pair"], r["start_id"]): r for r in rows
                if r["method"] == m and (case is None or r["case"] == case)
                and in_scope(r, scope)}
    ra, rb = pick(a), pick(b)
    both = [k for k in ra if k in rb and _mm(ra[k], target) is not None
            and _mm(rb[k], target) is not None]
    d = [_mm(ra[k], target) - _mm(rb[k], target) for k in both]
    flips = {g: sum(bool(ra[k][f"within_{g}_{target}"]) != bool(rb[k][f"within_{g}_{target}"])
                    for k in both) for g in GATES}
    return {"n": len(both), "closer": sum(x < 0 for x in d), "farther": sum(x > 0 for x in d),
            "median_diff_mm": _median(d), "median_abs_diff_mm": _median([abs(x) for x in d]),
            "flips": flips}


def object_counts(rows, method: str, case: str | None, scope: str, names,
                  key: str = "object") -> dict:
    """The score at the object over one cell: medians (and the 90th percentile of the grasp
    point), the PROPOSED gripper set (graspable = the object fits between the pads) and the
    width-free sweep, per offset. ``key``: ``object`` (against the depth-true pose) or
    ``object_approach`` (against the approach-normal target; the gripper set and the sweep
    are the depth-true pose's there and are not read)."""
    rs = [r for r in rows if r["method"] == method and (case is None or r["case"] == case)
          and in_scope(r, scope) and r.get(key) is not None]
    o = [r[key] for r in rs]
    per = {}
    for n in names:
        hit = [x["offsets"][n] for x in o if x["offsets"].get(n) is not None]
        gp = [r["gripper_pass"][n] for r in rs]
        per[n] = {"median_grasp_mm": _median([h["grasp_mm"] for h in hit]),
                  "p90_grasp_mm": _q([h["grasp_mm"] for h in hit], 0.9),
                  "median_abs_standoff_err_mm": _median([abs(h["standoff_err_mm"]) for h in hit]),
                  "graspable": sum(v is not None for v in gp),
                  "gripper_pass": sum(v is True for v in gp),
                  "sweep": {f"{x:g}": sum(r["sweep_pass"][n][f"{x:g}"] is True for r in rs)
                            for x in SWEEP_SIDE_MM}}
    return {"n": len(rs),
            "median_mm_depth_true": _median([r["mm_depth_true"] for r in rs]),
            "median_cam_side_mm": _median([x["cam_side_mm"] for x in o]),
            "median_abs_cam_height_mm": _median([abs(x["cam_height_mm"]) for x in o]),
            "median_pivot_mm": _median([x["pivot_mm"] for x in o]),
            "median_tilt_deg": _median([x["tilt_deg"] for x in o]),
            "median_abs_roll_deg": _median([abs(x["roll_deg"]) for x in o]),
            "offsets": per}


def targets_of(rows) -> tuple[str, ...]:
    """The targets scored in ``rows``: the goal pose, the depth-true pose and - when the rows
    carry it - the approach-normal target."""
    return TARGETS + ((APPROACH,) if any("mm_approach" in r for r in rows) else ())


def summarise(rows, methods, method: str, names=tuple(n for n, _ in DEFAULT_OFFSETS)) -> dict:
    groups = [*CASES, None]
    TARGETS = targets_of(rows)  # noqa: N806 - the module's tuple, plus the approach target
    out = {"targets": list(TARGETS),
           "counts": {t: {m: {c or "all": counts(rows, m, c, t) for c in groups}
                          for m in methods} for t in TARGETS},
           "method": method,
           "paired": {t: {o: {c or "all": paired(rows, method, o, c, t) for c in groups}
                          for o in methods if o != method} for t in TARGETS},
           "band": None, "offsets": list(names),
           "scopes": {sc: {
               "counts": {t: {m: {c or "all": counts(rows, m, c, t, sc) for c in groups}
                              for m in methods} for t in TARGETS},
               "paired": {t: {o: {c or "all": paired(rows, method, o, c, t, sc) for c in groups}
                              for o in methods if o != method} for t in TARGETS},
               "object": {m: {c or "all": object_counts(rows, m, c, sc, names) for c in groups}
                          for m in methods},
               "by_start": {t: {m: {st: {c or "all": counts(rows, m, c, t, sc, st)
                                          for c in groups} for st in START_GROUPS}
                                for m in methods} for t in TARGETS},
               **({"object_approach": {m: {c or "all": object_counts(rows, m, c, sc, names,
                                                                      "object_approach")
                                            for c in groups} for m in methods}}
                  if APPROACH in TARGETS else {})} for sc in SCOPES}}
    if all(any(r["method"] == b for r in rows) for b in BAND):
        out["band"] = {"a": BAND[0], "b": BAND[1],
                       **{t: {c or "all": paired(rows, BAND[0], BAND[1], c, t) for c in groups}
                          for t in TARGETS}}
    return out


def _fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _grid(head: str, groups, rows_) -> list[str]:
    return (["| " + head + " | " + " | ".join(groups) + " |",
             "| --- | " + " | ".join("---" for _ in groups) + " |"]
            + [f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows_] + [""])


TARGET_NAMES = {"goal": "the goal pose", "depth_true": "the depth-true pose",
                APPROACH: "the approach-normal target (ruling 5)"}


def table(summary: dict, methods, meta: dict) -> str:
    groups = [*CASES, "all"]
    TARGETS = tuple(summary.get("targets") or TARGETS_DEFAULT)  # noqa: N806
    lines = ["# The development set, start by start (`scripts/dev_compare.py`)", "",
             f"Flights: {meta['flights']} dev flights (6 forward scenes x {len(CASES)} live "
             f"objects x {len(DEV_START_IDS)} starts). Reading: an arm's own stop if it owns one "
             "and it fired (DINOBot), else the shared scored stop if it fired, else the step "
             "cap. Gates strict (`<`), as the benchmark's. Depth-true: "
             f"{meta['depth_true']}. Every table is given for all six kinds, without the lying "
             "bottle (owner 2026-10-16: out of scope for now) and for the bottle alone.", ""]
    for sc in SCOPES:
        S = summary["scopes"][sc]
        for t in TARGETS:
            name = TARGET_NAMES[t]
            lines += [f"## Camera pose against {name} - {sc}", "",
                      "Each cell: within 3 mm & 1 deg / 1 cm & 1 deg / 3 cm & 3 deg (of n), "
                      "median mm. The 3 mm tier is the owner's precision tier for the same "
                      "object and the smaller copy (2026-10-16); it is computed for every case.",
                      ""]
            lines += _grid("method", groups, [
                (m, ["-" if not (c := S["counts"][t][m][g])["n"] else
                     f"{c['3mm_1deg']} / {c['1cm_1deg']} / {c['3cm_3deg']} of {c['n']}, "
                     f"{_fmt(c['median_mm'])} mm" for g in groups]) for m in methods])
    meth = summary["method"]
    for sc in SCOPES:
        S = summary["scopes"][sc]
        for t in TARGETS:
            name = TARGET_NAMES[t]
            lines += [f"## Start by start: {meth} against each, to {name} - {sc}", "",
                      "Each cell: closer in k of n flights, median paired difference (method "
                      "minus other; negative = closer), median absolute difference.", ""]
            rows_ = [(o, S["paired"][t][o]) for o in S["paired"][t]]
            if sc == "all kinds" and summary["band"] is not None:
                rows_.append((f"_band: {summary['band']['a']} against {summary['band']['b']}_",
                              summary["band"][t]))
            lines += _grid("against", groups, [
                (o, ["-" if not (p := per[g])["n"] else
                     f"{p['closer']} of {p['n']}, {_fmt(p['median_diff_mm'])} mm "
                     f"(abs {_fmt(p['median_abs_diff_mm'])})" for g in groups])
                for o, per in rows_])
    for sc in ("without the lying bottle", "all kinds"):
        S = summary["scopes"][sc]
        for t in TARGETS:
            for st in START_GROUPS:
                lines += [f"## {st.capitalize()} starts against {TARGET_NAMES[t]} - {sc}", "",
                          "Each cell: within 3 mm & 1 deg / 1 cm & 1 deg / 3 cm & 3 deg (of n), "
                          "median mm / median deg (owner ruling 4: near and far separately).",
                          ""]
                lines += _grid("method", groups, [
                    (m, ["-" if not (c := S["by_start"][t][m][st][g])["n"] else
                         f"{c['3mm_1deg']} / {c['1cm_1deg']} / {c['3cm_3deg']} of {c['n']}, "
                         f"{_fmt(c['median_mm'])} mm / {_fmt(c['median_deg'], 2)} deg"
                         for g in groups]) for m in methods])
    if summary["band"] is not None:
        b = summary["band"]
        lines += ["## The run-to-run band (two identical DINOBot runs, all kinds)", "",
                  "Flights that changed side of a gate between the two runs, of n:", "",
                  "| target | gate | " + " | ".join(groups) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in groups) + " |"]
        for t in TARGETS:
            for g in GATES:
                lines.append(f"| {t} | {g} | " + " | ".join(
                    f"{b[t][c]['flips'][g]} of {b[t][c]['n']}" for c in groups) + " |")
        lines.append("")
    names = summary["offsets"]
    lines += ["## At the object", "",
              "Scored at the plane of the LIVE object's top against the depth-true pose "
              "(owner 2026-10-16). Grasp point: where the approach axis meets that plane, "
              "landing vs depth-true pose, sideways in the plane. Offsets (the gripper's TCP "
              "in the camera's optical frame, mm, axis parallel to the viewing axis): "
              + "; ".join(f"{n} = {tuple(v)}" for n, v in meta["offsets"]) + ". Tilt = angle "
              "between the landing's approach axis and the goal's; roll = the turn about it; "
              "standoff = distance along the axis from the TCP to the plane (error = landing "
              "minus depth-true; it depends on the offset only through the tilt). Pivot term = "
              "the depth-true camera standoff x tan(tilt): the sideways camera error a pure "
              "pivot about the object's top point would make.", ""]
    for sc in SCOPES:
        osum = summary["scopes"][sc]["object"]
        n0 = names[0]
        lines += [f"### The camera's miss against the grasp point - {sc}", "",
                  f"Each cell (medians): camera pose error mm | camera sideways mm | pivot term "
                  f"mm | grasp point mm ({n0}) | tilt deg | roll deg.", ""]
        lines += _grid("method", groups, [
            (m, ["-" if not (c := osum[m][g])["n"] else
                 f"{_fmt(c['median_mm_depth_true'])} \\| {_fmt(c['median_cam_side_mm'])} \\| "
                 f"{_fmt(c['median_pivot_mm'])} \\| "
                 f"{_fmt(c['offsets'][n0]['median_grasp_mm'])} \\| "
                 f"{_fmt(c['median_tilt_deg'], 2)} \\| {_fmt(c['median_abs_roll_deg'], 2)}"
                 for g in groups]) for m in methods])
        lines += [f"### Grasp point per offset - {sc}", "",
                  "Each cell: median grasp-point error mm (90th percentile) \\| median "
                  "|standoff error| mm.", ""]
        lines += _grid("method / offset", groups, [
            (f"{m} / {n}", ["-" if not (c := osum[m][g])["n"] else
                            f"{_fmt(c['offsets'][n]['median_grasp_mm'])} "
                            f"({_fmt(c['offsets'][n]['p90_grasp_mm'])}) \\| "
                            f"{_fmt(c['offsets'][n]['median_abs_standoff_err_mm'])}"
                            for g in groups]) for m in methods for n in names])
        lines += [f"### PROPOSED gripper set (2F-85) - {sc}", "",
                  "PROPOSED, not fixed (owner ruling (d): the thresholds are the owner's). "
                  f"Pass: grasp point < (85 - w) / 2 mm, |standoff error| < "
                  f"{PROPOSED_HEIGHT_MM:g} mm (half the 38 mm pad), tilt < "
                  f"{PROPOSED_TILT_DEG:g} deg, |roll| < the largest turn at which the object "
                  "still fits between the pads; w = the live object's smaller horizontal "
                  "extent (its bounding box). Each cell: pass of graspable (graspable = the "
                  "whole object fits between the pads; of n).", ""]
        lines += _grid("method / offset", groups, [
            (f"{m} / {n}", ["-" if not (c := osum[m][g])["n"] else
                            f"{c['offsets'][n]['gripper_pass']} of "
                            f"{c['offsets'][n]['graspable']} (of {c['n']})" for g in groups])
            for m in methods for n in names])
        lines += [f"### Width-free sweep - {sc}", "",
                  "PROPOSED. Pass: grasp point < X mm, |standoff error| < "
                  f"{PROPOSED_HEIGHT_MM:g} mm, tilt < {PROPOSED_TILT_DEG:g} deg, |roll| < "
                  f"{SWEEP_ROLL_DEG:g} deg (free for a round kind). Each cell: pass for X = "
                  + " / ".join(f"{x:g}" for x in SWEEP_SIDE_MM) + " mm, of n.", ""]
        lines += _grid("method / offset", groups, [
            (f"{m} / {n}", ["-" if not (c := osum[m][g])["n"] else
                            " / ".join(str(c["offsets"][n]["sweep"][f"{x:g}"])
                                       for x in SWEEP_SIDE_MM) + f" of {c['n']}"
                            for g in groups]) for m in methods for n in names])
    if APPROACH in TARGETS:
        lines += ["## At the object, against the approach-normal target (ruling 5)", "",
                  "As above, but the target is the approach-normal one: for the other-object "
                  "cases where it moved, its axis is tilted to the live top's normal (the grasp "
                  "point and standoff of the depth-true pose kept); elsewhere it is the "
                  "depth-true pose. Tilt and roll are to the target's axis.", ""]
        for sc in SCOPES:
            osum = summary["scopes"][sc]["object_approach"]
            lines += [f"### Against the approach-normal target - {sc}", "",
                      "Each cell (medians): camera sideways mm | grasp point mm per offset ("
                      + ", ".join(names) + ") | tilt deg | roll deg.", ""]
            lines += _grid("method", groups, [
                (m, ["-" if not (c := osum[m][g])["n"] else
                     f"{_fmt(c['median_cam_side_mm'])} \\| "
                     + " / ".join(_fmt(c["offsets"][n]["median_grasp_mm"]) for n in names)
                     + f" \\| {_fmt(c['median_tilt_deg'], 2)} \\| "
                     f"{_fmt(c['median_abs_roll_deg'], 2)}" for g in groups]) for m in methods])
        moved = meta.get("approach_targets") or {}
        if moved:
            lines += ["### The approach-normal targets per pair", "",
                      "| pair | moved | target tilt from the goal deg |", "| --- | --- | --- |"]
            for pair, t in sorted(moved.items()):
                lines.append(f"| {pair} | {'yes' if t['moved'] else 'kept'} | "
                             f"{_fmt(t.get('target_tilt_deg'), 2)} |")
            lines.append("")
    tols = meta.get("tolerances") or {}
    if tols:
        lines += ["### PROPOSED tolerances per live object (from the 2F-85 and the bounding box)",
                  "", "| pair | w mm | L mm | sideways mm | roll deg | height mm | tilt deg |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for pair, t in tols.items():
            lines.append(f"| {pair} | {t['width_mm']:.0f} | {t['length_mm']:.0f} | "
                         f"{_fmt(t['side_mm'])} | {_fmt(t['roll_deg'])} | "
                         f"{t['height_mm']:g} | {t['tilt_deg']:g} |")
        lines.append("")
    lines += ["## Sources", ""] + [f"- {m}: `{f}` arm `{a}`" for m, f, a in meta["sources"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True,
                    help="the folder holding the stored runs (the station's ~/genvs_ws/vssim_out)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=DIR:ARM",
                    help="one more run read the same way (repeatable)")
    ap.add_argument("--method", default=None,
                    help="the column compared start by start (default: the first --run, else "
                         "'ART-VS faithful')")
    ap.add_argument("--with-ported", action="store_true",
                    help="add the superseded 'as ported' ART-VS column (artvs_roi_scene)")
    ap.add_argument("--tcp-offset", action="append", default=[], metavar="NAME=x,y,z",
                    help="a camera-to-gripper offset to score at the object: the TCP in the "
                         "camera's optical frame, mm (repeatable; default: the three of "
                         "DEFAULT_OFFSETS)")
    ap.add_argument("--round-kind", action="append", default=[], metavar="KIND",
                    help="a kind whose roll does not matter (repeatable; default none)")
    ap.add_argument("--approach-targets", type=Path, default=None, metavar="FILE",
                    help="the approach-normal targets per pair (ruling 5; schema "
                         f"{APPROACH_SCHEMA}): adds the third target")
    args = ap.parse_args(argv)
    approach = load_approach_targets(args.approach_targets) if args.approach_targets else None
    offsets = tuple(parse_offset(t) for t in args.tcp_offset) or DEFAULT_OFFSETS
    if len({n for n, _ in offsets}) != len(offsets):
        raise SystemExit("two --tcp-offset share a name")
    round_kinds = frozenset(ROUND_KINDS | set(args.round_kind))
    root = args.root.expanduser()
    sources = [(lab, root / folder, arm) for lab, folder, arm in BASELINES]
    if args.with_ported:
        sources.append((PORTED[0], root / PORTED[1], PORTED[2]))
    extra = [parse_run(r) for r in args.run]
    sources += extra
    labels = [s[0] for s in sources]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"two columns share a label: {labels}")
    method = args.method or (extra[0][0] if extra else "ART-VS faithful")
    if method not in labels:
        raise SystemExit(f"--method {method!r} is not a column; have {labels}")
    rows, check = collect(sources, offsets, round_kinds, approach)
    methods = [m for m in labels if any(r["method"] == m for r in rows)]
    if method not in methods:
        raise SystemExit(f"--method {method!r} has no dev flight under its folder")
    summary = summarise(rows, methods, method, tuple(n for n, _ in offsets))
    n_dt = sum(r["depth_true_from"] == "computed" for r in rows)
    meta = {"flights": len(DEV_SCENES) * len(DEV_LIVE_NAMES) * len(DEV_START_IDS),
            "sources": [(m, str(f), a) for m, f, a in sources],
            "rows_per_method": {m: sum(r["method"] == m for r in rows) for m in labels},
            "depth_true": (f"{sum(r['depth_true_from'] == 'record' for r in rows)} rows from "
                           f"the record, {n_dt} computed from the swap's two tops; where both "
                           f"exist ({check['dt_compared']}) they differ by at most "
                           f"{check['dt_max_abs_diff_mm']:.2e} mm; {check['dt_missing']} "
                           "without"),
            "offsets": [(n, list(v)) for n, v in offsets], "round_kinds": sorted(round_kinds),
            "gripper": GRIPPER_2F85,
            "tolerances": {r["pair"]: r["gripper_tolerances"] for r in rows
                           if r["method"] == method and r["gripper_tolerances"] is not None},
            "approach_targets_file": (str(args.approach_targets) if args.approach_targets
                                      else None),
            "approach_targets": ({p: {"moved": t["moved"],
                                      "target_tilt_deg": t.get("target_tilt_deg")}
                                  for p, t in approach.items()} if approach else None),
            **check}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    (args.out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=1))
    md = table(summary, methods, meta)
    (args.out / "table.md").write_text(md)
    print(md)
    missing = {m: meta["flights"] - n for m, n in meta["rows_per_method"].items()
               if n != meta["flights"]}
    if missing:
        print(f"[dev_compare] flights missing per column: {missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
