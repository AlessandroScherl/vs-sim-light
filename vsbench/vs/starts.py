# vs-sim copy: vssim/starts.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The frozen start sets: sampled once, written with a checksum, never re-derived.

Purpose
-------
A benchmark campaign is **paired**: each method flies the same starts from the same
goal, so a contrast is a per-start difference rather than a difference of two
averages. That only works if the starts are a frozen artefact. This module samples
them, writes them with an md5, reads them back, and refuses to pair a start file
against a scene whose goal has drifted. The frozen files themselves live in
``benchmark/starts/`` (:mod:`vsbench.vs.benchmark` knows their pins).

The cells
---------
- **wide** (500): the DMLVS / ART-VS lineage distribution - a 1.2 x 1.2 x 0.3 m
  box in the goal camera frame, look-at focal points on four concentric circles
  of radius 0.08 x (1..4) m, and a deterministic per-circle roll linspace to
  +-179.5 deg. ``wide100`` / ``wide20`` / ``wideroll10`` are explicit id subsets.
- **narrow** (40): the historical precision band, +-18 % of the working distance
  per axis and up to 12 deg about a random axis.
- **near** (60): the terminal-phase stratum, 3-8 mm and up to 3 deg from the goal,
  in its own file so a probe start can never be mistaken for a campaign start.
- **simple** (10; the instance suite, owner 2026-09-23: "just do simple cases ...
  below 60 deg of rotation around the optical axis"): the camera 10-53 % of the goal
  distance away from the goal pose, always looking at the object, turned about its
  viewing axis by one of +-6 ... +-54 deg, never beyond 60 deg - so no rotation
  compensation is needed. Scaled to the working distance, so a close mug and a far drill
  get the same relative difficulty. Its own file, ``<scene>_starts_simple.json``.

Fences
------
- **Draw order is frozen.** ``sample_wide`` draws every position first, then the
  thetas circle by circle; ``sample_near`` draws all directions, then magnitudes,
  then axes, then angles. Re-ordering a single ``rng`` call re-deals every start
  in every campaign since 2026-08-20, silently. Tests pin both orders.
- **Regeneration is not bit-exact across machines.** The frozen files were built
  on the station; rebuilding them here reproduces every id, cell and count exactly
  and every rotation matrix to ~4e-16 (the residual is BLAS, not code). Compare
  with a tolerance; an md5 comparison alone is not a portable test.
- **The near stratum lives in its own file.** ``<scene>_starts_near.json``, never
  merged into the frozen ``<scene>_starts_v2.json``.

Provenance
----------
Extracted from the Refine-VS research repository (``refinevs/eval/starts.py`` plus
the two start-set readers of ``refinevs/eval/stats.py``), ported there from
``genvs-sim`` ``src/eval/starts_v2.py`` and ``src/eval/eval_scenes.py``.

Golden tests: ``tests/fast/test_starts.py`` - regenerating from the recorded seed
and goal reproduces the frozen hollywood files content for content.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "starts_v2"

#: The campaign seed (frozen 2026-08-20). Every ``<scene>_starts_v2.json`` carries it
#: (the two never-seen rooms, ``poster3`` and ``cluttered3``, were drawn with 20260827).
SEED = 20260820

#: The wide cell, from the published ART-VS constants
#: (``config_sift_hollywood_500.yaml``) with **one** deviation (2026-08-20): the roll
#: linspace is widened from the published +-120 deg to +-179.5 deg, so
#: nearly-upside-down starts occur inside the primary distribution instead of a
#: side cell. Exactly 180.0 stays impossible by construction (a 180 deg roll is a
#: retreat trap for an IBVS law), and the published-parity claim survives as the
#: ``|roll| <= 120`` subset, whose statistics are reported beside.
WIDE = {
    "n_circles": 4,
    "per_circle": 125,          # num_samples 500 / num_circles 4
    "radius_step_m": 0.08,      # circle_radius_aug
    "box_lateral_m": 1.2,       # box_sample_size[0:2]
    "box_depth_m": 0.3,         # box_sample_size[2]
    "roll_max_deg": 179.5,      # WIDENED (owner 2026-08-20); published 120
    "published_roll_max_deg": 120.0,
}
WIDE_CAMPAIGN_PER_CIRCLE = 25   # -> wide100
WIDE_ADMISSION_PER_CIRCLE = 5   # -> wide20, a prefix subset of wide100
EXTREME_ROLL_MIN_DEG = 150.0    # past the measured roll cliff of plain matching (~140 deg)
WIDEROLL_N = 10

#: The narrow precision band = the historical sampler, byte-identical.
NARROW = {"n": 40, "seed": 20260727, "translation_frac": 0.18, "rotation_deg": 12.0}

#: The terminal-phase stratum: starts already inside the parking regime, so an
#: episode spends its whole budget where the landing behaviour lives.
NEAR = {"n": 60, "seed": 20260902, "trans_min_mm": 3.0, "trans_max_mm": 8.0,
        "rot_max_deg": 3.0}

#: The simple cell (the instance suite, 2026-09-23). Offsets are fractions of the goal
#: working distance ``d``, in the goal camera's frame: sideways ``U(0.10, 0.35) d`` in a
#: uniform direction, along the viewing line ``U(-0.10, +0.40) d`` (positive = away from
#: the object), so a start is 0.10-0.53 d from the goal pose; the camera looks at the
#: goal's fixation point moved by at most ``0.05 d`` in the goal image plane, and is then
#: rolled about its optical axis by the fixed list below - the first five cover -54..+42
#: deg in 24-deg steps (the owner's "start with 5 starts"), the next five the rest.
SIMPLE = {"n": 10, "seed": 20260923, "lateral_frac": (0.10, 0.35),
          "axial_frac": (-0.10, 0.40), "aim_jitter_frac": 0.05,
          "rolls_deg": (-54.0, 18.0, -30.0, 42.0, -6.0, -42.0, 30.0, -18.0, 54.0, 6.0)}


# ============================================================ pure geometry
def rotation_geodesic_deg(R_a, R_b) -> float:
    """Geodesic angle between two rotation matrices, degrees."""
    R = np.asarray(R_a, float).T @ np.asarray(R_b, float)
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rot_axis_angle(axis, deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for ``deg`` degrees about ``axis``."""
    k = np.asarray(axis, float)
    n = float(np.linalg.norm(k))
    if n < 1e-12:
        raise ValueError("rot_axis_angle: zero axis")
    k = k / n
    a = np.radians(float(deg))
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def look_at_R(cam_pos, focal, y_ref_down) -> np.ndarray:
    """Camera rotation looking from ``cam_pos`` at ``focal`` (optical axis = +z column).

    ``y_ref_down`` fixes the roll-free image orientation - the goal camera's own
    image-down axis, so a start with zero applied roll is roll-aligned with the
    goal. (The published reference's ``world_up=[-1,0,0]`` contradicts its own
    comment; this is the well-defined form of the same intent.)
    """
    z = np.asarray(focal, float) - np.asarray(cam_pos, float)
    nz = float(np.linalg.norm(z))
    if nz < 1e-9:
        raise ValueError("look_at_R: camera position equals focal point")
    z = z / nz
    x = np.cross(np.asarray(y_ref_down, float), z)
    nx = float(np.linalg.norm(x))
    if nx < 1e-6:
        raise ValueError("look_at_R: y_ref parallel to the viewing ray")
    x = x / nx
    return np.column_stack((x, np.cross(z, x), z))


def roll_about_z(R, deg: float) -> np.ndarray:
    """Apply a roll about the camera's own optical axis."""
    a = np.radians(float(deg))
    c, s = np.cos(a), np.sin(a)
    return np.asarray(R, float) @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def goal_look_at(goal_pos, goal_R, distance_m: float) -> np.ndarray:
    """The goal camera's own fixation point: ``pos + d * optical axis``."""
    return np.asarray(goal_pos, float) + float(distance_m) * np.asarray(goal_R, float)[:, 2]


def roll_band(roll_deg: float) -> str:
    """``le120`` / ``b120_150`` / ``ge150`` - the stratification band of a start's roll."""
    a = abs(float(roll_deg))
    if a <= WIDE["published_roll_max_deg"]:
        return "le120"
    return "b120_150" if a < EXTREME_ROLL_MIN_DEG else "ge150"


# ============================================================ the cells
def sample_wide(goal_pos, goal_R, distance_m: float, seed: int = SEED) -> list[dict]:
    """The wide distribution: 500 starts (draw order frozen)."""
    g = np.asarray(goal_pos, float)
    R_g = np.asarray(goal_R, float)
    ex, ey, ez = R_g[:, 0], R_g[:, 1], R_g[:, 2]
    centre = goal_look_at(g, R_g, distance_m)
    n_c, per = WIDE["n_circles"], WIDE["per_circle"]
    half_lat, half_dep = WIDE["box_lateral_m"] / 2.0, WIDE["box_depth_m"] / 2.0

    rng = np.random.default_rng(seed)
    dxy = rng.uniform(-half_lat, half_lat, size=(n_c * per, 2))
    dz = rng.uniform(-half_dep, half_dep, size=(n_c * per, 1))
    positions = g[None, :] + dxy[:, :1] * ex + dxy[:, 1:2] * ey + dz * ez

    out: list[dict] = []
    for cn in range(n_c):
        radius = WIDE["radius_step_m"] * (cn + 1)
        thetas = rng.uniform(-np.pi, np.pi, size=per)
        rolls = np.linspace(-WIDE["roll_max_deg"], WIDE["roll_max_deg"], num=per)
        for i in range(per):
            idx = cn * per + i
            focal = centre + radius * np.cos(thetas[i]) * ex + radius * np.sin(thetas[i]) * ey
            R = roll_about_z(look_at_R(positions[idx], focal, y_ref_down=ey), rolls[i])
            out.append({
                "start_id": f"w{cn + 1}_{i:03d}", "cell": "wide", "circle": cn + 1,
                "tier": f"c{cn + 1}", "roll_deg": float(rolls[i]),
                "roll_band": roll_band(rolls[i]),
                "cam_pos": [float(v) for v in positions[idx]],
                "cam_R_opt": np.asarray(R, float).tolist(),
            })
    return out


def wide_subsets(wide_starts: list[dict], seed: int = SEED) -> tuple[list, list, list]:
    """Deterministic ``(wide100, wide20, wideroll10)`` id lists.

    ``wide20`` is a **prefix** subset of ``wide100``'s per-circle draw, so extending
    the admission set to the campaign set only appends starts. ``wideroll10`` takes
    the first extreme-roll starts in the same permutation order, cycling circles and
    skipping anything already in ``wide20``.
    """
    by_id = {s["start_id"]: s for s in wide_starts}
    rng = np.random.default_rng(seed + 1)
    ids100: list[str] = []
    ids20: list[str] = []
    perms = []
    for cn in range(WIDE["n_circles"]):
        perm = rng.permutation(WIDE["per_circle"])
        perms.append(perm)
        pick = perm[:WIDE_CAMPAIGN_PER_CIRCLE]
        ids100 += [f"w{cn + 1}_{i:03d}" for i in pick]
        ids20 += [f"w{cn + 1}_{i:03d}" for i in pick[:WIDE_ADMISSION_PER_CIRCLE]]
    taken = set(ids20)
    idsroll: list[str] = []
    cursors = [0] * WIDE["n_circles"]
    while len(idsroll) < WIDEROLL_N:
        progressed = False
        for cn in range(WIDE["n_circles"]):
            if len(idsroll) >= WIDEROLL_N:
                break
            perm = perms[cn]
            while cursors[cn] < len(perm):
                sid = f"w{cn + 1}_{perm[cursors[cn]]:03d}"
                cursors[cn] += 1
                if sid not in taken and abs(by_id[sid]["roll_deg"]) >= EXTREME_ROLL_MIN_DEG:
                    idsroll.append(sid)
                    taken.add(sid)
                    progressed = True
                    break
        if not progressed and all(c >= len(perms[i]) for i, c in enumerate(cursors)):
            raise ValueError(f"could not draw {WIDEROLL_N} extreme-roll starts")
    return ids100, ids20, idsroll


def sample_narrow(goal_pos, goal_R, distance_m: float) -> list[dict]:
    """The historical precision band, byte-identical to the first sampler.

    n = 40, seed 20260727, translation uniform in +-0.18 x the working distance per
    axis, rotation of a uniformly random axis and an angle uniform in [0, 12 deg],
    applied on the right so the magnitude is view-relative. The historical
    ``start_XX`` ids are kept for ledger continuity.
    """
    g = np.asarray(goal_pos, float)
    R_g = np.asarray(goal_R, float)
    rng = np.random.default_rng(NARROW["seed"])
    t_max = NARROW["translation_frac"] * float(distance_m)
    out: list[dict] = []
    for i in range(NARROW["n"]):
        dpos = rng.uniform(-t_max, t_max, size=3)
        axis = rng.normal(size=3)
        n = float(np.linalg.norm(axis))
        dR = (np.eye(3) if n < 1e-12
              else rot_axis_angle(axis / n, rng.uniform(0.0, NARROW["rotation_deg"])))
        out.append({
            "start_id": f"start_{i:02d}", "cell": "narrow", "circle": None,
            "tier": "narrow", "roll_deg": None, "roll_band": None,
            "cam_pos": (g + dpos).tolist(), "cam_R_opt": (R_g @ dR).tolist(),
        })
    return out


def sample_near(goal_pos, goal_R, distance_m: float | None = None,
                seed: int = NEAR["seed"], n: int = NEAR["n"],
                id_prefix: str = "n_") -> list[dict]:
    """The near-goal stratum: ``n`` starts a few millimetres off the goal.

    Position: an isotropic direction (a normalised Gaussian) with a magnitude
    uniform in [3, 8] mm, expressed in the **goal camera frame** like the wide
    cell. Orientation: a rotation of a uniform angle in [0, 3 deg] about a uniform
    axis, applied on the right, so the total geodesic error is exactly that angle
    and every component - roll included - is bounded by it. ``distance_m`` is
    unused geometrically and taken only so the signature matches the other samplers.
    ``id_prefix`` names the starts (``n_000``...; another prefix keeps a second near file's
    ids apart from the check's own near start ``n_000``); the draw does not depend on it.
    """
    g = np.asarray(goal_pos, float)
    R_g = np.asarray(goal_R, float)
    rng = np.random.default_rng(int(seed))
    dirs = rng.normal(size=(n, 3))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    mags_mm = rng.uniform(NEAR["trans_min_mm"], NEAR["trans_max_mm"], size=n)
    axes = rng.normal(size=(n, 3))
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    angs = rng.uniform(0.0, NEAR["rot_max_deg"], size=n)
    out: list[dict] = []
    for i in range(n):
        out.append({
            "start_id": f"{id_prefix}{i:03d}", "cell": "near", "circle": None, "tier": "near",
            # No roll GAUGE here (a single small rotation about a random axis, not
            # a look-at plus a roll), so the roll fields stay None as the narrow
            # cell's do; the roll COMPONENT is bounded by the total error.
            "roll_deg": None, "roll_band": None,
            "cam_pos": [float(v) for v in g + R_g @ (dirs[i] * (mags_mm[i] / 1000.0))],
            "cam_R_opt": (R_g @ rot_axis_angle(axes[i], angs[i])).tolist(),
        })
    return out


def sample_simple(goal_pos, goal_R, distance_m: float, seed: int = SIMPLE["seed"],
                  n: int = SIMPLE["n"]) -> list[dict]:
    """The simple cell: ``n`` starts near the goal, looking at the object, |roll| < 60 deg.

    Draw order is frozen (a test pins it): all sideways magnitudes, all sideways
    directions, all axial offsets, all aim-jitter magnitudes, all aim-jitter directions -
    each an array of ``n`` - then the roll of start ``i`` is ``rolls_deg[i % 10]``.
    """
    g = np.asarray(goal_pos, float)
    R_g = np.asarray(goal_R, float)
    d = float(distance_m)
    ex, ey, ez = R_g[:, 0], R_g[:, 1], R_g[:, 2]
    rng = np.random.default_rng(int(seed))
    lat = rng.uniform(*SIMPLE["lateral_frac"], size=n) * d
    phi = rng.uniform(-np.pi, np.pi, size=n)
    ax = rng.uniform(*SIMPLE["axial_frac"], size=n) * d
    jit = rng.uniform(0.0, SIMPLE["aim_jitter_frac"], size=n) * d
    psi = rng.uniform(-np.pi, np.pi, size=n)
    aim0 = goal_look_at(g, R_g, d)
    rolls = SIMPLE["rolls_deg"]
    out: list[dict] = []
    for i in range(n):
        pos = g + lat[i] * (np.cos(phi[i]) * ex + np.sin(phi[i]) * ey) - ax[i] * ez
        aim = aim0 + jit[i] * (np.cos(psi[i]) * ex + np.sin(psi[i]) * ey)
        roll = float(rolls[i % len(rolls)])
        R = roll_about_z(look_at_R(pos, aim, y_ref_down=ey), roll)
        out.append({
            "start_id": f"s_{i:02d}", "cell": "simple", "circle": None, "tier": "simple",
            "roll_deg": roll, "roll_band": roll_band(roll),
            "offset_frac": float(np.hypot(lat[i], ax[i]) / d),
            "cam_pos": [float(v) for v in pos],
            "cam_R_opt": np.asarray(R, float).tolist(),
        })
    return out


# ============================================================ payloads

def utc_today() -> str:
    """Today in UTC as ``YYYY-MM-DD`` - what a payload's ``created`` records.

    UTC and not ``date.today()``: the station and the laptop are in different places
    often enough that a local date would make two builds of the same file look like
    two different days.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _cell_stats(starts: list[dict], goal_pos, goal_R) -> dict:
    g = np.asarray(goal_pos, float)
    R_g = np.asarray(goal_R, float)
    pos = np.array([np.linalg.norm(np.asarray(s["cam_pos"]) - g) for s in starts])
    rot = np.array([rotation_geodesic_deg(R_g, np.asarray(s["cam_R_opt"])) for s in starts])
    return {"n": len(starts), "pos_cm_mean": float(pos.mean() * 100.0),
            "pos_cm_sd": float(pos.std() * 100.0),
            "rot_deg_mean": float(rot.mean()), "rot_deg_sd": float(rot.std())}


def _stamped(starts: list[dict], g, R_g) -> dict:
    """Start entries keyed by id, each carrying its measured initial errors."""
    out: dict[str, dict] = {}
    for s in starts:
        if s["start_id"] in out:
            raise ValueError(f"duplicate start_id {s['start_id']!r}")
        out[s["start_id"]] = dict(
            s, pos_err_m=float(np.linalg.norm(np.asarray(s["cam_pos"]) - g)),
            rot_err_deg=rotation_geodesic_deg(R_g, np.asarray(s["cam_R_opt"])))
    return out


def _goal_block(g, R_g, d: float) -> dict:
    return {"cam_pos": [float(v) for v in g], "cam_R_opt": np.asarray(R_g, float).tolist(),
            "distance_m": float(d), "look_at": [float(v) for v in goal_look_at(g, R_g, d)]}


def build_payload(scene_name: str, goal_pos, goal_R, distance_m: float,
                  seed: int = SEED, hardware: str | None = None,
                  created: str | None = None) -> dict:
    """The full per-scene campaign payload: all cells, id lists and statistics."""
    g = np.asarray(goal_pos, float)
    R_g = np.asarray(goal_R, float)
    d = float(distance_m)
    wide = sample_wide(g, R_g, d, seed=seed)
    ids100, ids20, idsroll = wide_subsets(wide, seed=seed)
    narrow = sample_narrow(g, R_g, d)
    starts = _stamped(wide + narrow, g, R_g)

    def by_id(ids):
        return [starts[i] for i in ids]

    return {
        "schema": SCHEMA, "version": 2, "scene": scene_name, "seed": int(seed),
        "created": created or utc_today(), "hardware": hardware,
        "goal": _goal_block(g, R_g, d),
        "scheme": {"wide": dict(WIDE), "extreme_roll_min_deg": EXTREME_ROLL_MIN_DEG,
                   "narrow": dict(NARROW),
                   "wide100_per_circle": WIDE_CAMPAIGN_PER_CIRCLE,
                   "wide20_per_circle": WIDE_ADMISSION_PER_CIRCLE,
                   "wideroll_n": WIDEROLL_N},
        "cells": {"wide": {"ids": [s["start_id"] for s in wide]},
                  "wide100": {"ids": ids100}, "wide20": {"ids": ids20},
                  "wideroll10": {"ids": idsroll},
                  "narrow": {"ids": [s["start_id"] for s in narrow]}},
        "starts": starts,
        "stats": {
            "wide": _cell_stats(wide, g, R_g),
            "wide100": _cell_stats(by_id(ids100), g, R_g),
            "wide20": _cell_stats(by_id(ids20), g, R_g),
            "wideroll10": _cell_stats(by_id(idsroll), g, R_g),
            "narrow": _cell_stats(narrow, g, R_g),
            "wide_le120_subset": _cell_stats(
                [s for s in wide if s["roll_band"] == "le120"], g, R_g),
            "wide_roll_band_shares": {
                b: sum(1 for s in wide if s["roll_band"] == b) / len(wide)
                for b in ("le120", "b120_150", "ge150")},
            "wide_per_circle": {
                f"c{c}": _cell_stats([s for s in wide if s["circle"] == c], g, R_g)
                for c in range(1, WIDE["n_circles"] + 1)},
            "reference": "papers report ~47+-16 cm / 74+-28 deg at their "
                         "+-120 deg roll cap (the wide_le120_subset row)",
        },
    }


def build_near_payload(scene_name: str, goal_pos, goal_R, distance_m: float,
                       seed: int = NEAR["seed"], n: int = NEAR["n"],
                       hardware: str | None = None, created: str | None = None,
                       id_prefix: str = "n_") -> dict:
    """A payload carrying **only** the near cell - its own file, deliberately."""
    g = np.asarray(goal_pos, float)
    R_g = np.asarray(goal_R, float)
    d = float(distance_m)
    near = sample_near(g, R_g, d, seed=seed, n=n, id_prefix=id_prefix)
    return {
        "schema": SCHEMA, "version": 2, "scene": scene_name, "seed": int(seed),
        "created": created or utc_today(), "hardware": hardware,
        "probe": "n38", "goal": _goal_block(g, R_g, d),
        "scheme": {"near": dict(NEAR, n=int(n), seed=int(seed))},
        "cells": {"near": {"ids": [s["start_id"] for s in near]}},
        "starts": _stamped(near, g, R_g),
        "stats": {"near": _cell_stats(near, g, R_g)},
    }


def build_simple_payload(scene_name: str, goal_pos, goal_R, distance_m: float,
                         seed: int = SIMPLE["seed"], n: int = SIMPLE["n"],
                         hardware: str | None = None, created: str | None = None) -> dict:
    """A payload carrying **only** the simple cell - ``<scene>_starts_simple.json``."""
    g = np.asarray(goal_pos, float)
    R_g = np.asarray(goal_R, float)
    d = float(distance_m)
    simple = sample_simple(g, R_g, d, seed=seed, n=n)
    return {
        "schema": SCHEMA, "version": 2, "scene": scene_name, "seed": int(seed),
        "created": created or utc_today(), "hardware": hardware,
        "probe": "instance_suite", "goal": _goal_block(g, R_g, d),
        "scheme": {"simple": {**{k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in SIMPLE.items()}, "n": int(n), "seed": int(seed)}},
        "cells": {"simple": {"ids": [s["start_id"] for s in simple]}},
        "starts": _stamped(simple, g, R_g),
        "stats": {"simple": _cell_stats(simple, g, R_g),
                  "simple5": _cell_stats(simple[:5], g, R_g)},
    }


# ============================================================ file I/O
def payload_bytes(payload: dict) -> bytes:
    """The exact bytes :func:`write_payload` writes - ``json.dumps(indent=1)``, as before."""
    return json.dumps(payload, indent=1).encode("utf-8")


def payload_md5(payload: dict) -> str:
    """md5 of :func:`payload_bytes` - the pin a runbook quotes."""
    return hashlib.md5(payload_bytes(payload)).hexdigest()


def write_payload(path, payload: dict) -> str:
    """Write the payload and its ``<name>.md5`` sidecar; returns the md5.

    A start file whose checksum travels with it cannot be quietly regenerated.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = payload_bytes(payload)
    p.write_bytes(blob)
    digest = hashlib.md5(blob).hexdigest()
    p.with_suffix(p.suffix + ".md5").write_text(f"{digest}  {p.name}\n", encoding="utf-8")
    return digest


def load_payload(path, expect_md5: str | None = None) -> dict:
    """Read a start file, check its schema and - when asked - its md5."""
    p = Path(path)
    blob = p.read_bytes()
    if expect_md5 is not None:
        got = hashlib.md5(blob).hexdigest()
        if got != expect_md5:
            raise ValueError(f"{p}: md5 {got} != expected {expect_md5} - the start file changed")
    payload = json.loads(blob.decode("utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{p}: not a {SCHEMA} file (schema={payload.get('schema')!r})")
    return payload


def check_goal_match(payload: dict, goal_pos, goal_R, distance_m: float,
                     pos_tol_m: float = 5e-4, rot_tol_deg: float = 0.05,
                     dist_tol_m: float = 1e-3) -> None:
    """Refuse to pair a start file against a scene whose goal drifted.

    The paired design's premise is that every method and every session replays the
    same geometry; a silently changed asset, fill or elevation breaks it. Loud
    :class:`ValueError`, never a warning.
    """
    gp = np.asarray(payload["goal"]["cam_pos"], float)
    gR = np.asarray(payload["goal"]["cam_R_opt"], float)
    gd = float(payload["goal"]["distance_m"])
    dp = float(np.linalg.norm(gp - np.asarray(goal_pos, float)))
    dr = rotation_geodesic_deg(gR, np.asarray(goal_R, float))
    dd = abs(gd - float(distance_m))
    if dp > pos_tol_m or dr > rot_tol_deg or dd > dist_tol_m:
        raise ValueError(
            f"starts file goal mismatch for scene {payload['scene']!r}: "
            f"|dpos|={dp:.2e} m (tol {pos_tol_m}), drot={dr:.4f} deg (tol {rot_tol_deg}), "
            f"|dd|={dd:.2e} m (tol {dist_tol_m}). The scene geometry changed since the "
            f"file was built - rebuild the start file (and expect re-gating) instead "
            f"of overriding.")


def cell_size_static(cell: str) -> int:
    """Planned size of a cell without reading a file - for dry-run accounting."""
    sizes = {"wide": WIDE["n_circles"] * WIDE["per_circle"],
             "wide100": WIDE["n_circles"] * WIDE_CAMPAIGN_PER_CIRCLE,
             "wide20": WIDE["n_circles"] * WIDE_ADMISSION_PER_CIRCLE,
             "wideroll10": WIDEROLL_N, "narrow": NARROW["n"], "near": NEAR["n"],
             "simple": SIMPLE["n"]}
    if cell in sizes:
        return sizes[cell]
    if "_" in cell:
        base, tail = cell.rsplit("_", 1)
        if base in sizes and tail.isdigit() and 0 < int(tail) <= sizes[base]:
            return int(tail)
    raise ValueError(f"unknown start cell {cell!r}; have {sorted(sizes)} (or <cell>_<K>)")


def starts_for_cell(payload: dict, cell: str) -> list[dict]:
    """Ordered starts of one cell (or a ``<cell>_<K>`` prefix, e.g. ``wide100_30``).

    The requested cell name is stamped into each copy's ``cell`` field: records then
    carry which invocation cell an episode belonged to, which the underlying field
    alone cannot say (``wideroll10`` starts are ``wide`` starts). The payload's seed
    is stamped as ``starts_seed`` for the same reason - the record must be able to
    name the file it was flown from without the runbook remembering.
    """
    name, n = cell, None
    if cell not in payload["cells"]:
        if "_" in cell:
            base, tail = cell.rsplit("_", 1)
            if base in payload["cells"] and tail.isdigit():
                name, n = base, int(tail)
        if n is None:
            raise ValueError(f"unknown start cell {cell!r}; have {sorted(payload['cells'])} "
                             f"(or <cell>_<K> prefixes)")
    ids = payload["cells"][name]["ids"]
    if n is not None:
        if n <= 0 or n > len(ids):
            raise ValueError(f"cell prefix {cell!r} out of range (cell has {len(ids)} starts)")
        ids = ids[:n]
    return [dict(payload["starts"][i], cell=name, starts_seed=payload["seed"])
            for i in ids]


# ============================================================ reading a set
THIRDS = ("small", "medium", "large")


def roll_thirds(items, key: str = "start_rotation_deg") -> dict[str, list]:
    """Split a start set into thirds by initial rotation error - CNS's sectioning.

    The wide-100 band is reported "in three thirds by initial rotation error (small
    / medium / large, ~33 each), the way CNS sections its 150 scenes". Deterministic
    by construction: sorted by ``(rotation, start_id)`` so equal rotations never
    reorder, and the remainder goes to the **large** third (100 -> 33 / 33 / 34).

    ``items`` may be records (``start_rotation_deg``) or start-file entries
    (``rot_err_deg``) - pass ``key`` for the latter. Returns
    ``{"small": [...], "medium": [...], "large": [...]}`` with the items themselves.
    """
    rows = [x for x in items if x.get(key) is not None]
    rows.sort(key=lambda r: (float(r[key]), str(r.get("start_id", ""))))
    n = len(rows)
    base, rem = divmod(n, 3)
    sizes = [base, base, base + rem]
    out, i = {}, 0
    for name, size in zip(THIRDS, sizes, strict=True):
        out[name] = rows[i:i + size]
        i += size
    return out


def capped_population(payload: dict, cap: float, n: int,
                      cell: str = "wide100") -> dict[str, float]:
    """``{start_id: |roll|}`` for the first ``n`` starts of ``cell`` with ``|roll| <= cap``.

    Stored order, not sorted: this is the screening population "the first 30 of the
    frozen wide-100 starts per scene with roll <= 90 deg".
    """
    starts = payload["starts"]
    out: dict[str, float] = {}
    for sid in payload["cells"][cell]["ids"]:
        roll = abs(float(starts[sid].get("roll_deg") or 0.0))
        if roll <= cap:
            out[sid] = roll
            if len(out) >= n:
                break
    return out
