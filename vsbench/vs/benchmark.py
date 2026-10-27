# vs-sim copy: vssim/benchmark.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The benchmark: six scenes x their frozen start files, md5-pinned.

Purpose
-------
"Test scenes" means two things frozen together: a scene (``configs/scenes.yaml``
+ its baked layout, which fix the goal pose) and the start poses flown from it
(``benchmark/starts/<scene>_starts_{v2,near}.json``). This module is the one
place that pairs them and pins the files, so another project can write::

    from vsbench.vs import benchmark
    for start in benchmark.select_starts("hollywood", "capped30"):
        ...   # start["cam_pos"], start["cam_R_opt"], start["start_id"]

and be flying exactly the population every stored campaign flew.

The start sets
--------------
============ ========= ================ ===== ==============================================
set          file      cell             n     use
============ ========= ================ ===== ==============================================
``capped30`` v2        wide100          30    screening: first 30 of wide100 with |roll| <= 90
``wide100``  v2        wide100          100   the headline band; report in thirds by rotation
``narrow20`` v2        narrow           20    precision band (posters only; ~78 mm / 5.8 deg)
``near60``   near      near             60    terminal-phase probes (posters only; 3-8 mm)
``simple5``  simple    simple           5     the instance suite (2026-09-23): |roll| < 60 deg
``simple10`` simple    simple           10    the same, all ten
============ ========= ================ ===== ==============================================

The instance suite's twelve scenes (:data:`INSTANCE_SCENES`) carry only a ``simple`` file,
pinned in :data:`INSTANCE_STARTS` once the station has built it. The development set
(owner 2026-10-10: :data:`DEV_SCENES` x :data:`DEV_LIVE_NAMES` x :data:`DEV_START_IDS`, 72
flights per method) is a fixed slice of it, not a new start set.

Fences
------
- **A start file is read through its pin.** :func:`load_starts` refuses a file
  whose md5 moved; :func:`check_scene_goal` refuses a scene whose derived goal
  drifted from the goal the file was built against.
- **Goal distances here are the table's, never the run's input.** A scene
  re-derives its goal from its own geometry every time; :data:`GOAL_DISTANCE_M`
  exists so a reader can see what the frozen files expect without booting Isaac.

Provenance
----------
The pairing and the pins are ``configs/protocol_v2.yaml`` sections 2 and 3 of the
Refine-VS research repository (evaluation protocol v2, 2026-09-02), with the same
files, md5s and seeds; the start-set reads are its ``scripts/fly.py:select_starts``.
"""

from __future__ import annotations

from pathlib import Path

from vsbench.vs import starts as startsmod
from vsbench.vs.camera import REPO_ROOT
from vsbench.vs.scene import Scene
from vsbench.vs.scenes import EVAL_SCENES

STARTS_DIR = REPO_ROOT / "benchmark" / "starts"

#: The benchmark scenes, in reporting order.
SCENES = EVAL_SCENES

#: The goal distance each frozen start file was built against (metres). A scene's
#: derived goal must reproduce it (:func:`check_scene_goal`).
GOAL_DISTANCE_M = {
    "hollywood": 0.4341815107823003,
    "poster2": 0.4341815107823003,
    "poster3": 0.4341815107823003,
    "industrial1_drill": 0.4422820341962433,
    "cluttered2": 1.168973419240573,
    "cluttered3": 1.2853190575513067,
}

#: Which scenes were never looked at before their first campaign ("never-seen rooms").
NEVER_SEEN = {"poster3", "cluttered3"}

#: ``{scene: {which: (file name, md5, seed)}}`` - the pins.
STARTS = {
    "hollywood": {
        "v2": ("hollywood_starts_v2.json", "9a242541b0d6d32f2e399bfc971e368d", 20260820),
        "near": ("hollywood_starts_near.json", "7e768de50e0e39346def7097115dfc63", 20260902),
    },
    "poster2": {
        "v2": ("poster2_starts_v2.json", "12087f69a22beb7c4615f96455a0da34", 20260820),
        "near": ("poster2_starts_near.json", "8652872908ff6af14c27bc15c713f737", 20260902),
    },
    "poster3": {
        "v2": ("poster3_starts_v2.json", "a9538e0ee2bf6f4c8ca42dbbe378ccbf", 20260827),
        "near": ("poster3_starts_near.json", "6452dfe278f7f97e56371fbbd5ef0aa7", 20260902),
    },
    "industrial1_drill": {
        "v2": ("industrial1_drill_starts_v2.json", "b8e07e2dac718332be91ffca2df167ea", 20260820),
    },
    "cluttered2": {
        "v2": ("cluttered2_starts_v2.json", "e258fabd8fe5b0d5fc623312fa10fd63", 20260820),
    },
    "cluttered3": {
        "v2": ("cluttered3_starts_v2.json", "c7b8edd1fe702e8fde7bcbf269ec0e68", 20260827),
    },
}

#: Scenes added BESIDE the frozen six (repository rule 4): the same file/md5/seed pins,
#: kept apart so the six stay the six. `shoe1` (2026-09-20): the instance-generalisation
#: scene - one GSO shoe; its start file is built on the station like a room's.
EXTRA_SCENES = ("shoe1", "shoe2")
EXTRA_GOAL_DISTANCE_M = {"shoe1": 0.38310132661345897,      # station build 2026-09-20
                         "shoe2": 0.215494}                  # station build 2026-09-22
EXTRA_STARTS: dict = {
    "shoe1": {"v2": ("shoe1_starts_v2.json", "0ae88a056ccc3d855197d472e5e96a3e", 20260820)},
    "shoe2": {"v2": ("shoe2_starts_v2.json", "ababa7e972cefbd471ef5520a6f3cbdc", 20260820)},
}


#: The instance suite's twelve scenes (2026-09-23; ``configs/scenes.yaml``, section "added
#: 2026-09-23"): six kinds x (goal = the big object, goal = the small one). Each flies
#: its own ``simple`` start file (:func:`vsbench.vs.starts.build_simple_payload`), built on the
#: station from the scene's measured goal - ``scripts/build_starts.py --kind simple
#: --scene inst_shoe inst_shoe_rev ...`` - and pinned here AFTER that build.
INSTANCE_SCENES = tuple(f"inst_{k}{d}" for k in ("shoe", "bottle", "teapot", "drill", "mug",
                                                  "hammer") for d in ("", "_rev"))
#: ``{scene: {"simple": (file name, md5, seed)}}`` - built on the station 2026-09-23 18:04 CEST
#: (``build_starts.py --kind simple``, the twelve scenes in one Isaac process; goal distances
#: 0.2526 / 0.1870 / 0.2672 / 0.1767 / 0.2542 / 0.1713 / 0.3678 / 0.2452 / 0.1413 / 0.0953 /
#: 0.2853 / 0.1595 m in the order below, each equal to the placement preview's to 1e-4 m).
#: md5 of each ``<scene>_starts_simple.json`` (seed ``vsbench.vs.starts.SIMPLE["seed"]`` = 20260923).
_INSTANCE_MD5 = {
    "inst_shoe": "443a58019d64acc4e3182abee5a2f533",
    "inst_shoe_rev": "62e927e6c0163b7b5b8a3210d04476bf",
    "inst_bottle": "d9e61ce2ece4ababc23ec6899e0db563",
    "inst_bottle_rev": "a11fd4a97d9887c04f1daf09d5140d65",
    "inst_teapot": "49987866a7458f62b662f12f5a2d26db",
    "inst_teapot_rev": "5317db9c130eb9aae9fea6be0dae4fcd",
    "inst_drill": "a426172acc243f7b5549d9e94cd468bf",
    "inst_drill_rev": "1a576c8b103856e38624e3ffc70261cf",
    "inst_mug": "5269faade1f2c8018c3b2e8a95b69fa5",
    "inst_mug_rev": "3af5e82c7500b2eb40c186fbd6821fac",
    "inst_hammer": "3718d8735f4ee1e53ed6e9667a2e03b9",
    "inst_hammer_rev": "01fe223637de50468e915cff6a0b8e87",
}
INSTANCE_STARTS: dict = {
    scene: {"simple": (f"{scene}_starts_simple.json", md5, 20260923)}
    for scene, md5 in _INSTANCE_MD5.items()
}
#: T1 Stage 2 (refine-vs, owner 2026-10-22): a second near file for the six dev scenes - the
#: benchmark's near sampler (3-8 mm, <= 3 deg) with a NEW seed and ids ``t1n_000``... (apart
#: from the check's own near start ``n_000``), the goal taken from each scene's pinned simple
#: file (``scripts/build_starts.py --kind near --seed 20261022 --goal-from simple --tag t1near
#: --id-prefix t1n_``). Added beside the frozen files, none of which changes.
_T1_NEAR_MD5 = {
    "inst_shoe": "7fea54f8378861977d1e44c686923240",
    "inst_bottle": "eafc1cc2ff07df3cf1098a33c2660822",
    "inst_teapot": "a9ea063b3c105c4b09e21b5a7f221d1e",
    "inst_drill": "f705f8556fffe614e2ba0dcbe981fa38",
    "inst_mug": "1b136b638058ca395872fad6e4714c14",
    "inst_hammer": "cb6e83c5c07ea550b972dacc165fc980",
}
for _scene, _md5 in _T1_NEAR_MD5.items():
    INSTANCE_STARTS[_scene]["t1near"] = (f"{_scene}_starts_t1near.json", _md5, 20261022)

#: The DEVELOPMENT set (owner 2026-10-10, ``docs/dev_eval_plan_2026-10-10.md`` section 2 at
#: the workspace root): a fixed slice of the instance suite for fast iterations while a
#: method is developed - the six kinds' FORWARD scenes (the reverse scenes, bigger live
#: objects, are left out), four named live objects each (the same object, its smaller
#: copy, the owner's pick of another object of the same size, another smaller object) =
#: 24 pairs, and three starts per pair: the near start (``n_000``, the check script's near
#: sampler) and the first two of the scene's simple file (``s_00``, ``s_01``; ``--far 2
#: --far-set simple5`` flies exactly these, a prefix of the stored order). 72 flights per
#: method; the quick variant (tier 1a) flies the near start only, 24 flights. Nothing here
#: changes a frozen file: the set is a selection of pairs and starts the suite already has
#: (``scripts/chains/instance_suite_plan.py --block dev``, ``scripts/chains/dev_suite.sh``,
#: ``scripts/dev_compare.py``).
DEV_SCENES = tuple(s for s in INSTANCE_SCENES if not s.endswith("_rev"))
DEV_LIVE_NAMES = ("same", "same_small", "mine1", "other_small")
#: the far starts per dev pair (the first ``DEV_FAR`` of ``simple5``) and the three start ids
DEV_FAR = 2
DEV_START_IDS = ("n_000", "s_00", "s_01")


def _pins(scene: str) -> dict:
    if scene in STARTS:
        return STARTS[scene]
    if scene in EXTRA_STARTS and EXTRA_STARTS[scene]:
        return EXTRA_STARTS[scene]
    if scene in INSTANCE_STARTS and INSTANCE_STARTS[scene]:
        return INSTANCE_STARTS[scene]
    have = {s: sorted(v) for s, v in {**STARTS, **EXTRA_STARTS, **INSTANCE_STARTS}.items()}
    raise KeyError(f"no start file pinned for scene {scene!r}; have {have}")


#: The start sets. ``file`` names the pinned file, ``source`` the cell inside it,
#: ``roll_cap_deg`` (when present) makes the set the first ``n`` **by stored order**
#: with ``|roll| <= cap`` rather than a prefix.
START_SETS = {
    "capped30": {"file": "v2", "source": "wide100", "roll_cap_deg": 90.0, "n": 30,
                 "scenes": "all", "use": "method screening"},
    "wide100": {"file": "v2", "source": "wide100", "n": 100, "thirds": True,
                "scenes": "all", "use": "the headline band"},
    "narrow20": {"file": "v2", "source": "narrow", "n": 20,
                 "scenes": ["hollywood", "poster2", "poster3"], "use": "precision band"},
    "near60": {"file": "near", "source": "near", "n": 60,
               "scenes": ["hollywood", "poster2", "poster3"],
               "use": "terminal-phase probes"},
    # the instance suite (2026-09-23): the first five / all ten of a scene's simple file
    "simple5": {"file": "simple", "source": "simple", "n": 5, "scenes": "all",
                "use": "instance suite: 5 simple starts (|roll| < 60 deg)"},
    "simple10": {"file": "simple", "source": "simple", "n": 10, "scenes": "all",
                 "use": "instance suite: 10 simple starts"},
    # T1 Stage 2 (refine-vs, owner 2026-10-22): three near starts per dev pair, a new seed
    "t1near3": {"file": "t1near", "source": "near", "n": 3, "scenes": sorted(_T1_NEAR_MD5),
                "use": "T1 Stage 2: 3 near starts (3-8 mm, <= 3 deg), seed 20261022"},
}


def starts_path(scene: str, which: str = "v2") -> Path:
    """Where the pinned start file of ``scene`` lives."""
    pins = _pins(scene)
    if which not in pins:
        raise KeyError(f"no {which!r} start file for scene {scene!r}; have {sorted(pins)}")
    name, _md5, _seed = pins[which]
    return STARTS_DIR / name


def load_starts(scene: str, which: str = "v2") -> dict:
    """The pinned start file of ``scene``, md5-checked on the way in."""
    pins = _pins(scene)
    if which not in pins:
        raise KeyError(f"no {which!r} start file for scene {scene!r}; have {sorted(pins)}")
    _name, md5, _seed = pins[which]
    return startsmod.load_payload(starts_path(scene, which), expect_md5=md5)


def select_starts(scene: str, start_set: str, limit: int | None = None) -> list[dict]:
    """The starts of one benchmark start set, in the order a campaign flies them.

    Each entry carries ``cam_pos``, ``cam_R_opt`` (the optical-frame pose - build a
    :class:`vsbench.vs.scene.Pose` with ``Pose.from_start``), ``start_id``, the cell it
    came from, its initial ``pos_err_m`` / ``rot_err_deg`` and ``roll_deg`` (wide
    cell only), plus ``starts_seed``.
    """
    if start_set not in START_SETS:
        raise KeyError(f"unknown start set {start_set!r}; have {sorted(START_SETS)}")
    spec = START_SETS[start_set]
    scenes = spec.get("scenes", "all")
    if scenes != "all" and scene not in scenes:
        raise ValueError(f"start set {start_set!r} is not defined for scene {scene!r} "
                         f"(it covers {scenes})")
    payload = load_starts(scene, spec["file"])
    n = int(spec["n"])
    if spec.get("roll_cap_deg") is not None:
        ids = list(startsmod.capped_population(payload, float(spec["roll_cap_deg"]), n,
                                               cell=spec["source"]))
        chosen = [dict(payload["starts"][i], cell=spec["source"],
                       starts_seed=payload["seed"]) for i in ids]
    else:
        chosen = startsmod.starts_for_cell(payload, f"{spec['source']}_{n}")
    return chosen[:limit] if limit else chosen


def check_scene_goal(scene: Scene, payload: dict) -> None:
    """Refuse to fly ``payload``'s starts on a scene whose goal drifted from its goal.

    :func:`vsbench.vs.starts.check_goal_match` under the scene's own facts: the built
    scene's ``goal_pose`` and ``working_distance`` against the file's ``goal`` block
    (5e-4 m / 0.05 deg / 1e-3 m tolerances). Call it once per scene, before the
    first episode - a 3-D scene's goal comes from the measured bounding box, so a
    drifted asset or layout shows up here and nowhere offline.
    """
    goal = scene.goal_pose
    startsmod.check_goal_match(payload, goal.position, goal.rotation, scene.working_distance)
