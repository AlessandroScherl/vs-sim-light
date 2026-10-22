#!/usr/bin/env python3
# vs-sim copy: scripts/chains/instance_suite_plan.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The instance suite's flight order, one ``<scene> <live name>`` pair per line.

    python3 scripts/chains/instance_suite_plan.py [--block core|gen|all|dev]

The CORE block first (owner 2026-09-23: it gives the first read): for every kind the
forward scene's ``same``, ``same_small``, ``other_small`` and its own picks ``mine*`` (the
mug also ``mine3``, GEN-VS's mug train reference) and the reverse scene's ``same``,
``same_big``, ``other_big`` - 49 pairs; then the GEN block: every forward scene's ``gen*``
(GEN-VS's held-out objects, 33 pairs). The DEV block (owner 2026-10-10, the development set,
``vsbench.vs.benchmark.DEV_SCENES`` x ``DEV_LIVE_NAMES``): the six forward scenes' ``same``,
``same_small``, ``other_small`` and ``mine1`` - 24 pairs, a subset of CORE, flown by
``scripts/chains/dev_suite.sh`` with three starts each. Read from ``configs/scenes.yaml`` - the
object lists live there and only there - so an edited ``live:`` entry changes the chain
without touching it. numpy + PyYAML only: runs under any interpreter, Isaac's included.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsbench.vs.benchmark import DEV_LIVE_NAMES, DEV_SCENES, INSTANCE_SCENES  # noqa: E402
from vsbench.vs.scenes import load_scenes  # noqa: E402

BLOCKS = ("core", "gen", "all", "dev")


def plan(block: str = "all") -> list[tuple[str, str]]:
    """``[(scene, live name)]`` in flight order for ``block`` (``core``, ``gen``, ``all`` or
    ``dev``). ``dev`` keeps each forward scene's file order and refuses a scene that lacks one
    of the four names (an edited ``live:`` entry must not shrink the dev set silently)."""
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")
    specs, _ = load_scenes()
    if block == "dev":
        out = []
        for scene in DEV_SCENES:
            names = specs[scene].live_names()
            missing = [n for n in DEV_LIVE_NAMES if n not in names]
            if missing:
                raise ValueError(f"dev set: scene {scene!r} has no live object(s) {missing}")
            out += [(scene, n) for n in names if n in DEV_LIVE_NAMES]
        return out
    core, gen = [], []
    for scene in INSTANCE_SCENES:
        for name in specs[scene].live_names():
            (gen if name.startswith("gen") else core).append((scene, name))
    return {"core": core, "gen": gen, "all": core + gen}[block]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--block", choices=BLOCKS, default="all")
    args = ap.parse_args(argv)
    for scene, name in plan(args.block):
        print(scene, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
