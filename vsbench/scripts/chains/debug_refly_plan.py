#!/usr/bin/env python3
# vs-sim copy: scripts/chains/debug_refly_plan.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The debug re-fly's flight order (owner 2026-09-30), one ``<scene> <live name>`` per line.

    python3 scripts/chains/debug_refly_plan.py

For each kind - shoe, bottle, teapot, drill, mug, hammer - the forward scene ``inst_<kind>``
with the live objects ``same``, ``same_small``, ``other_small`` and ``mine1``: 24 pairs, kind
by kind. Half of them keep a truth for the object's points (``same``: the same object;
``same_small``: the same object smaller), the other half are another object. Every name is
checked against ``configs/scenes.yaml`` and a missing one is refused. numpy + PyYAML only:
runs under any interpreter, Isaac's included. The instance suite's own plan
(``instance_suite_plan.py``) is not touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsbench.vs.scenes import load_scenes  # noqa: E402

KINDS = ("shoe", "bottle", "teapot", "drill", "mug", "hammer")
LIVE = ("same", "same_small", "other_small", "mine1")


def plan() -> list[tuple[str, str]]:
    """``[(scene, live name)]`` in flight order; refuses a name the scene does not have."""
    specs, _ = load_scenes()
    out = []
    for kind in KINDS:
        scene = f"inst_{kind}"
        if scene not in specs:
            raise SystemExit(f"debug re-fly: scene {scene!r} is not in configs/scenes.yaml")
        have = set(specs[scene].live_names())
        for name in LIVE:
            if name not in have:
                raise SystemExit(f"debug re-fly: {scene} has no live object {name!r} "
                                 f"(it has {sorted(have)})")
            out.append((scene, name))
    return out


def main() -> int:
    for scene, name in plan():
        print(scene, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
