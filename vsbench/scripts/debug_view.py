#!/usr/bin/env python3
# vs-sim copy: scripts/debug_view.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Draw a flight's debug view from its debug log (no GPU, no Isaac).

    python scripts/debug_view.py <episode dir> [--still STEP|landing|final] [--no-video]
                                 [--every N] [--out DIR]
    python scripts/debug_view.py --pair <pair dir> [--out DIR]

``<episode dir>`` is one ``<out>/films/<scene>_<arm>_<start>/`` written by
``check_baselines.py --telemetry --clean-film``. Writes ``debug.mp4`` (1600 x 1000, 10
frames a second, one frame per step; landings held 1 s, the final frame 2 s) and, with
``--still``, ``debug_<step>.png`` (``landing`` = the method's own stop, else the scored
stop). ``--pair`` draws the landing still of every episode of one pair
(``<out>/<scene>_<live>/``) and an ``index.html`` table of where each flight landed.

``--out DIR`` writes there instead of into the flight's own directory. On the station use
it: the flight directories belong to the container's user and the host cannot write into
them. Run it with the conda interpreter, which has OpenCV:

    ~/genvs_ws/miniconda3/envs/genvs/bin/python scripts/debug_view.py <dir> --out ~/debug_view

The drawing is :mod:`vsbench.vs.bench.debugview`; what each panel shows is in its docstring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsbench.vs.bench import debugview  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("episode", type=Path, nargs="?",
                    help="an episode directory with telemetry.jsonl.gz")
    ap.add_argument("--pair", type=Path, default=None,
                    help="a pair directory: the landing still of every episode + index.html "
                         "(stills only, no videos)")
    ap.add_argument("--still", default=None,
                    help="also write one picture: a step number, 'landing' or 'final'")
    ap.add_argument("--no-video", action="store_true", help="the still only, no debug.mp4")
    ap.add_argument("--every", type=int, default=1, help="draw every n-th step in the video "
                                                         "(landings always)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write here instead of into the flight's directory (needed on the "
                         "station's host: the flight directories are not writable there)")
    args = ap.parse_args(argv)
    if args.pair is not None:
        print(f"[debug_view] index {debugview.render_pair(args.pair, out_dir=args.out)}")
        return 0
    if args.episode is None:
        ap.error("give an episode directory, or --pair <pair dir>")
    if args.still is not None:
        print(f"[debug_view] still "
              f"{debugview.render_still(args.episode, args.still, out_dir=args.out)}")
    if not args.no_video:
        print(f"[debug_view] video "
              f"{debugview.render_episode(args.episode, every=args.every, out_dir=args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
