"""``python -m vsbench fly`` - one command for a published baseline and for your own method.

    python -m vsbench fly --scene poster2 --method sift_ransac --start-set near60 --limit 1 --film
    python -m vsbench fly --scene poster2 --method my_shift --plugin examples/my_method.py \\
        --start-set near60 --limit 1 --film

A thin router over ``python -m vsbench.world.fly run`` (which it calls, unchanged): a name
that is a cell of the benchmark protocol (``vitvs_dinov3``, ``artvs``, ``sift_ransac``, ...)
goes to ``--arms`` (the copied baseline, as published); any other name goes to ``--method``
(a registered vsbench method, flown through ``MethodArm``). ``--plugin`` imports a ``.py``
file or a module first, so the ``@register`` lines in it run. Every other flag is passed
through to ``vsbench.world.fly run`` (``--start-set``, ``--limit``, ``--max-iter``,
``--film``, ``--telemetry``, ``--param key=value``, ``--look``, ...); ``--out`` defaults to
``runs``. ``python -m vsbench list`` prints the names you can fly.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path


def load_plugin(spec: str) -> None:
    """Import a ``.py`` file (by path) or a module (by dotted name) for its ``@register``s."""
    p = Path(spec)
    if spec.endswith(".py") or p.is_file():
        if not p.is_file():
            raise SystemExit(f"--plugin {spec!r}: no such file")
        name = f"vsbench_plugin_{p.stem}"
        mod_spec = importlib.util.spec_from_file_location(name, p.resolve())
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = module
        mod_spec.loader.exec_module(module)
    else:
        importlib.import_module(spec)


def split_names(names, protocol_ids) -> tuple[list[str], list[str]]:
    """``(arms, methods)``: protocol cells to ``--arms``, everything else to ``--method``."""
    ids = set(protocol_ids)
    arms = [n for n in names if n in ids]
    methods = [n for n in names if n not in ids]
    return arms, methods


def protocol_ids() -> list[str]:
    """Every cell id of the copied protocol (a name among them always goes to ``--arms``)."""
    from vsbench.world.fly import load_protocol

    return [e["id"] for e in load_protocol()["arms"]]


def fly_argv(names, rest, *, out: str, plugins=()) -> list[str]:
    """The ``vsbench.world.fly`` argv for ``names`` (plugins already imported)."""
    arms, methods = split_names(names, protocol_ids())
    argv = ["run", *rest, "--out", out]
    if arms:
        argv += ["--arms", *arms]
    if methods:
        argv += ["--method", *methods]
    return argv


def build_parser() -> argparse.ArgumentParser:
    """The ``fly`` and ``list`` flags (``docs/BASELINES.md`` lists them from here)."""
    ap = argparse.ArgumentParser(prog="python -m vsbench", description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fly", help="fly baselines and/or your methods on one scene",
                       allow_abbrev=False, description="Every flag not listed here goes to `python -m "
                                   "vsbench.world.fly run` (see its --help).")
    f.add_argument("--method", nargs="+", required=True,
                   help="baseline names (vitvs_dinov3, artvs, sift_ransac, ...) and/or "
                        "registered method names")
    f.add_argument("--plugin", action="append", default=[],
                   help="a .py file or module that registers your method; repeatable")
    f.add_argument("--out", default="runs", help="output directory (default: runs)")
    lst = sub.add_parser("list", help="the baselines and methods you can fly", allow_abbrev=False)
    lst.add_argument("--plugin", action="append", default=[],
                     help="a .py file or module that registers your method; repeatable")
    return ap


def main(argv=None) -> int:
    args, rest = build_parser().parse_known_args(argv)

    for spec in args.plugin:
        load_plugin(spec)
    if args.cmd == "list":
        from vsbench.methods import available_methods, method_class
        from vsbench.methods.vssim_arms import ARM_IDS
        from vsbench.world.fly import flyable_ids, load_protocol

        ids = protocol_ids()
        flyable = flyable_ids(load_protocol())
        print("baselines:")
        print("  " + " ".join(ARM_IDS))
        print("variants of the baselines (no RANSAC, the 4-way pre-alignment, masks, ...):")
        print("  " + " ".join(i for i in flyable if i not in ARM_IDS))
        own = [m for m in available_methods()
               if m not in ids and not getattr(method_class(m), "privileged", False)]
        print("methods (registered with @register, flown through MethodArm):")
        print("  " + " ".join(own))
        return 0
    from vsbench.world import fly

    return int(fly.main(fly_argv(args.method, rest, out=args.out)))


if __name__ == "__main__":
    sys.exit(main())
