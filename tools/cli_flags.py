#!/usr/bin/env python
"""The flag tables of ``docs/BASELINES.md``, generated from the real argparse definitions.

    python tools/cli_flags.py            # print the tables
    python tools/cli_flags.py --write    # rewrite the tables in docs/BASELINES.md
    python tools/cli_flags.py --check    # exit 1 when the doc and argparse differ

Each table sits between ``<!-- flags:<name> -->`` and ``<!-- /flags:<name> -->`` in the doc;
``tests/test_baselines_doc.py`` fails when a flag exists in argparse but not in the doc, or
the other way round.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "BASELINES.md"


def _sub(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for act in parser._actions:
        if isinstance(act, argparse._SubParsersAction):
            return act.choices[name]
    raise KeyError(name)


def parsers() -> dict[str, argparse.ArgumentParser]:
    """``{table name: parser}`` - every command a user types."""
    from vsbench import __main__ as front
    from vsbench.world import check_baselines, fly

    top = front.build_parser()
    world = fly.build_parser()
    return {
        "vsbench-fly": _sub(top, "fly"),
        "vsbench-list": _sub(top, "list"),
        "fly-run": _sub(world, "run"),
        "fly-summary": _sub(world, "summary"),
        "fly-global": world,
        "check": check_baselines.build_parser(),
        "check-summary": check_baselines.build_summary_parser(),
    }


def options(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """The parser's own ``--flags`` (no ``-h``, no sub-commands)."""
    return [a for a in parser._actions
            if a.option_strings and not isinstance(a, argparse._HelpAction)]


def flag_names(parser: argparse.ArgumentParser) -> set[str]:
    return {a.option_strings[-1] for a in options(parser)}


def _values(a: argparse.Action) -> str:
    if a.nargs == 0:
        return "(switch)"
    if a.choices:
        v = " \\| ".join(str(c) for c in a.choices)
    elif a.metavar:
        v = str(a.metavar)
    elif a.type is int:
        v = "N"
    elif a.type is float:
        v = "X"
    else:
        v = a.dest.upper()
    if a.nargs in ("+", "*"):
        v += " ..."
    if isinstance(a, argparse._AppendAction):
        v += " (repeatable)"
    return v


def _default(a: argparse.Action) -> str:
    if a.required:
        return "required"
    if a.nargs == 0:
        return ""
    d = a.default
    if d is None or d == [] or d == argparse.SUPPRESS:
        return "-"
    if isinstance(d, (list, tuple)):
        return " ".join(str(x) for x in d)
    return str(d)


def table(parser: argparse.ArgumentParser) -> str:
    rows = ["| flag | values | default | what it does |", "| --- | --- | --- | --- |"]
    for a in options(parser):
        help_ = " ".join((a.help or "").replace("|", "\\|").split())
        help_ = help_.replace("%%", "%")
        rows.append(f"| `{a.option_strings[-1]}` | {_values(a)} | {_default(a)} | {help_} |")
    return "\n".join(rows)


def blocks(text: str) -> dict[str, str]:
    """``{name: table text}`` of every flag block in ``text``."""
    pat = re.compile(r"<!-- flags:([\w-]+) -->\n(.*?)\n?<!-- /flags:\1 -->", re.S)
    return {m.group(1): m.group(2).strip() for m in pat.finditer(text)}


def doc_flags(block: str) -> set[str]:
    return set(re.findall(r"^\| `(--[\w-]+)`", block, re.M))


def write(text: str) -> str:
    tables = {name: table(p) for name, p in parsers().items()}
    for name, body in tables.items():
        pat = re.compile(rf"(<!-- flags:{name} -->\n).*?\n?(<!-- /flags:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"{DOC}: no <!-- flags:{name} --> block")
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--write", action="store_true", help="rewrite the tables in the doc")
    ap.add_argument("--check", action="store_true", help="exit 1 when the doc is out of date")
    args = ap.parse_args(argv)
    if args.write:
        DOC.write_text(write(DOC.read_text()))
        print(f"wrote {DOC}")
        return 0
    if args.check:
        text = DOC.read_text()
        if write(text) != text:
            print(f"{DOC} is out of date: python tools/cli_flags.py --write", file=sys.stderr)
            return 1
        return 0
    for name, p in parsers().items():
        print(f"<!-- flags:{name} -->\n{table(p)}\n<!-- /flags:{name} -->\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
