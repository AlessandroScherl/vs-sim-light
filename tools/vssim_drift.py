"""Check that the copy of vs-sim in this repository is exactly what the pin says it is.

    python tools/vssim_drift.py                     # exit 0 = no drift
    python tools/vssim_drift.py --upstream <sha>    # what moving the pin to <sha> would change

docs/PORT_PLAN.md section 2.4. For every manifest entry: ``git show <pin>:<src>`` -> the
rewrite rules -> the patches -> the provenance line (``tools/vssim_copy.py::transform``, the
very function the copy used) must equal the file on disk, byte for byte. Also: the manifest
names exactly the files the pin's copy list yields; no file under the copy's roots is absent
from the manifest except the hand-written ones; every entry is at ``external/VSSIM_COMMIT``.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "vssim_copy", Path(__file__).resolve().with_name("vssim_copy.py"))
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def _first_difference(a: bytes, b: bytes) -> str:
    la, lb = a.splitlines(), b.splitlines()
    for i, (x, y) in enumerate(zip(la, lb, strict=False), 1):
        if x != y:
            return f"line {i}: expected {x[:120]!r}, found {y[:120]!r}"
    return f"length: expected {len(la)} lines, found {len(lb)} lines"


def check() -> list[str]:
    """Every problem found, one line each; empty = the copy is the pin's copy."""
    problems: list[str] = []
    if not vc.CLONE.is_dir():
        return [f"the source clone {vc.CLONE} is missing (docs/PORT_PLAN.md section 2, U0)"]
    if not vc.MANIFEST.is_file():
        return [f"{vc.MANIFEST.relative_to(vc.ROOT)} is missing - run tools/vssim_copy.py"]
    pin = vc.read_pin()
    manifest = json.loads(vc.MANIFEST.read_text())
    if manifest.get("vssim_commit") != pin:
        problems.append(f"manifest commit {manifest.get('vssim_commit')} != pin {pin}")
    entries = manifest["files"]
    for dest, e in entries.items():
        if e["vssim_commit"] != pin:
            problems.append(f"{dest}: copied from {e['vssim_commit']}, the pin is {pin}")
    want = vc.expected_copy(pin)
    for dest in sorted(set(want) - set(entries)):
        problems.append(f"{dest}: in the pin's copy list but not in the manifest")
    for dest in sorted(set(entries) - set(want)):
        problems.append(f"{dest}: in the manifest but not in the pin's copy list")
    for dest, (data, entry) in sorted(want.items()):
        path = vc.ROOT / dest
        if not path.is_file():
            problems.append(f"{dest}: missing")
            continue
        have = path.read_bytes()
        if have != data:
            first = _first_difference(data, have)
            problems.append(f"{dest}: differs from the pin's copy - {first}")
        old = entries.get(dest)
        if old is not None and (old["src"], old["src_blob"]) != (entry["src"], entry["src_blob"]):
            problems.append(f"{dest}: manifest source {old['src']}@{old['src_blob'][:10]} != "
                            f"{entry['src']}@{entry['src_blob'][:10]}")
    for rel in vc.present_files():
        if rel not in want and rel not in vc.HAND_WRITTEN:
            problems.append(f"{rel}: under the copy's roots but not in the manifest "
                            "(nothing hand-written lives there)")
    return problems


def upstream(new: str) -> int:
    """Per copied file: did vs-sim change it between the pin and ``new``? Prints the diffs."""
    pin = vc.read_pin()
    new = vc.resolve(new)
    old_files = {s: b for s, b, _m in vc.source_files(pin)}
    new_files = {s: b for s, b, _m in vc.source_files(new)}
    changed = 0
    for src in sorted(set(old_files) | set(new_files)):
        a, b = old_files.get(src), new_files.get(src)
        if a == b:
            continue
        changed += 1
        state = "added" if a is None else "removed" if b is None else "changed"
        print(f"== {src}: {state} upstream ({pin[:10]} -> {new[:10]})")
        if a and b:
            try:
                ta = vc.blob(pin, src).decode().splitlines(keepends=True)
                tb = vc.blob(new, src).decode().splitlines(keepends=True)
                sys.stdout.writelines(difflib.unified_diff(ta, tb, f"{src}@pin", f"{src}@new"))
            except UnicodeDecodeError:
                print("   (binary)")
    print(f"{changed} copied file(s) differ between the pin and {new[:12]}; to pull them in: "
          f"tools/vssim_copy.py --from {new[:12]} --set-pin, then pytest the copied tier")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", metavar="SHA", help="compare the pin with a newer vs-sim")
    args = ap.parse_args(argv)
    if args.upstream:
        return upstream(args.upstream)
    problems = check()
    for p in problems:
        print("DRIFT:", p)
    if not problems:
        print(f"no drift: the copy is vs-sim {vc.read_pin()[:12]} through the rules and "
              f"{len(vc.load_patches())} patch(es)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
