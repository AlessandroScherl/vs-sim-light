"""Copy vs-sim's baselines, bench, scenes, starts and fast tier into this repository.

    python tools/vssim_copy.py                 # re-copy at the pin in external/VSSIM_COMMIT
    python tools/vssim_copy.py --from <sha>    # must equal the pin ...
    python tools/vssim_copy.py --from <sha> --set-pin   # ... or move the pin with it

docs/PORT_PLAN.md section 2 (owner decisions D-1 copy, D-2 separate for good). Every source
blob is read with ``git -C external/vs-sim show <sha>:<path>`` - never the clone's working
tree - put through :func:`transform` (the mechanical rewrite rules of section 2.3, then the
reviewed patches of ``tools/vssim_copy_patches.json``, then one provenance line) and written
to its destination; ``vsbench/vs/COPY_MANIFEST.json`` records every file. The drift check
(``tools/vssim_drift.py``, ``tests/vs/test_drift.py``) applies the very same
:func:`transform`, so the copy and the check can never disagree.

Nothing under ``vsbench/vs/`` or ``tests/vs/`` is edited by hand except the files named in
:data:`HAND_WRITTEN`. A baseline is run as published; a change goes upstream into vs-sim and
comes back by moving the pin, never by an edit here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT / "external" / "vs-sim"
PIN_FILE = ROOT / "external" / "VSSIM_COMMIT"
PATCHES_FILE = ROOT / "tools" / "vssim_copy_patches.json"
MANIFEST = ROOT / "vsbench" / "vs" / "COPY_MANIFEST.json"
TOOL = "tools/vssim_copy.py"

#: (source prefix in vs-sim, destination prefix here, rewrite?) - docs/PORT_PLAN.md 2.2.
#: ``rewrite`` = the rules + the provenance line apply to the ``.py`` files under it; the
#: other trees are data and are copied byte for byte. ``tests/fixtures`` is data on purpose:
#: its ``reference/`` sources are vendored byte-for-byte from their releases (md5-pinned in
#: their PROVENANCE.md files) and its ``*_make_*`` generators ran once, in the source repo.
TREES: tuple[tuple[str, str, bool], ...] = (
    ("vssim/", "vsbench/vs/", True),
    ("configs/", "vsbench/vs/data/configs/", False),
    ("benchmark/", "vsbench/vs/data/benchmark/", False),
    ("assets/asset_library_v2.json", "vsbench/vs/data/assets/asset_library_v2.json", False),
    ("tests/conftest.py", "tests/vs/conftest.py", True),
    ("tests/fast/", "tests/vs/fast/", True),
    ("tests/fixtures/", "tests/vs/fixtures/", False),
)

#: The copy's two roots: nothing may live under them that the manifest does not name,
#: except these hand-written files (the manifest itself, the provenance page, the drift test).
DEST_ROOTS = ("vsbench/vs", "tests/vs")
HAND_WRITTEN = frozenset({
    "vsbench/vs/COPY_MANIFEST.json",
    "vsbench/vs/PROVENANCE.md",
    "tests/vs/test_drift.py",
})
#: Never part of the copy and never a drift: interpreter and tool caches.
IGNORED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})

#: docs/PORT_PLAN.md section 2.3, applied in this order to every rewritten ``.py``.
#: Hyphenated / underscored names (``vssim-config-1``, ``vssim_out``) never match: they are
#: schema names and record data, not module names.
RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("R1", re.compile(r"\bfrom vssim import\b"), "from vsbench.vs import"),
    ("R2", re.compile(r"\bfrom vssim\."), "from vsbench.vs."),
    ("R3a", re.compile(r"\bimport vssim\."), "import vsbench.vs."),
    ("R3b", re.compile(r"\bimport vssim\b(?![.\w-])"), "from vsbench import vs as vssim"),
    ("R4", re.compile(r"\bvssim\.(?=[A-Za-z_])"), "vsbench.vs."),
)
PROVENANCE_PREFIX = "# vs-sim copy: "


def provenance_line(src: str, sha: str, patch_id: str | None) -> str:
    """Line 1 of every rewritten ``.py`` (before the docstring; the drift check regenerates
    it). ``noqa: E501``: with a 40-character sha the line is longer than 100 columns."""
    edits = "rewrite rules" + (f" + patch {patch_id}" if patch_id else "")
    return f"{PROVENANCE_PREFIX}{src} @ {sha} ({TOOL}); edits: {edits}  # noqa: E501\n"


# --------------------------------------------------------------------------- the source
def git(*args: str, binary: bool = False) -> str | bytes:
    out = subprocess.run(["git", "-C", str(CLONE), *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def resolve(sha: str) -> str:
    return str(git("rev-parse", "--verify", f"{sha}^{{commit}}")).strip()


def read_pin() -> str:
    return PIN_FILE.read_text().strip()


def source_files(sha: str) -> list[tuple[str, str, str]]:
    """``(src path, blob sha, mode)`` for every file of the copy list at ``sha``."""
    out = []
    for line in str(git("ls-tree", "-r", "--full-tree", sha)).splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind != "blob":
            raise SystemExit(f"{path}: a {kind} in the copy list (submodule?) - refusing")
        if any(path == p.rstrip("/") or (p.endswith("/") and path.startswith(p))
               for p, _d, _r in TREES):
            out.append((path, blob, mode))
    return out


def dest_of(src: str) -> tuple[str, bool]:
    """The destination path of ``src`` and whether the rewrite applies to it."""
    for sp, dp, rewrite in TREES:
        if src == sp or (sp.endswith("/") and src.startswith(sp)):
            return dp + src[len(sp):], rewrite and src.endswith(".py")
    raise KeyError(src)


def blob(sha: str, src: str) -> bytes:
    return bytes(git("show", f"{sha}:{src}", binary=True))


# ------------------------------------------------------------------------ the transform
def load_patches() -> dict[str, dict]:
    """``dest -> {"id", "pairs": [{"old", "new", "count", "reason"}]}``; validated."""
    doc = json.loads(PATCHES_FILE.read_text())
    out: dict[str, dict] = {}
    for p in doc["patches"]:
        if p["dest"] in out:
            raise SystemExit(f"{PATCHES_FILE.name}: {p['dest']} is patched twice")
        if not p.get("reason") or not all(pair.get("reason") for pair in p["pairs"]):
            raise SystemExit(f"{PATCHES_FILE.name}: patch {p['id']} lacks a reason")
        out[p["dest"]] = p
    return out


def rewrite(text: str) -> tuple[str, list[str]]:
    """The mechanical rules; returns the text and the ids of the rules that fired."""
    fired = []
    for rid, pat, repl in RULES:
        text, n = pat.subn(repl, text)
        if n:
            fired.append(rid)
    return text, fired


def apply_patch(text: str, patch: dict, dest: str) -> str:
    for pair in patch["pairs"]:
        want = int(pair.get("count", 1))
        got = text.count(pair["old"])
        if got != want:
            raise ValueError(f"patch {patch['id']} on {dest}: {pair['old']!r} occurs {got} "
                             f"times after the rules, expected {want} - the source moved; "
                             "re-review the patch")
        text = text.replace(pair["old"], pair["new"])
    return text


def transform(src: str, data: bytes, sha: str, patches: dict[str, dict]) -> tuple[bytes, dict]:
    """The one function the copy and the drift check share: ``(dest bytes, manifest info)``."""
    dest, rewritten = dest_of(src)
    info = {"rules": [], "patch": None}
    if not rewritten:
        if dest in patches:
            raise ValueError(f"{dest}: a patch on a byte-identical (data) file")
        return data, info
    text = data.decode("utf-8")
    if text.startswith("#!"):
        raise ValueError(f"{src}: a shebang - the provenance line cannot be line 1")
    text, info["rules"] = rewrite(text)
    patch = patches.get(dest)
    if patch is not None:
        text = apply_patch(text, patch, dest)
        info["patch"] = patch["id"]
    out = provenance_line(src, sha, info["patch"]) + text
    return out.encode("utf-8"), info


def expected_copy(sha: str) -> dict[str, tuple[bytes, dict]]:
    """``dest -> (bytes, manifest entry)`` for the whole copy at ``sha``."""
    patches = load_patches()
    out: dict[str, tuple[bytes, dict]] = {}
    for src, blob_sha, mode in source_files(sha):
        data, info = transform(src, blob(sha, src), sha, patches)
        dest = dest_of(src)[0]
        out[dest] = (data, {
            "src": src, "src_blob": blob_sha, "src_mode": mode, "vssim_commit": sha,
            "rewritten": bool(dest_of(src)[1]), "rules": info["rules"], "patch": info["patch"],
            "dest_sha256": hashlib.sha256(data).hexdigest(),
        })
    unused = sorted(set(patches) - set(out))
    if unused:
        raise ValueError(f"patches for files that are not in the copy: {unused}")
    return out


def present_files() -> list[str]:
    """Every file under the copy's roots (caches excluded), repo-relative."""
    found = []
    for root in DEST_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_dir() or IGNORED_PARTS & set(rel.parts) or p.suffix == ".pyc":
                continue
            found.append(rel.as_posix())
    return sorted(found)


# ----------------------------------------------------------------------------- the copy
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="sha", default=None,
                    help="vs-sim commit to copy (default: external/VSSIM_COMMIT)")
    ap.add_argument("--set-pin", action="store_true",
                    help="write the resolved commit to external/VSSIM_COMMIT (a pin move)")
    args = ap.parse_args(argv)
    pin = read_pin()
    sha = resolve(args.sha or pin)
    if sha != pin and not args.set_pin:
        raise SystemExit(f"--from {sha} is not the pin {pin}; pass --set-pin to move the pin "
                         "(one pin for the whole copy - docs/PORT_PLAN.md 2.4)")
    want = expected_copy(sha)
    # clear the old copy (never the hand-written files), then write the new one
    for rel in present_files():
        if rel not in HAND_WRITTEN:
            (ROOT / rel).unlink()
    for root in DEST_ROOTS:
        for d in sorted((ROOT / root).rglob("*"), reverse=True):
            if d.is_dir() and d.name == "__pycache__":
                shutil.rmtree(d)
            elif d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    for dest, (data, entry) in want.items():
        path = ROOT / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if entry["src_mode"] == "100755":
            path.chmod(0o755)
    manifest = {
        "schema": "vsbench-vssim-copy-1",
        "vssim_commit": sha,
        "tool": TOOL,
        "rules": {rid: [pat.pattern, repl] for rid, pat, repl in RULES},
        "patches_file": PATCHES_FILE.relative_to(ROOT).as_posix(),
        "hand_written": sorted(HAND_WRITTEN),
        "files": {d: e for d, (_b, e) in sorted(want.items())},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    if args.set_pin and sha != pin:
        PIN_FILE.write_text(sha + "\n")
    n_rw = sum(e["rewritten"] for _b, e in want.values())
    n_p = sum(bool(e["patch"]) for _b, e in want.values())
    print(f"copied {len(want)} files from vs-sim {sha[:12]} ({n_rw} rewritten, {n_p} patched, "
          f"{len(want) - n_rw} byte-identical); manifest {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
