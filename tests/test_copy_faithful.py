"""The copied vs-sim files are the files that were copied: nothing edited, nothing added.

``vsbench/vs/COPY_MANIFEST.json`` lists every file copied from vs-sim (``vsbench/vs/PROVENANCE.md``)
with the sha256 of the bytes written here. These tests need no vs-sim clone:

* every listed file exists and still has its sha256 (an edit, even of whitespace, fails);
* no other file sits under the copied roots (a hand-written module there would pass for a
  copied one);
* every entry names the one vs-sim commit, and every rewritten file starts with the
  provenance line that says so.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "vsbench" / "vs" / "COPY_MANIFEST.json"
#: the directories that hold only copied files (plus the manifest's ``hand_written`` list)
ROOTS = ("vsbench/vs", "vsbench/configs", "vsbench/benchmark", "vsbench/assets",
         "vsbench/tests", "vsbench/scripts", "vsbench/experiments")
#: run output the copied run sink may write under experiments/ (git-ignored payload)
EXPERIMENTS_KEEP = {"vsbench/experiments/README.md", "vsbench/experiments/.gitkeep"}

M = json.loads(MANIFEST.read_text())
FILES: dict = M["files"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("rel", sorted(FILES))
def test_every_copied_file_is_unchanged(rel):
    path = REPO / rel
    assert path.is_file(), f"{rel} is in the manifest but missing"
    assert _sha256(path) == FILES[rel]["dest_sha256"], (
        f"{rel} differs from the copied bytes: copied files are never edited "
        "(vsbench/vs/PROVENANCE.md)")


def _present() -> list[str]:
    out = []
    for root in ROOTS:
        for p in (REPO / root).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            rel = p.relative_to(REPO).as_posix()
            if rel.startswith("vsbench/experiments/") and rel not in EXPERIMENTS_KEEP:
                continue
            out.append(rel)
    return sorted(out)


def test_nothing_else_sits_under_the_copied_roots():
    known = set(FILES) | set(M["hand_written"])
    stray = [r for r in _present() if r not in known]
    assert not stray, f"files under the copied roots that the manifest does not list: {stray}"


def test_one_vssim_commit_and_the_provenance_line():
    pin = M["vssim_commit"]
    assert len(pin) == 40 and {e["vssim_commit"] for e in FILES.values()} == {pin}
    for rel, e in FILES.items():
        if not e["rewritten"]:
            continue
        lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
        head = lines[1] if lines and lines[0].startswith("#!") else lines[0]
        assert head.startswith(f"# vs-sim copy: {e['src']} @ {pin}"), (rel, head)
        if e["patch"]:
            assert e["patch"] in M["patches"], (rel, e["patch"])


def test_the_hand_written_files_exist():
    for rel in M["hand_written"]:
        assert (REPO / rel).is_file(), rel
