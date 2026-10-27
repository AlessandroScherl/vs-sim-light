"""``docs/BASELINES.md`` against the code: every flag and every flyable name, both ways.

The flag tables are generated from the argparse definitions (``tools/cli_flags.py``); a flag
added to a parser and not to the doc - or left in the doc after it was removed - fails here.
The same for the names ``python -m vsbench list`` prints.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "BASELINES.md"


def _cli_flags():
    spec = importlib.util.spec_from_file_location("_cli_flags", REPO / "tools" / "cli_flags.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLI = _cli_flags()


@pytest.mark.parametrize("name", sorted(CLI.parsers()))
def test_every_flag_is_in_the_doc_and_back(name):
    parser = CLI.parsers()[name]
    blocks = CLI.blocks(DOC.read_text())
    assert name in blocks, f"docs/BASELINES.md has no <!-- flags:{name} --> table"
    code, doc = CLI.flag_names(parser), CLI.doc_flags(blocks[name])
    assert not code - doc, f"{name}: flags missing from the doc: {sorted(code - doc)}"
    assert not doc - code, f"{name}: flags in the doc that argparse lacks: {sorted(doc - code)}"


def test_flag_tables_are_up_to_date():
    text = DOC.read_text()
    assert CLI.write(text) == text, "run: python tools/cli_flags.py --write"


def test_every_flyable_name_is_in_the_doc_and_back():
    from vsbench.methods import available_methods, method_class
    from vsbench.world.fly import flyable_ids, load_protocol

    protocol = load_protocol()
    ids = {e["id"] for e in protocol["arms"]}
    own = {m for m in available_methods()
           if m not in ids and not getattr(method_class(m), "privileged", False)}
    code = set(flyable_ids(protocol)) | own
    text = DOC.read_text()
    section = text.split("## The names you can fly", 1)[1].split("\n## ", 1)[0]
    doc = set(re.findall(r"^\| `([a-z0-9_]+)` \|", section, re.M))
    assert not code - doc, f"names missing from the doc: {sorted(code - doc)}"
    assert not doc - code, f"names in the doc that cannot be flown: {sorted(doc - code)}"


def test_readme_points_to_the_doc():
    assert "docs/BASELINES.md" in (REPO / "README.md").read_text()
