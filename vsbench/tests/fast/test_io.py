# vs-sim copy: tests/fast/test_io.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The two shared file idioms: a write that is all-or-nothing, and a chunked md5.

Both existed three times over before review R30 finding 9. These tests are what makes
the extraction safe: they pin the properties the three call sites relied on - the
suffix-preserving temporary name (``cv2.imwrite`` and ``numpy.savez_compressed`` both
choose behaviour by extension), the parent-directory creation, and that a failed write
leaves the target absent rather than half-made.
"""

from __future__ import annotations

import hashlib

import pytest

from vsbench.vs.bench.io import atomic_write, file_md5


def test_the_file_appears_whole_or_not_at_all(tmp_path):
    path = tmp_path / "deep" / "frame.npz"
    atomic_write(path, lambda tmp: tmp.write_bytes(b"payload"))
    assert path.read_bytes() == b"payload"
    assert not list(tmp_path.rglob("tmp_*")), "a temporary file was left behind"


def test_a_failing_write_leaves_no_target(tmp_path):
    path = tmp_path / "frame.npz"

    def boom(tmp):
        tmp.write_bytes(b"half")
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        atomic_write(path, boom)
    assert not path.exists(), "a partial file is exactly what this helper prevents"


def test_the_temporary_name_keeps_the_suffix(tmp_path):
    """`cv2.imwrite` picks its encoder by extension and has no writer for `.tmp`;
    `numpy.savez_compressed` appends `.npz` to a name that lacks it. Both would break
    silently on a temporary name that dropped the suffix."""
    seen = []
    atomic_write(tmp_path / "x.png", lambda tmp: (seen.append(tmp.name),
                                                  tmp.write_bytes(b"1")))
    assert seen[0].endswith(".png") and seen[0].startswith("tmp_")


def test_the_write_happens_in_the_targets_own_directory(tmp_path):
    """The rename has to be within one filesystem to be atomic."""
    seen = []
    target = tmp_path / "sub" / "y.json"
    atomic_write(target, lambda tmp: (seen.append(tmp.parent), tmp.write_bytes(b"{}")))
    assert seen[0] == target.parent


def test_an_overwrite_replaces_the_old_bytes(tmp_path):
    path = tmp_path / "z.txt"
    atomic_write(path, lambda tmp: tmp.write_bytes(b"first"))
    atomic_write(path, lambda tmp: tmp.write_bytes(b"second"))
    assert path.read_bytes() == b"second"


def test_the_md5_is_the_md5(tmp_path):
    path = tmp_path / "blob.bin"
    payload = b"x" * (3 * (1 << 20) + 17)          # more than one chunk, not a multiple
    path.write_bytes(payload)
    assert file_md5(path) == hashlib.md5(payload).hexdigest()
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert file_md5(empty) == hashlib.md5(b"").hexdigest()
