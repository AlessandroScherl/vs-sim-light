# vs-sim copy: vssim/bench/io.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The two file idioms every writer in this repository needs, in one place.

Both existed three times over (review R30 finding 9): the tmp-then-``os.replace``
write in :mod:`refinevs.data.pairs` and :mod:`refinevs.eval.capture`, and the chunked
md5 in :mod:`vsbench.vs.bench.run` and :mod:`refinevs.eval.capture`. Rule 2 says the second
copy is where the helper gets extracted, so here they are.

- :func:`atomic_write` — a file appears complete or not at all. A killed station run
  leaves no half-written frame that a manifest would then checksum and a later job
  would read as data.
- :func:`file_md5` — the checksum every manifest and every ``MD5SUMS`` carries, read in
  1 MB chunks so a 15 MB frame or a 6 GB scene costs no memory.

Golden tests: ``tests/fast/test_io.py``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

#: Read size of the md5 loop. 1 MB: big enough that a 15 MB frame is 15 reads, small
#: enough that a 6 GB scene never holds more than that.
MD5_CHUNK = 1 << 20


def atomic_write(path, write) -> Path:
    """``write(tmp_path)``, then rename over ``path``. Returns ``path``.

    The temporary file keeps the target's **suffix** - some writers pick their encoder
    from the extension (``cv2.imwrite`` has no writer for ``.tmp``, and
    ``numpy.savez_compressed`` appends ``.npz`` to a name that lacks it) - and sits in
    the target's own directory, so the rename is within one filesystem and therefore
    atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp_" + path.name)
    write(tmp)
    os.replace(tmp, path)
    return path


def file_md5(path) -> str:
    """The md5 of a file's bytes, read in :data:`MD5_CHUNK` chunks."""
    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(MD5_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
