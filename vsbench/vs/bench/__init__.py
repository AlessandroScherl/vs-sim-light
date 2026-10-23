# vs-sim copy: vssim/bench/__init__.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The bench: fly an arm on a scene from the frozen starts and write records.

=============== ================================================================
module          what it owns
=============== ================================================================
``config``      the config loader: five required fields with no default
``arms``        the arm registry - a config builds a method, nothing parses a name
``runner``      the episode loop, the telemetry, the record, the episode sink
``scoring``     the three success rules, the outcome taxonomy, landing statistics
``stats``       the paired bootstrap and the record selectors
``run``         the run sidecar (resolved config, git hash, seed, MD5SUMS)
``io``          atomic writes and chunked md5s
=============== ================================================================

The driver is ``scripts/fly.py`` (``plan`` / ``run`` / ``summary``); the campaign
shape is ``configs/protocol.yaml``.
"""
