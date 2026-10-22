# vs-sim copy: vssim/bench/run.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules + patch P2  # noqa: E501
"""The run sidecar: every run records what produced it.

In the old workspace the resolved config, the code revision and the seed were
written by hand in a runbook, when they were written at all. Here the runner does
it, always, in the run's own directory:

===================== =========================================================
``resolved_config.yaml`` the validated config, exactly as the code saw it
``run.json``             git hash (+ dirty flag), UTC timestamp, seed, hostname,
                         GPU name, python and torch versions, the **numeric
                         environment** (thread counts, CUDA/cuDNN versions, the
                         TF32 and cuDNN flags) and the matcher tie order, plus
                         whatever the caller passed as ``extra``
``MD5SUMS``              a checksum for every file the run wrote
===================== =========================================================

A run refuses to start on a **dirty working tree** (untracked files count - code that
exists only in the working tree is not reproducible from the recorded commit) unless
``allow_dirty=True``, and refuses to start when the **git state cannot be read at all**
unless ``allow_no_git=True``. Both overrides are recorded in ``run.json``: a silent
``git_commit: null`` would break the one guarantee this module exists to make.

In plain words: you can always tell, from the output folder alone, which code and
which settings made these numbers.
"""

from __future__ import annotations

import copy
import json
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import config as cfgmod
from .io import file_md5

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = "resolved_config.yaml"
META_FILE = "run.json"
MD5_FILE = "MD5SUMS"


class RunError(RuntimeError):
    """A run could not be started (most often: the working tree is dirty)."""


def _git(*args: str, strip: bool = True) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if strip else out.stdout


def git_head() -> str | None:
    """The commit this code is at, or ``None`` when git could not be consulted."""
    return _git("rev-parse", "HEAD")


def _parse_porcelain(status: str) -> list[str]:
    """Paths out of ``git status --porcelain``: ``XY<space>path``, renames as ``old -> new``.

    The two status columns are load-bearing: ``' M README.md'`` starts with a space,
    so the output must NOT be stripped before it is split, or the path loses its
    first character and no prefix filter can ever match it.
    """
    paths = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip().strip('"'))
    return paths


def dirty_paths(ignore_prefix: Path | None = None) -> list[str] | None:
    """Uncommitted **or untracked** paths in this repo; ``None`` if git could not be read.

    Untracked files count as dirty: a run whose code exists only in the working tree
    is not reproducible from its recorded commit. Git-ignored files do not count.

    ``ignore_prefix`` drops paths under a directory - used so that the runs already
    written under ``experiments/`` do not make the next run refuse.
    """
    # NOT stripped: a porcelain line like ' M path' leads with a space.
    status = _git("-c", "color.status=false", "-c", "status.relativePaths=true",
                  "-c", "status.branch=false", "-c", "status.showStash=false",
                  "status", "--short", strip=False)
    if status is None:
        return None
    rel_prefix = None
    if ignore_prefix is not None:
        try:
            rel_prefix = str(Path(ignore_prefix).resolve().relative_to(REPO_ROOT))
        except ValueError:
            rel_prefix = None
    paths = []
    for path in _parse_porcelain(status):
        if rel_prefix and (path == rel_prefix or path.startswith(rel_prefix.rstrip("/") + "/")):
            continue
        paths.append(path)
    return paths


def _gpu_driver() -> str | None:
    """The NVIDIA driver version, or ``None`` where nvidia-smi is absent (C17: protocol
    v2 §7 asks every result to name machine, GPU, driver and code hash)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _gpu_name() -> str | None:
    try:
        import torch  # noqa: PLC0415  (optional, absent on the laptop CPU env)
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def _torch_version() -> str | None:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    return str(torch.__version__)


#: The numeric environment of the run - the settings that decide what arithmetic the
#: same code performs (review R24 section 7, correction 5). A mock episode is only
#: bit-reproducible at a **fixed CPU thread count**: R24 measured distinct encoder
#: outputs at 1, 4, 6 and 10 threads, and the same reduction-order effect is why U2's
#: signal tolerance had to be widened. TF32 and the cuDNN flags do the same thing on the
#: GPU side. None of it is a knob this repository sets; all of it is a fact a campaign
#: must record, because without it "the same code on the same input" is not a statement
#: anyone can check.
def _torch_numeric_env() -> dict:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return {}
    env: dict = {}
    def _try(key, fn):
        try:
            env[key] = fn()
        except Exception:                      # pragma: no cover - old/partial builds
            env[key] = None
    _try("torch_num_threads", lambda: int(torch.get_num_threads()))
    _try("torch_num_interop_threads", lambda: int(torch.get_num_interop_threads()))
    _try("cuda_version", lambda: torch.version.cuda)
    _try("cudnn_version", lambda: (torch.backends.cudnn.version()
                                   if torch.backends.cudnn.is_available() else None))
    _try("cuda_matmul_allow_tf32", lambda: bool(torch.backends.cuda.matmul.allow_tf32))
    _try("cudnn_allow_tf32", lambda: bool(torch.backends.cudnn.allow_tf32))
    _try("cudnn_benchmark", lambda: bool(torch.backends.cudnn.benchmark))
    _try("cudnn_deterministic", lambda: bool(torch.backends.cudnn.deterministic))
    return env


def _tie_order(config: dict) -> str | None:
    """What the run's matcher will sort ties by - echoed into the sidecar (R24).

    The config may leave it unstated, in which case the module default is what runs;
    the sidecar says which of the two it is rather than leaving the field empty.
    """
    stated = (config.get("matcher") or {}).get("tie_order")
    if stated is not None:
        return str(stated)
    try:
        from vsbench.vs.baselines.match import DEFAULT_TIE_ORDER  # noqa: PLC0415
    except Exception:                                 # pragma: no cover - torch absent
        return None
    return f"{DEFAULT_TIE_ORDER} (module default; the config does not state it)"


@dataclass
class RunDir:
    """One experiment directory, with its resolved config and provenance."""

    path: Path
    name: str
    config: dict = field(repr=False)
    meta: dict = field(repr=False)

    def finish(self) -> Path:
        """Write ``MD5SUMS`` over everything in the directory. Returns its path."""
        return finish_run(self)


def start_run(name: str, config: dict, out_root: str | Path | None = None,
              *, allow_dirty: bool = False, allow_no_git: bool = False,
              extra: dict | None = None) -> RunDir:
    """Create ``<out_root>/YYYY-MM-DD_<name>/`` and write the sidecar files.

    ``out_root`` defaults to **this repository's** ``experiments/`` (resolved from the
    package location, not from the current directory), so a runbook launched from
    anywhere writes into the same place.

    ``extra`` adds facts of the SESSION that are not in the arm's config - the goal
    capture's ``goal_render_warmup`` is the first - to ``run.json``. It may not
    overwrite a key this module writes: a sidecar whose ``git_commit`` came from a
    caller would be worth nothing.

    Raises :class:`RunError` when the working tree is dirty (untracked files included)
    and ``allow_dirty`` is false, or when the git state cannot be read and
    ``allow_no_git`` is false; raises :class:`vsbench.vs.bench.config.ConfigError` when the
    config is missing a required field. Both overrides are recorded in ``run.json``.
    """
    if config.get("_schema") == cfgmod.SCHEMA_VERSION:
        cfgmod.validate(config)
        resolved = copy.deepcopy(config)
    else:
        resolved = cfgmod.resolve(config)

    root = Path(out_root) if out_root is not None else REPO_ROOT / "experiments"
    head = git_head()
    dirty = dirty_paths(ignore_prefix=root)
    if (head is None or dirty is None) and not allow_no_git:
        raise RunError(
            "cannot read the git state of the code: no `git` on PATH, not a checkout, or git "
            f"refused the directory (on the station: `git config --global --add safe.directory "
            f"{REPO_ROOT}`). Pass allow_no_git=True to record git_commit=null; it is written "
            "to run.json."
        )
    dirty = dirty or []
    if dirty and not allow_dirty:
        shown = ", ".join(dirty[:5]) + (" ..." if len(dirty) > 5 else "")
        raise RunError(
            f"working tree is dirty ({len(dirty)} path(s): {shown}). "
            "Commit first, or pass allow_dirty=True (it is recorded in run.json)."
        )

    now = datetime.now(timezone.utc)
    stem = f"{now:%Y-%m-%d}_{name}"
    path = root / stem
    n = 2
    while path.exists():
        path = root / f"{stem}_{n:02d}"
        n += 1
    path.mkdir(parents=True)

    meta = {
        "name": name,
        "run_dir": str(path),
        "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": head,
        "git_available": head is not None,
        "git_dirty": bool(dirty),
        "git_dirty_paths": dirty if dirty else [],
        "allow_dirty": bool(allow_dirty),
        "allow_no_git": bool(allow_no_git),
        "seed": resolved["seed"],
        "hostname": socket.gethostname(),
        "gpu": _gpu_name(),
        "gpu_driver": _gpu_driver(),
        "finished_utc": None,
        "python_version": platform.python_version(),
        "torch_version": _torch_version(),
        "vssim_version": __import__("vsbench.vs", fromlist=["__version__"]).__version__,
        # the numeric environment (R24 correction 5) and the tie order the draw will use
        **_torch_numeric_env(),
        "matcher_tie_order": _tie_order(resolved),
    }
    clash = sorted(set(extra or {}) & set(meta))
    if clash:
        raise RunError(f"start_run(extra=...) may not overwrite the sidecar's own "
                       f"fields: {', '.join(clash)}")
    meta.update(extra or {})

    with (path / CONFIG_FILE).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=True, default_flow_style=False)
    with (path / META_FILE).open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    (path / MD5_FILE).write_text("", encoding="utf-8")

    return RunDir(path=path, name=name, config=resolved, meta=meta)


def finish_run(run: RunDir) -> Path:
    """Stamp ``finished_utc`` into ``run.json``, then write ``MD5SUMS`` over the directory.

    The manifest covers every file except itself, so ``md5sum -c MD5SUMS`` from inside
    the run directory verifies the whole payload including the stamped ``run.json``.
    """
    meta_path = run.path / META_FILE
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
            fh.write("\n")
        run.meta = meta

    md5_path = run.path / MD5_FILE
    lines = []
    for f in sorted(p for p in run.path.rglob("*") if p.is_file()):
        if f == md5_path:
            continue
        lines.append(f"{file_md5(f)}  {f.relative_to(run.path)}")
    md5_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return md5_path
