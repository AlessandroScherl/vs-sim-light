# vs-sim copy: tests/fast/test_run.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The run sidecar: provenance is written by the code, not by a runbook."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from vsbench.vs.bench import run as runmod
from vsbench.vs.bench.config import ConfigError

REPO_ROOT = runmod.REPO_ROOT


def _rev_parse_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def test_start_run_writes_the_three_sidecar_files(tmp_path, valid_config):
    rd = runmod.start_run("scaffold_smoke", valid_config, out_root=tmp_path, allow_dirty=True)

    assert rd.path.parent == tmp_path
    assert rd.path.name == f"{datetime.now(timezone.utc):%Y-%m-%d}_scaffold_smoke"
    for name in (runmod.CONFIG_FILE, runmod.META_FILE, runmod.MD5_FILE):
        assert (rd.path / name).is_file(), f"{name} missing"
    assert (rd.path / runmod.MD5_FILE).read_text() == "", "MD5SUMS starts empty"

    written = yaml.safe_load((rd.path / runmod.CONFIG_FILE).read_text())
    assert written["matcher"]["num_landmarks"] == 24
    assert written["_schema"] == rd.config["_schema"]

    meta = json.loads((rd.path / runmod.META_FILE).read_text())
    assert meta["seed"] == 20260902
    assert meta["hostname"]
    assert meta["python_version"]
    assert meta["allow_dirty"] is True
    assert "gpu" in meta and "torch_version" in meta
    assert meta["timestamp_utc"].endswith("Z")


def test_the_sidecar_records_the_numeric_environment_and_the_tie_order(tmp_path,
                                                                       valid_config):
    """Review R24 section 7, correction 5: a mock episode is only bit-reproducible at a
    FIXED CPU thread count - R24 measured distinct encoder outputs at 1, 4, 6 and 10
    threads - and TF32 and the cuDNN flags do the same thing on the GPU side. None of
    it is a knob this repository sets; all of it is a fact a campaign must record,
    because without it "the same code on the same input" is not a checkable claim."""
    rd = runmod.start_run("numeric_env", valid_config, out_root=tmp_path, allow_dirty=True)
    meta = json.loads((rd.path / runmod.META_FILE).read_text())
    assert meta["torch_num_threads"] >= 1
    for field in ("cuda_version", "cudnn_version", "cuda_matmul_allow_tf32",
                  "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic",
                  "torch_num_interop_threads"):
        assert field in meta, field
    # the draw's tie order is echoed, and says when it is the module default
    assert meta["matcher_tie_order"] == "stable (module default; the config does not state it)"
    stated = {**valid_config, "matcher": {**valid_config["matcher"], "tie_order": "legacy"}}
    rd = runmod.start_run("numeric_env", stated, out_root=tmp_path, allow_dirty=True)
    assert json.loads((rd.path / runmod.META_FILE).read_text())["matcher_tie_order"] \
        == "legacy"


def test_extra_facts_reach_the_sidecar_and_may_not_overwrite_its_own(tmp_path,
                                                                     valid_config):
    """`extra` carries facts of the SESSION that are not in the arm's config - protocol
    v2.2's goal capture is the first of them. A sidecar whose `git_commit` came from a
    caller would be worth nothing, so a clash raises."""
    rd = runmod.start_run("extra", valid_config, out_root=tmp_path, allow_dirty=True,
                          extra={"goal_render_warmup": 3, "goal_frame_md5": "abc",
                                 "goal_settle_series": {"n_captures": 3}})
    meta = json.loads((rd.path / runmod.META_FILE).read_text())
    assert meta["goal_render_warmup"] == 3 and meta["goal_frame_md5"] == "abc"
    assert meta["goal_settle_series"] == {"n_captures": 3}
    with pytest.raises(runmod.RunError, match="git_commit"):
        runmod.start_run("extra", valid_config, out_root=tmp_path, allow_dirty=True,
                         extra={"git_commit": "not mine"})


def test_run_json_git_hash_matches_rev_parse(tmp_path, valid_config):
    rd = runmod.start_run("githash", valid_config, out_root=tmp_path, allow_dirty=True)
    meta = json.loads((rd.path / runmod.META_FILE).read_text())
    assert meta["git_commit"] == _rev_parse_head()
    assert meta["git_commit"], "this repo must be a git checkout with at least one commit"


def test_dirty_tree_refuses_and_allow_dirty_records_it(tmp_path, valid_config,
                                                       writable_checkout):
    canary = REPO_ROOT / "_dirty_canary_for_tests.tmp"
    canary.write_text("temporary; the test removes it\n", encoding="utf-8")
    try:
        assert "_dirty_canary_for_tests.tmp" in runmod.dirty_paths(), "canary not seen by git"
        with pytest.raises(runmod.RunError, match="dirty"):
            runmod.start_run("refused", valid_config, out_root=tmp_path)
        assert not list(tmp_path.iterdir()), "a refused run must create nothing"

        rd = runmod.start_run("forced", valid_config, out_root=tmp_path, allow_dirty=True)
        meta = json.loads((rd.path / runmod.META_FILE).read_text())
        assert meta["git_dirty"] is True
        assert "_dirty_canary_for_tests.tmp" in meta["git_dirty_paths"]
    finally:
        canary.unlink(missing_ok=True)


def test_finish_run_writes_md5_of_every_file(tmp_path, valid_config):
    rd = runmod.start_run("manifest", valid_config, out_root=tmp_path, allow_dirty=True)
    payload = rd.path / "records.jsonl"
    payload.write_text('{"episode": 1}\n', encoding="utf-8")
    (rd.path / "sub").mkdir()
    nested = rd.path / "sub" / "telemetry.txt"
    nested.write_text("x\n", encoding="utf-8")

    md5_path = rd.finish()
    assert md5_path == rd.path / runmod.MD5_FILE

    listed = {}
    for line in md5_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    assert set(listed) == {
        runmod.CONFIG_FILE, runmod.META_FILE, "records.jsonl", "sub/telemetry.txt",
    }
    assert runmod.MD5_FILE not in listed
    for name, digest in listed.items():
        assert hashlib.md5((rd.path / name).read_bytes()).hexdigest() == digest


def test_start_run_validates_the_config(tmp_path, valid_config):
    del valid_config["matcher"]["num_landmarks"]
    with pytest.raises(ConfigError, match="matcher.num_landmarks"):
        runmod.start_run("bad", valid_config, out_root=tmp_path, allow_dirty=True)
    assert not list(tmp_path.iterdir())


def test_second_run_same_day_same_name_does_not_overwrite(tmp_path, valid_config):
    a = runmod.start_run("twice", valid_config, out_root=tmp_path, allow_dirty=True)
    b = runmod.start_run("twice", valid_config, out_root=tmp_path, allow_dirty=True)
    assert a.path != b.path
    assert b.path.name.endswith("_02")


def test_already_resolved_config_keeps_its_source(tmp_path, valid_config_file):
    from vsbench.vs.bench.config import load

    resolved = load(valid_config_file)
    rd = runmod.start_run("resolved", resolved, out_root=tmp_path, allow_dirty=True)
    written = yaml.safe_load((rd.path / runmod.CONFIG_FILE).read_text())
    assert written["_source"] == str(valid_config_file)


def test_run_dir_under_out_root_does_not_dirty_the_next_run(writable_checkout):
    """A run written into the repo's own experiments/ must not block the next one."""
    inside = Path(REPO_ROOT) / "experiments"
    assert inside.is_dir()
    canary = inside / "_run_canary_for_tests" / "README.md"
    canary.parent.mkdir(parents=True, exist_ok=True)
    canary.write_text("temporary; the test removes it\n", encoding="utf-8")
    try:
        seen = runmod.dirty_paths()
        assert any("_run_canary_for_tests" in p for p in seen), seen
        filtered = runmod.dirty_paths(ignore_prefix=inside)
        assert all("_run_canary_for_tests" not in p for p in filtered), filtered
    finally:
        canary.unlink(missing_ok=True)
        canary.parent.rmdir()


def test_porcelain_parser_keeps_the_first_character():
    """`git status --porcelain` uses two status columns: the path starts at column 4."""
    parsed = runmod._parse_porcelain(
        ' M README.md\n?? "new file.txt"\nR  old.md -> new.md\n M experiments/README.md\n'
    )
    assert parsed == ["README.md", "new file.txt", "new.md", "experiments/README.md"]


def test_dirty_paths_reads_unstripped_porcelain(monkeypatch):
    """The status columns are load-bearing: ' M README.md' leads with a space, so
    stripping the output costs the FIRST line its first character (README -> EADME)
    and the mangled path escapes every prefix filter."""
    captured = {}

    def fake_git(*args, strip=True):
        captured["args"] = args
        captured["strip"] = strip
        return " M README.md\n?? scratch/new.py\n M experiments/README.md\n"

    monkeypatch.setattr(runmod, "_git", fake_git)
    paths = runmod.dirty_paths()
    assert captured["strip"] is False, "dirty_paths must read porcelain unstripped"
    assert paths == ["README.md", "scratch/new.py", "experiments/README.md"]

    filtered = runmod.dirty_paths(ignore_prefix=REPO_ROOT / "experiments")
    assert filtered == ["README.md", "scratch/new.py"]


def test_dirty_paths_is_none_when_git_cannot_be_read(monkeypatch):
    monkeypatch.setattr(runmod, "_git", lambda *a, **k: None)
    assert runmod.dirty_paths() is None


def test_modified_tracked_file_inside_experiments_is_filtered(writable_checkout):
    """The bug this guards: a mangled path escapes the ignore-prefix filter."""
    target = REPO_ROOT / "experiments" / "README.md"
    original = target.read_bytes()
    target.write_bytes(original + b"\n# canary\n")
    try:
        assert "experiments/README.md" in runmod.dirty_paths()
        filtered = runmod.dirty_paths(ignore_prefix=REPO_ROOT / "experiments")
        assert "experiments/README.md" not in filtered
    finally:
        target.write_bytes(original)


def test_no_git_refuses_unless_allowed(tmp_path, valid_config, monkeypatch):
    """A missing git must stop the run, never record a silent git_commit: null."""
    monkeypatch.setenv("PATH", str(tmp_path))  # no git binary reachable
    with pytest.raises(runmod.RunError, match="git"):
        runmod.start_run("nogit", valid_config, out_root=tmp_path / "runs")
    rd = runmod.start_run(
        "nogit", valid_config, out_root=tmp_path / "runs", allow_no_git=True
    )
    meta = json.loads((rd.path / runmod.META_FILE).read_text())
    assert meta["git_commit"] is None
    assert meta["git_available"] is False
    assert meta["allow_no_git"] is True


def test_out_root_defaults_to_this_repos_experiments(valid_config, monkeypatch,
                                                     tmp_path, writable_checkout):
    """Launched from anywhere, a run lands in THIS repo's experiments/ — not in ./."""
    monkeypatch.chdir(tmp_path)
    rd = runmod.start_run("default_root", valid_config, allow_dirty=True)
    try:
        assert rd.path.parent == REPO_ROOT / "experiments"
        assert not (tmp_path / "experiments").exists()
    finally:
        for f in sorted(rd.path.rglob("*"), reverse=True):
            f.unlink()
        rd.path.rmdir()


def test_finish_run_stamps_finished_utc_and_records_the_driver(tmp_path, valid_config):
    rd = runmod.start_run("stamped", valid_config, out_root=tmp_path, allow_dirty=True)
    before = json.loads((rd.path / runmod.META_FILE).read_text())
    assert before["finished_utc"] is None
    assert "gpu_driver" in before  # None where nvidia-smi is absent
    rd.finish()
    after = json.loads((rd.path / runmod.META_FILE).read_text())
    assert after["finished_utc"] and after["finished_utc"].endswith("Z")
    listed = dict(
        reversed(line.split("  ", 1)) for line in
        (rd.path / runmod.MD5_FILE).read_text().splitlines()
    )
    assert hashlib.md5((rd.path / runmod.META_FILE).read_bytes()).hexdigest() == \
        listed[runmod.META_FILE], "MD5SUMS must hash run.json AFTER the stamp"
