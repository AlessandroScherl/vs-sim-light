# vs-sim copy: tests/fast/test_config.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The config loader refuses what the random-24 incident needed to be silent."""

from __future__ import annotations

import copy

import pytest
import yaml

from vsbench.vs.bench.config import (
    BACKBONE_VARIANTS,
    MATCHER_VARIANTS,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    STOP_RULES,
    SUCCESS_RULES,
    ConfigError,
    goal_render_warmup,
    load,
    resolve,
    validate,
)


def _drop(config: dict, dotted: str) -> dict:
    out = copy.deepcopy(config)
    node = out
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    del node[parts[-1]]
    return out


def test_valid_config_loads(valid_config_file, valid_config):
    resolved = load(valid_config_file)
    assert resolved["matcher"]["variant"] == valid_config["matcher"]["variant"]
    assert resolved["matcher"]["num_landmarks"] == 24
    assert resolved["seed"] == 20260902
    assert resolved["_schema"] == SCHEMA_VERSION
    assert resolved["_source"] == str(valid_config_file)


def test_resolve_is_a_copy_and_is_stamped(valid_config):
    resolved = resolve(valid_config)
    resolved["matcher"]["num_landmarks"] = 999
    assert valid_config["matcher"]["num_landmarks"] == 24
    assert resolved["_schema"] == SCHEMA_VERSION
    assert resolved["_source"] is None


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_required_field_is_named(valid_config, field):
    with pytest.raises(ConfigError) as exc:
        validate(_drop(valid_config, field))
    assert field in str(exc.value)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_null_required_field_is_named(valid_config, field):
    broken = copy.deepcopy(valid_config)
    node, parts = broken, field.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = None
    with pytest.raises(ConfigError) as exc:
        validate(broken)
    assert field in str(exc.value)


def test_missing_field_message_lists_all_required(valid_config):
    with pytest.raises(ConfigError) as exc:
        validate(_drop(valid_config, "seed"))
    for field in REQUIRED_FIELDS:
        assert field in str(exc.value)


def test_invalid_matcher_variant(valid_config):
    valid_config["matcher"]["variant"] = "random_24"
    with pytest.raises(ConfigError, match="matcher.variant"):
        validate(valid_config)


def test_invalid_stop_rule(valid_config):
    valid_config["stop"]["rule"] = "when_it_looks_done"
    with pytest.raises(ConfigError, match="stop.rule"):
        validate(valid_config)


def test_invalid_success_rule_entry(valid_config):
    valid_config["scoring"]["success_rule"] = ["predicate_a", "vibes"]
    with pytest.raises(ConfigError, match=r"scoring.success_rule\[1\]"):
        validate(valid_config)


@pytest.mark.parametrize("bad", [0, -3, "24", 24.0, True, None])
def test_num_landmarks_must_be_a_positive_int(valid_config, bad):
    valid_config["matcher"]["num_landmarks"] = bad
    with pytest.raises(ConfigError, match="matcher.num_landmarks"):
        validate(valid_config)


@pytest.mark.parametrize("bad", ["42", 4.2, True, None])
def test_seed_must_be_an_int(valid_config, bad):
    valid_config["seed"] = bad
    with pytest.raises(ConfigError, match="seed"):
        validate(valid_config)


@pytest.mark.parametrize("bad", [-1, 2**32, 2**64])
def test_seed_must_fit_the_rngs(valid_config, bad):
    """numpy's RandomState and OpenCV's RNG refuse these; catch it before the run."""
    valid_config["seed"] = bad
    with pytest.raises(ConfigError, match=r"seed.*2\*\*32"):
        validate(valid_config)


@pytest.mark.parametrize("ok", [0, 1, 2**32 - 1])
def test_seed_edges_are_accepted(valid_config, ok):
    valid_config["seed"] = ok
    validate(valid_config)


def test_empty_success_rule_list_rejected(valid_config):
    valid_config["scoring"]["success_rule"] = []
    with pytest.raises(ConfigError, match="scoring.success_rule"):
        validate(valid_config)


@pytest.mark.parametrize("variant", MATCHER_VARIANTS)
def test_every_declared_variant_is_accepted(valid_config, variant):
    valid_config["matcher"]["variant"] = variant
    validate(valid_config)


@pytest.mark.parametrize("rule", STOP_RULES)
def test_every_declared_stop_rule_is_accepted(valid_config, rule):
    valid_config["stop"]["rule"] = rule
    validate(valid_config)


def test_all_success_rules_together_accepted(valid_config):
    valid_config["scoring"]["success_rule"] = list(SUCCESS_RULES)
    validate(valid_config)


def test_extra_fields_are_allowed_and_carried_through(valid_config):
    valid_config["gain"] = 2.0
    valid_config["scenes"] = ["hollywood", "poster2"]
    resolved = resolve(valid_config)
    assert resolved["gain"] == 2.0
    assert resolved["scenes"] == ["hollywood", "poster2"]


def test_missing_file_and_empty_file(tmp_path, valid_config):
    with pytest.raises(ConfigError, match="not found"):
        load(tmp_path / "nope.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load(empty)
    listy = tmp_path / "list.yaml"
    listy.write_text(yaml.safe_dump([valid_config]), encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load(listy)


def test_duplicate_key_is_refused_with_its_line(tmp_path):
    """PyYAML's default is silent last-wins — exactly the override this loader forbids."""
    path = tmp_path / "dup.yaml"
    path.write_text(
        "seed: 20260902\n"
        "matcher:\n"
        "  variant: mutual_nn_all\n"
        "  num_landmarks: 24\n"
        "  num_landmarks: 48\n"
        "stop:\n"
        "  rule: cap_only\n"
        "scoring:\n"
        "  success_rule: [predicate_a]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load(path)
    assert "duplicate key" in str(exc.value)
    assert "num_landmarks" in str(exc.value)
    assert "line 5" in str(exc.value)


def test_duplicate_top_level_key_is_refused(tmp_path, valid_config):
    path = tmp_path / "dup_top.yaml"
    path.write_text(yaml.safe_dump(valid_config) + "seed: 7\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key"):
        load(path)


def test_yaml_syntax_error_is_a_config_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("matcher: {variant: [\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load(path)


def test_merge_keys_still_work(tmp_path):
    """`<<:` must survive the strict loader — it is not a duplicate key."""
    path = tmp_path / "merge.yaml"
    path.write_text(
        "defaults: &d\n"
        "  rule: cns_plateau\n"
        "seed: 20260902\n"
        "matcher: {variant: mutual_nn_all, num_landmarks: 48, device: cpu}\n"
        "stop:\n"
        "  <<: *d\n"
        "scoring: {success_rule: [predicate_a]}\n",
        encoding="utf-8",
    )
    assert load(path)["stop"]["rule"] == "cns_plateau"


# ------------------------------------------- the two conditional required fields
@pytest.mark.parametrize("variant", BACKBONE_VARIANTS)
def test_a_backbone_variant_must_state_its_device(valid_config, variant):
    """R22 correction 3. `FrozenBackbone` defaulted to `cpu`, no config or driver path
    passed a device, and the first station run therefore flew the whole learned chain on
    the CPU at 0.21 s/step - twice its budget - with nothing but
    `peak_vram_alloc_mb: 0.0` to show for it. The loader now refuses the absence, by
    name, exactly like the five that have never had a default."""
    cfg = copy.deepcopy(valid_config)
    cfg["matcher"]["variant"] = variant
    cfg["matcher"]["num_landmarks"] = 10**6 if variant == "mutual_nn_all" else 24
    del cfg["matcher"]["device"]
    with pytest.raises(ConfigError) as exc:
        validate(cfg)
    assert "matcher.device" in str(exc.value) and "no default" in str(exc.value)
    cfg["matcher"]["device"] = "cuda"
    validate(cfg)                                    # stated: accepted


def test_a_device_that_is_not_a_device_string_is_refused(valid_config):
    for bad in (0, True, "", "   ", None):
        cfg = copy.deepcopy(valid_config)
        cfg["matcher"]["device"] = bad
        with pytest.raises(ConfigError, match="matcher.device"):
            validate(cfg)


def test_a_matcher_without_an_encoder_needs_no_device(valid_config):
    """CNS and the classical arms load no encoder, so the field would be a knob that
    describes nothing - and an unread setting on a record is how a cell gets
    mislabelled."""
    for variant in ("classical_sift", "cns_v1"):
        cfg = copy.deepcopy(valid_config)
        cfg["matcher"]["variant"] = variant
        del cfg["matcher"]["device"]
        validate(cfg)


def test_the_goal_render_warmup_is_required_and_named(valid_config):
    """Protocol v2.2 (owner ruling 2026-09-05): the goal capture's warm-up is a stated
    field of the campaign's scene block, because it changes every stored token's
    provenance - `1` is "no warm-up", which is what N1-N3.8 and E0 carry."""
    with pytest.raises(ConfigError) as exc:
        goal_render_warmup(valid_config)
    assert "scene.render_warmup_steps" in str(exc.value)
    for bad in (0, -1, True, 1.5, "3", None):
        with pytest.raises(ConfigError, match="scene.render_warmup_steps"):
            goal_render_warmup({**valid_config, "scene": {"render_warmup_steps": bad}})
    assert goal_render_warmup({**valid_config, "scene": {"render_warmup_steps": 3}}) == 3
    assert goal_render_warmup({**valid_config, "scene": {"render_warmup_steps": 1}}) == 1
