# vs-sim copy: tests/fast/test_protocol.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""`configs/protocol.yaml` is the campaign in machine form, and `scripts/fly.py` reads it.

Every arm of the protocol builds with a weight-free stub encoder (or a stub CNS
pipeline), so a table's header and its numbers can never come from two different
protocols; the start sets are the benchmark's; the goal-capture block cannot fork from
the scenes yaml; `fly.py plan` prints the shape without loading a backbone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from vsbench.vs import benchmark
from vsbench.vs.baselines.cns import build_cns_arm
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.bench.arms import arm_label, build_arm
from vsbench.vs.scene import D435I
from vsbench.vs.scenes import goal_capture_settings, load_scenes

REPO = Path(__file__).resolve().parents[2]


def _fly():
    spec = importlib.util.spec_from_file_location("fly", REPO / "scripts" / "fly.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fly():
    return _fly()


@pytest.fixture(scope="module")
def protocol(fly):
    return fly.load_protocol()


class _Stub:
    name = "dinov3_vits16"
    input_size = 512
    patch_size = 16
    grid_side = 32
    dim = 384
    device = "cpu"

    def encode(self, frame):  # pragma: no cover
        raise AssertionError("builds only")


class _StubTiles(_Stub):
    name = "dinov2_vits14"
    input_size = 224
    patch_size = 14
    grid_side = 16
    dim = 3456


class _StubSim256(_Stub):
    """The faithful ART-VS arm's encoder geometry (256 px, 16 x 16 cells, 9 bins)."""

    name = "dinov3_vits16_artvs256"
    input_size = 256
    grid_side = 16
    dim = 3456


class _StubPipeline:
    def set_target(self, bgr, dist_scale=None, intrinsic=None):
        pass

    def get_control_rate(self, bgr):
        return [0.0] * 6, {}


def test_the_goal_capture_block_cannot_fork_from_the_scenes_yaml(protocol):
    assert (cfgmod.goal_render_warmup(protocol), cfgmod.goal_capture_keep(protocol)) == \
        goal_capture_settings()


def test_the_start_sets_are_the_benchmarks(protocol):
    assert set(protocol["start_sets"]) <= set(benchmark.START_SETS)
    for name in protocol["start_sets"]:
        assert benchmark.START_SETS[name]["n"] > 0


def test_every_arm_of_the_protocol_builds_and_derives_a_distinct_label(fly, protocol):
    specs, _ = load_scenes()
    labels = {}
    for entry in protocol["arms"]:
        cfg = fly.arm_config(protocol, entry["id"])
        if cfg["matcher"]["variant"] in cfgmod.BACKBONE_VARIANTS:
            cfg["matcher"]["device"] = "cpu"
        if cfg["matcher"]["variant"] == "cns_v1":
            arm = build_cns_arm(cfg, intrinsics=D435I, pipeline=_StubPipeline())
        else:
            stub = {"artvs": _StubTiles(), "artvs_roi_sim": _StubSim256()}.get(
                cfg["matcher"]["variant"], _Stub())
            kw = {"backbone": stub} if cfg["matcher"]["variant"] in cfgmod.BACKBONE_VARIANTS else {}
            if "scene_spec" in (entry.get("needs") or []):
                # a scene that states both a RANSAC model and a prompt (the instance
                # suite's `matcher.prompt: scene` cell reads the latter)
                kw["scene_spec"] = specs["inst_shoe" if entry["id"] in (
                    "artvs_roi_scene", "artvs_roi_sim", "artvs_roi_sim_dev") else "hollywood"]
            arm = build_arm(cfg, intrinsics=D435I, **kw)
        labels[entry["id"]] = arm_label(cfg)
        assert arm.name == labels[entry["id"]]
        assert "_source" in cfg and entry["base"] in cfg["_source"]
    # two cells, two labels: the DINOv2 and DINOv3 ViT-VS cells differ in the encoder
    # and say so; the pre-aligned twins carry `pre4way`
    assert len(set(labels.values())) == len(labels), labels
    assert labels["vitvs_dinov2"] != labels["vitvs_dinov3"]
    assert "pre4way" in labels["artvs_pre4way"] and "pre4way" not in labels["artvs"]


def test_arm_configs_are_the_files_plus_their_deltas_never_hand_assembled(fly, protocol):
    base = cfgmod.load(REPO / "configs" / "arms" / "classical.yaml")
    sift = fly.arm_config(protocol, "sift_ransac")
    orb = fly.arm_config(protocol, "orb_noransac")
    assert sift["matcher"]["variant"] == "classical_sift" and sift["matcher"]["ransac"] == \
        base["matcher"]["ransac"] == "scene_default"
    assert orb["matcher"]["variant"] == "classical_orb" and orb["matcher"]["ransac"] == "none"
    assert orb["controller"] == base["controller"]         # untouched by the delta
    with pytest.raises(SystemExit, match="unknown arm"):
        fly.arm_config(protocol, "nonsense")


def test_select_starts_is_the_benchmarks_population(fly, protocol):
    picked = fly.select_starts(protocol, "hollywood", "capped30", limit=5)
    assert [s["start_id"] for s in picked] == \
        [s["start_id"] for s in benchmark.select_starts("hollywood", "capped30")[:5]]
    with pytest.raises(SystemExit, match="not in the protocol"):
        fly.select_starts(protocol, "hollywood", "wide500")


def test_plan_prints_the_shape_without_a_backbone(fly, capsys):
    before = set(sys.modules)
    rc = fly.main(["plan", "--scenes", "hollywood", "--arms", "vitvs_dinov3", "cns_v1",
                   "--limit", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 scenes x 2 arms x 3 starts = 12 episodes" in out       # 3 + 3 x 3 repeats
    assert "label vitvs_dinov3 = mutual_nn_seeded_subsample@dinov3_vits16" in out
    assert "goal capture: 8 render(s), keep last" in out
    assert "transformers" not in set(sys.modules) - before


def test_the_protocol_file_is_strict_yaml(protocol):
    doc = yaml.safe_load((REPO / "configs" / "protocol.yaml").read_text())
    assert doc["schema"] == "vssim-protocol-1"
    assert doc["episode"]["scored_stop"] == "cns_plateau"
    assert doc["episode"]["dt"] == pytest.approx(1.0 / 60.0)
    assert doc["scoring"]["success_rules"] == list(cfgmod.SUCCESS_RULES)


#: Every protocol cell's derived label on `main` before 2026-10-05 (commit 4271faa,
#: `arm_label(fly.arm_config(protocol, id))` there). Two cells are added beside them
#: (`vitvs_dinov3_objmask_ema08`, `artvs_roi_sim`); not one of these may move.
CELL_LABELS_ON_MAIN = {
    "sift_ransac": "classical_sift/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "sift_noransac": "classical_sift/k24/ur/uniform/ransac-none/cns_plateau",
    "orb_ransac": "classical_orb/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "orb_noransac": "classical_orb/k24/ur/uniform/ransac-none/cns_plateau",
    "akaze_ransac": "classical_akaze/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "akaze_noransac": "classical_akaze/k24/ur/uniform/ransac-none/cns_plateau",
    "vitvs_dinov3": "mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/cns_plateau",
    "vitvs_dinov2": "mutual_nn_seeded_subsample@dinov2_vits14/k24/ur/uniform/cns_plateau",
    "artvs": "artvs@dinov2_vits14/k24/ur/uniform/cns_plateau",
    "artvs_pre4way": "artvs@dinov2_vits14/k24/ur/uniform/pre4way/cns_plateau",
    "vitvs_dinov3_objmask": "mutual_nn_seeded_subsample@dinov3_vits16/k24/objmask/ur/uniform/cns_plateau",  # noqa: E501 - a label, verbatim
    "artvs_objmask": "artvs@dinov2_vits14/k24/objmask/ur/uniform/cns_plateau",
    "sift_ransac_objmask": "classical_sift/k24/objmask/ur/uniform/ransac-scene_default/cns_plateau",
    "visp_ibvs_sift_objmask": "classical_sift/k24/objmask/ur/uniform/visp-current/ransac-scene_default/cns_plateau",  # noqa: E501 - a label, verbatim
    "cns_v1_objmask": "cns_v1_sift/x2.4/objmask/cns_plateau",
    "lightglue_ransac_objmask": "lightglue/k24/objmask/ur/uniform/ransac-scene_default/cns_plateau",
    "roma_ibvs_objmask": "roma/k24/objmask/ur/uniform/ransac-scene_default/cns_plateau",
    "romav2_ibvs_objmask": "romav2/k24/objmask/ur/uniform/ransac-scene_default/cns_plateau",
    "artvs_roi": "artvs_roi@dinov2_vits14/k24/ur/uniform/cns_plateau",
    "artvs_roi_scene": "artvs_roi@dinov2_vits14/k24/prompt-scene/ur/uniform/cns_plateau",
    "vitvs_dinov3_pre4way_objmask": "mutual_nn_seeded_subsample@dinov3_vits16/k24/objmask/ur/uniform/pre4way/cns_plateau",  # noqa: E501 - a label, verbatim
    "artvs_pre4way_objmask": "artvs@dinov2_vits14/k24/objmask/ur/uniform/pre4way/cns_plateau",
    "artvs_roi_pre4way": "artvs_roi@dinov2_vits14/k24/ur/uniform/pre4way/cns_plateau",
    "vitvs_dinov3_pre4way": "mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/pre4way/cns_plateau",  # noqa: E501 - a label, verbatim
    "lightglue_ransac": "lightglue/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "lightglue_noransac": "lightglue/k24/ur/uniform/ransac-none/cns_plateau",
    "roma_ibvs": "roma/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "roma_ibvs_noransac": "roma/k24/ur/uniform/ransac-none/cns_plateau",
    "romav2_ibvs": "romav2/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "romav2_ibvs_noransac": "romav2/k24/ur/uniform/ransac-none/cns_plateau",
    "visp_ibvs_dinov3": "mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/visp-current/cns_plateau",  # noqa: E501 - a label, verbatim
    "visp_ibvs_dinov3_mean": "mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/visp-mean/cns_plateau",  # noqa: E501 - a label, verbatim
    "visp_ibvs_sift": "classical_sift/k24/ur/uniform/visp-current/ransac-scene_default/cns_plateau",
    "dinobot": "dinobot_vits8/k8/jump1/cns_plateau",
    "dinobot_objmask": "dinobot_vits8/k8/jump1/objmask/cns_plateau",
    "dinobot_ibvs": "dinobot_corr@dino_vits8/k8/ur/uniform/cns_plateau",
    "dinobot_reffix": "dinobot_vits8/k8/jump1/reffix/cns_plateau",
    "cns_v1": "cns_v1_sift/x2.4/cns_plateau",
}


def test_existing_protocol_cells_keep_their_labels(fly, protocol):
    for cell, label in CELL_LABELS_ON_MAIN.items():
        assert arm_label(fly.arm_config(protocol, cell)) == label, cell
    new = {e["id"] for e in protocol["arms"]} - set(CELL_LABELS_ON_MAIN)
    assert new == {"vitvs_dinov3_objmask_ema08", "artvs_roi_sim", *CELLS_ADDED_2026_10_10}


#: The two cells added on 2026-10-05, as `main` at 5c9ebf6 derived them - before the faithful
#: arm's SAM path (`roi_sam_path`, 2026-10-06). The 2026-10-05 smoke's `artvs_roi_sim` records
#: carry the second (transformers' SAM).
CELL_LABELS_ADDED_2026_10_05 = {
    "vitvs_dinov3_objmask_ema08": "mutual_nn_seeded_subsample@dinov3_vits16/k24/objmask/ur/"
                                  "uniform/ema0.8/cns_plateau",
    "artvs_roi_sim": "artvs_roi_sim@dinov3_vits16_artvs256/k24/prompt-scene/lsmask/tr/uniform/"
                     "ema0.8/vcap10/hold/cns_plateau",
}


#: The cell added on 2026-10-10 (the development evaluation, owner rulings): the faithful ART-VS
#: cell with the simulator's outline as its goal mask and the switch by step 100.
CELLS_ADDED_2026_10_10 = {
    "artvs_roi_sim_dev": "artvs_roi_sim@dinov3_vits16_artvs256/k24/prompt-scene/objmask/"
                         "switch-by-step100/tr/uniform/ema0.8/vcap10/hold/cns_plateau",
}


def test_the_sam_path_moves_only_the_faithful_cell_by_one_token(fly, protocol):
    """2026-10-06 (the LangSAM check): every protocol cell keeps the label it had on main at
    5c9ebf6, except `artvs_roi_sim`, which gains `sam-langsam` after `lsmask` - so its records
    from before (the 2026-10-05 smoke) are told apart by label. No other cell carries a
    `sam-` token, and the cell's record field says the same."""
    before = {**CELL_LABELS_ON_MAIN, **CELL_LABELS_ADDED_2026_10_05}
    assert set(before) == {e["id"] for e in protocol["arms"]} - set(CELLS_ADDED_2026_10_10)
    for cell, label in before.items():
        now = arm_label(fly.arm_config(protocol, cell))
        if cell == "artvs_roi_sim":
            assert now == label.replace("/lsmask/", "/lsmask/sam-langsam/") != label
        else:
            assert now == label and "sam-" not in now, cell
    cfg = fly.arm_config(protocol, "artvs_roi_sim")
    assert cfg["matcher"]["roi_sam_path"] == "langsam"
    specs, _ = load_scenes()
    arm = build_arm({**cfg, "matcher": {**cfg["matcher"], "device": "cpu"}}, intrinsics=D435I,
                    scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    assert arm.record_fields()["roi_sam_path"] == "langsam"
    assert arm.matcher.tracker.detector.sam_path == "langsam"
    old = fly.arm_config(protocol, "artvs_roi_scene")             # the "as ported" cell
    old_arm = build_arm({**old, "matcher": {**old["matcher"], "device": "cpu"}},
                        intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubTiles())
    assert old_arm.matcher.tracker.detector.sam_path == "transformers"
    assert "roi_sam_path" not in old_arm.record_fields()       # its record, key for key


def test_the_vitvs_smoothing_cell_is_the_objmask_cell_plus_the_published_ema(fly, protocol):
    """Owner ruling 2026-10-05: ONE cell beside the instance suite's ViT-VS cell, the same in
    everything but `controller.ema_alpha: 0.8` (the release's
    config_dinov2_308_no_tiling.yaml:39) - and its label says so."""
    old = fly.arm_config(protocol, "vitvs_dinov3_objmask")
    new = fly.arm_config(protocol, "vitvs_dinov3_objmask_ema08")
    assert new["controller"]["ema_alpha"] == 0.8 and old["controller"]["ema_alpha"] == 0.0
    strip = ("_source", "controller")
    assert {k: v for k, v in new.items() if k not in strip} == \
        {k: v for k, v in old.items() if k not in strip}
    assert {k: v for k, v in new["controller"].items() if k != "ema_alpha"} == \
        {k: v for k, v in old["controller"].items() if k != "ema_alpha"}
    assert arm_label(new) == arm_label(old).replace("/uniform/", "/uniform/ema0.8/")
    assert arm_label(new) == ("mutual_nn_seeded_subsample@dinov3_vits16/k24/objmask/ur/"
                              "uniform/ema0.8/cns_plateau")
    cfg = dict(new, matcher={**new["matcher"], "device": "cpu"})
    arm = build_arm(cfg, intrinsics=D435I, backbone=_Stub())
    assert arm.controller.ema_alpha == 0.8 and arm.name == arm_label(new)


def test_the_artvs_dev_cell_is_the_faithful_cell_plus_the_two_rulings_of_2026_10_10(fly, protocol):
    """Owner 2026-10-10: ONE cell beside `artvs_roi_sim`, the same in everything but the goal
    mask (the simulator's outline through the release's branch, no SAM) and the switch (the
    release's rule or step 100, whichever first) - and its label says both; the faithful cell's
    label, config and record are untouched."""
    old = fly.arm_config(protocol, "artvs_roi_sim")
    new = fly.arm_config(protocol, "artvs_roi_sim_dev")
    assert {k: v for k, v in new.items() if k not in ("_source", "matcher")} == \
        {k: v for k, v in old.items() if k not in ("_source", "matcher")}
    changed = {k for k in set(old["matcher"]) | set(new["matcher"])
               if old["matcher"].get(k) != new["matcher"].get(k)}
    assert changed == {"goal_mask", "roi_mask", "roi_sam_path", "switch_max_steps"}
    assert (new["matcher"]["goal_mask"], new["matcher"]["roi_mask"],
            new["matcher"]["roi_sam_path"], new["matcher"]["switch_max_steps"]) == \
        ("objects", "objects", None, 100)
    assert arm_label(new) == CELLS_ADDED_2026_10_10["artvs_roi_sim_dev"]
    assert arm_label(old) == CELL_LABELS_ADDED_2026_10_05["artvs_roi_sim"].replace(
        "/lsmask/", "/lsmask/sam-langsam/")
    specs, _ = load_scenes()
    arm = build_arm({**new, "matcher": {**new["matcher"], "device": "cpu"}}, intrinsics=D435I,
                    scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    assert arm.goal_mask_mode == "objects"          # the runner hands it the outline
    assert arm.matcher.requires_goal_mask and arm.matcher.honours_goal_mask
    assert arm.matcher.tracker.goal_mask_wanted is False     # SAM is never asked for
    rec = arm.record_fields()
    assert rec["roi_mask"] == "objects" and rec["roi_sam_path"] is None
    assert rec["roi_switch_max_steps"] == 100 and rec["roi_switch_reason"] is None
    assert "SAM" not in rec["roi_detector"].split("(")[0]
    faithful = build_arm({**old, "matcher": {**old["matcher"], "device": "cpu"}},
                         intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    frec = faithful.record_fields()
    assert "roi_switch_max_steps" not in frec and "roi_switch_reason" not in frec
    assert faithful.matcher.requires_goal_mask is False and faithful.goal_mask_mode == "none"
    # the two keys go together: the mask from the bench without asking the runner for it, or
    # the runner's mask on the LangSAM cell, are both refused at build
    with pytest.raises(ValueError, match="requires|takes its goal mask from the bench"):
        build_arm({**new, "matcher": {**{k: v for k, v in new["matcher"].items()
                                         if k != "goal_mask"}, "device": "cpu"}},
                  intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    with pytest.raises(ValueError, match="roi_sam_path is 'langsam' under roi_mask 'objects'"):
        build_arm({**new, "matcher": {**new["matcher"], "device": "cpu",
                                      "roi_sam_path": "langsam"}},
                  intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
