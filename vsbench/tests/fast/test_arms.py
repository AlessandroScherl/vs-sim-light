# vs-sim copy: tests/fast/test_arms.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The arm registry: a config builds a method, and nothing parses a name.

Every arm file under ``configs/arms/`` is the whole surface: each builds with a
weight-free stub encoder (or a stub CNS pipeline, or cv2), derives its label from its
settings, and records what ran. The research repository's blocks are refused by name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vsbench.vs.baselines.classical import ClassicalMatcher
from vsbench.vs.baselines.cns import CNSArm, build_cns_arm
from vsbench.vs.baselines.control import CnsPlateau, IBVSController
from vsbench.vs.baselines.match import DenseMatcher
from vsbench.vs.baselines.tiling import TiledMatcher
from vsbench.vs.bench import arms
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.bench.truth import PlaneTruthMatcher
from vsbench.vs.scene import D435I, MockPlaneScene
from vsbench.vs.scenes import load_scenes

ARMS = Path(__file__).resolve().parents[2] / "configs" / "arms"


class _StubBackbone:
    """A weight-free stand-in so the fast tier can build a real DenseMatcher."""

    name = "dinov3_vits16"
    input_size = 512
    patch_size = 16
    grid_side = 32
    dim = 384
    device = "cpu"

    def encode(self, frame_rgb_u8):  # pragma: no cover - never called here
        raise AssertionError("this test builds arms, it does not run them")


class _StubDinoV2(_StubBackbone):
    name = "dinov2_vits14"
    input_size = 518
    patch_size = 14
    grid_side = 37
    dim = 3456


class _StubTileBackbone(_StubDinoV2):
    """ART-VS reads every tile at its `vit_input_size` (224 = 16 x 14)."""

    input_size = 224
    grid_side = 16


def _load(name: str, **over) -> dict:
    cfg = cfgmod.load(ARMS / name)
    cfg["matcher"] = {**cfg["matcher"], **over}
    return cfg


@pytest.fixture(scope="module")
def specs():
    return load_scenes()[0]


# ------------------------------------------------------------------ the arm files
def test_every_arm_file_builds_and_states_its_device():
    """The five files are the published baselines; each states the encoder's device."""
    for path in sorted(ARMS.glob("*.yaml")):
        cfg = cfgmod.load(path)
        assert cfg["stop"]["rule"] == "cns_plateau", path.name
        assert cfg["episode"]["max_iter"] == 600, path.name
        assert set(cfg["scoring"]["success_rule"]) == set(cfgmod.SUCCESS_RULES), path.name
        if cfg["matcher"]["variant"] in cfgmod.DEVICE_VARIANTS:
            assert cfg["matcher"]["device"] == "cuda", path.name
    assert {p.name for p in ARMS.glob("*.yaml")} == {
        "artvs.yaml", "classical.yaml", "cns_v1.yaml", "vitvs_dinov2.yaml", "vitvs_dinov3.yaml",
        "visp_ibvs.yaml", "visp_ibvs_sift.yaml", "lightglue.yaml", "roma.yaml", "artvs_roi.yaml",
        "dinobot.yaml", "dinobot_ibvs.yaml", "artvs_roi_sim.yaml"}


def test_vitvs_dinov3_is_a_dense_matcher_on_the_dinov3_backbone():
    cfg = _load("vitvs_dinov3.yaml", device="cpu")
    arm = arms.build_arm(cfg, intrinsics=D435I, backbone=_StubBackbone())
    assert isinstance(arm.matcher, DenseMatcher)
    assert isinstance(arm.controller, IBVSController) and isinstance(arm.stop, CnsPlateau)
    assert arm.name == "mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/cns_plateau"
    assert arm.kind == "method" and arm.checkpoint is None
    w = arm.controller.weighting
    assert (w.use_trust, w.use_margin, w.per_axis) == (False, False, False)   # as published
    assert arm.record_fields()["backbone"] == "dinov3_vits16"
    assert arm.record_fields()["rotation_mode"] == "none"


def test_vitvs_dinov2_is_the_same_matcher_on_the_published_chain():
    cfg = _load("vitvs_dinov2.yaml", device="cpu")
    arm = arms.build_arm(cfg, intrinsics=D435I, backbone=_StubDinoV2())
    assert isinstance(arm.matcher, DenseMatcher)
    assert arm.name == "mutual_nn_seeded_subsample@dinov2_vits14/k24/ur/uniform/cns_plateau"
    assert arm.record_fields()["backbone"] == "dinov2_vits14"


def test_artvs_is_the_tiled_matcher_with_the_published_tiling():
    cfg = _load("artvs.yaml", device="cpu")
    arm = arms.build_arm(cfg, intrinsics=D435I, backbone=_StubTileBackbone())
    assert isinstance(arm.matcher, TiledMatcher)
    assert arm.name == "artvs@dinov2_vits14/k24/ur/uniform/cns_plateau"
    fields = arm.record_fields()
    assert fields["artvs_tiles"] == 36 and fields["artvs_switch_threshold"] == 0.20
    assert fields["prealign"] is None and fields["rotation_mode"] == "none"
    pre = arms.build_arm(dict(cfg, rotation={"mode": "prealign_4way"}), intrinsics=D435I,
                         backbone=_StubTileBackbone())
    assert pre.name == "artvs@dinov2_vits14/k24/ur/uniform/pre4way/cns_plateau"
    assert pre.record_fields()["rotation_mode"] == "prealign_4way"


def test_the_classical_arm_needs_the_scene_for_its_ransac_model(specs):
    cfg = _load("classical.yaml")
    with pytest.raises(ValueError, match="scene_default"):
        arms.build_arm(cfg, intrinsics=D435I)
    arm = arms.build_arm(cfg, intrinsics=D435I, scene_spec=specs["hollywood"])
    assert isinstance(arm.matcher, ClassicalMatcher)
    assert arm.name == "classical_sift/k24/ur/uniform/ransac-scene_default/cns_plateau"
    assert arm.record_fields()["ransac_model"] == "homography"
    room = arms.build_arm(cfg, intrinsics=D435I, scene_spec=specs["cluttered2"])
    assert room.record_fields()["ransac_model"] == "fundamental"
    off = arms.build_arm(_load("classical.yaml", ransac="none"), intrinsics=D435I)
    assert off.name.endswith("/ransac-none/cns_plateau")
    for variant in ("classical_orb", "classical_akaze"):
        a = arms.build_arm(_load("classical.yaml", variant=variant, ransac="none"),
                           intrinsics=D435I)
        assert a.name.startswith(variant)


def test_a_weighted_controller_is_refused_on_an_unweighted_baseline():
    """The published classical and ART-VS chains carry no usable margin or trust."""
    cfg = _load("classical.yaml", ransac="none")
    cfg["controller"]["weighting"]["use_margin"] = True
    with pytest.raises(ValueError, match="use_margin"):
        arms.build_arm(cfg, intrinsics=D435I)
    cfg["controller"]["weighting"]["use_margin"] = False
    cfg["controller"]["weighting"]["use_trust"] = True
    with pytest.raises(ValueError, match="use_trust"):
        arms.build_arm(cfg, intrinsics=D435I)


class _StubPipeline:
    def set_target(self, bgr, dist_scale=None, intrinsic=None):
        self.target = bgr

    def get_control_rate(self, bgr):
        return np.zeros(6), {}


def test_cns_v1_is_an_arm_not_a_matcher():
    cfg = cfgmod.load(ARMS / "cns_v1.yaml")
    with pytest.raises(ValueError, match="whole pipeline"):
        arms.build_arm(cfg, intrinsics=D435I, backbone=_StubBackbone())
    arm = build_cns_arm(cfg, intrinsics=D435I, pipeline=_StubPipeline())
    assert isinstance(arm, CNSArm)
    fields = arm.record_fields()
    assert fields["cns_step_gain"] == 2.0 and fields["cns_rate_compensate"] is True
    assert fields["cns_step_scale"] == pytest.approx(2.4)
    assert arms.arm_label(cfg) == arm.name == "cns_v1_sift/x2.4/cns_plateau"


# ------------------------------------------------------------- the refusals
def test_the_research_repositorys_blocks_are_refused_by_name():
    cfg = _load("vitvs_dinov3.yaml", device="cpu")
    for block, value in (("refine", {"checkpoint": "x.pt", "tag": "v2cm"}),
                         ("oracle", {"mode": "grid"}),
                         ("head", {"arch": "v1_1"})):
        with pytest.raises(ValueError, match=f"config block '{block}'"):
            arms.build_arm(dict(cfg, **{block: value}), intrinsics=D435I,
                           backbone=_StubBackbone())
    with pytest.raises(ValueError, match="world="):
        arms.build_arm(cfg, intrinsics=D435I, backbone=_StubBackbone(), world=object())


@pytest.mark.parametrize("field", ["matcher.variant", "matcher.num_landmarks", "seed",
                                   "stop.rule", "scoring.success_rule"])
def test_a_config_missing_a_required_field_fails_in_the_builder(field):
    broken = _load("vitvs_dinov3.yaml", device="cpu")
    head, _, tail = field.partition(".")
    (broken[head].pop(tail) if tail else broken.pop(head))
    with pytest.raises(cfgmod.ConfigError, match=field):
        arms.build_arm(broken, intrinsics=D435I, backbone=_StubBackbone())


def test_an_injected_matcher_must_be_what_the_variant_says_or_declare_itself():
    """A cell may not be labelled as something it did not run: injecting a ground-truth
    matcher against `mutual_nn_seeded_subsample` is refused unless the arm declares
    `kind: oracle` or `kind: test`, which then travels into the record and the label."""
    cfg = _load("vitvs_dinov3.yaml", device="cpu")
    truth = PlaneTruthMatcher(MockPlaneScene(), num_landmarks=8, seed=1)
    with pytest.raises(ValueError, match="was injected"):
        arms.build_arm(cfg, intrinsics=D435I, matcher=truth)
    for kind in ("oracle", "test"):
        arm = arms.build_arm(dict(cfg, arm={"kind": kind}), intrinsics=D435I, matcher=truth)
        assert arm.kind == kind
        assert arm.matcher_name == "plane_truth" != arm.config["matcher"]["variant"]
        assert arm.name.startswith(f"{kind}:")
    with pytest.raises(ValueError, match="arm.kind"):
        arms.build_arm(dict(cfg, arm={"kind": "sneaky"}), intrinsics=D435I, matcher=truth)
    with pytest.raises(ValueError, match="silently ignored"):
        arms.build_arm(dict(cfg, arm={"kind": "oracle"}), intrinsics=D435I, matcher=truth,
                       tie_order="legacy")
    real = arms.build_arm(cfg, intrinsics=D435I, backbone=_StubBackbone())
    assert real.kind == "method" and real.matcher_name == "dense_dinov3_vits16_mutual_nn"


# ------------------------------------------------------------- the episode block
def test_the_episode_block_holds_the_budget_not_the_stop_block():
    cfg = _load("vitvs_dinov3.yaml", device="cpu")
    s = arms.episode_settings(cfg)
    assert (s["max_iter"], s["divergence_abort"]) == (600, True)
    assert s["record_stops"] == ["velocity_quiet", "dec_plateau"]
    assert cfg["stop"]["rule"] not in s["record_stops"]      # never also recorded
    assert arms.episode_settings(dict(cfg, episode={"max_iter": 10}))["record_stops"] == [
        "velocity_quiet"]                                     # recorded even unasked
    with pytest.raises(ValueError, match="unknown episode keys"):
        arms.episode_settings(dict(cfg, episode={"max_iters": 600}))
    agreeing = dict(cfg, episode={"max_iter": 10, "dt": 1.0 / 60.0})
    assert arms.episode_settings(agreeing)["dt"] == pytest.approx(1.0 / 60.0)
    with pytest.raises(ValueError, match="episode.dt"):
        arms.episode_settings(dict(cfg, episode={"max_iter": 10, "dt": 0.5}))


def test_the_record_of_the_stop_rule_is_its_effective_constants():
    cns = arms.build_arm(_load("vitvs_dinov3.yaml", device="cpu"), intrinsics=D435I,
                         backbone=_StubBackbone()).stop
    assert cns.params["waiting_time"] == 0.8 and cns.params["conduct_thresh"] == 0.01


# ------------------------------------------------------------ the image seam
def test_the_depth_lookup_truncates_and_never_rounds():
    """(100.4, 200.6) and (100.6, 200.4) must read the SAME pixel."""
    depth = np.arange(300 * 300, dtype=np.float64).reshape(300, 300) * 1e-5 + 0.3
    a = arms.depth_at(depth, np.array([[100.4, 200.6]]))
    b = arms.depth_at(depth, np.array([[100.6, 200.4]]))
    assert a == b == pytest.approx(depth[200, 100])
    assert a != pytest.approx(depth[201, 101])          # rounding would give this


def test_the_depth_lookup_clamps_to_the_image_and_cleans_invalid_samples():
    depth = np.full((8, 8), 0.4)
    depth[0, 0] = -1.0                                   # invalid: z <= 0
    depth[7, 7] = 99.0                                   # invalid: z > 10 m
    z = arms.depth_at(depth, np.array([[-5.0, -5.0], [900.0, 900.0], [3.0, 3.0]]))
    assert list(z) == [0.5, 0.5, 0.4]                    # DEPTH_FALLBACK_M twice
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        arms.depth_at(np.zeros((4, 4, 3)), np.zeros((1, 2)))

# ------------------------------------------------- additivity (2026-10-05, spec 4.9)
#: Every arm file's derived label on `main` before the faithful ART-VS arm was added
#: (commit 4271faa, computed there with `arm_label(load(path))`). The knobs of 2026-10-05
#: (`tr`, `ema`, `vcap`, `hold`, `lsmask`) are absent from all twelve, so not one may move.
LABELS_ON_MAIN = {
    "artvs.yaml": "artvs@dinov2_vits14/k24/ur/uniform/cns_plateau",
    "artvs_roi.yaml": "artvs_roi@dinov2_vits14/k24/ur/uniform/cns_plateau",
    "classical.yaml": "classical_sift/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "cns_v1.yaml": "cns_v1_sift/x2.4/cns_plateau",
    "dinobot.yaml": "dinobot_vits8/k8/jump1/cns_plateau",
    "dinobot_ibvs.yaml": "dinobot_corr@dino_vits8/k8/ur/uniform/cns_plateau",
    "lightglue.yaml": "lightglue/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "roma.yaml": "roma/k24/ur/uniform/ransac-scene_default/cns_plateau",
    "visp_ibvs.yaml": "mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/visp-current/cns_plateau",  # noqa: E501 - a label, verbatim
    "visp_ibvs_sift.yaml": "classical_sift/k24/ur/uniform/visp-current/ransac-scene_default/cns_plateau",  # noqa: E501 - a label, verbatim
    "vitvs_dinov2.yaml": "mutual_nn_seeded_subsample@dinov2_vits14/k24/ur/uniform/cns_plateau",
    "vitvs_dinov3.yaml": "mutual_nn_seeded_subsample@dinov3_vits16/k24/ur/uniform/cns_plateau",
}


def test_existing_arm_labels_are_unchanged():
    for name, label in LABELS_ON_MAIN.items():
        assert arms.arm_label(cfgmod.load(ARMS / name)) == label, name
    assert len(LABELS_ON_MAIN) == 12


class _StubSim256(_StubBackbone):
    """The faithful ART-VS arm's encoder geometry: 256 px, 16 x 16 cells, 9 x 384 bins."""

    name = "dinov3_vits16_artvs256"
    input_size = 256
    patch_size = 16
    grid_side = 16
    dim = 3456


def test_the_faithful_artvs_arm_file_builds_and_states_its_settings(specs):
    """`configs/arms/artvs_roi_sim.yaml` (spec 4.6, with the owner's evening rulings): the
    label is derived from its settings, the controller carries the four knobs, and the
    record states them; the matcher refuses the old port's mask rule and a mismatched
    encoder input."""
    from vsbench.vs.baselines.objroi import RoiTiledMatcherSim

    cfg = _load("artvs_roi_sim.yaml", device="cpu")
    arm = arms.build_arm(cfg, intrinsics=D435I, scene_spec=specs["inst_shoe"],
                         backbone=_StubSim256())
    assert isinstance(arm.matcher, RoiTiledMatcherSim)
    assert arm.name == ("artvs_roi_sim@dinov3_vits16_artvs256/k24/prompt-scene/lsmask/"
                        "sam-langsam/tr/uniform/ema0.8/vcap10/hold/cns_plateau")
    c = arm.controller
    assert (c.pixel_quantisation, c.max_velocity, c.degenerate_policy, c.depth_fallback_m,
            c.ema_alpha, c.solve_unrounded) == ("truncate", 10.0, "hold", 100.0, 0.8, False)
    rec = arm.record_fields()
    assert (rec["pixel_quantisation"], rec["max_velocity"], rec["degenerate_policy"],
            rec["depth_fallback_m"]) == ("truncate", 10.0, "hold", 100.0)
    assert rec["roi_prompt"] == "shoe" and rec["roi_mask"] == "langsam"
    assert rec["roi_sam_path"] == "langsam"                       # 2026-10-06, the LangSAM check
    assert arm.matcher.tracker.detector.sam_path == "langsam"     # the GroundingSam it built
    assert rec["roi_backbone"] == "dinov3_vits16_artvs256" and rec["roi_grid_from"] == "unpadded"
    assert arm.matcher.tracker.goal_mask_wanted is True
    assert "artvs_visual90" in arms.episode_settings(cfg)["record_stops"]
    with pytest.raises(ValueError, match="old port's mask rule"):
        arms.build_arm(_load("artvs_roi_sim.yaml", device="cpu", use_sam_mask=True),
                       intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    with pytest.raises(ValueError, match="only value is 'langsam'"):
        arms.build_arm(_load("artvs_roi_sim.yaml", device="cpu", roi_mask="none"),
                       intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    with pytest.raises(ValueError, match="roi_sam_path is 'transformers'"):
        arms.build_arm(_load("artvs_roi_sim.yaml", device="cpu", roi_sam_path="transformers"),
                       intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    with pytest.raises(ValueError, match="vit_input_size 224"):
        arms.build_arm(_load("artvs_roi_sim.yaml", device="cpu", vit_input_size=224),
                       intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    with pytest.raises(ValueError, match="D-A"):
        arms.build_arm(_load("artvs_roi_sim.yaml", device="cpu", auto_calculate_tiles=False),
                       intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    with pytest.raises(TypeError):        # a misspelt key reaches the constructor and fails
        arms.build_arm(_load("artvs_roi_sim.yaml", device="cpu", switch_min_frame=10),
                       intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())
    with pytest.raises(ValueError, match="ignores reset"):   # the bench's mask is refused
        arms.build_arm(_load("artvs_roi_sim.yaml", device="cpu", goal_mask="objects"),
                       intrinsics=D435I, scene_spec=specs["inst_shoe"], backbone=_StubSim256())


# ---------------------------------------- additivity (2026-10-06, the LangSAM check)
#: Every arm file's derived label on `main` at 5c9ebf6, before the faithful arm's SAM path
#: (`roi_sam_path`) existed - `arm_label(load(path))` there. The 2026-10-05 smoke's
#: `artvs_roi_sim` records carry the last one (transformers' SAM).
LABELS_ON_MAIN_5C9EBF6 = {
    **LABELS_ON_MAIN,
    "artvs_roi_sim.yaml": "artvs_roi_sim@dinov3_vits16_artvs256/k24/prompt-scene/lsmask/tr/"
                          "uniform/ema0.8/vcap10/hold/cns_plateau",
}


def test_the_sam_path_moves_the_faithful_arms_label_by_one_token_and_no_other_label():
    """`roi_sam_path: langsam` (SAM as the lang_sam package runs it) puts `sam-langsam` after
    `lsmask` in the faithful arm's label, so records flown before - the 2026-10-05 smoke -
    are told apart by label; every other arm file keeps its label. A sim config stating
    another SAM path would derive yet another label, and is refused at build."""
    files = {p.name for p in ARMS.glob("*.yaml")}
    assert set(LABELS_ON_MAIN_5C9EBF6) == files and len(files) == 13
    for name, label in LABELS_ON_MAIN_5C9EBF6.items():
        now = arms.arm_label(cfgmod.load(ARMS / name))
        if name == "artvs_roi_sim.yaml":
            assert now != label
            assert now == label.replace("/lsmask/", "/lsmask/sam-langsam/")
            assert now.split("/").count("sam-langsam") == 1
        else:
            assert now == label, name
            assert "sam-" not in now, name
    other = _load("artvs_roi_sim.yaml", roi_sam_path="transformers")
    assert "/lsmask/sam-transformers/" in arms.arm_label(other)
    assert cfgmod.load(ARMS / "artvs_roi_sim.yaml")["matcher"]["roi_sam_path"] == "langsam"
    for name in ("artvs_roi.yaml", "vitvs_dinov2.yaml", "vitvs_dinov3.yaml", "dinobot.yaml",
                 "dinobot_ibvs.yaml"):
        # the old object cell keeps transformers' SAM (it states no path); ViT-VS and DINOBot
        # run no SAM at all
        assert "roi_sam_path" not in cfgmod.load(ARMS / name)["matcher"], name
