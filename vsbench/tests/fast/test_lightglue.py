# vs-sim copy: tests/fast/test_lightglue.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""SuperPoint + LightGlue as a matcher: the seam with stub models, the real package where it is."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vsbench.vs.baselines import lightglue as lg  # noqa: E402
from vsbench.vs.baselines.match import Matches  # noqa: E402
from vsbench.vs.bench import arms  # noqa: E402
from vsbench.vs.bench import config as cfgmod  # noqa: E402
from vsbench.vs.scene import D435I  # noqa: E402

ARMS = Path(__file__).resolve().parents[2] / "configs" / "arms"


class _Extractor:
    """Returns 6 fixed keypoints per image (batched, as the package does)."""

    def extract(self, image, resize=None):
        self.resize = resize
        k = torch.tensor([[[100., 100.], [500., 300.], [900., 600.], [1300., 200.],
                           [1700., 900.], [300., 800.]]])
        return {"keypoints": k, "descriptors": torch.zeros(1, 6, 8)}


class _Matcher:
    """Pairs keypoint i with i, confidence decreasing with i; drops the last pair."""

    def __call__(self, data):
        return {"matches": torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]]),
                "scores": torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])}


def _rgb():
    return np.zeros((1080, 1920, 3), np.uint8)


def test_the_stub_seam_gives_pixel_lattice_matches_best_k_by_confidence():
    m = lg.LightGlueMatcher(num_landmarks=3, device="cpu", models=(_Extractor(), _Matcher()))
    m.set_episode("hollywood", "t0")
    m.reset(_rgb())
    out = m.match(_rgb())
    assert isinstance(out, Matches) and len(out) == 3
    rows = out.to_rows()
    np.testing.assert_allclose(rows["pts_goal"], [[100, 100], [500, 300], [900, 600]], atol=1e-9)
    np.testing.assert_allclose(rows["pts_cur"], rows["pts_goal"], atol=1e-9)   # identity pairing
    assert out.patch_size == 1 and out.side == 1920
    assert out.extras["n_raw"] == 5 and out.extras["n_keypoints_goal"] == 6
    assert out.extras["ransac_model"] == "none" and out.extras["ransac_applied"] is False
    fields = m.record_fields()
    assert fields["lg_max_keypoints"] == 2048 and fields["lg_resize"] == 1024
    assert fields["front_end"] == "superpoint_lightglue"
    # every match when num_landmarks is 0
    m0 = lg.LightGlueMatcher(num_landmarks=0, device="cpu", models=(_Extractor(), _Matcher()))
    m0.reset(_rgb())
    assert len(m0.match(_rgb())) == 5


def test_ransac_needs_the_scene_for_its_model_and_unknown_settings_are_refused():
    with pytest.raises(ValueError, match="unknown LightGlue settings"):
        lg.LightGlueMatcher(num_landmarks=24, device="cpu", nms=4)
    with pytest.raises(ValueError, match="scene_default"):
        lg.LightGlueMatcher(num_landmarks=24, device="cpu", ransac="scene_default")


def test_the_arm_file_builds_with_stub_models_and_is_labelled():
    cfg = cfgmod.load(ARMS / "lightglue.yaml")
    cfg["matcher"]["device"] = "cpu"
    m = lg.LightGlueMatcher(num_landmarks=24, device="cpu", ransac="none",
                            models=(_Extractor(), _Matcher()))
    arm = arms.build_arm(cfg, intrinsics=D435I, matcher=m)
    assert arm.name == "lightglue/k24/ur/uniform/ransac-scene_default/cns_plateau"
    assert arm.matcher_name == "superpoint_lightglue"
    with pytest.raises(cfgmod.ConfigError, match="matcher.device"):
        cfgmod.validate({**cfg, "matcher": {k: v for k, v in cfg["matcher"].items()
                                            if k != "device"}})


def test_the_real_package_where_installed():
    ok, why = lg.lightglue_available()
    if not ok:
        pytest.skip(why)
    m = lg.LightGlueMatcher(num_landmarks=24, device="cpu")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(270, 480, 3), dtype=np.uint8)
    m.reset(img)
    out = m.match(img)
    assert len(out) > 0
    rows = out.to_rows()
    np.testing.assert_allclose(rows["pts_goal"], rows["pts_cur"], atol=1.0)
    assert m.record_fields()["lg_weights_md5"]
