# vs-sim copy: tests/fast/test_roma.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""RoMa / RoMa v2 as matchers: the seam with a stub model (the real packages are station-only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vsbench.vs.baselines import roma as rm  # noqa: E402
from vsbench.vs.bench import arms  # noqa: E402
from vsbench.vs.bench import config as cfgmod  # noqa: E402
from vsbench.vs.scene import D435I  # noqa: E402

ARMS = Path(__file__).resolve().parents[2] / "configs" / "arms"
W, H = 1920, 1080


class _StubV1:
    """A RoMa-shaped model whose warp is the identity shifted by 10 px right, certainty 0.9
    everywhere except a low-certainty band on the left quarter."""

    def match(self, a, b, device=None):
        hh, ww = 64, 64          # RoMa's grid: pixel centres, align_corners=False
        ys, xs = torch.meshgrid(torch.linspace(-1 + 1 / hh, 1 - 1 / hh, hh),
                                torch.linspace(-1 + 1 / ww, 1 - 1 / ww, ww), indexing="ij")
        shift = 10.0 / W * 2
        warp = torch.stack([xs, ys, xs + shift, ys], -1)
        cert = torch.full((hh, ww), 0.9)
        cert[:, : ww // 4] = 0.1
        return warp, cert

    def sample(self, warp, certainty, num):
        flat = warp.reshape(-1, 4)
        c = certainty.reshape(-1)
        idx = torch.multinomial(c, num, replacement=False)
        return flat[idx], c[idx]

    @staticmethod
    def to_pixel_coordinates(matches, H_A, W_A, H_B, W_B):
        a = torch.stack([(matches[:, 0] + 1) / 2 * W_A, (matches[:, 1] + 1) / 2 * H_A], -1)
        b = torch.stack([(matches[:, 2] + 1) / 2 * W_B, (matches[:, 3] + 1) / 2 * H_B], -1)
        return a, b


def _rgb():
    return np.zeros((H, W, 3), np.uint8)


def test_sample_mode_is_seeded_per_frame_and_carries_the_certainty():
    m = rm.RoMaMatcher(version=1, num_landmarks=24, device="cpu", seed=5, model=_StubV1())
    m.set_episode("hollywood", "t0")
    m.reset(_rgb())
    a = m.match(_rgb()).to_rows()
    m.reset(_rgb())
    b = m.match(_rgb()).to_rows()
    np.testing.assert_array_equal(a["pts_goal"], b["pts_goal"])       # same frame, same draw
    c = m.match(_rgb()).to_rows()
    assert not np.array_equal(a["pts_goal"], c["pts_goal"])           # the next frame re-draws
    np.testing.assert_allclose(a["pts_cur"][:, 0] - a["pts_goal"][:, 0], 10.0, atol=1e-3)
    assert len(a["pts_goal"]) == 24 and m.name == "roma_v1"


def test_pinned_mode_keeps_the_same_goal_pixels_and_reads_the_warp_there():
    m = rm.RoMaMatcher(version=1, num_landmarks=8, device="cpu", seed=1, sampling="pinned",
                       model=_StubV1())
    m.set_episode("hollywood", "t0")
    m.reset(_rgb())
    a = m.match(_rgb()).to_rows()
    b = m.match(_rgb()).to_rows()
    np.testing.assert_array_equal(a["pts_goal"], b["pts_goal"])
    np.testing.assert_allclose(a["pts_cur"][:, 0] - a["pts_goal"][:, 0], 10.0, atol=0.5)
    np.testing.assert_allclose(a["pts_cur"][:, 1], a["pts_goal"][:, 1], atol=0.5)


def test_the_certainty_floor_drops_the_uncertain_band_and_the_settings_are_checked():
    m = rm.RoMaMatcher(version=1, num_landmarks=200, device="cpu", seed=2, certainty_min=0.5,
                       model=_StubV1())
    m.set_episode("s", 0)
    m.reset(_rgb())
    out = m.match(_rgb())
    rows = out.to_rows()
    assert (rows["pts_goal"][:, 0] > W / 4 - 40).all()               # the left band is gone
    assert out.extras["n_raw"] == 200 and out.extras["n_after_certainty"] == len(out)
    with pytest.raises(ValueError, match="sampling"):
        rm.RoMaMatcher(version=1, num_landmarks=24, device="cpu", sampling="grid")
    with pytest.raises(ValueError, match="version"):
        rm.RoMaMatcher(version=3, num_landmarks=24, device="cpu")
    with pytest.raises(ValueError, match="num_landmarks"):
        rm.RoMaMatcher(version=2, num_landmarks=0, device="cpu")


def test_the_arm_file_builds_both_versions_with_a_stub_and_labels_them():
    cfg = cfgmod.load(ARMS / "roma.yaml")
    cfg["matcher"]["device"] = "cpu"
    m = rm.RoMaMatcher(version=1, num_landmarks=24, device="cpu", ransac="none", model=_StubV1())
    arm = arms.build_arm(cfg, intrinsics=D435I, matcher=m)
    assert arm.name == "roma/k24/ur/uniform/ransac-scene_default/cns_plateau"
    assert arm.record_fields()["roma_version"] == 1
    cfg2 = {**cfg, "matcher": {**cfg["matcher"], "variant": "romav2"}}
    m2 = rm.RoMaMatcher(version=2, num_landmarks=24, device="cpu", ransac="none", model=_StubV1())
    arm2 = arms.build_arm(cfg2, intrinsics=D435I, matcher=m2)
    assert arm2.name.startswith("romav2/k24/") and arm2.matcher_name == "roma_v2"
