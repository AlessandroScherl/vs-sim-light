# vs-sim copy: tests/fast/test_match.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""U2 golden tests: the pool, the seeded draw, the signals, the registry.

All but one test runs on **stored tokens**, so the file is fast tier and needs no
weights: ``tests/fixtures/u2_matches_mug1.npz`` carries the old deploy tokens of
the current view, the U1 fixture the goal's. The exception
(``test_new_tokens_change_nothing_in_the_pool``) re-encodes both frames with
:class:`vsbench.vs.baselines.features.FrozenBackbone` and skips where the weights are not
cached. The fixture was produced once by ``genvs-sim`` @ tag
``legacy-pre-cleanup`` - ``tests/fixtures/u2_fixture_provenance.md``.

Measured 2026-09-02 in both environments (``refinevs`` torch 2.14.0+cpu, ``bot``
torch 2.11.0+cpu): pool indices, cyclic scores, similarities and margins are
**bit-identical** (max abs 0.0) to the frozen matcher's and the 24- and
48-landmark draws are the same pairs in the same order. The four signals sit
within 5e-6, which is exactly the ``round(x, 5)`` the old
``n2b_significance.row_significance`` applied to its records; rounded, equal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from vsbench.vs.baselines.match import (
    DEFAULT_TIE_ORDER,
    POOL_ALL,
    CorrespondenceMatcher,
    DenseMatcher,
    EpisodeDraw,
    Matches,
    build_matcher,
    episode_seed,
    margins_for_matches,
    mutual_nn_match,
    patch_mask_from_pixels,
    row_signals,
)
from vsbench.vs.bench.config import MATCHER_VARIANTS, ConfigError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
U2_NPZ = FIXTURES / "u2_matches_mug1.npz"
U1_NPZ = FIXTURES / "u1_dinov3_tokens_mug1_goal.npz"
SIDE = 32
#: how far a pool number may sit from the frozen matcher's. Measured: 0.0.
PARITY_TOL = 1e-6
#: The old ``row_significance`` rounded its records to **5 decimals**, so a comparison
#: against them is meaningful to **one unit in that last stored decimal**, not half of
#: one - and 5e-6 (the half-unit) was the bar until review R19 #16 found it failing at
#: 1 and 4 BLAS threads and passing at 12. Measured here, so the widening is arithmetic
#: rather than convenience:
#:
#: * the float32 similarity matmul reduces in thread-count order, so the signals move by
#:   at most **4.6e-7** between 1 and 12 threads (``s1`` 1.8e-7, ``second``/``margin``
#:   3.0e-7, ``ratio`` 3.0e-7, ``entropy_t05`` 4.6e-7, ``entropy_t1`` 8.7e-8);
#: * every signal sits within 4.9999e-6 of its own 5-decimal rounding - i.e. the stored
#:   fixture leaves **no headroom at all** above the half-unit;
#: * four values (1 ``ratio``, 2 ``entropy_t05``, 1 ``entropy_t1``) sit within **7.5e-8**
#:   of a rounding boundary, which is *inside* that drift. At 12 threads they round the way
#:   the frozen probe rounded; at 1 and 4 they round the other way, and the gap to the
#:   stored value jumps a whole unit to 5.07e-6.
#:
#: So the honest bar is one unit in the last stored decimal. Pinning the thread count
#: instead would hide a real property of the data and would not survive the station's
#: BLAS anyway.
SIGNAL_TOL = 1e-5
#: How far from a 5-decimal rounding boundary a value may sit and still be allowed to
#: round the other way, **in units of that last decimal**. It is the measured
#: thread-order drift (4.6e-7 absolute) expressed in those units - 0.046 - rounded up to
#: 0.05. The four boundary values sit 0.0075 or less from the boundary, well inside it;
#: a value further out than the drift that rounded differently would be a real
#: disagreement and this test would say so.
BOUNDARY_SLACK = 0.05
#: a goal row / current column whose best-vs-second similarity gap is below this
#: may legally change its argmax under token noise of order 1e-6. Measured on the
#: fixture: smallest goal-row gap 5.007e-6 (2 rows below 1e-5), smallest
#: current-column gap 7.689e-6 (1 column). See R11 correction 6.
NEAR_TIE_GAP = 1e-5


@pytest.fixture(scope="module")
def fx() -> dict:
    return dict(np.load(U2_NPZ, allow_pickle=False))


@pytest.fixture(scope="module")
def goal_tokens() -> torch.Tensor:
    return torch.from_numpy(np.load(U1_NPZ, allow_pickle=False)["tokens"])


@pytest.fixture(scope="module")
def cur_tokens(fx) -> torch.Tensor:
    return torch.from_numpy(fx["cur_tokens"])


@pytest.fixture(scope="module")
def pool(goal_tokens, cur_tokens):
    return mutual_nn_match(goal_tokens, cur_tokens, threshold=1.0, tie_order="legacy")


def _lin(patch: np.ndarray) -> np.ndarray:
    """``(col, row)`` -> ``row * side + col``."""
    p = np.asarray(patch, np.int64)
    return p[:, 1] * SIDE + p[:, 0]


class _TokenBackbone:
    """A weight-free stand-in for :class:`vsbench.vs.baselines.features.FrozenBackbone`.

    ``encode`` looks the frame up by the tag in its first pixel, so the fast tier
    drives the whole :class:`DenseMatcher` chain - reset, draw, pin, track,
    geometry - on stored tokens, without the backbone.
    """

    name = "dinov3_vits16"
    input_size = 512
    patch_size = 16
    grid_side = SIDE
    dim = 384

    def __init__(self, tokens: dict[int, torch.Tensor]) -> None:
        self.tokens = tokens

    @staticmethod
    def frame(tag: int) -> np.ndarray:
        """A 1080x1920 tag frame, allocated cheaply (only pixel [0,0,0] is read)."""
        f = np.zeros((1080, 1920, 3), np.uint8)
        f[0, 0, 0] = tag
        return f

    def encode(self, frame_rgb_u8: np.ndarray) -> torch.Tensor:
        return self.tokens[int(frame_rgb_u8[0, 0, 0])]


@pytest.fixture
def dense(goal_tokens, cur_tokens):
    """A :class:`DenseMatcher` over the fixture's tokens, episode ``(mug1, 7)``.

    ``tie_order="legacy"`` is explicit here and in every other parity test: it is
    what reproduces the frozen genvs-sim fixture on CPU. The repository default
    is ``"stable"`` (owner ruling 2026-09-03).
    """
    backbone = _TokenBackbone({0: goal_tokens, 1: cur_tokens})
    m = DenseMatcher(num_landmarks=24, seed=0, backbone=backbone, tie_order="legacy")
    m.set_episode("mug1", 7)
    m.reset(_TokenBackbone.frame(0))
    return m


# ------------------------------------------------------------- the fixture


def test_the_fixture_still_describes_the_committed_images(fx):
    """A silently swapped image would make every parity number below meaningless."""
    for name_key, md5_key in (("goal_image_file", "goal_image_md5"),
                              ("cur_image_file", "cur_image_md5")):
        blob = (FIXTURES / str(fx[name_key])).read_bytes()
        assert hashlib.md5(blob).hexdigest() == str(fx[md5_key])
    assert fx["cur_tokens"].shape == (SIDE, SIDE, 384)
    assert int(fx["subsample_seed"]) == episode_seed("mug1", 7, 0)


# ---------------------------------------------------------------- the pool


def test_pool_reproduces_the_frozen_matcher(pool, fx):
    """612 pairs, same order, same numbers as the vendored find_correspondences_batch.

    Order matters as much as membership: it is what ``randperm`` indexes.
    """
    assert pool.size == len(fx["pool_goal_lin"]) == 612
    np.testing.assert_array_equal(pool.goal_lin.numpy(), fx["pool_goal_lin"])
    np.testing.assert_array_equal(pool.cur_lin.numpy(), fx["pool_cur_lin"])
    for got, want in (
        (pool.cyclic.numpy(), fx["pool_cyclic"]),
        (pool.sim.numpy(), fx["pool_sim"]),
        (pool.margin.numpy(), fx["pool_margin"]),
        (pool.cyclic_all.numpy(), fx["cyclic_all"]),
    ):
        assert float(np.abs(got - want).max()) <= PARITY_TOL
    assert float(pool.cyclic.min()) == 1.0, "threshold 1.0 keeps only perfect round trips"


def test_the_pool_is_the_mutual_nearest_neighbour_set(pool, goal_tokens, cur_tokens):
    """Cross-check by definition, not by fixture: g is in the pool iff nn(nn(g)) == g."""
    normalize = torch.nn.functional.normalize
    g = normalize(goal_tokens.reshape(-1, 384), dim=-1)
    c = normalize(cur_tokens.reshape(-1, 384), dim=-1)
    sims = g @ c.T
    nn_goal, nn_cur = sims.argmax(dim=1), sims.argmax(dim=0)
    mutual = torch.nonzero(nn_cur[nn_goal] == torch.arange(SIDE * SIDE)).reshape(-1)
    assert set(pool.goal_lin.tolist()) == set(mutual.tolist())
    np.testing.assert_array_equal(pool.cur_lin.numpy(), nn_goal[pool.goal_lin].numpy())


def test_stable_is_the_default_tie_order_and_is_the_same_pool_sorted(
    goal_tokens, cur_tokens, pool
):
    """Default = `stable`: same 612 pairs, ascending goal index, any torch build.

    `legacy` reproduces the frozen record on CPU and is what the parity tests
    pass explicitly; it is not device-independent, because the sort runs where
    the tokens live. A different order draws a different (equally random) set.
    """
    assert DEFAULT_TIE_ORDER == "stable"
    stable = mutual_nn_match(goal_tokens, cur_tokens)
    assert stable.tie_order == "stable"
    np.testing.assert_array_equal(stable.goal_lin.numpy(), np.sort(pool.goal_lin.numpy()))
    assert not np.array_equal(stable.goal_lin.numpy(), pool.goal_lin.numpy())
    assert mutual_nn_match(goal_tokens, cur_tokens, tie_order="legacy").tie_order == "legacy"
    stable_draw = EpisodeDraw("mug1", 7, 0).draw(stable, 24)
    legacy_draw = EpisodeDraw("mug1", 7, 0).draw(pool, 24)
    assert not set(_lin(stable_draw.goal_patch)) & set(_lin(legacy_draw.goal_patch))
    with pytest.raises(ValueError, match="tie_order"):
        mutual_nn_match(goal_tokens, cur_tokens, tie_order="whatever")


def test_identical_grids_take_the_all_tied_branch(goal_tokens):
    """goal == current: every round trip is perfect, so every patch survives.

    The vendored ``max_dist < 1e-6`` branch (``torch.ones_like``), i.e. the
    parked-at-goal frame - and the input on which torch 2.11 and 2.14 return
    different non-stable sort orders, which is why ``tie_order`` exists.
    """
    p = mutual_nn_match(goal_tokens, goal_tokens)
    assert p.size == SIDE * SIDE
    np.testing.assert_array_equal(p.goal_lin.numpy(), np.arange(SIDE * SIDE))
    np.testing.assert_array_equal(p.cur_lin.numpy(), np.arange(SIDE * SIDE))
    assert float(p.cyclic_all.min()) == 1.0


def test_mask_and_threshold_are_honoured(goal_tokens, cur_tokens, pool):
    even = torch.zeros(SIDE * SIDE, dtype=torch.bool)
    even[pool.goal_lin[:100]] = True
    masked = mutual_nn_match(goal_tokens, cur_tokens, mask=even)
    assert set(masked.goal_lin.tolist()) == set(pool.goal_lin[:100].tolist())
    assert mutual_nn_match(goal_tokens, cur_tokens, mask=even.reshape(SIDE, SIDE)).size == 100
    assert mutual_nn_match(goal_tokens, cur_tokens, mask=torch.zeros_like(even)).size == 0
    # A loose threshold admits imperfect round trips; the fielded 1.0 admits none.
    # (The scores are normalised by the frame's own worst round trip - about 6
    # cells here - so 0.5 means "landed within ~3 cells of where it started".)
    loose = mutual_nn_match(goal_tokens, cur_tokens, threshold=0.5)
    assert loose.size > pool.size and float(loose.cyclic.min()) < 1.0


def test_non_square_token_counts_are_refused(goal_tokens):
    with pytest.raises(ValueError, match="not a square grid"):
        mutual_nn_match(goal_tokens.reshape(-1, 384)[:1000], goal_tokens)
    with pytest.raises(ValueError, match="descriptor dims differ"):
        mutual_nn_match(goal_tokens, goal_tokens[..., :100])


# ---------------------------------------------------------------- the draw


def test_seeded_draw_reproduces_the_frozen_landmarks(pool, fx):
    """The exact 24 and 48 pairs the fielded arm drew, in order."""
    for k in (24, 48):
        m = EpisodeDraw("mug1", 7, 0).draw(pool, k)
        assert len(m) == k and m.num_landmarks == k and m.pool_size == 612
        assert m.draw_seed == int(fx["subsample_seed"])
        np.testing.assert_array_equal(_lin(m.goal_patch), fx[f"draw{k}_goal_lin"])
        np.testing.assert_array_equal(_lin(m.cur_patch), fx[f"draw{k}_cur_lin"])
        assert float(np.abs(m.sim - fx[f"draw{k}_sim"]).max()) <= PARITY_TOL
        assert float(np.abs(m.margin - fx[f"draw{k}_margin"]).max()) <= PARITY_TOL


def test_the_draw_is_a_function_of_scene_start_seed_only(pool):
    """Same triple -> same draw, from a fresh object; a different seed -> a different one."""
    a = EpisodeDraw("mug1", 7, 0).draw(pool, 24)
    b = EpisodeDraw("mug1", 7, 0).draw(pool, 24)
    np.testing.assert_array_equal(a.goal_patch, b.goal_patch)
    for scene, start, seed in (("mug1", 8, 0), ("bottle1", 7, 0), ("mug1", 7, 1)):
        other = EpisodeDraw(scene, start, seed).draw(pool, 24)
        assert other.draw_seed != a.draw_seed
        assert not np.array_equal(other.goal_patch, a.goal_patch)


def test_larger_landmark_counts_are_supersets_at_the_first_frame(pool):
    """N3.7d's convention: one permutation per frame, sliced - so k48 contains k24."""
    sets = {}
    for k in (24, 48, 96, 192):
        m = EpisodeDraw("mug1", 7, 0).draw(pool, k)
        sets[k] = set(_lin(m.goal_patch).tolist())
        assert len(sets[k]) == k
    for small, large in ((24, 48), (48, 96), (96, 192)):
        assert sets[small] <= sets[large]


def test_the_draw_advances_frame_by_frame(pool):
    """Draw i uses seed ``episode_seed + i``: frame 2 is a new random 24 (the
    published re-draw-every-frame path) and reset() returns to frame 0."""
    draw = EpisodeDraw("mug1", 7, 0)
    first = draw.draw(pool, 24)
    second = draw.draw(pool, 24)
    assert second.draw_seed == first.draw_seed + 1
    assert not np.array_equal(first.goal_patch, second.goal_patch)
    draw.reset()
    np.testing.assert_array_equal(draw.draw(pool, 24).goal_patch, first.goal_patch)


def test_the_draw_clamps_to_the_pool_and_refuses_a_missing_count(pool):
    assert len(EpisodeDraw("mug1", 7, 0).draw(pool, POOL_ALL)) == pool.size
    with pytest.raises(ValueError, match="no default"):
        EpisodeDraw("mug1", 7, 0).draw(pool, None)
    with pytest.raises(ValueError, match=">= 1"):
        EpisodeDraw("mug1", 7, 0).draw(pool, 0)
    with pytest.raises(ValueError, match="no default"):
        EpisodeDraw("mug1", 7, None)


def test_episode_seed_is_crc32_and_shared_across_arms():
    assert episode_seed("mug1", 7, 0) == episode_seed("mug1", "7", 0)
    assert episode_seed("mug1", 7, 5) - episode_seed("mug1", 7, 0) == 5
    assert 0 <= episode_seed("cluttered2", 19, 0) < 2**31


def test_the_draw_arithmetic_is_the_vendored_one(pool, fx):
    """Isolate the permutation from the tie order.

    Re-order the canonical pool into the frozen record's own order and the draw
    must reproduce the record exactly - which proves the seeding, the
    ``randperm`` and the slice are the vendored ones, whatever the sort did.
    """
    stable = mutual_nn_match(
        torch.from_numpy(np.load(U1_NPZ, allow_pickle=False)["tokens"]),
        torch.from_numpy(fx["cur_tokens"]),
        tie_order="stable",
    )
    position = {int(g): i for i, g in enumerate(stable.goal_lin.tolist())}
    reordered = stable.take(torch.tensor([position[int(g)] for g in fx["pool_goal_lin"]]))
    m = EpisodeDraw("mug1", 7, 0).draw(reordered, 24)
    np.testing.assert_array_equal(_lin(m.goal_patch), fx["draw24_goal_lin"])


# ------------------------------------------------------------- the signals


def test_signals_match_the_old_probe(pool, fx):
    """margin / Lowe ratio / softmax entropy / look-alike count of all 612 pairs.

    Against ``n2b_significance.row_significance``, which rounded its records to 5
    decimals - so that rounding is the bar (:data:`SIGNAL_TOL`, and read its note: the
    bar is one unit in the last stored decimal, not half of one, because the matmul's
    thread-dependent reduction order moves four values across a rounding boundary).

    The 5-decimal round-trip is still asserted **exactly** for every value that is not
    sitting on such a boundary; a value that is may round either way, and the test says
    which ones those are instead of allowing any mismatch anywhere.
    """
    got = pool.signals()
    for key in ("s1", "second", "margin", "ratio", "n_lookalikes",
                "entropy_t05", "entropy_t1"):
        want = fx[f"sig_{key}"]
        assert float(np.abs(got[key] - want).max()) <= SIGNAL_TOL, key
        differs = np.round(got[key], 5) != want
        if differs.any():
            # every disagreeing value must be a boundary case, and off by exactly one
            # unit in the fifth decimal - never by more, and never for another reason
            scaled = got[key][differs] * 1e5
            on_boundary = np.abs(np.abs(scaled - np.floor(scaled)) - 0.5) <= BOUNDARY_SLACK
            assert on_boundary.all(), f"{key}: {int((~on_boundary).sum())} row(s) differ " \
                                      "in the 5th decimal for a reason other than rounding"
            # exactly ONE unit in the fifth decimal - stated as a unit count, because
            # `0.98562 - 0.98561` is 1.0000000000065512e-05 in binary floating point
            units = np.abs(np.round(got[key], 5) - want)[differs] / 1e-5
            assert np.array_equal(np.round(units), np.ones_like(units)), key
    # the deployed margin (`significance.margins_for_matches`) is unrounded
    assert float(np.abs(pool.margin.numpy() - fx["pool_margin"]).max()) <= PARITY_TOL


def test_margins_and_signals_equal_a_literal_loop():
    """The vectorised block exclusion against the old per-row Python loop - the
    one place U2 rewrote rather than transcribed."""
    rng = np.random.RandomState(20260902)
    side = 7
    rows = rng.rand(11, side * side)
    cols = rng.randint(0, side * side, 11)
    want = np.empty(11)
    second = np.empty(11)
    for i, c in enumerate(cols):
        r0, r1 = max(0, c // side - 1), min(side, c // side + 2)
        c0, c1 = max(0, c % side - 1), min(side, c % side + 2)
        scratch = rows[i].copy().reshape(side, side)
        scratch[r0:r1, c0:c1] = -np.inf
        second[i] = scratch.max()
        want[i] = rows[i, c] - second[i]
    got = margins_for_matches(torch.from_numpy(rows), torch.from_numpy(cols), side)
    np.testing.assert_allclose(got.numpy(), want, atol=0, rtol=0)
    sig = row_signals(torch.from_numpy(rows), torch.from_numpy(cols), side)
    np.testing.assert_allclose(sig["second"], second, atol=0, rtol=0)
    np.testing.assert_allclose(sig["ratio"], second / rows[np.arange(11), cols])
    with pytest.raises(ValueError, match="row width"):
        margins_for_matches(torch.from_numpy(rows), torch.from_numpy(cols), side + 1)


# --------------------------------------------------------- the dense matcher


def test_dense_matcher_reproduces_the_fielded_arm(dense, fx):
    """reset -> encode -> pool -> draw, with the two-hop pixel mapping."""
    m = dense.match(_TokenBackbone.frame(1))
    np.testing.assert_array_equal(_lin(m.goal_patch), fx["draw24_goal_lin"])
    np.testing.assert_array_equal(_lin(m.cur_patch), fx["draw24_cur_lin"])
    assert (m.input_size, m.patch_size, m.image_wh, m.frozen) == (512, 16, (1920, 1080), False)
    rows = m.to_rows()
    assert float(np.abs(rows["pts_goal"] - fx["draw24_goal_px"]).max()) <= PARITY_TOL
    assert float(np.abs(rows["pts_cur"] - fx["draw24_cur_px"]).max()) <= PARITY_TOL
    # 512-space centres: patch index * 16 + 8, and the second hop is anisotropic
    np.testing.assert_allclose(m.goal_xy, m.goal_patch * 16.0 + 8.0)
    np.testing.assert_allclose(rows["pts_goal"], m.goal_xy * [1920 / 512, 1080 / 512])


def test_frozen_landmarks_pins_frame_one_and_tracks_after(goal_tokens, cur_tokens, fx):
    """The `_fs` variant: frame 1 is the published draw, frames 2+ the tracker.

    Frame 2 re-reads the current view, frame 3 the goal view - where the one-way
    nearest neighbour of a pinned goal patch must be itself.
    """
    backbone = _TokenBackbone({0: goal_tokens, 1: cur_tokens})
    m = DenseMatcher(num_landmarks=24, seed=0, backbone=backbone, frozen_landmarks=True,
                     tie_order="legacy")
    m.set_episode("mug1", 7)
    m.reset(_TokenBackbone.frame(0))

    first = m.match(_TokenBackbone.frame(1))
    np.testing.assert_array_equal(_lin(first.goal_patch), fx["draw24_goal_lin"])
    assert not first.frozen, "frame 1 runs the published path, then pins it"

    for tag, frame in (("fs2", 1), ("fs3", 0)):
        out = m.match(_TokenBackbone.frame(frame))
        assert out.frozen and out.pool_size == 24
        np.testing.assert_array_equal(_lin(out.goal_patch), fx[f"{tag}_goal_lin"])
        np.testing.assert_array_equal(_lin(out.cur_patch), fx[f"{tag}_cur_lin"])
        assert float(np.abs(out.sim - fx[f"{tag}_sim"]).max()) <= PARITY_TOL
        assert float(np.abs(out.margin - fx[f"{tag}_margin"]).max()) <= PARITY_TOL
        assert np.isnan(out.cyclic).all(), "a one-way tracker has no cyclic score"
    # the pin never survives an episode
    m.reset(_TokenBackbone.frame(0))
    assert not m.match(_TokenBackbone.frame(1)).frozen


def test_dense_matcher_state_guards(goal_tokens, cur_tokens):
    backbone = _TokenBackbone({0: goal_tokens, 1: cur_tokens})
    m = DenseMatcher(num_landmarks=24, seed=0, backbone=backbone)
    with pytest.raises(RuntimeError, match="set_episode"):
        m.reset(_TokenBackbone.frame(0))
    m.set_episode("mug1", 7)
    with pytest.raises(RuntimeError, match="reset"):
        m.match(_TokenBackbone.frame(1))
    with pytest.raises(ValueError, match="no default"):
        DenseMatcher(seed=0, backbone=backbone)
    with pytest.raises(ValueError, match=">= 1"):
        DenseMatcher(num_landmarks=0, seed=0, backbone=backbone)
    with pytest.raises(ValueError, match="'seed'"):
        DenseMatcher(num_landmarks=24, backbone=backbone).set_episode("mug1", 7)
    assert isinstance(m, CorrespondenceMatcher)
    assert not hasattr(m, "set_intrinsics"), "the dead _dopt hook is not ported (R11 #8)"
    # the old `matcher_ibvs.step` called match(image, mask) positionally: the
    # shape survives, but a mask is refused loudly instead of ignored silently.
    m.reset(_TokenBackbone.frame(0))
    assert len(m.match(_TokenBackbone.frame(1), None)) == 24
    with pytest.raises(ValueError, match="current_mask"):
        m.match(_TokenBackbone.frame(1), np.ones((1080, 1920), bool))


def test_an_empty_pool_returns_empty_matches_and_still_burns_the_frame_seed(
    goal_tokens, cur_tokens
):
    """The old `_call_find_correspondences` advanced `_draw_idx` on every call,

    including a frame that matched nothing (vit_dense.py:365), so frame indices
    stayed aligned across arms. Two empty frames must therefore leave the third
    drawing at `episode_seed + 2`.
    """
    backbone = _TokenBackbone({0: goal_tokens, 1: cur_tokens})
    m = DenseMatcher(num_landmarks=24, seed=0, backbone=backbone,
                     mask=np.zeros(SIDE * SIDE, bool), tie_order="legacy")
    m.set_episode("mug1", 7)
    m.reset(_TokenBackbone.frame(0))
    for i in range(2):
        out = m.match(_TokenBackbone.frame(1))
        assert len(out) == 0 and out.pool_size == 0
        assert out.to_rows()["pts_goal"].shape == (0, 2)
        assert out.extras["reason"] == "no correspondences"  # the old debug field
        assert m.draw.draw_index == i + 1
    assert m.draw.frame_seed() == episode_seed("mug1", 7, 0) + 2
    # and an empty pool draws empty rather than raising
    empty = mutual_nn_match(goal_tokens, cur_tokens, mask=np.zeros(SIDE * SIDE, bool))
    assert len(EpisodeDraw("mug1", 7, 0).draw(empty, 24)) == 0


def test_goal_mask_is_honoured_the_way_the_old_matcher_honoured_it(goal_tokens, cur_tokens):
    """`reset(goal_mask=...)` is a pixel mask, converted exactly as

    `vit_dense._patch_mask_from_pixels` did (nearest-neighbour resize, >127, a
    patch eligible when more than 20 % of its pixels are set). An all-False mask
    yielded 0 matches in the old code and must here.
    """
    pytest.importorskip("cv2")
    backbone = _TokenBackbone({0: goal_tokens, 1: cur_tokens})
    m = DenseMatcher(num_landmarks=24, seed=0, backbone=backbone, tie_order="legacy")
    m.set_episode("mug1", 7)

    m.reset(_TokenBackbone.frame(0), goal_mask=np.zeros((1080, 1920), bool))
    assert len(m.match(_TokenBackbone.frame(1))) == 0

    left = np.zeros((1080, 1920), bool)
    left[:, :960] = True
    m.reset(_TokenBackbone.frame(0), goal_mask=left)
    out = m.match(_TokenBackbone.frame(1))
    assert 0 < len(out) <= 24 and (out.goal_patch[:, 0] < SIDE // 2).all()

    # the conversion itself, independent of the matcher
    assert patch_mask_from_pixels(left, input_size=512, patch_size=16).sum() == SIDE * SIDE // 2
    assert not patch_mask_from_pixels(
        np.zeros((1080, 1920), bool), input_size=512, patch_size=16
    ).any()


def test_matches_carries_the_hooks_the_head_needs(dense):
    """U5 attaches offsets and trust; `cur_xy` then reads the refined position."""
    m = dense.match(_TokenBackbone.frame(1))
    coarse = m.cur_xy.copy()
    m.offsets_patch = np.full((len(m), 2), 0.25)
    m.trust = np.linspace(0.1, 0.9, len(m))
    np.testing.assert_allclose(m.cur_xy, coarse + 0.25 * 16.0)
    assert "trust" in m.to_rows()
    m.extras["anything"] = np.arange(len(m))  # extras carries what U5/U7 add
    assert m.extras["anything"].shape == (len(m),)
    with pytest.raises(ValueError, match="row-aligned"):
        Matches(goal_patch=np.zeros((3, 2), np.int64), cur_patch=np.zeros((2, 2), np.int64),
                sim=np.zeros(2), cyclic=np.zeros(2), margin=np.zeros(2), side=SIDE,
                pool_size=2, num_landmarks=2, draw_seed=1, tie_order="legacy")


def test_matches_without_geometry_say_so(pool):
    """The training path (U5) gets patch indices and never touches pixels."""
    m = EpisodeDraw("mug1", 7, 0).draw(pool, 8)
    with pytest.raises(ValueError, match="input_size"):
        _ = m.goal_xy
    with pytest.raises(ValueError, match="image_wh"):
        m.input_size, m.patch_size = 512, 16
        m.to_rows()


# ---------------------------------------------------------------- registry


def _config(variant: str, n: int = 24) -> dict:
    return {
        "seed": 0,
        # required for every encoder variant, and stated even behind a stub backbone
        "matcher": {"variant": variant, "num_landmarks": n, "device": "cpu"},
        "stop": {"rule": "dec_plateau"},
        "scoring": {"success_rule": ["predicate_a"]},
    }


@pytest.mark.parametrize(
    ("variant", "n", "frozen"),
    [("mutual_nn_seeded_subsample", 24, False),
     ("frozen_landmarks", 24, True),
     ("mutual_nn_all", POOL_ALL, False)],
)
def test_the_registry_builds_the_right_dense_matcher(variant, n, frozen, goal_tokens):
    backbone = _TokenBackbone({0: goal_tokens})
    m = build_matcher(_config(variant, n), backbone=backbone, scene="mug1", start_id=7)
    assert isinstance(m, DenseMatcher)
    assert (m.num_landmarks, m.frozen_landmarks) == (n, frozen)
    assert m.tie_order == "stable", "the repository default, unless the config says otherwise"
    assert m.draw is not None and m.draw.episode_seed == episode_seed("mug1", 7, 0)


def test_mutual_nn_all_must_say_so_in_the_landmark_count(goal_tokens):
    with pytest.raises(ValueError, match=str(POOL_ALL)):
        build_matcher(_config("mutual_nn_all", 24), backbone=_TokenBackbone({0: goal_tokens}))


@pytest.mark.parametrize(
    ("variant", "cls"),
    [("classical_sift", "ClassicalMatcher"), ("classical_orb", "ClassicalMatcher"),
     ("classical_akaze", "ClassicalMatcher"), ("artvs", "TiledMatcher")],
)
def test_the_u3_variants_build_their_own_class(variant, cls):
    """U3 landed: these four are real matchers, built here and named by their class.

    `artvs` gets an injected backbone to keep this test weight-free; the real chain is
    exercised in `tests/fast/test_tiling.py`, and the classical arms in
    `tests/fast/test_classical.py`.
    """
    kw = {} if variant.startswith("classical") else {"backbone": _TokenBackbone({})}
    cfg = _config(variant)
    if variant.startswith("classical"):
        cfg["matcher"]["ransac"] = "none"
    assert type(build_matcher(cfg, **kw)).__name__ == cls


def test_cns_is_a_variant_but_not_a_matcher():
    """`cns_v1` is a whole pipeline - CNS replaces the control law, so it is an arm
    and `build_matcher` says where it IS built rather than claiming it is unported."""
    with pytest.raises(ValueError, match="vsbench.vs.baselines.cns.build_cns_arm"):
        build_matcher(_config("cns_v1"))


def test_the_registry_covers_the_config_enum():
    from vsbench.vs.baselines.match import MATCHER_BUILDERS

    assert set(MATCHER_BUILDERS) == set(MATCHER_VARIANTS)


def test_the_registry_refuses_a_config_without_a_landmark_count():
    """The loader already refuses it; so does the builder, naming the field."""
    bad = _config("mutual_nn_seeded_subsample")
    del bad["matcher"]["num_landmarks"]
    with pytest.raises(ConfigError, match="matcher.num_landmarks"):
        build_matcher(bad)
    no_variant = _config("mutual_nn_seeded_subsample")
    del no_variant["matcher"]["variant"]
    with pytest.raises(ConfigError, match="matcher.variant"):
        build_matcher(no_variant)


def test_the_matcher_config_block_can_carry_the_optional_knobs(goal_tokens):
    cfg = _config("mutual_nn_seeded_subsample")
    cfg["matcher"].update(threshold=0.9, tie_order="legacy")
    m = build_matcher(cfg, backbone=_TokenBackbone({0: goal_tokens}))
    assert (m.threshold, m.tie_order) == (0.9, "legacy")


# ------------------------------------------------------------------- the device
def test_a_matcher_that_must_load_an_encoder_refuses_to_guess_a_device(goal_tokens):
    """R22 correction 3. `FrozenBackbone` defaulted to `cpu`, so E0 ran the whole
    learned chain on the CPU at twice its budget with no config saying so. An injected
    backbone still needs no device - it has one of its own."""
    with pytest.raises(ValueError, match="matcher.device"):
        DenseMatcher(num_landmarks=24, seed=0)
    m = DenseMatcher(num_landmarks=24, seed=0, backbone=_TokenBackbone({0: goal_tokens}))
    assert m.record_fields()["backbone"] == "dinov3_vits16"


def test_the_record_states_the_device_the_encoder_actually_ran_on(dense):
    """The ENCODER's device, not the config's: a record that echoed the config would
    keep saying `cuda` while a stub ran on the CPU (the E0 failure, inverted)."""
    fields = dense.record_fields()
    assert "matcher_device" in fields
    assert fields["matcher_device"] is None            # the token stub owns no device

    class _OnCuda(_TokenBackbone):
        device = "cuda:0"

    m = DenseMatcher(num_landmarks=24, seed=0, backbone=_OnCuda({}))
    assert m.record_fields()["matcher_device"] == "cuda:0"


# ----------------------------------------------- new tokens vs old tokens


@pytest.fixture(scope="module")
def backbone():
    """The DINOv3 backbone on CPU, or a skip naming why it is unavailable."""
    pytest.importorskip("transformers", reason="the fast tier runs without the backbone")
    from vsbench.vs.baselines.features import FrozenBackbone

    try:
        return FrozenBackbone("dinov3_vits16", device="cpu")
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        pytest.skip(
            "DINOv3 ViT-S/16 weights are not in the local HF cache "
            f"({type(exc).__name__}: {str(exc).splitlines()[0]})"
        )


def _near_tie_rows(goal_tokens, cur_tokens):
    """Goal rows and current columns whose best-vs-second gap is below NEAR_TIE_GAP.

    Only these can legally change their argmax under a token perturbation of
    order 1e-6, so only these can move the mutual pool.
    """
    normalize = torch.nn.functional.normalize
    sims = normalize(goal_tokens.reshape(-1, 384), dim=-1) @ normalize(
        cur_tokens.reshape(-1, 384), dim=-1
    ).T
    gap_row = sims.topk(2, dim=1).values.diff(dim=1).abs().reshape(-1)
    gap_col = sims.topk(2, dim=0).values.diff(dim=0).abs().reshape(-1)
    return gap_row, gap_col


def test_new_tokens_only_move_near_ties_in_the_pool(backbone, fx, pool, goal_tokens, cur_tokens):
    """U1's `encode` sits ~1e-6 from the frozen tokens. What can that move?

    Only a goal row or a current column whose best-vs-second similarity gap is
    itself of that order. Measured on this fixture: the smallest goal-row gap is
    **5.007e-6** (2 rows below 1e-5) and the smallest current-column gap
    **7.689e-6** (1 column), against token differences of 8.8e-7 (goal) and
    2.3e-6 (current) - so a flip is possible in principle and **0 of 612 pool
    pairs and 0 of 24 drawn pairs actually flip on CPU**. The assertion is the
    honest one (R11 correction 6): whatever flips must sit on a near-tie row or
    column; a flip anywhere else is a port bug. On CUDA tokens some frames are
    expected to flip a near-tie pair.
    """
    from PIL import Image

    frames = [
        np.array(Image.open(FIXTURES / str(fx[key])).convert("RGB"))
        for key in ("goal_image_file", "cur_image_file")
    ]
    new_tokens = [backbone.encode(f) for f in frames]
    new_pool = mutual_nn_match(*new_tokens, tie_order="legacy")

    gap_row, gap_col = _near_tie_rows(goal_tokens, cur_tokens)
    fragile = set(torch.nonzero(gap_row < NEAR_TIE_GAP).reshape(-1).tolist())
    fragile |= {
        int(g)
        for g, c in zip(pool.goal_lin.tolist(), pool.cur_lin.tolist(), strict=True)
        if gap_col[c] < NEAR_TIE_GAP
    }
    old_pairs = dict(zip(pool.goal_lin.tolist(), pool.cur_lin.tolist(), strict=True))
    new_pairs = dict(
        zip(new_pool.goal_lin.tolist(), new_pool.cur_lin.tolist(), strict=True)
    )
    moved = {g for g in set(old_pairs) | set(new_pairs) if old_pairs.get(g) != new_pairs.get(g)}
    assert moved <= fragile, (
        f"{len(moved - fragile)} pool pairs moved without being a near tie "
        f"(gap >= {NEAR_TIE_GAP}); smallest gap on the fixture is "
        f"{float(gap_row.min()):.3e}"
    )
    assert not moved, f"recorded: 0 flips on CPU, got {len(moved)} (all near ties)"
    common = set(old_pairs) & set(new_pairs)
    assert [g for g in pool.goal_lin.tolist() if g in common] == [
        g for g in new_pool.goal_lin.tolist() if g in common
    ], "the surviving pairs must keep their relative order in the pool"
    assert float((new_pool.sim - pool.sim).abs().max()) <= 1e-5
    assert float((new_pool.margin - pool.margin).abs().max()) <= 1e-5
    old_draw = EpisodeDraw("mug1", 7, 0).draw(pool, 24)
    new_draw = EpisodeDraw("mug1", 7, 0).draw(new_pool, 24)
    assert set(_lin(new_draw.goal_patch)) - set(_lin(old_draw.goal_patch)) <= fragile


def test_the_matcher_builds_its_own_backbone_when_none_is_injected(backbone, fx):
    """The `backbone=None` construction path - the one the runner uses.

    `_TokenBackbone` covers everything else in this file; this is the only test
    that goes through `FrozenBackbone(name, device=...)` inside `DenseMatcher`,
    end to end from JPEG to full-image pixels.
    """
    from PIL import Image

    m = DenseMatcher(num_landmarks=24, seed=0, device="cpu", tie_order="legacy")
    assert m.backbone.name == "dinov3_vits16"
    m.set_episode("mug1", 7)
    m.reset(np.array(Image.open(FIXTURES / str(fx["goal_image_file"])).convert("RGB")))
    out = m.match(np.array(Image.open(FIXTURES / str(fx["cur_image_file"])).convert("RGB")))
    np.testing.assert_array_equal(_lin(out.goal_patch), fx["draw24_goal_lin"])
    rows = out.to_rows()
    assert float(np.abs(rows["pts_goal"] - fx["draw24_goal_px"]).max()) <= PARITY_TOL
    assert float(np.abs(rows["pts_cur"] - fx["draw24_cur_px"]).max()) <= PARITY_TOL


# ---------------------------------- the pool export (the matching study's T3)


def _dense_pair(goal_tokens, cur_tokens, **kw):
    """Two matchers over the same stored tokens, differing only in `kw`."""
    out = []
    for extra in ({}, kw):
        backbone = _TokenBackbone({0: goal_tokens, 1: cur_tokens})
        m = DenseMatcher(num_landmarks=24, seed=0, backbone=backbone,
                         tie_order="legacy", **extra)
        m.set_episode("mug1", 7)
        m.reset(_TokenBackbone.frame(0))
        out.append(m)
    return out


def test_keeping_the_pool_changes_nothing_on_the_fielded_path(goal_tokens, cur_tokens):
    """`keep_pool` is an export, not a setting: the landmarks and the record are equal.

    This is the guarantee T3's whole-pool candidate rests on - the study reads the pool
    of the SAME frames the fielded arm flies, so the flag must not be able to move a
    landmark or a record field.
    """
    off, on = _dense_pair(goal_tokens, cur_tokens, keep_pool=True)
    a, b = off.match(_TokenBackbone.frame(1)), on.match(_TokenBackbone.frame(1))
    np.testing.assert_array_equal(a.goal_patch, b.goal_patch)
    np.testing.assert_array_equal(a.cur_patch, b.cur_patch)
    for field in ("sim", "cyclic", "margin"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert (a.side, a.pool_size, a.num_landmarks, a.draw_seed, a.tie_order) == (
        b.side, b.pool_size, b.num_landmarks, b.draw_seed, b.tie_order)
    assert (a.input_size, a.patch_size, a.image_wh, a.frozen) == (
        b.input_size, b.patch_size, b.image_wh, b.frozen)
    assert a.extras == b.extras
    # the record must not gain a field either - `keep_pool` is deliberately absent
    assert off.record_fields() == on.record_fields()
    assert "keep_pool" not in off.record_fields()


def test_the_exported_pool_is_the_pool_the_draw_came_from(goal_tokens, cur_tokens):
    off, on = _dense_pair(goal_tokens, cur_tokens, keep_pool=True)
    assert off.last_pool is None, "off by default, and off means nothing is parked"
    matches = on.match(_TokenBackbone.frame(1))
    pool = on.last_pool
    assert pool is not None and pool.size == matches.pool_size
    reference = mutual_nn_match(goal_tokens, cur_tokens, threshold=1.0, tie_order="legacy")
    np.testing.assert_array_equal(pool.goal_lin.numpy(), reference.goal_lin.numpy())
    np.testing.assert_array_equal(pool.cur_lin.numpy(), reference.cur_lin.numpy())
    # every drawn landmark is a row of it, keyed by its goal patch
    assert set(_lin(matches.goal_patch).tolist()) <= set(pool.goal_lin.numpy().tolist())
    # and the signals the study's gate reads are there
    assert set(pool.signals()) >= {"margin", "ratio", "n_lookalikes"}


def test_the_exported_pool_is_cleared_where_there_is_none(goal_tokens, cur_tokens):
    """A reset drops it, and the one-way tracker has no pool to export."""
    backbone = _TokenBackbone({0: goal_tokens, 1: cur_tokens})
    m = DenseMatcher(num_landmarks=24, seed=0, backbone=backbone, tie_order="legacy",
                     frozen_landmarks=True, keep_pool=True)
    m.set_episode("mug1", 7)
    m.reset(_TokenBackbone.frame(0))
    m.match(_TokenBackbone.frame(1))
    assert m.last_pool is not None
    m.match(_TokenBackbone.frame(1))          # now pinned: a tracker, not a pool
    assert m.last_pool is None
    m.reset(_TokenBackbone.frame(0))
    assert m.last_pool is None


def test_the_registry_passes_keep_pool_through(goal_tokens, cur_tokens):
    cfg = {"seed": 0, "matcher": {"variant": "mutual_nn_all", "num_landmarks": POOL_ALL,
                                  "keep_pool": True, "tie_order": "legacy",
                                  "device": "cpu"},
           "stop": {"rule": "cap_only"}, "scoring": {"success_rule": ["predicate_a"]}}
    m = build_matcher(cfg, backbone=_TokenBackbone({0: goal_tokens, 1: cur_tokens}))
    assert isinstance(m, DenseMatcher) and m.keep_pool is True
