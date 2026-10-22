# vs-sim copy: vssim/baselines/match.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Explicit matching: goal tokens x live tokens -> landmarks (unit U2).

Mutual nearest-neighbour matching with the cyclic-consistency filter, the score
threshold, the seeded landmark draw, and the per-match significance signals
(margin / Lowe ratio / softmax entropy / look-alike count) the head and the
controller weight by. **One matcher serves training and deployment** - the old
repo had two and they had to be kept in step by hand.

Ported from (``../genvs-sim`` @ tag ``legacy-pre-cleanup`` = commit ``905d42b``)
-------------------------------------------------------------------------------
- ``_vitvs_vendor/.../features/matcher.py::find_correspondences_batch``
  (L122-258) -> :func:`mutual_nn_match` + :meth:`EpisodeDraw.draw`. Verbatim
  arithmetic: ``chunk_cosine_sim``, nearest neighbour both ways, round-trip
  distance in grid cells, shift by the minimum and divide by the maximum
  ``+ 1e-8`` (or all-1.0 when that maximum is under ``1e-6`` - identical
  frames), descending sort, the ``threshold - 1e-6`` epsilon at ``>= 0.99``, the
  goal patch mask, then ``torch.randperm`` over **all** survivors sliced to
  ``num_pairs`` - random from the whole filtered pool, not top-K, re-drawn every
  frame (``../docs/paper3_reset_2026-09-02.md`` section 2.1 item 2).
- ``src/control/matchers/vit_dense.py`` -> :class:`DenseMatcher`: goal cached
  once at ``reset`` (L368-443) with the pixel-mask -> patch-mask conversion
  (``_patch_mask_from_pixels``, L672-695); encode -> match -> patch-to-pixel per
  frame (L445-635); the seeded per-episode subsample (``set_subsample_seed``
  L231-247, ``_call_find_correspondences`` L345-366); ``emit_margins``
  (L589-606); the full-pool request ``FREEZE_ALL_PAIRS`` (L252) ->
  :data:`POOL_ALL`; the ``pin_first_frame`` (``_fs``) pin and its one-way
  tracker (L296-343) -> ``frozen_landmarks``; the ``_k<N>`` landmark-count path
  (``baseline_check.py`` L1978-1983); the two-hop patch-centre mapping
  ``_patch_to_full_image`` (L641-654).
- ``significance.py::margins_for_matches`` (the deployed, unrounded margin) and
  ``eval/n2b_significance.py::row_significance`` (margin, Lowe ratio, softmax
  entropy at tau 0.05/0.1, look-alike count) -> :func:`margins_for_matches` and
  :func:`row_signals`, vectorised. The signal *computation* is ported; none of
  the selection built on it is.
- ``src/refine/coarse_matcher.py`` - the TRAIN-side mirror, **folded in here**:
  training (U5) calls :func:`mutual_nn_match` and stops, deployment calls it and
  then draws. Its inlined copy of ``margins_for_matches`` existed only to avoid a
  ``control -> refine -> control`` import cycle (``coarse_matcher.py:49-57``);
  with one matcher module there is no cycle and no second copy - **resolved**.
- ``matchers/base.py`` -> :class:`CorrespondenceMatcher` and the coordinate
  contract (full-image pixels, ``(column, row)``); ``matchers/__init__.py``
  ``build_matcher`` -> :func:`build_matcher`, re-keyed on the ``matcher.variant``
  enum of :mod:`vsbench.vs.bench.config`.

Deliberately dropped
--------------------
Every retired landmark-choice mechanism and its grammar - ``_sel`` / ``_sel2``
(``significance.goal_selfsim`` / ``selection_order`` / ``selection_order_spaced``),
``_dopt`` (all of ``matchers/dopt.py`` and the goal-depth sampling ``reset`` did
for it), ``_tsel`` / ``_tsel2`` (``matchers/refine.py:470-500``), ``_cons`` /
``_consr``. Five selection rules, 1.7-5.6 mm worse at landing; the record's
lesson is **weight, do not choose, do not reject**, so the signals come across
and the selectors do not (``../docs/paper3_reset_2026-09-02.md`` section 2.2).
Also dropped: ``freeze_landmarks()`` / ``unfreeze_landmarks()`` (the N1a
mid-episode ``frz`` stop mode, which pinned the FULL pool of whatever frame
armed it - only the stop-mechanics bench used it; ``frozen_landmarks`` keeps the
variant measured on the deploy chain); RANSAC
(``matchers/geometric.filter_by_ransac`` - off for every ViT/ART arm by owner
decision; U3 ports it as :func:`vsbench.vs.baselines.classical.filter_by_ransac`, the switch
the classical arms need); ``_rot*`` and the pre-align hooks (U8); and the rospy
stub + vendored ``sys.path`` injection (retired with U1). Two things that read
like omissions are simply elsewhere: the tiled ART-VS subclass is
:class:`vsbench.vs.baselines.tiling.TiledMatcher` (this module's :class:`DenseMatcher` is its
coarse phase), and the published DINOv2 descriptor chain is
:class:`vsbench.vs.baselines.features.PublishedDinoV2Backbone`, so ``vitvs_dinov2`` is this
matcher on that backbone and needs no code of its own.

Fences kept (each cost real debugging in the old repo)
------------------------------------------------------
1. **Register/CLS masking is upstream now.** The old ``_full_image_patch_mask``
   had to put ``False`` in front of the goal mask for register tokens, or
   position-free tokens became matchable. U1's ``encode`` strips them and
   *checks* that exactly ``grid_side**2`` remain, so :func:`mutual_nn_match` gets
   a square grid by construction and refuses anything else.
2. **The half-patch centre.** Patch (col, row) -> ``idx * (input_size / side) +
   patch_size / 2``, and only then -> full-image pixels. Dropping the
   ``+ patch_size/2`` biases every correspondence by half a patch. Hop 1 is
   :attr:`Matches.goal_xy` (float, so U5's sub-patch offsets add straight on),
   both hops are :meth:`Matches.to_rows`.
3. **The seeded draw is reproducibility, not content-determinism.** It pins
   *which* random 24 a re-run picks, so paired arms at one ``(scene, start)``
   compare identical landmark sets. It does not make the choice "the best 24" -
   choosing was tested five ways and lost. One draw per frame, including a frame
   whose pool is empty (``vit_dense.py:365``), so frame indices stay aligned
   across arms.
4. **The pool's tie order is a sort implementation detail.** At the fielded
   ``threshold=1.0`` every survivor scores exactly 1.0 (float32 rounds
   ``max_dist / (max_dist + 1e-8)`` to 1.0 - which is also why the vendored
   threshold carries a ``- 1e-6`` epsilon), so the vendored
   ``sort(descending=True)`` orders a block of **equal keys**, unspecified by
   PyTorch for ``stable=False`` and ordered differently by the CUDA kernel - and
   it is that order ``randperm`` indexes, i.e. it picks the 24 patches the loop
   steers on. :data:`TIE_ORDERS` makes the choice explicit and every
   :class:`Matches` records it. Same pairs either way, a different (equally
   random) 24. **Full record - which setting buys what, and which stored
   campaigns can and cannot be replayed: ``tests/fixtures/u2_fixture_provenance.md``.**

Golden tests
------------
``tests/fast/test_match.py`` against ``tests/fixtures/u2_matches_mug1.npz``
(produced once by the frozen old code - ``u2_fixture_provenance.md``): the
612-pair pool, its cyclic scores / similarities / margins, the exact 24 and 48
drawn pairs, the ``frozen_landmarks`` tracker output, the four signals, the seed
conventions and the registry. **The landmark count has no default** anywhere in
this module or the registry.
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

__all__ = [
    "DEFAULT_TIE_ORDER",
    "MISSING_DEVICE",
    "CorrespondenceMatcher",
    "DenseMatcher",
    "EpisodeDraw",
    "MATCHER_BUILDERS",
    "Matches",
    "POOL_ALL",
    "Pool",
    "TIE_ORDERS",
    "build_matcher",
    "device_of",
    "episode_seed",
    "margins_for_matches",
    "mutual_nn_match",
    "patch_mask_from_pixels",
    "row_signals",
]

#: What a matcher says when it is asked to build an encoder and no device was stated.
#: One text, because both encoder-owning matchers raise it (:class:`DenseMatcher` and
#: :class:`vsbench.vs.baselines.tiling.TiledMatcher`) and the loader refuses the same absence a
#: config earlier (:func:`vsbench.vs.bench.config.validate`). R22 correction 3.
MISSING_DEVICE = (
    "no device: the encoder's device is a stated setting with no default, and the "
    "config field is 'matcher.device' ('cuda' on the station, 'cpu' in the fast tier). "
    "The first station run flew the whole learned chain on the CPU at 0.21 s/step - "
    "twice its budget - because this defaulted to 'cpu' (R22 correction 3). Pass "
    "`device=` or inject a ready-made `backbone=`, whose own device then counts."
)

#: The ``num_landmarks`` request that means "every survivor": the draw clamps
#: with ``min(num_landmarks, pool_size)``, so anything past the pool size is the
#: whole pool. Verbatim from ``vit_dense.ViTDenseMatcher.FREEZE_ALL_PAIRS``.
POOL_ALL = 10**6

#: How the pool orders survivors that tie on the cyclic score - it decides which
#: landmarks a seeded draw returns. ``"stable"`` (the default, owner ruling
#: 2026-09-03) is ascending goal-patch index, identical on every torch build and
#: device; ``"legacy"`` is the byte-identical old call, which reproduces the
#: frozen ``genvs-sim`` fixture on CPU and is what the U2 parity tests pass.
#: Module docstring fence 4; full record in ``tests/fixtures/u2_fixture_provenance.md``.
TIE_ORDERS = ("stable", "legacy")
DEFAULT_TIE_ORDER = "stable"

#: Second-peak exclusion radius: the (2r+1)^2 block around the matched cell is
#: excluded before the second peak is taken, so sub-patch ambiguity between
#: neighbouring cells never counts as insignificance. The N2.b pre-registered
#: constant (``significance.EXCLUDE_RADIUS`` = ``n2b_significance.EXCLUDE_RADIUS``).
EXCLUDE_RADIUS = 1

#: Softmax temperatures of the row-entropy signal, primary first (N2.b ``TAUS``).
ENTROPY_TAUS = (0.05, 0.1)

#: Cells within ``s1 - band`` of the best count as look-alikes (N2.b ``LOOKALIKE_BAND``).
LOOKALIKE_BAND = 0.02


# --------------------------------------------------------------- helpers


def _as_grid(tokens: torch.Tensor, what: str) -> tuple[torch.Tensor, int]:
    """``(side, side, D)`` or ``(N, D)`` tokens -> ``(N, D)`` float32 and ``side``.

    A non-square token count means the caller did not hand over a patch grid
    (U1's ``encode`` strips and counts CLS + registers) - fence 1.
    """
    t = torch.as_tensor(tokens)
    if t.ndim < 2:
        raise ValueError(f"{what} must be (side, side, D) or (N, D), got {tuple(t.shape)}")
    t = t.reshape(-1, t.shape[-1]).float()
    n = int(t.shape[0])
    side = int(round(n**0.5))
    if side * side != n:
        raise ValueError(
            f"{what} has {n} tokens, which is not a square grid - CLS and register "
            "tokens must be stripped upstream (vsbench.vs.baselines.features.FrozenBackbone.encode)"
        )
    return t, side


def _excluded_block_mask(cur_lin: torch.Tensor, side: int, radius: int) -> torch.Tensor:
    """Bool ``[K, side*side]``: True inside each match's (2r+1)^2 neighbourhood.

    Vectorised twin of ``significance._excluded_block`` - same clipped block,
    ``(2r+1)^2`` scatters instead of a Python loop over K rows.
    """
    k = cur_lin.shape[0]
    out = torch.zeros(k, side, side, dtype=torch.bool, device=cur_lin.device)
    rows, cols = cur_lin // side, cur_lin % side
    idx = torch.arange(k, device=cur_lin.device)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = rows + dr, cols + dc
            ok = (rr >= 0) & (rr < side) & (cc >= 0) & (cc < side)
            out[idx[ok], rr[ok], cc[ok]] = True
    return out.reshape(k, side * side)


def _peaks(sim_rows, cur_lin, side: int, radius: int):
    """Validate, then ``(rows, cols, s1, second, outside)`` in float64.

    The shared half of :func:`margins_for_matches` and :func:`row_signals`:
    the matched cell's similarity and the best one outside its (2r+1)^2 block.
    """
    rows = torch.as_tensor(sim_rows).double()
    cols = torch.as_tensor(cur_lin, device=rows.device).reshape(-1).long()
    if rows.ndim != 2 or rows.shape[0] != cols.shape[0]:
        raise ValueError(f"sim_rows {tuple(rows.shape)} vs cur_lin {tuple(cols.shape)}")
    if rows.shape[1] != side * side:
        raise ValueError(f"row width {rows.shape[1]} != side^2 {side * side}")
    outside = ~_excluded_block_mask(cols, side, radius)
    return (
        rows,
        cols,
        rows.gather(1, cols.unsqueeze(1)).squeeze(1),
        torch.where(outside, rows, torch.full_like(rows, -np.inf)).max(dim=1).values,
        outside,
    )


def margins_for_matches(
    sim_rows: torch.Tensor,
    cur_lin: torch.Tensor,
    side: int,
    radius: int = EXCLUDE_RADIUS,
) -> torch.Tensor:
    """Margin per match: ``row[c] - max(row outside c's (2r+1)^2 block)``. ``[K]`` float64.

    The pre-registered PRIMARY margin, and the only significance statistic the
    deployed loop computes every frame: ``_mw`` feeds ``eps + margin`` into the
    weighted least squares (U4), the ``_m`` head variants take it as an input
    channel (U5). >= 0 for a mutual-best match, negative for a forced one
    (training's mismatch injection) - informative, kept. Vectorised port of
    ``significance.margins_for_matches``: float64 throughout, same clipped
    block, same ``s1 - max``, ``+inf`` on an all-excluded row.
    """
    _, _, s1, second, _ = _peaks(sim_rows, cur_lin, side, radius)
    return s1 - second


def row_signals(
    sim_rows: torch.Tensor,
    cur_lin: torch.Tensor,
    side: int,
    *,
    radius: int = EXCLUDE_RADIUS,
    taus: tuple[float, ...] = ENTROPY_TAUS,
    lookalike_band: float = LOOKALIKE_BAND,
) -> dict[str, np.ndarray]:
    """The four per-match significance signals of ``n2b_significance.row_significance``.

    ``sim_rows`` ``[K, side^2]`` is each match's goal patch against every current
    cell, ``cur_lin`` ``[K]`` the matched cell. Returns float64 arrays keyed
    ``s1``, ``second``, ``margin``, ``ratio``, ``n_lookalikes``, ``entropy_t05``,
    ``entropy_t1`` (the old key spelling: ``tau`` with ``"0."`` stripped).

    In plain words: **margin** = how much the best cell beats the best cell that
    is not next to it; **ratio** = Lowe-style second-over-best; **entropy** = how
    spread the whole row is (nats, full row, no exclusion - a monotone
    transform, so the probe's AUROC is unaffected); **n_lookalikes** = cells
    within ``lookalike_band`` of the best, outside the block. The N2.b probe
    measured the margin separating wrong matches at AUROC 0.756. Definitions are
    frozen by that pre-registration; the old function rounded to 5 decimals for
    its records, this one does not.
    """
    rows, _, s1, second, outside = _peaks(sim_rows, cur_lin, side, radius)
    out = {
        "s1": s1,
        "second": second,
        "margin": s1 - second,
        "ratio": torch.where(s1 > 0, second / s1, torch.full_like(s1, np.nan)),
        "n_lookalikes": ((rows >= (s1 - lookalike_band).unsqueeze(1)) & outside).sum(1).double(),
    }
    for tau in taus:
        p = torch.exp((rows - rows.max(dim=1, keepdim=True).values) / float(tau))
        p = p / p.sum(dim=1, keepdim=True)
        key = f"entropy_t{str(tau).replace('0.', '')}"
        out[key] = -(p * torch.log(p.clamp_min(1e-300))).sum(dim=1)
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in out.items()}


def patch_mask_from_pixels(
    pixel_mask: np.ndarray, *, input_size: int, patch_size: int
) -> np.ndarray:
    """Full-image pixel mask -> bool ``[side*side]`` over goal patches.

    Verbatim port of ``vit_dense._patch_mask_from_pixels`` (L672-695), which is
    itself the old ``ViTVSMethod._mask_from_pixels``: nearest-neighbour resize to
    the encoder input, ``> 127``, then a patch counts as eligible when **more
    than 20 %** of its pixels are set. The register/CLS prefix the old version
    concatenated is gone with U1's fence (the grid is square by construction).
    """
    import cv2

    m = np.asarray(pixel_mask)
    m_u8 = (m.astype(np.uint8) * 255) if m.dtype == bool else m
    side = int(input_size) // int(patch_size)
    if side * int(patch_size) != int(input_size):
        raise ValueError(f"input_size {input_size} is not {side} x patch_size {patch_size}")
    resized = cv2.resize(m_u8, (int(input_size), int(input_size)),
                         interpolation=cv2.INTER_NEAREST) > 127
    blocks = resized.reshape(side, int(patch_size), side, int(patch_size))
    return (blocks.mean(axis=(1, 3)) > 0.2).reshape(-1)


def episode_seed(scene: str, start_id: Any, base: int) -> int:
    """Per-``(scene, start)`` draw seed. Verbatim ``baseline_check.episode_subsample_seed``.

    ``crc32``, never ``hash()`` - ``PYTHONHASHSEED`` must not be able to change a
    campaign. Every arm at the same start gets the same value, which is what
    makes the paired reads ("A beats B by x mm") compare identical landmark sets.
    """
    return (int(base) + zlib.crc32(f"{scene}/{start_id}".encode())) % (2**31)


# ------------------------------------------------------------------ the pool


@dataclass(frozen=True)
class Pool:
    """Every mutual-nearest-neighbour pair that survives the cyclic filter.

    One frame's complete candidate set, in the order the draw indexes (module
    docstring, ``tie_order``). Row i of every array is the same pair.

    - ``goal_lin`` / ``cur_lin`` ``[P]`` int64 linear patch indices ``row*side+col``
    - ``cyclic`` ``[P]`` float32 normalised cyclic score. A perfect round trip
      (goal -> current -> goal lands back on the same patch) scores
      ``max_dist / (max_dist + 1e-8)``, which float32 rounds **to exactly 1.0** -
      that rounding is why the vendored threshold carries a ``- 1e-6`` epsilon
    - ``sim`` ``[P]`` float32 cosine similarity of the matched tokens
    - ``margin`` ``[P]`` float64, :func:`margins_for_matches`
    - ``sim_rows`` ``[P, side^2]`` float32, each pair's goal patch against every
      current cell: the rows the margin and :meth:`signals` read, and the ones
      U5's mismatch-injection re-scoring needs. A slice of the similarity matrix
      the match already computed, so they are free.
    - ``cyclic_all`` ``[side^2]`` float32, the score of *every* goal patch,
      survivor or not - how close the rejected ones were, and the strongest
      thing the parity fixture can pin.
    """

    goal_lin: torch.Tensor
    cur_lin: torch.Tensor
    cyclic: torch.Tensor
    sim: torch.Tensor
    margin: torch.Tensor
    sim_rows: torch.Tensor
    cyclic_all: torch.Tensor
    side: int
    threshold: float
    tie_order: str

    @property
    def size(self) -> int:
        return int(self.goal_lin.shape[0])

    def __len__(self) -> int:
        return self.size

    def take(self, order) -> Pool:
        """The same pool, rows reordered/subset by ``order`` (a long index tensor).

        Used by :meth:`EpisodeDraw.draw`, and by the parity test to re-order a
        canonical pool into the frozen record's order and check the draw
        arithmetic on its own.
        """
        idx = torch.as_tensor(order, device=self.goal_lin.device).reshape(-1).long()
        return Pool(
            goal_lin=self.goal_lin[idx],
            cur_lin=self.cur_lin[idx],
            cyclic=self.cyclic[idx],
            sim=self.sim[idx],
            margin=self.margin[idx],
            sim_rows=self.sim_rows[idx],
            cyclic_all=self.cyclic_all,
            side=self.side,
            threshold=self.threshold,
            tie_order=self.tie_order,
        )

    def signals(self) -> dict[str, np.ndarray]:
        """:func:`row_signals` over the whole pool."""
        return row_signals(self.sim_rows, self.cur_lin, self.side)


@dataclass(frozen=True)
class CountOnlyPool:
    """What :func:`mutual_nn_match` returns under ``with_rows=False``.

    A pool's *identity* without its evidence: how many pairs survived the cyclic
    filter and how similar they are on average - which is everything U8's bank score
    reads (``count`` and ``count_x_sim``) and nothing a controller could steer on. A
    separate type rather than a :class:`Pool` with empty arrays, so a caller that meant
    to steer fails on the attribute instead of on a silently empty solve.
    """

    size: int
    #: the mean matched similarity as a **0-dim tensor on the pool's device**. U8 scores
    #: twelve banks a frame and reads this once per bank; on CUDA each read is a device
    #: sync, so the caller stacks the twelve and syncs once (review R21 item 8). The
    #: ``mean_sim`` property below is the convenient single-use read. The *count* cannot
    #: be batched away: it is the shape of a boolean-indexed tensor, which is already a
    #: sync wherever the tokens live.
    mean_sim_t: torch.Tensor

    @property
    def mean_sim(self) -> float:
        return float(self.mean_sim_t.item())

    def __len__(self) -> int:
        return int(self.size)


def mutual_nn_match(
    goal_tokens: torch.Tensor,
    cur_tokens: torch.Tensor,
    *,
    threshold: float = 1.0,
    mask: torch.Tensor | None = None,
    tie_order: str = DEFAULT_TIE_ORDER,
    with_rows: bool = True,
) -> Pool | CountOnlyPool:
    """Mutual nearest neighbours with the cyclic-consistency filter -> :class:`Pool`.

    Pure function of the two token grids: no state, no RNG, no draw; CPU or GPU,
    wherever the tokens are. The one matcher - training (U5) calls it and stops,
    deployment calls it and then :meth:`EpisodeDraw.draw`. The arithmetic is
    ``find_correspondences_batch`` line for line (module docstring).

    - ``goal_tokens`` / ``cur_tokens``: ``(side, side, D)`` or ``(N, D)``,
      normalised or not - they are normalised again here, as
      ``chunk_cosine_sim`` did (idempotent on U1's tokens).
    - ``threshold``: the vendored ``distance_threshold`` on the normalised
      cyclic score. ``1.0``, the fielded value, is pure mutual nearest
      neighbour; ``>= 0.99`` gets the vendored ``- 1e-6`` epsilon because a
      perfect score can come back as 0.9999999 in float32.
    - ``mask``: optional bool eligibility over **goal** patches, ``[side, side]``
      or ``[N]``; ``None`` (fielded) means every patch is eligible.
    - ``tie_order``: ``"stable"`` (default) or ``"legacy"``, recorded on the
      :class:`Pool` - module docstring, fence 4.
    - ``with_rows``: ``False`` returns a :class:`CountOnlyPool` - the same survivors,
      counted, without the ``[P, N]`` evidence block or the margins. It exists for U8's
      bank selection, where eleven of twelve pools are scored and thrown away (review
      R19 item 9); every path that STEERS on a pool asks for the full one.
    """
    if tie_order not in TIE_ORDERS:
        raise ValueError(f"tie_order must be one of {TIE_ORDERS}, got {tie_order!r}")
    dg, side = _as_grid(goal_tokens, "goal_tokens")
    dc, side_cur = _as_grid(cur_tokens, "cur_tokens")
    if side_cur != side:
        raise ValueError(f"goal grid is {side}x{side} but current grid is {side_cur}x{side_cur}")
    if dc.shape[-1] != dg.shape[-1]:
        raise ValueError(f"descriptor dims differ: {dg.shape[-1]} vs {dc.shape[-1]}")
    n = side * side
    dc = dc.to(dg.device)

    normalize = torch.nn.functional.normalize
    sims = normalize(dg, dim=-1) @ normalize(dc, dim=-1).T  # [N, N] float32
    sim_goal, nn_goal = sims.max(dim=1)  # per goal patch: its best current patch
    _, nn_cur = sims.max(dim=0)  # per current patch: its best goal patch
    cyclical = nn_cur[nn_goal]  # goal -> current -> goal

    own = torch.arange(n, device=sims.device)
    delta = torch.stack(
        [(cyclical % side) - (own % side), (cyclical // side) - (own // side)], dim=-1
    ).float()
    scores = -torch.norm(delta, p=2, dim=-1)
    scores = scores - scores.min()
    span = scores.max()
    scores = torch.ones_like(scores) if float(span) < 1e-6 else scores / (span + 1e-8)

    sorted_scores, order = scores.sort(descending=True, stable=(tie_order == "stable"))
    eff_threshold = threshold - 1e-6 if threshold >= 0.99 else threshold
    keep = sorted_scores >= eff_threshold
    if mask is not None:
        keep = keep & torch.as_tensor(mask, device=sims.device).reshape(-1)[order].bool()
    pool_idx = order[keep]

    cur_lin = nn_goal[pool_idx]
    if not with_rows:
        # The count-only path (U8's bank selection, review R19 item 9): everything above
        # is the pool's IDENTITY, and everything below is its per-row evidence block -
        # a [P, N] slice plus the margin's two peak passes over it. A bank that is not
        # selected never steers on anything, so it needs the identity and nothing else.
        n = int(pool_idx.shape[0])
        mean = (sim_goal[pool_idx].mean() if n
                else torch.zeros((), dtype=sim_goal.dtype, device=sim_goal.device))
        return CountOnlyPool(size=n, mean_sim_t=mean)
    sim_rows = sims[pool_idx]
    return Pool(
        goal_lin=pool_idx,
        cur_lin=cur_lin,
        cyclic=scores[pool_idx],
        sim=sim_goal[pool_idx],
        margin=margins_for_matches(sim_rows, cur_lin, side),
        sim_rows=sim_rows,
        cyclic_all=scores,
        side=side,
        threshold=float(threshold),
        tie_order=tie_order,
    )


# ------------------------------------------------------------- the landmarks


@dataclass
class Matches:
    """The landmarks one frame steers on - the matcher's output contract.

    Row i of every array is the same landmark. Patch indices are ``(column,
    row)`` in the token grid (the vendored order); the ``_xy`` properties are
    patch **centres in encoder-input pixels** - the "512-space" the head
    predicts its offsets in - float and sub-pixel ready, and :meth:`to_rows`
    does the second hop to full-image pixels for the controller. U5 fills
    ``offsets_patch`` (patch units, bounded +-2) and ``trust``; :attr:`cur_xy`
    adds the offset when it is there, so U4 is identical with and without the
    head. ``extras`` carries anything else row-aligned.
    """

    goal_patch: np.ndarray  # [K, 2] int64 (col, row)
    cur_patch: np.ndarray  # [K, 2] int64 (col, row)
    sim: np.ndarray  # [K] float64 cosine similarity
    cyclic: np.ndarray  # [K] float64 cyclic score (NaN under frozen_landmarks)
    margin: np.ndarray  # [K] float64
    side: int
    pool_size: int
    num_landmarks: int
    draw_seed: int | None
    tie_order: str
    input_size: int | None = None
    patch_size: int | None = None
    image_wh: tuple[int, int] | None = None
    frozen: bool = False
    offsets_patch: np.ndarray | None = None  # [K, 2] float, filled by U5
    #: the goal side's sub-cell remainder. ``None`` on every dense arm - a goal
    #: landmark IS a patch there - and filled by :meth:`from_pixel_pairs` for a
    #: matcher whose goal answer is a pixel (U3's classical arms, whose OpenCV
    #: keypoints are sub-pixel on both sides).
    goal_offsets_patch: np.ndarray | None = None  # [K, 2] float
    trust: np.ndarray | None = None  # [K] float, filled by U5
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        k = len(self.goal_patch)
        for name in ("cur_patch", "sim", "cyclic", "margin"):
            if len(getattr(self, name)) != k:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} rows for {k} matches - "
                    "every array here is row-aligned with goal_patch"
                )

    def __len__(self) -> int:
        return len(self.goal_patch)

    # ------------------------------------------------------------ geometry
    def _require(self, *names: str) -> None:
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ValueError(
                f"Matches.{', '.join(missing)} is unset: these matches came straight "
                "from mutual_nn_match (the training path, patch indices only). "
                "DenseMatcher fills the encoder geometry and the image size."
            )

    @property
    def goal_xy(self) -> np.ndarray:
        """``[K, 2]`` float64 goal patch centres in encoder-input pixels.

        Plus ``goal_offsets_patch`` when a matcher filled it (U3's classical arms),
        exactly as :attr:`cur_xy` adds the head's.
        """
        self._require("input_size", "patch_size")
        xy = self._centres(self.goal_patch)
        if self.goal_offsets_patch is not None:
            xy = xy + np.asarray(self.goal_offsets_patch, np.float64) * (
                self.input_size / self.side)
        return xy

    @property
    def cur_xy(self) -> np.ndarray:
        """``[K, 2]`` float64 current patch centres in encoder-input pixels.

        Plus ``offsets_patch * (input_size / side)`` when the head has filled
        them in - so this is the *refined* position whenever there is one.
        """
        self._require("input_size", "patch_size")
        xy = self._centres(self.cur_patch)
        if self.offsets_patch is not None:
            xy = xy + np.asarray(self.offsets_patch, np.float64) * (self.input_size / self.side)
        return xy

    def _centres(self, patch: np.ndarray) -> np.ndarray:
        """``vit_dense._patch_to_full_image`` hop 1: centre in encoder-input pixels."""
        return np.asarray(patch, np.float64) * (self.input_size / self.side) + (
            self.patch_size / 2.0
        )

    @staticmethod
    def _split(px, *, side: int, input_size: int, patch_size: int,
               image_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Full-image pixels -> ``(integer cell, sub-cell remainder)``, exactly.

        The remainder is taken **after** the clip to the lattice, so the pair
        reconstructs the input to the last bit even for a point that fell off the
        grid. The one place that split lives.
        """
        w, h = int(image_wh[0]), int(image_wh[1])
        p = np.asarray(px, np.float64).reshape(-1, 2)
        p_in = p * np.asarray([input_size / w, input_size / h], np.float64)
        exact = (p_in - patch_size / 2.0) / (input_size / side)
        patch = np.clip(np.round(exact), 0, side - 1).astype(np.int64)
        return patch, exact - patch

    @classmethod
    def from_pixels(cls, goal_patch, cur_px, *, side: int, input_size: int,
                    patch_size: int, image_wh: tuple[int, int], **fields) -> Matches:
        """Build from **full-image** current pixels - the exact inverse of :meth:`to_rows`.

        A matcher whose answer is a pixel rather than a cell (an oracle's GT
        reprojection, the mock plane's truth, U7b's snapped positions) still has to
        hand the controller the shape U5's head produces: an integer ``cur_patch``
        plus the sub-patch remainder in ``offsets_patch``. The goal side stays a
        cell here; :meth:`from_pixel_pairs` is the both-sides-are-pixels twin.
        """
        patch, offsets = cls._split(cur_px, side=side, input_size=input_size,
                                    patch_size=patch_size, image_wh=image_wh)
        return cls(goal_patch=np.asarray(goal_patch, np.int64).reshape(-1, 2),
                   cur_patch=patch, side=int(side), input_size=int(input_size),
                   patch_size=int(patch_size),
                   image_wh=(int(image_wh[0]), int(image_wh[1])),
                   offsets_patch=offsets, **fields)

    @classmethod
    def from_pixel_pairs(cls, goal_px, cur_px, *, side: int, input_size: int,
                         patch_size: int, image_wh: tuple[int, int], **fields) -> Matches:
        """Both sides given as **full-image** pixels (U3).

        A classical detector has no patch grid at all and its keypoints are
        sub-pixel on the goal side too, so quantising the goal to a cell centre
        would move every published baseline landmark by up to half a cell. Both
        sides go through :meth:`_split`, so ``to_rows()`` returns the two pixel
        arrays it was given.
        """
        goal_patch, goal_off = cls._split(goal_px, side=side, input_size=input_size,
                                          patch_size=patch_size, image_wh=image_wh)
        patch, offsets = cls._split(cur_px, side=side, input_size=input_size,
                                    patch_size=patch_size, image_wh=image_wh)
        return cls(goal_patch=goal_patch, cur_patch=patch, side=int(side),
                   input_size=int(input_size), patch_size=int(patch_size),
                   image_wh=(int(image_wh[0]), int(image_wh[1])),
                   offsets_patch=offsets, goal_offsets_patch=goal_off, **fields)

    def to_rows(self) -> dict[str, np.ndarray]:
        """Row-aligned arrays in **full-image pixels** ``(column, row)``, for U4.

        The contract of ``matchers/base.py``: a matcher working at a reduced ViT
        resolution maps back itself, and the controller applies intrinsics with
        no rescaling. Hop 2 is ``_patch_to_full_image``'s anisotropic
        ``* (img_w / input_size, img_h / input_size)`` - the 1920x1080 frame is
        squashed to 512x512, so the axes really do scale differently.
        """
        self._require("input_size", "patch_size", "image_wh")
        w, h = self.image_wh
        scale = np.asarray([w / self.input_size, h / self.input_size], np.float64)
        rows = {
            "pts_goal": self.goal_xy * scale,
            "pts_cur": self.cur_xy * scale,
            "conf": np.asarray(self.sim, np.float64),
            "margin": np.asarray(self.margin, np.float64),
            "goal_patch": np.asarray(self.goal_patch, np.int64),
            "cur_patch": np.asarray(self.cur_patch, np.int64),
        }
        if self.trust is not None:
            rows["trust"] = np.asarray(self.trust, np.float64)
        return rows


def _to_patch(lin: torch.Tensor, side: int) -> np.ndarray:
    """Linear patch index -> ``(col, row)``, the vendored ``_to_cartesian`` order."""
    a = torch.as_tensor(lin).reshape(-1).long().cpu().numpy()
    return np.column_stack([a % side, a // side]).astype(np.int64)


def track_pinned(pinned_desc: torch.Tensor, cur_tokens: torch.Tensor):
    """One-way nearest neighbour of pinned goal descriptors over a current token grid.

    ``vit_dense._match_frozen``'s arithmetic, extracted so the frozen-landmark variant
    has ONE implementation (CLAUDE.md rule 2). The ``[K, N]`` rows this path computes
    anyway **are** the margin rows, so the margin definition and the 3x3 exclusion are
    the mutual path's, with no extra token pass.

    ``pinned_desc`` is ``[K, D]``, already L2-normalised; ``cur_tokens`` is the current
    grid. Returns ``(best_lin [K], sim [K], margin [K] float64, side)``.
    """
    cur, side = _as_grid(cur_tokens, "cur_tokens")
    cur = torch.nn.functional.normalize(cur, p=2, dim=-1).to(pinned_desc.device)
    with torch.no_grad():
        sims = pinned_desc @ cur.T                        # [K, N]
        best = sims.argmax(dim=1)
        sim = sims.gather(1, best.reshape(-1, 1)).reshape(-1)
        margin = margins_for_matches(sims, best, side)
    return best, sim, margin, side


class EpisodeDraw:
    """The seeded per-episode landmark subsample - state, not policy.

    The vendored ``find_correspondences_batch`` drew from torch's **global** RNG,
    so the same episode re-run in a fresh process picked different landmarks
    (measured: the same baseline arm re-ran at 5.33 vs 6.22 mm median on
    byte-identical starts). Here the draw is a function of
    ``(scene, start_id, seed)`` and the frame number after :meth:`reset` and
    nothing else: draw ``i`` runs under a private generator seeded
    ``(episode_seed + i) % 2**31`` - the old ``_call_find_correspondences``
    arithmetic, on an explicit ``torch.Generator`` instead of a fork of the
    process RNG, so no global state is touched and the device does not matter.

    ``num_landmarks`` is required at :meth:`draw` and has no default. One
    permutation per frame, sliced - so a 48-landmark arm's draw is a **superset**
    of the 24-landmark arm's at the same start (the N3.7d convention that keeps a
    landmark-count sweep paired).
    """

    def __init__(self, scene: str, start_id: Any, seed: int) -> None:
        if seed is None:
            raise ValueError("EpisodeDraw needs a seed - it has no default (config field 'seed')")
        self.scene = scene
        self.start_id = start_id
        self.base_seed = int(seed)
        self.episode_seed = episode_seed(scene, start_id, self.base_seed)
        self.draw_index = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"EpisodeDraw(scene={self.scene!r}, start_id={self.start_id!r}, "
            f"seed={self.base_seed}, episode_seed={self.episode_seed}, "
            f"draw_index={self.draw_index})"
        )

    def reset(self) -> None:
        """Back to frame 0 of the episode. Called by :meth:`DenseMatcher.reset`."""
        self.draw_index = 0

    def frame_seed(self, draw_index: int | None = None) -> int:
        idx = self.draw_index if draw_index is None else int(draw_index)
        return (self.episode_seed + idx) % (2**31)

    def draw(self, pool: Pool, num_landmarks: int) -> Matches:
        """``num_landmarks`` landmarks drawn at random from the **whole** pool.

        Not the top-K: that is the published selection, and every rule that
        replaced it harmed landing (module docstring, "Deliberately dropped").
        An empty pool returns empty :class:`Matches` and **still consumes the
        frame's seed**, as the old code did (fence 3).
        """
        if num_landmarks is None:
            raise ValueError(
                "num_landmarks is required and has no default - the config field "
                "'matcher.num_landmarks' states it (vsbench.vs.bench.config)"
            )
        num_landmarks = int(num_landmarks)
        if num_landmarks < 1:
            raise ValueError(f"num_landmarks must be >= 1, got {num_landmarks}")
        seed = self.frame_seed()
        generator = torch.Generator().manual_seed(seed)
        take = min(num_landmarks, pool.size)
        perm = torch.randperm(pool.size, generator=generator)[:take]
        # One draw per FRAME, empty pool included: the old
        # `_call_find_correspondences` advanced `_draw_idx` before it could know
        # whether the frame yielded anything (vit_dense.py:365), so a frame that
        # matched nothing still consumed a seed and later frames stayed aligned
        # across arms. Fence 3.
        self.draw_index += 1
        drawn = pool.take(perm.to(pool.goal_lin.device))
        return Matches(
            goal_patch=_to_patch(drawn.goal_lin, pool.side),
            cur_patch=_to_patch(drawn.cur_lin, pool.side),
            sim=drawn.sim.detach().cpu().numpy().astype(np.float64),
            cyclic=drawn.cyclic.detach().cpu().numpy().astype(np.float64),
            margin=drawn.margin.detach().cpu().numpy().astype(np.float64),
            side=pool.side,
            pool_size=pool.size,
            num_landmarks=num_landmarks,
            draw_seed=seed,
            tie_order=pool.tie_order,
        )


# ------------------------------------------------------------- the interface


class CorrespondenceMatcher(ABC):
    """Turns (goal view, current view) into :class:`Matches`.

    The pluggable half of a VS method: a matcher turns an image pair into
    landmarks, :mod:`vsbench.vs.baselines.control` turns landmarks into a twist. Ported from
    ``matchers/base.py`` with three changes: the return type is :class:`Matches`
    (which carries the patch grid the head needs, not only pixels);
    ``set_episode`` is part of the interface, because the draw is a function of
    ``(scene, start, seed)``, and it replaces the old ``set_subsample_seed``; and
    the old duck-typed ``set_intrinsics`` hook (``matcher_ibvs.py:253-256``) is
    **not** here - its only consumer was the dropped ``_dopt`` rule, so no
    matcher in this repo needs the camera. Intrinsics belong to
    :mod:`vsbench.vs.baselines.control`; if a future matcher needs them, add the hook back
    with the consumer that justifies it.
    """

    #: short stable identifier used in configs, telemetry and result tables
    name: str = "matcher"
    #: True on a matcher whose :meth:`reset` USES ``goal_mask`` (the ViT matchers);
    #: `vsbench.vs.bench.arms.build_arm` refuses `matcher.goal_mask: objects` on any other,
    #: so a masked label never sits on a run that ignored the mask.
    honours_goal_mask: bool = False

    @abstractmethod
    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        """Cache what the goal view needs. Once per episode, before any :meth:`match`."""

    @abstractmethod
    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        """Match the current view against the cached goal view.

        ``current_mask`` is the second **positional** argument the old
        ``matcher_ibvs.step`` passed (``matcher_ibvs.py:259``); the shape is kept
        so U4/U7 can call exactly as before. This suite passes ``None``, and an
        implementation that cannot honour a mask must refuse it loudly.

        Returns an **empty** :class:`Matches` (K = 0) rather than raising when
        matching fails; the controller commands a zero twist on that.
        """

    def set_episode(self, scene: str, start_id: Any) -> None:  # noqa: B027
        """Name the episode, so the draw is reproducible and shared across arms.

        Optional on purpose, not abstract: a matcher without a random draw (the
        classical arms of U3, the oracles of U7) has nothing to seed. It replaces
        the old ``set_subsample_seed``, so **every wrapper must delegate it** -
        U5's deploy wrapper and U7's oracle wrapper, exactly as the old wrappers
        delegated ``set_subsample_seed``.
        """

    def close(self) -> None:  # noqa: B027 - optional hook, most matchers hold nothing
        """Release heavy resources. Default: nothing."""


class DenseMatcher(CorrespondenceMatcher):
    honours_goal_mask = True
    """The fielded chain: frozen ViT tokens -> mutual pool -> seeded draw.

    ``reset`` encodes the goal once per episode and caches its tokens (the old
    ``vit_dense.reset``); every ``match`` encodes the current frame, runs
    :func:`mutual_nn_match` and draws ``num_landmarks`` landmarks. Margins are
    **always** emitted: they are a controller weight and a head input channel and
    cost one gather over rows the match already computed - the old
    ``emit_margins`` flag existed only because ``_mw`` was an experiment.

    ``frozen_landmarks=True`` is the N3.6 ``_fs`` variant: frame 1 runs the
    published path unchanged and then **pins** exactly the landmarks it drew;
    every later frame stops re-drawing and tracks those pinned goal patches by
    one-way nearest neighbour. Not a default, and it should not become one - it
    measured **-2.26 mm** paired in N3.6 and **+1.41 mm worse parking** on
    hollywood in N3.8. It is kept so protocol v2 has the per-frame-churn axis
    available as a declared variant, not because it helps.

    ``num_landmarks`` is **required, no default** (:data:`POOL_ALL` = the whole
    pool); ``seed`` is the base of ``episode_seed(scene, start, seed)``;
    ``backbone`` is a :class:`vsbench.vs.baselines.features.FrozenBackbone` or ``None`` to
    build one on ``device``, which is then **required and has no default**
    (:data:`MISSING_DEVICE`) - injecting a backbone instead lets arms share one
    loaded encoder and lets a test drive the matcher on stored tokens, and the
    device is then the injected encoder's own.
    """

    def __init__(
        self,
        *,
        num_landmarks: int | None = None,
        seed: int | None = None,
        backbone=None,
        backbone_name: str = "dinov3_vits16",
        device: str | None = None,
        threshold: float = 1.0,
        frozen_landmarks: bool = False,
        tie_order: str = DEFAULT_TIE_ORDER,
        mask: np.ndarray | None = None,
        keep_pool: bool = False,
    ) -> None:
        if num_landmarks is None:
            raise ValueError(
                "DenseMatcher needs num_landmarks and it has no default - the "
                "config field 'matcher.num_landmarks' states it, and a run that "
                "does not say how many landmarks it steered on is the incident "
                "this repository exists to make impossible (CLAUDE.md rule 3)"
            )
        num_landmarks = int(num_landmarks)
        if num_landmarks < 1:
            raise ValueError(f"num_landmarks must be >= 1, got {num_landmarks}")
        if tie_order not in TIE_ORDERS:
            raise ValueError(f"tie_order must be one of {TIE_ORDERS}, got {tie_order!r}")

        if backbone is None:
            if device is None:
                raise ValueError(MISSING_DEVICE)
            from vsbench.vs.baselines.features import build_backbone

            # `build_backbone`, not `FrozenBackbone`: `dinov2_vits14` is the published
            # torch.hub + log-binned chain (U3) and loads through a different class.
            backbone = build_backbone(backbone_name, device=device)
        self.backbone = backbone
        self.num_landmarks = num_landmarks
        self.seed = None if seed is None else int(seed)
        self.threshold = float(threshold)
        self.frozen_landmarks = bool(frozen_landmarks)
        self.tie_order = tie_order
        self.mask = None if mask is None else np.asarray(mask).astype(bool)
        #: the mask actually used: `mask` ANDed with `reset(goal_mask=...)`.
        self._mask = self.mask
        self.name = (
            f"dense_{backbone.name}_"
            f"{'frozen_landmarks' if frozen_landmarks else 'mutual_nn'}"
        )
        #: OFF by default, and off is the fielded path (the matching study's T3).
        #: `match()` drops the `Pool` it built; an offline read that wants the whole
        #: pool - its `sim_rows`, its margins, the look-alike counts - would otherwise
        #: have to re-run the matcher on the same tokens. With this on, the pool is
        #: parked on :attr:`last_pool` and NOTHING else changes: the returned
        #: :class:`Matches` and :meth:`record_fields` are byte-identical either way
        #: (`tests/fast/test_match.py::test_keeping_the_pool_changes_nothing_on_the_fielded_path`),
        #: which is why it is not a record field - it is an export, not a setting a
        #: number depends on.
        self.keep_pool = bool(keep_pool)
        #: the last frame's :class:`Pool` when `keep_pool` is on, else always `None`.
        self.last_pool: Pool | None = None
        self.draw: EpisodeDraw | None = None
        self.goal_tokens: torch.Tensor | None = None
        self.cur_tokens: torch.Tensor | None = None  # U5 reads this for its windows
        self.image_wh: tuple[int, int] | None = None
        self._pinned_lin: torch.Tensor | None = None
        self._pinned_desc: torch.Tensor | None = None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"DenseMatcher(num_landmarks={self.num_landmarks}, seed={self.seed}, "
            f"threshold={self.threshold}, frozen_landmarks={self.frozen_landmarks}, "
            f"keep_pool={self.keep_pool}, "
            f"tie_order={self.tie_order!r}, backbone={self.backbone.name!r})"
        )

    # ------------------------------------------------------------- episode
    def set_episode(self, scene: str, start_id: Any) -> None:
        """Name the episode; every arm at this ``(scene, start)`` then draws alike."""
        if self.seed is None:
            raise ValueError(
                "DenseMatcher was built without a seed - the config field 'seed' "
                "has no default (vsbench.vs.bench.config)"
            )
        self.draw = EpisodeDraw(scene, start_id, self.seed)

    def reset(self, goal_rgb: np.ndarray, goal_depth=None, goal_mask=None) -> None:
        """Encode and cache the goal view; clear any pin; restart the draw sequence.

        ``goal_mask`` is the old pixel-space goal mask and is **honoured**, via
        :func:`patch_mask_from_pixels`, exactly as ``vit_dense.reset`` did - it
        restricts which goal patches may match, and it is ANDed with the
        constructor's patch-grid ``mask``. This suite passes ``None``, which
        selects every patch; note that ``None`` here does **not** mean the old
        ``ViTVSMethod`` depth-Otsu mask, which would have made matching silently
        object-centric. ``goal_depth`` is accepted for the
        :class:`CorrespondenceMatcher` contract and unused (its only old consumer
        was the dropped ``_dopt`` rule).
        """
        if self.draw is None:
            raise RuntimeError(
                "call set_episode(scene, start_id) before reset() - the landmark "
                "draw is a function of (scene, start, seed) and nothing else"
            )
        h, w = goal_rgb.shape[:2]
        self.image_wh = (int(w), int(h))
        self._mask = self.mask
        if goal_mask is not None:
            from_pixels = patch_mask_from_pixels(
                goal_mask,
                input_size=self.backbone.input_size,
                patch_size=self.backbone.patch_size,
            )
            self._mask = from_pixels if self._mask is None else (
                self._mask.reshape(-1) & from_pixels)
        self.goal_tokens = self.backbone.encode(goal_rgb)
        self.cur_tokens = None
        self.last_pool = None
        self._pinned_lin = None
        self._pinned_desc = None
        self.draw.reset()

    # --------------------------------------------------------------- match
    def match(self, current_rgb: np.ndarray, current_mask: np.ndarray | None = None) -> Matches:
        """Encode the frame, match it against the goal, draw the landmarks.

        ``current_mask`` keeps the old two-positional-argument call shape
        (``matcher_ibvs.py:259``) and must be ``None``: the old dense matcher
        accepted it and ignored it, which is exactly the silent behaviour this
        repository refuses, so a non-``None`` mask is refused loudly instead.
        """
        if current_mask is not None:
            raise ValueError(
                "DenseMatcher does not support a per-frame current_mask - the old "
                "dense matcher accepted the argument and silently ignored it "
                "(vit_dense.match). Mask the GOAL side instead: mask= at "
                "construction, or reset(goal_mask=...)"
            )
        if self.goal_tokens is None:
            raise RuntimeError("reset() must be called before match()")
        self.cur_tokens = self.backbone.encode(current_rgb)
        if self._pinned_lin is not None:
            self.last_pool = None   # a one-way tracker has no pool to export
            return self._match_pinned()

        pool = mutual_nn_match(
            self.goal_tokens,
            self.cur_tokens,
            threshold=self.threshold,
            mask=self._mask,
            tie_order=self.tie_order,
        )
        if self.keep_pool:
            self.last_pool = pool
        # draw() runs even on an empty pool: it consumes the frame seed, as the
        # old code did, so a frame that matched nothing does not desync the arms.
        matches = self.draw.draw(pool, self.num_landmarks)
        self._stamp(matches)
        if not len(matches):
            matches.extras["reason"] = "no correspondences"  # the old debug field
            return matches
        if self.frozen_landmarks:
            self.pin_landmarks(matches.goal_patch[:, 1] * pool.side + matches.goal_patch[:, 0])
        return matches

    def pin_landmarks(self, goal_lin) -> None:
        """Pin these goal patches; every later :meth:`match` tracks them one-way.

        What ``frozen_landmarks`` does after frame 1 (``_activate_freeze``),
        exposed so a test or U7 can pin an explicit set.
        """
        if self.goal_tokens is None:
            raise RuntimeError("reset() must be called before pin_landmarks()")
        lin = torch.as_tensor(np.asarray(goal_lin).reshape(-1), dtype=torch.long)
        grid = self.goal_tokens.reshape(-1, self.goal_tokens.shape[-1])
        self._pinned_lin = lin
        self._pinned_desc = torch.nn.functional.normalize(grid[lin].float(), p=2, dim=-1)

    def _match_pinned(self) -> Matches:
        """One-way nearest neighbour of the pinned goal patches over the current tokens.

        The arithmetic is :func:`track_pinned`, shared with U8's banked twin so the
        tracker has one implementation. The cyclic score is ``NaN`` on purpose - a
        one-way tracker has none.
        """
        best, sim, margin, side = track_pinned(self._pinned_desc, self.cur_tokens)
        k = int(best.shape[0])
        matches = Matches(
            goal_patch=_to_patch(self._pinned_lin, side),
            cur_patch=_to_patch(best, side),
            sim=sim.detach().cpu().numpy().astype(np.float64),
            cyclic=np.full(k, np.nan),
            margin=margin.detach().cpu().numpy(),
            side=side,
            pool_size=k,
            num_landmarks=self.num_landmarks,
            draw_seed=None if self.draw is None else self.draw.episode_seed,
            tie_order=self.tie_order,
            frozen=True,
        )
        self._stamp(matches)
        return matches

    def record_fields(self) -> dict:
        """Published knobs of the dense arm - never a measurement.

        ``prealign`` is the 4-way pre-alignment, stated on **every** ViT row - ``None``
        when it did not run - because protocol v2 section 4 asks for its presence to be
        stated in every table and the published numbers effectively start roll-aligned.
        A row that simply omitted the field would read as "not applicable" rather than
        "off". U8 attaches a :class:`vsbench.vs.baselines.prealign.Prealign4Way` to this attribute
        when a config's ``rotation:`` block asks for it; the ``label`` lookup keeps this
        module free of an import it would otherwise need for one string.
        :class:`vsbench.vs.baselines.tiling.TiledMatcher` states the same thing for ART-VS.
        """
        return {"backbone": self.backbone.name,
                "matcher_device": device_of(self.backbone),
                "prealign": getattr(getattr(self, "prealign", None), "label", None)}

    def _stamp(self, matches: Matches) -> None:
        matches.input_size = int(self.backbone.input_size)
        matches.patch_size = int(self.backbone.patch_size)
        matches.image_wh = self.image_wh


# ---------------------------------------------------------------- registry


def _not_a_matcher(variant: str, **_: Any) -> CorrespondenceMatcher:
    """``cns_v1`` exists as a variant and is not built by ``build_matcher``.

    It is a whole pipeline - CNS replaces the control law, so it is an arm, not a
    matcher (:func:`vsbench.vs.baselines.cns.build_cns_arm`).
    """
    raise ValueError(
        "matcher.variant 'cns_v1' is not a matcher: CNS v1 is a graph-network "
        "policy that replaces the control law. vsbench.vs.bench.arms.build_arm "
        "builds it through vsbench.vs.baselines.cns.build_cns_arm")


def _dinobot_arm_only(variant: str, **_: Any) -> CorrespondenceMatcher:
    """``dinobot`` exists as a variant and is not built by ``build_matcher``: DINOBot's
    alignment replaces the control law (:func:`vsbench.vs.baselines.dinobot.build_dinobot_arm`)."""
    raise ValueError(
        "matcher.variant 'dinobot' is not a matcher: DINOBot's alignment moves the camera by "
        "a fitted rigid transform instead of a velocity law. vsbench.vs.bench.arms.build_arm "
        "builds it through vsbench.vs.baselines.dinobot.build_dinobot_arm; its front end alone is "
        "matcher.variant 'dinobot_corr'")


def _dinobot_corr(variant: str, *, dinobot: dict | None = None, **kw: Any) -> CorrespondenceMatcher:
    """``dinobot_corr`` - DINOBot's front end (DINO v1 best buddies + K-means to 8 pairs) as a
    matcher; its knobs are the config's ``dinobot:`` block, the arm's own."""
    from vsbench.vs.baselines.dinobot import (  # noqa: PLC0415 - torch-heavy, sklearn on use
        FRONT_END_KEYS,
        DinoBotCorrespondences,
        resolve_block,
    )

    fe = resolve_block({"dinobot": dict(dinobot or {})}, align=False)
    return DinoBotCorrespondences(**{k: fe[k] for k in FRONT_END_KEYS}, **kw)


def _classical(variant: str, **kw: Any) -> CorrespondenceMatcher:
    """``classical_sift`` / ``classical_orb`` / ``classical_akaze`` (U3)."""
    from vsbench.vs.baselines.classical import ClassicalMatcher  # noqa: PLC0415 - cv2 stays lazy

    return ClassicalMatcher(variant.removeprefix("classical_"), **kw)


def _tiled(variant: str, **kw: Any) -> CorrespondenceMatcher:
    """``artvs`` - the published coarse->tiled ViT matcher (U3)."""
    from vsbench.vs.baselines.tiling import TiledMatcher  # noqa: PLC0415 - avoids an import cycle

    return TiledMatcher(**kw)


def _roi(variant: str, **kw: Any) -> CorrespondenceMatcher:
    """``artvs_roi`` - ART-VS's published LangSAM + LightTrack + ROI-tiling pipeline."""
    from vsbench.vs.baselines.objroi import RoiTiledMatcher  # noqa: PLC0415 - avoids an import cycle

    return RoiTiledMatcher(**kw)


def _roi_sim(variant: str, **kw: Any) -> CorrespondenceMatcher:
    """``artvs_roi_sim`` - ART-VS as the release's SIMULATION program runs object mode
    (2026-10-05; :class:`vsbench.vs.baselines.objroi.RoiTiledMatcherSim`)."""
    from vsbench.vs.baselines.objroi import RoiTiledMatcherSim  # noqa: PLC0415 - import cycle

    return RoiTiledMatcherSim(**kw)


def _dense(variant: str, **kw: Any) -> CorrespondenceMatcher:
    return DenseMatcher(frozen_landmarks=(variant == "frozen_landmarks"), **kw)


def _dense_all(variant: str, **kw: Any) -> CorrespondenceMatcher:
    if kw.get("num_landmarks") != POOL_ALL:
        raise ValueError(
            "matcher.variant 'mutual_nn_all' steers on every surviving pair, so it "
            f"requires matcher.num_landmarks: {POOL_ALL} (vsbench.vs.baselines.match.POOL_ALL, the "
            f"vendored FREEZE_ALL_PAIRS request) - got {kw.get('num_landmarks')!r}. "
            "Use 'mutual_nn_seeded_subsample' for a fixed landmark count."
        )
    return DenseMatcher(frozen_landmarks=False, **kw)


#: ``matcher.variant`` (the enum of :mod:`vsbench.vs.bench.config`) -> builder.
def _lightglue(variant: str, **kw: Any) -> CorrespondenceMatcher:
    """``lightglue``: SuperPoint + LightGlue on the pixel lattice."""
    from vsbench.vs.baselines.lightglue import LightGlueMatcher  # noqa: PLC0415 - torch-heavy

    return LightGlueMatcher(**kw)


def _roma(variant: str, **kw: Any) -> CorrespondenceMatcher:
    """``roma`` / ``romav2``: the dense matchers, sampled at the goal landmarks."""
    from vsbench.vs.baselines.roma import RoMaMatcher  # noqa: PLC0415 - torch-heavy

    return RoMaMatcher(version=1 if variant == "roma" else 2, **kw)


MATCHER_BUILDERS = {
    "mutual_nn_seeded_subsample": _dense,
    "mutual_nn_all": _dense_all,
    "frozen_landmarks": _dense,
    "classical_sift": _classical,
    "classical_orb": _classical,
    "classical_akaze": _classical,
    "artvs": _tiled,
    "artvs_roi": _roi,
    "artvs_roi_sim": _roi_sim,
    "cns_v1": _not_a_matcher,
    "lightglue": _lightglue,
    "roma": _roma,
    "romav2": _roma,
    "dinobot": _dinobot_arm_only,
    "dinobot_corr": _dinobot_corr,
}


def device_of(backbone: Any) -> str | None:
    """The device an encoder actually lives on, for the record - ``None`` for a stub.

    One reader, two writers (:meth:`DenseMatcher.record_fields` and
    :meth:`vsbench.vs.baselines.tiling.TiledMatcher.record_fields`), because a record that states
    the *config's* device rather than the encoder's would keep saying ``cuda`` when a
    test injected a CPU stand-in.
    """
    dev = getattr(backbone, "device", None)
    return None if dev is None else str(dev)


def build_matcher(config: dict, **overrides: Any) -> CorrespondenceMatcher:
    """Build the matcher a resolved config asks for.

    Validated first (:func:`vsbench.vs.bench.config.validate`), so a config without
    ``matcher.variant`` or ``matcher.num_landmarks`` fails **here** too, naming
    the field, not only in the loader. ``overrides`` reach the constructor;
    ``scene=`` / ``start_id=`` call :meth:`set_episode` on the built matcher.
    """
    from vsbench.vs.bench.config import MATCHER_VARIANTS, validate

    validate(config)
    variant = config["matcher"]["variant"]
    if variant not in MATCHER_BUILDERS:  # pragma: no cover - kept in step by a test
        raise ValueError(
            f"no builder registered for matcher.variant {variant!r}; "
            f"the enum is {', '.join(MATCHER_VARIANTS)}"
        )
    scene = overrides.pop("scene", None)
    start_id = overrides.pop("start_id", None)
    kw: dict[str, Any] = {
        "num_landmarks": config["matcher"]["num_landmarks"],
        "seed": config["seed"],
    }
    # EVERY other key of the matcher block reaches the constructor (U3). An allow-list
    # here would silently swallow a misspelt knob - `tie_oder: legacy` would have run the
    # default while the record still said `legacy` - whereas a constructor that takes no
    # `**kwargs` turns the same typo into a TypeError naming the argument.
    kw.update({k: v for k, v in config["matcher"].items()
               if k not in ("variant", "num_landmarks")})
    if variant == "dinobot_corr":
        # DINOBot's front-end knobs live in the `dinobot:` block, where the arm reads them
        # too, so the IBVS cell and the arm cannot be configured apart by accident
        kw["dinobot"] = dict(config.get("dinobot") or {})
    kw.update(overrides)
    matcher = MATCHER_BUILDERS[variant](variant, **kw)
    if scene is not None or start_id is not None:
        matcher.set_episode(scene, start_id)
    return matcher
