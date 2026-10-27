"""Produce the U2 parity fixture with the FROZEN OLD CODE. Run once; see
``u2_fixture_provenance.md`` for the exact command and the resulting md5s.

Never imports ``refinevs``. It drives the old
``genvs-sim/src/control/matchers/vit_dense.py::ViTDenseMatcher`` exactly as
``src/eval/baseline_check.py`` builds the fielded dense arm (dinov3,
``num_pairs=24``, ``ransac="none"``, ``emit_margins=True``, the seeded draw of
``episode_subsample_seed(scene, start_id, base)``) on one (goal, current) pair
and records: (1) the FULL mutual pool in the vendored order with its similarity
and margin per pair, plus the vendored cyclic score of **every** goal patch;
(2) the 24 pairs the seeded draw returns, and the 48-pair draw for the N3.7d
superset convention; (3) the ``pin_first_frame`` (``_fs``) variant's frames 2
and 3 - the pinned set tracked one-way over the current frame and over the goal
frame; (4) the pool's significance signals from the OLD
``src/eval/n2b_significance.py::row_significance``.

Two old-code internals are captured by monkeypatching, so the recorded numbers
are the vendored function's own rather than a transcription: ``torch.Tensor.sort``
is wrapped to keep ``cyclical_dists_norm`` and to record both the non-stable
order the old code takes and the stable one; ``torch.randperm`` is replaced by
``arange`` for the pool pass only, so ``filtered_points`` come back unpermuted -
that is, in the vendored sorted order.

Regenerating a fixture is a decision, not a fix: it means the port changed a
number (``tests/fixtures/README.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD_REPO = Path("~/Desktop/runpod_dev/genvs-sim")
CUR_FRAME_SOURCE = Path(
    "~/Desktop/runpod_dev/docs/results/2026-08-16_rotsym_wpc_preview"
    "/bench_frames/mug1/roll010.jpg"
)
GOAL_FRAME = HERE / "u1_frame_mug1_goal.jpg"          # the U1 fixture's image
CUR_FRAME = HERE / "u2_frame_mug1_roll010.jpg"
OUT = HERE / "u2_matches_mug1.npz"

SCENE = "mug1"
START_ID = 7
SEED_BASE = 0
NUM_LANDMARKS = 24
NUM_LANDMARKS_SUPERSET = 48


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from PIL import Image

    if str(OLD_REPO) not in sys.path:
        sys.path.insert(0, str(OLD_REPO))

    if not CUR_FRAME.exists():
        CUR_FRAME.write_bytes(CUR_FRAME_SOURCE.read_bytes())
        print(f"wrote {CUR_FRAME.name}  md5 {md5_file(CUR_FRAME)}")

    import torch
    import transformers
    from src.control.matchers.significance import margins_for_matches
    from src.control.matchers.vit_dense import ViTDenseMatcher
    from src.eval.baseline_check import episode_subsample_seed
    from src.eval.n2b_significance import row_significance

    goal = np.asarray(Image.open(GOAL_FRAME).convert("RGB"))
    cur = np.asarray(Image.open(CUR_FRAME).convert("RGB"))
    assert goal.shape == cur.shape == (1080, 1920, 3)

    sub_seed = episode_subsample_seed(SCENE, START_ID, SEED_BASE)
    print(f"episode_subsample_seed({SCENE!r}, {START_ID}, {SEED_BASE}) = {sub_seed}")

    matcher = ViTDenseMatcher(
        backbone="dinov3", num_pairs=NUM_LANDMARKS, device=args.device,
        ransac="none", emit_margins=True,
    )
    out: dict = {}

    def lin(pts_patch):
        """(col, row) patch indices -> linear index row*side + col (side 32)."""
        p = np.asarray(pts_patch, np.int64)
        return p[:, 1] * 32 + p[:, 0]

    # ---------------------------------------------------------------- pool
    # `num_pairs = FREEZE_ALL_PAIRS` is the vendored "give me every survivor"
    # request (`vit_dense.ViTDenseMatcher.FREEZE_ALL_PAIRS`); with randperm
    # replaced by arange the survivors come back in the vendored sorted order.
    captured: dict = {}
    orig_sort = torch.Tensor.sort
    orig_randperm = torch.randperm

    def spy_sort(self, *a, **kw):
        captured["cyclic_all"] = self.detach().clone()
        result = orig_sort(self, *a, **kw)
        captured["order_vendored"] = result[1].detach().clone()
        captured["order_stable"] = orig_sort(self, *a, **{**kw, "stable": True})[1].clone()
        return result

    torch.Tensor.sort = spy_sort
    torch.randperm = lambda n, **kw: torch.arange(n, device=kw.get("device", None))
    try:
        matcher.num_pairs = ViTDenseMatcher.FREEZE_ALL_PAIRS
        matcher.set_subsample_seed(sub_seed)
        matcher.reset(goal)
        pool_corr = matcher.match(cur)
    finally:
        torch.Tensor.sort = orig_sort
        torch.randperm = orig_randperm
    matcher.num_pairs = NUM_LANDMARKS

    pool_goal_lin = lin(pool_corr.debug["points_ref_patch"])
    pool_cur_lin = lin(pool_corr.debug["points_cur_patch"])
    n_pool = len(pool_goal_lin)
    cyclic_all = captured["cyclic_all"].reshape(-1).cpu().numpy().astype(np.float32)
    order_vendored = captured["order_vendored"].reshape(-1).cpu().numpy().astype(np.int64)
    order_stable = captured["order_stable"].reshape(-1).cpu().numpy().astype(np.int64)
    print(f"pool: {n_pool} pairs of {cyclic_all.size}; "
          f"vendored order == stable order over the pool: "
          f"{np.array_equal(order_vendored[:n_pool], order_stable[:n_pool])}")

    out.update(
        pool_goal_lin=pool_goal_lin,
        pool_cur_lin=pool_cur_lin,
        pool_sim=np.asarray(pool_corr.conf, np.float32),
        pool_margin=np.asarray(pool_corr.debug["margin"], np.float64),
        pool_cyclic=cyclic_all[pool_goal_lin],
        cyclic_all=cyclic_all,
        sorted_order_vendored=order_vendored,
        sorted_order_stable=order_stable,
    )

    # The old deploy tokens of the CURRENT frame, taken HERE - `match()`
    # overwrites `_last_cur_descriptors` every call and the `_fs` phase below
    # ends on the goal frame. The goal side is the U1 fixture (same image, same
    # chain), so it is not duplicated.
    cur_tokens = (matcher._last_cur_descriptors.reshape(32, 32, 384)
                  .float().cpu().numpy().astype(np.float32))

    # The similarity rows of the pool's goal patches, from the matcher's own
    # normalized tokens - the exact rows `emit_margins` matches on. Used for
    # the significance signals below; not stored (P x 1024 is recomputable).
    goal_tok = matcher._goal_tok_norm
    cur_tok, side = matcher._normalized_patch_tokens(matcher._last_cur_descriptors)
    assert side == 32
    with torch.no_grad():
        sim_rows = (goal_tok[torch.as_tensor(pool_goal_lin)] @ cur_tok.T).cpu().numpy()
    np.testing.assert_allclose(
        margins_for_matches(sim_rows, pool_cur_lin, 32), out["pool_margin"], atol=0, rtol=0)

    sig = [row_significance(sim_rows[i], int(pool_cur_lin[i]), 32) for i in range(n_pool)]
    for key in ("s1", "second", "margin", "ratio", "n_lookalikes",
                "entropy_t05", "entropy_t1"):
        out[f"sig_{key}"] = np.asarray([s[key] for s in sig], np.float64)

    # ------------------------------------------------------------ the draws
    for k in (NUM_LANDMARKS, NUM_LANDMARKS_SUPERSET):
        matcher.num_pairs = k
        matcher.set_subsample_seed(sub_seed)
        matcher.reset(goal)
        c = matcher.match(cur)
        assert len(c) == k, (k, len(c))
        out[f"draw{k}_goal_lin"] = lin(c.debug["points_ref_patch"])
        out[f"draw{k}_cur_lin"] = lin(c.debug["points_cur_patch"])
        out[f"draw{k}_sim"] = np.asarray(c.conf, np.float32)
        out[f"draw{k}_margin"] = np.asarray(c.debug["margin"], np.float64)
        out[f"draw{k}_goal_px"] = np.asarray(c.pts_goal, np.float64)
        out[f"draw{k}_cur_px"] = np.asarray(c.pts_cur, np.float64)
    matcher.num_pairs = NUM_LANDMARKS
    assert np.array_equal(
        np.sort(out[f"draw{NUM_LANDMARKS}_goal_lin"]),
        np.sort(np.intersect1d(out[f"draw{NUM_LANDMARKS}_goal_lin"],
                               out[f"draw{NUM_LANDMARKS_SUPERSET}_goal_lin"]))), \
        "the k=48 draw must be a superset of the k=24 draw at the same seed"

    # -------------------------------------------------- frozen_landmarks (_fs)
    # Frame 1 runs the published path and pins its 24; frames 2 and 3 track the
    # pinned goal patches one-way. Frame 2 re-reads the same current frame,
    # frame 3 the goal frame (whose answer must be the identity).
    matcher.pin_first_frame = True
    matcher.set_subsample_seed(sub_seed)
    matcher.reset(goal)
    f1 = matcher.match(cur)
    f2 = matcher.match(cur)
    f3 = matcher.match(goal)
    matcher.pin_first_frame = False
    assert np.array_equal(lin(f1.debug["points_ref_patch"]), out["draw24_goal_lin"]), \
        "the _fs frame-1 pin must BE the published seeded draw"
    for tag, c in (("fs2", f2), ("fs3", f3)):
        out[f"{tag}_goal_lin"] = lin(c.debug["points_ref_patch"])
        out[f"{tag}_cur_lin"] = lin(c.debug["points_cur_patch"])
        out[f"{tag}_sim"] = np.asarray(c.conf, np.float64)
        out[f"{tag}_margin"] = np.asarray(c.debug["margin"], np.float64)

    old_commit = subprocess.run(
        ["git", "-C", str(OLD_REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    meta = dict(
        cur_tokens=cur_tokens,
        goal_image_file=np.asarray(GOAL_FRAME.name),
        goal_image_md5=np.asarray(md5_file(GOAL_FRAME)),
        cur_image_file=np.asarray(CUR_FRAME.name),
        cur_image_md5=np.asarray(md5_file(CUR_FRAME)),
        cur_image_pixels_md5=np.asarray(hashlib.md5(cur.tobytes()).hexdigest()),
        cur_image_source=np.asarray(
            "docs/results/2026-08-16_rotsym_wpc_preview/bench_frames/mug1/roll010.jpg"),
        scene=np.asarray(SCENE),
        start_id=np.asarray(START_ID),
        seed_base=np.asarray(SEED_BASE),
        subsample_seed=np.asarray(sub_seed),
        num_landmarks=np.asarray(NUM_LANDMARKS),
        old_repo=np.asarray("genvs-sim @ tag legacy-pre-cleanup"),
        old_commit=np.asarray(old_commit),
        chain=np.asarray(
            "src/control/matchers/vit_dense.py::ViTDenseMatcher(backbone='dinov3', "
            "num_pairs=24, ransac='none', emit_margins=True) -> "
            "_vitvs_vendor/.../matcher.find_correspondences_batch; margins from "
            "src/control/matchers/significance.margins_for_matches; signals from "
            "src/eval/n2b_significance.row_significance"),
        model_id=np.asarray("facebook/dinov3-vits16-pretrain-lvd1689m"),
        device=np.asarray(args.device),
        dtype=np.asarray("float32"),
        torch_version=np.asarray(torch.__version__),
        transformers_version=np.asarray(transformers.__version__),
        numpy_version=np.asarray(np.__version__),
        python_version=np.asarray(sys.version.split()[0]),
        created_utc=np.asarray(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
    )
    np.savez_compressed(OUT, **out, **meta)
    print(f"wrote {OUT.name}  {OUT.stat().st_size / 1e6:.2f} MB  md5 {md5_file(OUT)}")

    (HERE / "u2_matches_mug1.json").write_text(json.dumps({
        "fixture": OUT.name,
        "unit": "U2 (refinevs/match.py)",
        "created_utc": str(meta["created_utc"]),
        "md5": md5_file(OUT),
        "bytes": OUT.stat().st_size,
        "command": (
            "HF_HUB_OFFLINE=1 PYTHONPATH=~/Desktop/runpod_dev/genvs-sim "
            "python tests/fixtures/u2_make_match_fixture.py --device cpu"
            "   (conda env refinevs)"),
        "source_repo_commit": f"genvs-sim {old_commit} (tag legacy-pre-cleanup = 905d42b)",
        "pair": {
            "goal": f"{GOAL_FRAME.name} (md5 {md5_file(GOAL_FRAME)}) - the U1 fixture image",
            "current": f"{CUR_FRAME.name} (md5 {md5_file(CUR_FRAME)})",
            "current_source": str(CUR_FRAME_SOURCE),
        },
        "episode": {"scene": SCENE, "start_id": START_ID, "seed_base": SEED_BASE,
                    "subsample_seed": int(sub_seed)},
        "pool_size": int(n_pool),
        "vendored_order_equals_stable_order": bool(
            np.array_equal(order_vendored[:n_pool], order_stable[:n_pool])),
        "arrays": {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)}
                   for k, v in out.items()},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": transformers.__version__, "numpy": np.__version__,
                "device": args.device},
        "provenance_note": "tests/fixtures/u2_fixture_provenance.md",
        "read_by": ["tests/fast/test_match.py"],
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
