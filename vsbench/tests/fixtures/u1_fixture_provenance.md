# U1 parity fixture — provenance

The tokens under `u1_dinov3_tokens_mug1_goal.npz` were produced **once**, on
2026-09-02, by the frozen old code, not by `refinevs`. Machine-readable copy of
the same facts: `u1_dinov3_tokens_mug1_goal.json` (and inside the `.npz` itself).

## The exact command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate refinevs
cd ~/Desktop/runpod_dev/refine-vs
HF_HUB_OFFLINE=1 PYTHONPATH=/home/alessandro/Desktop/runpod_dev/genvs-sim \
  python tests/fixtures/u1_make_dinov3_tokens.py --device cpu
```

`u1_make_dinov3_tokens.py` never imports `refinevs`. It walks the old deploy
chain exactly as `genvs-sim/src/control/matchers/vit_dense.py:391-395` does:

```python
pil    = Image.fromarray(rgb).resize((512, 512))          # Pillow, antialiased BICUBIC
tensor = _DinoV3ExtractorAdapter(device="cpu", input_size=512).preprocess_pil(pil)
desc   = extractor.extract_descriptors(tensor, layer=11, facet="token", bin=False)
tokens = desc.reshape(32, 32, 384).float()                # CLS + 4 registers already stripped
```

i.e. `src/control/vit_vs_dinov3.py:59-92` → the vendored
`_vitvs_vendor/visual_servoing/features/multi_backbone_extractor.py`
(`MultiBackboneViTExtractor`, `model_name="dinov3-small"`), which strips
`[CLS] + 4 registers` and L2-normalizes.

## What was pinned

| what | value |
| --- | --- |
| old repo | `../../../genvs-sim`, read-only, clean worktree at HEAD `2b11ab8`. The tag `legacy-pre-cleanup` is `905d42b`; `2b11ab8` is one commit later and touches only `CLAUDE.md` and `README.md` (`git diff --name-only legacy-pre-cleanup..HEAD`), so every ported source file is the tagged one. |
| weights | `facebook/dinov3-vits16-pretrain-lvd1689m`, snapshot `114c1379950215c8b35dfcd4e90a5c251dde0d32`, from `~/.cache/huggingface` — gated, never downloaded here (`HF_HUB_OFFLINE=1`) |
| env | conda `refinevs`: python 3.11.16, torch 2.14.0+cpu, transformers 4.57.1, timm 1.0.21, numpy 1.26.4, Pillow 12.3.0 |
| device | `cpu` |

## Files and md5s

| file | md5 | size | what |
| --- | --- | --- | --- |
| `u1_frame_mug1_goal.jpg` | `0ce3c308f2323b85a6e8ee5724783ed3` | 197 KB | the fixture image: a real 1920×1080 Isaac harness render, copied byte-for-byte from `docs/results/2026-08-16_rotsym_wpc_preview/bench_frames/mug1/goal.jpg` (the WP-C preview bench, scene `mug1`, goal view) |
| `u1_pattern_synthetic.png` | `1af75cf0cb02b8f3e2c8da26b36845fe` | 273 KB | the synthetic pattern: a deterministic 640×480 chart (three checkerboards, radial rings, two ramps, two flat blocks) written by `make_pattern()` in the generator |
| `u1_dinov3_tokens_mug1_goal.npz` | `73f1fdbe3e60a11313b6ff356d09df8d` | 1.47 MB | float32 `(32, 32, 384)` tokens of the frame above, plus the metadata in the table above |

The image's decoded pixels (not the JPEG bytes) hash to
`161879133226b37fcef558814b5ae424`; the npz carries that as `image_pixels_md5`,
so a re-encoded JPEG cannot pass unnoticed.

## What the fixture proves, and how close

`refinevs.features.FrozenBackbone("dinov3_vits16", device="cpu").encode(frame)`
must reproduce it to **max abs diff ≤ 1e-5**. Measured 2026-09-02: **8.8e-7** in
both `refinevs` (transformers 4.57.1) and `bot` (5.8.0) — per-token cosine
≥ 0.9999998. The **model side is bit-exact**: fed the old chain's own
`pixel_values`, the new model reproduces the fixture at 0.0. The 8.8e-7 is
entirely preprocessing — the old *deploy* chain's HF `DINOv3ViTImageProcessorFast`
fuses rescale and normalize into one affine map, while `refinevs` (like the old
*training* encoder `src/refine/encoder.py`) computes the plain
`(x/255 − mean)/std`; the two differ by 7.15e-7 in `pixel_values`. That is also
why the old training and deploy chains were themselves 8.8e-7 apart, and why the
new `encode` matches the training encoder **bit-for-bit** (max abs 0.0) — i.e.
it is exactly the tokens the refinement head was trained on. (The earlier
"~7e-8" recorded here was wrong; it did not reproduce. Corrected after review R10,
which re-derived the old chain three ways at 0.0.) On another device or dtype the
bar is **max abs ≤ 1e-2 and per-token cosine ≥ 0.999**; the GPU preprocessing
twin `preprocess_torch` lands at 5.7e-4 and is not the fielded path. Both rules are in
`refinevs/features.py`'s module docstring; the checks are
`tests/fast/test_features.py::test_tokens_match_legacy_dinov3` and
`tests/slow/test_features_parity.py::test_tokens_match_legacy_dinov3_on_gpu`.

Regenerating this fixture is a decision, not a fix: it means the port changed a
number. Say so in the commit message and in `../../RESULTS.md`.
