# `langsam_masks/` — the goal masks the real `lang_sam` package makes, and today's

*Written 2026-10-06 with the SAM path of the faithful ART-VS arm (`GroundingSam(sam_path=...)`,
`roi_sam_path`). Used by `tests/slow/test_langsam_masks.py`. The check behind it:
`docs/results/2026-10-06_langsam_check/` at the workspace root (owner approval 2026-10-06).*

## What is here

| file | what |
| --- | --- |
| `index.json` | byte-for-byte copy of the check's `fixtures/index.json`: per picture the source path and its md5, the prompt, `lang_sam`'s best box and score, the mask's pixel count |
| `*_langsam_mask.png` (4) | byte-for-byte copies of the check's `fixtures/` masks: 1-bit PNG, 1920 × 1080, `lang_sam`'s mask of its best box |
| `*_transformers_mask.png` (4) | today's masks: the bench's `GroundingSam` on its default (transformers) path - the old `artvs_roi*` cells' mask - same pictures, same box |
| `transformers_masks.json` | their box, score, pixel count, pixels differing from `lang_sam`'s, md5 |
| `make_transformers_masks.py` | the calls that made them, kept for the record |

## The `lang_sam` masks

- Made in the check by the real package: **lang-sam 0.2.1 at git `918043ed4666eea04da88aa179eb8d27ef4b1a1d`**
  and **SAM-2 1.0 at git `c2ec8e14a185632b0a5d8b161928ceb50197eddc`** (lang_sam's pin), both
  `--no-deps`, in the conda env **`langsam_check`** (`~/.conda/envs/langsam_check`, cloned from
  `genvs`: torch 2.11.0+cu128, torchvision 0.26.0, transformers 4.57.1, numpy 2.2.6, Pillow 12.1.1).
- **CPU fp32, 4 threads**, CUDA hidden, weights from the HF cache offline. `LangSAM` built as
  the ART-VS release builds it: the GroundingDINO-base snapshot
  `12bdfa3120f3e7ec7b434d90674b3396eccf88eb` for model and processor, and the cached
  `sam2.1_hiera_small.pt` (sha256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`, 184,416,285 bytes) as `sam_ckpt_path`.
  Then `LangSAM.predict([picture], [prompt])` with its defaults (box 0.3, text 0.25): the mask
  of the highest-scoring box, `> 0` (`scripts/compare.py` of the check, its `mask_ls`).
- The source pictures are the instance suite's goal renders on the station
  (`~/genvs_ws/vssim_out/<pair>/`, `$ARTVS_STILLS_ROOT` in the tests); the md5s are in `index.json`
  and the tests refuse a picture that does not match:

| fixture | picture | prompt | mask px |
| --- | --- | --- | --- |
| `inst_shoe_same_goal` | `inst_shoe_same/inst_shoe_goal.png` (`fb588247…`) | shoe | 433,768 |
| `inst_mug_rev_other_big_goal` | `inst_mug_rev_other_big/inst_mug_rev_goal.png` (`4b5eb41a…`) | mug | 604,050 |
| `inst_teapot_rev_other_big_goal` | `inst_teapot_rev_other_big/inst_teapot_rev_goal.png` (`08a1eadc…`) | teapot | 440,922 |
| `inst_hammer_same_goal` | `inst_hammer_same/inst_hammer_goal.png` (`76443221…`) | hammer | 287,802 |

What they cover: the shoe is the stability fallback's case (it fired on every shoe goal and
takes in the insole); the mug and the teapot are the resize rounding's largest and a middling
case (the mug's dark interior, the teapot's lid knob); the hammer is a stable goal (the old
path is 185 px from `lang_sam`'s there).

## Today's (transformers-path) masks

- Made 2026-10-06 on the station host with this repository at **`5c9ebf6`** (main, before the
  SAM path existed): `GroundingSam(device="cpu")` → `detect(picture, prompt, with_mask=True)`,
  conda `genvs` (the same versions as above, without lang_sam), CPU fp32, 4 threads, CUDA hidden -
  the calls of `make_transformers_masks.py`, run from a scratch copy.
- They reproduce the check's numbers for the bench's path exactly (its `results.jsonl`, `px_b`
  and `xor_px`): shoe 478,074 px (45,244 differ from `lang_sam`'s), mug 626,487 (22,793), teapot
  444,996 (4,174), hammer 287,701 (185). Boxes and scores equal `lang_sam`'s to the last digit.
- Re-made with `make_transformers_masks.py` at `d75db77` (after the change, the default path):
  the four PNGs byte-identical (same md5s).

## In plain words

The four `*_langsam_mask.png` are the outlines the ART-VS authors' own detector package draws
around the object on four goal pictures. The four `*_transformers_mask.png` are the outlines our
benchmark drew before this fix. The slow test checks that the faithful ART-VS arm now draws the
first set exactly, and that the old cells still draw the second set exactly.
