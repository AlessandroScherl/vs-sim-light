# Vendored references for the DINOBot baseline — provenance

These files are the **published sources** `vssim.baselines.dinobot` is tested against
(`tests/fast/test_dinobot.py`, `tests/slow/test_dinobot_parity.py`). They are
third-party code kept byte-for-byte: `pyproject.toml` excludes this directory from ruff,
and nothing here may be reformatted or "fixed". If a port disagrees with them, the port
changes. Fetched 2026-09-22 for the build of `docs/dinobot_baseline_spec_2026-09-22.md`
(workspace root).

## `dino_vit_features/` — ShirAmir/dino-vit-features (MIT)

Amir, Gandelsman, Bagon, Dekel, *Deep ViT Features as Dense Visual Descriptors*
(arXiv 2112.05814). Upstream `https://github.com/ShirAmir/dino-vit-features`, commit
**`1779649078744e7d9be903325355fd481cca8103`** (2022-09-09, the last commit), cloned with
`git clone https://github.com/ShirAmir/dino-vit-features && git checkout 1779649`.
The DINOBot gist says "run this file inside that repo": this library does all of
DINOBot's matching.

| file here | upstream file (md5 at that commit) | kept (upstream line numbers) | left out, and why |
| --- | --- | --- | --- |
| `extractor.py` | `extractor.py` (`a30886446a4745d3a5d3eea998e547e2`) | imports L2, L4, L6, L7, L9, L11; class `ViTExtractor` L15-56 (`__init__`), L81-139 (`_fix_pos_enc`, `patch_vit_resolution`), L159-318 (`_get_hook`, `_register_hooks`, `_unregister_hooks`, `_extract_features`, `_log_bin`, `extract_descriptors`, `extract_saliency_maps`) | `create_model` L57-80 (torch.hub / timm loading), `preprocess` L140-158 (reads an image path), `str2bool` and the CLI L319-357, and the imports only they use (argparse, torchvision, timm, pathlib, PIL). Tests inject a model, as the class's own `model=` argument allows. |
| `correspondences.py` | `correspondences.py` (`f078cb463f1f53607614bcfa9a56398b`) | imports L2, L6, L7; `find_correspondences` body **L46-110** under a wrapper line of ours; `chunk_cosine_sim` L152-165 | L14-45 of `find_correspondences` (it builds a `ViTExtractor` from torch.hub and preprocesses two image *paths*): replaced by ONE line of ours, `def find_correspondences_core(...)`, whose parameters supply exactly the names L46-110 read (`descriptors1/2`, `saliency_map1/2`, `num_patches1/2`, `extractor` - only `.stride` and `.p` are read -, `num_pairs`, `thresh`, `device`, and `image1_pil/image2_pil` so that the return line stays upstream's). `draw_correspondences`, `str2bool`, the CLI and their imports. |
| `LICENSE` | `LICENSE` | whole file | - |

The library's pinned environment (`env/dino-vit-feats-env.yml`): python 3.9.7, torch
1.10.1, torchvision 0.11.2, **scikit-learn 1.0.1**, numpy 1.21.2, timm 0.4.12. Under
scikit-learn 1.0.1, `KMeans(n_clusters=n, random_state=0)` ran `n_init=10` and
`algorithm="auto"` (= `"elkan"` for dense data). Today's scikit-learn defaults to
`n_init="auto"` (one k-means++ run) and `algorithm="lloyd"`. The port states
`n_init=10` explicitly (owner decision D3) and the fast test binds the vendored module's
`KMeans` name to `functools.partial(KMeans, n_init=...)` for the comparison, instead of
editing the vendored call. Elkan and Lloyd are the same algorithm in exact arithmetic
(Elkan skips distance computations by the triangle inequality); the choice of
`algorithm` is left at the installed default and recorded (`dinobot_sklearn_version`).

## `dinobot_oneshot_il.py` — the DINOBot gist, final revision (not importable)

Di Palo & Johns, *DINOBot* (arXiv 2402.13181, ICRA 2024). The code the project page
links is a gist: `https://gist.github.com/normandipalo/fbc21f23606fbe3d407e22c363cb134e`,
file `dinobot_oneshot_il.py`. Fetched with
`curl -sL https://gist.githubusercontent.com/normandipalo/fbc21f23606fbe3d407e22c363cb134e/raw/`
= revision **`60d5adbea956d36d6a2d039a59f9524875b8aa0c`** (2024-05-21, the fifth and last;
md5 `2ea5880fb9b76efc540d0f492a2c2950`, identical to
`.../raw/60d5adbea956d36d6a2d039a59f9524875b8aa0c/dinobot_oneshot_il.py`). The five
revisions (GitHub API `gists/<id>/commits`): `4349383` 2024-02-29 14:34, `3044933`
2024-02-29 14:36, `3287cc9` 2024-04-23, `d29ad51` 2024-05-13, `60d5adb` 2024-05-21.

**Kept verbatim, and it does not run as published** - four defects, left as they are:

1. `if __name__ == “__main__”:` (L112) uses typographic quotes - a `SyntaxError`;
2. `def record_demo()` (L72) and `def replay_demo(demo)` (L79) have no colon - two more;
3. `robot.move(t_meters,R)` (L147): `t_meters` is not defined in this revision (it was
   `convert_pixels_to_meters(t)` in revision 1 and the definition was deleted in
   revision 4, the call was not), and `robot` is not defined anywhere (the placeholder
   above it is `robot_move`);
4. `intrinsics` (L140-141) is not defined anywhere.

So `tests/fixtures/reference/dinobot_gist_functions.py` carries the two functions the
port reproduces - `find_transformation` (L87-107) and `compute_error` (L109-110) - byte
for byte, with the gist's own `import numpy as np` (L10), so the tests can import them.
`tests/fast/test_package.py` does not parse this directory: it is third-party text, and
one file of it is not Python that parses.
