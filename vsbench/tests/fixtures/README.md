# tests/fixtures — the parity fixtures

Each is the **same input run once through the frozen old code** (`genvs-sim` at tag
`legacy-pre-cleanup`, via the Refine-VS port), stored here so the port is proven equal
rather than assumed equal, on any machine, without the old checkout. The `*_make_*.py`
generators ran in the source repository and are kept for the record; the
`*_provenance.md` files say what was measured and to what tolerance.

| fixture | proves |
| --- | --- |
| `u1_*` | the DINOv3 ViT-S/16 token grid of one committed 1920 × 1080 frame, to 8.8e-7 (bar 1e-5), and the Pillow-exact 512 resize bit for bit |
| `u2_*` | the mutual-nearest-neighbour pool of a second frame of the same scene, bit-identical (612 pairs, cyclic scores, similarities, margins), and the seeded landmark draws |
| `u3_*` | SIFT / ORB / AKAZE (cv2 4.10–4.11), ART-VS's tiled matcher on DINOv3, and the DINOv2 chain's shapes |
| `u4_*` | the IBVS solve against 300 recorded steps of a station episode, to 0.0; `reference/cns_stop_policy.py` is CNS's own stop policy (MIT), run beside ours on every test trace |
| `u7_*` | the frozen episode record schema (84 fields) and one stored extended-telemetry step |
| `u8_*` | the published 4-way pre-servo search: the hypothesis rotate (md5s), `unwarp_points` tables, and one frozen `RollPrealigner` pick |
| `reference/artvs_release/` | the ART-VS release (MIT, git `9b1208c`): `matcher.py` and `controller.py` byte for byte, `_log_bin` and the simulation controller's / detector's ROS-bound code assembled from the release's lines (`PROVENANCE.md`; `make_*.py` regenerate them) - run beside the faithful ART-VS arm (`test_objroi_sim.py`, `tests/slow/test_artvs_release_parity.py`) |
| `langsam_masks/` | the goal masks the real `lang_sam` package makes on four instance-suite goals (lang-sam `918043e`, SAM-2 `c2ec8e1`, the LangSAM check of 2026-10-06) and the masks the bench's default SAM path made there at `5c9ebf6` (`PROVENANCE.md`) - the faithful ART-VS arm's mask equals the first bit for bit, the old cells' still equals the second (`tests/slow/test_langsam_masks.py`) |
| `frozen_goals.json` | the goal block of each benchmark scene's frozen start file |

Stored tokens carry the "no warm-up" goal capture (one render); the settle rule
(`scene.render_warmup_steps: 8`) came later and does not change these files.
