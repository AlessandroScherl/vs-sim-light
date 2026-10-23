"""Maker of ``*_transformers_mask.png`` + ``transformers_masks.json`` (kept for the record).

"Today's masks" of the old ``artvs_roi*`` cells: the bench's ``GroundingSam(device="cpu")``
on its default (transformers) path, on the four source pictures of ``index.json``, CPU
fp32, 4 threads, CUDA hidden, conda ``genvs``. The stored files were made on 2026-10-06 by
these same calls with the repository at ``5c9ebf6`` (main, before the SAM path existed);
``PROVENANCE.md`` says how this script re-made them after the change, bit for bit.
``tests/slow/test_langsam_masks.py`` checks the default path still makes them.

    CUDA_VISIBLE_DEVICES= HF_HOME=~/genvs_ws/cache/hf HF_HUB_OFFLINE=1 \\
        python tests/fixtures/langsam_masks/make_transformers_masks.py [OUT_DIR]

``OUT_DIR`` defaults to this directory (which overwrites the stored files).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STILLS = Path(os.environ.get("ARTVS_STILLS_ROOT", Path.home() / "genvs_ws" / "vssim_out"))


def main(argv: list[str]) -> int:
    import torch
    from PIL import Image

    sys.path.insert(0, str(REPO))
    from vssim.baselines.objroi import GroundingSam

    out = Path(argv[0]) if argv else HERE
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    gs = GroundingSam(device="cpu")
    rows = []
    for f in json.loads((HERE / "index.json").read_text()):
        pic = STILLS.joinpath(*Path(f["picture"]).parts[-2:])
        if hashlib.md5(pic.read_bytes()).hexdigest() != f["picture_md5"]:
            raise SystemExit(f"{pic}: not the picture the lang_sam mask was made from")
        d = gs.detect(np.asarray(Image.open(pic).convert("RGB")), f["prompt"], with_mask=True)
        mask = np.asarray(d["mask"], bool)
        name = f"{f['name']}_transformers_mask.png"
        Image.fromarray(mask).save(out / name)
        ref = np.asarray(Image.open(HERE / Path(f["mask"]).name)).astype(bool)
        rows.append({"name": f["name"], "box": [float(v) for v in d["box"]],
                     "score": float(d["score"]), "mask_px": int(mask.sum()),
                     "xor_px_vs_langsam": int((mask ^ ref).sum()), "mask": name,
                     "mask_md5": hashlib.md5((out / name).read_bytes()).hexdigest(),
                     "made_with": f"vs-sim {rev} GroundingSam (transformers path), "
                                  f"torch {torch.__version__} CPU fp32, 4 threads"})
        print(json.dumps(rows[-1]), flush=True)
    (out / "transformers_masks.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
