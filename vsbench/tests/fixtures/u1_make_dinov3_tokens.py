"""Produce the U1 parity fixture with the FROZEN OLD CODE. Run once; see
``u1_fixture_provenance.md`` for the exact command and the resulting md5s.

This script never imports ``refinevs``. It walks the old chain exactly as
``genvs-sim/src/control/matchers/vit_dense.py`` lines 391-395 do it:

    pil    = Image.fromarray(rgb).resize((512, 512))
    tensor = _DinoV3ExtractorAdapter(device, 512).preprocess_pil(pil)
    desc   = extractor.extract_descriptors(tensor, layer=11, facet="token", bin=False)

and stores ``desc.reshape(32, 32, 384)`` (already CLS + 4-register stripped and
L2-normalized by the vendored ``MultiBackboneExtractor``) as float32, together
with the metadata that says what produced it.

It also writes the synthetic pattern the fast tests use, so both fixture images
are reproducible from this file alone.

Regenerating a fixture is a decision, not a fix: it means the port changed a
number (``tests/fixtures/README.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD_REPO = Path("~/Desktop/runpod_dev/genvs-sim")
REAL_FRAME_SOURCE = Path(
    "~/Desktop/runpod_dev/docs/results/2026-08-16_rotsym_wpc_preview"
    "/bench_frames/mug1/goal.jpg"
)
REAL_FRAME = HERE / "u1_frame_mug1_goal.jpg"
PATTERN = HERE / "u1_pattern_synthetic.png"
TOKENS = HERE / "u1_dinov3_tokens_mug1_goal.npz"


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def make_pattern(size=(480, 640)) -> np.ndarray:
    """A deterministic, noise-free 480x640 RGB chart.

    Deliberately structured rather than random: checkerboards of three cell
    sizes, radial rings and two ramps give the resize sharp edges to disagree
    on and give the backbone real texture, while staying a small PNG.
    """
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    checks = sum(
        (((xx // c).astype(int) + (yy // c).astype(int)) % 2) * (255.0 / (i + 2))
        for i, c in enumerate((37, 11, 3))
    )
    rings = 127.5 * (1 + np.cos(np.hypot(xx - 0.35 * w, yy - 0.6 * h) / 9.0))
    ramp_x = 255.0 * xx / (w - 1)
    ramp_y = 255.0 * yy / (h - 1)
    img = np.stack(
        [
            0.5 * checks + 0.5 * ramp_x,
            0.5 * rings + 0.5 * ramp_y,
            0.34 * checks + 0.33 * rings + 0.33 * (255.0 - ramp_x),
        ],
        axis=-1,
    )
    img[40:120, 40:200] = (12.0, 240.0, 30.0)  # a few flat blocks
    img[300:380, 420:600] = (250.0, 8.0, 190.0)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from PIL import Image

    if str(OLD_REPO) not in sys.path:
        sys.path.insert(0, str(OLD_REPO))

    if not PATTERN.exists():
        Image.fromarray(make_pattern()).save(PATTERN, optimize=True)
        print(f"wrote {PATTERN.name}  md5 {md5_file(PATTERN)}")

    if not REAL_FRAME.exists():
        REAL_FRAME.write_bytes(REAL_FRAME_SOURCE.read_bytes())
        print(f"wrote {REAL_FRAME.name}  md5 {md5_file(REAL_FRAME)}")

    import torch
    import transformers
    from src.control.vit_vs_dinov3 import _DinoV3ExtractorAdapter

    rgb = np.asarray(Image.open(REAL_FRAME).convert("RGB"))
    assert rgb.shape == (1080, 1920, 3) and rgb.dtype == np.uint8, rgb.shape

    extractor = _DinoV3ExtractorAdapter(device=args.device, input_size=512)
    pil = Image.fromarray(rgb).resize((512, 512))
    tensor = extractor.preprocess_pil(pil).to(args.device)
    with torch.no_grad():
        desc = extractor.extract_descriptors(tensor, layer=11, facet="token", bin=False)
    assert tuple(desc.shape) == (1, 1, 1024, 384), tuple(desc.shape)
    tokens = desc.reshape(32, 32, 384).float().cpu().numpy().astype(np.float32)

    old_commit = subprocess.run(
        ["git", "-C", str(OLD_REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    np.savez_compressed(
        TOKENS,
        tokens=tokens,
        image_file=np.asarray(REAL_FRAME.name),
        image_md5=np.asarray(md5_file(REAL_FRAME)),
        image_pixels_md5=np.asarray(hashlib.md5(rgb.tobytes()).hexdigest()),
        image_source=np.asarray(
            "docs/results/2026-08-16_rotsym_wpc_preview/bench_frames/mug1/goal.jpg"
        ),
        old_repo=np.asarray("genvs-sim @ tag legacy-pre-cleanup"),
        old_commit=np.asarray(old_commit),
        chain=np.asarray(
            "src/control/matchers/vit_dense.py:391-395 -> "
            "vit_vs_dinov3._DinoV3ExtractorAdapter -> "
            "_vitvs_vendor/.../multi_backbone_extractor.MultiBackboneViTExtractor"
        ),
        model_id=np.asarray("facebook/dinov3-vits16-pretrain-lvd1689m"),
        device=np.asarray(args.device),
        dtype=np.asarray("float32"),
        torch_version=np.asarray(torch.__version__),
        transformers_version=np.asarray(transformers.__version__),
        numpy_version=np.asarray(np.__version__),
        pillow_version=np.asarray(Image.__version__ if hasattr(Image, "__version__") else ""),
        python_version=np.asarray(sys.version.split()[0]),
        created_utc=np.asarray(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
    )
    print(f"wrote {TOKENS.name}  {TOKENS.stat().st_size / 1e6:.2f} MB  md5 {md5_file(TOKENS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
