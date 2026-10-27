# vs-sim copy: vssim/baselines/__init__.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The published baselines: ViT-VS, ART-VS, CNS v1 and the classical detectors.

=============== ================================================================
module          what it owns
=============== ================================================================
``features``    the frozen ViT backbones: DINOv3 ViT-S/16 (transformers) and the
                published ViT-VS DINOv2 ViT-S/14 chain (torch.hub, log-binned);
                the Pillow-exact 512 preprocessing
``match``       mutual-nearest-neighbour patch matching = **ViT-VS**, the
                ``Matches`` record, the seeded landmark draw
``tiling``      **ART-VS**: the tile grid and the coarse -> tiled latch
``objroi``      **ART-VS object mode**: LangSAM + LightTrack + ROI tiling - the
                2026-09-21 port (``artvs_roi``) and the release's simulation
                program (``artvs_roi_sim``, 2026-10-05)
``prealign``    the published 4-way pre-servo roll search (ViT-VS / ART-VS)
``classical``   SIFT / ORB / AKAZE with RANSAC as a switch
``control``     the analytic IBVS law, the weighted least squares, the stop rules
``cns``         **CNS v1** (github.com/hhcaz/CNS), run as published, in process
``dinobot``     **DINOBot**'s alignment phase (ICRA 2024): DINO v1 best buddies,
                depth on both sides, a Kabsch jump per step - an arm, like CNS
=============== ================================================================

Everything here imports ``torch`` at module level; ``import vsbench.vs`` does not import
this package, so the simulator stays numpy-only. Weights are never downloaded:
DINOv3 comes from the local HF cache, DINOv2 and DINO v1 from the local torch.hub
cache, CNS from a checkout named by ``$CNS_ROOT``. scikit-learn (DINOBot's K-means) is
imported only where it runs.

Provenance: extracted from the Refine-VS research repository, whose modules were
ported from ``genvs-sim`` with parity fixtures (the same fixtures are under
``tests/fixtures/`` here). A docstring that names a ``refinevs.*`` module not
present in this repository refers to that project.
"""
