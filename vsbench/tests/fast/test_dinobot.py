# vs-sim copy: tests/fast/test_dinobot.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""DINOBot's alignment phase, with no weights: the vendored sources run beside the port.

Spec: ``../docs/dinobot_baseline_spec_2026-09-22.md`` section 5.1 (workspace root); the
nine numbered items there are the sections below. The references are vendored under
``tests/fixtures/reference/`` (``dino_vit_features/`` - the library DINOBot runs inside,
commit 1779649, MIT - and ``dinobot_gist_functions.py`` - the gist's ``find_transformation``
and ``compute_error``, byte for byte); a port written from a reading diverged in five
places once (the CNS stop policy, review R12), a port tested against its source does not.

No test here touches a GPU or a weight file: the network is either a stub extractor
(descriptors from the pixels) or a tiny randomly initialised network with DINO v1's
exact module structure, run through the port and through the vendored extractor alike.
``tests/slow/test_dinobot_parity.py`` runs the real library with the real weights.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vsbench.vs.baselines import dinobot as db
from vsbench.vs.baselines.control import CapOnly, CnsPlateau, StopObservation, build_stop_rule
from vsbench.vs.baselines.features import log_bin
from vsbench.vs.baselines.match import Matches
from vsbench.vs.bench import arms, runner
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.scene import MockPlaneScene, Pose, rodrigues, rotation_error_deg

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "tests" / "fixtures" / "reference"
ARMS = REPO / "configs" / "arms"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref_ext():
    """The vendored ``extractor.py`` excerpt (the library's ``ViTExtractor``)."""
    return _module("dvf_ref_extractor", REF / "dino_vit_features" / "extractor.py")


@pytest.fixture(scope="module")
def ref_cor():
    """The vendored ``correspondences.py`` excerpt, its ``KMeans`` bound to ``n_init=10``.

    The library's pinned scikit-learn (1.0.1) ran ``n_init=10`` for
    ``KMeans(n_clusters, random_state=0)``; today's default is one run. The call is not
    edited - the module-level name is bound, as the library's environment bound it.
    """
    from sklearn.cluster import KMeans

    mod = _module("dvf_ref_correspondences", REF / "dino_vit_features" / "correspondences.py")
    mod.KMeans = functools.partial(KMeans, n_init=10)
    return mod


@pytest.fixture(scope="module")
def gist():
    return _module("dinobot_gist_functions", REF / "dinobot_gist_functions.py")


# ------------------------------------------------------------ stand-ins
class TinyDinoViT(torch.nn.Module):
    """DINO v1's ``VisionTransformer`` structure at toy size, random weights.

    The same modules and forward as ``facebookresearch/dino`` ``vision_transformer.py``
    (``PatchEmbed`` with an int ``patch_size`` and a ``proj`` conv, ``Attention`` that
    returns ``(x, attn)`` and runs ``attn_drop`` on the probabilities, ``Block`` with
    ``norm1``/``attn``/``norm2``/``mlp``, ``interpolate_pos_encoding`` over a 28 x 28
    training grid, ``prepare_tokens`` naming the tensor's dims ``w, h`` as DINO does):
    24-d, 6 heads (the saliency reads heads 0, 2, 4, 5), 12 blocks (the key facet is
    block 9, the saliency block 11).
    """

    class _Attn(torch.nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.num_heads = heads
            self.scale = (dim // heads) ** -0.5
            self.qkv = torch.nn.Linear(dim, dim * 3, bias=True)
            self.attn_drop = torch.nn.Dropout(0.0)
            self.proj = torch.nn.Linear(dim, dim)
            self.proj_drop = torch.nn.Dropout(0.0)

        def forward(self, x):
            B, N, C = x.shape  # noqa: N806 - DINO's own names
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
                2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj_drop(self.proj(x)), attn

    class _Block(torch.nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(dim, eps=1e-6)
            self.attn = TinyDinoViT._Attn(dim, heads)
            self.norm2 = torch.nn.LayerNorm(dim, eps=1e-6)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, 4 * dim), torch.nn.GELU(),
                                           torch.nn.Linear(4 * dim, dim))

        def forward(self, x):
            y, _attn = self.attn(self.norm1(x))
            x = x + y
            return x + self.mlp(self.norm2(x))

    class _PatchEmbed(torch.nn.Module):
        def __init__(self, patch, dim):
            super().__init__()
            self.patch_size = patch
            self.proj = torch.nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)

    def __init__(self, dim: int = 24, heads: int = 6, depth: int = 12, patch: int = 8):
        super().__init__()
        self.patch_embed = TinyDinoViT._PatchEmbed(patch, dim)
        n = (224 // patch) ** 2
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.pos_drop = torch.nn.Dropout(0.0)
        self.blocks = torch.nn.ModuleList(TinyDinoViT._Block(dim, heads) for _ in range(depth))
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)

    def interpolate_pos_encoding(self, x, w, h):  # pragma: no cover - replaced at stride 4
        raise AssertionError("the stride patch must replace this")

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape  # noqa: N806
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


class StubExtractor:
    """:class:`vsbench.vs.baselines.dinobot.DinoV1Extractor`'s ``encode`` without a network.

    Each token's descriptor is its own 8 x 8 patch of the resized grey picture
    (zero-mean, unit-norm) plus a small code of its lattice position, so descriptors are
    unique and a picture matches itself; the saliency is the patch's contrast,
    min-max normalised (a flat background is not foreground). Deterministic.
    """

    device = "cpu"
    weights = weights_md5 = hub_commit = hub_code_md5 = None

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, rgb, *, load_size, layer, facet, bin_hierarchy) -> db.DinoFeatures:
        self.calls += 1
        pil = Image.fromarray(np.asarray(rgb, np.uint8))
        w, h = pil.size
        short, long = (w, h) if w <= h else (h, w)
        nw, nh = ((load_size, int(load_size * long / short)) if w <= h
                  else (int(load_size * long / short), load_size))
        g = np.asarray(pil.resize((nw, nh), Image.Resampling.LANCZOS).convert("L"), np.float32)
        gh, gw = 1 + (nh - 8) // 4, 1 + (nw - 8) // 4
        win = np.lib.stride_tricks.sliding_window_view(g, (8, 8))[::4, ::4][:gh, :gw]
        win = win.reshape(gh, gw, 64)
        zm = win - win.mean(axis=-1, keepdims=True)
        desc = zm / (np.linalg.norm(zm, axis=-1, keepdims=True) + 1e-3)
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
        desc = np.concatenate([desc, 0.05 * np.stack([yy / gh, xx / gw], -1)], -1)
        con = win.std(axis=-1).reshape(-1)
        sal = (con - con.min()) / max(float(con.max() - con.min()), 1e-12)
        return db.DinoFeatures(desc=torch.from_numpy(desc.reshape(gh * gw, -1).astype(np.float32)),
                               saliency=torch.from_numpy(sal.astype(np.float32)),
                               grid=(gh, gw), load_hw=(nh, nw), p=8, stride=(4, 4), binned=False)


def dinobot_config(**over) -> dict:
    """``configs/arms/dinobot.yaml`` with the fast tier's device and ``over`` merged in."""
    cfg = cfgmod.load(ARMS / "dinobot.yaml")
    cfg["matcher"]["device"] = "cpu"
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfgmod.resolve(cfg)


def test_importing_the_module_pulls_no_scikit_learn():
    """scikit-learn is imported where the K-means runs (owner rule), never at import."""
    import subprocess
    import sys

    code = ("import sys, vsbench.vs.baselines.dinobot, vsbench.vs.bench.arms; "
            "print('sklearn' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=REPO)
    assert out.stdout.strip() == "False"


# ======================================================= 1. log_bin(hierarchy=2)
@pytest.mark.parametrize(("h", "w"), [(7, 11), (55, 98), (1, 5), (4, 1)])
@pytest.mark.parametrize("hierarchy", [1, 2])
def test_log_bin_equals_the_vendored_loop_bit_for_bit(ref_ext, h, w, hierarchy):
    """17 bins (the 3x3 at step 1, then the step-3 ring of a 3x3-averaged map, borders
    clamped) and the published 9 - against ``ViTExtractor._log_bin`` itself."""
    torch.manual_seed(h * 100 + w)
    x = torch.randn(1, 6, h * w, 4)                      # B x heads x tokens x d, CLS gone
    fake = types.SimpleNamespace(num_patches=(h, w), device="cpu")
    want = ref_ext.ViTExtractor._log_bin(fake, x, hierarchy=hierarchy)[0, 0]
    tokens = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)   # the library's order
    got = log_bin(tokens, (h, w), hierarchy=hierarchy)[0]
    assert got.shape == want.shape == (h * w, (1 + 8 * hierarchy) * 24)
    assert torch.equal(got, want)


def test_log_bin_hierarchy_1_is_what_it_was():
    """The published ViT-VS gather is untouched by the second level (its own parity test
    against the frozen ViT-VS extractor lives in test_features_dinov2.py)."""
    torch.manual_seed(3)
    x = torch.randn(5 * 7, 16)
    two = log_bin(x, (5, 7), hierarchy=2)
    assert torch.equal(two[:, : 9 * 16], log_bin(x, (5, 7), hierarchy=1))


# ============================================= the extractor chain (tiny network)
def test_the_extractor_port_is_the_librarys_chain(ref_ext):
    """Stride patch, position-encoding interpolation, the key hook, the 17-bin log-binning
    and the saliency map: the port and the vendored ``ViTExtractor`` on the same network
    (DINO v1's structure, random weights), bit for bit."""
    torch.manual_seed(0)
    base = TinyDinoViT()
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (54, 96, 3), dtype=np.uint8)          # a 16:9 frame
    port = db.DinoV1Extractor("dino_vits8", 4, device="cpu", model=copy.deepcopy(base))
    lib = ref_ext.ViTExtractor("dino_vits8", 4, model=copy.deepcopy(base), device="cpu")
    assert (port.p, port.stride) == (lib.p, lib.stride) == (8, (4, 4))
    feats = port.encode(rgb, load_size=32, layer=9, facet="key", bin_hierarchy=2)
    batch = port.preprocess(rgb, 32)
    with torch.no_grad():
        want_desc = lib.extract_descriptors(batch, 9, "key", True)[0, 0]
        want_sal = lib.extract_saliency_maps(batch)[0]
    assert feats.load_hw == (32, 56) and feats.grid == lib.num_patches == (7, 13)
    assert feats.desc.shape == want_desc.shape == (7 * 13, 17 * 24)
    assert torch.equal(feats.desc, want_desc)
    assert torch.equal(feats.saliency, want_sal)
    # the other facets and the unbinned path go through the same hooks
    for facet, hier in (("query", 2), ("value", 0), ("token", 2)):
        got = port.encode(rgb, load_size=32, layer=9, facet=facet, bin_hierarchy=hier).desc
        with torch.no_grad():
            want = lib.extract_descriptors(batch, 9, facet, bool(hier))[0, 0]
        assert torch.equal(got, want), facet


def test_the_preprocess_is_torchvisions_resize_totensor_normalize():
    """``Resize(224, LANCZOS)`` on the PIL picture + ToTensor + Normalize, reproduced with
    Pillow and ``(x/255 - mean)/std``: identical, and 1920 x 1080 becomes 398 x 224."""
    tv = pytest.importorskip("torchvision.transforms")
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    ex = db.DinoV1Extractor("dino_vits8", 4, device="cpu", model=TinyDinoViT())
    got = ex.preprocess(rgb, 224)
    pil = tv.Resize(224, interpolation=tv.InterpolationMode.LANCZOS)(Image.fromarray(rgb))
    want = tv.Compose([tv.ToTensor(), tv.Normalize(mean=db.IMAGENET_MEAN,
                                                   std=db.IMAGENET_STD)])(pil)[None, ...]
    assert tuple(got.shape) == (1, 3, 224, 398) and got.is_contiguous()
    assert torch.equal(got, want)
    square = ex.preprocess(rgb, [224, 224])                     # the paper's squash
    assert tuple(square.shape) == (1, 3, 224, 224)


def test_the_network_is_never_downloaded(monkeypatch, tmp_path):
    """An empty torch.hub cache is refused BEFORE torch.hub.load is called, with the
    staging command; nothing is fetched."""
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    def _no(*_a, **_k):  # pragma: no cover - reaching it is the failure
        raise AssertionError("torch.hub.load must not be reached without a cache")

    monkeypatch.setattr(torch.hub, "load", _no)
    with pytest.raises(RuntimeError, match="never downloads weights") as exc:
        db.DinoV1Extractor("dino_vits8", 4, device="cpu")
    assert "dino_deitsmall8_pretrain.pth" in str(exc.value)
    assert "torch.hub.load('facebookresearch/dino:main', 'dino_vits8')" in str(exc.value)


def test_the_direct_loader_is_torch_hubs_network():
    """The cached repository's ``vision_transformer`` + ``utils``, executed directly, is
    the network ``torch.hub.load`` builds: the same parameters bit for bit and the same
    forward. (The direct build exists because inside the Isaac application a module
    named ``utils`` - OpenCV's bundled package - shadows the repository's, and torch.hub's
    own import then fails; station rig check 2026-09-23.) Skips where the cache is absent."""
    hub = Path(torch.hub.get_dir())
    repo, ckpt = hub / db.HUB_DIR_NAME, hub / "checkpoints" / db.CHECKPOINTS["dino_vits8"]
    if not (repo / "hubconf.py").is_file() or not ckpt.is_file():
        pytest.skip("DINO v1 ViT-S/8 is not in this machine's torch.hub cache (staged by hand)")
    saved_utils = sys.modules.get("utils")
    ours = db.DinoV1Extractor("dino_vits8", 8, device="cpu")       # stride 8: the patch is a no-op
    assert sys.modules.get("utils") is saved_utils                  # nothing leaked
    theirs = torch.hub.load(db.HUB_REPO, "dino_vits8", source="github", trust_repo=True,
                            skip_validation=True, verbose=False).eval()
    a, b = ours.model.state_dict(), theirs.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(ours.model(x), theirs(x))
    assert ours.weights_md5 == "230cd55a58a3273221a39401ff6cb904"


# ============================ 2. best buddies + foreground + K-means + point centres
def _random_features(seed: int, grid, heads=6, d=4, sal=None) -> tuple:
    """Descriptors from random keys through the vendored ``_log_bin`` - so the reference
    sees the library's own layout - and the port's contiguous copy of the same numbers."""
    g = torch.Generator().manual_seed(seed)
    h, w = grid
    x = torch.randn(1, heads, h * w, d, generator=g)
    fake = types.SimpleNamespace(num_patches=grid, device="cpu")
    ref_ext = _module("dvf_ref_extractor_2", REF / "dino_vit_features" / "extractor.py")
    lib_desc = ref_ext.ViTExtractor._log_bin(fake, x)           # 1 x 1 x N x 17 hd, a view
    if sal is None:
        sal = torch.rand(h * w, generator=g)
        sal = (sal - sal.min()) / (sal.max() - sal.min())
    feats = db.DinoFeatures(desc=lib_desc[0, 0].contiguous(), saliency=sal.float(),
                            grid=grid, load_hw=(4 * h + 4, 4 * w + 4), p=8, stride=(4, 4),
                            binned=True)
    return lib_desc, feats


def _port_vs_vendored(ref_cor, goal, live, lib1, lib2, num_pairs=8, thresh=0.05):
    """The vendored core and the port on the same inputs; the vendored K-means runs on the
    port's one thread (a multi-threaded float32 fit is not repeatable, see KMEANS_THREADS)."""
    from threadpoolctl import threadpool_limits

    stand_in = types.SimpleNamespace(stride=(4, 4), p=8)
    with threadpool_limits(limits=db.KMEANS_THREADS):
        p1, p2, _, _ = ref_cor.find_correspondences_core(
            lib1, lib2, goal.saliency, live.saliency, goal.grid, live.grid, stand_in,
            num_pairs=num_pairs, thresh=thresh, device="cpu")
    got = db.select_pairs(goal, live, num_pairs=num_pairs, thresh=thresh, n_init=10)
    return np.asarray(p1, np.int64).reshape(-1, 2), np.asarray(p2, np.int64).reshape(-1, 2), got


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_selection_is_the_librarys_on_random_descriptors(ref_cor, seed):
    lib1, goal = _random_features(10 + seed, (9, 14))
    lib2, live = _random_features(20 + seed, (9, 14))
    want1, want2, got = _port_vs_vendored(ref_cor, goal, live, lib1, lib2)
    assert len(want1) == 8
    np.testing.assert_array_equal(got["points1"], want1)
    np.testing.assert_array_equal(got["points2"], want2)
    # the point centres are 4 * index + 4 in the resized picture, (y, x) order
    np.testing.assert_array_equal(got["points1"],
                                  np.column_stack([got["idx1"] // 14, got["idx1"] % 14]) * 4 + 4)


def test_the_selection_matches_a_shifted_copy_and_ports_the_foreground_quirk(ref_cor):
    """A live picture that is the goal shifted by one column: every buddy is the shifted
    token; the goal foreground is kept when its token is the nearest neighbour of ANY
    foreground live token (``nn_2[fg_mask2]``), exactly as the library applies it."""
    lib1, goal = _random_features(5, (8, 12))
    grid = goal.grid
    d = goal.desc.reshape(8, 12, -1)
    shifted = torch.roll(d, shifts=1, dims=1).reshape(96, -1).contiguous()
    lib2 = db.library_layout(shifted, grid, True)
    sal2 = torch.roll(goal.saliency.reshape(8, 12), 1, dims=1).reshape(-1)
    live = db.DinoFeatures(desc=shifted, saliency=sal2, grid=grid, load_hw=goal.load_hw,
                           p=8, stride=(4, 4), binned=True)
    want1, want2, got = _port_vs_vendored(ref_cor, goal, live, lib1, lib2)
    np.testing.assert_array_equal(got["points1"], want1)
    np.testing.assert_array_equal(got["points2"], want2)
    assert np.all((want2[:, 1] - want1[:, 1]) % 48 == 4)      # one column = 4 resized px


def test_near_ties_are_decided_on_the_librarys_own_rows(ref_cor):
    """Planted exact duplicates and 1e-7 near-duplicates in the live descriptors: the
    product and the library's per-row cosine may order those differently in the last bit,
    and the port must still return the library's argmaxes and pairs - which it does by
    recomputing the ambiguous rows with the library's own arithmetic. Every token is
    foreground here, so every row and every column is one the library reads."""
    ones = torch.ones(9 * 14)
    lib1, goal = _random_features(31, (9, 14), sal=ones)
    _, live = _random_features(32, (9, 14), sal=ones)
    desc = live.desc.clone()
    g = torch.Generator().manual_seed(7)
    for src in (3, 17, 40, 77):                     # each goal token gets a twin in live
        desc[src + 1] = goal.desc[src] + 1e-7 * torch.randn(desc.shape[1], generator=g)
        desc[src + 2] = goal.desc[src]
    live = db.DinoFeatures(desc=desc, saliency=ones, grid=live.grid, load_hw=live.load_hw,
                           p=8, stride=(4, 4), binned=True)
    lib2 = db.library_layout(desc, live.grid, True)
    fg = torch.ones(9 * 14, dtype=torch.bool)
    _sim_1, nn_1, nn_2, info = db.best_buddies(goal, live, fg, fg)
    assert info["n_exact_rows"] >= 4                        # the planted rows were re-decided
    assert info["similarity_route"] == "split"              # the fast form passed its check
    want = ref_cor.chunk_cosine_sim(lib1, lib2)[0, 0]
    assert torch.equal(nn_1, torch.max(want, dim=-1)[1])
    assert torch.equal(nn_2, torch.max(want, dim=-2)[1])
    want1, want2, got = _port_vs_vendored(ref_cor, goal, live, lib1, lib2)
    np.testing.assert_array_equal(got["points1"], want1)
    np.testing.assert_array_equal(got["points2"], want2)


def test_the_library_rows_are_the_librarys_call_bit_for_bit(ref_cor):
    """`_library_rows` (the live normalisation formed once per frame) is the library's
    `chunk_cosine_sim` row for row, bit for bit; and a broken premise falls back to the
    library's whole loop rather than returning a product argmax."""
    lib1, goal = _random_features(71, (9, 14))
    lib2, live = _random_features(72, (9, 14))
    rows, route = db._library_rows(lib1, lib2, [0, 5, 60, 125])
    want = ref_cor.chunk_cosine_sim(lib1, lib2)[0, 0]
    assert route == "split"
    for i, row in rows.items():
        assert torch.equal(row, want[i])
    fg = torch.ones(9 * 14, dtype=torch.bool)
    _, nn_1, _, info = db.best_buddies(goal, live, fg, fg, tie_tol=2.0)   # every row "ties"
    assert info["n_exact_rows"] == 9 * 14
    assert torch.equal(nn_1, torch.max(want, dim=-1)[1])
    # a band narrower than the product's own rounding: an exact duplicate ties at 0, its row
    # is recomputed, moves by more than half the band - and the whole loop takes over
    desc = live.desc.clone()
    desc[10] = goal.desc[3]
    desc[11] = goal.desc[3]
    twin = db.DinoFeatures(desc=desc, saliency=live.saliency, grid=live.grid,
                           load_hw=live.load_hw, p=8, stride=(4, 4), binned=True)
    _, nn_1, nn_2, info = db.best_buddies(goal, twin, fg, fg, tie_tol=1e-12)
    want = ref_cor.chunk_cosine_sim(lib1, db.library_layout(desc, twin.grid, True))[0, 0]
    assert info["similarity_route"] == "loop" and info["n_exact_rows"] == 9 * 14
    assert torch.equal(nn_1, torch.max(want, dim=-1)[1])
    assert torch.equal(nn_2, torch.max(want, dim=-2)[1])


def test_fewer_buddies_than_eight_are_all_kept(ref_cor):
    sal = torch.zeros(9 * 14)
    sal[[5, 30, 61, 90, 111]] = 1.0                          # five foreground goal tokens
    lib1, goal = _random_features(41, (9, 14), sal=sal)
    lib2 = db.library_layout(goal.desc, goal.grid, True)     # the goal as the live picture
    live = db.DinoFeatures(desc=goal.desc.clone(), saliency=torch.ones(9 * 14),
                           grid=goal.grid, load_hw=goal.load_hw, p=8, stride=(4, 4),
                           binned=True)
    want1, want2, got = _port_vs_vendored(ref_cor, goal, live, lib1, lib2)
    assert len(want1) == got["n_clusters"] == got["n_buddies"] == 5
    np.testing.assert_array_equal(got["points1"], want1)
    np.testing.assert_array_equal(got["points2"], want2)


def test_an_empty_pool_is_no_pairs_where_the_library_crashes(ref_cor):
    """Deviation 2: the library calls KMeans(n_clusters=0) and raises; the port returns no
    pairs (the arm then commands no motion and says `degenerate`)."""
    lib1, goal = _random_features(51, (9, 14), sal=torch.zeros(9 * 14))
    lib2, live = _random_features(52, (9, 14))
    stand_in = types.SimpleNamespace(stride=(4, 4), p=8)
    with pytest.raises(ValueError):
        ref_cor.find_correspondences_core(lib1, lib2, goal.saliency, live.saliency,
                                          goal.grid, live.grid, stand_in, num_pairs=8,
                                          thresh=0.05, device="cpu")
    got = db.select_pairs(goal, live, num_pairs=8, thresh=0.05, n_init=10)
    assert got["n_buddies"] == 0 and got["points1"].shape == (0, 2)


def test_n_init_is_passed_through_as_stated(ref_cor):
    """``kmeans_n_init`` reaches KMeans: at 1 the port equals the library run with 1."""
    from sklearn.cluster import KMeans

    lib1, goal = _random_features(61, (9, 14))
    lib2, live = _random_features(62, (9, 14))
    from threadpoolctl import threadpool_limits

    one = _module("dvf_ref_correspondences_n1", REF / "dino_vit_features" / "correspondences.py")
    one.KMeans = functools.partial(KMeans, n_init=1)
    with threadpool_limits(limits=db.KMEANS_THREADS):
        p1, p2, _, _ = one.find_correspondences_core(lib1, lib2, goal.saliency, live.saliency,
                                                     goal.grid, live.grid,
                                                     types.SimpleNamespace(stride=(4, 4), p=8),
                                                     num_pairs=8, thresh=0.05, device="cpu")
    got = db.select_pairs(goal, live, num_pairs=8, thresh=0.05, n_init=1)
    np.testing.assert_array_equal(got["points1"], np.asarray(p1).reshape(-1, 2))
    np.testing.assert_array_equal(got["points2"], np.asarray(p2).reshape(-1, 2))


# ======================================================================= 3. kabsch
def _rigid(seed):
    """Eight 3-D points in front of a camera, and a rigid motion."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-0.15, 0.15, 8), rng.uniform(-0.1, 0.1, 8),
                         rng.uniform(0.4, 0.8, 8)])
    ax = rng.normal(size=3)
    R = rodrigues(ax / np.linalg.norm(ax), np.deg2rad(rng.uniform(1, 40)))
    t = rng.normal(scale=0.05, size=3)
    return X, R, t


@pytest.mark.parametrize("seed", range(5))
def test_kabsch_is_the_gists_find_transformation_and_recovers_a_rigid_motion(gist, seed):
    X, R, t = _rigid(seed)
    Y = X @ R.T + t
    Rg, tg = gist.find_transformation(X, Y)
    Rp, tp = db.kabsch(X, Y)
    assert np.array_equal(Rp, Rg) and np.array_equal(tp, tg)       # the same arithmetic
    np.testing.assert_allclose(Rp, R, atol=1e-9)
    np.testing.assert_allclose(tp, t, atol=1e-9)
    assert db.compute_error(X, Y) == gist.compute_error(X, Y)


def test_a_mirrored_set_gets_the_improper_matrix_and_the_arm_does_not_move(gist):
    """The published formula has no reflection check: a mirrored set comes back with
    det(R) = -1, exactly as the gist returns it. The arm then commands no motion and
    flags it; `reflection: fix` (Sorkine-Hornung) returns a proper rotation instead."""
    X, _, _ = _rigid(9)
    Y = X * np.array([-1.0, 1.0, 1.0])                        # a mirror image, still in view
    Rg, _ = gist.find_transformation(X, Y)
    Rp, _ = db.kabsch(X, Y)
    assert np.array_equal(Rp, Rg) and np.linalg.det(Rp) == pytest.approx(-1.0)
    Rf, _ = db.kabsch(X, Y, reflection="fix")
    assert np.linalg.det(Rf) == pytest.approx(1.0)
    intr = MockPlaneScene(render_scale=0.125).intrinsics
    align = db.DinoBotAlign(intrinsics=intr, err_space="metric")   # pairs given in pixels only
    px_g, zg = _project(X, intr)
    px_c, zc = _project(Y, intr)
    align.set_goal_depth(_depth_map(px_g, zg, intr))
    m = db.pairs_to_matches(px_g, px_c, image_wh=(intr.width, intr.height), sim=np.ones(8),
                            pool_size=8, num_landmarks=8)
    twist, debug, keep = align.step(m, _depth_map(px_c, zc, intr))
    assert debug["reflection"] and not debug["degenerate"] and keep.all()
    assert np.all(twist.as_array() == 0.0) and debug["fit_det"] < 0
    assert np.isfinite(debug["feature_error_norm"])           # the pairs were real


def test_on_a_plane_the_published_fit_is_improper_about_half_the_time():
    """Why the posters matter: eight COPLANAR points leave the third singular direction to
    rounding, so the published formula returns a reflection on about half of all fits
    (49.7 % of these 2000 on the laptop) - and the no-motion guard then holds the camera
    still. The identity (the same set twice) is always proper."""
    rng = np.random.default_rng(0)
    improper = 0
    for _ in range(2000):
        X = np.column_stack([rng.uniform(-0.3, 0.3, 8), rng.uniform(-0.2, 0.2, 8),
                             np.full(8, 0.434)])
        ax = rng.normal(size=3)
        R = rodrigues(ax / np.linalg.norm(ax), np.deg2rad(rng.uniform(0, 20)))
        Rh, _ = db.kabsch(X, X @ R.T + rng.normal(scale=0.03, size=3))
        improper += np.linalg.det(Rh) < 0
        Ri, _ = db.kabsch(X, X.copy())
        assert np.linalg.det(Ri) > 0
    assert 0.3 < improper / 2000 < 0.7


# ================================================== 4. the round trip on the mock
def _project(P_cam: np.ndarray, intr) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P_cam, np.float64)
    return (np.column_stack([P[:, 0] / P[:, 2] * intr.fx + intr.cx,
                             P[:, 1] / P[:, 2] * intr.fy + intr.cy]), P[:, 2])


def _depth_map(px, z, intr, fill=2.0) -> np.ndarray:
    """A depth map holding each point's depth at its truncated pixel (the bench's lookup)."""
    d = np.full((intr.height, intr.width), fill, np.float64)
    u = np.clip(px[:, 0].astype(np.int64), 0, intr.width - 1)
    v = np.clip(px[:, 1].astype(np.int64), 0, intr.height - 1)
    assert len(set(zip(u.tolist(), v.tolist(), strict=True))) == len(u), "pixels must differ"
    d[v, u] = z
    return d


class _PairsMatcher:
    """A matcher that returns given pixel pairs (with the library's resized-picture points)."""

    name = "given_pairs"
    tie_order = "given"
    frozen_landmarks = False

    def __init__(self, goal_px, cur_px, image_wh, scale=1.0):
        self.goal_px, self.cur_px = np.asarray(goal_px), np.asarray(cur_px)
        self.image_wh = tuple(image_wh)
        self.scale = scale

    def set_episode(self, scene, start_id):
        pass

    def reset(self, goal_rgb, goal_depth=None, goal_mask=None):
        pass

    def match(self, rgb, mask=None):
        k = len(self.goal_px)
        return db.pairs_to_matches(
            self.goal_px, self.cur_px, image_wh=self.image_wh, sim=np.ones(k), pool_size=k,
            num_landmarks=k, extras={"goal_load_yx": self.goal_px[:, ::-1] * self.scale,
                                     "cur_load_yx": self.cur_px[:, ::-1] * self.scale})

    def record_fields(self):
        return {}

    def close(self):
        pass


def _round_trip(step_fraction: float, reflection: str = "none"):
    """Eight NON-coplanar world points (on and above the mock's plate), seen from the
    mock's goal pose and from a start pose; one arm step + one scene step."""
    scene = MockPlaneScene(render_scale=0.25)
    intr, goal = scene.intrinsics, scene.goal_pose
    ax = np.array([0.3, -0.5, 1.0])
    start = Pose(position=goal.position + np.array([0.021, -0.013, 0.034]),
                 rotation=goal.rotation @ rodrigues(ax / np.linalg.norm(ax), np.deg2rad(6.0)))
    rng = np.random.default_rng(4)
    world = np.column_stack([rng.uniform(-0.12, 0.12, 8), rng.uniform(-0.07, 0.07, 8),
                             rng.uniform(0.0, 0.12, 8)])            # heights above the plate

    def in_cam(pose):
        return (world - pose.position) @ pose.rotation              # R^T (P - p), row-wise

    Xg, Xc = in_cam(goal), in_cam(start)
    pg, zg = _project(Xg, intr)
    pc, zc = _project(Xc, intr)
    arm = db.DinoBotArm(_PairsMatcher(pg, pc, (intr.width, intr.height)),
                        db.DinoBotAlign(intrinsics=intr, step_fraction=step_fraction,
                                        reflection=reflection),
                        CapOnly(), {}, name="test", err_threshold=50.0)
    arm.reset(np.zeros((intr.height, intr.width, 3), np.uint8), _depth_map(pg, zg, intr))
    twist, debug, used = arm.step(np.zeros((intr.height, intr.width, 3), np.uint8),
                                  _depth_map(pc, zc, intr))
    scene.reset(start)
    scene.apply(twist.as_array(), arm.dt)
    return scene, start, goal, debug, used


def test_one_step_lands_the_camera_on_the_goal():
    """The sign convention (goal -> live fit = the move in the camera's own frame) and the
    jump -> twist conversion, proven together: to 1e-6 m and 1e-6 rad (measured ~1e-15)."""
    scene, start, goal, debug, used = _round_trip(1.0)
    assert not debug["degenerate"] and not debug["reflection"] and debug["n_matches"] == 8
    assert np.linalg.norm(scene.pose.position - goal.position) < 1e-6
    assert np.radians(rotation_error_deg(scene.pose.rotation, goal.rotation)) < 1e-6
    assert debug["jump_t_mm"] == pytest.approx(
        np.linalg.norm(start.position - goal.position) * 1000.0, rel=1e-9)
    assert isinstance(used, Matches) and len(used) == 8


def test_half_a_jump_lands_half_way():
    """``step_fraction 0.5`` = ``(R^0.5, 0.5 t)``: the midpoint, and half the rotation."""
    scene, start, goal, _, _ = _round_trip(0.5)
    np.testing.assert_allclose(scene.pose.position, 0.5 * (start.position + goal.position),
                               atol=1e-9)
    full = rotation_error_deg(start.rotation, goal.rotation)
    assert rotation_error_deg(scene.pose.rotation, goal.rotation) == pytest.approx(full / 2,
                                                                                    rel=1e-6)


def test_the_jump_twist_is_the_transform_over_one_period():
    ax = np.array([0.0, 0.0, 1.0])
    R = rodrigues(ax, np.deg2rad(30.0))
    v = db.jump_twist(R, np.array([0.05, 0.0, 0.0]), dt=1 / 60)
    np.testing.assert_allclose(v, [3.0, 0, 0, 0, 0, np.deg2rad(30.0) * 60], atol=1e-12)


# ======================================================================= 5. identity
@pytest.fixture(scope="module")
def mock():
    scene = MockPlaneScene(render_scale=0.25)
    scene.reset(scene.goal_pose)
    rgb, depth, _ = scene.render()
    return scene, rgb, depth


def _stub_arm(cfg=None, extractor=None, intr=None):
    ex = extractor or StubExtractor()
    arm = db.build_dinobot_arm(cfg or dinobot_config(dinobot={"load_size": 128}),
                               intrinsics=intr, extractor=ex)
    return arm, ex


def test_the_goal_picture_as_the_live_one_commands_no_motion(mock):
    """All pairs at zero displacement, a zero twist, and the error 0 in both spaces."""
    scene, rgb, depth = mock
    arm, _ = _stub_arm(intr=scene.intrinsics)
    arm.set_episode(scene.name, "identity")
    arm.reset(rgb, depth)
    twist, debug, used = arm.step(rgb, depth)
    assert debug["n_matches"] == 8 and not debug["degenerate"] and not debug["reflection"]
    np.testing.assert_array_equal(debug["pts_goal"], debug["pts_cur"])
    assert np.abs(twist.as_array()).max() < 1e-9
    assert debug["err_pixel_load"] == 0.0 and debug["err_metric"] == 0.0
    assert debug["feature_error_norm"] == 0.0
    np.testing.assert_array_equal(used.to_rows()["pts_goal"], used.to_rows()["pts_cur"])


def test_the_goal_is_encoded_once_per_episode(mock):
    """The library re-encodes the goal on every call; the port caches it at reset - the
    same numbers (the encoder is deterministic), one forward instead of two per step."""
    scene, rgb, depth = mock
    arm, ex = _stub_arm(intr=scene.intrinsics)
    arm.reset(rgb, depth)
    first = arm.matcher.goal.desc.clone()
    for _ in range(3):
        arm.step(rgb, depth)
    assert ex.calls == 4                                     # 1 goal + 3 live frames
    assert torch.equal(ex.encode(rgb, load_size=128, layer=9, facet="key",
                                 bin_hierarchy=2).desc, first)


# ========================================== 6. the stops: dinobot_err and cns_plateau
def _obs(step, feat):
    return StopObservation(step=step, feature_error_norm=feat)


def test_dinobot_err_fires_on_the_step_after_the_first_reading_under_and_latches():
    """The gist tests `error > ERR_THRESHOLD` after executing the move computed on that
    observation, so DINOBot ends one jump after its first reading under the threshold -
    the step at which this rule latches, and where the runner reads the pose."""
    rule = build_stop_rule({"stop": {"rule": "dinobot_err"}})
    assert rule.name == "dinobot_err" and rule.params["threshold"] == 50.0
    rule.reset()
    seq = [(1, 120.0), (2, float("nan")), (3, 60.0), (4, 49.9), (5, 80.0), (6, 10.0)]
    fired = [rule.update(_obs(s, e)) for s, e in seq]
    assert [v.would_stop for v in fired] == [False, False, False, False, True, True]
    assert fired[-1].fired_step == 5 and fired[-1].detail["under_step"] == 4
    exact = build_stop_rule({"stop": {"rule": "dinobot_err", "threshold": 50.0}})
    exact.reset()
    assert not exact.update(_obs(1, 50.0)).would_stop       # strictly below, as the gist
    assert not exact.update(_obs(2, 50.0)).would_stop
    with pytest.raises(ValueError, match="space"):
        build_stop_rule({"stop": {"rule": "dinobot_err", "space": "pixels"}})


def test_the_arm_reports_its_error_in_the_stated_space(mock):
    scene, rgb, depth = mock
    shifted = scene.goal_pose.position + np.array([0.004, 0.002, 0.0])
    scene2 = MockPlaneScene(render_scale=0.25)
    scene2.reset(Pose(position=shifted, rotation=scene.goal_pose.rotation))
    live, live_depth, _ = scene2.render()
    out = {}
    for space in ("pixel_load", "metric"):
        arm, _ = _stub_arm(dinobot_config(dinobot={
            "load_size": 128, "err_space": space, "reflection": "fix",
            "err_threshold": db.ERR_THRESHOLDS[space]}), intr=scene.intrinsics)
        arm.reset(rgb, depth)
        _, debug, _ = arm.step(live, live_depth)
        out[space] = debug
        assert debug["feature_error_norm"] == debug[
            "err_pixel_load" if space == "pixel_load" else "err_metric"]
        assert arm.recorded_stop_constants()["dinobot_err"] == {
            "threshold": db.ERR_THRESHOLDS[space], "space": space}
    assert out["pixel_load"]["err_metric"] == out["metric"]["err_metric"]


def test_cns_plateau_runs_on_the_arms_per_pair_normalised_errors(mock):
    """The scored rule reads the same per-pair quantity the IBVS arms hand it:
    ((u_goal - u_live)/fx, (v_goal - v_live)/fy), one norm per pair, row i = node i."""
    scene, rgb, depth = mock
    arm, _ = _stub_arm(intr=scene.intrinsics)
    arm.reset(rgb, depth)
    _, debug, used = arm.step(rgb, depth)
    rows = used.to_rows()
    intr = scene.intrinsics
    want = np.hypot((rows["pts_goal"][:, 0] - rows["pts_cur"][:, 0]) / intr.fx,
                    (rows["pts_goal"][:, 1] - rows["pts_cur"][:, 1]) / intr.fy)
    np.testing.assert_array_equal(debug["normalised_errors"], want)
    assert runner._landmark_ids(used, debug) is None
    rule = CnsPlateau(dt=arm.dt)
    rule.reset()
    v = rule.update(StopObservation.from_debug(debug, twist=np.zeros(6), dt=arm.dt))
    assert v.detail["armed"] and v.detail["cns_error"] == 0.0


def test_the_runner_records_dinobot_err_at_the_arms_own_threshold(mock):
    """The recorded rule is built from the arm's `dinobot:` block, not the rule's default:
    a metric-space arm must not be judged against the pixel constant 50. At the goal the
    metric error is 0 (under 0.02 m from step 1, so the rule fires on step 2); 30 mm off
    it is centimetres - over 0.02 m, far under the default 50, so a default-built rule
    would fire there and the arm's own must not."""
    scene, _, _ = mock
    cfg = dinobot_config(dinobot={"load_size": 128, "err_space": "metric",
                                  "err_threshold": 0.02, "reflection": "fix"},
                         episode={"max_iter": 4})
    g = scene.goal_pose
    recs = {}
    for name, offset in (("at_goal", [0.0, 0.0, 0.0]), ("off", [0.03, -0.02, 0.0])):
        arm, _ = _stub_arm(cfg, intr=scene.intrinsics)
        start = {"start_id": name, "cam_pos": g.position + np.array(offset),
                 "cam_R_opt": g.rotation.tolist(), "starts_seed": 1}
        recs[name] = runner.run_episode(scene, arm, start, cfg)
    rec = recs["at_goal"]
    assert rec["dinobot_err_space"] == "metric" and rec["dinobot_err_threshold"] == 0.02
    assert rec["stops"]["dinobot_err"]["step"] == 2
    assert rec["stop_mode"] == "cns_plateau" and rec["gain"] == 1.0
    assert rec["weighting"] == {"method": "dinobot_kabsch_unweighted"}
    assert rec["feature_error_first"] == 0.0
    off = recs["off"]
    assert 0.02 < off["feature_error_first"] < 50.0
    assert "dinobot_err" not in off["stops"]


# ===================================================================== 7. refusals
@pytest.mark.parametrize(("over", "match"), [
    ({"controller": {"gain": 2.0}}, "controller.gain"),
    ({"controller": {"weighting": {"use_trust": False}}}, "controller.weighting"),
    ({"controller": {"law": "visp"}}, "controller.law"),
    ({"stop": {"rule": "dec_plateau"}}, "`dec` stop"),
    ({"episode": {"record_stops": ["velocity_quiet", "dec_plateau"]}}, "dec_plateau"),
    ({"dinobot": {"step_fractoin": 1.0}}, "unknown dinobot config keys"),
    ({"dinobot": {"model_type": "dino_vitb8"}}, "D2"),
    ({"dinobot": {"min_pairs": 2}}, "three non-collinear"),
    ({"rotation": {"mode": "prealign_4way"}}, "no rotation search"),
    ({"matcher": {"backbone_name": "dinov3_vits16"}}, "not read by the DINOBot arm"),
    ({"head": {"arch": "v1"}}, "config block 'head'"),
])
def test_what_dinobot_does_not_have_is_refused_by_name(over, match):
    cfg = dinobot_config(**over)
    with pytest.raises((ValueError, cfgmod.ConfigError), match=match):
        db.build_dinobot_arm(cfg, intrinsics=None, extractor=StubExtractor())


def test_dinobot_err_is_refused_on_every_other_arm():
    cfg = cfgmod.load(ARMS / "vitvs_dinov3.yaml")
    cfg["matcher"]["device"] = "cpu"
    for over in ({"stop": {"rule": "dinobot_err"}},
                 {"episode": {**cfg["episode"], "record_stops": ["dinobot_err"]}}):
        with pytest.raises(ValueError, match="DINOBot's own stop"):
            arms.build_arm({**cfg, **over}, intrinsics=MockPlaneScene().intrinsics)


def test_the_ibvs_cell_refuses_the_look_then_move_keys_and_the_pre_roll():
    cfg = cfgmod.load(ARMS / "dinobot_ibvs.yaml")
    cfg["matcher"]["device"] = "cpu"
    with pytest.raises(ValueError, match="look-then-move arm"):
        arms.build_arm({**cfg, "dinobot": {**cfg["dinobot"], "reflection": "fix"}},
                       intrinsics=MockPlaneScene().intrinsics, extractor=StubExtractor())
    with pytest.raises(ValueError, match="no rotation search"):
        arms.build_arm({**cfg, "rotation": {"mode": "prealign_4way"}},
                       intrinsics=MockPlaneScene().intrinsics, extractor=StubExtractor())


def test_dinobot_is_an_arm_not_a_matcher():
    from vsbench.vs.baselines.match import MATCHER_BUILDERS

    with pytest.raises(ValueError, match="not a matcher"):
        MATCHER_BUILDERS["dinobot"]("dinobot")
    with pytest.raises(ValueError, match="whole arm"):
        arms.build_arm(dinobot_config(), intrinsics=MockPlaneScene().intrinsics,
                       matcher=object())


# ============================================== 8. the arm file, the label, the record
def test_the_arm_file_builds_states_its_device_and_derives_its_label(mock):
    scene, rgb, depth = mock
    raw = cfgmod.load(ARMS / "dinobot.yaml")
    assert raw["matcher"]["device"] == "cuda" and raw["matcher"]["num_landmarks"] == 8
    assert raw["dinobot"] == {"model_type": "dino_vits8", "load_size": 224, "stride": 4,
                              "layer": 9, "facet": "key", "bin_hierarchy": 2,
                              "saliency_thresh": 0.05, "kmeans_n_init": 10,
                              "step_fraction": 1.0, "min_pairs": 3, "reflection": "none",
                              "err_space": "pixel_load", "err_threshold": 50}
    arm = arms.build_arm(raw, intrinsics=scene.intrinsics, extractor=StubExtractor())
    assert isinstance(arm, db.DinoBotArm) and arm.kind == "method"
    assert arm.name == arms.arm_label(raw) == "dinobot_vits8/k8/jump1/cns_plateau"
    assert arm.matcher.device == "cuda"                      # stated; the stub never uses it
    assert isinstance(arm.stop, CnsPlateau) and arm.stop.dt == pytest.approx(1 / 60)
    for over, token in (({"step_fraction": 0.5}, "jump0.5"), ({"reflection": "fix"}, "reffix"),
                        ({"err_space": "metric"}, "errm"), ({"load_size": [224, 224]},
                                                            "load224x224")):
        cfg = {**raw, "dinobot": {**raw["dinobot"], **over}}
        if over.get("err_space"):
            cfg["dinobot"].pop("err_threshold")               # the space's stated 0.02
        assert token in arms.arm_label(cfg).split("/"), token
    assert "thr0.5" in arms.arm_label({**raw, "dinobot": {**raw["dinobot"],
                                                          "err_threshold": 0.5}})


SPEC_RECORD_FIELDS = (
    "dinobot_model_type", "dinobot_load_size", "dinobot_stride", "dinobot_layer",
    "dinobot_facet", "dinobot_bin_hierarchy", "dinobot_saliency_thresh", "dinobot_num_pairs",
    "dinobot_kmeans_n_init", "dinobot_step_fraction", "dinobot_min_pairs",
    "dinobot_reflection", "dinobot_err_space", "dinobot_err_threshold", "dinobot_weights",
    "dinobot_weights_md5", "dinobot_sklearn_version", "dinobot_torch_hub_commit")


def test_the_record_carries_every_setting_and_no_measurement(mock):
    scene, rgb, depth = mock
    cfg = dinobot_config(dinobot={"load_size": 128}, episode={"max_iter": 3})
    arm, _ = _stub_arm(cfg, intr=scene.intrinsics)
    g = scene.goal_pose
    start = {"start_id": "t1", "cam_pos": g.position + np.array([0.002, 0.001, 0.0]),
             "cam_R_opt": g.rotation.tolist(), "starts_seed": 1}
    telem: list = []
    rec = runner.run_episode(scene, arm, start, cfg, telemetry=telem)
    for f in SPEC_RECORD_FIELDS:
        assert f in rec, f
    import sklearn

    assert rec["dinobot_sklearn_version"] == sklearn.__version__
    assert rec["dinobot_kmeans_threads"] == 1
    assert rec["dinobot_num_pairs"] == 8 and rec["dinobot_bin_hierarchy"] == 2
    assert rec["matcher_variant"] == "dinobot" and rec["matcher_name"] == "dinobot_dino_vits8"
    assert rec["tie_order"] == db.TIE_ORDER and rec["min_matches"] == 3
    assert rec["arm"] == "dinobot_vits8/k8/jump1/load128/cns_plateau"
    fields = arm.record_fields()
    for key in ("n_buddies", "err_pixel_load", "jump_t_mm", "n_matches"):
        assert key not in fields                            # measurements live per step
    assert len(telem) == 3 and "gu" in telem[0]              # the points reach the telemetry


def test_the_matches_are_full_image_pixels_of_the_librarys_points(mock):
    """``to_rows()`` returns the library's resized-picture points times the two axis scales
    (the frame is not squashed square: 1920/398 and 1080/224 differ by 0.06 %)."""
    scene, rgb, depth = mock
    arm, _ = _stub_arm(intr=scene.intrinsics)
    arm.reset(rgb, depth)
    m = arm.matcher.match(rgb)
    assert isinstance(m, Matches) and len(m) == 8
    rows = m.to_rows()
    h, w = rgb.shape[:2]
    lh, lw = m.extras["load_hw"]
    want = m.extras["cur_load_yx"][:, ::-1] * np.array([w / lw, h / lh])
    np.testing.assert_allclose(rows["pts_cur"], want, atol=1e-9)
    assert np.all(np.isnan(rows["margin"])) and np.all(rows["conf"] <= 1.0 + 1e-6)
    assert m.pool_size == m.extras["n_buddies"] >= 8


# ============================================================== 9. the masked cell
def test_the_objects_mask_is_anded_into_the_goal_foreground_at_the_token_lattice(mock):
    scene, rgb, depth = mock
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), bool)
    mask[: h // 2, : w // 2] = True                          # the top-left quarter only
    cfg = dinobot_config(dinobot={"load_size": 128}, matcher={"goal_mask": "objects"})
    arm = arms.build_arm(cfg, intrinsics=scene.intrinsics, extractor=StubExtractor())
    assert arm.goal_mask_mode == "objects" and "objmask" in arm.name.split("/")
    arm.reset(rgb, depth, mask)
    g = arm.matcher.goal
    tok = db.token_mask_from_pixels(mask, load_hw=g.load_hw, grid=g.grid, p=g.p,
                                    stride=g.stride)
    np.testing.assert_array_equal(arm.matcher.goal_fg.numpy(), tok)
    _, debug, used = arm.step(rgb, depth)
    rows = used.to_rows()
    assert len(used) and np.all(rows["pts_goal"][:, 0] < w / 2 + 40)
    assert np.all(rows["pts_goal"][:, 1] < h / 2 + 40)
    free, _ = _stub_arm(intr=scene.intrinsics)
    free.reset(rgb, depth)
    _, debug_free, _ = free.step(rgb, depth)
    assert debug["n_fg_goal"] < debug_free["n_fg_goal"]
    assert debug["n_fg_goal"] == int((tok & (g.saliency > 0.05).numpy()).sum())


def test_the_token_mask_keeps_a_token_when_more_than_a_fifth_is_set():
    """patch_mask_from_pixels's rule (nearest resize, > 127, > 20 %) on the overlapping
    stride-4 lattice of an anisotropic resize."""
    mask = np.zeros((108, 192), bool)
    mask[:, :96] = True                                      # the left half
    tok = db.token_mask_from_pixels(mask, load_hw=(32, 56), grid=(7, 13), p=8,
                                    stride=(4, 4)).reshape(7, 13)
    # window columns [4c, 4c+8) of the 56-wide resize; the half is at 28
    assert tok[:, :6].all() and not tok[:, 7:].any()


# ================================================= the IBVS cell (owner decision D7)
def test_the_ibvs_cell_is_the_same_front_end_behind_the_bench_law(mock):
    scene, rgb, depth = mock
    cfg = cfgmod.load(ARMS / "dinobot_ibvs.yaml")
    cfg["dinobot"]["load_size"] = 128
    cfg["episode"]["max_iter"] = 3
    arm = arms.build_arm(cfg, intrinsics=scene.intrinsics, extractor=StubExtractor())
    assert isinstance(arm.matcher, db.DinoBotCorrespondences)
    assert arm.name == "dinobot_corr@dino_vits8+load128/k8/ur/uniform/cns_plateau"
    raw = cfgmod.load(ARMS / "dinobot_ibvs.yaml")
    assert arms.arm_label(raw) == "dinobot_corr@dino_vits8/k8/ur/uniform/cns_plateau"
    w = arm.controller.weighting
    assert (w.use_trust, w.use_margin, w.per_axis) == (False, False, False)
    assert arm.gain == 2.0 and arm.controller.solve_unrounded
    arm.set_episode(scene.name, "identity")
    arm.reset(rgb, depth)
    twist, debug, matches = arm.step(rgb, depth)
    assert len(matches) == 8 and np.abs(twist.as_array()).max() < 1e-12
    g = scene.goal_pose
    start = {"start_id": "t2", "cam_pos": g.position + np.array([0.002, 0.0, 0.0]),
             "cam_R_opt": g.rotation.tolist(), "starts_seed": 1}
    rec = runner.run_episode(scene, arm, start, cfg)
    assert rec["matcher_variant"] == "dinobot_corr" and rec["dinobot_num_pairs"] == 8
    assert rec["arm"].startswith("dinobot_corr@dino_vits8")


def test_the_protocol_carries_the_cells():
    import yaml

    doc = yaml.safe_load((REPO / "configs" / "protocol.yaml").read_text())
    cells = {e["id"]: e for e in doc["arms"]}
    assert cells["dinobot"]["base"] == "dinobot.yaml"
    assert cells["dinobot_objmask"]["overrides"] == {"matcher": {"goal_mask": "objects"}}
    assert cells["dinobot_ibvs"]["base"] == "dinobot_ibvs.yaml"
    assert cells["dinobot_reffix"]["overrides"] == {"dinobot": {"reflection": "fix"}}
    assert not any("prealign" in str(e.get("overrides")) for k, e in cells.items()
                   if k.startswith("dinobot"))


def test_the_pos_encoding_interpolation_hits_the_stride_4_grid():
    """398 x 224 at stride 4 is 55 x 98 = 5390 tokens (the spec's numbers)."""
    m = TinyDinoViT()
    db.patch_vit_resolution(m, 4)
    x = torch.zeros(1, 1 + 55 * 98, 24)
    pe = m.interpolate_pos_encoding(x, 224, 398)             # DINO names (H, W) as (w, h)
    assert tuple(pe.shape) == (1, 1 + 5390, 24)
    assert m.patch_embed.proj.stride == (4, 4)
    assert math.isclose(float(pe[0, 0, 0]), float(m.pos_embed[0, 0, 0]))
