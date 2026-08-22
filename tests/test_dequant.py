"""Tests for fast reconstruction of packed 4-bit weights.

The fused path replaces a reference implementation, so the bar is bit-exact
agreement rather than approximate. A dequantization that is subtly wrong does
not fail — it produces a model that generates slightly worse text, which no
assertion would catch.
"""

import pytest
import torch

from corestream.dequant import _can_use_fast_path, decompress_weight

pytest.importorskip("compressed_tensors")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class Weights:
    def __init__(self, num_bits=4, symmetric=True, strategy="group", group_size=128):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.strategy = strategy
        self.group_size = group_size


class Scheme:
    def __init__(self, **kwargs):
        self.weights = Weights(**kwargs)


def make_packed(rows, in_features, group_size=128, actorder=False, device="cuda"):
    torch.manual_seed(0)
    packed = torch.randint(
        -(2**31), 2**31 - 1, (rows, in_features // 8), dtype=torch.int32, device=device
    )
    groups = in_features // group_size
    scale = (
        torch.randn(rows, groups, dtype=torch.bfloat16, device=device).abs() + 0.01
    )
    compressed = {
        "weight_packed": packed,
        "weight_scale": scale,
        "weight_shape": torch.tensor([rows, in_features], device=device),
    }
    if actorder:
        # A valid g_idx assigns exactly group_size columns to each group --
        # it is a permutation of the column order, not arbitrary labels.
        # Random labels are not a checkpoint any quantizer would produce.
        labels = torch.arange(groups, device=device).repeat_interleave(group_size)
        permutation = torch.randperm(in_features, device=device)
        compressed["weight_g_idx"] = labels[permutation].to(torch.int32)
    return compressed


def reference(compressed, scheme, dtype):
    from compressed_tensors.compressors.pack_quantized.base import (
        PackedQuantizationCompressor,
    )

    return PackedQuantizationCompressor.decompress(dict(compressed), scheme)[
        "weight"
    ].to(dtype)


@cuda_only
@pytest.mark.parametrize("rows,in_features", [(256, 512), (512, 1024)])
def test_fused_matches_reference_exactly(rows, in_features):
    compressed = make_packed(rows, in_features)
    scheme = Scheme()
    assert _can_use_fast_path(scheme, compressed)
    fast = decompress_weight(compressed, scheme, torch.bfloat16)
    assert torch.equal(fast, reference(compressed, scheme, torch.bfloat16))


@cuda_only
def test_actorder_matches_reference_exactly():
    """Activation ordering reorders columns, so scales come from a gather.

    Treating them as contiguous groups yields a plausible-looking weight built
    from the wrong scales, which is exactly the kind of error that survives
    every test and only shows up as a slightly worse model.
    """
    compressed = make_packed(256, 512, actorder=True)
    scheme = Scheme()
    assert _can_use_fast_path(scheme, compressed)
    fast = decompress_weight(compressed, scheme, torch.bfloat16)
    assert torch.equal(fast, reference(compressed, scheme, torch.bfloat16))


@cuda_only
def test_output_shape_and_dtype():
    compressed = make_packed(128, 256)
    out = decompress_weight(compressed, Scheme(), torch.bfloat16)
    assert out.shape == (128, 256)
    assert out.dtype is torch.bfloat16


def test_guard_rejects_unsupported_configurations():
    """Anything outside the fused path's assumptions must fall back.

    The fallback is slower but correct; guessing would be fast and wrong.
    """
    compressed = {
        "weight_packed": torch.zeros(1, 1, dtype=torch.int32),
        "weight_scale": torch.zeros(1, 1),
        "weight_shape": torch.tensor([1, 8]),
    }
    assert _can_use_fast_path(Scheme(), compressed)

    assert not _can_use_fast_path(Scheme(num_bits=8), compressed)
    assert not _can_use_fast_path(Scheme(symmetric=False), compressed)
    assert not _can_use_fast_path(Scheme(strategy="channel"), compressed)
    assert not _can_use_fast_path(Scheme(group_size=0), compressed)


def test_guard_rejects_zero_point():
    """A zero point means asymmetric reconstruction, which is not implemented."""
    compressed = {
        "weight_packed": torch.zeros(1, 1, dtype=torch.int32),
        "weight_scale": torch.zeros(1, 1),
        "weight_shape": torch.tensor([1, 8]),
        "weight_zero_point": torch.zeros(1, 1, dtype=torch.int32),
    }
    assert not _can_use_fast_path(Scheme(), compressed)


def test_guard_accepts_activation_ordering():
    compressed = {
        "weight_packed": torch.zeros(1, 1, dtype=torch.int32),
        "weight_scale": torch.zeros(1, 1),
        "weight_shape": torch.tensor([1, 8]),
        "weight_g_idx": torch.zeros(8, dtype=torch.int32),
    }
    assert _can_use_fast_path(Scheme(), compressed)
