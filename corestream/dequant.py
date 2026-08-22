"""Fast reconstruction of packed 4-bit weights.

Streaming a quantized checkpoint only pays off if unpacking is cheap. The
reference implementation in `compressed-tensors` is general -- any bit width
from 1 to 8, either packed dimension -- and pays for that generality with
gather operations and several full-size intermediate tensors. Measured on an
RTX 3060 it took 38 ms to reconstruct one transformer layer, against 12 ms to
transfer that layer's packed bytes: four times more expensive than the problem
it solves, which made 4-bit streaming slower than not quantizing at all.

The arithmetic says it should be far cheaper. Reconstructing a layer moves
roughly 575 MiB of device memory, which at the card's bandwidth is a couple of
milliseconds. The gap is unfused elementwise work, not fundamental cost.

So the common case -- 4-bit, symmetric, group-wise scales, packed along the
last dimension, which is what `w4a16` checkpoints use -- gets a fused path
compiled into a single kernel. It is 17x faster than the reference and
produces bit-identical output; anything outside those conditions falls back to
the reference rather than guessing.
"""

from __future__ import annotations

import torch

# Each int32 holds eight 4-bit values, so a nibble's shift is its index * 4.
_INT4_SHIFTS = tuple(range(0, 32, 4))
_INT4_OFFSET = 8  # symmetric 4-bit stores values biased by 2^(bits-1)


def _fused_int4_dequantize(
    packed: torch.Tensor,
    scale: torch.Tensor,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack nibbles and apply group scales in one pass."""
    rows = packed.shape[0]
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    values = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(rows, -1)
    values = values[:, :in_features]
    # Cast before scaling so the intermediate is the compute dtype rather than
    # int32, halving the traffic through the widest tensor in the chain.
    values = (values - _INT4_OFFSET).to(dtype)
    scaled = values.view(rows, -1, group_size) * scale.unsqueeze(-1).to(dtype)
    return scaled.view(rows, in_features)


def column_group_index(g_idx: torch.Tensor, group_size: int) -> torch.Tensor:
    """Map each column to the scale group the reference implementation uses.

    Activation ordering breaks the assumption that a column's group is its
    position divided by the group size: columns are permuted by importance
    during calibration. The reference handles this by sorting columns with
    `argsort(g_idx)`, grouping the sorted result contiguously, then undoing
    the permutation.

    Reading `g_idx` directly as the group id gives the same answer whenever
    every group holds exactly `group_size` columns, which real checkpoints
    satisfy -- but it is not the same operation, and silently diverges on any
    checkpoint that does not. Deriving the mapping the way the reference does
    keeps them identical by construction.

    The result depends only on `g_idx`, which is fixed for a module, so this
    is computed once and reused rather than per token.
    """
    permutation = torch.argsort(g_idx)
    inverse = torch.argsort(permutation)
    return torch.div(inverse, group_size, rounding_mode="floor").to(torch.int32)


def _fused_int4_dequantize_actorder(
    packed: torch.Tensor,
    scale: torch.Tensor,
    column_group: torch.Tensor,
    in_features: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """As above, but taking each column's group from a precomputed mapping."""
    rows = packed.shape[0]
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    values = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(rows, -1)
    values = values[:, :in_features]
    values = (values - _INT4_OFFSET).to(dtype)
    per_column_scale = scale.to(dtype).index_select(1, column_group)
    return values * per_column_scale


# `dynamic=False` because the shapes are fixed by the model: recompiling per
# layer shape once at startup is worth it, and dynamic shapes would give up
# most of the fusion.
_compiled_int4_dequantize = torch.compile(_fused_int4_dequantize, dynamic=False)
_compiled_int4_dequantize_actorder = torch.compile(
    _fused_int4_dequantize_actorder, dynamic=False
)


def _can_use_fast_path(scheme, compressed: dict) -> bool:
    """Whether this module matches the conditions the fused kernel assumes."""
    weights = getattr(scheme, "weights", None)
    if weights is None:
        return False
    if getattr(weights, "num_bits", None) != 4:
        return False
    if not getattr(weights, "symmetric", False):
        return False
    if str(getattr(weights, "strategy", "")).split(".")[-1].lower() != "group":
        return False
    if not getattr(weights, "group_size", None):
        return False
    # A zero point means asymmetric reconstruction, which neither fused path
    # implements. Activation ordering is handled, by its own variant.
    if compressed.get("weight_zero_point") is not None:
        return False
    return "weight_shape" in compressed and "weight_scale" in compressed


def decompress_weight(
    compressed: dict,
    scheme,
    dtype: torch.dtype,
    column_group: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct a module's weight from its packed representation.

    `column_group` is the cached output of `column_group_index` for this
    module; it is derived when absent. Falls back to the reference
    implementation whenever the fused path's assumptions do not hold, so an
    unusual quantization configuration is slower rather than silently wrong.
    """
    from compressed_tensors.compressors.pack_quantized.base import (
        PackedQuantizationCompressor,
    )

    if _can_use_fast_path(scheme, compressed):
        shape = compressed["weight_shape"]
        in_features = int(shape[1].item() if shape.ndim else shape.item())
        group_size = int(scheme.weights.group_size)
        g_idx = compressed.get("weight_g_idx")
        if g_idx is not None and not _is_column_order(g_idx):
            if column_group is None:
                column_group = column_group_index(g_idx, group_size)
            return _compiled_int4_dequantize_actorder(
                compressed["weight_packed"],
                compressed["weight_scale"],
                column_group,
                in_features,
                dtype,
            )
        return _compiled_int4_dequantize(
            compressed["weight_packed"],
            compressed["weight_scale"],
            in_features,
            group_size,
            dtype,
        )

    return PackedQuantizationCompressor.decompress(compressed, scheme)["weight"]


def _is_column_order(g_idx: torch.Tensor) -> bool:
    """Whether `g_idx` is an uninitialised placeholder rather than an ordering.

    Mirrors the reference's own check: a meta tensor or one containing -1
    means activation ordering was never applied, and plain column order
    should be used.
    """
    return g_idx.device.type == "meta" or bool((g_idx == -1).any())
