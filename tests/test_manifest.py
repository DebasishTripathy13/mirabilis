"""Tests for the checkpoint manifest and its packing layout.

The manifest decides where every parameter sits inside its chunk. If an offset
is wrong the model does not crash -- it computes on misaligned bytes and
returns plausible nonsense, which is far harder to notice than a failure. So
the round trip through pack-and-slice is tested directly.
"""

import json
import struct

import pytest
import torch

from corestream.manifest import (
    ALIGNMENT,
    ManifestChunkSource,
    build_manifest,
    _align,
    _layer_index,
)


def write_safetensors(path, tensors: dict[str, torch.Tensor]) -> None:
    """Write a minimal safetensors file without depending on the writer API."""
    header = {}
    offset = 0
    blobs = []
    dtype_names = {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
    }
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        header[name] = {
            "dtype": dtype_names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)

    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for blob in blobs:
            f.write(blob)


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    tensors = {
        "model.embed_tokens.weight": torch.randn(16, 8, dtype=torch.float32),
        "model.norm.weight": torch.randn(8, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.float32),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 12, dtype=torch.float32),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.float32),
        "model.layers.1.mlp.down_proj.weight": torch.randn(8, 12, dtype=torch.float32),
    }
    path = tmp_path / "model.safetensors"
    write_safetensors(path, tensors)
    return str(tmp_path), tensors


def test_layer_index_extraction():
    assert _layer_index("model.layers.7.mlp.up_proj.weight") == 7
    assert _layer_index("model.embed_tokens.weight") is None
    assert _layer_index("model.layers.notanumber.weight") is None


def test_alignment_rounds_up():
    assert _align(0) == 0
    assert _align(1) == ALIGNMENT
    assert _align(ALIGNMENT) == ALIGNMENT
    assert _align(ALIGNMENT + 1) == 2 * ALIGNMENT


def test_manifest_groups_by_layer(checkpoint):
    path, _ = checkpoint
    manifest = build_manifest(path)
    assert manifest.num_layers == 2
    assert manifest.layer_keys == ["layer.0", "layer.1"]
    assert set(manifest.chunks["layer.0"].names) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }


def test_non_layer_weights_are_resident(checkpoint):
    """Weights read on every token must not be streamed.

    Embeddings and the final norm are needed regardless of which layer runs,
    so streaming them would pay transfer cost on every step for no benefit.
    """
    path, _ = checkpoint
    manifest = build_manifest(path)
    assert set(manifest.resident_tensors) == {
        "model.embed_tokens.weight",
        "model.norm.weight",
    }


def test_tensor_offsets_are_aligned(checkpoint):
    path, _ = checkpoint
    manifest = build_manifest(path)
    for chunk in manifest.chunks.values():
        for spec in chunk.tensors:
            assert spec.offset % ALIGNMENT == 0


def test_packed_chunk_round_trips(checkpoint):
    """The central invariant: pack then slice must return the originals.

    A wrong offset here yields misaligned but well-formed tensors, so the
    model would produce confident nonsense rather than an error.
    """
    path, originals = checkpoint
    manifest = build_manifest(path)
    source = ManifestChunkSource(manifest)

    for key, chunk in manifest.chunks.items():
        flat = source.host_tensor(key)
        assert flat.numel() == chunk.total_bytes
        for spec in chunk.tensors:
            torch.testing.assert_close(spec.view_from(flat), originals[spec.name])


def test_chunk_size_matches_declared(checkpoint):
    path, _ = checkpoint
    manifest = build_manifest(path)
    source = ManifestChunkSource(manifest)
    for key in manifest.chunks:
        assert source.nbytes(key) == source.host_tensor(key).numel()


def test_dominant_dtype_follows_the_bytes(checkpoint):
    path, _ = checkpoint
    assert build_manifest(path).dominant_dtype is torch.float32


def test_dominant_dtype_weighs_by_size(tmp_path):
    """Dtype is decided by bytes, not tensor count.

    A checkpoint may hold a handful of small fp32 norms beside large bf16
    weight matrices; the dtype the model computes in is the one the bulk of
    the bytes are stored in.
    """
    write_safetensors(
        tmp_path / "m.safetensors",
        {
            "model.layers.0.tiny_a.weight": torch.randn(2, dtype=torch.float32),
            "model.layers.0.tiny_b.weight": torch.randn(2, dtype=torch.float32),
            "model.layers.0.big.weight": torch.randn(256, 64, dtype=torch.bfloat16),
        },
    )
    assert build_manifest(str(tmp_path)).dominant_dtype is torch.bfloat16


def test_unknown_chunk_raises(checkpoint):
    path, _ = checkpoint
    source = ManifestChunkSource(build_manifest(path))
    with pytest.raises(KeyError):
        source.host_tensor("layer.99")


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest(str(tmp_path))
