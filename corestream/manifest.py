"""Model layout: which tensors form a chunk, and where they sit inside it.

The store moves a chunk as one contiguous buffer, because issuing a separate
copy per tensor would give up most of the achievable bandwidth to per-transfer
overhead -- a transformer layer is seven or more tensors, and an MoE expert
three. That means each chunk needs a layout describing how to slice the buffer
back into named parameters on the far side.

Offsets are computed once, at load, by reading only the safetensors headers.
No weights are touched, so a manifest for a 30 GB checkpoint costs milliseconds
and a few kilobytes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch

# safetensors records dtypes as short strings in its header.
_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# Every tensor starts on a 16-byte boundary. Sub-buffer views must satisfy the
# element alignment of their dtype, and aligned starts also let the copy engine
# and vectorised loads work at full width.
ALIGNMENT = 16


def _align(offset: int) -> int:
    return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass
class TensorSpec:
    """One parameter's identity and position inside its chunk."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int  # byte offset within the packed chunk
    nbytes: int

    def view_from(self, flat: torch.Tensor) -> torch.Tensor:
        """Slice this tensor out of a packed uint8 buffer, without copying."""
        window = flat[self.offset : self.offset + self.nbytes]
        return window.view(self.dtype).view(self.shape)


@dataclass
class ChunkSpec:
    """A group of tensors transferred together as one buffer."""

    key: str
    tensors: list[TensorSpec] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tensors]


@dataclass
class ModelManifest:
    """The full chunk layout for a checkpoint."""

    chunks: dict[str, ChunkSpec]
    layer_keys: list[str]  # chunk keys in execution order
    resident_tensors: list[str]  # weights needed on every token
    tensor_to_file: dict[str, str]
    num_layers: int

    @property
    def total_bytes(self) -> int:
        return sum(c.total_bytes for c in self.chunks.values())

    @property
    def dominant_dtype(self) -> torch.dtype:
        """The dtype most of the streamed weight bytes are stored in.

        Streaming hands the model zero-copy views of the checkpoint's own
        bytes, so the checkpoint's dtype is not a preference -- it is what the
        model will compute in. Requesting a different one cannot be honoured
        without a conversion on every transfer, which would reintroduce
        exactly the per-chunk CPU work the design exists to avoid.
        """
        weight: dict[torch.dtype, int] = {}
        for chunk in self.chunks.values():
            for spec in chunk.tensors:
                weight[spec.dtype] = weight.get(spec.dtype, 0) + spec.nbytes
        if not weight:
            return torch.float32
        return max(weight.items(), key=lambda kv: kv[1])[0]

    def chunk_bytes(self, key: str) -> int:
        return self.chunks[key].total_bytes


def _read_headers(paths: list[str]) -> dict[str, tuple[str, dict]]:
    """Read every shard's safetensors header without materialising tensors.

    The format is an 8-byte little-endian header length followed by that many
    bytes of JSON, so the metadata can be read with two small reads per file
    regardless of how large the shard is.
    """
    out: dict[str, tuple[str, dict]] = {}
    for path in paths:
        with open(path, "rb") as f:
            length = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(length))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            out[name] = (path, info)
    return out


def _layer_index(name: str) -> int | None:
    """Extract the decoder layer index from a parameter name, if it has one."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def find_shards(model_path: str) -> list[str]:
    if os.path.isfile(model_path):
        return [model_path]
    files = sorted(
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith(".safetensors")
    )
    if not files:
        raise FileNotFoundError(f"no .safetensors shards in {model_path}")
    return files


def build_manifest(model_path: str) -> ModelManifest:
    """Group a checkpoint's tensors into per-layer chunks.

    Anything without a layer index -- embeddings, the final norm, the LM head
    -- is marked resident rather than chunked. Those are read on every token
    regardless of routing, so streaming them would pay transfer cost on every
    step for no benefit; they are placed on the device once and left there.
    """
    shards = find_shards(model_path)
    headers = _read_headers(shards)

    chunks: dict[str, ChunkSpec] = {}
    resident: list[str] = []
    tensor_to_file: dict[str, str] = {}
    max_layer = -1

    for name in sorted(headers):
        path, info = headers[name]
        tensor_to_file[name] = path
        dtype = _DTYPE_MAP.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported dtype {info['dtype']!r} for {name}")
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        nbytes = end - start

        layer = _layer_index(name)
        if layer is None:
            resident.append(name)
            continue

        max_layer = max(max_layer, layer)
        key = f"layer.{layer}"
        chunk = chunks.setdefault(key, ChunkSpec(key=key))
        offset = _align(chunk.total_bytes)
        chunk.tensors.append(
            TensorSpec(
                name=name, dtype=dtype, shape=shape, offset=offset, nbytes=nbytes
            )
        )
        chunk.total_bytes = offset + nbytes

    # Pad each chunk so its length is aligned too, keeping slot arithmetic in
    # the pinned cache uniform.
    for chunk in chunks.values():
        chunk.total_bytes = _align(chunk.total_bytes)

    layer_keys = [f"layer.{i}" for i in range(max_layer + 1) if f"layer.{i}" in chunks]

    return ModelManifest(
        chunks=chunks,
        layer_keys=layer_keys,
        resident_tensors=resident,
        tensor_to_file=tensor_to_file,
        num_layers=len(layer_keys),
    )


class ManifestChunkSource:
    """Reads chunks from safetensors shards, packed per the manifest.

    Packing costs one CPU copy per chunk. That is paid once when the pinned
    host tier adopts the chunk, after which transfers are direct DMA out of
    the packed, page-locked copy -- so the two mechanisms compose: packing
    makes the transfer contiguous, pinning makes it copy-free.
    """

    def __init__(self, manifest: ModelManifest):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.manifest = manifest
        self._handles: dict[str, object] = {}

    def keys(self) -> list[str]:
        return list(self.manifest.chunks.keys())

    def nbytes(self, key: str) -> int:
        spec = self.manifest.chunks.get(key)
        if spec is None:
            raise KeyError(f"unknown chunk: {key}")
        return spec.total_bytes

    def _get_tensor(self, name: str) -> torch.Tensor:
        path = self.manifest.tensor_to_file[name]
        # safe_open mmaps the shard, so repeated opens are cheap and the
        # kernel page cache -- not this code -- decides what stays in RAM.
        with self._safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def host_tensor(self, key: str) -> torch.Tensor:
        spec = self.manifest.chunks.get(key)
        if spec is None:
            raise KeyError(f"unknown chunk: {key}")
        flat = torch.empty(spec.total_bytes, dtype=torch.uint8)
        for tensor_spec in spec.tensors:
            source = self._get_tensor(tensor_spec.name)
            window = flat[
                tensor_spec.offset : tensor_spec.offset + tensor_spec.nbytes
            ]
            window.copy_(
                source.contiguous().view(torch.uint8).reshape(-1)
            )
        return flat

    def resident_tensors(self) -> dict[str, torch.Tensor]:
        return {name: self._get_tensor(name) for name in self.manifest.resident_tensors}
