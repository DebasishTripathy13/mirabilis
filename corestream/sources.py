"""Chunk sources: where weights are read from.

`SafetensorsSource` is the production path. It reads through `safetensors`,
which mmaps the file, so the kernel page cache serves as the WARM tier and
performs readahead without any cache code here.

`SyntheticSource` generates weights of realistic shape without a download. It
exists because the property being validated -- how much of the memory
bandwidth ceiling the engine captures -- depends on transfer sizes and access
patterns, not on the numeric content of the weights. It makes the streaming
core measurable before committing to a multi-gigabyte download.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch


class SyntheticSource:
    """Fixed-size chunks of arbitrary data, for benchmarking and tests."""

    def __init__(
        self,
        chunk_keys: Iterable[str],
        chunk_bytes: int,
        dtype: torch.dtype = torch.float16,
        seed: int = 0,
    ):
        self._keys = list(chunk_keys)
        self.dtype = dtype
        element_size = torch.empty(0, dtype=dtype).element_size()
        self._numel = max(1, chunk_bytes // element_size)
        self._nbytes = self._numel * element_size

        # One backing buffer shared by every chunk. Distinct buffers would
        # exhaust host RAM at realistic model sizes, and the transfer cost --
        # the thing under measurement -- does not depend on the contents.
        generator = torch.Generator().manual_seed(seed)
        self._buffer = torch.empty(self._numel, dtype=dtype)
        self._buffer.uniform_(-1, 1, generator=generator)

    def keys(self) -> list[str]:
        return list(self._keys)

    def nbytes(self, key: str) -> int:
        return self._nbytes

    def host_tensor(self, key: str) -> torch.Tensor:
        if key not in self._keys:
            raise KeyError(f"unknown chunk: {key}")
        return self._buffer


class SafetensorsSource:
    """Reads chunks from one or more `.safetensors` shards via mmap.

    A chunk maps to one or more tensor names -- a transformer layer is many
    tensors, an MoE expert is typically three. Multi-tensor chunks are
    returned as a single flat buffer so the whole chunk moves in one
    transfer; issuing separate small copies per tensor would give up most of
    the achievable PCIe bandwidth to per-transfer overhead.
    """

    def __init__(self, paths: str | list[str], chunk_map: dict[str, list[str]]):
        from safetensors import safe_open

        self._safe_open = safe_open
        if isinstance(paths, str):
            paths = (
                [os.path.join(paths, f) for f in sorted(os.listdir(paths))
                 if f.endswith(".safetensors")]
                if os.path.isdir(paths)
                else [paths]
            )
        self.paths = paths
        self.chunk_map = chunk_map

        # Which shard holds which tensor, resolved once so lookups do not
        # reopen every file.
        self._tensor_to_path: dict[str, str] = {}
        for path in self.paths:
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    self._tensor_to_path[name] = path

        self._nbytes_cache: dict[str, int] = {}

    def keys(self) -> list[str]:
        return list(self.chunk_map.keys())

    def _tensors(self, key: str) -> list[torch.Tensor]:
        names = self.chunk_map.get(key)
        if names is None:
            raise KeyError(f"unknown chunk: {key}")
        out = []
        for name in names:
            path = self._tensor_to_path.get(name)
            if path is None:
                raise KeyError(f"tensor {name!r} not present in any shard")
            with self._safe_open(path, framework="pt", device="cpu") as f:
                out.append(f.get_tensor(name))
        return out

    def nbytes(self, key: str) -> int:
        cached = self._nbytes_cache.get(key)
        if cached is None:
            cached = sum(t.numel() * t.element_size() for t in self._tensors(key))
            self._nbytes_cache[key] = cached
        return cached

    def host_tensor(self, key: str) -> torch.Tensor:
        tensors = self._tensors(key)
        if len(tensors) == 1:
            return tensors[0]
        flat = torch.empty(self.nbytes(key), dtype=torch.uint8)
        offset = 0
        for t in tensors:
            size = t.numel() * t.element_size()
            flat[offset : offset + size] = t.contiguous().view(torch.uint8).reshape(-1)
            offset += size
        return flat


def dense_chunk_map(tensor_names: Iterable[str], num_layers: int) -> dict[str, list[str]]:
    """Group tensor names into one chunk per transformer layer.

    Matches the common `...layers.<n>....` convention used by Llama, Qwen,
    Mistral, and their derivatives.
    """
    chunks: dict[str, list[str]] = {f"layer.{i}": [] for i in range(num_layers)}
    for name in tensor_names:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer = int(parts[i + 1])
                if 0 <= layer < num_layers:
                    chunks[f"layer.{layer}"].append(name)
                break
    return {k: v for k, v in chunks.items() if v}


def moe_chunk_map(
    tensor_names: Iterable[str], num_layers: int, num_experts: int
) -> dict[str, list[str]]:
    """Split tensor names into per-layer shared chunks and per-expert chunks.

    Anything inside a layer that is not attributable to a specific expert --
    attention, norms, the router itself -- lands in that layer's shared
    chunk, because it is needed on every token regardless of routing.
    """
    chunks: dict[str, list[str]] = {}
    for name in tensor_names:
        parts = name.split(".")
        layer = None
        expert = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer = int(parts[i + 1])
            if part in {"experts", "expert"} and i + 1 < len(parts) and parts[i + 1].isdigit():
                expert = int(parts[i + 1])
        if layer is None or not (0 <= layer < num_layers):
            continue
        if expert is not None and 0 <= expert < num_experts:
            key = f"expert.{layer}.{expert}"
        else:
            key = f"shared.{layer}"
        chunks.setdefault(key, []).append(name)
    return chunks
