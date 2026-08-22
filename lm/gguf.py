"""Minimal GGUF header reader.

Only the metadata is read -- a few kilobytes at the front of the file -- so
inspecting a 30 GiB model costs nothing. What is needed downstream is the
layer count and whether the model is a mixture of experts, because those two
facts decide how the tensors should be placed across GPU and RAM.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# GGUF metadata value type tags.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4), _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}


class _Reader:
    def __init__(self, handle):
        self.f = handle

    def raw(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF header")
        return data

    def scalar(self, fmt: str, size: int):
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        length = self.scalar("<Q", 8)
        return self.raw(length).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _FIXED:
            fmt, size = _FIXED[vtype]
            return self.scalar(fmt, size)
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            item_type = self.scalar("<I", 4)
            count = self.scalar("<Q", 8)
            # Long arrays (token lists) are skipped rather than materialised:
            # nothing here needs them and a vocabulary can be hundreds of
            # thousands of entries.
            if count > 4096:
                for _ in range(count):
                    self.value(item_type)
                return None
            return [self.value(item_type) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")


@dataclass
class GGUFInfo:
    architecture: str
    layers: int
    experts: int
    experts_used: int
    context_length: int
    embedding_length: int
    name: str

    @property
    def is_moe(self) -> bool:
        return self.experts > 1

    def describe(self) -> str:
        kind = (
            f"MoE, {self.experts} experts, top-{self.experts_used}"
            if self.is_moe else "dense"
        )
        return f"{self.architecture} ({kind}), {self.layers} layers"


def read_info(path: str) -> GGUFInfo:
    """Read architecture metadata from a GGUF file's header."""
    with open(path, "rb") as handle:
        reader = _Reader(handle)
        if reader.raw(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        reader.scalar("<I", 4)              # format version
        reader.scalar("<Q", 8)              # tensor count
        kv_count = reader.scalar("<Q", 8)

        meta: dict[str, object] = {}
        for _ in range(kv_count):
            key = reader.string()
            vtype = reader.scalar("<I", 4)
            try:
                meta[key] = reader.value(vtype)
            except ValueError:
                break

    arch = str(meta.get("general.architecture", "unknown"))

    def get(suffix: str, default: int = 0) -> int:
        value = meta.get(f"{arch}.{suffix}")
        return int(value) if isinstance(value, (int, float)) else default

    return GGUFInfo(
        architecture=arch,
        layers=get("block_count"),
        experts=get("expert_count"),
        experts_used=get("expert_used_count"),
        context_length=get("context_length", 4096),
        embedding_length=get("embedding_length"),
        name=str(meta.get("general.name", "")) or "",
    )
