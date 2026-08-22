"""Dense execution plan: every layer, every token, in fixed order.

Because the access pattern is a known, repeating cycle, prefetching needs no
prediction at all -- this is the ideal case for readahead, exactly as
sequential file access is the ideal case for OS page-cache readahead.

It is also the case where caching cannot help. Every layer is touched exactly
once per token, so no layer is hotter than any other, and a VRAM cache
holding a fraction of the model saves that same fraction of transfers no
matter which layers it holds. Throughput is therefore pinned near

    bandwidth / (model_bytes - cached_bytes)

which is the arithmetic that makes large dense models slow to stream
regardless of engine quality.
"""

from __future__ import annotations


class DensePlan:
    """Sequential layer-by-layer execution."""

    def __init__(self, num_layers: int, prefix: str = "layer"):
        self._num_layers = num_layers
        self.prefix = prefix

    @property
    def num_steps(self) -> int:
        return self._num_layers

    def key(self, layer: int) -> str:
        return f"{self.prefix}.{layer}"

    def chunks_for_step(self, step: int) -> list[str]:
        return [self.key(step % self._num_layers)]

    def lookahead(self, step: int, depth: int) -> list[str]:
        # Wraps past the final layer: the next token restarts at layer 0, so
        # the cycle continues rather than ending.
        return [self.key((step + i) % self._num_layers) for i in range(1, depth + 1)]

    def observe(self, step: int, chunks: list[str]) -> None:
        return  # Deterministic; nothing to learn.
