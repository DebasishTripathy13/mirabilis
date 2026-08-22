"""The streaming engine: drives a plan through the store and scheduler.

Per step the loop is:

1.  Ask the plan what this step needs, and fetch it (usually a cache hit or
    an already-completed prefetch).
2.  Immediately hint the scheduler about upcoming chunks, so their transfers
    begin *before* this step's compute rather than after it.
3.  Run compute.
4.  Tell the plan what was actually used, so its predictions improve.

Step 2 preceding step 3 is the whole point: it is what converts
`load -> compute -> load -> compute` into overlapping streams.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from .loaders.base import ExecutionPlan
from .loaders.moe import MoEPlan
from .scheduler import PrefetchScheduler
from .store import (
    AdmissionPolicy,
    EvictionPolicy,
    LFUAdmission,
    LRUEviction,
    StaticPinning,
    TieredWeightStore,
)


@dataclass
class RunReport:
    """Outcome of a benchmark run, measured against the bandwidth ceiling."""

    tokens: int
    wall_seconds: float
    bytes_moved: int
    bytes_served_from_cache: int
    stall_free_rate: float  # requests that found the chunk already on device
    admissions: int
    evictions: int
    prefetched: int
    prefetch_wasted: int  # bytes moved beyond what the model demanded
    roofline_tokens_per_sec: float
    measured_bandwidth_bytes_per_sec: float

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def savings_rate(self) -> float:
        """Share of demanded bytes the cache kept off the bus."""
        demanded = self.bytes_moved + self.bytes_served_from_cache
        return self.bytes_served_from_cache / demanded if demanded else 0.0

    @property
    def roofline_utilization(self) -> float:
        """Share of the theoretical ceiling actually achieved.

        The honest headline metric. "Faster than the previous engine" can be
        satisfied by a rounding error; this says how much of the hardware's
        available bandwidth the design converted into tokens.
        """
        if not self.roofline_tokens_per_sec:
            return 0.0
        return self.tokens_per_sec / self.roofline_tokens_per_sec

    def summary(self) -> str:
        gib = 1024**3
        lines = [
            f"tokens:               {self.tokens}",
            f"wall time:            {self.wall_seconds:.2f} s",
            f"throughput:           {self.tokens_per_sec:.2f} tok/s",
            f"roofline:             {self.roofline_tokens_per_sec:.2f} tok/s",
            f"roofline utilization: {self.roofline_utilization:.1%}",
            f"bytes moved:          {self.bytes_moved / gib:.2f} GiB",
            f"served from cache:    {self.bytes_served_from_cache / gib:.2f} GiB",
            f"prefetch coverage:    {self.stall_free_rate:.1%}",
            f"bandwidth saved:      {self.savings_rate:.1%}",
            f"admissions/evictions: {self.admissions} / {self.evictions}",
            f"achieved bandwidth:   {self.measured_bandwidth_bytes_per_sec / gib:.1f} GiB/s",
        ]
        demanded = self.bytes_moved + self.bytes_served_from_cache
        if demanded > 0 and self.prefetch_wasted:
            lines.append(
                f"wasted traffic:       {self.prefetch_wasted / gib:.2f} GiB "
                f"({self.prefetch_wasted / demanded:.1%} over demand)"
            )
        return "\n".join(lines)


@dataclass
class EngineConfig:
    hot_budget_bytes: int
    prefetch_depth: int = 2
    workers: int = 2
    device: str | None = None
    # Peak host->device bandwidth for this machine, measured independently by
    # `hardware.profile`. The roofline must be computed against a fixed
    # hardware capability; deriving it from the run's own throughput would be
    # circular and would report ~100% utilization no matter how much traffic
    # the engine wasted.
    reference_bandwidth_bytes_per_sec: float = 0.0


class StreamingEngine:
    """Executes a plan while streaming its weights through the tier stack."""

    def __init__(
        self,
        source,
        plan: ExecutionPlan,
        config: EngineConfig,
        compute_fn: Callable[[int, list[torch.Tensor]], None] | None = None,
        admission: AdmissionPolicy | None = None,
        eviction: EvictionPolicy | None = None,
    ):
        self.plan = plan
        self.config = config

        # The plan's access pattern dictates the right policies, so they are
        # chosen from it rather than left to the caller to get right.
        is_moe = isinstance(plan, MoEPlan)
        if admission is None:
            # Dense access is a fixed cycle with no cold tail, so delaying
            # admission only wastes the first pass. MoE has a long tail of
            # rarely-routed experts that must not evict the hot ones.
            admission = LFUAdmission(threshold=2 if is_moe else 1)
        if eviction is None:
            # Recency predicts reuse only when access is skewed. Under the
            # cyclic access of a dense model it predicts the opposite, so
            # a fixed resident subset beats LRU outright.
            eviction = LRUEviction() if is_moe else StaticPinning()

        self.store = TieredWeightStore(
            source=source,
            hot_budget_bytes=config.hot_budget_bytes,
            device=config.device,
            admission=admission,
            eviction=eviction,
        )
        self.scheduler = PrefetchScheduler(self.store, workers=config.workers)
        self.compute_fn = compute_fn or (lambda step, tensors: None)
        self._used_keys: set[str] = set()

    def step(self, step_index: int, extra_keys: list[str] | None = None) -> None:
        """Execute one step, prefetching the next ones first."""
        keys = self.plan.chunks_for_step(step_index)
        if extra_keys:
            keys = keys + extra_keys

        # Issue lookahead before fetching, so background transfers are
        # already moving while this step blocks on its own data.
        self.scheduler.hint(self.plan.lookahead(step_index, self.config.prefetch_depth))

        tensors = [self.store.get(k) for k in keys]
        self._used_keys.update(keys)
        self.compute_fn(step_index, tensors)
        self.plan.observe(step_index, keys)

    def run(
        self,
        tokens: int,
        router: Callable[[int], list[int]] | None = None,
    ) -> RunReport:
        """Generate `tokens` tokens and report against the roofline.

        `router` supplies expert selections for MoE plans, taking the layer
        index and returning selected expert ids.
        """
        is_moe = isinstance(self.plan, MoEPlan)
        if is_moe and router is None:
            raise ValueError("an MoE plan requires a router callable")

        before_promoted = self.store.stats.bytes_promoted
        before_demanded = self.store.stats.bytes_demanded
        before_prefetched = self.scheduler.stats.chunks_prefetched

        if self.store.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(tokens):
            for layer in range(self.plan.num_steps):
                extra = None
                if is_moe:
                    extra = self.plan.commit_experts(layer, router(layer))
                self.step(layer, extra_keys=extra)

        if self.store.device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - start

        moved = self.store.stats.bytes_promoted - before_promoted
        demanded = self.store.stats.bytes_demanded - before_demanded
        prefetched = self.scheduler.stats.chunks_prefetched - before_prefetched

        # Bytes the model demands per token. A property of the model and the
        # plan, not of how well the engine ran -- which is what makes it a
        # valid denominator for the ceiling.
        bytes_per_token = demanded / tokens if tokens else 0
        bandwidth = moved / wall if wall else 0.0

        reference = self.config.reference_bandwidth_bytes_per_sec or bandwidth
        roofline = reference / bytes_per_token if bytes_per_token else 0.0

        return RunReport(
            tokens=tokens,
            wall_seconds=wall,
            bytes_moved=moved,
            bytes_served_from_cache=max(0, demanded - moved),
            stall_free_rate=self.store.stats.stall_free_rate,
            admissions=self.store.stats.admissions,
            evictions=self.store.stats.evictions,
            prefetched=prefetched,
            prefetch_wasted=max(0, moved - demanded),
            roofline_tokens_per_sec=roofline,
            measured_bandwidth_bytes_per_sec=bandwidth,
        )

    def close(self) -> None:
        self.scheduler.shutdown()
        self.store.clear_hot()

    def __enter__(self) -> "StreamingEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def zipf_router(
    num_experts: int, experts_per_token: int, skew: float = 1.1, seed: int = 0
) -> Callable[[int], list[int]]:
    """A router whose expert selection follows a Zipf distribution.

    Real MoE routing is neither uniform nor fixed: a minority of experts
    absorb a large share of tokens. Benchmarking against uniform routing
    would understate what a VRAM cache is worth, while benchmarking against a
    fixed selection would overstate it. Zipf sits between the two and is the
    shape reported for trained routers.

    `skew` near 1.0 is mild; higher concentrates traffic on fewer experts.
    """
    generator = torch.Generator().manual_seed(seed)
    ranks = torch.arange(1, num_experts + 1, dtype=torch.float32)
    weights = 1.0 / ranks.pow(skew)
    weights /= weights.sum()

    def route(layer: int) -> list[int]:
        # Layer-dependent permutation: different layers favour different
        # experts, so a cache cannot simply hold one global top-k.
        offset = layer * 7 % max(1, num_experts)
        picked = torch.multinomial(
            weights, experts_per_token, replacement=False, generator=generator
        )
        return [int((p + offset) % num_experts) for p in picked]

    return route
