"""Execution plans: which chunks does this step actually need?

The engine has one execution path, not a dense path and an MoE path. Each
step asks the plan two questions:

    chunks_for_step(step)   -- what must be resident right now
    lookahead(step, depth)  -- what is probably needed next, for prefetch

A dense model answers the first exactly and the second exactly, because layer
order is fixed and fully known. An MoE model answers the first exactly only
once its router has run, and answers the second probabilistically from recent
routing history.

That difference is the only thing separating the two families here: **a dense
model is an MoE in which every expert is always active**, so it is the
degenerate case of the same interface rather than a separate implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutionPlan(Protocol):
    """Maps execution steps to the chunk keys they require."""

    @property
    def num_steps(self) -> int:
        """Steps in one forward pass -- usually the layer count."""
        ...

    def chunks_for_step(self, step: int) -> list[str]:
        """Chunks that must be resident before step `step` can execute."""
        ...

    def lookahead(self, step: int, depth: int) -> list[str]:
        """Chunks worth staging now, highest confidence first.

        May over-predict: a wrong guess costs a wasted transfer, while a
        missing guess costs a synchronous stall. Under a bandwidth surplus
        the asymmetry favours over-prediction, so implementations should lean
        that way only when the bus is not already saturated.
        """
        ...

    def observe(self, step: int, chunks: list[str]) -> None:
        """Record what a step actually used, to improve future predictions.

        A no-op for deterministic plans.
        """
        ...
