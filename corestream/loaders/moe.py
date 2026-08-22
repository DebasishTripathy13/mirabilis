"""MoE execution plan: shared weights plus whichever experts the router picks.

Two properties make MoE the case worth optimising for streaming inference:

*   **Only a fraction of the weights move per token.** A 30B model activating
    3B parameters transfers roughly a tenth of what a dense 30B would, which
    is a tenfold difference in the bandwidth-bound ceiling.

*   **Expert usage is skewed.** Routing is not uniform -- some experts are
    selected far more often than others, and consecutive tokens tend to reuse
    experts. Skew is what makes a VRAM cache pay: caching the head of the
    distribution removes most transfers, unlike the dense case where every
    chunk is equally warm.

Prediction here is modelled on cellular handover pre-staging. A network does
not wait for a handset to lose signal before preparing the next cell; it
watches the trend and stages the session ahead of the event. Likewise, the
router's decision for layer N is unknowable until layer N-1 has produced its
output, so instead of waiting we stage the experts recent history says are
most likely, and treat a wrong guess as a cheap wasted transfer.
"""

from __future__ import annotations

from collections import Counter, deque


class MoEPlan:
    """Shared-weight plus routed-expert execution with predictive prefetch.

    `chunks_for_step` returns only what is certain -- the shared weights for
    the layer. Experts become known when `commit_experts` is called with the
    router's actual selection, after which `observe` folds them into the
    history used for future predictions.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        experts_per_token: int,
        history_window: int = 64,
        shared_prefix: str = "shared",
        expert_prefix: str = "expert",
    ):
        self._num_layers = num_layers
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.shared_prefix = shared_prefix
        self.expert_prefix = expert_prefix

        # Per-layer routing history. A bounded deque keeps predictions
        # responsive to the current context: expert preference shifts with
        # subject matter, so a lifetime histogram would track a distribution
        # the model has already moved on from.
        self._history: list[deque[int]] = [
            deque(maxlen=history_window) for _ in range(num_layers)
        ]
        self._counts: list[Counter] = [Counter() for _ in range(num_layers)]

    @property
    def num_steps(self) -> int:
        return self._num_layers

    def shared_key(self, layer: int) -> str:
        return f"{self.shared_prefix}.{layer}"

    def expert_key(self, layer: int, expert: int) -> str:
        return f"{self.expert_prefix}.{layer}.{expert}"

    def chunks_for_step(self, step: int) -> list[str]:
        """Only the shared weights are certain before the router runs."""
        return [self.shared_key(step % self._num_layers)]

    def commit_experts(self, layer: int, experts: list[int]) -> list[str]:
        """Chunk keys for the experts the router actually selected."""
        return [self.expert_key(layer, e) for e in experts]

    def predict_experts(self, layer: int, k: int) -> list[int]:
        """The `k` experts most likely to be selected at this layer.

        Falls back to low-numbered experts before any history exists, which
        is arbitrary but harmless: early predictions are wrong either way,
        and the histogram converges within a few tokens.
        """
        layer %= self._num_layers
        counts = self._counts[layer]
        if not counts:
            return list(range(min(k, self.num_experts)))
        return [expert for expert, _ in counts.most_common(k)]

    def lookahead(self, step: int, depth: int) -> list[str]:
        """Stage upcoming shared weights, then likely experts.

        Shared weights come first because they are certain to be needed --
        spending bandwidth on a guaranteed hit before a probabilistic one is
        strictly better when the bus is the constraint.
        """
        keys: list[str] = []
        for i in range(1, depth + 1):
            layer = (step + i) % self._num_layers
            keys.append(self.shared_key(layer))

        for i in range(1, depth + 1):
            layer = (step + i) % self._num_layers
            # Predicting exactly top-k would leave no margin for the router
            # disagreeing with history; one extra covers the common near-miss
            # without materially increasing wasted bandwidth.
            for expert in self.predict_experts(layer, self.experts_per_token + 1):
                keys.append(self.expert_key(layer, expert))
        return keys

    def observe(self, step: int, chunks: list[str]) -> None:
        """Fold an actual routing decision into the history."""
        layer = step % self._num_layers
        prefix = f"{self.expert_prefix}.{layer}."
        for key in chunks:
            if not key.startswith(prefix):
                continue
            try:
                expert = int(key[len(prefix) :])
            except ValueError:
                continue
            history = self._history[layer]
            if len(history) == history.maxlen:
                # Decay the entry leaving the window so the histogram tracks
                # the window rather than accumulating without bound.
                self._counts[layer][history[0]] -= 1
                if self._counts[layer][history[0]] <= 0:
                    del self._counts[layer][history[0]]
            history.append(expert)
            self._counts[layer][expert] += 1

    def prediction_accuracy(self, layer: int, actual: list[int]) -> float:
        """Share of actually-selected experts that prediction anticipated.

        The metric that decides whether predictive prefetch earns its
        bandwidth: below roughly 1/num_experts it is no better than guessing.
        """
        if not actual:
            return 0.0
        predicted = set(self.predict_experts(layer, self.experts_per_token + 1))
        return len(predicted & set(actual)) / len(actual)
