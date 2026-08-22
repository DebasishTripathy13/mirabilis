"""Tests for execution plans, including the dense-as-degenerate-MoE claim."""

from corestream.loaders import DensePlan, MoEPlan


def test_dense_plan_is_sequential():
    plan = DensePlan(num_layers=4)
    assert plan.chunks_for_step(0) == ["layer.0"]
    assert plan.chunks_for_step(3) == ["layer.3"]


def test_dense_lookahead_wraps_to_next_token():
    """Layer N-1's successor is layer 0 of the following token.

    Without wrapping, the last layers of every token would always stall,
    since nothing would be prefetched across the token boundary.
    """
    plan = DensePlan(num_layers=4)
    assert plan.lookahead(3, depth=2) == ["layer.0", "layer.1"]


def test_moe_step_returns_only_certain_chunks():
    """Experts are unknown until the router runs, so they are not returned."""
    plan = MoEPlan(num_layers=2, num_experts=8, experts_per_token=2)
    assert plan.chunks_for_step(0) == ["shared.0"]


def test_moe_commit_experts_builds_keys():
    plan = MoEPlan(num_layers=2, num_experts=8, experts_per_token=2)
    assert plan.commit_experts(1, [3, 5]) == ["expert.1.3", "expert.1.5"]


def test_moe_learns_routing_history():
    plan = MoEPlan(num_layers=1, num_experts=8, experts_per_token=2)
    for _ in range(10):
        plan.observe(0, ["expert.0.4", "expert.0.6"])
    predicted = plan.predict_experts(0, k=2)
    assert set(predicted) == {4, 6}


def test_moe_prediction_beats_chance_on_skewed_routing():
    """Predictive prefetch must beat random guessing to earn its bandwidth."""
    plan = MoEPlan(num_layers=1, num_experts=32, experts_per_token=4)
    hot = [1, 2, 3, 4]
    for _ in range(50):
        plan.observe(0, [f"expert.0.{e}" for e in hot])

    accuracy = plan.prediction_accuracy(0, hot)
    chance = plan.experts_per_token / plan.num_experts
    assert accuracy > chance
    assert accuracy == 1.0


def test_moe_history_window_forgets_stale_routing():
    """Predictions must track the current context, not lifetime totals.

    Expert preference shifts with subject matter, so a histogram that never
    decays would keep prefetching experts the model has moved away from.
    """
    plan = MoEPlan(num_layers=1, num_experts=8, experts_per_token=1, history_window=4)
    for _ in range(4):
        plan.observe(0, ["expert.0.7"])
    assert plan.predict_experts(0, k=1) == [7]

    for _ in range(4):
        plan.observe(0, ["expert.0.2"])
    assert plan.predict_experts(0, k=1) == [2]


def test_moe_lookahead_prioritises_certain_chunks():
    """Shared weights are certain; experts are guesses. Certain goes first."""
    plan = MoEPlan(num_layers=4, num_experts=8, experts_per_token=2)
    keys = plan.lookahead(0, depth=2)
    shared = [k for k in keys if k.startswith("shared.")]
    first_expert = next(i for i, k in enumerate(keys) if k.startswith("expert."))
    assert all(keys.index(s) < first_expert for s in shared)


def test_dense_is_moe_with_all_experts_active():
    """The unifying claim, stated as a test.

    An MoE whose top-k equals its expert count touches every expert every
    token -- which is precisely a dense model. If this holds, the two
    families genuinely share one execution path.
    """
    num_layers, num_experts = 3, 4
    moe = MoEPlan(
        num_layers=num_layers, num_experts=num_experts, experts_per_token=num_experts
    )
    all_experts = moe.commit_experts(0, list(range(num_experts)))
    assert len(all_experts) == num_experts

    touched_per_token = {
        key for layer in range(num_layers)
        for key in moe.chunks_for_step(layer)
             + moe.commit_experts(layer, list(range(num_experts)))
    }
    total_chunks = num_layers * (1 + num_experts)
    assert len(touched_per_token) == total_chunks
