"""Tests for the Hugging Face integration.

The parts that can be tested without a checkpoint -- name resolution,
parameter replacement, layer discovery -- are tested directly. The end-to-end
correctness check needs a real model and is skipped when one is not cached
locally, so the suite stays runnable offline.
"""

import os

import pytest
import torch

from corestream.hf import _assign, _resolve, find_decoder_layers

transformers = pytest.importorskip("transformers")


class Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.register_buffer("scale", torch.ones(2))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Inner()
        self.mlp = Inner()


class Tiny(torch.nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block() for _ in range(depth)])
        self.model.embed = torch.nn.Embedding(4, 2)


def test_resolve_walks_dotted_path():
    model = Tiny()
    module, attr = _resolve(model, "model.layers.1.attn.weight")
    assert attr == "weight"
    assert module is model.model.layers[1].attn


def test_assign_replaces_parameter():
    model = Tiny()
    target = torch.ones(3, 3)
    module, attr = _resolve(model, "model.layers.0.attn.weight")
    _assign(module, attr, target)
    assert torch.equal(module.weight.detach(), target)
    assert isinstance(module.weight, torch.nn.Parameter)
    assert not module.weight.requires_grad


def test_assign_handles_buffers():
    """Buffers are not in `_parameters`, so they need plain attribute assignment."""
    model = Tiny()
    module, attr = _resolve(model, "model.layers.0.attn.scale")
    _assign(module, attr, torch.full((2,), 5.0))
    assert torch.equal(module.scale, torch.full((2,), 5.0))
    assert "scale" not in module._parameters


def test_assign_bridges_meta_to_real():
    """A meta parameter cannot take a real tensor via `.data`.

    `set_data` treats meta and CUDA/CPU tensors as different tensor types and
    rejects the assignment, which is why the parameter object is replaced
    rather than repointed the first time.
    """
    module = Inner()
    module._parameters["weight"] = torch.nn.Parameter(
        torch.empty(2, 2, device="meta")
    )
    with pytest.raises(RuntimeError):
        module.weight.data = torch.ones(2, 2)

    _assign(module, "weight", torch.ones(2, 2))
    assert torch.equal(module.weight.detach(), torch.ones(2, 2))


def test_same_device_swap_is_cheap_after_assign():
    """Once real, parameters can be repointed without allocating.

    This is what keeps the per-layer hooks cheap: the swap in the hot path is
    a same-device `.data` assignment, not a new Parameter per tensor per token.
    """
    module = Inner()
    _assign(module, "weight", torch.zeros(2, 2))
    before = id(module.weight)
    module.weight.data = torch.ones(4, 4)
    assert id(module.weight) == before
    assert module.weight.shape == (4, 4)


def test_find_decoder_layers():
    model = Tiny(depth=5)
    layers = find_decoder_layers(model)
    assert layers is not None
    assert len(layers) == 5


def test_find_decoder_layers_returns_none_without_stack():
    assert find_decoder_layers(torch.nn.Linear(2, 2)) is None


# -- end-to-end ------------------------------------------------------------

CACHED_MODEL = os.environ.get("CORESTREAM_TEST_MODEL")
needs_model = pytest.mark.skipif(
    not CACHED_MODEL or not os.path.isdir(CACHED_MODEL),
    reason="set CORESTREAM_TEST_MODEL to a local checkpoint directory",
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@needs_model
@needs_cuda
def test_streamed_logits_match_reference():
    """The gate that proves the integration is real.

    Streaming rewires how weights reach the model, so the only convincing
    check is that the numbers come out identical to an ordinary forward pass.
    """
    from transformers import AutoModelForCausalLM

    from corestream.hf import StreamingConfig, StreamingModel
    from corestream.manifest import build_manifest

    dtype = build_manifest(CACHED_MODEL).dominant_dtype
    streamer = StreamingModel(
        CACHED_MODEL,
        StreamingConfig(hot_budget_bytes=256 * 1024**2, pinned_budget_bytes=512 * 1024**2),
    )
    ids = streamer.tokenizer("The capital of France is", return_tensors="pt").input_ids
    streamed = streamer.forward_logits(ids).float().cpu()
    streamer.close()
    del streamer
    torch.cuda.empty_cache()

    # The reference must run on the same device. Comparing a CUDA run against
    # a CPU one is not a correctness check: bf16 accumulates in a different
    # order on each, so they differ by ~1e-1 even when both are right.
    reference_model = (
        AutoModelForCausalLM.from_pretrained(CACHED_MODEL, dtype=dtype).cuda().eval()
    )
    with torch.inference_mode():
        reference = reference_model(ids.cuda()).logits.float().cpu()

    assert torch.equal(streamed, reference)


def test_coverage_check_rejects_unbacked_parameters():
    """A parameter with no source of weights must stop the model starting.

    Nothing raises on its own when a parameter is never filled: the layer
    computes on an empty or meta tensor and generation emits fluent-looking
    garbage at full speed. A 4-bit checkpoint whose module layout did not
    match produced "!!!!!!!!" at 20 tok/s while every test passed, which is
    why this is a hard refusal rather than a warning.
    """
    from corestream.hf import StreamingModel

    model = Tiny(depth=1)
    # Mimic a checkpoint that supplies nothing for this parameter.
    model.model.layers[0].attn._parameters["weight"] = torch.nn.Parameter(
        torch.empty(0)
    )

    stub = type("Stub", (), {})()
    stub.model = model
    stub.manifest = type("M", (), {"chunks": {}, "resident_tensors": []})()

    with pytest.raises(RuntimeError, match="never receive weights"):
        StreamingModel._verify_coverage(stub)


def test_coverage_check_flags_meta_parameters():
    """Meta tensors report their full numel, so length alone misses them."""
    from corestream.hf import StreamingModel

    model = Tiny(depth=1)
    model.model.layers[0].attn._parameters["weight"] = torch.nn.Parameter(
        torch.empty(2, 2, device="meta")
    )
    assert model.model.layers[0].attn.weight.numel() == 4  # looks well-formed

    stub = type("Stub", (), {})()
    stub.model = model
    stub.manifest = type("M", (), {"chunks": {}, "resident_tensors": []})()

    with pytest.raises(RuntimeError, match="never receive weights"):
        StreamingModel._verify_coverage(stub)
