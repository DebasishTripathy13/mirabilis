"""Model topology inspection: what does this model touch per token?

The whole engine turns on one number -- bytes touched per generated token --
because streaming inference is bandwidth-bound. For a dense model that number
is the entire weight set. For a mixture-of-experts model it is the shared
parameters plus only the experts the router actually selects, which is
typically a tenth of the total.

That is why the two model families land in completely different performance
regimes on the same hardware, and why this module computes the figure from
`config.json` before anything is downloaded.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from enum import Enum


class ModelKind(Enum):
    DENSE = "dense"
    MOE = "moe"


# Keys different model families use for the same concept. Qwen, Mixtral,
# DeepSeek, and OLMoE each picked their own spelling.
_EXPERT_COUNT_KEYS = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "moe_num_experts",
)
_EXPERTS_PER_TOK_KEYS = (
    "num_experts_per_tok",
    "n_activated_experts",
    "moe_top_k",
    "top_k",
)
_MOE_INTERMEDIATE_KEYS = (
    "moe_intermediate_size",
    "expert_intermediate_size",
    "intermediate_size",
)


def _first(config: dict, keys: tuple[str, ...], default=None):
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


@dataclass
class ModelTopology:
    """Everything needed to plan streaming for a model."""

    kind: ModelKind
    name: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    bytes_per_param: float
    # MoE only; 0 for dense models.
    num_experts: int
    experts_per_token: int
    expert_params: int  # per expert, per layer
    # Derived.
    total_params: int
    active_params_per_token: int

    @property
    def total_bytes(self) -> int:
        return int(self.total_params * self.bytes_per_param)

    @property
    def bytes_per_token(self) -> int:
        """Weight bytes that must be reachable to produce one token."""
        return int(self.active_params_per_token * self.bytes_per_param)

    @property
    def activation_ratio(self) -> float:
        """Fraction of the model touched per token. 1.0 for dense models.

        The reciprocal of this is roughly the speedup an MoE model enjoys
        over a dense model of the same total size under streaming.
        """
        if not self.total_params:
            return 1.0
        return self.active_params_per_token / self.total_params

    def summary(self) -> str:
        lines = [
            f"model:            {self.name}",
            f"kind:             {self.kind.value}",
            f"layers:           {self.num_layers}",
            f"total params:     {self.total_params / 1e9:.1f} B",
        ]
        if self.kind is ModelKind.MOE:
            lines += [
                f"experts:          {self.num_experts} per layer, "
                f"top-{self.experts_per_token} active",
                f"active params:    {self.active_params_per_token / 1e9:.1f} B "
                f"({self.activation_ratio:.1%} of total)",
            ]
        return "\n".join(lines)


def _bytes_per_param(config: dict) -> float:
    """Infer the on-disk size of one parameter.

    Quantization config is checked first because it is what actually
    determines transfer volume; `torch_dtype` describes the compute dtype and
    frequently disagrees with how the weights are stored.
    """
    quant = config.get("quantization_config") or {}
    bits = quant.get("bits") or quant.get("w_bit") or quant.get("weight_bits")
    if bits:
        # Group-quantized formats carry scales and zero-points alongside the
        # packed weights; ~12% covers the common group sizes of 64-128.
        return (float(bits) / 8.0) * 1.12
    if quant.get("quant_method") in {"fp8", "fbgemm_fp8"}:
        return 1.0
    if quant.get("load_in_4bit"):
        return 0.5 * 1.12
    if quant.get("load_in_8bit"):
        return 1.0

    dtype = str(config.get("torch_dtype") or config.get("dtype") or "bfloat16")
    if "float32" in dtype:
        return 4.0
    if "float8" in dtype or "fp8" in dtype:
        return 1.0
    return 2.0  # bfloat16 / float16


def _attention_params(config: dict, hidden: int) -> int:
    """Parameter count for one layer's attention block, accounting for GQA."""
    heads = config.get("num_attention_heads") or max(1, hidden // 128)
    kv_heads = config.get("num_key_value_heads") or heads
    head_dim = config.get("head_dim") or (hidden // heads if heads else 128)

    q = hidden * heads * head_dim
    k = hidden * kv_heads * head_dim
    v = hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    return q + k + v + o


def from_config(config: dict, name: str = "unknown") -> ModelTopology:
    """Derive topology from a Hugging Face `config.json` dict.

    Nested `text_config` is unwrapped first: multimodal checkpoints put the
    language model's real dimensions there, and the outer object describes
    the wrapper.
    """
    if "text_config" in config and isinstance(config["text_config"], dict):
        merged = {**config, **config["text_config"]}
        config = merged

    hidden = config.get("hidden_size") or config.get("d_model") or 4096
    layers = (
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or 32
    )
    vocab = config.get("vocab_size") or 32000
    intermediate = config.get("intermediate_size") or hidden * 4
    bpp = _bytes_per_param(config)

    num_experts = int(_first(config, _EXPERT_COUNT_KEYS, 0) or 0)
    is_moe = num_experts > 1

    attn = _attention_params(config, hidden)
    embeddings = vocab * hidden
    if not config.get("tie_word_embeddings", False):
        embeddings *= 2

    if is_moe:
        top_k = int(_first(config, _EXPERTS_PER_TOK_KEYS, 2) or 2)
        moe_inter = int(_first(config, _MOE_INTERMEDIATE_KEYS, intermediate))
        # Gate, up, and down projections per expert (SwiGLU-style FFN).
        expert_params = 3 * hidden * moe_inter
        shared = int(config.get("n_shared_experts") or 0) * expert_params

        per_layer_experts = num_experts * expert_params
        per_layer_dense = attn + shared + (hidden * num_experts)  # + router

        total = embeddings + layers * (per_layer_dense + per_layer_experts)
        active = embeddings + layers * (per_layer_dense + top_k * expert_params)
        kind = ModelKind.MOE
    else:
        top_k = 0
        expert_params = 0
        ffn = 3 * hidden * intermediate
        total = embeddings + layers * (attn + ffn)
        active = total  # A dense model touches every weight, every token.
        kind = ModelKind.DENSE

    return ModelTopology(
        kind=kind,
        name=name,
        num_layers=int(layers),
        hidden_size=int(hidden),
        vocab_size=int(vocab),
        bytes_per_param=bpp,
        num_experts=num_experts,
        experts_per_token=top_k,
        expert_params=expert_params,
        total_params=int(total),
        active_params_per_token=int(active),
    )


def from_path(path: str) -> ModelTopology:
    """Load topology from a local directory containing `config.json`."""
    config_path = path if path.endswith(".json") else os.path.join(path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    name = os.path.basename(os.path.dirname(config_path) or path)
    return from_config(config, name=name or "unknown")


def from_hub(model_id: str, token: str | None = None) -> ModelTopology:
    """Fetch just `config.json` from the Hub -- a few kilobytes, not gigabytes.

    This is what makes `corestream doctor <model-id>` able to tell you whether
    a model is worth downloading before you spend an hour on it.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, "config.json", token=token)
    with open(path) as f:
        config = json.load(f)
    return from_config(config, name=model_id)


def estimate_kv_cache_bytes(
    topology: ModelTopology, context_length: int, config: dict | None = None
) -> int:
    """Bytes of KV cache for a given context, at fp16 per element.

    Reserved out of the VRAM budget before any weights are cached, because a
    KV cache that grows into space the weight cache already claimed is an OOM
    partway through generation rather than at load time.
    """
    config = config or {}
    heads = config.get("num_attention_heads") or max(1, topology.hidden_size // 128)
    kv_heads = config.get("num_key_value_heads") or heads
    head_dim = config.get("head_dim") or topology.hidden_size // max(1, heads)
    per_token = 2 * topology.num_layers * kv_heads * head_dim * 2  # K and V, fp16
    return per_token * context_length


def suggest_context_length(available_bytes: int, topology: ModelTopology) -> int:
    """Largest power-of-two context that fits the KV reserve."""
    per_token = max(1, estimate_kv_cache_bytes(topology, 1))
    raw = available_bytes // per_token
    if raw < 512:
        return 0
    return 2 ** int(math.log2(raw))
