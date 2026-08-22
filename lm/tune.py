"""Turn hardware plus model shape into engine flags.

The rules here are the measured findings from `LAPTOP-INFERENCE.md`, not
guesses:

*   Decode is memory-bound, so throughput is bandwidth divided by bytes read
    per token. Placement decides which bandwidth applies to which bytes.
*   For a mixture of experts, place by *tensor role* rather than layer index:
    attention and norms are small and read on every token, so they belong on
    the GPU; expert banks are large and mostly idle per token, so they belong
    in RAM.
*   Threads should be physical cores. On a hybrid CPU, adding the E-cores
    measured 25% *slower* on this workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gguf import GGUFInfo
from .hardware import Hardware


@dataclass
class Plan:
    gpu_layers: int
    cpu_moe_layers: int          # -ncmoe; 0 means the flag is not used
    threads: int
    context: int
    flash_attention: bool = True
    # Use the chat template embedded in the GGUF rather than llama.cpp's guess.
    # It matters for models whose template drives behaviour -- reasoning
    # toggles, tool-call syntax -- where the wrong template makes the model
    # ramble or emit malformed calls instead of failing visibly.
    jinja: bool = True
    # Quantizing the KV cache frees VRAM, which on a tight card buys another
    # expert layer or two on the GPU. The two settings interact, so they have
    # to be searched together rather than tuned one at a time.
    cache_type: str = ""
    # Pinning threads to specific CPUs. Without it the scheduler is free to
    # migrate a memory-bound thread onto a slow core mid-run.
    cpu_mask: str = ""
    projector: str = ""
    notes: list[str] = field(default_factory=list)

    def to_args(self) -> list[str]:
        args = ["-ngl", str(self.gpu_layers), "-t", str(self.threads),
                "-c", str(self.context)]
        if self.cpu_moe_layers > 0:
            args += ["-ncmoe", str(self.cpu_moe_layers)]
        if self.flash_attention:
            args += ["-fa", "on"]
        if self.jinja:
            args += ["--jinja"]
        if self.cache_type:
            args += ["-ctk", self.cache_type, "-ctv", self.cache_type]
        if self.cpu_mask:
            args += ["-C", self.cpu_mask, "--cpu-strict", "1"]
        if self.projector:
            args += ["--mmproj", self.projector]
        return args

    def explain(self) -> str:
        return "\n".join(f"  {n}" for n in self.notes)


# Hybrid architectures interleave linear-attention layers, whose recurrent
# state is a few megabytes, with a minority of full-attention layers that carry
# a real KV cache. Estimating every layer as full attention overstates the
# reservation several times over and starves the weight budget.
_HYBRID_ATTENTION = ("qwen3next", "qwen35", "granitemoehybrid", "falcon_h1",
                     "jamba", "bamba", "nemotronh", "plamo2", "lfm2",
                     "mamba", "rwkv", "recurrentgemma")

# No architecture should reserve so much of the card for context that no
# weights fit. Getting this wrong is silent and expensive: a 65-layer model
# estimated as all-full-attention reserved 5.1 GiB of a 5.6 GiB card, leaving
# 0.0 GiB for weights, and the planner duly put every layer on the CPU.
_MAX_KV_SHARE_OF_VRAM = 0.5


def _kv_cache_gib(info: GGUFInfo, context: int, vram_gib: float = 0.0) -> float:
    """Rough KV cache size at f16, used to reserve VRAM before placing weights."""
    if not info.layers or not info.embedding_length:
        return 0.5
    layers = info.layers
    if any(tag in info.architecture.lower() for tag in _HYBRID_ATTENTION):
        # Hybrid stacks interleave linear-attention layers whose recurrent
        # state is a few megabytes with a minority carrying a real KV cache.
        layers = max(1, round(layers / 4))
    per_token = 2 * layers * info.embedding_length * 2  # K and V, 2 bytes
    estimate = per_token * context / 1024**3
    if vram_gib > 0:
        return min(estimate, vram_gib * _MAX_KV_SHARE_OF_VRAM)
    return estimate


def plan(hw: Hardware, info: GGUFInfo, file_gib: float,
         context: int | None = None, override_ncmoe: int | None = None,
         override_threads: int | None = None,
         override_cache_type: str | None = None,
         override_gpu_layers: int | None = None,
         override_cpu_mask: str | None = None) -> Plan:
    """Choose placement for this model on this machine.

    `override_ncmoe` comes from `lm tune`, which measures candidate placements
    rather than predicting them. A measurement always beats this model, which
    has to guess at things the file does not state -- how much of the file is
    experts, and how many layers carry a real KV cache.
    """
    # 4096 is the default rather than the model's advertised maximum: context
    # competes with weights for the same VRAM, and a long context that pushes
    # expert layers off the GPU costs more than it gains.
    context = context or 4096
    notes: list[str] = []
    threads = override_threads or hw.threads
    mask = override_cpu_mask if override_cpu_mask is not None else hw.affinity_mask(threads)
    if override_threads:
        notes.append(f"{threads} threads (measured best by `lm tune`).")
    elif hw.fast_core_ids:
        notes.append(
            f"{threads} threads: one per physical fast core "
            f"({len(hw.fast_core_ids)}) plus "
            f"{max(0, threads - len(hw.fast_core_ids))} slower core(s). "
            "Hyperthread siblings share the load/store units decode waits on, "
            "so they add contention rather than bandwidth."
        )
    else:
        notes.append(f"{threads} threads (physical cores).")

    if mask:
        notes.append(f"Threads pinned to CPUs (mask {mask}) so the scheduler "
                     "cannot migrate them onto slow cores.")

    if not hw.has_gpu:
        notes.append("No GPU detected; running entirely on CPU.")
        return Plan(0, 0, threads, context, flash_attention=False,
                    cpu_mask=mask, notes=notes)

    kv = _kv_cache_gib(info, context, hw.usable_vram_gib)
    weight_budget = max(0.0, hw.usable_vram_gib - kv)
    notes.append(
        f"{hw.vram_free_gib:.1f} GiB VRAM free, ~{kv:.1f} GiB reserved for KV "
        f"cache at {context} context, leaving {weight_budget:.1f} GiB for weights."
    )

    if file_gib <= weight_budget:
        notes.append("Whole model fits in VRAM; everything on GPU.")
        return Plan(999, 0, threads, context, cpu_mask=mask, notes=notes)

    if not info.is_moe:
        # Dense: every layer is read per token, so the only choice is how many
        # live on the fast tier. Layers are near enough equal in size.
        per_layer = file_gib / max(1, info.layers)
        fit = int(weight_budget / per_layer) if per_layer else 0
        fit = max(0, min(info.layers, fit))
        notes.append(
            f"Dense model: {fit} of {info.layers} layers on GPU "
            f"(~{per_layer:.2f} GiB each), remainder computed on CPU from RAM."
        )
        if file_gib > hw.usable_ram_gib + weight_budget:
            notes.append(
                "WARNING: model exceeds RAM+VRAM, so part of every token comes "
                "from disk at roughly a tenth of RAM speed. A smaller "
                "quantization will be far faster."
            )
        if override_gpu_layers is not None:
            fit = max(0, min(info.layers, override_gpu_layers))
            notes.append(f"Using measured placement from `lm tune`: {fit} layers "
                         "on GPU, which beat llama.cpp's own conservative fit.")
        return Plan(fit, 0, threads, context, cpu_mask=mask, notes=notes)

    # MoE: attention and norms are small and read every token -- always GPU.
    # Expert banks dominate size; place as many layers' experts on the GPU as
    # the remaining budget allows and leave the rest in RAM.
    expert_fraction = 0.9          # experts are the overwhelming majority
    expert_gib = file_gib * expert_fraction
    non_expert_gib = file_gib - expert_gib
    per_layer_experts = expert_gib / max(1, info.layers)

    room = max(0.0, weight_budget - non_expert_gib)
    layers_on_gpu = int(room / per_layer_experts) if per_layer_experts else 0
    layers_on_gpu = max(0, min(info.layers, layers_on_gpu))
    cpu_moe = info.layers - layers_on_gpu

    if override_ncmoe is not None:
        cpu_moe = max(0, min(info.layers, override_ncmoe))
        layers_on_gpu = info.layers - cpu_moe
        notes.append(f"Using measured placement from `lm tune`.")

    notes.append(
        f"MoE: attention and norms on GPU (~{non_expert_gib:.1f} GiB); experts "
        f"for {cpu_moe} of {info.layers} layers stay in RAM, "
        f"{layers_on_gpu} on GPU."
    )
    active_note = (
        f"Only ~{info.experts_used}/{info.experts} experts run per token, so "
        "the model stores like its full size but reads like a fraction of it."
    )
    notes.append(active_note)

    # The single most valuable thing a user can act on, and the least
    # visible: if the working set does not fit in *available* RAM, the
    # shortfall is re-read from disk on every token at roughly a twelfth of
    # RAM speed. Freeing a few gigabytes by closing a browser measured a
    # larger gain here than any placement flag.
    if file_gib > hw.ram_available_gib:
        short = file_gib - hw.ram_available_gib
        notes.append(
            f"NOTE: {short:.1f} GiB of this model cannot stay cached "
            f"({file_gib:.1f} GiB model, {hw.ram_available_gib:.1f} GiB "
            "available). That share is re-read from disk every token at about "
            "a twelfth of RAM speed. Closing other applications is usually the "
            "biggest single speedup available."
        )

    if expert_gib > hw.usable_ram_gib:
        shortfall = expert_gib - hw.usable_ram_gib
        notes.append(
            f"WARNING: about {shortfall:.0f} GiB of experts will not fit in RAM "
            "and will be read from disk, which is roughly 12x slower. A smaller "
            "quantization that fits RAM will be substantially faster."
        )
    result = Plan(999, cpu_moe, threads, context, cpu_mask=mask, notes=notes)
    if override_cache_type:
        result.cache_type = override_cache_type
        notes.append(f"KV cache quantized to {override_cache_type}, freeing "
                     "VRAM for more expert layers.")
    return result
