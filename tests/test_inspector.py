"""Tests for topology inspection and the roofline calculation.

The roofline is the number that decides whether a model is worth downloading,
so its inputs -- bytes per parameter and active parameters per token -- need
to be right for the config shapes real checkpoints use.
"""

import pytest

from corestream import hardware
from corestream.hardware import HardwareProfile
from corestream.inspector import ModelKind, from_config

GIB = 1024**3


def dense_config(**overrides):
    config = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "torch_dtype": "bfloat16",
    }
    config.update(overrides)
    return config


def moe_config(**overrides):
    config = dense_config(
        num_local_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
    )
    config.update(overrides)
    return config


def test_dense_is_classified_dense():
    topo = from_config(dense_config())
    assert topo.kind is ModelKind.DENSE


def test_dense_activates_everything():
    """The defining property of a dense model under streaming."""
    topo = from_config(dense_config())
    assert topo.activation_ratio == pytest.approx(1.0)
    assert topo.bytes_per_token == topo.total_bytes


def test_moe_is_classified_moe():
    assert from_config(moe_config()).kind is ModelKind.MOE


def test_moe_activates_a_fraction():
    """This ratio is the entire reason MoE is the interesting case here."""
    topo = from_config(moe_config())
    assert topo.activation_ratio < 0.5
    assert topo.bytes_per_token < topo.total_bytes


def test_quantization_reduces_bytes_per_param():
    bf16 = from_config(dense_config())
    q4 = from_config(
        dense_config(quantization_config={"bits": 4, "quant_method": "awq"})
    )
    assert q4.total_bytes < bf16.total_bytes / 3


def test_fp8_is_one_byte_per_param():
    topo = from_config(dense_config(quantization_config={"quant_method": "fp8"}))
    assert topo.bytes_per_param == pytest.approx(1.0)


def test_expert_count_key_variants():
    """Qwen, Mixtral, and DeepSeek spell the same field differently."""
    for key in ("num_local_experts", "num_experts", "n_routed_experts"):
        topo = from_config(
            dense_config(**{key: 32, "num_experts_per_tok": 4, "moe_intermediate_size": 768})
        )
        assert topo.kind is ModelKind.MOE, key
        assert topo.num_experts == 32, key


def test_nested_text_config_is_unwrapped():
    """Multimodal checkpoints hide the language model's dimensions inside."""
    topo = from_config({"model_type": "wrapper", "text_config": dense_config()})
    assert topo.num_layers == 32
    assert topo.hidden_size == 4096


def fake_profile(pcie_gib=20.0, disk_gib=3.5, ram_gib=30.0):
    return HardwareProfile(
        has_cuda=True,
        gpu_name="test",
        vram_total_bytes=6 * GIB,
        vram_free_bytes=6 * GIB,
        ram_total_bytes=int(ram_gib * GIB),
        ram_available_bytes=int(ram_gib * GIB),
        disk_free_bytes=500 * GIB,
        pcie_pinned_bytes_per_sec=pcie_gib * GIB,
        disk_read_bytes_per_sec=disk_gib * GIB,
    )


def test_roofline_is_pcie_bound_when_model_fits_ram():
    prof = fake_profile()
    rl = hardware.roofline(
        prof, bytes_per_token=2 * GIB, model_total_bytes=10 * GIB
    )
    assert rl.bound_by == "pcie"
    assert rl.max_tokens_per_sec == pytest.approx(10.0, rel=0.01)


def test_roofline_is_disk_bound_when_model_exceeds_ram():
    """Exceeding RAM is a regime change, not a gradual slowdown."""
    prof = fake_profile(ram_gib=8.0)
    rl = hardware.roofline(
        prof, bytes_per_token=2 * GIB, model_total_bytes=40 * GIB
    )
    assert rl.bound_by == "disk"
    assert rl.max_tokens_per_sec < 2.0


def test_cached_bytes_raise_the_ceiling():
    """Caching works by removing bytes from the transfer, not by going faster."""
    prof = fake_profile()
    uncached = hardware.roofline(prof, 2 * GIB, 10 * GIB, hot_cache_bytes=0)
    cached = hardware.roofline(prof, 2 * GIB, 10 * GIB, hot_cache_bytes=GIB)
    assert cached.max_tokens_per_sec == pytest.approx(
        2 * uncached.max_tokens_per_sec, rel=0.01
    )


def test_moe_ceiling_beats_dense_at_equal_size():
    """The finding that reshaped the design, asserted as a test."""
    prof = fake_profile()
    total = 16 * GIB
    dense = hardware.roofline(prof, bytes_per_token=total, model_total_bytes=total)
    moe = hardware.roofline(
        prof, bytes_per_token=int(total * 0.1), model_total_bytes=total
    )
    assert moe.max_tokens_per_sec > 5 * dense.max_tokens_per_sec


def test_kv_reserve_is_carved_out_of_vram():
    """Weights may not claim the VRAM the KV cache will need."""
    prof = fake_profile()
    assert prof.hot_budget_bytes() < prof.vram_free_bytes
