"""Command line interface.

`doctor` is the command that matters most in practice. Streaming inference is
bandwidth-bound, so whether a given model is usable on a given machine is
decided by arithmetic that can be done from `config.json` alone -- a few
kilobytes. Finding out after a 30 GB download that the ceiling is 0.2 tok/s is
avoidable, and this is what avoids it.
"""

from __future__ import annotations

import argparse
import sys

from . import hardware, inspector
from .engine import EngineConfig, StreamingEngine, zipf_router
from .hardware import format_bytes
from .inspector import ModelKind
from .loaders import DensePlan, MoEPlan
from .sources import SyntheticSource

GIB = 1024**3


def _print_profile(prof: hardware.HardwareProfile) -> None:
    print("hardware")
    print(f"  gpu:              {prof.gpu_name}")
    if prof.has_cuda:
        print(
            f"  vram:             {format_bytes(prof.vram_free_bytes)} free "
            f"of {format_bytes(prof.vram_total_bytes)}"
        )
    print(
        f"  ram:              {format_bytes(prof.ram_available_bytes)} available "
        f"of {format_bytes(prof.ram_total_bytes)}"
    )
    print(f"  disk free:        {format_bytes(prof.disk_free_bytes)}")
    if prof.pcie_pinned_bytes_per_sec:
        print(f"  pcie (measured):  {prof.pcie_pinned_bytes_per_sec / GIB:.1f} GiB/s")
    if prof.disk_read_bytes_per_sec:
        print(f"  disk (measured):  {prof.disk_read_bytes_per_sec / GIB:.1f} GiB/s")
    print(f"  weight cache:     {format_bytes(prof.hot_budget_bytes())} of VRAM")
    print(f"  ram for weights:  {format_bytes(prof.warm_budget_bytes())}")


def cmd_doctor(args: argparse.Namespace) -> int:
    prof = hardware.profile(probe_path=args.probe_path, run_benchmarks=not args.fast)
    _print_profile(prof)

    if not args.model:
        print("\nPass a model id or path to estimate achievable throughput.")
        return 0

    try:
        if args.model.startswith(".") or "/" in args.model and args.local:
            topo = inspector.from_path(args.model)
        else:
            topo = inspector.from_hub(args.model)
    except Exception as exc:
        print(f"\ncould not read model config: {exc}", file=sys.stderr)
        return 1

    print("\n" + topo.summary())

    kv = inspector.estimate_kv_cache_bytes(topo, args.context)
    hot_budget = prof.hot_budget_bytes()
    cacheable = min(hot_budget, topo.bytes_per_token)

    rl = hardware.roofline(
        prof,
        bytes_per_token=topo.bytes_per_token,
        model_total_bytes=topo.total_bytes,
        hot_cache_bytes=cacheable if topo.kind is ModelKind.MOE else 0,
    )

    print("\nprojection")
    print(f"  weights on disk:  {format_bytes(topo.total_bytes)}")
    print(f"  touched/token:    {format_bytes(topo.bytes_per_token)}")
    print(f"  kv cache @ {args.context}: {format_bytes(kv)}")
    print(f"  bound by:         {rl.bound_by}")
    print(f"  ceiling:          {rl.max_tokens_per_sec:.2f} tok/s")
    for note in rl.notes:
        print(f"  note:             {note}")

    if topo.total_bytes > prof.disk_free_bytes:
        print("\n  WARNING: model exceeds free disk space.")
    if topo.kind is ModelKind.DENSE and rl.max_tokens_per_sec < 2:
        print(
            "\n  A dense model touches every weight per token, so this ceiling\n"
            "  cannot be raised by caching or overlap -- only by a smaller model\n"
            "  or a heavier quantization. An MoE model of similar total size\n"
            "  would run several times faster here."
        )
    return 0


def _make_compute_fn(matrix_dim: int):
    """Synthetic GPU work standing in for a layer's compute.

    A benchmark with no compute cannot show what overlap is worth: with
    nothing to hide transfers behind, prefetching only adds contention. This
    keeps the GPU genuinely busy so the measurement reflects the interleaving
    the real engine would do.
    """
    import torch

    if matrix_dim <= 0 or not torch.cuda.is_available():
        return lambda step, tensors: None

    a = torch.randn(matrix_dim, matrix_dim, device="cuda", dtype=torch.float16)
    b = torch.randn(matrix_dim, matrix_dim, device="cuda", dtype=torch.float16)

    def compute(step: int, tensors) -> None:
        torch.mm(a, b)

    return compute


def cmd_bench(args: argparse.Namespace) -> int:
    """Benchmark the streaming core with synthetic weights of realistic size."""
    prof = hardware.profile(run_benchmarks=not args.no_probe)
    hot_budget = (
        int(args.hot_budget_gib * GIB)
        if args.hot_budget_gib
        else prof.hot_budget_bytes()
    )

    if args.moe:
        keys = [f"shared.{i}" for i in range(args.layers)]
        keys += [
            f"expert.{i}.{e}"
            for i in range(args.layers)
            for e in range(args.experts)
        ]
        plan = MoEPlan(
            num_layers=args.layers,
            num_experts=args.experts,
            experts_per_token=args.top_k,
        )
        router = zipf_router(args.experts, args.top_k, skew=args.skew)
        chunk_bytes = int(args.chunk_mib * 1024 * 1024)
    else:
        keys = [f"layer.{i}" for i in range(args.layers)]
        plan = DensePlan(num_layers=args.layers)
        router = None
        chunk_bytes = int(args.chunk_mib * 1024 * 1024)

    source = SyntheticSource(keys, chunk_bytes=chunk_bytes)
    total = len(keys) * chunk_bytes

    print(f"chunks:        {len(keys)} x {format_bytes(chunk_bytes)}")
    print(f"working set:   {format_bytes(total)}")
    print(f"weight cache:  {format_bytes(hot_budget)}")
    print(f"prefetch:      depth {args.depth}, {args.workers} workers\n")

    config = EngineConfig(
        hot_budget_bytes=hot_budget,
        prefetch_depth=args.depth,
        workers=args.workers,
        reference_bandwidth_bytes_per_sec=prof.pcie_pinned_bytes_per_sec,
    )
    with StreamingEngine(
        source, plan, config, compute_fn=_make_compute_fn(args.compute_dim)
    ) as engine:
        report = engine.run(tokens=args.tokens, router=router)
    print(report.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="corestream",
        description="Tiered-memory streaming inference for consumer GPUs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor", help="profile this machine and project a model's throughput"
    )
    doctor.add_argument("model", nargs="?", help="Hugging Face model id or local path")
    doctor.add_argument("--local", action="store_true", help="treat model as a path")
    doctor.add_argument("--context", type=int, default=8192)
    doctor.add_argument("--probe-path", default=".")
    doctor.add_argument("--fast", action="store_true", help="skip bandwidth probes")
    doctor.set_defaults(func=cmd_doctor)

    bench = sub.add_parser("bench", help="benchmark the streaming core")
    bench.add_argument("--moe", action="store_true", help="benchmark an MoE shape")
    bench.add_argument("--layers", type=int, default=32)
    bench.add_argument("--experts", type=int, default=64)
    bench.add_argument("--top-k", type=int, default=8)
    bench.add_argument("--skew", type=float, default=1.1)
    bench.add_argument("--chunk-mib", type=float, default=16.0)
    bench.add_argument("--tokens", type=int, default=16)
    bench.add_argument("--depth", type=int, default=2)
    bench.add_argument("--workers", type=int, default=2)
    bench.add_argument("--hot-budget-gib", type=float, default=None)
    bench.add_argument(
        "--compute-dim",
        type=int,
        default=2048,
        help="size of the synthetic matmul per step; 0 disables compute",
    )
    bench.add_argument("--no-probe", action="store_true", help="skip bandwidth probes")
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
