"""Find the best placement by measuring it.

The planner in `tune.py` has to guess at things a GGUF file does not state:
what fraction of the weights are expert banks, and how many layers carry a real
KV cache rather than a small recurrent state. Those guesses were off by enough
on a real model to cost about 2% throughput and, in other configurations,
could cost much more.

Measuring a handful of candidates takes a few minutes once, and the answer is
then correct for this machine and this model rather than approximately right.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field

from . import server
from .gguf import GGUFInfo
from .hardware import Hardware
from .tune import Plan

PROMPT = "Explain how a mixture-of-experts transformer routes tokens:"


@dataclass
class Result:
    label: str
    args: list[str]
    tokens_per_second: float
    ncmoe: int | None
    threads: int
    gpu_layers: int | None = None
    cache_type: str = ""
    cpu_mask: str = ""
    samples: list = field(default_factory=list)

    @property
    def spread(self) -> str:
        if len(self.samples) < 2:
            return ""
        return f"  (min {min(self.samples):.1f})"


def _measure(port: int, tokens: int) -> float:
    body = {"prompt": PROMPT, "n_predict": tokens, "temperature": 0,
            "cache_prompt": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r).get("timings", {}).get("predicted_per_second", 0.0)


def candidates(hw: Hardware, info: GGUFInfo, base: Plan) -> list[tuple[str, list[str], int | None, int]]:
    """Placements worth trying, cheapest-to-load first."""
    out: list[tuple[str, list[str], int | None, int]] = []
    threads = base.threads
    context = base.context
    base_mask = base.cpu_mask
    # The vision projector is a real VRAM tenant -- nearly a gigabyte on some
    # models -- and it is attached at run time. Measuring without it picks a
    # layer count that then fails to allocate the moment it is added.
    extras: list[str] = []
    if base.projector:
        extras += ["--mmproj", base.projector]
    if base.jinja:
        extras += ["--jinja"]

    def args_for(ncmoe: int | None, t: int, cache: str = "",
                 mask: str = "") -> list[str]:
        a = ["-ngl", "999", "-t", str(t), "-c", str(context), "-fa", "on"]
        if ncmoe is not None:
            a += ["-ncmoe", str(ncmoe)]
        if cache:
            a += ["-ctk", cache, "-ctv", cache]
        if mask:
            a += ["-C", mask, "--cpu-strict", "1"]
        return a + extras

    if info.is_moe and info.layers:
        # Walk down from "all experts in RAM", moving a few layers at a time
        # onto the GPU. The interesting region is right at the top: attention
        # already occupies most of the VRAM, so only a handful of expert
        # layers fit beside it, and one step too far fails to allocate rather
        # than degrading gracefully. Coarse fractions miss this entirely --
        # sweeping 85/75/65% found only failures on a 6 GB card.
        layers = info.layers
        steps = [layers]
        for delta in (2, 4, 6, 9, 12):
            n = layers - delta
            if n > 0:
                steps.append(n)
        for n in steps:
            label = ("all experts in RAM" if n == layers
                     else f"{layers - n} expert layers on GPU")
            out.append((f"{label} (ncmoe={n})", args_for(n, threads, "", base_mask),
                        n, threads, "", None, base_mask))
        # Quantizing the KV cache frees VRAM, which lets the split go one or
        # two layers further than it otherwise could. Measured worth ~8%:
        # ncmoe=43 alone reached 22.4 tok/s, with q8_0 it reached 23.3 and
        # ncmoe=42 became reachable at all.
        for delta in (3, 5, 7):
            n = layers - delta
            if n > 0:
                out.append((f"{delta} expert layers on GPU + KV q8_0",
                            args_for(n, threads, "q8_0", base_mask), n, threads,
                            "q8_0", None, base_mask))
    else:
        # Dense: every layer is read per token, so the only question is how
        # many sit on the fast tier. llama.cpp's own fitter is the baseline;
        # pushing past it is worth trying because quantizing the KV cache
        # frees room the fitter did not know it would have.
        def dense_args(ngl: int | None, cache: str = "") -> list[str]:
            a = ["-t", str(threads), "-c", str(context), "-fa", "on"]
            if ngl is not None:
                a += ["-ngl", str(ngl)]
            if cache:
                a += ["-ctk", cache, "-ctv", cache]
            if base_mask:
                a += ["-C", base_mask, "--cpu-strict", "1"]
            return a + extras

        out.append(("auto placement", dense_args(None), None, threads, "",
                    None, base_mask))
        out.append(("auto placement + KV q8_0",
                    dense_args(None, "q8_0"), None, threads, "q8_0", None,
                    base_mask))
        if info.layers:
            # llama.cpp's own fitter reserves conservatively, and quantizing
            # the KV cache frees room it did not account for, so it is worth
            # walking up until the allocation fails. The starting point is
            # included explicitly and the range extends below it as well: once
            # a previous tune has raised `-ngl`, an upward-only sweep tests
            # nothing but configurations that fail and silently abandons the
            # value that was already winning.
            fitted = base.gpu_layers if 0 < base.gpu_layers < info.layers else None
            if fitted:
                seen = set()
                for delta in (-8, -4, 0, 4, 8, 12, 16, 22):
                    n = min(info.layers, fitted + delta)
                    if n <= 0 or n in seen:
                        continue
                    seen.add(n)
                    out.append((f"{n}/{info.layers} layers on GPU + KV q8_0",
                                dense_args(n, "q8_0"), None, threads, "q8_0", n,
                                base_mask))

    return out


def thread_candidates(hw: Hardware, base: Plan, winner: Result,
                      context: int) -> list[tuple]:
    """Thread and affinity options, built on an already-measured placement.

    This has to be a second phase rather than part of one flat sweep: the
    right thread count depends on where the weights ended up, and the sweep is
    generated before anything has been measured. Testing threads against
    whichever placement happened to be listed first produced a saved
    configuration that combined the best thread count with a placement that
    had already lost.
    """
    out: list[tuple] = []
    if not hw.fast_core_ids:
        return out
    fast = len(hw.fast_core_ids)
    for extra in (0, 2, 4):
        threads = fast + min(extra, len(hw.slow_core_ids))
        mask = hw.affinity_mask(threads)
        if not mask:
            continue
        args = list(winner.args)
        for flag in ("-t", "-C", "--cpu-strict"):
            while flag in args:
                index = args.index(flag)
                del args[index:index + 2]
        args += ["-t", str(threads), "-C", mask, "--cpu-strict", "1"]
        label = (f"{threads} threads pinned ({fast} fast"
                 + (f" + {threads - fast} slow)" if threads > fast else ")"))
        out.append((label, args, winner.ncmoe, threads, winner.cache_type,
                    winner.gpu_layers, mask))
    return out


def run(name: str, model_path: str, hw: Hardware, info: GGUFInfo, base: Plan,
        tokens: int = 48, repeats: int = 3, on_result=None) -> list[Result]:
    """Measure each candidate, reporting the best of `repeats` runs.

    Best-of rather than mean: throughput noise here is one-sided. Nothing makes
    a run faster than the hardware allows, while page-cache pressure, other
    processes, and clock throttling all make individual runs slower. The
    fastest observed run is therefore the closest estimate of what the
    configuration can actually sustain.

    This matters more than it sounds. A model whose size is close to available
    RAM is reloaded into a page cache that is already nearly full, so
    single-sample measurements of the same configuration were observed to vary
    by over 50%, enough to pick the wrong winner.
    """
    def measure_all(items) -> list[Result]:
        found: list[Result] = []
        for label, args, ncmoe, threads, cache, ngl, mask in items:
            running = None
            rates: list[float] = []
            try:
                running = server.start(name, model_path, args)
                _measure(running.port, 24)             # warm the page cache
                for _ in range(max(1, repeats)):
                    rates.append(_measure(running.port, tokens))
            except Exception:                          # noqa: BLE001
                pass
            finally:
                if running is not None:
                    server.stop(running)
                time.sleep(2)
            result = Result(label, args, max(rates, default=0.0), ncmoe,
                            threads, gpu_layers=ngl, cache_type=cache,
                            cpu_mask=mask, samples=rates)
            found.append(result)
            if on_result:
                on_result(result)
        return found

    # Phase one: where the weights live. Phase two: how many threads work on
    # them, applied to whichever placement actually won. Searching them as one
    # flat list would test thread counts against an arbitrary placement.
    results = measure_all(candidates(hw, info, base))
    best = max(results, key=lambda r: r.tokens_per_second, default=None)
    if best is not None and best.tokens_per_second > 0:
        results += measure_all(
            thread_candidates(hw, base, best, base.context))
    return results
