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
    cache_type: str = ""
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

    def args_for(ncmoe: int | None, t: int, cache: str = "") -> list[str]:
        a = ["-ngl", "999", "-t", str(t), "-c", str(context), "-fa", "on"]
        if ncmoe is not None:
            a += ["-ncmoe", str(ncmoe)]
        if cache:
            a += ["-ctk", cache, "-ctv", cache]
        return a

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
            out.append((f"{label} (ncmoe={n})", args_for(n, threads), n,
                        threads, ""))
        # Quantizing the KV cache frees VRAM, which lets the split go one or
        # two layers further than it otherwise could. Measured worth ~8%:
        # ncmoe=43 alone reached 22.4 tok/s, with q8_0 it reached 23.3 and
        # ncmoe=42 became reachable at all.
        for delta in (3, 5, 7):
            n = layers - delta
            if n > 0:
                out.append((f"{delta} expert layers on GPU + KV q8_0",
                            args_for(n, threads, "q8_0"), n, threads, "q8_0"))
    else:
        out.append(("auto placement", base.to_args(), None, threads, ""))
        out.append(("auto placement + KV q8_0",
                    base.to_args() + ["-ctk", "q8_0", "-ctv", "q8_0"],
                    None, threads, "q8_0"))

    # Thread count matters on hybrid CPUs; test one alternative either side.
    alt = max(1, hw.performance_cores or threads // 2)
    if alt != threads:
        best_ncmoe = out[0][2] if out else None
        out.append((f"{alt} threads (performance cores only)",
                    args_for(best_ncmoe, alt), best_ncmoe, alt, ""))
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
    results: list[Result] = []
    for label, args, ncmoe, threads, cache in candidates(hw, info, base):
        running = None
        rates: list[float] = []
        try:
            running = server.start(name, model_path, args)
            _measure(running.port, 24)                 # warm the page cache
            for _ in range(max(1, repeats)):
                rates.append(_measure(running.port, tokens))
        except Exception:                              # noqa: BLE001
            pass
        finally:
            if running is not None:
                server.stop(running)
            time.sleep(2)
        result = Result(label, args, max(rates, default=0.0), ncmoe, threads,
                        cache_type=cache, samples=rates)
        results.append(result)
        if on_result:
            on_result(result)
    return results
