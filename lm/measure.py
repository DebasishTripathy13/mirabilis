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

    def args_for(ncmoe: int | None, t: int) -> list[str]:
        a = ["-ngl", "999", "-t", str(t), "-c", str(context), "-fa", "on"]
        if ncmoe is not None:
            a += ["-ncmoe", str(ncmoe)]
        return a

    if info.is_moe and info.layers:
        # Sweep how many layers keep their experts in RAM. Fewer means more
        # experts on the GPU, which is faster until VRAM runs out and the load
        # either fails or starts evicting.
        steps = sorted({info.layers,
                        int(info.layers * 0.85),
                        int(info.layers * 0.75),
                        int(info.layers * 0.65)}, reverse=True)
        for n in steps:
            out.append((f"experts in RAM: {n}/{info.layers} layers",
                        args_for(n, threads), n, threads))
    else:
        out.append(("auto placement", base.to_args(), None, threads))

    # Thread count matters on hybrid CPUs; test one alternative either side.
    alt = max(1, hw.performance_cores or threads // 2)
    if alt != threads:
        best_ncmoe = out[0][2] if out else None
        out.append((f"{alt} threads (performance cores only)",
                    args_for(best_ncmoe, alt), best_ncmoe, alt))
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
    for label, args, ncmoe, threads in candidates(hw, info, base):
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
                        samples=rates)
        results.append(result)
        if on_result:
            on_result(result)
    return results
