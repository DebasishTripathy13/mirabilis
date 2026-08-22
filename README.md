# lm — run an 80B model on a 6 GB laptop

**Qwen3-Next-80B at ~26 tok/s** on an RTX 3060 Laptop (6 GB VRAM, 30 GB RAM) —
the throughput Ollama gets on an *8B* model on the same machine, from one ten
times larger.

```bash
pip install -e .
lm pull unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF   # picks the quant that fits
lm tune qwen3-next                                  # measures the fastest setup
lm run  qwen3-next                                  # chat
```

Ollama-style commands, with placement measured on the machine it runs on
rather than guessed. Full reference in [`LM.md`](LM.md).

| | model | sustained |
|---|---|---|
| **this configuration** | Qwen3-Next-80B-A3B (80B total, ~3B active) | **26.4 tok/s** |
| Ollama, tuned | `ministral-3:8b` (8B dense) | 11.7 tok/s |
| Ollama, tuned | `qwen2.5-coder:32b` (32B dense) | 2.4 tok/s |

## Why an 80B beats a 32B here

Total size decides what must be **stored**. Active parameters decide what must
be **read per token** — and only the second one costs time, because decoding is
memory-bound. A dense 32B reads all 18.5 GiB every token. An 80B
mixture-of-experts reads about 1 GiB.

So the thing to shop for is a low active-parameter count, not a small total
size. A dense 70B cannot be fast on this hardware at any quantization; an 80B
MoE can.

## What the tool does differently

- **Picks the quantization by what fits memory**, not by a quality preset. RAM
  reads ~12x faster than NVMe, so a larger quant that spills to disk is slower
  than a smaller one that does not.
- **Places tensors by role.** Attention and norms on the GPU (small, read every
  token); expert banks in RAM (large, mostly idle per token).
- **Finds the GPU backend.** Ollama hides `libggml-cuda.so` in a subdirectory
  ggml does not search, so a directly-launched server silently runs on CPU.
  Worth 1.65x here, and invisible without checking.
- **Measures instead of guessing.** `lm tune` sweeps placement, then threads,
  and keeps the best of several runs per candidate — throughput noise is
  one-sided, so the fastest observed run is the honest estimate.

## Where the speed came from

Each step measured on the same model, from a starting point of 12.98 tok/s:

| change | gain |
|---|---|
| CUDA backend actually loading | **1.65x** |
| freeing RAM so the model stays cached | 1.14x |
| thread pinning (6 physical P-cores + 2 E-cores) | 1.15x |
| expert split + KV cache quantization | 1.08x |
| *(throughput only)* 4 concurrent streams | 1.67x aggregate |

**Tested and gave nothing:** huge pages, `--no-mmap`, MTP and n-gram
speculative decoding, expert layer choice via `-ot`, trading context for VRAM.
Negative results are recorded alongside the wins in
[`LAPTOP-INFERENCE.md`](LAPTOP-INFERENCE.md), which also has the measured
bandwidth of every memory tier on this class of machine and the arithmetic that
predicts throughput before downloading anything.

## Repository layout

| path | what it is |
|---|---|
| [`LM.md`](LM.md) | the CLI: commands, options, model selection |
| [`LAPTOP-INFERENCE.md`](LAPTOP-INFERENCE.md) | measured hardware model and every finding, including the failures |
| `lm/` | the tool itself |
| `run-80b.sh` | standalone launcher, no CLI needed |
| `corestream/` | an earlier streaming engine that lost to llama.cpp — see its [post-mortem](docs/corestream-postmortem.md) |
| `tests/` | 93 tests |

## Requirements

`llama-server`, which Ollama already bundles — if Ollama is installed there is
nothing else to get. Point `LM_LLAMA_SERVER` at a different binary to use your
own llama.cpp build. `lm doctor` reports what it found, including whether a GPU
backend is present.

Everything here was measured on one machine: RTX 3060 Laptop (6 GB),
i9-12900H, 30 GB DDR5, NVMe. The numbers will differ on yours; `lm tune`
exists so the *configuration* does not have to be copied along with them.
