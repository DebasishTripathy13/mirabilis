# `lm` — run large models on a laptop

Ollama-style commands, with placement tuned to the machine it is running on.
On an RTX 3060 Laptop (6 GB) with 30 GB of RAM it runs an **80B model at
~13 tok/s** — the same speed Ollama gets on an 8B here.

```
lm pull unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF
lm tune qwen3-next
lm run  qwen3-next
```

## Install

Needs `llama-server`, which Ollama already bundles. If Ollama is installed,
nothing else is required.

```bash
pip install -e .        # from this repository
lm doctor               # check hardware and that the engine was found
```

Point `LM_LLAMA_SERVER` at a different binary to use your own llama.cpp build.

## Commands

| command | what it does |
|---|---|
| `lm pull <hf-repo>` | download from Hugging Face, choosing the quantization that fits |
| `lm list` | installed models |
| `lm run <name>` | interactive chat |
| `lm run <name> "prompt"` | one-shot answer |
| `lm serve <name>` | start the server, OpenAI-compatible API |
| `lm tune <name>` | measure the fastest placement and remember it |
| `lm ps` / `lm stop` | what is running / stop it |
| `lm rm <name>` | uninstall (`--purge` also deletes the weights) |
| `lm doctor [name]` | hardware report, and the plan for a model |

Names accept unambiguous prefixes, so `lm run qwen3-next` is enough.

## What it does differently

**Picks the quantization by what fits memory, not by a quality preset.**
RAM reads about twelve times faster than NVMe on a typical laptop, so a larger
quantization that spills to disk is *slower* than a smaller one that does not.
`lm pull` shows every quantization with whether it fits and takes the
best-quality one that does:

```
quant                          size   fits memory
IQ2_XXS                      24.4 GiB yes
Q2_K_XL                      28.1 GiB yes <-- selected
Q4_K_M                       45.2 GiB  no
```

**Places tensors by role, not by layer index.** For a mixture-of-experts
model, attention and norms are small and read on every token, so they go on the
GPU; expert banks are large and mostly idle per token, so they stay in RAM.
This is what makes an 80B usable on a 6 GB card — it *stores* like an 80B and
*reads* like a 3B.

**Uses physical cores, not every thread.** On an i9-12900H, 20 threads measured
9.2 tok/s against 12.3 for 14. Decode is memory-bound, so the slow E-cores
stall the fast P-cores instead of adding throughput.

**Measures instead of guessing, when asked.** `lm tune` runs several placements
and keeps the fastest. It reports the best of several runs per candidate rather
than the average, because throughput noise is one-sided — nothing makes a run
faster than the hardware allows, while page-cache pressure and thermal
throttling make individual runs slower. On a model whose size is close to
available RAM, single-sample measurements varied by over 50%, enough to pick
the wrong winner.

## Choosing a model

Total size decides what must be **stored**. Active parameters decide what must
be **read per token**, and only the second one costs time.

| | total | active/token | on this laptop |
|---|---|---|---|
| Llama-70B dense | 35 GiB @ 4-bit | 35 GiB | ~0.1 tok/s |
| Qwen2.5-32B dense | 18.5 GiB | 18.5 GiB | 2.4 tok/s |
| **Qwen3-Next-80B-A3B** | 28 GiB @ Q2_K_XL | ~1 GiB | **~13 tok/s** |

A dense 70B cannot be fast here at any quantization — 35 GiB has to cross a
2.9 GiB/s link every token. A large MoE can be, because it reads a small
fraction of itself. **Shop for low active-parameter count, not small total
size.**

## Notes

- Weights live in the shared Hugging Face cache, so `lm` and other tools use
  one copy. `lm rm` leaves them alone unless given `--purge`.
- `lm serve` exposes `/v1/chat/completions`, so existing OpenAI clients work
  against it unchanged.
- The reasoning behind every default is measured in
  [`LAPTOP-INFERENCE.md`](LAPTOP-INFERENCE.md).
