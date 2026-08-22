# Evidence

Verbatim terminal output from the machine the README describes: an RTX 3060
Laptop (6 GB VRAM), i9-12900H, 30 GB DDR5, NVMe, Linux.

Captured with `mirabilis` at the commit that added this directory. Absolute paths are
rewritten to `~` and nothing else is edited — including the parts that make the
setup look worse than a tuned demo would, such as the RAM shortfall warning and
the wide `(min ...)` spread on individual tune candidates.

| file | what it shows |
|---|---|
| `doctor.txt` | hardware detection, GPU backend discovery, RAM shortfall warning |
| `plan-80b.txt` | the placement plan for the 80B and the reasoning behind each choice |
| `pull-quants.txt` | every quantization on Hugging Face with fit and predicted speed, before downloading |
| `tune-80b.txt` | the placement sweep for the MoE, failures included |
| `tune-27b.txt` | the same sweep for a dense model, which prefers a different thread count |
| `run-80b.txt` | a live generation with the server's own timings |
| `gpu-idle.txt` | GPU utilisation sampled during decode |
| `tiers.txt` | measured bandwidth of every memory tier and both compute units |

Two things in here are worth reading closely because they are easy to miss.

**The tune tables contain failures.** Rows marked `failed` are configurations
that would not allocate. They are kept because the boundary is a cliff rather
than a slope: `-ncmoe 44` runs at 22.9 tok/s and `-ncmoe 42` does not run at
all. A sweep that hid its failures would look tidier and teach less.

**The `(min ...)` column is the noise.** Each candidate is measured several
times and the best is reported, because throughput noise is one-sided — nothing
runs faster than the hardware allows, while cache pressure and thermal
throttling only slow things down. On a model sized near available RAM, single
samples varied by over 50% and picked the wrong winner.
