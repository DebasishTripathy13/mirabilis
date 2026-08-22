"""Finding and fetching GGUF models from Hugging Face.

The useful part is quantization choice. Near the memory boundary, picking a
quant is not really a quality/size tradeoff -- it decides whether the model is
read from RAM or from disk, and those differ by more than a factor of ten on a
typical laptop. So the default is the largest quant whose working set still
fits in memory, and nothing smaller: going below that gives up quality without
buying speed, because the bottleneck has already moved.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass

GIB = 1024**3

# Ordered worst to best. Used to prefer higher quality when two candidates
# both fit.
_QUALITY_ORDER = [
    "TQ1", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "Q2_K", "Q2_K_L", "Q2_K_XL", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_XL", "IQ4_XS", "IQ4_NL",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q4_K_XL",
    "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q6_K", "Q6_K_XL", "Q8_0", "BF16", "F16", "F32",
]
_SPLIT = re.compile(r"-0000\d+-of-0000\d+", re.IGNORECASE)


@dataclass
class Quant:
    label: str
    size_gib: float
    files: list[str]

    @property
    def quality_rank(self) -> int:
        best = -1
        upper = self.label.upper()
        for index, name in enumerate(_QUALITY_ORDER):
            if name in upper:
                best = max(best, index)
        return best


def _label_from(filename: str) -> str:
    base = os.path.basename(filename)
    base = _SPLIT.sub("", base)
    base = re.sub(r"\.gguf$", "", base, flags=re.IGNORECASE)
    # Prefer the trailing quant token, which is how these files are named.
    parts = base.split("-")
    for tail in range(len(parts) - 1, -1, -1):
        candidate = "-".join(parts[tail:])
        if any(name in candidate.upper() for name in _QUALITY_ORDER):
            return candidate
    return base


def repo_metadata(repo: str, token: str | None = None) -> dict:
    """Architecture and parameter count, as Hugging Face already parsed them.

    Hugging Face reads the GGUF header server-side and exposes the result, so
    the architecture is known before any weights are fetched. That is what lets
    `pull` warn that a dense model will be slow while there is still a choice.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, token=token)
    meta = getattr(info, "gguf", None) or {}
    return {
        "architecture": meta.get("architecture", ""),
        "parameters": meta.get("total", 0),
        "context_length": meta.get("context_length", 0),
    }


def is_companion(filename: str) -> bool:
    """Files that sit beside a model rather than being one.

    `mmproj-*.gguf` is a vision projector and `*imatrix*.gguf` is calibration
    data. Both are GGUF files in the same repo, and a projector is small enough
    that treating it as a quantization makes it look like the best thing that
    fits -- an `mmproj-…-f16.gguf` at 0.9 GiB was selected over the real 15 GiB
    model before this check existed.
    """
    base = os.path.basename(filename).lower()
    return base.startswith("mmproj") or "imatrix" in base


def find_projector(repo: str, token: str | None = None) -> str | None:
    """Repo-relative path of the vision projector, if the model has one."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True, token=token)
    projectors = [
        s.rfilename for s in info.siblings
        if s.rfilename.lower().endswith(".gguf")
        and os.path.basename(s.rfilename).lower().startswith("mmproj")
    ]
    # Prefer the f16 projector when several precisions are published; it is
    # under a gigabyte and quantizing it saves nothing worth the quality.
    projectors.sort(key=lambda p: ("f16" not in p.lower(), p))
    return projectors[0] if projectors else None


def download_file(repo: str, filename: str, token: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    return os.path.realpath(hf_hub_download(repo, filename, token=token))


def list_quants(repo: str, token: str | None = None) -> list[Quant]:
    """Group a repo's GGUF files into quantizations, largest last."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True, token=token)
    grouped: dict[str, list] = defaultdict(list)
    for sibling in info.siblings:
        name = sibling.rfilename
        if not name.lower().endswith(".gguf"):
            continue
        if is_companion(name):
            continue
        grouped[_label_from(name)].append(sibling)

    quants = []
    for label, siblings in grouped.items():
        total = sum((s.size or 0) for s in siblings)
        quants.append(
            Quant(label=label, size_gib=total / GIB,
                  files=sorted(s.rfilename for s in siblings))
        )
    return sorted(quants, key=lambda q: q.size_gib)


def choose_quant(quants: list[Quant], budget_gib: float, *,
                 rate_for=None, min_rate: float = 0.0) -> Quant | None:
    """Pick the best-quality quant that fits, and is not painfully slow.

    Fitting memory is the first constraint, but it is not the only one. For a
    dense model every parameter is read on every token, so a larger quant is
    directly slower -- "the best quality that fits" would happily choose a
    20 GiB file that runs at 2 tok/s over a 10 GiB one at 5. For a
    mixture-of-experts model size barely affects speed once it fits, so the
    floor never binds and quality wins as it should.

    `rate_for` estimates tok/s for a size; `min_rate` is the floor worth
    insisting on. If nothing clears the floor the fastest candidate is
    returned, since the alternative is recommending something unusable.
    """
    if not quants:
        return None
    fitting = [q for q in quants if q.size_gib <= budget_gib]
    if not fitting:
        return None
    if rate_for is None or min_rate <= 0:
        return max(fitting, key=lambda q: (q.quality_rank, q.size_gib))

    acceptable = [q for q in fitting if rate_for(q.size_gib) >= min_rate]
    if acceptable:
        return max(acceptable, key=lambda q: (q.quality_rank, q.size_gib))
    return min(fitting, key=lambda q: q.size_gib)


def download(repo: str, quant: Quant, token: str | None = None) -> str:
    """Fetch one quantization; returns the local path of its first shard."""
    from huggingface_hub import snapshot_download

    patterns = [f"*{os.path.basename(f)}" for f in quant.files]
    path = snapshot_download(repo, allow_patterns=patterns, token=token,
                             max_workers=8)
    for name in quant.files:
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return os.path.realpath(candidate)
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if name.lower().endswith(".gguf"):
                return os.path.realpath(os.path.join(root, name))
    raise FileNotFoundError(f"no GGUF found after downloading {repo}")
