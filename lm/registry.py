"""Which models are installed, and where.

The GGUF files themselves stay in the Hugging Face cache so that `lm` and any
other tool share one copy. This registry only records names, paths, and what
was learned about each model at install time.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

HOME = os.path.expanduser(os.environ.get("LM_HOME", "~/.lm"))
REGISTRY = os.path.join(HOME, "models.json")


@dataclass
class Entry:
    name: str
    repo: str
    quant: str
    path: str
    size_gib: float
    architecture: str = ""
    layers: int = 0
    experts: int = 0
    experts_used: int = 0
    installed_at: float = field(default_factory=time.time)
    # Filled in by `lm tune`: a measured placement always beats a predicted one.
    tuned_ncmoe: int | None = None
    tuned_threads: int | None = None
    tuned_tokens_per_second: float = 0.0
    tuned_cache_type: str = ""
    # Vision projector, when the model has one. Kept beside the weights so
    # `lm run` can restore image input without a second flag.
    projector: str = ""

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path)

    @property
    def kind(self) -> str:
        if self.experts > 1:
            return f"MoE {self.experts}x top-{self.experts_used}"
        return "dense"


def _load_raw() -> dict:
    try:
        with open(REGISTRY) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load() -> dict[str, Entry]:
    out = {}
    for name, data in _load_raw().items():
        known = {k: v for k, v in data.items() if k in Entry.__dataclass_fields__}
        out[name] = Entry(**known)
    return out


def save(entries: dict[str, Entry]) -> None:
    os.makedirs(HOME, exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w") as f:
        json.dump({n: asdict(e) for n, e in entries.items()}, f, indent=2)
    os.replace(tmp, REGISTRY)      # atomic, so an interrupted write cannot
                                   # leave a half-written registry behind


def add(entry: Entry) -> None:
    entries = load()
    entries[entry.name] = entry
    save(entries)


def remove(name: str) -> Entry | None:
    entries = load()
    entry = entries.pop(name, None)
    if entry is not None:
        save(entries)
    return entry


def get(name: str) -> Entry | None:
    entries = load()
    if name in entries:
        return entries[name]
    # Allow unambiguous prefixes, the way container tooling does.
    matches = [e for n, e in entries.items() if n.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def default_name(repo: str, quant: str) -> str:
    """Short, memorable name: the model part of the repo plus the quant."""
    base = repo.split("/")[-1]
    for suffix in ("-GGUF", "-gguf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    short = quant
    for token in (base, base.lower()):
        short = short.replace(token + "-", "").replace(token, "")
    short = short.strip("-.") or quant
    return f"{base}:{short}".lower()
