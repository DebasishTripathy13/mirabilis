"""Hugging Face integration: stream a real model's layers during generation.

The model is built on the `meta` device, so no weights are allocated at load.
Each decoder layer then gets a pair of forward hooks:

*   the pre-hook fetches that layer's chunk from the store and points the
    layer's parameters at slices of it,
*   the post-hook points them back at meta placeholders, releasing the buffer
    unless the store chose to cache it.

Because the hooks leave an ordinary `nn.Module` behind, everything downstream
-- `generate`, sampling, KV caching, batching -- works unmodified. There is no
custom generation loop here, which is deliberate: reimplementing sampling
would be a large surface for subtle divergence from stock behaviour, and the
whole point is that output should be indistinguishable from an unstreamed run.

Weights without a layer index (embeddings, final norm, LM head) are placed on
the device once and left there. They are read on every token regardless of
routing, so streaming them would pay transfer cost on every step for nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .loaders import DensePlan
from .manifest import ManifestChunkSource, ModelManifest, build_manifest
from .scheduler import PrefetchScheduler
from .store import LFUAdmission, StaticPinning, TieredWeightStore

# Local parameter names that make up a packed quantized weight.
_COMPRESSED_LOCAL_NAMES = frozenset(
    {"weight_packed", "weight_scale", "weight_shape", "weight_zero_point",
     "weight_g_idx"}
)


def decompress_weight(compressed, scheme, dtype, column_group=None):
    from .dequant import decompress_weight as _impl

    return _impl(compressed, scheme, dtype, column_group)


def _resolve(root: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    """Map a dotted parameter name to its owning module and attribute."""
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def _assign(module: torch.nn.Module, attr: str, tensor: torch.Tensor) -> None:
    """Attach a real tensor where a meta parameter or buffer currently sits.

    `param.data = tensor` is rejected across devices -- a meta parameter and a
    CUDA tensor are different tensor types as far as `set_data` is concerned --
    so the parameter object itself has to be replaced the first time.
    """
    if attr in module._parameters:
        module._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        setattr(module, attr, tensor)


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList | None:
    """Locate the stack of decoder layers.

    Identified structurally -- the largest `ModuleList` reached by a path
    ending in `layers` -- rather than by matching known architecture names, so
    it keeps working for models this code has never seen.
    """
    best: torch.nn.ModuleList | None = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers"):
            if best is None or len(module) > len(best):
                best = module
    return best


def resolve_model_path(model_id: str, token: str | None = None) -> str:
    """Return a local directory for a model id or path, downloading if needed."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id,
        token=token,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )


@dataclass
class StreamingConfig:
    hot_budget_bytes: int
    pinned_budget_bytes: int = 0
    prefetch_depth: int = 2
    workers: int = 2
    device: str = "cuda"
    dtype: torch.dtype | None = None


class StreamingModel:
    """An HF causal LM whose decoder layers are streamed from disk/RAM."""

    def __init__(self, model_path: str, config: StreamingConfig):
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.path = model_path
        self.config = config
        self.device = torch.device(config.device)

        hf_config = AutoConfig.from_pretrained(model_path)
        self.manifest: ModelManifest = build_manifest(model_path)

        # The checkpoint's storage dtype wins. Parameters are zero-copy views
        # over the transferred bytes, so building the model in another dtype
        # would silently mix precisions -- the model would compute in the
        # checkpoint's dtype anyway while reporting the requested one.
        dtype = self.manifest.dominant_dtype
        if config.dtype is not None and config.dtype != dtype:
            raise ValueError(
                f"checkpoint stores {dtype}; cannot stream it as {config.dtype}. "
                "Streaming uses views over the stored bytes, so conversion is "
                "not free. Convert the checkpoint offline instead."
            )
        self.dtype = dtype

        # `include_buffers=False` matters: buffers such as rotary-embedding
        # inverse frequencies are computed at init and absent from the
        # checkpoint. Left on meta they would never be filled, and the first
        # forward would fail on a meta tensor.
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(hf_config, dtype=dtype)
            # A quantized checkpoint stores packed integers, scales, and zero
            # points under parameter names that only exist once `nn.Linear`
            # has been swapped for the quantized module. `from_config` does
            # not do that swap -- it happens inside `from_pretrained` -- so it
            # has to be invoked here, or every quantized tensor in the
            # manifest would have nowhere to land.
            self.quantizer = self._apply_quantizer(model, hf_config)
        model.eval()

        self.source = ManifestChunkSource(self.manifest)

        self.store = TieredWeightStore(
            source=self.source,
            hot_budget_bytes=config.hot_budget_bytes,
            device=config.device,
            # A dense stack revisits every layer once per token in fixed
            # order, which is the pathological case for LRU. Pin a fixed
            # subset instead. Admission on first touch, since there is no
            # cold tail to guard against.
            admission=LFUAdmission(threshold=1),
            eviction=StaticPinning(),
            pinned_budget_bytes=config.pinned_budget_bytes,
        )
        self.scheduler = PrefetchScheduler(self.store, workers=config.workers)
        self.plan = DensePlan(num_layers=self.manifest.num_layers)

        self._meta_placeholder: dict[str, torch.Tensor] = {}
        self._resolved: dict[str, tuple[torch.nn.Module, str]] = {}
        self._live: dict[int, torch.Tensor] = {}
        self._handles: list = []

        self.model = model
        self._materialise_resident()
        self._move_buffers()
        self._install_hooks()

        self._verify_coverage()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -- setup ---------------------------------------------------------

    def _empty_for(self, dtype: torch.dtype) -> torch.Tensor:
        """A shared zero-length placeholder of the given dtype."""
        placeholder = self._empties.get(dtype)
        if placeholder is None:
            placeholder = torch.empty(0, dtype=dtype, device=self.device)
            self._empties[dtype] = placeholder
        return placeholder

    @staticmethod
    def _apply_quantizer(model: torch.nn.Module, hf_config) -> object | None:
        """Swap in quantized modules so quantized weights have a home.

        Returns the quantizer, or None for an unquantized checkpoint.
        """
        quant_config = getattr(hf_config, "quantization_config", None)
        if quant_config is None:
            return None

        from transformers.quantizers import AutoHfQuantizer

        quantizer = AutoHfQuantizer.from_config(quant_config, pre_quantized=True)
        method = (
            quant_config.get("quant_method")
            if isinstance(quant_config, dict)
            else getattr(quant_config, "quant_method", None)
        ) or "quantization"

        if method == "compressed-tensors":
            # Build the decompressed module layout, but skip the model-wide
            # compress/decompress that transformers pairs it with. That pass
            # expands every weight at once -- which needs the whole model
            # resident, and would undo the compression that makes streaming
            # worthwhile. The layers are decompressed one at a time in the
            # forward hooks instead.
            from compressed_tensors.quantization import apply_quantization_config

            apply_quantization_config(
                model, quantizer.compressor.quantization_config, run_compressed=False
            )
            return quantizer

        try:
            quantizer.validate_environment(device_map=None)
        except ImportError as exc:
            raise RuntimeError(
                f"this checkpoint is {method}-quantized, and that format needs a "
                f"backend this environment does not have: {exc}. The streaming "
                "path itself is format-agnostic -- it moves whatever tensors the "
                "checkpoint contains -- but the quantized modules that decode "
                "them come from that backend."
            ) from exc
        quantizer._process_model_before_weight_loading(model)
        return quantizer

    def _materialise_resident(self) -> None:
        """Place always-needed weights on the device permanently."""
        tensors = self.source.resident_tensors()
        for name, tensor in tensors.items():
            try:
                module, attr = _resolve(self.model, name)
            except AttributeError:
                continue  # Present in the checkpoint but not in this config.
            if getattr(module, attr, None) is None:
                continue
            _assign(module, attr, tensor.to(self.device, non_blocking=False))

        # A tied LM head has no entry of its own in the checkpoint; point it
        # at the embedding it shares rather than leaving it on meta.
        if getattr(self.model.config, "tie_word_embeddings", False):
            head = getattr(self.model, "lm_head", None)
            embed = self.model.get_input_embeddings()
            if head is not None and embed is not None and head.weight.is_meta:
                head.weight = embed.weight

    def _column_group(self, module, compressed: dict, scheme):
        """Cached column-to-group mapping for an activation-ordered module.

        Depends only on `weight_g_idx`, which is fixed for the module, so
        deriving it once avoids two sorts per weight per token.
        """
        g_idx = compressed.get("weight_g_idx")
        if g_idx is None:
            return None
        cached = self._column_groups.get(id(module))
        if cached is None:
            from .dequant import _is_column_order, column_group_index

            if _is_column_order(g_idx):
                return None
            cached = column_group_index(g_idx, int(scheme.weights.group_size))
            self._column_groups[id(module)] = cached
        return cached

    def _build_quant_plan(self) -> dict[str, list[tuple]]:
        """Group each layer's compressed tensors by the module they belong to.

        A packed weight is not a parameter the model can use directly -- it is
        `weight_packed` plus `weight_scale` and a shape, which together
        reconstruct `weight`. This records, per layer, which modules need that
        reconstruction and which checkpoint tensors feed it, so the forward
        hook can do the work for one layer at a time rather than the whole
        model at once.
        """
        self._compressed_tensor_names: set[str] = set()
        plan: dict[str, list[tuple]] = {}
        if self.quantizer is None:
            return plan

        for key, chunk in self.manifest.chunks.items():
            by_module: dict[str, dict[str, object]] = {}
            for spec in chunk.tensors:
                module_path, _, local = spec.name.rpartition(".")
                if local not in _COMPRESSED_LOCAL_NAMES:
                    continue
                try:
                    module = self.model.get_submodule(module_path)
                except AttributeError:
                    continue
                scheme = getattr(module, "quantization_scheme", None)
                if scheme is None or getattr(scheme, "weights", None) is None:
                    continue
                by_module.setdefault(module_path, {})[local] = spec
                self._compressed_tensor_names.add(spec.name)

            entries = []
            for module_path, tensors in by_module.items():
                if "weight_packed" not in tensors:
                    # Not actually a packed module; leave its tensors to the
                    # ordinary assignment path.
                    for spec in tensors.values():
                        self._compressed_tensor_names.discard(spec.name)
                    continue
                module = self.model.get_submodule(module_path)
                entries.append((module, module.quantization_scheme, tensors))
            if entries:
                plan[key] = entries
        return plan

    def _verify_coverage(self) -> None:
        """Refuse to run unless every parameter has a source of weights.

        A parameter the checkpoint never fills does not raise on its own. It
        stays at zero length, the layer computes on nothing, and generation
        emits fluent-looking garbage at full speed -- a 4-bit checkpoint whose
        module layout did not match produced "!!!!!!!!" at 20 tok/s while
        every test passed. Silence is the worst failure mode available here,
        so it is converted into a refusal to start.
        """
        supplied = {spec.name for chunk in self.manifest.chunks.values()
                    for spec in chunk.tensors}
        supplied.update(self.manifest.resident_tensors)
        # A packed weight has no parameter of its own; it reconstructs
        # `weight`, so credit that instead.
        for entries in self._quant_plan.values():
            for module, _scheme, _tensors in entries:
                for name, candidate in self.model.named_modules():
                    if candidate is module:
                        supplied.add(f"{name}.weight")
                        break

        # A parameter is unbacked if it is still on meta -- never materialised
        # at all -- or is a zero-length placeholder with nothing to fill it.
        # Checking length alone is not enough: a meta tensor reports its full
        # nominal `numel`, so an entirely absent weight looks well-formed.
        unfilled = [
            name
            for name, param in self.model.named_parameters()
            if name not in supplied and (param.is_meta or param.numel() == 0)
        ]
        if not unfilled:
            return

        orphans = sorted(name for name in supplied
                         if name not in dict(self.model.named_parameters()))
        detail = ""
        if orphans:
            detail = (
                f" The checkpoint also carries {len(orphans)} tensors the model "
                f"has no parameter for (e.g. {orphans[0]!r}), so the module "
                "layout does not match the checkpoint's -- typical when a "
                "quantized format needs modules this build did not create."
            )
        raise RuntimeError(
            f"{len(unfilled)} parameters would never receive weights "
            f"(e.g. {unfilled[0]!r}).{detail} Refusing to run: the model would "
            "generate confident nonsense rather than fail."
        )

    def _move_buffers(self) -> None:
        for name, buffer in list(self.model.named_buffers()):
            if buffer.is_meta:
                continue
            module, attr = _resolve(self.model, name)
            setattr(module, attr, buffer.to(self.device))

    def _install_hooks(self) -> None:
        layers = find_decoder_layers(self.model)
        if layers is None:
            raise RuntimeError("could not locate decoder layers in this model")
        if len(layers) != self.manifest.num_layers:
            raise RuntimeError(
                f"model has {len(layers)} layers but the checkpoint describes "
                f"{self.manifest.num_layers}"
            )

        # Swap every streamed parameter from meta to a real -- but empty --
        # device tensor now. Once the parameter lives on the right device, the
        # per-step swaps in the hooks are same-device `.data` assignments,
        # which cost nothing and allocate nothing. Doing it the other way
        # round would mean constructing a fresh Parameter for every tensor of
        # every layer on every token.
        # One placeholder per dtype, not one overall. A quantized checkpoint
        # mixes dtypes within a layer -- packed weights are int32 while their
        # scales are floating point -- and handing an int32 parameter a
        # float placeholder makes the quantized module reject it ("Expected
        # torch.int32 but got torch.bfloat16") the moment it inspects one.
        self._empties: dict[torch.dtype, torch.Tensor] = {}
        self._column_groups: dict[int, torch.Tensor] = {}
        self._quant_plan = self._build_quant_plan()

        for chunk in self.manifest.chunks.values():
            for spec in chunk.tensors:
                if spec.name in self._compressed_tensor_names:
                    continue  # consumed by decompression, not assigned directly
                module, attr = _resolve(self.model, spec.name)
                self._resolved[spec.name] = (module, attr)
                _assign(module, attr, self._empty_for(spec.dtype))
        self._empty = self._empty_for(self.dtype)

        # Quantized weights arrive packed, so the parameter they decompress
        # into starts as a placeholder of the compute dtype.
        for entries in self._quant_plan.values():
            for module, _scheme, _tensors in entries:
                _assign(module, "weight", self._empty_for(self.dtype))

        # Precompute each layer's parameter objects and slice geometry. The
        # hooks run on every layer of every token, so the dictionary lookups
        # and getattr chains they would otherwise repeat are hoisted here --
        # the parameter objects are stable once assigned above, so they can be
        # held directly.
        self._layer_plan: dict[str, list[tuple]] = {}
        for key, chunk in self.manifest.chunks.items():
            entries = []
            for spec in chunk.tensors:
                if spec.name not in self._resolved:
                    continue  # decompressed rather than assigned
                module, attr = self._resolved[spec.name]
                entries.append(
                    (
                        getattr(module, attr),
                        spec.offset,
                        spec.offset + spec.nbytes,
                        spec.dtype,
                        spec.shape,
                        self._empty_for(spec.dtype),
                    )
                )
            self._layer_plan[key] = entries

        for index, layer in enumerate(layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._make_pre_hook(index))
            )
            self._handles.append(
                layer.register_forward_hook(self._make_post_hook(index))
            )

    # -- streaming hooks -----------------------------------------------

    def _make_pre_hook(self, index: int):
        key = self.plan.key(index)
        plan = self._layer_plan[key]
        quant = self._quant_plan.get(key, ())
        lookahead = self.plan.lookahead(index, self.config.prefetch_depth)
        dtype = self.dtype

        def pre_hook(module, args):
            # Issue the lookahead before blocking on this layer, so the next
            # transfers are already moving while this one is waited on. The
            # order is the whole mechanism -- reversed, every transfer would
            # start only after the previous layer had finished computing.
            self.scheduler.hint(lookahead)
            flat = self.store.get(key)
            for param, start, end, dtype, shape, _ in plan:
                param.data = flat[start:end].view(dtype).view(shape)
            for module, scheme, tensors in quant:
                # Reconstruct this module's weight from its packed form. The
                # cache and the bus only ever see the packed bytes; the
                # expansion happens here, on the device, for one layer at a
                # time and is discarded when the layer is done.
                compressed = {
                    local: spec.view_from(flat) for local, spec in tensors.items()
                }
                module.weight.data = decompress_weight(
                    compressed, scheme, dtype, self._column_group(module, compressed,
                                                                 scheme)
                )
            self._live[index] = flat
            return None

        return pre_hook

    def _make_post_hook(self, index: int):
        key = self.plan.key(index)
        plan = self._layer_plan[key]
        quant = self._quant_plan.get(key, ())
        empty = self._empty_for(self.dtype)

        def post_hook(module, args, output):
            for quantized, _scheme, _tensors in quant:
                quantized.weight.data = empty
            for entry in plan:
                # Repointing at a same-dtype empty tensor drops this layer's
                # only reference to the buffer, so VRAM is reclaimed unless
                # the store decided to keep the chunk resident.
                entry[0].data = entry[5]
            self._live.pop(index, None)
            return output

        return post_hook

    # -- use -----------------------------------------------------------

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 64, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @torch.inference_mode()
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids.to(self.device)).logits

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.scheduler.shutdown()
        self.store.clear_hot()

    def __enter__(self) -> "StreamingModel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
