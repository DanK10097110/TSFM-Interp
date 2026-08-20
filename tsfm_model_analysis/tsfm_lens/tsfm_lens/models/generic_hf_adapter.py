"""A zero-code attempt at any Hugging Face checkpoint (ROADMAP.md sec 16 E3(b)).

Every other adapter in this package is a hand-written statement of four
things: how to feed the model, which modules carry the residual stream, what
time interval each token covers, and how to get a forecast out. This adapter
*probes* for all four instead, so a checkpoint nobody has written an adapter
for can be attempted immediately -- and, when a probe cannot resolve, refuses
with a message naming exactly what it tried rather than guessing.

The design rule throughout is that a resolved strategy is **recorded, not
assumed**: `describe_strategies()` returns which path was taken at each of the
four seams, `run.py --probe-adapter` prints it, and anything unresolvable is
an error at the seam that needed it rather than a plausible-looking number
downstream (`CLAUDE.md` sec 2.5). Two consequences worth stating up front,
because both are places a generic path can quietly produce a wrong result:

- **Covariates are never fabricated.** A checkpoint whose forward pass
  requires `past_time_features` (TimeSeriesTransformer, Informer, Autoformer)
  is refused at load, because a zeros tensor is a forecast-changing input, not
  a neutral default. Write an adapter for those.
- **A point-only forecast head yields degenerate quantiles**, which would make
  L0's calibration section report a perfect-looking reliability curve that
  means nothing. This is logged as a warning and recorded in
  `describe_strategies()["quantiles"]`.

Span discovery is `extraction/span_discovery.py` (E3(a)), run lazily on first
use and cached; a checkpoint whose impulse response is not time-localized gets
`time_localization()["localized"] = False` and a `token_time_spans()` that
raises, which is E3(c)'s refusal signal at the adapter seam. Routing the
pipeline to L0-only on that signal is not built here.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from ..utils import log
from .base import (ModelAdapter, NotTimeLocalized, _scan_attention, _scan_mlp,
                   random_init_like)

# Ordered by how unambiguous each name is. `inputs_embeds` is deliberately
# absent even though nearly every HF forward accepts it: it takes an
# already-embedded `[B, T, D]` tensor, so feeding it a raw series skips the
# model's own tokenizer/patch embedder. It looks like a valid input name and
# is never the right one -- Timer picks it over `input_ids` if it is listed,
# and then fails with a matmul shape error rather than a legible one.
# `input_ids` is last and separate:
# it is a *text* convention that several time-series checkpoints (TimeMoE,
# Timer, and the thuml lineage generally) reuse for a float series, so it must
# be reachable -- but a text model reaching it would be fed floats, which is
# why choosing it is warned about rather than silent.
_INPUT_KWARGS = ("past_values", "context", "input_values", "x", "input_ids")

# Only masks defined on the *raw series* axis, where all-ones is genuinely
# neutral. `attention_mask` is excluded despite being the most common name in
# HF: on a patch-tokenizing model it is a mask over *tokens*, so an all-ones
# tensor shaped like the series is the wrong length -- Timer fails with an
# unpack error rather than anything legible, and a model that broadcast it
# instead would be silently mis-masked.
_SYNTHESIZABLE = ("past_observed_mask", "observed_mask")

# `logits` is last for the same reason: it is a regression head's output on a
# forecasting model and a vocabulary distribution on a text model. The shape
# checks in `_try_forward_field` separate them -- a `[B, T, V]` logit tensor
# is not `[B, H]` -- but the ordering means a purpose-named field always wins.
_FORECAST_FIELDS = ("prediction_outputs", "prediction_logits", "predictions",
                    "sequences", "forecast", "prediction", "logits")


class GenericHFAdapter(ModelAdapter):
    """Probe-and-record adapter for checkpoints with no hand-written support."""

    measures_own_spans = True

    default_layer_regex = r"^$"

    def load(self) -> None:
        """Load by checkpoint id, then resolve the input and layer strategies.

        Both are resolved here rather than lazily so an unsupportable
        checkpoint fails at load with an actionable message, instead of part
        way through an extraction that has already spent minutes.
        """
        self._model = self._instantiate()
        if self.cfg.random_init:
            log.warning("generic_hf '%s': random_init=True -- discarding pretrained "
                        "weights (ROADMAP.md sec 16 E9)", self.name)
            self._model = random_init_like(self._model)
        self._model = self._model.to(self.device).eval()
        self._spans: Optional[np.ndarray] = None
        self._discovery = None
        self._forecast_strategy: Optional[str] = None
        self._input_kwarg, self._aux_kwargs = self._resolve_input()
        self._input_rank = self._resolve_input_rank()
        if not self.cfg.layer_regex:
            self.default_layer_regex = self._discover_layer_regex()
        self._warming_spans = False

    def ensure_loaded(self) -> None:
        """Load, then measure spans before any caller can install a hook.

        `token_slice` -- and so every `token_patch` hook -- asks for
        `token_time_spans`, and discovery answers by running its own forward
        pass. If the first such request arrives while a patching hook is
        already installed, that inner forward re-enters the same hook and
        recurses until the stack ends. It is not hypothetical: the lens stage
        hits it whenever `run.keep_models_loaded: false` has released and
        reloaded the model, clearing this cache. Discovery is one forward pass
        and every analysis path needs the spans anyway, so it is done here,
        eagerly, alongside the strategies `load` already resolves up front.

        This has to sit after `super().ensure_loaded()` rather than inside
        `load`, because `_loaded` is only set once `load` returns -- warming
        from inside it sends discovery's own `ensure_loaded` back into `load`
        for an unbounded reload loop. `_warming_spans` guards the same
        re-entry through this method.

        A checkpoint that is not time-localized still raises from
        `token_time_spans` at its normal call site, not from here: the refusal
        stays where E3(c) put it, and the model stays loadable for the
        behavioral (L0) comparison that refusal explicitly still permits.
        """
        super().ensure_loaded()
        if self._spans is None and not getattr(self, "_warming_spans", False):
            self._warming_spans = True
            try:
                self.token_time_spans()
            except ValueError:
                pass
            finally:
                self._warming_spans = False

    def _instantiate(self) -> nn.Module:
        """Construct the checkpoint. The one seam a test replaces, so every
        probe below is exercised without a network round trip.

        `kwargs.auto_class` names the `transformers` class to build with, and
        exists because `AutoModel` resolves to the *bare backbone* for most
        time-series architectures -- `PatchTSTModel`, not
        `PatchTSTForPrediction` -- which has no forecast head at all and so
        cannot produce L0. Naming the class is a config line rather than
        adapter code, so the zero-code claim survives; guessing at a
        `ForPrediction` suffix would not, since the naming is not a rule.
        """
        import transformers
        if not self.cfg.checkpoint:
            raise ValueError(
                f"model '{self.name}': adapter 'generic_hf' needs an explicit "
                f"`checkpoint:` -- there is no default to guess at")
        cls_name = self.cfg.kwargs.get("auto_class", "AutoModel")
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            raise ValueError(
                f"model '{self.name}': transformers has no class '{cls_name}' "
                f"(from `kwargs.auto_class`)")
        model, info = cls.from_pretrained(
            self.cfg.checkpoint, trust_remote_code=True, torch_dtype=self.dtype,
            output_loading_info=True)
        missing = [k for k in info.get("missing_keys", [])
                   if not k.endswith("num_batches_tracked")]
        if missing and not (self.cfg.random_init
                            or self.cfg.kwargs.get("allow_uninitialized")):
            # The most dangerous thing a zero-code path can do quietly. Asking
            # for a `*ForPrediction` class against a backbone-only pretrain
            # checkpoint loads, warns on stderr, and returns a model whose
            # head -- often whose whole stack -- is random. Every number
            # downstream is then real-looking output from an untrained model,
            # which is exactly `ModelConfig.random_init`'s deliberate null
            # arrived at by accident.
            raise ValueError(
                f"model '{self.name}' ({self.cfg.checkpoint}) loaded as {cls_name} with "
                f"{len(missing)} randomly-initialized parameters "
                f"(e.g. {missing[:3]}); this checkpoint does not contain that class's "
                f"weights. Pick the class the checkpoint was saved as, or set "
                f"`kwargs.allow_uninitialized: true` if a partly-random model is "
                f"intended (see `random_init` for the deliberate null).")
        return model

    @property
    def module(self) -> nn.Module:
        return self._model

    def _release(self) -> None:
        self._model = None
        self._spans = None
        self._discovery = None

    def describe_strategies(self) -> dict:
        """What resolved at each of the four probed seams, for the run record.

        Written as a plain dict rather than logged only, because "which path
        did this checkpoint actually take" is the one question a reader of a
        generic-adapter run cannot answer from the numbers.
        """
        loc = self.time_localization()
        return {
            "checkpoint": self.cfg.checkpoint,
            "input_kwarg": getattr(self, "_input_kwarg", None),
            "input_rank": getattr(self, "_input_rank", None),
            "aux_kwargs": list(getattr(self, "_aux_kwargs", ())),
            "layer_regex": self.cfg.layer_regex or self.default_layer_regex,
            "n_capture_layers": len(self.layer_names()) if self._loaded else None,
            "forecast": self._forecast_strategy,
            "quantiles": ("degenerate_point"
                          if self._forecast_strategy == "forward_field"
                          else "sampled" if self._forecast_strategy else None),
            "time_localization": loc,
        }

    def _forward_signature(self) -> dict:
        return inspect.signature(type(self._model).forward).parameters

    def _resolve_input(self) -> tuple:
        """Pick the float-input parameter, and refuse rather than fabricate covariates.

        A required parameter this adapter cannot synthesize is a hard stop:
        `past_time_features` carries the calendar covariates a seasonal model
        conditions on, so passing zeros produces a forecast for a series that
        does not exist -- a silently wrong number, which is worse than no
        number (`CLAUDE.md` sec 2.5).
        """
        params = self._forward_signature()
        chosen = next((k for k in _INPUT_KWARGS if k in params), None)
        if chosen is None:
            raise ValueError(
                f"model '{self.name}' ({self.cfg.checkpoint}): forward() accepts none of "
                f"{list(_INPUT_KWARGS)}; it takes {sorted(k for k in params if k != 'self')}. "
                f"This checkpoint needs a hand-written adapter.")
        required = [k for k, p in params.items()
                    if k not in ("self", chosen)
                    and p.default is inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        unsupported = [k for k in required if k not in _SYNTHESIZABLE]
        if unsupported:
            raise ValueError(
                f"model '{self.name}' ({self.cfg.checkpoint}): forward() requires "
                f"{unsupported}, which this adapter will not fabricate -- a synthesized "
                f"covariate changes the forecast rather than leaving it neutral. "
                f"Write an adapter for this checkpoint.")
        if chosen == "input_ids":
            log.warning("generic_hf '%s': feeding a float series to 'input_ids' -- correct "
                        "for time-series checkpoints that reuse the text convention "
                        "(TimeMoE, Timer), wrong for an actual language model. Check "
                        "--probe-adapter output before trusting any number.", self.name)
        self._no_cache = "use_cache" in params
        aux = tuple(k for k in _SYNTHESIZABLE if k in params)
        log.info("generic_hf '%s': input kwarg '%s', synthesizable aux %s",
                 self.name, chosen, list(aux))
        return chosen, aux

    def _resolve_input_rank(self) -> int:
        """Whether this model wants `[B, T]` or `[B, T, 1]`, measured by trying both.

        Univariate-native checkpoints (Chronos-like) take a flat `[B, T]`;
        channel-aware ones (PatchTST, PatchTSMixer and most HF time-series
        classes) take `[B, T, n_channels]` and this repo is univariate by
        decision (`CLAUDE.md` sec 12), so the channel axis is a singleton.
        Nothing in a config or signature distinguishes the two reliably, and a
        wrong guess is not a silent failure -- it is a shape error -- so
        trying is both cheap and safe here, unlike the covariate case above.
        """
        errors = {}
        for rank in (2, 3):
            self._input_rank = rank
            try:
                with torch.no_grad():
                    self.forward(self.prepare(_dummy_context(self.data_cfg.context_len)))
                log.info("generic_hf '%s': input rank %d", self.name, rank)
                return rank
            except Exception as exc:
                errors[rank] = f"{type(exc).__name__}: {exc}"
        # Both errors, not just the last: they are usually different failures
        # (a channel-count mismatch vs. a rank mismatch) and reporting one
        # sends the reader after the wrong cause.
        raise ValueError(
            f"model '{self.name}' ({self.cfg.checkpoint}): forward() rejected both input "
            f"ranks for '{self._input_kwarg}'.\n  [B, T]    -> {errors[2]}\n"
            f"  [B, T, 1] -> {errors[3]}")

    def _discover_layer_regex(self) -> str:
        """Find the repeated `[B, T, D]` block stack by measuring output shapes.

        Groups module names by their digit-free skeleton and keeps the largest
        group whose members all emit one consistent `[B, T, D]` shape. The
        regex is anchored (`CLAUDE.md` sec 11.30): an unanchored one silently
        swallows a second stack whose name ends with the first's.

        Two tie-breaks, in order, and the second is the one that matters. A
        block's own *components* -- its MLP output projection, its attention
        output, a residual-width Linear -- emit the identical `[B, T, D]`
        shape at the identical member count as the block containing them, so
        shape alone cannot separate `layers.#` from `layers.#.lin` and picking
        either is a coin flip. Capturing a component instead of the block
        would silently put a partial computation on the residual axis that
        every downstream analysis reads as the residual stream. So among tied
        candidates, prefer the **shallowest** path: the residual stream is the
        block's own output, and a component is by construction nested below
        it. Width matching against the config's declared hidden size is tried
        first, since it separates a real residual stack from an equally
        repeated non-residual one.
        """
        shapes: dict = {}
        handles = [mod.register_forward_hook(
            lambda m, i, o, n=name: shapes.__setitem__(n, _out_shape(o)))
            for name, mod in self._model.named_modules() if name]
        try:
            with torch.no_grad():
                self.forward(self.prepare(_dummy_context(self.data_cfg.context_len)))
        finally:
            for h in handles:
                h.remove()
        groups: dict = {}
        for name, shape in shapes.items():
            if shape is None or len(shape) != 3 or shape[1] < 2 or shape[2] < 2:
                continue
            groups.setdefault(_skeleton(name), []).append((name, shape))
        width = getattr(self._model.config, "hidden_size",
                        getattr(self._model.config, "d_model", None))
        best, best_key = None, None
        for key, members in groups.items():
            uniform = {s[1:] for _, s in members}
            if len(members) < 2 or len(uniform) != 1:
                continue
            score = (len(members), 1 if members[0][1][2] == width else 0,
                     -key.count("."))
            if best is None or score > best:
                best, best_key = score, key
        if best_key is None:
            raise ValueError(
                f"model '{self.name}' ({self.cfg.checkpoint}): no repeated module group "
                f"emitting a consistent [B, T, D] output was found among "
                f"{len(shapes)} modules; set `layer_regex:` explicitly after "
                f"inspecting --discover-layers.")
        regex = "^" + r"\.".join(r"\d+" if p == "#" else re.escape(p)
                                 for p in best_key.split(".")) + "$"
        log.info("generic_hf '%s': discovered layer regex %s (%d blocks, width %s)",
                 self.name, regex, best[0], groups[best_key][0][1][2])
        return regex

    def prepare(self, contexts: np.ndarray) -> Any:
        """Model-ready kwargs: the resolved float input plus any all-ones masks."""
        x = torch.from_numpy(np.ascontiguousarray(contexts)).to(
            self.device, dtype=self.dtype)
        if getattr(self, "_input_rank", 2) == 3:
            x = x[:, :, None]
        prepared = {self._input_kwarg: x}
        for k in self._aux_kwargs:
            prepared[k] = torch.ones_like(x)
        if getattr(self, "_no_cache", False):
            # Not a preference. Several time-series checkpoints ship remote
            # code written against a pre-4.41 `DynamicCache` and crash on
            # `get_usable_length` with caching on -- `CLAUDE.md` sec 11.22
            # records this for Sundial, and it reproduces identically on
            # TimeMoE and Timer, so it is a lineage-wide break rather than one
            # checkpoint's bug. Disabling a cache cannot change a forecast,
            # which is what separates this from the covariates above.
            prepared["use_cache"] = False
        return prepared

    def forward(self, prepared: Any) -> None:
        with torch.no_grad():
            self._model(**prepared)

    @property
    def _min_contrast(self) -> float:
        """The peak:pedestal floor below which this model is refused.

        Read from `is_time_localized`'s own default signature rather than
        restated here, so the number the refusal message quotes is always the
        number the gate actually applied (`CLAUDE.md` sec 11.33's statistic
        lives in one place).
        """
        import inspect as _inspect
        from ..extraction.span_discovery import SpanDiscovery
        default = _inspect.signature(SpanDiscovery.is_time_localized).parameters["min_contrast"]
        return float(self.cfg.kwargs.get("min_contrast", default.default))

    @property
    def _min_contiguity(self) -> float:
        """The share of non-empty tokens that must read one interval, or refusal.

        Same single-source-of-truth arrangement as `_min_contrast` above, for
        the second, orthogonal gate (ROADMAP.md sec 19 G2): a model can be
        sharply time-localized and still read a *set of lags* per token, which
        contrast cannot see because each of those lags is a clean peak.
        """
        import inspect as _inspect
        from ..extraction.span_discovery import SpanDiscovery
        default = _inspect.signature(SpanDiscovery.is_contiguous).parameters["min_contiguity"]
        return float(self.cfg.kwargs.get("min_contiguity", default.default))

    def token_time_spans(self) -> np.ndarray:
        """Discovered spans, measured once and cached; raises when not time-localized.

        This is E3(c)'s refusal at the adapter seam: a model whose impulse
        response is diffuse has no contiguous token->time map, and handing the
        pooling matrix an invented one would put every cross-model number on a
        fiction (`CLAUDE.md` sec 6.3).
        """
        if self._spans is None:
            from ..extraction.span_discovery import discover_spans
            self._discovery = discover_spans(self, context_len=self.data_cfg.context_len)
            reason = self._discovery.refusal_reason(self._min_contrast, self._min_contiguity)
            if reason is not None:
                raise NotTimeLocalized(
                    f"model '{self.name}' ({self.cfg.checkpoint}): {reason}, so window "
                    f"pooling -- and every cross-model analysis built on it -- is not "
                    f"defined for this model. It can still be compared behaviorally (L0).",
                    model=self.name, checkpoint=self.cfg.checkpoint,
                    contrast=float(self._discovery.contrast),
                    min_contrast=float(self._min_contrast),
                    diffuseness=float(self._discovery.diffuseness),
                    contiguity=float(self._discovery.contiguity),
                    min_contiguity=float(self._min_contiguity))
            self._spans = self._discovery.spans
        return self._spans

    def time_localization(self) -> Optional[dict]:
        """The measured refusal signal, or None before anything has been measured.

        Deliberately safe on an unloaded adapter: this is the accessor the
        pipeline's routing (E3(c)) reads to decide what a model is eligible
        for, and a query that has to load a checkpoint to answer "not measured
        yet" would make that decision expensive enough to skip.
        """
        d = getattr(self, "_discovery", None)
        if d is None:
            return None
        return {"localized": d.refusal_reason(self._min_contrast,
                                              self._min_contiguity) is None,
                "min_contrast": self._min_contrast, "contrast": float(d.contrast),
                "diffuseness": float(d.diffuseness), "n_tokens": int(d.n_tokens),
                "flagged_tokens": len(d.flagged_tokens),
                "contiguity": float(d.contiguity),
                "min_contiguity": self._min_contiguity,
                "noncontiguous_tokens": len(d.noncontiguous_tokens),
                "empty_tokens": len(d.empty_tokens),
                "per_amplitude_agreement": float(d.per_amplitude_agreement)}

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Pass captures through until spans exist, then assert against them.

        Discovery itself runs before any span is known, so the base
        implementation's token-count assertion cannot apply yet -- but once a
        span table has been measured, a capture that disagrees with it is the
        same silent-misalignment failure the base guards, and stays loud.
        """
        if self._spans is None:
            return hidden
        return super().postprocess_tokens(layer_name, hidden)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Forecast via the first forecast strategy that resolves, recorded once.

        Two strategies, in order: a `generate()` that returns sampled
        `sequences` (real quantiles), then a forward pass carrying a named
        prediction field (a point forecast only). Neither resolving is an
        error, not an empty forecast.
        """
        prepared = self.prepare(contexts)
        samples = self._try_generate(prepared, horizon)
        if samples is not None:
            self._forecast_strategy = "generate"
            point = np.median(samples, axis=1)
            q = np.stack([np.quantile(samples, lv, axis=1) for lv in quantiles], axis=-1)
            return {"point": point, "quantiles": q}
        point = self._try_forward_field(prepared, horizon)
        if point is None:
            raise ValueError(
                f"model '{self.name}' ({self.cfg.checkpoint}): no forecast path resolved -- "
                f"generate() produced no sampled `sequences`, and forward() returned no "
                f"field in {list(_FORECAST_FIELDS)} shaped to the horizon. Capture and "
                f"alignment still work; L0 needs a hand-written predict().")
        if self._forecast_strategy != "forward_field":
            self._forecast_strategy = "forward_field"
            log.warning("generic_hf '%s': point-only forecast head -- quantiles are "
                        "degenerate (every level equals the point forecast). Read L0's "
                        "calibration section for this model as unmeasured, not perfect.",
                        self.name)
        return {"point": point,
                "quantiles": np.repeat(point[:, :, None], len(quantiles), axis=2)}

    def _try_generate(self, prepared: dict, horizon: int) -> Optional[np.ndarray]:
        """Sampled `[B, n_samples, H]` from a generation head, or None."""
        if not hasattr(self._model, "generate"):
            return None
        try:
            with torch.no_grad():
                out = self._model.generate(**prepared)
        except Exception as exc:
            log.info("generic_hf '%s': generate() unavailable (%s)", self.name, exc)
            return None
        seq = getattr(out, "sequences", None)
        if seq is None or seq.ndim != 3:
            return None
        return _fit_horizon(seq.float().cpu().numpy(), horizon, self.name, axis=2)

    def _try_forward_field(self, prepared: dict, horizon: int) -> Optional[np.ndarray]:
        """Point forecast `[B, H]` from a named forward output field, or None."""
        with torch.no_grad():
            out = self._model(**prepared)
        for field in _FORECAST_FIELDS:
            val = out.get(field) if hasattr(out, "get") else getattr(out, field, None)
            if not isinstance(val, torch.Tensor) or val.ndim < 2:
                continue
            arr = val.float().cpu().numpy()
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            if arr.ndim != 2:
                continue
            log.info("generic_hf '%s': forecast read from forward field '%s'",
                     self.name, field)
            return _fit_horizon(arr, horizon, self.name, axis=1)
        return None

    def attention_info(self) -> Optional[list]:
        """Best-effort head map via the shared block scan, None if any block fails."""
        self.ensure_loaded()
        infos = []
        for block in self.all_layer_names():
            info = _scan_attention(self.module, block)
            if info is None:
                log.info("generic_hf '%s': no head map for block %s; head-level "
                         "analyses disabled", self.name, block)
                return None
            infos.append(info)
        return infos

    def mlp_info(self) -> Optional[dict]:
        """Best-effort block -> pre-residual MLP map via the shared block scan."""
        self.ensure_loaded()
        mapping = {}
        for block in self.all_layer_names():
            name = _scan_mlp(self.module, block)
            if name is None:
                log.info("generic_hf '%s': no MLP module for block %s; MLP ablation "
                         "disabled", self.name, block)
                return None
            mapping[block] = name
        return mapping


def _out_shape(output: Any) -> Optional[tuple]:
    """Shape of a module's primary output tensor, or None when it has none.

    Mirrors `extraction/hooks.py::_primary`'s tensor-first rule so shape
    probing and capture agree about what a block's output is -- including for
    HF `ModelOutput` dataclasses, which are indexable but are not tuples
    (`CLAUDE.md` sec 11.21).
    """
    if isinstance(output, torch.Tensor):
        return tuple(output.shape)
    try:
        first = output[0]
    except Exception:
        return None
    return tuple(first.shape) if isinstance(first, torch.Tensor) else None


def _skeleton(name: str) -> str:
    """A module name with every all-digit path segment replaced by `#`."""
    return ".".join("#" if p.isdigit() else p for p in name.split("."))


def _dummy_context(context_len: int) -> np.ndarray:
    """A two-row probe batch for shape discovery only; values are irrelevant."""
    t = np.arange(context_len, dtype=np.float32)
    return np.stack([np.sin(2 * np.pi * t / 32.0), np.cos(2 * np.pi * t / 32.0)])


def _fit_horizon(arr: np.ndarray, horizon: int, name: str, axis: int) -> np.ndarray:
    """Trim a forecast to the requested horizon, refusing when it is too short."""
    have = arr.shape[axis]
    if have < horizon:
        raise ValueError(
            f"model '{name}': forecast head produces {have} steps but the run asks for "
            f"{horizon}; lower `data.horizon` or write an adapter that rolls this head "
            f"forward -- padding it would invent values.")
    return arr[:, :horizon] if axis == 1 else arr[:, :, :horizon]
