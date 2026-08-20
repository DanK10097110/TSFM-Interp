"""The architecture-agnostic model contract.

Every model enters the pipeline through a `ModelAdapter`, which owns four
responsibilities: loading, running a forward pass that fires capture hooks,
mapping internal tokens to time spans (the key to cross-architecture
alignment), and producing forecasts. New models plug in by subclassing this
and registering in `models/__init__.py`; nothing downstream changes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from ..config import DataConfig, ModelConfig
from ..utils import log


class CapabilityUnavailable(NotImplementedError):
    """A model was asked for something its capability tier does not provide.

    `ROADMAP.md` sec 19 G1. Before tiers, `forward`/`prepare`/`module`/
    `token_time_spans` were `@abstractmethod`, so a black-box model -- one that
    can only forecast, which is every hosted or API-only model and plenty of
    local ones -- could not be expressed at all: you wrote the whole adapter or
    got nothing, even though L0, calibration, horizon metrics and cost need
    none of it. They are now concrete methods raising this, and the pipeline
    routes around a model that raises it rather than crashing on one.

    Distinct from `NotTimeLocalized` on purpose: that says a model HAS internals
    the pipeline could read and its token->time map is unusable, which is a
    measured verdict about the model. This says the adapter never claimed to
    expose them, which is a fact about the adapter. Conflating them would let a
    missing implementation read as a finding about a model.
    """

    def __init__(self, message: str, *, model: str, capability: str,
                 tier_actual: Optional[int] = None,
                 tier_required: Optional[int] = None):
        super().__init__(message)
        self.model = model
        self.capability = capability
        self.tier_actual = tier_actual
        self.tier_required = tier_required

    def as_record(self) -> dict:
        return {"reason": str(self), "model": self.model,
                "capability": self.capability, "tier_actual": self.tier_actual,
                "tier_required": self.tier_required}


TIER_NAMES = {0: "black box", 1: "observable", 2: "steerable", 3: "decomposable"}


class NotTimeLocalized(ValueError):
    """A model whose tokens do not each read one contiguous time interval.

    Raised by `token_time_spans()` when an adapter has *measured* its own
    impulse response and found it diffuse, which makes window pooling --
    and therefore every cross-model analysis in this repo -- undefined for
    that model (`CLAUDE.md` sec 12's envelope edge, made machine-checkable
    by ROADMAP.md sec 16 E3).

    A distinct type, not a bare ValueError, because the pipeline *catches*
    this one to route the model to L0-only (E3(c)); catching ValueError
    there would swallow unrelated bugs and turn a real defect into a
    plausible-looking "this model is diffuse" verdict. It subclasses
    ValueError so that existing call sites that guard against a bad span
    table keep behaving exactly as before.
    """

    def __init__(self, message: str, *, model: str, checkpoint: str = "",
                 contrast: Optional[float] = None, min_contrast: Optional[float] = None,
                 diffuseness: Optional[float] = None,
                 contiguity: Optional[float] = None,
                 min_contiguity: Optional[float] = None):
        super().__init__(message)
        self.model = model
        self.checkpoint = checkpoint
        self.contrast = contrast
        self.min_contrast = min_contrast
        self.diffuseness = diffuseness
        self.contiguity = contiguity
        self.min_contiguity = min_contiguity

    def as_record(self) -> dict:
        """The measured facts behind the refusal, for `routing.json`."""
        return {"model": self.model, "checkpoint": self.checkpoint,
                "contrast": self.contrast, "min_contrast": self.min_contrast,
                "diffuseness": self.diffuseness, "contiguity": self.contiguity,
                "min_contiguity": self.min_contiguity, "reason": str(self)}


class ModelAdapter(ABC):
    """Wraps one model behind a uniform capture/align/predict interface."""

    default_layer_regex: str = r".*"

    # True only for adapters that DERIVE their token->time map by measuring
    # an impulse response (`GenericHFAdapter`). Those are the only ones that
    # can refuse themselves, so they are the only ones the pipeline's routing
    # preflight probes -- probing a declared-spans adapter would load a
    # checkpoint to learn nothing, and would hold every model resident at
    # once, which extraction deliberately avoids for VRAM.
    measures_own_spans: bool = False

    def __init__(self, cfg: ModelConfig, data_cfg: DataConfig,
                 device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.device = device
        self.dtype = dtype
        self._loaded = False

    @property
    def name(self) -> str:
        return self.cfg.name

    @abstractmethod
    def load(self) -> None:
        """Instantiate the underlying model on the target device."""

    # ---- tier 1 (observable). Concrete, not abstract, since ROADMAP.md sec 19
    # G1: a tier-0 adapter implements `load` and `predict` and nothing else,
    # and must be constructible. Each raises `CapabilityUnavailable`, which the
    # pipeline routes around; overriding them all is what makes an adapter
    # tier 1. Keeping them abstract would make "black box" unrepresentable,
    # which was the actual barrier -- not any analysis code downstream.

    @property
    def module(self) -> nn.Module:
        """The torch module hooks attach to."""
        raise self._unavailable("module", 1)

    def prepare(self, contexts: np.ndarray) -> Any:
        """Convert raw contexts [B, context_len] into model-ready inputs."""
        raise self._unavailable("prepare", 1)

    def forward(self, prepared: Any) -> None:
        """Run one forward pass over prepared inputs so capture hooks fire."""
        raise self._unavailable("forward", 1)

    def token_time_spans(self) -> np.ndarray:
        """Per-token (start, end) time coverage in context steps, shape [n_tokens, 2].

        This is the single piece of information that makes representations
        from patch-based and per-step tokenizations comparable.
        """
        raise self._unavailable("token_time_spans", 1)

    def _unavailable(self, capability: str, tier_required: int) -> CapabilityUnavailable:
        return CapabilityUnavailable(
            f"model '{self.name}' is tier {self.capability_tier()} "
            f"({TIER_NAMES[self.capability_tier()]}) and does not implement "
            f"`{capability}`, which requires tier {tier_required} "
            f"({TIER_NAMES[tier_required]})",
            model=self.name, capability=capability,
            tier_actual=self.capability_tier(), tier_required=tier_required)

    @abstractmethod
    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Forecast, returning {'point': [B, H], 'quantiles': [B, H, Q]}."""

    # Tier 2 ("steerable") means the whole context is processed in ONE forward
    # pass, so a cached clean state can be written back into a corrupted run at
    # a fixed token position (`CLAUDE.md` sec 12's envelope edge, and what
    # `hooks.token_patch` assumes). An adapter that chunks or re-enters its
    # context sets this False and drops to tier 1: its activations are readable
    # but not patchable. Every adapter written so far is single-pass, which is
    # why the default is True and why this is a declaration rather than a probe
    # -- nothing in this repo has ever exercised the False branch on a real
    # checkpoint, and saying so is more honest than a heuristic that has never
    # been tested against a counterexample.
    single_pass_context: bool = True

    @classmethod
    def _overrides(cls, *names: str) -> bool:
        """True when this subclass provides its own version of every `name`."""
        return all(getattr(cls, n, None) is not getattr(ModelAdapter, n, None)
                   for n in names)

    @classmethod
    def capability_tier(cls) -> int:
        """This adapter's declared tier, 0-3 (`ROADMAP.md` sec 19 G1).

        Derived from what the subclass actually implements rather than from a
        hand-set integer, because a hand-set one is a claim checked nowhere --
        `CLAUDE.md` sec 11.34's lesson about declarations, applied to the
        contract itself. `models/conformance.py` cross-checks the derived tier
        against what each stage then asks for, and `capability_matrix.py`
        already covers the separate question of whether a declared capability
        actually WORKS on a live checkpoint. Tier says what the adapter
        offers; the matrix says whether it delivers.

        Tier 3 deliberately does NOT require `mlp_info`. TimesFM's feed-forward
        block is two bare `nn.Linear`s with no wrapping module to name, so it
        has no MLP ablation and full attention-head ablation and pattern
        analysis (`CLAUDE.md` sec 6.2) -- requiring `mlp_info` would demote the
        model the attention taxonomy works best on.
        """
        if not cls._overrides("module", "prepare", "forward", "token_time_spans"):
            return 0
        if not cls.single_pass_context:
            return 1
        if cls._overrides("attention_info", "attention_patterns"):
            return 3
        return 2

    @classmethod
    def tier_report(cls) -> dict:
        """The tier plus what each higher tier would additionally require."""
        tier = cls.capability_tier()
        missing = {}
        if tier < 1:
            missing[1] = [n for n in ("module", "prepare", "forward", "token_time_spans")
                          if not cls._overrides(n)]
        if tier < 2:
            missing[2] = (["single_pass_context"] if tier == 1 and not cls.single_pass_context
                          else ["tier 1 first"])
        if tier < 3:
            missing[3] = ([n for n in ("attention_info", "attention_patterns")
                           if not cls._overrides(n)] if tier == 2 else ["tier 2 first"])
        return {"tier": tier, "name": TIER_NAMES[tier],
                "adapter": cls.__name__, "missing": missing}

    def ensure_loaded(self) -> None:
        """Load lazily on first use."""
        if not self._loaded:
            log.info("loading model '%s' (%s)", self.name, self.cfg.adapter)
            self.load()
            # A tier-0 adapter has no `module` to put in eval mode. Guarded on
            # the declared tier rather than on catching CapabilityUnavailable,
            # so a tier-1+ adapter whose `module` is genuinely broken still
            # fails here instead of being quietly excused (`CLAUDE.md` sec 2.5).
            if self.capability_tier() >= 1:
                self.module.eval()
            self._loaded = True

    def unload(self) -> None:
        """Release the model and free accelerator memory."""
        self._release()
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release(self) -> None:
        """Drop references to the underlying model; subclasses extend as needed."""

    def layer_names(self) -> list:
        """Capture points: named modules matching the layer regex, in forward order."""
        return self.all_layer_names()[:: max(1, self.cfg.capture_layer_stride)]

    def all_layer_names(self) -> list:
        """Every layer-regex match in forward order, ignoring the capture stride.

        The forecast lens patches states into the model's true final block,
        which the strided capture list may skip, so it needs the full list.
        """
        self.ensure_loaded()
        pattern = re.compile(self.cfg.layer_regex or self.default_layer_regex)
        names = [n for n, _ in self.module.named_modules() if pattern.search(n)]
        if not names:
            raise ValueError(
                f"model '{self.name}': layer regex matched nothing; "
                f"run discover_layers() to inspect module names")
        return names

    def final_block_name(self) -> str:
        """The last block on the residual path, used as the skip-lens patch target."""
        return self.all_layer_names()[-1]

    def uncaptured_surfaces(self) -> dict:
        """Blocks this model runs that the layer regex never matches, `{name: n}`.

        Optional, defaulting to nothing, so no existing adapter breaks and a
        model with a fully-captured stack says nothing. The canonical entry is
        Chronos-T5's decoder: `layer_names()` covers the encoder, so without
        this declaration nothing downstream can distinguish a 12-block model
        from the observed half of a 24-block one -- which is precisely the
        confusion `CLAUDE.md` sec 12 items 1-2 describe and the depth axis
        (`analysis/depth_axis.py`, ROADMAP.md sec 18 F1) exists to make
        visible on a plot.

        Declared surfaces are taken to run **after** the matched blocks; see
        `depth_axis.adapter_uncaptured_surfaces` for why that assumption is
        stated rather than inferred, and what an adapter with a leading
        uncaptured surface would need instead.
        """
        return {}

    def attention_info(self) -> Optional[list]:
        """Standardized per-block attention map for head-level interventions.

        Returns a list of dicts, one per block in forward order, each with
        keys `block` (the capture-layer module name), `o_proj` (qualified name
        of the attention output projection, whose input is the head
        concatenation so slicing it isolates single heads), `n_heads`, and
        `head_dim`. Returns None when the architecture does not expose a
        hookable output projection; head-level analyses then skip this model.
        """
        return None

    def mlp_info(self) -> Optional[dict]:
        """Map of block capture name -> qualified MLP module name (pre-residual).

        The named module's output must be the MLP contribution before the
        residual addition, so mean-ablating it removes only that component.
        Returns None when not resolvable for this architecture.
        """
        return None

    def attention_patterns(self, prepared: Any) -> Optional[dict]:
        """Self-attention probabilities per block for one prepared batch.

        Returns {block_name: tensor [B, n_heads, T, T]} restricted to the
        postprocessed token positions (specials stripped), or None when the
        architecture cannot expose patterns robustly.
        """
        return None

    def cross_attention_patterns(self, prepared: Any) -> Optional[torch.Tensor]:
        """First-forecast-step decoder cross-attention over context tokens.

        For encoder-decoder models only: returns [n_decoder_layers, B,
        n_heads, T_enc] for the first decoding step, giving a view into the
        otherwise-invisible decoder read of the encoder. None for
        decoder-only models or when unsupported.
        """
        return None

    def time_localization(self) -> Optional[dict]:
        """Measured evidence that this model's tokens each read one time interval.

        None means "not measured", which is every hand-written adapter's
        default: those declare their spans, and `--check-alignment` /
        `--discover-spans` are how a human verifies the declaration. An
        adapter that *derives* its spans by measurement (`GenericHFAdapter`)
        returns `{'localized': bool, 'contrast': float, ...}` here, which is
        the signal ROADMAP.md sec 16 E3(c)'s routing reads to drop a
        non-time-localized model to L0 only rather than pooling it onto a
        window axis that does not describe it.
        """
        return None

    def token_ids(self, prepared: Any) -> Optional[np.ndarray]:
        """Discretized token ids for a prepared batch, `[B, n_tokens]`.

        Only meaningful for architectures whose tokenizer re-quantizes the
        input (e.g. Chronos-T5's context-adaptive scalar quantization,
        `MeanScaleUniformBins` -- CLAUDE.md sec 11.16). Returns None for
        continuous-embedding architectures (TimesFM/Sundial/Chronos-Bolt/
        Chronos-2's patch-MLP tokenizers), which have nothing for an
        impulse probe to re-quantize -- `calibrate_impulse_amplitude`
        (ROADMAP.md sec 15 A20) treats None as "not applicable" and skips
        calibration rather than guessing.
        """
        return None

    def discover_layers(self, contains: str = "") -> list:
        """List candidate module names, for choosing a layer regex on a new checkpoint."""
        self.ensure_loaded()
        return [n for n, _ in self.module.named_modules() if contains in n]

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Strip special or padded positions so tokens match `token_time_spans`.

        The default trusts the raw capture; adapters whose sequence carries
        EOS or padding override this. A shape mismatch here is loud on
        purpose: silent misalignment would corrupt every downstream analysis.
        """
        expected = self.token_time_spans().shape[0]
        if hidden.shape[1] == expected:
            return hidden
        raise ValueError(
            f"model '{self.name}' layer '{layer_name}': captured {hidden.shape[1]} tokens, "
            f"alignment expects {expected}; override postprocess_tokens for this adapter")

    def find_module(self, name: str) -> nn.Module:
        """Fetch a submodule by its qualified name."""
        return dict(self.module.named_modules())[name]

    def token_slice(self, live_len: int) -> slice:
        """Positions of the postprocessed tokens within a live captured sequence.

        The inverse of `postprocess_tokens` for patching: activation patching
        writes cached clean token states back into a running forward, and this
        says where they belong. Default assumes tokens lead the sequence with
        any specials trailing; adapters with front padding override.
        """
        expected = self.token_time_spans().shape[0]
        if live_len < expected:
            raise ValueError(f"model '{self.name}': live sequence {live_len} shorter "
                             f"than expected {expected} tokens")
        return slice(0, expected)

    def hidden_size(self) -> Optional[int]:
        """Best-effort hidden dimension, resolved from the first capture on the fly if unknown."""
        return None


def random_init_like(model: nn.Module) -> nn.Module:
    """An architecture-matched twin of `model` with freshly, randomly initialized
    weights (ROADMAP.md sec 16 E9's untrained-weights null baseline).

    Reconstructs from the model's own `.config` via `type(model)(model.config)`
    rather than a generic reinitialization heuristic. This matters: many
    architectures (e.g. HF's `T5LayerNorm`) hold learnable parameters but
    define no `reset_parameters()`, so a heuristic that only reinitializes
    modules exposing that method would silently leave those specific weights
    at their pretrained values -- exactly the kind of silent partial failure
    `CLAUDE.md` sec 2.5 forbids. `type(model)(model.config)` instead reruns
    the class's own constructor (and, for `transformers.PreTrainedModel`
    subclasses, its own `_init_weights` scheme via `post_init()`), which is
    the standard, library-provided way to get "same config, untrained
    weights" -- exactly what `AutoModel.from_config()` does versus
    `from_pretrained()`. Raises if `model` has no `.config` to reconstruct
    from, rather than silently falling back to a weaker heuristic.
    """
    if not hasattr(model, "config"):
        raise ValueError(
            f"random_init_like: {type(model).__name__} has no `.config` attribute to "
            f"reconstruct from; this adapter needs its own random_init handling")
    fresh = type(model)(model.config)
    fresh.eval()
    return fresh


def count_blocks_matching(root: nn.Module, pattern: str) -> int:
    """How many of `root`'s named modules match `pattern`.

    Exists so an adapter declaring an uncaptured surface can *count* its
    blocks off the loaded model rather than hardcode a number that a
    checkpoint swap would silently invalidate -- the failure mode
    `CLAUDE.md` sec 11.8 records for hardcoded layer counts.
    """
    compiled = re.compile(pattern)
    return sum(1 for name, _ in root.named_modules() if compiled.search(name))


def _scan_attention(root, block_name: str):
    """Locate an attention submodule and its output projection inside one block."""
    block = dict(root.named_modules())[block_name]
    for sub_name, sub in block.named_modules():
        leaf = sub_name.rsplit(".", 1)[-1].lower()
        if not sub_name or not ("attn" in leaf or "attention" in leaf):
            continue
        n_heads = _first_attr(sub, ("num_heads", "n_heads", "num_attention_heads"))
        for proj_name, proj in sub.named_modules():
            proj_leaf = proj_name.rsplit(".", 1)[-1].lower()
            if isinstance(proj, nn.Linear) and proj_leaf in ("o_proj", "o", "out", "out_proj", "wo"):
                if not n_heads or proj.in_features % n_heads:
                    return None
                return {"block": block_name,
                        "o_proj": f"{block_name}.{sub_name}.{proj_name}",
                        "n_heads": int(n_heads),
                        "head_dim": proj.in_features // int(n_heads)}
    return None


def _scan_mlp(root, block_name: str):
    """Locate the pre-residual feed-forward submodule inside one block."""
    block = dict(root.named_modules())[block_name]
    for sub_name, _sub in block.named_modules():
        leaf = sub_name.rsplit(".", 1)[-1].lower()
        if sub_name and leaf in ("mlp", "ff", "ffn", "feed_forward",
                                 "transformer_feedforward", "densereludense"):
            return f"{block_name}.{sub_name}"
    return None


def _first_attr(obj, names: tuple):
    """First present integer attribute among candidate names, else None."""
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None
