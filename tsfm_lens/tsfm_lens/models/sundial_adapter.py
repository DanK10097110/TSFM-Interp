"""Adapter for Sundial (Tsinghua THUML; decoder-only transformer over input
patches with a flow-matching probabilistic head, "TimeFlow Loss").

`thuml/sundial-base-128m` (12 layers, hidden 768, 12 heads, patch=16, trained
on ~1T time points, Apache-2.0). Architecturally a third distinct pattern
next to TimesFM (deterministic quantile head) and Chronos (discrete-token
sampling): Sundial's decoder-only backbone (`SundialModel`, plain causal
self-attention + SwiGLU-style MLP blocks over 16-step patches, RoPE
position embeddings) is capped by a `FlowLoss` head that samples full
`horizon`-length continuous trajectories directly from the *final* patch's
hidden state via a diffusion-style ODE sampler -- there is no
per-timestep/per-patch autoregressive decoding step at all for horizons
within one output length (checkpoint default 720 steps), unlike Chronos-T5's
real decoder or TimesFM's own decode loop.

Capture and prediction both call the model's own forward paths directly
(never `.generate()`) for two independent, verified reasons:

1. **A real bug in Sundial's own remote code makes `.generate()` unusable
   against this checkpoint on transformers>=4.4x.** `TSGenerationMixin`'s
   `.generate()` funnels through `SundialForPrediction.prepare_inputs_for_
   generation`, which reads `past_key_values.seen_tokens` -- an attribute
   removed from `transformers.DynamicCache` well before 4.57.6 (the
   version installed in this repo's shared `cudaPy` env; the HF card
   recommends the much older 4.40.1, where that attribute still existed).
   Confirmed directly (not assumed) against the installed 4.57.6: a bare
   `model.generate(seqs, max_new_tokens=64, num_samples=20)` raises
   `AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'`
   on the very first call.
2. **A single direct forward call is also strictly sufficient and cheaper**
   for this repo's needs. `SundialForPrediction.forward(labels=None)`
   already produces `[B, num_samples, output_token_len]` predictions in
   *one* non-autoregressive call by sampling the flow-matching head off the
   final patch's hidden state -- exactly the `.generate()`-documented
   `(batch_size, num_samples, forecast_length)` shape, with `horizon`
   always well under the 720-step ceiling for this repo's configs. The
   *capture* path (`forward()` below) goes one level lower still, calling
   `SundialForPrediction.model(...)` (the decoder-only backbone alone,
   `use_cache=False`) directly -- mirroring `ChronosAdapter`'s own
   "encoder-only" pattern -- so extraction never pays for the flow-matching
   sampler's ~50 diffusion steps at all; hooks on `model.layers.{i}` still
   fire identically either way, since they are the same submodule
   instances regardless of which top-level `forward` reaches them.

`use_cache=False` on every call (both capture and predict) is required, not
optional: `SundialModel.forward`'s cache-enabled branch calls
`past_key_values.get_usable_length(...)` -- also removed from
`transformers.DynamicCache` in the installed version -- so leaving the
default `use_cache=True` (the checkpoint's own config default) crashes the
same way `.generate()` does, even on a single non-generation forward call.
Neither of these breakages is a bug in this repo's shared abstraction: both
are the remote checkpoint's own code assuming an older `transformers.Cache`
API than is installed. Since a single forward call never needs incremental
decoding, routing around the whole `Cache` machinery via `use_cache=False`
sidesteps both breakages entirely rather than patching or downgrading a
shared library version other adapters (TimesFM, Chronos, Chronos-Bolt,
Chronos-2) already depend on -- see `CLAUDE.md` sec 11.8-11.10 for why that
downgrade path is exactly the trap to avoid.

Weights load at the checkpoint's own default dtype (float32) rather than
forcing `torch_dtype=self.dtype` at load time the way the Chronos adapters
do: verified directly that doing so breaks Sundial's own patch-embedding
code (`SundialPatchEmbedding.forward` hardcodes
`mask = torch.ones_like(x, dtype=torch.float32)` regardless of `x`'s actual
dtype, so a bf16-loaded model crashes concatenating a float32 mask against
a bf16 input on the very first call). Extraction still gets its bf16
compute the normal way, through `extract.py`'s own `torch.autocast` region
around every `adapter.forward()` call -- confirmed empirically to work
cleanly with fp32 master weights, matching `TimesFMAdapter`'s identical
choice for an unrelated reason (compile-safety there, this dtype
incompatibility here).

Attention patterns are deliberately left unsupported (inherits the base
class's `None`): `SundialAttention.forward` runs a fused
`F.scaled_dot_product_attention` call with no exposed intermediate weights,
and confirmed directly that its own `output_attentions=True` branch is
broken independent of that -- it reads a local `attn_weights` variable that
is only ever assigned inside the `if not output_attentions` branch, so
requesting attentions raises `UnboundLocalError: local variable
'attn_weights' referenced before assignment` before any patching trick
would even matter. Recovering patterns would require globally monkeypatching
`torch.nn.functional.scaled_dot_product_attention` for the call (there is no
per-instance swappable `attention_fn` seam the way TimesFM's
`MultiHeadAttention` has) -- a broader, less contained hack than any
existing adapter uses, for a fragile, upstream-broken code path. Chronos-Bolt
already sets the precedent for declining an attention surface on
reliability grounds rather than reliability-at-any-cost; the same call is
made here. `attention_info`/`mlp_info` (head/MLP mean-ablation) are fully
supported -- both are plain hookable `nn.Linear` modules, verified directly
against the loaded module tree, independent of the attention-pattern issue
above. No decoder, no cross-attention: `cross_attention_patterns` stays
unsupported for the same reason it does for TimesFM.

**Manual per-series input normalization (`revin`), applied in exactly one
place (`prepare()`), is load-bearing -- not a style choice.** The checkpoint's
own remote code (`ts_generation_mixin.py`) normalizes each series
(`x = (x - mean) / std`, denormalizing the sampled output afterward) only on
its `.generate()` path (`revin=True` there by default); `SundialForCausalLM.
forward` itself defaults to `revin=False`, and its own `revin=True` branch is
independently broken for `num_samples > 1` (`predictions * stdev + means`
fails to broadcast -- confirmed directly: `RuntimeError: size of tensor a
(20) must match ... (64) at dim 1`). Because this repo's `predict()` and
`forward()` both call the checkpoint's forward paths directly (never
`.generate()`, for the two independent reasons above), **every Sundial
forecast and captured activation in this repo was, before this fix, computed
on raw-scale input** -- the checkpoint's intended per-series normalization
never ran. Diagnosed from the frontend stage's own scale-equivariance probe
reading Sundial's worst residual at 38.4318 context-scale units against
<=0.0086 for every other model in `runs/full_report_run_4model`, and
confirmed at the input level: 10x/100x-scaling a context collapses Sundial's
forecast to flat (0.002 forecast/context sd ratio, MASE 4.13) where manual
z-scoring the input does not (`CLAUDE.md` sec 8's "silent failures produce
well-formed output" -- a flat forecast on an out-of-distribution scale reads
as ordinary model behavior until this probe is read as a bug, not a
property). The checkpoint's own rule, reproduced from its `generate()` path
(`ts_generation_mixin.py`, the official API: `mu = x.mean(1)`, `sd =
x.std(1, unbiased=False) + 1e-5`, normalize the input by `(x - mu) / sd`,
denormalize sampled outputs by `* sd + mu`), is
applied once in `prepare()` so that `forward()` (capture), `predict()` and
anything patched through `hooks.token_patch` (whose clean cache is written
via `prepare()` + `forward()` too, see `extract.py::capture_raw_tokens`) all
see an identical normalized tensor -- there is no second code path into this
checkpoint's backbone that could see raw-scale input instead. `prepare()`
returns a small `_Prepared` container (the normalized tensor plus its
per-row `mu`/`sd`) rather than a bare tensor specifically so `predict()`
denormalizes with the *same* stats `forward()` normalized with, rather than
recomputing them from a second `contexts.mean()/.std()` call that could
drift from the first if either were ever changed independently. The
checkpoint's `forward(revin=True)` branch uses a DIFFERENT rule (an absolute
floor, `sd = sd if sd > 1e-2 else 1`) and is broken for `num_samples > 1`;
its floor also breaks scale-equivariance for small-magnitude series (at a
0.001x rescale every one of the frontend probe's 64 series fell below it,
residual 39.47 context-sd units even with seeded sampling), so the
`generate()` rule is the one reproduced here. This is
unconditional (not a config knob): the checkpoint's own intended use always
normalizes, nothing in this repo's stages needs the raw-scale behavior as a
comparison arm, and CLAUDE.md sec 2.1's "old behavior stays reproducible
behind a knob" is about behavior something else *depends on* -- reproducing
a diagnosed bug is not that.

Written and verified against `transformers==4.57.6`, `torch==2.12.0`,
`thuml/sundial-base-128m`; run `--check-alignment` again on any future
library or checkpoint bump, per every other adapter in this file.

**`--check-alignment` reads low at this model's own default impulse
amplitude -- verified to be a real, amplitude-dependent property of this
architecture, not a broken `token_time_spans` mapping (`CLAUDE.md` sec
11.16's playbook, applied to a third distinct failure mode).** At
`impulse_alignment_check`'s current 0.25x-base-amplitude default, Sundial's
diagonal-hit fraction is a perfect 1.00 at the first 2-3 layers and decays
to ~0.06-0.12 by mid-depth -- looks exactly like sec 6.3's "near-zero
everywhere, fix the adapter" warning sign at first glance. Diagnosed
directly (not guessed) two ways before concluding otherwise: (1) the full
per-probe-window delta row is *exactly* zero for every window before the
perturbed one at every layer and every amplitude tested, at every depth --
i.e. zero backward leakage, exactly what a correctly-causal, correctly-
aligned decoder must show, which a broken span mapping would not
reliably reproduce; (2) sweeping the impulse amplitude from 0.02x to 4x
base amplitude shows the diagonal-hit fraction is *strictly amplitude-
dependent* -- perfect 1.00 at every one of the 12 layers at 0.02x-0.05x,
decaying steadily as amplitude grows -- which a genuinely wrong mapping
would not do (a wrong mapping is wrong regardless of how hard you push on
it). Side-by-side against `TimesFMAdapter` with the exact same check code
at the exact same 0.25x amplitude, all 20 of TimesFM's layers stay at a
perfect 1.00 -- so this is a real, verified difference in how far a given-
size perturbation's causal echo propagates forward through Sundial's 12
plain-residual decoder blocks (no per-adapter QK-norm the way TimesFM's
`MultiHeadAttention` docstring notes it has) relative to how large the
perturbation itself was, not an artifact of the check being run on a new
architecture. Read a low default-amplitude number for this adapter as "a
real forward-accumulation effect, confirmed benign by the amplitude sweep,"
not as "the span mapping is broken" -- but note the in-pipeline gate
(`extraction/alignment.py::run_alignment_gate`) only checks the *shallowest*
captured layer's fraction, which stays a perfect 1.00 for Sundial at every
amplitude tested, so a real run will not surface this mid-depth pattern on
its own; a human re-running `--check-alignment sundial` and reading every
layer (not just the gate's pass/fail) is still the way to see it, exactly
per invariant 7's "verified empirically," not inferred from a single
aggregate.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch

from ..utils import log
from .base import ModelAdapter, _scan_attention, random_init_like


class _Prepared(NamedTuple):
    """Sundial's `prepare()` output: the checkpoint's own per-series
    normalization rule applied once, plus the exact per-row stats used to
    apply it -- carried alongside the tensor (rather than recomputed) so
    `predict()` denormalizes with precisely the same `mu`/`sd` that
    `forward()` normalized the identical batch with."""
    normalized: torch.Tensor
    mu: torch.Tensor
    sd: torch.Tensor


class SundialAdapter(ModelAdapter):

    default_layer_regex = r"model\.layers\.\d+$"

    def load(self) -> None:
        """Load the HF `trust_remote_code` model at its own default (float32)
        dtype and read patch/horizon geometry from its config.

        `random_init` (ROADMAP.md sec 16 E9) reconstructs the model from its
        own config via the shared `random_init_like` helper -- identical
        mechanism to every other adapter in this file, verified directly to
        work against this checkpoint's custom remote-code class.
        """
        from transformers import AutoModelForCausalLM
        repo = self.cfg.checkpoint or "thuml/sundial-base-128m"
        self._inner = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        if self.cfg.random_init:
            log.warning("sundial '%s': random_init=True -- discarding pretrained "
                        "weights, using an architecture-matched random-weight twin "
                        "(ROADMAP.md sec 16 E9)", self.name)
            self._inner = random_init_like(self._inner)
        self._inner = self._inner.to(self.device)
        self._inner.eval()
        ccfg = self._inner.config
        self._patch = int(ccfg.input_token_len)
        self._max_horizon = int(max(ccfg.output_token_lens))
        if self.data_cfg.horizon > self._max_horizon:
            raise ValueError(
                f"sundial '{self.name}': data.horizon {self.data_cfg.horizon} exceeds this "
                f"checkpoint's max output_token_len {self._max_horizon}")
        log.info("sundial '%s': patch=%d hidden=%d layers=%d heads=%d max_horizon=%d",
                 self.name, self._patch, int(ccfg.hidden_size), int(ccfg.num_hidden_layers),
                 int(ccfg.num_attention_heads), self._max_horizon)

    @property
    def module(self):
        return self._inner

    def _release(self) -> None:
        self._inner = None

    def _geometry(self):
        """(n_tokens, front_pad) for the current data context -- mirrors
        `SundialPatchEmbedding.forward`'s own left-padding formula exactly
        (`padding_length = (patch - (length % patch)) % patch`), verified
        against the loaded module's source rather than assumed."""
        ctx = self.data_cfg.context_len
        front_pad = (self._patch - (ctx % self._patch)) % self._patch
        n_tokens = (ctx + front_pad) // self._patch
        return n_tokens, front_pad

    def prepare(self, contexts: np.ndarray) -> _Prepared:
        """Apply the checkpoint's own per-series `revin` rule (module
        docstring above) -- the single normalization site every other method
        in this class reads from, so capture, prediction and patching never
        disagree about what the model saw."""
        raw = torch.from_numpy(np.ascontiguousarray(contexts)).float().to(self.device)
        mu = raw.mean(dim=1, keepdim=True)
        sd = raw.std(dim=1, keepdim=True, unbiased=False) + 1e-5
        return _Prepared((raw - mu) / sd, mu, sd)

    def forward(self, prepared: _Prepared) -> None:
        """One non-cached backbone-only pass over the NORMALIZED tensor --
        skips the flow-matching head entirely (not needed to fire capture
        hooks on `model.layers.{i}`), mirroring `ChronosAdapter`'s "encoder
        only" capture pattern."""
        with torch.no_grad():
            self._inner.model(input_ids=prepared.normalized, use_cache=False)

    def token_time_spans(self) -> np.ndarray:
        """One span per 16-step input patch; the leading patch is clipped
        where left-padding covers no real data."""
        n_tokens, front_pad = self._geometry()
        starts = np.arange(n_tokens) * self._patch - front_pad
        spans = np.stack([starts, starts + self._patch], axis=1).astype(np.float64)
        return np.clip(spans, 0, self.data_cfg.context_len)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Sample flow-matching forecast trajectories in one forward call over
        the NORMALIZED context, denormalize the sampled trajectories with the
        same per-row `mu`/`sd` `prepare()` computed, then reduce to point
        (median) and quantiles -- identical reduction to
        `ChronosAdapter.predict`'s handling of its own sampled decoder."""
        num_samples = int(self.cfg.kwargs.get("num_samples", 20))
        prepared = self.prepare(contexts)
        with torch.no_grad():
            out = self._inner(input_ids=prepared.normalized, max_output_length=horizon,
                              num_samples=num_samples, use_cache=False, return_dict=True)
        samples = out.logits[..., :horizon].float().cpu()  # [B, num_samples, horizon]
        mu, sd = prepared.mu.cpu(), prepared.sd.cpu()
        samples = samples * sd[:, :, None] + mu[:, :, None]
        q = torch.quantile(samples, torch.tensor(quantiles, dtype=torch.float32), dim=1)
        return {"point": samples.median(dim=1).values.numpy(),
                "quantiles": q.permute(1, 2, 0).numpy()}

    def attention_info(self) -> list:
        """Best-effort head map via the shared block scan -- every block's
        `self_attn.o_proj` is a plain hookable Linear, verified directly."""
        self.ensure_loaded()
        infos = []
        for block in self.all_layer_names():
            info = _scan_attention(self.module, block)
            if info is None:
                log.info("sundial '%s': no head map for block %s; "
                         "head-level analyses disabled", self.name, block)
                return None
            infos.append(info)
        return infos

    def mlp_info(self) -> dict:
        """Block -> pre-residual MLP map. Not resolved via the shared
        `_scan_mlp` helper -- Sundial's feed-forward submodule is named
        `ffn_layer`, which that helper's fixed name set does not match (a
        naming-convention gap, not a bug in the helper) -- so the qualified
        name is built directly and checked for existence, verified against
        the loaded module tree rather than assumed."""
        self.ensure_loaded()
        modules = dict(self.module.named_modules())
        mapping = {}
        for block in self.all_layer_names():
            name = f"{block}.ffn_layer"
            if name not in modules:
                log.info("sundial '%s': no MLP module for block %s; "
                         "MLP ablation disabled", self.name, block)
                return None
            mapping[block] = name
        return mapping
