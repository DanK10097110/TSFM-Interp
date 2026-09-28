"""Adapter for Lag-Llama (decoder-only transformer over disjoint lag features).

`ROADMAP.md` Item D2: this checkpoint was picked specifically because its
tokenization is architecturally unlike every hand-written adapter in this
package. Every existing model's tokens (TimesFM's 32-step patches, Chronos's
1-step scalars, Sundial's 16-step patches) each read one *contiguous* time
interval, which is the premise `extraction/span_discovery.py::is_contiguous`
(`CLAUDE.md` sec 12's second envelope edge, ROADMAP.md sec 19 G2) exists to
check rather than assume. Lag-Llama's per-timestep feature vector is built
from `lags_seq` -- 84 hand-picked historical offsets (0, 7, 8, ..., 1092), so
step t's token reads `{t-0, t-7, t-8, ..., t-1092}`: a *set of disjoint lags*,
not one interval. Whether that measures as time-localized-but-noncontiguous
(the gate this checkpoint exists to exercise on real weights, not a mock) is
recorded by `--discover-spans`, not asserted here.

Architecture: an 8-block, hidden=144 (n_head=9 x n_embd_per_head=16),
RoPE-positional, SDPA-causal-attention decoder (`transformer.h.0..7`), fed a
92-dim engineered feature per timestep (84 lags + 2 static scale/loc + 6
calendar features) through a `RobustScaler` (median/IQR, not mean/std) and a
`StudentTOutput` sampling head (3-parameter Student-t, matching Chronos-T5's
own "the forecast head samples" precedent -- ROADMAP.md sec 25/sec 11.50's
seeded-predict discipline applies here too). ~2.45M parameters, by far the
smallest checkpoint in this repo.

Two load-bearing implementation choices, recorded here rather than only in
the ROADMAP.md write-up because they are the difference between a correct
adapter and a silently-wrong one (`CLAUDE.md` sec 2.5):

1. **Front-padding for the lag window.** The largest lag (1092) exceeds any
   `context_len` this repo typically configures (512), so `prepare()` zero-
   pads the front of `past_target` with `past_observed_values=0` for the
   padded region. `RobustScaler` is NaN-aware (`torch.where(weights==1, data,
   nan)` then `nanmedian`/`nanquantile`), so the padding is excluded from the
   scale statistics it computes -- verified against the vendored scaler
   directly, not assumed. This makes every repo-standard `context_len` work
   and yields exactly `context_len` tokens (1 token/timestep, Chronos-T5-
   shaped granularity), at the cost of the model always seeing `max_lag`
   fewer *real* history steps than its own pretraining checkpoint's longest
   lag would ideally want for the shortest configured contexts.
2. **Weights stay float32 regardless of the run's configured autocast
   dtype.** At ~2.45M parameters this model's compute cost is negligible, so
   the precision-mismatch class of bug `CLAUDE.md` sec 11.49 found in
   TimesFM (fp32 weights computed under bf16 autocast during capture,
   differing from `predict()`'s bf16-native forward) is avoided by never
   participating in bf16 at all: `load()` casts to `torch.float32`
   unconditionally, independent of `self.dtype`. Prepare/forward/predict
   all build their own tensors as float32 for the same reason -- the
   Student-t head's `df` parameter goes through a positivity map whose
   precision near small values is exactly the kind of thing bf16 would blur.

`predict()` is a from-scratch, no-kv-cache autoregressive sampling loop (the
vendored `use_kv_cache` path exists but recomputing the full sequence each
step was chosen for simplicity and correctness at this model's negligible
size, not for speed): the upstream checkpoint's own pretraining pipeline
(`lag_llama/gluon/estimator.py`) hardcodes `time_features_from_frequency_str
("S")` for every dataset regardless of its real frequency, so this adapter
does the same, against a dummy `pd.date_range` -- this repo's synthetic
corpora carry no real timestamps for any adapter to read anyway.

`attention_patterns()` is deliberately unsupported (this adapter stays tier
2, not tier 3): `CausalSelfAttention.forward` (in the vendored module) calls
`F.scaled_dot_product_attention` directly with RoPE-rotated q/k, not through
a swappable `attention_fn` seam the way TimesFM's does (`CLAUDE.md` sec 6.2's
correction). Recovering the intermediate weights would mean hand-replicating
the RoPE + causal-mask + softmax arithmetic outside SDPA's fused kernel, and
a subtle mistake there would produce a plausible-looking but wrong attention
map rather than an honest refusal -- the same "release fragility not worth
it" call this repo already made for Chronos-Bolt and Chronos-2's own
attention_patterns (`CLAUDE.md` sec 6.2's support matrix). `attention_info`
IS implemented (hand-written, mirroring Chronos-T5's own hard-coded pattern,
since `_scan_attention`'s generic heuristic looks for an attribute literally
named `num_heads`/`n_heads`/`num_attention_heads` and a Linear leaf named
`o_proj`/`o`/`out`/`out_proj`/`wo` -- this architecture has neither: the
head-count attribute is `n_head` singular and the output projection is
`c_proj`), so head-level mean-ablation still works.

Vendored support code (the model architecture and the RobustScaler) lives in
`models/contrib/_lag_llama_vendor/`, copied verbatim (license header and
exact source commit preserved on each file) from
https://github.com/time-series-foundation-models/lag-llama
(`gluonts.torch.modules.loss` -- a training-only loss-object attribute of the
Lightning checkpoint's pickled hyperparameters, unrelated to inference -- was
removed from gluonts between the upstream repo's pin (<=0.14.4) and this
repo's installed 0.17.0; `_stub_removed_loss_module` below installs a
throwaway stand-in into `sys.modules` before `torch.load` so the checkpoint
still unpickles, and is not a modification of any vendored file).
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from tsfm_lens.utils import log
from tsfm_lens.models.base import ModelAdapter, NotTimeLocalized

ADAPTER_NAME = "lag_llama"
ADAPTER_CLASS = "LagLlamaAdapter"

_CKPT_FILENAME = "lag-llama.ckpt"
_N_TIME_FEAT = 6  # fixed: the checkpoint's own pretraining pipeline hardcodes freq="S"


def _stub_removed_loss_module() -> None:
    """Install a throwaway `gluonts.torch.modules.loss` so `torch.load` can unpickle.

    The real module existed only through gluonts<=0.14.4 (the upstream repo's
    own pin) and was removed by 0.17.0 (this repo's installed version). The
    Lightning checkpoint's pickled `hyper_parameters` reference
    `NegativeLogLikelihood`/`DistributionLoss` by import path as training-only
    metadata -- nothing at inference time reads the reconstructed object's
    behavior, only its presence, so a minimal stand-in sufficient for
    `__setstate__` is enough. Idempotent and side-effect-free if the real
    module is ever reintroduced (only installed when absent).
    """
    if "gluonts.torch.modules.loss" in sys.modules:
        return
    mod = types.ModuleType("gluonts.torch.modules.loss")

    class DistributionLoss:
        def __new__(cls, *a, **k):
            return object.__new__(cls)

        def __setstate__(self, state):
            self.__dict__.update(state or {})

    class NegativeLogLikelihood(DistributionLoss):
        pass

    mod.DistributionLoss = DistributionLoss
    mod.NegativeLogLikelihood = NegativeLogLikelihood
    sys.modules["gluonts.torch.modules.loss"] = mod


class LagLlamaAdapter(ModelAdapter):

    default_layer_regex = r"^transformer\.h\.\d+$"
    measures_own_spans = True

    def __init__(self, *args, **kwargs) -> None:
        # `pipeline.resolve_routing` calls `token_time_spans()` on a freshly
        # constructed, not-yet-loaded adapter (it relies on `discover_spans`
        # to call `ensure_loaded()` itself) -- so `_spans`/`_discovery` must
        # exist before `load()` ever runs, not only be set inside it.
        super().__init__(*args, **kwargs)
        self._spans: Optional[np.ndarray] = None
        self._discovery = None
        self._warming_spans = False

    def ensure_loaded(self) -> None:
        """Warm `token_time_spans()` eagerly, before any caller can attach a hook.

        Bug found and fixed in THIS file during ROADMAP.md Item D2's own
        real paired-model run (not a shared-infrastructure defect -- see
        `GenericHFAdapter.ensure_loaded`'s identical, already-documented
        fix, which this mirrors verbatim): `discover_spans`'s own internal
        probe forward passes go through `adapter.forward`, and if a capture
        or patching hook is already installed on the module (extraction,
        the lens stage, or `pipeline.resolve_routing`'s own hooks) that hook
        calls `token_slice` -> `token_time_spans` -> (still `None`) ->
        `discover_spans` -> another `adapter.forward` -> the same hook again,
        recursing until the interpreter's stack limit raises `RecursionError`
        (observed live: `lag_llama_vs_chronos` run, mid layer_screen /
        second `extract` after a `keep_models_loaded: false` reload cleared
        `self._spans`). Discovery is one cheap forward pass and every
        analysis path needs the spans anyway, so it is done here, right
        after the model loads and before any hook can be attached.
        `_warming_spans` guards re-entry through this method itself (a
        forward call inside discovery can trigger `ensure_loaded` again via
        `hub.get()`-style lazy loading elsewhere in the pipeline).
        """
        super().ensure_loaded()
        if self._spans is None and not getattr(self, "_warming_spans", False):
            self._warming_spans = True
            try:
                self.token_time_spans()
            except NotTimeLocalized:
                pass
            finally:
                self._warming_spans = False

    def load(self) -> None:
        from huggingface_hub import hf_hub_download
        from gluonts.torch.distributions import StudentTOutput

        from tsfm_lens.models.contrib._lag_llama_vendor.model import LagLlamaModel

        checkpoint = self.cfg.checkpoint or "time-series-foundation-models/Lag-Llama"
        _stub_removed_loss_module()
        ckpt_path = hf_hub_download(repo_id=checkpoint, filename=_CKPT_FILENAME)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_kwargs = dict(ckpt["hyper_parameters"]["model_kwargs"])
        # The checkpoint's own pickled distr_output is an instance tied to the
        # training-time loss machinery; rebuild fresh (same class/args the
        # upstream repo's own inference path uses) rather than trust the
        # unpickled one, since only its *shape* (3 StudentT params) matters.
        model_kwargs["distr_output"] = StudentTOutput()
        model = LagLlamaModel(**model_kwargs)
        if not self.cfg.random_init:
            state_dict = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                          if k.startswith("model.")}
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            assert not missing and not unexpected, (missing, unexpected)
        else:
            log.info("model '%s': random_init=True, skipping checkpoint weights "
                     "(architecture built fresh with random init, matching "
                     "TimesFM's own random_init precedent of skipping its "
                     "checkpoint-loading step)", self.name)
        # float32 unconditionally -- see module docstring point 2.
        self._model = model.to(self.device, dtype=torch.float32).eval()
        self._lags_seq = list(model.lags_seq)
        self._max_lag = max(self._lags_seq)
        self._n_head = model.transformer.h[0].attn.n_head
        self._head_dim = model.transformer.h[0].attn.n_embd_per_head
        self._spans: Optional[np.ndarray] = None
        self._discovery = None

    @property
    def module(self):
        return self._model

    def _release(self) -> None:
        self._model = None

    def _time_features(self, length: int) -> np.ndarray:
        """The checkpoint's own hardcoded freq='S' calendar features, over a dummy calendar.

        This repo's synthetic corpora carry no real timestamps, and the
        upstream pretraining pipeline used this exact frequency string for
        every dataset regardless of its real one -- so a dummy calendar is
        not a lesser stand-in than what the checkpoint was trained against.
        """
        from gluonts.time_feature import time_features_from_frequency_str
        feats = time_features_from_frequency_str("S")
        idx = pd.date_range("2000-01-01", periods=length, freq="s")
        return np.stack([f(idx) for f in feats], axis=-1).astype(np.float32)

    def prepare(self, contexts: np.ndarray) -> Any:
        """Front-padded, single (non-autoregressive) forward's worth of input.

        Used only for the capture pass (`forward`, below) -- one token per
        real context timestep, `max_lag` padded steps prepended and marked
        unobserved so `RobustScaler` excludes them from its statistics.
        """
        contexts = np.ascontiguousarray(contexts, dtype=np.float32)
        B, L = contexts.shape
        device = self.device
        total_len = self._max_lag + L
        past_target = torch.zeros(B, total_len, device=device, dtype=torch.float32)
        past_target[:, self._max_lag:] = torch.from_numpy(contexts).to(device)
        observed = torch.zeros(B, total_len, device=device, dtype=torch.float32)
        observed[:, self._max_lag:] = 1.0
        time_feat = torch.from_numpy(self._time_features(total_len)).to(device)
        time_feat = time_feat.unsqueeze(0).expand(B, -1, -1).contiguous()
        # `prepare_input`'s own time_feat branch only assigns its local
        # `time_feat` variable when BOTH past and future time features are
        # not None, then unconditionally references it whenever past is not
        # None -- an UnboundLocalError on any call with future_time_feat=None
        # (this vendored file is left byte-identical to upstream; worked
        # around here, not patched in the vendored copy). A [B,1,N] dummy
        # whose `[..., :-1, :]` slice is empty satisfies the branch with no
        # effect on the result, matching every one of our calls having no
        # real future target.
        future_time_feat = torch.zeros(B, 1, _N_TIME_FEAT, device=device, dtype=torch.float32)
        return {"past_target": past_target, "past_observed_values": observed,
                "past_time_feat": time_feat, "future_time_feat": future_time_feat,
                "future_target": None}

    def forward(self, prepared: Any) -> None:
        with torch.no_grad():
            self._model(**prepared)

    @property
    def _min_contrast(self) -> float:
        import inspect as _inspect
        from tsfm_lens.extraction.span_discovery import SpanDiscovery
        default = _inspect.signature(SpanDiscovery.is_time_localized).parameters["min_contrast"]
        return float(self.cfg.kwargs.get("min_contrast", default.default))

    @property
    def _min_contiguity(self) -> float:
        import inspect as _inspect
        from tsfm_lens.extraction.span_discovery import SpanDiscovery
        default = _inspect.signature(SpanDiscovery.is_contiguous).parameters["min_contiguity"]
        return float(self.cfg.kwargs.get("min_contiguity", default.default))

    def token_time_spans(self) -> np.ndarray:
        """Discovered spans, measured once and cached; raises when not usable.

        This is the seam ROADMAP.md Item D2 exists to exercise: Lag-Llama's
        tokens are, by construction, a set of disjoint lags rather than one
        interval, so whichever gate fires (or neither) is a real measurement
        on real weights, not an assumption -- see the module docstring.
        """
        if self._spans is None:
            from tsfm_lens.extraction.span_discovery import discover_spans
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
        if self._spans is None:
            return hidden
        return super().postprocess_tokens(layer_name, hidden)

    def attention_info(self) -> Optional[list]:
        """Hand-written: `_scan_attention`'s generic heuristic matches neither

        this architecture's head-count attribute (`n_head`, not
        `n_heads`/`num_heads`/`num_attention_heads`) nor its output
        projection's leaf name (`c_proj`, not `o_proj`/`o`/`out`/`out_proj`/
        `wo`) -- the same class of gap Chronos-T5's `ff0`/`ff1` MLP has
        relative to `_scan_mlp`, worked around the same way (`CLAUDE.md`
        sec 6.2): hard-code rather than extend the shared scanner.
        """
        return [{"block": n, "o_proj": f"{n}.attn.c_proj",
                 "n_heads": self._n_head, "head_dim": self._head_dim}
                for n in self.layer_names()]

    def mlp_info(self) -> Optional[dict]:
        """`Block.forward` is `y = x + self.mlp(self.rms_2(x))`; the `mlp`

        submodule's own output IS the pre-residual MLP contribution (its last
        internal op is `c_proj` with nothing after), so naming the submodule
        itself is exact, not an approximation.
        """
        return {n: f"{n}.mlp" for n in self.layer_names()}

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Autoregressive sampling from the StudentT head, no kv-cache.

        The head samples (`CLAUDE.md` sec 11.50's precedent: any correctness
        probe that calls `predict()` more than once and diffs the results
        needs its own seeding, which `models/conformance.py` already applies
        uniformly). Recomputes the full sequence at every one of `horizon`
        steps rather than using the vendored `use_kv_cache` path -- simpler
        and, at ~2.45M parameters, cheap enough that the correctness benefit
        (no cache-indexing bug class to get wrong under a tight session) was
        judged worth the wasted compute.
        """
        num_samples = int(self.cfg.kwargs.get("num_samples", 20))
        contexts = np.ascontiguousarray(contexts, dtype=np.float32)
        B, L = contexts.shape
        device = self.device
        total_len = self._max_lag + L + horizon

        contexts_t = torch.from_numpy(contexts).to(device)
        contexts_rep = contexts_t.repeat_interleave(num_samples, dim=0)
        Bs = contexts_rep.shape[0]

        past_target = torch.zeros(Bs, total_len, device=device, dtype=torch.float32)
        past_target[:, self._max_lag:self._max_lag + L] = contexts_rep
        observed = torch.zeros(Bs, total_len, device=device, dtype=torch.float32)
        observed[:, self._max_lag:self._max_lag + L] = 1.0
        time_feat = torch.from_numpy(self._time_features(total_len)).to(device)
        time_feat = time_feat.unsqueeze(0).expand(Bs, -1, -1).contiguous()
        future_time_feat = torch.zeros(Bs, 1, _N_TIME_FEAT, device=device, dtype=torch.float32)

        model = self._model
        with torch.no_grad():
            for t in range(horizon):
                cur_len = self._max_lag + L + t
                params, loc, scale = model(
                    past_target=past_target[:, :cur_len],
                    past_observed_values=observed[:, :cur_len],
                    past_time_feat=time_feat[:, :cur_len, :],
                    future_time_feat=future_time_feat,
                    future_target=None)
                df, loc_p, scale_p = params
                distr = model.distr_output.distribution(
                    (df[:, -1], loc_p[:, -1], scale_p[:, -1]),
                    loc=loc.squeeze(-1), scale=scale.squeeze(-1))
                sample = distr.sample()
                past_target[:, self._max_lag + L + t] = sample
                observed[:, self._max_lag + L + t] = 1.0

        samples = past_target[:, self._max_lag + L:].view(B, num_samples, horizon)
        samples_np = samples.detach().cpu().numpy()
        point = np.median(samples_np, axis=1)
        q = np.quantile(samples_np, quantiles, axis=1)
        q = np.moveaxis(q, 0, -1)  # [Q, B, H] -> [B, H, Q]
        return {"point": point, "quantiles": q}
