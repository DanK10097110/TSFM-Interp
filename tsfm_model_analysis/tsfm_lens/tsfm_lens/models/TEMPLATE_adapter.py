r"""Copy-paste skeleton for a new `ModelAdapter` (`ROADMAP.md` sec 34.6 Item E1.3).

**How to use this file.** Copy it into `models/contrib/<your_name>_adapter.py`,
rename the class, fill in `load()`/`predict()` against your real checkpoint,
then work your way up through the tiers below -- each one unlocks a specific
set of pipeline stages (`CLAUDE.md` sec 6.2's tier table; run
`run.py --check-adapter <your_name>` once registered, ROADMAP.md sec 34.6
Item E4, to see exactly which stages your adapter currently unlocks and what
each check found).

**As shipped, this file runs.** `TemplateAdapter` subclasses
`models.mock.MockBlackBoxAdapter` -- the tier-0 mock built for exactly this
purpose (it forecasts with a real, if weak, seasonal-naive-plus-drift
algorithm and implements nothing else) -- so before you write a single line
against your real checkpoint you can register this file (copy it into
`contrib/`, it is already a valid ADAPTER_NAME/ADAPTER_CLASS pair) and run
the checklist against a model that genuinely works, to see what "pass"
actually looks like. Then replace the inherited `load()`/`predict()` with
your real ones and start climbing tiers.

**The four tiers, and what each buys** (`CLAUDE.md` sec 6.2, ROADMAP.md sec
19 G1). Tier is *derived* from which methods you override -- never hand-set
it, a hand-set tier is a claim nothing checks (`CLAUDE.md` sec 11.34):

- **Tier 0 (black box)** -- only `load()` + `predict()`. Unlocks `l0`,
  `budget`, `frontend`, `report`, `register`, `confirm`, `corpus`. This is
  the only tier available for a hosted/API-only model with no readable
  internals at all.
- **Tier 1 (observable)** -- also override `module`, `prepare`, `forward`,
  `token_time_spans`. Unlocks `extract`, `layer_screen`, `internals`, `l1`,
  `l2`, `cluster`, `sae`, `attention`, `exemplars`.
- **Tier 2 (steerable)** -- tier 1, plus the whole context is processed in
  ONE forward pass (`single_pass_context = True`, the default -- only flip
  it to False if your model genuinely chunks or re-enters its context).
  Unlocks `lens`, `l3` (activation patching needs a single pass to patch).
- **Tier 3 (decomposable)** -- tier 2, plus override `attention_info` AND
  `attention_patterns`. Unlocks per-head ablation and attention-pattern
  analysis. `mlp_info` is optional at every tier (TimesFM has no hookable
  MLP submodule and is still tier 3 -- CLAUDE.md sec 6.2).

**A refusal is a valid contribution, not a failure.** If your model's tokens
genuinely do not each cover one contiguous time interval -- a lag-feature
tokenizer reading several disjoint lags per token, say -- `token_time_spans`
should raise `models.base.NotTimeLocalized`, not silently return a wrong
map. The pipeline routes a model that raises it to `l0`-only rather than
pooling activations onto a window axis that does not describe it
(`CLAUDE.md` sec 12's envelope edge). Never catch `ValueError` broadly there:
only `NotTimeLocalized` is a considered verdict; anything else is a bug.

**One warning worth repeating verbatim: anchor your layer regex.**
`default_layer_regex` (or `model.layer_regex` in a config) is matched with
`re.search`, not `fullmatch` -- an unanchored pattern like `blocks\.\d+$`
will also match inside `decoder_blocks.0` and silently merge two distinct
stacks (`CLAUDE.md` sec 11.30). Prefer `^blocks\.\d+$` or a prefix specific
enough that nothing else in the checkpoint could match it, and check the
match list with `run.py --discover-layers <your_name>` before trusting it.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .mock import MockBlackBoxAdapter

# --- Registration: read by AST at discovery time, never executed. Change
# both before copying this file into contrib/ -- ADAPTER_NAME must not
# collide with an existing registry name (built-in or another contrib file),
# and ADAPTER_CLASS must exactly match the class name below.
ADAPTER_NAME = "template"
ADAPTER_CLASS = "TemplateAdapter"


class TemplateAdapter(MockBlackBoxAdapter):
    """Rename this class (and update ADAPTER_CLASS above) before real use."""

    # ---- Tier 1: uncomment and fill in to unlock extract/l1/l2/attention/... ----
    #
    # @property
    # def module(self):
    #     """The torch module hooks attach to -- usually the loaded model itself."""
    #     return self._model
    #
    # def prepare(self, contexts: np.ndarray) -> Any:
    #     """Raw [B, context_len] float32 -> whatever your model's forward wants."""
    #     return self._tokenizer(contexts)
    #
    # def forward(self, prepared: Any) -> None:
    #     """One forward pass so capture hooks fire. Discard the return value --
    #     hooks are how activations are read, not this method's output."""
    #     with torch.no_grad():
    #         self._model(prepared)
    #
    # def token_time_spans(self) -> np.ndarray:
    #     """Per-token [start, end) time coverage, shape [n_tokens, 2].
    #
    #     OPTION A -- fixed patch (TimesFM-shaped: one token per `patch`
    #     contiguous timesteps, no overlap, no gaps):
    #         n_tokens = self.data_cfg.context_len // self.patch
    #         starts = np.arange(n_tokens) * self.patch
    #         return np.stack([starts, starts + self.patch], axis=1).astype(np.float32)
    #
    #     OPTION B -- per-step quantized (Chronos-shaped: one token per
    #     timestep, so each span has width 1):
    #         n_tokens = self.data_cfg.context_len
    #         starts = np.arange(n_tokens)
    #         return np.stack([starts, starts + 1], axis=1).astype(np.float32)
    #
    #     If your tokenizer is neither -- e.g. it reads a set of DISJOINT lags
    #     per token rather than one interval -- do not force one of the above.
    #     Measure it instead: `run.py --discover-spans <your_name>` empirically
    #     probes the map with an impulse sweep and tells you which of the two
    #     refusal gates (contrast vs contiguity) would fire, if either
    #     (ROADMAP.md sec 19 G2). Raise `models.base.NotTimeLocalized` from
    #     here if it does; that is a correct, useful contribution.
    #     raise self._unavailable("token_time_spans", 1)

    # ---- Tier 2: only needed if your model does NOT process the whole
    # context in one forward pass (every adapter in this repo so far does,
    # so the base class default of True is almost always correct). ----
    #
    # single_pass_context = False

    # ---- Tier 3: uncomment to unlock per-head ablation + attention patterns ----
    #
    # def attention_info(self) -> Optional[list]:
    #     """One dict per captured block: {'block', 'o_proj', 'n_heads', 'head_dim'}.
    #     `o_proj` must name a real nn.Linear whose INPUT is the concatenated
    #     heads -- that is the tensor head ablation slices. Return None if your
    #     architecture has no such hookable projection."""
    #     return None
    #
    # def attention_patterns(self, prepared: Any) -> Optional[dict]:
    #     """{block_name: tensor [B, n_heads, T, T]} of softmax attention
    #     weights for the given prepared batch. Return None if your framework
    #     fuses attention into a kernel with no exposed intermediate weights
    #     you can recover (TimesFM's adapter shows how to temporarily swap in
    #     an unfused implementation for one forward call, if that's available
    #     to you -- CLAUDE.md sec 6.2's TimesFM correction)."""
    #     return None

    # ---- Optional, any tier: only override if genuinely true for your model ----
    #
    # def mlp_info(self) -> Optional[dict]:
    #     """block capture name -> qualified MLP module name, pre-residual."""
    #     return None
    #
    # def cross_attention_patterns(self, prepared: Any) -> Optional["torch.Tensor"]:
    #     """Encoder-decoder only: first-decode-step cross-attention, shape
    #     [n_decoder_layers, B, n_heads, T_enc]. None for decoder-only models."""
    #     return None
