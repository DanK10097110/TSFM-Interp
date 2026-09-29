"""Measure a model's token->time map instead of trusting the adapter's declaration.

`ROADMAP.md` sec 16 E3. `ModelAdapter.token_time_spans()` is the single thing
every cross-architecture comparison in this repo rests on (`CLAUDE.md` sec
6.3), and today it is *asserted* by whoever wrote the adapter and checked by
`impulse_alignment_check`, which answers only "is the declaration consistent
with the model's behavior, yes or no". That check is this algorithm with the
answer thrown away: it already perturbs one timestep per window and finds
which pooled window moved most. Keeping the argmax instead of collapsing it
to a hit rate turns a pass/fail into the map itself.

Why this matters beyond tidiness: a declared map can only be written by
someone who has read the checkpoint's tokenizer source, so "add a model"
is currently an adapter-authoring task. A measured map makes it a run.

Three properties of the measurement are load-bearing and none is optional:

- **The impulse is relative to the probe signal's own amplitude, never
  absolute.** `CLAUDE.md` sec 11.16 is what an absolute constant costs: an
  8.0 impulse re-quantized 472 of 513 of Chronos-T5's tokens through its
  context-adaptive bin edges, and the resulting near-chance diagonal-hit
  fraction read as a broken adapter for as long as nobody measured it.
- **Amplitude is swept, not fixed.** Sundial's alignment is genuinely
  amplitude-dependent (sec 11.22) and Chronos's tokenizer rescales globally
  (sec 11.16), so any single amplitude gives a per-model-arbitrary answer.
  Agreement across amplitudes is reported as its own number; a map that
  moves with the probe is not a map.
- **Refusal is a real outcome.** A model whose tokens are not time-localized
  at all -- a spectral tokenizer, latent-query attention -- has no correct
  span map, and inventing one would silently corrupt every pooled
  cross-model number downstream. `diffuseness` exists to make that case
  detectable, and `CLAUDE.md` sec 12's "L0 only" envelope edge is what it
  routes to.

The captured layer is deliberately the **first** one: attention mixes
positions with depth, so the shallowest block is where a perturbation is
still closest to where it entered. Later blocks would blur the argmax and
inflate diffuseness for models that are perfectly localized at their input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..models.base import ModelAdapter
from ..utils import log
from .hooks import ActivationCatcher


@dataclass
class SpanDiscovery:
    """A measured token->time map plus the numbers that say whether to believe it.

    `spans` follows the same `[n_tokens, 2]` half-open `[start, end)` contract
    `ModelAdapter.token_time_spans` declares, so a caller can substitute one
    for the other. A token no probed timestep selected gets a zero-width span
    and is listed in `flagged_tokens` -- an empty span is not silently
    widened to something plausible.
    """
    spans: np.ndarray
    diffuseness: float
    per_amplitude_agreement: float
    flagged_tokens: list
    n_tokens: int
    amplitudes: list
    stride: int
    concentration: float
    contrast: float = 0.0
    participation_ratio: float = 0.0
    per_amplitude_spans: dict = field(default_factory=dict)
    layer: str = ""
    empty_tokens: list = field(default_factory=list)
    noncontiguous_tokens: list = field(default_factory=list)
    contiguity: float = 1.0

    def is_time_localized(self, min_contrast: float = 4.0) -> bool:
        """Whether these spans are trustworthy enough to pool activations on.

        Gates on `contrast` (peak delta over mean off-peak delta), **not** on
        `diffuseness`, and the reason is a measured one rather than a
        preference. `diffuseness` is share-of-mass-off-the-peak, so a fixed
        per-token leakage costs `(n_tokens - 1) x` of it: at the live
        acceptance run (sec 16 E3's Findings) Chronos-T5-Small scored
        diffuseness 0.794 with a **bit-exact** span map, purely because it has
        512 one-timestep tokens for the leakage to spread across, and a 0.5
        cut on that number would have refused three of the five adapters whose
        maps had just been proven exact.

        `contrast` inverts the dependence in the direction a gate needs: its
        **refusal end is pinned near 1.0 regardless of token count** (a
        genuinely uniform response has peak == mean off-peak by definition),
        while a localized model's value only grows as tokens are added. So
        the dangerous case has a fixed signature and the safe case gets
        further from it, rather than the reverse. Measured: 1.07 for a
        deliberately non-localized net, 12.0 for the smallest real localized
        case (a 4-token mock), 16.4-320 everywhere else.

        Deliberately a method on the result rather than a flag computed at
        discovery time: the threshold is a policy call belonging to whoever
        consumes the map (`GenericHFAdapter` vs. a human reading a table),
        and baking one number into the artifact would make a later
        recalibration silently rewrite what old artifacts meant
        (`CLAUDE.md` sec 11.24's class of trap). `diffuseness` is still
        recorded, unchanged, so no already-reported value moves.
        """
        return self.contrast >= min_contrast

    def is_contiguous(self, min_contiguity: float = 0.95) -> bool:
        """Whether each token reads ONE interval rather than a set of lags.

        A second, independent gate, and deliberately not folded into
        `is_time_localized`: the two failures are orthogonal and want
        different messages. A diffuse model has no time structure to pool;
        a **sharp but non-contiguous** model has plenty -- it simply does
        not read contiguous intervals, so `[min, max]` describes a range it
        mostly does not touch (ROADMAP.md sec 19 G2's Lag-Llama case).
        Coercing that into a span is not a lossy approximation, it is a
        fabrication, and contrast alone cannot see it: every one of those
        lags can be a clean peak.

        `contiguity` is the fraction of **non-empty** tokens whose owned
        timesteps have no holes, at the worst probed amplitude. Empty tokens
        are excluded rather than counted as failures -- see
        `_spans_from_argmax` for why those two are not the same event.
        """
        return self.contiguity >= min_contiguity

    def refusal_reason(self, min_contrast: float = 4.0,
                       min_contiguity: float = 0.95) -> str | None:
        """`None` if these spans may be pooled on, else which gate refused and why.

        One place that knows both gates, so a caller cannot check contrast,
        forget contiguity, and admit a Lag-Llama-shaped model by omission.
        Contrast is reported first when both fail: a diffuse response makes
        the contiguity number meaningless rather than merely also-bad.
        """
        if not self.is_time_localized(min_contrast):
            return (f"impulse response is not time-localized (peak:pedestal contrast "
                    f"{self.contrast:.2f}, below the {min_contrast:.2f} floor); its "
                    f"tokens do not each read one contiguous time interval")
        if not self.is_contiguous(min_contiguity):
            return (f"tokens are time-localized but not to contiguous intervals "
                    f"({len(self.noncontiguous_tokens)} of "
                    f"{self.n_tokens - len(self.empty_tokens)} non-empty tokens read "
                    f"disjoint timesteps; contiguity {self.contiguity:.3f}, below the "
                    f"{min_contiguity:.3f} floor); the pooling premise does not hold")
        return None

    def to_dict(self) -> dict:
        return {"spans": self.spans.tolist(), "diffuseness": self.diffuseness,
                "per_amplitude_agreement": self.per_amplitude_agreement,
                "flagged_tokens": self.flagged_tokens, "n_tokens": self.n_tokens,
                "amplitudes": list(self.amplitudes), "stride": self.stride,
                "concentration": self.concentration, "contrast": self.contrast,
                "participation_ratio": self.participation_ratio, "layer": self.layer,
                "empty_tokens": self.empty_tokens,
                "noncontiguous_tokens": self.noncontiguous_tokens,
                "contiguity": self.contiguity}


def _probe_signal(context_len: int) -> np.ndarray:
    """The same sine probe `impulse_alignment_check` uses, for comparability.

    Sharing the probe means a `--discover-spans` table and a
    `--check-alignment` hit rate on the same checkpoint are two readings of
    one measurement rather than two measurements that might disagree for
    reasons unrelated to alignment.
    """
    t = np.arange(context_len, dtype=np.float32)
    return np.sin(2 * np.pi * t / (context_len / 4)).astype(np.float32)


def _token_deltas(adapter: ModelAdapter, layer: str, base: np.ndarray,
                  probe_positions: np.ndarray, impulse: float,
                  batch_size: int) -> np.ndarray:
    """Per (probed timestep, token position) L2 norm of the activation change.

    Runs the unperturbed row in every chunk rather than once, because a model
    whose forward pass depends on batch composition (any input-adaptive
    normalization, and Chronos's whole-sequence quantization is one) would
    otherwise have its baseline computed under different conditions than the
    perturbed rows it is subtracted from -- a confound that looks exactly
    like diffuseness and is not.
    """
    out = []
    for start in range(0, len(probe_positions), batch_size - 1):
        chunk = probe_positions[start:start + batch_size - 1]
        batch = np.tile(base, (len(chunk) + 1, 1))
        for i, t in enumerate(chunk):
            batch[i + 1, t] += impulse
        with torch.no_grad(), ActivationCatcher(adapter.module, [layer]) as catcher:
            adapter.forward(adapter.prepare(batch))
            hidden = catcher.collect()[layer]
        hidden = _tokens_only(adapter, layer, hidden).float()
        out.append((hidden[1:] - hidden[0:1]).norm(dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _tokens_only(adapter: ModelAdapter, layer: str, hidden: torch.Tensor) -> torch.Tensor:
    """Strip specials/padding when the adapter knows how, else take the tensor as-is.

    `postprocess_tokens` is a statement about *which positions are real
    tokens*, which is structural, not about which time each token covers,
    which is the thing being measured here -- so using it does not make the
    discovery circular. It is wrapped because an adapter with no declared
    spans at all (the `GenericHFAdapter` case this module exists to enable)
    cannot run the base implementation's token-count assertion.
    """
    try:
        return adapter.postprocess_tokens(layer, hidden)
    except Exception as exc:
        log.info("span discovery '%s': postprocess_tokens unavailable (%s); "
                 "using raw token positions", adapter.name, exc)
        return hidden


def _spans_from_argmax(owner: np.ndarray, positions: np.ndarray, n_tokens: int,
                       stride: int) -> tuple:
    """Invert a timestep->token map into contiguous per-token spans, flagging what won't invert.

    The `ModelAdapter` contract requires each token to cover **one contiguous
    interval**, so a raw argmax map has to be coerced -- and the coercion is
    where the information about whether the model actually satisfies that
    contract lives. A token whose chosen timesteps have holes in them is not
    described by `[min, max]`; widening it silently would hand the pooling
    matrix a fiction. Such tokens are flagged, and the flagged fraction is
    what a caller reads to decide whether the whole map is meaningful.

    The two reasons a token cannot be described by `[min, max]` are returned
    **separately**, because they mean opposite things (ROADMAP.md sec 19 G2).
    A token no probed timestep selected is *empty* -- routine and benign: a
    stripped special, a padding position, a token the stride never landed on.
    A token whose owned timesteps have holes is *non-contiguous* -- the model
    reads a set of disjoint lags at one position, which is exactly the case
    (Lag-Llama-style lag features) that is sharply time-localized and still
    breaks the pooling premise. Pooled into one `flagged_tokens` count, a
    model with the second problem is indistinguishable from one with a few
    specials, and would pass a contrast gate cleanly.
    """
    spans = np.zeros((n_tokens, 2), dtype=np.float64)
    empty, noncontiguous = [], []
    for tok in range(n_tokens):
        chosen = positions[owner == tok]
        if len(chosen) == 0:
            empty.append(tok)
            continue
        lo, hi = int(chosen.min()), int(chosen.max())
        spans[tok] = (lo, hi + stride)
        # A contiguous owner set, sampled at `stride`, has exactly
        # (hi - lo) / stride + 1 members. Fewer means holes: some timestep
        # inside this token's own range was claimed by a different token.
        if len(chosen) != (hi - lo) // stride + 1:
            noncontiguous.append(tok)
    return spans, empty, noncontiguous


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-token intersection-over-union between two `[n_tokens, 2]` span tables."""
    n = min(len(a), len(b))
    inter = np.maximum(0.0, np.minimum(a[:n, 1], b[:n, 1]) - np.maximum(a[:n, 0], b[:n, 0]))
    union = np.maximum(a[:n, 1], b[:n, 1]) - np.minimum(a[:n, 0], b[:n, 0])
    return np.divide(inter, union, out=np.zeros(n), where=union > 0)


def discover_spans(adapter: ModelAdapter, context_len: int | None = None,
                   amplitudes: tuple = (0.25, 0.15, 0.05), stride: int = 1,
                   layer: str | None = None,
                   batch_size: int | None = None) -> SpanDiscovery:
    """Measure which time interval each token position actually reads.

    Sweeps a relative-amplitude impulse across the context (every `stride`
    timesteps), records which token position of the first captured block
    moves most, and inverts that map into contiguous spans. Repeats the whole
    sweep at each amplitude and reports how much the resulting maps agree:
    the returned `spans` come from the **largest** amplitude, which has the
    best signal-to-noise, but a low `per_amplitude_agreement` means that
    choice is arbitrary and the map should not be trusted regardless of how
    clean it looks.

    `diffuseness` is 0 when every probed timestep moves exactly one token and
    1 when it moves every token equally, normalized so the "equally" end does
    not drift with token count. It is the number the refusal path reads.
    """
    adapter.ensure_loaded()
    context_len = context_len or adapter.data_cfg.context_len
    layer = layer or adapter.layer_names()[0]
    batch_size = batch_size or max(2, int(getattr(adapter.cfg, "batch_size", 8)))
    base = _probe_signal(context_len)
    impulse_unit = float(np.max(np.abs(base)))
    positions = np.arange(0, context_len, stride)

    per_amp_spans, per_amp_flagged, per_amp_conc = {}, {}, {}
    per_amp_contrast, per_amp_pr = {}, {}
    per_amp_empty, per_amp_noncontig, per_amp_contiguity = {}, {}, {}
    n_tokens = 0
    for amp in amplitudes:
        deltas = _token_deltas(adapter, layer, base, positions,
                               amp * impulse_unit, batch_size)
        n_tokens = deltas.shape[1]
        total = deltas.sum(axis=1)
        top = deltas.max(axis=1)
        # A probe that moved nothing carries no information about ownership;
        # counting it as "perfectly diffuse" would penalize a model for a
        # timestep its tokenizer legitimately discards (padding, a stripped
        # special) rather than for being non-localized.
        live = total > 0
        share = np.divide(top, total, out=np.zeros_like(total), where=live)
        owner = np.where(live, deltas.argmax(axis=1), -1)
        spans, empty, noncontig = _spans_from_argmax(owner, positions, n_tokens, stride)
        per_amp_spans[amp] = spans
        per_amp_flagged[amp] = sorted(empty + noncontig)
        per_amp_empty[amp], per_amp_noncontig[amp] = empty, noncontig
        n_live_tokens = n_tokens - len(empty)
        per_amp_contiguity[amp] = (1.0 - len(noncontig) / n_live_tokens
                                   if n_live_tokens else 1.0)
        per_amp_conc[amp] = float(share[live].mean()) if live.any() else 0.0
        # Peak-to-pedestal contrast and the participation ratio, both over the
        # live probes only. `contrast` is what the refusal gate reads (see
        # `SpanDiscovery.is_time_localized` for why it and not `concentration`);
        # `participation_ratio` is recorded beside it because it answers a
        # different question -- how many tokens the response is effectively
        # spread over -- and the two disagreeing is itself informative.
        d_live, tot_live, top_live = deltas[live], total[live], top[live]
        if len(d_live):
            pedestal = (tot_live - top_live) / max(n_tokens - 1, 1)
            per_amp_contrast[amp] = float(np.mean(top_live / np.maximum(pedestal, 1e-12)))
            pr = tot_live ** 2 / np.maximum((d_live ** 2).sum(axis=1), 1e-12)
            per_amp_pr[amp] = float(pr.mean() / max(n_tokens, 1))
        else:
            per_amp_contrast[amp], per_amp_pr[amp] = 0.0, 1.0

    uniform = 1.0 / max(n_tokens, 1)
    concentration = float(np.mean(list(per_amp_conc.values())))
    diffuseness = float(np.clip((1.0 - concentration) / max(1.0 - uniform, 1e-9), 0.0, 1.0))
    # The gate reads the WORST amplitude, not the mean: an impulse size at
    # which the response smears is a real property of the model, and averaging
    # it away with two better amplitudes is how a probe passes a model it
    # should have flagged.
    contrast = float(min(per_amp_contrast.values())) if per_amp_contrast else 0.0
    participation_ratio = float(max(per_amp_pr.values())) if per_amp_pr else 1.0
    # Worst amplitude, for the same reason `contrast` takes the worst: an
    # impulse size at which tokens start reading disjoint timesteps is a
    # property of the model, not of the probe.
    contiguity = float(min(per_amp_contiguity.values())) if per_amp_contiguity else 1.0

    amps = list(amplitudes)
    pairs = [float(_iou(per_amp_spans[a], per_amp_spans[b]).mean())
             for i, a in enumerate(amps) for b in amps[i + 1:]]
    agreement = float(np.mean(pairs)) if pairs else 1.0

    chosen = amps[0]
    result = SpanDiscovery(
        spans=per_amp_spans[chosen], diffuseness=diffuseness,
        per_amplitude_agreement=agreement, flagged_tokens=per_amp_flagged[chosen],
        n_tokens=n_tokens, amplitudes=amps, stride=stride,
        concentration=concentration, contrast=contrast,
        participation_ratio=participation_ratio,
        per_amplitude_spans={a: s.tolist() for a, s in per_amp_spans.items()},
        layer=layer, empty_tokens=per_amp_empty[chosen],
        noncontiguous_tokens=per_amp_noncontig[chosen], contiguity=contiguity)
    log.info("span discovery '%s' @%s: %d tokens, contrast=%.2f, diffuseness=%.3f, "
             "contiguity=%.3f, amplitude agreement=%.3f, %d flagged (%d empty, "
             "%d non-contiguous)",
             adapter.name, layer, n_tokens, contrast, diffuseness, contiguity,
             agreement, len(result.flagged_tokens), len(result.empty_tokens),
             len(result.noncontiguous_tokens))
    return result


def compare_declared(adapter: ModelAdapter, discovered: SpanDiscovery) -> dict:
    """Cross-check an adapter's declared spans against the measured ones (E3(d)).

    Returns per-token IoU and its mean. A hand-written adapter that has gone
    stale against a library update is exactly the failure `CLAUDE.md` sec 11.8
    describes, and this is the check that catches it as a number rather than
    as a downstream result that looks slightly wrong for months. Returns
    `available: False` rather than raising for an adapter that declares no
    spans -- that is the generic-adapter case, not an error.
    """
    try:
        declared = adapter.token_time_spans()
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    iou = _iou(np.asarray(declared, dtype=np.float64), discovered.spans)
    return {"available": True, "mean_iou": float(iou.mean()),
            "per_token_iou": iou.tolist(),
            "n_declared": int(len(declared)), "n_discovered": int(discovered.n_tokens),
            "worst_token": int(iou.argmin()) if len(iou) else -1,
            "worst_iou": float(iou.min()) if len(iou) else 0.0}
