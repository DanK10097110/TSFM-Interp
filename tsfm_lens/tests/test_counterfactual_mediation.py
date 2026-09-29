"""ROADMAP.md sec 37.9 P6a -- measurement-side input response tests.

Mirrors `test_shared_input_agreement.py`'s style: a stub adapter/SAE pair
with `capture_raw_tokens` monkeypatched at the module level, so the
DECISION logic (series selection, dose ladder, response/null scoring, BH)
runs through the real code without a real forward pass. The generator side
(`tsfm_benchmark.build_pipeline.counterfactual.regenerate`) is used for
real -- it is Part A's own machinery, already tested in
`tsfm_benchmark/tests/test_counterfactual.py` -- so these tests exercise the
genuine dose ladder, not a second, hand-rolled one.

One token spans the whole context (`token_time_spans` = `[[0, CONTEXT_LEN]]`,
`alignment.window = CONTEXT_LEN`), so `pooling_matrix`/`align` (both real,
not mocked -- pure tensor ops) collapse to an identity: the "window-pooled
activation" the mock SAE sees is exactly the raw context. The mock SAE's
`encode` is a hand-written function of that raw context, so every feature's
exact value -- and thus its exact per-series Spearman response -- is known
in closed form, not merely "plausible".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The env's editable `tsfm_benchmark` install maps to the MAIN checkout, not
# this worktree (CLAUDE.md sec 3 / the P6a spec's own rule) -- prepend the
# worktree's own repo root so the ordinary sys.path search finds ITS
# `tsfm_benchmark/build_pipeline/counterfactual.py` ahead of the editable
# install's meta-path finder, which would otherwise resolve to a copy that
# does not have this module at all. A plain sys.path insert is not enough
# when this file runs in the SAME pytest session as another test module that
# already imported `tsfm_benchmark` transitively (e.g. `test_shared_input_
# agreement.py` -> `sae/ground_truth.py` -> `tsfm_benchmark.build_pipeline.
# seal`): Python resolves `import tsfm_benchmark` from `sys.modules` first,
# so the earlier (main-checkout) module object wins regardless of sys.path
# order (CLAUDE.md sec 11.52's import-cache trap, one level removed).
#
# Deleting the stale `sys.modules` entries (an earlier version of this fix)
# "worked" in isolation but broke a DIFFERENT, unrelated suite the same way
# when applied on the `tsfm_benchmark` side (`tsfm_benchmark/tests/
# test_counterfactual.py`'s own fix, see its comment): a test collected
# earlier in the session that had already bound a module-level reference for
# `monkeypatch` gets orphaned when that module is purged, and a later,
# unrelated re-import creates a second, unpatched copy. `test_sample_rows.py`
# is the only other file in this suite that mentions `tsfm_benchmark` (in a
# comment, not an import) and nothing here monkeypatches into it, so a purge
# has not been observed to break anything in THIS suite -- but the same
# non-destructive fix applies just as well and removes the risk entirely:
# extend `build_pipeline`'s own `__path__` in place instead of touching
# `sys.modules`. `counterfactual` does not exist in the main checkout, so
# there is no name collision to order between the two search locations.
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WORKTREE_ROOT))
import tsfm_benchmark.build_pipeline as _bp

_worktree_bp_dir = str(_WORKTREE_ROOT / "tsfm_benchmark" / "build_pipeline")
if _worktree_bp_dir not in _bp.__path__:
    _bp.__path__.append(_worktree_bp_dir)

from tsfm_benchmark.build_pipeline.generators import random_parametric

from tsfm_lens.sae import counterfactual as cfmod
from tsfm_lens.sae import shared_input_agreement as sia

CONTEXT_LEN = 32
DICT_SIZE = 4
MODEL, LAYER = "M", "blocks.0"
TARGET = f"{MODEL}/{LAYER}"


# ---------------------------------------------------------------------------
# Fixtures: real random_parametric samples (seasonal_dominant guarantees a
# seasonality on every draw -- see tsfm_benchmark/tests/test_counterfactual.py),
# a stub adapter/SAE/hub/store, and a lightweight data object.
# ---------------------------------------------------------------------------

def _make_samples(seeds):
    return [random_parametric(seed=s, archetypes=["seasonal_dominant"]) for s in seeds]


class _Data:
    def __init__(self, samples, context_len=CONTEXT_LEN):
        self.context_len = context_len
        self._values = np.stack([np.asarray(s.values[:context_len], dtype=np.float32)
                                 for s in samples])
        archetypes = [s.provenance.generator_params.get("archetype") for s in samples]
        self.meta = pd.DataFrame({"series_id": [s.sample_id for s in samples],
                                  "archetype": archetypes})

    def contexts(self):
        return self._values


class _Adapter:
    def __init__(self):
        self.module = object()
        self.cfg = type("C", (), {"batch_size": 999})()

    def ensure_loaded(self):
        pass

    def token_time_spans(self):
        return np.array([[0.0, float(CONTEXT_LEN)]])


class _SAE:
    def __init__(self, encode_fn, dict_size=DICT_SIZE, d_in=CONTEXT_LEN):
        self._encode_fn = encode_fn
        self.dict_size = dict_size
        self.d_in = d_in

    def encode(self, x):
        return self._encode_fn(x)

    def to(self, device):
        return self


class _Hub:
    def __init__(self, adapter):
        self._adapter = adapter

    def get(self, name):
        return self._adapter


class _Store:
    def __init__(self, pooled, raw_pooled=None):
        self._pooled = pooled
        # `space="act"` (raw, pre-SAE): with `alignment.window == CONTEXT_LEN`
        # (one window spans the whole context, `_Cfg.alignment.window`) and
        # `_Adapter.token_time_spans` a single full-context span, `align`'s
        # pooling is the identity, so the "stored" series-level pooled
        # activation the real store would hold is exactly the context itself
        # -- matching what a fresh `_raw_pooled_activation` call computes
        # through the same mocked `capture_raw_tokens`. Defaults to the
        # contexts already implied by `pooled`'s own construction when a
        # test does not care about the raw-activation diagnostic.
        self._raw_pooled = raw_pooled

    def load(self, model, layer, level="series", space="sae", rows=None, **kw):
        src = self._pooled if space == "sae" else self._raw_pooled
        if src is None:
            raise KeyError(f"_Store: no {space!r} array configured for this test")
        return src if rows is None else src[np.asarray(rows)]


class _Cfg:
    class run:
        seed = 0
    class data:
        path = "/nonexistent/corpus"
    class alignment:
        window = CONTEXT_LEN
    class concepts:
        cf_max_series = 64
        cf_doses = [0.0, 0.5, 1.0, 1.5, 2.0]
        cf_n_null = 3
        transfer_fdr_q = 0.05
        cf_encode_min_spearman = 0.98


def _std_encode(x: torch.Tensor) -> torch.Tensor:
    """Feature 0 reads the input's amplitude (its std) directly -- the
    planted responsive feature for `seasonal_amplitude`. Features 1-3 are
    deterministic but weaker/mixed-sign wobbles of the same quantity
    (`sin(freq*std+phase)`, tuned so their Spearman response against the
    dose ladder is nowhere near the perfect +1.0 the linear readout gets),
    giving a genuine, non-degenerate matched-null pool."""
    std = x.std(dim=1, unbiased=False)
    d1 = torch.sin(3.1 * std + 0.3)
    d2 = torch.sin(7.7 * std + 1.1)
    d3 = torch.sin(11.3 * std + 2.7)
    return torch.stack([std, d1, d2, d3], dim=1)


def _constant_encode(x: torch.Tensor) -> torch.Tensor:
    """Every feature ignores the input entirely -- the "absent input change"
    fixture: whatever the dose, the encoded score cannot move."""
    b = x.shape[0]
    return torch.ones(b, DICT_SIZE, dtype=torch.float32) * torch.arange(
        1, DICT_SIZE + 1, dtype=torch.float32)


def _wire(monkeypatch, samples, encode_fn, cf_n_null=3, targets=None, store_perturb=None):
    data = _Data(samples)
    adapter = _Adapter()
    sae = _SAE(encode_fn)

    calls = []

    def _fake_capture(adapter_, contexts, layers, autocast=False):
        calls.append({"autocast": autocast, "n": len(contexts)})
        t = torch.as_tensor(np.asarray(contexts, dtype=np.float32))
        return {layers[0]: t[:, None, :]}

    monkeypatch.setattr(cfmod, "capture_raw_tokens", _fake_capture)
    monkeypatch.setattr(sia, "load_sae_checkpoint", lambda path: sae)
    monkeypatch.setattr(sia, "load_all_windows",
                        lambda store, model, layer: data.contexts())
    monkeypatch.setattr(sia, "reach_probe",
                        lambda cfg, adapter_, layer, data_, device: {"reachable": True, "reason": ""})

    pooled = _std_encode_reference(encode_fn, data.contexts())
    stored_pooled = store_perturb(pooled.copy()) if store_perturb is not None else pooled
    # Identity pooling (one window == the whole context, see `_Store`'s own
    # comment), so the real store's "act" space would hold exactly the
    # contexts themselves.
    store = _Store(stored_pooled, raw_pooled=data.contexts().astype(np.float64))
    hub = _Hub(adapter)

    cfg = _Cfg
    cfg.concepts.cf_n_null = cf_n_null

    monkeypatch.setattr(cfmod, "_load_samples_by_id",
                        lambda load_sealed_fn, path: {s.sample_id: s for s in samples})

    atlas = {"rows": [{"model": MODEL, "layer": LAYER, "feature": 0, "concept": 1}]}
    atlas_path = None
    return cfg, hub, store, data, atlas, calls


def _std_encode_reference(encode_fn, contexts_native: np.ndarray) -> np.ndarray:
    """The store's persisted dose=1.0 series-level SAE features, computed with
    the SAME encode function as the mock so the encode-check trivially
    matches (this fixture is not testing the encode-check itself)."""
    with torch.no_grad():
        return encode_fn(torch.as_tensor(contexts_native, dtype=torch.float32)).numpy().astype(np.float64)


def _write_atlas(run_dir: Path, atlas: dict) -> None:
    from tsfm_lens.utils import save_json
    save_json(run_dir / "sae" / "concept_atlas.json", atlas)


# ---------------------------------------------------------------------------
# Encode-check fixtures (ROADMAP sec 37.9 P6a review): seeds chosen so their
# std values (feature 0's own quantity, `_std_encode`) are already in
# ascending order, with a minimum gap of 0.142 between consecutive values --
# 0.199, 0.506, 0.698, 0.840, 1.279, 2.044 -- so a `store_perturb` that only
# nudges values by ~1e-3 relative cannot flip any pair's rank, and one that
# reverses row order flips every pair's rank (Spearman exactly -1.0).
# ---------------------------------------------------------------------------

_ENCODE_CHECK_SEEDS = [7, 21, 12, 25, 10, 9]


def _small_order_preserving_offset(pooled: np.ndarray) -> np.ndarray:
    """A uniform constant additive offset to the STORED array only: it
    cannot change which series ranks where for ANY column (adding the same
    constant to every value never inverts a pairwise comparison), which is
    the property a real bf16-precision perturbation also has when it stays
    below a feature's own resolution -- but 0.05 is well above `atol=1e-3,
    rtol=1e-2`'s allowance for every value in this fixture (largest allowed
    diff here is ~0.0214, for the biggest std value 2.044), so the OLD
    whole-vector `allclose` gate would reject it while the new per-concept
    Spearman gate (correctly) does not care, because rank is exactly
    preserved. This is deliberately not modeled as literal bf16 rounding
    (whose error scales down for values near zero, so it would not
    reliably violate the old gate for THIS demo) -- the point demonstrated
    is the general one CLAUDE.md sec 8 names: a fixed absolute/relative
    epsilon on raw values penalizes a perturbation that is functionally
    harmless (order-preserving) and real bf16 noise on a sparse dictionary's
    near-zero entries does exactly this in practice (measured: 0.17%
    relative on the raw activation still failed a whole-vector encode-check
    by up to 0.8 absolute on a real run, ROADMAP sec 37.9 P6a's go/no-go)."""
    return pooled + 0.05


def _reversed_rows(pooled: np.ndarray) -> np.ndarray:
    """Row-reversal of the STORED array: every row now holds a DIFFERENT
    series' feature vector than the fresh recompute does at that index --
    the general shape of "wrong space" (a misaligned array, a different
    layer's activations, a stale row mapping). For a fixture built in
    std-ascending order this decorrelates feature 0 completely (Spearman
    -1.0), never merely perturbs it."""
    return pooled[::-1].copy()


def test_encode_check_small_precision_noise_passes(monkeypatch, tmp_path):
    """The scale-aware encode-check (fresh-vs-stored Spearman on each atlas
    concept part's own pooled score, not a whole-vector absolute/relative
    tolerance on raw SAE-feature values -- CLAUDE.md sec 8, 'Absolute
    epsilons') must NOT block scoring on a small, order-preserving
    perturbation that stands in for real bf16 precision noise. `_wire`'s
    reference stored array is built from the SAME encode function as the
    fresh recompute, so without `store_perturb` this would pass trivially
    with zero noise at all; perturbing only the stored copy makes the
    fresh-vs-stored comparison genuine."""
    samples = _make_samples(_ENCODE_CHECK_SEEDS)
    cfg, hub, store, data, atlas, calls = _wire(monkeypatch, samples, _std_encode, cf_n_null=3,
                                                store_perturb=_small_order_preserving_offset)
    run_dir = tmp_path
    _write_atlas(run_dir, atlas)

    out = cfmod.run_counterfactual_response(cfg, run_dir, hub, store, data, "cpu",
                                            targets=[TARGET])
    rec = out["targets"][0]
    assert rec["status"] == "ok", rec
    assert rec["encode_check"]["ok"] is True, rec["encode_check"]
    part = rec["encode_check"]["concept_parts"][0]
    assert part["concept"] == 1
    assert part["pass"] is True, part
    assert part["spearman"] is not None and part["spearman"] >= 0.98, part
    # This is the whole point of the fix (ROADMAP sec 37.9 P6a review): the
    # OLD whole-vector encode-check would have rejected this same fixture
    # (see the diagnostic it leaves behind, `sae_feature_diagnostic`) --
    # confirmed separately by reverting to `ok = sae_diag_ok` and rerunning
    # this test (planted regression, see the P6a report).
    assert rec["encode_check"]["sae_feature_diagnostic"]["ok"] is False, (
        "this fixture's offset must be big enough to fail the OLD whole-vector "
        "allclose check -- otherwise this test cannot demonstrate the new gate's "
        "value over the old one")
    scored = next(t for t in rec["tests"] if t["concept"] == 1 and t["knob"] == "seasonal_amplitude")
    assert scored["status"] == "scored", scored


def test_encode_check_wrong_space_fails(monkeypatch, tmp_path):
    """A row-reversed stored array (the shape of a misaligned/wrong-space
    array) must be caught by the same encode-check and must block scoring
    for that target, with a stated reason -- never a silent wrong-space
    score."""
    samples = _make_samples(_ENCODE_CHECK_SEEDS)
    cfg, hub, store, data, atlas, calls = _wire(monkeypatch, samples, _std_encode, cf_n_null=3,
                                                store_perturb=_reversed_rows)
    run_dir = tmp_path
    _write_atlas(run_dir, atlas)

    out = cfmod.run_counterfactual_response(cfg, run_dir, hub, store, data, "cpu",
                                            targets=[TARGET])
    rec = out["targets"][0]
    assert rec["status"] == "encode_mismatch", rec
    assert rec["encode_check"]["ok"] is False
    part = rec["encode_check"]["concept_parts"][0]
    assert part["pass"] is False, part
    assert part["spearman"] is not None and part["spearman"] < 0.98, part
    assert "reason" in rec and "reason" in part
    assert "tests" not in rec, "a blocked target must not carry any scored/excluded tests"


# ---------------------------------------------------------------------------
# 1. Identical contexts across the dose ladder -> exactly 0.0 score change.
# ---------------------------------------------------------------------------

def test_unsupported_when_tsfm_benchmark_not_importable(monkeypatch, tmp_path):
    def _raise():
        raise ImportError("no module named tsfm_benchmark (planted)")

    monkeypatch.setattr(cfmod, "_load_counterfactual_module", _raise)
    out = cfmod.run_counterfactual_response(_Cfg, tmp_path, None, None, None, "cpu")
    assert out["status"] == "unsupported"
    assert "reason" in out
    written = cfmod.load_json(cfmod.counterfactual_response_path(tmp_path))
    assert written == out


def test_identity_check_raises_on_mismatch(monkeypatch, tmp_path):
    """Dose 1.0 must reproduce `data.contexts()` bit-identically (sec 37.9
    design item 2) -- asserted inside the measurement itself. Swap which
    sample two series_ids resolve to (as if the corpus/data row mapping had
    drifted) and confirm the mismatch is a loud `AssertionError`, not a
    silently-wrong response."""
    samples = _make_samples([101, 202, 303])
    cfg, hub, store, data, atlas, calls = _wire(monkeypatch, samples, _std_encode, cf_n_null=3)
    run_dir = tmp_path
    _write_atlas(run_dir, atlas)

    swapped = {samples[0].sample_id: samples[1], samples[1].sample_id: samples[0],
              samples[2].sample_id: samples[2]}
    monkeypatch.setattr(cfmod, "_load_samples_by_id", lambda load_sealed_fn, path: swapped)

    with pytest.raises(AssertionError, match="does not reproduce the corpus series"):
        cfmod.run_counterfactual_response(cfg, run_dir, hub, store, data, "cpu",
                                          targets=[TARGET])


def test_absent_input_change_gives_zero_response(monkeypatch, tmp_path):
    """A dose ladder whose CONTEXTS are identical (the mock SAE ignores the
    input entirely, standing in for "knob on a zero-amplitude seasonality is
    refused, so build the identity directly") must encode to exactly the
    same score at every dose -- confirms `_encode_series_level` introduces no
    spurious nondeterminism (autocast, dropout, ...) that could masquerade as
    a real response."""
    contexts = np.stack([np.linspace(-1, 1, CONTEXT_LEN, dtype=np.float32)] * 5)
    adapter = _Adapter()
    sae = _SAE(_constant_encode)

    def _fake_capture(adapter_, ctx, layers, autocast=False):
        t = torch.as_tensor(np.asarray(ctx, dtype=np.float32))
        return {layers[0]: t[:, None, :]}

    monkeypatch.setattr(cfmod, "capture_raw_tokens", _fake_capture)
    pool = cfmod.pooling_matrix(adapter.token_time_spans(), CONTEXT_LEN, CONTEXT_LEN)
    feats = cfmod._encode_series_level(adapter, sae, LAYER, contexts, pool, "cpu", 64)
    assert feats.shape == (5, DICT_SIZE)
    score_change = feats.max(axis=0) - feats.min(axis=0)
    assert np.array_equal(score_change, np.zeros(DICT_SIZE)), \
        f"identical contexts produced a nonzero score change: {score_change!r}"


# ---------------------------------------------------------------------------
# 2. Full driver: a planted responsive feature clears the matched null.
# ---------------------------------------------------------------------------

def test_planted_responsive_feature(monkeypatch, tmp_path):
    samples = _make_samples([101, 202, 303])
    cfg, hub, store, data, atlas, calls = _wire(monkeypatch, samples, _std_encode, cf_n_null=3)
    run_dir = tmp_path
    _write_atlas(run_dir, atlas)

    out = cfmod.run_counterfactual_response(cfg, run_dir, hub, store, data, "cpu",
                                            targets=[TARGET])
    assert out["status"] == "ok"
    tests = out["targets"][0]["tests"]
    rec = next(t for t in tests if t["concept"] == 1 and t["knob"] == "seasonal_amplitude")
    assert rec["status"] == "scored"
    assert rec["response_mean"] == pytest.approx(1.0, abs=1e-9), rec
    assert rec["null_p95_abs"] is not None and rec["null_p95_abs"] < 1.0, rec
    assert rec["ci_excludes_zero"] is True
    assert rec["responds"] is True, rec


# ---------------------------------------------------------------------------
# 3. Encode path always uses autocast=True (no patch path exists in P6a).
# ---------------------------------------------------------------------------

def test_store_regime_used_for_encoding(monkeypatch, tmp_path):
    samples = _make_samples([101, 202])
    cfg, hub, store, data, atlas, calls = _wire(monkeypatch, samples, _std_encode, cf_n_null=2)
    run_dir = tmp_path
    _write_atlas(run_dir, atlas)

    cfmod.run_counterfactual_response(cfg, run_dir, hub, store, data, "cpu", targets=[TARGET])
    assert calls, "capture_raw_tokens was never called"
    assert all(c["autocast"] is True for c in calls), \
        f"counterfactual encoding must always run with autocast=True: {calls!r}"


# ---------------------------------------------------------------------------
# 4. A constant score across doses is recorded as undefined, never 0.
# ---------------------------------------------------------------------------

def test_constant_score_recorded_as_undefined(monkeypatch, tmp_path):
    samples = _make_samples([101, 202, 303])
    cfg, hub, store, data, atlas, calls = _wire(monkeypatch, samples, _constant_encode,
                                                cf_n_null=3)
    run_dir = tmp_path
    _write_atlas(run_dir, atlas)

    out = cfmod.run_counterfactual_response(cfg, run_dir, hub, store, data, "cpu",
                                            targets=[TARGET])
    rec = next(t for t in out["targets"][0]["tests"]
              if t["concept"] == 1 and t["knob"] == "seasonal_amplitude")
    assert rec["status"] == "undefined", rec
    assert "response_mean" not in rec, \
        "an undefined response must not carry a numeric response_mean (never silently 0)"


# ---------------------------------------------------------------------------
# 5. BH is applied within a target, across every (concept, knob) pair there.
# ---------------------------------------------------------------------------

def test_bh_applied_within_target(monkeypatch, tmp_path):
    """Two concepts at the SAME target, each its own (concept, knob) test.
    Load-bearing: `benjamini_hochberg` must be called ONCE with a dict
    holding both tests' p-values (one family), never once per test (which
    would silently skip the correction -- a lone borderline p that only
    survives in company with a second one would then wrongly read as
    'survives' or 'fails' independently of its family)."""
    samples = _make_samples([101, 202, 303])
    cfg, hub, store, data, atlas, calls = _wire(monkeypatch, samples, _std_encode, cf_n_null=3)
    atlas["rows"].append({"model": MODEL, "layer": LAYER, "feature": 1, "concept": 2})
    run_dir = tmp_path
    _write_atlas(run_dir, atlas)

    bh_calls = []
    real_bh = cfmod.benjamini_hochberg

    def _spy_bh(pvals, q):
        bh_calls.append(dict(pvals))
        return real_bh(pvals, q)

    monkeypatch.setattr(cfmod, "benjamini_hochberg", _spy_bh)

    out = cfmod.run_counterfactual_response(cfg, run_dir, hub, store, data, "cpu",
                                            targets=[TARGET])
    scored = [t for t in out["targets"][0]["tests"] if t["status"] == "scored"]
    assert len(scored) >= 2, "need at least two scored tests to exercise a real BH family"
    assert len(bh_calls) == 1, f"benjamini_hochberg must be called once per target, got {len(bh_calls)}"
    assert len(bh_calls[0]) == len(scored), \
        f"benjamini_hochberg's family must include every scored test at this target: " \
        f"{bh_calls[0].keys()} vs {[(t['concept'], t['knob']) for t in scored]}"
    for t in scored:
        assert "p_bh" in t and "survives_fdr" in t


def test_spearman_rows_matches_per_series_spearman():
    """The vectorized null path must give exactly what the per-series
    `_spearman` gives, including ties (quantized concept scores tie often)
    and a constant row (undefined, never 0). Plant: ordinal instead of
    average ranks in `_spearman_rows` fails the tied rows."""
    rng = np.random.default_rng(0)
    doses = [0.0, 0.5, 1.0, 1.5, 2.0]
    rows = rng.normal(size=(40, 5))
    rows[:10] = np.round(rows[:10])
    rows[10] = 3.0
    got = cfmod._spearman_rows(rows, doses)
    for i in range(rows.shape[0]):
        want = cfmod._spearman(rows[i], np.asarray(doses))
        if want is None:
            assert np.isnan(got[i])
        else:
            assert got[i] == pytest.approx(want, abs=1e-12)
    assert np.isnan(got[10])


def test_bh_family_satisfiability_recorded(monkeypatch, tmp_path):
    """With few null draws, no test can survive BH however strong its
    response; that must be recorded, never left to read as a negative.
    Plant: `satisfiable` always True fails this."""
    samples = _make_samples([101, 202, 303])
    cfg, hub, store, data, atlas, calls = _wire(monkeypatch, samples, _std_encode, cf_n_null=3)
    atlas["rows"].append({"model": MODEL, "layer": LAYER, "feature": 1, "concept": 2})
    _write_atlas(tmp_path, atlas)
    out = cfmod.run_counterfactual_response(cfg, tmp_path, hub, store, data, "cpu",
                                            targets=[TARGET])
    fam = out["targets"][0]["bh_family"]
    assert fam["n_tests"] >= 2
    assert fam["min_attainable_q_single"] == pytest.approx(fam["n_tests"] / 4)
    assert fam["satisfiable"] is False
