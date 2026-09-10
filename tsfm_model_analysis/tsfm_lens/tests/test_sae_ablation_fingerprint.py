"""ROADMAP.md sec 27: the feature-ABLATION fingerprint -- what each SAE
feature causally does on the series it actually fires on.

This is deliberately a second pass beside `feature_response_fingerprints`
(Component A), not a replacement for it, and the two differ on two axes:

  intervention   ablate the atom out of the reconstruction, vs INJECT a
                 direction at a fixed strength
  conditioning   the atom's OWN top-firing series, vs a representative
                 family-stratified sample

Both matter for the thing this pass is for -- distinguishing two features
that fire on the same series for different reasons -- and neither is
answerable from the other's numbers, which is why sec 2.1 says keep both.

Every fixture here plants a known answer. The load-bearing negative is (3):
the null must be recomputed over EACH candidate's own rows. A single pooled
p95 is the natural implementation and is wrong for the same reason
`CLAUDE.md` sec 11.33's share-of-total statistic was wrong -- a candidate
whose top-firing series happen to sit in a high-response regime would clear
a pooled threshold on that regime alone, and one in a quiet regime would be
refused despite a real effect. The plant makes the two answers disagree.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae import response as R


HORIZON = 4
D_IN = 3
DICT = 4
N_SERIES = 6

# Rows 0-2 sit in a loud regime, rows 3-5 in a quiet one: any perturbation,
# real or null, is amplified 50x on the first three. This is the whole point
# of the fixture -- it is what makes a pooled null p95 the wrong reference.
ROW_GAIN = np.array([50.0, 50.0, 50.0, 1.0, 1.0, 1.0])

# Feature 0 fires only in the quiet regime, feature 1 only in the loud one.
ACTS = np.zeros((N_SERIES, DICT))
ACTS[3:, 0] = [9.0, 8.0, 7.0]
ACTS[:3, 1] = [9.0, 8.0, 7.0]

# ...and only feature 0 decodes to anything. Feature 1 is the sham: it fires
# hard, in the loudest regime, and removing it changes the reconstruction by
# exactly nothing.
W_DEC = np.zeros((DICT, D_IN))
W_DEC[0] = 30.0


class _SAE:
    d_in, dict_size = D_IN, DICT

    def encode(self, x):
        # clean_tokens[i, 0, 0] carries the row index, so the stub can look
        # up the planted activation matrix rather than learn anything.
        idx = x[:, 0].round().long().cpu().numpy().astype(int)
        return torch.as_tensor(ACTS[idx], dtype=torch.float32)

    def decode(self, f):
        return torch.as_tensor(np.asarray(f.cpu()) @ W_DEC, dtype=torch.float32)

    def __call__(self, x):
        f = self.encode(x)
        return self.decode(f), f


class _Cfg:
    class run:
        seed = 0
    class data:
        horizon = HORIZON
    class l0:
        quantiles = [0.5]


class _Data:
    n = N_SERIES
    families = np.array(["f"] * N_SERIES)
    series_ids = np.array([f"s{i}" for i in range(N_SERIES)])

    def contexts(self):
        # Row index in column 0 so `_SAE.encode` can find it after reshape.
        c = np.zeros((N_SERIES, 8), dtype=np.float32)
        c[:, 0] = np.arange(N_SERIES)
        return c

    def targets(self):
        return np.zeros((N_SERIES, HORIZON), dtype=np.float32)


class _Adapter:
    """Forecast is a per-row gain times the sum of whatever was patched in.

    That makes `level` (mean of the forecast) a direct, monotone readout of
    the intervention, so the plant is legible in the artifact rather than
    mediated by a statistic.
    """

    name = "stub"

    def __init__(self, holder):
        self.module = object()
        self.cfg = type("C", (), {"batch_size": 999})()
        self._holder = holder

    def ensure_loaded(self):
        pass

    def token_slice(self, live_len):
        return slice(0, live_len)

    def predict(self, contexts, horizon, quantiles):
        repl = self._holder.get("repl")
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        if repl is None:
            vals = np.zeros(len(rows))
        else:
            vals = np.asarray(repl).reshape(len(rows), -1).sum(axis=1)
        point = (ROW_GAIN[rows] * vals)[:, None] * np.ones((1, horizon))
        return {"point": point.astype(np.float32)}


@pytest.fixture
def wired(monkeypatch):
    holder: dict = {}

    def _capture(adapter, contexts, layers, **_kw):
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        tok = np.zeros((len(rows), 1, D_IN), dtype=np.float32)
        tok[:, 0, 0] = rows
        return {layers[0]: torch.as_tensor(tok)}

    @contextmanager
    def _patch(module, layer, slicer, replacement):
        holder["repl"] = replacement
        try:
            yield
        finally:
            holder["repl"] = None

    monkeypatch.setattr(R, "capture_raw_tokens", _capture)
    monkeypatch.setattr(R, "token_patch", _patch)
    monkeypatch.setattr(R, "reach_probe", lambda *a, **k: {
        "reachable": True, "reason": "", "self_patch_delta": 0.0})
    return _Adapter(holder)


def _run(adapter, **kw):
    return R.feature_ablation_fingerprints(
        _Cfg, adapter, "blocks.0", _SAE(), _Data(), "cpu",
        candidates=[{"feature": 0, "rules": ["planted-real"]},
                    {"feature": 1, "rules": ["planted-sham"]}],
        activations=ACTS, top_k_series=3, n_null_directions=12, **kw)


# ---------------------------------------------------------------------------
# 1. Row selection
# ---------------------------------------------------------------------------

def test_top_firing_rows_ranks_by_activation_strongest_first():
    assert R.top_firing_rows(ACTS, 0, 3).tolist() == [3, 4, 5]
    assert R.top_firing_rows(ACTS, 1, 3).tolist() == [0, 1, 2]


def test_top_firing_rows_excludes_non_firing_rows_rather_than_padding():
    # A rare atom must return FEWER rows than asked for. Padding with
    # zero-activation series would put the intervention's own control group
    # inside the treatment group.
    sparse = np.zeros((N_SERIES, DICT))
    sparse[2, 0] = 1.0
    assert R.top_firing_rows(sparse, 0, 4).tolist() == [2]
    assert R.top_firing_rows(sparse, 3, 4).tolist() == []


# ---------------------------------------------------------------------------
# 2. The planted causal answer
# ---------------------------------------------------------------------------

def test_real_feature_clears_and_sham_feature_does_not(wired):
    out = _run(wired)
    assert out["withheld"] is False
    real, sham = out["candidates"][0], out["candidates"][1]

    assert real["feature"] == 0 and sham["feature"] == 1
    assert real["channels"]["level"]["clears_null"] is True
    assert sham["channels"]["level"]["effect"] == pytest.approx(0.0)
    assert sham["channels"]["level"]["clears_null"] is False
    assert real["n_channels_clearing"] > sham["n_channels_clearing"]


def test_each_candidate_is_scored_on_its_own_firing_rows(wired):
    out = _run(wired)
    real, sham = out["candidates"]
    assert real["n_top_series"] == 3 and sham["n_top_series"] == 3
    assert [f["row"] for f in real["forecasts"]] == [3, 4, 5]
    assert [f["row"] for f in sham["forecasts"]] == [0, 1, 2]
    assert real["mean_activation_on_top"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 3. The null is ROW-MATCHED -- the load-bearing negative
# ---------------------------------------------------------------------------

def test_null_p95_is_recomputed_over_each_candidates_own_rows(wired):
    out = _run(wired)
    real, sham = out["candidates"]
    p95_quiet = real["channels"]["level"]["null_p95"]
    p95_loud = sham["channels"]["level"]["null_p95"]

    # A pooled null would hand both candidates the SAME threshold. These are
    # measured on disjoint regimes and must differ by roughly the gain ratio.
    assert p95_loud > 10 * p95_quiet

    # And the plant makes the two answers disagree: the real feature's
    # genuine effect sits BELOW the loud regime's threshold, so a pooled p95
    # (which that regime dominates) would have refused it.
    assert real["channels"]["level"]["effect"] < p95_loud
    assert real["channels"]["level"]["effect"] > p95_quiet


# ---------------------------------------------------------------------------
# 4. Withholding, not zeros
# ---------------------------------------------------------------------------

def test_unreachable_target_is_withheld_rather_than_scored_as_no_effect(monkeypatch, wired):
    monkeypatch.setattr(R, "reach_probe", lambda *a, **k: {
        "reachable": False, "reason": "patch never lands"})
    out = _run(wired)
    assert out["withheld"] is True
    assert "candidates" not in out


def test_feature_that_fires_nowhere_is_withheld_with_a_reason(wired):
    out = R.feature_ablation_fingerprints(
        _Cfg, wired, "blocks.0", _SAE(), _Data(), "cpu",
        candidates=[{"feature": 2, "rules": []}],
        activations=ACTS, top_k_series=3, n_null_directions=4)
    assert out["withheld"] is True
    assert "fires on any series" in out["reason"]


# ---------------------------------------------------------------------------
# 5. The with/without pair the exemplar overlay renders
# ---------------------------------------------------------------------------

def test_forecast_pairs_contrast_full_reconstruction_against_ablation(wired):
    out = _run(wired, keep_forecasts=2)
    real, sham = out["candidates"]

    assert len(real["forecasts"]) == 2
    for f in real["forecasts"]:
        assert len(f["with_feature"]) == HORIZON
        assert len(f["without_feature"]) == HORIZON
        # The baseline arm is the FULL reconstruction, not the raw clean
        # forecast, so the picture and the channel number describe the same
        # contrast. Both are kept so a reader can see the SAE's own cost.
        assert "unpatched" in f
        assert f["with_feature"] != f["without_feature"]
        assert f["activation"] == pytest.approx(ACTS[f["row"], 0])

    # The sham atom decodes to nothing, so its two arms must coincide
    # exactly -- a "without" curve that moved would mean the ablation had
    # touched something other than the named atom.
    for f in sham["forecasts"]:
        assert f["with_feature"] == f["without_feature"]


# ---------------------------------------------------------------------------
# 6. The batch cap packs candidates into chunks; it never head-slices rows
# ---------------------------------------------------------------------------

def test_batch_cap_packs_candidates_rather_than_truncating_the_union(wired):
    # Two candidates on disjoint regimes, three rows each, against a batch
    # that holds four. One union truncated to the cap would keep rows 0,1,2,3
    # -- every row of the loud candidate and ONE of the quiet one -- so the
    # quiet candidate would be scored on a third of its regime while looking
    # like it had been measured.
    wired.cfg = type("C", (), {"batch_size": 4})()
    out = _run(wired)
    assert out["n_chunks"] == 2
    real, sham = out["candidates"]
    assert real["n_top_series"] == 3 and sham["n_top_series"] == 3
    assert [f["row"] for f in real["forecasts"]] == [3, 4, 5]
    assert real["channels"]["level"]["clears_null"] is True


def test_candidate_order_is_preserved_across_chunks(wired):
    wired.cfg = type("C", (), {"batch_size": 4})()
    out = _run(wired)
    assert [c["feature"] for c in out["candidates"]] == [0, 1]


# ---------------------------------------------------------------------------
# 7. A window-level activation matrix must refuse, not mis-index
# ---------------------------------------------------------------------------

def test_window_level_activations_are_refused_rather_than_indexed_as_series(wired):
    # `train.load_all_windows` returns [n_series * n_windows, dict], whose row
    # indices name windows. Indexing `data` with them selects a different
    # population, and for a run with more windows than series it happens to
    # raise far downstream -- so the shape is checked at the door.
    windows = np.repeat(ACTS, 4, axis=0)
    with pytest.raises(ValueError, match="series-level"):
        R.feature_ablation_fingerprints(
            _Cfg, wired, "blocks.0", _SAE(), _Data(), "cpu",
            candidates=[{"feature": 0, "rules": []}],
            activations=windows, top_k_series=2, n_null_directions=2)


# ---------------------------------------------------------------------------
# 8. A null with no spread is not a threshold
# ---------------------------------------------------------------------------

class _InertAdapter(_Adapter):
    """Forecast responds to a REAL ablation and is bit-identical under every
    null direction -- the quantized-decoding case (a small perturbation
    leaves the emitted tokens unchanged), measured on the first real target
    as 43 zero-spread cells, 10 of which were being counted as clearing.
    """

    def predict(self, contexts, horizon, quantiles):
        repl = self._holder.get("repl")
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        vals = (np.zeros(len(rows)) if repl is None
                else np.asarray(repl).reshape(len(rows), -1).sum(axis=1))
        # Snap to the reconstruction's own two levels: full recon and the
        # ablation differ, every null lands back on the full-recon level.
        snapped = np.where(np.abs(vals) > 1e-6, 1.0, 0.0)
        return {"point": (ROW_GAIN[rows] * snapped)[:, None] * np.ones((1, horizon))}


def test_zero_spread_null_is_reported_not_cleared(wired):
    inert = _InertAdapter(wired._holder)
    out = _run(inert)
    real = out["candidates"][0]
    lvl = real["channels"]["level"]
    assert lvl["null_p95"] == 0.0
    assert lvl["null_degenerate"] is True
    assert lvl["clears_null"] is False
    assert lvl["margin"] is None
    assert "no spread to clear" in lvl["reason"]
    # The effect itself is still recorded -- withholding the verdict is not
    # the same as discarding the measurement.
    assert lvl["effect"] > 0.0


def test_channel_with_no_finite_value_on_this_features_rows_is_unavailable(wired):
    # `periods_full` all-NaN makes `seasonal` unavailable for the whole
    # batch; a per-candidate NaN is the narrower case, and neither may reach
    # the artifact as a NaN effect.
    out = _run(wired)
    for cand in out["candidates"]:
        for ch, rec in cand["channels"].items():
            if rec["available"]:
                assert np.isfinite(rec["effect"]), f"{ch} recorded a non-finite effect"
                assert np.isfinite(rec["signed_effect"])
            else:
                assert rec["effect"] is None and rec["reason"]


def test_the_two_batteries_do_not_use_one_field_name_for_two_quantities():
    """`excess_over_chance` means a DIFFERENCE in the Stage 2 artifact.

    The ablation artifact originally wrote a RATIO under the same key, so the
    same target read 70.65 in one file and 4.65 in the other -- both correct,
    and side by side in a comparison a reader will certainly make. The ratio
    now carries its units in its name.
    """
    import inspect
    from tsfm_lens.sae import response as _r
    src = inspect.getsource(_r)
    assert '"clearing_cells_over_chance_ratio"' in src
    # Both writers of `excess_over_chance` must compute a difference.
    for line in src.splitlines():
        if '"excess_over_chance":' in line:
            assert "-" in line.split(":", 1)[1], line
            assert "/" not in line.split(":", 1)[1], line
