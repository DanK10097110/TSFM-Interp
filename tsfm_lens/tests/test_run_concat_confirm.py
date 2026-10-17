"""Known-answer tests for run_concat_confirm.py (ROADMAP.md sec 47.3): planted truths and a decoy per test."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_concat_confirm as rcc  # noqa: E402
from tsfm_lens.analysis.window_concat import centered_gram, cka_from_grams  # noqa: E402

NS, NW = 40, 3


def _gram(x):
    return centered_gram([x])


def _shared(seed=0, d=6, noise=0.3):
    """Two representations that share per-series structure (windows of a series share a vector)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(NS, d, generator=g).repeat_interleave(NW, 0)
    a = s + noise * torch.randn(NS * NW, d, generator=g)
    b = s @ torch.randn(d, d, generator=g) + noise * torch.randn(NS * NW, d, generator=g)
    return a, b


def test_registered_text_parses_to_21_pairs_and_4_hc2_pairs():
    section = rcc.registered_section_at_commit()
    pairs = rcc.parse_pairs(section)
    assert len(pairs) == 21 and len({rcc.pair_id(p) for p in pairs}) == 21
    assert [(p["layer_a"], p["layer_b"]) for p in rcc.hc2_pairs(pairs)] == [
        ("stacked_xf.0", "encoder.block.3"), ("stacked_xf.0", "encoder.block.7"),
        ("stacked_xf.0", "encoder.block.2"), ("stacked_xf.0", "model.layers.0")]
    assert "H-C1 (21 tests)" in section and "Holm over 4" in section


def test_gram_cka_equals_feature_space_cka_and_library():
    a, b = _shared()
    ka, kb = _gram(a), _gram(b)
    assert rcc.gram_cka(ka, kb) == pytest.approx(rcc.linear_cka(a, b), abs=1e-5)
    assert rcc.gram_cka(ka, kb) == pytest.approx(cka_from_grams(ka, kb), abs=1e-12)


def test_window_rows_moves_a_series_windows_together():
    assert rcc.window_rows(np.array([2, 0, 1]), 3).tolist() == [6, 7, 8, 0, 1, 2, 3, 4, 5]


def test_permutation_identical_reps_reach_the_p_floor():
    a, _ = _shared(1)
    ka = _gram(a)
    r = rcc.permutation_test(ka, ka.clone(), NS, NW, 200, 0)
    assert r["observed"] == pytest.approx(1.0, abs=1e-5)
    assert r["p"] == pytest.approx(1 / 201) and r["n_null_ge_obs"] == 0
    assert r["null_p95"] < 0.5


def test_permutation_independent_noise_is_not_significant_but_decoy_shared_structure_is():
    g = torch.Generator().manual_seed(5)
    a = torch.randn(NS * NW, 6, generator=g)
    b = torch.randn(NS * NW, 6, generator=g)
    noise = rcc.permutation_test(_gram(a), _gram(b), NS, NW, 200, 0)
    assert noise["p"] > 0.05 and noise["observed"] <= noise["null_p95"] * 1.5
    sa, sb = _shared(5)
    shared = rcc.permutation_test(_gram(sa), _gram(sb), NS, NW, 200, 0)
    assert shared["p"] == pytest.approx(1 / 201) and shared["observed"] > shared["null_p95"]


def test_permutation_is_deterministic_in_seed():
    a, b = _shared(2)
    r1 = rcc.permutation_test(_gram(a), _gram(b), NS, NW, 50, 7)
    r2 = rcc.permutation_test(_gram(a), _gram(b), NS, NW, 50, 7)
    assert r1 == r2


def _order_signal(seed=0, d=5, noise=0.05):
    """Within-window order carries the shared signal: position 0 = +s, position 1 = -s, so the MEAN cancels it."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(NS * NW, d, generator=g)
    sb = s @ torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    pa = [s + noise * torch.randn_like(s, generator=g), -s + noise * torch.randn_like(s, generator=g)]
    pb = [sb + noise * torch.randn_like(s, generator=g), -sb + noise * torch.randn_like(s, generator=g)]
    return pa, pb


def test_bootstrap_concat_beats_mean_when_order_carries_the_signal():
    pa, pb = _order_signal()
    ka, kb = centered_gram(pa), centered_gram(pb)
    xa, xb = (pa[0] + pa[1]) / 2, (pb[0] + pb[1]) / 2
    r = rcc.concat_minus_mean_bootstrap(ka, kb, xa, xb, NS, NW, 100, 0)
    assert r["concat"] > 0.9 and r["mean"] < 0.5 and r["diff"] > 0.4
    assert r["p"] == pytest.approx(1 / 101) and r["diff_ci95"][0] > 0


def test_bootstrap_identical_representations_give_zero_difference_and_p_one():
    a, b = _shared(3)
    r = rcc.concat_minus_mean_bootstrap(_gram(a), _gram(b), a, b, NS, NW, 60, 0)
    assert abs(r["diff"]) < 1e-5 and r["p"] > 0.05


def test_bootstrap_decoy_equal_representations_are_not_confirmed():
    pa, pb = _order_signal(4)
    ka = centered_gram(pa)
    kb = centered_gram(pb)
    xa, xb = (pa[0] + pa[1]) / 2, (pb[0] + pb[1]) / 2
    rev = rcc.concat_minus_mean_bootstrap(ka, kb, xa, xb, NS, NW, 60, 0)
    assert rev["diff"] > 0
    r_swapped_rep = rcc.concat_minus_mean_bootstrap(centered_gram([xa]), centered_gram([xb]), xa, xb, NS, NW, 60, 0)
    assert r_swapped_rep["p"] > 0.05 and abs(r_swapped_rep["diff"]) < 1e-5


def test_holm_known_answer_and_decide():
    adj = rcc.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}
    _, verdict = rcc.decide({"a": 0.01, "b": 0.04, "c": 0.03})
    assert verdict == {"a": "confirmed", "b": "not confirmed", "c": "not confirmed"}


def test_hc1_floor_survives_holm_over_21_but_not_a_floor_of_0_003():
    floor = 1 / 1001
    _, v = rcc.decide({str(i): floor for i in range(21)})
    assert set(v.values()) == {"confirmed"}
    _, v2 = rcc.decide({str(i): 0.003 for i in range(21)})
    assert set(v2.values()) == {"not confirmed"}


def test_tampered_registry_is_refused(tmp_path):
    section = rcc.registered_section_at_commit()
    reg = {"hypotheses_text_verbatim": section.replace("21 tests", "20 tests"),
           "hypotheses_sha256": rcc.sha256_text(section)}
    (tmp_path / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(ValueError):
        rcc.check_registry(tmp_path)
    reg = {"hypotheses_text_verbatim": section, "hypotheses_sha256": rcc.sha256_text(section)}
    (tmp_path / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
    assert rcc.check_registry(tmp_path)["hypotheses_sha256"] == rcc.sha256_text(section)


def test_results_are_never_overwritten(tmp_path):
    (tmp_path / "concat_confirmation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        rcc.run_confirmation(tmp_path, tmp_path / "cfg.yaml", "cpu")


def test_registry_is_written_once(tmp_path):
    (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        rcc.write_registry(tmp_path, tmp_path)
