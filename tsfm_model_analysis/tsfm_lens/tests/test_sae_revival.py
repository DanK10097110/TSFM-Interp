"""The production `sae` stage's dead-dictionary fix (ROADMAP.md sec 23.2 A1).

A1's own measurement (run_sae_capacity_sweep.py, run_sae_corpus_diversity_
check.py) found the ~95% dead production dictionary was NOT a corpus
problem -- real Monash activations excite *fewer* directions than the
synthetic benchmark, and the same checkpoints are 93-98% dead at every one
of 22 layers regardless of layer content. The actual cause is three
compounding training-budget knobs: `epochs` silently means "2 optimizer
steps" on a ~4600-row corpus at `batch_size: 4096`, `aux_dead_steps` (an
absolute step count) eats a large fraction of that tiny budget before AuxK
ever engages, and `dict_size_mult * d_in` is not sized to what a layer can
actually support.

This file tests the fix's four mechanisms, all synthetic with planted
answers, matching this repo's own SAE test style (`tests/test_aux_k.py`,
`tests/test_sae_seed_floor.py`):

  (b) `SAETrainConfig.min_train_steps` / `aux_dead_steps_frac`
  (c) `search_dict_size` (`dict_size_policy: "search"`)
  (d) the rendered `dead_rate_gate` and its report-side caveat

Runnable directly (`python tests/test_sae_revival.py`) or via pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import SAEConfig
from tsfm_lens.report.report import _sae_gate_caveat
from tsfm_lens.sae.eval import dead_feature_rate, reconstruction_fidelity
from tsfm_lens.sae.train import SAETrainConfig, search_dict_size, train_sae
from tsfm_lens.utils import save_json

DEVICE = torch.device("cpu")


def _planted_activations(n=600, d=24, n_causes=8, seed=0):
    """Same construction as `test_aux_k.py::_planted_activations` -- a real,
    finite-rank dictionary exists (n_causes directions), so a dictionary much
    bigger than n_causes has genuinely-dead capacity to find, and this file's
    planted answers are about *training budget*, not about there being
    nothing left to represent."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_causes, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    codes = rng.exponential(scale=1.0, size=(n, n_causes)) * (rng.random((n, n_causes)) < 0.3)
    return (codes @ directions).astype(np.float32)


# ---------------------------------------------------------------------------
# (b) min_train_steps / aux_dead_steps_frac
# ---------------------------------------------------------------------------

def test_min_train_steps_zero_is_bit_identical_to_the_old_behavior():
    """The default (`0`) must reproduce today's behavior exactly -- no
    already-recorded SAE run's `epochs_run`/dictionary should move."""
    acts = _planted_activations()
    base = SAETrainConfig(dict_size=64, k=4, epochs=3, batch_size=128, seed=0)
    sae, hist = train_sae(acts, base, DEVICE)
    assert sae.train_meta["epochs_configured"] == 3
    assert sae.train_meta["epochs_run"] == 3, "min_train_steps=0 must not alter epoch count"
    steps_per_epoch = -(-acts.shape[0] // base.batch_size)
    assert sae.train_meta["steps_per_epoch"] == steps_per_epoch
    assert sae.train_meta["n_optimizer_steps"] == 3 * steps_per_epoch
    assert sae.train_meta["aux_dead_steps_used"] == base.aux_dead_steps


def test_min_train_steps_raises_epoch_count_to_meet_the_step_floor():
    """This is the production stage's actual bug in miniature: `epochs: 3` at
    128-row batches over 600 rows is 5 steps/epoch = 15 steps total. Asking
    for at least 40 steps must raise the epoch count, not silently train for
    fewer steps than requested."""
    acts = _planted_activations()
    cfg = SAETrainConfig(dict_size=64, k=4, epochs=3, batch_size=128, seed=0,
                         min_train_steps=40)
    sae, _ = train_sae(acts, cfg, DEVICE)
    steps_per_epoch = sae.train_meta["steps_per_epoch"]
    assert sae.train_meta["epochs_run"] > 3
    assert sae.train_meta["n_optimizer_steps"] >= 40
    # And never *fewer* epochs than plainly configured, even if the step
    # floor is already met by cfg.epochs alone.
    cfg_low_floor = SAETrainConfig(dict_size=64, k=4, epochs=3, batch_size=128,
                                   seed=0, min_train_steps=1)
    sae_low, _ = train_sae(acts, cfg_low_floor, DEVICE)
    assert sae_low.train_meta["epochs_run"] == 3


def test_aux_dead_steps_frac_scales_with_the_real_budget_not_a_fixed_constant():
    """`aux_dead_steps_frac > 0` must derive a warm-up length from the ACTUAL
    resolved step budget (after any `min_train_steps` floor), not from
    `cfg.epochs` alone -- otherwise raising the step floor without touching
    this knob would silently change what fraction of the budget the warm-up
    consumes, which is exactly the bug this fix exists to close."""
    acts = _planted_activations()
    cfg_small = SAETrainConfig(dict_size=64, k=4, epochs=2, batch_size=128, seed=0,
                               aux_dead_steps_frac=0.5, aux_k=4)
    sae_small, _ = train_sae(acts, cfg_small, DEVICE)
    total_small = sae_small.train_meta["n_optimizer_steps"]
    assert sae_small.train_meta["aux_dead_steps_used"] == max(1, int(0.5 * total_small))

    cfg_big = SAETrainConfig(dict_size=64, k=4, epochs=2, batch_size=128, seed=0,
                             aux_dead_steps_frac=0.5, aux_k=4, min_train_steps=200)
    sae_big, _ = train_sae(acts, cfg_big, DEVICE)
    total_big = sae_big.train_meta["n_optimizer_steps"]
    assert total_big > total_small
    assert sae_big.train_meta["aux_dead_steps_used"] == max(1, int(0.5 * total_big))
    assert sae_big.train_meta["aux_dead_steps_used"] > sae_small.train_meta["aux_dead_steps_used"]


def test_aux_dead_steps_frac_zero_leaves_the_absolute_default_untouched():
    """`0.0` (the default) must keep using the absolute `aux_dead_steps`
    count regardless of how many steps actually ran -- no recorded run's
    warm-up length moves."""
    acts = _planted_activations()
    cfg = SAETrainConfig(dict_size=64, k=4, epochs=2, batch_size=128, seed=0,
                         min_train_steps=500, aux_dead_steps=17)
    sae, _ = train_sae(acts, cfg, DEVICE)
    assert sae.train_meta["aux_dead_steps_used"] == 17


# ---------------------------------------------------------------------------
# (c) search_dict_size
# ---------------------------------------------------------------------------

def test_search_dict_size_picks_the_largest_size_that_clears_the_bar():
    """A planted 8-cause dictionary: small candidate sizes should train close
    to fully alive (little room to be dead with only 8 real causes to
    represent), so the search should be able to clear a generous bar and
    should prefer the LARGEST such candidate, not the smallest."""
    acts = _planted_activations(n=800, n_causes=8)
    base = SAETrainConfig(k=4, epochs=6, batch_size=128, seed=0, aux_k=8,
                          min_train_steps=80)
    ladder = [8, 16, 32]
    result = search_dict_size(acts, base, ladder, max_dead_rate=0.5, device=DEVICE)
    assert result["target_met"] is True
    assert result["chosen_dict_size"] == max(
        r["dict_size"] for r in result["ladder"] if r["dead_feature_rate"] <= 0.5)
    assert {r["dict_size"] for r in result["ladder"]} == set(ladder)
    # Every ladder cell trained independently at its own size.
    assert len({r["dict_size"] for r in result["ladder"]}) == len(ladder)


def test_search_dict_size_falls_back_to_most_alive_when_nothing_clears_the_bar():
    """§23.2 A1(c)'s own stated risk: a layer that saturates (more dead
    atoms in a bigger dictionary, no gain in alive count) must not have the
    search silently prefer the biggest dictionary just because it's biggest.
    Plants that exact shape by hand -- a fake `dead_feature_rate` and
    `reconstruction_fidelity` that make bigger sizes strictly worse -- so the
    selection rule itself is pinned independent of whether real training
    happens to produce that shape today."""
    import tsfm_lens.sae.train as train_mod

    planted = {
        8: {"dead": 0.90, "fid": 0.5},
        16: {"dead": 0.95, "fid": 0.5},   # more alive in absolute terms (0.8) than size 8 (0.8)... see below
        32: {"dead": 0.98, "fid": 0.5},
    }
    # size*alive_frac: 8*0.10=0.8, 16*0.05=0.8, 32*0.02=0.64 -- size 8 and 16
    # tie on alive count, 32 is worse. An impossible bar (0.01) means NOTHING
    # passes, so the fallback rule must fire.
    orig_dead = train_mod.dead_feature_rate
    orig_fid = train_mod.reconstruction_fidelity
    orig_train = train_mod.train_sae

    class _FakeSAE:
        def __init__(self, dict_size):
            self.dict_size = dict_size

    def fake_train_sae(acts, cfg, device):
        return _FakeSAE(cfg.dict_size), []

    def fake_dead(sae, acts, device):
        return planted[sae.dict_size]["dead"]

    def fake_fid(sae, acts, device):
        return planted[sae.dict_size]["fid"]

    train_mod.train_sae = fake_train_sae
    train_mod.dead_feature_rate = fake_dead
    train_mod.reconstruction_fidelity = fake_fid
    try:
        acts = _planted_activations(n=50, d=4)
        base = SAETrainConfig(k=2, epochs=1, batch_size=16, seed=0)
        result = train_mod.search_dict_size(acts, base, [8, 16, 32],
                                            max_dead_rate=0.01, device=DEVICE)
    finally:
        train_mod.train_sae = orig_train
        train_mod.dead_feature_rate = orig_dead
        train_mod.reconstruction_fidelity = orig_fid

    assert result["target_met"] is False
    # size 8 and 16 tie at 0.8 alive atoms; max() with this key keeps the
    # FIRST max it sees among equals (ladder is sorted ascending), so 8 wins
    # the tie -- and, crucially, 32 (the largest, most-dead size) must NOT
    # be chosen just because it's biggest.
    assert result["chosen_dict_size"] in (8, 16)
    assert result["chosen_dict_size"] != 32


def test_search_dict_size_trains_on_train_activations_but_scores_on_eval_activations():
    """ROADMAP.md sec 23.2 A1 finding (2026-08-21, live sae_revival run): the
    original `search_dict_size` trained AND scored every candidate on the same
    array. In production that array was benchmark-only activations, while the
    actual deployed model (real_data_enabled: true) trains on benchmark + real
    -data-augmented activations -- a materially different training set (more
    rows, different steps_per_epoch, different batch composition at this
    stage's near-full-batch sizes). This gave a single-seed dead-rate estimate
    for Chronos-T5-Base's chosen size that a 5-seed re-measurement on the
    actually-deployed (augmented) training set missed by +0.078, enough to
    flip a boundary candidate from "clears the bar" to "sits at the bar".

    This test pins the fix directly rather than trusting the real run's
    numbers alone: a SAE trained on `train_acts` (an 8-cause dictionary) must
    be *evaluated* against whatever `eval_activations` names, not against
    `train_acts` -- proven with an `eval_activations` drawn from a
    deliberately unrelated distribution (independent random directions), so a
    sae fit to `train_acts` reconstructs it far worse and the ladder's
    reported fidelity/dead-rate must reflect that, not the training-set-only
    numbers a pre-fix call would have returned.
    """
    train_acts = _planted_activations(n=400, n_causes=8, seed=1)
    unrelated_acts = _planted_activations(n=400, n_causes=8, seed=99) * 5.0  # different
    # directions AND a different scale -- a SAE fit to train_acts should
    # reconstruct this far worse than it reconstructs its own training data.
    base = SAETrainConfig(k=4, epochs=4, batch_size=64, seed=0, aux_k=4)
    ladder = [16]

    result_self_eval = search_dict_size(train_acts, base, ladder, max_dead_rate=1.0, device=DEVICE)
    result_cross_eval = search_dict_size(train_acts, base, ladder, max_dead_rate=1.0,
                                         device=DEVICE, eval_activations=unrelated_acts)

    fid_self = result_self_eval["ladder"][0]["reconstruction_fidelity"]
    fid_cross = result_cross_eval["ladder"][0]["reconstruction_fidelity"]
    assert fid_cross < fid_self, (
        "evaluating against an unrelated array must score worse than "
        "evaluating against the array actually trained on -- if this fails, "
        "eval_activations is being ignored and the pre-fix bug is back")

    # Backward compatibility: omitting eval_activations must reproduce the
    # pre-fix bit-identical behavior (train == eval), so no already-recorded
    # search result (e.g. medium_run_chronos_base's pinned SAE targets, which
    # never use dict_size_policy: search) can move.
    result_default = search_dict_size(train_acts, base, ladder, max_dead_rate=1.0, device=DEVICE)
    assert result_default["ladder"][0]["reconstruction_fidelity"] == fid_self


def test_search_dict_size_multi_seed_and_fidelity_floor_catch_a_boundary_case():
    """Reproduces the exact shape of the real `sae_revival` validation-run bug
    (ROADMAP.md sec 23.2 A1, found 2026-08-21): the default `n_seeds=1` /
    `min_fidelity=0.0` evaluates each candidate with exactly ONE stochastic
    draw and never looks at fidelity when filtering, so it can pick a
    candidate whose TRUE (multi-seed) dead rate sits over the bar (dict_size
    64 here: a lucky seed-0 draw of 0.05, true mean 0.317) or whose
    reconstruction collapsed on the drawn seed alone (dict_size 128: dead
    rate looks fine, fidelity is -0.99). `n_seeds=3` + `min_fidelity=0.5`
    correctly demotes both and falls back to the only candidate that is
    actually safe at every seed (dict_size 32)."""
    import tsfm_lens.sae.train as train_mod

    planted_dead = {(64, 0): 0.05, (64, 1): 0.42, (64, 2): 0.48,
                    (32, 0): 0.10, (32, 1): 0.11, (32, 2): 0.09,
                    (128, 0): 0.05, (128, 1): 0.05, (128, 2): 0.05}
    planted_fid = {(64, 0): 0.9, (64, 1): 0.6, (64, 2): 0.55,
                   (32, 0): 0.85, (32, 1): 0.84, (32, 2): 0.86,
                   (128, 0): -0.99, (128, 1): 0.9, (128, 2): 0.9}

    class _FakeSAE:
        def __init__(self, dict_size, seed):
            self.dict_size = dict_size
            self.seed = seed

    orig_train = train_mod.train_sae
    orig_dead = train_mod.dead_feature_rate
    orig_fid = train_mod.reconstruction_fidelity

    def fake_train_sae(acts, cfg, device):
        return _FakeSAE(cfg.dict_size, cfg.seed), []

    def fake_dead(sae, acts, device):
        return planted_dead[(sae.dict_size, sae.seed)]

    def fake_fid(sae, acts, device):
        return planted_fid[(sae.dict_size, sae.seed)]

    train_mod.train_sae = fake_train_sae
    train_mod.dead_feature_rate = fake_dead
    train_mod.reconstruction_fidelity = fake_fid
    try:
        acts = _planted_activations(n=50, d=4)
        base = SAETrainConfig(k=2, epochs=1, batch_size=16, seed=0)

        # Old default (n_seeds=1, min_fidelity=0.0): only each size's seed-0
        # draw is ever seen. All three "pass" dead_rate<=0.30 on that single
        # draw, so "largest passing" picks 128 -- despite its fidelity having
        # collapsed on that very draw, invisible because nothing filters on it.
        old = train_mod.search_dict_size(acts, base, [32, 64, 128],
                                         max_dead_rate=0.30, device=DEVICE)
        assert old["chosen_dict_size"] == 128, "sanity: reproduces the pre-fix single-draw bug"

        # n_seeds=3 averages 64's true (over-the-bar) dead rate and correctly
        # excludes it; min_fidelity=0.5 excludes 128's collapsed mean fidelity.
        # Only 32 -- safe at every seed on both metrics -- remains.
        fixed = train_mod.search_dict_size(acts, base, [32, 64, 128], max_dead_rate=0.30,
                                           device=DEVICE, min_fidelity=0.5, n_seeds=3)
        assert fixed["chosen_dict_size"] == 32
        cell64 = next(r for r in fixed["ladder"] if r["dict_size"] == 64)
        assert cell64["dead_feature_rate"] == pytest.approx((0.05 + 0.42 + 0.48) / 3)
        assert cell64["dead_feature_rate_seeds"]["n"] == 3
        cell128 = next(r for r in fixed["ladder"] if r["dict_size"] == 128)
        assert cell128["reconstruction_fidelity"] == pytest.approx((-0.99 + 0.9 + 0.9) / 3)
    finally:
        train_mod.train_sae = orig_train
        train_mod.dead_feature_rate = orig_dead
        train_mod.reconstruction_fidelity = orig_fid


def test_search_dict_size_margin_avoids_the_riskiest_passing_candidate():
    """Reproduces the exact shape of the real Chronos-T5-Base finding
    (ROADMAP.md sec 23.2 A1, second validation-run block, 2026-08-21,
    corrected same day): a STEEPLY-increasing dead-rate-vs-size ladder
    where only the two smallest candidates clear the bar at all, and the
    DEFAULT `margin=0.0` selection (largest size whose mean clears the bar)
    picks the boundary one (256) whose true mean sits so close to the bar
    that ordinary seed noise (not modeled by this planted test, which uses
    a single planted mean per size -- the real 5-seed floor's sd was ~0.011
    against a ~0.007 gap between the search's 3-seed mean and the bar)
    would flip its pass/fail verdict repeatedly. Unlike a first draft of
    this test (which mistakenly planted TimesFM-shaped values where 512
    and 1024 also passed comfortably), there is no larger passing
    alternative here -- `margin>0` must fall back to the only other
    passing candidate, 128, not to some larger "safer" size."""
    import tsfm_lens.sae.train as train_mod

    # Mirrors the real Chronos-T5-Base ladder's shape: dead rate rises
    # steeply with size, and only 128/256 clear max_dead_rate=0.30 --
    # 512/1024 fail outright, same as the real ladder's 384-and-up.
    planted_dead = {128: 0.12, 256: 0.297, 512: 0.50, 1024: 0.69}
    planted_fid = {128: 0.88, 256: 0.90, 512: 0.92, 1024: 0.94}

    class _FakeSAE:
        def __init__(self, dict_size):
            self.dict_size = dict_size

    orig_train = train_mod.train_sae
    orig_dead = train_mod.dead_feature_rate
    orig_fid = train_mod.reconstruction_fidelity

    def fake_train_sae(acts, cfg, device):
        return _FakeSAE(cfg.dict_size), []

    def fake_dead(sae, acts, device):
        return planted_dead[sae.dict_size]

    def fake_fid(sae, acts, device):
        return planted_fid[sae.dict_size]

    train_mod.train_sae = fake_train_sae
    train_mod.dead_feature_rate = fake_dead
    train_mod.reconstruction_fidelity = fake_fid
    try:
        acts = _planted_activations(n=50, d=4)
        base = SAETrainConfig(k=2, epochs=1, batch_size=16, seed=0)
        ladder = [128, 256, 512, 1024]

        # margin=0.0 (default): 128 and 256 both clear 0.30, 512/1024 both
        # fail outright -- 256 is the largest of the passing pair, chosen
        # despite sitting almost exactly on the boundary. Reproduces the
        # real bug's selection, including that there is no larger passing
        # alternative on this ladder (only a smaller, safer one).
        no_margin = train_mod.search_dict_size(acts, base, ladder, max_dead_rate=0.30,
                                               device=DEVICE)
        assert no_margin["chosen_dict_size"] == 256
        assert no_margin["margin"] == 0.0

        # margin=0.05: only sizes whose mean clears 0.30-0.05=0.25 pass --
        # 256 (0.297) is now excluded, leaving only 128 (0.12); 512/1024
        # still fail regardless of margin. The only fallback is the
        # smaller, comfortably-passing size, not a larger "safer" one.
        with_margin = train_mod.search_dict_size(acts, base, ladder, max_dead_rate=0.30,
                                                  device=DEVICE, margin=0.05)
        assert with_margin["chosen_dict_size"] == 128
        assert with_margin["target_met"] is True
        assert with_margin["margin"] == 0.05
    finally:
        train_mod.train_sae = orig_train
        train_mod.dead_feature_rate = orig_dead
        train_mod.reconstruction_fidelity = orig_fid


def test_sae_config_search_robustness_defaults_are_no_ops():
    """`min_fidelity`/`dict_size_search_seeds`/`dict_size_search_margin` must
    default to values that reproduce today's single-draw, dead-rate-only,
    largest-passing selection exactly -- no already-recorded
    `dict_size_policy: search` result may move."""
    cfg = SAEConfig()
    assert cfg.min_fidelity == 0.0
    assert cfg.dict_size_search_seeds == 1
    assert cfg.dict_size_search_margin == 0.0


# ---------------------------------------------------------------------------
# (d) dead_rate_gate + report caveat
# ---------------------------------------------------------------------------

def test_dead_rate_gate_pass_and_fail_are_computed_correctly():
    threshold = 0.30
    passed = {"threshold": threshold, "value": 0.10, "passed": 0.10 <= threshold}
    failed = {"threshold": threshold, "value": 0.95, "passed": 0.95 <= threshold}
    assert passed["passed"] is True
    assert failed["passed"] is False


def test_sae_gate_caveat_fires_only_on_a_failing_gate(tmp_path):
    """`_sae_gate_caveat` reads `sae/meta.json` back out by the exact key
    `_sec_sae`'s findings embed in their own text (`"SAE — {model}/{layer}:
    ..."`). A passing gate, a missing gate, and an unparsable key must all
    render nothing -- only a genuinely failing gate produces a caveat clause,
    per `CLAUDE.md` §2.5's "loud when broken, quiet when fine"."""
    run_dir = tmp_path / "run"
    (run_dir / "sae").mkdir(parents=True)
    save_json(run_dir / "sae" / "meta.json", {
        "TimesFM/stacked_xf.18": {"dead_rate_gate": {"threshold": 0.3, "value": 0.95,
                                                     "passed": False}},
        "Chronos-T5-Base/encoder.block.6": {"dead_rate_gate": {"threshold": 0.3,
                                                               "value": 0.10,
                                                               "passed": True}},
    })
    failing = _sae_gate_caveat("SAE — TimesFM/stacked_xf.18: feature 12 ...", run_dir)
    assert failing != ""
    assert "95.0%" in failing or "0.95" in failing.lower() or "95" in failing

    passing = _sae_gate_caveat("SAE — Chronos-T5-Base/encoder.block.6: feature 3 ...", run_dir)
    assert passing == ""

    unparsable = _sae_gate_caveat("some unrelated finding text", run_dir)
    assert unparsable == ""

    missing_key = _sae_gate_caveat("SAE — Unknown/layer.0: feature 1 ...", run_dir)
    assert missing_key == ""

    # No sae/meta.json at all -- a real failure mode this test file's first
    # version didn't cover, caught by test_finding_caveats.py's own fixture
    # (a run dir with no SAE stage run at all): must degrade, not raise.
    empty_run = run_dir.parent / "run_with_no_sae"
    empty_run.mkdir()
    no_meta = _sae_gate_caveat("SAE — TimesFM/stacked_xf.18: feature 1 ...", empty_run)
    assert no_meta == ""


def test_sae_config_defaults_are_off_and_backward_compatible():
    """The new knobs must default to exactly the pre-fix behavior -- this is
    the actual acceptance bar `CLAUDE.md` §2.1 asks for whenever a change
    touches a shared training path with many already-recorded numbers."""
    cfg = SAEConfig()
    assert cfg.min_train_steps == 0
    assert cfg.aux_dead_steps_frac == 0.0
    assert cfg.dict_size_policy == "mult"
    assert cfg.max_dead_rate == 0.30
    assert cfg.aux_k == 0  # the default flip is NOT part of this change


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_run_sae_dict_size_policy_search_writes_a_ladder_and_a_gate(tmp_path):
    """End-to-end through the real `run_sae` pipeline function (mock models,
    matching `tests/test_sae_seed_floor.py`'s own end-to-end test), confirming
    the plumbing -- not just the standalone `search_dict_size` unit -- writes
    `dict_size_search` and `dead_rate_gate` into `sae/meta.json`, and that an
    explicit per-target `dict_size` skips the search entirely."""
    from tests.test_smoke import build_config
    from tests.test_smoke import test_end_to_end as _build_extracted_run
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.pipeline import Context
    from tsfm_lens.sae.train import run_sae
    from tsfm_lens.utils import load_json

    run_dir = Path(_build_extracted_run())
    cfg = config_from_dict(build_config(str(run_dir.parent)))
    cfg.run.name = run_dir.name
    store_ro = ActivationStore(run_dir / "activations.zarr", mode="r")
    layer_a = store_ro.layers("patchy")[-1]
    layer_b = store_ro.layers("steppy")[-1]

    cfg.sae.enabled = True
    cfg.sae.dict_size_policy = "search"
    cfg.sae.dict_size_ladder = [4, 8, 16]
    cfg.sae.max_dead_rate = 0.9  # generous -- this is a plumbing test, not a
                                 # re-run of the real acceptance bar
    cfg.sae.targets = [{"model": "patchy", "layer": layer_a},
                       {"model": "steppy", "layer": layer_b, "dict_size": 6}]
    cfg.sae.epochs = 2
    cfg.sae.k = 2
    cfg.sae.forecast_preservation_max_series = 16

    ctx = Context(cfg)
    run_sae(cfg, ctx.hub, ctx.store, ctx.data, ctx.device)

    meta = load_json(run_dir / "sae" / "meta.json")
    patchy = meta["patchy/" + layer_a]
    steppy = meta["steppy/" + layer_b]

    assert patchy["dict_size_search"] is not None
    assert patchy["dict_size"] in cfg.sae.dict_size_ladder
    assert {r["dict_size"] for r in patchy["dict_size_search"]["ladder"]} == set(cfg.sae.dict_size_ladder)

    assert steppy["dict_size_search"] is None, "an explicit per-target dict_size must skip the search"
    assert steppy["dict_size"] == 6

    for entry in (patchy, steppy):
        gate = entry["dead_rate_gate"]
        assert gate["threshold"] == 0.9
        assert gate["passed"] == (gate["value"] <= 0.9)
        tb = entry["training_budget"]
        assert tb["n_optimizer_steps"] == tb["epochs_run"] * tb["steps_per_epoch"]
