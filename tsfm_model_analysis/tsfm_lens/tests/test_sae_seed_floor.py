"""The SAE stage's own seed-to-seed noise floor (ROADMAP.md §13).

`sae.n_seeds > 1` retrains each target against the identical frozen
activations and records the spread, so a headline ΔMASE is rendered beside
the size of the retraining noise it has to clear rather than read against
zero. The measured motivation is on record: TimesFM's forecast-preservation
ΔMASE has a five-seed sd of 0.121 against single-seed values of +0.05 and
+0.110 -- two numbers that look like different results and are one.

What is covered here: the spread statistic itself, the tri-state floor
lookup the report's warning branches on, both rendering cases (including
that an unmeasured floor renders a sentence rather than nothing, per
`CLAUDE.md` §2.5), that the seed is not inert, and that the primary seed and
its replicates cannot drift into different key sets. What is not covered
here is the assembled warning sentence inside `_sec_sae`, which needs a real
run directory with an activation store; it branches on exactly the
`resolvable is False` value pinned below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import SAEConfig
from tsfm_lens.report.report import _sae_seed_floor_block
from tsfm_lens.report.derived import seed_floor_verdict
from tsfm_lens.sae.eval import reconstruction_fidelity, seed_spread
from tsfm_lens.sae.train import SEED_FLOOR_METRICS, _metric_row, _train_config, train_sae


class _NS:
    pass


def _cfg(**sae_kwargs):
    cfg = _NS()
    cfg.sae = SAEConfig(**sae_kwargs)
    return cfg


def test_default_is_one_seed_so_no_recorded_run_changes():
    """The flag exists to be opted into. A default above 1 would silently
    multiply every existing SAE run's cost and rewrite its artifact shape."""
    assert SAEConfig().n_seeds == 1


def test_seed_spread_reports_the_sample_sd_and_the_range_beside_it():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    spread = seed_spread(values)
    assert spread["n"] == 5
    assert spread["mean"] == pytest.approx(0.3)
    assert spread["sd"] == pytest.approx(np.std(values, ddof=1))
    assert spread["range"] == pytest.approx(0.4)
    assert spread["values"] == pytest.approx(values)


def test_seed_spread_drops_failed_seeds_rather_than_poisoning_the_summary():
    """A forecast-preservation check that raised contributes `None`, and a
    non-finite value must not turn the whole floor into NaN -- but `n` has to
    fall so a reader can see how many seeds actually counted."""
    spread = seed_spread([0.2, None, float("nan"), 0.4])
    assert spread["n"] == 2
    assert spread["mean"] == pytest.approx(0.3)
    assert seed_spread([None, None]) == {"n": 0}
    assert seed_spread([])["n"] == 0


def test_a_single_seed_has_no_spread_rather_than_a_fabricated_one():
    assert seed_spread([0.4]) == {"n": 1, "mean": 0.4, "sd": 0.0, "min": 0.4,
                                  "max": 0.4, "range": 0.0, "values": [0.4]}


def _entry(mean, sd_values):
    """A `sae/meta.json` entry whose window ΔMASE has the given seed values."""
    return {"seed_floor": {"n_seeds": len(sd_values),
                           "per_seed": [],
                           "spread": {"mase_delta_window": seed_spread(sd_values),
                                      "mase_delta_token": seed_spread(sd_values)}}}


def test_the_floor_lookup_is_tri_state_and_unmeasured_is_not_failed():
    """The report warns on `False` only. If an absent floor collapsed to
    `False`, every single-seed run would sprout a warning it has no evidence
    for; if it collapsed to `True`, a genuinely unresolvable delta would read
    as cleared. Neither is acceptable, so this stays tri-state the way
    `in_floor_units`' `interpretable` does."""
    suffix, resolvable = seed_floor_verdict({}, "mase_delta_window")
    assert (suffix, resolvable) == ("", None)

    one_seed = {"seed_floor": {"n_seeds": 1, "spread": {"mase_delta_window": seed_spread([0.5])}}}
    assert seed_floor_verdict(one_seed, "mase_delta_window") == ("", None)

    clear = _entry(None, [0.90, 0.95, 1.00, 1.05, 1.10])
    suffix, resolvable = seed_floor_verdict(clear, "mase_delta_window")
    assert resolvable is True
    assert "over 5 seeds" in suffix and "±" in suffix

    swamped = _entry(None, [-0.20, 0.35, 0.02, -0.30, 0.18])
    assert seed_floor_verdict(swamped, "mase_delta_window")[1] is False


def test_an_unmeasured_floor_renders_a_sentence_not_an_empty_block():
    """CLAUDE.md §2.5: a bare single-seed ΔMASE with nothing beside it reads
    exactly like one that cleared a floor. The absent case has to say so."""
    html = _sae_seed_floor_block({"m/l": {"reconstruction_fidelity": 0.8}})
    assert "Not measured" in html
    assert "sae.n_seeds" in html


def test_a_measured_floor_renders_every_metric_for_every_target():
    meta = {"a/l0": _entry(None, [0.1, 0.2, 0.3]), "b/l1": _entry(None, [1.0, 1.1, 1.2])}
    html = _sae_seed_floor_block(meta)
    assert "Seed-to-seed noise floor" in html
    assert "a/l0" in html and "b/l1" in html
    assert "mase_delta_window" in html and "mase_delta_token" in html
    assert "Not measured" not in html


def test_the_metric_row_covers_exactly_what_the_spread_is_taken_over():
    """The primary seed and its replicates both build their row here, so the
    two paths cannot record different keys -- but the row still has to carry
    every metric `run_sae` then asks `seed_spread` for."""
    row = _metric_row(3, 0.8, 0.1, {"mase_delta": 0.2, "mase_clean": 1.5},
                      {"mase_delta": 0.3, "mase_clean": 1.5})
    assert row["seed"] == 3
    assert set(SEED_FLOOR_METRICS) <= set(row)
    failed = _metric_row(4, 0.8, 0.1, {"error": "boom"}, {"error": "boom"})
    assert failed["mase_delta_window"] is None and failed["mase_clean_token"] is None


def test_the_training_seed_is_not_inert():
    """A floor built from seeds that all produce the same dictionary would be
    identically zero and would license any delta at all."""
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(4, 16))
    activations = (rng.normal(size=(1500, 4)) @ basis).astype(np.float32)
    cfg = _cfg(dict_size_mult=4, k=4, lr=1e-2, epochs=6, batch_size=256)
    device = torch.device("cpu")

    assert _train_config(cfg, 0).seed == 0 and _train_config(cfg, 7).seed == 7
    first, _ = train_sae(activations, _train_config(cfg, 0), device)
    second, _ = train_sae(activations, _train_config(cfg, 7), device)
    assert not torch.allclose(first.W_dec, second.W_dec), \
        "two training seeds must not produce the same dictionary"

    same, _ = train_sae(activations, _train_config(cfg, 0), device)
    assert torch.allclose(first.W_dec, same.W_dec), \
        "one seed must still reproduce, or the floor measures nondeterminism instead"

    fidelities = [reconstruction_fidelity(s, activations, device) for s in (first, second)]
    assert seed_spread(fidelities)["n"] == 2


def test_every_configured_hyperparameter_reaches_the_replicate_trainings():
    """A replicate trained at different settings than the primary seed would
    measure the wrong thing entirely."""
    cfg = _cfg(dict_size_mult=3, k=5, lr=7e-3, epochs=2, batch_size=64,
               resample_dead_every_epochs=1, aux_k=8, aux_coef=0.5, aux_dead_steps=4)
    primary, replicate = _train_config(cfg, 0), _train_config(cfg, 1)
    for field in ("dict_size_mult", "k", "lr", "epochs", "batch_size",
                  "resample_dead_every_epochs", "aux_k", "aux_coef", "aux_dead_steps"):
        assert getattr(primary, field) == getattr(replicate, field) == getattr(cfg.sae, field)


def test_train_config_dict_size_defaults_to_zero_so_dict_size_mult_still_governs():
    """ROADMAP.md sec 13 item 9: the override must be opt-in -- omitting it
    must reproduce the pre-existing dict_size_mult-only behavior exactly, so
    no already-recorded SAE run's dictionary size moves underneath it."""
    cfg = _cfg(dict_size_mult=8)
    assert _train_config(cfg, 0).dict_size == 0


def test_train_config_dict_size_override_reaches_the_train_config():
    """A per-target absolute size (e.g. Stage 0's matched 576/512) must reach
    SAETrainConfig.dict_size, which train_sae's own `cfg.dict_size or
    cfg.dict_size_mult * d_in` resolution already treats as an override
    when nonzero -- this only needs to verify the value is plumbed through."""
    cfg = _cfg(dict_size_mult=8)
    assert _train_config(cfg, 0, dict_size=576).dict_size == 576
    # And leaves every other hyperparameter, including dict_size_mult itself
    # (kept for provenance even though dict_size wins), untouched.
    assert _train_config(cfg, 0, dict_size=576).dict_size_mult == 8


def test_run_sae_resolves_each_targets_own_dict_size_independently():
    """`run_sae` (tsfm_lens/sae/train.py) must resolve each target's own
    `dict_size` key separately -- a single train_cfg hoisted above the
    per-target loop (the pre-fix shape) would apply whichever target
    computed it last to every target, silently sizing the wrong dictionary
    for models earlier in the list."""
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
    # patchy gets an explicit absolute override; steppy is left to the
    # default dict_size_mult -- the pre-fix bug would have carried whichever
    # target's train_cfg the loop computed last onto the other target too.
    cfg.sae.targets = [{"model": "patchy", "layer": layer_a, "dict_size": 12},
                       {"model": "steppy", "layer": layer_b}]
    cfg.sae.epochs = 2
    cfg.sae.dict_size_mult = 2
    cfg.sae.k = 2
    cfg.sae.forecast_preservation_max_series = 16

    ctx = Context(cfg)
    d_in_steppy = ctx.store.load("steppy", layer_b, level="window",
                                 rows=np.array([0])).shape[-1]
    run_sae(cfg, ctx.hub, ctx.store, ctx.data, ctx.device)

    patchy_ckpt = load_json(run_dir / "sae" / "meta.json")["patchy/" + layer_a]
    steppy_ckpt = load_json(run_dir / "sae" / "meta.json")["steppy/" + layer_b]
    assert patchy_ckpt["dict_size"] == 12
    assert steppy_ckpt["dict_size"] == 2 * d_in_steppy
    assert steppy_ckpt["dict_size"] != 12


if __name__ == "__main__":
    test_default_is_one_seed_so_no_recorded_run_changes()
    test_seed_spread_reports_the_sample_sd_and_the_range_beside_it()
    test_the_floor_lookup_is_tri_state_and_unmeasured_is_not_failed()
    test_an_unmeasured_floor_renders_a_sentence_not_an_empty_block()
    test_the_training_seed_is_not_inert()
    print("SAE seed-floor tests passed")
