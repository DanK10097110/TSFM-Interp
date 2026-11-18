# Register a feature screen and freeze its replication

This CPU-only workflow connects prepared [feature-interference cases](feature_interference.md) to a fixed discovery family and a fresh replication plan. It never launches a model. All output directories must be new. Run commands from `tsfm_lens/` in the project environment.

The example below uses only the registered CPU mock adapters. Save this as `runs/plan_input.yaml` after creating `runs/` if needed:

```yaml
design:
  context: 128
  horizon: 32
  periods: [32, 16]
  n_bundles: 32
  seed_base: 2500000
  native_background: true
models:
  - {name: mock_patch, adapter: mock_patch, checkpoint: ""}
  - {name: mock_wave, adapter: mock_wave, checkpoint: ""}
inference:
  primary_estimand: sign_probability
  n_boot: 199
  n_permutations: 999
  seed: 113
  ci: 0.95
```

Prepare inputs before forecasts:

```bash
python run_feature_interference.py --config runs/plan_input.yaml \
  --prepare-only --out runs/plan_prepared
```

Save this complete catalog as `runs/plan_catalog.json`. Add every intended prepared case before registration, including cases whose results might remain unavailable. The expected model order must match each case's frozen protocol.

```json
{
  "primary_estimand": "sign_probability",
  "alpha": 0.05,
  "minimum_absolute_mean": {
    "untouched_phase_phi": 0.1,
    "untouched_phase_psi": 0.1,
    "mixed_scaled_mse": 0.05,
    "model_by_composition": 0.05
  },
  "cases": [
    {
      "id": "period32",
      "prepared": "runs/plan_prepared",
      "report": "runs/plan_forecasts",
      "model_order": ["mock_patch", "mock_wave"]
    }
  ]
}
```

Register **before screening reads any results**. Registration freezes catalog membership, primary estimand, thresholds, model order, prepared config/protocol/corpus/metadata hashes, and expected forecast paths. It does not open completed reports. This is an auditable declaration, not cryptographic proof that nobody previously inspected outcomes.

```bash
python run_feature_search_plan.py register --catalog runs/plan_catalog.json \
  --out runs/plan_registered
python run_feature_interference.py --prepared runs/plan_prepared \
  --device cpu --out runs/plan_forecasts
python run_feature_search_plan.py screen \
  --registration runs/plan_registered/registration.json --out runs/plan_screen
```

Screening recomputes one global Holm correction from **raw selected-endpoint p-values across every registered case × model pair × four endpoints**. Missing reports and untestable endpoints remain at p=1; they never shrink the family. The report lists every candidate and negative. Casewise significance is insufficient.

The phase thresholds above are **radians**. The other thresholds use MSE divided by the common per-seed mixed energy `(a²+b²)/2`, including pure conditions. `model_by_composition` is the mixed model gap minus the equal-weight pure model gap; it is not a difference in native-minus-component-sum gains. Contrasts use later-listed minus earlier-listed models.

The practical gate uses the **absolute point mean**, not a confidence bound establishing that the practical margin is exceeded. With sign probability selected, the global exact sign test concerns `q=P(D>0 | D≠0, jointly identifiable)`, not expected-MSE superiority. Advancement additionally requires the mean and `q−0.5` directions to agree. Tiny significant effects and opposing mean/majority directions are not advanced. Mean mode retains its symmetry/exchangeability assumption; no secondary endpoint rescues the selected family.

Freeze a replication using an explicit fresh range of actual **generating IDs**:

```bash
python run_feature_search_plan.py plan --screen runs/plan_screen/screen.json \
  --fresh-seed-base 2600000 --n-bundles 32 --device cpu \
  --out runs/plan_replication
```

The range must not overlap any registered discovery case's `generator_seed` values. Local indices such as `seed=0` may recur in a fresh corpus. This check covers the registered discovery catalog, not a global seed ledger; independently avoid ranges used in any earlier inspected replication. If nothing qualifies, the result is `closed_no_replication_needed` and no native configs or commands are emitted.

Otherwise, inspect `plan_replication/plan.json`. It freezes only selected claims, their directions and thresholds, the new design/model order, and exact prepare/infer commands. For this single-case example those commands have this form:

```bash
python run_feature_interference.py \
  --config runs/plan_replication/period32/config.yaml --prepare-only \
  --out runs/plan_replication/period32/prepared
python run_feature_interference.py \
  --prepared runs/plan_replication/period32/prepared --device cpu \
  --out runs/plan_replication/period32/forecast_report
python run_feature_search_plan.py analyze-replication \
  --plan runs/plan_replication/plan.json --out runs/plan_replication_analysis
```

The planner never executes these commands. Replication analysis uses only the frozen selected family, retaining missing claims at p=1. It refuses modified plans/configs/designs, reused generating IDs, mismatched corpus bytes or row identities, and mismatched predictor declarations. Missing reports, artifacts, or required declarations remain unavailable at p=1 rather than being guessed. Reports must carry the producer's exact source hashes and explicit model/config/checkpoint-identifier bindings; identifiers are not weight checksums. Nonstandard forecast filenames must be declared in the catalog's optional `predictions` mapping before registration.

No endpoint is reselected or replaced after replication results arrive. Success concerns the declared statistical question and gates; it does not establish feature uniqueness, hidden mediation, or deployment utility. The mock walkthrough validates workflow behavior, not a scientific discovery.

## Real-background pilot continuation (ROADMAP §49.24–49.25)

The preferred §49.23 experiments 2–6 now have a bounded two-model implementation:
`experiments/feature_search/real_background_poc.py`. It freezes source-series
splits, controls, effect-size screening gates and CPU limits before inference.
[The pilot report](../../examples/feature_search_stage49/real_background/index.html)
and [reproduction instructions](../../examples/feature_search_stage49/real_background/README.md)
retain every model/experiment cell. None qualified for expansion. Transfer and
history have validity limitations, and the tiny SAEs fail response fidelity;
these do not refute the corresponding broader research hypotheses.

## Fourier decomposition diagnostic (ROADMAP §49.29–49.30)

`experiments/feature_search/fourier_adapter.py` implements input-only exact
spectral leaves and hierarchical forecast recombination. The bounded
`fourier_poc.py` study compares native/decomposed forecasting, phase and grouping,
distractor responses, real additions and nonstationary stress cases on both
models. Positive and negative findings are retained; no expansion is automatic.
[Results](../../examples/feature_search_stage49/fourier_poc/index.html) and
[reproduction instructions](../../examples/feature_search_stage49/fourier_poc/README.md)
include source-level uncertainty, control degeneracies and pretraining limits.
