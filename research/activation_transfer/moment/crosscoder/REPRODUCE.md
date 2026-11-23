# Reproducing the paired-dictionary comparison

Run from the repository root in its native forecasting environment. The MOMENT study's reconstruction instructions stage the pinned weights, M4 fit and evaluation captures, native SAE checkpoints, and USCRN station/date panel. The two families of crosscoders are fit on M4. All plotted aggregate comparisons and most case-level scores can also be audited from the persisted result arrays without downloading the models. [MOMENT materials](../REPRODUCE.md) · [Original cross-family activation captures](../../shape_transfer/REPRODUCE.md)

For the MOMENT checkpoint runner, include `.`, `tsfm_lens`, `tsfm_benchmark` and the staged `momentfm` installation on `PYTHONPATH`. Use the repo's pinned MOMENT-compatible packages, local model cache and CPU PyTorch. During the recorded runs we limited Torch, OpenBLAS and MKL to two threads. In these commands `python` names that environment. Scratch activation and training-weight paths are identified in the study scripts. The small compressed per-case results, fit diagnostics and figures live under this directory.

## Dictionaries, target doses, and causal weather checks

```bash
suite=research.activation_transfer.moment.crosscoder
for method in joint masked shuffled
do
  for seed in 22112401 22112402
  do
    python -m "$suite.dictionaries" "$method" "$seed"
  done
done
python -m "$suite.mechanism" capture
python -m "$suite.mechanism" run
python -m "$suite.evaluate" raw sae joint1 joint2 masked1 masked2 shuffled1 shuffled2 oracle_joint1 oracle_joint2
python -m "$suite.evaluation_scores"
PYTHON=python bash research/activation_transfer/moment/crosscoder/reproduce_causal.sh /tmp/moment-causal-reproduction
```

The causal wrapper computes into its output directory. Separate processes build the two-view diagnostics for comparison, and only small-view inputs generate doses in the source-only arms. The published causal summary includes all native forecasts, closed-gate costs, seed comparisons and crossed resampling. [Dictionary training](DICTIONARY_FIDELITY.md) · [Dose scores](EVALUATION.md) · [Native weather interventions](CAUSAL_RESULTS.md)

## Shared directions, measurement at depth, and exceptions

```bash
for method in joint masked
do
  python -m "$suite.directions" "$method" --finite
done
python -m "$suite.directions" shuffled
python -m "$suite.native_components"
python -m "$suite.patch_probe"
for method in joint masked shuffled
do
  python -m "$suite.readout_slots" "$method" --slot 61
done
python -m "$suite.readout_slots" masked --slot 60
for method in joint masked
do
  python -m "$suite.readout_dense" "$method"
done
for size in small large
do
  python -m "$suite.depth_trace" prepare "$size"
done
for method in joint masked shuffled
do
  python -m "$suite.depth_trace" run "$method"
done
python -m "$suite.selection" search
python -m "$suite.selection" finite
```

The published fitters reuse the 96/32 native derivative cache from the preceding functional calibration study, strictly separate from the 256 evaluation histories. [Historical derivatives and independent finite-difference checks](../dictionary_transfer/functional/README.md)

## Across-family comparison and independent measurements

Use the TimesFM–Bolt directory's runner or stage commands for its actual checkpoints, native alignment, ridge probes, dictionary fits and finite-dose replays. These fits supply the cross-family archive used by the joint figure and the disjoint-probe sensitivity. The other directories document the fresh USCRN experiment, physical input intervention and the independent target-dose exercise. [TimesFM–Bolt experiment](cross_family_directions/README.md) · [Fresh USCRN readouts](fresh_readout/README.md) · [Input sensitivity](input_sensitivity/README.md) · [Across-family dose checks](architecture_check/README.md)

```bash
for model in timesfm bolt
do
  python -m "$suite.cross_family_directions.traffic_phase" "$model"
done
python -m "$suite.cross_family_directions.traffic_phase" score
python -m "$suite.disjoint_probes"
python -m "$suite.overview"
python -m "$suite.motifs"
python -m "$suite.verify_claims"
python -m pytest -q research/activation_transfer/moment/crosscoder
```

The disjoint-probe control uses different M4 histories for the two linear readouts and excludes all 768 crosscoder-training histories from both probe fits. The plotting command recomputes the paired history-and-coordinate intervals from event-level measurements. The feature order and the M4 or traffic series must agree across both measurements before plotting.

An independent implementation materialized the entire 32-by-32 within-history cosine matrix for each observed-patch comparison. It recovered the 72 figure cells within 1.12 × 10⁻¹⁶ and the 48 traced native/probe contrasts within 2.23 × 10⁻¹⁶. Separate scripts in the component studies replay the trained weights, live interventions and final-head outputs. [Aggregate verification](results/overview/audit.json)

## Four-view sparse dictionaries

The depth capture step above stages the 60 observed final-layer states per M4 history and checkpoint. The four-view training runner adds both final layers to the paired middle-layer crosscoder and repeats the experiment with only the depth correspondence shuffled. At evaluation the small middle layer is the sole encoder input. The downstream runner reconstructs the six readouts and compares all real writes. [Training runner and artifact contract](multilayer/DICTIONARY_RESULTS.md) · [Evaluation results and reproduction](multilayer/EVALUATION.md)
