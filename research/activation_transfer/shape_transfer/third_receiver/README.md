# Can Timer reproduce forecast changes caused by removing an SAE feature in another forecaster?

The donor features are the original sixteen training-selected atoms for Chronos-Bolt-small and sixteen for TimesFM 2.5-200M. Each donor reference is its actual eight-step forecast change after removing one feature at that series' observed activation. Each recipient result is a live change to the pinned 84M-parameter Timer forecast. The donors used the last 32 observations' tokens at their dictionary training sites. Timer received a write to the post-block residual at `model.layers.4`, at the last of five data tokens, which covers 96 observations. Its deterministic head supplies point forecasts. All models saw the same standardized 480-step histories. [Locked design](PROTOCOL.md) · [Original donor material and splits](../../replication/README.md)

**Timer reproduced both donor models’ eight-lead effects, including the changes in forecast shape.** The adaptive method received each donor's and Timer's derivatives on the current item, with no recipient SAE. On M4 daily, it recovered 94.6% of the Bolt removals' horizon-centered energy by squared-error skill, and 88.9% for TimesFM removals. Scrambling the target lead order lost that shape. [Measured per-item Bolt removals](results/bolt.npz) · [Measured per-item TimesFM removals](results/timesfm.npz)

| Donor | Domain | Frequent atoms / active cases | Full skill | Centered skill [95% interval] | Wrong lead, centered skill | Mean donor, centered skill | Adaptive centered cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| Chronos-Bolt | M4 daily | 15 / 520 | 0.992 | 0.946 [0.927, 0.974] | -0.985 | 0.390 | 0.976 |
| Chronos-Bolt | Bitbrains | 4 / 127 | 0.948 | 0.892 [0.837, 0.975] | -0.944 | -0.967 | 0.961 |
| TimesFM | M4 daily | 14 / 691 | 0.981 | 0.889 [0.871, 0.921] | -1.082 | 0.494 | 0.951 |
| TimesFM | Bitbrains | 2 / 156 | 0.906 | 0.745 [0.554, 0.847] | -0.833 | 0.332 | 0.898 |

On M4 daily, the full forecast changes can hide the lead-order failure. Scrambling the recipient's gradients retained full skill of .957 for Bolt and .905 for TimesFM, even while centered skill fell below zero for both. Using the current donor derivative also improved centered skill over using the 96-calibration-history mean. Paired improvements were +.556 [.494, .668] for daily Bolt, +1.858 [1.404, 2.820] for Bitbrains Bolt, +.395 [.348, .466] for daily TimesFM and +.413 [.105, .785] for Bitbrains TimesFM. [All macro and per-feature scores, firing counts and paired intervals](results/scores.json)

The Bitbrains TimesFM result covers only two frequent atoms and 79 of their 156 adaptive writes hit the cap. Across the other three cells, ten of 1,338 active cases hit it. The experiment establishes that Timer has usable local directions for reproducing these specified responses. The same eight leads guide the solve and are then evaluated on a real nonlinear forecast. Shared SAE features and unconstrained forecast leads would require separate tests.

## Measurement

We use the donor's ordinary-state Jacobian and active decoder vector to predict the source removal's eight-lead forecast response. For the current series, we solve for a single Timer hidden vector producing that response under Timer's own Jacobian. Source calibration histories set one relative mean-versus-shape weight. The solve and the 20% hidden-norm cap are inherited verbatim from the two-recipient experiment. We compare the adaptive solve with the same method after scrambling Timer's lead order, and after replacing each item's donor derivative by the mean donor derivative across 96 distinct calibration histories. Every arm still uses the current Timer gradient and actual per-item donor SAE activation. The source model's observed ablated forecast only enters evaluation. [Gradient collection](timer.py) · [Transfer code](run.py) · [Original full-space methods](../full_space/PROTOCOL.md)

Each score is the equally weighted average across donor atoms that fired on at least eight histories in that domain. Signed cosine pools forecast entries across that feature’s active items. Recovery skill equals one minus target-versus-source squared error divided by source-removal energy. Zero skill is the score of making no change. Centering removes each item’s own eight-lead mean from both effects. Paired 95% intervals resample whole series 4,000 times and keep the feature set fixed. Domains and selected source features were previously studied. This comparison introduces a new checkpoint as a recipient of these atom removals. [Scoring](score.py)

An exact ordinary-versus-self-patch check on 416 donor-calibration and evaluation inputs yielded a maximum forecast error of zero. Across 24 prespecified signed finite-difference probes, relative error of the automatic derivative was 0.000349. The write cap was 2.28645 hidden L2 units, 20% of the recipient training median state norm over 1,024 separate histories. [Timer site and validation](results/timer.json) · [Measured finite differences](results/timer_validation.npz)

## Reproduction

Use the 480-step replication data and the original donor Jacobians. The following command collects Timer states and gradients, executes all recipient writes, computes paired uncertainty and checks selected real interventions using a separate additive hook and solver. The checkpoint’s trusted Python files are already present locally. Transformers copies these to the writable `HF_MODULES_CACHE` set by the script. [Replication data instructions](../../replication/README.md) · [Regenerate donor Jacobians](../full_space/jacobians.py) · [Reproduction script](reproduce.sh)

```bash
PYTHON=python bash research/activation_transfer/shape_transfer/third_receiver/reproduce.sh \
  /tmp/tsfm_atom_replication_20261120 /tmp/tsfm_full_space_sae_20261120 \
  /tmp/tsfm-third-fresh-jac /tmp/tsfm-third-fresh-results
```

The independent replay recomputed a singleton Timer Jacobian and all three writes on each of two deterministic active cases per donor and domain. Across the 24 physical writes, maximum forecast-effect disagreement was 7.15e-07 history SD, and maximum derivative component disagreement was 1.04e-07. It used a live additive hook and an independently constructed eight-dimensional weighted solve. [Replay measurements](results/replay.json) · [Independent replay code](replay.py) · [Control-test verification](MUTATION_CHECK.md)
