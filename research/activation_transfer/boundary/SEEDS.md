# Replication across independently trained dictionaries

We fitted two TopK SAEs to exactly the same 34,515 public-development activation rows from each backbone, changing the initialization and minibatch seed. Both used width 6,144, k=32 and 720 optimization steps. The archived dictionaries also saw 3,990 Monash augmentation rows whose saved activations were unavailable. Seeds 1 and 2 form the controlled same-data comparison. The archived checkpoint adds a sensitivity test across training recipes. The fresh fits used the archived public-development training rows. Historical exposure through the pretrained backbones or the earlier online Monash augmentation is unknown. Both fresh fits share the downstream calibration and evaluation series. [Training split and method](replica.py) · [Training hashes and learning curves](replica_training.json) · [Protocol](PROTOCOL.md)

## Can the same backbone’s atoms be matched across seeds?

We fitted one-to-one Hungarian assignments of calibration-active atoms between seed 1 and seed 2. The decoder assignment maximizes decoder-direction cosine in the shared hidden coordinates. The activation assignment maximizes centered clean-code profile cosine over 128 daily calibration histories. Each paired coefficient gets a nonnegative, clean-calibrated scale. We then predicted the complete seed-2 code and its edit on the same 80 untouched-by-fitting daily reserve items. The shuffled comparison permutes the assigned target indices within calibration frequency bands and fits the same scales. The dense states in this same-backbone comparison coincide exactly. [Matching code](seed_identity.py) · [Bolt measurements](seed_identity_bolt.json) · [TimesFM measurements](seed_identity_timesfm.json)

| Backbone | Assignment | Clean hard cosine | Hard edit cosine | Shoulder edit cosine | Clean-locked edit cosine | Shuffled hard edit cosine | Correct active IDs | Clean − hard edit, paired 95% interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bolt | decoder | 0.842 | 0.115 | 0.168 | 0.291 | 0.006 | 0.272 | +0.727 [+0.696, +0.756] |
| Bolt | activation | 0.883 | 0.120 | 0.183 | 0.372 | 0.028 | 0.520 | +0.763 [+0.740, +0.786] |
| TimesFM | decoder | 0.836 | 0.397 | 0.412 | 0.506 | 0.233 | 0.718 | +0.439 [+0.402, +0.479] |
| TimesFM | activation | 0.983 | 0.367 | 0.376 | 0.494 | 0.276 | 0.905 | +0.616 [+0.575, +0.658] |

The correct-active-ID column is the fraction of the target’s 32 clean active IDs present after remapping the independent source support. It differs from coefficient-weighted cosine, which can be high when a few large features align. Holding clean support fixed measures local slopes for the two dictionaries. The detailed files include the achievable cosine when the true target changes on all the matched coordinates are supplied. All direct matching results condition on 128 calibration histories.

## Can many-to-many maps recover the code change?

Independent rank-at-most-32 maps learn from paired perturbation changes on 128 daily histories and select their ridge coefficient using the original 48 validation histories. Each row below uses the same complete target code change. Each column changes what the map receives. The dense source receives only the displacement. Access to both dense endpoints and the known target encoder would reproduce the target exactly. [Method](seed_translation.py) · [Bolt](seed_translation_bolt.json) · [TimesFM](seed_translation_timesfm.json)

| Backbone | Seed-2 target | Hard source | Shoulder source | Locked source | Dense source | Input-only |
|---|---|---:|---:|---:|---:|---:|
| Bolt | hard | 0.142 | 0.204 | 0.240 | 0.256 | 0.198 |
| Bolt | shoulder | 0.326 | 0.422 | 0.461 | 0.498 | 0.373 |
| Bolt | locked | 0.472 | 0.608 | 0.672 | 0.690 | 0.520 |
| TimesFM | hard | 0.409 | 0.507 | 0.547 | 0.506 | 0.235 |
| TimesFM | shoulder | 0.611 | 0.747 | 0.755 | 0.754 | 0.460 |
| TimesFM | locked | 0.685 | 0.779 | 0.806 | 0.799 | 0.416 |

## Does smoothing reproduce across the cross-model seed pairs?

Here each row refits the original cross-model maps with that seed’s source code, then patches the target model. The target remains the receiving forecaster’s actual dense hidden change and its eight-step native forecast-patch response. Scores are mean signed cosines over 80 target-domain series. The intervals apply to the paired forecast cosine difference. [Seed 1 weekly hidden](weekly_seed1.json) · [Seed 2 weekly hidden](weekly_seed2.json) · [Seed 1 forecast audit](seed1_forecast_audit.json) · [Seed 2 forecast audit](seed2_forecast_audit.json)

| Dataset | Target | Source SAE | Hard → shoulder hidden | Hard → shoulder forecast | Paired forecast improvement, 95% interval |
|---|---|---|---:|---:|---:|
| Weekly | TimesFM | archived | 0.144 → 0.238 | 0.237 → 0.401 | +0.164 [+0.068, +0.271] |
| Weekly | TimesFM | seed 1 | 0.206 → 0.262 | 0.374 → 0.556 | +0.182 [+0.085, +0.278] |
| Weekly | TimesFM | seed 2 | 0.210 → 0.244 | 0.391 → 0.391 | +0.000 [-0.066, +0.071] |
| Weekly | Bolt | archived | 0.571 → 0.665 | 0.571 → 0.682 | +0.112 [+0.012, +0.218] |
| Weekly | Bolt | seed 1 | 0.364 → 0.673 | 0.434 → 0.694 | +0.260 [+0.124, +0.396] |
| Weekly | Bolt | seed 2 | 0.580 → 0.656 | 0.609 → 0.686 | +0.076 [-0.027, +0.189] |
| Bitbrains | TimesFM | archived | 0.052 → 0.117 | 0.108 → 0.187 | +0.079 [-0.030, +0.195] |
| Bitbrains | TimesFM | seed 1 | 0.083 → 0.140 | 0.109 → 0.227 | +0.118 [+0.022, +0.214] |
| Bitbrains | TimesFM | seed 2 | 0.104 → 0.113 | 0.178 → 0.223 | +0.045 [-0.042, +0.136] |
| Bitbrains | Bolt | archived | 0.458 → 0.539 | 0.550 → 0.596 | +0.046 [-0.050, +0.140] |
| Bitbrains | Bolt | seed 1 | 0.283 → 0.529 | 0.328 → 0.541 | +0.213 [+0.072, +0.351] |
| Bitbrains | Bolt | seed 2 | 0.486 → 0.539 | 0.583 → 0.607 | +0.024 [-0.060, +0.116] |

Results include the actual normalization and uncertainty for each native effect in the forecast audits, plus the unchanged waveform-only control. Both fresh seeds share the same sampled calibration and evaluation items. A truly independent population replication would require further checkpoints and series.
