# One small-model state predicts final observed-token states in both MOMENT checkpoints

A four-view crosscoder trained on aligned histories could reconstruct the **large model’s final observed-token state** from the small model’s middle state alone, at held-out R² **.850 and .852** across two seeds. When the small and large final states remained paired with each other but came from a different training history than the two middle states, the same test gave **.326 and .277**. Alignment across depth therefore mattered for inferring final states. This outcome concerns an internal representation of the observed past. Forecast responses and future values were absent from training. [Protocol](DICTIONARY_PROTOCOL.md)

The training rows were 46,080 observed eight-value patches across the first 768 M4 histories. Native observed slots 2–61 were aligned by item and position. The final view is the output of the post-encoder layer normalization at the observed patch. The code had 1,024 coordinates and selected up to 32 per row. Exactly 500 of 1,000 updates encoded only the small model’s middle state; the other 500 encoded all four views. Every batch supervised reconstruction of all four views with equal coefficients after dividing each view by its own training activation RMS. Both control fits used the same fixed derangement of the final views and the same seed-specific sample and masking schedules as the aligned fits. All 768 final-view training histories changed identity under the control, while patch position and within-depth pairing were preserved. [Implementation](train.py)

## Validation reconstruction

Each value is R² across the last observed patch of 256 held-out M4 histories, relative to that view’s **training** mean. Training and validation histories are disjoint.

| Training pairs and inference inputs | Seed | Small middle | Large middle | Small final | Large final |
|---|---:|---:|---:|---:|---:|
| Aligned, small middle only | 1 | .904 | .810 | .914 | .850 |
| Aligned, small middle only | 2 | .902 | .809 | .913 | .852 |
| Shuffled across depth, small middle only | 1 | .882 | .785 | .358 | .326 |
| Shuffled across depth, small middle only | 2 | .879 | .784 | .312 | .277 |
| Aligned, all four views | 1 | .898 | .852 | .916 | .884 |
| Aligned, all four views | 2 | .897 | .853 | .914 | .884 |
| Shuffled across depth, all four views | 1 | .848 | .816 | .854 | .838 |
| Shuffled across depth, all four views | 2 | .840 | .812 | .842 | .821 |

Supplying the actual final states makes them substantially easier to reconstruct in the shuffled controls. That all-view metric measures access to those states during encoding. The source-only metric measures whether the code inferred from the small middle state carries the same information. The mean Jaccard overlap of the source-only and all-view supports was .632/.635 when aligned and .359/.342 when shuffled. [Aligned fit, seed 1](results/aligned_22112401.json) · [Aligned fit, seed 2](results/aligned_22112402.json) · [Shuffled control, seed 1](results/shuffled_depth_22112401.json) · [Shuffled control, seed 2](results/shuffled_depth_22112402.json)

The earlier two-view masked fit scored .825/.822 for source-only reconstruction of **large middle**, slightly above .810/.809 with four aligned views. The four-view experiment adds a predictable final representation but shares its code budget with two additional objectives. The four-view model has **6,295,552** trainable parameters versus **3,148,288** in the original two-view fit. It also uses four reconstruction targets per update and more compute. Latent width, active-feature count, training histories, steps, minibatch size and masking fraction match. These budgets do not support a parameter-matched dominance claim. [Earlier two-view results](../DICTIONARY_FIDELITY.md)

## Released artifacts and genuine training checks

Each compressed artifact includes every trainable encoder/decoder weight and bias, four training means/scales, source-only M4 codes [1,024 × 1,024], blind daily/traffic codes [256 × 1,024], and NOAA codes [385 × 1,024]. NOAA rows use station-major/date-minor order for the 35 stations and eleven dates. `decoder_small` and `decoder_large` are the original **middle-view** decoder vectors in their respective raw hidden units. `decoder_small_final` and `decoder_large_final` are the final-view counterparts. The archives also include row IDs, M4 train/validation indices, the exact step mask and the control permutation. Neither held-out nor NOAA source codes require any large-model state. [Aligned artifact, seed 1](results/aligned_22112401.npz) · [Aligned artifact, seed 2](results/aligned_22112402.npz) · [Shuffled artifact, seed 1](results/shuffled_depth_22112401.npz) · [Shuffled artifact, seed 2](results/shuffled_depth_22112402.npz)

An independent implementation recreated every stored source-only activation and gate from the released weights on all 1,665 histories per fit, bit for bit. It independently reconstructed predictions against the archived aligned validation states and verified the source-only/full-view R² values. The maximum concatenated decoder squared-norm error across all four dictionaries was 2.66 × 10⁻⁷. The original final-state training endpoint archive and the separate native middle/final archive agreed exactly for the small model and to 5.37 × 10⁻⁷ for the large model. Two tests caught actual planted regressions where the small and large final views were assigned different histories or the large middle view leaked into the source-only encoder. [Independent reconstruction audit](results/training_audit.json) · [Code](audit_training.py) · [Regression checks](TRAINING_MUTATION_CHECK.md)

With the frozen M4/NOAA and MOMENT state caches from the parent study staged, run from the repository root:

```bash
PYTHON=python bash research/activation_transfer/moment/crosscoder/multilayer/reproduce_training.sh /tmp/four-view-reproduction
```
