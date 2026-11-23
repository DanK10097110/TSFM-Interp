# Independent audit of the two SAE refits

I reconstructed the donor selection, candidate support, projection arithmetic, dose limits and representative 16-lead physical interventions separately from the study drivers. All eight combinations of checkpoint, donor seed and recipient seed were included. The model weights, context, tokens and readout are fixed within each comparison. The intervention dictionary changes when the recipient seed changes. [Independent replay code](audit_independent.py) · [Per-model checks and case IDs](results/independent_replay.json)

| Check | Bolt | TimesFM |
|---|---:|---:|
| SAE site | encoder.block.3 | stacked_xf.10 |
| Time intervals written within each 480-observation input | [448,464), [464,480) | [448,480) |
| Independently recomputed training median state radius | 5.638598 | 106.847778 |
| Max discrepancy between two independent cached captures | 0.00000036 | 0.00004578 |
| Representative recipient / uncapped source replays | 96 / 16 | 96 / 16 |
| Max discrepancy across the source and receiver 16-lead forecasts | 0.00000167 | 0.00000334 |
| Independent macro scores from individual firing events | 192 | 192 |

The two fresh dictionaries on each backbone contain 6,144 decoder rows and use TopK 32. The exact training sample-identity and activation-content hashes match across the two fits of that backbone. The 34,515 training rows, 80 actual epochs and 720 optimizer steps match, while the checkpoint-file hashes differ. These are **different training RNG seeds**. The seeded generator controls initialization and epoch minibatch order. The historical donor selection reproduced from only the training contexts on each seed and yielded the same 16 IDs as the original physical removal archives. The 320 daily evaluation series are distinct from one another and from the downstream donor-selection series. They are previously studied evaluation cohorts. This audit measures reproducibility and the prospective within-history design, without claiming pretraining-data independence. [Refit training implementation](../../../boundary/replica.py) · [Original donor selection and ablation](../../../replication/experiment.py)

For every firing query, the independently encoded source supports reproduced the archived row/atom mask. Every reported natural, attenuated and hidden-nearest ID was a positive member of the recipient’s current 32-candidate set. The natural reference was the median of the actual 16-step source removal at its native coefficient, without a radius cap. The blinded matcher saw only the eight measured early effects, the recipient's natural candidate moves and early derivatives. Its natural and bounded-attenuation choices were reconstructed independently. The zero-prior dense inverse and the previously measured source outcome are the same for both receiver-seed choices at fixed donor seed. Each recipient arm applies the same 20% cap to its final hidden vector. The score archive counts diagonal top-one only for donor features firing on at least eight histories in that cohort. This differs from the all-firing identification denominator in the original same-dictionary study.

The single-line and 32-direction-span controls know the full donor hidden vector in a coordinate system shared by both SAE fits. They permit arbitrary signed coefficients, limited only by the total receiver-vector radius. The squared-norm fractions describe the orthogonal projections before the final cap. The single-line projection was independently recovered with normalized inner products and the span with a separate SVD. A favorable multi-direction projection demonstrates attainable coverage with privileged information. It supplies neither a blind cross-seed partner nor an identified semantic decomposition. A matched generic 32-dimensional control would be required to attribute the result specifically to the learned receiving SAE span. Similarly, the two seed-specific donor draws can differ in feature identities and firing frequencies. Pairing the own and alternate receiver within a donor seed controls those differences. Whole-history and crossed history/feature intervals give conditional uncertainty; two training seeds do not establish a population-level seed interval.

Across all four ordered pairs on both models and both daily cohorts, direct per-donor squared-error calculations reproduced the 384 primary macro outcomes to within 2.3 × 10⁻¹⁶. On 32 deliberately sampled donor queries, direct real-model hooks reproduced 192 recipient writes as well as all 32 uncapped source ablations.

## Encoding the exact hidden endpoint

The added hard/locked comparison is a second privileged task. It uses the complete true donor edit and clean state in the same hidden frame. The locked code retains the starting support and rectifies its own new preactivations. It permits an old feature to fall to zero. The eighth-dose arm extrapolates one eighth of the displacement by a factor of eight. This is a finite difference that can cross a discontinuous boundary, so its hard-code version cannot be interpreted as an ordinary infinitesimal Jacobian. Both decoded differences are added to the original running state, preserving the original reconstruction residual.

An independent NumPy implementation constructed preactivations directly from the four checkpoint parameters, sorted the rectified activations, decoded hard and locked changes at both amounts, reapplied the correct single final cap, and ran the actual median forecast through 16 leads. Forty true, full-dose and eighth-dose physical writes across both source seeds and both backbones reproduced shipped outputs to within 0.00000573 normalized units. Switching and stable cases were both represented where available. The full-dose hard-versus-locked comparison changes the gate while keeping the case fixed. Any comparison of switching histories with stable histories remains observational and also depends on donor frequency in each subset. [Weight-level reconstruction](audit_encoding_independent.py) · [Physical encoding replays](results/independent_encoding.json)

Across all 2,420 full/eighth-dose endpoint checks, both clean and changed states had exactly 32 positive codes and the number of entering IDs equaled the number leaving. All stored entrant counts matched. Neither Bolt seed supplied eight stable full-dose observations for any one donor feature, so that conditional macro endpoint is unestimated. A separate sum-of-squares calculation reproduced 200 hard/locked/true scores across the all, switching and stable subsets (maximum error 3.6e-15).

Run both independent checks from the repository root after the study has produced the per-event archives and summaries.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
pkg=research.activation_transfer.shape_transfer.dictionary_prior.refit
python -m "$pkg.audit_independent"
python -m "$pkg.audit_encoding_independent"
```
