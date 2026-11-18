# Dictionary pilot 4951000

The source is `runs/selective_replication4942000`, with 224 dependent case rows,
32 FIT and 32 TEST independent sources (16 synthetic plus four per real domain in
each split). Training uses only full-token FIT block-output states. No future
targets or TEST-selected model settings enter training. All token weights sum
to equal mass per source, then divide across that source's cases/tokens.

The primary cut is block0: Bolt has nine tokens ×512 residual coordinates;
TimesFM has four tokens ×1280. Compare signed magnitude-TopK native coordinates,
random orthogonal coordinates, FIT PCA, the repository TopKSAE, and a simplified
Matryoshka-inspired shared-code two-prefix TopK SAE. Learned widths equal d;
k is 8/16; steps500; batch64; seeds4951000–4951002. Native/PCA are deterministic
and fitted once per k; rotation and learned dictionaries use all three seeds.
Every learned candidate also gets direct-coordinate controls at its actual
per-token L0; realized zero/tie shortfalls are reported, never assumed matched.

TopK and nested share parameters, initializations, sampled token sequences and
optimizer steps. Two nested reconstructions add compute: dense forward matmul
FLOPs are analytically 1.25× TopK at even width, excluding TopK, backward and
optimizer costs. Direct controls have no neural training passes. Recorded counts
include parameters, sampled tokens, equivalent token passes, reconstruction
terms, wall time, RSS and observed activity/dead rates. No equal-FLOP claim.

Continuous probes use FIT correlation screening plus ridge .01 with 1/4 chosen
features; mean and RMS token pooling are separate. Native-vs-rotation sparse
selectivity can identify a useful basis even when their full spans agree.
Targets include physical spectral coefficients, coefficients in full-history-SD
units, coefficients of the actual model-normalized input, and level/scale/trend
nuisances. Input generic and spectral-oracle controls are explicit. Every feature
arm also gets the same normalizer metadata as its metadata-only comparator,
with all regression/scaling and feature selection fitted on FIT only.

Analytical metadata respects Bolt's full-history population mean/SD and TimesFM's
cumulative32-step-patch statistics. FP64 history-derived metadata approximates
native FP32 arithmetic; it is not a hook measurement. Original real-unit source
normalization was applied upstream: physical coefficients here mean the cached
input units. The paired perturbation diagnostic uses background-locked states.
Phase slices and background/domain TEST results are descriptive; fixed injected
phase is not crossed-phase generalization. Noise response is a nuisance control.

Native preservation is mandatory. A separate monitored stage takes the first
recorded case from four TEST sources/domain (20 receivers, including four of16
synthetic TEST sources), uses singleton forecasts, exact repeat/self/saved-state
gates at the same cut, and exact locked-counterfactual suffix replay. It measures
reconstruction distortion and FIT-selected1/4-feature removal against two
Gram-preserving random edits; invalid random edits cannot support a positive.
Sinusoid-vs-noise native response changes measure operational selectivity.
Independent random edits on +/- branches make that response null conservative.
No scalar threshold automatically declares usefulness or monosemanticity.

The missing-FFN-hook worker captures/replays the input to TimesFM `ff1` or Bolt
`DenseReluDense.wo`. These are actual activated/gated expansion coordinates,
distinct from residual coordinates, and are not labeled PatchTST post-GELU.
It is prepared separately; the main queue does not launch it. PM35's random-init
decodability warning still applies: trained-vs-init causal controls are missing.

Literature scope: [PatchTST post-GELU evidence](https://arxiv.org/html/2605.05151v1)
does not imply general TSFM monosemanticity. [Matryoshka](https://arxiv.org/html/2503.17547v1)
motivates nested reconstruction; this pilot omits its BatchTopK/AuxK recipe.
[BatchTopK](https://arxiv.org/html/2412.06410v1) is deferred; any future inference
threshold must be FIT-frozen and batch-independent. [TSAE](https://arxiv.org/html/2610.04925v1)
motivates temporal/forecast checks; its main controlled PatchTST/ETT comparison
and variable routing do not transfer directly to these univariate adapters.
This is neither a TSAE nor a complete Matryoshka reproduction.

From `tsfm_lens`, with the cudaPy interpreter and all thread env vars set to1:

```bash
python -m pytest tests/test_dictionary_pilot.py -q
python -m experiments.feature_search.dictionary_pilot prepare
python -m experiments.feature_search.dictionary_queue --slot-granted
```

The coordinator explicitly granted slot1 after tests/protocol freeze. Every
stage has an external600s/3GiB parent+worker RSS monitor and one compute thread.
The serial queue stops on failure. It writes stage `knowledge.json` with concise
results/pointers, preserves detailed `results.json`, and records slot release.
Existing sources and outputs are never overwritten; no commits or ledger edits.
