# Crosscoder reconstruction with one or both models present

On the last observed patch of 256 held-out M4 histories, training half the batches with only the small encoder present raised small-to-large reconstruction R² from **.498/.518** across the two initializations to **.825/.822**. Permuting the large training histories while preserving patch position reduced it to **.349/.301**. These are predictions of the complete large hidden vector against a training-mean baseline. Each model saw the same 768 training histories and used 1,024 latent coordinates with per-row TopK at 32. [Training protocol](DICTIONARY_PROTOCOL.md) · [Training, encoding and archive contract](dictionaries.py)

| Training | Seed | Small R², small read | Large R², small read | Small R², both read | Large R², both read | Code support Jaccard, small vs both |
|---|---:|---:|---:|---:|---:|---:|
| Joint | 22112401 | .795 | .498 | .904 | .882 | .349 |
| Joint | 22112402 | .799 | .518 | .903 | .885 | .354 |
| 50% small-only | 22112401 | .911 | .825 | .905 | .864 | .675 |
| 50% small-only | 22112402 | .908 | .822 | .901 | .861 | .681 |
| 50% small-only, shuffled histories | 22112401 | .896 | .349 | .860 | .829 | .362 |
| 50% small-only, shuffled histories | 22112402 | .899 | .301 | .861 | .815 | .372 |

Reconstruction with both inputs available permits direct encoding of the receiving model. The shuffled fit can therefore recover much of the receiver variance with both inputs present. Source-only decoding instead tests which receiver variation the source state predicts. Masked training optimized exactly this missing-view configuration on 500 of its 1,000 updates. Its gain extends across the 15,360 held-out patches, with small-only large R² .768/.770 against .411/.417 for standard joint training. These diagnostics concern continuous hidden states. Target feature gates, activation doses and forecast responses require downstream measurement. [Seed-specific diagnostics](dictionary_results)

Each `<method>_<seed>.npz` in `/tmp/tsfm_moment_crosscoder` stores `m4_source`, `m4_both`, `noaa_source` and `noaa_both` on the last observed H16 token, plus `m4_ids`, `station_ids`, `dates`, `decoder_small` and `decoder_large`. M4 rows are in native order with 768 training and 256 held-out histories. NOAA rows are station-major, with 35 stations and 11 dates. Each decoder row is already expressed in the original hidden-state units per code unit. All six can be rebuilt from the original hidden captures with `python -m research.activation_transfer.moment.crosscoder.dictionaries <joint|masked|shuffled> <22112401|22112402>`. The `.pt` sidecar stores the raw-state means, per-view scales, and complete PyTorch state dictionary. [Archive audit](audit_dictionary_archive.py) · [Eight histories, two read modes per fit](dictionary_results/replay.json)

The source-only encoding test passed. A planted implementation that inserted a zero-valued target view and retained its learned target-bias contribution failed the test’s numerical assertion. [Missing-view test](test_dictionaries.py)
