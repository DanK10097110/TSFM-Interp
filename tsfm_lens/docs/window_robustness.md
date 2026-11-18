# Cached window robustness diagnostic

Run this opt-in diagnostic from `tsfm_lens/`. It reads an existing activation
store and writes JSON plus a self-contained interactive HTML report to a **new**
directory. It does not run models or change pipeline stages or defaults.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python run_window_robustness.py \
  --run runs/panel7_v3_concat \
  --target TimesFM:stacked_xf.10:32 \
  --target Chronos-Bolt:encoder.block.3:16 \
  --target Chronos-2:encoder.block.6:16 \
  --windows 32,64,128 --crop 96:480 \
  --series 128 --train-series 64 --ranks 8,32 \
  --out runs/my_window_robustness
```

The CLI also caps PyTorch and BLAS to one thread internally. The library API
leaves thread control with its caller. The default working-array estimate limit
is 2048 MiB (`--memory-limit-mb`); this is an estimate, not an operating-system
memory limit. Cached-store decompression can read chunks larger than the selected
rows.

Each target requires `MODEL:LAYER:NATIVE_WIDTH`. Supply known regular native token
widths; the tool checks them against cached fine-position layout. It cannot infer
unknown token spans, establish empirical temporal locality, or reconstruct
native tokens wider than the cached window. Such source-contract mismatches fail
explicitly. Requested windows below a target's resolution are visibly skipped.

All window sizes use identical selected series and an identical physical extent.
The 480-step example explicitly crops to the trailing `[96,480)` interval, which
all three requested window sizes tile. Without an explicit crop, every window
must tile the full context; the tool never silently truncates separately for each
window. The source context and crop remain in the report.

The default sample contains 128 common cached IDs, selected with family
stratification. Sixty-four whole series train PCA; the other 64 provide similarity
measurements. `protocol.json` records source row IDs, human-readable series IDs,
the split, random seed, resolutions, and available alignment provenance.
`--rows 1,5,9,...` supplies an explicit common sample; missing or duplicated IDs
are refused. A subset cache without source-ID metadata is refused.
The report shows a compact source, crop, split, and native-width summary before
the comparison plots. Its collapsed scientific-provenance section retains the
full protocol, source row IDs, and any hashes supplied by the source artifacts.

Methods are `mean`, `native_concat`, and `common_grid`. The common physical grid
defaults to the least common multiple of declared native widths; override it with
`--grid-width` only when that width tiles every native span and requested window.
The effective physical grid is shown in the report separately from the requested
setting. Legacy auto-grid reports derive an explicitly labeled LCM from recorded
native widths; a requested `null` is not displayed as the effective resolution.
New JSON records `requested_grid_width`, `effective_grid_width`, and
`grid_width_source` separately. A legacy `common32_concat` label identifies its
32-step grid, while the report separately states the derived native-width LCM
and that the original request was not recorded.
`--methods` and `--ranks` accept comma-separated lists. Layers are explicit choices;
the tool does not search for the largest CKA.

The report includes raw CKA, training-only PCA rank controls, retained spectrum
participation ratios, and whole-series permutation nulls (999 draws by default).
Null p values are descriptive and unadjusted over the displayed grid. Windows
within a series stay together in each permutation. The identity-shuffle null does
not condition on inputs or generator family: significance cannot establish
learned shared features. Equal PCA rank does not imply equal spectrum or task
information.

Changing the window size changes the scientific observation: each row covers a
different physical extent, and the number of rows per series changes. Retaining
the same source series and crop does not hold that row-level estimand fixed.
Whole-series permutations preserve dependence but do not remove this change of
estimand. A higher score need not indicate more shared task information.
Matched-rank sensitivity controls nominal feature budget; it does not match the
effective spectra or make the different window observations interchangeable.

The complete-sequence native-concatenation estimate is computed once, with one row
per series. Its partition-invariance check separates retained content from the
changing window-row observational unit. It is a different statistic from
window-row CKA, and the report labels that distinction. There is no universal
best representation; use separate held-out concept probes to assess information
loss. This diagnostic establishes no causal use or shared concept claim.

A one-window-per-series sensitivity would answer a narrower question at a fixed
row count. It would also change the physical coverage or require an explicit
position-selection rule, so it is not a replacement for the current full-crop
diagnostic. Such a sensitivity is needed before attributing a score change to
window size while claiming the observational unit was held fixed; this tool
makes no such attribution.

Render an existing result without reading activations or repeating analysis:

```bash
python run_window_robustness.py \
  --report runs/my_window_robustness/results.json \
  --out runs/my_window_robustness_report
```

The report mode also reads neighboring `protocol.json` when present. Both modes
require a new output directory and write `report.html`, which embeds Plotly for
offline use.

The repository includes a [portable cached report](../../examples/window_robustness/report.html),
its [exact results](../../examples/window_robustness/results.json), and the
[recorded protocol](../../examples/window_robustness/protocol.json). No activation
store is included or needed to rebuild this report. From `tsfm_lens/`, run:

```bash
python run_window_robustness.py \
  --report ../examples/window_robustness/results.json \
  --out runs/window_portable_rebuild
```

The original source-cache paths remain provenance, rather than dependencies of
render-only mode. Existing score values are preserved; this command creates a new
HTML report without recomputing any measurement.
