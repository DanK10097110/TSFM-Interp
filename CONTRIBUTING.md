# Contributing

This repo runs a `dev` → `main` workflow: `dev` carries the full research
history (experiment configs, standalone study-driver scripts, the raw
research log); `main` is the stable, publishable branch (the two packages,
their tests, and curated documentation and example configs). Land new work
on `dev` first; a change lands on `main` once it's decided and cleaned up.

## Before you open a PR

- **Tests use a planted, known answer, with a decoy on purpose, and a plant
  that must fail.** A test whose load-bearing assertion has never been
  confirmed to fail against a broken version of the code isn't actually
  testing that code. When you add a test, plant the regression it's meant to
  catch, run the test, confirm it fails, then revert the plant and confirm it
  passes. Report which plant failed which test in your PR description — see
  the two test fixes in this repo's own `git log` for the pattern.
- **Run the doc-render checks after any doc or config-schema edit:**

  ```bash
  cd tsfm_lens
  python render_stage_docs.py --check
  python render_glossary.py --check
  python render_adapter_docs.py   # should produce no diff
  ```

  `tsfm_lens/README.md`'s stage-docs and glossary sections, and
  `tsfm_lens/ADAPTERS.md` in full, are generated from `stage_docs.py`,
  `glossary.py`, and `adapter_docs.py` respectively — edit those files, never
  the generated markdown between the `BEGIN`/`END` markers.
- **New generator options are opt-in.** A change to `tsfm_benchmark`'s
  default archetype list, corruption pipeline, or any other draw sequence
  changes what every existing seed produces, silently breaking bit-exact
  regeneration of every sealed corpus already built from it (golden hashes
  pinned in `tsfm_benchmark/tests`). Default a new knob to reproduce today's
  behavior exactly; require an explicit opt-in for the new one.
- **Config fingerprints.** Adding a field to a whole-section config key
  (`tsfm_lens/manifest.py`'s `config_keys`) marks every existing run whose
  artifacts depend on that section stale — reran runs then need
  `--allow-stale`. A field no stage artifact actually depends on should be
  declared `metadata={"stage_input": False}` instead, so it doesn't
  needlessly invalidate old runs.
- **No hardcoded absolute paths.** Every path is relative to the repo, to
  `__file__`, or to a CLI argument — never a literal `/home/...` or
  `/common/...`. Config placeholders like `../benchmark_medium/public_dev`
  are relative and fine; a literal machine path is not.
- **Always pass `encoding="utf-8"`** to `read_text`/`write_text` and any
  other text I/O — a missing encoding silently picks up the platform default.
- **Degrade gracefully and loudly, never silently.** An unsupported
  capability skips with a stated reason that's *rendered*, not only logged.
  A broken assumption fails loudly rather than producing a plausible-looking
  wrong answer.

## Running the tests

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate <your-env>
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# tsfm_benchmark
cd tsfm_benchmark && PYTHONPATH=. python -m pytest tests -q

# tsfm_lens
cd tsfm_lens && python -m pytest tests/test_smoke.py -q       # full pipeline, mock models
python -m pytest tests/<file>.py -q                            # a specific test file
```

`tests/conftest.py` caps threads itself, but exporting the thread-count env
vars first avoids oversubscription on a shared machine.

## Adding a model adapter

See [`tsfm_lens/docs/ADDING_A_MODEL.md`](tsfm_lens/docs/ADDING_A_MODEL.md).

## Adding a benchmark generator or real-data source

See [`tsfm_benchmark/README.md`](tsfm_benchmark/README.md)'s "Generators and
archetypes" and "Adding a real-data source" sections.

## Style

- Docstrings at module and function level; module docstrings say *why* the
  module exists and what it inherits from the previous stage. No per-line
  comments in Python. Config YAML is the deliberate exception — it's the
  user-facing surface, so it's documented inline.
- Prefer established libraries (numpy, scipy, sklearn, torch, zarr v2,
  plotly, jinja2) over hand-rolled equivalents. The two deliberate
  exceptions are the metrics in `tsfm_lens/analysis/l0_behavioral.py` and the
  bootstrap designs in `tsfm_lens/analysis/stats.py`: short, auditable code,
  because the formulas *are* the substance being tested.
- Never bootstrap or split on windows — the series is always the resampling
  unit (L0: paired bootstrap; L1: cluster bootstrap; L2: validation-series CI
  with probes held fixed; L3: paired cluster bootstrap).

## Reporting a bug or requesting a model

Use the issue templates under `.github/ISSUE_TEMPLATE/` — there's a
dedicated one for requesting or contributing a new model adapter.
