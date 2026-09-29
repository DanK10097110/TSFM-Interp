---
name: Bug report
about: Something behaved unexpectedly (a stage crashed, a number looks wrong, a check failed that shouldn't have)
title: ""
labels: bug
assignees: ""
---

## What happened

A clear description of the unexpected behavior.

## Which package and stage

- [ ] `tsfm_benchmark` (generation) / `benchmark_validation`
- [ ] `tsfm_lens` — stage: `_____` (e.g. `extract`, `l3`, `report`)

## Command run

```
(exact command line, including --config and any flags)
```

## Expected vs. actual

What you expected to see, and what you actually saw (paste the relevant log
lines, error message, or a screenshot of the report section).

## Config

Paste the relevant config section, or attach the config file. Redact any
absolute paths specific to your machine.

## Environment

- OS:
- Python version:
- Key library versions (`pip show torch zarr transformers` etc., or paste
  the output of `python run.py --doctor` / your `DEPENDENCIES.md` diff):
- GPU (if applicable):

## Is this reproducible on `configs/smoke.yaml` (mock models, no GPU)?

If yes, that's the easiest reproduction for a maintainer to run — please
include the exact steps.
