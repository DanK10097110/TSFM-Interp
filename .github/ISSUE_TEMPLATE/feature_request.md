---
name: Feature request
about: Suggest a new capability, stage, metric, or benchmark generator
title: ""
labels: enhancement
assignees: ""
---

## What question would this answer that the pipeline can't answer today?

This repo is built around specific research questions (see the root
README's "What it answers"). A new feature request is strongest when it's
framed as a question the current stages/generators can't address.

## Proposed approach

How would this fit into the existing structure?

- A new `tsfm_lens` stage, or an addition to an existing one?
- A new `tsfm_benchmark` generator, corruption, or real-data source?
- Something else (a new report section, a new statistic, a new CLI flag)?

## Evidence class

If this produces a new kind of claim, which rung of the evidence ladder does
it belong on — descriptive, L1 (geometric), L2 (linear-translatable), L3
(causal within-model), illustrative, or confirmatory? (See the root README's
"The evidence-class ladder.")

## Backward compatibility

Would this change behavior for an existing config or a previously-sealed
corpus? New generator options must be opt-in (see `CONTRIBUTING.md`); a new
stage config field that changes an existing stage's fingerprint marks old
runs stale.

## Alternatives considered

Anything you already tried or ruled out.
