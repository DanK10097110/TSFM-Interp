---
name: New model adapter
about: Request, or announce your own contribution of, a new TSFM adapter
title: "Adapter: <model name>"
labels: adapter
assignees: ""
---

Read [`tsfm_lens/docs/ADDING_A_MODEL.md`](../../tsfm_lens/docs/ADDING_A_MODEL.md)
first — most of this template is that page's checklist, condensed.

## Model

- Checkpoint id (Hugging Face or other):
- Architecture family (decoder-only / encoder-decoder / encoder-only /
  something else):
- License:

## Have you tried the zero-code path?

```
python run.py --config <a copy of configs/generic_hf_timer.yaml, retargeted> --probe-adapter <name>
```

- [ ] All four probes resolved — no hand-written adapter needed.
- [ ] It refused. Paste the refusal reason(s) here — this *is* the spec for
      the hand-written adapter, not a dead end.
- [ ] It crashed below this repo's own code (bad checkpoint id, unrecognized
      `config.json`, etc.) — paste the traceback.

## If writing a hand-written adapter

- [ ] Scaffolded via `python run.py --new-adapter <name> --checkpoint <id>`
- [ ] `--check-adapter <name>` passes (or every failing row has a stated,
      understood reason — e.g. a genuine tier-2 ceiling)
- [ ] `--check-alignment <name>` diagonal-hit fraction is near its own
      measured ceiling at shallow layers
- [ ] Checked against `tsfm_lens/docs/ADDING_A_MODEL.md` §3's checklist
      (normalization, sampled-decoder seeding, precision, batch chunking,
      layer regex, encoder-decoder capture, reach controls, `random_init`
      causal-floor caveat, output scale)
- [ ] Added a mock-inner-module test with a planted answer + a decoy (see
      `tests/test_sundial_adapter.py` as the template)
- [ ] `render_adapter_docs.py` run, `tsfm_lens/ADAPTERS.md` updated
- [ ] Checkpoint license recorded on the model config

## What architecture assumption does this model break?

Per `ADDING_A_MODEL.md` §5: a model that exercises a genuinely new envelope
assumption (non-contiguous lag tokens, no attention at all, a non-causal
encoder, ...) is worth more than one that looks identical to an existing
adapter. What's new here?
