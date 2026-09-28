# Curated example configs

A small, purpose-built set of configs for someone landing on this repo —
each demonstrates one thing, is fully commented inline, and was validated
against this checkout (loaded through the config loader; the real-model ones
were not run to completion here — see each file's own header for exactly
what was checked). They are copies or close derivatives of existing,
actively-used configs (see each header for its source), not new
experiments — the full config catalog and its dev/main split are documented
in [`../README.md`](../README.md).

`--list-configs` only scans `configs/*.yaml` directly, so these do not
appear in its output; validate one with:

```bash
python run.py --config configs/examples/<name>.yaml --doctor
```

| Config | Demonstrates | Models | Shape |
|---|---|---|---|
| `smoke_mock.yaml` | The fastest possible complete run (mock architectures, CPU, <1 min) | 2 mock | pair |
| `short_1model.yaml` | One real, deterministic model; the run-shape gate dropping comparison stages | Chronos-Bolt | solo |
| `medium_2model.yaml` | A real pair with the full SAE + concept-atlas chain at modest scale | TimesFM, Chronos-2 | pair |
| `large_4model.yaml` | The flagship concept-atlas run: 4 models, confirm on a sealed private split | TimesFM, Chronos-2, Sundial, Chronos-Bolt | panel |
| `sundial_solo.yaml` | A sampled decoder with an unsupported attention capability, solo shape | Sundial | solo |
| `lag_llama_solo.yaml` | A contrib (out-of-tree) hand-written adapter, solo shape | Lag-Llama | solo |
| `chronos_t5_solo.yaml` | An encoder-decoder architecture (only the encoder captured), solo shape | Chronos-T5 | solo |
| `generic_hf_timer.yaml` | The zero-code `generic_hf` adapter path — no adapter file at all | Timer (generic_hf) + Chronos-T5-Small | pair |

Every config beyond `smoke_mock.yaml` needs a real, sealed benchmark corpus
(see [`../../../tsfm_benchmark/README.md`](../../../tsfm_benchmark/README.md))
and a real checkpoint download; `data.path` in each is a placeholder
matching the convention used throughout this repo's configs
(`../benchmark_medium/public_dev` etc.) — point it at your own corpus.
