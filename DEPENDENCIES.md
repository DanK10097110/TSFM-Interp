# DEPENDENCIES.md — Environment & library versions

> **Purpose.** The authoritative, empirically-verified record of what's
> actually installed and working for this repo, on this machine, right now.
> `tsfm_benchmark/pyproject.toml`, `tsfm_benchmark/benchmark_validation/requirements.txt`,
> and `tsfm_model_analysis/tsfm_lens/requirements.txt`/`pyproject.toml` declare
> **loose minimum bounds** for pip installability; this file is the **exact,
> tested-together set** — the one to reach for when recreating the
> environment or diagnosing a version-drift bug. Several exact pins below
> exist only because a looser bound broke in a documented, sometimes
> nasty way (`CLAUDE.md` §11) — don't loosen them without re-reading why.
>
> **Last verified:** 2026-08-04, against the `tsfmPy` conda-forge
> environment, Windows 11, NVIDIA RTX 5070 (12 GB VRAM, driver 580.88,
> CUDA 13.0). Re-verify (see "Sanity checks" below) after any package
> upgrade and update this date.
>
> **Third-environment note (2026-08-05).** `tsfm_lens` was also run on a
> third, previously-undocumented machine — Linux, a pre-existing `cudaPy`
> conda env (not `tsfmPy`), 8× NVIDIA RTX A5000 — for `ROADMAP.md` §5.3's
> `medium_run_chronos_base.yaml` run. That env had **zarr 3.2.1** installed
> (violating the `<3` pin below) and would have crashed extraction
> immediately; fixed with `pip install "zarr>=2.16,<3"`, which resolved to
> the same `2.18.7` already verified here — no code or version-target
> change needed, this is a third confirmation the pin is load-bearing, not
> a new gotcha. `torch` in that env was `2.12.0` (vs. this file's verified
> `2.9.1+cu130`) and worked with no observed issue for that run, but was
> not deliberately re-verified package-by-package the way the `tsfmPy` env
> below was — **this file's exact-version table (§3) still describes only
> `tsfmPy`**; treat any other environment as "spot-check the pins in §5
> before trusting it," not as covered by this file.
>
> **Update this file whenever a library version changes** — a new pin, a
> loosened/tightened bound, a newly-discovered fragile interaction. See
> `CLAUDE.md` §11 for the trap-log pattern this file's "why pinned" notes
> follow, and `ROADMAP.md` §0 for the "don't let a doc silently drift stale"
> discipline this file is held to as well.

---

## 1. One shared environment, not two

Earlier sessions considered a second, GPU-only environment to avoid a
suspected MKL/CUDA conflict (`ROADMAP.md` §5 Findings, "CUDA pytorch is not
installable in the same conda-forge env..."). **That concern is resolved as
of 2026-08-04** — verified directly, not assumed:

```
numpy 2.1.0 matmul  ->  OK
torch 2.9.1+cu130, torch.cuda.is_available()  ->  True
sdv 1.14.0 (imports ctgan/deepecho)  ->  imports cleanly
torch.randn(1000,1000, device="cuda") @ itself  ->  OK
```

All four coexist in the single `tsfmPy` conda-forge environment, on Python
3.12.13. **`tsfmPy` is the one environment for the whole repo** — both
`tsfm_benchmark`/`benchmark_validation` (CPU-only workloads: generation,
validation, leakage audit) and `tsfm_model_analysis/tsfm_lens` (GPU-capable:
real-checkpoint extraction and analysis) run out of it. Nothing here needs a
second environment.

**GPU/CUDA status: fully integrated, not partial.** `torch.cuda.is_available()
== True`; the medium-scale real-checkpoint `tsfm_lens` run already exercised
this GPU end-to-end (`ROADMAP.md` §5.4, ~10.5 min for the full 12-stage
pipeline against `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small`).
Also newly confirmed this session: SDV's `PARSynthesizer` (the
`sequential_par` generator, `CLAUDE.md` §11.11/§11.14) defaults to
`cuda=True` in its constructor and imports/runs fine alongside CUDA torch and
numpy/MKL in this same environment — the GPU path for `sequential_par` that
`ROADMAP.md` §5 marked "currently unusable... decision deferred" is not
blocked by anything found in this environment today; that note in
`ROADMAP.md` is stale and should be read as superseded by this file.

---

## 2. Recreating the environment

```bash
conda create -n tsfmPy python=3.12 -y
conda activate tsfmPy

# Core numeric/ML stack — pin mkl explicitly (see §4 "why pinned")
conda install -c conda-forge numpy=2.1.0 mkl=2024.2.2 scipy pandas \
    scikit-learn pyyaml jinja2 tqdm zarr=2.18.7 plotly=5.24.1 \
    pycatch22 umap-learn dtaidistance datasets=2.21.0 tsbootstrap=0.7.1 \
    sdv=1.14.0 -y

# GPU-capable torch stack (CUDA 13.0 build; swap the index URL / build for
# a different CUDA version or CPU-only — see PyTorch's own install matrix)
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130

# Model libraries (real-checkpoint adapters)
pip install transformers==4.57.6 timesfm==2.0.2 chronos-forecasting==2.3.1

# Contrib adapter (models/contrib/) libraries -- ONLY needed for the specific
# checkpoint that contrib adapter targets, not for the core five hand-written
# adapters above. gluonts + lightning installed cleanly against this env's
# torch==2.9.1+cu130 / transformers==4.57.6 with no downgrade of either
# (ROADMAP.md Item D2).
pip install gluonts==0.17.0 lightning==2.6.6   # models/contrib/lag_llama_adapter.py

# tsfm_benchmark + tsfm_lens themselves, editable -- two SEPARATE packages,
# deliberately (CLAUDE.md sec 3, README.md "Install"): their dependency sets
# barely overlap and tsfm_lens's (torch/zarr/plotly/sklearn) are an order of
# magnitude heavier, so a single merged package would force every install to
# pull both. Install whichever half you need into this one shared env.
pip install -e .                              # from repo root: tsfm_benchmark + example_runs
pip install -e tsfm_model_analysis/tsfm_lens   # tsfm_lens
```

This is a reconstruction from what's verified installed today (§3), not a
byte-exact replay of the original install order — `conda`/`pip` will resolve
compatible transitive versions on a fresh install that may differ slightly
from the exact build strings in §3. If a fresh install disagrees with §3 in
a way that breaks something, trust §3 (it's what was actually tested) and
pin harder.

---

## 3. Exact verified versions (load-bearing packages)

| Package | Version | Source | Why this exact version matters |
|---|---|---|---|
| `python` | 3.12.13 | conda-forge | — |
| `numpy` | 2.1.0 | conda-forge | Golden-hash regression test (`CLAUDE.md` §7 invariant 1, `ROADMAP.md` sec 15 A8) is **currently green** against this exact version in this environment (re-verified 2026-08-06). The leading cross-version-instability hypothesis was checked directly (`default_rng(0).choice(100,5,replace=False)` and `.normal(size=5)` compared bit-for-bit against numpy 1.26.4 in an isolated venv) and **did not reproduce** — both primitives are identical across 1.26.4 and 2.1.0. The originally-reported mismatch (`CLAUDE.md` §11.13, a from-scratch environment on an unspecified earlier session) remains unexplained; still don't bump numpy casually, but the specific "choice isn't stream-stable" story is refuted for these two versions, not confirmed. |
| `mkl` | 2024.2.2 | conda-forge | Must match `numpy` 2.1.0's build era. A newer MKL (2025+, e.g. pulled in transitively by some `sdv`/pytorch installs) causes an unhandled SEH crash (`0xc06d007f`) on **any** matmul with this numpy build (`CLAUDE.md` §11.13). Verified together with CUDA torch 2.9.1+cu130 today with no crash (§1). |
| `scipy` | 1.18.0 | pip (pypi) | — |
| `pandas` | 2.3.3 | conda-forge | — |
| `scikit-learn` | 1.9.0 | conda-forge | — |
| `pyyaml` | 6.0.3 | conda-forge | — |
| `jinja2` | 3.1.6 | conda-forge | Report templating (`tsfm_lens/report/report.py`). |
| `plotly` | 5.24.1 | conda-forge | Report figures; `tsfm_lens/report/report.py` loads plotly.js 2.32.0 from CDN separately (embedded `<script>` tag, not this pip package). |
| `zarr` | 2.18.7 | pip (pypi) | **Must stay `<3`.** v3 renamed `Group.create_array`→ this code calls `create_dataset` (the v2 API) — v3 would break `extraction/store.py` immediately (`CLAUDE.md` §11.15). |
| `pycatch22` | 0.4.5 | conda-forge | catch22/catch24 feature extraction (`benchmark_validation/features.py`). |
| `umap-learn` | 0.5.12 | conda-forge | 3D embedding for the validation report / `tsfm_lens` clustering map. |
| `dtaidistance` | 2.4.0 | conda-forge | Fast C DTW for the leakage gate and shape matcher; both have a numpy fallback if this is absent. |
| `datasets` | 2.21.0 | conda-forge | **Must stay `<3`.** `datasets>=3` removed script-based dataset loading entirely, and `Monash-University/monash_tsf` is script-backed — a hard `RuntimeError`, not a warning (`CLAUDE.md` §11.9). |
| `tsbootstrap` | 0.7.1 | pip (pypi) | Public API was fully rewritten at some version before this; `generators.py:block_bootstrap` targets this version's functional `bootstrap(X, method=MovingBlock(...), ...)` API, not the older class-based one (`CLAUDE.md` §11.10). |
| `sdv` | 1.14.0 | conda-forge | `PARSynthesizer` (the `sequential_par` generator, `CLAUDE.md` §11.11). No `random_state` parameter at this version — that generator's reproducibility is best-effort, not bit-exact, by design. Pulls in `ctgan` 0.12.1 / `deepecho` 0.8.1. |
| `torch` | 2.9.1+cu130 | pip (pytorch.org wheel) | CUDA 13.0 build; confirmed working against the installed driver (580.88) and this numpy/mkl pair (§1). A CPU-only build works too for anything not needing GPU, but this repo's real-checkpoint `tsfm_lens` runs need the CUDA build. |
| `transformers` | 4.57.6 | pip (pypi) | Backend for the Chronos adapter. Installing this + `timesfm`/`chronos-forecasting` downgraded `huggingface-hub` from a previously-installed 1.26.0 to 0.36.2 — re-verified this doesn't break `datasets`-based Monash/ETT loading. |
| `huggingface-hub` | 0.36.2 | pip (pypi) | See `transformers` note above — a `timesfm`/`chronos-forecasting` transitive constraint, not chosen directly. |
| `timesfm` | 2.0.2 | pip (pypi) | **Not the same API as pre-2.0 releases.** `timesfm>=2.0` removed the old `TimesFmHparams`/`TimesFmCheckpoint`/`TimesFm` class API entirely; `models/timesfm_adapter.py` targets the new `TimesFM_2p5_200M_torch` class-based API and defaults to checkpoint `google/timesfm-2.5-200m-pytorch` (20 layers), not the older 2.0/500M checkpoint (50 layers) — see `CLAUDE.md` §11.8. An older pre-2.0 `timesfm` install needs a different (unmaintained-here) adapter. |
| `chronos-forecasting` | 2.3.1 | pip (pypi) | Backend for the Chronos-T5/Chronos-Bolt adapters. |
| `gluonts` | 0.17.0 | pip (pypi) | Backend for the `lag_llama` **contrib** adapter (`models/contrib/lag_llama_adapter.py`, ROADMAP.md Item D2) — not needed for the five hand-written adapters above. Installed clean against `torch==2.9.1+cu130` / `transformers==4.57.6` with no downgrade of either. ⚠️ **`gluonts.torch.modules.loss` was removed between gluonts<=0.14.4 (what the published `lag-llama.ckpt` was pickled against) and 0.17.0** — `torch.load()` on that checkpoint fails to unpickle its `hyper_parameters['loss']` entry with this version installed unless a `sys.modules` stub for the removed module is injected first (`lag_llama_adapter.py::_stub_removed_loss_module`, called before every `torch.load()` in that file). Same trap class as `CLAUDE.md` §11.9/§11.10 (a dependency's own breaking release invalidating a checkpoint pickled against an older version), just for an unpickling path instead of a public API. |
| `lightning` | 2.6.6 | pip (pypi) | Transitive via `gluonts[torch]`'s Lag-Llama-era checkpoint format (the vendored `LagLlamaModel`'s training code references PyTorch-Lightning conventions even though this repo's adapter only ever calls it in inference mode). |
| `einops` | 0.8.2 | pip (pypi) | Transitive (model library dependency). |
| `safetensors` | 0.8.0 | pip (pypi) | Transitive (checkpoint loading). |
| `accelerate` | 1.14.0 | pip (pypi) | Transitive (model library dependency). |
| `ctgan` | 0.12.1 | conda-forge | Transitive via `sdv`. |
| `deepecho` | 0.8.1 | conda-forge | Transitive via `sdv`; the actual RNN backend `sequential_par` trains (`CLAUDE.md` §11.14 — per-timestep training cost is why `max_train_length` truncation exists). |
| `tqdm` | 4.70.0 | conda-forge | Progress bars across both packages. |

Full transitive package list (~286 packages) can be regenerated any time
with `conda list -n tsfmPy`; it's intentionally not dumped here in full —
the table above is every package this repo's own code directly imports or
that has a documented version-sensitive gotcha (cross-checked against
`import`/`from` statements across both packages, not just recalled from
memory).

---

## 4. Sanity checks (run after any environment change)

```bash
# 1. numpy/MKL ABI crash check (CLAUDE.md §11.13) — should print OK, not crash
python -c "import numpy as np; print((np.random.rand(500,500) @ np.random.rand(500,500)).shape)"

# 2. CUDA availability
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 3. sdv + CUDA torch + numpy coexisting (the concern ROADMAP.md §5 once deferred)
python -c "
import numpy as np; np.random.rand(300,300) @ np.random.rand(300,300)
import torch; from sdv.sequential import PARSynthesizer
x = torch.randn(1000,1000, device='cuda'); (x @ x); torch.cuda.synchronize()
print('all clear')
"

# 4. tsfm_benchmark test suite (expect 40 passed / 1 known-failing golden-hash
#    test — CLAUDE.md §11.13/§9; anything else failing is a real regression)
python -m pytest tsfm_benchmark/tests -q

# 5. tsfm_lens end-to-end smoke test (expect all three to print "passed")
cd tsfm_model_analysis/tsfm_lens && python tests/test_smoke.py
```

---

## 5. Known fragile spots (see `CLAUDE.md` §11 for full detail)

- **numpy 2.1.0 + MKL 2025+ = crash.** Pin `mkl==2024.2.2` alongside it (§3).
- **numpy version *might* affect golden-hash reproducibility**, but the
  leading hypothesis (`Generator.choice(..., replace=False)` cross-version
  instability) was directly checked 2026-08-06 and did **not** reproduce
  between numpy 1.26.4 and 2.1.0 — treat any numpy bump as a reproducibility
  risk to re-verify (`ROADMAP.md` sec 15 A8's golden-hash test), not as a
  confirmed-broken mechanism.
- **`datasets>=3` breaks Monash loading entirely** (script-based datasets
  removed). Keep `<3`.
- **`zarr>=3` breaks `extraction/store.py`** (renamed `Group` API). Keep `<3`.
- **`tsbootstrap` and `timesfm` have both had full public-API rewrites**
  across major versions — a version bump on either is not routine, it needs
  the adapter/generator code re-verified against the new API, not just a
  version-string change.
- **`sdv`'s `PARSynthesizer` has no `random_state`** at this version —
  `sequential_par`'s reproducibility is best-effort (global seeding only),
  not bit-exact like the rest of the generation pipeline.
- **Chronos-2's own remote code (`amazon/chronos-2`, `trust_remote_code=True`)
  returns HF `ModelOutput` dataclasses from its encoder blocks, not plain
  tuples** — invisible across the three prior adapters (Chronos-T5,
  Chronos-Bolt, TimesFM), which never happened to return anything else, and
  surfaced only once a fourth architecture with a genuinely different
  forward-output convention was added. Fixed in `extraction/hooks.py`'s
  output-unwrapping helper (`CLAUDE.md` §11.21); not a version-pin issue,
  but the kind of gap a `transformers`/adapter bump elsewhere could
  reintroduce for the *next* new model, so worth re-checking then.
- **Sundial's own remote code (`thuml/sundial-base-128m`,
  `trust_remote_code=True`) is broken against `transformers` 4.57.6** — its
  documented `.generate()` path, and even a single cached `forward()` call,
  crash on `transformers.DynamicCache` attributes (`seen_tokens`,
  `get_usable_length`) that this checkpoint's remote code still expects but
  that were removed from `DynamicCache` in a release newer than what the
  model card recommends (pre-4.41) and older than this repo's pinned 4.57.6
  (`CLAUDE.md` §11.22). **Not fixed by changing the `transformers` pin**
  (would risk the same §11.8-class regression against the other three
  already-working adapters) — worked around in
  `models/sundial_adapter.py` by calling
  `SundialForPrediction.forward(..., use_cache=False)` directly instead of
  `.generate()` (sufficient since this repo's horizon, 64, is well under
  Sundial's 720-token one-shot flow-matching sampling limit). If
  `transformers` is ever bumped, re-check this specific checkpoint's remote
  code against the new version before trusting Sundial again — this is a
  third-party checkpoint's own code being version-fragile, not something a
  version bump in this repo's control can pre-empt.
  🔴 **Correction (2026-08-19): this is not Sundial-specific — it is
  lineage-wide.** Probing two further checkpoints for the generic-adapter work
  (`ROADMAP.md` §16 E3(b)) found `Maple728/TimeMoE-50M` and
  `thuml/timer-base-84m` crashing with the **identical**
  `'DynamicCache' object has no attribute 'get_usable_length'` under this same
  `transformers==4.57.6`, and both are fixed by the **identical**
  `use_cache=False`. So the entry above should be read as describing a class
  of checkpoint (thuml-lineage decoder-only time-series remote code written
  against pre-4.41 `transformers`), not one model's bug.
  `models/generic_hf_adapter.py` therefore passes `use_cache=False` to any
  forward whose signature accepts it, as a default rather than a per-adapter
  workaround — safe because disabling a cache cannot change a forecast, which
  is what separates it from the covariates that adapter refuses to fabricate.
  **Expect the next thuml-lineage checkpoint to need this too**, and expect a
  future `transformers` bump to change *which* attribute is missing rather
  than to fix it.
- **`datasets`/`AutoModel` note for the generic path (2026-08-19).**
  `transformers.AutoModel` resolves to the **bare backbone** for most
  time-series architectures (`PatchTSTModel`, not `PatchTSTForPrediction`;
  `TimerModel`, not `TimerForPrediction`), and a backbone has no forecast head,
  so a generic-adapter run against one produces no L0 at all. Worse, asking for
  the `*ForPrediction` class against a *backbone-only pretrain checkpoint*
  succeeds: `PatchTSTForPrediction.from_pretrained('ibm/patchtst-etth1-pretrain')`
  returns a model with ~70 randomly-initialized parameters after printing a
  warning to stderr and continuing. `GenericHFAdapter` turns that into a hard
  error via `output_loading_info=True`; if a future `transformers` changes the
  shape of that loading-info dict, that guard is the thing to re-check.
