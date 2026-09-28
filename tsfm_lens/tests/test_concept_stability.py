"""Tests for ROADMAP.md sec 37 P2: seed stability of atlas concepts, and the
within-model transfer ceiling (`extraction/store.py`'s `replicate` axis,
`sae/train.py::train_sae_replicate`, `sae/stability.py::concept_stability`).

All synthetic, with planted answers, CPU-only. Each load-bearing assertion is
confirmed to discriminate against a planted regression (`CLAUDE.md` sec 9/
sec 11.55's corollary -- a plant that changes nothing is indistinguishable
from a working guard, so every negative below states what it would have
missed).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.store import ActivationStore, save_meta
from tsfm_lens.sae import train as train_mod
from tsfm_lens.sae.stability import concept_stability
from tsfm_lens.sae.transfer import _seed


def _cfg(n_sae_seeds=2, top_k=15, n_null=100, transfer_seed=0):
    return SimpleNamespace(
        concepts=SimpleNamespace(n_sae_seeds=n_sae_seeds),
        sae=SimpleNamespace(transfer_top_k=top_k, transfer_n_null=n_null,
                            transfer_seed=transfer_seed),
    )


def _meta(n, strata):
    """`strata`: list of (label, count) pairs summing to n."""
    labels = []
    for label, count in strata:
        labels += [label] * count
    assert len(labels) == n
    return pd.DataFrame({"archetype": labels, "family": labels})


def _make_store(tmp_path, n=120, n_windows=2, window=8, context_len=16):
    store = ActivationStore.create(tmp_path / "activations.zarr", n_series=n,
                                    n_windows=n_windows, window=window, context_len=context_len)
    return store


def _write_sae(store, model, layer, pooled, replicate=0):
    """`pooled`: `(n, F)`. Broadcasts to a trivial window axis of size
    `n_windows` (identical at every window) so `write_sae_batch`'s own
    mean-pooling reproduces `pooled` exactly."""
    n, f = pooled.shape
    n_windows = store.root.attrs["n_windows"]
    store.init_sae_layer(model, layer, f, replicate=replicate)
    window_feats = np.broadcast_to(pooled[:, None, :], (n, n_windows, f)).astype(np.float32)
    store.write_sae_batch(model, layer, 0, window_feats, replicate=replicate)


def _atlas_one_part(concept_id, model, layer, feature_ids, n_models=1):
    """A minimal `concept_atlas.json`-shaped dict with exactly one concept
    whose members all live at one (model, layer) target -- i.e. one PART."""
    rows = [{"model": model, "layer": layer, "feature": fid, "concept": concept_id}
           for fid in feature_ids]
    return {"rows": rows, "concepts": [{
        "concept": concept_id, "n_models": n_models,
        "members": list(range(len(feature_ids))),
    }]}


# ---------------------------------------------------------------------------
# 1. replicate=0 backward compatibility
# ---------------------------------------------------------------------------

def test_replicate_zero_is_byte_identical_to_unreplicated_keys(tmp_path):
    store = _make_store(tmp_path, n=10, n_windows=2)
    pooled = np.arange(30, dtype=np.float32).reshape(10, 3)
    _write_sae(store, "ModelA", "layer1", pooled, replicate=0)

    loaded_default = store.load("ModelA", "layer1", level="series", space="sae")
    loaded_r0 = store.load("ModelA", "layer1", level="series", space="sae", replicate=0)
    assert np.array_equal(loaded_default, loaded_r0)
    assert "sae/ModelA/layer1" in store.root
    assert "sae_pooled/ModelA/layer1" in store.root
    # negative: replicate=0 must NOT create a suffixed sibling group -- a
    # regression that always suffixed (even at 0) would still pass the
    # array-equality check above but would miss this.
    assert "sae_r0/ModelA/layer1" not in store.root
    assert store.has_sae_features("ModelA", "layer1")
    assert store.has_sae_features("ModelA", "layer1", replicate=0)


def test_replicate_nonzero_uses_a_sibling_group(tmp_path):
    store = _make_store(tmp_path, n=10, n_windows=2)
    pooled0 = np.zeros((10, 3), dtype=np.float32)
    pooled1 = np.ones((10, 3), dtype=np.float32)
    _write_sae(store, "ModelA", "layer1", pooled0, replicate=0)
    _write_sae(store, "ModelA", "layer1", pooled1, replicate=1)

    assert "sae_r1/ModelA/layer1" in store.root
    assert store.has_sae_features("ModelA", "layer1", replicate=1)
    loaded0 = store.load("ModelA", "layer1", level="series", space="sae", replicate=0)
    loaded1 = store.load("ModelA", "layer1", level="series", space="sae", replicate=1)
    # negative: if replicate=1 aliased replicate=0's array, this would fail.
    assert not np.array_equal(loaded0, loaded1)
    assert np.array_equal(loaded0, pooled0)
    assert np.array_equal(loaded1, pooled1)


def test_missing_replicate_raises_with_actionable_hint(tmp_path):
    store = _make_store(tmp_path, n=10, n_windows=2)
    _write_sae(store, "ModelA", "layer1", np.zeros((10, 3), dtype=np.float32), replicate=0)
    with pytest.raises(KeyError, match="replicate=1"):
        store.load("ModelA", "layer1", level="series", space="sae", replicate=1)


# ---------------------------------------------------------------------------
# 2. Identical-copy replicate -> ceiling exactly 1.0
# ---------------------------------------------------------------------------

def test_identical_replicate_gives_ceiling_one(tmp_path):
    """A replicate whose pooled features are a BYTE COPY of the primary's own
    must be perfectly reciprocal -- the transfer test asks whether the
    replicate's best feature separates the primary's own top-k, and a copy's
    best feature IS the primary's own feature."""
    rng = np.random.default_rng(0)
    n = 150
    strata = [("a", 50), ("b", 50), ("c", 50)]
    meta = _meta(n, strata)
    save_meta(tmp_path, meta)
    store = _make_store(tmp_path, n=n, n_windows=2)

    f = 6
    pooled = rng.normal(size=(n, f)).astype(np.float32)
    # give feature 0 real structure so its top-k is non-degenerate
    boosted = rng.choice(n, size=20, replace=False)
    pooled[boosted, 0] += 4.0
    _write_sae(store, "ModelA", "layer1", pooled, replicate=0)
    _write_sae(store, "ModelA", "layer1", pooled.copy(), replicate=1)

    atlas = _atlas_one_part(0, "ModelA", "layer1", [0])
    cfg = _cfg(n_sae_seeds=2, top_k=15, n_null=80)
    out = concept_stability(tmp_path, atlas, cfg)

    assert out["measured"] is True
    part = out["concepts"][0]["stability"]["parts"][0]
    assert part["reciprocal_at"] == [True]
    assert part["stable"] is True
    assert out["concepts"][0]["stability"]["stable"] is True
    assert out["ceiling"]["by_model"]["ModelA"]["rate"] == pytest.approx(1.0)

    # negative: corrupting the "identical" copy (shuffling series order,
    # which any real bug in `write_sae_batch`'s row alignment could do)
    # must NOT still read as a perfect ceiling.
    shuffled = pooled.copy()
    rng.shuffle(shuffled)
    store2 = _make_store(tmp_path / "shuffled_store", n=n, n_windows=2)
    save_meta(tmp_path / "shuffled_store", meta)
    _write_sae(store2, "ModelA", "layer1", pooled, replicate=0)
    _write_sae(store2, "ModelA", "layer1", shuffled, replicate=1)
    out_shuffled = concept_stability(tmp_path / "shuffled_store", atlas, cfg)
    assert out_shuffled["ceiling"]["by_model"]["ModelA"]["rate"] < 1.0


# ---------------------------------------------------------------------------
# 3. Random-features replicate -> reciprocal rate stays within its own null
#    (load-bearing negative: a broken statistic would systematically inflate
#    this above chance, exactly CLAUDE.md sec 11.37's shape).
# ---------------------------------------------------------------------------

def test_pure_noise_replicate_reciprocal_rate_is_not_inflated(tmp_path):
    """The primary's concept is REAL (feature 0 separates a planted subset);
    the replicate dictionary is PURE, unrelated noise. The reciprocal rate
    across many independent concepts drawn this way must land near the
    forward leg's own false-positive rate (p95 threshold means the forward
    leg alone should clear at roughly 5% under pure noise), not near 1.0 --
    a bug that always returns `reciprocal=True` (e.g. `clears` or
    `rev_clears` accidentally hardcoded, or the null comparison inverted)
    would push this to 100%."""
    rng = np.random.default_rng(1)
    n = 200
    strata = [("a", 100), ("b", 100)]
    meta = _meta(n, strata)
    save_meta(tmp_path, meta)
    store = _make_store(tmp_path, n=n, n_windows=2)

    n_concepts = 25
    f = 8
    primary = rng.normal(size=(n, f)).astype(np.float32)
    replicate = rng.normal(size=(n, f)).astype(np.float32)  # unrelated noise
    _write_sae(store, "ModelA", "layer1", primary, replicate=0)
    _write_sae(store, "ModelA", "layer1", replicate, replicate=1)

    atlas = {"rows": [], "concepts": []}
    for cid in range(n_concepts):
        fid = int(rng.integers(0, f))
        atlas["rows"].append({"model": "ModelA", "layer": "layer1", "feature": fid, "concept": cid})
        atlas["concepts"].append({"concept": cid, "n_models": 1, "members": [len(atlas["rows"]) - 1]})

    cfg = _cfg(n_sae_seeds=2, top_k=20, n_null=150)
    out = concept_stability(tmp_path, atlas, cfg)

    rate = out["ceiling"]["by_model"]["ModelA"]["rate"]
    # A single-leg p95 null clears at ~5% by construction; reciprocal (two
    # independent-ish legs, one of which -- the reverse -- is scored against
    # an UNRELATED primary score here) should not exceed a generous multiple
    # of that under pure noise.
    assert rate <= 0.30, (
        f"pure-noise replicate reciprocal rate {rate} is not distinguishable "
        f"from a systematically inflated statistic")


# ---------------------------------------------------------------------------
# 4. Planted two-latent-factor test
# ---------------------------------------------------------------------------

def test_two_latent_factors_only_the_shared_one_is_stable(tmp_path):
    """Primary has two concepts: concept A (feature 0) encodes a REAL latent
    factor that also exists in the replicate's dictionary (as a different
    feature index, with noise); concept B (feature 1) is primary-only noise
    with nothing corresponding in the replicate. Stability must separate
    them: A stable, B not."""
    rng = np.random.default_rng(2)
    n = 240
    strata = [("a", 80), ("b", 80), ("c", 80)]
    meta = _meta(n, strata)
    save_meta(tmp_path, meta)
    store = _make_store(tmp_path, n=n, n_windows=2)

    f = 5
    latent_members = rng.choice(n, size=40, replace=False)
    latent = np.zeros(n)
    latent[latent_members] = 1.0

    primary = rng.normal(scale=0.3, size=(n, f)).astype(np.float32)
    primary[:, 0] += 3.0 * latent          # concept A: real, shared latent
    primary[:, 1] += rng.normal(scale=3.0, size=n)  # concept B: primary-only noise

    replicate = rng.normal(scale=0.3, size=(n, f)).astype(np.float32)
    replicate[:, 2] += 3.0 * latent        # SAME latent, different feature slot
    # replicate has nothing corresponding to concept B by construction

    _write_sae(store, "ModelA", "layer1", primary.astype(np.float32), replicate=0)
    _write_sae(store, "ModelA", "layer1", replicate.astype(np.float32), replicate=1)

    atlas = {
        "rows": [
            {"model": "ModelA", "layer": "layer1", "feature": 0, "concept": 0},
            {"model": "ModelA", "layer": "layer1", "feature": 1, "concept": 1},
        ],
        "concepts": [
            {"concept": 0, "n_models": 1, "members": [0]},
            {"concept": 1, "n_models": 1, "members": [1]},
        ],
    }
    cfg = _cfg(n_sae_seeds=2, top_k=25, n_null=150)
    out = concept_stability(tmp_path, atlas, cfg)

    by_concept = {c["concept"]: c["stability"] for c in out["concepts"]}
    assert by_concept[0]["stable"] is True, "the shared latent factor must be stable"
    assert by_concept[1]["stable"] is False, "primary-only noise must NOT be stable"
    assert out["n_stable"] == 1 and out["n_scored"] == 2
    assert out["frac_stable"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4b. Stability requires ALL replicates reciprocal, not a majority
#     (sec 37 P2's own pre-registered "all-of-n, not 1-of-n" decision).
# ---------------------------------------------------------------------------

def test_stability_requires_all_replicates_not_a_majority(tmp_path):
    """Three seeds -> two replicates. Replicate 1 shares the primary's real
    latent factor (reciprocal clears); replicate 2 is unrelated noise (does
    not clear). A majority-vote implementation would call this part stable
    (1 of 2 clearing looks like "mostly agrees"); the spec requires ALL."""
    rng = np.random.default_rng(4)
    n = 240
    strata = [("a", 80), ("b", 80), ("c", 80)]
    meta = _meta(n, strata)
    save_meta(tmp_path, meta)
    store = _make_store(tmp_path, n=n, n_windows=2)

    f = 5
    latent_members = rng.choice(n, size=40, replace=False)
    latent = np.zeros(n)
    latent[latent_members] = 1.0

    primary = rng.normal(scale=0.3, size=(n, f)).astype(np.float32)
    primary[:, 0] += 3.0 * latent

    replicate1 = rng.normal(scale=0.3, size=(n, f)).astype(np.float32)
    replicate1[:, 2] += 3.0 * latent        # shares the real latent -> reciprocal

    replicate2 = rng.normal(scale=0.3, size=(n, f)).astype(np.float32)  # pure noise

    _write_sae(store, "ModelA", "layer1", primary, replicate=0)
    _write_sae(store, "ModelA", "layer1", replicate1, replicate=1)
    _write_sae(store, "ModelA", "layer1", replicate2, replicate=2)

    atlas = _atlas_one_part(0, "ModelA", "layer1", [0])
    cfg = _cfg(n_sae_seeds=3, top_k=25, n_null=150)
    out = concept_stability(tmp_path, atlas, cfg)

    part = out["concepts"][0]["stability"]["parts"][0]
    assert part["reciprocal_at"] == [True, False], (
        "expected replicate 1 to clear (shares the planted latent) and "
        "replicate 2 not to (pure noise) -- got a different pattern, so "
        "this fixture does not actually discriminate all-vs-majority")
    assert part["stable"] is False, "1-of-2 clearing must NOT count as stable"


# ---------------------------------------------------------------------------
# 5. search_dict_size is never called for a replicate
# ---------------------------------------------------------------------------

def test_train_sae_replicate_never_calls_search_dict_size(monkeypatch):
    """`train_sae_replicate`'s signature REQUIRES an explicit `dict_size`,
    with no code path back to `run_sae`'s own `dict_size_policy: search`
    branch -- monkeypatch `search_dict_size` in the module namespace to raise
    and confirm the replicate helper never reaches it, then confirm the mock
    was never invoked (not merely "no exception", since an unrelated no-op
    branch could also produce no exception)."""
    sentinel = MagicMock(side_effect=AssertionError("search_dict_size must never be called "
                                                     "for a replicate"))
    monkeypatch.setattr(train_mod, "search_dict_size", sentinel)

    rng = np.random.default_rng(3)
    activations = rng.normal(size=(64, 8)).astype(np.float32)
    cfg = SimpleNamespace(sae=SimpleNamespace(
        k=2, dict_size_mult=4, lr=1e-3, epochs=1, batch_size=32,
        resample_dead_every_epochs=0,
        aux_k=0, aux_coef=0.0, aux_dead_steps=0, min_train_steps=0,
        aux_dead_steps_frac=0.0, dict_size_policy="search",
        dict_size_ladder=[4, 8, 16]))
    device = torch.device("cpu")

    sae, history = train_mod.train_sae_replicate(activations, cfg, seed=1, dict_size=6, device=device)

    assert sentinel.call_count == 0
    assert sae.dict_size == 6


# ---------------------------------------------------------------------------
# 6. Seed derivation for the per-replicate reverse leg is deterministic
#    across process boundaries (CLAUDE.md sec 11.2/sec 11.52).
# ---------------------------------------------------------------------------

def test_replicate_seed_tag_is_bit_identical_across_processes():
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from tsfm_lens.sae.transfer import _seed
        print(_seed("ModelA/layer1", 0, "@r1", base=0))
    """)
    out1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    out2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out1.returncode == 0 and out2.returncode == 0
    assert out1.stdout.strip() == out2.stdout.strip()
    assert out1.stdout.strip() != ""
    # negative: different replicate indices must give different seeds -- the
    # failure `_seed`'s own "different seeds give different draws" style
    # test guards against (sec 11.53's postscript: a plant that changes
    # nothing is indistinguishable from a working guard).
    assert _seed("ModelA/layer1", 0, "@r1", base=0) != _seed("ModelA/layer1", 0, "@r2", base=0)


# ---------------------------------------------------------------------------
# n_sae_seeds < 2 degrades to "not measured", never a partial measurement
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regression: `ablation_run.py::ablation_targets` must not treat a replicate
# checkpoint (a SIBLING file `run_sae_replicates` now writes beside the
# primary one) as a target of its own. Found live by this session's own
# `test_concept_stage.py::test_script_and_stage_produce_identical_ablation_json`
# after wiring replicate training into the `concepts` stage -- `run_sae_
# ablation.py --all` crashed with `KeyError: store.load('patchy',
# 'blocks_5@r1', ...)` because its glob over `sae/*/*.pt` picked up
# `blocks_5@r1.pt` as if it named a real, never-extracted layer
# (`CLAUDE.md` sec 11.40's exact shape: a fix that widens what one stage
# emits widens what everything downstream that reads that directory sees).
# ---------------------------------------------------------------------------

def test_ablation_targets_excludes_replicate_checkpoints(tmp_path):
    from tsfm_lens.sae.ablation_run import ablation_targets
    sae_dir = tmp_path / "sae" / "ModelA"
    sae_dir.mkdir(parents=True)
    (sae_dir / "layer1.pt").write_bytes(b"")
    (sae_dir / "layer1@r1.pt").write_bytes(b"")
    (sae_dir / "layer1@r2.pt").write_bytes(b"")

    targets = ablation_targets(tmp_path)
    assert targets == [("ModelA", "layer1")], (
        f"replicate checkpoints leaked into the target list: {targets}")


def test_n_sae_seeds_below_two_is_not_measured(tmp_path):
    save_meta(tmp_path, _meta(10, [("a", 10)]))
    _make_store(tmp_path, n=10, n_windows=2)
    atlas = _atlas_one_part(0, "ModelA", "layer1", [0])
    cfg = _cfg(n_sae_seeds=1)
    out = concept_stability(tmp_path, atlas, cfg)
    assert out["measured"] is False
    assert out["concepts"] == [{"concept": 0, "stability": "not measured"}]
    assert out["ceiling"] is None
    (tmp_path / "sae" / "concept_stability.json").exists()
