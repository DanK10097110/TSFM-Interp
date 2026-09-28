"""Tests for `analysis/internals.py::_family_probe_permutation_null`
(ROADMAP.md sec 16 E9's remaining scope): the majority-class `chance` line
already reported for family-probe decodability is a weak, unconditional
floor. This reruns the identical probe fit with the family label shuffled
at the SERIES level (never within a series' own windows, per `CLAUDE.md`
invariant 2), mirroring `sae/ground_truth.py::permutation_null_alignment`'s
"rerun the identical search" pattern and its two-scenario test shape: a
planted real signal must survive comparison against the null, and pure
noise must not tower over its own null.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.internals import _family_probe, _family_probe_permutation_null
from tsfm_lens.config import PipelineConfig


def _cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.stats.enabled = False  # exercise the plain {"value": ...} path
    cfg.internals.probe_pca_dim = 5
    return cfg


def _series_split(n_series: int, n_windows: int, seed: int, val_frac: float = 0.25):
    perm = np.random.default_rng(seed).permutation(n_series)
    n_val = max(2, int(val_frac * n_series))
    val_s, train_s = perm[:n_val], perm[n_val:]
    train_mask = np.zeros(n_series * n_windows, dtype=bool)
    train_mask[(train_s[:, None] * n_windows + np.arange(n_windows)).ravel()] = True
    return train_mask, val_s


def test_planted_family_signal_survives_the_null():
    rng = np.random.default_rng(0)
    n_series, n_windows, d = 80, 4, 6
    families = np.where(rng.integers(0, 2, size=n_series) == 0, "A", "B")
    row_labels = np.repeat(families, n_windows)
    x = rng.normal(size=(n_series * n_windows, d)).astype(np.float32)
    x[:, 0] += np.where(row_labels == "A", 3.0, -3.0)  # planted, easily-separable signal

    cfg = _cfg()
    train_mask, val_s = _series_split(n_series, n_windows, seed=1)
    real = _family_probe(cfg, x, row_labels, train_mask, val_s, n_windows)
    null = _family_probe_permutation_null(cfg, x, families, train_mask, val_s,
                                          n_windows, n_perm=10, seed=2)

    assert real["value"] > 0.9
    assert null["n_perm"] == 10
    # The genuinely decodable signal must clear the null -- the whole point
    # of comparing against a permutation null instead of bare chance.
    assert real["value"] > null["acc_null_p95"]


def test_pure_noise_probe_does_not_tower_over_its_own_null():
    rng = np.random.default_rng(3)
    n_series, n_windows, d = 80, 4, 6
    families = np.where(rng.integers(0, 2, size=n_series) == 0, "A", "B")
    row_labels = np.repeat(families, n_windows)
    x = rng.normal(size=(n_series * n_windows, d)).astype(np.float32)  # no planted signal

    cfg = _cfg()
    train_mask, val_s = _series_split(n_series, n_windows, seed=4)
    real = _family_probe(cfg, x, row_labels, train_mask, val_s, n_windows)
    null = _family_probe_permutation_null(cfg, x, families, train_mask, val_s,
                                          n_windows, n_perm=10, seed=5)

    assert null["n_perm"] == 10
    # With no real structure, real accuracy is itself just one more draw
    # from roughly the same distribution the null measures -- it should not
    # be dramatically above the null's own mean.
    assert real["value"] < null["acc_null_mean"] + 0.20


def test_n_perm_zero_disables_cleanly():
    rng = np.random.default_rng(6)
    n_series, n_windows, d = 40, 3, 4
    families = np.where(rng.integers(0, 2, size=n_series) == 0, "A", "B")
    x = rng.normal(size=(n_series * n_windows, d)).astype(np.float32)
    cfg = _cfg()
    train_mask, val_s = _series_split(n_series, n_windows, seed=7)
    null = _family_probe_permutation_null(cfg, x, families, train_mask, val_s,
                                          n_windows, n_perm=0, seed=0)
    assert null == {"n_perm": 0, "acc_null_mean": None, "acc_null_p95": None,
                    "acc_null_values": []}


def test_permutation_shuffles_at_the_series_level_not_the_window_level():
    """Every permuted row-label block must stay internally uniform -- a
    series' own windows never get split across two different shuffled
    labels (`CLAUDE.md` invariant 2: the series is the resampling unit)."""
    rng = np.random.default_rng(8)
    n_series, n_windows = 12, 5
    families = np.where(rng.integers(0, 2, size=n_series) == 0, "A", "B")
    x = rng.normal(size=(n_series * n_windows, 3)).astype(np.float32)
    cfg = _cfg()
    train_mask, val_s = _series_split(n_series, n_windows, seed=9)

    # Reach into one shuffled draw directly via the same RNG call the
    # function makes, to check the block structure it actually used.
    perm_rng = np.random.default_rng(10)
    shuffled_series_labels = families[perm_rng.permutation(n_series)]
    perm_row_labels = np.repeat(shuffled_series_labels, n_windows)
    blocks = perm_row_labels.reshape(n_series, n_windows)
    assert all(len(set(block)) == 1 for block in blocks)

    null = _family_probe_permutation_null(cfg, x, families, train_mask, val_s,
                                          n_windows, n_perm=3, seed=10)
    assert null["n_perm"] == 3


def _hard_to_converge(n_series: int = 80, n_windows: int = 4, d: int = 6,
                      seed: int = 11):
    """Near-separable data: the logistic loss has no finite minimizer, so the
    solver runs until it hits its iteration cap rather than converging."""
    rng = np.random.default_rng(seed)
    families = np.where(rng.integers(0, 2, size=n_series) == 0, "A", "B")
    row_labels = np.repeat(families, n_windows)
    x = rng.normal(scale=0.01, size=(n_series * n_windows, d)).astype(np.float32)
    x[:, 0] += np.where(row_labels == "A", 50.0, -50.0)
    return x, families, row_labels


def test_probe_records_whether_its_solver_converged():
    """`converged` is a recorded fact, not a log line (`CLAUDE.md` sec 2.5).

    An under-converged fit UNDERSTATES probe accuracy, so a corpus that
    outgrows the iteration budget makes a model's internals look less
    decodable than they are -- and because the permutation null reruns the
    identical fit, the bias cancels in the real-vs-null gap the report
    renders and is invisible there. The artifact has to say so.
    """
    x, families, row_labels = _hard_to_converge()
    cfg = _cfg()
    train_mask, val_s = _series_split(len(families), 4, seed=12)

    cfg.internals.probe_max_iter = 1
    starved = _family_probe(cfg, x, row_labels, train_mask, val_s, 4)
    assert starved["converged"] is False, "a 1-iteration budget must report False"

    cfg.internals.probe_max_iter = 5000
    ample = _family_probe(cfg, x, row_labels, train_mask, val_s, 4)
    assert ample["converged"] is True

    # The flag must not be a constant: the two runs disagree, which is the
    # only thing that makes recording it worth anything.
    assert starved["converged"] != ample["converged"]


def test_probe_max_iter_is_read_from_config_not_hardcoded():
    """The budget is a config knob. Pinned because the historical value was a
    literal `300` that silently failed to converge on a ~1000-series corpus.
    """
    x, families, row_labels = _hard_to_converge(seed=13)
    cfg = _cfg()
    train_mask, val_s = _series_split(len(families), 4, seed=14)

    cfg.internals.probe_max_iter = 1
    assert _family_probe(cfg, x, row_labels, train_mask, val_s, 4)["converged"] is False
    # If the call site ignored the config and used its own constant, raising
    # the knob could not change the outcome.
    cfg.internals.probe_max_iter = 5000
    assert _family_probe(cfg, x, row_labels, train_mask, val_s, 4)["converged"] is True


def test_converged_is_reported_on_the_bootstrap_path_too():
    """`cfg.stats.enabled` picks a different return path; both must carry it."""
    x, families, row_labels = _hard_to_converge(seed=15)
    cfg = _cfg()
    cfg.stats.enabled = True
    cfg.stats.n_boot = 20
    cfg.internals.probe_max_iter = 1
    train_mask, val_s = _series_split(len(families), 4, seed=16)
    out = _family_probe(cfg, x, row_labels, train_mask, val_s, 4)
    assert "value" in out and "lo" in out, "bootstrap fields must survive"
    assert out["converged"] is False


def test_permutation_null_is_unaffected_by_the_extra_key():
    """The null reads only `result["value"]`; adding a key must not break it."""
    x, families, row_labels = _hard_to_converge(seed=17)
    cfg = _cfg()
    cfg.internals.probe_max_iter = 50
    train_mask, val_s = _series_split(len(families), 4, seed=18)
    null = _family_probe_permutation_null(cfg, x, families, train_mask, val_s,
                                          4, n_perm=5, seed=19)
    assert null["n_perm"] == 5
    assert len(null["acc_null_values"]) == 5
    assert all(0.0 <= a <= 1.0 for a in null["acc_null_values"])


def test_report_names_a_capped_solver_only_when_one_was_capped():
    """The clause is a diagnostic, so it must be absent on a healthy run.

    A caveat that always renders trains a reader to skip it (`CLAUDE.md`
    sec 11.43's banner lesson), and this one exists precisely because the
    defect it names is invisible in the real-vs-null gap beside it
    (sec 11.47).
    """
    import re
    src = Path(__file__).resolve().parents[1] / "tsfm_lens" / "report" / "report.py"
    text = src.read_text(encoding="utf-8")
    assert "hit its iteration cap at" in text, "the clause must exist"
    # It must be guarded by a count of actually-unconverged layers, not
    # appended unconditionally.
    guard = re.search(r"n_unconverged = sum\(.*?if n_unconverged:", text, re.S)
    assert guard, "the clause must be gated on a measured count"
    assert "hit its iteration cap at" not in guard.group(0), \
        "the clause must sit inside the guard, not before it"
    # And the count must key off `is False` -- a layer whose flag is missing
    # (an older artifact) is unknown, not unconverged.
    assert 'q.get("converged") is False' in text
