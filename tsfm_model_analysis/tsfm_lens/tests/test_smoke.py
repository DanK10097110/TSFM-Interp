"""End-to-end smoke test: every stage on mock models and generated data.

Runnable directly (`python tests/test_smoke.py`) or via pytest. Passing
means extraction, alignment, all four analysis levels, clustering, patching,
and the report compose correctly; it says nothing about real checkpoints,
which is what `--check-alignment` exists for.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import config_from_dict
from tsfm_lens.pipeline import Context, run_pipeline


def build_config(out_dir: str) -> dict:
    """Small but complete configuration exercising every stage."""
    return {
        "run": {"name": "smoke", "out_dir": out_dir, "device": "cpu",
                "dtype": "float32", "seed": 0},
        "data": {"source": "smoke", "context_len": 128, "horizon": 32,
                 "smoke_series_per_family": 20},
        "alignment": {"window": 32, "sanity_check": True},
        "models": [
            {"name": "patchy", "adapter": "mock_patch", "batch_size": 64},
            {"name": "steppy", "adapter": "mock_step", "batch_size": 64},
        ],
        "l1": {"layer_stride": 1, "max_rows": 20000, "min_family_series": 10,
               "rsa": True, "rsa_max_series": 120},
        "l2": {"layer_stride": 1, "max_rows": 8000, "val_frac": 0.3},
        "l3": {"max_series": 48,
               "patching": {"layer_stride": 1, "max_series": 32,
                            "corruptions": ["deseasonalize", "noise"],
                            "per_window": True, "window_stride": 1}},
        "lens": {"layer_stride": 1, "max_series": 24, "tuned": True,
                 "tuned_max_series": 200, "lambdas": [1e-2, 1e-1, 1.0]},
        "attention": {"max_series": 40, "batch_series": 16,
                      "max_lag_tokens": 64, "ablation_max_series": 32,
                      "top_k": 3},
        "exemplars": {"per_family": 2, "max_families": 4},
        "clustering": {"use_umap": False, "max_series": 200},
        "stats": {"enabled": True, "n_boot": 150, "n_boot_heavy": 100,
                  "min_series": 8},
        "internals": {"enabled": True, "max_rows": 8000, "probe_pca_dim": 20},
        "confirm": {"enabled": True, "source": "smoke", "max_series": 120,
                    "require_seal": False},
        # verbose explicitly requested: `report.verbose` now defaults to False
        # (ROADMAP.md §10/§16), but this test exists to exercise every stage
        # and every report section, verbose L3 case studies included.
        "report": {"title": "Smoke comparison", "verbose": True},
    }


def test_end_to_end(tmp_path=None):
    out = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    cfg = config_from_dict(build_config(out))
    run_pipeline(cfg)
    run_dir = cfg.run_dir()

    expected = [
        "activations.zarr", "meta.parquet", "config_resolved.yaml",
        "l0/metrics.parquet", "l0/summary.json",
        "layer_screen/selection.json",
        "l1/cka.npz", "l1/meta.json",
        "l2/stitching.json",
        "l3/sensitivity.npz", "l3/meta.json", "l3/patching.npz", "l3/patching.json",
        "lens/curves.npz", "lens/lens.json",
        "attention/arrays.npz", "attention/meta.json",
        "exemplars/exemplars.npz", "exemplars/exemplars.json",
        "clustering/embedding.parquet", "clustering/clusters.json",
        "clustering/comparison.json",
        "internals/profile.json",
        "hypotheses.json", "confirm/behavioral.parquet", "confirm/confirmation.json",
        "report.html", "report/coverage.json", "report/findings.json",
    ]
    missing = [p for p in expected if not (run_dir / p).exists()]
    assert not missing, f"missing artifacts: {missing}"

    html = (run_dir / "report.html").read_text(encoding="utf-8")
    for token in ("Findings", "Representational geometry", "Activation clusters",
                  "Model internals", "Private benchmark confirmation",
                  "Forecast lens", "Attention structure", "Exemplar case studies",
                  "Layer screening", "per-window restoration", "p (Holm)", "plotly",
                  "How to read this report", "mask-fraction baseline",
                  "How to read these case studies", "patched at"):
        assert token in html, f"report missing '{token}'"

    # `report/findings.json` (ROADMAP.md sec 21 E6/J1): one structured `Finding`
    # record per rendered claim -- `plain`/`text`/`caveat`, the three-register
    # claim contract -- machine-readable without re-parsing the HTML `<li>`
    # list.
    findings_payload = json.loads((run_dir / "report" / "findings.json").read_text(encoding="utf-8"))
    findings_list = findings_payload["findings"]
    assert findings_list, "findings.json has no entries"
    # Isolated by the first `<section ` tag, not the first `</div>` -- J1's
    # per-finding collapsed caveat (`<details class="note">...<div
    # class="note-body">...</div></details>`) nests a `</div>` inside every
    # `<li>`, so a naive first-`</div>` split would truncate the block after
    # the very first finding's caveat instead of at the findings block's own
    # close (which precedes the first rendered `<section>`).
    findings_block = html.split('<div class="findings">', 1)[1].split("<section ", 1)[0]
    n_findings_html = findings_block.count("<li>")
    assert len(findings_list) == n_findings_html, (
        f"findings.json has {len(findings_list)} entries but the report "
        f"rendered {n_findings_html} <li> findings")
    allowed_evidence_classes = {"geometric", "translatable", "causal_within_model",
                                "descriptive", "illustrative", "behavioral"}
    for entry in findings_list:
        assert entry["evidence_class"] in allowed_evidence_classes, (
            f"finding {entry.get('claim_id')} has unexpected evidence_class "
            f"{entry.get('evidence_class')!r}")
        # J1: every finding must carry a genuine plain-English headline and a
        # generated caveat -- neither should ever be silently empty.
        assert entry.get("plain"), f"finding {entry.get('claim_id')} has empty plain"
        assert entry.get("caveat"), f"finding {entry.get('claim_id')} has empty caveat"
        assert entry["plain"] != entry["text"], (
            f"finding {entry.get('claim_id')}: plain should not be identical to text")
    # The plain headline and the caveat's collapsed <details> must both
    # actually be present in the rendered HTML, not just in findings.json.
    assert 'class="finding-plain"' in findings_block
    assert findings_block.count('class="finding-plain"') == len(findings_list)
    assert '<details class="note"><summary>Caveats</summary>' in findings_block
    for entry in findings_list:
        assert entry["text"], f"finding {entry.get('claim_id')} has empty text"
        assert isinstance(entry["registered"], bool)

    print(f"smoke test passed: {run_dir}")
    return run_dir


def test_per_window_and_lens_artifacts(run_dir=None):
    """Shape checks on the new per-window patching and lens/attention arrays."""
    import numpy as np

    from tsfm_lens.utils import load_json

    run_dir = Path(run_dir) if run_dir else Path(test_end_to_end())
    pmeta = load_json(run_dir / "l3" / "patching.json")
    parrs = np.load(run_dir / "l3" / "patching.npz")
    for model, info in pmeta.items():
        rw = parrs[f"restoration_windows_{model}"]
        assert rw.shape == (len(info["corruptions"]), len(info["rel_depth"]),
                            len(info["windows"])), rw.shape
        rh = parrs[f"restoration_by_horizon_{model}"]
        assert rh.shape[:2] == (len(info["corruptions"]), len(info["rel_depth"])), rh.shape
        assert rh.shape[2] > 0, rh.shape
        assert np.isfinite(rh).all(), "restoration_by_horizon has non-finite entries"
        for cname, vmeta in info.get("verbose", {}).items():
            prefix = f"verbose_{model}_{cname}_"
            grid = parrs[prefix + "grid"]
            n_series = grid.shape[-1]
            assert grid.shape[:2] == (len(info["rel_depth"]), len(info["windows"])), grid.shape
            assert n_series == len(vmeta["series_ids"]) == len(vmeta["families"])
            for key in ("context", "target", "clean", "corrupted", "patched"):
                assert parrs[prefix + key].shape[0] == n_series, (prefix, key)

    lmeta = load_json(run_dir / "lens" / "lens.json")
    larrs = np.load(run_dir / "lens" / "curves.npz")
    for model, m in lmeta.items():
        assert larrs[f"skip_mase_{model}"].shape == (len(m["layers"]),)
        assert np.isfinite(larrs[f"skip_mase_{model}"]).all()
        if f"skip_mase_by_horizon_{model}" in larrs:
            mh = larrs[f"skip_mase_by_horizon_{model}"]
            assert mh.shape[0] == len(m["layers"]) and mh.shape[1] > 0, mh.shape
            assert np.isfinite(mh).all()
            curve = m.get("crystallization_depth_by_horizon")
            assert curve is not None and len(curve) == mh.shape[1]

    ameta = load_json(run_dir / "attention" / "meta.json")
    aarrs = np.load(run_dir / "attention" / "arrays.npz")
    for model, m in ameta.items():
        assert "head_scores" in m["patterns"], m
        assert f"head_delta_{model}" in aarrs

    emeta = load_json(run_dir / "exemplars" / "exemplars.json")
    earrs = np.load(run_dir / "exemplars" / "exemplars.npz")
    n_ex = len(emeta["exemplars"])
    assert earrs["contexts"].shape[0] == n_ex
    for model in emeta["models"]:
        assert earrs[f"lens_mase_{model}"].shape[1] == n_ex
    print("per-window/lens/attention/exemplar artifact test passed")


def test_confirm_hypothesis_path():
    """The verdict machinery must fire correctly when dev strengths exist.

    Mocks rarely produce significant dev strengths, so the smoke run
    exercises only the empty-hypothesis branch; this test feeds synthetic
    dev claims and private metrics with one real and one spurious effect.
    """
    import numpy as np
    import pandas as pd

    from tsfm_lens.analysis.confirm import _test_registered_hypotheses
    from tsfm_lens.utils import save_json

    out = Path(tempfile.mkdtemp())
    cfg = config_from_dict(build_config(str(out)))
    (cfg.run_dir() / "l0").mkdir(parents=True)
    save_json(cfg.run_dir() / "l0" / "summary.json", {
        "strengths": {"patchy": ["trend"], "steppy": ["spiky"]},
        "mase_ratio": {"trend": 0.7, "spiky": 1.4},
    })
    # A hand-built registry (sec 15 A15), standing in for what the `register`
    # stage would have written from the same `l0/summary.json` above.
    registry = {"hypotheses": [
        {"id": "l0_family::trend::patchy", "stage": "l0", "family": "trend",
        "favored": "patchy", "replicable": True},
        {"id": "l0_family::spiky::steppy", "stage": "l0", "family": "spiky",
        "favored": "steppy", "replicable": True},
    ]}
    rng = np.random.default_rng(0)
    rows = []
    for fam, (mu_a, mu_b) in {"trend": (0.8, 1.2), "spiky": (1.0, 1.0)}.items():
        for i in range(40):
            sid = f"{fam}_{i}"
            rows.append({"series_id": sid, "family": fam, "model": "patchy",
                         "mase": mu_a + rng.normal(0, 0.1)})
            rows.append({"series_id": sid, "family": fam, "model": "steppy",
                         "mase": mu_b + rng.normal(0, 0.1)})
    res = _test_registered_hypotheses(cfg, pd.DataFrame(rows), registry, "patchy", "steppy")
    verdicts = {t["family"]: t["confirmed"] for t in res["tests"]}
    assert verdicts == {"trend": True, "spiky": False}, verdicts
    assert all("p_holm" in t for t in res["tests"])
    print("confirm hypothesis-path test passed")


def test_sae_stage_integration(run_dir=None):
    """The `sae` stage against mock adapters: trains, evaluates, and degrades
    gracefully on `ground_truth_alignment` (smoke data has no sealed corpus
    to load ground truth from) without breaking `forecast_preservation`,
    which needs no corpus -- only the model and stored activations."""
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.sae.train import run_sae
    from tsfm_lens.utils import load_json

    run_dir = Path(run_dir) if run_dir else Path(test_end_to_end())
    cfg = config_from_dict(build_config(str(run_dir.parent)))
    cfg.run.name = run_dir.name
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    layer = store.layers("patchy")[-1]
    cfg.sae.enabled = True
    cfg.sae.targets = [{"model": "patchy", "layer": layer}]
    cfg.sae.epochs = 3
    cfg.sae.dict_size_mult = 2
    cfg.sae.k = 4
    cfg.sae.forecast_preservation_max_series = 16
    cfg.sae.feature_ablation_enabled = True
    cfg.sae.feature_ablation_max_series = 8

    ctx = Context(cfg)
    run_sae(cfg, ctx.hub, ctx.store, ctx.data, ctx.device)

    meta = load_json(run_dir / "sae" / "meta.json")
    key = f"patchy/{layer}"
    assert key in meta, meta.keys()
    entry = meta[key]
    assert entry["dict_size"] == entry["d_in"] * 2
    assert 0.0 <= entry["dead_feature_rate"] <= 1.0

    fp = entry["forecast_preservation"]
    assert "error" not in fp, fp
    assert fp["n_series"] == 16
    assert fp["mase_clean"] > 0 and fp["mase_reconstructed"] > 0
    assert fp["granularity"] == "window"

    fp_token = entry["forecast_preservation_token"]
    assert "error" not in fp_token, fp_token
    assert fp_token["granularity"] == "token"
    assert fp_token["n_series"] == 16
    assert fp_token["mase_clean"] > 0 and fp_token["mase_reconstructed"] > 0

    gt = entry["ground_truth_alignment"]
    assert "error" in gt, "smoke data has no sealed corpus; ground truth must degrade, not crash"

    # Feature ablation reuses `gt`'s matches as its candidate set; since smoke
    # data has no sealed corpus (`gt` is an error dict), there are no matched
    # features to ablate -- it must degrade to a clean no-op, not crash.
    assert entry["feature_ablation"] is None, (
        "no ground-truth matches available on smoke data; feature_ablation must skip, not run")

    from tsfm_lens.report.report import run_report
    run_report(cfg)
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Sparse feature dictionary" in html, "report missing the SAE section"
    assert "reconstruction fidelity" in html
    assert "no ground-truth-matched features to illustrate" in html, (
        "smoke data has no ground truth, so the exemplar panel must say so, not crash or "
        "silently render an empty table")
    print("sae stage integration test passed")


def test_layer_screen_stage_and_sae_auto_targets(run_dir=None):
    """`layer_screen` runs by default in a full pipeline (ROADMAP.md §6.1.1)
    and `sae.targets: []` ("auto") must resolve from its selection.json
    rather than falling back to the old arbitrary final-layer default."""
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.sae.train import _default_targets
    from tsfm_lens.utils import load_json

    run_dir = Path(run_dir) if run_dir else Path(test_end_to_end())
    screen = load_json(run_dir / "layer_screen" / "selection.json")
    assert set(screen) == {"patchy", "steppy"}, screen.keys()
    for model, sel in screen.items():
        assert sel["method"] == "work_bend"
        assert 0 < len(sel["selected"]) <= len(sel["layers"])
        assert len(sel["score_per_layer"]) == len(sel["layers"])

    cfg = config_from_dict(build_config(str(run_dir.parent)))
    cfg.run.name = run_dir.name
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    resolved = _default_targets(cfg, store)
    expected = [{"model": m, "layer": l} for m, sel in screen.items() for l in sel["selected"]]
    assert resolved == expected, (resolved, expected)
    print("layer_screen stage + sae auto-target resolution test passed")


if __name__ == "__main__":
    run_dir = test_end_to_end()
    test_per_window_and_lens_artifacts(run_dir)
    test_confirm_hypothesis_path()
    test_sae_stage_integration(run_dir)
    test_layer_screen_stage_and_sae_auto_targets(run_dir)
