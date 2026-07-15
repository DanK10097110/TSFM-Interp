"""Exercise benchmark_validation against real BenchmarkBuilder output.

Every record here comes from an actual sealed corpus produced by the
generation pipeline (mocking only the network-touching source load), not from
hand-built SeqRecords -- the point is to check the validation package reads
and plots what the pipeline really writes, including task/tier/domain
provenance that only exists once builder.py records it.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("pycatch22")
pytest.importorskip("plotly")

from tsfm_benchmark.build_pipeline.builder import BenchmarkBuilder, TaskSpec
from tsfm_benchmark.build_pipeline.schema import SourceRef
import tsfm_benchmark.benchmark_validation as bv


def _build_sealed_corpus(tmp_path, monkeypatch):
    fake_pool = [(SourceRef(corpus=f"hf/x/domain_{i % 3}/train", item_id=str(i), sha256="a" * 64), np.sin(np.linspace(0, 10, 300) + i)) for i in range(12)]
    monkeypatch.setattr("tsfm_benchmark.build_pipeline.builder.load_sources", lambda config: fake_pool)

    specs = [
        TaskSpec(name="synth_task", generator="random_parametric", count=14, tier="synthetic", generator_params={"length": 128}),
        TaskSpec(
            name="real_task",
            generator="mixture",
            count=6,
            tier="realism_stress",
            generator_params={"mode": "weighted_sum"},
            source_sample_size=(2, 4),
            source_config={"kind": "huggingface", "dataset": "fake/dataset"},
        ),
    ]
    builder = BenchmarkBuilder()
    result = builder.build(specs, seed=0, epoch=0)
    pub_dir = tmp_path / "public_dev"
    from tsfm_benchmark.build_pipeline import seal as seal_mod

    seal_mod.seal_corpus(result.public_dev, str(pub_dir), epoch=0, visibility="public")
    return str(pub_dir), result


def test_from_sealed_recovers_task_tier_and_domain_provenance(tmp_path, monkeypatch):
    pub_dir, result = _build_sealed_corpus(tmp_path, monkeypatch)
    records = bv.from_sealed(pub_dir)

    assert len(records) == len(result.public_dev) == 20

    by_task = {r.task for r in records}
    assert by_task == {"synth_task", "real_task"}

    synth = [r for r in records if r.task == "synth_task"]
    real = [r for r in records if r.task == "real_task"]
    assert len(synth) == 14 and len(real) == 6
    assert all(r.tier == "synthetic" for r in synth)
    assert all(r.tier == "realism_stress" for r in real)
    assert all(r.domains == [] for r in synth)
    assert all(set(r.domains) <= {"domain_0", "domain_1", "domain_2"} and r.domains for r in real)


def test_full_validation_pipeline_runs_against_real_pipeline_output(tmp_path, monkeypatch):
    pub_dir, _ = _build_sealed_corpus(tmp_path, monkeypatch)
    records = bv.from_sealed(pub_dir)

    match = bv.match_all(records, method="xcorr")
    fm = bv.extract_features(records)
    coords, embed_method = bv.embed_3d(fm.features_scaled, method="pca")
    diversity = bv.diversity_metrics(fm)
    report = bv.build_report(match, fm, diversity, embed_method, records=records)

    assert report["composition"]["by_tier"]["synthetic"]["count"] == 14
    assert report["composition"]["by_tier"]["realism_stress"]["count"] == 6
    assert set(report["composition"]["by_domain"]) <= {"domain_0", "domain_1", "domain_2"}
    assert sum(report["composition"]["by_domain"].values()) > 0

    out = tmp_path / "outputs"
    out.mkdir()
    paths = [
        bv.plot_embedding(coords, fm, embed_method, str(out / "embed.html")),
        bv.plot_group_composition(records, str(out / "composition.html")),
        bv.plot_example_sequences(records, str(out / "examples.html")),
        bv.plot_feature_variance(diversity, str(out / "variance.html")),
        bv.plot_redundancy_histogram(match, str(out / "redundancy.html")),
    ]
    domain_path = bv.plot_domain_composition(records, str(out / "domains.html"))
    assert domain_path is not None
    paths.append(domain_path)

    for p in paths:
        assert Path(p).exists()
        assert Path(p).stat().st_size > 500


def test_plot_domain_composition_is_none_for_purely_synthetic_corpus():
    records = [bv.SeqRecord(seq_id=str(i), values=np.random.randn(50), group="random_parametric", task="t", tier="synthetic", domains=[]) for i in range(5)]
    assert bv.plot_domain_composition(records, "/tmp/should-not-be-written.html") is None
