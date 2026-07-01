from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.example_runs.run_full import load_specs
from tsfm_benchmark.build_pipeline.builder import BenchmarkBuilder, TaskSpec


def test_load_specs_keeps_real_derived_tasks_when_sources_are_injected(monkeypatch):
    cfg = {
        "tasks": [
            {"name": "mixture_task", "generator": "mixture", "count": 3, "tier": "realism_stress", "generator_params": {"mode": "multiplicative"}},
        ]
    }

    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh)
        path = fh.name

    monkeypatch.setattr("tsfm_benchmark.example_runs.run_full.load_sources", lambda kind, limit, subset="tourism_monthly": [("fake", [1.0, 2.0])])

    try:
        specs = load_specs(path, max_count=None, source_kind="monash", source_limit=2)
    finally:
        Path(path).unlink(missing_ok=True)

    assert len(specs) == 1
    assert specs[0].name == "mixture_task"


def test_builder_uses_task_source_config(monkeypatch):
    def fake_generator(**kwargs):
        assert kwargs["sources"] == [("fake", [1.0, 2.0])]
        sample = type("Sample", (), {})()
        sample.values = [0.0]
        sample.ground_truth = None
        sample.provenance = type("Prov", (), {"generator_params": {}, "library_versions": {}, "source_refs": []})()
        sample.sample_id = ""
        sample.content_hash = lambda: "x"
        return sample

    monkeypatch.setattr("tsfm_benchmark.build_pipeline.builder.GENERATORS.get", lambda name: fake_generator)
    monkeypatch.setattr("tsfm_benchmark.build_pipeline.builder.load_sources", lambda config: [("fake", [1.0, 2.0])])

    spec = TaskSpec(name="hf_task", generator="mixture", count=1, source_config={"kind": "huggingface", "dataset": "demo"})
    builder = BenchmarkBuilder()
    builder._make_one(spec, seed=0, epoch=0)
