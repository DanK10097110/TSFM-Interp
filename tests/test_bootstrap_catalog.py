from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from tsfm_benchmark.build_pipeline import sources as sources_mod
from tsfm_benchmark.build_pipeline.schema import SourceRef


def test_bootstrap_catalog_samples_n_domains_and_skips_broken_ones(monkeypatch):
    catalog = ["weather", "broken_freq", "bitcoin", "covid_deaths", "also_broken", "us_births"]

    fake_datasets = type(sys)("datasets")
    fake_datasets.get_dataset_config_names = lambda dataset_name, trust_remote_code=True: catalog
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    def fake_load_single_config(dataset_name, config_name, split, field_name, streaming, trust_remote_code, limit, license):
        if "broken" in config_name:
            raise RuntimeError("KeyError: 'h' (upstream frequency alias bug)")
        return [(SourceRef(corpus=f"hf/x/{config_name}/train", item_id=str(i), sha256="a" * 64), [1.0, 2.0, 3.0]) for i in range(limit)]

    monkeypatch.setattr(sources_mod, "_load_single_config", fake_load_single_config)

    pool = sources_mod.bootstrap_catalog(n_domains=3, total_limit=30, seed=0)

    domains_used = {ref.corpus.split("/")[2] for ref, _ in pool}
    assert len(domains_used) == 3
    assert domains_used.issubset({"weather", "bitcoin", "covid_deaths", "us_births"})
    assert "broken_freq" not in domains_used
    assert "also_broken" not in domains_used


def test_bootstrap_catalog_works_against_an_arbitrary_dataset_name(monkeypatch):
    # bootstrap_catalog must not be Monash-specific: any dataset name works.
    catalog = ["h1", "h2", "m1", "m2"]
    fake_datasets = type(sys)("datasets")
    fake_datasets.get_dataset_config_names = lambda dataset_name, trust_remote_code=True: catalog
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    def fake_load_single_config(dataset_name, config_name, split, field_name, streaming, trust_remote_code, limit, license):
        assert dataset_name == "ETDataset/ett"
        return [(SourceRef(corpus=f"hf/{dataset_name}/{config_name}/train", item_id="0", sha256="a" * 64), [1.0, 2.0])]

    monkeypatch.setattr(sources_mod, "_load_single_config", fake_load_single_config)

    pool = sources_mod.bootstrap_catalog(dataset_name="ETDataset/ett", n_domains=2, total_limit=10, seed=0)
    assert len(pool) == 2


def test_bootstrap_catalog_defaults_to_every_discovered_domain(monkeypatch):
    catalog = ["a", "b", "c", "d"]
    fake_datasets = type(sys)("datasets")
    fake_datasets.get_dataset_config_names = lambda dataset_name, trust_remote_code=True: catalog
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    def fake_load_single_config(dataset_name, config_name, split, field_name, streaming, trust_remote_code, limit, license):
        return [(SourceRef(corpus=f"hf/x/{config_name}/train", item_id="0", sha256="a" * 64), [1.0, 2.0])]

    monkeypatch.setattr(sources_mod, "_load_single_config", fake_load_single_config)

    # n_domains left unspecified -> every discovered config should be used.
    pool = sources_mod.bootstrap_catalog(total_limit=40, seed=0)
    domains_used = {ref.corpus.split("/")[2] for ref, _ in pool}
    assert domains_used == set(catalog)


def test_bootstrap_catalog_is_reproducible_given_same_seed(monkeypatch):
    catalog = ["a", "b", "c", "d", "e"]
    fake_datasets = type(sys)("datasets")
    fake_datasets.get_dataset_config_names = lambda dataset_name, trust_remote_code=True: catalog
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    def fake_load_single_config(dataset_name, config_name, split, field_name, streaming, trust_remote_code, limit, license):
        return [(SourceRef(corpus=f"hf/x/{config_name}/train", item_id="0", sha256="a" * 64), [1.0, 2.0])]

    monkeypatch.setattr(sources_mod, "_load_single_config", fake_load_single_config)

    pool_a = sources_mod.bootstrap_catalog(n_domains=2, total_limit=10, seed=42)
    pool_b = sources_mod.bootstrap_catalog(n_domains=2, total_limit=10, seed=42)
    assert [ref.corpus for ref, _ in pool_a] == [ref.corpus for ref, _ in pool_b]


def test_bootstrap_catalog_raises_when_every_domain_fails(monkeypatch):
    catalog = ["broken_a", "broken_b"]
    fake_datasets = type(sys)("datasets")
    fake_datasets.get_dataset_config_names = lambda dataset_name, trust_remote_code=True: catalog
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    def fake_load_single_config(dataset_name, config_name, split, field_name, streaming, trust_remote_code, limit, license):
        raise RuntimeError("nope")

    monkeypatch.setattr(sources_mod, "_load_single_config", fake_load_single_config)

    with pytest.raises(RuntimeError):
        sources_mod.bootstrap_catalog(n_domains=2, total_limit=10, seed=0)


def test_load_sources_dispatches_to_bootstrap_when_no_subset_given(monkeypatch):
    calls = []

    def fake_bootstrap_catalog(**kwargs):
        calls.append(kwargs)
        return [(SourceRef(corpus="hf/x/y/train", item_id="0", sha256="a" * 64), [1.0, 2.0])]

    monkeypatch.setattr(sources_mod, "bootstrap_catalog", fake_bootstrap_catalog)

    result = sources_mod.load_sources({"kind": "huggingface", "dataset": "some/arbitrary-dataset"})
    assert len(calls) == 1
    assert calls[0]["dataset_name"] == "some/arbitrary-dataset"
    assert calls[0]["n_domains"] is None  # large/unbounded default when unspecified
    assert result


def test_load_sources_requires_dataset_name_for_huggingface_kind():
    with pytest.raises(ValueError):
        sources_mod.load_sources({"kind": "huggingface"})
