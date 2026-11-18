"""Execution-only cap amendment never changes the scientific schedule."""
import pytest
from experiments.feature_search import ffn_selectivity_causal_replication_300s as runner


def test_amendment_requires300_and_sets_internal_worker_deadline(tmp_path,monkeypatch):
    runner.study.write(tmp_path/'protocol.json',dict(cells=1088,model='bolt'))
    amendment=dict(base_protocol_sha256=runner.study.sha(tmp_path/'protocol.json'),launcher_sha256=runner.study.sha(runner.__file__),
        seconds_per_worker=300,cells=1088,science_and_controls_unchanged=True)
    runner.study.write(tmp_path/'execution_amendment.json',amendment)
    def toy_worker(out):
        assert out==tmp_path and runner.study.SECONDS==300
    monkeypatch.setattr(runner.study,'worker',toy_worker)
    monkeypatch.setattr(runner.study,'SECONDS',900)
    runner.worker(tmp_path)
    amendment['seconds_per_worker']=900
    runner.study.write(tmp_path/'execution_amendment.json',amendment)
    with pytest.raises(ValueError,match='300-second'):
        runner.validate(tmp_path)


def test_amended_dispatcher_holds_without_grant(tmp_path):
    with pytest.raises(RuntimeError,match='HOLD'):
        runner.run(tmp_path)
