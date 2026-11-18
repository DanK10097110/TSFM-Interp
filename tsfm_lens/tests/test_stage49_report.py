"""Independent saved-forecast replay and confusable primary-flip decoy."""
import json
from pathlib import Path
import pytest
from experiments.feature_search.stage49_report import replay

ROOT=Path(__file__).resolve().parents[2]
PREPARED=ROOT/'tsfm_lens/runs/feature_search_stage49_crossover3400000_prepared'
RUN=ROOT/'tsfm_lens/runs/feature_search_stage49_crossover3400000'
pytestmark=pytest.mark.skipif(not PREPARED.is_dir() or not RUN.is_dir(),reason='optional independent replay requires the controlled saved-forecast run; portable figure rebuilding does not')


def test_independent_replay_matches_all_registered_effects():
    """Independent losses, Gram solve, exact signs and global correction agree."""
    result=replay(PREPARED,RUN)
    assert result['passed']
    assert result['claims_replayed']==22
    assert result['per_seed_values_checked']==6144
    assert result['maximum_per_seed_absolute_error']<1e-10
    assert result['maximum_full_gram_energy_closure']<1e-10


def test_false_primary_flip_cannot_hide_behind_sensitivity_success(tmp_path):
    """An outcome summary must agree with the frozen primary sign/gap gates."""
    for path in RUN.iterdir():
        if path.name!='results.json':(tmp_path/path.name).symlink_to(path.resolve())
    result=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
    assert not result['primary_ranking_flip'] and result['sensitivity_ranking_flip']
    result['primary_ranking_flip']=True
    (tmp_path/'results.json').write_text(json.dumps(result),encoding='utf-8')
    with pytest.raises(AssertionError):replay(PREPARED,tmp_path)
