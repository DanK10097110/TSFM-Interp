"""A reused-cohort rank screen is neither confirmation nor an impossibility proof."""
import copy
import pytest
from experiments.feature_search import stage49_report as report


def fixture():
    protocol=dict(ranks=[1,2,4],cut=15,seed_ids=list(range(3600000,3600016)),
        gates=dict(recovery=.8,mean_loss=.06,random_advantage=.05,mean_only_advantage=.05))
    discovery=dict(pilot=dict(native_loss=[.5]*16,descriptive={'15':dict(anchor_loss=[.04]*16)}))
    result=dict(status='complete',execution_valid=True,total_wall_seconds=1.,execution_phases={'all':dict(returncode=0,conservative_peak_sum=100)},
        full_edit_mean_improvement=.46,mean_only_mean_loss=.3,smallest_passing_rank=None,
        ranks={str(rank):dict(valid_all16=True,mean_loss=loss,recovery=(.5-loss)/.46,random_advantage=.1,mean_only_advantage=.3-loss,passed=False)
               for rank,loss in [(1,.3),(2,.2),(4,.14)]})
    return result,protocol,discovery


def test_rank_gains_do_not_rescue_failed_fixed_screen_or_imply_broad_absence():
    summary=report.summarize_compactness(*fixture())
    assert summary['status']=='closed_negative' and set(summary['ranks'])=={'1','2','4'}
    text=report.compactness_html(summary)
    assert 'not</strong> disprove all low-dimensional or nonlinear mechanisms' in text
    assert 'no p-values or confirmatory claims' in text
    assert 'full-edit improvement' in text and 'not matched-rank' in text


def test_frozen_gates_and_complete_family_cannot_be_relabelled():
    result,protocol,discovery=fixture()
    bad=copy.deepcopy(result);bad['ranks']['4']['passed']=True
    with pytest.raises(ValueError,match='gate mismatch'):report.summarize_compactness(bad,protocol,discovery)
    bad=copy.deepcopy(protocol);bad['gates']['mean_loss']=.2
    with pytest.raises(ValueError,match='fixed compactness'):report.summarize_compactness(result,bad,discovery)
    bad=copy.deepcopy(result);bad['execution_valid']=False
    with pytest.raises(ValueError,match='not valid'):report.summarize_compactness(bad,protocol,discovery)


def test_recovery_and_loss_are_both_required_not_interchangeable():
    result,protocol,discovery=fixture()
    result['ranks']['4'].update(mean_loss=.1,recovery=.4/.46,mean_only_advantage=.2)
    assert report.summarize_compactness(result,protocol,discovery)['status']=='closed_negative'  # Recovery alone passes.
    discovery['pilot']['native_loss']=[.1]*16;discovery['pilot']['descriptive']['15']['anchor_loss']=[.01]*16
    result['full_edit_mean_improvement']=.09
    result['ranks']['4'].update(mean_loss=.05,mean_only_advantage=.25)
    for row in result['ranks'].values():row['recovery']=(.1-row['mean_loss'])/.09
    assert report.summarize_compactness(result,protocol,discovery)['status']=='closed_negative'  # Loss alone passes.
