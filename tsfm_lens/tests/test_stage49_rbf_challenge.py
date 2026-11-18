"""Nonlinear known answers and confusable deterministic-input duplicates."""
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np

folder=Path(__file__).parents[1]/'experiments/feature_search'
sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location('stage49_rbf_challenge',folder/'stage49_rbf_challenge.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_rbf_nonlinear_answer_and_prediction_batch_invariance():
    x=np.linspace(-3,3,60)[:,None];y=np.sin(x[:,0]);folds=np.arange(60)%5
    fit,grid,oof=audit.train(x,y,folds)
    held=np.array([[-2.4],[.4],[1.8]])
    prediction=audit.predict(fit,held)
    assert np.mean((prediction-np.sin(held[:,0]))**2)<.005
    assert len(grid)==15
    np.testing.assert_allclose(fit['setup']['mean'],x.mean(0),atol=1e-15)
    # Evaluation rows must not change one another's scaling or kernel centering.
    np.testing.assert_allclose(prediction,audit.predict(fit,np.vstack([held,[[1e6]]]))[:3],atol=1e-12)


def fixture(seed,n=64):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,2));z=rng.normal(size=(n,2))
    return x,dict(loss=np.sin(x[:,0])+3*z[:,0],response=z,alternate_response=z.copy())


def test_joint_known_signal_and_evaluation_labels_do_not_select_models():
    dx,d=fixture(1);ex,e=fixture(2)
    result,saved=audit.assess(dx,ex,d,e,shuffled=2)
    effect=result['contrasts']['joint_vs_input']
    assert effect['relative_gain']>.5
    assert effect['passes_frozen_robustness_gate']
    assert effect['pointwise_bootstrap_ci95'][0]>0
    altered=copy.deepcopy(e);altered['loss']*=100
    other,arrays=audit.assess(dx,ex,d,altered,shuffled=2)
    for name in result['models']:
        assert result['models'][name]['selected']==other['models'][name]['selected']
        np.testing.assert_array_equal(saved[name+'_eval_prediction'],arrays[name+'_eval_prediction'])
    # A null second derivative coordinate should not be called the positive signal.
    assert result['contrasts']['on_only_vs_input']['relative_gain']<.05


def test_confusable_input_duplicate_has_no_incremental_information():
    dx,d=fixture(4);ex,e=fixture(5)
    for x,cohort in [(dx,d),(ex,e)]:
        cohort['loss']=np.sin(x[:,0])+np.cos(x[:,1])
        cohort['response']=x.copy();cohort['alternate_response']=x.copy()
    result,saved=audit.assess(dx,ex,d,e,shuffled=2)
    # Duplicating every standardized descriptor scales all distances and their
    # median equally. The joint kernel is exactly the same input-only function.
    np.testing.assert_allclose(saved['joint_rbf_eval_prediction'],saved['input_rbf_eval_prediction'],atol=1e-12)
    assert not result['contrasts']['joint_vs_input']['passes_frozen_robustness_gate']
