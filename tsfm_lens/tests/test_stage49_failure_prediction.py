"""Known-answer, confusable input decoy and held-cohort leakage checks."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).parents[1]/'experiments/feature_search/stage49_failure_prediction.py'
spec = importlib.util.spec_from_file_location('stage49_failure_prediction', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_true_secant_and_cross_frequency_decoy_with_varying_scale():
    conditions=[['base',None,0,0.],['plus','psi',1,.01],['minus','psi',-1,.01]]
    metadata=[]; prediction=[]; coupled=[]; t=np.arange(128,160)
    for i,(a,b,psi) in enumerate([(1.,2.,.8),(3.,.4,-1.2)]):
        for name,_,sign,delta in conditions:
            q=psi+sign*delta; scale=np.sqrt((a*a+b*b)/2)
            row=dict(a=a,b=b,scale=scale,psi=q);metadata.append(row)
            clean=a*np.sin(2*np.pi*t/32+.4)+b*np.sin(2*np.pi*t/16+q)
            prediction.append(clean)
            coupled.append(clean+.7*b*np.sin(q)*np.sin(2*np.pi*t/32))
    response,off=audit.phase_response(prediction,metadata,conditions,.01)
    assert np.max(np.abs(response)) < 1e-22
    contaminated,_=audit.phase_response(coupled,metadata,conditions,.01)
    expected=np.log1p((.7*np.cos(np.array([.8,-1.2]))*np.sin(.01)/.01)**2)
    np.testing.assert_allclose(contaminated[:,0],expected,atol=1e-11)
    assert np.max(contaminated[:,1]) < 1e-22


def synthetic(seed,offset,n=96):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,3));z=rng.normal(size=n)
    response=np.column_stack([z,rng.normal(size=n)])
    models={}
    for model in audit.MODELS:
        # This unobserved native response adds predictive information.
        y=3+2*x[:,0]+z
        models[model]=dict(loss=y,response=response,alternate_response=response)
    return dict(ids=list(range(offset,offset+n)),descriptors=x,models=models)


def test_incremental_known_answer_input_decoy_and_eval_labels_cannot_choose_fit():
    discovery=synthetic(12,0);evaluation=synthetic(13,1000)
    report,saved=audit.assess(discovery,evaluation,shuffled=3)
    for model in audit.MODELS:
        effect=report[model]['augmentations']['off_only']
        assert effect['gain_fraction_of_baseline']>.95
        assert effect['exploratory_promotion']
        assert effect['sign_p_holm6']>effect['sign_p']
        assert effect['mean']>max(effect['shuffled_gain_distribution'])
    altered=copy.deepcopy(evaluation)
    for model in audit.MODELS: altered['models'][model]['loss'] *= -100
    other,other_saved=audit.assess(discovery,altered,shuffled=3)
    for key in saved:
        if key.endswith('prediction'): np.testing.assert_array_equal(saved[key],other_saved[key])
    for model in audit.MODELS:
        assert report[model]['augmentations']['off_only']['alpha']==other[model]['augmentations']['off_only']['alpha']
    # Confusable decoy is already fully predictable from the input baseline.
    decoy_d,decoy_e=copy.deepcopy(discovery),copy.deepcopy(evaluation)
    for cohort in [decoy_d,decoy_e]:
        for model in audit.MODELS:
            cohort['models'][model]['loss']=cohort['descriptors'][:,0]*2+3+.1*np.random.default_rng(cohort['ids'][0]+15).normal(size=len(cohort['ids']))
            cohort['models'][model]['response'][:,0]=cohort['descriptors'][:,0]
    decoy,_=audit.assess(decoy_d,decoy_e,shuffled=3)
    assert all(not decoy[m]['augmentations']['off_only']['exploratory_promotion'] for m in audit.MODELS)
    fit=audit.fit_ridge(discovery['descriptors'],discovery['models']['bolt']['loss'],1.)
    np.testing.assert_allclose(fit['mean'],discovery['descriptors'].mean(0))


def test_generator_seed_reuse_and_pseudoreplicated_outcomes_refused():
    with pytest.raises(ValueError,match='generating seeds overlap'):
        audit.assess(synthetic(1,0),synthetic(2,20),shuffled=1)
    with pytest.raises(ValueError,match='seed-level'):
        audit.fit_ridge(np.ones((5,2)),np.ones((5,5)),1.)
