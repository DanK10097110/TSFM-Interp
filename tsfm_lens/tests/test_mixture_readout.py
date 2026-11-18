"""Known-answer spectral coefficient ordering, nuisance decoys and heldout isolation."""
import importlib.util
import sys
from pathlib import Path
import numpy as np


def _module():
    """Load the standalone CPU readout diagnostic."""
    root=Path(__file__).parents[1]/'experiments/feature_search';sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('mixture_readout',root/'mixture_readout.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_analytic_reconstruction_and_input_oracle_known_answer():
    """A louder second harmonic and offset cannot swap phase coefficient order."""
    m=_module();rng=np.random.default_rng(17);metadata=[];contexts=[];future=[]
    for i in range(20):
        a,b=.6,1.9;phi,psi=rng.uniform(-np.pi,np.pi,2);offset=rng.uniform(-2,2);scale=np.sqrt((a*a+b*b)/2)
        metadata.append(dict(a=a,b=b,phi=phi,psi=psi,scale=scale));fn=lambda t:offset+a*np.sin(2*np.pi*t/32+phi)+b*np.sin(2*np.pi*t/16+psi)
        contexts.append(fn(np.arange(128)));future.append(fn(np.arange(131,163)))
    context=np.array(contexts);truth=np.array(future);reconstruction=m.reconstruct(m.semantic_targets(metadata),context,start=131)
    assert np.max(abs(reconstruction-truth))<1e-12
    assert np.max(abs(m.input_oracle(context,start=131)-truth))<1e-12


def test_supervised_readout_ignores_test_labels_and_variance_decoy():
    """A noisy thousand-scale coordinate and changed test labels cannot drive fitting."""
    m=_module();rng=np.random.default_rng(6);targets=rng.normal(size=(400,4));noise=rng.normal(size=(400,6));noise[:,0]*=1000;features=np.column_stack([noise,targets]);train=np.arange(400)<200;validation=(np.arange(400)>=200)&(np.arange(400)<300)
    fit=m.fit_candidates(features,targets,train,validation)
    altered=targets.copy();altered[300:]*=-100
    changed=m.fit_candidates(features,altered,train,validation)
    assert np.array_equal(fit['prediction'],changed['prediction'])
    assert fit['alpha']==changed['alpha']
    assert np.mean((fit['prediction'][300:]-targets[300:])**2)<1e-6


def test_seed_summary_equal_bundle_weighting():
    """Replicated cells are a decoy for independent generating-bundle counts."""
    m=_module();summary=m.seed_summary(np.array([0]*100+[1,2]),np.array([0]*100+[1,2]))
    assert summary['n_seeds']==3
    assert summary['mean']==1
