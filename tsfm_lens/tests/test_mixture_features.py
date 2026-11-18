"""Known-answer factorial target/nuisance tests with a louder confusable harmonic."""
import importlib.util
from pathlib import Path
import numpy as np


def _module():
    """Load the standalone experiment without requiring model libraries."""
    path=Path(__file__).parents[1]/'experiments/feature_search/mixture_features.py'
    spec=importlib.util.spec_from_file_location('mixture_features',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_joint_harmonic_fit_separates_louder_nuisance():
    """Loud period16 cannot be mistaken for the period32 target."""
    m=_module();t=np.arange(128,160);x=(.3+ .6*np.sin(2*np.pi*t/32+.7)+2*np.sin(2*np.pi*t/16-1.2))[None,:]
    fit=m.harmonic_fit(x)
    assert abs(fit['phi'][0]-.7)<1e-10 and abs(fit['psi'][0]+1.2)<1e-10
    assert abs(fit['a'][0]-.6)<1e-10 and abs(fit['b'][0]-2)<1e-10


def test_factor_pairs_keep_nuisance_and_bundle_partition():
    """Matching by target must not accidentally swap the other phase cell."""
    m=_module();context,future,meta=m.make_corpus()
    for target,other in [('phi','psi'),('psi','phi')]:
        receiver,donor,groups=m.pairs(meta,'test',target)
        assert len(receiver)==96 and len(set(groups))==48
        assert all(meta[i]['split']=='test' for i in receiver)
        assert all(meta[i][other]==meta[j][other] for i,j in zip(receiver,donor))
        assert all(abs(meta[j][target]-meta[i][target]-np.pi/2)<1e-12 for i,j in zip(receiver,donor))
    patch=context.reshape(192,4,4,32)
    assert np.ptp(patch.mean(-1),axis=1).max()<1e-6
    assert np.ptp(patch.std(-1),axis=1).max()<1e-6


def test_selective_metric_catches_target_plus_nuisance_decoy():
    """A donor that restores target while changing nuisance is explicitly caught."""
    m=_module();_,future,meta=m.make_corpus();receiver,donor,_=m.pairs(meta,'test','phi')
    rows=[meta[i] for i in receiver]
    selective=m.metrics(future[donor],future[receiver],future[donor],rows,'phi')
    nuisance_donor=np.array([i+1 if meta[i]['cell_psi']==0 else i-1 for i in donor])
    decoy=m.metrics(future[nuisance_donor],future[receiver],future[donor],rows,'phi')
    assert selective['target_error'].max()<1e-10 and selective['other_phase_change'].max()<1e-10
    assert decoy['target_error'].max()<1e-10 and decoy['other_phase_change'].min()>1.5
