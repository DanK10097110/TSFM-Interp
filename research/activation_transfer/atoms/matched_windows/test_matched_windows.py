import numpy as np

from research.activation_transfer.atoms.matched_windows.prepare import adjacent
from research.activation_transfer.atoms.matched_windows import patch
from research.activation_transfer.atoms.matched_windows.score import paired


def test_adjacent_uses_the_original_center_and_an_actual_predecessor():
    raw=np.arange(500,dtype=float)**2
    clean,new,end,step,mu,sd=adjacent(raw)
    assert (end,step)==(492,-1)
    np.testing.assert_allclose(clean,(raw[12:492]-mu)/sd)
    np.testing.assert_allclose(new,(raw[11:491]-mu)/sd)
    assert abs(new.mean())>.001
    raw=np.arange(488,dtype=float)
    _,new,end,step,mu,sd=adjacent(raw)
    assert (end,step)==(480,1)
    np.testing.assert_allclose(new,(raw[1:481]-mu)/sd)


def test_natural_write_matches_each_feature_dose_and_keeps_the_series_direction(monkeypatch):
    # Non-axis-aligned signs and unequal decoder norms catch transposed row/feature indexing.
    delta=np.array([[-3.,4.],[0.,-2.],[0.,0.]])
    directions,norm=patch.unit_windows(delta)
    np.testing.assert_allclose(directions,[[-.6,.8],[0.,-1.],[0.,0.]])
    np.testing.assert_allclose(norm,[5.,2.,0.])
    z=np.array([[2.,3.],[4.,0.],[1.,5.]])
    dose=patch.requested_doses(z,np.array([[2.,0.],[0.,3.]]))
    np.testing.assert_allclose(dose,[[4.,9.],[8.,0.],[2.,15.]])
    def forecast(adapter,layer,x,tokens,d,selector,batch):
        return d+7.
    monkeypatch.setattr(patch,'change_forecasts',forecast)
    effect=patch.patch_vectors(None,None,None,np.ones((3,5)),np.ones((3,1,2)),np.full((3,2),7.),
                               z>0,dose,directions,batch=2)
    np.testing.assert_allclose(effect[0],[[ -2.4,3.2],[-5.4,7.2]],atol=1e-6)
    np.testing.assert_allclose(effect[1],[[0.,-8.],[0.,0.]])
    np.testing.assert_array_equal(effect[2],0.)


def test_comparison_keeps_both_references_measurable_and_uses_paired_draws():
    active=np.ones((3,2),dtype=bool)
    ref=np.array([[[1.,2.],[2.,3.]],[[1.,-1.],[4.,5.]],[[3.,-1.],[1.,-5.]]])
    w=ref.copy();w[:,1]=0.
    references={'atom':ref,'window':w}
    candidates={'atom':-ref,'window':w*2}
    counts=np.array([[1,1,1],[2,0,1],[0,2,1]])
    out=paired(active,references,candidates,counts)
    assert out['common_indices']==[0]
    assert out['eligible_indices']=={'atom':[0,1],'window':[0]}
    assert out['window_minus_atom']['cosine']['value']==2.
    np.testing.assert_allclose(out['window_minus_atom']['cosine']['ci95'],[2.,2.])
    assert out['window_minus_atom']['skill']['value']==3.
