import numpy as np
from research.activation_transfer.shape_transfer.physical_ramp.study import edits,wrong_rows,match_norm
from research.activation_transfer.shape_transfer.physical_ramp.score import energy


def test_ramp_has_zero_mean_and_only_last_eight_change():
    clean=np.zeros((2,480),'f4');ramp,step=edits(clean)
    np.testing.assert_equal(ramp[:,:-8],0.);np.testing.assert_equal(step[:,:-8],0.)
    np.testing.assert_allclose(ramp[:,-8:],np.broadcast_to(np.linspace(-2.,2.,8),(2,8)),atol=1e-7)
    np.testing.assert_equal(step[:,-8:],2.);np.testing.assert_allclose(ramp[:,-8:].mean(1),0.,atol=1e-7)


def test_other_station_preserves_season_and_physical_dose():
    st=np.repeat(range(5),4);dt=np.tile(range(4),5);wrong=wrong_rows(st,dt)
    np.testing.assert_equal(dt[wrong],dt);assert (st[wrong]!=st).all()
    np.testing.assert_equal(np.sort(wrong),np.arange(20))
    vectors=np.array([[3.,4.],[0.,2.]])
    np.testing.assert_allclose(match_norm(vectors,[2.,3.]),[[1.2,1.6],[0.,3.]])


def test_date_specific_template_is_reestimated_per_station_bootstrap():
    a=np.array([[[1.,3.],[0.,2.]],[[5.,3.],[1.,8.]],[[3.,7.],[4.,5.]]])
    w=np.array([[1,1,1],[2,0,1],[0,3,0]])
    expected=[]
    for count in w:
        v=np.repeat(a,count,axis=0);expected.append(np.sum((v-v.mean(0))**2))
    np.testing.assert_allclose(energy(a,w,True),expected,atol=1e-12)
