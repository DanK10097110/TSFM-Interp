import numpy as np

from research.activation_transfer.moment.dictionary_transfer.weather.score import matched_effect,metrics
from research.activation_transfer.moment.dictionary_transfer.weather.study import last_observed


def test_small_measurable_response_retains_signed_geometry():
    sums=np.array([[2e-20,2e-20,-2e-20,6.],[1e-20,0.,0.,6.],[0.,1.,0.,3.]])
    output=metrics(sums)
    np.testing.assert_allclose(output['skill'][:2],[-3.,0.])
    assert np.isnan(output['skill'][2])
    np.testing.assert_equal(output['cosine'][0],-1.)
    assert np.isnan(output['cosine'][1])


def test_physical_shift_uses_identical_batch_zero_write():
    records={'noop':np.array([[.15,.10]]),'donor':np.array([[.25,-.15]]),
             'noop_head':np.array([[.09,.11]]),'donor_head':np.array([[.20,-.09]])}
    np.testing.assert_allclose(matched_effect(records,'donor'),[[.10,-.25]])
    np.testing.assert_allclose(matched_effect(records,'donor',True),[[.11,-.20]])


def test_model_reported_observed_slot_changes_with_horizon():
    class Layout:
        def patch_layout(self,length,horizon):
            stop=64-horizon//8
            return slice(stop-length//8,stop),slice(stop,64)
        def token_time_spans(self):
            edge=np.arange(0,481,8)
            return np.stack((edge[:-1],edge[1:]),1)
    assert [last_observed(Layout(),480,h)[0] for h in (8,16,24)]==[62,61,60]


def test_example_median_uses_only_donor():
    from research.activation_transfer.moment.dictionary_transfer.weather.example import select
    station=np.repeat(np.arange(9),2)
    codes=np.zeros((len(station),2))
    codes[:17,0]=1.;codes[::2,1]=1.
    rows,cols=np.nonzero(codes)
    magnitude=(rows+1)*(1+5*cols)
    donor=np.stack([-magnitude,magnitude],axis=1)
    event,median=select(codes,station,rows,cols,donor)
    assert (int(rows[event]),int(cols[event]),median)==(8,0,9.)
