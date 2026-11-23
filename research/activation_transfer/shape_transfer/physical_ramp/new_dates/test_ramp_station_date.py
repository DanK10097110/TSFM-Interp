import numpy as np
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.score import energy

def test_independent_station_date_resampling_reestimates_site_mean():
    rng=np.random.default_rng(93);data=rng.normal(size=(4,3,8))
    station=np.array([[1,1,1,1],[0,3,1,0],[0,1,1,2]]);date=np.array([[1,1,1],[2,0,1],[0,0,3]])
    for center in (True,False):
        expected=[]
        for a,b in zip(station,date):
            x=np.repeat(np.repeat(data,a,axis=0),b,axis=1)
            if center:x=x-x.mean(0)
            expected.append(np.square(x).sum())
        np.testing.assert_allclose(energy(data,station,date,center),expected,atol=1e-11)
