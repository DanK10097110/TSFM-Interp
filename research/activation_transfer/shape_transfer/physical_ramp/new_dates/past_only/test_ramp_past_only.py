import numpy as np
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.past_only.study import calendar_mask

def test_template_requires_strictly_prior_date_including_at_boundary():
    prior=['2026-01-10','2026-01-10','2026-04-10','2026-10-10']
    targets=['2026-01-01','2026-01-10','2026-01-11','2026-04-10','2026-08-01','2026-11-01']
    wanted=[[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]]
    np.testing.assert_equal(calendar_mask(prior,targets),np.asarray(wanted,dtype=bool))
