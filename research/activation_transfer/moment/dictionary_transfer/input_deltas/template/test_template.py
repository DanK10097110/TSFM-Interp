import numpy as np

from research.activation_transfer.moment.dictionary_transfer.input_deltas.template.run import generic
from research.activation_transfer.moment.dictionary_transfer.input_deltas.template.score import measurements


def test_historical_template_ignores_new_orientation_and_preserves_each_scalar_dose():
    old=np.array([[4.,1.],[0.,-1.]])
    new=np.array([[0.,3.],[-4.,0.],[0.,0.]])
    fixed,matched=generic(old,new)
    np.testing.assert_array_equal(fixed,[[2,0]]*3)
    np.testing.assert_array_equal(matched,[[3,0],[4,0],[0,0]])


def test_paired_positive_minus_negative_shape_at_both_horizons():
    reference=np.tile([-1.,1.],(6,8))*np.arange(1,7)[:,None]
    wave=dict(reference=reference,mapped=reference+181.,historical_fixed=.5*reference,historical_matched=-reference)
    out=measurements(wave,3,2,["a","b"])
    for cell in ("early_shape","late_shape"):
        result=out[cell]
        np.testing.assert_allclose([result["arms"][k]["skill"]["value"] for k in ("mapped","historical_fixed","historical_matched")],[1,.75,-3])
        v=result["paired"]["mapped_minus_historical_matched"]
        np.testing.assert_allclose([v["value"],*v["ci95"]],[4,4,4])
