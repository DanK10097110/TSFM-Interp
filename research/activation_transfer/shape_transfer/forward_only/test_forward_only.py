"""Unit contracts for disjoint source probes and eight generic, calibrated target writes."""
import numpy as np

from research.activation_transfer.shape_transfer.forward_only.probe import source_steps
from research.activation_transfer.shape_transfer.forward_only.transfer import fit_library, shape_weight


def test_forward_probe_excludes_removal_even_for_small_activations():
    decoder_norm=np.array([.5,2.])
    codes=np.array([[0.,.01],[200.,100.]])
    np.testing.assert_allclose(source_steps(codes,decoder_norm,10.),[[0.,.005],[.02,.02]])


def test_generic_impulses_fit_all_history_outputs_in_weighted_space():
    rng=np.random.default_rng(4649)
    jac=rng.normal(size=(4,8,11))
    basis=np.linalg.qr(rng.normal(size=(19,11)))[0].T
    weight=3.5
    dc=np.ones((8,8))/8
    w=dc+weight*(np.eye(8)-dc)
    a=np.vstack([w@j for j in jac]); b=np.tile(w,(4,1))
    ridge=.001*np.linalg.svd(a,compute_uv=False)[0]**2
    joined=np.vstack([a,np.sqrt(ridge)*np.eye(11)])
    goals=np.vstack([b,np.zeros((11,8))])
    reference=basis.T@np.linalg.lstsq(joined,goals,rcond=None)[0]
    library,stats=fit_library(jac,basis,weight)
    np.testing.assert_allclose(library,reference,atol=1e-10)
    np.testing.assert_allclose(stats['unweighted_mean_output_operator'],np.mean(jac@(basis@library),axis=0))


def test_generic_random_probes_weight_dc_against_per_history_shape():
    j=np.array([[[1.,2.],[3.,4.]], [[4.,0.],[2.,2.]]])
    weighted=shape_weight(j)
    level=np.repeat(j.mean(axis=1,keepdims=True),2,axis=1)
    np.testing.assert_allclose(np.linalg.norm(level), weighted*np.linalg.norm(j-level))
