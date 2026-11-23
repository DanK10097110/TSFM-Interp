"""Contracts for isolated cue changes, physical decoder clamping, and signed readouts."""
import numpy as np

from research.activation_transfer.shape_transfer.level_pair.inputs import isolated_histories
from research.activation_transfer.shape_transfer.level_pair.run import term_corrections
from research.activation_transfer.shape_transfer.level_pair.score import compare,half_difference


def test_isolated_edits_hold_other_values_and_use_the_same_feasible_dose_for_both_signs():
    rng=np.random.default_rng(4845);x=rng.normal(size=(3,480));x[1]=rng.normal(0,1e-3,480);x[1,-1]=8.
    result=isolated_histories(x);moved=result['edits'].astype(float)
    def measures(z):
        last=z[...,-32:];rise=(last@(np.arange(32)-15.5))*31/np.sum((np.arange(32)-15.5)**2)
        return np.stack([last.mean(-1)-z.mean(-1),z[...,-1]-z.mean(-1),rise],axis=-1)
    baseline=measures(x);diff=measures(moved)-baseline[:,None,None]
    goal=np.zeros_like(diff)
    sd=x.std(-1);requested=np.array([.25,.5,.5])
    for k in range(3):goal[:,k,:,k]=np.array([-1.,1.])[None,:]*(result['scales'][:,k]*sd*requested[k])[:,None]
    np.testing.assert_allclose(diff,goal,atol=2e-6)
    np.testing.assert_allclose(moved.std(-1),np.broadcast_to(sd[:,None,None],moved.shape[:3]),atol=2e-6)
    np.testing.assert_allclose(moved[:,3,1]-x,(.25*sd[:,None])+np.zeros_like(x),atol=1e-6)
    assert 0 < result['scales'][1,1] < .01


def test_named_lock_restores_decoder_term_and_matches_control_norms():
    d=np.array([[1.,2.,0.],[0.,0.,4.],[2.,0.,1.],[0.,3.,1.]])
    old=np.array([2.,0.]);edited=np.array([.3,1.]);dec0=np.array([[3.,1.,2.],[2.,2.,4.]]);dec1=dec0+.7
    out=term_corrections(old,edited,dec0,dec1,d,0,[1,2,3]);delta=(old-edited)[:,None]*d[0]
    np.testing.assert_allclose(out['named'],delta);np.testing.assert_allclose(out['all']+dec1,dec0)
    for j in range(3):
        np.testing.assert_allclose(np.linalg.norm(out['control'+str(j)],axis=1),np.linalg.norm(delta,axis=1))


def test_signed_level_loss_and_centered_shape_projection():
    curves=np.array([[[1.,3.],[3.,9.]],[[0.,2.],[4.,6.]]]);reference=half_difference(curves)
    np.testing.assert_allclose(reference,[[1,3],[2,2]])
    lock=reference-np.array([[.5,1.5],[1,1]])
    scored=compare(reference,lock,np.array([[1,1],[2,0],[0,2]]))
    np.testing.assert_allclose(scored['removed_level']['value'],1.)
    np.testing.assert_allclose(scored['fraction_mean_lift_removed']['value'],.5)
    np.testing.assert_allclose(scored['fraction_shape_projected_away']['value'],.5)


def test_subset_order_average_divides_interactions_among_its_participants():
    from research.activation_transfer.shape_transfer.level_pair.factorial import order_average
    lift={}
    for bits in range(8):
        named,other,residual=[int(bool(bits&k)) for k in (1,2,4)]
        removed=2*named+other-3*residual+4*named*residual+3*named*other*residual
        lift[bits]=np.array([10.-removed])
    values=order_average(lift)
    np.testing.assert_allclose([values[k][0] for k in (1,2,4)],[5,2,0],atol=1e-10)


def test_threshold_excludes_named_feature_and_counts_strictly_higher_competitors():
    import torch
    from research.activation_transfer.shape_transfer.level_pair.gating import cutoffs

    class TestDictionary:
        k=2
        def pre_activations(self,x):
            return x.clone()
        def sparsify(self,x):
            vals,idx=x.relu().topk(self.k)
            return torch.zeros_like(x).scatter(1,idx,vals)

    h=np.array([[[4.,2.5,1.]],[[1.,2.,3.]]],dtype='f4')
    report=cutoffs(TestDictionary(),h,0)
    np.testing.assert_equal(report['rank'],[1,3])
    np.testing.assert_equal(report['cutoff'],[1,2])
    np.testing.assert_equal(report['margin'],[3,-1])
    np.testing.assert_equal(report['z'],[4,0])


def test_common_grid_holds_all_registered_coordinates_and_history_variance():
    from research.activation_transfer.shape_transfer.level_pair.gating import coordinate_targets
    r=np.random.default_rng(8144);x=3+2*r.normal(size=(3,480))
    waves,eligible,_=coordinate_targets(x)
    t=np.arange(32)-15.5
    for mode in ('joint','native'):
        for condition,value in [('low',0.),('high',1.25)]:
            key=mode+'_'+condition;z=waves[key].astype(float)
            assert eligible[key].all()
            np.testing.assert_allclose(z.mean(1),x.mean(1),atol=1e-6)
            np.testing.assert_allclose(z.std(1),x.std(1),atol=1e-6)
            norm=(z-x.mean(1)[:,None])/x.std(1)[:,None]
            np.testing.assert_allclose(norm[:,-32:].mean(1),value,atol=1e-6)
            if mode=='native':
                reference=(x-x.mean(1)[:,None])/x.std(1)[:,None]
                target_end=reference[:,-1];target_slope=reference[:,-32:]@t
            else:
                target_end=.5;target_slope=0.
            np.testing.assert_allclose(norm[:,-1],target_end,atol=1e-6)
            np.testing.assert_allclose(norm[:,-32:]@t,target_slope,atol=5e-6)


def test_hybrid_preserves_donor_pattern_and_separately_allocates_recent_residual_energy():
    from research.activation_transfer.shape_transfer.level_pair.splice import hybrid_grid
    rng=np.random.default_rng(9304)
    x=rng.normal(size=(4,480));x[:2,-32:]*=np.array([2.,.25])[:,None]
    day=np.array([0,1]);traffic=np.array([3,2]);waves,shares,_=hybrid_grid(x,day,traffic)
    t=np.arange(32)-15.5
    tail_rows=np.column_stack([np.ones(32),np.eye(32)[:,-1],t])
    def pieces(series):
        a=series[:,:-32].copy();a-=a.mean(1)[:,None]
        b=series[:,-32:].copy()
        b-=np.linalg.lstsq(tail_rows,b.T,rcond=None)[0].T@tail_rows.T
        return a,b
    aa,bb=pieces(x)
    def sq(z):return np.einsum('ij,ij->i',z,z)
    truth=sq(bb)/(sq(aa)+sq(bb))
    np.testing.assert_allclose(shares,truth)
    for bits,y in waves.items():
        a,b=pieces(y.astype(float));pa=day if bits&1 else traffic;pb=day if bits&2 else traffic;pe=day if bits&4 else traffic
        for actual,donor in [(a,aa[pa]),(b,bb[pb])]:
            np.testing.assert_allclose(actual/np.sqrt(sq(actual))[:,None],donor/np.sqrt(sq(donor))[:,None],atol=1e-7)
        np.testing.assert_allclose(sq(b)/(sq(a)+sq(b)),truth[pe],atol=1e-7)
