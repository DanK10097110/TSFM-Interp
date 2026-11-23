"""Independent production replay of per-history SAE writes and their headline score.

No derivative solver, patch helper, aggregation routine or model-runner from the
full-space study is imported. Reuse only checkpoint constructors, the trained
SAE loader, fixed manifests and held-out inputs.
"""
from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
RAW=Path('/tmp/tsfm_atom_replication_20261120')
JACS=Path('/tmp/tsfm_full_space_sae_20261120')
REPLICATION=HERE.parents[1]/'replication/results'
MODELS_USED=('bolt','timesfm')
ARMS=('adaptive','unweighted','mean_donor','wrong_leads')


def weight_from_calibration(jac):
    level=np.mean(jac,axis=1,keepdims=True)
    return float(np.linalg.norm(np.broadcast_to(level,jac.shape))/np.linalg.norm(jac-level))


def svd_write(jac,response,shape_weight):
    # Construct the lead-space weighting matrix explicitly, independently of boost_shape.
    h=len(response); mean=np.ones((h,h))/h
    metric=mean+shape_weight*(np.eye(h)-mean)
    left,s,right=np.linalg.svd(metric@jac,full_matrices=False)
    factors=s/(s*s+.001*s[0]**2)
    projected=left.T@(metric@response)
    return right.T@(factors*projected if projected.ndim==1 else factors[:,None]*projected)


def cap(v,radius,fraction):
    return v*min(1.,fraction*radius/max(np.linalg.norm(v),1e-16))


def check_selection(labels,active,clipped,seed):
    rng=np.random.default_rng(seed);chosen=[]
    for domain in ('daily','machines'):
        pairs=np.argwhere(active&(labels==domain)[:,None]);pairs=pairs[rng.permutation(len(pairs))]
        group=[];seen=set(); represented=set()
        # Cover both clipping states when observed, then diversify source atom and item.
        for state in (False,True):
            block=[(int(i),int(j)) for i,j in pairs if bool(clipped[i,j])==state and int(i) not in seen]
            if block:
                group.append(block[0]);seen.add(block[0][0]);represented.add(block[0][1])
        for unique_atom in (True,False):
            for ri,ai in pairs:
                i,j=int(ri),int(ai)
                if len(group)>=6: break
                if i in seen or (unique_atom and j in represented): continue
                seen.add(i);represented.add(j);group.append((i,j))
        if len(group)!=6: raise ValueError('Insufficient distinct active cases for audit')
        chosen.extend(group)
    return chosen


@contextmanager
def hook(adapter,name,delta,capture=None):
    """Add the displacement to live model output without replacing a cached state."""
    block=dict(adapter.module.named_modules())[MODELS[name][-1]]
    # The final context interval is [448,480). Bolt appends a register after its 30 historical patches.
    slots=[28,29] if name=='bolt' else [14]
    spans=adapter.token_time_spans()
    np.testing.assert_array_equal(spans[slots],[[448,464],[464,480]] if name=='bolt' else [[448,480]])
    length=31 if name=='bolt' else 15
    calls=[]
    def replace(module,inputs,output):
        if not isinstance(output,tuple) or output[0].shape[1]!=length: raise ValueError('Unexpected production path')
        hidden=output[0]
        if capture is not None: capture.append(hidden[:,slots].detach().cpu().numpy().copy())
        # Preserve both original Bolt states and every untouched registration/position token.
        new=hidden.detach().clone();new[:,slots,:]=hidden.detach()[:,slots,:]+delta[:,None,:]
        calls.append(1);return (new,*output[1:])
    handle=block.register_forward_hook(replace)
    try: yield
    finally: handle.remove()
    if len(calls)!=1: raise ValueError('Expected the selected block exactly once')


def ordinary(adapter,name,x,v,leads):
    if v is None: return adapter.predict(x,leads,[.5])['quantiles'][0,:,0].astype(float)
    with hook(adapter,name,torch.as_tensor(v[None],dtype=torch.float32)):
        return adapter.predict(x,leads,[.5])['quantiles'][0,:,0].astype(float)


def fresh_jacobian(adapter,name,x,width,captured):
    """Independent single-row differentiable model call and complete autograd Jacobian."""
    def derivative_point(v):
        with hook(adapter,name,v[None],captured):
            return adapter.predict_differentiable(x,8)[0]
    return torch.autograd.functional.jacobian(derivative_point,torch.zeros(width),create_graph=False).detach().numpy().astype(float)


def direct_scores(ref,pred,cases,bootstrap=False):
    """Compute per-feature SSE or cosine directly on firing rows, then average features."""
    n,atoms,h=ref.shape; all_skill=[];all_cos=[];num=[];den=[];n_active=[]
    for j in range(atoms):
        use=cases[:,j];p=pred[use,j];r=ref[use,j]
        energy=np.sum(r*r)
        if not len(r) or energy <= 1e-10*len(r)*h: continue
        all_skill.append(1.-np.sum((r-p)**2)/energy)
        all_cos.append(np.sum(r*p)/max(np.linalg.norm(r)*np.linalg.norm(p),1e-20))
        a=np.zeros(n);b=np.zeros(n);a[use]=np.sum((p-r)**2,axis=1);b[use]=np.sum(r*r,axis=1)
        num.append(a);den.append(b);n_active.append(use.astype(float))
    result=dict(skill=float(np.mean(all_skill)),cosine=float(np.mean(all_cos)),n_scored=len(all_skill))
    if bootstrap:
        draws=np.random.default_rng(2026112021).multinomial(n,np.full(n,1/n),size=4000)
        sse=draws@np.stack(num).T; truth=draws@np.stack(den).T; count=draws@np.stack(n_active).T
        kept=(count>0)&(truth>1e-10*count*h)
        s=np.divide(sse,truth,out=np.zeros_like(truth),where=kept)
        avg=np.sum(np.where(kept,1-s,0),axis=1)/np.maximum(1,kept.sum(axis=1))
        result['ci95']=np.quantile(avg,[.025,.975]).tolist()
    return result


def archive_metrics(results,source,d,a,fn):
    report=json.loads((results/'results.json').read_text())[source]['domains']
    ev=a['split']!='cal'
    np.testing.assert_array_equal(a['split'][ev],d['split']);np.testing.assert_array_equal(a['ids'][ev],d['ids']);np.testing.assert_array_equal(fn['ids'],a['ids'])
    np.testing.assert_array_equal(a['atoms'],fn['atoms']);np.testing.assert_array_equal(a['active'],fn['active'])
    np.testing.assert_array_equal(a['source_codes'][ev]>0,d['active']);np.testing.assert_array_equal(d['atoms'],a['atoms'])
    truth=a['native'][ev].astype(float); active=d['active'];est={k:d[k] for k in ['fixed',*ARMS,'source_linear']}
    est.update(clean=a['dense'][ev],feature_fit=fn['derivative_write'][ev],template=fn['template'][ev]); max_point=0.;max_ci=0.; output={}
    for key,v in {**est,'source':truth}.items():
        if np.any(v[~active]!=0): raise ValueError('Unexpected nonzero forecast change on a nonfiring feature: '+key)
    for domain in ('daily','machines'):
        use=d['split']==domain;output[domain]={}
        for minimum in (1,8):
            cols=active[use].sum(0)>=minimum
            if not cols.any():continue
            out={};m=active[use][:,cols]
            for metric in ('full','shape'):
                q=truth[use][:,cols];q=q-q.mean(-1,keepdims=True) if metric=='shape' else q
                for arm,v in est.items():
                    p=v[use][:,cols].astype(float);p=p-p.mean(-1,keepdims=True) if metric=='shape' else p
                    now=direct_scores(q,p,m,bootstrap=arm=='adaptive')
                    old=report[domain]['subsets'][str(minimum)][metric]['methods'][arm]
                    max_point=max(max_point,*[abs(now[k]-old[k]['value']) for k in ('skill','cosine')]);
                    if arm=='adaptive':
                        max_ci=max(max_ci,float(np.max(np.abs(np.array(now['ci95'])-old['skill']['ci95']))));out[metric]=now
            output[domain][str(minimum)]=out
    if max(max_point,max_ci)>1e-8: raise ValueError(('Direct aggregate mismatch',max_point,max_ci))
    return dict(max_macro_error=max_point,max_adaptive_interval_error=max_ci,metrics=output)


def cached(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    start=time.monotonic(); out=HERE/'results';raw=cached(RAW/'data.npz');train=raw['split']=='train'; keep=~train
    ref={m:cached(REPLICATION/(m+'_seed0.npz')) for m in MODELS_USED}; current={m:cached(out/(m+'.npz')) for m in MODELS_USED}
    settings={};jacs={};models={};saes={};counts={}; radius={}; raw_states={};selected={};ids=raw['ids'][keep]; raw_ev=np.isin(raw['split'],['daily','machines']);ev=raw['split'][keep]!='cal'; ev_ids=raw['ids'][raw_ev]
    np.testing.assert_array_equal(ids[ev],ev_ids)
    for k,m in enumerate(MODELS_USED):
        jacs[m]=cached(JACS/(m+'_jac.npz'));np.testing.assert_array_equal(ids,jacs[m]['ids']);np.testing.assert_array_equal(raw['split'][keep],jacs[m]['split'])
        material=cached(RAW/(m+'.npz'));np.testing.assert_array_equal(raw['ids'],material['ids']);raw_states[m]=material['tokens']
        radius[m]=float(np.median(np.linalg.norm(raw_states[m][train].mean(axis=1),axis=1)))
        np.testing.assert_allclose(radius[m],jacs[m]['radius'],rtol=1e-7)
        models[m]=adapter_for(m);saes[m]=load_sae_checkpoint(str(dictionary_path(m))); counts[m]=ref[m]['source_codes'][ev]
        selected[m]=check_selection(current[m]['split'],counts[m]>0,current[m]['adaptive_clipped'],11202026+k)
        fn=cached(REPLICATION/(m+'_functional.npz'));settings[m]=archive_metrics(out,m,current[m],ref[m],fn)
    cases=sorted(set(i for pairs in selected.values() for i,j in pairs)); fresh={};cal_j={m:jacs[m]['jac'][~ev].astype(float) for m in MODELS_USED}
    raw_idx=np.flatnonzero(raw_ev);loc=np.flatnonzero(ev)
    for i in cases:
        x=raw['x'][raw_idx[i]:raw_idx[i]+1]
        for m in MODELS_USED:
            cap_state=[];width=jacs[m]['jac'].shape[-1];jac=fresh_jacobian(models[m],m,x,width,cap_state)
            live=cap_state[0][0];arch=jacs[m]['jac'][loc[i]].astype(float)
            err=np.linalg.norm(jac-arch)/max(np.linalg.norm(jac),1e-15)
            if err>.001:raise ValueError(('Fresh single-case gradient mismatch',m,i,err))
            np.testing.assert_allclose(live,raw_states[m][raw_idx[i]],atol=4e-5,rtol=2e-4)
            with torch.inference_mode():z=saes[m].encode(torch.from_numpy(live.mean(0)[None])).numpy()[0,ref[m]['atoms']]
            np.testing.assert_allclose(z,counts[m][i],atol=3e-4,rtol=2e-4)
            base={h:ordinary(models[m],m,x,None,h) for h in (8,16)}
            fresh[m,i]=(jac,z,err,base)
    rows={m:[] for m in MODELS_USED}; followup_files={}
    for source in MODELS_USED:
        target=next(m for m in MODELS_USED if m!=source);r=ref[source];z=current[source]
        side={}
        for key in ('leads','dose'):
            path=out/(source+'_'+key+'.npz')
            if path.exists():
                side[key]=cached(path);np.testing.assert_array_equal(z['ids'],side[key]['ids']);np.testing.assert_array_equal(z['active'],side[key]['active'])
        followup_files[source]=sorted(side)
        weight=weight_from_calibration(cal_j[source]);np.testing.assert_allclose(weight,z['shape_weight'],rtol=1e-12)
        selected_atoms=r['atoms']; decoder=saes[source].W_dec.detach().numpy()[selected_atoms].astype(float)
        for i,j in selected[source]:
            x=raw['x'][raw_idx[i]:raw_idx[i]+1];js,zz,ds,base_s=fresh[source,i];jt,_,dt,base_t=fresh[target,i]
            amplitude=zz[j];w=decoder[j];neg=-float(amplitude)*w;pred=js@w
            writes={};unclipped={};order=[3,6,1,7,0,2,5,4]
            for k,design,desired,alpha in [('adaptive',jt,pred,weight),('unweighted',jt,pred,1.),('wrong_leads',jt[order],pred,weight),('mean_donor',jt,cal_j[source].mean(0)@w,weight)]:
                v=-float(amplitude)*svd_write(design,desired,alpha);unclipped[k]=v;writes[k]=cap(v,radius[target],.2)
                if bool(np.linalg.norm(v)>.2*radius[target])!=bool(z[k+'_clipped'][i,j]):raise ValueError('Clip flag changed')
            errors={}; measures={}
            def measure(name,bone,v,h,expected):
                baseline=base_s if bone==source else base_t
                y=ordinary(models[bone],bone,x,v,h)-baseline[h];errors[name]=float(abs(y-expected).max());return y
            sr=measure('native',source,neg,8,r['native'][ev][i,j]);
            for method,delta in writes.items():
                y=measure(method,target,delta,8,z[method][i,j]);
                if method=='adaptive': adapted=y
            # Directional finite differences on actual API, along the requested SAE removal and reconstructed receiver write.
            for m,jac,v in [(source,js,neg),(target,jt,writes['adaptive'])]:
                step=.001*radius[m]/np.linalg.norm(v); u=v*step
                f=(ordinary(models[m],m,x,u,8)-ordinary(models[m],m,x,-u,8))/2
                measures[m]=dict(predicted=(jac@u).tolist(),measured=f.tolist())
            if 'leads' in side:
                lead=side['leads']; odd=np.arange(0,8,2); part_weight=weight_from_calibration(cal_j[source][:,odd]); partial=-float(amplitude)*svd_write(jt[odd],pred[odd],part_weight)
                measure('native16',source,neg,16,lead['source'][i,j]);measure('eight16',target,writes['adaptive'],16,lead['eight'][i,j]);measure('odd16',target,cap(partial,radius[target],.2),16,lead['odd'][i,j])
            if 'dose' in side:
                dd=side['dose'];measure('source_quarter',source,neg*.25,8,dd['source_quarter'][i,j]);measure('quarter',target,writes['adaptive']*.25,8,dd['quarter'][i,j]);measure('larger_cap',target,cap(unclipped['adaptive'],radius[target],.4),8,dd['larger_cap'][i,j]);errors['linear_target']=float(abs(jt@writes['adaptive']-dd['linear_target'][i,j]).max())
            errors['source_linear']=float(abs(js@neg-z['source_linear'][i,j]).max())
            if max(errors.values())>8e-5:raise ValueError((source,str(ev_ids[i]),int(selected_atoms[j]),errors))
            rows[source].append(dict(id=str(ev_ids[i]),domain=str(z['split'][i]),atom=int(selected_atoms[j]),activation=float(amplitude),
                    native=sr.tolist(),recipient=adapted.tolist(),source_jac_relative_error=ds,target_jac_relative_error=dt,
                    capped=bool(z['adaptive_clipped'][i,j]),norm_fraction={k:float(np.linalg.norm(v)/radius[target]) for k,v in writes.items()},
                    max_abs_delta_by_arm=errors,central_fd=measures))
    fd={}
    for source,rr in rows.items():
        fd[source]={}
        for m in MODELS_USED:
            pred=np.asarray([v['central_fd'][m]['predicted'] for v in rr]);actual=np.asarray([v['central_fd'][m]['measured'] for v in rr]);rel=float(np.linalg.norm(pred-actual)/np.linalg.norm(actual));
            if rel>.08:raise ValueError(('Production FD disagrees',source,m,rel))
            fd[source][m]=rel
    result=dict(seed=11202026,selection='Six distinct histories per domain and source. Cover both clipping states if observed, then favor distinct features. No outcomes used for selection.',
                derivative='Fresh singleton full hidden Jacobians from live absolute additions. Independent thin-SVD inverse and public one-sided plus central forecasts.',
                n_unique_histories=len(cases),n_pairs=sum(map(len,selected.values())),available_followups=followup_files,
                train_median_radius=radius,full_archive_point_and_interval_checks=settings,pooled_directional_fd_relative_error=fd,
                max_abs_replay_error=max(c for g in rows.values() for r in g for c in r['max_abs_delta_by_arm'].values()),
                max_fresh_jacobian_relative_error=max(v[2] for v in fresh.values()),sampled_pairs=rows,elapsed_seconds=time.monotonic()-start)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def extended():
    """Independent all-row cap checks, CAL-fixed inversion, follow-up metrics and cost ledger."""
    output=HERE/'results';record=output/'independent_audit.json'; report=json.loads(record.read_text()); stats={}
    parent=json.loads((output/'followups.json').read_text()); raw=cached(RAW/'data.npz'); ev_inputs=np.isin(raw['split'],['daily','machines']);
    ev_ids=raw['ids'][ev_inputs]; histories=raw['x'][ev_inputs];
    for source in MODELS_USED:
        target=next(m for m in MODELS_USED if m!=source); model=adapter_for(target)
        z=cached(output/(source+'.npz')); dose=cached(output/(source+'_dose.npz'));leads=cached(output/(source+'_leads.npz'))
        r=cached(REPLICATION/(source+'_seed0.npz')); ev=r['split']!='cal';amplitudes=r['source_codes'][ev]
        a=cached(JACS/(source+'_jac.npz')); b=cached(JACS/(target+'_jac.npz')); js=a['jac'].astype(float); jt=b['jac'].astype(float)
        for part in (z,leads,dose):
            np.testing.assert_array_equal(ev_ids,part['ids']);np.testing.assert_array_equal(amplitudes>0,part['active']);np.testing.assert_array_equal(z['split'],part['split']);np.testing.assert_array_equal(r['atoms'],part['atoms'])
        source_sae=load_sae_checkpoint(str(dictionary_path(source)));w=source_sae.W_dec.detach().numpy()[r['atoms']].T.astype(float)
        profile=js@w;mean_profile=profile[~ev].mean(0);weight=weight_from_calibration(js[~ev]);radius=float(b['radius'])
        h=8;ident=np.eye(h);avg=np.ones((h,h))/h;Q=avg+weight*(ident-avg)
        A=(Q@jt[~ev]).reshape(-1,jt.shape[-1]);Y=(Q@profile[~ev]).reshape(-1,w.shape[-1]);U,s,V=np.linalg.svd(A,full_matrices=False)
        fixed=V.T@((s/(s*s+.001*s[0]**2))[:,None]*(U.T@Y)); sel=report['sampled_pairs'][source]; fixed_errors=[]
        for case in sel:
            i=int(np.flatnonzero(ev_ids==case['id'])[0]);j=int(np.flatnonzero(r['atoms']==case['atom'])[0]);d=cap(-amplitudes[i,j]*fixed[:,j],radius,.2)
            obs=histories[i:i+1];ref=ordinary(model,target,obs,None,8); pred=ordinary(model,target,obs,d,8)-ref
            fixed_errors.append(float(abs(pred-z['fixed'][i,j]).max()))
        if max(fixed_errors)>8e-5:raise ValueError('Independent fixed write replay mismatch')
        methods=['adaptive','unweighted','mean_donor','wrong_leads','fixed']; frac={k:np.zeros_like(amplitudes,dtype=float) for k in methods}; js_eval=profile[ev]; jt_eval=jt[ev]
        for i in range(len(amplitudes)):
            natural=js_eval[i];J=jt_eval[i]
            for k in methods:
                pred=fixed if k=='fixed' else svd_write(J[[3,6,1,7,0,2,5,4]] if k=='wrong_leads' else J,
                        mean_profile if k=='mean_donor' else natural,1. if k=='unweighted' else weight)
                frac[k][i]=amplitudes[i]*np.linalg.norm(pred,axis=0)/radius
        for k in methods:np.testing.assert_array_equal(frac[k]>.2,z[k+'_clipped'])
        np.testing.assert_allclose(frac['adaptive'],dose['raw_fraction'],atol=1e-9,rtol=1e-6)
        np.testing.assert_array_equal(frac['adaptive']>.4,dose['larger_cap_clipped'])
        result=dict(direct_fixed_max_error=max(fixed_errors),cohorts={},followup_score_error=0.,followup_95_interval_error=0.)
        for domain in ('daily','machines'):
            take=z['split']==domain;active=z['active'][take];at=active.sum(0)>=8;mask=active[:,at]
            scope={k:dict(median_raw_fraction=float(np.median(v[take][active])),median_applied_fraction=float(np.median(np.minimum(v[take][active],.2))),n_clipped=int((v[take][active]>.2).sum())) for k,v in frac.items()}
            gt=leads['source']; close=lambda arr:arr[take][:,at].astype(float)
            groups={}
            for tag,indices in [('trained_odd',[0,2,4,6]),('held_even',[1,3,5,7]),('late',list(range(8,16)))]:
                groups[tag]={k:(close(leads[k])[:,:,indices],close(gt)[:,:,indices]) for k in ('odd','eight','odd_interpolation','last8_tile')}
            full=close(gt)[:,:,:8];quarter=close(dose['source_quarter']);groups['dose']={'original':(close(z['adaptive']),full), 'larger_cap':(close(dose['larger_cap']),full),'linear_target':(close(dose['linear_target']),full),'quarter':(close(dose['quarter']),quarter),'linear_quarter':(.25*close(dose['linear_target']),quarter),'donor_linear_quarter':(.25*close(z['source_linear']),quarter)}
            followup={}
            for tag,arms in groups.items():
                for kind in ('full','shape'):
                    for arm,(p,y) in arms.items():
                        P=p-p.mean(-1,keepdims=True) if kind=='shape' else p;Y=y-y.mean(-1,keepdims=True) if kind=='shape' else y
                        now=direct_scores(Y,P,mask,bootstrap=True); old=parent[source][domain][tag][kind][arm]
                        result['followup_score_error']=max(result['followup_score_error'],abs(now['skill']-old['value']));result['followup_95_interval_error']=max(result['followup_95_interval_error'],float(np.max(np.abs(np.asarray(now['ci95'])-old['ci95']))))
                        if kind=='shape' and arm==('odd' if tag!='dose' else 'quarter'):followup[tag]=dict(value=now['skill'],n_scored=now['n_scored'])
            result['cohorts'][domain]=dict(cap_and_norm_by_arm=scope,independently_scored_centered_followups=followup)
        if max(result['followup_score_error'],result['followup_95_interval_error'])>1e-7:raise ValueError('Independent follow-up metric discrepancy')
        stats[source]=result
    report['full_archive_norms_followups_and_fixed_replay']=stats
    report['access_and_limits']=['Each adaptive rule consumes eight source and eight target lead gradients on each evaluation history, amortized across all 16 selected atom directions. A fixed clean bridge does not have this test-time access.',
         'The mean-donor control retains the actual evaluation source coefficient and evaluation target Jacobian. Caps are common, but applied direction norms may differ.',
         'Horizon centering subtracts the eight-lead mean separately per intervention. It does not subtract a cross-history atom-specific template.',
         'For the four-lead arm, odd_interpolation uses only the four disclosed source derivative leads. last8_tile uses all eight source derivative leads, including held-even ones.',
         'The quarter-dose recovery uses the actual quarter-dose source counterfactual as its reference; the full-dose comparison uses a different response.']
    record.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extend-existing-replay',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    if not args.extend_existing_replay: main()
    extended()
