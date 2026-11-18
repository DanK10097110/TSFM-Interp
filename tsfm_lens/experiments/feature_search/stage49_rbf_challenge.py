"""One exploratory Bolt input-control challenge; cached cohorts, no native inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from stage49_failure_prediction import phase_response

MULTIPLIERS = (.5, 1., 2.)
ALPHAS = (.1, 1., 10., 100., 1000.)


def distance2(x,y):
    """Squared Euclidean distance with roundoff guarded at zero."""
    return np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(y*y,axis=1)[None,:]-2*x@y.T,0.)


def kernel_setup(x,multiplier):
    """Fit standardization and median bandwidth using training rows only."""
    x=np.asarray(x,float)
    if x.ndim!=2 or not np.isfinite(x).all(): raise ValueError('Finite seed feature matrix required')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-12,scale,1.);z=(x-mean)/scale
    d2=distance2(z,z);dist=np.sqrt(d2[np.triu_indices(len(x),1)])
    median=float(np.median(dist));median=median if median>1e-12 else 1.
    lengthscale=median*multiplier;kernel=np.exp(-d2/(2*lengthscale**2))
    column_mean=kernel.mean(0);grand=float(kernel.mean());centered=kernel-column_mean[None,:]-column_mean[:,None]+grand
    return dict(mean=mean,scale=scale,z=z,median=median,lengthscale=lengthscale,column_mean=column_mean,grand=grand,kernel=centered)


def solve_kernel(setup,y,alpha):
    """Fit a centered kernel with unpenalized training outcome mean."""
    y=np.asarray(y,float)
    if y.shape!=(len(setup['z']),) or not np.isfinite(y).all():raise ValueError('One finite outcome per seed required')
    center=float(y.mean());weights=np.linalg.solve(setup['kernel']+alpha*np.eye(len(y)),y-center)
    return dict(setup=setup,center=center,weights=weights,alpha=float(alpha))


def predict(fit,x):
    """Apply frozen train scaling, bandwidth and kernel-centering statistics."""
    setup=fit['setup'];z=(np.asarray(x)-setup['mean'])/setup['scale']
    kernel=np.exp(-distance2(z,setup['z'])/(2*setup['lengthscale']**2))
    centered=kernel-kernel.mean(1)[:,None]-setup['column_mean'][None,:]+setup['grand']
    return fit['center']+centered@fit['weights']


def train(x,y,folds):
    """Choose a fixed grid only by discovery seed OOF prediction error."""
    scores=[];predictions=[]
    for multiplier in MULTIPLIERS:
        setups={fold:kernel_setup(x[folds!=fold],multiplier) for fold in np.unique(folds)}
        for alpha in ALPHAS:
            oof=np.empty(len(y))
            for fold,setup in setups.items():
                held=folds==fold;fit=solve_kernel(setup,y[~held],alpha);oof[held]=predict(fit,x[held])
            scores.append(dict(lengthscale_multiplier=multiplier,alpha=alpha,mse=float(np.mean((oof-y)**2))))
            predictions.append(oof)
    best=int(np.argmin([row['mse'] for row in scores]));selected=scores[best]
    fit=solve_kernel(kernel_setup(x,selected['lengthscale_multiplier']),y,selected['alpha'])
    return fit,scores,predictions[best]


def raw_inputs(path):
    """Use the identical nine physical descriptors as the earlier quadratic model."""
    metadata=json.loads((path/'metadata.json').read_text(encoding='utf-8'))[::5]
    a,b,scale,offset,phi,psi=[np.array([row[k] for row in metadata]) for k in ['a','b','scale','offset','phi','psi']]
    return np.column_stack([np.log(a/b),np.log(scale),offset/scale,np.sin(phi),np.cos(phi),np.sin(psi),np.cos(psi),np.sin(psi-2*phi),np.cos(psi-2*phi)])



def bolt_cohort(path):
    """Read Bolt only; retain all natural perturbations inside each seed bundle."""
    protocol=json.loads((path/'protocol.json').read_text(encoding='utf-8'))
    metadata=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    conditions=protocol['conditions'];n=len(conditions)
    if n!=5 or len(metadata)!=128*n:raise ValueError('Need128 complete seed bundles')
    ids=[]
    for i,m in enumerate(metadata[::n]):
        rows=metadata[i*n:(i+1)*n]
        if any(row['seed']!=m['seed'] or row['condition']!=condition[0] for row,condition in zip(rows,conditions)):raise ValueError('Seed bundle mismatch')
        ids.append(m.get('generator_seed',protocol['seed_base']+i))
    if len(set(ids))!=128:raise ValueError('Duplicate generating IDs')
    with np.load(path/'bolt_capture.npz') as data:prediction=data['prediction'].copy()
    with np.load(path/'corpus.npz') as data:future=data['clean_future'].reshape(128,n,32)[:,0]
    if prediction.shape!=(640,32) or not np.isfinite(prediction).all():raise ValueError('Invalid Bolt predictions')
    scale=np.array([m['scale'] for m in metadata[::n]])
    response,_=phase_response(prediction,metadata,conditions,.01)
    alternate,_=phase_response(prediction,metadata,conditions,.02)
    loss=np.mean((prediction.reshape(128,n,32)[:,0]-future)**2,axis=1)/scale**2
    return dict(ids=ids,response=response,alternate_response=alternate,loss=loss)


def gain_summary(baseline,prediction,y):
    """Paired seed losses; descriptive pointwise mean bootstrap, no new p claim."""
    base=(baseline-y)**2;gain=base-(prediction-y)**2
    rng=np.random.default_rng(490201);draws=gain[rng.integers(0,len(gain),(2000,len(gain)))].mean(1)
    interval=list(np.quantile(draws,[.025,.975]))
    return dict(mean_prediction_loss_gain=float(gain.mean()),pointwise_bootstrap_ci95=interval,relative_gain=float(gain.mean()/base.mean()),interval_excludes_zero=bool(interval[0]>0),n_seeds=len(y)),gain


def assess(dx,ex,d,e,shuffled=32):
    """Fit input, joint and on-only models before evaluation, with fixed negative controls."""
    folds=np.random.default_rng(4901).permutation(len(dx))%5
    variants={'input_rbf':[], 'joint_rbf':[0,1], 'on_only_rbf':[1]}
    models={};saved={};fits={};ep={};alternates={}
    for name,columns in variants.items():
        tx=np.column_stack([dx,d['response'][:,columns]]);vx=np.column_stack([ex,e['response'][:,columns]])
        fit,grid,oof=train(tx,d['loss'],folds);prediction=predict(fit,vx)
        alternate=predict(fit,np.column_stack([ex,e['alternate_response'][:,columns]]))
        fits[name]=fit;ep[name]=prediction;alternates[name]=alternate
        models[name]=dict(selected=dict(lengthscale_multiplier=fit['setup']['lengthscale']/fit['setup']['median'],alpha=fit['alpha'],train_median_distance=fit['setup']['median']),cv_grid=grid,discovery_oof_mse=float(np.mean((oof-d['loss'])**2)),evaluation_prediction_mse=float(np.mean((prediction-e['loss'])**2)),discovery_mean_predictor_mse=float(np.mean((d['loss'].mean()-e['loss'])**2)))
        saved[name+'_eval_prediction']=prediction;saved[name+'_discovery_oof_prediction']=oof
    contrasts={}
    for name,aug,base in [('joint_vs_input','joint_rbf','input_rbf'),('on_only_vs_input','on_only_rbf','input_rbf'),('joint_vs_on_only','joint_rbf','on_only_rbf')]:
        value,gain=gain_summary(ep[base],ep[aug],e['loss']);alternate,_=gain_summary(ep[base],alternates[aug],e['loss']);value['delta02_gain']=alternate['mean_prediction_loss_gain'];contrasts[name]=value;saved[name+'_gain']=gain
    for name,columns in [('joint_vs_input',[0,1]),('on_only_vs_input',[1])]:
        rng=np.random.default_rng(490202+len(columns));controls=[]
        for _ in range(shuffled):
            tx=np.column_stack([dx,d['response'][rng.permutation(len(dx))][:,columns]])
            vx=np.column_stack([ex,e['response'][rng.permutation(len(ex))][:,columns]])
            fit,_,_=train(tx,d['loss'],folds);prediction=predict(fit,vx)
            controls.append(float(np.mean((ep['input_rbf']-e['loss'])**2-(prediction-e['loss'])**2)))
        value=contrasts[name];value['promotion_eligible']=(name=='joint_vs_input');value['shuffle_gain_distribution']=controls;value['shuffle_gain_quantiles']=list(np.quantile(controls,[0,.5,.95,1]))
        value['passes_frozen_robustness_gate']=bool(value['relative_gain']>=.05 and value['mean_prediction_loss_gain']>value['shuffle_gain_quantiles'][2] and value['pointwise_bootstrap_ci95'][0]>0)
    contrasts['joint_vs_on_only']['scope']='secondary coordinate ablation;no promotion or p-value claim'
    saved['eval_outcome']=e['loss'];saved['discovery_outcome']=d['loss']
    return dict(models=models,contrasts=contrasts,propose_fresh_replication=contrasts['joint_vs_input']['passes_frozen_robustness_gate'],secondary_on_only_cannot_rescue=True),saved


def run(discovery,evaluation,out):
    """Freeze all specifications and input bytes before computing associations."""
    out.mkdir(parents=True,exist_ok=False)
    paths={}
    for label,path in [('discovery',discovery),('evaluation',evaluation)]:
        paths[label]={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ['protocol.json','metadata.json','corpus.npz','bolt_capture.npz']}
    protocol=dict(scope='one exploratory Bolt RBF robustness challenge;both cohorts previously inspected;no new primary inference',sources=paths,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),response_helper_sha256=hashlib.sha256(Path(phase_response.__code__.co_filename).read_bytes()).hexdigest(),input_features='nine original raw descriptors;no quadratic expansion;RBF on standardized input,or combined standardized input+response',multipliers=MULTIPLIERS,alphas=ALPHAS,bandwidth='multiplier times median training-fold pairwise Euclidean distance;fold-only scaling',kernel='Gaussian exp(-distance²/(2lengthscale²));train-only centering;unpenalized outcome intercept',fold_seed=4901,folds=5,selection='discovery1659000only;refit128seeds;fixed evaluation2225000',models=['input_rbf','joint_rbf','on_only_rbf'],response='exact earlier log1p relative off-energy and on-target secant error;delta.01',controls=32,shuffle_seeds=[490204,490203],bootstrap_seed=490201,bootstrap_draws=2000,gate='joint relative prediction-loss gain>=5% AND exceeds95th shuffled gain percentile;AND pointwisebootstrap lower>0;no new p-value claim',secondary='on-only and joint-minus-on-only;no selected branch rescue',units='outcome is scaled forecastMSE;prediction losses and gains are squared scaled forecastMSE',caveats=['Finite input function class is not complete confounder adjustment','Shuffling is negative control,not a conditional-randomization null','Both historical outcome cohorts were already inspected','Perturbation-derived features require additional forecasts'])
    text=json.dumps(protocol,indent=2);(out/'protocol.json').write_text(text,encoding='utf-8')
    d,e=bolt_cohort(discovery),bolt_cohort(evaluation)
    if set(d['ids'])&set(e['ids']):raise ValueError('Generating seed overlap')
    result,saved=assess(raw_inputs(discovery),raw_inputs(evaluation),d,e)
    result.update(scope=protocol['scope'],protocol_sha256=hashlib.sha256(text.encode()).hexdigest())
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'source_data.npz',discovery_generator_ids=d['ids'],evaluation_generator_ids=e['ids'],**saved)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--discovery',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.discovery,a.evaluation,a.out)
