"""Exploratory held-cohort prediction from cached native phase responses; CPU only.

Each observation is one generating seed. This analysis asks incremental predictive
value conditional on a fixed input descriptor model, not causal mediation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from tsfm_lens.analysis.stats import holm

ALPHAS = (0.1, 1., 10., 100., 1000.)
MODELS = ('bolt', 'timesfm', 'chronos2')


def quadratic(values):
    """Fixed input-only interactions allow modest nonlinear confounding adjustment."""
    x = np.asarray(values, dtype=float)
    return np.column_stack([x] + [x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])])


def fit_ridge(x, y, alpha):
    """Fit every preprocessing parameter on training observations alone."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Need finite seed-level feature matrix and outcome vector')
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x - mean) / scale
    center = float(y.mean())
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - center))
    return dict(mean=mean, scale=scale, center=center, weights=weights, alpha=float(alpha))


def predict(fit, x):
    """Apply frozen training scaling without touching evaluation outcomes."""
    return fit['center'] + ((np.asarray(x) - fit['mean']) / fit['scale']) @ fit['weights']


def train(x, y, folds):
    """Choose ridge penalty using only prespecified discovery seed folds."""
    scores = []
    for alpha in ALPHAS:
        losses = np.empty(len(y))
        for fold in np.unique(folds):
            held = folds == fold
            fit = fit_ridge(x[~held], y[~held], alpha)
            losses[held] = (predict(fit, x[held]) - y[held]) ** 2
        scores.append(float(losses.mean()))
    chosen = int(np.argmin(scores))
    fit = fit_ridge(x, y, ALPHAS[chosen])
    fit['cv_mse'] = scores
    return fit


def coefficients(values):
    """Joint harmonic OLS at native future origin128, in physical output units."""
    t = np.arange(128, 160)
    design = np.column_stack([np.sin(2*np.pi*t/32), np.cos(2*np.pi*t/32), np.sin(2*np.pi*t/16), np.cos(2*np.pi*t/16), np.ones(32)])
    return np.asarray(values, float) @ np.linalg.pinv(design).T


def phase_response(prediction, metadata, conditions, delta):
    """Use centered natural-input differences, one feature vector per entire seed."""
    plus = next(i for i, c in enumerate(conditions) if c[1:] == ['psi', 1, delta])
    minus = next(i for i, c in enumerate(conditions) if c[1:] == ['psi', -1, delta])
    n = len(conditions)
    pred = np.asarray(prediction, float).reshape(-1, n, 32)
    base = metadata[::n]
    derivative = (pred[:, plus] - pred[:, minus]) / (2*delta)
    coeff = coefficients(derivative)[:, :4]
    scale = np.array([m['scale'] for m in base])
    b = np.array([m['b'] for m in base])
    psi = np.array([m['psi'] for m in base])
    target = np.column_stack([-b*np.sin(psi), b*np.cos(psi)]) * (np.sin(delta)/delta)
    strength = (b/scale)**2
    off = np.sum((coeff[:, :2]/scale[:, None])**2, axis=1) / strength
    on_error = np.sum(((coeff[:, 2:]-target)/scale[:, None])**2, axis=1) / strength
    return np.column_stack([np.log1p(off), np.log1p(on_error)]), coeff[:, :2]/scale[:, None]


def load_cohort(path, chronos2_path=None):
    """Validate full bundles and derive covariates without using perturbed rows as replicates."""
    protocol = json.loads((path/'protocol.json').read_text(encoding='utf-8'))
    metadata = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    conditions = protocol['conditions']; n = len(conditions)
    if n != 5 or len(metadata) != protocol['n_seeds']*n or protocol['n_seeds'] != 128:
        raise ValueError('Require complete128seed five-condition cohorts')
    base = metadata[::n]
    ids = []
    for seed, m in enumerate(base):
        rows = metadata[seed*n:(seed+1)*n]
        if any(row['seed'] != m['seed'] or row['condition'] != c[0] for row, c in zip(rows, conditions)):
            raise ValueError('Seed bundle identity/order mismatch')
        for key in ['a','b','offset','scale','phi']:
            if any(row[key] != m[key] for row in rows): raise ValueError('Nuisance changed within bundle')
        for row, c in zip(rows, conditions):
            if not np.isclose(row['psi'], m['psi'] + c[2]*c[3], atol=1e-12, rtol=0): raise ValueError('Wrong phase perturbation')
        ids.append(m.get('generator_seed', protocol['seed_base']+seed))
    if len(set(ids)) != len(ids): raise ValueError('Duplicate generating seeds')
    a,b,scale,offset,phi,psi = [np.array([m[k] for m in base]) for k in ['a','b','scale','offset','phi','psi']]
    if np.any(a <= 0) or np.any(b <= 0) or not np.allclose(scale**2, (a*a+b*b)/2): raise ValueError('Invalid analytic energy')
    raw = np.column_stack([np.log(a/b), np.log(scale), offset/scale, np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi), np.sin(psi-2*phi), np.cos(psi-2*phi)])
    with np.load(path/'corpus.npz') as data: future = data['clean_future'].reshape(-1,n,32)[:,0]
    expected = offset[:,None]+a[:,None]*np.sin(2*np.pi*np.arange(128,160)/32+phi[:,None])+b[:,None]*np.sin(2*np.pi*np.arange(128,160)/16+psi[:,None])
    if not np.allclose(future, expected, atol=1e-6): raise ValueError('Future origin or labels mismatch')
    models = {}
    if chronos2_path is not None:
        binding = json.loads((chronos2_path/'protocol.json').read_text(encoding='utf-8'))
        for name in ['corpus','metadata']:
            suffix = '.npz' if name == 'corpus' else '.json'
            if hashlib.sha256((path/(name+suffix)).read_bytes()).hexdigest() != binding[name+'_sha256']:
                raise ValueError('Chronos2 discovery corpus binding mismatch')
    for model in MODELS:
        capture_path = chronos2_path if model == 'chronos2' and chronos2_path is not None else path
        with np.load(capture_path/f'{model}_capture.npz') as data: prediction = data['prediction'].copy()
        if prediction.shape != (len(metadata),32) or not np.isfinite(prediction).all(): raise ValueError('Bad native predictions')
        response, off = phase_response(prediction, metadata, conditions, .01)
        alternate, _ = phase_response(prediction, metadata, conditions, .02)
        loss = np.mean((prediction.reshape(-1,n,32)[:,0]-future)**2, axis=1)/scale**2
        models[model] = dict(response=response, alternate_response=alternate, loss=loss)
        # The archived branch audit stores [cos,sin], unlike OLS [sin,cos].
        cached_path = path/'source_data.npz'
        if cached_path.is_file():
            with np.load(cached_path) as data:
                if not np.allclose(off[:,[1,0]], data[model+'_0.01_native_off_derivative'], atol=1e-8): raise ValueError('Cached derivative disagrees')
    return dict(ids=ids, descriptors=quadratic(raw), models=models)


def effect(values, seed=4901):
    """Seed-paired mean interval and distinct exact conditional sign estimand."""
    x = np.asarray(values)
    rng = np.random.default_rng(seed); draws = x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
    positive, negative = int(np.sum(x>0)), int(np.sum(x<0)); nonzero = positive+negative
    return dict(mean=float(x.mean()), ci95=list(np.quantile(draws,[.025,.975])), n_seeds=len(x), positive=positive, negative=negative, ties=int(np.sum(x==0)), sign_p=1. if nonzero==0 else float(binomtest(positive,nonzero,.5).pvalue), scope='pointwise mean bootstrap; exact two-sided sign test concerns conditional majority, not mean')


def assess(discovery, evaluation, shuffled=32):
    """Freeze discovery fits then evaluate all six fixed augmentation comparisons."""
    if set(discovery['ids']) & set(evaluation['ids']): raise ValueError('Discovery/evaluation generating seeds overlap')
    folds = np.random.default_rng(4901).permutation(len(discovery['ids'])) % 5
    report, saved, claims = {}, {}, {}
    for model in MODELS:
        d,e = discovery['models'][model], evaluation['models'][model]
        dx,ex = discovery['descriptors'],evaluation['descriptors']
        baseline = train(dx,d['loss'],folds); bp = predict(baseline,ex); bl = (bp-e['loss'])**2
        mean_loss = (d['loss'].mean()-e['loss'])**2
        rows = dict(input_baseline=dict(eval_prediction_mse=float(bl.mean()), mean_predictor_mse=float(mean_loss.mean()), outcome_mean=float(e['loss'].mean()), discovery_outcome_mean=float(d['loss'].mean()), discovery_outcome_variance=float(d['loss'].var()), r2_vs_discovery_mean=float(1-bl.mean()/mean_loss.mean()), outcome_variance=float(e['loss'].var()), worse_than_mean_predictor=bool(bl.mean()>mean_loss.mean()), alpha=baseline['alpha'], cv_mse=baseline['cv_mse']), augmentations={})
        saved[model+'_outcome'] = e['loss']; saved[model+'_baseline_prediction'] = bp
        for count,name in [(1,'off_only'),(2,'off_and_on_error')]:
            fit = train(np.column_stack([dx,d['response'][:,:count]]),d['loss'],folds)
            prediction = predict(fit,np.column_stack([ex,e['response'][:,:count]])); losses=(prediction-e['loss'])**2
            gain=bl-losses; result=effect(gain); result.update(eval_prediction_mse=float(losses.mean()),alpha=fit['alpha'],cv_mse=fit['cv_mse'],gain_fraction_of_baseline=float(gain.mean()/bl.mean()))
            rng=np.random.default_rng(490100+count); controls=[]
            for draw in range(shuffled):
                ds=d['response'][rng.permutation(len(dx)),:count]; es=e['response'][rng.permutation(len(ex)),:count]
                nullfit=train(np.column_stack([dx,ds]),d['loss'],folds)
                nullpred=predict(nullfit,np.column_stack([ex,es])); controls.append(float(np.mean(bl-(nullpred-e['loss'])**2)))
            result['shuffled_gain_distribution']=controls; result['shuffled_gain_quantiles']=list(np.quantile(controls,[0,.5,.95,1]))
            alternate = predict(fit,np.column_stack([ex,e['alternate_response'][:,:count]]))
            result['delta02_fixed_fit_gain']=float(np.mean(bl-(alternate-e['loss'])**2))
            rows['augmentations'][name]=result; claims[model+'/'+name]=result['sign_p']; saved[model+'_'+name+'_prediction']=prediction; saved[model+'_'+name+'_gain']=gain
        report[model]=rows
    adjusted=holm(claims)
    for key,p in adjusted.items():
        model,name=key.split('/')
        report[model]['augmentations'][name]['sign_p_holm6']=p
        result=report[model]['augmentations'][name]
        result['exploratory_promotion']=bool(result['gain_fraction_of_baseline'] >= .05 and p < .05 and result['mean'] > result['shuffled_gain_quantiles'][2])
    return report,saved


def run(discovery_path, evaluation_path, out, discovery_chronos2, prepare_only=False):
    """Write protocol before loading any associations; outputs are never overwritten."""
    out.mkdir(parents=True,exist_ok=False)
    sources={}
    for label,path in [('discovery',discovery_path),('evaluation',evaluation_path)]:
        names=['protocol.json','metadata.json','corpus.npz']+[f'{m}_capture.npz' for m in MODELS if m != 'chronos2' or label == 'evaluation']
        if (path/'source_data.npz').is_file(): names.append('source_data.npz')
        sources[label]=dict(path=str(path.resolve()),sha256={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in names})
    sources['discovery_chronos2']=dict(path=str(discovery_chronos2.resolve()),sha256={name:hashlib.sha256((discovery_chronos2/name).read_bytes()).hexdigest() for name in ['protocol.json','chronos2_capture.npz','source_data.npz']})
    protocol=dict(stage=49,scope='exploratory, both cohort outcomes previously inspected; no fresh scientific confirmation',sources=sources,models=MODELS,unit='one whole generating seed;128 discovery and128 evaluation',outcome='unperturbed native forecast MSE / common mix variance',prediction_loss_unit='squared scaled forecast MSE;gain is input-only squared prediction error minus augmented squared prediction error',input_descriptors='log(a/b),log(scale),offset/scale,sin/cos(phi),sin/cos(psi),sin/cos(psi-2phi);all pairwise products including squares',response='centered native psi derivative delta.01;log1p(off-period32 energy / analytic period16 derivative strength);secondary on-target coefficient error versus true finite-difference sin(delta)/delta secant /strength',training='5 fixed discovery-only folds, scaler within each training fold;refit all discovery;separate model fits',alphas=ALPHAS,shuffle_controls=32,shuffle_seed_by_augmentation=[490101,490102],promotion='at least5% held-evaluation predictionMSE improvement,selected signHolm6<.05,and gain exceeds95th shuffled-feature percentile;exploratory only',bootstrap_seed=4901,n_bootstrap=2000,family='six fixed model x augmentation exact two-sided conditional sign tests;globalHolm6;no assumption-free mean inference',delta02='secondary fixed-fit input-feature stability;no tuning',caveat='Predictive association conditional on finite quadratic input baseline, not causal explanation or complete confounder removal; perturbation probes require additional forecasts',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if prepare_only: return
    discovery,evaluation=load_cohort(discovery_path,discovery_chronos2),load_cohort(evaluation_path)
    report,saved=assess(discovery,evaluation)
    (out/'results.json').write_text(json.dumps(dict(protocol=protocol,models=report),indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'source_data.npz',discovery_generator_ids=discovery['ids'],evaluation_generator_ids=evaluation['ids'],**saved)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--discovery',type=Path,required=True);parser.add_argument('--evaluation',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--discovery-chronos2',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();run(args.discovery,args.evaluation,args.out,args.discovery_chronos2,args.prepare_only)
