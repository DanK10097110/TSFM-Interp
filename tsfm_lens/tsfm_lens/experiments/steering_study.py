"""Reproducible short-to-long experiments for selective internal forecast control.

Synthetic contexts have independently generated train/calibration/test seeds.
Test optimization consumes only the model's response and the requested change,
never a future label. It is adaptive output control, not a zero-shot semantic
vector. Raw results and a short findings file are written after every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from ..sae.train import SAETrainConfig, train_sae
from ..steering import interval_shift, seasonal_shift, quantile_shift, solve_direction, response_metrics
from ..steering.native import BoltHead


def make_contexts(n, seed_base, context=128, *, shifted=False):
    """Independent mixed trend, periodicity, AR noise and level-shift contexts."""
    rows, metadata = [], []
    t = np.arange(context)
    for i in range(n):
        seed = seed_base+i
        rng = np.random.default_rng(seed)
        period = float(rng.choice([13.7, 21.3, 47.1] if shifted else [8., 16., 32.]))
        phase = rng.uniform(-np.pi, np.pi)
        amplitude = rng.uniform(.3, 2.)
        noise_scale = rng.uniform(.02, .5)
        noise = rng.normal(0, noise_scale, context)
        for k in range(1, context):
            noise[k] += .5*noise[k-1]
        slope = rng.uniform(-.012, .012)
        x = (rng.uniform(-2, 2)+slope*(t-context/2)+amplitude*np.sin(2*np.pi*t/period+phase)
             +rng.uniform(0, .6)*np.cos(2*np.pi*t/(period*1.7)+rng.uniform(-np.pi,np.pi))+noise)
        if i % 4 == 0:
            x[t > context*.7] += rng.uniform(-1, 1)
        rows.append(x)
        metadata.append(dict(seed=seed, period=period, phase=phase, amplitude=amplitude,
                             slope=slope, noise_scale=noise_scale))
    return np.asarray(rows, dtype=np.float32), metadata


def summarize(rows):
    """Group paired measurements without treating tasks as independent samples."""
    output = {}
    for key in sorted({(r['task'], r['method'], r['dose']) for r in rows}):
        task, method, dose = key
        subset = [r for r in rows if (r['task'],r['method'],r['dose']) == key]
        metrics = ['gain','collateral','relative_error','linearization_error','relative_hidden_norm',
                   'median_drift_rms','width_change','crossing_fraction']
        output[f'{task}/{method}/{dose:g}'] = dict(n=len(subset), **{
            name:dict(mean=float(np.mean([r[name] for r in subset])),
                      median=float(np.median([r[name] for r in subset]))) for name in metrics},
            success_fraction=float(np.mean([r['relative_error'] < .1 for r in subset])))
    return output


def run(args):
    """Run one immutable protocol, saving all decisions before seeing test outcomes."""
    started = time.time()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = vars(args).copy()
    protocol.update(evidence='causal within-model; exploratory fresh synthetic seeds; not sealed confirmation',
                    intervention='native decoder head input; original hidden + signed decoder combination',
                    test_selection='no test-based selection; all methods and doses reported',
                    success='full requested change relative RMS error < 0.1; no crossing postprocessing',
                    scope='width is not calibration; seasonal prior is an external desired edit',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if args.model == 'bolt':
        head = BoltHead(horizon=args.horizon)
    else:
        from ..steering.timesfm import TimesFMHead
        head = TimesFMHead(horizon=args.horizon)
    x, metadata = make_contexts(args.train+args.calibration, args.seed)
    test, test_meta = make_contexts(args.test, args.seed+100000, shifted=args.shifted)
    captured = head.capture(x)
    held = head.capture(test)
    htrain = captured['hidden'][:args.train]
    sae_cfg = SAETrainConfig(dict_size=args.dictionary, k=args.k, epochs=1,
                            min_train_steps=args.steps, batch_size=64, seed=args.seed,
                            aux_k=args.k, aux_dead_steps=20)
    if args.sae_kind == 'legacy':
        sae, history = train_sae(htrain, sae_cfg, torch.device('cpu'), context='steering native head')
    else:
        from ..sae.modern import train_modern_sae
        sae, history = train_modern_sae(htrain, sae_cfg, kind=args.sae_kind)
    sae.eval()
    dictionary = sae.W_dec.detach().cpu().numpy().astype(float)
    rng = np.random.default_rng(args.seed+17)
    random = rng.normal(size=dictionary.shape)
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    with torch.no_grad():
        recon, codes = sae(torch.from_numpy(held['hidden']))
    reconstruction = float(np.mean((recon.numpy()-held['hidden'])**2)/np.mean(
        (held['hidden']-htrain.mean(0))**2))
    dictionary_report = dict(normalized_reconstruction_mse=reconstruction,
                             live_test_fraction=float((codes != 0).any(0).float().mean()),
                             measured_l0=float((codes != 0).sum(1).float().mean()),
                             train_history=history, train_meta=sae.train_meta)
    torch.save(dict(state_dict=sae.state_dict(), config=vars(sae_cfg), kind=args.sae_kind), out/'sae.pt')
    np.savez_compressed(out/'corpus.npz', train=x, test=test, train_hidden=captured['hidden'],
                        test_hidden=held['hidden'], baseline=held['normalized'], dictionary=dictionary)
    (out/'metadata.json').write_text(json.dumps(dict(train=metadata,test=test_meta), indent=2)+'\n')
    print(json.dumps(dict(stage='capture_and_sae', seconds=time.time()-started,
                         reconstruction=dictionary_report['normalized_reconstruction_mse'])), flush=True)
    cal_js = np.stack([head.jacobian(h).astype(float) for h in captured['hidden'][args.train:]])
    width = htrain.shape[1]
    frozen = linalg.solve(np.einsum('nmd,nme->de', cal_js, cal_js)/len(cal_js)
                          +args.ridge*np.eye(width), cal_js.mean(0).T, assume_a='pos')
    np.savez_compressed(out/'frozen.npz', controller=frozen)
    methods = [('dense', None, None), ('frozen_dense', None, None)]
    for count in args.budgets:
        methods += [(f'sae_{count}',dictionary,count), (f'neurons_{count}',np.eye(width),count),
                    (f'random_{count}',random,count)]
    rows, arrays, spectra = [], {}, []
    full_model_error = 0.
    for index, (h, baseline, context, meta) in enumerate(zip(
            held['hidden'],held['normalized'],test,test_meta)):
        j = head.jacobian(h).astype(float)
        singular = linalg.svdvals(j)
        spectra.append(dict(seed=meta['seed'],largest=float(singular[0]),smallest=float(singular[-1]),
                            rank=int((singular > max(j.shape)*np.finfo(np.float32).eps*singular[0]).sum())))
        norm_limit = args.radius*np.linalg.norm(h)
        for dose in args.doses:
            tasks = dict(widen=interval_shift(baseline,head.quantiles,dose),
                         narrow=interval_shift(baseline,head.quantiles,-dose),
                         season=seasonal_shift(context,args.horizon,head.quantiles,
                                               period=meta['period'],amplitude=dose),
                         upper=quantile_shift(args.horizon,head.quantiles,channel=.9,amount=dose))
            for task, target in tasks.items():
                for method, basis, count in methods:
                    if method == 'frozen_dense':
                        delta = frozen @ target.flatten()
                        clipped = np.linalg.norm(delta) > norm_limit
                        if clipped:
                            delta *= norm_limit/np.linalg.norm(delta)
                        predicted = j @ delta
                        selected = width
                    else:
                        solution = solve_direction(j,target.flatten(),basis,max_features=count,
                                                   ridge=args.ridge,max_norm=norm_limit)
                        delta, predicted, clipped = solution.delta, solution.predicted, solution.clipped
                        selected = len(solution.selected)
                    with torch.no_grad():
                        edited = head.forward(head.tensor(h+delta)).cpu().numpy()
                    actual = edited-baseline
                    metrics = response_metrics(actual,target,predicted.reshape(target.shape))
                    row = dict(seed=meta['seed'],task=task,method=method,dose=dose,**metrics,
                               relative_hidden_norm=float(np.linalg.norm(delta)/np.linalg.norm(h)),
                               clipped=bool(clipped),selected=selected,
                               median_drift_rms=float(np.sqrt(np.mean(actual[:,4]**2))),
                               width_change=float(np.mean(actual[:,-1]-actual[:,0])),
                               crossing_fraction=float(np.mean(np.diff(edited,axis=-1)<0)),
                               baseline_crossing_fraction=float(np.mean(np.diff(baseline,axis=-1)<0)))
                    rows.append(row)
                    key=f'{index}_{task}_{method}_{dose:g}'
                    arrays[key+'_edited']=edited
                    arrays[key+'_target']=target
                    arrays[key+'_delta']=delta.astype(np.float32)
                    if index < 2 and dose == args.doses[0] and method == 'dense':
                        actual_full=head.predict_edited(context[None],delta[None])[0]
                        replay=edited*held['scale'][index]+held['location'][index]
                        error=float(np.max(np.abs(actual_full-replay)))
                        full_model_error=max(full_model_error,error)
                        if not np.allclose(actual_full,replay,atol=1e-5,rtol=1e-5):
                            raise RuntimeError(f'Edited native replay disagrees with full model: {error}')
        print(json.dumps(dict(stage='test',completed=index+1,total=args.test,
                              seconds=time.time()-started)),flush=True)
    summary=summarize(rows)
    result=dict(protocol=protocol,dictionary=dictionary_report,summary=summary,rows=rows,
                spectra=spectra,identity=dict(capture=max(captured['replay_max_abs'],held['replay_max_abs']),
                                            edited_full_model_max_abs=full_model_error),
                environment=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
                                 cuda_available=torch.cuda.is_available()),seconds=time.time()-started)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'interventions.npz',**arrays)
    lines=['# Steering experiment findings','',f'Elapsed: {result["seconds"]:.2f} seconds.',
           '', 'Causal internal edits on fresh synthetic contexts; no forecast accuracy or calibration claim.',
           '', '| Task / method / dose | Relative error | Collateral | Gain | Success |',
           '| --- | ---: | ---: | ---: | ---: |']
    for key,values in summary.items():
        lines.append(f'| {key} | {values["relative_error"]["mean"]:.5f} | '
                     f'{values["collateral"]["mean"]:.5f} | {values["gain"]["mean"]:.5f} | '
                     f'{values["success_fraction"]:.3f} |')
    lines += ['', 'Read protocol.json for frozen choices, results.json for every sample and diagnostics,',
              'and interventions.npz for actual forecasts, targets and hidden additions.',
              'Sparse greedy failures are not impossibility results; dense success is not monosemanticity.']
    (out/'findings.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(out=str(out),seconds=result['seconds'],identity=result['identity'])),flush=True)


def main():
    """Command-line short pilot; use explicit larger budgets for follow-ups."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--train',type=int,default=128)
    p.add_argument('--calibration',type=int,default=8)
    p.add_argument('--test',type=int,default=8)
    p.add_argument('--seed',type=int,default=5160000)
    p.add_argument('--dictionary',type=int,default=512)
    p.add_argument('--k',type=int,default=16)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--sae-kind',choices=['legacy','topk','abstopk','batchtopk'],default='legacy')
    p.add_argument('--model',choices=['bolt','timesfm'],default='bolt')
    p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--ridge',type=float,default=.001)
    p.add_argument('--radius',type=float,default=.2)
    p.add_argument('--budgets',type=int,nargs='+',default=[1,8,32])
    p.add_argument('--doses',type=float,nargs='+',default=[.2])
    p.add_argument('--shifted',action='store_true')
    args=p.parse_args()
    if min(args.train,args.calibration,args.test,args.dictionary,args.steps,args.threads) < 1:
        p.error('sample counts and compute budgets must be positive')
    if args.train+args.calibration > 100000:
        p.error('training/calibration seeds would overlap the test seed range')
    if any(not 0 < dose < 1 for dose in args.doses):
        p.error('doses must lie strictly between 0 and 1')
    run(args)


if __name__ == '__main__':
    main()
