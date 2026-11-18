"""Frozen reuse-only comparability pilot: freeze, validate, then analyze saved data.

No native adapters, torch, or prior experiment modules are imported. Resource
limits and provenance are checked before fitting; only new run artifacts change.
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_info
from experiments.feature_search.comparability_core import (descriptors, folds, group_values, interval,
    kernel_bank, select_predict, shuffled_indices, weights)

ROOT = Path(__file__).resolve().parents[2]
RUNS = ('selective_pilot4941000', 'selective_replication4942000')
MODELS = ('bolt', 'timesfm')
ENDPOINTS = ('forecast', 'fusion_response', 'log_failure', 'failure_flag')


def read(path):
    """Load an explicit UTF-8 JSON artifact."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    """Never replace an existing result artifact."""
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    """Stream hashes without loading state archives into memory."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_paths():
    """Limit ownership and executable provenance to this pilot's new files."""
    return sorted(list((ROOT/'experiments/feature_search').glob('comparability_*.py')) + list((ROOT/'tests').glob('test_comparability_*.py')))


def freeze(out):
    """Bind a fully specified protocol to immutable upstream files before validation."""
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in RUNS:
        p = ROOT/'runs'/name
        for file in ['rows.json', 'plans.json', 'inputs.npz', 'targets.npz', 'protocol.json'] + [f'{m}_{suffix}' for m in MODELS for suffix in ['states.npz', 'forecasts.json', 'worker.json']]:
            files[str((p/file).relative_to(ROOT))] = sha(p/file)
    protocol = dict(seed=4951000, evidence='Exploratory source-held-out predictive comparison on previously studied data; not sealed confirmation or causality.',
        resources=dict(cpu_threads=1, address_space_limit_bytes=3*2**30, native_inference=False),
        source_paths=list(RUNS), source_hashes=files,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in code_paths()},
        split='Use only original FIT sources from both runs. Replication TEST primary; pilot TEST separate sensitivity only. No TEST used to choose settings. Source IDs must be globally disjoint across FIT/TEST.',
        cohorts='Synthetic separately; pooled four-domain natural real separately; injected real separately (artificial contexts, not natural-real evidence). Each real domain TEST has four sources. No domain extrapolation claim.',
        features='Full original token flattening at last saved cut: Bolt block3 9x512, TimesFM block10 4x1280. Same 128-step context; no token/window pseudo-replication. Fixed role-depth, not homologous layers or actual head-input ceiling.',
        input='128 z-scored past values, 65 FFT magnitudes, ACF lags1..32, 9 quantiles, context mean/SD and log SD, saved past-only decomposition tone descriptors. FIT standardized. Untruncated full-history ridge and full-descriptor ridge, PCA8/16, or RBF gamma .5/2 divided by FIT median squared distance.',
        normalization='Both input and hidden probes predict identical normalized targets. Forecast inverse is history_mean + history_SD * predicted_normalized_forecast for every arm; response inverse is history_SD * predicted_normalized_response. Log failure uses normalized MSE, flag compares native/persistence on the same row. No hidden-only arm is asked to recover physical scale/offset omitted by native normalization.',
        fitting='Weighted FIT-only dual PCA rank8/16 capped at training source count minus1, preserving orthogonal geometry with one scalar normalization. Four domain-stratified source folds choose kernel and ridge alpha .001/.01/.1/1 independently per endpoint/arm; then refit all FIT. Each source equal total weight.',
        arms=['input', 'A_hidden', 'input+A_hidden', 'B_hidden', 'input+B_hidden', 'input+A_forecast'],
        endpoint_definitions=dict(forecast='32-step B native forecast centered by past mean and scaled by past SD',
            fusion_response='B saved depth1, first-cut, equal-weight, blend1 fusion minus B native, divided by past SD. If tree abstains, response exactly0; no TEST selection.',
            log_failure='log1p(mean squared B-native error / past variance)',
            failure_flag='B native MSE exceeds last-value persistence MSE; ridge probability clipped[0,1], Brier loss'),
        same_model='B hidden is a same-capacity empirical comparator, not a guaranteed upper bound; sampled layer does not include whole decoder.',
        controls='19 uniform FIT source-block hidden permutations within cohort/domain/case signature, TEST unchanged; refit PCA and reselect ridge for every permutation. Orthogonal Householder rotations of actual full-token hidden features must preserve Gram/probe predictions. General nonorthogonal transforms need not preserve PCA/ridge regularization.',
        uncertainty='Mean row loss within source then equal source mean. Paired 4000 source-bootstrap draws, frozen predictions; 95% exploratory intervals and 98.75% Bonferroni intervals for four synthetic primary contrasts. No training-set or new-domain uncertainty.',
        primary='Synthetic replication TEST: both directions x forecast/log_failure, input+A_hidden vs input. All other endpoints/cohorts descriptive. 19 null draws give coarse minimum tail rank .05; no formal permutation p-value claim.',
        promotion='Recommend targeted new-data expansion only if a synthetic primary contrast has >=5% lower loss, 98.75% gain interval lower>0, exceeds all19 shuffled alignment gains, passes actual-state rotation, and same-model input+B_hidden improves over input. Otherwise no broad expansion; negative result is recipe/power bounded.',
        limitations=['Prior data/results inspected historically; frozen protocol does not restore unseen-data status.', 'Real4/domain and unknown pretraining overlap; synthetic generator families narrow and related.', 'Failure prediction is supervised target-risk estimation; future labels only enter FIT fitting and TEST scoring.', 'Hidden states are deterministic input transforms; incremental value depends on baseline and probe class, not information-theoretic independence.', 'No raw cross-model L2, neuron matching, causal mediation, or mechanistic identity claim.', 'No layer/PCA/hyperparameter changes after inspecting TEST.'])
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', dict(protocol_sha256=sha(out/'protocol.json'), unix_time=time.time()))
    print('Frozen', out, flush=True)


def verify(out):
    """Reject changed protocol, code, or upstream evidence."""
    p = read(out/'protocol.json')
    if sha(out/'protocol.json') != read(out/'freeze.json')['protocol_sha256']:
        raise ValueError('Protocol changed')
    if (out/'amendment_01.json').exists():
        amendment = read(out/'amendment_01.json')
        assert sha(out/'amendment_01.json') == read(out/'amendment_freeze.json')['sha256']
        assert amendment['protocol_sha256'] == sha(out/'protocol.json')
        p['code_hashes'] = amendment['code_hashes']
    for key in ('source_hashes', 'code_hashes'):
        for path, digest in p[key].items():
            if sha(ROOT/path) != digest:
                raise ValueError('Frozen artifact changed: '+path)
    return p


def validate(out):
    """Run the known-answer and regression-plant suite before opening study outcomes."""
    protocol = verify(out)
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_comparability_pilot.py', '-q'], cwd=ROOT, text=True, capture_output=True, env=dict(os.environ, TSFM_TEST_THREADS='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'))
    stem = 'validation_v2' if (out/'amendment_01.json').exists() else 'validation'
    (out/f'{stem}.log').write_text(result.stdout+result.stderr, encoding='utf-8')
    write(out/f'{stem}.json', dict(passed=result.returncode == 0, returncode=result.returncode,
        code_hashes=protocol['code_hashes'], plants='Known linear shared signal and independent decoy; deleted signal, broken rotation, row split, TEST-fit leakage, row bootstrap, coordinate-scaling and unstable CV ties explicitly rejected in tests.'))
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError('Validation failed; analysis forbidden')


def load_data():
    """Read small saved slices one at a time; never load native model libraries."""
    rows, xs, ys, plans = [], [], [], []
    hidden = {m:[] for m in MODELS}
    pred = {m:[] for m in MODELS}
    responses = {m:[] for m in MODELS}
    availability = []
    offset = 0
    shapes = {}
    for name in RUNS:
        p = ROOT/'runs'/name
        rs, ps = read(p/'rows.json'), read(p/'plans.json')
        original_protocol = read(p/'protocol.json')
        for filename, digest in original_protocol['hashes'].items():
            assert sha(p/filename) == digest, 'Upstream input binding changed'
        with np.load(p/'inputs.npz') as a:
            x = np.asarray(a['history'], float)
        with np.load(p/'targets.npz') as a:
            y = np.asarray(a['future'], float)
        assert len(rs) == len(x) == len(y) == len(ps)
        rows.extend([dict(r, run=name, local_row=i, index=offset+i) for i,r in enumerate(rs)])
        xs.extend(x); ys.extend(y); plans.extend(ps)
        availability.extend(len(plan['levels']) > 1 for plan in ps)
        for m in MODELS:
            w = read(p/f'{m}_worker.json')
            for suffix in ('states.npz', 'forecasts.json'):
                assert sha(p/f'{m}_{suffix}') == w['hashes'][f'{m}_{suffix}'], 'Upstream worker binding changed'
            assert all(w['gates'][k] == 0 for k in ('self_patch', 'normalizer_identity', 'repeat'))
            records = read(p/f'{m}_forecasts.json')
            native = {r['row']:np.asarray(r['forecast'], float) for r in records if r['method'] == 'native'}
            fused = {r['row']:np.asarray(r['forecast'], float) for r in records if r['method'] == 'fusion' and r['depth'] == 1 and r['cut'] == w['cuts'][0] and r['weight'] == 'equal' and r['blend'] == 1}
            assert len(native) == len(rs)
            with np.load(p/f'{m}_states.npz') as a:
                for i in range(len(rs)):
                    h = a[f'{i}_full'][-1].astype(float)
                    shapes[m] = list(h.shape)
                    hidden[m].append(h.ravel())
                    pred[m].append(native[i])
                    if len(ps[i]['levels']) > 1 and i not in fused:
                        raise ValueError('Missing declared fixed fusion')
                    responses[m].append(fused.get(i, native[i]) - native[i])
        offset += len(rs)
    for s in {r['source'] for r in rows}:
        assert len({r['split'] for r in rows if r['source'] == s}) == 1, 'FIT/TEST source overlap'
    x,y = np.asarray(xs),np.asarray(ys)
    hidden = {m:np.asarray(v) for m,v in hidden.items()}
    forecasts = {m:np.asarray(v) for m,v in pred.items()}
    sd,mean = x.std(1),x.mean(1)
    targets = {}
    for m in MODELS:
        mse = np.mean((forecasts[m]-y)**2, axis=1)
        persistence = np.mean((x[:, -1, None]-y)**2, axis=1)
        targets[m] = dict(forecast=(forecasts[m]-mean[:, None])/sd[:, None],
            fusion_response=np.asarray(responses[m])/sd[:, None],
            log_failure=np.log1p(mse/sd**2)[:, None], failure_flag=(mse>persistence).astype(float)[:, None])
    return rows, descriptors(x, plans), hidden, targets, availability, shapes


def make_banks(x, grams, rows, train):
    """Cache label-free training transforms for inner folds and final refit."""
    return [(tr,va,kernel_bank(x, grams, rows,tr)) for tr,va in folds(rows,train)], kernel_bank(x,grams,rows,train)


def analyze(out):
    """Fit frozen probes, then publish every direction/endpoint including negatives."""
    started = time.monotonic()
    protocol = verify(out)
    stem = 'validation_v2' if (out/'amendment_01.json').exists() else 'validation'
    validation = read(out/f'{stem}.json')
    if not validation['passed'] or validation['code_hashes'] != protocol['code_hashes']:
        raise ValueError('Validation gate not passed')
    if (out/'results.json').exists():
        raise ValueError('Refuse repeated result look')
    rows,x,hidden,targets,available,shapes = load_data()
    rotations = {}
    for m,h in hidden.items():
        rng = np.random.default_rng(4951000)
        rotated = h.copy()
        for _ in range(3):
            v = rng.normal(size=h.shape[1]); v /= np.linalg.norm(v)
            rotated -= 2*np.outer(rotated@v,v)
        k,kr = h@h.T,rotated@rotated.T
        error = float(np.max(abs(k-kr))/max(np.max(abs(k)),1e-14))
        assert error < 1e-10
        rotations[m] = dict(relative_gram_error=error, method='three dense Householder reflections in flattened coordinates', passed=True)
    all_results, predictions, selections, source_losses, inventories = [], {}, [], [], []
    cohorts = {'synthetic':lambda r:r['domain']=='synthetic',
               'real_natural':lambda r:r['case']=='real', 'real_injected':lambda r:r['case']=='real_injected'}
    for cohort,keep in cohorts.items():
        ids = np.array([i for i,r in enumerate(rows) if keep(r)])
        rs = [rows[i] for i in ids]; xx = x[ids]
        train = np.array([i for i,r in enumerate(rs) if r['split']=='fit'])
        tests = {leg:np.array([i for i,r in enumerate(rs) if r['split']=='test' and r['run']==run]) for leg,run in [('replication',RUNS[1]),('pilot_sensitivity',RUNS[0])]}
        inventories.append(dict(cohort=cohort, fit_sources=len({rs[i]['source'] for i in train}), fit_rows=len(train), tests={leg:dict(rows=len(ii),sources=len({rs[i]['source'] for i in ii})) for leg,ii in tests.items()}, fusion_eligible=int(sum(available[i] for i in ids))))
        for a,b in [('bolt','timesfm'),('timesfm','bolt')]:
            grams = {'A_hidden':hidden[a][ids]@hidden[a][ids].T, 'B_hidden':hidden[b][ids]@hidden[b][ids].T,
                     'A_forecast':targets[a]['forecast'][ids]@targets[a]['forecast'][ids].T}
            banks,final = make_banks(xx,grams,rs,train)
            h = hidden[a][ids]
            v = np.random.default_rng(4951001).normal(size=h.shape[1]); v /= np.linalg.norm(v)
            rotated = h - 2*np.outer(h@v,v)
            rb,rf = make_banks(xx,dict(grams, A_hidden=rotated@rotated.T),rs,train)
            evaluate = np.arange(len(rs))
            fitted = {}
            for ep in ENDPOINTS:
                yy = targets[b][ep][ids]
                arms = ['input','A_hidden','input+A_hidden','B_hidden','input+B_hidden','input+A_forecast']
                fitted[ep] = {}
                for arm in arms:
                    prediction,selection = select_predict(banks,final,yy,rs,train,evaluate,arm,ep=='failure_flag')
                    fitted[ep][arm] = prediction
                    selections.append(dict(cohort=cohort,A=a,B=b,endpoint=ep,arm=arm,**selection))
                    predictions[f'{cohort}__{a}_to_{b}__{ep}__{arm}'] = prediction
                rp,rsel = select_predict(rb,rf,yy,rs,train,evaluate,'input+A_hidden',ep=='failure_flag')
                rotation_error = float(np.max(abs(rp-fitted[ep]['input+A_hidden'])))
                assert rotation_error < 1e-7, 'Actual-state probe rotation failed'
                rotations[a]['max_prediction_error'] = max(rotation_error,rotations[a].get('max_prediction_error',0))
                fitted[ep]['FIT_constant'] = np.repeat((weights(rs,train)@yy[train])[None,:],len(rs),axis=0)
            del rb,rf
            nulls = {ep:[] for ep in ENDPOINTS}
            for seed in range(19):
                perm = shuffled_indices(rs,train,4951000+seed)
                shuffled = {'A_hidden':grams['A_hidden'][np.ix_(perm,perm)]}
                nb,nf = make_banks(xx,shuffled,rs,train)
                for ep in ENDPOINTS:
                    yy = targets[b][ep][ids]
                    pp,_ = select_predict(nb,nf,yy,rs,train,evaluate,'input+A_hidden',ep=='failure_flag')
                    nulls[ep].append(pp)
            for leg,test in tests.items():
                if not len(test):
                    continue
                domains = ['ALL'] if cohort=='synthetic' else ['ALL']+sorted({r['domain'] for r in rs})
                for domain in domains:
                    ti = np.array([i for i in test if domain=='ALL' or rs[i]['domain']==domain])
                    for ep in ENDPOINTS:
                        yy = targets[b][ep][ids]
                        losses = {arm:np.mean((pp[ti]-yy[ti])**2,axis=1) for arm,pp in fitted[ep].items()}
                        base = losses['input']
                        gain = base-losses['input+A_hidden']
                        grouped = group_values(gain,rs,ti)
                        null_gains = [float(np.mean(list(group_values(base-np.mean((pp[ti]-yy[ti])**2,axis=1),rs,ti).values()))) for pp in nulls[ep]]
                        summary = interval(list(grouped.values()))
                        family = interval(list(grouped.values()),coverage=.9875)
                        arm_losses = {arm:interval(list(group_values(v,rs,ti).values())) for arm,v in losses.items()}
                        baseline = arm_losses['input']['mean']
                        same_gain = baseline-arm_losses['input+B_hidden']['mean']
                        tail = (1+sum(v>=summary['mean'] for v in null_gains))/20
                        promoted = (cohort=='synthetic' and leg=='replication' and ep in ('forecast','log_failure') and summary['mean']>=.05*baseline and family['low']>0 and tail==.05 and same_gain>0)
                        result = dict(cohort=cohort,leg=leg,domain=domain,A=a,B=b,endpoint=ep,losses=arm_losses,
                            gain=summary,primary_family_interval=family,relative_gain=summary['mean']/max(baseline,1e-14),
                            shuffled_gain_mean=float(np.mean(null_gains)),shuffled_gain_max=float(max(null_gains)),shuffled_tail_rank=tail,
                            shuffled_gains=null_gains,same_model_increment=same_gain,expand=promoted,
                            target_sd=float(np.std(yy[ti])),target_mean=float(np.mean(yy[ti])))
                        all_results.append(result)
                        for arm,v in losses.items():
                            source_losses.extend(dict(cohort=cohort,leg=leg,domain=domain,A=a,B=b,endpoint=ep,arm=arm,source=s,loss=value) for s,value in group_values(v,rs,ti).items())
            print(cohort,a,'to',b,'complete',round(time.monotonic()-started,1),'seconds',flush=True)
    write(out/'inventory.json',dict(cohorts=inventories,hidden_shapes=shapes,rows=rows,rotation=rotations))
    write(out/'selections.json',selections)
    write(out/'source_losses.json',source_losses)
    np.savez_compressed(out/'predictions.npz',**predictions)
    write(out/'results.json',all_results)
    execution = dict(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     os_threads=int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:'))),
                     pools=threadpool_info(),protocol_sha256=sha(out/'protocol.json'),code_hashes=protocol['code_hashes'])
    assert execution['peak_rss_bytes']<3*2**30 and execution['os_threads']==1
    write(out/'execution.json',execution)
    report(out,all_results,execution)


def report(out,results,execution):
    """Render all primary and pooled descriptive outcomes with negative evidence visible."""
    lines=['# Reuse-only comparability pilot 4951000', '',
        'Exploratory predictive evidence on previously studied data. FIT/TEST sources are disjoint; this is not sealed confirmation, causal mediation, unit correspondence, or a claim about raw hidden L2 units.', '',
        'Code, upstream hashes and choices were frozen in protocol.json before validation and scoring. validation.log records tests and planted regression controls. All input transforms, PCA, regularization and kernel choices use FIT only.', '',
        'Amendment 01 preserves the initial failed analysis.log: actual-state rotation exposed machine-precision ties between equivalent inner-CV candidates whose final refits differed. A fixed 1e-12 relative/absolute CV tie rule now chooses the first declared candidate. No TEST risk was inspected to choose this correction; original protocol and validation remain intact, and validation_v2 records the expanded regression suite.', '',
        '| Cohort | A → B | Endpoint | Input loss | A hidden loss | Input+A loss | Input+B loss | Input+A forecast | Gain [95% source CI] | Null tail rank | Expand |',
        '|---|---|---|---:|---:|---:|---:|---:|---|---:|---|']
    for r in results:
        if r['leg']!='replication' or r['domain']!='ALL':
            continue
        l,g=r['losses'],r['gain']
        lines.append(f"| {r['cohort']} | {r['A']} → {r['B']} | {r['endpoint']} | {l['input']['mean']:.5g} | {l['A_hidden']['mean']:.5g} | {l['input+A_hidden']['mean']:.5g} | {l['input+B_hidden']['mean']:.5g} | {l['input+A_forecast']['mean']:.5g} | {g['mean']:.5g} [{g['low']:.5g}, {g['high']:.5g}] | {r['shuffled_tail_rank']:.2f} | {r['expand']} |")
    promoted=[r for r in results if r['expand']]
    lines.extend(['',f"Frozen expansion screen: {len(promoted)} of four primary synthetic contrasts qualify. This screen uses 98.75% family intervals, >=5% loss reduction, gain above all 19 source shuffles, and same-model headroom. Null tail ranks are coarse diagnostics, not calibrated confirmatory p-values.", '',
        'Full results.json includes every real domain (only four TEST sources each), pilot TEST sensitivity, all source intervals, target variation and null draws. source_losses.json exposes dependence and outliers. Synthetic and natural real are never pooled. Injected real is an artificial-input challenge on the same real sources, not new independent evidence.', '',
        'Full-token flattening compares prediction from the same 128-step input extent without aligning neuron coordinates. Three dense orthogonal reflections preserve actual hidden Gram matrices (inventory.json). PCA/ridge is invariant to such rotations; arbitrary nonorthogonal reparameterizations can change its regularization metric. This does not establish meaningful neuron identities.', '',
        'Same-model B hidden is an empirical capacity comparator at a saved intermediate layer, not a mathematical ceiling. The decoder may use information outside that capture. Input+A forecast is a separate observable-only comparator requiring A inference, reused here from disk.', '',
        'Forecast and fusion-response losses are squared normalized observable errors; log_failure loss is squared error of log1p(normalized native forecast MSE); failure_flag uses Brier loss relative to last-value persistence. Zero fusion responses include declared tree abstentions. No native model inference was performed.', '',
        'Bootstrap intervals condition on trained probes, sampled datasets, and existing sources. They omit training instability, dataset-level uncertainty, and unknown pretraining overlap. Failure flag prevalence and low response variance can create trivial prediction; consult target_mean/target_sd and source losses.', '',
        f"Execution: {execution['seconds']:.1f} s, peak RSS {execution['peak_rss_bytes']/2**20:.1f} MiB, {execution['os_threads']} OS thread. Reproduce from tsfm_lens with the cudaPy Python: -m experiments.feature_search.comparability_pilot freeze/validate/analyze --out runs/<fresh_name>. Never overwrite this run."])
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    """Enforce one CPU affinity and a hard 3-GiB address-space limit before work."""
    resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','validate','analyze'])
    p.add_argument('--out',type=Path,default=ROOT/'runs/comparability_pilot4951000')
    a=p.parse_args()
    {'freeze':freeze,'validate':validate,'analyze':analyze}[a.command](a.out)


if __name__=='__main__':
    main()
