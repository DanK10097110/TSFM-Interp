"""Frozen two-arm response-aware loss PoC; existing TopKSAE, saved states only.

No architecture fork, native inference, parameter grid or TSAE reproduction.
Both arms execute identical full-state and paired branches, including backward.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SEED = 4965000


def read(path):
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomically persist finite JSON."""
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    """Hash an immutable input."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def response_loss(predicted, target, floor):
    """Normalize each entire [tokens, dimensions] difference, never tiny tokens."""
    energy = target.square().mean(dim=(-2, -1))
    return ((predicted-target).square().mean(dim=(-2, -1))/energy.clamp_min(floor)).mean()


def schedule(weights, token_count, steps=500, seed=SEED):
    """Independent identical full-token and per-direction whole-pair draws."""
    import numpy as np
    rng = np.random.default_rng(seed)
    tokens = rng.choice(len(weights)*token_count, (steps, 64), p=np.repeat(weights/token_count, token_count))
    pairs = rng.choice(len(weights), (steps, 2, 4), p=weights)
    return tokens, pairs


def gate(control, response):
    """Prespecified descriptive expansion gate; no parameter selection."""
    gains = {d: 1-response['old_response'][d]['mean_source_nmse']/control['old_response'][d]['mean_source_nmse']
             for d in ('sinusoid', 'noise')}
    focused_gain = control['focused']['mean_source_nmse']-response['focused']['mean_source_nmse']
    adequate = control['global']['nmse'] <= .05
    return dict(lambda0_adequate=adequate, lambda1_global_pass=response['global']['nmse'] <= .05,
                old_relative_gains=gains, focused_absolute_gain=focused_gain,
                passed=bool(adequate and response['global']['nmse'] <= .05 and min(gains.values()) >= .2 and focused_gain > 0))


def focused_pairs(signals):
    """Opposite phase center states: 0-vs-pi and pi/2-vs-3pi/2, both frequencies."""
    output = []
    for row in signals:
        if row['phase_id'] not in (0, 1):
            continue
        match = [r for r in signals if r['source']==row['source'] and r['frequency']==row['frequency']
                 and r['phase_id']==row['phase_id']+2]
        if len(match) != 1:
            raise ValueError('Incomplete focused opposite-phase pair')
        output.append(dict(source=row['source'], frequency=row['frequency'], quadrature=row['phase_id'],
                           plus=row['zero'], minus=match[0]['zero']))
    if len(output) != len(signals)//2:
        raise ValueError('Incomplete focused bank')
    return output


def prepare(out):
    """Freeze provenance and choices after tests, before any fitting."""
    base = Path('runs')
    old = base/'selective_replication4942000'
    focused = base/'response_surface_focused4954100'
    pca = base/'dictionary_pilot4951000/recovery_exit_monitor/fit_bolt_cut0_pca_k16_s4951000/dictionary.npz'
    files = [old/n for n in ('rows.json','bolt_states.npz','bolt_worker.json')]
    files += [focused/n for n in ('signals.json','sources.json','bolt_locked_hidden_0.npy','bolt_worker.json','execution.json')]
    files += [pca, Path(__file__), Path('experiments/feature_search/dictionary_core.py'),
              Path('tsfm_lens/sae/models.py'), Path('tests/test_dictionary_response_poc.py')]
    assert read(focused/'execution.json')['bolt']['valid']
    pairs = focused_pairs(read(focused/'signals.json'))
    old_sources = {r['source'] for r in read(old/'rows.json')}
    assert not old_sources.intersection(r['source'] for r in pairs)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', dict(seed=SEED, model='bolt', cut='0', width=512, k=16, steps=500,
        full_batch_tokens=64, pair_batch_per_direction=4, directions=['sinusoid','noise'], lambdas=[0,1],
        optimizer='Adam lr=.001 default betas/eps; unit decoder normalization after each step',
        initialization='same existing TopKSAE seeded generator; no warm start',
        weighting='uniform source, then case; uniform token for full-state; equal two direction means',
        scaling='FIT full-state vector center and one scalar RMS, shared both arms; no metadata target',
        loss='full normalized state MSE + lambda * mean(direction mean(pair full-token delta MSE / max(pair full-token delta energy, floor)))',
        floor='per direction max(1e-12, .01 * source-weighted FIT mean full-token delta energy), in common scaled coordinates',
        branches='both arms run full state and both signs of BOTH directions with autograd; lambda0 multiplies paired loss by zero',
        budgets=dict(threads=1, rss_bytes=2*2**30, total_worker_seconds=180, steps_per_arm=500,
                     full_tokens_per_arm=32000, paired_endpoint_tokens_per_arm=72000,
                     dense_forward_matmul_flops_per_step=4*208*512*512,
                     flop_scope='same 208 token forwards/step both arms; 3.25x original 64-token TopK step; backward/Adam/topk excluded from analytical FLOPs'),
        evaluation='all32 old TEST sources, source-equal global NMSE; response mean of sourcewise error-energy/target-energy ratios; no evaluation flooring, flag zero energy',
        focused='all16 fresh sources, opposite-phase LOCKED first-block CENTER pairs at periods24/19.2, two quadratures; four dependent cells/source, mean cell NMSE then mean sources; no local path/smoothness duplication',
        gate='lambda0 global NMSE<=.05 required for adequate control; lambda1 global<=.05 AND >=20% relative old response NMSE improvement EACH direction AND positive mean focused NMSE gain; descriptive only',
        old=str(old.resolve()), focused_dir=str(focused.resolve()), pca=str(pca.resolve()),
        files={str(p.resolve()):sha(p) for p in files},
        limits=['one new seed; old spectral TEST reused, not confirmatory','four old real TEST sources/domain;16 focused synthetic sources',
                'hidden response fidelity predicts model response, not true signal semantics or monosemanticity',
                'response-aware inspired, not TSAE replication; residual output, not FFN neurons',
                'no native authorization; no lambda grid; no selection/refitting on focused holdout']))
    write(out/'knowledge.json', dict(question='Can paired response loss improve weak hidden response fidelity without losing global reconstruction?',
        status='frozen_ready', exact_results={'protocol':'protocol.json'}, controls=['paired initialization/batches/budgets','frozen adaptive-support PCA k16'],
        limitations=read(out/'protocol.json')['limits'], next_step='Run only authorized two arms, then close or request separate native review.'))


def score_pairs(pair_values, artifact, metadata):
    """FP64 difference arithmetic, retaining sources and true response energy."""
    import numpy as np
    from experiments.feature_search import dictionary_core as core
    rows = []
    for h, meta in zip(pair_values, metadata):
        x = (h-artifact['center'])/float(artifact['scale'])
        z = core.encode(x, artifact)
        rebuilt = core.reconstruct(z, artifact).astype(float)*float(artifact['scale'])
        target = h[0].astype(float)-h[1].astype(float)
        error = rebuilt[0]-rebuilt[1]-target
        energy = float(np.mean(target**2))
        rows.append(dict(meta, error_energy=float(np.mean(error**2)), target_energy=energy,
                         nmse=None if energy==0 else float(np.mean(error**2)/energy),
                         mean_l0=float(np.mean(np.count_nonzero(z, axis=-1)))))
    return rows


def aggregate(rows, mode):
    """Old source ratio-of-energies vs focused equal dependent-cell means."""
    import numpy as np
    sources = []
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source']==source]
        if any(r['nmse'] is None for r in group):
            raise ValueError('Zero response energy: do not silently floor evaluation')
        nmse = (sum(r['error_energy'] for r in group)/sum(r['target_energy'] for r in group)
                if mode=='old' else float(np.mean([r['nmse'] for r in group])))
        sources.append(dict(source=source, nmse=nmse, dependent_cells=len(group)))
    return dict(mean_source_nmse=float(np.mean([r['nmse'] for r in sources])), n_sources=len(sources), sourcewise=sources, cells=rows)


def evaluate(artifact, full, oldpairs, rows, test, focused, focused_meta):
    """Evaluate frozen dictionaries once, no optimizer or selection."""
    import numpy as np
    from experiments.feature_search import dictionary_core as core
    x = (full[test]-artifact['center'])/float(artifact['scale'])
    z = core.encode(x, artifact)
    rec = core.reconstruct(z, artifact)
    weights = core.source_weights(np.repeat([rows[i]['source'] for i in test], full.shape[1]))
    health = core.health(x.reshape(-1,512), z.reshape(-1,512), rec.reshape(-1,512), weights)
    metadata = [dict(row=int(i), source=rows[i]['source'], domain=rows[i]['domain']) for i in test]
    return dict(global_metrics=health, old_response={d:aggregate(score_pairs(h[test],artifact,metadata),'old') for d,h in oldpairs.items()},
                focused=aggregate(score_pairs(focused,artifact,focused_meta),'focused'))


def worker(out):
    """Sequential immutable arm children; entire worker under external180s/2GiB cap."""
    import numpy as np
    import torch
    from experiments.feature_search import dictionary_core as core
    from tsfm_lens.sae.models import TopKSAE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    p = read(out/'protocol.json')
    for name, digest in p['files'].items():
        if sha(name)!=digest:
            raise ValueError('Frozen input changed: '+name)
    old = Path(p['old'])
    rows = read(old/'rows.json')
    fit, test = core.split_indices(rows)
    cut = read(old/'bolt_worker.json')['cuts'].index('0')
    with np.load(old/'bolt_states.npz') as a:
        full = np.stack([a[f'{i}_full'][cut] for i in range(len(rows))])
        pairs = {d:np.stack([a[f'{i}_{d}_0.1_locked'][cut] for i in range(len(rows))]) for d in p['directions']}
    assert full.shape[1:]==(9,512)
    weights = core.source_weights([rows[i]['source'] for i in fit])
    center, scale = core.fit_scaler(full[fit].reshape(-1,512), np.repeat(weights/9,9))
    x = torch.from_numpy(((full[fit]-center)/scale).reshape(-1,512))
    pair_data = {d:torch.from_numpy((h[fit]-center)/scale) for d,h in pairs.items()}
    floors = {d:max(1e-12,.01*float(weights @ (h[:,0]-h[:,1]).square().mean(dim=(-2,-1)).numpy())) for d,h in pair_data.items()}
    tokens, batches = schedule(weights,9)
    np.savez_compressed(out/'fit_schedule.npz',tokens=tokens,pairs=batches,center=center,scale=scale)
    focused_dir = Path(p['focused_dir'])
    meta = focused_pairs(read(focused_dir/'signals.json'))
    raw = np.load(focused_dir/'bolt_locked_hidden_0.npy',mmap_mode='r')
    focused = np.stack([raw[[r['plus'],r['minus']]] for r in meta])
    summaries = {}
    training_started = time.monotonic()
    for lam in p['lambdas']:
        dest = out/f'lambda{lam}'
        dest.mkdir(exist_ok=False)
        write(dest/'knowledge.json',dict(question='Paired response loss arm',status='running',exact_results={},controls=['frozen protocol'],limitations=p['limits'],next_step='Finish bounded arm'))
        model = TopKSAE(512,512,16,generator=torch.Generator().manual_seed(SEED))
        initial_hash = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        history = []
        start = time.monotonic()
        for step in range(p['steps']):
            value = x[tokens[step]]
            full_loss = (model(value)[0]-value).square().mean()
            response_terms = []
            for j,d in enumerate(p['directions']):
                h = pair_data[d][batches[step,j]]
                rebuilt = model(h.reshape(-1,512))[0].reshape(h.shape)
                response_terms.append(response_loss(rebuilt[:,0]-rebuilt[:,1],h[:,0]-h[:,1],floors[d]))
            response = torch.stack(response_terms).mean()
            loss = full_loss+lam*response
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_decoder_()
            history.append(dict(step=step+1,global_loss=float(full_loss.detach()),response_loss=float(response.detach())))
        fit_seconds = time.monotonic()-start
        artifact = {k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
        artifact.update(center=center,scale=np.array(scale),k=np.array(16))
        np.savez_compressed(dest/'dictionary.npz',**artifact)
        write(dest/'losses.json',history)
        result = evaluate(artifact,full,pairs,rows,test,focused,meta)
        result['global'] = result.pop('global_metrics')
        result.update(initial_hash=initial_hash,lambda_value=lam,fit_seconds=fit_seconds,steps=500,
                      trainable_parameters=sum(v.numel() for v in model.parameters()),floors=floors,
                      dictionary_sha256=sha(dest/'dictionary.npz'),schedule_sha256=sha(out/'fit_schedule.npz'),protocol_sha256=sha(out/'protocol.json'))
        write(dest/'results.json',result)
        summaries[str(lam)] = {k:result[k] for k in ('global','old_response','focused','initial_hash','fit_seconds')}
        write(dest/'knowledge.json',dict(question='Does paired response loss improve frozen held-out response fidelity?',status='complete',
            exact_results=dict(table='results.json',global_nmse=result['global']['nmse'],old_response={d:r['mean_source_nmse'] for d,r in result['old_response'].items()},
                               focused_nmse=result['focused']['mean_source_nmse'],fit_seconds=fit_seconds,steps=500),
            controls=['same initialization, schedule, full and paired branches','FIT-only scalars and floors','no native inference'],
            limitations=p['limits'],next_step='Compare fixed lambda0/lambda1 against frozen gate; no grid.'))
        print(json.dumps(dict(arm=lam,global_nmse=result['global']['nmse'],response={d:r['mean_source_nmse'] for d,r in result['old_response'].items()},focused=result['focused']['mean_source_nmse'],fit_seconds=fit_seconds)),flush=True)
    baseline = evaluate(dict(np.load(p['pca'])),full,pairs,rows,test,focused,meta)
    baseline['global'] = baseline.pop('global_metrics')
    write(out/'pca_results.json',baseline)
    assert summaries['0']['initial_hash']==summaries['1']['initial_hash']
    verdict = gate(summaries['0'],summaries['1'])
    compact = {name:dict(global_nmse=r['global']['nmse'],old_response={d:v['mean_source_nmse'] for d,v in r['old_response'].items()},
                        focused_nmse=r['focused']['mean_source_nmse'],mean_l0=r['global']['mean_l0'],dead_fraction=r['global']['dead_fraction'])
               for name,r in [('lambda0',summaries['0']),('lambda1',summaries['1']),('frozen_pca',baseline)]}
    write(out/'comparison.json',dict(table=compact,gate=verdict,postload_seconds=time.monotonic()-training_started))
    write(out/'knowledge.json',dict(question='Does paired response reconstruction justify a separately authorized native preservation test?',
        status='complete',slot1='released',native_launched=False,exact_results=dict(comparison='comparison.json',gate=verdict,table=compact),
        controls=['same initial hash and full/pair schedule, matched computed branches','sourceweighted FIT-only scalars/floors','all32 old TEST sources and16 fresh focused sources','frozen PCA is adaptive-support full basis, not fixed rank'],
        limitations=p['limits'],next_step='Request separate native review; no launch authorized.' if verdict['passed'] else 'Gate failed: close PoC, retain tradeoff/negative, no lambda grid or native launch.'))


def launch(out):
    """Stdlib monitor counts parent+worker RSS and180s total including loading/eval."""
    p = read(out/'protocol.json')
    start,peak,reason = time.monotonic(),0,None
    env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',DICTIONARY_RESPONSE_GRANTED='1')
    with (out/'worker.log').open('w') as log:
        child = subprocess.Popen([sys.executable,'-m','experiments.feature_search.dictionary_response_poc','worker','--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            rss = 0
            for pid in (os.getpid(),child.pid):
                try:
                    rss += int(next((line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmRSS:')),'0'))*1024
                except FileNotFoundError:
                    pass
            peak = max(peak,rss)
            if peak>p['budgets']['rss_bytes'] or time.monotonic()-start>180:
                reason='resource cap'
                os.killpg(child.pid,9)
                child.wait()
                break
            time.sleep(.05)
    record=dict(returncode=child.returncode,seconds=time.monotonic()-start,peak_rss=peak,reason=reason,threads=1)
    write(out/'execution.json',record)
    if child.returncode:
        for dest in [out]+[d for d in out.glob('lambda*') if d.is_dir() and read(d/'knowledge.json')['status']=='running']:
            write(dest/'knowledge.json',dict(question='Bounded paired-loss PoC',status='failed',slot1='released',exact_results=record,
                controls=['external monitor'],limitations=['incomplete; no scientific verdict'],next_step='Inspect logs; no automatic retry or native.'))
    print(json.dumps(record),flush=True)


def main():
    """Prepare separately from explicitly authorized fit; native is absent."""
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','worker'])
    parser.add_argument('--out',type=Path,default=Path('runs/dictionary_response4965000'))
    parser.add_argument('--slot-granted',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.out)
    elif args.command=='run':
        if not args.slot_granted:
            parser.error('Explicit fit slot grant required')
        launch(args.out)
    else:
        if os.environ.get('DICTIONARY_RESPONSE_GRANTED')!='1':
            parser.error('Use bounded supervisor')
        worker(args.out)


if __name__=='__main__':
    main()
