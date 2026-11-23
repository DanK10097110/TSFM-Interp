"""Train six paired-view TopK dictionaries on a fixed history split of MOMENT middle states."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.weather.new_dates import run as weather
from research.activation_transfer.moment.token_paths.geometry import CACHE as GEOMETRY
from tsfm_lens.sae.crosscoder import CrosscoderSAE

HERE = Path(__file__).resolve().parent
OUT = HERE / 'dictionary_results'
CACHE = Path(os.environ.get('TSFM_MOMENT_CROSSCODER', '/tmp/tsfm_moment_crosscoder'))
SEEDS = (22112401, 22112402)
METHODS = ('joint', 'masked', 'shuffled')
SIZES = ('small', 'large')
WIDTH = 1024
TOPK = 32
TRAIN = 768
POSITIONS = 60
BATCH = 256
STEPS = 1000


def paired_m4():
    ids = None
    arrays = []
    for size in SIZES:
        with np.load(GEOMETRY/size/'train.npz') as capture:
            current = capture['ids']
            if ids is None:
                ids = current
            else:
                np.testing.assert_array_equal(ids, current)
        x = np.load(GEOMETRY/size/'train_observed.npy', mmap_mode='r')
        assert x.shape[:2] == (1024, POSITIONS)
        arrays.append(x)
    assert len(np.unique(ids)) == 1024
    return ids, arrays


def paired_noaa():
    stations, dates, x = weather.panel()
    with np.load(weather.WORK/'physical_ramp_confirmation/small_base.npz') as src, np.load(weather.OUT/'ordinary.npz') as tgt:
        digest = sha256(x.tobytes()).hexdigest()
        assert str(src['sig']) == digest == str(tgt['input_sha256'])
        np.testing.assert_array_equal(stations,tgt['ids'])
        np.testing.assert_array_equal(dates,tgt['dates'])
        return stations, dates, [src['arr'],tgt['last']]


def normalizers(arrays):
    means, scales = [], []
    for x in arrays:
        train = x[:TRAIN].reshape(-1, x.shape[-1])
        mean = train.mean(axis=0, dtype=np.float64)
        energy = 0.
        for i in range(0,len(train),1024):
            energy += np.square(train[i:i+1024]-mean).sum()
        means.append(torch.from_numpy(mean.astype('f4')))
        scales.append(float(np.sqrt(energy/train.size)))
    return means, scales


def encode_views(model, means, small, large=None):
    """Accept raw states; with only small present the large contribution is identically zero."""
    small = torch.as_tensor(small, dtype=model.W_enc[0].dtype, device=model.W_enc[0].device)-means[0]
    if large is None:
        return model.sparsify(model.pre_activations([small]))
    large = torch.as_tensor(large, dtype=small.dtype, device=small.device)-means[1]
    return model.encode([small, large])


def match_indices(method):
    # A single derangement, common to both seeds. It never permutes patch position.
    groups = np.arange(TRAIN)
    if method == 'shuffled':
        cycle = np.random.default_rng(22112406).permutation(TRAIN)
        groups[cycle] = np.roll(cycle,1)
        assert np.all(groups != np.arange(TRAIN))
    return groups


def train_dict(arrays,method,seed,means,scales):
    dims = [x.shape[-1] for x in arrays]
    model = CrosscoderSAE(dims, WIDTH, TOPK, source_scale=scales, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    groups = match_indices(method)
    draw = np.random.default_rng(seed+100)
    single = np.zeros(STEPS, dtype=bool)
    if method != 'joint':
        single[np.random.default_rng(seed+200).permutation(STEPS)[:STEPS//2]] = True
    history, running = [], np.zeros(2)
    order = np.empty(0, dtype=int)
    for step in range(STEPS):
        if len(order) < BATCH:
            order = np.r_[order,draw.permutation(TRAIN*POSITIONS)]
        group, pos = np.divmod(order[:BATCH],POSITIONS)
        order = order[BATCH:]
        values = [torch.from_numpy(arrays[0][group,pos]), torch.from_numpy(arrays[1][groups[group],pos])]
        code = encode_views(model,means,values[0],None if single[step] else values[1])
        recon = model.decode(code)
        losses = [((hat-(x-mu))/s).square().mean() for hat,x,mu,s in zip(recon,values,means,scales)]
        optimizer.zero_grad(set_to_none=True)
        loss = losses[0]+losses[1]
        if not bool(torch.isfinite(loss)):
            raise ValueError(f'Nonfinite loss at {method}, {seed}, step {step}')
        loss.backward()
        optimizer.step()
        model.normalize_decoder_()
        running += [v.item() for v in losses]
        if (step+1)%250 == 0:
            history.append(dict(through=step+1, normalized_mse=(running/250).tolist()))
            print(method,seed,history[-1],flush=True)
            running.fill(0)
    return model.eval(), groups, int(single.sum()), history


@torch.inference_mode()
def diagnostics(model,means,arrays,last):
    val = [x[TRAIN:,-1] if last else x[TRAIN:].reshape(-1,x.shape[-1]) for x in arrays]
    y = np.zeros(2)
    residual = {read:np.zeros(2) for read in ('source','both')}
    support_overlap = 0.
    for i in range(0,len(val[0]),BATCH):
        xs = [torch.from_numpy(x[i:i+BATCH].copy()) for x in val]
        truth = [v-mu for v,mu in zip(xs,means)]
        y += [v.double().square().sum().item() for v in truth]
        source, both = encode_views(model,means,xs[0]), encode_views(model,means,*xs)
        support_overlap += (((source>0)&(both>0)).sum(1)/((source>0)|(both>0)).sum(1).clamp_min(1)).sum().item()
        for read,z in [('source',source),('both',both)]:
            residual[read] += [(r-v).double().square().sum().item() for r,v in zip(model.decode(z),truth)]
    return dict(rows=len(val[0]),source_r2=dict(zip(SIZES,(1-residual['source']/y).tolist())),
                both_r2=dict(zip(SIZES,(1-residual['both']/y).tolist())),mean_support_jaccard=support_overlap/len(val[0]))


@torch.inference_mode()
def code_archive(model,means,states):
    parts = {read:[] for read in ('source','both')}
    for i in range(0,len(states[0]),BATCH):
        small,large = [torch.from_numpy(x[i:i+BATCH].copy()) for x in states]
        parts['source'].append(encode_views(model,means,small).numpy())
        parts['both'].append(encode_views(model,means,small,large).numpy())
    return {k:np.concatenate(v) for k,v in parts.items()}


def load_artifact(method,seed):
    f = torch.load(CACHE/f'{method}_{seed}.pt', map_location='cpu', weights_only=True)
    model = CrosscoderSAE(f['dims'],WIDTH,TOPK,source_scale=f['scales'])
    model.load_state_dict(f['state'])
    return model.eval(), f['means']


def run(method,seed):
    OUT.mkdir(exist_ok=True,parents=True)
    CACHE.mkdir(exist_ok=True,parents=True)
    ids,arrays = paired_m4()
    means,scales = normalizers(arrays)
    model,groups,single_batches,history = train_dict(arrays,method,seed,means,scales)
    stem = f'{method}_{seed}'
    state = dict(state=model.state_dict(),dims=[x.shape[-1] for x in arrays],means=means,scales=scales)
    tmp = CACHE/f'{stem}.partial.pt'
    torch.save(state,tmp)
    tmp.replace(CACHE/f'{stem}.pt')
    report = dict(method=method,seed=seed,patches=TRAIN*POSITIONS,train_histories=TRAIN,validation_histories=len(ids)-TRAIN,
                  train_ids_sha256=sha256(np.ascontiguousarray(ids[:TRAIN]).tobytes()).hexdigest(),
                  train_permutation=groups.tolist() if method=='shuffled' else None,
                  features=WIDTH,top_k=TOPK,steps=STEPS,batch=BATCH,small_only_batches=single_batches,centered_train_rms=scales,
                  loss_history=history,validation_all=diagnostics(model,means,arrays,False),
                  validation_last=diagnostics(model,means,arrays,True))
    stations,dates,noaa = paired_noaa()
    output = dict(m4_ids=ids,station_ids=stations,dates=dates)
    for name,z in [('m4',[x[:,-1] for x in arrays]),('noaa',noaa)]:
        for read,code in code_archive(model,means,z).items():
            output[name+'_'+read] = code
    for i,size in enumerate(SIZES):
        output['decoder_'+size] = (model.W_dec[i].detach()*scales[i]).numpy()
    tmp = CACHE/f'{stem}.partial.npz'
    np.savez_compressed(tmp,**output)
    tmp.replace(CACHE/f'{stem}.npz')
    (OUT/f'{stem}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('DONE',stem,report['validation_last'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('method',choices=METHODS);p.add_argument('seed',type=int,choices=SEEDS)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    run(a.method,a.seed)
