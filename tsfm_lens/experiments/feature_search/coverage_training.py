"""Parked ID7: matched tiny models trained with/without withheld frequency pairs.

Both arms see exactly the same single components and component exposure counts;
only co-occurrence changes. Two architectures and three matched initialization
seeds. No pretrained-model training data or architecture attribution is claimed.
"""
from __future__ import annotations
import argparse
import json
import time
import resource
from pathlib import Path
import numpy as np

FREQUENCIES = (3, 5, 9, 17)
CYCLES = (((0, 1), (1, 2), (2, 3), (0, 3)),
          ((0, 1), (1, 3), (2, 3), (0, 2)),
          ((0, 2), (1, 2), (1, 3), (0, 3)))


def dataset(seed, blocks, covered):
    """Every latent block exposes each frequency once alone and twice in pairs."""
    rng = np.random.default_rng(seed); t = np.arange(-128, 32)
    values, labels = [], []
    for block in range(blocks):
        amplitudes = rng.uniform(.4, 1.6, 4); phases = rng.uniform(-np.pi, np.pi, 4)
        components = amplitudes[:, None]*np.sin(2*np.pi*np.asarray(FREQUENCIES)[:, None]*t/128+phases[:, None])
        noise = rng.normal(0, .03, (8, len(t)))
        pairs = CYCLES[block % 3] if covered else CYCLES[0]
        groups = [(i,) for i in range(4)]+list(pairs)
        for index, group in enumerate(groups):
            values.append(components[list(group)].sum(0)+noise[index])
            labels.append(dict(block=block, components=list(group)))
    value = np.asarray(values, dtype=np.float32)
    return value[:, :128], value[:, 128:], labels


def evaluation(seed, blocks=128):
    """New generating blocks, with singles, seen pairs and the two withheld pairs."""
    rng = np.random.default_rng(seed); t = np.arange(-128, 32)
    xs, ys, labels = [], [], []
    groups = [(i,) for i in range(4)]+list(CYCLES[0])+[(0, 2), (1, 3)]
    for block in range(blocks):
        amp = rng.uniform(.4, 1.6, 4); phase = rng.uniform(-np.pi, np.pi, 4)
        comp = amp[:, None]*np.sin(2*np.pi*np.asarray(FREQUENCIES)[:, None]*t/128+phase[:, None])
        for g in groups:
            value = comp[list(g)].sum(0)+rng.normal(0, .03, len(t))
            xs.append(value[:128]); ys.append(value[128:]); labels.append(dict(block=block, group='single' if len(g)==1 else 'heldout' if g in [(0, 2), (1, 3)] else 'seen'))
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), labels


def make_model(architecture, seed):
    """Same initialization within paired coverage arms; 32-step prediction in source units."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            if architecture == 'mlp': self.net = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 32))
            else:
                self.embed = nn.Linear(16, 32)
                self.position = nn.Parameter(torch.randn(1, 8, 32)*.02)
                self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, dropout=0., batch_first=True), 2, enable_nested_tensor=False)
                self.head = nn.Linear(32, 32)
        def forward(self, x):
            mu = x.mean(-1, keepdim=True); sd = x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            z = (x-mu)/sd
            if architecture == 'mlp': y = self.net(z)
            else: y = self.head(self.encoder(self.embed(z.reshape(-1, 8, 16))+self.position)[:, -1])
            return y*sd+mu
    return Forecaster()


def run(out, steps=400):
    """Fixed budget; preserve all seeds, fit diagnostics and exact exposure matching."""
    import torch
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = dict(steps=steps, batch=64, learning_rate=.002, seeds=[49371,49372,49373], architectures=['mlp','transformer'],
                    train_blocks=96, train_rows=768, test_blocks=128, normalization='per-history native-like mean/SD',
                    pairs_withheld=[[0,2],[1,3]], matched='single histories, per-component latent realizations/exposure counts, rows, initialization, optimization budget',
                    scope='Small controlled training study; no attribution of pretrained TSFM failures', cpu_threads=1,
                    cap_seconds=600, cap_rss=2*2**30)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    train = {arm: dataset(493700,96,arm=='covered') for arm in ['withheld','covered']}
    ex, ey, labels = evaluation(493800)
    results = []; predictions = {}; loss_rows = []
    for architecture in protocol['architectures']:
        for seed in protocol['seeds']:
            for arm in ['withheld','covered']:
                x,y,meta = train[arm]; model = make_model(architecture,seed)
                optimizer = torch.optim.Adam(model.parameters(), lr=.002)
                tx,ty = torch.tensor(x),torch.tensor(y); generator = torch.Generator().manual_seed(seed+1)
                trace=[]
                for step in range(steps):
                    ix = torch.randint(len(tx),(64,),generator=generator)
                    pred = model(tx[ix]); loss = ((pred-ty[ix])**2).mean()
                    optimizer.zero_grad();loss.backward();optimizer.step();trace.append(float(loss.detach()))
                    if time.monotonic()-started > 600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 2*2**30: raise RuntimeError('Training resource cap exceeded')
                model.eval()
                with torch.no_grad(): pred = model(torch.tensor(ex)).numpy(); fitted = model(tx).numpy()
                scale = np.var(ex,axis=1); errors = np.mean((pred-ey)**2,axis=1)/scale
                summary = {group:float(np.mean(errors[[i for i,r in enumerate(labels) if r['group']==group]])) for group in ['single','seen','heldout']}
                key=f'{architecture}_{seed}_{arm}'; predictions[key]=pred; loss_rows.append(errors)
                results.append(dict(architecture=architecture,seed=seed,arm=arm,parameters=sum(p.numel() for p in model.parameters()),
                                    error=summary,training_mse=float(np.mean((fitted-y)**2)),final_loss_mean=float(np.mean(trace[-50:]))))
                torch.save(model.state_dict(),out/f'{key}.pt')
                print(key,summary,flush=True)
    from sklearn.linear_model import Ridge
    ridge={}
    for arm,(x,y,_) in train.items():
        f=Ridge(alpha=.01).fit(x,y);pred=f.predict(ex);e=np.mean((pred-ey)**2,axis=1)/np.var(ex,axis=1)
        ridge[arm]={group:float(np.mean(e[[i for i,r in enumerate(labels) if r['group']==group]])) for group in ['single','seen','heldout']}
    np.savez_compressed(out/'predictions.npz',test_history=ex,test_future=ey,errors=np.asarray(loss_rows),**predictions)
    (out/'test_labels.json').write_text(json.dumps(labels),encoding='utf-8')
    output=dict(results=results,ridge=ridge,seconds=time.monotonic()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                evidence='Exploratory paired training seeds; test blocks are not independent training runs')
    (out/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
