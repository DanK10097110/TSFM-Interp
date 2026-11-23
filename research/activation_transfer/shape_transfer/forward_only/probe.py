"""Measure generic CAL directions and natural test atom responses using signed forward calls."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
RANK = 24


def source_steps(codes, decoder_norm, radius):
    return np.minimum(.002 * radius, .25 * codes * decoder_norm[None, :])


def central(adapter, layer, selector, x, tokens, axes, active, steps, batch):
    """Return dy per hidden L2 unit, the observed signed half difference, and its exact size."""
    rows, columns = np.nonzero(active)
    out = np.zeros((*active.shape, 8), dtype='f4')
    for start in range(0, len(rows), batch):
        i, j = rows[start:start+batch], columns[start:start+batch]
        radius = steps[i, j]
        assert np.all(radius > 0)
        delta = (axes[j] * radius[:,None]).astype('f4')
        high, low = [change_forecasts(adapter, layer, x[i], tokens[i], sign*delta, selector, batch)
                     for sign in (1., -1.)]
        out[i,j] = (high-low)/2.
        if (start // batch + 1) % 96 == 0:
            print(layer, min(start+batch,len(rows)), 'of', len(rows), 'signed probe pairs', flush=True)
    derivative = np.divide(out, steps[:,:,None], out=np.zeros_like(out), where=active[:,:,None])
    return derivative, out


def collect(name, cache, out, batch):
    clock = time.perf_counter()
    with np.load(cache/'data.npz') as f:
        ids, split, x = f['ids'], f['split'], f['x']
    tokens, pos, token_ids = read_materials(cache,name)
    np.testing.assert_array_equal(ids,token_ids)
    training = tokens[split=='train'].mean(axis=1).astype(float)
    radius = float(np.median(np.linalg.norm(training,axis=1)))
    basis = np.linalg.svd(training-training.mean(axis=0),full_matrices=False)[2][:RANK]
    rng = np.random.default_rng({'bolt':2026112088, 'timesfm':2026112089}[name])
    random_axes = np.linalg.qr(rng.normal(size=(training.shape[1], RANK)))[0].T
    with np.load(REPLICATION/'results'/(name+'_seed0.npz')) as f:
        np.testing.assert_array_equal(ids[split!='train'],f['ids'])
        np.testing.assert_array_equal(split[split!='train'],f['split'])
        atoms, codes = f['atoms'], f['source_codes'][f['split']!='cal']
    decoder = load_sae_checkpoint(str(dictionary_path(name))).W_dec.detach().numpy()[atoms].astype(float)
    decoder_norm = np.linalg.norm(decoder,axis=1)
    decoder_unit = decoder/decoder_norm[:,None]
    axes = np.vstack((basis,random_axes,decoder_unit))
    np.testing.assert_allclose(np.linalg.norm(axes,axis=1),1.,atol=1e-10)
    adapter = adapter_for(name)
    layer = MODELS[name][-1]
    selector = last_window_slice(adapter,pos)
    cal = split=='cal'
    test = (split!='train') & ~cal
    assert [cal.sum(), test.sum()] == [96, 320]
    active_cal = np.ones((cal.sum(),len(axes)),bool)
    cal_steps = np.full(active_cal.shape,.002*radius)
    cal_der, _ = central(adapter,layer,selector,x[cal],tokens[cal],axes,active_cal,cal_steps,batch)
    cal_seconds = time.perf_counter()-clock
    print(name,'CAL complete',round(cal_seconds,1),flush=True)
    active = codes>0
    steps = source_steps(codes,decoder_norm,radius)
    test_der, half_difference = central(adapter,layer,selector,x[test],tokens[test],decoder_unit,active,steps,batch)
    test_seconds = time.perf_counter()-clock-cal_seconds
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(name+'_probes.npz'),model=name,site=layer,ids=ids[test],split=split[test],
        cal_ids=ids[cal],train_ids=ids[split=='train'],positions=pos,atoms=atoms,codes=codes,decoder_norm=decoder_norm,
        radius=radius,cal_step=.002*radius,pc_basis=basis,random_basis=random_axes,
        pc_jac=cal_der[:,:RANK].transpose(0,2,1),random_jac=cal_der[:,RANK:2*RANK].transpose(0,2,1),
        cal_decoder_der=cal_der[:,2*RANK:]*decoder_norm[None,:,None],
        test_decoder_der=test_der*decoder_norm[None,:,None],test_steps=steps,half_difference=half_difference,
        cal_seconds=cal_seconds,test_seconds=test_seconds)
    print(name,'test complete',active.sum(),'active probes',round(test_seconds,1),'sec',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=MODELS)
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--batch',type=int,default=16)
    a=p.parse_args()
    torch.set_num_threads(1)
    collect(a.model,a.cache,a.out,a.batch)
