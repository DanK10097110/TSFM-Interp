"""Physically test the generic direction built only from earlier original-panel origins."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.physical_ramp.study import hashed, match_norm
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.run import ARMS, OLD, OUT as NEW, panel

OUT=Path(__file__).resolve().parent/'results'

def calendar_mask(source_dates, target_dates):
    earlier=np.asarray(source_dates,dtype='datetime64[D]')
    requested=np.asarray(target_dates,dtype='datetime64[D]')
    return earlier[None,:] < requested[:,None]

def run(out,batch):
    x,ids,station,origin,sites,dates=panel()
    with np.load(OLD/'timesfm_native.npz') as source:
        history=source['ramp_mean']
        historical_ids=source['ids']
    original_dates=np.array([row.rsplit('/',1)[1] for row in historical_ids])
    original_stations=np.array([row.split('/')[1] for row in historical_ids])
    assert len(set(original_stations))==46 and set(original_stations).isdisjoint(sites)
    allowed=calendar_mask(original_dates,dates)
    available=np.flatnonzero(allowed.any(axis=1))
    np.testing.assert_equal(available,np.arange(1,11))
    counts=allowed[available].sum(axis=1)
    np.testing.assert_equal(counts,[46,46,46,92,92,92,138,138,138,184])
    with np.load(OLD/'timesfm_map.npz') as fit:
        matrix=fit['linear'];map_hash=str(fit['linear_sha'])
        assert hashed(matrix)==map_hash
        target_pos=fit['target_pos']
    with np.load(NEW/'transfer.npz') as archive:
        assert map_hash==str(archive['map_sha']) and hashed(x)==str(archive['raw_input_sha'])
        np.testing.assert_equal(ids,archive['ids'])
        keep=np.isin(origin,available)
        assert keep.sum()==350
        dose=archive['budget'][keep]
        records={k:archive[k][keep] for k in (*ARMS,'truth','whole')}
    directions=np.stack([history[allowed[j]].mean(axis=0)@matrix for j in available])
    writes=match_norm(directions[origin[keep]-1],dose)
    np.testing.assert_allclose(np.linalg.norm(writes,axis=1),dose,atol=1e-10)
    with np.load(NEW/'bolt_native.npz') as native:
        np.testing.assert_equal(native['ids'],ids)
        np.testing.assert_equal(native['pos'],target_pos)
        base=native['base'][keep];clean=native['window'][keep]
    receiver=adapter_for('bolt');site=MODELS['bolt'][-1]
    result=forecast16(receiver,site,x[keep],clean,writes.astype('f4'),last_window_slice(receiver,target_pos),batch)
    records['past_only']=result.astype(float)-base
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'cases.npz',**records,ids=ids[keep],station=station[keep],date=origin[keep]-1,dates=dates[available],
                        old_ids=historical_ids,admissible=allowed[available],past_directions=directions,
                        write=writes,dose=dose,case_indices=np.flatnonzero(keep),map_sha=map_hash,context_sha=hashed(x[keep]))
    print('350 physical calendar-strict writes complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=24)
    args=p.parse_args();torch.set_num_threads(1);run(args.out,args.batch)
