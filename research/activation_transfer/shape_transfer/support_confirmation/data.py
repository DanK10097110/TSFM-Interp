"""Freeze daily source identities and measure ordinary eight-lead derivatives at the physical edit sites."""
import argparse
from hashlib import file_digest,sha256
import json
from pathlib import Path
import re

import numpy as np
import torch
import pyarrow as pa

from research.activation_transfer.atoms.materials import CACHE as ATOMS,MODELS,adapter_for,contexts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as REP,HUB_REVISION
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.dictionary_prior.refit.confirmation import TMP as GATE
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as EXTRA,exclusions
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.rolling.data import CACHE as ROLL
from research.activation_transfer.shape_transfer.continuation.probe import SCRATCH as CAL
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.config import ModelConfig,DataConfig
from tsfm_lens.models import build_adapter
from research.activation_transfer.boundary.capture import HUB

HERE=Path(__file__).resolve().parent
SUITE=HERE.parents[1];ROOT=SUITE.parents[1]
OUT=HERE/'results'
CACHE=Path('/tmp/tsfm_support_confirmation_20261122')
SEED=20261122011
SOURCE=Path.home()/'.cache/huggingface/hub/datasets--Salesforce--GiftEval/snapshots'/HUB_REVISION/'m4_daily/data-00000-of-00001.arrow'


def exclusion_catalog():
    previous,details=exclusions()
    by_source={'Earlier daily exclusion registries and named study memberships':previous['daily']}
    def add(label,records):
        by_source[label]=set(map(str,records))
    seen=set()
    for name,loc in [('atom train/cal/test',ATOMS),('replication train/cal/test',REP),('additional',EXTRA),('blind',BLIND),('rolling',ROLL),('gate confirmation',GATE)]:
        with np.load(loc/'data.npz') as d:
            day=[str(key).split('/')[-1] for key in d['ids'] if str(key).split('/')[-2] in ('daily','m4_daily')]
            add(name,day)
            seen.update(sha256(x.tobytes()).digest() for x in d['x'])
    fixed=json.loads((SUITE/'boundary/fixed.json').read_text(encoding='utf-8'))['datasets']
    for label in ('reserve','m4'):add('boundary plain IDs '+label,fixed[label]['ids'])
    benchmark=ROOT/'benchmark_v3/external_real/corpus.jsonl'
    records=[json.loads(line)['provenance']['generator_params'] for line in benchmark.read_text(encoding='utf-8').splitlines()]
    add(str(benchmark.relative_to(ROOT)),[r['item_id'] for r in records if r['dataset']=='m4_daily'])
    for path in sorted(SUITE.rglob('*.json')):
        if HERE in path.parents:continue
        ids=set(re.findall(rb'(?:m4_)?daily/(\d+)(?=[/"\\,])',path.read_bytes()))
        if ids:add(str(path.relative_to(ROOT)),[i.decode() for i in ids])
    return by_source,seen,details


def prepare(cache,out):
    by_source,seen,details=exclusion_catalog()
    manifest=out/'manifest.json'; prior=set.union(*by_source.values())
    if manifest.exists():
        frozen=json.loads(manifest.read_text(encoding='utf-8'))
        excluded=set(frozen['excluded_daily_ids']);n=len(frozen['items'])
        # A later follow-up might deliberately reuse this panel. Recreate from the locked exclusion list.
    else:
        excluded=prior;n=160
    original=seen.copy()
    try:items,x=contexts(SOURCE,n,SEED,seen,excluded)
    except ValueError as exc:
        match=re.fullmatch(r'Only (\d+) eligible in .+',str(exc))
        if manifest.exists() or not match or int(match[1])<1:raise
        n=int(match[1]);items,x=contexts(SOURCE,n,SEED,original.copy(),excluded)
    assert not set(items)&excluded
    assert not any(sha256(a.tobytes()).digest() in original for a in x)
    assert len(set(items))==len(x)
    ids=np.array(['daily/'+item for item in items])
    digest=sha256(x.tobytes()).hexdigest()
    if manifest.exists():
        np.testing.assert_array_equal(items,frozen['items'])
        assert digest==frozen['input_sha256']
    else:
        with SOURCE.open('rb') as f:source_digest=file_digest(f,'sha256').hexdigest()
        snapshot={name:dict(n=len(group),intersection=sorted(set(items)&group)) for name,group in by_source.items()}
        assert not any(g['intersection'] for g in snapshot.values())
        frozen=dict(items=items,ids=ids.tolist(),excluded_daily_ids=sorted(excluded),seed=SEED,
                    n_requested=160,n=len(x),source=str(SOURCE),source_sha256=source_digest,
                    input_sha256=digest,audited_source_ids=snapshot,baseline_catalog=details,
                    historical_distinct_window_hashes=len(original))
        out.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps(frozen,indent=2)+'\n',encoding='utf-8')
    cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache/'data.npz',x=x,ids=ids,split=np.full(len(x),'daily'))
    print('locked',len(x),'of 160',len(excluded),'excluded IDs',digest,flush=True)


def prepare_short(cache,out):
    long=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    excluded=set(long['excluded_daily_ids'])|set(long['items'])
    meta_path=out/'short_manifest.json'
    with pa.memory_map(str(SOURCE)) as stream:
        t=pa.ipc.open_stream(stream).read_all()
    result=[];ids=[];seen=set()
    for row in np.random.default_rng(20261122013).permutation(len(t)):
        key=str(t['item_id'][int(row)].as_py())
        if key in excluded:continue
        raw=t['target'][int(row)].values
        if len(raw)<264:continue
        obs=raw.slice(len(raw)-264,256).to_numpy(zero_copy_only=False).astype(float)
        if not np.isfinite(obs).all() or obs.std()<=1e-8:continue
        wave=((obs-obs.mean())/obs.std()).astype('f4')
        sig=sha256(wave.tobytes()).digest()
        if sig in seen:continue
        seen.add(sig);ids.append(key);result.append(wave)
        if len(ids)==160:break
    assert len(ids)==160 and len(set(ids))==160 and set(ids).isdisjoint(excluded)
    x=np.stack(result);keys=['daily/'+i for i in ids]
    doc=dict(items=ids,ids=keys,n=len(ids),seed=20261122013,context=256,source=str(SOURCE),
             source_sha256=long['source_sha256'],input_sha256=sha256(x.tobytes()).hexdigest(),
             exclude_manifest='manifest.json',original_length_intersection=[],earlier_source_intersection=[])
    if meta_path.exists():assert doc==json.loads(meta_path.read_text(encoding='utf-8'))
    else:meta_path.write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
    cache.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(cache/'data.npz',x=x,ids=keys,split=np.full(len(x),'daily'))
    print('short panel',len(ids),doc['input_sha256'],flush=True)


def adapter_for_length(model,length):
    if length==480:return adapter_for(model)
    name,kind,repo,revision,_=MODELS[model]
    path=HUB/('models--'+repo.replace('/','--'))/'snapshots'/revision
    adapter=build_adapter(ModelConfig(name=name,adapter=kind,checkpoint=str(path),batch_size=16),
                          DataConfig(context_len=length,horizon=8),torch.device('cpu'),torch.float32)
    adapter.ensure_loaded()
    return adapter


def probe(model,cache,batch):
    with np.load(cache/'data.npz') as d:
        x=d['x'];ids=d['ids']
    size=x.shape[1];enc=adapter_for_length(model,size);site=MODELS[model][-1]
    spans=enc.token_time_spans()
    pos=np.flatnonzero((spans[:,0]>=size-32)&(spans[:,1]<=size))
    assert spans[pos[0],0]==size-32 and spans[pos[-1],1]==size
    np.testing.assert_array_equal(spans[pos[1:],0],spans[pos[:-1],1])
    h=np.concatenate([capture_raw_tokens(enc,x[i:i+batch],[site])[site][:,pos].numpy()
                      for i in range(0,len(x),batch)])
    np.savez_compressed(cache/(model+'.npz'),ids=ids,tokens=h,positions=pos)
    sel=last_window_slice(enc,pos)
    values=[]
    for i in range(0,len(x),batch):
        state=torch.from_numpy(h[i:i+batch])
        write=torch.zeros_like(state[:,0],requires_grad=True)
        with torch.enable_grad(),token_patch(enc.module,site,sel,state+write[:,None]):
            y=enc.predict_differentiable(x[i:i+batch],8)
            grad=[torch.autograd.grad(y[:,j].sum(),write,retain_graph=j<7)[0] for j in range(8)]
        values.append(torch.stack(grad,dim=1).detach().numpy())
    jac=np.concatenate(values)
    base=forecast16(enc,site,x,h,None,sel,batch)
    with np.load(CAL/(model+'_jac16.npz')) as f:radius=float(f['radius'])
    idx=np.linspace(0,len(x)-1,12,dtype=int)
    v=np.random.default_rng(20261122012).normal(size=(len(idx),h.shape[-1]))
    v=(.002*radius*v/np.linalg.norm(v,axis=1)[:,None]).astype('f4')
    plus,minus=[forecast16(enc,site,x[idx],h[idx],sign*v,sel,batch) for sign in (1,-1)]
    physical=(plus[:,:8]-minus[:,:8])/2
    pred=np.einsum('nhd,nd->nh',jac[idx],v)
    error=np.linalg.norm(physical-pred)/np.linalg.norm(physical)
    assert np.isfinite(jac).all() and error<.08,error
    np.savez_compressed(cache/(model+'_early.npz'),ids=ids,jac=jac,forecast=base,radius=radius,
                        fd_rows=idx,fd_actual=physical,fd_linear=pred,fd_error=error)
    print(model,'finite difference',error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('data',*MODELS))
    p.add_argument('--cohort',choices=['long','short'],default='long');p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=8)
    a=p.parse_args();torch.set_num_threads(1)
    cache=a.cache if a.cohort=='long' else a.cache/'short'
    if a.stage=='data':(prepare if a.cohort=='long' else prepare_short)(cache,a.out)
    else:probe(a.stage,cache,a.batch)
