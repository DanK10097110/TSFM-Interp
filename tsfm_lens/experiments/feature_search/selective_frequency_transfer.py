"""Frequency-specificity control for successful near-daily effect transfer.

A fixed19.2-hour oscillation tests whether the near-daily result extends away
from a familiar seasonal period. Same real histories, donors, radius and cuts;
no future targets, no frequency search and no retraining.
"""
from pathlib import Path
import argparse,time,resource
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha,normalization


def run(out,model):
    """Capture fresh common-normalizer effects, then test same-domain donor transfer."""
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.selective_study import blocks,capture,patched
    from experiments.feature_search.composition_restoration import random_edits
    start=time.monotonic();p=read(out/'protocol.json');rows=read(out/'rows.json');inputs=np.load(out/'inputs.npz')['history'];native=Native(model,p['checkpoints'][model]['path']);sites=blocks(native);period=19.2;epsilon=.1
    prefix=model+'_transfer_period19p2'
    write(out/f'{prefix}_protocol.json',dict(period=period,epsilon=epsilon,scope='Exploratory frequency control after near-daily Bolt transfer; fixed before these runs; no target access',source_sha256=sha(__file__),oracle_gate=1e-4,semantic_gate=.5))
    saved={};cache={};t=np.arange(-128,32);wave=np.sin(2*np.pi*t/period+.3);normal=np.sqrt(np.mean(wave[:128]**2))
    for i,row in enumerate(rows):
        if row['case']!='real':continue
        x=inputs[i:i+1];base,h=capture(native,x,sites);query=x+epsilon*x.std()*wave[:128]/normal
        with normalization(native,x,'background'):natural,hp=capture(native,query,sites)
        cache[i]=(base[0],natural[0],h,hp);saved[f'{i}_full']=np.stack([h[k][0] for k in sites]);saved[f'{i}_perturbed']=np.stack([hp[k][0] for k in sites])
    results=[]
    for i,row in enumerate(rows):
        if row['case']!='real' or row['split']!='test':continue
        eligible=[j for j,r in enumerate(rows) if r['case']=='real' and r['split']=='fit' and r['domain']==row['domain']];ordered=sorted(eligible,key=lambda j:float(np.mean((inputs[j]-inputs[i])**2)))
        base,natural,h,hp=cache[i];response=natural-base;energy=float(np.mean(response**2));target=epsilon*inputs[i].std()*wave[128:]/normal;native_error=float(np.mean((response-target)**2)/np.mean(target**2))
        for c,(cut,module) in enumerate(sites.items()):
            oracle=patched(native,inputs[i:i+1],module,hp[cut])[0];oracle_error=float(np.mean((oracle-natural)**2)/max(energy,1e-20));valid=energy>1e-12*float(inputs[i].var()) and oracle_error<=1e-4
            for distance,j in [('near',ordered[0]),('far',ordered[-1])]:
                edit=cache[j][3][cut][0].astype(float)-cache[j][2][cut][0].astype(float);receiver=h[cut][0].astype(float)
                randomized,invalid=random_edits(receiver[None],edit[None],np.random.default_rng(4946000+i*10+c),count=1)
                forecasts=patched(native,np.repeat(inputs[i:i+1],2,axis=0),module,np.stack([receiver+edit,receiver+randomized[0,0]]))
                desired=hp[cut][0].astype(float)-receiver
                results.append(dict(row=i,source=row['source'],domain=row['domain'],donor=rows[j]['source'],distance=distance,cut=cut,epsilon=epsilon,period=period,oracle_response_nmse=oracle_error,oracle_valid=valid,native_signal_nmse=native_error,semantic_valid=valid and native_error<=.5,random_valid=not bool(invalid[0]),state_effect_nmse=float(np.mean((edit-desired)**2)/max(float(np.mean(desired**2)),1e-20)),transfer_response_nmse=float(np.mean((forecasts[0]-natural)**2)/max(energy,1e-20)),random_response_nmse=float(np.mean((forecasts[1]-natural)**2)/max(energy,1e-20)),response_energy=energy,base=base.tolist(),natural=natural.tolist(),oracle=oracle.tolist(),transferred=forecasts[0].tolist(),random_forecast=forecasts[1].tolist()))
        if time.monotonic()-start>300 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>3*2**30:raise RuntimeError('Frequency control cap')
    write(out/f'{prefix}.json',results);np.savez_compressed(out/f'{prefix}_states.npz',**saved);write(out/f'{prefix}_execution.json',dict(seconds=time.monotonic()-start,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,sha256=sha(out/f'{prefix}.json')));print(model,'frequency controls',len(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=['bolt','timesfm'],required=True);a=p.parse_args();run(a.out,a.model)
