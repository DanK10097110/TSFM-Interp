"""Transfer the same small sinusoidal hidden effect between real backgrounds.

Uses saved locked-normalizer states, with natural-counterfactual oracle closure
at each receiving layer. No actual future labels are read. Nearest and farthest
FIT donors are chosen by observed history only, within the receiver's dataset.
"""
from pathlib import Path
import argparse,time,resource
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha,normalization


def run(out,model):
    """Probe contextual variation with source-disjoint donors and matched controls."""
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.selective_study import blocks,patched
    from experiments.feature_search.composition_restoration import random_edits
    started=time.monotonic();p=read(out/'protocol.json');native=Native(model,p['checkpoints'][model]['path']);sites=blocks(native)
    rows=read(out/'rows.json');x=np.load(out/'inputs.npz')['history'];states=np.load(out/f'{model}_states.npz');cuts=read(out/f'{model}_worker.json')['cuts'];result=[]
    declaration=dict(model=model,source_sha256=sha(__file__),radii=[.01,.1],donors='nearest/farthest FIT same-domain history',oracle_gate='response-normalized MSE<=1e-4 and nonzero natural response',semantic_gate='native known-sinusoid response NMSE<=.5; all other cases remain descriptive',scope='Transfer native behavior, not cross-model concept alignment; no target file read')
    write(out/f'{model}_transfer_protocol.json',declaration)
    for i,receiver in enumerate(rows):
        if receiver['case']!='real' or receiver['split']!='test':continue
        eligible=[j for j,r in enumerate(rows) if r['case']=='real' and r['split']=='fit' and r['domain']==receiver['domain']]
        ordered=sorted(eligible,key=lambda j:float(np.mean((x[i]-x[j])**2)));donors={'near':ordered[0],'far':ordered[-1]}
        full=states[f'{i}_full'].astype(float);base=native.predict(x[i:i+1])[0][0]
        raw=np.sin(2*np.pi*5.35*np.arange(-128,0)/128+.3);normal=float(np.sqrt(np.mean(raw**2)));sd=float(x[i].std())
        for epsilon in [.01,.1]:
            query=x[i:i+1]+epsilon*sd*raw/normal
            with normalization(native,x[i:i+1],'background'):natural=native.predict(query)[0][0]
            target=epsilon*sd*np.sin(2*np.pi*5.35*np.arange(32)/128+.3)/normal
            response=natural-base;energy=float(np.mean(response**2));native_signal_nmse=float(np.mean((response-target)**2)/np.mean(target**2))
            counter=states[f'{i}_sinusoid_{epsilon}_locked'][:,0].astype(float)
            for c,(cut,module) in enumerate(sites.items()):
                oracle=patched(native,x[i:i+1],module,counter[c:c+1])[0]
                oracle_error=float(np.mean((oracle-natural)**2)/max(energy,1e-20));oracle_valid=energy>1e-12*sd**2 and oracle_error<=1e-4
                for distance,j in donors.items():
                    donorfull=states[f'{j}_full'][c].astype(float);donorplus=states[f'{j}_sinusoid_{epsilon}_locked'][c,0].astype(float);edit=donorplus-donorfull
                    randomized,invalid=random_edits(full[c:c+1],edit[None],np.random.default_rng(4945000+i*10+c),count=1)
                    variants=np.stack([full[c]+edit,full[c]+randomized[0,0]])
                    forecasts=patched(native,np.repeat(x[i:i+1],2,axis=0),module,variants)
                    desired=counter[c]-full[c];state_error=float(np.mean((edit-desired)**2)/max(float(np.mean(desired**2)),1e-20))
                    result.append(dict(row=i,source=receiver['source'],domain=receiver['domain'],donor=rows[j]['source'],distance=distance,cut=cut,epsilon=epsilon,oracle_response_nmse=oracle_error,oracle_valid=oracle_valid,native_signal_nmse=native_signal_nmse,semantic_valid=native_signal_nmse<=.5 and oracle_valid,random_valid=not bool(invalid[0]),state_effect_nmse=state_error,transfer_response_nmse=float(np.mean((forecasts[0]-natural)**2)/max(energy,1e-20)),random_response_nmse=float(np.mean((forecasts[1]-natural)**2)/max(energy,1e-20)),response_energy=energy,base=base.tolist(),natural=natural.tolist(),oracle=oracle.tolist(),transferred=forecasts[0].tolist(),random_forecast=forecasts[1].tolist()))
        if time.monotonic()-started>300 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>3*2**30:raise RuntimeError('Transfer resource cap')
    write(out/f'{model}_transfer.json',result);write(out/f'{model}_transfer_execution.json',dict(seconds=time.monotonic()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,sha256=sha(out/f'{model}_transfer.json')))
    print(model,'transfers',len(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=['bolt','timesfm'],required=True);a=p.parse_args();run(a.out,a.model)
