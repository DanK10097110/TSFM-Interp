"""Does p0 signal recoverability exceed the same architecture before training?"""
from pathlib import Path
import argparse,time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.coverage_training import make_model


def run(source,out):
    """Frozen same-seed initialization, same supervised decoding and test histories."""
    import torch
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();p=read(source/'protocol.json');z=np.load(source/'predictions.npz');trained=read(source/'analysis.json')
    write(out/'protocol.json',dict(source=str(source),source_artifact_sha256=sha(source/'predictions.npz'),code_sha256=sha(__file__),scope='Exploratory initialization control prompted by p0 hidden-recovery differences; same test sources, no new independent confirmation',ridge_alpha=10.))
    x=z['train_history'].astype(float);ex=z['test_history'].astype(float);y=z['train_tone'].astype(float);ey=z['test_tone'].astype(float);mu=x.mean(1,keepdims=True);sd=x.std(1,keepdims=True);emu=ex.mean(1,keepdims=True);esd=ex.std(1,keepdims=True);target=(y-mu)/sd;energy=np.mean(ey**2,axis=1);records=[];arrays={}
    for arch in p['architectures']:
        for seed in p['seeds']:
            model=make_model(arch,seed).eval();head=model.head if arch=='transformer' else model.net[-1];captured=[]
            handle=head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy().copy()))
            try:
                with torch.no_grad():model(torch.tensor(x,dtype=torch.float32));model(torch.tensor(ex,dtype=torch.float32))
            finally:handle.remove()
            probe=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(captured[0],target);decoded=probe.predict(captured[1])*esd+emu
            error=float(np.mean(np.mean((decoded-ey)**2,axis=1)/energy));key=f'{arch}_{seed}';arrays[key]=decoded
            p0=next(r for r in trained['records'] if r['architecture']==arch and r['seed']==seed and r['training_probability']==0.)
            records.append(dict(architecture=arch,seed=seed,initial_decoded_nmse=error,p0_decoded_nmse=p0['decoded_tone_nmse'],initial_minus_p0=error-p0['decoded_tone_nmse']))
    np.savez_compressed(out/'predictions.npz',**arrays);write(out/'results.json',dict(records=records,seconds=time.monotonic()-start));print(records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source,a.out)
