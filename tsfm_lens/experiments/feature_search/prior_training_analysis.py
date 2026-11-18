"""Independent risk audit and signal-recovery probes for matched-prior training."""
from pathlib import Path
import argparse
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from experiments.feature_search.coherence_study import read,write,sha


def run(out):
    """Compare observable predictions and supervised retained information separately."""
    p=read(out/'protocol.json');r=read(out/'results.json');z=np.load(out/'predictions.npz')
    assert sha(out/'predictions.npz')==r['predictions_sha256']
    assert sha(Path(__file__).with_name('prior_training.py'))==p['source_sha256']
    y=z['test_tone'].astype(float);energy=np.mean(y*y,axis=1);records=[]
    train_x=z['train_history'].astype(float);test_x=z['test_history'].astype(float)
    mu=train_x.mean(1,keepdims=True);sd=train_x.std(1,keepdims=True);emu=test_x.mean(1,keepdims=True);esd=test_x.std(1,keepdims=True)
    target=(z['train_tone']-mu)/sd
    input_probe=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit((train_x-mu)/sd,target)
    baseline=float(np.mean(np.mean((input_probe.predict((test_x-emu)/esd)*esd+emu-y)**2,axis=1)/energy))
    for row in r['records']:
        key=row['key'];pred=z[key].astype(float);prob=row['training_probability']
        error=np.mean((pred-prob*y)**2,axis=1)
        np.testing.assert_allclose(float(np.mean(error/energy)),row['conditional_mean_nmse'],rtol=1e-5,atol=1e-7)
        for q in p['probabilities']:
            risk=np.mean((pred-q*y)**2,axis=1)+q*(1-q)*energy
            np.testing.assert_allclose(float(np.mean(risk/energy)),row['risks'][str(q)],rtol=1e-7,atol=1e-7)
        features=z[key+'_train_hidden'];test=z[key+'_test_hidden']
        probe=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(features,target);decoded=probe.predict(test)*esd+emu
        decoded_error=float(np.mean(np.mean((decoded-y)**2,axis=1)/energy))
        records.append(dict(row,decoded_tone_nmse=decoded_error,input_probe_nmse=baseline,scope='Supervised fixed ridge on normalized latent future, native input normalizer restored identically for hidden and input probes; additional task labels; not zero-shot prediction nor causal head repair'))
    adequate={}
    for arch in p['architectures']:
        group=[a for a in records if a['architecture']==arch and a['training_probability']==1.]
        adequate[arch]=all(a['conditional_mean_nmse']<=.15 for a in group)
    write(out/'analysis.json',dict(records=records,adequate_p1_fit=adequate,audit_passed=True,analysis_sha256=sha(__file__),unit='Training seed for training interventions; TEST histories for conditional forecast risk only'))
    print('Adequacy:',adequate,'Input probe:',baseline)
    for row in records:print(row['key'],'gain',row['mean_gain'],'meanerror',row['conditional_mean_nmse'],'decoded',row['decoded_tone_nmse'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
