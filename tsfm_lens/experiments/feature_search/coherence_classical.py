"""Two-frequency history-only estimator: identifiability reference, not an oracle.

The number of tones is a privileged structural assumption, but their frequencies,
phases and amplitudes are fitted from each observed history only.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from experiments.feature_search.coherence_study import load_bank,read,write,sha


def forecast(x):
    """Greedy spectral initialization followed by joint variable projection."""
    x=np.asarray(x,float);t=np.arange(-len(x),0)/128;future=np.arange(32)/128
    def design(f,time): return np.column_stack([np.ones(len(time))]+[z for a in f for z in [np.sin(2*np.pi*a*time),np.cos(2*np.pi*a*time)]])
    def residual(f):
        a=design(f,t);return x-a@np.linalg.lstsq(a,x,rcond=None)[0]
    frequencies=[]
    for _ in range(2):
        resid=residual(frequencies);power=abs(np.fft.rfft(resid,n=4096))**2;grid=np.fft.rfftfreq(4096)*128
        power[(grid<.5)|(grid>60)]=0
        initial=grid[np.argmax(power)];width=128/len(x)
        result=minimize_scalar(lambda f:np.mean(residual(frequencies+[f])**2),bounds=(max(.1,initial-width),min(63.9,initial+width)),method='bounded')
        frequencies.append(result.x)
    fit=least_squares(residual,frequencies,bounds=(.05,63.95),max_nfev=50)
    a=design(fit.x,t);coef=np.linalg.lstsq(a,x,rcond=None)[0]
    return design(fit.x,future)@coef


def run(out):
    """Fit synthetic histories only; save explicit IDs and provenance."""
    rows=read(out/'rows.json');bank=load_bank(out/'bank.npz');ids=sorted({r['root'] for r in rows if r['kind']=='interference'})
    predictions=np.stack([forecast(bank[i]) for i in ids])
    np.savez_compressed(out/'classical.npz',ids=ids,predictions=predictions)
    write(out/'classical.json',dict(bank_sha256=sha(out/'bank.npz'),sha256=sha(out/'classical.npz'),source_sha256=sha(__file__),scope='Two-tone structural assumption; all frequencies fitted, no generator labels read'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
