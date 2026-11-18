"""Exploratory heldout paired forecast errors and known-period harmonic fits."""
import argparse
import json
from pathlib import Path
import numpy as np


def estimate(values):
    """Bootstrap generating seeds, never individual context tokens."""
    values=np.asarray(values);rng=np.random.default_rng(3401)
    means=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(values))


def analyze(root):
    """Compare saved forecasts on heldout synthetic seeds without refitting models."""
    meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    corpus=np.load(root/'corpus.npz');truth=corpus['clean_future']
    keep=np.array([m['split']=='test' for m in meta]);period=np.array([m['period'] for m in meta]);dose=np.array([m['dose'] for m in meta]);scale=np.array([m['scale'] for m in meta])
    forecasts={name:np.load(root/f'{name}.npz')['prediction'] for name in ['bolt','timesfm']}
    error={name:np.mean((pred-truth)**2,axis=1)/scale for name,pred in forecasts.items()}
    gap=error['timesfm']-error['bolt'];result={}
    for label,mask in [('absent',keep&(dose==0)),('present',keep&(dose==1))]+[(f'present_period{p}',keep&(dose==1)&(period==p)) for p in [16,24,32,48]]:
        result[label]={name:estimate(err[mask]) for name,err in error.items()}
        result[label]['timesfm_minus_bolt']=estimate(gap[mask])
    result['model_by_dose_interaction']=estimate(gap[keep&(dose==1)]-gap[keep&(dose==0)])
    harmonics={}
    for name,pred in forecasts.items():
        amplitude_error=[];phase_error=[];trend_error=[]
        for row in np.flatnonzero(keep&(dose==1)):
            m=meta[row];t=np.arange(128,160)
            design=np.column_stack([np.sin(2*np.pi*t/m['period']),np.cos(2*np.pi*t/m['period']),np.ones(32),(t-64)/128])
            coef=np.linalg.lstsq(design,pred[row],rcond=None)[0]
            amplitude_error.append(abs(np.hypot(coef[0],coef[1])/m['amplitude']-1))
            phase_error.append(abs(np.angle(np.exp(1j*(np.arctan2(coef[1],coef[0])-m['phase'])))))
            trend_error.append(abs(coef[3]-m['slope']))
        harmonics[name]=dict(relative_amplitude_abs_error=estimate(amplitude_error),phase_abs_error_radians=estimate(phase_error),slope_abs_error=estimate(trend_error))
    result['harmonic_fit']=harmonics
    result['evidence']='exploratory behavioral; harmonic slopes confounded on less than one period of future'
    (root/'behavior.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);analyze(parser.parse_args().root)
