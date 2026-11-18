"""Conservative continuous-frequency peeling with an intact residual.

Unlike the earlier fixed FFT partitions, a split must explain meaningful energy,
stand above the spectral floor and reproduce its phase/amplitude in past halves.
No future or generator decomposition enters this API.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def candidate(x):
    """Fit one dominant oscillation and assess its past-only separability."""
    x=np.asarray(x,float);n=len(x);t=np.arange(-n,0)/128
    def basis(f,time=t):return np.column_stack([np.ones(len(time)),np.sin(2*np.pi*f*time),np.cos(2*np.pi*f*time)])
    def fit(f):
        a=basis(f);coef=np.linalg.lstsq(a,x,rcond=None)[0];return coef,x-a@coef
    power=abs(np.fft.rfft(x-x.mean(),n=4096))**2;grid=np.fft.rfftfreq(4096)*128
    power[(grid<2)|(grid>48)]=0;freq=float(grid[np.argmax(power)])
    opt=minimize_scalar(lambda f:np.mean(fit(f)[1]**2),bounds=(max(2,freq-.75),min(48,freq+.75)),method='bounded',options={'xatol':1e-9})
    freq=float(opt.x);coef,residual=fit(freq);tone=basis(freq)[:,1:]@coef[1:]
    halves=[]
    for sl in [slice(0,n//2),slice(n//2,n)]:halves.append(np.linalg.lstsq(basis(freq)[sl],x[sl],rcond=None)[0][1:])
    stability=float(np.clip(1-np.linalg.norm(halves[0]-halves[1])/(np.linalg.norm(coef[1:])+1e-12),0,1))
    explained=float(np.clip(1-np.mean(residual**2)/(np.var(x)+1e-12),0,1))
    coarse=abs(np.fft.rfft(x-x.mean()))**2;bins=np.fft.rfftfreq(n)*128
    floor=coarse[(bins>=2)&(bins<=48)&(abs(bins-freq)>2)]
    prominence=float(coarse[np.argmin(abs(bins-freq))]/max(float(np.median(floor)),1e-12))
    confidence=float(stability*np.sqrt(explained)*min(1.,prominence/20))
    return tone,dict(frequency=freq,coefficients=coef[1:].tolist(),stability=stability,explained=explained,prominence=prominence,confidence=confidence)


def decompose(x,depth=3,stability=.65,energy=.15,prominence=8.):
    """An unbalanced nested tree: peel a clear tone, recurse only on the remainder."""
    x=np.asarray(x,float);residual=x.copy();tones=[];metadata=[];levels=[[x.copy()]]
    for _ in range(depth):
        if np.var(residual)<=1e-10*max(np.var(x),1e-12):break
        tone,meta=candidate(residual)
        if meta['stability']<stability or meta['explained']<energy or meta['prominence']<prominence:break
        tones.append(tone);metadata.append(meta);residual=residual-tone
        leaves=tones+[residual.copy()]
        np.testing.assert_allclose(np.sum(leaves,axis=0),x,rtol=1e-12,atol=1e-12)
        levels.append(leaves)
    return levels,metadata


def continuation(metadata,depth):
    """Classical extrapolation of extracted tones only; residual remains modelled."""
    t=np.arange(32)/128;value=np.zeros(32)
    for item in metadata[:depth]:
        a,b=item['coefficients'];value+=a*np.sin(2*np.pi*item['frequency']*t)+b*np.cos(2*np.pi*item['frequency']*t)
    return value


def fuse(states,zero,weights):
    """Translation-invariant anchored composition, retaining residual weight one."""
    states=np.asarray(states,float);zero=np.asarray(zero,float)
    return zero+np.sum((states-zero)*np.asarray(weights).reshape((-1,)+(1,)*zero.ndim),axis=0)


def local_metrics(base,positive,negative,epsilon):
    """Dimensionless secant sensitivity and relative nonlinearity at one radius."""
    base,positive,negative=[np.asarray(a,float) for a in [base,positive,negative]]
    scale=max(float(np.sqrt(np.mean(base**2))),1e-12)
    first=(positive-negative)/2;second=positive+negative-2*base
    return dict(sensitivity=float(np.sqrt(np.mean(first**2))/scale/epsilon),
                curvature_ratio=float(np.sqrt(np.mean(second**2))/(2*np.sqrt(np.mean(first**2))+1e-12)),
                relative_change=float(np.sqrt(np.mean((positive-base)**2))/scale))
