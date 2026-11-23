"""Generic learned time continuation, history controls, and fixed frequency scores."""
import numpy as np
from scipy.fft import dct, idct


def permute_histories(n):
    order=np.random.default_rng(20261121073).permutation(n)
    perm=np.empty(n,dtype=int)
    perm[order]=np.roll(order,1)
    return perm


def blinded(cal):
    """Eliminate late DCT modes 3–7 from calibration derivative rows."""
    x=cal.copy()
    freq=dct(x[:,8:],axis=1,norm="ortho")
    freq[:,3:]=0
    x[:,8:]=idct(freq,axis=1,norm="ortho")
    return x


def within_model(cal,current):
    """Predict late derivative per channel with a lead map trained across that model's channels."""
    early=cal[:,:8]
    late=cal[:,8:]
    mean_e,mean_l=early.mean(0),late.mean(0)
    e=(early-mean_e).transpose(0,2,1).reshape(-1,8)
    l=(late-mean_l).transpose(0,2,1).reshape(-1,8)
    gram=e.T@e
    strength=.001*np.linalg.eigvalsh(gram)[-1]
    if strength<=0:
        raise ValueError("No varying early directions for regression")
    coeff=np.linalg.solve(gram+strength*np.eye(8),e.T@l)
    return mean_l+np.einsum("el,ned->nld",coeff,current-mean_e,optimize=True)


def bands(x):
    """Orthonormal coordinates; energy sums of early and late bands equal centered energy."""
    z=dct(x[...,8:],axis=-1,norm="ortho")
    return dict(early=x[...,:8]-x[...,:8].mean(-1,keepdims=True),
                late=z[...,1:],slow=z[...,1:3],fast=z[...,3:])
