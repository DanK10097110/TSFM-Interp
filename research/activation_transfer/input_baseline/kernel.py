"""Fit smooth maps with a target mean learned from the training histories alone."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist


FAMILIES = (("linear", 1.), ("rbf", .25), ("rbf", 1.), ("rbf", 4.))
PENALTIES = (1e-4, .001, .01, .1, 1.)


def raw_state(x, recent=False):
    """Keep full standardized history, optionally giving the final 32 values equal total weight."""
    x = np.asarray(x, dtype=float)
    z = (x - x.mean(axis=-1,keepdims=True)) / np.maximum(x.std(axis=-1,keepdims=True),1e-8)
    if recent:
        return np.concatenate([z, z[..., -32:] * np.sqrt(z.shape[-1] / 32)],axis=-1)
    return z


@dataclass
class Kernel:
    x: np.ndarray
    family: str
    width: float
    column_mean: np.ndarray
    grand_mean: float
    eig: np.ndarray
    vec: np.ndarray
    scale: float

    @classmethod
    def fit(cls,x,family,multiplier):
        x = np.asarray(x,dtype=float)
        if family == "rbf":
            widths = pdist(x, "sqeuclidean")
            width = max(float(np.median(widths)), 1e-12) * multiplier
            gram = np.exp(-cdist(x,x,"sqeuclidean") / width)
        elif family == "linear":
            width = 1.
            gram = x @ x.T
        else:
            raise ValueError("Unknown kernel")
        col = gram.mean(axis=0)
        centered = gram - col[:,None] - col[None,:] + col.mean()
        eig, vec = eigh(centered)
        return cls(x,family,width,col,float(col.mean()),np.maximum(eig,0),vec,
                   max(float(np.trace(centered)/len(x)),1e-12))

    def cross(self,x):
        if self.family == "rbf":
            k = np.exp(-cdist(x,self.x,"sqeuclidean") / self.width)
        else:
            k = x @ self.x.T
        return k - k.mean(axis=1,keepdims=True) - self.column_mean + self.grand_mean


@dataclass
class Map:
    kernel: Kernel
    coef: np.ndarray
    mean: np.ndarray

    def predict(self,x):
        return self.kernel.cross(x) @ self.coef + self.mean


def fit(x,y):
    """Compare all candidates with foldwise refitted preprocessing and intercept."""
    y = np.asarray(y, dtype=float)
    loss = np.zeros((len(FAMILIES),len(PENALTIES)))
    folds = np.arange(len(y)) % 4
    for f in range(4):
        train = folds != f
        avg = y[train].mean(axis=0)
        truth = y[~train]-avg
        for m,(fam,mult) in enumerate(FAMILIES):
            k = Kernel.fit(x[train],fam,mult)
            left = k.cross(x[~train]) @ k.vec
            right = k.vec.T @ (y[train]-avg)
            for j,p in enumerate(PENALTIES):
                pred = (left / (k.eig+k.scale*p)) @ right
                loss[m,j] += np.square(pred-truth).mean()/4
    i,j = np.unravel_index(np.argmin(loss),loss.shape)
    family,mult = FAMILIES[i]
    k = Kernel.fit(x,family,mult)
    avg = y.mean(axis=0)
    coef = k.vec @ ((k.vec.T @ (y-avg)) / (k.eig+k.scale*PENALTIES[j])[:,None])
    return Map(k,coef,avg), dict(family=family,width_multiplier=mult,relative_penalty=PENALTIES[j],
                                  cv_mse=float(loss[i,j]),grid=loss.tolist())
