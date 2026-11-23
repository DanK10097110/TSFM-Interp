"""Model-independent linear algebra for generic response-matched intervention transport."""
import numpy as np


def goals():
    """First three registered requests and two unit-RMS orthogonal holdout shapes."""
    t=np.arange(8,dtype=float)
    rows=[np.ones(8),np.linspace(-1,1,8),np.sin(2*np.pi*t/8)]
    for wave in [np.exp(-.5*((t-1)/.85)**2),np.sin(2*np.pi*t/4)]:
        basis,_=np.linalg.qr(np.stack(rows).T)
        residual=wave-basis@(basis.T@wave)
        if np.linalg.norm(residual)<1e-7: raise ValueError('Degenerate novel forecast request')
        rows.append(residual)
    waves=np.stack(rows)
    return .05*waves/np.sqrt(np.mean(waves**2,axis=1,keepdims=True))


def ridge_response(target_jac, source_jac):
    """Column-vector map W minimizing sum_i ||J_t[i] W - J_s[i]||² plus ridge."""
    if target_jac.shape[:2]!=source_jac.shape[:2]:
        raise ValueError('FIT histories and forecast leads must align')
    a=target_jac.reshape(-1,target_jac.shape[-1]); b=source_jac.reshape(-1,source_jac.shape[-1])
    eigen=np.linalg.eigvalsh(a.T@a); penalty=.01*eigen[-1]
    if penalty<=0: raise ValueError('Inactive target Jacobian')
    w=np.linalg.solve(a.T@a+penalty*np.eye(a.shape[1]),a.T@b)
    return w, float(penalty)


def limit_norm(vectors,radius):
    n=np.linalg.norm(vectors,axis=1,keepdims=True)
    if not np.isfinite(n).all() or np.any(n<=1e-12): raise ValueError('Degenerate candidate write')
    return vectors*np.minimum(1.,radius/n)


def match_norm(vectors,reference):
    n=np.linalg.norm(vectors,axis=1,keepdims=True)
    if np.any(n<=1e-12): raise ValueError('Cannot match a zero write')
    return vectors*(np.linalg.norm(reference,axis=1,keepdims=True)/n)


def policies(jac,basis,radius,original):
    waves=goals(); wanted=np.broadcast_to(waves.T[None],(len(jac),waves.shape[1],len(waves)))
    coeff,_=ridge_response(jac,wanted)
    physical=limit_norm(coeff.T@basis,.20*radius)
    first=physical[:len(original)]
    relative=np.linalg.norm(first-original,axis=1)/np.linalg.norm(original,axis=1)
    if np.max(relative)>.005: raise ValueError(f'Original source fit mismatch: {relative}')
    physical[:len(original)]=original
    projection=physical@basis.T@basis
    np.testing.assert_allclose(physical,projection,atol=2e-7,rtol=2e-6)
    return physical,relative
