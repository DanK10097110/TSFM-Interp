"""Independent dense frozen-core replay of prescribed saved-state paths."""
import numpy as np
from experiments.feature_search import dictionary_smoothness_math as math


def canonical(h,artifact,method,dtype):
    """Original frozen encoder/dense decoder, or direct fixed-PC linear projection."""
    x=np.asarray(h,dtype=dtype);shape=x.shape
    if method=='identity':
        return x.copy(),None
    a={k:(np.asarray(v,dtype=dtype) if k!='k' else v) for k,v in artifact.items()}
    flat=(x.reshape(-1,shape[-1])-a['center'])/dtype(a['scale'])
    if method=='pca_fixed':
        basis=a['basis'][:,:int(a['k'])]
        result=(flat@basis)@basis.T
        support=None
    else:
        from experiments.feature_search import dictionary_core
        z=dictionary_core.encode(flat,a)
        result=dictionary_core.reconstruct(z,a)
        support=(z!=0).reshape(shape[:-1]+(z.shape[-1],))
    result=result*dtype(a['scale'])+a['center']
    return result.reshape(shape),support


def replay(h,artifact,method,expected64,expected32,support64,support32):
    """Audit FP64 tightly, FP32 at explicit rounding tolerance; repeat each exactly."""
    results={}
    for dtype,label,expected,support in ((np.float64,'FP64',expected64,support64),(np.float32,'FP32',expected32,support32)):
        actual,active=canonical(h,artifact,method,dtype)
        repeated,again=canonical(h,artifact,method,dtype)
        if not np.array_equal(actual,repeated) or (active is not None and not np.array_equal(active,again)):
            raise ValueError('Dictionary canonical repeat failed')
        difference=np.asarray(actual,float)-expected
        scale=max(1.,float(math.rms(actual)))
        limit=1e-10 if dtype==np.float64 else 5e-5
        same_support=active is None or np.array_equal(active,support)
        if not same_support or np.max(abs(difference))>limit*scale:
            raise ValueError('Independent frozen dictionary replay mismatch: '+label)
        results[label]=dict(max_abs_error=float(np.max(abs(difference))),RMS_error=float(math.rms(difference)),
            support_identical=same_support,repeat_exact=True,tolerance=limit*scale)
    isolated=actual[0].astype(float)+actual[1]-2*actual[3]
    results['FP32_isolated_RMS_mean']=float(math.rms(isolated,(-2,-1)).mean())
    return dict(valid=True,precision=results)
