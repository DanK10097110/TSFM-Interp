"""Frozen dictionary inference and chord geometry; no fitting or native calls.

Actual path curvature and off-manifold activation-chord nonlinearity are kept
separate. Adaptive signed PCA TopK is a nonlinear selector, unlike first-k PCs.
"""
import numpy as np


def rms(x,axis=None):
    """FP64 measurement arithmetic even when inference uses FP32."""
    return np.sqrt(np.mean(np.asarray(x,float)**2,axis=axis))


def top_indices(pre,k):
    """Fast magnitude TopK with exact stable-index resolution of positive ties."""
    scores=-np.abs(pre)
    order=np.argpartition(scores,k-1,axis=1)[:,:k].copy()
    threshold=np.min(np.abs(np.take_along_axis(pre,order,axis=1)),axis=1)
    tied=(threshold>0)&(np.sum(np.abs(pre)==threshold[:,None],axis=1)>1)
    for row in np.flatnonzero(tied):
        order[row]=np.argsort(-abs(pre[row]),kind='stable')[:k]
    order.sort(axis=1)
    return order


def evaluate(h,artifact,method,dtype,coefficients=None):
    """Decode frozen maps in native residual units; chunk sparse decoder gathers."""
    h=np.asarray(h,dtype=dtype)
    shape=h.shape;flat=h.reshape(-1,shape[-1]);d=shape[-1]
    if method=='identity':
        return h.copy(),None
    center=np.asarray(artifact['center'],dtype=dtype);scale=dtype(artifact['scale']);k=int(artifact['k'])
    if coefficients is None:
        x=(flat-center)/scale
        if method.startswith('pca'):
            basis=np.asarray(artifact['basis'],dtype=dtype)
            pre=x@(basis[:,:k] if method=='pca_fixed' else basis)
        else:
            x-=np.asarray(artifact['b_dec'],dtype=dtype)
            pre=x@np.asarray(artifact['W_enc'],dtype=dtype)+np.asarray(artifact['b_enc'],dtype=dtype)
            np.maximum(pre,0,out=pre)
        del x
    else:
        pre=coefficients
    decoder=np.asarray(artifact['basis'].T if method.startswith('pca') else artifact['W_dec'],dtype=dtype)
    if method=='pca_fixed':
        output=pre[:,:k]@decoder[:k]
        support=None
    else:
        indices=top_indices(pre,k);values=np.take_along_axis(pre,indices,axis=1)
        support=np.zeros(pre.shape,bool)
        np.put_along_axis(support,indices,values!=0,axis=1)
        output=np.empty((len(flat),d),dtype=dtype)
        for start in range(0,len(flat),128):
            sl=slice(start,start+128)
            output[sl]=np.einsum('nk,nkd->nd',values[sl],decoder[indices[sl]],optimize=False)
        support=support.reshape(shape[:-1]+(pre.shape[-1],))
    if not method.startswith('pca'):
        output+=np.asarray(artifact['b_dec'],dtype=dtype)
    output*=scale;output+=center
    return output.reshape(shape),support


def turnover(a,b):
    """Mean token Jaccard distance; all-zero support contributes zero."""
    union=np.sum(a|b,axis=-1);different=np.sum(a^b,axis=-1)
    value=np.divide(different,union,out=np.zeros(union.shape,float),where=union>0)
    return value.mean(axis=-1)


def floor_ratio(numerator,denominator,floor,multiplier):
    """Explicit refloor sensitivity; never hide zero/raw-unresolved denominators."""
    if multiplier not in (1,10,100) or np.any(np.asarray(floor)<=0):
        raise ValueError('Positive resolution floor and frozen multiplier required')
    return np.asarray(numerator)/np.maximum(denominator,multiplier*np.asarray(floor))


def measure(h,g,support,reference,radii):
    """Minus/plus/actual midpoint/interpolated midpoint geometry for each path."""
    h=np.asarray(h,float);g=np.asarray(g,float)
    raw=h[0]+h[1]-2*h[2]
    actual=g[0]+g[1]-2*g[2]
    isolated=g[0]+g[1]-2*g[3]
    transported=2*(g[3]-g[2])
    axes=(-2,-1)
    raw_rms=rms(raw,axes);actual_rms=rms(actual,axes);isolated_rms=rms(isolated,axes)
    floor=np.maximum(np.finfo(np.float32).eps*(rms(h[0],axes)+rms(h[1],axes)+2*rms(h[2],axes)),1e-30)
    secant=rms((h[1]-h[0])/2,axes)
    output=dict(raw_chord_RMS=raw_rms,decoded_actual_chord_RMS=actual_rms,isolated_dictionary_chord_RMS=isolated_rms,
        transported_midpoint_deviation_RMS=rms(transported,axes),state_FP32_resolution_proxy=floor,
        raw_curvature=raw_rms/reference/radii**2,decoded_curvature=actual_rms/reference/radii**2,
        isolated_curvature=isolated_rms/reference/radii**2,
        midpoint_reconstruction_relative_RMS=rms(g[2]-h[2],axes)/reference,
        decoded_secant_to_raw=rms((g[1]-g[0])/2,axes)/np.maximum(secant,floor),
        chord_decomposition_error=rms(actual-isolated-transported,axes),
        raw_above_10x_proxy=(raw_rms>10*floor).astype(float),
        isolated_above_10x_proxy=(isolated_rms>10*floor).astype(float),
        actual_above_raw=(actual_rms>raw_rms).astype(float))
    for factor in (1,10,100):
        output[f'isolated_to_raw_floor{factor}']=floor_ratio(isolated_rms,raw_rms,floor,factor)
        output[f'actual_to_raw_floor{factor}']=floor_ratio(actual_rms,raw_rms,floor,factor)
    if support is None:
        output.update(endpoint_support_turnover=np.zeros(len(raw_rms)),chord_support_turnover=np.zeros(len(raw_rms)),
            any_chord_support_switch=np.zeros(len(raw_rms)))
    else:
        output.update(endpoint_support_turnover=turnover(support[0],support[1]),
            chord_support_turnover=(turnover(support[0],support[3])+turnover(support[1],support[3]))/2,
            any_chord_support_switch=np.any((support[0]!=support[3])|(support[1]!=support[3]),axis=(-2,-1)).astype(float))
    return output,actual,isolated


def aggregate(metrics,paths):
    """Average dependent phase/frequency paths within source before group summaries."""
    summaries=[]
    for axis in ('amplitude','frequency'):
        for radius in (.002,.005,.01):
            sources=sorted({p['source'] for p in paths})
            ids=[[i for i,p in enumerate(paths) if p['source']==s and p['axis']==axis and p['radius']==radius] for s in sources]
            if any(not group for group in ids):
                raise ValueError('Incomplete source/axis/radius geometry')
            row=dict(axis=axis,radius=radius,n_sources=len(sources),source_ids=sources,metrics={})
            for name,values in metrics.items():
                per_source=np.asarray([np.mean(np.asarray(values)[group]) for group in ids])
                row['metrics'][name]=dict(mean=float(per_source.mean()),median=float(np.median(per_source)),
                    min=float(per_source.min()),max=float(per_source.max()),per_source=per_source.tolist())
            summaries.append(row)
    return summaries
