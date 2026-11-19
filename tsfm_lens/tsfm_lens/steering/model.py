"""Minimal model registration and architecture-independent input-site hooks.

A prediction callable owns preprocessing, device movement, native horizon
handling and output selection. Discovery does not guess those contracts. Hook
sites come from an observed execution, including invocation indices for reused
modules. Only floating batched 2D/3D inputs are inferred automatically.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import copy
import hashlib
import threading

import numpy as np
import torch


class SteeringUnavailable(RuntimeError):
    """A declared model/intervention capability could not be verified."""


@dataclass
class InterventionSite:
    """Serializable observed site; position modes describe tensor axes, not time."""
    module: str
    path: list
    occurrence: int
    shape: list
    mode: str
    source: str = 'module_input'
    module_type: str = ''
    occurrences: list | None = None
    batch_multiplier: int = 1
    invocation_multipliers: list | None = None

    @property
    def width(self):
        single=int(np.prod(self.shape)) if self.mode=='full' else self.shape[-1]
        return single*(len(self.occurrences) if self.occurrences is not None else 1)

    @property
    def label(self):
        invocation=self.occurrences if self.occurrences is not None else self.occurrence
        return f'{self.source}:{self.module}:{invocation}:{self.path}:{self.mode}'


def _leaves(value, path=()):
    """Enumerate tensor leaves without assuming a model-specific argument name."""
    if isinstance(value,torch.Tensor):
        yield path,value
    elif isinstance(value,dict):
        for key,item in value.items():
            yield from _leaves(item,path+(key,))
    elif isinstance(value,(tuple,list)):
        for key,item in enumerate(value):
            yield from _leaves(item,path+(key,))


def _get(value,path):
    for key in path:value=value[key]
    return value


def _replace(value,path,replacement):
    """Rebuild argument containers, leaving caller-owned inputs untouched."""
    if not path:return replacement
    key,*rest=path
    if isinstance(value,dict):
        out=copy.copy(value);out[key]=_replace(value[key],rest,replacement);return out
    out=list(value);out[key]=_replace(value[key],rest,replacement)
    if isinstance(value,tuple):
        return type(value)(*out) if hasattr(value,'_fields') else tuple(out)
    return out


class SteeringModel:
    """Register any deterministic univariate forecast callable.

    predict(contexts[B,C], prediction_length) returns [B,H,Q] or [B,H] for
    point-only outputs. A {'quantiles': ...} or {'point': ...} result is accepted.
    Supply native quantile levels in the exact output order, including 0.5.
    Optional module access enables internal discovery. With no module, control
    can only act on the input; it is always labeled as such. A custom predictor
    should be deterministic and must not mutate the supplied contexts.
    An adapter may supply readout_rows(horizon, quantiles), mapping Linear
    module names to row indices in flattened [step, quantile] order. Declare
    only affine readouts up to a positive context-dependent scale. Return an
    empty mapping for unsupported geometry.
    """
    def __init__(self,predict,*,module=None,quantiles=(.5,),model_id='custom',readout_rows=None):
        if not callable(predict):raise TypeError('predict must be callable')
        if module is not None and not isinstance(module,torch.nn.Module):
            raise TypeError('module must be a torch.nn.Module or None')
        q=np.asarray(quantiles,dtype=float)
        if q.ndim!=1 or not len(q) or not np.isfinite(q).all() or np.any(np.diff(q)<=0) or not np.any(np.isclose(q,.5,atol=1e-7,rtol=0)) or q[0]<=0 or q[-1]>=1:
            raise ValueError('need increasing native quantiles including0.5; point-only uses [0.5]')
        self.predict_fn=predict;self.module=module;self.quantiles=q
        if readout_rows is not None and not callable(readout_rows):
            raise TypeError("readout_rows must be callable")
        self.readout_rows = readout_rows
        self.model_id=str(model_id);self.calls=0;self.evaluated_rows=0
        self.lock=threading.RLock()

    @classmethod
    def from_adapter(cls,adapter,*,quantiles=(.1,.5,.9)):
        """Use any existing repo adapter, without model-name dispatch in discovery."""
        adapter.ensure_loaded()
        module=adapter.module if adapter.capability_tier()>=1 else None
        return cls(lambda x,h:adapter.predict(x,h,list(quantiles)),module=module,
                   quantiles=quantiles,model_id=f'{adapter.cfg.adapter}:{adapter.cfg.checkpoint}' + getattr(adapter, 'forecast_revision', ''),
                   readout_rows=getattr(adapter, 'steering_readout_rows', None))

    def forecast(self,contexts,prediction_length):
        """Validate the public output contract and count actual model calls."""
        x=np.asarray(contexts,dtype=np.float32)
        if x.ndim!=2 or not x.size or not np.isfinite(x).all():
            raise ValueError('need nonempty finite [batch,context] inputs')
        if not isinstance(prediction_length,(int,np.integer)) or prediction_length<1:
            raise ValueError('prediction_length must be a positive integer')
        self.calls+=1;self.evaluated_rows+=len(x)
        value=self.predict_fn(x.copy(),int(prediction_length))
        if isinstance(value,dict):
            value=value.get('quantiles') if value.get('quantiles') is not None else value.get('point')
        if isinstance(value,torch.Tensor):value=value.detach().float().cpu().numpy()
        value=np.asarray(value,dtype=float)
        if value.ndim==2:value=value[...,None]
        if value.shape!=(len(x),prediction_length,len(self.quantiles)) or not np.isfinite(value).all():
            raise SteeringUnavailable(f'forecast must be finite {(len(x),prediction_length,len(self.quantiles))}; got {value.shape}')
        return value

    def fingerprint(self,*,legacy=False):
        """Bind saved policies to exact tensor weights, structure and caller identity.

        Without module access, model_id must identify the immutable model version;
        its server-side weights cannot be inspected by this library.
        """
        digest=hashlib.sha256(self.model_id.encode())
        digest.update(self.quantiles.tobytes())
        if self.module is not None:
            if not legacy:
                for name,module in self.module.named_modules():
                    digest.update(str((name,type(module).__module__,type(module).__qualname__,module.extra_repr())).encode())
            for name,value in self.module.state_dict().items():
                if not isinstance(value,torch.Tensor):
                    raise SteeringUnavailable('non-tensor state requires a simpler registered module')
                digest.update(name.encode());digest.update(str((value.shape,value.dtype)).encode())
                tensor=value.detach().cpu().contiguous()
                if tensor.is_quantized and not legacy:
                    digest.update(str(tensor.qscheme()).encode())
                    if tensor.qscheme() in (torch.per_tensor_affine,torch.per_tensor_symmetric):
                        digest.update(str((tensor.q_scale(),tensor.q_zero_point())).encode())
                    else:
                        digest.update(str(tensor.q_per_channel_axis()).encode())
                        digest.update(tensor.q_per_channel_scales().numpy().tobytes())
                        digest.update(tensor.q_per_channel_zero_points().numpy().tobytes())
                    tensor=tensor.int_repr()
                elif tensor.is_quantized:tensor=tensor.int_repr()
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def trace(self,context,prediction_length,*,min_width=4,max_width=8192,modes=('last','shared'),max_batch_multiplier=64):
        """Observe candidate inputs from parameter-owning modules in execution order."""
        if self.module is None:return []
        if any(m.training for m in self.module.modules()):
            raise SteeringUnavailable('call model.eval() before discovering steering sites')
        x=np.asarray(context,dtype=np.float32)
        if x.shape[0]!=1:raise ValueError('trace expects one context')
        hits=[];counts={};handles=[]
        def make_hook(name,mod):
            def hook(module,args,kwargs):
                occurrence=counts.get(name,0);counts[name]=occurrence+1
                tree={'args':args,'kwargs':kwargs}
                for path,value in _leaves(tree):
                    if (value.is_floating_point() and value.ndim in (2,3) and 1<=value.shape[0]<=max_batch_multiplier
                            and min_width<=value.shape[-1]<=max_width and torch.isfinite(value).all()):
                        choices=('single',) if value.ndim==2 else modes
                        for mode in choices:
                            hits.append(InterventionSite(name,list(path),occurrence,list(value.shape[1:]),mode,
                                                         module_type=type(mod).__qualname__,batch_multiplier=value.shape[0]))
                        break
            return hook
        try:
            for name,module in self.module.named_modules():
                if any(True for _ in module.parameters(recurse=False)):
                    handles.append(module.register_forward_pre_hook(make_hook(name,module),with_kwargs=True))
            self.forecast(x,prediction_length)
        finally:
            for handle in handles:handle.remove()
        # Autoregressive forecasts reuse readouts. Joint control needs distinct
        # coefficients for their invocations, discovered from this execution.
        groups={}
        for site in hits:
            key=(site.module,str(site.path),tuple(site.shape),site.mode)
            groups.setdefault(key,[]).append(site)
        for group in groups.values():
            if len(group)>1:
                first=group[0]
                joint=InterventionSite(**asdict(first));joint.occurrences=[s.occurrence for s in group]
                joint.invocation_multipliers=[s.batch_multiplier for s in group]
                if joint.width<=max_width:hits.append(joint)
        return hits

    def evaluate(self,contexts,prediction_length,site,basis,coefficients):
        """Apply a real input/internal intervention; never modify returned forecasts."""
        x=np.asarray(contexts,dtype=np.float32)
        coefficients=np.asarray(coefficients,dtype=float)
        basis=np.asarray(basis,dtype=float)
        if coefficients.shape!=(len(x),len(basis)) or basis.shape[1]!=site.width:
            raise ValueError('coefficient/basis dimensions do not match the site')
        if not np.isfinite(coefficients).all() or not np.isfinite(basis).all():
            raise ValueError('intervention must be finite')
        if site.source=='input':
            if list(x.shape[1:])!=site.shape:raise SteeringUnavailable('input context geometry changed')
            scales=np.maximum(np.linalg.norm(x-x.mean(1,keepdims=True),axis=1),1e-6)
            return self.forecast(x+(coefficients@basis)*scales[:,None],prediction_length)
        if self.module is None:raise SteeringUnavailable('internal intervention requires module access')
        module=self.module.get_submodule(site.module)
        counts=[0,0]
        selected=site.occurrences if site.occurrences is not None else [site.occurrence]
        multipliers=site.invocation_multipliers if site.occurrences is not None else [site.batch_multiplier]
        def hook(mod,args,kwargs):
            occurrence=counts[0];counts[0]+=1
            if occurrence not in selected:return None
            slot=selected.index(occurrence);multiplier=multipliers[slot]
            tree={'args':args,'kwargs':kwargs};value=_get(tree,site.path)
            if not isinstance(value,torch.Tensor) or list(value.shape[1:])!=site.shape or value.shape[0]!=len(x)*multiplier:
                raise SteeringUnavailable('observed site geometry or batch axis changed')
            counts[1]+=1
            direction=(coefficients@basis).reshape(len(x),len(selected),-1)[:,slot]
            # Sample-major expansion is checked against singleton forecasts by
            # the response estimator. Other layouts fall back to singleton use.
            direction=np.repeat(direction,multiplier,axis=0)
            row=torch.as_tensor(direction,dtype=value.dtype,device=value.device)
            if site.mode=='last':
                norm=value[:,-1].float().norm(dim=-1).clamp_min(1e-6).to(value.dtype)
                edited=value.clone();edited[:,-1]=edited[:,-1]+row*norm[:,None]
            elif site.mode=='shared':
                norm=value.float().flatten(1).norm(dim=-1).clamp_min(1e-6).to(value.dtype)
                edited=value+row[:,None,:]*norm[:,None,None]/np.sqrt(value.shape[1])
            else:
                norm=value.float().flatten(1).norm(dim=-1).clamp_min(1e-6).to(value.dtype)
                edited=value+(row*norm[:,None]).reshape_as(value)
            tree=_replace(tree,site.path,edited)
            return tree['args'],tree['kwargs']
        handle=module.register_forward_pre_hook(hook,with_kwargs=True)
        try:result=self.forecast(x,prediction_length)
        finally:handle.remove()
        if counts[1]!=len(selected):raise SteeringUnavailable('selected invocations did not all execute exactly once')
        return result
