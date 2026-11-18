"""TimesFM 2.5 point-head control using the repo adapter's native quantiles.

Only the final context token is edited and only a single output patch is
supported. The auxiliary long-horizon spread head is not used by this repo's
adapter and is not a target here. Running normalization is replayed using the
installed library's own routine, then checked against the full decode path.
"""

from contextlib import contextmanager

import numpy as np
import torch

from .native import BoltHead


class TimesFMHead(BoltHead):
    """Native [horizon, nine quantiles] head with explicit scope checks."""

    def __init__(self, checkpoint='google/timesfm-2.5-200m-pytorch', horizon=32, device='cpu'):
        from ..config import ModelConfig, DataConfig
        from ..models.timesfm_adapter import TimesFMAdapter
        self.device=torch.device(device)
        self.adapter=TimesFMAdapter(ModelConfig(name='timesfm',checkpoint=checkpoint),
                                    DataConfig(context_len=128,horizon=horizon),self.device,torch.float32)
        self.adapter.ensure_loaded()
        self.model=self.adapter.module.eval()
        self.model.requires_grad_(False)
        self.head=self.model.output_projection_point
        self.quantiles=np.asarray(self.model.config.quantiles)
        self.horizon=horizon
        self.native_horizon=int(self.model.o)
        if not 1 <= horizon <= self.native_horizon or self.model.q != len(self.quantiles)+1:
            raise ValueError('expected one native output patch with extra point channel plus quantiles')

    def forward(self, hidden):
        """Replay actual point projection and select the adapter's native quantiles."""
        raw=self.head(hidden)
        return raw.reshape(*hidden.shape[:-1],self.native_horizon,self.model.q)[...,:self.horizon,1:]

    def _normalization(self, batch):
        """Use the same cumulative patch statistics as TimesFM.decode."""
        from timesfm.torch.util import update_running_stats
        n=mu=sigma=torch.zeros(len(batch),device=batch.device)
        for patch in batch.split(self.model.p,dim=-1):
            (n,mu,sigma),_=update_running_stats(n,mu,sigma,patch,torch.zeros_like(patch,dtype=torch.bool))
        return mu[:,None],sigma[:,None]

    def capture(self, contexts, batch_size=16):
        """Audit direct replay against model.decode for every captured batch."""
        x=np.asarray(contexts,dtype=np.float32)
        if x.ndim != 2 or x.shape[1] % self.model.p or not np.isfinite(x).all():
            raise ValueError('need finite complete-patch contexts')
        states,forecasts,locations,scales=[],[],[],[]
        max_error=0.
        for start in range(0,len(x),batch_size):
            batch=self.tensor(x[start:start+batch_size])
            hits=[]
            handle=self.head.register_forward_pre_hook(
                lambda module,args:hits.append(args[0].detach().clone()))
            try:
                with torch.no_grad():
                    outputs,_,extra=self.model.decode(self.horizon,batch,torch.zeros_like(batch,dtype=torch.bool))
            finally:
                handle.remove()
            if len(hits)!=1 or hits[0].ndim!=3 or hits[0].shape[:2]!=(len(batch),x.shape[1]//self.model.p) or extra is not None:
                raise RuntimeError('unexpected native TimesFM head capture geometry')
            full=outputs[:,-1,:self.horizon,1:]
            with torch.no_grad():
                loc,scale=self._normalization(batch)
                norm=self.forward(hits[0][:,-1])
                replay=norm*scale[:,None]+loc[:,None]
            error=float((replay-full).abs().max())
            max_error=max(max_error,error)
            if not torch.allclose(replay,full,atol=2e-6,rtol=2e-6):
                raise RuntimeError(f'TimesFM native replay mismatch: {error}')
            states.append(hits[0][:,-1].cpu().numpy())
            forecasts.append(norm.cpu().numpy())
            locations.append(loc.cpu().numpy())
            scales.append(scale.cpu().numpy())
        return dict(hidden=np.concatenate(states),normalized=np.concatenate(forecasts),
                    location=np.concatenate(locations),scale=np.concatenate(scales),replay_max_abs=max_error)

    @contextmanager
    def _addition(self, delta):
        """Add only at the native read position and remove the hook on every exit."""
        change=self.tensor(delta)
        hits=[]
        def hook(module,args):
            value=args[0]
            if value.ndim!=3 or value[:,-1].shape!=change.shape:
                raise ValueError('delta must match [batch,hidden] exactly')
            edited=value.clone()
            edited[:,-1]=edited[:,-1]+change
            hits.append(1)
            return (edited,)
        handle=self.head.register_forward_pre_hook(hook)
        try:
            yield hits
        finally:
            handle.remove()

    def predict_edited(self, contexts, delta):
        """Full decode with an internal addition at the final context token."""
        batch=self.tensor(contexts)
        with self._addition(delta) as hits,torch.no_grad():
            output,_,extra=self.model.decode(self.horizon,batch,torch.zeros_like(batch,dtype=torch.bool))
        if len(hits)!=1 or extra is not None:
            raise RuntimeError('expected one native intervention without autoregressive extension')
        return output[:,-1,:self.horizon,1:].cpu().numpy()
