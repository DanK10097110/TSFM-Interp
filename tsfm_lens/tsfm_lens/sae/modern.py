"""Opt-in signed and batch-adaptive SAEs for controlled TSFM comparisons.

AbsTopK (Zhu et al., arXiv:2510.00404) motivates signed, magnitude-selected
codes. BatchTopK (Bussmann et al., arXiv:2412.06410) motivates a batch activity
budget and a fixed inference threshold. These small implementations reuse the
repo's decoder and AuxK mechanism; they are not reproductions of the papers'
complete training recipes. The existing pipeline's TopK defaults are untouched.

All three comparison arms use the same mean initialization, optimizer, decoder
normalization and training budget. Held-out reconstruction is a diagnostic,
not a substitute for causal steering and raw-neuron/random-basis controls.
"""

from __future__ import annotations

import numpy as np
import torch

from .models import TopKSAE, auxiliary_dead_loss


class AbsTopKSAE(TopKSAE):
    """Keep the k largest absolute preactivations, preserving both signs."""

    def sparsify(self, pre):
        """Magnitude selection avoids splitting positive/negative polarity by design."""
        indices = pre.abs().topk(min(self.k, pre.shape[-1]), dim=-1).indices
        result = torch.zeros_like(pre)
        return result.scatter(-1, indices, pre.gather(-1, indices))


class BatchTopKSAE(TopKSAE):
    """Batch activity budget in training; frozen scalar threshold in inference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('threshold', torch.tensor(float('nan')))

    def sparsify(self, pre):
        """Inference never depends on other examples sharing an evaluation batch."""
        positive = pre.relu()
        if not self.training:
            if not torch.isfinite(self.threshold):
                raise RuntimeError('calibrate the BatchTopK inference threshold on training data first')
            return positive * (positive > self.threshold)
        count = min(positive.numel(), self.k * positive.shape[0])
        flat = positive.flatten()
        values, indices = flat.topk(count)
        result = torch.zeros_like(flat)
        return result.scatter(0, indices, values).reshape_as(positive)

    def calibrate(self, train):
        """Freeze train-only threshold targeting mean activity k; ties may lower L0."""
        with torch.no_grad():
            values = self.pre_activations(train).relu().flatten()
            count = min(values.numel(), self.k*len(train))
            top = values.topk(min(count+1, values.numel())).values
            threshold = (top[count-1]+top[count])/2 if count < values.numel() else values.new_tensor(0.)
            self.threshold.copy_(threshold)


def train_modern_sae(activations, cfg, *, kind='topk'):
    """Train a matched-budget CPU comparison with explicit train-only centering.

    Accepts the existing SAETrainConfig. Checkpoints retain class kind separately.
    This research entry point does not change pipeline checkpoint loading.
    """
    classes = dict(topk=TopKSAE, abstopk=AbsTopKSAE, batchtopk=BatchTopKSAE)
    if kind not in classes:
        raise ValueError(f'unknown SAE kind: {kind}')
    x = torch.as_tensor(np.asarray(activations), dtype=torch.float32)
    if x.ndim != 2 or not x.numel() or not torch.isfinite(x).all():
        raise ValueError('training requires nonempty finite [rows,hidden] activations')
    size = cfg.dict_size or cfg.dict_size_mult*x.shape[1]
    if not 0 < cfg.k <= size or cfg.batch_size < 1 or cfg.lr <= 0:
        raise ValueError('invalid sparsity, batch size or learning rate')
    rng = torch.Generator().manual_seed(cfg.seed)
    model = classes[kind](x.shape[1],size,cfg.k,generator=rng)
    with torch.no_grad():
        model.b_dec.copy_(x.mean(0))
    opt = torch.optim.Adam(model.parameters(),lr=cfg.lr)
    per_epoch = (len(x)+cfg.batch_size-1)//cfg.batch_size
    epochs = max(cfg.epochs,(cfg.min_train_steps+per_epoch-1)//per_epoch)
    dead = torch.zeros(size,dtype=torch.long)
    history = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(x),generator=rng)
        total = 0.
        for ids in permutation.split(cfg.batch_size):
            batch=x[ids]
            reconstruction, code, pre=model.forward_with_pre(batch)
            mse=(reconstruction-batch).square().mean()
            aux=auxiliary_dead_loss(pre,batch-reconstruction,model.W_dec,
                                    dead >= cfg.aux_dead_steps,cfg.aux_k)
            loss=mse if aux is None else mse+cfg.aux_coef*aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{kind} training diverged at epoch {epoch}')
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_decoder_()
            dead=torch.where((code.detach().abs()>1e-8).any(0),0,dead+1)
            total+=mse.item()*len(ids)
        history.append(total/len(x))
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('nonfinite trained dictionary')
    if isinstance(model,BatchTopKSAE):
        model.calibrate(x)
    model.eval()
    model.train_meta=dict(kind=kind,initialization='train mean decoder bias',
                          epochs_run=epochs,n_optimizer_steps=epochs*per_epoch,
                          batch_size=cfg.batch_size,training_dead_fraction=float((dead>=cfg.aux_dead_steps).float().mean()),
                          inference='fixed train-calibrated threshold' if kind=='batchtopk' else 'per-example top k')
    return model,history


def load_steering_dictionary(path, device='cpu'):
    """Load a study checkpoint without confusing signed/adaptive codes with TopK.

    This is separate from the existing pipeline checkpoint format. Inference
    thresholds are buffers and must be restored for BatchTopK. Restricted torch
    loading accepts tensor/basic-data checkpoints without arbitrary pickles.
    """
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    kind=checkpoint.get('kind','legacy')
    classes=dict(legacy=TopKSAE,topk=TopKSAE,abstopk=AbsTopKSAE,batchtopk=BatchTopKSAE)
    if kind not in classes:
        raise ValueError(f'unsupported dictionary kind: {kind}')
    state=checkpoint['state_dict']
    width,size=state['W_enc'].shape
    model=classes[kind](width,size,checkpoint['config']['k'],
                        generator=torch.Generator().manual_seed(0)).to(device)
    model.load_state_dict(state,strict=True)
    model.eval()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('checkpoint has nonfinite dictionary parameters')
    if isinstance(model,BatchTopKSAE) and not torch.isfinite(model.threshold):
        raise ValueError('checkpoint lacks a calibrated inference threshold')
    return model
