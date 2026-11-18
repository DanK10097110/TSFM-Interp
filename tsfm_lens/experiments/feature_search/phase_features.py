"""Phase representation/readout instrument, conditional on a behavioral gate.

The noiseless period32 design fixes each model's normalization state across
phase donors: four complete context periods, one period per TimesFM patch.
Probes are correlational; donor edits require the separately validated patch
driver and its equal-energy controls before any phase-specific causal claim.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROTOCOL = dict(seed_base=993000, n_pairs=192, period=32, context=128, horizon=32,
                split='seed index<96 train;96..143 validation;144..191 test',
                phase_difference=float(np.pi/2),noise=0,conditional_noise_replication=.1,
                amplitude=[.5,2],offset=[-2,2],slope=0,
                representation='final context token at first/middle/final layers',
                ridge_alpha=[.01,1,100],candidate_subspaces=['phase_probe_rank2','response_svd_rank2','response_svd_rank4'],
                selection='train fit;validation layer/rank/direction;frozen test',
                primary='phase-selective donor restoration vs equal-energy topPC/random',
                multiplicity='Holm4claims:2models*2direction;no phase claim from decode',
                nuisance=['relative_amplitude_change','offset_change_divided_by_latent_amplitude'],
                control='whole donor,self,zero,equal-energy topPC/random,rotated-neuron')


def save_protocol(path: Path):
    """Persist the frozen design before conditionally authorized extraction."""
    path.write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')


def make_phase_corpus(noise=0.):
    """Generate phase pairs with equal amplitude/offset and coupled noise draws."""
    contexts, future, records=[],[],[]
    t=np.arange(160)
    for index in range(PROTOCOL['n_pairs']):
        rng=np.random.default_rng(PROTOCOL['seed_base']+index)
        phase=rng.uniform(-np.pi,np.pi);amplitude=rng.uniform(.5,2);offset=rng.uniform(-2,2)
        eps=rng.normal(0,noise,160)
        for direction in [0,1]:
            angle=phase+direction*np.pi/2
            latent=offset+amplitude*np.sin(2*np.pi*t/32+angle)
            contexts.append((latent+eps)[:128]);future.append(latent[128:])
            records.append(dict(seed=index,generator_seed=993000+index,split='train' if index<96 else 'validation' if index<144 else 'test',
                                dose=direction,phase=angle,amplitude=amplitude,offset=offset,noise=noise))
    return np.asarray(contexts,dtype=np.float32),np.asarray(future),records


def harmonic_fit(values, start=128):
    """Fit known-period sine, cosine and offset in a common global time frame."""
    t=np.arange(start,start+values.shape[-1]);design=np.column_stack([np.sin(2*np.pi*t/32),np.cos(2*np.pi*t/32),np.ones(len(t))])
    coefficients=np.linalg.lstsq(design,values.T,rcond=None)[0].T
    return dict(phase=np.arctan2(coefficients[:,1],coefficients[:,0]),
                amplitude=np.hypot(coefficients[:,0],coefficients[:,1]),offset=coefficients[:,2],coefficients=coefficients)


def fit_phase_probe(x, phase, train, validation, coordinate=False, rotation_seed=None):
    """Fit phase sine/cosine and return a raw-coordinate intervention rowspace."""
    y=np.column_stack([np.sin(phase),np.cos(phase)])
    coordinate_map=np.eye(x.shape[1])
    if rotation_seed is not None:
        q,_=np.linalg.qr(np.random.default_rng(rotation_seed).normal(size=(x.shape[1],x.shape[1])))
        x=x@q;coordinate_map=q
    selected_coordinate=None
    if coordinate:
        centered=x[train]-x[train].mean(0)
        covariance=centered.T@(y[train]-y[train].mean(0))
        score=np.linalg.norm(covariance,axis=1)/np.maximum(np.linalg.norm(centered,axis=0),1e-12)
        selected_coordinate=int(np.argmax(score));x=x[:,[selected_coordinate]]
        coordinate_map=coordinate_map[:,[selected_coordinate]]
    scaler=StandardScaler().fit(x[train]);z=scaler.transform(x)
    fits=[Ridge(alpha=alpha).fit(z[train],y[train]) for alpha in PROTOCOL['ridge_alpha']]
    loss=[np.mean((m.predict(z[validation])-y[validation])**2) for m in fits]
    chosen=int(np.argmin(loss));fit=fits[chosen]
    raw_coefficients=fit.coef_/scaler.scale_[None,:]
    raw_coefficients=raw_coefficients@coordinate_map.T
    frame,_=np.linalg.qr(raw_coefficients.T)
    rank=int(np.linalg.matrix_rank(raw_coefficients))
    return dict(prediction=fit.predict(z),frame=frame[:,:rank],raw_coefficients=raw_coefficients,
                validation_loss=float(loss[chosen]),alpha=PROTOCOL['ridge_alpha'][chosen],
                coordinate=selected_coordinate,rotation_seed=rotation_seed)


def phase_metrics(edited, receiver, donor, amplitude):
    """Measure edited donor-phase restoration and explicit nuisance changes.

    Restoration is reduction in circular phase error to the actual donor
    forecast, not assumed perfect π/2 behavior. Latent target errors are a
    separate behavioral quantity; no effect is divided by near-zero amount.
    """
    e,r,d=[harmonic_fit(z) for z in [edited,receiver,donor]]
    circular=lambda a,b:np.abs(np.angle(np.exp(1j*(a-b))))
    return dict(phase_restoration=circular(r['phase'],d['phase'])-circular(e['phase'],d['phase']),
                donor_phase_error=circular(e['phase'],d['phase']),
                receiver_phase_change=circular(e['phase'],r['phase']),
                amplitude_relative_change=np.abs(e['amplitude']-r['amplitude'])/amplitude,
                offset_change_normalized=np.abs(e['offset']-r['offset'])/amplitude)


def normalization_diagnostics(contexts):
    """Expose pairwise normalizer differences rather than assuming matched states."""
    patches=contexts.reshape(len(contexts),4,32)
    means=np.column_stack([contexts.mean(1),*[patches[:,:i].reshape(len(contexts),-1).mean(1) for i in range(1,5)]])
    scales=np.column_stack([contexts.std(1),*[patches[:,:i].reshape(len(contexts),-1).std(1) for i in range(1,5)]])
    return dict(mean_pair_max_abs=float(abs(means[1::2]-means[::2]).max()),
                std_pair_max_abs=float(abs(scales[1::2]-scales[::2]).max()))
