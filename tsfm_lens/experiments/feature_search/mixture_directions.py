"""Train-only generative patterns and discriminative phase filters.

Haufe et al. (2014), NeuroImage87, Eq6/8, distinguishes filters from patterns:
A=Cov(H) W Cov(HW)^-1. The factor covariance is that of PREDICTED factors.
https://doi.org/10.1016/j.neuroimage.2013.10.067

Patterns describe linear representation associations; using them as causal
edit directions still needs matched donor, nuisance and generic-PC controls.
Physical token starts align local phase coordinates across different patches.
Special tokens without physical time must be analyzed separately.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd


def semantic_design(metadata, token_starts, *, amplitude_weighted=True):
    """Create local two-frequency coordinates and explicit amplitude nuisances.

    Relative amplitudes a/scale,b/scale follow the known normalized-input
    generator, scale=sqrt((a²+b²)/2). They are provided corpus quantities, not
    estimates using confirmation activations. Raw phase decoder controls can
    request unweighted sine/cosine coordinates.
    """
    starts = np.asarray(token_starts, dtype=float)
    if starts.ndim != 1 or len(starts) == 0 or not np.isfinite(starts).all():
        raise ValueError('Need one finite physical start for each canonical token')
    phi, psi, a, b, offset, scale = (np.array([m[key] for m in metadata], dtype=float) for key in ('phi','psi','a','b','offset','scale'))
    if not all(np.isfinite(v).all() for v in (phi,psi,a,b,offset,scale)) or np.any(scale <= 0):
        raise ValueError('Need finite phase/amplitude metadata with positive scale')
    p = phi[:,None]+2*np.pi*starts[None,:]/32
    q = psi[:,None]+2*np.pi*starts[None,:]/16
    first = a[:,None]/scale[:,None] if amplitude_weighted else np.ones((len(phi),1))
    second = b[:,None]/scale[:,None] if amplitude_weighted else np.ones((len(phi),1))
    labels = np.stack((first*np.sin(p),first*np.cos(p),second*np.sin(q),second*np.cos(q)), axis=-1)
    nuisance = np.broadcast_to(np.column_stack((offset/scale,a/scale,b/scale))[:,None,:], (len(phi),len(starts),3))
    return labels, nuisance


def _frame(matrix, rank):
    """Return only numerically supported channel directions, never arbitrary QR fill."""
    matrix = np.asarray(matrix, dtype=float)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    actual = min(rank, int(np.sum(singular > max(float(singular[0]) if len(singular) else 0, 1.)*1e-10)))
    if actual == 0:
        raise ValueError('No supported semantic direction in training activations')
    return u[:,:actual]


def _residualize(h, target, nuisance):
    """Condition both factors and activations on the same training nuisance design."""
    design = np.column_stack((np.ones(len(h)), nuisance))
    residual_h = h-design@np.linalg.lstsq(design,h,rcond=None)[0]
    residual_target = target-design@np.linalg.lstsq(design,target,rcond=None)[0]
    return residual_h, residual_target


def _decoder(h, target, alpha):
    """Fit standardized ridge and express its filters in raw channel coordinates."""
    scale = h.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    model = Ridge(alpha=alpha, fit_intercept=False).fit(h/scale,target)
    return model.coef_.T/scale[:,None]


def haufe_patterns(h, decoder):
    """Compute Eq6 without allocating a channel-by-channel covariance matrix.

    Inputs are centered/residualized TRAIN rows. Pseudoinversion handles
    dependent predicted factors; the caller records actual pattern rank.
    """
    h, decoder = np.asarray(h,dtype=float), np.asarray(decoder,dtype=float)
    if h.ndim != 2 or decoder.ndim != 2 or h.shape[1] != decoder.shape[0] or len(h) < 2:
        raise ValueError('Need aligned activation rows and raw-coordinate decoder')
    h = h-h.mean(axis=0)
    prediction = h@decoder
    covariance = prediction.T@prediction/(len(h)-1)
    return (h.T@prediction/(len(h)-1))@np.linalg.pinv(covariance,rcond=1e-10)


def _target_contrasts(tokens, metadata, target):
    """Average matched nuisance cells before taking target donor differences."""
    target_key = 'cell_'+target
    other_key = 'cell_'+('psi' if target == 'phi' else 'phi')
    lookup = {}
    for i,row in enumerate(metadata):
        key = (row['seed'],row[target_key],row[other_key])
        if key in lookup:
            raise ValueError('Duplicate generating-seed factorial cell')
        lookup[key] = i
    differences = []
    for seed in sorted({row['seed'] for row in metadata}):
        if any((seed,a,b) not in lookup for a in (0,1) for b in (0,1)):
            raise ValueError('Training split must retain complete four-cell generating bundles')
        absent = (tokens[lookup[seed,0,0]]+tokens[lookup[seed,0,1]])/2
        present = (tokens[lookup[seed,1,0]]+tokens[lookup[seed,1,1]])/2
        differences.append(present-absent)
    return np.stack(differences).reshape(-1,tokens.shape[-1])


def fit_direction_frames(tokens, metadata, token_starts, *, target='phi', alpha=1., train_mask=None, seed=73):
    """Fit conditional encoding/Haufe, raw phase filter and paired-SVD controls.

    Only complete TRAIN bundles enter any regression, standardization, covariance
    or SVD. No layer/rank selection occurs here. Canonical channel frames apply
    per token; the driver selects candidates on validation and tests once.
    """
    tokens = np.asarray(tokens)
    if target not in ('phi','psi') or tokens.ndim != 3 or len(tokens) != len(metadata) or tokens.shape[1] != len(token_starts):
        raise ValueError('Need aligned canonical [series,physical_token,channel] activations and target phi/psi')
    train = np.array([row['split'] == 'train' for row in metadata]) if train_mask is None else np.asarray(train_mask)
    if train.dtype != bool or train.shape != (len(tokens),) or not train.any() or not np.isfinite(tokens[train]).all() or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Need finite training activations, aligned boolean train mask and positive alpha')
    for identity in {row['seed'] for row in metadata}:
        membership = train[np.array([row['seed'] == identity for row in metadata])]
        if membership.any() and not membership.all():
            raise ValueError('A generating-seed bundle crosses the training boundary')
    labels, extra = semantic_design(metadata,token_starts)
    raw_labels, _ = semantic_design(metadata,token_starts,amplitude_weighted=False)
    selection = [0,1] if target == 'phi' else [2,3]
    other = [2,3] if target == 'phi' else [0,1]
    h = tokens[train].reshape(-1,tokens.shape[-1]).astype(float)
    y, raw, extra = (value[train].reshape(-1,value.shape[-1]) for value in (labels,raw_labels,extra))
    residual_h, residual_y = _residualize(h,y[:,selection],np.column_stack((y[:,other],extra)))
    native_h, native_y = _residualize(h,raw[:,selection],np.column_stack((raw[:,other],extra)))
    encoding = np.linalg.lstsq(residual_y,residual_h,rcond=None)[0].T
    decoder = _decoder(residual_h,residual_y,alpha)
    native_decoder = _decoder(native_h,native_y,alpha)
    patterns = haufe_patterns(residual_h,decoder)
    differences = _target_contrasts(tokens[train].astype(float),[m for m,keep in zip(metadata,train) if keep],target)
    _, singular, vt = randomized_svd(differences,n_components=min(4,*differences.shape),random_state=seed)
    actual = int(np.sum(singular > max(float(singular[0]),1.)*1e-10))
    if actual == 0:
        raise ValueError('No paired target contrast in training activations')
    frames = dict(encoding2=_frame(encoding,2),haufe2=_frame(patterns,2),decoder2=_frame(native_decoder,2),
                  contrast2=vt[:min(2,actual)].T,contrast4=vt[:min(4,actual)].T)
    return dict(frames=frames,diagnostics=dict(target=target,alpha=float(alpha),training_seeds=len({m['seed'] for m,keep in zip(metadata,train) if keep}),
                physical_tokens=tokens.shape[1],ranks={key:value.shape[1] for key,value in frames.items()},
                contrast_singular_values=singular.tolist(),encoding_label_rank=int(np.linalg.matrix_rank(residual_y)),
                amplitude_weighted_methods=['encoding2','haufe2'],unweighted_methods=['decoder2'],
                conditioning='other-phase sine/cosine and offset/scale,a/scale,b/scale;training only',
                haufe='Cov(H) W pinv(Cov(HW));predicted-factor covariance',evidence='linear association;causal selectivity requires independent donor controls'))
