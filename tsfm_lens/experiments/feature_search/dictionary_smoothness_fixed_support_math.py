"""Same-weight, actual-midpoint gate controls for saved dictionary path geometry."""
import numpy as np
from experiments.feature_search.dictionary_smoothness_math import rms, top_indices


def coefficients(h, artifact, method, dtype=np.float64):
    """Keep signed pre-ReLU coefficients so fixed ReLU gates define an affine map."""
    h = np.asarray(h, dtype=dtype)
    a = {key: np.asarray(value, dtype=dtype) for key, value in artifact.items()}
    x = (h - a['center']) / a['scale']
    if method == 'pca_adaptive':
        return x @ a['basis']
    return (x - a['b_dec']) @ a['W_enc'] + a['b_enc']


def central_masks(pre, k, method):
    """Freeze exactly k indices per path/token at the saved actual-input midpoint."""
    central = pre[2]
    ranked = central if method == 'pca_adaptive' else np.maximum(central, 0)
    indices = np.argsort(-abs(ranked), axis=-1, kind='stable')[..., :k]
    gate = np.take_along_axis(central, indices, axis=-1) > 0
    return indices, gate


def decode_control(pre, artifact, method, mode, indices, gate):
    """Apply hard, frozen-support/ReLU, or frozen-support/central-ReLU mapping."""
    if mode not in ('hard', 'fixed_support', 'fixed_affine'):
        raise ValueError('Unknown frozen control')
    dtype = pre.dtype
    k = int(artifact['k'])
    flat = pre.reshape(-1, pre.shape[-1])
    if mode == 'hard':
        values = flat if method == 'pca_adaptive' else np.maximum(flat, 0)
        selected = top_indices(values, k)
        values = np.take_along_axis(values, selected, axis=-1)
    else:
        selected = np.broadcast_to(indices, pre.shape[:-1] + (k,)).reshape(-1, k)
        values = np.take_along_axis(flat, selected, axis=-1)
        if method != 'pca_adaptive':
            if mode == 'fixed_support':
                values = np.maximum(values, 0)
            else:
                values = values * np.broadcast_to(gate, pre.shape[:-1] + (k,)).reshape(-1, k)
    decoder = np.asarray(artifact['basis'].T if method == 'pca_adaptive' else artifact['W_dec'], dtype=dtype)
    output = np.empty((len(flat), decoder.shape[1]), dtype=dtype)
    for start in range(0, len(flat), 128):
        sl = slice(start, start + 128)
        output[sl] = np.einsum('nk,nkd->nd', values[sl], decoder[selected[sl]], optimize=False)
    if method != 'pca_adaptive':
        output += np.asarray(artifact['b_dec'], dtype=dtype)
    output *= dtype.type(artifact['scale'])
    output += np.asarray(artifact['center'], dtype=dtype)
    return output.reshape(pre.shape[:-1] + (decoder.shape[1],))


def fidelity(h, g, hard, reference):
    """State reconstruction and hidden-response errors, never forecast fidelity."""
    h, g, hard = (np.asarray(x, float) for x in (h, g, hard))
    axes = (-2, -1)
    state = np.sqrt(np.mean((g[:3] - h[:3]) ** 2, axis=(0, 2, 3)))
    raw_response = h[:2] - h[2]
    decoded_response = g[:2] - g[2]
    hard_response = hard[:2] - hard[2]
    response_error = np.sqrt(np.mean((decoded_response - raw_response) ** 2, axis=(0, 2, 3)))
    raw_response_rms = np.sqrt(np.mean(raw_response ** 2, axis=(0, 2, 3)))
    floor = np.maximum(np.finfo(np.float32).eps * reference, 1e-30)
    return dict(
        state_reconstruction_RMS=state, state_reconstruction_relative_RMS=state / reference,
        response_reconstruction_RMS=response_error,
        response_reconstruction_relative_RMS=response_error / np.maximum(raw_response_rms, floor),
        raw_response_RMS=raw_response_rms,
        response_RMS=np.sqrt(np.mean(decoded_response ** 2, axis=(0, 2, 3))),
        response_change_from_hard_RMS=np.sqrt(np.mean((decoded_response - hard_response) ** 2, axis=(0, 2, 3))),
        state_change_from_hard_RMS=np.sqrt(np.mean((g[:3] - hard[:3]) ** 2, axis=(0, 2, 3))),
        central_output_change_from_hard_RMS=rms(g[2] - hard[2], axes))


def source_summary(values, paths):
    """Equal-source aggregation of the balanced dependent paths."""
    ids = sorted({p['source'] for p in paths})
    means = [float(np.mean(values[[i for i, p in enumerate(paths) if p['source'] == source]])) for source in ids]
    return dict(source_ids=ids, n_sources=len(ids), per_source=means, mean=float(np.mean(means)))
