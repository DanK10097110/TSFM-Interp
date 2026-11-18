"""Independent dense reference for frozen midpoint support and rectifier controls."""
import numpy as np


def reference(h, a, method, mode, dtype):
    """Recompute central indices by full stable sort and decode dense coefficients."""
    a = {key: np.asarray(value, dtype=dtype) for key, value in a.items()}
    shape = h.shape
    x = (np.asarray(h, dtype=dtype) - a['center']) / a['scale']
    pre = x @ a['basis'] if method == 'pca_adaptive' else (x - a['b_dec']) @ a['W_enc'] + a['b_enc']
    activated = pre if method == 'pca_adaptive' else np.maximum(pre, 0)
    central = np.argsort(-abs(activated[2]), axis=-1, kind='stable')[..., :int(a['k'])]
    mask = np.zeros(pre.shape[1:], dtype=bool)
    np.put_along_axis(mask, central, True, axis=-1)
    if mode == 'hard':
        chosen = np.argsort(-abs(activated), axis=-1, kind='stable')[..., :int(a['k'])]
        codes = np.zeros_like(pre)
        np.put_along_axis(codes, chosen, np.take_along_axis(activated, chosen, axis=-1), axis=-1)
    elif mode == 'fixed_support':
        codes = activated * mask
    elif mode == 'fixed_affine':
        codes = pre * mask
        if method != 'pca_adaptive':
            codes *= pre[2] > 0
    else:
        raise ValueError(mode)
    result = codes @ (a['basis'].T if method == 'pca_adaptive' else a['W_dec'])
    if method != 'pca_adaptive':
        result += a['b_dec']
    return (result * a['scale'] + a['center']).reshape(shape)


def check(h, a, method, mode, expected64, expected32):
    """Check independent reference, deterministic repeats and arithmetic precision."""
    result = {}
    for dtype, label, expected, tolerance in ((np.float64, 'FP64', expected64, 1e-10),
                                             (np.float32, 'FP32', expected32, 5e-5)):
        actual = reference(h, a, method, mode, dtype)
        repeated = reference(h, a, method, mode, dtype)
        assert np.array_equal(actual, repeated), 'Reference repeat mismatch'
        error = float(np.max(abs(actual.astype(float) - expected)))
        limit = tolerance * max(1., float(np.sqrt(np.mean(actual.astype(float) ** 2))))
        assert error <= limit, (label, mode, error, limit)
        result[label] = dict(max_abs_error=error, tolerance=limit, repeat_exact=True)
    return result
