"""Patch matched compact dictionary approximations of dense transported atom removals."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from scipy.linalg import helmert

HERE = Path(__file__).resolve().parent
ATOMS = HERE.parent / 'atoms'
SIZES = (1, 4, 16)
BASES = ('pca', 'isotropic', 'activation_covariance', 'decoder_covariance')
SEED = {'bolt': 2026112024, 'timesfm': 2026112025}


def unit_rows(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-15)


def scramble_rows(atoms, rng):
    """An orthogonal row rotation fixing ones preserves mean and centered covariance."""
    n = len(atoms)
    h = helmert(n)
    q, r = np.linalg.qr(rng.normal(size=(n-1, n-1)))
    q *= np.sign(np.diag(r))
    return atoms.mean(0, keepdims=True) + h.T @ (q @ (h @ atoms))


def covariance_error(a, b):
    """Relative Frobenius distance between centered covariance matrices, common n."""
    a, b = a-a.mean(0), b-b.mean(0)
    aa, bb, ab = np.linalg.norm(a@a.T)**2, np.linalg.norm(b@b.T)**2, np.linalg.norm(a@b.T)**2
    return np.sqrt(max(0., aa+bb-2*ab) / max(aa, 1e-30))


def make_bases(states, decoders, seed):
    x = states.astype(float) - states.mean(axis=0, dtype=float)
    decoders = decoders.astype(float)
    n, width = decoders.shape
    _, s, vh = np.linalg.svd(x, full_matrices=False)
    if n > np.sum(s>s[0]*1e-7):
        raise ValueError('Requested cardinality exceeds numerical rank of clean PCA basis')
    streams = [np.random.default_rng(r) for r in np.random.SeedSequence(seed).spawn(3)]
    rotated = scramble_rows(decoders, streams[2])
    covariance = (streams[1].normal(size=(n, len(s))) * (s/np.sqrt(len(x)-1))) @ vh
    bases = dict(pca=vh[:n], isotropic=streams[0].normal(size=(n,width)),
                 activation_covariance=covariance, decoder_covariance=rotated)
    return bases, dict(seed=seed, n_eligible=n,
                       direct_covariance_relative_error=covariance_error(decoders,rotated),
                       unit_covariance_relative_error=covariance_error(unit_rows(decoders),unit_rows(rotated)),
                       mean_max_error=float(np.max(np.abs(decoders.mean(0)-rotated.mean(0)))))


def approximations(directions, basis, solver):
    approximated = {k: [] for k in SIZES}
    for d in directions:
        result = solver(d, basis, np.arange(len(basis)))
        for k in SIZES:
            approximated[k].append(result[k][0])
    return {k: np.stack(rows) for k, rows in approximated.items()}


def vector_metrics(w, pred):
    return dict(cosine=np.sum(unit_rows(w)*unit_rows(pred),axis=1).tolist(),
                recovery=(1-np.sum((pred-w)**2,axis=1)/np.sum(w*w,axis=1)).tolist())


def collect(source, cache):

    from research.activation_transfer.atoms.intervene import run_arm
    from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
    from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
    from tsfm_lens.sae.train import load_sae_checkpoint
    from research.activation_transfer.atoms.vectors import approximate
    import torch
    torch.set_num_threads(3)
    target = next(m for m in MODELS if m != source)
    with np.load(cache / 'data.npz') as f:
        train = f['split'] == 'train'; ids = f['ids']; x = f['x'][~train]
    with np.load(cache / (source + '_vectors.npz')) as f:
        np.testing.assert_array_equal(ids, f['ids'])
        expected = int(f['target_eligible']); strength = f['source_codes'][~train]
        dense = f['target_dense']; source_atoms = f['source_atoms']; sha = str(f['target_sae_sha'])
        sae_vectors = {k: f['target_omp'+str(k)] for k in SIZES}
    with np.load(cache / (target + '.npz')) as f:
        np.testing.assert_array_equal(ids, f['ids'])
        tokens = f['tokens']; positions = f['positions']
    path = dictionary_path(target)
    with path.open('rb') as reader:
        if file_digest(reader,'sha256').hexdigest() != sha:
            raise ValueError('Target SAE weights do not match original execution')
    sae = load_sae_checkpoint(str(path))
    states = tokens[train].mean(axis=1)
    with torch.inference_mode():
        codes = sae.encode(torch.from_numpy(states)).numpy()
        decoders = sae.W_dec.numpy().copy()
    eligible = np.flatnonzero((codes>0).mean(0) >= .01)
    if len(eligible) != expected:
        raise ValueError('Original dictionary eligibility changed')
    runs = {'sae': approximations(dense, decoders[eligible], approximate)}
    for k in SIZES:
        np.testing.assert_allclose(runs['sae'][k], sae_vectors[k], atol=1e-10, rtol=1e-8)
    bases, metadata = make_bases(states, decoders[eligible], SEED[target])
    for name,basis in bases.items():
        runs[name] = approximations(dense, basis, approximate)
    metadata['vector_metrics'] = {name: {str(k): vector_metrics(dense,v) for k,v in group.items()}
                                  for name,group in runs.items()}
    metadata['target'] = target; metadata['source'] = source
    metadata['source_atoms'] = source_atoms.tolist()
    metadata['target_sae_sha256'] = sha
    enc = adapter_for(target); layer = MODELS[target][-1]
    selector = last_window_slice(enc, positions); t = tokens[~train]
    baseline = change_forecasts(enc, layer, x, t, np.zeros((len(x),t.shape[-1]),'f4'), selector, 24)
    effects = {}
    for name in BASES:
        for k in SIZES:
            effects[name+'_'+str(k)] = run_arm(enc, layer, selector, x, t, baseline, strength>0,
                                               strength, runs[name][k], 24)
            print(source, target, name, k, flush=True)
    np.savez_compressed(HERE / ('dictionary_'+source+'_effects.npz'),ids=ids[~train],source_atoms=source_atoms,**effects)
    return metadata


def analyze(source, metadata):

    from research.activation_transfer.atoms.scores import aggregate, resampling_counts
    with np.load(ATOMS/'results'/(source+'_effects.npz')) as f:
        target = f['target_dense']; active = f['active']; split = f['split']; ids = f['ids']
        predictions = {'sae_'+str(k): f['target_omp'+str(k)] for k in SIZES}
    with np.load(HERE / ('dictionary_'+source+'_effects.npz')) as f:
        np.testing.assert_array_equal(ids, f['ids'])
        np.testing.assert_array_equal(metadata['source_atoms'], f['source_atoms'])
        predictions.update({n:f[n] for n in f.files if n not in ('source_atoms','ids')})
    out = {}
    for name in ('daily','electricity'):
        use = split==name; n=int(use.sum()); boot=resampling_counts(n)
        group = {}
        for center in (False, True):
            transform = lambda z: z-z.mean(axis=-1,keepdims=True) if center else z
            reference = transform(target[use]); scores={}; draws={}
            for key,p in predictions.items():
                per, avg, sam = aggregate(active[use],transform(p[use]),reference,boot)
                draws[key] = sam['skill']
                scores[key] = dict(recovery=float(avg['skill']), cosine=float(avg['cosine']),
                    recovery_ci95=np.nanquantile(sam['skill'],[.025,.975]).tolist(),
                    atom_recovery=[float(y) if np.isfinite(y) else None for y in per['skill']],
                    atom_cosine=[float(y) if np.isfinite(y) else None for y in per['cosine']])
            differences={}
            for base in BASES:
                for k in SIZES:
                    orig,other='sae_'+str(k),base+'_'+str(k)
                    delta = draws[orig]-draws[other]
                    differences[other]=dict(recovery=float(scores[orig]['recovery']-scores[other]['recovery']),
                        ci95=np.nanquantile(delta,[.025,.975]).tolist(),
                        simultaneous_lower=float(np.nanquantile(delta,.00625)) if not center and k==16 and base in ('pca','decoder_covariance') else None)
            group['shape' if center else 'full']=dict(methods=scores,sae_minus_alternative=differences)
        out[name]=dict(active_count=active[use].sum(0).tolist(),**group)
    return dict(**metadata, results=out)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',choices=['bolt','timesfm'])
    parser.add_argument('--cache',type=Path,default=Path('/tmp/tsfm_atoms_20261120'))
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    dest=HERE/('dictionary_'+args.source+'_results.json')
    if args.analyze_only:
        metadata=json.loads(dest.read_text()); metadata.pop('results')
    else:
        metadata=collect(args.source,args.cache)
    dest.write_text(json.dumps(analyze(args.source,metadata),indent=2,allow_nan=False)+'\n')
