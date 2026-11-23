"""Capture Chronos-2 context tokens in the original literal-atom series order."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import CACHE as ORIGINAL
from research.activation_transfer.boundary.capture import ALL_MODELS, adapter_for, digest, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.extraction.hooks import ActivationCatcher

HERE = Path(__file__).resolve().parent
CACHE = Path('/tmp/tsfm_third_atoms_20261120')


def collect(cache, original):
    cache.mkdir(parents=True, exist_ok=True)
    with np.load(original/'data.npz') as d:
        x, ids = d['x'], d['ids']
    model = 'chronos2'
    enc = adapter_for(model)
    if enc.data_cfg.horizon != 64:
        raise ValueError('Chronos-2 archived SAE expects a 64-step prediction path')
    layer = ALL_MODELS[model][-1]
    spans = enc.token_time_spans()
    positions = np.flatnonzero(spans[:,0] >= 448)
    np.testing.assert_array_equal(positions, [28,29])
    np.testing.assert_array_equal(spans[positions], [[448,464],[464,480]])
    def capture(rows):
        return capture_raw_tokens(enc,rows,[layer])[layer].numpy()[:,positions].copy()
    # Cross-check the capture with the block's live output during the actual forecast API.
    with ActivationCatcher(enc.module,[layer]) as catcher:
        enc.predict(x[:6],64,[.5])
        live=catcher.collect()[layer].detach().cpu().numpy()
    select=last_window_slice(enc,positions); chosen=select(live.shape[1])
    captured=capture(x[:24])[:6]
    discrepancy=float(np.max(abs(live[:,chosen]-captured)))
    np.testing.assert_allclose(live[:,chosen],captured,atol=2e-5,rtol=2e-5)
    tokens=np.concatenate([capture(x[i:i+24]) for i in range(0,len(x),24)])
    np.savez_compressed(cache/'chronos2.npz',tokens=tokens,positions=positions,ids=ids,
                        sae_sha256=digest(dictionary_path(model)))
    metadata=dict(model=model,checkpoint_revision=ALL_MODELS[model][-2],layer=layer,forecast_horizon=64,
                  score_horizon=8,live_tokens=live.shape[1],n_context=len(spans),absolute_time_spans=spans[positions].tolist(),
                  live_selected=[chosen.start,chosen.stop],max_live_state_error=discrepancy,
                  source_manifest='../results/manifest.json',sae_sha256=digest(dictionary_path(model)))
    (HERE/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--original',type=Path,default=ORIGINAL)
    a=parser.parse_args(); torch.set_num_threads(3); collect(a.cache,a.original)
