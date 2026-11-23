from types import SimpleNamespace

import numpy as np
import torch
from momentfm.models.layers.revin import RevIN

from research.activation_transfer.moment.dictionary_transfer.input_deltas.order.normalization.run import capture_fixed, prior


def test_edited_input_uses_clean_moments_and_preserves_untouched_samples(monkeypatch):
    normalizer = RevIN(num_features=1)
    def prepare(history):
        padded = torch.nn.functional.pad(torch.as_tensor(history), (32, 0))
        mask = torch.zeros_like(padded, dtype=torch.long)
        mask[:, 32:] = 1
        return padded[:, None], mask
    model = SimpleNamespace(module=SimpleNamespace(normalizer=normalizer), prepare=prepare)
    def read_normalized(adapter, history):
        x, mask = adapter.prepare(history)
        return adapter.module.normalizer(x=x, mask=mask, mode='norm').numpy()[:,0,32:]
    monkeypatch.setattr(prior, 'capture', read_normalized)
    clean = np.tile(np.arange(480, dtype='f4') % 5, (2,1))
    edit = clean.copy()
    edit[:, -8:] += np.linspace(-20,20,8)
    state, mu, sd, actual_sd = capture_fixed(model, clean, edit)
    expected = (edit - clean.mean(1)[:,None]) / (clean.std(1)[:,None] + normalizer.eps)
    np.testing.assert_allclose(state, expected, atol=2e-6)
    assert np.all(actual_sd > sd*1.2)
    np.testing.assert_allclose(mu, clean.mean(1))
