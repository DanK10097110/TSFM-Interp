"""Feature-space CKA between two models' baseline SAE dictionaries.

The consumer side of the encode-store seam (ROADMAP.md sec 6.2.1 Stage 3d):
once `train.py::encode_and_persist_features` has written encoded features
back into the store, this asks L1's own question -- do these two models
share representational geometry? -- of the *learned dictionaries* instead
of the raw residual stream. A genuinely different measurement, not a
replacement for `analysis/l1_geometry.py`'s activation-space CKA: two
models could share little raw-activation geometry yet have SAEs that
converge on similar interpretable directions, or the reverse.

Series-level only, unlike L1's window-level global pass -- a TopK
dictionary is typically 8x-16x wider than the activation it was trained on
(`sae.dict_size_mult` defaults to 8), so a window-level [N*W, F] pass here
would multiply L1's own memory footprint by that same factor for a first
pass at this measurement. Window-level is a natural, separately-scoped
follow-up if this one proves useful.

Same evidence-class caveat as L1 (CLAUDE.md sec 6.1): correlational,
evidences shared geometry, not shared mechanism.
"""

from __future__ import annotations

import numpy as np
import torch

from ..analysis.l1_geometry import linear_cka
from ..analysis.stats import bootstrap_ci
from ..config import PipelineConfig
from ..extraction.store import ActivationStore
from ..utils import log, save_json


def run_sae_feature_cka(cfg: PipelineConfig, store: ActivationStore, targets: list,
                        device: torch.device) -> dict | None:
    """Linear CKA between every (model_a target, model_b target) SAE feature-space pair.

    Returns `None` (and logs why) rather than raising when either
    comparison-pair model has no persisted feature space to compare --
    e.g. `sae.targets` covering only one model, or `encode_and_persist_features`
    failing for every target of one model (CLAUDE.md sec 2.5: skip and log,
    don't crash the whole `sae` stage over an optional, additive measurement).
    """
    if cfg.run_shape() == "solo":
        log.info("sae: feature-space CKA skipped -- solo run (1 model), and this "
                 "measurement compares two models' persisted feature spaces "
                 "(ROADMAP.md sec 24.3)")
        return None
    a, b = cfg.comparison_pair()
    targets_a = [t["layer"] for t in targets
                if t["model"] == a.name and store.has_sae_features(a.name, t["layer"])]
    targets_b = [t["layer"] for t in targets
                if t["model"] == b.name and store.has_sae_features(b.name, t["layer"])]
    if not targets_a or not targets_b:
        log.info(f"sae: feature-space CKA skipped -- no persisted SAE features for both "
                 f"{a.name} and {b.name} (targets_a={targets_a}, targets_b={targets_b})")
        return None

    n = store.root.attrs["n_series"]

    def bank(model: str, layers: list) -> dict:
        out = {}
        for layer in layers:
            arr = store.load(model, layer, level="series", rows=None, space="sae")
            out[layer] = torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)
        return out

    bank_a, bank_b = bank(a.name, targets_a), bank(b.name, targets_b)
    cka = np.zeros((len(targets_a), len(targets_b)), dtype=np.float32)
    for i, la in enumerate(targets_a):
        for j, lb in enumerate(targets_b):
            cka[i, j] = linear_cka(bank_a[la], bank_b[lb])
    best = np.unravel_index(int(cka.argmax()), cka.shape)

    ci = None
    if cfg.stats.enabled:
        xa, xb = bank_a[targets_a[best[0]]], bank_b[targets_b[best[1]]]

        def stat(idx: np.ndarray) -> float:
            sel = torch.from_numpy(idx).to(device)
            return linear_cka(xa[sel], xb[sel])

        ci = bootstrap_ci(stat, n, min(cfg.stats.n_boot, cfg.stats.n_boot_heavy),
                          cfg.run.seed + 17, cfg.stats.ci)

    result = {
        "model_a": a.name, "model_b": b.name,
        "targets_a": targets_a, "targets_b": targets_b,
        "cka": cka.tolist(),
        "best_pair": {"layer_a": targets_a[best[0]], "layer_b": targets_b[best[1]],
                      "cka": float(cka[best]), "ci": ci},
        "n_rows": int(n),
        "level": "series",
        "evidence_class": "correlational, feature-space (ROADMAP.md sec 6.2.1 Stage 3d) -- "
                          "shares L1's own limitation (CLAUDE.md sec 6.1): geometric "
                          "similarity of the learned dictionaries, not shared mechanism",
    }
    out_dir = cfg.run_dir() / "l1"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "cka_sae.json", result)
    log.info(f"sae: feature-space CKA peak={cka[best]:.3f} at "
             f"({targets_a[best[0]]}, {targets_b[best[1]]}) over {n} series")
    return result
