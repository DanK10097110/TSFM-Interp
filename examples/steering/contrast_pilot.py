"""Fit one hidden trend contrast per checkpoint, then intervene on unseen paired histories."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tsfm_lens.config import load_config
from tsfm_lens.pipeline import Context
from tsfm_lens.steering import ActivationContrast, SteeringModel
from tsfm_lens.utils import set_seed


def histories(seed, count):
    rng = np.random.default_rng(seed)
    t = np.arange(64)
    base = np.array([rng.normal(0, .3, 64) + np.sin(t * rng.uniform(.1, .5) + rng.uniform(0, 6))
                     for _ in range(count)])
    # Opposite slopes, identical recent noise and last observed value.
    ramp = np.r_[np.zeros(32), np.linspace(0, -1, 16), np.linspace(-1, 0, 16)]
    return base + ramp, base - ramp


def effects(contrast, x, baseline, mode, dose):
    forecast = contrast.predict(x, mode=mode, strength=dose)
    delta = forecast - baseline
    t = np.linspace(-.5, .5, forecast.shape[1])
    tilt = delta[:, :, 1] @ t / (t @ t)
    rising, falling = np.split(tilt, 2)
    # Pair is the independent unit for intervals. A constant upward tilt cancels here.
    return dict(tilt_bias=(rising + falling) / 2, sensitivity=(rising - falling) / 2,
                level_shift=delta[:, :, 1].mean(axis=1).reshape(2, -1).mean(axis=0),
                width_shift=(delta[:, :, 2] - delta[:, :, 0]).mean(axis=1).reshape(2, -1).mean(axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(Path(__file__).with_name("all_types.yaml"))
    set_seed(731)
    ctx = Context(cfg)
    train = histories(42, 32)
    x = np.concatenate(histories(43, 32)).astype(np.float32)
    draws = np.random.default_rng(44).integers(32, size=(4000, 32))
    groups = []
    readouts = []
    for spec in cfg.models:
        adapter = ctx.hub.get(spec.name)
        model = SteeringModel.from_adapter(adapter, quantiles=[.1, .5, .9])
        layers = adapter.all_layer_names()
        layer = layers[len(layers)//2]
        profile = ActivationContrast.fit(model, *train, layer=layer, prediction_length=8)
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "profile.npz"
            profile.save(file)
            profile = ActivationContrast.load(file, model)
        up, down = np.split(profile.score(x), 2)
        readouts.append(dict(model=spec.name, positive_above_negative=int(np.sum(up > down)),
                             differences=(up - down).tolist()))
        baseline = np.stack([model.forecast(row[None], 8)[0] for row in x])
        np.testing.assert_allclose(profile.predict(x, strength=0), baseline, rtol=1e-5, atol=1e-6)
        unit = np.random.default_rng(45).normal(size=profile.difference.shape)
        control = replace(profile, difference=unit * np.linalg.norm(profile.difference) / np.linalg.norm(unit))
        for mode in ("add", "amplify"):
            for label, vector, dose in (("learned", profile, -1.), ("learned", profile, .5),
                                        ("learned", profile, 1.), ("random", control, 1.)):
                values = effects(vector, x, baseline, mode, dose)
                stats = {k: dict(mean=float(v.mean()), ci95=np.quantile(v[draws].mean(axis=1), [.025,.975]).tolist(),
                                 paired_effects=v.tolist()) for k,v in values.items()}
                groups.append(dict(model=spec.name, layer=layer, mode=mode, vector=label, strength=dose, **stats))
        ctx.hub.release(spec.name)
    args.out.write_text(json.dumps(dict(training_pairs=32, evaluation_pairs=32, readouts=readouts, groups=groups), indent=2) + "\n")


if __name__ == "__main__":
    main()
