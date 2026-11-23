"""Summarize fixed-direction experiments from the individual steering responses."""
import argparse
import json
from pathlib import Path

import numpy as np


def skill_terms(effects, reference):
    """Per-series numerator and denominator for skill relative to doing nothing."""
    return np.mean((effects - reference) ** 2, axis=-1), np.mean(reference ** 2, axis=-1)


def paired_score(loss, energy, draws):
    """Ratio of pooled energies and conditional series-resampling interval."""
    den = energy.mean()
    if den < 1e-15:
        return None
    sample_denom = energy[draws].mean(axis=1)
    estimates = 1 - loss[draws].mean(axis=1) / np.maximum(sample_denom, 1e-15)
    return dict(value=float(1 - loss.mean() / den), ci95=np.quantile(estimates, [.025,.975]).tolist())


def template_removed_skill(estimate, target, draws):
    """Re-estimate the across-item response template on every bootstrap resample."""
    residual = estimate - target

    def variance(a, groups):
        return (np.mean(a[groups]**2, axis=(-2, -1))
                - np.mean(a[groups].mean(axis=-2)**2, axis=-1))

    all_items = np.arange(len(target))
    energy = variance(target, all_items)
    if energy < 1e-15:
        return None
    values = 1 - variance(residual, draws)/np.maximum(variance(target, draws), 1e-15)
    return dict(value=float(1 - variance(residual, all_items)/energy),
                ci95=np.quantile(values, [.025, .975]).tolist())


def agreement(effect, reference):
    """Keep gain, direction, magnitude and signed level bias distinct."""
    def cosine(a,b):
        scale = np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.sum(a*b)/scale) if scale > 1e-12 else None
    ref_energy = np.sum(reference**2)
    return dict(shape_cosine=cosine(effect-effect.mean(1,keepdims=True),reference-reference.mean(1,keepdims=True)),
                cosine=cosine(effect,reference),
                gain=float(np.sum(reference*effect)/ref_energy) if ref_energy>1e-12 else None,
                mapped_rms=float(np.sqrt(np.mean(effect**2))), reference_rms=float(np.sqrt(np.mean(reference**2))),
                mapped_level=float(effect.mean()), reference_level=float(reference.mean()))


def measure(target_effects, source_reference, draws):
    """Compare with the recipient-native intervention and the source-native behavior."""
    target, mapped, *shams = target_effects.transpose(1,0,2)
    loss, energy = skill_terms(mapped, target)
    sham_loss = np.stack([skill_terms(sham,target)[0] for sham in shams]).mean(0)
    result = dict(target_native=agreement(mapped,target), source_native=agreement(mapped,source_reference),
                  native_cross_model=agreement(target,source_reference),
                  target_skill=paired_score(loss,energy,draws), sham_skill=paired_score(sham_loss,energy,draws),
                  source_skill=paired_score(*skill_terms(mapped,source_reference),draws),
                  template_removed_skill=template_removed_skill(mapped,target,draws))
    # A difference in relative skill uses the same target energies on each bootstrap draw.
    contrast = 1 - (loss-sham_loss)[draws].mean(1)/np.maximum(energy[draws].mean(1),1e-15)
    result["advantage_over_shams"] = dict(value=float((sham_loss-loss).mean()/energy.mean()),
                                          ci95=(np.quantile(contrast,[.025,.975])-1).tolist())
    return result, (loss, sham_loss, energy)


def summarize(cache, destination):
    """Save a small results table and an auditable compressed per-item forecast-effect array."""
    metadata = json.loads((cache/"metadata.json").read_text())
    rows, pooled, arm_data = [], {}, {}
    for depth_index, depth in enumerate(metadata["depths"]):
        for domain_index, domain in enumerate(["daily","weekly"]):
            size = len(metadata["series_ids"][domain])
            section = slice(domain_index*size,(domain_index+1)*size)
            draws = np.random.default_rng(202611204).integers(size,size=(4000,size))
            contrasts = []
            for cue in metadata["cues"]:
                batch = {}
                for model in metadata["models"]:
                    key = f"d{depth_index}_{model}_{cue}"
                    all_effects = np.load(cache/(key+".npz"))["effects"]
                    batch[model] = all_effects[section].astype(float)
                    if domain_index == 0:
                        arm_data[key] = all_effects
                for target in metadata["models"]:
                    source = next(name for name in metadata["models"] if name != target)
                    result, terms = measure(batch[target],batch[source][:,0,:],draws)
                    rows.append(dict(domain=domain,depth=depth,source=source,target=target,cue=cue,**result))
                    loss, sham_loss, energy = terms
                    contrasts.append((loss,sham_loss,energy))
            cells = np.stack(contrasts)  # arm × term × series
            def scores(selection):
                means = cells[:,:,selection].mean(axis=-1)
                skills = 1-means[:,0]/np.maximum(means[:,2],1e-15)
                gains = (means[:,1]-means[:,0])/np.maximum(means[:,2],1e-15)
                return np.stack([skills.mean(axis=0),gains.mean(axis=0)])
            aggregate = scores(np.arange(size))
            intervals = np.quantile(scores(draws),[.025,.975],axis=-1)
            pooled[f"{domain}_{depth}"] = dict(zip(["skill","advantage_over_shams"],
                [dict(value=float(v),ci95=intervals[:,i].tolist()) for i,v in enumerate(aggregate)]))
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"results.json").write_text(json.dumps(dict(**metadata,rows=rows,pooled=pooled),indent=2,allow_nan=False)+"\n")
    # NumPy's losslessly compressed archive is independent of local model caches.
    np.savez_compressed(destination/"effects.npz", **arm_data)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    summarize(args.cache,args.destination)
