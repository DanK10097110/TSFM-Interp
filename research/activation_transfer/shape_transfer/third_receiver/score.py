"""Score Timer's live eight-lead responses against the observed donor SAE removals."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.shape_transfer.third_receiver.run import ARMS, HERE


def summarize(archive):
    report = {}
    for label in ('daily', 'machines'):
        rows = archive['split'] == label
        mask = archive['active'][rows]
        counts = mask.sum(axis=0)
        frequent = counts >= 8
        active = mask[:, frequent]
        draws = resampling_counts(len(active))
        report[label] = result = dict(n=len(active), n_atoms=int(frequent.sum()), n_active=int(active.sum()),
                                      atoms=archive['atoms'][frequent].tolist(), all_atoms=archive['atoms'].tolist(),
                                      counts=counts.tolist(), capped={arm:int(archive[arm+'_clipped'][rows][:,frequent].sum())
                                                                     for arm in ARMS})
        for metric in ('full', 'centered'):
            results, boots = {}, {}
            ref = archive['donor_reference'][rows][:,frequent].astype(float)
            if metric == 'centered':
                ref -= ref.mean(axis=-1, keepdims=True)
            for arm in ARMS:
                pred = archive[arm][rows][:,frequent].astype(float)
                if metric == 'centered':
                    pred -= pred.mean(axis=-1, keepdims=True)
                per, point, boots[arm] = aggregate(active, pred, ref, draws)
                results[arm] = {key:ci(point[key], boots[arm][key]) for key in ('skill', 'cosine')}
                results[arm]['atom_skills'] = per['skill'].tolist()
            for control in ('wrong_leads', 'mean_donor'):
                delta = results['adaptive']['skill']['value'] - results[control]['skill']['value']
                results['adaptive_minus_'+control] = ci(delta, boots['adaptive']['skill']-boots[control]['skill'])
            result[metric] = results
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    args = parser.parse_args()
    report = {}
    for donor in ('bolt', 'timesfm'):
        with np.load(args.out/(donor+'.npz')) as data:
            report[donor] = summarize(data)
    (args.out/'scores.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
