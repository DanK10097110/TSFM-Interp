"""Generate a readable comparison table from both independently fitted dictionaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def number(x):
    return f'{x:.3f}'


def interval(r):
    return f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"


def generate(data,scratch):
    lines=['# Replication across independently trained dictionaries','',
           'We fitted two TopK SAEs to exactly the same 34,515 public-development activation rows from each backbone, changing the initialization and minibatch seed. Both used width 6,144, k=32 and 720 optimization steps. The archived dictionaries also saw 3,990 Monash augmentation rows whose saved activations were unavailable. Seeds 1 and 2 form the controlled same-data comparison. The archived checkpoint adds a sensitivity test across training recipes. The fresh fits used the archived public-development training rows. Historical exposure through the pretrained backbones or the earlier online Monash augmentation is unknown. Both fresh fits share the downstream calibration and evaluation series. [Training split and method](replica.py) · [Training hashes and learning curves](replica_training.json) · [Protocol](PROTOCOL.md)','',
           '## Can the same backbone’s atoms be matched across seeds?','',
           'We fitted one-to-one Hungarian assignments of calibration-active atoms between seed 1 and seed 2. The decoder assignment maximizes decoder-direction cosine in the shared hidden coordinates. The activation assignment maximizes centered clean-code profile cosine over 128 daily calibration histories. Each paired coefficient gets a nonnegative, clean-calibrated scale. We then predicted the complete seed-2 code and its edit on the same 80 untouched-by-fitting daily reserve items. The shuffled comparison permutes the assigned target indices within calibration frequency bands and fits the same scales. The dense states in this same-backbone comparison coincide exactly. [Matching code](seed_identity.py) · [Bolt measurements](seed_identity_bolt.json) · [TimesFM measurements](seed_identity_timesfm.json)','',
           '| Backbone | Assignment | Clean hard cosine | Hard edit cosine | Shoulder edit cosine | Clean-locked edit cosine | Shuffled hard edit cosine | Correct active IDs | Clean − hard edit, paired 95% interval |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    meta={}
    for k,model in [('bolt','Bolt'),('timesfm','TimesFM')]:
        d=read(data/f'seed_identity_{k}.json')
        for match in ['decoder','activation']:
            row=d[match]; c=row['match']
            columns=[model,match]+[number(c[v]['mean']) for v in ['hard_clean_cosine','hard_delta_cosine','shoulder_delta_cosine','locked_delta_cosine']]
            columns.extend([number(row['shuffle']['hard_delta_cosine']['mean']),number(c['correct_target_active_fraction']['mean']),interval(c['hard_clean_minus_delta_cosine'])])
            lines.append('| '+' | '.join(columns)+' |')
        for seed in [1,2]:
            meta[f'seed{seed}_{k}']=read(scratch/f'seed{seed}_{k}.json')
    lines+=['','The correct-active-ID column is the fraction of the target’s 32 clean active IDs present after remapping the independent source support. It differs from coefficient-weighted cosine, which can be high when a few large features align. Holding clean support fixed measures local slopes for the two dictionaries. The detailed files include the achievable cosine when the true target changes on all the matched coordinates are supplied. All direct matching results condition on 128 calibration histories.','',
            '## Can many-to-many maps recover the code change?','',
            'Independent rank-at-most-32 maps learn from paired perturbation changes on 128 daily histories and select their ridge coefficient using the original 48 validation histories. Each row below uses the same complete target code change. Each column changes what the map receives. The dense source receives only the displacement. Access to both dense endpoints and the known target encoder would reproduce the target exactly. [Method](seed_translation.py) · [Bolt](seed_translation_bolt.json) · [TimesFM](seed_translation_timesfm.json)','',
            '| Backbone | Seed-2 target | Hard source | Shoulder source | Locked source | Dense source | Input-only |',
            '|---|---|---:|---:|---:|---:|---:|']
    for k,m in [('bolt','Bolt'),('timesfm','TimesFM')]:
        d=read(data/f'seed_translation_{k}.json')
        for a in ['hard','shoulder','locked']:
            cols=[m,a]+[number(d['targets'][a]['edit_'+s]['cosine']['mean']) for s in ['hard','shoulder','locked','dense','input']]
            lines.append('| '+' | '.join(cols)+' |')
    lines+=['','## Does smoothing reproduce across the cross-model seed pairs?','',
            'Here each row refits the original cross-model maps with that seed’s source code, then patches the target model. The target remains the receiving forecaster’s actual dense hidden change and its eight-step native forecast-patch response. Scores are mean signed cosines over 80 target-domain series. The intervals apply to the paired forecast cosine difference. [Seed 1 weekly hidden](weekly_seed1.json) · [Seed 2 weekly hidden](weekly_seed2.json) · [Seed 1 forecast audit](seed1_forecast_audit.json) · [Seed 2 forecast audit](seed2_forecast_audit.json)','',
            '| Dataset | Target | Source SAE | Hard → shoulder hidden | Hard → shoulder forecast | Paired forecast improvement, 95% interval |',
            '|---|---|---|---:|---:|---:|']
    for ds,title in [('weekly','Weekly'),('fast','Bitbrains')]:
        for recipient,other,model in [('timesfm','bolt','TimesFM'),('bolt','timesfm','Bolt')]:
            for label,tag in [('archived','smooth'),('seed 1','seed1'),('seed 2','seed2')]:
                d=read(data/f'{ds}_{tag}.json')['transfer'][other+'_to_'+recipient]
                f=read(data/f'{ds}_{tag}_{recipient}_forecasts.json')
                hc=' → '.join(number(d[a]['cosine']['mean']) for a in ['code','shoulder'])
                fc=' → '.join(number(f['cross_'+a]['cosine']['mean']) for a in ['code','shoulder'])
                lines.append('| '+' | '.join([title,model,label,hc,fc,interval(f['cross_shoulder_minus_cross_code'])])+' |')
    lines+=['','Results include the actual normalization and uncertainty for each native effect in the forecast audits, plus the unchanged waveform-only control. Both fresh seeds share the same sampled calibration and evaluation items. A truly independent population replication would require further checkpoints and series.','']
    (data/'replica_training.json').write_text(json.dumps(meta,indent=2)+'\n')
    (data/'SEEDS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args();generate(a.results,a.scratch)
