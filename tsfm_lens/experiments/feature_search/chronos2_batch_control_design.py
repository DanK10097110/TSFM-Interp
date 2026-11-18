"""CPU-only design of fixed-B8 counterfactual-neighbor measurement controls.

There is deliberately no model-loading/GPU inference entry point here.
"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from experiments.feature_search.phase_orbit_native import digest

BASE_ROWS=(0,73,146,219,292,365,438,511)
NEIGHBOR_ROWS=((63,126,189,252,315,378,441,504),tuple(reversed(BASE_ROWS)))


def fixed_batches(context):
    context=np.asarray(context)
    if context.shape!=(512,128) or not np.isfinite(context).all():
        raise ValueError('Require exact finite512×128 saved phase corpus')
    base=context[list(BASE_ROWS)].copy()
    result=[]
    for target in range(8):
        for neighbor,indices in enumerate(NEIGHBOR_ROWS):
            counterfactual=context[list(indices)].copy()
            counterfactual[target]=base[target]
            if not np.array_equal(counterfactual[target],base[target]):raise AssertionError('Target changed')
            result.append(dict(target=target,neighbor_bank=neighbor,batch=counterfactual))
    return base,result


def known_group_forecast(batch,group_ids):
    """Synthetic readout using installed native group mask: a known-bad decoy."""
    import torch
    from chronos.chronos2.model import Chronos2Encoder
    groups=torch.tensor(group_ids,dtype=torch.long)
    mask=Chronos2Encoder._construct_and_invert_group_time_mask(groups,torch.ones((8,1)),torch.float32)
    allowed=(mask[0,0]==0).double().numpy()
    levels=np.asarray(batch,dtype=np.float64).mean(axis=1)
    return allowed@levels/allowed.sum(axis=1)


def prepare(source,out):
    if out.exists():raise ValueError('Preserve earlier designs')
    import chronos.chronos2.model as model
    import chronos.chronos2.layers as layers
    import chronos.chronos_bolt as normalization
    from tsfm_lens.models.chronos2_adapter import Chronos2Adapter
    protocol=dict(role='DESIGN ONLY;measurement diagnosis;no model inference authorized here',
        corpus_source=str(source),pairs=['p32_16','p40_20'],context=128,horizon=64,
        batch_size=8,dtype='float32',base_rows=BASE_ROWS,neighbor_rows=NEIGHBOR_ROWS,
        target='Every target0..7 held bit-identical in its SAME slot;onlyother7 changed;allcontexts finite/no padding changes',
        primary='defaultgroupIDs neighbor replacement maxabs EXACT0 for every target/case/bank',
        separate_controls='B8native repeat EXACT0;default vs explicit arange(8) groups EXACT0 onbase+all16counterfactualbatches;cyclicrowpermutations1..7 inversealigned EXACT0;positioncheck reported separately',
        group_source='Adapter passes no groupIDs;model defaults arange(B);groupattention permits equalIDs only;normalization reduces last(time)dimension perrow',
        masks='unique arangeIDs produce diagonal group mask;finite contexts keep allvalid time masks and native shapes unchanged',
        known_bad='Allzero groupIDs in synthetic installed-mask readout mixes target withneighbors andmust fail;properuniqueIDs must pass',
        budget='Perpair:2base default repeats+1base explicit+16neighbor default+16neighbor explicit+7cyclic default=42B8 forwards;2pairs=84forwards=672contexts;onecachedC2model',
        resources=dict(gpu=3,cpu_threads=2,batch=8,cap_bytes=6*2**30,timeout_seconds=180),
        interpretation='Neighborchange atfixedshape is scientificbatchdependence;positionchange alone may be numericpositioneffect;singleton-vsB8 discrepancy remains originalfailure;sourceindependence is structural evidence,not numeric control pass',
        no_rescue='No singletonthreshold change,no oldglobalcontrol relabel,no mode/featureclaim,no aliasvalidation or captures',
        next_step='A separatelyreviewednative runner+explicitrootauthorization required;this source canONLYprepare',
        design_source_sha256=digest(__file__),sources={inspect.getfile(m):digest(inspect.getfile(m)) for m in (model,layers,normalization,Chronos2Adapter)},
        bindings={stem:{suffix:digest(source/f'{stem}.{suffix}') for suffix in ('npz','json')} for stem in ('p32_16','p40_20')})
    out.mkdir(parents=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(args.source,args.out)
