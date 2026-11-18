"""External-validity and public-API check on cached hourly real contexts.

Uses the existing real-background corpus's evaluation sources only, with no
future values and no setting selection. These windows were used by earlier
research and may overlap model pretraining; this is not sealed confirmation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..steering import ForecastSteerer


def run(args):
    """Apply the frozen discovery settings, saving source-level diagnostics."""
    torch.set_num_threads(2)
    started=time.time()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source)
    metadata=json.loads((source/'metadata.json').read_text())
    series=np.load(source/'corpus.npz')['series']
    indices=[i for i,m in enumerate(metadata) if m['split']=='evaluation']
    contexts=np.stack([series[i,128:256]*metadata[i]['scale']+metadata[i]['center'] for i in indices]).astype('float32')
    selected=[metadata[i] for i in indices]
    protocol=dict(model=args.model,source=str(source),selected_indices=indices,
                  corpus_sha256=hashlib.sha256((source/'corpus.npz').read_bytes()).hexdigest(),
                  metadata_sha256=hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),
                  context_slice=[128,256],future_used=False,period=24.,strength='interval fraction .2; seasonal/upper .2 context SD',
                  radius=.2,steps=6,ridge=1e-5,scope='external validity on reused real sources; pretraining overlap unknown')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    steerer=ForecastSteerer.from_pretrained(args.model)
    rows,arrays=[],{}
    for i,(x,meta) in enumerate(zip(contexts,selected)):
        for task,name,strength in [('interval','widen',.2),('interval','narrow',-.2),
                                    ('seasonal','season',.2*float(np.std(x))),
                                    ('quantile','upper',.2*float(np.std(x)))]:
            result=steerer.steer(x[None],task=task,strength=strength,period=24.)
            rows.append(dict(dataset=meta['dataset'],source_id=meta['id'],task=name,
                             **result.diagnostics[0]))
            arrays[f'{i}_{name}_baseline']=result.baseline[0]
            arrays[f'{i}_{name}_forecast']=result.forecast[0]
            arrays[f'{i}_{name}_target']=result.requested_change[0]
        print(json.dumps(dict(completed=i+1,total=len(contexts),seconds=time.time()-started)),flush=True)
    summary={}
    for dataset in sorted({r['dataset'] for r in rows}):
        for task in ['widen','narrow','season','upper']:
            group=[r for r in rows if r['dataset']==dataset and r['task']==task]
            summary[f'{dataset}/{task}']=dict(n=len(group),error=float(np.mean([r['relative_error'] for r in group])),
                successes=sum(r['achieved'] for r in group),
                crossing_fraction=float(np.mean([r['crossing_fraction'] for r in group])),
                baseline_crossing_fraction=float(np.mean([r['baseline_crossing_fraction'] for r in group])))
    result=dict(protocol=protocol,rows=rows,summary=summary,seconds=time.time()-started)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'interventions.npz',contexts=contexts,**arrays)
    (out/'findings.md').write_text('# Real-context steering findings\n\n'+json.dumps(summary,indent=2)+
                                  '\n\nNo future labels, no per-dataset retuning, no calibration claim.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--model',choices=['bolt','timesfm'],required=True)
    p.add_argument('--source',default='runs/real_background_poc492306')
    run(p.parse_args())
