"""Bounded cached-dev screen before promoting window diagnostics into library code."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.utils import sample_rows
from window_alignment import linear_cka, matched_rank, represent
from window_alignment_phase import kernel_null


def run():
    """Read only three named dev layers and crop identical trailing384steps."""
    source=Path('runs/panel7_v3_concat')
    root=Path('runs/feature_search_20261111_window_cached_dev')
    root.mkdir(exist_ok=False)
    store=ActivationStore(source/'activations.zarr',mode='r')
    targets=[('TimesFM','stacked_xf.10',32),('Chronos-Bolt','encoder.block.3',16),('Chronos-2','encoder.block.6',16)]
    metadata=pd.read_parquet(source/'meta.parquet')
    if metadata.series_id.duplicated().any():raise ValueError('duplicate series identifiers')
    held=[np.asarray(store.root[f'fine/{m}/{l}'].attrs['rows']) for m,l,w in targets]
    common=sorted(set(held[0]).intersection(*[set(r) for r in held[1:]]))
    common=np.asarray(common)
    rows=common[sample_rows(len(common),128,20261111,strata=metadata.family.to_numpy()[common])]
    train_positions=sample_rows(len(rows),64,20261112,strata=metadata.family.to_numpy()[rows])
    train=np.zeros(len(rows),dtype=bool);train[train_positions]=True;test=~train
    manifest=dict(source=str(source),context=480,crop=[96,480],n_common_ids=len(common),
                  rows=rows.tolist(),series_ids=metadata.series_id.iloc[rows].tolist(),
                  train_rows=rows[train].tolist(),test_rows=rows[test].tolist(),
                  strata='metadata.family',sample_seed=20261111,split_seed=20261112,
                  targets=targets,windows=[32,64,128],methods=['mean','native_concat','common32_concat'],
                  ranks=[8,32],permutations=999,inference='descriptive; no adjusted multiplicity claims',
                  limits='windowrows dependent; PCA matches rank not spectrum; fixedspan nativegrid contract')
    (root/'protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    tokens={}
    for model,layer,width in targets:
        array=store.root[f'fine/{model}/{layer}']
        if array.shape[0]!=32//width or array.shape[2]!=15:
            raise ValueError('stored grid incompatible with declared native span width')
        positions=[store.load_fine_position(model,layer,p,rows).astype(np.float32)[:,3:]
                   for p in range(array.shape[0])]
        tokens[model]=(np.stack(positions,axis=2).reshape(128,-1,array.shape[-1]),width)
        print('read '+model+' '+str(tokens[model][0].shape),flush=True)
    pairs=list(combinations(tokens,2))
    sequence={f'{a}|{b}':linear_cka(tokens[a][0][test].reshape(64,-1),tokens[b][0][test].reshape(64,-1)) for a,b in pairs}
    output=dict(sequence_cka=sequence,sequence_partition_max_difference={},cells={})
    for name,(values,width) in tokens.items():
        output['sequence_partition_max_difference'][name]=float(max(np.max(np.abs(
            represent(values,width,w,'native_concat').reshape(128,-1)-values.reshape(128,-1))) for w in (32,64,128)))
    for window in (32,64,128):
        nw=384//window
        for method in ('mean','native_concat','common32_concat'):
            raw={};controls={8:{},32:{}};diagnostics={}
            for name,(values,width) in tokens.items():
                features=represent(values,width,window,method).reshape(128,nw,-1)
                raw[name]=features[test].reshape(64*nw,-1)
                diagnostics[name]={'dimension':features.shape[-1],'pca':{}}
                for rank in (8,32):
                    _,b,diag=matched_rank(features[train].reshape(64*nw,-1),raw[name],rank)
                    controls[rank][name]=b;diagnostics[name]['pca'][rank]=diag
            record=dict(n_test_series=64,n_windows_per_series=nw,models=diagnostics,pairs={})
            for a,b in pairs:
                record['pairs'][f'{a}|{b}']={'raw':kernel_null(raw[a],raw[b],64,nw),
                    **{f'rank{rank}':kernel_null(controls[rank][a],controls[rank][b],64,nw) for rank in (8,32)}}
                for score in record['pairs'][f'{a}|{b}'].values():
                    score['resample_unit']='whole_source_series_allwindows'
            output['cells'][f'{window}_{method}']=record
            (root/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
            print('complete '+str(window)+' '+method,flush=True)
    return output


if __name__=='__main__':run()
