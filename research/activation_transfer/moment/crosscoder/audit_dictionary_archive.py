"""Independently reconstruct selected codes and raw-unit decoder changes from the saved tensors."""
import json

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder.dictionaries import CACHE, METHODS, OUT, SEEDS, paired_m4, paired_noaa


def run():
    ids, m4 = paired_m4()
    stations, dates, noaa = paired_noaa()
    reports = {}
    for method in METHODS:
        for seed in SEEDS:
            stem = f'{method}_{seed}'
            f = torch.load(CACHE / f'{stem}.pt',weights_only=True,map_location='cpu')
            w = f['state']
            gaps = []
            with np.load(CACHE / f'{stem}.npz') as z:
                np.testing.assert_array_equal(z['m4_ids'],ids)
                np.testing.assert_array_equal(z['station_ids'],stations)
                np.testing.assert_array_equal(z['dates'],dates)
                for i,size in enumerate(('small','large')):
                    raw = (w[f'W_dec.{i}'] * f['scales'][i]).numpy()
                    np.testing.assert_array_equal(z['decoder_'+size], raw)
                for name, arrays, ix in [('m4',[x[:,-1] for x in m4],[0,767,768,1023]),('noaa',noaa,[0,97,192,384])]:
                    components = []
                    for i,vals in enumerate(arrays):
                        h = torch.from_numpy(vals[ix].copy())
                        normalized = (h-f['means'][i])/f['scales'][i]-w[f'b_dec.{i}']
                        components.append(normalized@w[f'W_enc.{i}'])
                    for read in ('source','both'):
                        pre = w['b_enc']+components[0]
                        if read=='both':
                            pre += components[1]
                        values, indices = torch.topk(pre.clamp_min(0),32,dim=1)
                        expected = torch.zeros_like(pre).scatter_(1,indices,values).numpy()
                        archived = z[name+'_'+read][ix]
                        np.testing.assert_array_equal(expected>0,archived>0)
                        np.testing.assert_allclose(expected,archived,atol=7e-5,rtol=1e-5)
                        gaps.append(float(np.max(np.abs(expected-archived))))
            reports[stem] = dict(histories=8,encoder_readings=2,max_abs_code_gap=max(gaps),all_support_exact=True,raw_decoders_exact=True)
    (OUT/'replay.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
