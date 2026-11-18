"""Independently recompute published errors/sensitivity from portable raw arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np


def audit():
    root=Path(__file__).resolve().parents[1]/'examples/dynamic_steering'
    manifest=json.loads((root/'manifest.json').read_text())
    for name,record in manifest.items():
        raw=(root/name).read_bytes()
        assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
    count=0;sensitivities=0;maximum=0.;replay=0.
    for folder in sorted(root.glob('dynamic_d*')):
        if not (folder/'forecasts.npz').exists():continue
        rows=json.loads((folder/'rows.json').read_text())
        with np.load(folder/'forecasts.npz',allow_pickle=False) as arrays:
            if 'achieved' in rows[0]:
                kinds=list(dict.fromkeys(row['kind'] for row in rows))
                for row in rows:
                    k=row.get('request_index',kinds.index(row['kind']));i=row['context_index']
                    base=arrays[f'{k}_baseline'][i];actual=arrays[f'{k}_forecast'][i];target=arrays[f'{k}_target'][i]
                    norm=np.linalg.norm(target);error=float(np.linalg.norm(actual-base-target)/norm) if norm else 0.
                    maximum=max(maximum,abs(error-row['relative_error']))
                    assert np.isclose(error,row['relative_error'],rtol=1e-10,atol=1e-10),(folder.name,k,i,error,row['relative_error'])
                    assert row['achieved']==(error<=.1),(folder.name,k,i,'success flag')
                    if actual.shape[-1]>1:
                        assert np.mean(np.diff(actual,axis=-1)<0)==row['crossing_fraction']
                    replay=max(replay,row['replay_max_abs']);count+=1
            else:
                for row in rows:
                    k=row['case'];i=row['context']
                    difference=arrays[f'{k}_perturbed_forecast'][i]-arrays[f'{k}_forecast'][i]
                    base_difference=arrays[f'{k}_perturbed_baseline'][i]-arrays[f'{k}_baseline'][i]
                    assert np.isclose(np.linalg.norm(difference),row['controlled_sensitivity'])
                    assert np.isclose(np.linalg.norm(base_difference),row['baseline_sensitivity'])
                    replay=max(replay,row['replay_max_abs']);sensitivities+=1
    report=dict(checked_artifact_hashes=len(manifest),steering_cases=count,sensitivity_pairs=sensitivities,
        max_error_discrepancy=maximum,max_recorded_native_replay_discrepancy=replay,
        scope='offline recomputation from raw arrays and hash verification; native replay was performed during each experiment')
    (root/'artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':audit()
