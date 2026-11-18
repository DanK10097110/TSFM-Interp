"""Serial slot-one coordinator: no model imports; immutable bounded stage outputs.

Requires the explicit slot flag. Stops on the first failed stage and records slot
release. Each subprocess invokes the independent 600-second/3GiB stage monitor.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    """Atomic small execution ledger; detailed numbers stay in each stage file."""
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def main():
    """Execute the prespecified grid, preserving completed stages on resumption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/dictionary_pilot4951000'))
    parser.add_argument('--slot-granted', action='store_true')
    parser.add_argument('--models', nargs='+', choices=('bolt', 'timesfm'), default=['bolt', 'timesfm'])
    args = parser.parse_args()
    if not args.slot_granted:
        parser.error('Coordinator slot grant required')
    history = []
    started = time.monotonic()
    status = 'running'
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    try:
        for model in args.models:
            for k in (8, 16):
                for method in ('native', 'rotation', 'pca', 'topk', 'nested'):
                    seeds = (4951000,) if method in ('native', 'pca') else (4951000, 4951001, 4951002)
                    for seed in seeds:
                        for stage in ('fit', 'native'):
                            name = f'{stage}_{model}_cut0_{method}_k{k}_s{seed}'
                            dest = args.out/name
                            if dest.exists():
                                record = json.loads((dest/'execution.json').read_text(encoding='utf-8'))
                                if record['returncode'] != 0:
                                    raise RuntimeError('Prior failed stage requires review: '+name)
                                continue
                            command = [sys.executable, '-m', 'experiments.feature_search.dictionary_pilot', 'run',
                                '--slot-granted', '--out', str(args.out), '--stage', stage, '--model', model,
                                '--method', method, '--k', str(k), '--seed', str(seed)]
                            process = subprocess.run(command, env=env)
                            entry = dict(stage=name, returncode=process.returncode, seconds_since_start=time.monotonic()-started)
                            history.append(entry)
                            if (dest/'knowledge.json').exists():
                                evidence = json.loads((dest/'knowledge.json').read_text(encoding='utf-8'))
                                entry['exact_results'] = evidence['exact_results']
                            write(args.out/'queue.json', dict(status='running', slot=1, stages=history))
                            print(json.dumps(entry), flush=True)
                            if process.returncode:
                                raise RuntimeError('Stage failed: '+name)
                command = [sys.executable, '-m', 'experiments.feature_search.dictionary_pilot', 'compare',
                    '--slot-granted', '--out', str(args.out), '--model', model, '--k', str(k)]
                subprocess.run(command, env=env, check=True)
        status = 'complete'
    finally:
        if status != 'complete':
            status = 'stopped_failure'
        write(args.out/'queue.json', dict(status=status, slot=1, released=True, stages=history))
        write(args.out/'knowledge.json', dict(question='Bounded direct-coordinate versus sparse-dictionary spectral pilot',
            exact_results=dict(queue='queue.json', completed_stages=sum(r['returncode']==0 for r in history), elapsed_seconds=time.monotonic()-started),
            controls=['FIT-only models/scalers/probes; source weighting; normalized targets and metadata controls',
                      'native exact self/repeat/saved-state gates; nuisance response and matched random edits'],
            limitations=['spectral cache; four TEST real sources/domain; no learned-concept or general monosemanticity claim',
                         'nested SAE has 1.25x estimated forward dense matmul FLOPs; equal optimizer steps/parameters are not equal compute'],
            next_step='Coordinator reviews stage records before any expansion or claim.', status=status,
            slot=dict(id=1, released=True)))


if __name__ == '__main__':
    main()
