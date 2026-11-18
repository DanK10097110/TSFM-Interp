"""Opt-in, read-only cached window geometry screen; writes only a NEW output directory.

Example (from tsfm_lens):
python run_window_robustness.py --run runs/panel7_v3_concat   --target TimesFM:stacked_xf.10:32 --target Chronos-Bolt:encoder.block.3:16   --windows 32,64,128 --crop 96:480 --out runs/new_window_diagnostic

Native widths are explicit regular-span declarations; they are verified against
fine-array layout, not inferred from array dimensions or treated as locality.
"""
from __future__ import annotations
import argparse
import json
from functools import reduce
from math import lcm
from pathlib import Path
import numpy as np
import pandas as pd
from tsfm_lens.analysis.window_robustness import WindowTarget, available_rows, compare_windows, physical_extent, render_window_report
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.utils import sample_rows

def target_spec(value):
    """Parse an explicit model:layer:native_width target without guessing layers."""
    try:
        model, layer, width = value.rsplit(':', 2)
        if not model or not layer:
            raise ValueError('empty target')
        return WindowTarget(model, layer, int(width))
    except ValueError as error:
        raise argparse.ArgumentTypeError('target must be MODEL:LAYER:NATIVE_WIDTH') from error

def _main(argv=None):
    """Select shared cached IDs, record the split, and stream progress artifacts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target', type=target_spec, action='append')
    parser.add_argument('--report', type=Path, help='render an existing results JSON only; no activation reads or analysis')
    parser.add_argument('--windows', default='32,64,128')
    parser.add_argument('--crop', help='explicit START:END physical extent')
    parser.add_argument('--ranks', default='8,32')
    parser.add_argument('--methods', default='mean,native_concat,common_grid')
    parser.add_argument('--grid-width', type=int)
    parser.add_argument('--series', type=int, default=128)
    parser.add_argument('--train-series', type=int, default=64)
    parser.add_argument('--rows', help='explicit comma-separated source row IDs')
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--memory-limit-mb', type=float, default=2048)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('--out must be a NEW directory; existing runs are read-only')
    if args.report:
        result = json.loads(args.report.read_text(encoding='utf-8'))
        protocol_path = args.report.parent / 'protocol.json'
        protocol = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.exists() else {}
        args.out.mkdir(parents=True, exist_ok=False)
        render_window_report(result, args.out / 'report.html', protocol)
        return result
    if args.run is None or args.target is None:
        parser.error('--run and at least two explicit --target values are required unless --report is used')
    if len(args.target) < 2:
        parser.error('provide at least two distinct explicit --target values')
    store = ActivationStore(args.run / 'activations.zarr', mode='r')
    windows = [int(w) for w in args.windows.split(',')]
    ranks = [int(r) for r in args.ranks.split(',')]
    crop = tuple((int(x) for x in args.crop.split(':'))) if args.crop else None
    extent = physical_extent(store.root.attrs['context_len'], windows, crop)
    common = reduce(np.intersect1d, [available_rows(store, t) for t in args.target])
    metadata = pd.read_parquet(args.run / 'meta.parquet')
    if len(metadata) != store.root.attrs['n_series'] or metadata.series_id.duplicated().any():
        parser.error('metadata has inconsistent or duplicate source series identifiers')
    families = metadata.family.to_numpy()
    if args.rows:
        rows = np.array([int(r) for r in args.rows.split(',')])
        if len(set(rows)) != len(rows) or not set(rows) <= set(common):
            parser.error('explicit rows must be unique and present in every target')
    else:
        if args.series > len(common):
            parser.error(f'only {len(common)} common stored rows available; reduce --series explicitly')
        rows = common[sample_rows(len(common), args.series, args.seed, strata=families[common])]
    if not 2 <= args.train_series <= len(rows) - 3:
        parser.error('need at least two train and three test series')
    train_positions = sample_rows(len(rows), args.train_series, args.seed + 1, strata=families[rows])
    train_rows = rows[train_positions]
    dimensions = [store.root[f'fine/{t.model}/{t.layer}'].shape[-1] for t in args.target]
    source_context = store.root.attrs['context_len']
    n_test = len(rows) - len(train_rows)
    source_sizes = [4 * len(rows) * (source_context // t.native_width) * d for t, d in zip(args.target,
            dimensions)]
    max_nw = max(((extent[1] - extent[0]) // w for w in windows))
    estimate = sum(source_sizes) + 3 * max(source_sizes) + 8 * (n_test * max_nw) ** 2 * (len(args.target) * (len(ranks) + 1) + 3)
    if estimate > args.memory_limit_mb * 1024 ** 2:
        parser.error(f'estimated working arrays {estimate / 1024 ** 2:.1f}MiB exceed --memory-limit-mb')
    alignment_path = args.run / 'alignment' / 'alignment_check.json'
    alignment = json.loads(alignment_path.read_text(encoding='utf-8')) if alignment_path.exists() else None
    protocol = dict(source=str(args.run), targets=[vars(t) for t in args.target], windows=windows,
        crop=list(extent), ranks=ranks, methods=args.methods.split(','), grid_width=args.grid_width,
        requested_grid_width=args.grid_width,
        effective_grid_width=args.grid_width if args.grid_width is not None else lcm(*(t.native_width for t in args.target)),
        grid_width_source='explicit' if args.grid_width is not None else 'auto_lcm',
        source_context=int(source_context), rows=rows.tolist(), series_ids=metadata.series_id.iloc[rows].tolist(),
        train_rows=train_rows.tolist(), test_rows=rows[~np.isin(rows, train_rows)].tolist(),
        selection='family-stratified common cached IDs; explicit rows retained verbatim when supplied',
        seed=args.seed, permutations=args.permutations, estimated_working_mib=estimate / 1024 ** 2,
        alignment_provenance={t.key: alignment.get(t.model, {}).get('per_layer', {}).get(t.layer,
                'unavailable') if alignment else 'unavailable' for t in args.target}, source_read_only=True,
        inference='descriptive; null p values unadjusted across all cells')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

    def progress(cell, output):
        """Persist each completed cell so a long diagnostic remains reviewable."""
        (args.out / 'results.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
        print('completed ' + cell, flush=True)
    result = compare_windows(store, args.target, rows, train_rows, windows, crop=extent,
        ranks=ranks, methods=args.methods.split(','), grid_width=args.grid_width, permutations=args.permutations,
        seed=args.seed, progress=progress)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    render_window_report(result, args.out / 'report.html', protocol)
    return result

def main(argv=None):
    """Cap CLI torch and BLAS threads even when shell thread settings are absent."""
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        return _main(argv)
if __name__ == '__main__':
    main()
