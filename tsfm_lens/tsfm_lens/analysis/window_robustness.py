"""Read-only window geometry diagnostics with explicit physical-time provenance.

Native concatenation preserves available token order; common-grid pooling and
means answer different geometric questions. Matched PCA rank controls feature
budget, not spectrum or task information. No representation is a gold standard.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import lcm
import numpy as np
import torch
from sklearn.decomposition import PCA
from ..extraction.alignment import pooling_matrix
from ..extraction.store import ActivationStore
from .window_concat import position_features

@dataclass(frozen=True)
class WindowTarget:
    """A named layer with explicitly verified regular native span width."""
    model: str
    layer: str
    native_width: int

    @property
    def key(self):
        """A target identity retaining both model and layer."""
        return f'{self.model}/{self.layer}'

def available_rows(store: ActivationStore, target: WindowTarget) -> np.ndarray:
    """Audit fine-array IDs and declared regular resolution without reading data."""
    array = store.root[f'fine/{target.model}/{target.layer}']
    base = int(store.root.attrs['window'])
    context = int(store.root.attrs['context_len'])
    width = target.native_width
    if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width <= 0:
        raise ValueError('native width must be an explicit positive integer')
    if context % base or base % width or array.shape[0] != base // width or (array.shape[2] != context // base):
        raise ValueError(f'{target.key}: declared native width does not match stored fine-grid contract; replicated coarse tokens unsupported')
    if 'rows' not in array.attrs and array.shape[1] != store.root.attrs['n_series']:
        raise ValueError(f'{target.key}: subset fine cache lacks source row IDs; refusing to guess identities')
    ids = np.asarray(array.attrs.get('rows', list(range(array.shape[1]))))
    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError(f'{target.key}: stored source row IDs must be integers; refusing to coerce identities')
    if len(ids) != array.shape[1] or len(set(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= store.root.attrs['n_series']):
        raise ValueError(f'{target.key}: inconsistent or duplicate stored series IDs')
    return ids

def validate_rows(store, targets, rows, train_rows):
    """Refuse ID mismatches instead of silently intersecting requested samples."""
    rows = np.asarray(rows)
    train_rows = np.asarray(train_rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(set(rows)) != len(rows):
        raise ValueError('rows must be unique integer source-series IDs')
    if train_rows.ndim != 1 or train_rows.dtype.kind not in 'iu' or len(set(train_rows)) != len(train_rows) or (not set(train_rows) <= set(rows)):
        raise ValueError('train rows must be unique requested source-series IDs')
    if len(train_rows) < 2 or len(rows) - len(train_rows) < 3:
        raise ValueError('need at least two training and three test series')
    if len({t.key for t in targets}) != len(targets) or len(targets) < 2:
        raise ValueError('need at least two distinct model/layer targets')
    for target in targets:
        missing = set(rows) - set(available_rows(store, target))
        if missing:
            raise ValueError(f'{target.key}: inconsistent IDs; missing requested rows {sorted(missing)[:5]}')
    return np.isin(rows, train_rows)

def physical_extent(context, windows, crop=None):
    """Require one common extent; never silently truncate separately per window."""
    if not windows or any((isinstance(w, bool) or not isinstance(w, (int, np.integer)) or w <= 0 for w in windows)):
        raise ValueError('windows must be positive integers')
    if len(set(windows)) != len(windows):
        raise ValueError('duplicate windows would overwrite diagnostic cells')
    start, end = (0, int(context)) if crop is None else tuple(crop)
    if not (isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)) and (0 <= start < end <= context)):
        raise ValueError('crop must be an explicit valid [start,end) physical interval')
    if any(((end - start) % w for w in windows)):
        raise ValueError('all windows must tile the SAME physical extent; supply an explicit common crop')
    return (int(start), int(end))

def load_native_tokens(store, target, rows, crop):
    """Read one fine position at a time and restore chronological native order."""
    width = target.native_width
    start, end = crop
    if start % width or end % width:
        raise ValueError('crop boundaries must align with every declared native span')
    pieces = [store.load_fine_position(target.model, target.layer, p, np.asarray(rows)).astype(np.float32) for p in range(store.n_fine_positions(target.model,
                target.layer))]
    values = np.stack(pieces, axis=2).reshape(len(rows), -1, pieces[0].shape[-1])
    values = values[:, start // width:end // width]
    if not np.isfinite(values).all():
        raise ValueError(f'{target.key}: nonfinite cached activation values')
    return values

def window_features(tokens, native_width, window, method, grid_width):
    """Use existing span helpers on explicitly declared, regular native spans."""
    if window < native_width or window % native_width:
        raise ValueError('window is below native resolution or does not tile native spans')
    context = tokens.shape[1] * native_width
    if context % window:
        raise ValueError('window does not tile the common extent')
    spans = np.column_stack((np.arange(tokens.shape[1]) * native_width, (np.arange(tokens.shape[1]) + 1) * native_width))
    hidden = torch.from_numpy(np.asarray(tokens, dtype=np.float32))
    if method == 'mean':
        return torch.einsum('wt,btd->bwd', pooling_matrix(spans, context, window), hidden).numpy()
    if method == 'native_concat':
        positions = position_features(hidden, spans, context, window)
    elif method == 'common_grid':
        if grid_width < native_width or grid_width % native_width or window % grid_width:
            raise ValueError('common grid must tile the window and every native span')
        positions = position_features(hidden, spans, context, window, grid=grid_width)
    else:
        raise ValueError(f'unknown window method {method}')
    return positions.numpy().reshape(len(tokens), context // window, -1)

def project_rank(train, test, rank, seed=0):
    """Fit PCA on training series only and expose retained spectrum diagnostics."""
    if rank < 1 or rank > min(train.shape):
        raise ValueError('rank exceeds training rows or feature dimension')
    pca = PCA(n_components=rank, svd_solver='randomized', random_state=seed)
    pca.fit(train)
    variance = pca.explained_variance_
    if not np.isfinite(variance).all() or variance.sum() <= 0:
        raise ValueError('zero-variance PCA training representation')
    return (pca.transform(test), dict(rank=int(rank), participation_ratio=float(variance.sum() ** 2 / (variance @ variance)),
            variance_retained=float(pca.explained_variance_ratio_.sum())))

def centered_kernel(features):
    """Return a centered row kernel with float64 reduction precision."""
    features = np.asarray(features, dtype=np.float64)
    features = features - features.mean(0)
    return features @ features.T

def kernel_similarity(ka, kb, n_series, n_windows, permutations=999, seed=0):
    """Linear CKA plus a null permuting complete series with their window order."""
    if ka.shape != kb.shape or ka.shape != (n_series * n_windows,) * 2:
        raise ValueError('kernel rows do not match the same series/window identities')
    if not isinstance(permutations, (int, np.integer)) or permutations < 1:
        raise ValueError('need positive permutation count')
    denominator = np.linalg.norm(ka) * np.linalg.norm(kb)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('CKA undefined for zero-variance representations')
    observed = float(np.einsum('ij,ij->', ka, kb) / denominator)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        idx = (rng.permutation(n_series)[:, None] * n_windows + np.arange(n_windows)).ravel()
        null.append(float(np.einsum('ij,ij->', ka, kb[np.ix_(idx, idx)]) / denominator))
    return dict(value=observed, null_mean=float(np.mean(null)), null_p95=float(np.quantile(null,
                0.95)), null_p_plusone=float((1 + np.sum(np.asarray(null) >= observed)) / (permutations + 1)),
        n_permutations=int(permutations), resample_unit='whole_source_series')

def compare_windows(store, targets, rows, train_rows, windows, *, crop=None, ranks=(8,
        32), methods=('mean', 'native_concat', 'common_grid'), grid_width=None, permutations=999,
    seed=0, progress=None):
    """Compare fixed targets/rows over a shared extent without modifying a run.

    Callers choose layers prospectively. Raw CKA is dimension dependent;
    dependent windows require the whole-series null. Null p values here are
    descriptive and unadjusted. This diagnostic makes no task or causal claim.
    """
    targets = list(targets)
    rows = np.asarray(rows)
    windows = list(windows)
    if not methods or len(set(methods)) != len(methods) or any((m not in ('mean', 'native_concat',
                'common_grid') for m in methods)):
        raise ValueError('methods must be distinct mean/native_concat/common_grid choices')
    if not ranks or len(set(ranks)) != len(ranks) or any((isinstance(r, bool) or not isinstance(r,
                (int, np.integer)) or r < 1 for r in ranks)):
        raise ValueError('ranks must be distinct positive integers')
    train = validate_rows(store, targets, rows, np.asarray(train_rows))
    test = ~train
    extent = physical_extent(store.root.attrs['context_len'], windows, crop)
    requested_grid_width = grid_width
    grid_width = lcm(*(t.native_width for t in targets)) if grid_width is None else grid_width
    if not isinstance(grid_width, (int, np.integer)) or grid_width < 1:
        raise ValueError('common grid width must be positive integer')
    tokens = {t.key: load_native_tokens(store, t, rows, extent) for t in targets}
    pairs = list(combinations(targets, 2))
    n_test = int(test.sum())
    length = extent[1] - extent[0]
    sequence = {}
    for a, b in pairs:
        ka = centered_kernel(tokens[a.key][test].reshape(n_test, -1))
        kb = centered_kernel(tokens[b.key][test].reshape(n_test, -1))
        denominator = np.linalg.norm(ka) * np.linalg.norm(kb)
        if denominator <= 0:
            raise ValueError('whole-sequence CKA undefined for zero-variance representation')
        sequence[f'{a.key}|{b.key}'] = float(np.einsum('ij,ij->', ka, kb) / denominator)
    output = dict(source=str(store.path), source_context=int(store.root.attrs['context_len']),
        crop=list(extent), rows=rows.tolist(), train_rows=rows[train].tolist(), test_rows=rows[test].tolist(),
        grid_width=int(grid_width), requested_grid_width=None if requested_grid_width is None else int(requested_grid_width),
        effective_grid_width=int(grid_width), grid_width_source='auto_lcm' if requested_grid_width is None else 'explicit',
        native_widths={t.key: int(t.native_width) for t in targets},
        source_contract='explicit regular native widths verified against stored fine-position layout',
        sequence_native_cka=sequence, sequence_partition_max_difference={}, cells={}, evidence='descriptive geometric; no universal method ranking, task or causal claim',
        limits=['raw CKA has dimensional bias', 'windows within source series are dependent',
            'PCA rank does not match spectra', 'native widths are caller declarations, not empirical locality checks',
            'common-grid pooling loses fine native variation', 'null p values unadjusted across the reported grid',
            'identity-shuffle null does not condition on input or generator family; significance is not evidence of learned shared features'])
    for target in targets:
        values = tokens[target.key]
        differences = []
        for w in windows:
            if w >= target.native_width and w % target.native_width == 0:
                flat = window_features(values, target.native_width, w, 'native_concat',
                    grid_width).reshape(len(rows), -1)
                differences.append(float(np.max(np.abs(flat - values.reshape(len(rows), -1)))))
        output['sequence_partition_max_difference'][target.key] = max(differences, default=None)
    for window in windows:
        nw = length // window
        for method in methods:
            record = dict(n_test_series=n_test, n_windows_per_series=nw, targets={}, pairs={})
            kernels = {}
            projected = {rank: {} for rank in ranks}
            for target in targets:
                key = target.key
                try:
                    features = window_features(tokens[key], target.native_width, window, method, grid_width)
                except ValueError as error:
                    record['targets'][key] = {'status': 'skipped', 'reason': str(error)}
                    continue
                held = features[test].reshape(n_test * nw, -1)
                kernels[key] = centered_kernel(held)
                detail = dict(status='measured', dimension=features.shape[-1], ranks={})
                for rank in ranks:
                    try:
                        low, diag = project_rank(features[train].reshape(int(train.sum()) * nw,
                                -1), held, rank, seed)
                        projected[rank][key] = centered_kernel(low)
                        detail['ranks'][str(rank)] = diag
                    except ValueError as error:
                        detail['ranks'][str(rank)] = {'status': 'skipped', 'reason': str(error)}
                record['targets'][key] = detail
            for a, b in pairs:
                key = f'{a.key}|{b.key}'
                scores = {}
                if a.key in kernels and b.key in kernels:
                    scores['raw'] = kernel_similarity(kernels[a.key], kernels[b.key],
                        n_test, nw, permutations, seed)
                    for rank in ranks:
                        if a.key in projected[rank] and b.key in projected[rank]:
                            scores[f'rank{rank}'] = kernel_similarity(projected[rank][a.key],
                                projected[rank][b.key], n_test, nw, permutations, seed)
                record['pairs'][key] = scores or {'status': 'skipped', 'reason': 'target representation unavailable; see target diagnostics'}
            cell = f'{window}_{method}'
            output['cells'][cell] = record
            if progress is not None:
                progress(cell, output)
    return output

def effective_grid_metadata(result, protocol, native_widths):
    """Resolve recorded grids or honestly derive legacy auto LCM from declared widths."""
    missing = object()
    requested = result.get('requested_grid_width', protocol.get('requested_grid_width', protocol.get('grid_width', missing)))
    effective = result.get('effective_grid_width', result.get('grid_width', protocol.get('effective_grid_width')))
    derived = effective is None
    if isinstance(native_widths, dict):
        widths = list(native_widths.values())
    elif isinstance(native_widths, list):
        widths = [row.get('native_width') if isinstance(row, dict) else row[2] if isinstance(row, (list, tuple)) and len(row) == 3 else None for row in native_widths]
    else:
        widths = []
    auto_width = lcm(*widths) if widths and all(not isinstance(w, bool) and isinstance(w, (int, np.integer)) and w > 0 for w in widths) else None
    legacy_common32 = effective is None and requested is missing and ('common32_concat' in protocol.get('methods', []) or any(cell.endswith('_common32_concat') for cell in result.get('cells', {})))
    if legacy_common32:
        effective = 32
    if effective is None and requested is not missing and requested is not None:
        effective = requested
    if effective is None and requested is None:
        effective = auto_width
    if effective is None:
        return None, 'unavailable; no effective width or valid auto-grid declarations recorded'
    if legacy_common32:
        comparison = f'auto LCM = {auto_width} steps derived from declared native widths' if auto_width is not None else 'declared native-width LCM unavailable'
        note = f'32 steps (legacy common32_concat; {comparison}; request not recorded)'
    elif requested is None:
        note = f'auto LCM = {effective} steps'
        if derived:
            note += ' (derived from declared native widths; legacy metadata)'
    elif requested is missing:
        note = f'{effective} steps (effective width recorded; request not recorded)'
    else:
        note = f'{effective} steps (explicit request)'
    return int(effective), note


def render_window_report(result, output_path, protocol=None):
    """Render saved measurements only, with an offline chart and auditable tables."""
    from html import escape
    from pathlib import Path
    import json
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    protocol = protocol or {}
    cells = result['cells']
    pairs = sorted({pair for cell in cells.values() for pair in cell['pairs']})
    sequence = result.get('sequence_native_cka', result.get('sequence_cka', {}))
    figure = make_subplots(rows=max(1, len(pairs)), cols=1, subplot_titles=pairs,
        vertical_spacing=min(0.12, 0.5 / max(1, len(pairs) - 1)))
    lines = {}
    score_rows = []
    spectrum_rows = []
    skip_rows = []
    for cell_name, cell in cells.items():
        window_text, method = cell_name.split('_', 1)
        window = int(window_text)
        targets = cell.get('targets', cell.get('models', {}))
        for target, detail in targets.items():
            if detail.get('status') == 'skipped':
                skip_rows.append([cell_name, target, detail['reason']])
                continue
            for rank, diagnostic in detail.get('ranks', detail.get('pca', {})).items():
                if diagnostic.get('status') == 'skipped':
                    skip_rows.append([cell_name, target + ' rank' + str(rank), diagnostic['reason']])
                    continue
                spectrum_rows.append([cell_name, target, detail.get('dimension'), rank,
                        diagnostic.get('participation_ratio'), diagnostic.get('variance_retained', diagnostic.get('explained_variance_fraction'))])
        for pair, metrics in cell['pairs'].items():
            if metrics.get('status') == 'skipped':
                skip_rows.append([cell_name, pair, metrics['reason']])
                continue
            for metric, score in metrics.items():
                lines.setdefault((pair, method, metric), []).append((window, score['value']))
                score_rows.append([cell_name, pair, metric, cell['n_test_series'], cell['n_windows_per_series'],
                        score['value'], score['null_p95'], score['null_p_plusone'], score['n_permutations']])
    colors = {'mean': '#2563eb', 'native_concat': '#ea580c', 'common_grid': '#16a34a',
        'common32_concat': '#16a34a'}
    dashes = {'raw': 'dot', 'rank8': 'solid', 'rank32': 'dash'}
    for (pair, method, metric), values in sorted(lines.items()):
        values = sorted(values)
        row = pairs.index(pair) + 1
        figure.add_trace(go.Scatter(x=[x for x, y in values], y=[y for x, y in values],
                mode='lines+markers', name=f'{method} · {metric}', legendgroup=f'{method}_{metric}',
                showlegend=row == 1, line=dict(color=colors.get(method), dash=dashes.get(metric, 'dashdot'))),
            row=row, col=1)
    all_windows = sorted({int(key.split('_', 1)[0]) for key in cells})
    for index, pair in enumerate(pairs, 1):
        if pair in sequence:
            figure.add_trace(go.Scatter(x=all_windows, y=[sequence[pair]] * len(all_windows),
                    mode='lines', name='complete sequence (different row unit)', legendgroup='sequence',
                    showlegend=index == 1, line=dict(color='#333333', dash='longdash')), row=index, col=1)
        figure.update_xaxes(title_text='Physical window (steps)', tickvals=all_windows, row=index, col=1)
        figure.update_yaxes(title_text='Linear CKA', row=index, col=1)
    figure.update_layout(height=max(450, 350 * len(pairs)), title='Window representations at fixed target layers and source series',
        template='plotly_white', legend=dict(orientation='h', y=-0.08))
    chart = figure.to_html(full_html=False, include_plotlyjs=True)

    def table(headers, rows):
        """Escape every artifact value before embedding the static audit tables."""

        def value(x):
            return f'{x:.9g}' if isinstance(x, float) else str(x)
        return '<table><thead><tr>' + ''.join(('<th>' + escape(h) + '</th>' for h in headers)) + '</tr></thead><tbody>' + ''.join(('<tr>' + ''.join(('<td>' + escape(value(x)) + '</td>' for x in row)) + '</tr>' for row in rows)) + '</tbody></table>'
    crop = result.get('crop', protocol.get('crop'))
    context = result.get('source_context', protocol.get('context', protocol.get('source_context')))
    train = result.get('train_rows', protocol.get('train_rows', []))
    test = result.get('test_rows', protocol.get('test_rows', []))
    provenance = dict(source=result.get('source', protocol.get('source')), source_context=context,
        crop=crop, n_train_series=len(train), n_test_series=len(test), native_widths=result.get('native_widths',
            protocol.get('targets')), common_grid_width=result.get('grid_width'), alignment_provenance=protocol.get('alignment_provenance',
            'see source alignment artifacts'), sequence_partition_max_difference=result.get('sequence_partition_max_difference'))
    effective_width, grid_note = effective_grid_metadata(result, protocol, provenance['native_widths'])
    provenance['common_grid_width'] = effective_width
    provenance['common_grid_width_note'] = grid_note
    html = '<!doctype html><html><head><meta charset="utf-8"><title>Window robustness diagnostic</title><style>body{font-family:system-ui,sans-serif;max-width:1250px;margin:auto;padding:24px;line-height:1.5}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto;margin:20px 0}td,th{border:1px solid #ddd;padding:6px;text-align:left}th{background:#f1f5f9}pre{white-space:pre-wrap;background:#f8fafc;padding:16px}.notice{border-left:4px solid #d97706;padding:12px;background:#fffbeb}</style></head><body>'
    html += '<h1>Window robustness diagnostic</h1><p class="notice">Descriptive geometric evidence. Changing window size changes the physical extent of each observation and the number of window rows, even when the same complete source series and crop are retained. Whole-series permutations handle dependence; they do not make these scientific units identical. A higher CKA need not mean more shared task information. Matched PCA rank controls nominal feature budget, while effective spectra can still differ. These measurements do not establish shared concepts or causal use.</p>'
    widths = provenance['native_widths']
    if isinstance(widths, dict):
        width_text = '; '.join(f'{target}: {width} steps' for target, width in widths.items())
    elif isinstance(widths, list):
        width_text = '; '.join(f'{target[0]}/{target[1]}: {target[2]} steps' if isinstance(target, list) and len(target) == 3 else str(target) for target in widths)
    else:
        width_text = str(widths)
    html += '<h2>Physical extent and source provenance</h2><p><strong>Source:</strong> ' + escape(str(provenance['source']))
    html += '<br><strong>Extent:</strong> ' + escape(str(context)) + ' context steps; physical crop ' + escape(str(crop))
    html += '. <strong>Series:</strong> ' + str(len(train)) + ' train / ' + str(len(test)) + ' test.'
    html += '<br><strong>Native widths:</strong> ' + escape(width_text) + '</p>'
    html += '<p><strong>Common physical grid:</strong> ' + escape(grid_note) + '</p>'
    full_provenance = dict(summary=provenance, protocol=protocol, source_row_ids={key: result.get(key, protocol.get(key)) for key in ('rows', 'train_rows', 'test_rows')})
    html += '<details><summary>Full scientific provenance, protocol, and source row IDs</summary><pre>' + escape(json.dumps(full_provenance, indent=2)) + '</pre></details>'
    html += '<h2>Shared geometry across windows</h2>' + chart
    html += '<p>Every window size uses the same source series and physical crop. Window rows change with window size; adjacent windows within a series are dependent. The complete-sequence line uses one row per series and is invariant to window partition, so it is a different observational unit. Raw CKA has dimensional bias. PCA matches nominal rank, not effective spectrum or task information.</p>'
    html += '<h2>Similarity and whole-series nulls</h2><p>Nulls permute complete source series and preserve window order. P values are descriptive, plus-one corrected, and unadjusted across the full displayed grid. No best window, layer, rank, or method is selected.</p>'
    html += table(['Window/method', 'Target pair', 'Feature control', 'Test series',
            'Windows/series', 'CKA', 'Null p95', 'Null p (unadjusted)', 'Permutations'], score_rows)
    html += '<h2>Feature dimensions and effective spectra</h2><p>Participation ratio exposes effective rank after train-only PCA; equal nominal rank can still leave unequal spectra.</p>'
    html += table(['Window/method', 'Target', 'Input dimension', 'PCA rank', 'Participation ratio',
            'Variance retained'], spectrum_rows)
    html += '<h2>Unsupported or skipped measurements</h2>' + (table(['Cell', 'Target/control',
                'Reason'], skip_rows) if skip_rows else '<p>No requested measurements were skipped.</p>')
    html += '<h2>Interpretation limits</h2><p>Native widths are explicit regular-span declarations checked against the cached fine-array contract. This check does not establish empirical temporal locality. Common-grid pooling discards fine native variation; concatenation can increase feature dimension. Use separate held-out task probes to investigate information loss. The identity-shuffle null does not condition on input or generator family; significance cannot establish learned shared features.</p></body></html>'
    Path(output_path).write_text(html, encoding='utf-8')
