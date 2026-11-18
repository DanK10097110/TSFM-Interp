"""Render a portable research overview from selected, already reported results.

No models or statistical analyses run here. --summary rebuilds without run stores.
The whole examples tree is portable. Images are copied beside the rebuilt HTML;
links to reports outside an arbitrary output location are visibly unavailable.
"""
import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    assets = root / 'examples/feature_search_20261111'
    spectral = json.loads((assets / 'source_data.json').read_text())
    native = json.loads((assets / 'native_head_replication_source_data.json').read_text())
    interference = json.loads((root / 'examples/feature_interference/results.json').read_text())
    rows = [
        ['Controlled composition', 'Fresh two-frequency replication',
         'Mixed-minus-pure model-gap interaction: ' + str(spectral['panel_a']['interaction']['value']),
         'Promoted behavioral input-intervention diagnostic; not a hidden-feature cause.'],
        ['Actual native head', 'Fresh 128-seed, three-model replication',
         '; '.join(f"{name}: {model['deltas']['0.01']['absolute_attribution']['total_energy_ratio']['value']:.9g}"
                   for name, model in native['results']['models'].items()),
         'Absolute off-frequency energy ratios at delta 0.01; recoverability and native use differ.'],
        ['Window sensitivity', 'Matched physical extent, whole-series grouped comparisons',
         'Mean, native concatenation and common-grid concatenation; fixed PCA ranks',
         'Choice, rank and data dependence; no universal best representation.'],
        ['Frozen SAE hierarchy', 'ROADMAP §48.2',
         'All three promotion conditions fail; uncompressed codes score 1.0 on both tasks',
         'Closed limited frozen-dictionary ablation; not a full SAE++ rejection.'],
        ['Specialized phase subspaces', 'ROADMAP §48.6', 'All four specialization gates fail',
         'Accurate phase decoding does not establish a privileged causal direction.'],
        ['Frozen TimesFM probe', 'ROADMAP §48.13',
         f"Off-target relative RMS {spectral['failed_tangent_gate']['timesfm_offtarget_relative_RMS']:.6g} exceeds frozen threshold {spectral['failed_tangent_gate']['threshold']}",
         'Retention comparison fails; significant energy contrasts do not rescue the gate.'],
        ['Trend–seasonality', 'ROADMAP §48.16', '0/12 pass the absolute practical threshold 0.05',
         'Closed without promotion; two-frequency behavior is not universal composition failure.'],
    ]
    sources = {}
    for relative in ['FINDINGS.md', 'ROADMAP.md', 'examples/feature_search_20261111/source_data.json',
                     'examples/feature_search_20261111/native_head_replication_source_data.json',
                     'examples/feature_interference/results.json', 'examples/window_robustness/results.json']:
        sources[relative] = sha(root / relative)
    return dict(rows=rows, sources=sources, native_comparisons=native['results']['comparisons'],
                interference_primary_estimand=interference.get('primary_estimand', 'mean'),
                literature=[['Pandey et al.: Internal Semantics', 'https://arxiv.org/abs/2511.15324'],
                            ['Bao et al.: Universal Redundancies', 'https://proceedings.mlr.press/v306/bao26c.html'],
                            ['Yildirim: PatchTST superposition study', 'https://arxiv.org/abs/2605.05151'],
                            ['SAE++: Cascaded Sparse Autoencoders', 'https://arxiv.org/abs/2606.16193']],
                scope='Selected reported results only; this manifest is not a new scientific analysis.')


def render(summary):
    escape = html.escape
    rows = ''.join('<tr>' + ''.join('<td>' + escape(str(cell)) + '</td>' for cell in row) + '</tr>'
                   for row in summary['rows'])
    links = ' · '.join(f'<a href="{escape(url)}">{escape(label)}</a>' for label, url in summary['literature'])
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Controlled feature comparison in time-series models</title><style>
body{font:17px/1.55 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#182332;background:#fafbfd}
h1{font-size:2.25rem;line-height:1.15}h2{margin-top:2rem}a{color:#0755a2}figure{margin:24px 0}img{max-width:100%;height:auto;background:white}figcaption{font-size:.9rem;color:#475569}table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{text-align:left;vertical-align:top;border:1px solid #cbd5e1;padding:10px}th{background:#e7eef6}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.75rem}.scope{padding:16px;background:#e7eef6;border-left:4px solid #365d83}</style>
<h1>Compare what models retain, use and mix</h1>
<p>A controlled feature-comparison workflow links known input composition, actual forecast-head recoverability and native use, context/patch boundaries, and robustness to representation choices. These are complementary measurements: accurate decoding alone does not establish effective native use.</p>
<p><a href="../feature_interference/report.html">Reproduce the feature-interference audit</a> · <a href="../window_robustness/report.html">Inspect window robustness</a> · <a href="study_summary.json">Portable source summary</a> · <a href="../../tsfm_lens/docs/feature_search_plan.md">Replication planner usage</a></p>
<p class="scope">No reliable model-exclusive causal feature has been established. Behavioral input interventions are not isolated hidden-feature interventions. This comparison does not establish a universal winning model or pooling method.</p>
<figure><img src="spectral_composition_summary.png" alt="Four panels comparing composition behavior, head information, context boundaries and native branch attribution"><figcaption>Existing four-panel synthesis. Each panel retains its own seed cohort, units and gate; panels are not a pooled statistical test. Native component sums are diagnostics, not a deployment improvement guarantee.</figcaption></figure>
<h2>Evidence and decisions</h2><table><thead><tr><th>Branch</th><th>Scope</th><th>Reported evidence</th><th>Decision and limit</th></tr></thead><tbody>''' + rows + '''</tbody></table>
<figure><img src="native_head_replication.png" alt="Fresh native-head energy ratios and exact conditional sign probabilities"><figcaption>Fresh native-head replication: 128 matched seeds, three cached models, no fitted probes. Energy intervals are pointwise seed-bootstrap intervals. Exact conditional majority probabilities answer a different question from mean energy superiority; branch attribution does not isolate a unique feature.</figcaption></figure>
<h2>Statistical and literature boundaries</h2>
<p>The historical native interference example retains its registered mean contrasts. Its sign-flip inference requires symmetry/exchangeability and is not calibrated for arbitrary skew. New prospective studies can predeclare exact two-sided q = P(D &gt; 0 | D ≠ 0, jointly identifiable). Exact q evidence does not establish mean-MSE superiority or practical relevance. Endpoint families receive separate Holm correction; secondary evidence cannot rescue the selected primary family. Confidence intervals are pointwise. Undefined phase excludes the matched bundle and changes the conditional estimand.</p>
<p>Whole-series window permutations preserve dependent windows and assess identity alignment, not learned features conditional on inputs. Raw CKA depends on dimensionality and covariance spectrum; matched PCA rank does not match the full spectrum. Sequence-level concatenation supplies a window-partition invariance control.</p>
<p>Prior work already studies TSFM concepts, composition and component attribution. No novelty or universal monosemanticity claim is made. ''' + links + '''.</p>
<p>Rebuild this page without model caches or run stores, from the repository root:</p><pre>python tsfm_lens/experiments/feature_search/research_overview.py --summary examples/feature_search_20261111/study_summary.json --out examples/feature_search_20261111_rebuild</pre>
<details><summary>Source hashes and selected replication comparisons</summary><pre>''' + escape(json.dumps(summary, indent=2)) + '''</pre></details></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.repo_root) == bool(args.summary):
        parser.error('provide exactly one of --repo-root or --summary')
    summary = json.loads(args.summary.read_text()) if args.summary else collect(args.repo_root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'study_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    source_dir = args.summary.parent if args.summary else args.repo_root / 'examples/feature_search_20261111'
    for name in ('spectral_composition_summary.png', 'native_head_replication.png'):
        source = source_dir / name
        target = args.out / name
        if not source.is_file():
            raise FileNotFoundError(f'Portable overview requires image beside summary: {source}')
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    document = render(summary)
    def local_link(match):
        target, label = match.groups()
        if target.startswith(('https:', '#')) or (args.out / target).is_file():
            return match.group(0)
        return '<span>' + label + ' (unavailable in this output location)</span>'
    document = re.sub(r'<a href="([^"]+)">(.*?)</a>', local_link, document)
    (args.out / 'index.html').write_text(document)


if __name__ == '__main__':
    main()
