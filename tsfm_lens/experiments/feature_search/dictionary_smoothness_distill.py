"""Distill the audited saved-only geometry without further dictionary computation."""
from pathlib import Path
import numpy as np
from experiments.feature_search.dictionary_smoothness_study import read, write, sha, knowledge


def distill(out):
    """Preserve exact source-level effect ranges and limitations in persistent knowledge."""
    if (out / 'leading_findings.json').exists():
        raise FileExistsError('Refuse to overwrite the completed distillation')
    record = read(out / 'compact_record.json')
    assert record['status'] == 'COMPLETE_SAVED_ONLY_AUDITED'
    assert record['audit']['valid']
    arms = record['arms']
    metrics = [
        'raw_chord_RMS', 'decoded_actual_chord_RMS', 'isolated_dictionary_chord_RMS',
        'actual_to_raw_floor100', 'isolated_to_raw_floor100',
        'any_chord_support_switch', 'chord_support_turnover',
        'isolated_resolved_vs_precision', 'support_FP32_FP64_disagreement',
        'midpoint_reconstruction_relative_RMS', 'decoded_secant_to_raw',
        'raw_above_10x_proxy',
    ]
    groups = []
    details = []
    for model in ('bolt', 'timesfm'):
        for policy in ('native', 'locked'):
            rows = [a for a in arms if a['model'] == model and a['policy'] == policy
                    and a['method'] in ('pca_adaptive', 'topk', 'nested')]
            assert len(rows) == 6 and all(a['n_sources'] == 16 for a in rows)
            ratios = []
            counts = []
            for arm in rows:
                overall = arm['overall']
                actual = np.asarray(overall['decoded_actual_chord_RMS']['per_source'])
                raw = np.asarray(overall['raw_chord_RMS']['per_source'])
                ratio = float(actual.mean() / raw.mean())
                ratios.append(ratio)
                counts.append(int(np.sum(actual > raw)))
                details.append(dict(
                    model=model, policy=policy, method=arm['method'], k=arm['k'],
                    source_mean_metrics={key: overall[key]['mean'] for key in metrics},
                    actual_to_raw_ratio_of_source_means=ratio,
                    sources_with_larger_actual_chord=int(np.sum(actual > raw)),
                    source_count=16,
                    support_switch_conditioning=arm['support_switch_conditioning']))
            groups.append(dict(
                model=model, policy=policy,
                actual_to_raw_ratio_of_source_means_range=[min(ratios), max(ratios)],
                sources_with_larger_actual_chord_range=[min(counts), max(counts)],
                source_mean_metric_ranges={key: [min(a['overall'][key]['mean'] for a in rows),
                                                 max(a['overall'][key]['mean'] for a in rows)]
                                           for key in metrics},
                smallest_axis_radius_mean_actual_to_raw_floor100=min(
                    g['metrics']['actual_to_raw_floor100']['mean']
                    for a in rows for g in a['by_axis_radius']),
                no_support_switch_mean_isolated_RMS_range=[
                    min(a['support_switch_conditioning']['no_switch']['mean_isolated_RMS'] for a in rows),
                    max(a['support_switch_conditioning']['no_switch']['mean_isolated_RMS'] for a in rows)]))
    findings = dict(
        conclusion='Frozen hard-sparse maps introduce substantial chord roughness in these saved first-block residual states; raw model states also have nonzero curvature.',
        ratio_definition='Ratio of source-averaged actual decoded chord RMS to source-averaged raw chord RMS, pooling the balanced frozen paths; not mean pathwise curvature ratio.',
        groups=groups, sparse_arms=details, audit=record['audit'], execution=record['execution'],
        tests=dict(passed=6, corrected_regression_plants_detected=6,
                   provenance='Completed preflight tests/plants; first ineffective mutant preserved in verification_failure.json and failed_plants_v1.py.'),
        qualifications=[
            'All 24 sparse arm means exceed raw-state chord means, but individual sources and paths need not do so.',
            'Isolated dictionary chords use off-manifold activation interpolation; actual-path chords use saved native responses at the actual input midpoint.',
            'Identity and fixed first-k PC projection have isolated chord RMS at roundoff; adaptive signed PCA TopK is nonlinear, not fixed rank-k projection.',
            'Support switches coincide with resolved isolated curvature; unswitched learned maps can still include ReLU nonlinearity in principle.',
            'FP64 only changes dictionary arithmetic over saved FP32 model states; it does not establish native-model FP64 accuracy.',
            'These compressed representations have nonzero reconstruction error and altered secant response; results do not rank semantic quality.',
            'Source n=16 synthetic backgrounds. One dictionary seed, k8/16, first-block residual coordinates, no inferential significance claim or general monosemanticity claim.',
            'Nested uses the frozen simplified shared-code two-prefix recipe, not a full Matryoshka reproduction.',
        ],
        verification_failure_preserved=record['verification_failure_preserved'],
        evidence_class='L1 saved-array methodological geometry', native_calls=0, fit_calls=0,
        compact_record_sha256=sha(out / 'compact_record.json'),
        distiller_sha256=sha(__file__),
        next_decision='Close this saved-only comparison. Treat decoded sparse-path roughness as encoder-dependent and require raw-state, fixed-linear, support-turnover and precision controls in later interpretation; no new native/fitting/grid launch.')
    write(out / 'leading_findings.json', findings)
    knowledge(out, 'distillation', 'complete', findings, findings['next_decision'])
    persisted = read(out / 'knowledge.json')
    assert persisted['exact_findings'] == findings
    assert any(event['status'] == 'failed' for event in persisted['events'])
    print(dict(status='COMPLETE_SAVED_ONLY_AUDITED', groups=groups,
               knowledge=str(out / 'knowledge.json')), flush=True)


if __name__ == '__main__':
    distill(Path('runs/dictionary_smoothness4965000'))
