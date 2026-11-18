"""The readout must compare random controls on matching valid receivers only."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from experiments.feature_search import dictionary_review as report


def test_invalid_random_receiver_is_not_a_paired_comparator():
    """A large invalid-control decoy must not enter either matched mean."""
    with TemporaryDirectory(prefix='dictionary-review-test-') as temporary:
        root = Path(temporary)
        fit = root/'fit_bolt_cut0_native_k8_s4951000'
        native = root/'native_bolt_cut0_native_k8_s4951000'
        fit.mkdir()
        native.mkdir()
        def write(path, value):
            path.write_text(json.dumps(value), encoding='utf-8')
        execution = dict(returncode=0, seconds=1., peak_rss=1024)
        for dest in (fit, native):
            write(dest/'execution.json', execution)
        write(fit/'results.json', dict(model='bolt', cut='0', method='native', k=8, seed=4951000,
              metrics={'test': dict(nmse=.1, mean_l0=8, dead_fraction=0)},
              budget=dict(trainable_parameters=0),
              probes={v: [] for v in ('code_mean', 'code_rms', 'input_generic', 'metadata_only')}))
        records = []
        for error, null, valid in [(2., 3., True), (100., 1000., False)]:
            effect = dict(random_valid=valid, response_change_nmse=error, random_response_change_nmse=null)
            causal = {direction: dict(reach=True, budgets={'1': effect, '4': effect})
                      for direction in ('sinusoid', 'noise')}
            records.append(dict(domain='d', exact_gates=True, causal_selectivity=causal,
                measurements={'reconstruction': dict(history_variance_nmse=error, random_valid=valid,
                                random_metrics=[dict(history_variance_nmse=null)])}))
        write(native/'results.json', records)
        report.review(root, root/'review')
        row = json.loads((root/'review/forecast.json').read_text(encoding='utf-8'))[0]
        assert row['history_variance_nmse'] == 51.
        assert row['matched_native_history_variance_nmse'] == 2.
        assert row['random_history_variance_nmse'] == 3.
        assert row['random_valid'] == 1
        assert row['ablate1_sinusoid_matched_response_change_nmse'] == 2.
        assert row['ablate1_sinusoid_random_response_change_nmse'] == 3.
