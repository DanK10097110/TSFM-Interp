"""Bound scientific declarations, global screening and frozen fresh mock replication."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
import run_feature_interference as producer
import tsfm_lens.analysis.feature_interference as core
from tsfm_lens.config import ModelConfig
from tsfm_lens.experiments.feature_search import replication_planner as planner


@pytest.fixture(autouse=True)
def compact_render(monkeypatch):
    """The planner checks artifacts, so avoid embedding Plotly in every test report."""
    monkeypatch.setattr(core, 'render_interference_report', lambda report, path, *args: Path(path).write_text('test report', encoding='utf-8'))


def prepared_case(root, name, seed_base, n):
    """Use the actual prepare contract without loading any model."""
    directory = root / name
    directory.mkdir()
    config = dict(design=dict(n_bundles=n, seed_base=seed_base, native_background=True),
        models=[dict(name=name, adapter='mock_patch', checkpoint='') for name in ('A', 'B')],
        inference=dict(primary_estimand='sign_probability', n_boot=19, n_permutations=19))
    config_path = directory / 'input.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    prepared = directory / 'prepared'
    producer.prepare(config_path, prepared)
    return dict(id=name, prepared=str(prepared), report=str(directory / 'report'), model_order=['A', 'B'])


def emit_report(case, differences):
    """Create explicitly bound synthetic forecasts and run actual saved-file analysis."""
    prepared, out = Path(case['prepared']), Path(case['report'])
    out.mkdir()
    config = json.loads((prepared / 'protocol.json').read_text(encoding='utf-8'))
    (out / 'protocol.json').write_bytes((prepared / 'protocol.json').read_bytes())
    metadata = json.loads((prepared / 'metadata.json').read_text(encoding='utf-8'))
    with np.load(prepared / 'corpus.npz') as corpus:
        truth = corpus['component_future'].sum(1) + np.array([row['offset'] for row in metadata])[:, None]
        row_ids = corpus['row_ids'].copy()
    groups = np.array([row['seed'] for row in metadata])
    scale = np.array([row['scale'] for row in metadata])[:, None]
    difference = np.asarray(differences)
    losses = dict(A=np.maximum(-difference, 0) + .1, B=np.maximum(difference, 0) + .1)
    paths = {}
    for model in config['models']:
        name = model['name']
        effective = asdict(ModelConfig(**model))
        model_hash = hashlib.sha256(json.dumps(effective, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        paths[name] = out / f'{name}_predictions.npz'
        np.savez_compressed(paths[name], prediction=truth + scale * np.sqrt(losses[name][groups])[:, None],
            row_ids=row_ids, corpus_sha256=np.asarray(producer.file_sha256(prepared / 'corpus.npz')),
            model_name=np.asarray(name), model_config_sha256=np.asarray(model_hash),
            checkpoint_identifier_sha256=np.asarray(hashlib.sha256(model['checkpoint'].encode()).hexdigest()))
    return producer.analyze_files(config, prepared / 'corpus.npz', prepared / 'metadata.json', paths, out, prepared_source=prepared)


def catalog_file(root, cases):
    """Declare the whole family and unit-specific thresholds before result reads."""
    path = root / 'catalog.json'
    path.write_text(json.dumps(dict(primary_estimand='sign_probability', alpha=.05,
        minimum_absolute_mean={key: .1 if key.startswith('untouched_phase') else .05 for key in planner.ENDPOINTS}, cases=cases)), encoding='utf-8')
    return path


def test_registration_then_global_holm_tiny_effect_and_closed_screen(tmp_path):
    """Casewise significance, tiny effects and an opposing mean cannot advance."""
    cases = [prepared_case(tmp_path, 'casewise', 5000, 8), prepared_case(tmp_path, 'tiny', 6000, 64),
             prepared_case(tmp_path, 'opposing', 7000, 64), prepared_case(tmp_path, 'missing', 8000, 8)]
    malformed = Path(cases[-1]['report'])
    malformed.mkdir()
    (malformed / 'results.json').write_text('not JSON', encoding='utf-8')
    registered = planner.register(catalog_file(tmp_path, cases), tmp_path / 'registered')
    assert registered['timing'].startswith('Registration precedes screen result reads')
    local = emit_report(cases[0], np.full(8, .2))
    emit_report(cases[1], np.full(64, .001))
    emit_report(cases[2], np.r_[np.ones(63), -1000.])
    endpoint = 'B-minus-A/mixed_scaled_mse'
    assert local['sign_comparisons'][endpoint]['p_holm'] < .05
    screened = planner.screen(tmp_path / 'registered/registration.json', tmp_path / 'screened')
    rows = {row['claim']: row for row in screened['claims']}
    assert screened['family_size'] == 16
    assert rows['casewise::' + endpoint]['global_p_holm'] >= .05
    assert rows['tiny::' + endpoint]['global_p_holm'] < .05
    assert 'below_prespecified_absolute_mean' in rows['tiny::' + endpoint]['reasons']
    assert 'mean_and_selected_direction_disagree' in rows['opposing::' + endpoint]['reasons']
    assert all(row['raw_selected_p'] == 1 for row in screened['claims'] if row['case_id'] == 'missing')
    assert not screened['replication_needed']
    closed = planner.plan(tmp_path / 'screened/screen.json', tmp_path / 'closed')
    assert closed['status'] == 'closed_no_replication_needed'
    assert not list((tmp_path / 'closed').rglob('config.yaml'))


def positive_screen(root):
    """Freeze a single practically large native-error endpoint for mock replication."""
    case = prepared_case(root, 'positive', 10000, 16)
    planner.register(catalog_file(root, [case]), root / 'registered')
    emit_report(case, np.full(16, .2))
    result = planner.screen(root / 'registered/registration.json', root / 'screened')
    assert sum(row['qualified'] for row in result['claims']) == 1
    return root / 'screened/screen.json'


def execute_mock_plan(plan_value, difference):
    """Explicitly execute only CPU prepare/analysis stages, never a model launch."""
    for case in plan_value['cases']:
        producer.prepare(Path(case['config_path']), Path(case['prepared']))
        emit_report(case, np.full(plan_value['n_bundles'], difference))


def test_two_stage_mock_replication_preserves_only_frozen_family_and_direction(tmp_path):
    """Fresh local seed indices can recur; generating IDs and frozen signs govern."""
    screen_path = positive_screen(tmp_path)
    frozen = planner.plan(screen_path, tmp_path / 'replication_plan', fresh_seed_base=20000, n_bundles=16)
    assert len(frozen['claims']) == 1
    assert len(frozen['commands']) == 2
    assert not Path(frozen['cases'][0]['prepared']).exists()
    missing = planner.analyze_replication(tmp_path / 'replication_plan/plan.json', tmp_path / 'missing_replication')
    assert missing['family_size'] == 1
    assert missing['claims'][0]['raw_selected_p'] == 1
    assert not missing['all_confirmed']
    execute_mock_plan(frozen, .2)
    confirmed = planner.analyze_replication(tmp_path / 'replication_plan/plan.json', tmp_path / 'confirmed')
    assert confirmed['family_size'] == 1
    assert confirmed['all_confirmed']
    reverse = planner.plan(screen_path, tmp_path / 'reverse_plan', fresh_seed_base=21000, n_bundles=16)
    execute_mock_plan(reverse, -.2)
    refused = planner.analyze_replication(tmp_path / 'reverse_plan/plan.json', tmp_path / 'reverse_result')
    assert not refused['all_confirmed']
    assert 'failed_frozen_replication_direction' in refused['claims'][0]['reasons']


def test_generator_seed_overlap_plan_tampering_and_forecast_binding_are_refused(tmp_path):
    """Local IDs and a valid-looking NPZ cannot substitute for frozen source identity."""
    screen_path = positive_screen(tmp_path)
    with pytest.raises(ValueError, match='generator_seed range overlaps'):
        planner.plan(screen_path, tmp_path / 'overlap', fresh_seed_base=10008, n_bundles=16)
    frozen = planner.plan(screen_path, tmp_path / 'frozen', fresh_seed_base=22000, n_bundles=16)
    wrong_config = json.loads(json.dumps(frozen['cases'][0]['config']))
    wrong_config['design']['amplitudes'] = [.75, 2.]
    wrong_path = tmp_path / 'altered_replication.yaml'
    wrong_path.write_text(yaml.safe_dump(wrong_config), encoding='utf-8')
    producer.prepare(wrong_path, Path(frozen['cases'][0]['prepared']))
    with pytest.raises(ValueError, match='frozen plan'):
        planner.analyze_replication(tmp_path / 'frozen/plan.json', tmp_path / 'altered_design_result')
    path = tmp_path / 'frozen/plan.json'
    altered = json.loads(path.read_text(encoding='utf-8'))
    altered['n_bundles'] += 1
    path.write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(ValueError, match='tampered'):
        planner.analyze_replication(path, tmp_path / 'tampered_result')
    original = json.loads((tmp_path / 'registered/registration.json').read_text(encoding='utf-8'))
    prediction = Path(original['cases'][0]['predictions']['B'])
    with np.load(prediction) as artifact:
        fields = {key: artifact[key] for key in artifact.files}
    fields['corpus_sha256'] = np.asarray('0' * 64)
    np.savez_compressed(prediction, **fields)
    with pytest.raises(ValueError, match='Forecast bytes changed'):
        planner.screen(tmp_path / 'registered/registration.json', tmp_path / 'rebound_result')

    prepared = Path(original['cases'][0]['prepared'])
    fields['corpus_sha256'] = np.asarray(producer.file_sha256(prepared / 'corpus.npz'))
    fields['model_name'] = np.asarray('wrong_predictor')
    np.savez_compressed(prediction, **fields)
    provenance_path = Path(original['cases'][0]['report']) / 'analysis_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    provenance['forecasts']['B']['sha256'] = producer.file_sha256(prediction)
    provenance_path.write_text(json.dumps(provenance), encoding='utf-8')
    with pytest.raises(ValueError, match='Mismatched explicit predictor'):
        planner.screen(tmp_path / 'registered/registration.json', tmp_path / 'wrong_predictor_result')
