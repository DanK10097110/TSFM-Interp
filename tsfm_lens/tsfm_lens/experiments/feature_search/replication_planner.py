"""Auditable discovery registration and frozen replication planning, without inference.

Only explicitly catalogued prepared feature-interference artifacts are read.
Registration never reads results. Content hashes detect edits, but are not
cryptographic proof that nobody inspected outcomes before declaring a catalog.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import yaml
import run_feature_interference as producer
from ...analysis.feature_interference import validate_factorial
from ...analysis.stats import holm
from ...config import ModelConfig

ENDPOINTS = ('untouched_phase_phi', 'untouched_phase_psi', 'mixed_scaled_mse', 'model_by_composition')


def _canonical(value):
    """Hash semantic JSON without accepting NaN or infinity."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    """Name exact semantic declarations by their SHA256 digest."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path):
    """Read only the explicitly supplied JSON artifact."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sealed(path):
    """Refuse a modified declaration before reading any dependent results."""
    value = _read(path)
    claimed = value.pop('declaration_sha256', None)
    if claimed != _digest(value):
        raise ValueError('Declaration or plan was tampered with')
    return dict(value, declaration_sha256=claimed)


def _save(out, name, value):
    """Create a new output directory and a write-once, content-hashed declaration."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    value = dict(value, declaration_sha256=_digest(value))
    with (out / name).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    return value


def _prepared(directory, expected=None):
    """Validate the producer's exact prepared hashes, physical rows and generator IDs."""
    directory = Path(directory).resolve()
    manifest = _read(directory / 'provenance.json')
    paths = dict(protocol=directory / 'protocol.json', corpus=directory / 'corpus.npz',
                 metadata=directory / 'metadata.json', config=directory / 'config.yaml')
    hashes = {name: producer.file_sha256(path) for name, path in paths.items()}
    if any(hashes[name] != manifest.get(name + '_sha256') for name in hashes):
        raise ValueError('Prepared source/config/corpus hash binding changed')
    design, config = producer.validate_config(_read(paths['protocol']))
    _, original_config = producer.load_config(paths['config'])
    if _canonical(config) != _canonical(original_config):
        raise ValueError('Prepared config and protocol disagree')
    if expected is not None and _canonical(config) != _canonical(expected):
        raise ValueError('Replication design/model order/inference differs from the frozen plan')
    metadata = _read(paths['metadata'])
    with np.load(paths['corpus'], allow_pickle=False) as corpus:
        validate_factorial(corpus['context'], corpus['component_future'], metadata, corpus['row_ids'], design)
        row_ids, shape = corpus['row_ids'].copy(), corpus['component_future'].shape[::2]
    return dict(prepared=str(directory), config=config, hashes=hashes,
                generator_ids=sorted({row['generator_seed'] for row in metadata}),
                row_ids=row_ids, prediction_shape=shape)


def _keys(order):
    """Keep all four endpoints per unordered model pair in the frozen model order."""
    return [f'{later}-minus-{earlier}/{endpoint}' for earlier, later in combinations(order, 2) for endpoint in ENDPOINTS]


def _report(case, source):
    """Require exact producer/corpus/forecast/model bindings; never guess identities."""
    directory = Path(case['report'])
    required = [directory / name for name in ('results.json', 'analysis_provenance.json', 'protocol.json')]
    required += [Path(path) for path in case['predictions'].values()]
    if not all(path.is_file() for path in required):
        return None, 'missing report, provenance, or forecast artifact'
    report, provenance = _read(required[0]), _read(required[1])
    if producer.file_sha256(required[2]) != source['hashes']['protocol'] or provenance.get('protocol_sha256') != source['hashes']['protocol']:
        raise ValueError('Completed report protocol binding changed')
    for name in ('corpus', 'metadata'):
        if provenance.get(name, {}).get('sha256') != source['hashes'][name]:
            raise ValueError('Completed report is bound to another corpus or metadata')
    config = source['config']
    order = [model['name'] for model in config['models']]
    if report.get('model_order') != order or report.get('primary_estimand') != config['inference']['primary_estimand'] or _canonical(report.get('design')) != _canonical(config['design']):
        raise ValueError('Report changed declared design/model order/primary estimand')
    if any(report.get('settings', {}).get(key) != value for key, value in config['inference'].items() if key != 'primary_estimand'):
        raise ValueError('Report inference settings changed after registration')
    if set(provenance.get('forecasts', {})) != set(order):
        raise ValueError('Report forecast model set changed')
    for model in config['models']:
        name, path = model['name'], Path(case['predictions'][model['name']])
        if producer.file_sha256(path) != provenance['forecasts'][name].get('sha256'):
            raise ValueError('Forecast bytes changed after completed analysis')
        artifact = producer.load_forecast_artifact(path, source['hashes']['corpus'])
        effective = asdict(ModelConfig(**model))
        expected = dict(model_name=name, model_config_sha256=_digest(effective),
                        checkpoint_identifier_sha256=hashlib.sha256(model['checkpoint'].encode()).hexdigest())
        if any(key not in artifact for key in expected):
            return None, 'missing explicit predictor declarations'
        if any(artifact[key] != value for key, value in expected.items()):
            raise ValueError('Mismatched explicit predictor declarations')
        if not np.array_equal(artifact['row_ids'], source['row_ids']) or artifact['prediction'].shape != source['prediction_shape'] or not np.isfinite(artifact['prediction']).all():
            raise ValueError('Forecast row identity, shape or finiteness mismatch')
    report['_bound_report_sha256'] = producer.file_sha256(required[0])
    report['_bound_provenance_sha256'] = producer.file_sha256(required[1])
    return report, None


def _rows(case, source, estimand, allowed=None):
    """Represent every registered endpoint, including missing and untestable ones."""
    report, unavailable = _report(case, source)
    order = [model['name'] for model in source['config']['models']]
    family = {} if report is None else report['mean_comparisons' if estimand == 'mean' else 'sign_comparisons']
    means = {} if report is None else report['mean_comparisons']
    if set(family) - set(_keys(order)):
        raise ValueError('Completed report contains an altered endpoint family')
    rows = []
    for key in _keys(order):
        if allowed is not None and f"{case['id']}::{key}" not in allowed:
            continue
        effect, mean_effect = family.get(key), means.get(key, {})
        missing = unavailable or ('missing endpoint' if effect is None else effect.get('untestable'))
        p = 1. if missing else float(effect['p'])
        mean = mean_effect.get('value')
        value = None if missing else effect.get('value')
        if not np.isfinite(p) or not 0 <= p <= 1 or (mean is not None and not np.isfinite(mean)) or (value is not None and (not np.isfinite(value) or (estimand == 'sign_probability' and not 0 <= value <= 1))):
            raise ValueError('Invalid selected-endpoint p or mean estimate')
        direction = 0 if value is None else int(np.sign(value - (.5 if estimand == 'sign_probability' else 0)))
        rows.append(dict(claim=f"{case['id']}::{key}", case_id=case['id'], endpoint=key,
            mean=mean, raw_selected_p=p, selected_value=value, selected_direction=direction, unavailable=missing,
            source_report_sha256=None if report is None else report['_bound_report_sha256'],
            source_analysis_provenance_sha256=None if report is None else report['_bound_provenance_sha256']))
    return rows


def _decide(rows, thresholds, alpha, directions=None):
    """Correct the whole fixed family before effect-size and direction gates."""
    adjusted = holm({row['claim']: row['raw_selected_p'] for row in rows})
    for row in rows:
        minimum = thresholds[row['endpoint'].split('/')[-1]]
        mean_direction = 0 if row['mean'] is None else int(np.sign(row['mean']))
        agreement = mean_direction != 0 and mean_direction == row['selected_direction']
        reasons = []
        if row['unavailable']:
            reasons.append('missing_or_untestable')
        if adjusted[row['claim']] >= alpha:
            reasons.append('not_significant_in_global_family')
        if row['mean'] is None or abs(row['mean']) < minimum:
            reasons.append('below_prespecified_absolute_mean')
        if not agreement:
            reasons.append('mean_and_selected_direction_disagree')
        if directions is not None and mean_direction != directions[row['claim']]:
            reasons.append('failed_frozen_replication_direction')
        row.update(global_p_holm=adjusted[row['claim']], minimum_absolute_mean=minimum,
                   mean_unit='radians' if row['endpoint'].split('/')[-1].startswith('untouched_phase') else 'common mixed-variance scaled MSE',
                   mean_direction=mean_direction, direction_agreement=agreement,
                   qualified=not reasons, reasons=reasons)
    return rows


def register(catalog_path, out):
    """Freeze membership/settings/bindings without opening any completed results."""
    catalog = _read(catalog_path)
    estimand, thresholds = catalog['primary_estimand'], catalog['minimum_absolute_mean']
    alpha = catalog.get('alpha', .05)
    if estimand not in ('mean', 'sign_probability') or not 0 < alpha < 1 or set(thresholds) != set(ENDPOINTS) or any(not np.isfinite(value) or value <= 0 for value in thresholds.values()):
        raise ValueError('Declare one estimand and positive endpoint-specific mean thresholds before screening')
    cases, seen = [], set()
    for item in catalog['cases']:
        name = item['id']
        if not name or name in ('.', '..') or any(not (c.isalnum() or c in '._-') for c in name) or name in seen:
            raise ValueError('Case IDs must be unique safe names')
        seen.add(name)
        source = _prepared(item['prepared'])
        config = source['config']
        order = [model['name'] for model in config['models']]
        if item['model_order'] != order or len(order) < 2 or len(set(_keys(order))) != len(_keys(order)) or config['inference']['primary_estimand'] != estimand:
            raise ValueError('All prepared cases must match their expected model order and common estimand')
        report = str(Path(item['report']).resolve())
        predictions = item.get('predictions', {name: str(Path(report) / f'{name}_predictions.npz') for name in order})
        if set(predictions) != set(order):
            raise ValueError('Freeze exactly one expected prediction path for every model')
        cases.append(dict(id=name, prepared=source['prepared'], report=report,
            predictions={name: str(Path(path).resolve()) for name, path in predictions.items()},
            config=config, hashes=source['hashes'], generator_ids=source['generator_ids']))
    if not cases:
        raise ValueError('Register at least one explicit prepared case')
    return _save(out, 'registration.json', dict(kind='registered_discovery', cases=cases,
        primary_estimand=estimand, minimum_absolute_mean=thresholds, alpha=alpha,
        catalog_sha256=producer.file_sha256(catalog_path),
        timing='Registration precedes screen result reads; auditable declaration, not cryptographic proof of no prior inspection'))


def screen(registration_path, out):
    """Read only the frozen catalog, with no screen-time thresholds or reselection."""
    registration = _sealed(registration_path)
    if registration.get('kind') != 'registered_discovery':
        raise ValueError('Screen requires a registered discovery catalog')
    rows = []
    for case in registration['cases']:
        source = _prepared(case['prepared'], case['config'])
        if source['hashes'] != case['hashes']:
            raise ValueError('Registered prepared-case bytes changed')
        rows.extend(_rows(case, source, registration['primary_estimand']))
    _decide(rows, registration['minimum_absolute_mean'], registration['alpha'])
    return _save(out, 'screen.json', dict(kind='discovery_screen',
        registration=str(Path(registration_path).resolve()), registration_sha256=registration['declaration_sha256'],
        family_size=len(rows), claims=rows, primary_estimand=registration['primary_estimand'],
        minimum_absolute_mean=registration['minimum_absolute_mean'], alpha=registration['alpha'],
        replication_needed=any(row['qualified'] for row in rows),
        caution='All cases and negatives retained; qualification is not uniqueness or confirmation'))


def plan(screen_path, out, *, fresh_seed_base=None, n_bundles=None, device='cpu'):
    """Write only frozen configs/commands; never prepare inputs or launch models."""
    screened = _sealed(screen_path)
    if screened.get('kind') != 'discovery_screen':
        raise ValueError('Plan requires a completed discovery screen')
    registration = _sealed(screened['registration'])
    if registration['declaration_sha256'] != screened['registration_sha256']:
        raise ValueError('Screen and registration declarations disagree')
    selected = [row for row in screened['claims'] if row['qualified']]
    if not selected:
        return _save(out, 'plan.json', dict(kind='replication_plan', status='closed_no_replication_needed',
            screen_sha256=screened['declaration_sha256'], claims=[], cases=[], commands=[]))
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (fresh_seed_base, n_bundles)) or fresh_seed_base < 0 or n_bundles < 3:
        raise ValueError('Selected claims require an explicit fresh seed base and bundle count>=3')
    fresh = set(range(fresh_seed_base, fresh_seed_base + n_bundles))
    discovery = {seed for case in registration['cases'] for seed in case['generator_ids']}
    if fresh & discovery:
        raise ValueError('Fresh generator_seed range overlaps discovery generating IDs')
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    cases, commands = [], []
    runner = str(Path(producer.__file__).resolve())
    for original in registration['cases']:
        if not any(row['case_id'] == original['id'] for row in selected):
            continue
        config = json.loads(json.dumps(original['config']))
        config['design'].update(seed_base=fresh_seed_base, n_bundles=n_bundles)
        _, config = producer.validate_config(config)
        directory = out / original['id']
        config_path, prepared, report = directory / 'config.yaml', directory / 'prepared', directory / 'forecast_report'
        text = yaml.safe_dump(config, sort_keys=False)
        cases.append(dict(id=original['id'], config=config, config_path=str(config_path),
            config_sha256=hashlib.sha256(text.encode()).hexdigest(), prepared=str(prepared), report=str(report),
            predictions={model['name']: str(report / f"{model['name']}_predictions.npz") for model in config['models']},
            config_text=text))
        commands.extend([shlex.join([sys.executable, runner, '--config', str(config_path), '--prepare-only', '--out', str(prepared)]),
            shlex.join([sys.executable, runner, '--prepared', str(prepared), '--out', str(report), '--device', device, '--batch-size', '8'])])
    payload = dict(kind='replication_plan', status='frozen_before_replication_forecasts',
        screen_sha256=screened['declaration_sha256'], discovery_generator_ids=sorted(discovery),
        fresh_seed_base=fresh_seed_base, n_bundles=n_bundles,
        primary_estimand=registration['primary_estimand'], minimum_absolute_mean=registration['minimum_absolute_mean'], alpha=registration['alpha'],
        claims=[dict(claim=row['claim'], case_id=row['case_id'], endpoint=row['endpoint'], expected_direction=row['mean_direction']) for row in selected],
        cases=cases, commands=commands,
        timing='Freeze before external prepare/infer commands; exact new prepared hashes and native predictor declarations must bind replication reports',
        caution='Only this selected family may be analyzed; no alternative endpoint rescue or uniqueness claim')
    value = _save(out, 'plan.json', payload)
    for case in cases:
        Path(case['config_path']).parent.mkdir()
        Path(case['config_path']).write_text(case['config_text'], encoding='utf-8')
    return value


def analyze_replication(plan_path, out):
    """Use only the frozen claims, fresh generator IDs and newly bound reports."""
    frozen = _sealed(plan_path)
    if frozen.get('kind') != 'replication_plan' or frozen.get('status') != 'frozen_before_replication_forecasts':
        raise ValueError('No active frozen replication family')
    expected_ids = set(range(frozen['fresh_seed_base'], frozen['fresh_seed_base'] + frozen['n_bundles']))
    if expected_ids & set(frozen['discovery_generator_ids']):
        raise ValueError('Replication generating IDs overlap discovery')
    rows = []
    wanted = {claim['claim']: claim['expected_direction'] for claim in frozen['claims']}
    if not wanted:
        raise ValueError('Active replication plan has no frozen claims')
    for case in frozen['cases']:
        if producer.file_sha256(case['config_path']) != case['config_sha256']:
            raise ValueError('Frozen replication config changed')
        if not Path(case['prepared']).is_dir():
            source = dict(config=case['config'])
            available = [dict(claim=f"{case['id']}::{key}", case_id=case['id'], endpoint=key,
                mean=None, raw_selected_p=1., selected_direction=0, unavailable='missing prepared replication case') for key in _keys([m['name'] for m in case['config']['models']])]
        else:
            source = _prepared(case['prepared'], case['config'])
            if set(source['generator_ids']) != expected_ids:
                raise ValueError('Replication generator IDs differ from frozen fresh range')
            available = _rows(case, source, frozen['primary_estimand'], wanted)
        rows.extend(row for row in available if row['claim'] in wanted)
    if {row['claim'] for row in rows} != set(wanted):
        raise ValueError('Frozen replication family changed')
    _decide(rows, frozen['minimum_absolute_mean'], frozen['alpha'], wanted)
    return _save(out, 'replication.json', dict(kind='frozen_replication_analysis',
        plan_sha256=frozen['declaration_sha256'], family_size=len(rows), claims=rows,
        primary_estimand=frozen['primary_estimand'],
        question='Selected estimand with a prespecified point-mean size/direction gate; sign probability does not establish expected-MSE superiority',
        all_confirmed=all(row['qualified'] for row in rows),
        caution='Only the prospectively selected family was tested; missing claims remain p=1; no reselection or uniqueness inference'))
