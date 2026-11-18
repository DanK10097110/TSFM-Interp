"""Opt-in controlled feature-interference audit with immutable prepared inputs.

Prepare: --config YAML --prepare-only --out NEWDIR
Infer:   --prepared DIR --out NEWDIR [--device cpu|cuda:0]
Analyze: --config YAML --corpus NPZ --metadata JSON --predictions NAME=NPZ ... --out NEWDIR

All prediction and corpus artifacts require explicit matching row_ids. Cached
models run sequentially through the existing adapter registry; downloads are off.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import yaml
from tsfm_lens.analysis.feature_interference import InterferenceDesign, audit_interference, make_factorial, validate_factorial

def file_sha256(path):
    """Hash exact artifact bytes in bounded chunks for the provenance manifest."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_config(path):
    """Validate explicit registry model configs and freeze all analysis settings."""
    return validate_config(yaml.safe_load(Path(path).read_text(encoding='utf-8')))

def validate_config(cfg):
    """Apply the same strict checks to YAML and hash-consistent prepared JSON."""
    from tsfm_lens.config import ModelConfig
    from tsfm_lens.models import resolve_adapter_class
    if not isinstance(cfg, dict) or set(cfg) - {'design', 'models', 'inference'}:
        raise ValueError('config accepts design,models,inference sections only')
    design = InterferenceDesign(**cfg.get('design', {}))
    models = cfg.get('models', [])
    if not models:
        raise ValueError('name at least one explicit model configuration')
    for model in models:
        if not {'name', 'adapter', 'checkpoint'} <= set(model):
            raise ValueError('models require explicit name,adapter,checkpoint; use empty checkpoint only for mocks')
        parsed = ModelConfig(**model)
        resolve_adapter_class(parsed.adapter)
    names = [m['name'] for m in models]
    if len(set(names)) != len(names) or any((not n or '/' in n or '\\' in n for n in names)):
        raise ValueError('model names must be unique nonempty artifact-safe names')
    inference = dict(n_boot=2000, n_permutations=1999, seed=113, ci=0.95, primary_estimand='mean')
    if set(cfg.get('inference', {})) - set(inference):
        raise ValueError('unknown inference setting')
    inference.update(cfg.get('inference', {}))
    if inference['primary_estimand'] not in ('mean', 'sign_probability'):
        raise ValueError('primary_estimand must be mean or sign_probability')
    if any((isinstance(inference[k], bool) or not isinstance(inference[k], int) or inference[k] < 1 for k in ('n_boot',
                'n_permutations'))) or not 0 < inference['ci'] < 1:
        raise ValueError('need positive integer resampling counts and CI between 0 and 1')
    cfg = dict(design=asdict(design), models=models, inference=inference)
    return (design, cfg)

def check_working_memory(design, cfg, limit_mib):
    """Bound estimated corpus/forecast/bootstrap arrays before materializing them."""
    rows = design.rows_per_bundle * design.n_bundles
    estimate = rows * (16 * design.context + 32 * design.horizon + 8 * len(cfg['models']) * design.horizon + 2048) + 8 * cfg['inference']['n_boot'] * design.n_bundles
    if limit_mib <= 0 or estimate > limit_mib * 1024 ** 2:
        raise ValueError(f'estimated working arrays {estimate / 1024 ** 2:.1f} MiB exceed --memory-limit-mb')
    return estimate / 1024 ** 2


def load_forecast_artifact(path, expected_corpus_sha256):
    """Bind saved forecasts to exact corpus bytes as well as ordered row IDs.

    Factorial IDs recur across contexts and period settings; they cannot identify
    the physical input corpus by themselves. A hash-consistent artifact remains
    a producer declaration, not cryptographic proof that inference was performed.
    """
    with np.load(path, allow_pickle=False) as saved:
        required = {'prediction', 'row_ids', 'corpus_sha256'}
        if not required <= set(saved.files):
            raise ValueError('forecast NPZ requires prediction, row_ids AND corpus_sha256; no source-corpus guessing')
        digest = saved['corpus_sha256']
        if digest.shape != () or digest.dtype.kind != 'U':
            raise ValueError('forecast corpus_sha256 must be one Unicode SHA256 scalar')
        if digest.item() != expected_corpus_sha256:
            raise ValueError('forecast corpus hash mismatch: factorial row IDs alone do not bind physical inputs')
        artifact = {'prediction': saved['prediction'], 'row_ids': saved['row_ids']}
        for key in ('model_name', 'model_config_sha256', 'checkpoint_identifier_sha256'):
            if key in saved.files:
                value = saved[key]
                if value.shape != () or value.dtype.kind != 'U':
                    raise ValueError(f'forecast {key} must be one Unicode scalar')
                artifact[key] = value.item()
        return artifact

def prepare(config_path, out, memory_limit_mib=2048):
    """Write the complete prospective protocol before any forecasting occurs."""
    design, cfg = load_config(config_path)
    estimated = check_working_memory(design, cfg, memory_limit_mib)
    out.mkdir(parents=True, exist_ok=False)
    corpus, metadata = make_factorial(design)
    np.savez_compressed(out / 'corpus.npz', **corpus)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out / 'protocol.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (out / 'config.yaml').write_bytes(Path(config_path).read_bytes())
    manifest = dict(config_sha256=file_sha256(config_path), protocol_sha256=file_sha256(out / 'protocol.json'),
        corpus_sha256=file_sha256(out / 'corpus.npz'), metadata_sha256=file_sha256(out / 'metadata.json'),
        prepared_before_forecasts=True, seed_base=design.seed_base, estimated_working_mib=estimated,
        seed_caution='caller must choose fresh disjoint generating seeds for replication; no private corpus is accessed')
    (out / 'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return cfg

def analyze_files(cfg, corpus_path, metadata_path, prediction_paths, out, *,
                  prepared_source=None, producer_provenance=None):
    """Require artifact-level row IDs and analyze saved forecasts without a GPU."""
    from tsfm_lens.analysis.feature_interference import render_interference_report
    design = InterferenceDesign(**cfg['design'])
    corpus = np.load(corpus_path, allow_pickle=False)
    if not {'context', 'component_future', 'row_ids'} <= set(corpus.files):
        raise ValueError('corpus NPZ requires context,component_future,row_ids')
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    artifacts = {}
    names = [m['name'] for m in cfg['models']]
    if set(prediction_paths) != set(names):
        raise ValueError('predictions must name exactly the frozen config model set')
    corpus_sha256 = file_sha256(corpus_path)
    for name in names:
        artifacts[name] = load_forecast_artifact(prediction_paths[name], corpus_sha256)
    report, contrasts = audit_interference(corpus['context'], corpus['component_future'],
        artifacts, metadata, corpus['row_ids'], design, model_order=names, **cfg['inference'])
    provenance = dict(protocol_sha256=file_sha256(out / 'protocol.json'), analysis_code_sha256=file_sha256(audit_interference.__code__.co_filename),
        runner_code_sha256=file_sha256(__file__), corpus=dict(path=str(corpus_path), sha256=file_sha256(corpus_path)),
        metadata=dict(path=str(metadata_path), sha256=file_sha256(metadata_path)), forecasts={name: dict(path=str(path),
                sha256=file_sha256(path)) for name, path in prediction_paths.items()}, prepared_source=None if prepared_source is None else str(prepared_source),
        row_identity='ordered IDs checked against metadata; EVERY forecast corpus_sha256 also matches exact supplied corpus bytes',
        procedure_timing='prepared input hashes verified before this inference' if prepared_source is not None else 'saved-forecast analysis; preregistration timing not verified by this command')
    attestation = Path(corpus_path).parent / 'legacy_order_attestation.json'
    if attestation.exists():
        provenance['legacy_order_attestation'] = dict(path=str(attestation), sha256=file_sha256(attestation),
            contents=json.loads(attestation.read_text(encoding='utf-8')))
    provenance['forecast_model_declarations'] = {
        name: {key: artifact[key] for key in
               ('model_name', 'model_config_sha256', 'checkpoint_identifier_sha256')
               if key in artifact}
        for name, artifact in artifacts.items()
    }
    if producer_provenance is not None:
        producer_path = Path(producer_provenance)
        producer = json.loads(producer_path.read_text(encoding='utf-8'))
        if producer['corpus']['sha256'] != corpus_sha256:
            raise ValueError('producer provenance does not bind the supplied corpus')
        if producer['metadata']['sha256'] != file_sha256(metadata_path):
            raise ValueError('producer provenance does not bind the supplied metadata')
        if set(producer['forecasts']) != set(prediction_paths):
            raise ValueError('producer provenance model set differs from analysis inputs')
        for name, path in prediction_paths.items():
            if producer['forecasts'][name]['sha256'] != file_sha256(path):
                raise ValueError(f'{name}: producer forecast hash mismatch')
        producer_protocol_path = producer_path.parent / 'protocol.json'
        if file_sha256(producer_protocol_path) != producer['protocol_sha256']:
            raise ValueError('producer protocol bytes do not match producer provenance')
        producer_protocol = json.loads(producer_protocol_path.read_text(encoding='utf-8'))
        _, normalized_producer = validate_config(producer_protocol)
        same_plan = normalized_producer == cfg
        provenance['producer_inference'] = {
            'provenance_path': str(producer_path),
            'provenance_sha256': file_sha256(producer_path),
            'original_protocol_sha256': producer['protocol_sha256'],
            'original_primary_estimand': normalized_producer['inference']['primary_estimand'],
            'analysis_primary_estimand': cfg['inference']['primary_estimand'],
            'registered_plan_preserved': same_plan,
            'contents': producer,
            'interpretation': (
                'Original registered plan retained; additional endpoint comparisons are secondary.'
                if same_plan else
                'Alternate-plan reanalysis; not prospective primary confirmation of the new endpoint.'
            ),
        }
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'seed_contrasts.npz', **contrasts)
    render_interference_report(report, out / 'report.html', cfg, provenance, corpus['context'],
        corpus['component_future'], artifacts, metadata)
    return report

def infer_prepared(prepared, out, device, batch_size, memory_limit_mib=2048):
    """Run one cached adapter at a time; reject nonrepeatable native predictions."""
    import torch
    from tsfm_lens.config import ModelConfig, DataConfig
    from tsfm_lens.models import build_adapter
    design, cfg = validate_config(json.loads((prepared / 'protocol.json').read_text(encoding='utf-8')))
    manifest = json.loads((prepared / 'provenance.json').read_text(encoding='utf-8'))
    check_working_memory(design, cfg, memory_limit_mib)
    for name in ('corpus', 'metadata', 'protocol'):
        path = prepared / (name + ('.npz' if name == 'corpus' else '.json'))
        if file_sha256(path) != manifest[name + '_sha256']:
            raise ValueError('prepared artifact changed after protocol freeze: ' + name)
    if batch_size < 1 or batch_size > 8:
        raise ValueError('batch size must be between 1 and 8')
    corpus = np.load(prepared / 'corpus.npz', allow_pickle=False)
    corpus_sha256 = manifest['corpus_sha256']
    metadata = json.loads((prepared / 'metadata.json').read_text(encoding='utf-8'))
    validate_factorial(corpus['context'], corpus['component_future'], metadata, corpus['row_ids'], design)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    paths = {}
    model_provenance = {}
    target_device = torch.device(device)
    if target_device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('explicit CUDA device unavailable; choose CPU or an assigned visible GPU')
        torch.cuda.set_device(target_device)
        torch.cuda.set_per_process_memory_fraction(min(1.0, 6 * 1024 ** 3 / torch.cuda.get_device_properties(target_device).total_memory),
            target_device)
    for model in cfg['models']:
        mcfg = ModelConfig(**model)
        adapter = build_adapter(mcfg, DataConfig(context_len=design.context, horizon=design.horizon),
            target_device, torch.float32)
        adapter.ensure_loaded()
        prediction = []
        for start in range(0, len(corpus['context']), batch_size):
            x = corpus['context'][start:start + batch_size]
            prediction.append(adapter.predict(x, design.horizon, [0.5])['point'])
        prediction = np.concatenate(prediction)
        repeat = adapter.predict(corpus['context'][:batch_size], design.horizon, [0.5])['point']
        if not np.array_equal(prediction[:batch_size], repeat):
            raise RuntimeError(f'{mcfg.name}: nonrepeatable native point forecast; sampled forecasts require a separate draw-matched protocol')
        path = out / (mcfg.name + '_predictions.npz')
        effective_model_config = asdict(mcfg)
        model_config_sha256 = hashlib.sha256(
            json.dumps(effective_model_config, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        checkpoint_identifier_sha256 = hashlib.sha256(mcfg.checkpoint.encode()).hexdigest()
        np.savez_compressed(
            path, prediction=prediction, row_ids=corpus['row_ids'],
            corpus_sha256=np.asarray(corpus_sha256), model_name=np.asarray(mcfg.name),
            model_config_sha256=np.asarray(model_config_sha256),
            checkpoint_identifier_sha256=np.asarray(checkpoint_identifier_sha256),
        )
        paths[mcfg.name] = path
        model_provenance[mcfg.name] = dict(config=model, device=str(target_device),
            dtype='float32', batch_size=batch_size, repeat_max_abs=0.0, sha256=file_sha256(path))
        model_provenance[mcfg.name].update(
            effective_model_config=effective_model_config,
            model_config_sha256=model_config_sha256,
            checkpoint_identifier_sha256=checkpoint_identifier_sha256,
            checkpoint_identifier_hash_scope='identifier string only, not checkpoint weight bytes',
            corpus_sha256=corpus_sha256,
        )
        (out / 'model_provenance.json').write_text(json.dumps(model_provenance, indent=2), encoding='utf-8')
        print('forecast complete: ' + mcfg.name, flush=True)
        del adapter
        gc.collect()
        if target_device.type == 'cuda':
            torch.cuda.empty_cache()
    return analyze_files(cfg, prepared / 'corpus.npz', prepared / 'metadata.json', paths,
        out, prepared_source=prepared)

def main(argv=None):
    """Dispatch immutable prepare/infer/analyze modes with capped CPU resources."""
    import torch
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--predictions', action='append', help='NAME=prediction.npz with explicit row_ids')
    parser.add_argument('--producer-provenance', type=Path,
                        help='original producer analysis provenance; verifies its neighboring protocol and input hashes')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--memory-limit-mb', type=float, default=2048)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('--out must be a NEW directory')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.prepared:
            if args.config or args.prepare_only or args.predictions or args.corpus or args.metadata or args.producer_provenance:
                parser.error('--prepared cannot be combined with prepare or analysis inputs')
            return infer_prepared(args.prepared, args.out, args.device, args.batch_size, args.memory_limit_mb)
        if not args.config:
            parser.error('--config is required for prepare or analysis')
        if args.predictions:
            if args.prepare_only or not args.corpus or (not args.metadata):
                parser.error('analysis requires --corpus and --metadata')
            design, cfg = load_config(args.config)
            check_working_memory(design, cfg, args.memory_limit_mb)
            paths = {}
            for value in args.predictions:
                name, path = value.split('=', 1)
                if name in paths:
                    parser.error('duplicate prediction name')
                paths[name] = Path(path)
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / 'protocol.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            return analyze_files(cfg, args.corpus, args.metadata, paths, args.out,
                                 producer_provenance=args.producer_provenance)
        if not args.prepare_only:
            parser.error('choose --prepare-only, --prepared, or complete saved-prediction analysis inputs')
        if args.corpus or args.metadata or args.producer_provenance:
            parser.error('prepare mode does not accept corpus/metadata inputs')
        return prepare(args.config, args.out, args.memory_limit_mb)
if __name__ == '__main__':
    main()
