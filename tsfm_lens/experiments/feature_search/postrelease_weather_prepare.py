"""Bounded public provenance/download preparation; never loads a native model.

The eight airport identities and October origins are fixed before observations.
Public HTTPS only; no auth, no weights download, and a40,000,000-byte total body
budget including metadata. Missing data or chronology blocks the claimed cohort.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import io
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from experiments.feature_search.persistence_protocol import read, sha, write

AIRPORTS = [
    ('KSEA', 'Seattle Tacoma', '72793024233'),
    ('KPHX', 'Phoenix Sky Harbor', '72278023183'),
    ('KDEN', 'Denver International', '72565003017'),
    ('KMSP', 'Minneapolis Saint Paul', '72658014922'),
    ('KDFW', 'Dallas Fort Worth', '72259003927'),
    ('KJFK', 'New York JFK', '74486094789'),
    ('KATL', 'Atlanta Hartsfield Jackson', '72219013874'),
    ('KMIA', 'Miami International', '72202012839'),
]
CAP = 40_000_000
HOSTS = {'huggingface.co', 'www.ncei.noaa.gov', 'noaa-global-hourly-pds.s3.amazonaws.com'}


def utcnow():
    """Record real retrieval time with timezone, independently of data timestamps."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    """Keep redirects on explicit public data hosts; never follow arbitrary locations."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_url(url):
    """Reject non-HTTPS, credentials and endpoints outside the public allowlist."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.username or parsed.password:
        raise ValueError('Not an allowed public HTTPS URL')


def fetch(out, url, name, limit):
    """Download within per-file/aggregate bounds and journal every successful/failing read."""
    check_url(url)
    manifest = read(out/'retrievals.json')
    remaining = CAP-sum(r.get('bytes_received', 0) for r in manifest)
    allowed = min(limit, remaining)
    if allowed <= 0:
        raise ValueError('Aggregate40MB budget exhausted')
    target = out/'raw'/name
    if target.exists():
        raise ValueError('Refuse raw download overwrite')
    record = dict(url=url, name=name, started_utc=utcnow(), bytes_received=0, limit_bytes=allowed)
    request = urllib.request.Request(url, headers={'User-Agent': 'TSFM-postrelease-weather-research/1.0', 'Accept-Encoding': 'identity'})
    opener = urllib.request.build_opener(PublicRedirect())
    try:
        with opener.open(request, timeout=45) as response:
            record.update(status=response.status, final_url=response.url,
                          headers={k: response.headers.get(k) for k in ['Content-Type', 'Content-Length', 'ETag', 'Last-Modified', 'Date']})
            size = response.headers.get('Content-Length')
            if size and int(size) > allowed:
                raise ValueError('Advertised body exceeds bounded download limit')
            with target.open('xb') as handle:
                while record['bytes_received'] < allowed:
                    part = response.read(min(65536, allowed-record['bytes_received']))
                    if not part:
                        break
                    handle.write(part); record['bytes_received'] += len(part)
                else:
                    raise ValueError('Body reached limit; conservatively reject possible truncation')
            record.update(valid=True, sha256=sha(target))
    except Exception as error:
        record.update(valid=False, error_type=type(error).__name__, error=str(error))
        if isinstance(error, urllib.error.HTTPError):
            record['status'] = error.code
        if target.exists():
            record['partial_sha256'] = sha(target)
    record['finished_utc'] = utcnow()
    manifest.append(record); write(out/'retrievals.json', manifest)
    return record


def initialize(out, checkpoint_protocol):
    """Freeze question, airport choices, preprocessing and forecast recipes before reads."""
    out.mkdir(parents=True, exist_ok=False); (out/'raw').mkdir()
    checkpoints = read(checkpoint_protocol)['checkpoints']
    protocol = dict(question='Are reproducible October2026 airport temperature observations accessible and chronologically after the exact cached weight revisions?',
                    stage='PREPARATION ONLY; native HOLD', created_utc=utcnow(), airports=[dict(icao=a, name=b, station_id=c) for a,b,c in AIRPORTS],
                    checkpoints=checkpoints, max_download_bytes=CAP, credentials='none; public HTTPS GET only; no model weight downloads',
                    month='2026-10', origins_utc=[f'2026-10-{d:02d}T00:00:00Z' for d in [8,15,22,29]],
                    context_hours=128, horizon_hours=32,
                    cohort='8 predetermined airports x4 predetermined origins; no replacements based on missingness or outcomes',
                    sampling_unit='station, n<=8; four origins remain dependent, not32 independent samples',
                    chronology='Exact HF revision commit timestamp before2026-10-01 AND model.safetensors LFS SHA256 at that revision equals cached bytes. No cached mtime inference.',
                    scope='Temporal observation holdout only; no new-location or unseen-temperature-distribution claim. Cached Monash pretraining overlap remains unknown.',
                    access='NOAA global-hourly month subset API with TMP attributes; if unavailable, same-station official2026 annual CSV within remaining budget. No alternative station/month substitution.',
                    quality='Verify official ISD TMP tenths-Celsius and quality definitions; accept codes0,1,4,5 only; reject+9999 and malformed values.',
                    aggregation='UTC left-closed hour bins[h,h+1); average distinct accepted temperatures at each exact timestamp, then average timestamps within hour; all history timestamps strictly before origin.',
                    history_missing='At most2 missing hourly bins total per128h history; forward-fill only from preceding observed bins within context. Leading missing or>2 missing makes history ineligible. No backward/interpolated/future fill.',
                    future_missing='No future imputation. Preserve every scheduled origin and missing mask; score only complete32h futures, report excluded origins/stations. Do not replace them. History eligibility is independent of future completeness.',
                    methods=['native median', 'fixed depth1 selective tones + native residual median', 'seasonal24 repeat last24h'],
                    hybrid='Reuse frozen selective_tree.decompose(depth=1), stability>=.65,energy>=.15,prominence>=8,continuous frequencies2..48/128; continuation on horizon32; abstention exactly native; no veto/refit/new tuning or forecast selection.',
                    metrics='Future squared error in Celsius^2 and divided by history variance; station averages across complete fixed origins, then equally weight stations. Missingness counts and actual station n always reported.',
                    caveats=['Archived quality control may be retrospective; this is not a real-time data-vintage test.', 'Post-checkpoint dates do not imply unfamiliar geography, climate, seasonality, or distribution.'],
                    next_decision='First verify public accessibility, station IDs, exact checkpoint chronology and eligible-window counts; then wait for native slot grant.')
    write(out/'protocol.json', protocol); write(out/'protocol_seal.json', dict(sha256=sha(out/'protocol.json')))
    write(out/'retrievals.json', [])
    write(out/'knowledge.json', dict(question=protocol['question'],design=protocol,exact_findings=dict(status='prepared; metadata/data availability unverified',native_launched=False),controls='Fixed cohort and recipes; public download cap; no credentials',limitations=protocol['caveats'],evidence_class='prospective feasibility/provenance investigation',next_decision=protocol['next_decision']))


def metadata(out):
    """Download only small official model/station/format provenance resources."""
    p = read(out/'protocol.json')
    for model, item in p['checkpoints'].items():
        repo, revision = item['repo'], item['revision']
        fetch(out, f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', f'{model}_revision.json', 1_000_000)
        fetch(out, f'https://huggingface.co/api/models/{repo}/commits/{revision}', f'{model}_commits.json', 1_000_000)
    fetch(out, 'https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv', 'isd-history.csv', 4_000_000)
    fetch(out, 'https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf', 'isd-format-document.pdf', 2_000_000)
    fetch(out, 'https://www.ncei.noaa.gov/data/global-hourly/access/2026/', 'noaa_2026_index.html', 1_000_000)
    k = read(out/'knowledge.json'); k['metadata_retrievals'] = read(out/'retrievals.json'); k['next_decision'] = 'Verify retrieved official station IDs, checkpoint dates/hash binding and quality documentation before weather downloads'; write(out/'knowledge.json', k)


def verify_metadata(out):
    """Verify exact revision dates/weight bytes and fixed ICAO+station identities."""
    p = read(out/'protocol.json'); models = {}
    for model, item in p['checkpoints'].items():
        info = read(out/'raw'/f'{model}_revision.json'); commits = read(out/'raw'/f'{model}_commits.json')
        if info['sha'] != item['revision']:
            raise ValueError('Exact revision mismatch')
        commit = next(c for c in commits if c['id'] == item['revision'])
        stamp = dt.datetime.fromisoformat(commit['date'].replace('Z', '+00:00'))
        if stamp >= dt.datetime(2026,10,1,tzinfo=dt.timezone.utc):
            raise ValueError('Checkpoint does not predate October2026')
        weight = next(s for s in info['siblings'] if s['rfilename'] == 'model.safetensors')
        public = weight['lfs']['sha256']; local = sha(Path(item['path'])/'model.safetensors')
        if public != local or local != item['weights']['model.safetensors']:
            raise ValueError('Public LFS / cached weight SHA256 mismatch')
        models[model] = dict(revision=item['revision'],commit_utc=commit['date'],weight_sha256=local,
                             weights_match=True,predates_observations=True,
                             commit_url=f"https://huggingface.co/{item['repo']}/commit/{item['revision']}")
    registry = list(csv.DictReader((out/'raw/isd-history.csv').open(encoding='utf-8')))
    stations = []
    for item in p['airports']:
        matches = [r for r in registry if r['USAF']+r['WBAN'] == item['station_id'] and r['ICAO'] == item['icao']]
        if len(matches) != 1:
            raise ValueError('Official station identity mismatch: '+str(item))
        row = matches[0]
        if row['CTRY'] != 'US':
            raise ValueError('Not a US station')
        stations.append(dict(item,official_record=row))
    result = dict(models=models,stations=stations,verified=True,verified_utc=utcnow())
    write(out/'metadata_verification.json', result)
    k = read(out/'knowledge.json'); k['exact_findings'] = dict(status='Checkpoint chronology/weight hashes and8 official station IDs verified; observation availability pending',metadata=result,native_launched=False);write(out/'knowledge.json',k)
    print(json.dumps(result, indent=2))


def observations(out):
    """Fetch exactly October observations for fixed stations, with bounded same-source fallback."""
    verified = read(out/'metadata_verification.json')
    if not verified['verified']:
        raise ValueError('Provenance gates not passed')
    records = []
    for item in verified['stations']:
        station = item['station_id']
        params = dict(dataset='global-hourly',stations=station,startDate='2026-10-01T00:00:00Z',endDate='2026-10-31T23:59:59Z',
                      dataTypes='TMP',format='csv',includeAttributes='true',includeStationName='true',includeStationLocation='true')
        url = 'https://www.ncei.noaa.gov/access/services/data/v1?'+urllib.parse.urlencode(params)
        record = fetch(out, url, station+'_202610_api.csv', 3_000_000)
        schema_ok = False
        if record['valid']:
            with (out/'raw'/record['name']).open(encoding='utf-8-sig') as handle:
                reader = csv.DictReader(handle)
                schema_ok = {'STATION','DATE','TMP'}.issubset(reader.fieldnames or []) and next(reader, None) is not None
        if not schema_ok:
            record = fetch(out, f'https://noaa-global-hourly-pds.s3.amazonaws.com/2026/{station}.csv',station+'_2026_archive.csv',5_000_000)
        records.append(dict(icao=item['icao'],station_id=station,retrieval=record))
        write(out/'weather_retrievals.json',records)
        print(item['icao'],record.get('status'),record.get('bytes_received'),record.get('valid'),flush=True)
    k = read(out/'knowledge.json'); k['weather_retrievals'] = records;k['next_decision']='Validate dates/quality schema and fixed windows; no native launch';write(out/'knowledge.json',k)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['initialize','metadata','verify','observations'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint-protocol',type=Path,default=Path('tsfm_lens/runs/persistence_replication4953000/inference/protocol.json'))
    args=parser.parse_args()
    if args.command=='initialize': initialize(args.out,args.checkpoint_protocol)
    elif args.command=='metadata': metadata(args.out)
    elif args.command=='verify': verify_metadata(args.out)
    else: observations(args.out)
