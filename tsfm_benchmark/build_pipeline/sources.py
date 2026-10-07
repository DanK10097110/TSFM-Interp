"""Adapters that load real corpora into the canonical form.

Real data is used here only as (1) calibration targets for the realism report
and (2) the reference corpus the leakage auditor keeps the benchmark away from.
It is never copied into a benchmark sample directly. Each loaded item is hashed
and tagged with its license so derived outputs remain auditable and the source
license can be honoured when the benchmark is published.

``load_sources`` works against any Hugging Face dataset repo id, not just
Monash: give it a ``dataset`` name and, if you also pin a ``config``/``subset``,
it loads exactly that one; if you don't, it calls ``bootstrap_catalog`` to
discover every config the dataset exposes and pool a random sample of them.
``kind: monash`` is a convenience alias that just presets
``dataset: Monash-University/monash_tsf``; the actual loading logic is the
same generic path either way. TIME, BOOM, and ARFBench are left as
registration points for adapters that need a different yield contract than
row-per-series HF datasets provide.

``kind: chronos_datasets`` is a second real source, added for ROADMAP D1: a
flatness investigation found the real-derived tier was ~42% of
``benchmark_large``'s dev split and almost entirely Monash ``weather`` (61%
zeros for `block_bootstrap`, 36% for `sequential_par`), because `weather` is
the only Monash-via-``monash_tsf`` domain whose series are both long enough
(``min_length``) and loadable at all -- the long *hourly* Monash domains fail
to load through that dataset's own loading script (a frequency-alias bug
parsing pandas offset strings like ``'h'``, see ``configs/large_run.yaml``).
``autogluon/chronos_datasets`` mirrors several of those same Monash domains
(plus others) as plain parquet, so it sidesteps that bug entirely and is not
script-backed, which also matters for the ``datasets<3`` pin (`datasets>=3`
dropped script-based loading outright). Measured here (see
``configs/large_run_v2.yaml``'s docstring for the numbers): its
``monash_electricity_hourly`` and ``monash_traffic`` configs are strongly
periodic and almost never zero or flat, unlike `weather`. Like every other
real-derived source, these datasets may overlap TSFM pretraining corpora --
that exposure cannot be verified (CLAUDE.md sec 10) and is a known limitation,
not something this loader can detect or control.

Unlike the streaming, take-the-first-``limit``-rows single-config path,
``chronos_datasets`` subsets have few enough rows (tens to low thousands) to
load non-streamed and index directly, so item *selection* can be a real
seeded random sample of the pool rather than always the same leading slice --
mixed via sha256 (``_mix_seed``), never Python's salted, per-process
``hash()``. The same mixing deterministically windows any series longer than
``max_length`` (several of these domains are single, multi-year hourly
recordings 15,000-100,000+ points long) so one real-derived task's build cost
and stored series length stay bounded without truncating every series to the
same fixed leading window, which would only ever sample the same calendar
season across the whole pool.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
from typing import Any, Iterator

import numpy as np

from .registry import SOURCES
from .schema import SourceRef


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.round(values, 6).tobytes()).hexdigest()


def _mix_seed(seed: int, *parts: str) -> int:
    """Derive a reproducible child seed from an integer seed plus string tags.

    Used wherever a source needs more than one independent random draw
    (which rows to keep, which window to cut a long series to) from a single
    config-level ``seed``: mixing in a tag keeps those draws independent of
    each other without needing a second config field per draw. sha256-based,
    never Python's ``hash()``, which is randomly salted per process and would
    make "the same seed" not actually reproduce anything.
    """
    digest = hashlib.sha256(f"{seed}:{':'.join(parts)}".encode()).hexdigest()
    return int(digest, 16) % (2**32 - 1)


def _to_series(values: Any) -> np.ndarray | None:
    """Coerce a candidate field to a clean 1D numeric array, or reject it.

    Real-world series commonly record missing observations as NaN (Monash
    domains like tourism/traffic do this) -- Python's ``float`` accepts NaN,
    so a naive numeric check alone would let it through. Every downstream
    consumer (mixture blending, the leakage audit's KD-tree, catch22
    features) assumes finite values, so a series with any non-finite entry
    is rejected here rather than silently poisoning everything built from it.
    """
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)) and values and all(isinstance(v, (int, float, np.number, bool)) for v in values):
        arr = np.asarray(values, dtype=float)
        if not np.isfinite(arr).all():
            return None
        return arr
    return None


def _iter_series_candidates(obj: Any) -> list[np.ndarray]:
    if isinstance(obj, dict):
        candidates: list[np.ndarray] = []
        for value in obj.values():
            candidates.extend(_iter_series_candidates(value))
        return candidates

    if isinstance(obj, (list, tuple)):
        series = _to_series(obj)
        if series is not None:
            return [series]
        candidates = []
        for item in obj:
            candidates.extend(_iter_series_candidates(item))
        return candidates

    return []


def _filter_min_length(sources: list, source_config: dict) -> list:
    """Drop source series shorter than ``min_length``, loudly.

    Several real-derived generators inherit their output length from the real
    series they consume -- ``block_bootstrap`` reorders one source in place and
    has no length parameter at all. A pooled Monash catalog spans yearly series
    of ~14 points alongside multi-thousand-point hourly ones, so without this
    filter a task can emit an entire family whose every member is shorter than
    the analysis needs. That failure is silent at both ends: the builder admits
    the samples happily, and ``tsfm_lens``'s loader drops every one of them as
    "shorter than context+horizon", leaving a family the config promises and no
    analysis ever sees.

    Off unless asked for (no ``min_length`` key leaves every pool byte-identical,
    so no existing config's corpus changes). Raises rather than returning an
    empty pool: a task configured against a source pool that cannot satisfy it
    is a config error, and returning nothing would surface much later as an
    unexplained empty family.
    """
    min_length = source_config.get("min_length")
    if not min_length:
        return sources
    min_length = int(min_length)
    kept = [(ref, arr) for ref, arr in sources if len(np.asarray(arr)) >= min_length]
    dropped = len(sources) - len(kept)
    if dropped:
        print(f"load_sources: dropped {dropped} of {len(sources)} source series "
              f"shorter than min_length={min_length}")
    if not kept:
        raise ValueError(
            f"no source series survived min_length={min_length} (of {len(sources)} "
            f"loaded). Lower min_length, raise the source limit, or pin a `subset` "
            f"whose series are long enough."
        )
    return kept


def load_sources(source_config: dict[str, Any], limit: int | None = None, license: str = "unknown") -> list[tuple[SourceRef, np.ndarray]]:
    """Load time-series sources from a declarative config.

    See ``configs/example.yaml`` for a fully annotated reference of every
    ``source_config`` key and what values each one accepts. In short: name a
    ``dataset`` (any Hugging Face repo id); if you also pin ``config``/
    ``subset`` to one of that dataset's configs, exactly that one is loaded;
    if you don't, every config the dataset exposes is discovered and a random
    sample is pooled instead (``bootstrap_catalog``). The single-config path
    walks each record and extracts the named field (or, absent a field name,
    prefers the ``target`` convention used by nearly every Hugging Face
    forecasting dataset, falling back to scanning the whole record for
    numeric series if that key isn't present either) so it works across
    differently-shaped datasets without per-dataset code.
    """
    if not isinstance(source_config, dict) or not source_config:
        raise ValueError("source_config must be a non-empty mapping")

    kind = str(source_config.get("kind", source_config.get("source_kind", "huggingface"))).lower()
    if kind not in {"huggingface", "hf", "monash", "chronos_datasets"}:
        raise ValueError(f"unsupported source kind '{kind}'")

    if kind == "chronos_datasets":
        subset = source_config.get("subset") or source_config.get("config") or source_config.get("config_name")
        if not subset:
            raise ValueError(
                "source_config with kind 'chronos_datasets' must include 'subset' "
                "(one of autogluon/chronos_datasets' config names, e.g. "
                "'monash_electricity_hourly' or 'monash_traffic')"
            )
        dataset_name = source_config.get("dataset") or source_config.get("dataset_name") or "autogluon/chronos_datasets"
        field_name = source_config.get("field_name") or source_config.get("series_field") or "target"
        limit = limit if limit is not None else int(source_config.get("limit", 100))
        return _filter_min_length(
            _load_chronos_datasets_config(
                dataset_name=dataset_name,
                subset=subset,
                field_name=field_name,
                split=source_config.get("split", "train"),
                limit=limit,
                seed=int(source_config.get("seed", 0)),
                max_length=source_config.get("max_length"),
                license=license,
                windows_per_row=int(source_config.get("windows_per_row", 1)),
                finite_windows=bool(source_config.get("finite_windows", False)),
            ),
            source_config,
        )

    if kind == "monash":
        dataset_name = source_config.get("dataset") or source_config.get("dataset_name") or source_config.get("name") or "Monash-University/monash_tsf"
    else:
        dataset_name = source_config.get("dataset") or source_config.get("dataset_name") or source_config.get("name")
        if not dataset_name:
            raise ValueError(
                "source_config must include 'dataset' (a Hugging Face repo id, e.g. "
                "'Monash-University/monash_tsf' or 'ETDataset/ett' -- any dataset works)"
            )

    subset = source_config.get("subset") or source_config.get("config") or source_config.get("config_name")
    split = source_config.get("split", "train")
    field_name = source_config.get("field_name") or source_config.get("series_field")
    streaming = bool(source_config.get("streaming", True))
    trust_remote_code = bool(source_config.get("trust_remote_code", True))

    if not subset:
        # No single config named: rather than guessing one (which is exactly
        # the "benchmark only reflects the domain I happened to pick"
        # problem), discover every config this dataset exposes and bootstrap
        # a random sample of them instead.
        total_limit = limit if limit is not None else int(source_config.get("limit", 500))
        n_domains = source_config.get("n_domains")
        if isinstance(n_domains, str) and n_domains.lower() == "all":
            n_domains = None
        return _filter_min_length(bootstrap_catalog(
            dataset_name=dataset_name,
            n_domains=n_domains,
            total_limit=total_limit,
            seed=int(source_config.get("seed", 0)),
            field_name=field_name or "target",
            split=split,
            license=license,
            domain_timeout_s=float(source_config.get("domain_timeout_s", 30.0)),
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        ), source_config)

    limit = limit if limit is not None else int(source_config.get("limit", 100))
    return _filter_min_length(
        _load_single_config(dataset_name, subset, split, field_name, streaming,
                            trust_remote_code, limit, license),
        source_config)


def _load_single_config(
    dataset_name: str,
    config_name: str | None,
    split: str,
    field_name: str | None,
    streaming: bool,
    trust_remote_code: bool,
    limit: int | None,
    license: str,
) -> list[tuple[SourceRef, np.ndarray]]:
    """Load up to ``limit`` series from one dataset config."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install 'datasets' to load Hugging Face data") from exc

    try:
        if config_name is None:
            ds = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
        else:
            ds = load_dataset(dataset_name, config_name, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
    except TypeError:
        if config_name is None:
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
        else:
            ds = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    except Exception as exc:
        raise RuntimeError(f"could not load Hugging Face dataset '{dataset_name}'") from exc

    # Config name is folded into the corpus tag so a source loaded as part of
    # a multi-domain pool (e.g. bootstrap_catalog) still records which domain
    # it actually came from, not just the parent dataset.
    corpus = f"hf/{dataset_name}/{config_name}/{split}" if config_name else f"hf/{dataset_name}/{split}"

    out: list[tuple[SourceRef, np.ndarray]] = []
    for idx, row in enumerate(ds):
        if limit is not None and len(out) >= limit:
            break

        if field_name is not None:
            if isinstance(row, dict) and field_name in row:
                values = _to_series(row[field_name])
                if values is not None:
                    ref = SourceRef(corpus=corpus, item_id=f"{idx}:{field_name}", sha256=_hash(values), license=license)
                    out.append((ref, values))
            continue

        # No field named: "target" is the field almost every Hugging Face
        # forecasting dataset uses (Monash TSF, ETT, and most GluonTS-style
        # repos), so prefer it over blindly walking the whole record, which
        # would also sweep up dynamic covariates as if they were independent
        # series. Only fall back to the generic walk if "target" isn't there.
        if isinstance(row, dict) and "target" in row:
            values = _to_series(row["target"])
            if values is not None:
                ref = SourceRef(corpus=corpus, item_id=f"{idx}:target", sha256=_hash(values), license=license)
                out.append((ref, values))
            continue

        for values in _iter_series_candidates(row):
            if len(out) >= (limit or 10**9):
                break
            ref = SourceRef(corpus=corpus, item_id=f"{idx}:{len(out)}", sha256=_hash(values), license=license)
            out.append((ref, values))

    return out


def _load_chronos_datasets_config(
    dataset_name: str,
    subset: str,
    field_name: str,
    split: str,
    limit: int,
    seed: int,
    max_length: int | None,
    license: str,
    windows_per_row: int = 1,
    finite_windows: bool = False,
) -> list[tuple[SourceRef, np.ndarray]]:
    """Load a deterministic, seeded sample from one ``chronos_datasets`` config.

    Every config this dataset exposes is plain parquet with few enough rows
    (tens to a few thousand, one full series per row) to index non-streamed,
    which is what makes a real seeded sample of the pool possible: the
    streaming, take-the-first-``limit``-rows path every other kind here uses
    would always draw the same leading slice in file order, seed or no seed.
    Row selection is mixed from ``seed`` via ``_mix_seed`` (sha256, not
    ``hash()``) so the same config reproduces the same pool; a series longer
    than ``max_length`` is windowed to exactly that length at an
    independently seeded per-row offset, so a task built from several rows
    doesn't sample the same calendar slice out of every one of them.

    Two opt-in options (ROADMAP sec 41, V3-A; both off by default, which leaves
    the pool byte-identical to every earlier config) serve subsets that have
    few, very long rows with scattered missing values (``ercot``,
    ``monash_kdd_cup_2018``): ``windows_per_row`` cuts up to that many
    *non-overlapping* ``max_length`` windows from each row (one per equal
    segment of the row, each at a seeded offset inside its segment), and
    ``finite_windows`` accepts a row containing NaN as long as a window free of
    them can be found (up to 20 seeded draws per segment) instead of rejecting
    the whole row. A segment with no finite window is skipped and the skip is
    visible as fewer pool items, never filled in. Both require ``max_length``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install 'datasets' to load Hugging Face data") from exc

    try:
        ds = load_dataset(dataset_name, subset, split=split, streaming=False, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(f"could not load '{dataset_name}' config '{subset}'") from exc

    n_total = len(ds)
    if n_total == 0:
        return []

    corpus = f"hf/{dataset_name}/{subset}/{split}"
    n_keep = min(limit, n_total) if limit is not None else n_total
    if n_keep < n_total:
        rng = np.random.default_rng(_mix_seed(seed, dataset_name, subset, "rows"))
        row_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
    else:
        row_idx = np.arange(n_total)

    max_length = int(max_length) if max_length else None
    multi_window = windows_per_row > 1 or finite_windows
    if multi_window and not max_length:
        raise ValueError("windows_per_row / finite_windows require max_length")

    out: list[tuple[SourceRef, np.ndarray]] = []
    for i in row_idx:
        row = ds[int(i)]
        if not isinstance(row, dict) or field_name not in row:
            continue
        if multi_window:
            out.extend(_row_windows(row[field_name], int(i), field_name, corpus, dataset_name, subset, seed,
                                    max_length, windows_per_row, finite_windows, license))
            continue
        values = _to_series(row[field_name])
        if values is None:
            continue
        item_id = f"{int(i)}:{field_name}"
        if max_length and len(values) > max_length:
            span = len(values) - max_length
            offset_rng = np.random.default_rng(_mix_seed(seed, dataset_name, subset, "window", str(int(i))))
            offset = int(offset_rng.integers(0, span + 1))
            values = values[offset:offset + max_length]
            item_id = f"{item_id}:w{offset}"
        ref = SourceRef(corpus=corpus, item_id=item_id, sha256=_hash(values), license=license)
        out.append((ref, values))

    return out


def _row_windows(raw: Any, row_index: int, field_name: str, corpus: str, dataset_name: str, subset: str,
                 seed: int, max_length: int, windows_per_row: int, finite_windows: bool,
                 license: str) -> list[tuple[SourceRef, np.ndarray]]:
    """Cut up to ``windows_per_row`` non-overlapping ``max_length`` windows from one row.

    The row is split into ``n = min(windows_per_row, len // max_length)`` equal
    segments; window ``w`` starts at a sha256-seeded offset inside segment ``w``
    so windows never overlap. A row shorter than ``max_length`` yields nothing.
    With ``finite_windows`` a window containing NaN/inf is redrawn (20 tries)
    and then skipped; without it a row with any non-finite value is rejected,
    as in the single-window path.
    """
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) == 0:
        return []
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 1 or len(arr) < max_length:
        return []
    if not finite_windows and not np.isfinite(arr).all():
        return []
    n = max(1, min(windows_per_row, len(arr) // max_length))
    seg = len(arr) // n
    out: list[tuple[SourceRef, np.ndarray]] = []
    for w in range(n):
        lo, hi = w * seg, w * seg + seg - max_length
        rng = np.random.default_rng(_mix_seed(seed, dataset_name, subset, "window", str(row_index), str(w)))
        for _ in range(20 if finite_windows else 1):
            offset = lo + int(rng.integers(0, hi - lo + 1))
            values = arr[offset:offset + max_length]
            if np.isfinite(values).all():
                ref = SourceRef(corpus=corpus, item_id=f"{row_index}:{field_name}:w{offset}",
                                sha256=_hash(values), license=license)
                out.append((ref, values))
                break
    return out


def _load_with_timeout(
    dataset_name: str,
    config_name: str,
    split: str,
    field_name: str,
    streaming: bool,
    trust_remote_code: bool,
    limit: int,
    license: str,
    timeout_s: float,
) -> list[tuple[SourceRef, np.ndarray]]:
    """Run ``_load_single_config`` off-thread and give up after ``timeout_s``.

    Some Hub configs are script-backed datasets that aren't actually
    streamable and instead download a large archive up front; a network hiccup
    there triggers huggingface_hub's own multi-attempt retry with backoff,
    which can stall for minutes on one domain. There is no clean way to cancel
    a blocked network call, so the worker thread is abandoned on timeout
    (``shutdown(wait=False)``) rather than waited on -- acceptable for a
    short-lived build script, and far better than hanging the whole build on
    one unlucky domain draw.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_load_single_config, dataset_name, config_name, split, field_name, streaming, trust_remote_code, limit, license)
    try:
        return future.result(timeout=timeout_s)
    finally:
        executor.shutdown(wait=False)


def bootstrap_catalog(
    dataset_name: str = "Monash-University/monash_tsf",
    n_domains: int | None = None,
    total_limit: int = 500,
    seed: int = 0,
    field_name: str = "target",
    split: str = "train",
    license: str = "unknown",
    domain_timeout_s: float = 30.0,
    streaming: bool = True,
    trust_remote_code: bool = True,
) -> list[tuple[SourceRef, np.ndarray]]:
    """Randomly bootstrap a cross-domain source pool from an entire dataset's config catalog.

    Works against any Hugging Face dataset repo id, not just Monash --
    ``dataset_name`` is just a string. Hand-picking which domains go into a
    realism-stress task makes the benchmark only as generalizable as the
    curator's domain list; this instead discovers every config
    ``dataset_name`` exposes and draws a ``seed``-controlled random sample of
    ``n_domains`` of them. ``n_domains=None`` (the default) means use every
    config the dataset has -- deliberately a large/unbounded default so a
    config that doesn't specify it still gets broad coverage rather than one
    arbitrary domain. ``total_limit`` series are split roughly evenly across
    whichever domains are actually sampled.

    Some datasets have configs that fail to load (e.g. several Monash TSF
    configs error because the dataset's own loading script parses frequency
    strings like "H"/"min"/"ME" using pandas aliases that changed upstream)
    or are effectively huge, non-streamed downloads that can hang on a slow
    connection. Either kind of failure is skipped in favor of the next
    randomly drawn candidate (bounded by ``domain_timeout_s`` per attempt) so
    one broken or oversized domain can't fail or stall the whole build; every
    skip is a plain print so the caller can see the substitution.
    """
    try:
        from datasets import get_dataset_config_names
    except ImportError as exc:
        raise ImportError("install 'datasets' to load Hugging Face data") from exc

    try:
        catalog = get_dataset_config_names(dataset_name, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"could not list configs for '{dataset_name}'") from exc
    if not catalog:
        raise RuntimeError(f"'{dataset_name}' exposes no configs to bootstrap from")

    target = len(catalog) if n_domains is None else min(n_domains, len(catalog))
    per_domain = max(1, total_limit // max(1, target))

    rng = np.random.default_rng(seed)
    order = [catalog[i] for i in rng.permutation(len(catalog))]

    pool: list[tuple[SourceRef, np.ndarray]] = []
    used: list[str] = []
    for domain in order:
        if len(used) >= target:
            break
        try:
            items = _load_with_timeout(dataset_name, domain, split, field_name, streaming, trust_remote_code, per_domain, license, domain_timeout_s)
        except concurrent.futures.TimeoutError:
            print(f"bootstrap_catalog: skipping domain '{domain}' (timed out after {domain_timeout_s:.0f}s); trying another")
            continue
        except Exception as exc:
            print(f"bootstrap_catalog: skipping domain '{domain}' ({exc}); trying another")
            continue
        if not items:
            continue
        pool.extend(items)
        used.append(domain)

    if not pool:
        raise RuntimeError(f"could not bootstrap any domain from '{dataset_name}' ({len(catalog)} candidates tried)")

    print(f"bootstrap_catalog: pooled {len(pool)} series across {len(used)} domains from '{dataset_name}': {sorted(used)}")
    return pool


@SOURCES.register("monash")
def monash(subset: str = "tourism_monthly", limit: int = 100, license: str = "CC-BY-4.0") -> Iterator[tuple[SourceRef, np.ndarray]]:
    """Yield series from one Monash subset via the HuggingFace datasets hub.

    Verify the subset license before redistributing anything derived from it;
    Monash subsets do not all share the same terms.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install 'datasets' to load Monash data") from exc

    try:
        ds = load_dataset("Monash-University/monash_tsf", subset, split="train", streaming=True, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            "could not load Monash dataset; Monash-University/monash_tsf is script-backed and may not be accessible in this environment. "
            "Install a compatible datasets version and ensure network/HF access is available."
        ) from exc

    for i, row in enumerate(ds):
        if i >= limit:
            break
        values = np.asarray(row["target"], dtype=float)
        ref = SourceRef(corpus=f"monash/{subset}", item_id=str(i), sha256=_hash(values), license=license)
        yield ref, values
