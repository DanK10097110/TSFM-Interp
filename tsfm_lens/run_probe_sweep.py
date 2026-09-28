"""D1's regenerable driver (`ROADMAP.md` sec 34.5 Item D1.2).

Answers "can this repo analyse model X" for a fixed list of candidate TSFM
checkpoints, without writing a line of adapter code for any of them --
reusing `GenericHFAdapter` (the zero-code probe path, sec 34.5's own
premise) and `models/adapter_check.py`'s existing checklist rather than
building a second probing mechanism (`CLAUDE.md` sec 2.2).

Three outcomes per candidate, not two, because the first live sweep
(2026-09-18) found a real third one: `resolved` (all four seams probed
successfully), `refused:<gate>` (one of `GenericHFAdapter`'s own two
refusal gates fired -- contiguity or multivariate), and
`crashed_upstream:<cause>` (the checkpoint's own `config.json` never
reached `transformers.AutoConfig.from_pretrained` in a recognizable form,
so no seam of this repo's own code ever ran at all). Collapsing the third
into "refused" would misattribute a `transformers`-level compatibility gap
to this repo's own gates, which is exactly the confusion the first sweep's
Moirai/Lag-Llama rows exist to correct.

Writes `docs/probe_sweep.md` (the human-readable table) and
`docs/probe_sweep.json` (the same rows, machine-readable) -- never hand-
edit either; rerun this driver instead (`CLAUDE.md` sec 11.34's "a
hand-written index is a claim checked nowhere", the same discipline
`ADAPTERS.md`/`render_adapter_docs.py` already follow for the adapter
table one level up).

Usage: `python run_probe_sweep.py [--quick]`. `--quick` skips the full
`adapter_check.run_adapter_checklist` pass (which downloads and loads real
weights for whichever candidates construct successfully) and only measures
construction -- i.e. whether `transformers.AutoConfig.from_pretrained`
resolves at all. Default mode runs the full checklist for anything that
constructs, since that is the only way to get a live-measured contrast/
contiguity/tier for the ones that make it past `AutoConfig`.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CANDIDATES = [
    {
        "name": "Timer",
        "config": "configs/generic_hf_timer.yaml",
        "checkpoint": "thuml/timer-base-84m",
        "license": "apache-2.0",
        "note": "decoder-only, per-timestep-adjacent patch (32-step) tokenization",
    },
    {
        "name": "Toto",
        "config": "configs/generic_hf_toto.yaml",
        "checkpoint": "Datadog/Toto-Open-Base-1.0",
        "license": "apache-2.0",
        "note": "multivariate-capable by design, used degenerate-univariate here; "
                "standalone `toto-ts` package, config.json has no model_type",
    },
    {
        "name": "Moment",
        "config": "configs/generic_hf_moment.yaml",
        "checkpoint": "AutonLab/MOMENT-1-small",
        "license": "mit",
        "note": "T5-encoder backbone nested one level inside a bespoke wrapper config; "
                "standalone `momentfm` package",
    },
    {
        "name": "LagLlama",
        "config": "configs/generic_hf_lagllama.yaml",
        "checkpoint": "time-series-foundation-models/Lag-Llama",
        "license": "apache-2.0",
        "note": "univariate, non-contiguous lag-feature tokens (the G2 contiguity-gate "
                "target); repo has no config.json at all, gluonts-based",
    },
    {
        "name": "TTM",
        "config": "configs/generic_hf_ttm.yaml",
        "checkpoint": "ibm-granite/granite-timeseries-ttm-r1",
        "license": "apache-2.0",
        "note": "univariate-capable mixer (num_input_channels=1); config.json IS "
                "transformers-shaped but model_type 'tinytimemixer' is unregistered "
                "without the separate `granite-tsfm` package",
    },
    {
        "name": "TimeMoE",
        "config": "configs/generic_hf_timemoe.yaml",
        "checkpoint": "Maple728/TimeMoE-50M",
        "license": "apache-2.0",
        "note": "univariate, per-timestep scalar tokens, MoE routing; real auto_map, "
                "genuinely trust_remote_code-compatible",
    },
    {
        "name": "Moirai",
        "config": "configs/generic_hf_moirai.yaml",
        "checkpoint": "Salesforce/moirai-1.0-R-small",
        "license": "cc-by-nc-4.0",
        "note": "any-variate (the multivariate-refusal-gate target); standalone "
                "`uni2ts` package, config.json has no model_type",
    },
    {
        "name": "VisionTS",
        "config": None,
        "checkpoint": None,
        "license": "mit (VisionTS++ successor only)",
        "note": "no probeable Hugging Face checkpoint exists for either VisionTS or "
                "its VisionTS++ successor (Lefei/VisionTSpp has no config.json either)",
    },
]


def _classify_construct_failure(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, ValueError) and "Unrecognized model" in msg:
        return "crashed_upstream:autoconfig_unrecognized_model_type"
    if isinstance(exc, ValueError) and "does not recognize this architecture" in msg:
        return "crashed_upstream:autoconfig_unregistered_model_type"
    if isinstance(exc, KeyError):
        return f"crashed_upstream:autoconfig_keyerror:{msg}"
    return f"crashed_upstream:other:{type(exc).__name__}"


def _truncate_detail(msg: str, limit: int = 400) -> str:
    # `AutoConfig`'s own "Unrecognized model" message enumerates every one of
    # ~400 registered model_type strings -- real, useful once, and not worth
    # repeating byte-for-byte in every failing row of a generated artifact.
    if msg is None or len(msg) <= limit:
        return msg
    return msg[:limit] + f" ... [truncated, {len(msg)} chars total]"


def probe_one(candidate: dict, quick: bool) -> dict:
    row = {
        "name": candidate["name"],
        "checkpoint": candidate["checkpoint"],
        "license": candidate["license"],
        "note": candidate["note"],
        "outcome": None,
        "detail": None,
    }
    if candidate["config"] is None:
        row["outcome"] = "not_found"
        row["detail"] = "no probeable Hugging Face checkpoint id exists"
        return row

    from tsfm_lens.config import load_config
    from tsfm_lens.pipeline import Context

    cfg = load_config(candidate["config"])
    model_name = candidate["name"]
    # `hub.get` only *constructs* the adapter object (`build_adapter`'s own
    # docstring: "without loading") -- the AutoConfig/AutoModel resolution
    # this driver actually cares about happens inside `ensure_loaded()`,
    # exactly the call `run.py --probe-adapter` makes before doing anything
    # else. Calling it explicitly here, rather than letting the checklist
    # trigger it implicitly inside `prepare()`, is what lets a crash here be
    # told apart from a crash inside the conformance check itself.
    adapter = Context(cfg).hub.get(model_name)
    try:
        adapter.ensure_loaded()
    except Exception as exc:  # noqa: BLE001 - classified below, not swallowed
        row["outcome"] = _classify_construct_failure(exc)
        row["detail"] = _truncate_detail(f"{type(exc).__name__}: {exc}")
        return row

    from tsfm_lens.models.base import NotTimeLocalized

    tier = adapter.capability_tier()
    if quick:
        row["outcome"] = "constructed"
        row["detail"] = f"tier {tier}, adapter class {type(adapter).__name__} " \
                         "(--quick: seams not probed)"
        return row

    from tsfm_lens.models.adapter_check import overall_status, run_adapter_checklist

    try:
        checklist = run_adapter_checklist(cfg, model_name)
    except NotTimeLocalized as exc:
        row["outcome"] = "refused:not_time_localized"
        row["detail"] = str(exc)
        return row
    status = overall_status(checklist)
    fail_rows = [r for r in checklist if r["status"] == "fail"]
    if fail_rows:
        row["outcome"] = f"refused:{fail_rows[0]['name']}"
        row["detail"] = fail_rows[0]["detail"]
    else:
        row["outcome"] = "resolved"
        row["detail"] = f"checklist status={status}; " + "; ".join(
            f"{r['name']}={r['status']}" for r in checklist
        )
    return row


def render_markdown(rows: list) -> str:
    """Pure reduction over already-computed rows -- no network, no model load.

    Split out from `main()` so `--quick`/full-checklist re-runs, and the
    determinism test in `tests/test_probe_sweep.py`, exercise the exact same
    rendering code the real driver uses (`CLAUDE.md` sec 2.2/sec 11.34 --
    never a second, hand-maintained rendering of the same data).
    """
    lines = [
        "<!-- GENERATED by run_probe_sweep.py -- do not hand-edit; rerun the driver. -->",
        "# Candidate checkpoint probe sweep",
        "",
        "Answers \"can this repo analyse model X\" for each candidate in "
        "`ROADMAP.md` sec 19.1's landscape table, using only the zero-code "
        "`generic_hf` path (`ROADMAP.md` sec 34.5 Item D1). No adapter code "
        "was written for any row below.",
        "",
        "| model | checkpoint | license | outcome | detail |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        detail = (row["detail"] or "").replace("\n", " ").replace("|", "\\|")
        if len(detail) > 200:
            detail = detail[:200] + " ..."
        lines.append(
            f"| {row['name']} | `{row['checkpoint']}` | {row['license']} | "
            f"**{row['outcome']}** | {detail} |"
        )
    return "\n".join(lines) + "\n"


def write_artifacts(rows: list, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    (out_dir / "probe_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "probe_sweep.md").write_text(render_markdown(rows), encoding="utf-8")


def run_sweep(quick: bool) -> list:
    rows = []
    for candidate in CANDIDATES:
        print(f"probing {candidate['name']} ...", file=sys.stderr)
        try:
            rows.append(probe_one(candidate, quick))
        except Exception:  # noqa: BLE001 - a driver bug must not lose the other rows
            traceback.print_exc()
            rows.append({
                "name": candidate["name"], "checkpoint": candidate["checkpoint"],
                "license": candidate["license"], "note": candidate["note"],
                "outcome": "driver_error", "detail": traceback.format_exc(),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                     help="only measure construction (AutoConfig resolution), "
                          "skip the full seam checklist")
    args = ap.parse_args()

    rows = run_sweep(args.quick)
    out_dir = Path(__file__).resolve().parent / "docs"
    write_artifacts(rows, out_dir)
    print(f"wrote {out_dir / 'probe_sweep.md'} and {out_dir / 'probe_sweep.json'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
