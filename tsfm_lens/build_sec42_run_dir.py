"""Build and narrow the ROADMAP.md sec 42 step 6 run directory from `panel7_v3_dev`.

`build` makes `runs/<dst>` from a finished run WITHOUT copying its bulk or its conclusions:

  * `activations.zarr` and each target's PRIMARY SAE checkpoint (`sae/<model>/<layer>.pt`) are
    HARDLINKED (the 11 GB that would not fit the home quota). Safe only because nothing on the
    sec 42 path writes into an existing file in place: zarr's DirectoryStore writes every key to a
    temp file and `os.replace`s it, so a rewrite gives the new directory its own inode, and the
    stages run (`concepts`, `exemplars`, `report`) never call `torch.save` on a primary
    checkpoint. `torch.save` WOULD write through a hardlink, so the stability replicates
    (`<layer>@r<i>.pt`, retrained by the concepts stage) are NOT linked or copied; the stage
    trains fresh ones into the new directory;
  * every other input is a real copy (JSON, parquet, yaml), except what the new run must
    recompute: the concepts stage's products (`sae/` apart from `meta.json`, the per-target
    `*_ablation.json`), `confirm/`, `hypotheses.json`, `report/`, `report.html`. A stage
    self-skips when its artifact exists, so a copied artifact would silently stand in for the
    rerun.

`select` writes the narrowed target set: `sae/meta.json` keeps only the selected targets (the
concepts stage and every reader of `sae/meta.json` then see exactly that set), the full record
(every target, with the `admission_sec42` verdicts) is kept as `sae/meta_full.json`, and
`sae/target_selection.json` states the rule and the reason for every target.

    python build_sec42_run_dir.py build --src runs/panel7_v3_dev --dst runs/panel7_v3_sec42
    python run_admission_reeval.py --config configs/panel7_v3_sec42.yaml
    python build_sec42_run_dir.py select --run runs/panel7_v3_sec42
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.utils import load_json, save_json  # noqa: E402

HARDLINK_TOP = ("activations.zarr",)
SKIP_TOP = ("report", "report.html", "confirm", "hypotheses.json")
KEEP_FROM_SAE = ("meta.json",)
PER_TARGET = 2


def is_primary_checkpoint(name: str) -> bool:
    return name.endswith(".pt") and "@r" not in name


def build(src: Path, dst: Path) -> dict:
    """Create `dst` from `src` per the module docstring; refuses an existing `dst`."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists; a run directory is never written into twice")
    dst.mkdir(parents=True)
    n = {"hardlinked": 0, "copied": 0, "skipped": 0}
    for entry in sorted(src.iterdir()):
        if entry.name in SKIP_TOP:
            n["skipped"] += 1
        elif entry.name in HARDLINK_TOP:
            shutil.copytree(entry, dst / entry.name, copy_function=os.link)
            n["hardlinked"] += sum(1 for p in (dst / entry.name).rglob("*") if p.is_file())
        elif entry.name == "sae":
            for p in sorted(entry.rglob("*")):
                rel = p.relative_to(entry)
                if p.is_dir():
                    (dst / "sae" / rel).mkdir(parents=True, exist_ok=True)
                elif is_primary_checkpoint(p.name):
                    (dst / "sae" / rel).parent.mkdir(parents=True, exist_ok=True)
                    os.link(p, dst / "sae" / rel)
                    n["hardlinked"] += 1
                elif rel.as_posix() in KEEP_FROM_SAE:
                    shutil.copy2(p, dst / "sae" / rel)
                    n["copied"] += 1
                else:
                    n["skipped"] += 1
        elif entry.is_dir():
            shutil.copytree(entry, dst / entry.name)
            n["copied"] += sum(1 for p in (dst / entry.name).rglob("*") if p.is_file())
        else:
            shutil.copy2(entry, dst / entry.name)
            n["copied"] += 1
    return n


def select_targets(meta: dict, screen: dict, per_model: int = PER_TARGET) -> dict:
    """The sec 42 step 6 target set: `{"targets": [key, ...], "reasons": {key: [str, ...]}}`.

    A target is kept when (a) its `admission_sec42` verdict is `passed` True, or (b) it is one of the
    `per_model` highest-scoring layers (`layer_screen`'s own work_bend score) among that model's
    `layer_screen`-selected layers that were also trained. A pick is kept whatever its admission
    verdict, and its reason says what the verdict was. Order follows `meta`.
    """
    reasons: dict = {}
    for key, entry in meta.items():
        verdict = entry.get("admission_sec42")
        if verdict is None:
            raise ValueError(f"{key} has no admission_sec42; run run_admission_reeval.py first")
        if verdict["passed"] is True:
            reasons.setdefault(key, []).append("admitted under the sec 42 gate")
    for model, rec in screen.items():
        score = dict(zip(rec["layers"], rec["score_per_layer"]))
        trained = [k for k in meta if k.split("/", 1)[0] == model]
        cand = [(score[k.split("/", 1)[1]], k) for k in trained
                if k.split("/", 1)[1] in rec["selected"] and k.split("/", 1)[1] in score]
        for rank, (sc, key) in enumerate(sorted(cand, key=lambda t: -t[0])[:per_model], start=1):
            v = meta[key]["admission_sec42"]
            note = "admitted" if v["passed"] is True else f"not admitted ({v['reason']})"
            reasons.setdefault(key, []).append(
                f"layer_screen pick {rank} of {per_model} for {model} "
                f"(work_bend score {sc:.6g}); sec 42 verdict: {note}")
    return {"targets": [k for k in meta if k in reasons], "reasons": reasons}


def select(run: Path, per_model: int = PER_TARGET) -> dict:
    """Write `sae/meta_full.json`, `sae/target_selection.json` and the narrowed `sae/meta.json`."""
    run = Path(run)
    meta_path, full_path = run / "sae" / "meta.json", run / "sae" / "meta_full.json"
    if full_path.exists():
        raise SystemExit(f"{full_path} exists; the target set was already narrowed")
    meta = load_json(meta_path)
    sel = select_targets(meta, load_json(run / "layer_screen" / "selection.json"), per_model)
    shutil.copy2(meta_path, full_path)
    save_json(meta_path, {k: meta[k] for k in sel["targets"]})
    record = {"rule": (f"admitted under the sec 42 gate (admission_sec42.passed is true), plus up to "
                       f"{per_model} layer_screen picks per model: the highest work_bend scores "
                       f"among that model's layer_screen-selected layers that were trained, kept "
                       f"whatever their admission verdict"),
              "n_trained": len(meta), "n_selected": len(sel["targets"]),
              "not_selected": [k for k in meta if k not in sel["reasons"]], **sel}
    save_json(run / "sae" / "target_selection.json", record)
    return record


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--dst", required=True)
    s = sub.add_parser("select")
    s.add_argument("--run", required=True)
    s.add_argument("--per-model", type=int, default=PER_TARGET)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(build(Path(args.src), Path(args.dst)))
    else:
        rec = select(Path(args.run), args.per_model)
        print(rec["n_selected"], "of", rec["n_trained"], "targets selected")


if __name__ == "__main__":
    main()
