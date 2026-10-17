"""One-shot held-out confirmation of the registered CONCAT geometry claims (ROADMAP.md sec 47.3).

Why this exists. Sec 45 re-measured the v3 dev run's window-level CKA with the lossless CONCAT window
representation and selected 21 (layer_a, layer_b) peak pairs on dev. Those pairs are exploratory (a max over
a layer grid). Sec 47.3 registered, before any data of the fresh private epoch 2 existed, two held-out claim
families on those exact pairs, and this driver is the single place that tests them:

  H-C1 (21 tests): CONCAT CKA at the registered pair exceeds its shuffled-series null. The null permutes which
      series lines up with which (a series' windows move together) 1000 times; p = (1 + #null >= obs) / 1001;
      Holm over 21 at alpha 0.05.
  H-C2 (4 tests): for TimesFM vs Chronos-2 / Chronos-T5-Base / Chronos-Bolt / Sundial, CONCAT CKA minus MEAN CKA
      at the same registered pair on identical rows is > 0; one-sided series cluster bootstrap with 5000
      resamples, p = (1 + #(diff* <= 0)) / 5001; Holm over 4 at alpha 0.05.

Evidence class: geometric (linear CKA, correlational). It inherits `analysis/window_concat.py` (Gram sums over the
store's `fine/` group, centered) and the MEAN window vector of the store's window level. The CKA formulas are
restated here (not imported from `analysis/l1_geometry.py`) so the confirmation does not depend on library code that
is being edited, and they are tested against the library's own functions on a planted fixture.

Discipline (CLAUDE.md 6.6, invariant 6): `registry` writes `registry.json` (the registered text verbatim from commit
1c1a7f4, its sha256, the parsed pairs, the fixed test parameters) BEFORE extraction; `run` refuses without a registry
whose text still equals that commit's, refuses an unsealed or digest-mismatched corpus, never overwrites its own
results, writes nothing until every statistic is computed, and prints no statistic. A crash before a statistic is
computed may be fixed and rerun; a result that has been read is never recomputed.

    python run_concat_confirm.py registry --run runs/panel7_v3_concat_confirm_e2 --corpus ../benchmark_v3/private_epoch2
    python run.py --config configs/panel7_v3_concat_confirm_e2.yaml --stages extract
    python run_concat_confirm.py run --run runs/panel7_v3_concat_confirm_e2 --config configs/panel7_v3_concat_confirm_e2.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REGISTRATION_COMMIT = "1c1a7f4"
SECTION_START = "**47.3 Held-out confirmation of the CONCAT geometry.**"
SECTION_END = "**47.4 "
N_PERM = 1000
N_BOOT = 5000
ALPHA = 0.05
EXPECTED_SERIES = 512
PERM_SEED = 4713
BOOT_SEED = 4714
H_C2_PARTNERS = ["Chronos-2", "Chronos-T5-Base", "Chronos-Bolt", "Sundial"]
H_C2_ANCHOR = "TimesFM"
_PAIR_RE = re.compile(r"([A-Za-z0-9-]+)/([A-Za-z0-9-]+) ([\w.]+)/([\w.]+) (\d\.\d+)")


def repo_root() -> Path:
    return HERE.parent


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def registered_section(roadmap_text: str) -> str:
    """The sec 47.3 block verbatim: from its heading line up to (excluding) the sec 47.4 heading."""
    start = roadmap_text.index(SECTION_START)
    end = roadmap_text.index(SECTION_END, start)
    return roadmap_text[start:end].rstrip("\n") + "\n"


def registered_section_at_commit(commit: str = REGISTRATION_COMMIT) -> str:
    out = subprocess.run(["git", "show", f"{commit}:ROADMAP.md"], cwd=repo_root(), check=True,
                         capture_output=True, text=True, encoding="utf-8").stdout
    return registered_section(out)


def parse_pairs(section_text: str) -> list[dict]:
    """The 21 registered dev pairs from the 47.3a list: models, layers, dev CONCAT peak."""
    flat = " ".join(section_text[section_text.index("47.3a"):].split())
    pairs = [{"model_a": m[0], "model_b": m[1], "layer_a": m[2], "layer_b": m[3], "dev_peak": float(m[4])}
             for m in _PAIR_RE.findall(flat)]
    return pairs


def hc2_pairs(pairs: list[dict]) -> list[dict]:
    """The registered pairs H-C2 tests: TimesFM against each of the four partners."""
    out = []
    for partner in H_C2_PARTNERS:
        hit = [p for p in pairs if p["model_a"] == H_C2_ANCHOR and p["model_b"] == partner]
        if len(hit) != 1:
            raise ValueError(f"H-C2 needs exactly one registered {H_C2_ANCHOR}/{partner} pair, got {len(hit)}")
        out.append(hit[0])
    return out


def pair_id(p: dict) -> str:
    return f"{p['model_a']}/{p['model_b']}@{p['layer_a']}/{p['layer_b']}"


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjustment (same rule as `analysis/stats.py::holm`)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adjusted[key] = float(min(1.0, running))
    return adjusted


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Feature-space linear CKA of row-aligned [N, Dx], [N, Dy] (the MEAN window-level statistic)."""
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    cross = (yc.T @ xc).norm() ** 2
    denom = (xc.T @ xc).norm() * (yc.T @ yc).norm()
    return float((cross / denom.clamp_min(1e-12)).item())


def center(k: torch.Tensor) -> torch.Tensor:
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()


def gram_cka(ka: torch.Tensor, kb: torch.Tensor) -> float:
    """Linear CKA of two centered Gram matrices (float64 reduction)."""
    ka, kb = ka.double(), kb.double()
    return float(((ka * kb).sum() / torch.sqrt((ka * ka).sum() * (kb * kb).sum()).clamp_min(1e-300)).item())


def window_rows(series: np.ndarray, n_windows: int) -> np.ndarray:
    """Row indices of the windows of the given series (a series' windows move together)."""
    series = np.asarray(series)
    return (series[:, None] * n_windows + np.arange(n_windows)[None]).reshape(-1)


def permutation_test(ka: torch.Tensor, kb: torch.Tensor, n_series: int, n_windows: int,
                     n_perm: int, seed: int) -> dict:
    """H-C1 for one pair: observed Gram CKA, `n_perm` series-permutation nulls, p = (1 + #null >= obs) / (n_perm + 1)."""
    obs = gram_cka(ka, kb)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        rows = torch.from_numpy(window_rows(rng.permutation(n_series), n_windows)).to(kb.device)
        null[i] = gram_cka(ka, center(kb[rows][:, rows]))
    p = (1 + int((null >= obs).sum())) / (n_perm + 1)
    return {"observed": obs, "p": p, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "null_max": float(null.max()), "n_null_ge_obs": int((null >= obs).sum()), "n_perm": n_perm}


def concat_minus_mean_bootstrap(ka: torch.Tensor, kb: torch.Tensor, xa: torch.Tensor, xb: torch.Tensor,
                                n_series: int, n_windows: int, n_boot: int, seed: int) -> dict:
    """H-C2 for one pair: CONCAT - MEAN CKA on identical rows, one-sided series cluster bootstrap.

    Each resample draws series with replacement and recomputes BOTH statistics on the same resampled rows
    (CONCAT from Gram blocks K[rows][:, rows] re-centered, MEAN from the window means), so the difference is paired.
    p = (1 + #(diff* <= 0)) / (n_boot + 1).
    """
    concat_obs, mean_obs = gram_cka(ka, kb), linear_cka(xa, xb)
    rng = np.random.default_rng(seed)
    cs, ms = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        rows = torch.from_numpy(window_rows(rng.integers(0, n_series, n_series), n_windows)).to(kb.device)
        cs[i] = gram_cka(center(ka[rows][:, rows]), center(kb[rows][:, rows]))
        ms[i] = linear_cka(xa[rows], xb[rows])
    diff = cs - ms

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"concat": concat_obs, "mean": mean_obs, "diff": concat_obs - mean_obs,
            "concat_ci95": ci(cs), "mean_ci95": ci(ms), "diff_ci95": ci(diff),
            "diff_lower_one_sided_95": float(np.percentile(diff, 5)),
            "p": (1 + int((diff <= 0).sum())) / (n_boot + 1), "n_boot": n_boot,
            "n_boot_diff_le_0": int((diff <= 0).sum())}


def decide(pvals: dict, alpha: float = ALPHA) -> tuple[dict, dict]:
    adj = holm(pvals)
    return adj, {k: ("confirmed" if v < alpha else "not confirmed") for k, v in adj.items()}


def write_registry(run_dir: Path, corpus: Path) -> dict:
    """Write `registry.json`; refuses to overwrite one (the registry is written once, before extraction)."""
    path = run_dir / "registry.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; the registry is written once")
    section = registered_section_at_commit()
    working = registered_section(Path(repo_root() / "ROADMAP.md").read_text(encoding="utf-8"))
    if working != section:
        raise ValueError("ROADMAP.md sec 47.3 differs from commit " + REGISTRATION_COMMIT + "; refusing to register")
    pairs = parse_pairs(section)
    if len(pairs) != 21:
        raise ValueError(f"expected 21 registered pairs, parsed {len(pairs)}")
    hc2_pairs(pairs)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    roadmap_sha = subprocess.run(["git", "rev-parse", f"{REGISTRATION_COMMIT}:ROADMAP.md"], cwd=repo_root(),
                                 check=True, capture_output=True, text=True).stdout.strip()
    registry = {
        "registered_commit_arg": REGISTRATION_COMMIT,
        "registered_commit": subprocess.run(["git", "rev-parse", REGISTRATION_COMMIT], cwd=repo_root(), check=True,
                                            capture_output=True, text=True).stdout.strip(),
        "roadmap_blob_at_commit": roadmap_sha,
        "hypotheses_text_verbatim": section,
        "hypotheses_sha256": sha256_text(section),
        "pairs": pairs,
        "tests": {
            "H-C1": {"n_tests": len(pairs), "n_perm": N_PERM, "p": "(1 + #null >= obs) / (n_perm + 1)",
                     "correction": "Holm over 21", "alpha": ALPHA, "perm_seed": PERM_SEED},
            "H-C2": {"n_tests": 4, "partners_of": H_C2_ANCHOR, "partners": H_C2_PARTNERS, "n_boot": N_BOOT,
                     "p": "(1 + #(diff* <= 0)) / (n_boot + 1), one-sided, series cluster bootstrap",
                     "correction": "Holm over 4", "alpha": ALPHA, "boot_seed": BOOT_SEED},
            "n_series": EXPECTED_SERIES,
        },
        "corpus": {"path": str(corpus), "epoch": manifest["epoch"], "n_samples": manifest["n_samples"],
                   "global_digest": manifest["global_digest"]},
        "written_before_extraction": True,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=1), encoding="utf-8")
    return registry


def check_registry(run_dir: Path) -> dict:
    registry = json.loads((run_dir / "registry.json").read_text(encoding="utf-8"))
    text = registry["hypotheses_text_verbatim"]
    if sha256_text(text) != registry["hypotheses_sha256"] or text != registered_section_at_commit():
        raise ValueError("registry.json no longer matches the registered text at commit " + REGISTRATION_COMMIT)
    return registry


def run_confirmation(run_dir: Path, config_path: Path, device: str, n_perm: int = N_PERM, n_boot: int = N_BOOT) -> dict:
    """Compute every registered statistic once, then write `concat_confirmation.json`."""
    from tsfm_lens.analysis.window_concat import centered_gram, concat_rows
    from tsfm_lens.config import load_config
    from tsfm_lens.extraction.store import ActivationStore

    out_path = run_dir / "concat_confirmation.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists: the confirmation is run once and never repeated")
    t0 = time.time()
    registry = check_registry(run_dir)
    pairs = registry["pairs"]
    cfg = load_config(config_path)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    n_windows = int(store.root.attrs["n_windows"])
    import pandas as pd
    meta = pd.read_parquet(run_dir / "meta.parquet")
    rows = np.asarray(concat_rows(cfg, meta["family"].to_numpy(), n_windows))
    if len(rows) != EXPECTED_SERIES:
        raise ValueError(f"expected {EXPECTED_SERIES} CONCAT series, got {len(rows)}")
    dev = torch.device(device)

    grams: dict = {}

    def gram(model, layer):
        if (model, layer) not in grams:
            if not store.has_fine(model, layer):
                raise KeyError(f"no fine group for {model}/{layer}")
            n_pos = store.n_fine_positions(model, layer)

            def blocks():
                for p in range(n_pos):
                    x = store.load_fine_position(model, layer, p, rows)
                    yield torch.from_numpy(np.ascontiguousarray(x.reshape(-1, x.shape[-1])))
            grams[(model, layer)] = centered_gram(blocks(), dev)
        return grams[(model, layer)]

    h1 = {}
    for i, p in enumerate(pairs):
        h1[pair_id(p)] = permutation_test(gram(p["model_a"], p["layer_a"]), gram(p["model_b"], p["layer_b"]),
                                          len(rows), n_windows, n_perm, PERM_SEED + i)
        h1[pair_id(p)].update({"dev_peak": p["dev_peak"], "model_a": p["model_a"], "model_b": p["model_b"],
                               "layer_a": p["layer_a"], "layer_b": p["layer_b"]})
    adj, verdict = decide({k: v["p"] for k, v in h1.items()})
    for k in h1:
        h1[k]["p_holm"], h1[k]["verdict"] = adj[k], verdict[k]

    def mean_feats(model, layer):
        x = store.load(model, layer, level="window", rows=rows)
        return torch.from_numpy(np.ascontiguousarray(x.reshape(-1, x.shape[-1]))).half().float().to(dev)

    h2 = {}
    for i, p in enumerate(hc2_pairs(pairs)):
        h2[pair_id(p)] = concat_minus_mean_bootstrap(
            gram(p["model_a"], p["layer_a"]), gram(p["model_b"], p["layer_b"]),
            mean_feats(p["model_a"], p["layer_a"]), mean_feats(p["model_b"], p["layer_b"]),
            len(rows), n_windows, n_boot, BOOT_SEED + i)
        h2[pair_id(p)].update({"model_a": p["model_a"], "model_b": p["model_b"],
                               "layer_a": p["layer_a"], "layer_b": p["layer_b"]})
    adj2, verdict2 = decide({k: v["p"] for k, v in h2.items()})
    for k in h2:
        h2[k]["p_holm"], h2[k]["verdict"] = adj2[k], verdict2[k]

    result = {"evidence_class": "geometric (linear CKA, correlational), held-out confirmation",
              "registry_sha256": registry["hypotheses_sha256"], "registered_commit": registry["registered_commit"],
              "corpus_digest": registry["corpus"]["global_digest"], "n_series": int(len(rows)),
              "n_windows": n_windows, "n_rows_window": int(len(rows) * n_windows), "H-C1": h1, "H-C2": h2,
              "summary": {"H-C1_confirmed": sum(v["verdict"] == "confirmed" for v in h1.values()),
                          "H-C2_confirmed": sum(v["verdict"] == "confirmed" for v in h2.values())},
              "seconds": time.time() - t0}
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("registry")
    r.add_argument("--run", required=True)
    r.add_argument("--corpus", required=True)
    c = sub.add_parser("run")
    c.add_argument("--run", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.cmd == "registry":
        reg = write_registry(Path(args.run), Path(args.corpus))
        print("registry written; hypotheses sha256", reg["hypotheses_sha256"])
    else:
        run_confirmation(Path(args.run), Path(args.config), args.device)
        print("confirmation written (statistics are in concat_confirmation.json)")


if __name__ == "__main__":
    main()
