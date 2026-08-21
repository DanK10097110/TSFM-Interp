"""Build a genuine known-lineage pair for provenance detection (`ROADMAP.md`
sec 6.3.1 Option E, minimally scoped by sec 23.3 D2).

The provenance-detection thread (`analysis/error_fingerprint.py`, sec
6.3.1 Option C) failed three of its own controls on its first real sweep,
and the root cause traces to its only "positive" example --
`chronos-t5-small` vs `chronos-t5-base` -- which, as that same Findings
block discovered, do **not** share initialization weights at all; they
only share a training recipe and corpus. So every provenance method in
this repo has been evaluated against a positive case that was never
actually a positive case for the thing being detected (shared lineage).

This module fixes that by fine-tuning one real parent checkpoint
(`amazon/chronos-t5-small`) on two **disjoint** data splits, producing two
children that share initialization but nothing else -- a real, unambiguous
positive pair for the first time. Minimally scoped per D2: one parent, two
children, reusing the corpus this repo already has sealed rather than
building anything new.

Two things are deliberately separated so the mechanism is testable without
a network or a real checkpoint download:

- `_train_steps` is the actual training loop, over already-tokenized
  (input_ids, attention_mask, labels) batches -- pure `torch`, testable
  offline against a tiny synthetic `T5Config` (mirrors
  `tests/test_random_init.py`'s pattern for the same reason).
- `_build_batches` does the real tokenization via the checkpoint's own
  `ChronosTokenizer` (`context_input_transform`/`label_input_transform`,
  the same two calls the upstream chronos-forecasting training recipe
  uses) -- this needs a real pipeline object and is not unit-tested here;
  it is exercised end-to-end by actually running a fine-tune, the same
  division of labour `CLAUDE.md` sec 6.2 already draws for adapter code
  ("the adapters themselves are one-line callers ... not run in CI").
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import DataConfig
from ..data import BenchmarkData, load_benchmark
from ..utils import log, save_json


def _train_steps(model, batches: list, steps: int, lr: float, seed: int,
                 device: str = "cpu") -> list:
    """Fine-tune `model` for `steps` optimizer steps over `batches` (cycled
    if `steps > len(batches)`), returning the per-step loss.

    Plain AdamW + the model's own seq2seq LM loss (`labels` with `-100` at
    ignored positions) -- no scheduler, no mixed precision, nothing beyond
    what a "modest fine-tune" (`ROADMAP.md` sec 6.3.1 Option E) needs. Kept
    deliberately simple: the point of this experiment is a genuine lineage
    label, not a well-tuned model.
    """
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        input_ids, attention_mask, labels = batches[step % len(batches)]
        opt.zero_grad()
        out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                    labels=labels.to(device))
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach().cpu()))
    model.eval()
    return losses


def check_horizon_matches_checkpoint(prediction_length: int, horizon: int, checkpoint: str) -> None:
    """Reconcile the requested training horizon against the parent
    checkpoint's own baked-in `prediction_length` (`ROADMAP.md` sec 23.3 D2's
    second Findings).

    `chronos-forecasting`'s `ChronosTokenizer.label_input_transform` hard-
    asserts `label.shape[-1] == self.config.prediction_length` unconditionally
    -- this repo's first live run of this module hit that assertion as a bare
    `AssertionError` deep inside tokenization, with no indication of *why*.
    Checking it here, before any tokenization happens, turns that into an
    actionable error naming the exact mismatch and the fix.
    """
    if horizon != prediction_length:
        raise ValueError(
            f"finetune_child: checkpoint {checkpoint!r} was trained with "
            f"prediction_length={prediction_length}, but a training horizon of "
            f"{horizon} was requested. chronos-forecasting's "
            f"ChronosTokenizer.label_input_transform hard-asserts these match "
            f"exactly, so fine-tuning this checkpoint requires --horizon "
            f"{prediction_length} (and a matching horizon in whatever data config "
            f"scores the result afterward) -- it is not safe to derive/change "
            f"silently since a config elsewhere may assume the requested value.")


def _build_batches(pipeline, contexts: np.ndarray, targets: np.ndarray,
                   batch_size: int) -> list:
    """Tokenize (context, target) series pairs into (input_ids,
    attention_mask, labels) batches via the checkpoint's own `ChronosTokenizer`
    -- the same two calls (`context_input_transform`, `label_input_transform`)
    the upstream chronos-forecasting training recipe uses, so this reproduces
    what the checkpoint was originally trained with rather than a fresh
    tokenization convention.
    """
    tok = pipeline.tokenizer
    batches = []
    for start in range(0, len(contexts), batch_size):
        ctx = torch.as_tensor(contexts[start:start + batch_size], dtype=torch.float32)
        tgt = torch.as_tensor(targets[start:start + batch_size], dtype=torch.float32)
        input_ids, attention_mask, scale = tok.context_input_transform(ctx)
        label_ids, label_mask = tok.label_input_transform(tgt, scale)
        label_ids = label_ids.clone()
        label_ids[label_mask == 0] = -100
        batches.append((input_ids, attention_mask, label_ids))
    return batches


@dataclass
class ChildResult:
    out_dir: str
    row_indices: list
    steps: int
    lr: float
    seed: int
    losses: list
    corpus_digest: Optional[str]


def finetune_child(parent_checkpoint: str, data: BenchmarkData, row_indices: np.ndarray,
                   *, steps: int, lr: float, batch_size: int, seed: int, out_dir: Path,
                   device: str = "cpu") -> ChildResult:
    """Fine-tune `parent_checkpoint` on `data`'s rows at `row_indices` for
    `steps` steps, saving the resulting checkpoint to `out_dir`.

    `row_indices` is the disjointness mechanism: two calls with
    non-overlapping `row_indices` (over the same `data`) produce two
    children that share initialization and nothing else -- the positive
    pair Option E exists to build. Recorded alongside the checkpoint so a
    later session can verify the splits never overlapped.
    """
    from chronos import ChronosPipeline

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ChronosPipeline.from_pretrained(parent_checkpoint, device_map=device,
                                               torch_dtype=torch.float32)
    contexts = data.contexts()[row_indices]
    targets = data.targets()[row_indices]
    check_horizon_matches_checkpoint(
        pipeline.model.config.prediction_length, targets.shape[-1], parent_checkpoint)
    batches = _build_batches(pipeline, contexts, targets, batch_size)
    losses = _train_steps(pipeline.model.model, batches, steps, lr, seed, device)

    pipeline.model.model.save_pretrained(out_dir)
    manifest = ChildResult(
        out_dir=str(out_dir), row_indices=[int(i) for i in row_indices], steps=steps,
        lr=lr, seed=seed, losses=losses, corpus_digest=data.corpus_digest)
    save_json(out_dir / "finetune_manifest.json", manifest.__dict__)
    log.info("finetune_child: %s -> %s, %d steps, final loss %.4f",
            parent_checkpoint, out_dir, steps, losses[-1] if losses else float("nan"))
    return manifest


def disjoint_split(n: int, seed: int) -> tuple:
    """Two non-overlapping row-index halves of `range(n)`, shuffled by
    `seed` so the split isn't just "first half / second half" of a corpus
    that may itself be grouped by family/generator (`CLAUDE.md` sec 11.24's
    class of trap -- a structured corpus makes a naive positional split a
    hidden confound, not a random one).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    half = n // 2
    return idx[:half], idx[half:]


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", default="amazon/chronos-t5-small")
    p.add_argument("--corpus", required=True, help="path to a sealed corpus dir")
    p.add_argument("--out", required=True, help="output base dir; children saved to "
                                               "<out>/child_light, <out>/child_drifted")
    p.add_argument("--context-len", type=int, default=64)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--light-steps", type=int, default=50)
    p.add_argument("--drifted-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_cfg = DataConfig(source="sealed", path=args.corpus, context_len=args.context_len,
                          horizon=args.horizon)
    data = load_benchmark(data_cfg, seed=args.seed)
    light_idx, drifted_idx = disjoint_split(data.n, args.seed)
    log.info("finetune_child: %d rows -> light=%d, drifted=%d (disjoint)",
            data.n, len(light_idx), len(drifted_idx))

    out = Path(args.out)
    light = finetune_child(args.parent, data, light_idx, steps=args.light_steps, lr=args.lr,
                           batch_size=args.batch_size, seed=args.seed,
                           out_dir=out / "child_light", device=args.device)
    drifted = finetune_child(args.parent, data, drifted_idx, steps=args.drifted_steps,
                             lr=args.lr, batch_size=args.batch_size, seed=args.seed + 1,
                             out_dir=out / "child_drifted", device=args.device)
    save_json(out / "lineage.json", {
        "parent": args.parent, "corpus_digest": data.corpus_digest,
        "child_light": light.__dict__, "child_drifted": drifted.__dict__,
        "disjoint_verified": bool(set(light_idx.tolist()).isdisjoint(set(drifted_idx.tolist()))),
    })


if __name__ == "__main__":
    _cli()
