"""Test exported mixed profiles on unseen M4 daily histories without refitting."""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
from scipy.stats import binomtest

from tsfm_lens.steering import SteeredForecaster
from tsfm_lens.steering.requests import requested_change


def sample(path, count, seed):
    with pa.memory_map(str(path)) as source:
        table = pa.ipc.open_stream(source).read_all()
    histories = []
    seen = set()
    for index in np.random.default_rng(seed).permutation(len(table)):
        values = np.asarray(table["target"][int(index)].as_py(), dtype=np.float32)[-72:]
        item = str(table["item_id"][int(index)].as_py())
        if len(values) != 72 or not np.isfinite(values).all() or item in seen:
            continue
        histories.append((item, values[:64], values[64:]))
        seen.add(item)
        if len(histories) == count:
            return histories
    raise ValueError("insufficient distinct M4 daily series with 72 finite final observations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("arrow", type=Path, help="GIFT-Eval m4_daily data-00000-of-00001.arrow")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    text = args.report.read_text()
    rows = json.loads(re.search(r'id="steering-results">(.*?)</script>', text).group(1))
    bundles = re.findall(r'data:application/zip;base64,([^"]+)', text)
    assert len(rows) == len(bundles)
    seed = 42
    histories = sample(args.arrow, 24, seed)
    data = dict(dataset="Salesforce/GiftEval m4_daily", source_sha256=sha256(args.arrow.read_bytes()).hexdigest(),
                seed=seed, context_length=64, horizon=8, groups=[])
    with TemporaryDirectory() as tmp:
        for row, encoded in zip(rows, bundles):
            if row["kind"] != "combined":
                continue
            path = Path(tmp) / "model.zip"
            path.write_bytes(base64.b64decode(encoded))
            predictor = SteeredForecaster.load(path)
            cases = []
            for item, history, future in histories:
                result = predictor.compare(history)
                replay = predictor.steerer.replay(history[None], result)[0]
                baseline, target = result.baseline[0], result.requested_change[0]
                size = np.linalg.norm(target)
                error = float(np.linalg.norm(replay - baseline - target) / size) if size else None
                matches = bool(np.allclose(replay, result.forecast[0], rtol=1e-5, atol=1e-6))
                median = list(result.quantiles).index(.5)
                patterns = [requested_change(baseline, history, result.quantiles, term)
                            for term in predictor.request.terms]
                matrix = np.column_stack([p.ravel() for p in patterns])
                gains, _, rank, _ = np.linalg.lstsq(matrix, (replay - baseline).ravel(), rcond=None)
                cases.append(dict(component_gains=gains.tolist() if rank == len(patterns) else None, item=item, relative_error=error, replay_matches=matches,
                                  verified=matches and error is not None and error <= predictor.steerer.config.tolerance,
                                  baseline=baseline.tolist(), requested_change=target.tolist(), replay=replay.tolist(),
                                  forecast=result.forecast[0].tolist(), coefficient_norm=result.diagnostics[0]["coefficient_norm"],
                                  baseline_mae=float(np.abs(baseline[:, median] - future).mean()),
                                  steered_mae=float(np.abs(replay[:, median] - future).mean())))
            hits = sum(c["verified"] for c in cases)
            ci = binomtest(hits, len(cases)).proportion_ci(method="wilson")
            data["groups"].append(dict(model=row["model"], request=row["request"], verified=hits, total=len(cases),
                                       wilson95=list(ci), cases=cases))
    args.out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
