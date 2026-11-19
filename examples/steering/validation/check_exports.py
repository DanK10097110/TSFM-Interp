"""Reload report ZIPs on real inputs and vary only the edit cap on the largest misses."""
import argparse
import base64
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from audit import read
from tsfm_lens.config import load_config
from tsfm_lens.pipeline import Context
from tsfm_lens.steering import SteeredForecaster
from tsfm_lens.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    data = Context(cfg).data
    ids = list(data.meta.series_id.astype(str))
    evidence = read(args.report)
    bundles = re.findall(r'data:application/zip;base64,([^"]+)', args.report.read_text())
    groups = evidence["groups"]
    assert len(bundles) == len(groups)
    results = []
    with TemporaryDirectory() as directory:
        for group, encoded in zip(groups, bundles):
            summary = group["summary"]
            if summary["kind"] not in ("bounds", "combined"):
                continue
            path = Path(directory) / "model.zip"
            path.write_bytes(base64.b64decode(encoded))
            predictor = SteeredForecaster.load(path)
            cases = [c for c in group["evaluations"] if c["relative_error"] is not None]
            if not cases:
                continue
            worst = max(cases, key=lambda c: c["relative_error"])
            context = data.contexts()[ids.index(worst["series"])]
            original_radius = predictor.steerer.config.radius
            trials = []
            for factor in (1, 2, 4):
                predictor.steerer.config.radius = factor * original_radius
                result = predictor.compare(context)
                replay = predictor.steerer.replay(context[None], result)
                np.testing.assert_allclose(replay, result.forecast, rtol=1e-5, atol=1e-6)
                if factor == 1:
                    np.testing.assert_allclose(replay[0], worst["replay"], rtol=1e-5, atol=1e-6)
                info = result.diagnostics[0]
                trials.append(dict(radius=info["radius"], error=info["relative_error"], norm=info["coefficient_norm"]))
            results.append(dict(model=summary["model"], request=summary["request"], series=worst["series"], trials=trials))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
