"""Exploratory single receiver vocabularies shared by all donor features."""
import argparse
import json
import numpy as np
import torch

from research.activation_transfer.shape_transfer.feature_groups.data import OUT, dictionaries, load
from research.activation_transfer.shape_transfer.feature_groups.fit import greedy, gain_tables


def fit_global(source):
    dic = dictionaries(source)
    data = load(source, dic, "cal")
    groups = {}
    for name, level in (("global_functional", False), ("global_level", True)):
        normalized = []
        for g in gain_tables(data, dic, level):
            total = g.max(1).sum()
            if total > 0:
                normalized.append(g / total)
        groups[name] = greedy(np.vstack(normalized)).tolist()
    seen = (data["strengths"] > 0).sum(0)
    groups["global_prevalent"] = np.argsort(-seen, kind="stable")[:4].tolist()
    lists = {key:[cols for _ in dic.atoms] for key,cols in groups.items()}
    packed = dict(atoms=dic.atoms.tolist(), calibration_ids=data["ids"].tolist(), groups=lists)
    (OUT / (source + "_global_lists.json")).write_text(json.dumps(packed, indent=2) + "\n")
    print(source, groups, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["bolt", "timesfm"])
    torch.set_num_threads(2)
    fit_global(parser.parse_args().source)
