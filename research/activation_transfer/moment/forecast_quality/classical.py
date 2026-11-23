"""History-only classical forecasts with an inner walk-forward selector."""
import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT, SPECS, hash_array


def seasonal(history, horizon, period, cycles):
    """Predict the next phase from the latest one or several complete seasonal cycles."""
    complete = min(cycles, len(history) // period)
    if complete < 1:
        raise ValueError("Insufficient complete seasonal cycle")
    past = history[-complete*period:].reshape(complete, period)
    phase = np.median(past, axis=0)
    return phase[np.arange(horizon) % period]


def candidates(history, horizon, periods):
    steps = np.arange(1, horizon + 1)
    runs = {"last": np.repeat(history[-1], horizon),
            "drift": history[-1] + steps * ((history[-1] - history[0]) / (len(history) - 1))}
    for j, period in enumerate(periods, 1):
        runs[f"season{j}"] = seasonal(history, horizon, period, 1)
        runs[f"median{j}"] = seasonal(history, horizon, period, 4)
    return runs


def predict_one(history, horizon, periods):
    complete = candidates(history, horizon, periods)
    losses = {k: 0. for k in complete}
    for stop in range(len(history) - 8*horizon, len(history), horizon):
        train, actual = history[:stop], history[stop:stop+horizon]
        for name, pred in candidates(train, horizon, periods).items():
            losses[name] += np.abs(actual - pred).mean()
    winner = min(losses, key=losses.get)
    return {**complete, "history_select": complete[winner]}, winner


def main():
    with np.load(OUT / "panel.npz") as archive:
        contexts = archive["context"]
        ids = archive["id"]
        domains = archive["domain"]
    arrays = dict(ids=ids, context_sha256=hash_array(contexts))
    for h in (8, 16):
        choices = np.full(contexts.shape[:2], "", dtype="U16")
        for i, windows in enumerate(contexts):
            for j, history in enumerate(windows):
                results, choices[i, j] = predict_one(history, h, SPECS[domains[i]][1])
                for name, pred in results.items():
                    key = f"{name}_{h}"
                    if key not in arrays:
                        arrays[key] = np.full((*contexts.shape[:2], h), np.nan)
                    arrays[key][i, j] = pred
        arrays[f"selected_{h}"] = choices
    np.savez_compressed(OUT / "classical.npz", **arrays)


if __name__ == "__main__":
    main()
