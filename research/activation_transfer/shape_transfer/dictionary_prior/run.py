"""Blind a one-atom matcher to donor identity and execute priors and residual corrections."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.geometry import histories
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as FULL
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.self_transfer.run import HERE as SELF, SCRATCH, geometry
from research.activation_transfer.shape_transfer.sparse_stability.run import active_codes
from research.activation_transfer.shape_transfer.sparse_writes.run import weight_leads
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ARMS = (
    "dense", "natural", "natural_corrected", "attenuated", "attenuated_corrected",
    "wrong1", "wrong1_corrected", "wrong2", "wrong2_corrected",
)


def blind_match(responses, goal, weight):
    """Rank candidate natural removals with only eight predictions and the observed eight-vector."""
    observed = weight_leads(goal, weight)
    predicted = weight_leads(responses, weight)
    ranks = np.argsort(np.square(predicted - observed).sum(1), kind="stable")

    energy = np.square(predicted).sum(1)
    fractions = np.clip((predicted @ observed) / np.maximum(energy, 1e-24), 0., 1.)
    attenuated_loss = np.square(fractions[:, None] * predicted - observed).sum(1)
    attenuated_ranks = np.argsort(attenuated_loss, kind="stable")
    level_ranks = np.argsort(np.abs(responses.mean(1) - goal.mean()), kind="stable")
    return ranks, attenuated_ranks, fractions, level_ranks


def inverse(jac, weight):
    """Return the early weighted Jacobian and its damped right inverse."""
    jacobian = weight_leads(jac.T, weight).T
    gram = jacobian @ jacobian.T
    damping = .001 * np.linalg.eigvalsh(gram)[-1]
    if damping <= 0:
        raise ValueError("Degenerate receiver derivative")
    return jacobian, np.linalg.solve(gram + damping * np.eye(8), jacobian).T


def correct(prior, jacobian, lift, weighted_goal):
    return prior + lift @ (weighted_goal - jacobian @ prior)


def make_directions(decoder, doses, jac, goal, weight, jacobian, lift):
    """Use the receiver's 32 active rows, coefficients, early derivative and the measured donor goal."""
    signatures = (-doses[:, None] * decoder) @ jac.T
    ranks, attenuated_ranks, fractions, level_ranks = blind_match(signatures, goal, weight)
    best = ranks[0]
    attenuated_best = attenuated_ranks[0]
    vector = -doses[best] * decoder[best]
    shrunk = -fractions[attenuated_best] * doses[attenuated_best] * decoder[attenuated_best]

    priors = {"natural": vector, "attenuated": shrunk}
    for number, offset in enumerate((7, 17), 1):
        alternative = decoder[(best + offset) % len(decoder)]
        priors[f"wrong{number}"] = -np.linalg.norm(vector) * alternative / np.linalg.norm(alternative)
    weighted_goal = weight_leads(goal, weight)
    moves = {"dense": lift @ weighted_goal}
    for name, prior in priors.items():
        moves[name] = prior
        moves[name + "_corrected"] = correct(prior, jacobian, lift, weighted_goal)
    return moves, (ranks, attenuated_ranks, level_ranks), fractions[attenuated_best]


def load_target(target):
    with np.load(geometry(target, SCRATCH)) as archive:
        keep = np.isin(archive["split"], ["daily", "extra_daily"])
        ids = archive["ids"][keep]
        split = archive["split"][keep]
        params = dict(
            jac=archive["jac"][keep].astype(float), states=archive["states"][keep],
            base=archive["base"][keep], radius=float(archive["radius"]), positions=archive["positions"],
        )
    all_ids, x, all_split = histories()
    active = np.isin(all_split, ["daily", "extra_daily"])
    np.testing.assert_array_equal(all_ids[active], ids)
    np.testing.assert_array_equal(all_split[active], split)
    sae = load_sae_checkpoint(str(dictionary_path(target)))
    support, codes = active_codes(sae, params["states"])
    return dict(
        ids=ids, split=split, x=x[active], support=support,
        doses=np.take_along_axis(codes, support, axis=1),
        decoder=sae.W_dec.detach().numpy().astype(float), **params,
    )


def run(source, target):
    data = load_target(target)
    with np.load(SELF / "results" / f"{source}_to_{source}.npz") as archive:
        keep = np.isin(archive["split"], ["daily", "extra_daily"])
        np.testing.assert_array_equal(archive["ids"][keep], data["ids"])
        live = archive["active"][keep]
        atoms = archive["atoms"]
        observed = archive["native"][keep]
    with np.load(FULL / "results" / (source + ".npz")) as archive:
        weight = float(archive["shape_weight"])

    early = observed[:, :, :8].copy()
    arms = list(ARMS) + (["known_capped"] if source == target else [])
    out = dict(
        ids=data["ids"], split=data["split"], atoms=atoms, active=live,
        receiver_support=data["support"], receiver_strength=data["doses"],
        radius=data["radius"], shape_weight=weight,
    )
    rank_keys = ("natural_rank", "attenuated_rank", "level_rank")
    for key in rank_keys:
        out[key] = np.full((*live.shape, 32), -1, dtype=np.int16)
    out["attenuation_fraction"] = np.zeros(live.shape, dtype="f4")
    for arm in arms:
        out[arm + "_effect"] = np.zeros((*live.shape, 16), dtype="f4")
        out[arm + "_norm"] = np.zeros(live.shape, dtype="f4")
        out[arm + "_clipped"] = np.zeros(live.shape, dtype=bool)

    adapter = adapter_for(target)
    layer = MODELS[target][-1]
    selector = last_window_slice(adapter, data["positions"])
    total = 0
    for start in range(0, len(live), 24):
        # Forwards may be shared only for exactly identical writes on the same history.
        unique = {}
        for row in range(start, min(start + 24, len(live))):
            support = data["support"][row]
            jac = data["jac"][row]
            decoder = data["decoder"][support]
            doses = data["doses"][row].astype(float)
            jacobian, lift = inverse(jac, weight)
            for query in np.flatnonzero(live[row]):
                moves, rankings, fraction = make_directions(
                    decoder, doses, jac, early[row, query], weight, jacobian, lift,
                )
                for key, ranks in zip(rank_keys, rankings):
                    out[key][row, query] = ranks
                out["attenuation_fraction"][row, query] = fraction
                if source == target:
                    # The correct ID enters only this separate oracle after every blind ranking and move exists.
                    actual = np.flatnonzero(support == atoms[query])
                    if len(actual) != 1:
                        raise ValueError("True firing source absent from self receiver")
                    moves["known_capped"] = -doses[actual[0]] * decoder[actual[0]]
                for arm, vector in moves.items():
                    norm = float(np.linalg.norm(vector))
                    out[arm + "_norm"][row, query] = norm
                    scale = min(1., .2 * data["radius"] / max(norm, 1e-20))
                    out[arm + "_clipped"][row, query] = scale < 1
                    if norm > 0.:
                        key = (row, (vector * scale).astype("f4").tobytes())
                        unique.setdefault(key, []).append((arm, row, query))

        keys = list(unique)
        for offset in range(0, len(keys), 32):
            subset = keys[offset:offset + 32]
            rows = np.array([row for row, _ in subset])
            delta = np.stack([np.frombuffer(vector, dtype="f4") for _, vector in subset])
            patched = forecast16(
                adapter, layer, data["x"][rows], data["states"][rows], delta, selector, 32,
            ) - data["base"][rows]
            for key, wave in zip(subset, patched):
                for arm, row, query in unique[key]:
                    out[arm + "_effect"][row, query] = wave
        total += len(keys)
        if (start // 24) % 4 == 3:
            print(source, target, min(start + 24, len(live)), "histories done", flush=True)

    out["measured_source"] = observed
    out["unique_physical_writes"] = total
    if source == target:
        uncapped = live & ~out["known_capped_clipped"]
        np.testing.assert_allclose(out["known_capped_effect"][uncapped], observed[uncapped], atol=1e-4, rtol=.001)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{source}_to_{target}.npz", **out)
    print(source, target, total, "physical writes complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=MODELS)
    parser.add_argument("target", choices=MODELS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.source, args.target)
