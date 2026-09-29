"""The planted forecaster: a mock whose internal concepts are known in advance.

Why this module exists (`ROADMAP.md` sec 38.1, K1). Every other mock is a
default-initialised network with no structure in its weights, so nothing in
this repo could show that the SAE -> ablation battery -> concept atlas ->
transfer -> shared-input-agreement chain *finds* a causal direction that is
known to exist, or *rejects* one known not to. The paper's causal counts were
validated only against the chain's own nulls. `mock_planted` is a network
built by hand so that the answer key exists: a designated **planted block**
writes known directions into the residual stream, and a head reads a known
linear function of them.

What it inherits. It is built from `mock._MockBlock`, so hooks, `token_patch`,
spans, attention patterns and tier derivation behave exactly as for
`mock_patch` (tier 3, by derivation: nothing here is declared). It is a NEW
adapter; no existing mock, stage or default changes.

The construction (all randomness is sha256-seeded from `construction_seed`,
never Python `hash()`):

* **Embedding.** Each 32-step patch of the per-series-normalised context is
  mapped to `N_FEATURES` hand-set statistics (patch mean, least-squares slope,
  residual sd, phase-free amplitude at the corpus's two most common periods,
  spike statistic, range, curvature), standardised on the calibration corpus,
  and written to the first `N_FEATURES` coordinates. The remaining coordinates
  are a random projection of the patch, restricted to the complement of the
  planted span.
* **Planted block** (`blocks.P`). For each planted object `k` it adds
  `amp * a_k(t) * d_k` to every token, where `a_k(t) = ReLU(w_k . f(t) - tau_k)`
  is a thresholded readout of that token's own features (so a concept is
  time-local and sparse), `tau_k` is calibrated on the corpus so that the
  concept fires on `target_fraction` in [0.12, 0.28] of series, and the `d_k`
  are unit vectors with pairwise cosine in [0, 0.3] (superposition, not an
  orthogonal basis).
* **Everything else is blind to the planted span.** The other blocks read and
  write only the orthogonal complement of `span(D)` and are scaled by 0.05
  (near identity). The planted coefficient of concept `k` at the head is
  therefore exactly `c_k = mean_t (u_k . h_t)` with `u_k` the dual basis
  (`u_j . d_k = delta_jk`), and removing `c_k d_k` from the residual at the
  planted block changes the forecast by exactly the planted component of `k`.
* **Head.** `forecast = last value + sd_ctx * (gamma * W_bg P_perp h_bar +
  sum_k beta_k c_k shape_kind(k))`. The shapes have unit RMS and map one to one
  onto battery channels: level (constant), trend (centred ramp), seasonal (a
  cosine with one cycle over the horizon, symmetric about the horizon centre so
  it is exactly orthogonal to the ramp in the FFT bin the battery reads),
  dispersion (an alternating +-1 pattern, which moves the point forecast's sd
  and scales the quantile band). The dose `s` multiplies every real concept's
  `beta_k`.
* **Effect size is defined, not tuned.** `beta_k` is set so the RMS planted
  forecast displacement on the concept's own top-8 firing series is
  `DOSE_UNIT_EFFECT * s` context standard deviations, with `DOSE_UNIT_EFFECT`
  the real models' median causal effect (0.0395 context sd, FINDINGS MN-15).
  Dose 1 therefore reads directly as "the size of a real feature's effect".
* **Known negatives.** `input_only` decoys are written exactly like concepts but
  `beta = 0`: correlated with the input, causally inert. `sub_null` decoys have
  a fixed RMS effect of `SUB_NULL_FRACTION * DOSE_UNIT_EFFECT`, independent of
  dose, chosen from the calibration run recorded in `run_known_answer.py`'s
  output to sit below the random-direction null's p95.
* **The pair.** `plant_set` in {"A", "B"} gives the two members of a pair that
  share a vocabulary: `shared` (same readout, same effect), `convergent`
  (different readout, same effect), `opposite` (same readout, opposite sign;
  level and trend only, because a seasonal or dispersion effect has no sign),
  and `unique` (model A only).

`MockPlantedAdapter.manifest()` is the ground truth: per planted object its id,
class, kind, `d_k`, readout weights and threshold, effect channel, sign, beta,
dose and the exact series-level activation `a_k(x)` on every corpus series. The
scorer (`analysis/known_answer.py`) reads only that and the run's artifacts.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

from .mock import _MockAdapterBase, _MockBlock, _normal_ppf

KINDS = ("level", "trend", "seasonal", "dispersion")
FEATURE_NAMES = ("mean", "slope", "resid_sd", "amp_p1", "amp_p2", "spike", "range", "curvature")
N_FEATURES = len(FEATURE_NAMES)
DOSE_UNIT_EFFECT = 0.0395
SUB_NULL_FRACTION = 0.1
AMPLITUDE = 10.0
EMBED_SCALE = 0.03
TOP_ROWS = 8
COSINE_RANGE = (0.0, 0.3)
FIRING_RANGE = (0.12, 0.28)
_MAX_CALIBRATION_ROWS = 2000
_CALIBRATION_CACHE: dict = {}


def _seed(*parts: Any) -> int:
    """A stable 63-bit seed from `parts` (sha256, never Python `hash()`)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def concept_table() -> list:
    """Every planted object the pair vocabulary defines, model-independent.

    Each row: `id`, `cls` (shared, convergent, opposite, unique, decoy_input_only,
    decoy_sub_null), `kind` (an effect channel, or None for an input-only
    decoy), `roles` (which models carry it), and `sign` per role.

    The vocabulary is SIGNATURE-MAJOR: every effect signature (kind and sign)
    belongs to exactly one class, with three items each. The atlas clusters
    causal features by their effect vector, so an effect signature is what an
    atlas concept is; giving one signature to two classes (a shared and a
    unique level-down concept, say) would make the class of the atlas concept
    it lands in undefined. Signatures: `shared` trend-down; `convergent`
    dispersion-up and trend-up; `opposite` level-up in A and level-down in B
    (so each is a single-model atlas concept whose INPUTS coincide); `unique`
    seasonal-up in A only. Seasonal and dispersion effects are magnitudes, so
    their sign is always +1 and neither has an opposite-effect twin.
    """
    rows = []
    for i in (1, 2, 3):
        rows.append({"id": f"shared_trend_{i}", "cls": "shared", "kind": "trend",
                     "roles": ("A", "B"), "sign": {"A": -1, "B": -1}})
    for i in (1, 2, 3):
        rows.append({"id": f"convergent_dispersion_{i}", "cls": "convergent", "kind": "dispersion",
                     "roles": ("A", "B"), "sign": {"A": 1, "B": 1}})
    for i in (1, 2, 3):
        rows.append({"id": f"convergent_trend_{i}", "cls": "convergent", "kind": "trend",
                     "roles": ("A", "B"), "sign": {"A": 1, "B": 1}})
    for i in (1, 2, 3):
        rows.append({"id": f"opposite_level_{i}", "cls": "opposite", "kind": "level",
                     "roles": ("A", "B"), "sign": {"A": 1, "B": -1}})
    for i in (1, 2, 3):
        rows.append({"id": f"unique_seasonal_{i}", "cls": "unique", "kind": "seasonal",
                     "roles": ("A",), "sign": {"A": 1}})
    for i in (1, 2, 3):
        rows.append({"id": f"inert_{i}", "cls": "decoy_input_only", "kind": None,
                     "roles": ("A", "B"), "sign": {"A": 0, "B": 0}})
    for i, kind in ((1, "level"), (2, "trend"), (3, "seasonal")):
        rows.append({"id": f"subnull_{i}", "cls": "decoy_sub_null", "kind": kind,
                     "roles": ("A", "B"), "sign": {"A": 1, "B": 1}})
    return rows


_INERT_READOUTS = {"inert_1": {"slope": 1.0}, "inert_2": {"amp_p1": 1.0}, "inert_3": {"resid_sd": 1.0}}


def _readout_key(row: dict, role: str) -> str:
    """Shared, opposite and decoy objects read the same input in both models;
    a convergent object reads a different input in each."""
    return f"{row['id']}@{role}" if row["cls"] == "convergent" else row["id"]


def _draw_readouts(rows: list, construction_seed: int) -> dict:
    """`{readout_key: weight vector [N_FEATURES]}`, distinct across keys.

    A random readout reads two features with random signs, unit norm. The
    input-only decoys read a single feature chosen to correlate with a
    ground-truth field (slope -> trend, amplitude -> seasonality, residual sd
    -> noise), so a correlational probe has something to find.
    """
    combos = [(i, j, si, sj) for i in range(N_FEATURES) for j in range(i + 1, N_FEATURES)
              for si in (1, -1) for sj in (1, -1)]
    rng = np.random.default_rng(_seed(construction_seed, "readouts"))
    keys = sorted({_readout_key(r, role) for r in rows for role in r["roles"]
                   if r["cls"] != "decoy_input_only"})
    picks = rng.choice(len(combos), size=len(keys), replace=False)
    out = {}
    for key, p in zip(keys, picks):
        i, j, si, sj = combos[int(p)]
        w = np.zeros(N_FEATURES)
        w[i], w[j] = si, sj
        out[key] = w / np.linalg.norm(w)
    for key, spec in _INERT_READOUTS.items():
        w = np.zeros(N_FEATURES)
        for name, val in spec.items():
            w[FEATURE_NAMES.index(name)] = val
        out[key] = w
    return out


def _draw_directions(ids: list, dim: int, construction_seed: int, role: str) -> dict:
    """Unit directions in the non-feature coordinates with pairwise cosine in
    `COSINE_RANGE`, by sequential rejection sampling around a common axis."""
    n_free = dim - N_FEATURES
    rng = np.random.default_rng(_seed(construction_seed, "directions", role))
    common = rng.normal(size=n_free)
    common /= np.linalg.norm(common)
    chosen: dict = {}
    for cid in ids:
        for _attempt in range(200000):
            rho = rng.uniform(0.08, 0.2)
            e = rng.normal(size=n_free)
            e -= (e @ common) * common
            e /= np.linalg.norm(e)
            v = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * e
            v /= np.linalg.norm(v)
            if all(COSINE_RANGE[0] <= float(v @ u) <= COSINE_RANGE[1] for u in chosen.values()):
                chosen[cid] = v
                break
        else:
            raise RuntimeError(f"could not place direction '{cid}' with cosine in {COSINE_RANGE}")
    full = {}
    for cid, v in chosen.items():
        d = np.zeros(dim)
        d[N_FEATURES:] = v
        full[cid] = d
    return full


def patch_features(xn_patches: torch.Tensor, periods: tuple) -> torch.Tensor:
    """`[N, patch] -> [N, N_FEATURES]`: hand-set statistics of a normalised patch.

    Deterministic, differentiable-free and shared by the network, the
    calibration and the manifest, so all three compute one function.
    """
    n, p = xn_patches.shape
    x = xn_patches.double()
    t = torch.arange(p, dtype=torch.float64) - (p - 1) / 2.0
    mean = x.mean(dim=1)
    slope = (x * t).sum(dim=1) / (t ** 2).sum()
    resid = x - mean[:, None] - slope[:, None] * t
    resid_sd = resid.pow(2).mean(dim=1).sqrt()
    idx = torch.arange(p, dtype=torch.float64)
    amps = []
    for period in periods:
        ph = 2.0 * np.pi * idx / float(period)
        c = (resid * torch.cos(ph)).mean(dim=1) * 2.0
        s = (resid * torch.sin(ph)).mean(dim=1) * 2.0
        amps.append((c ** 2 + s ** 2).sqrt())
    med = x.median(dim=1).values
    mad = (x - med[:, None]).abs().median(dim=1).values
    spike = (x - med[:, None]).abs().max(dim=1).values / (mad + 1e-6)
    rng_ = x.max(dim=1).values - x.min(dim=1).values
    q = t ** 2 - (t ** 2).mean()
    curv = (resid * q).sum(dim=1) / (q ** 2).sum()
    return torch.stack([mean, slope, resid_sd, amps[0], amps[1], spike, rng_, curv], dim=1)


def normalise_contexts(x: torch.Tensor) -> tuple:
    """Per-series normalisation: `(xn, mean, sd)`, `sd` floored by 1e-6."""
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, unbiased=False, keepdim=True) + 1e-6
    return (x - mu) / sd, mu, sd


def contexts_to_features(contexts: np.ndarray, patch: int, periods: tuple) -> torch.Tensor:
    """`[B, T] -> [B, T // patch, N_FEATURES]` raw (unstandardised) features."""
    x = torch.from_numpy(np.ascontiguousarray(contexts)).double()
    xn, _, _ = normalise_contexts(x)
    b, t = xn.shape
    feats = patch_features(xn.reshape(b * (t // patch), patch), periods)
    return feats.reshape(b, t // patch, N_FEATURES)


def planted_activations(f_z: torch.Tensor, W: torch.Tensor, tau: torch.Tensor,
                        amp: float) -> torch.Tensor:
    """`amp * ReLU(f_z W^T - tau)`, `[B, n_tokens, K]`: the one definition of a
    planted coefficient, used by the network and by the manifest alike."""
    return amp * torch.relu(f_z @ W.t() - tau)


def build_spec(construction_seed: int, dose: float, plant_set: str, dim: int, patch: int,
               contexts: np.ndarray, series_ids: list, periods: tuple, planted_block: int,
               families: np.ndarray | None = None) -> dict:
    """Everything that defines one member of the pair, calibrated on `contexts`.

    Calibration (feature standardisation, per-concept threshold, per-concept
    beta) uses at most `_MAX_CALIBRATION_ROWS` rows, chosen with a stratified
    `sample_rows` when the corpus is larger; the series-level activations in
    the manifest are computed on every row.
    """
    from ..utils import sample_rows
    if plant_set not in ("A", "B"):
        raise ValueError(f"plant_set must be 'A' or 'B', got {plant_set!r}")
    role = plant_set
    rows = [r for r in concept_table() if role in r["roles"]]
    all_rows = concept_table()
    readouts = _draw_readouts(all_rows, construction_seed)
    directions = _draw_directions([r["id"] for r in rows], dim, construction_seed, role)

    raw = contexts_to_features(contexts, patch, periods)
    n = raw.shape[0]
    cal = (sample_rows(n, _MAX_CALIBRATION_ROWS, _seed(construction_seed, "cal") % (2 ** 31),
                       strata=families) if n > _MAX_CALIBRATION_ROWS else np.arange(n))
    flat = raw[cal].reshape(-1, N_FEATURES)
    feat_mean, feat_std = flat.mean(dim=0), flat.std(dim=0) + 1e-9
    f_z = (raw - feat_mean) / feat_std

    rng_frac = np.random.default_rng(_seed(construction_seed, "firing"))
    targets = {r["id"]: float(rng_frac.uniform(*FIRING_RANGE)) for r in all_rows}

    concepts = []
    for r in rows:
        w = torch.from_numpy(readouts[_readout_key(r, role)]).double()
        z = f_z[cal] @ w
        series_max = z.max(dim=1).values.numpy()
        tau = float(np.quantile(series_max, 1.0 - targets[r["id"]]))
        a_all = planted_activations(f_z, w[None, :], torch.tensor([tau], dtype=torch.float64),
                                    AMPLITUDE)[..., 0]
        a_series = a_all.mean(dim=1).numpy()
        top = np.sort(a_series)[::-1][:TOP_ROWS]
        top_mean = float(top.mean())
        if r["cls"] == "decoy_input_only":
            unit = 0.0
        elif r["cls"] == "decoy_sub_null":
            unit = SUB_NULL_FRACTION * DOSE_UNIT_EFFECT
        else:
            unit = float(dose) * DOSE_UNIT_EFFECT
        beta = (unit / top_mean if top_mean > 0 else 0.0) * r["sign"][role]
        concepts.append({
            "id": r["id"], "cls": r["cls"], "kind": r["kind"], "sign": int(r["sign"][role]),
            "roles": list(r["roles"]), "target_fraction": targets[r["id"]],
            "readout_weights": w.numpy().tolist(), "readout_threshold": tau,
            "direction": directions[r["id"]].tolist(), "beta": float(beta),
            "rms_effect_on_top_series_ctx_sd": float(abs(beta) * top_mean),
            "firing_fraction": float(np.mean(a_series > 0.0)),
            "series_activation": a_series.tolist(),
        })
    return {
        "construction_seed": int(construction_seed), "dose": float(dose), "role": role,
        "plant_set": plant_set, "planted_block": int(planted_block), "dim": int(dim),
        "patch": int(patch), "n_features": N_FEATURES, "feature_names": list(FEATURE_NAMES),
        "periods": [float(p) for p in periods], "amplitude": AMPLITUDE,
        "dose_unit_effect_ctx_sd": DOSE_UNIT_EFFECT, "sub_null_fraction": SUB_NULL_FRACTION,
        "feature_mean": feat_mean.numpy().tolist(), "feature_std": feat_std.numpy().tolist(),
        "series_ids": [str(s) for s in series_ids], "concepts": concepts,
    }


def _shapes(horizon: int) -> dict:
    """Unit-RMS forecast shapes, one per effect channel."""
    t = np.arange(horizon, dtype=np.float64)
    centred = t - (horizon - 1) / 2.0
    ramp = centred / np.sqrt((centred ** 2).mean())
    cosine = np.sqrt(2.0) * np.cos(2.0 * np.pi * centred / horizon)
    alt = np.where(np.arange(horizon) % 2 == 0, 1.0, -1.0)
    return {"level": np.ones(horizon), "trend": ramp, "seasonal": cosine, "dispersion": alt}


class _PlantedBlock(_MockBlock):
    """A near-identity residual block that only reads and writes the
    complement of the planted span; the planted block additionally writes."""

    def __init__(self, dim: int, n_heads: int, p_perp: torch.Tensor, gen: torch.Generator):
        super().__init__(dim, n_heads)
        self.register_buffer("p_perp", p_perp.clone())
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(0.05 * torch.randn(p.shape, generator=gen) / np.sqrt(p.shape[-1])
                        if p.dim() == 2 else torch.zeros_like(p))
        self.stash: dict | None = None
        self.write: nn.Module | None = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(h @ self.p_perp) @ self.p_perp
        h = h + self.mlp(h @ self.p_perp) @ self.p_perp
        if self.write is not None:
            h = h + self.write(self.stash)
        return h


class _PlantedWrite(nn.Module):
    """Adds `sum_k amp a_k(t) d_k` to every token, from the input's own features."""

    def __init__(self, W: torch.Tensor, tau: torch.Tensor, D: torch.Tensor):
        super().__init__()
        self.register_buffer("W", W)
        self.register_buffer("tau", tau)
        self.register_buffer("D", D)

    def forward(self, stash: dict) -> torch.Tensor:
        a = planted_activations(stash["f_z"], self.W, self.tau, AMPLITUDE)
        stash["a"] = a
        return (a @ self.D.t()).float()


class _PlantedNet(nn.Module):
    """Feature-carrying embedding, planted block, near-identity blocks, known head."""

    def __init__(self, spec: dict, n_layers: int, n_heads: int, horizon: int, seed: int):
        super().__init__()
        dim, patch = spec["dim"], spec["patch"]
        self.patch, self.dim, self.horizon = patch, dim, horizon
        self.periods = tuple(spec["periods"])
        concepts = spec["concepts"]
        self.concepts = concepts
        D = torch.tensor(np.stack([c["direction"] for c in concepts], axis=1), dtype=torch.float64)
        G = D.t() @ D
        U = torch.linalg.solve(G, D.t())
        p_perp = torch.eye(dim, dtype=torch.float64) - D @ U
        self.register_buffer("feat_mean", torch.tensor(spec["feature_mean"], dtype=torch.float64))
        self.register_buffer("feat_std", torch.tensor(spec["feature_std"], dtype=torch.float64))
        self.register_buffer("U", U)
        self.register_buffer("p_perp64", p_perp)
        gen = torch.Generator().manual_seed(seed)
        R = torch.randn(patch, dim, generator=gen) * EMBED_SCALE
        R[:, :N_FEATURES] = 0.0
        R = R @ p_perp.float()
        self.register_buffer("embed_R", R)
        self.blocks = nn.ModuleList(
            _PlantedBlock(dim, n_heads, p_perp.float(), gen) for _ in range(n_layers))
        W = torch.tensor(np.stack([c["readout_weights"] for c in concepts]), dtype=torch.float64)
        tau = torch.tensor([c["readout_threshold"] for c in concepts], dtype=torch.float64)
        self.stash: dict = {}
        planted = self.blocks[spec["planted_block"]]
        planted.stash = self.stash
        planted.write = _PlantedWrite(W, tau, D)
        self.register_buffer("W_bg", torch.randn(horizon, dim, generator=gen, dtype=torch.float64)
                             / np.sqrt(dim))
        self.gamma = 0.05
        shapes = _shapes(horizon)
        self.register_buffer("shape_mat", torch.tensor(
            np.stack([shapes[c["kind"]] if c["kind"] else np.zeros(horizon) for c in concepts]),
            dtype=torch.float64))
        self.register_buffer("beta", torch.tensor([c["beta"] for c in concepts], dtype=torch.float64))
        self.register_buffer("is_dispersion", torch.tensor(
            [1.0 if c["kind"] == "dispersion" else 0.0 for c in concepts], dtype=torch.float64))

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        xn, _, _ = normalise_contexts(x.double())
        n_tok = t // self.patch
        patches = xn.reshape(b * n_tok, self.patch)
        f = patch_features(patches, self.periods)
        f_z = ((f - self.feat_mean) / self.feat_std).reshape(b, n_tok, N_FEATURES)
        self.stash["f_z"] = f_z
        h = patches.float() @ self.embed_R
        h = h.reshape(b, n_tok, self.dim)
        h[..., :N_FEATURES] = f_z.float()
        return h

    def forecast(self, x: torch.Tensor) -> tuple:
        """`[B, T] -> (point [B, H], quantile_scale [B])`, all blocks and the head."""
        h = self._embed(x)
        for block in self.blocks:
            h = block(h)
        hbar = h.double().mean(dim=1)
        coef = hbar @ self.U.t()
        bg = (hbar @ self.p_perp64) @ self.W_bg.t()
        contrib = (coef * self.beta) @ self.shape_mat
        xd = x.double()
        sd = xd.std(dim=1, unbiased=False, keepdim=True) + 1e-6
        point = xd[:, -1:] + sd * (self.gamma * bg + contrib)
        qscale = torch.exp((coef * self.beta * self.is_dispersion).sum(dim=1))
        return point.float(), qscale.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forecast(x)[0]


class MockPlantedAdapter(_MockAdapterBase):
    """Tier-3 patch-tokenizing mock with planted, ground-truth-labelled concepts.

    Config kwargs (all optional): `plant_set` ("A" or "B", default "A"),
    `construction_seed` (int, default 0), `dose` (float, default 1.0),
    `periods` (two periods for the amplitude features, default [24, 50], the
    two most common periods of the synthetic known-answer corpus),
    `planted_block` (default 2). The calibration corpus is the run's own data
    (`data:`), so the firing thresholds are calibrated on exactly the series
    the pipeline will feed the model.
    """

    patch, dim, n_layers, n_heads, seed = 32, 64, 5, 2, 11

    def _kwarg(self, name: str, default: Any) -> Any:
        return self.cfg.kwargs.get(name, default)

    def load(self) -> None:
        from ..data import load_benchmark
        plant_set = str(self._kwarg("plant_set", "A"))
        cseed = int(self._kwarg("construction_seed", 0))
        dose = float(self._kwarg("dose", 1.0))
        periods = tuple(float(p) for p in self._kwarg("periods", (24, 50)))
        block = int(self._kwarg("planted_block", 2))
        d = self.data_cfg
        key = (plant_set, cseed, dose, periods, block, d.source, d.path, d.context_len,
               d.horizon, d.max_series, d.smoke_series_per_family, d.family_key)
        if key not in _CALIBRATION_CACHE:
            data = load_benchmark(d)
            _CALIBRATION_CACHE[key] = build_spec(
                cseed, dose, plant_set, self.dim, self.patch, data.contexts(),
                data.meta["series_id"].tolist(), periods, block, families=data.families)
        self._spec = _CALIBRATION_CACHE[key]
        net_seed = _seed(cseed, "net", plant_set)
        if self.cfg.random_init:
            net_seed += self._RANDOM_INIT_SEED_OFFSET
        self._net = _PlantedNet(self._spec, self.n_layers, self.n_heads,
                                self.data_cfg.horizon, net_seed % (2 ** 31)).to(self.device)

    def manifest(self) -> dict:
        """The ground truth for this member of the pair (JSON-serialisable)."""
        self.ensure_loaded()
        return {**self._spec, "model": self.name, "planted_layer": f"blocks.{self._spec['planted_block']}"}

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Deterministic point forecast; the quantile band is `sd_ctx * ppf(q)`
        scaled by the planted dispersion concepts' `exp(sum beta c)`."""
        with torch.no_grad():
            point, qscale = self._net.forecast(self.prepare(contexts))
        point = point.cpu().numpy()[:, :horizon]
        scale = contexts.std(axis=1, keepdims=True) * qscale.cpu().numpy()[:, None] + 1e-6
        offsets = np.array([_normal_ppf(q) for q in quantiles], dtype=np.float32)
        q = point[:, :, None] + offsets[None, None, :] * scale[:, :, None]
        return {"point": point, "quantiles": q.astype(np.float32)}
