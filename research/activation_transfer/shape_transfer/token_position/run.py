"""Direct steering maps at the same block using time-aligned observed patches."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.forecast_requests.run import (
    ALIGN, CAL, CUES, DATA_ROOT, FIT, N_ALIGN, N_FIT, RANK, TEST, adapter_for, data, desired,
    learn_directions, scale_factors,
)
from research.activation_transfer.forecast_requests.units.run import normalize_dose
from tsfm_lens.steering import SteeringModel, solve_direction
from tsfm_lens.steering.model import InterventionSite, _get, _replace

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "forecast_requests"
CACHE = Path("/tmp/tsfm_forecast_requests_20261120")


def observed_positions(spans, live_data, count):
    if len(spans) != len(live_data) or count not in (1, 2) or len(spans) < count:
        raise ValueError("Invalid token registration")
    covered = spans[-count:]
    if covered[-1, 1] != 192 or not np.array_equal(covered[1:, 0], covered[:-1, 1]):
        raise ValueError("Selected tokens do not cover a contiguous final window")
    return live_data[-count:]


class Window:
    def __init__(self, name, count):
        adapter = adapter_for(name)
        self.model = SteeringModel.from_adapter(adapter, quantiles=[.5])
        record = json.loads((OLD / "results" / (name + ".json")).read_text(encoding="utf-8"))
        self.site = InterventionSite(**record["source_fits"][name]["site"][0])
        spans = adapter.token_time_spans()
        live_data = np.arange(self.site.shape[0])[adapter.token_slice(self.site.shape[0])]
        self.positions = observed_positions(spans, live_data, count)
        if name == "bolt" and (self.positions[-1] != self.site.shape[0]-2 or not adapter._use_reg):
            raise ValueError("Expected one trailing Bolt [REG] token")
        self.spans = spans[-count:].tolist()
        self.block = self.model.module.get_submodule(self.site.module)

    def forecast(self, x, displacement=None):
        captured = []
        def edit(module, args, kwargs):
            packed = dict(args=args, kwargs=kwargs)
            state = _get(packed, self.site.path)
            if state.shape[0] != len(x) or list(state.shape[1:]) != self.site.shape:
                raise ValueError("Unexpected block geometry")
            captured.append(state[:, self.positions].mean(dim=1).detach().numpy().copy())
            if displacement is None:
                return
            changed = state.clone()
            changed[:, self.positions] += torch.as_tensor(displacement, dtype=state.dtype)[:,None,:]
            packed = _replace(packed, self.site.path, changed)
            return packed["args"], packed["kwargs"]
        with self.block.register_forward_pre_hook(edit, with_kwargs=True):
            y = self.model.forecast(x, 8)[...,0]
        if len(captured) != 1:
            raise ValueError("Window did not execute once")
        return y, captured[0]

    def effects(self, x, base, directions, batch):
        predictions = []
        for start in range(0, len(x)*len(directions), batch):
            indices = np.arange(start, min(start+batch, len(x)*len(directions)))
            rows, arms = np.divmod(indices, len(directions))
            predictions.append(self.forecast(x[rows], directions[arms])[0] - base[rows])
        return np.concatenate(predictions).reshape(len(x),len(directions),8)


def prepare(name, count, x, ids, batch):
    window = Window(name, count)
    clean = [window.forecast(x[i:i+batch]) for i in range(0,len(x),batch)]
    base, states = [np.concatenate(a) for a in zip(*clean)]
    if name == "timesfm":
        with np.load(CACHE / "timesfm_fit.npz") as f:
            np.testing.assert_array_equal(f["ids"],ids)
            np.testing.assert_allclose(states, f["s0"], rtol=2e-4, atol=1e-4)
            native = f["d0"]
    else:
        centered = states[ALIGN].astype(float) - states[ALIGN].mean(axis=0, dtype=float)
        basis = np.linalg.svd(centered, full_matrices=False)[2][:RANK]
        radius = np.median(np.linalg.norm(states[ALIGN].astype(float),axis=1))
        step = .01*radius
        probes = np.concatenate((step*basis,-step*basis))
        observed = window.effects(x[FIT], base[FIT], probes, batch)
        jac = ((observed[:,:RANK] - observed[:,RANK:])/(2*step)).transpose(0,2,1).reshape(-1,RANK)
        ridge = .01*np.linalg.svd(jac, compute_uv=False)[0]**2
        native = np.stack([solve_direction(jac,np.tile(wave,N_FIT),ridge=ridge,max_norm=.2*radius).delta @ basis
                           for wave in desired()])
    return window, base, states, native


def run(count, out, batch):
    ids,x = data(DATA_ROOT)
    models = {name:prepare(name, count if name == "bolt" else 1, x, ids, batch) for name in ("bolt","timesfm")}
    rng = np.random.default_rng(202611207)
    perms = [rng.permutation(N_ALIGN) for _ in range(4)]
    arrays = dict(ids=ids[TEST], requests=desired())
    for source,target in (("bolt","timesfm"),("timesfm","bolt")):
        window,base,states,native = models[target]
        _,_,source_states,source_native = models[source]
        mapped, shams = learn_directions(source_states[ALIGN],states[ALIGN],source_native,perms)
        if len(CUES) != 3:
            raise ValueError("Original cue set changed")
        units,_ = normalize_dose(mapped,shams,native)
        shape = units.shape
        cal = window.effects(x[CAL],base[CAL],units.reshape(-1,shape[-1]),batch).reshape(-1,*shape[:2],8)
        dose = np.stack([scale_factors(cal[:,:,a],desired()) for a in range(shape[1])],axis=1)
        edits = (units*dose[:,:,None]).reshape(-1,shape[-1])
        live = window.effects(x[TEST],base[TEST],edits,batch).reshape(-1,*shape[:2],8)
        live[:,dose==0] = 0.
        normal = window.effects(x[TEST],base[TEST],native,batch)
        if target == "timesfm":
            with np.load(OLD / "results" / "timesfm.npz") as f:
                np.testing.assert_allclose(normal,f["0_native"],atol=3e-4,rtol=.002)
        for k,val in dict(live=live,native=normal,dose=dose,spans=window.spans,positions=window.positions,
                          native_norm=np.linalg.norm(native,axis=1),mapped_norm=np.linalg.norm(mapped,axis=1)).items():
            arrays[target+"_"+k] = val
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out / f"bolt_{count}_patches.npz",**arrays)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bolt_patches",type=int,choices=[1,2])
    p.add_argument("--out",type=Path,default=HERE / "results")
    p.add_argument("--batch",type=int,default=32)
    a = p.parse_args()
    torch.set_num_threads(2)
    run(a.bolt_patches,a.out,a.batch)
