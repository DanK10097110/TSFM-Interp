"""Compare independent observed-value probes and native head on identical finite final-state edits."""
from contextlib import nullcontext
from hashlib import file_digest
import json

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.evaluate import write_npz
from research.activation_transfer.moment.crosscoder.fresh_readout.panel import OUT
from research.activation_transfer.moment.crosscoder.fresh_readout.run import BATCH, SLOT, ordinary, sample, sha
from research.activation_transfer.moment.crosscoder.readout_slots import native_slots, patch, source_events, writes
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import cached_middle
from research.activation_transfer.moment.token_paths.geometry import layers, load

ARMS=("small","paired","decoy")
METHODS=("masked","shuffled")
PROBES=OUT.parents[1]/"results/depth_trace"


@torch.inference_mode()
def capture(a,x,hs=None,move=None):
    states=[];site=layers(a)["middle"]
    cache=nullcontext() if hs is None else cached_middle(a,hs,site)
    edit=nullcontext() if move is None else patch(a,SLOT,move)
    def save(_,args,out):
        states.append(out[:,SLOT].detach().cpu().clone())
    with cache,edit:
        handle=a.module.encoder.final_layer_norm.register_forward_hook(save)
        try:native,_=native_slots(a,x)
        finally:handle.remove()
    assert len(states)==1;final=states[0]
    # The live head must recover the actual native edited-patch reading from these same final states.
    normalized=a.module.head.linear(final)[:,None,:]
    manual=a.module.normalizer(x=normalized,mode="denorm")[:,0].numpy()
    actual=native[:,2]  # native slots [59,60,61,62,63].
    err=float(np.max(np.abs(manual-actual)))
    assert err<.0001
    return final.numpy(),actual,err


def observe(a,x,hs,move=None):
    hidden=[];waves=[];errors=[]
    for lo in range(0,len(x),BATCH):
        ix=slice(lo,lo+BATCH);mv=None if move is None else move[ix]
        state,native,err=capture(a,x[ix],hs[ix],mv)
        hidden.append(state);waves.append(native);errors.append(err)
    return np.concatenate(hidden),np.concatenate(waves),max(errors)


def main():
    torch.set_num_threads(2)
    with np.load(OUT/"panel.npz") as f:x=f["x"].reshape(-1,480);ids=f["ids"];dates=f["dates"]
    adapters={s:load(s) for s in cd.SIZES}
    h={};base_final={};base_native={};weight={};unit={};replay_rows=np.array([4,20,115,201,286])
    std=x.astype(float).std(1);mu=x.astype(float).mean(1)
    clean=dict(ids=ids,dates=dates,sd=std,mu=mu,truth=(x[:,-8:]-mu[:,None])/std[:,None],input_sha256=sha(x))
    for size,a in adapters.items():
        probe=PROBES/(size+"_probes.npz")
        with probe.open("rb") as f:clean[size+"_probe_sha256"]=file_digest(f,"sha256").hexdigest()
        with np.load(probe) as f:
            weight[size]=f["final"];bias=f["final_bias"]
            assert len(f["train_ids"])==768 and len(f["val_ids"])==256
            if "train_ids" in clean:np.testing.assert_array_equal(clean["train_ids"],f["train_ids"])
            else:clean["train_ids"]=f["train_ids"];clean["val_ids"]=f["val_ids"]
        h[size],ordinary_native=ordinary(a,x,size)
        base_final[size],base_native[size],err=observe(a,x,h[size])
        np.testing.assert_allclose(base_native[size],ordinary_native[:,1],atol=.00004,rtol=0)
        clean[size+"_probe"]=base_final[size].astype(float)@weight[size]+bias
        clean[size+"_native"]=(base_native[size]-mu[:,None])/std[:,None]
        clean[size+"_head_replay_error_C"]=err
        # A literal zero write on identical native arithmetic is also replayed without any cached blocks.
        direct,actual,err=capture(a,x[replay_rows])
        zero,zeroy,err2=capture(a,x[replay_rows],move=np.zeros_like(h[size][replay_rows,SLOT]))
        np.testing.assert_array_equal(zero,direct);np.testing.assert_array_equal(zeroy,actual)
        unit[size]=(direct,actual);print(size,"ordinary final state",flush=True)
    write_npz(OUT/"final_clean.npz",**clean)
    pick,other=sample(len(x))
    for method in METHODS:
        for seed in cd.SEEDS:
            stem=f"{method}_{seed}";result=dict(ids=ids,dates=dates,sd=std,replay_rows=replay_rows);errors={}
            with np.load(OUT/(stem+".npz")) as old,np.load(cd.CACHE/(stem+".npz")) as f:
                np.testing.assert_array_equal(ids,old["ids"]);np.testing.assert_array_equal(dates,old["dates"])
                atoms,alt,dose=(old[k] for k in ("source_atom","other_atom","dose"))
                native_ref={k:old[k][:,1] for k in ARMS}
                model,means=cd.load_artifact(method,seed)
                for actual,expected in zip(source_events(model,means,h["small"][:,SLOT],pick,other),(atoms,alt,dose)):
                    np.testing.assert_array_equal(actual,expected)
                moves=writes(f["decoder_small"],f["decoder_large"],atoms,alt,dose)
                result.update(source_atom=atoms,other_atom=alt,dose=dose,input_sha256=sha(x))
            for arm in ARMS:
                size="small" if arm=="small" else "large";a=adapters[size]
                after,physical,head_error=observe(a,x,h[size],moves[arm])
                drift=after.astype(float)-base_final[size]
                pr=drift@weight[size];native=physical.astype(float)-base_native[size]
                previous=float(np.max(np.abs(native-native_ref[arm])))
                assert previous<.00006
                direct,actual,head_check=capture(a,x[replay_rows],move=moves[arm][replay_rows])
                d0,y0=unit[size]
                direct_probe=(direct.astype(float)-d0)@weight[size]
                p_gap=float(np.max(abs(direct_probe-pr[replay_rows])))
                c_gap=float(np.max(abs(actual-y0-native[replay_rows])))
                assert p_gap<.0001 and c_gap<.0002
                result[arm+"_probe"]=pr;result[arm+"_native_C"]=native
                errors[arm]=dict(previous_native_max_C=previous,
                    native_head_from_final_max_C=max(head_error,head_check),direct_native_max_C=c_gap,
                    direct_final_probe_max_history_SD=p_gap)
                print(stem,arm,"complete",flush=True)
            result["replay"]=json.dumps(errors);write_npz(OUT/("final_"+stem+".npz"),**result)


if __name__=="__main__":
    main()
