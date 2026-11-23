"""Compare released native short forecasts and controlled masked or visible patch reconstructions."""
import argparse
import json

import numpy as np
import torch
from momentfm import MOMENTPipeline
from safetensors import safe_open

from research.activation_transfer.moment.forecast_quality.data import OUT
from research.activation_transfer.moment.forecast_quality.predict import MOMENT_ROOT
from research.activation_transfer.moment.forecast_quality.score import boot_weights, percentile


def inspect_head(where):
    with safe_open(where / "model.safetensors", framework="pt") as f:
        key = next(k for k in f.keys() if k.startswith("head.") and k.endswith(".weight"))
        w = f.get_tensor(key).double()
        b = f.get_tensor(key.replace(".weight",".bias")).double()
        mask = next(k for k in f.keys() if "mask_embedding" in k)
        mask_norm = f.get_tensor(mask).double().norm().item()
    gram = w @ w.T
    return dict(weight_key=key, shape=list(w.shape), zero_bias_terms=int((b==0).sum()), bias_norm=b.norm().item(),
                max_abs_deviation_from_initializer_gram=(gram-torch.eye(8,dtype=w.dtype)*1.41**2).abs().max().item(),
                gram_eigenvalues=torch.linalg.eigvalsh(gram).tolist(), adjacent_row_inner_products=gram.diag(1).tolist(),
                mask_embedding_norm=mask_norm)


def manual(net, x, alternate, start, flank, visible=False):
    """Place the eight-point target in a fixed slot and fix RevIN to the shared past."""
    n = len(x)
    stop = start+8
    raw = torch.zeros((n,1,512))
    attention = torch.zeros((n,512),dtype=torch.long)
    mask = torch.ones_like(attention)
    raw[:,0,start-416:start] = x[:,:416]
    attention[:,start-416:stop] = 1
    if visible:
        raw[:,0,start:stop] = x[:,416:424]
    else:
        mask[:,start:stop] = 0
    if flank is not None:
        attention[:,stop:stop+56] = 1
        raw[:,0,stop:stop+56] = x[:,424:] if flank == "true" else alternate
    norm_mask = torch.zeros_like(attention)
    norm_mask[:,start-416:start] = 1
    normed = net.normalizer(x=raw,mask=norm_mask,mode="norm")
    emb = net.patch_embedding(net.tokenizer(normed),mask=mask)[:,0]
    amask = attention.reshape(n,64,8).amin(-1)
    hidden = net.encoder(inputs_embeds=emb,attention_mask=amask).last_hidden_state
    out = net.head(hidden[:,None])
    return net.normalizer(x=out,mode="denorm")[:,0,start:stop]


def batch(net, x, alternate):
    n = len(x)
    pad = torch.nn.functional.pad(x[:,:416], (96,0))
    attention = torch.ones((n,512),dtype=torch.long)
    attention[:,:96] = 0
    native = net.short_forecast(x_enc=pad[:,None],input_mask=attention,forecast_horizon=8).forecast[:,0]
    pad = torch.nn.functional.pad(x,(32,0))
    att = torch.ones((n,512), dtype=torch.long)
    att[:,:32] = 0
    mask = torch.ones_like(att)
    mask[:,448:456] = 0
    middle = net.reconstruction(x_enc=pad[:,None],input_mask=att,mask=mask).reconstruction[:,0,448:456]
    return dict(native_causal=native, manual_terminal=manual(net,x,alternate,504,None),
                interior_causal=manual(net,x,alternate,448,None),
                interior_true_right=manual(net,x,alternate,448,"true"),
                interior_wrong_right=manual(net,x,alternate,448,"wrong"),
                interior_visible=manual(net,x,alternate,448,"true",True), native_interior=middle)


def run(model):
    with np.load(OUT / "panel.npz") as f:
        d, split = f["domain"], f["split"]
        rows = np.concatenate([np.flatnonzero((d==key)&(split=="diagnostic"))[:12] for key in np.unique(d)])
        x = f["context"][rows,1]
        ids, d = f["id"][rows],d[rows]
    p = x[:,:416]
    ref = x[:,416:424]
    sd, mean = p.std(1,keepdims=True), p.mean(1,keepdims=True)
    wrong = np.zeros_like(x[:,424:])
    for key in np.unique(d):
        use = np.flatnonzero(d==key)
        alt = np.roll(use,1)
        wrong[use] = (x[alt,424:] - mean[alt])/sd[alt]*sd[use]+mean[use]
    path = MOMENT_ROOT/model.split("_")[-1]
    head = inspect_head(path)
    net = MOMENTPipeline.from_pretrained(str(path),model_kwargs={"task_name":"reconstruction"},strict=True).eval()
    output = {}
    with torch.inference_mode():
        for j in range(0,len(x),12):
            current = batch(net,torch.from_numpy(x[j:j+12].astype("f4")),torch.from_numpy(wrong[j:j+12].astype("f4")))
            for k, v in current.items():
                output.setdefault(k,[]).append(v.double().numpy())
    output = {k:np.concatenate(v) for k,v in output.items()}
    output.update(last=np.tile(p[:,-1:],(1,8)),prefix_mean=np.tile(mean,(1,8)),
                  two_sided_linear=p[:,-1:] + np.arange(1,9)/9*(x[:,424:425]-p[:,-1:]))
    report = dict(head=head,selected_ids=ids.tolist(),domains={})
    gap = np.abs(output["native_causal"]-output["manual_terminal"])/sd
    report["max_terminal_replay_error_in_past_sd"] = gap.max()
    if gap.max() > 2e-4:
        raise ValueError(f"Independent causal paths differ by {gap.max():.6g} past SD")
    for tag in [*np.unique(d), "all"]:
        use = np.ones(len(d),bool) if tag=="all" else d==tag
        actual = ((ref-mean)/sd)[use].mean(1)
        group = {}
        for k,v in output.items():
            level = ((v-mean)/sd)[use].mean(1)
            c = np.corrcoef(actual,level)[0,1] if level.std()>1e-12 else None
            group[k] = dict(mae_sd=float((np.abs(v-ref)/sd)[use].mean()),
                            correlation_with_target_level=c,
                            predicted_on_actual_level_slope=float(np.cov(actual,level,ddof=0)[0,1]/np.var(actual)))
        report["domains"][tag]=group
    each={k:(np.abs(v-ref)/sd).mean(1) for k,v in output.items()}
    report["paired_domain_balanced_mae_sd_difference"]={}
    for a,b in (("native_causal","last"),("interior_causal","manual_terminal"),
                ("interior_true_right","interior_wrong_right"),("interior_true_right","interior_causal"),
                ("interior_visible","prefix_mean"),("native_interior","two_sided_linear")):
        diff=each[a]-each[b]
        draws=[boot_weights("reconstruction|"+key,int((d==key).sum()))@diff[d==key] for key in np.unique(d)]
        report["paired_domain_balanced_mae_sd_difference"][a+"_minus_"+b]=percentile(np.mean(draws,axis=0))
    np.savez_compressed(OUT/(model+"_reconstruction.npz"),ids=ids,domain=d,target=ref,**output)
    (OUT/(model+"_reconstruction.json")).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",choices=("moment_small","moment_base","moment_large"))
    torch.set_num_threads(2)
    run(parser.parse_args().model)
