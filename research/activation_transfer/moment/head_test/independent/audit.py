"""Cross-runtime check of a linear reconstruction readout using the original MOMENT research code."""
import argparse
from argparse import Namespace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from scipy import linalg
import torch
from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "results"
QUALITY = HERE.parents[1] / "forecast_quality/results"
RIDGE = np.logspace(-6, 1, 8)


def model_from_original(root):
    from moment.models.moment import MOMENT

    raw = json.loads((root / "small/config.json").read_text())
    archive = root / "small/model.safetensors"
    with archive.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == "785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877"
    t5 = T5Config.from_pretrained(str(root / "flan_small"), local_files_only=True)
    args = Namespace(**{**raw, "task_name": "short-horizon-forecasting", "finetuning_mode": "zero-shot",
                        "d_model": t5.d_model, "debug": False})
    with patch.object(T5EncoderModel, "from_pretrained", side_effect=lambda _: T5EncoderModel(t5)):
        model = MOMENT(args).eval()
    model.load_state_dict(load_file(str(archive)), strict=True)
    model.requires_grad_(False)
    return model


def public_input(histories):
    values = torch.zeros(len(histories), 1, 512)
    valid = torch.zeros(len(histories), 512, dtype=torch.int64)
    values[:, 0, -histories.shape[1]:] = torch.as_tensor(histories, dtype=torch.float32)
    valid[:, -histories.shape[1]:] = 1
    return values, valid


def reconstruction(model, x, missing):
    """Build masks in scalar time coordinates and capture inputs to the actual linear output head."""
    values, valid = public_input(x)
    indices = 32 + 8*missing[:, :, None] + np.arange(8)
    mask = torch.ones_like(valid)
    rows = np.arange(len(x))
    mask[rows[:, None, None], indices] = 0
    records = []
    hook = model.head.linear.register_forward_pre_hook(lambda m, args: records.append(args[0].detach().numpy().copy()))
    try:
        out = model.pretraining(x_enc=values, input_mask=valid, mask=mask).reconstruction[:, 0].numpy()
    finally:
        hook.remove()
    hidden = records[0][rows[:, None], 0, indices[:, :, 0]//8]
    mu = model.normalizer.mean.reshape(-1).numpy().astype(float)
    sd = model.normalizer.stdev.reshape(-1).numpy().astype(float)
    visible = np.where((valid*mask).bool().numpy(), values[:, 0].numpy().astype(float), np.nan)
    ref_mu = np.nanmean(visible, axis=1)
    ref_sd = np.nanstd(visible, axis=1) + 1e-5
    assert np.max(np.abs(mu-ref_mu)/ref_sd) < 2e-4
    assert np.max(np.abs(sd-ref_sd)/ref_sd) < 2e-5
    raw = np.take_along_axis(x[:,None,:], indices-32, axis=2)
    labels = (raw-mu[:, None, None])/sd[:, None, None]
    return hidden, out, mu, sd, labels


def read_fit_data(model, scratch):
    with np.load(scratch / "train.npz") as f:
        x, holes, ids, split = (f[k] for k in ("context", "holes", "ids", "split"))
    with np.load(QUALITY / "panel.npz") as f:
        panel_ids = f["id"]
        panel_hashes = {hashlib.sha256(row.tobytes()).digest() for row in f["context"].reshape(-1, 480)}
    training_hashes = {hashlib.sha256(row.tobytes()).digest() for row in x}
    assert len(set(ids)) == len(training_hashes) == 1152
    assert np.sum(split == "train") == 1024 and np.sum(split == "cal") == 128
    assert not set("m4_daily/"+s for s in ids).intersection(panel_ids) and not training_hashes.intersection(panel_hashes)
    assert all(len(set(row)) == 8 and min(row) > 0 and max(row) < 59 for row in holes)
    assert (holes[:, 1::2] == holes[:, ::2]+1).all()
    pick = np.r_[np.arange(0, 1024, 128), np.arange(1024, 1152, 16)]
    clean = reconstruction(model, x[pick], holes[pick])
    altered = x[pick].copy().reshape(-1,60,8)
    altered[np.arange(len(pick))[:,None], holes[pick]] = np.random.default_rng(45162).normal(32000, 5000, (len(pick),8,8))
    wrong = reconstruction(model, altered.reshape(-1,480), holes[pick])
    for a,b in zip(clean[:4], wrong[:4]):np.testing.assert_array_equal(a,b)
    assert np.linalg.norm(clean[-1]-wrong[-1]) > 1.
    states=[]; labels=[]
    for lo in range(0,len(x),16):
        h, _, _, _, y=reconstruction(model,x[lo:lo+16],holes[lo:lo+16])
        states.append(h);labels.append(y)
    states, labels = np.concatenate(states).astype(float), np.concatenate(labels)
    with np.load(scratch / "small_0/interior.npz") as f:
        np.testing.assert_array_equal(f['ids'],ids)
        hidden_max=float(np.max(np.abs(states-f['h'])))
        label_max=float(np.max(np.abs(labels-f['y'])))
        archived_states=f['h'].astype(float);archived_labels=f['y'].astype(float)
    assert hidden_max < 4e-4 and label_max < 1e-4,(hidden_max,label_max)
    data_summary = dict(train=1024,validation=128,patches_per_history=8,panel_id_overlap=0,panel_history_overlap=0,
                        occlusion_test_items=len(pick),max_masked_output_change=0.,max_masked_hidden_change=0.,
                        max_masked_mu_or_sd_change=0.,max_cross_library_hidden_difference=hidden_max,
                        max_cross_library_normalized_target_difference=label_max)
    return states, labels, archived_states, archived_labels, split=='train',data_summary


def svd_ridge(features, labels, mask):
    d = features.shape[-1]
    x=features[mask].reshape(-1,d);y=labels[mask].reshape(-1,8)
    v=features[~mask].reshape(-1,d);t=labels[~mask].reshape(-1,8)
    mx,my=x.mean(0),y.mean(0)
    u,s,vt=linalg.svd(x-mx,full_matrices=False,check_finite=False,lapack_driver='gesdd')
    left=u.T@(y-my)
    scores=[];params=[]
    for penalty in RIDGE:
        w=vt.T@((s/(s*s+penalty*s[0]**2))[:,None]*left)
        b=my-mx@w
        scores.append(float(np.mean((v@w+b-t)**2)));params.append((w,b))
    winner=int(np.argmin(scores))
    w, b = params[winner]
    return w, b, scores, float(RIDGE[winner])


def refit(states,labels,train,head,heads,archived_fit):
    w0=head.weight.detach().numpy().astype(float)
    b0=head.bias.detach().numpy().astype(float)
    null=linalg.null_space(w0)
    assert null.shape==(512,504) and np.max(np.abs(w0@null)) < 1e-12
    summary={}
    for name,features in [('hidden',states),('published_space',states@w0.T+b0),('orthogonal',states@null)]:
        w,b,errors,penalty=svd_ridge(features,labels,train)
        if name=='published_space':b=b0@w+b;w=w0.T@w
        if name=='orthogonal':w=null@w
        wc=heads['interior_'+name+'_weight'];bc=heads['interior_'+name+'_bias']
        rel=float(np.linalg.norm(w-wc)/np.linalg.norm(wc))
        curve_err=float(np.max(np.abs(np.asarray(errors)-archived_fit[name]['mse_by_penalty'])))
        assert penalty==archived_fit[name]['relative'] and curve_err<1e-4 and rel<.001,(name,penalty,curve_err,rel)
        record=dict(selected_penalty=penalty,independent_validation_mse=min(errors),max_validation_curve_difference=curve_err,
                    relative_weight_difference=rel,max_bias_difference=float(np.max(np.abs(b-bc))))
        if name=='orthogonal':
            record['released_decoder_times_archived_complement_max']=float(np.max(np.abs(w0@wc)))
            assert record['released_decoder_times_archived_complement_max']<1e-12
        summary[name]=record
    return summary


def physical_forecasts(model,heads):
    with np.load(QUALITY/'panel.npz') as f:
        ids,domains,splits,x,y=(f[k] for k in ('id','domain','split','context','future'))
    chosen=[]
    for domain in sorted(set(domains)):
        indices=np.flatnonzero((domains==domain)&(splits=='holdout'))
        if not len(indices):continue
        chosen.extend(indices if domain=='m4_daily' else indices[np.linspace(0,len(indices)-1,8,dtype=int)])
    chosen=np.asarray(chosen);context=x[chosen];daily=domains[chosen]=='m4_daily'
    hsd=context.std(axis=-1);lag=np.abs(np.diff(context,axis=-1)).mean(axis=-1)
    source=json.loads((STUDY/'summary.json').read_text())['h16']['m4_daily']['scores']
    with np.load(STUDY/'small_0_pred.npz') as f:
        np.testing.assert_array_equal(f['ids'],ids)
        refs={k:f[k][chosen] for k in f.files if k!='ids'}
    raw_base=None;records={}
    for arm in ('interior_published','interior_hidden','interior_orthogonal','interior_published_space','terminal_hidden'):
        model.head.linear.weight.copy_(torch.from_numpy(heads[arm+'_weight'].T))
        model.head.linear.bias.copy_(torch.from_numpy(heads[arm+'_bias']))
        records[arm]={}
        for horizon in (8,16):
            flat=context.reshape(-1,480);chunks=[]
            for lo in range(0,len(flat),16):
                value,valid=public_input(flat[lo:lo+16]);yhat=model.short_forecast(x_enc=value,input_mask=valid,forecast_horizon=horizon).forecast[:,0]
                chunks.append(yhat.numpy())
            raw=np.concatenate(chunks).reshape(-1,2,horizon).astype(float)
            err=np.abs(raw-refs[arm+'_'+str(horizon)])
            discrepancy=float(np.max(err/hsd[:,:,None]));assert discrepancy<3e-4,(arm,horizon,discrepancy)
            metrics=dict(raw_unit_max_by_domain={k:float(err[domains[chosen]==k].max()) for k in sorted(set(domains[chosen]))},
                         max_error_in_history_sd=discrepancy)
            if horizon==16:
                truth=y[chosen][daily]
                item=(np.abs(raw[daily]-truth).mean(-1)/lag[daily]).mean(-1)
                score=float(item.mean());key='moment_small' if arm=='interior_published' else 'small_0_'+arm
                official_score=source[key]['mase1']['value'];assert abs(score-official_score)<2e-4
                metrics.update(daily_holdout_h16_mase1=score,archived_primary_score=official_score)
                if arm=='interior_published':raw_base=raw
                else:metrics['max_physical_change_in_history_sd']=float(np.max(np.abs(raw-raw_base)/hsd[:,:,None]))
            records[arm][str(horizon)]=metrics
    return dict(selected_items=[str(i) for i in ids[chosen]],origins_per_item=2,
                counts={d:int(np.sum(domains[chosen]==d)) for d in sorted(set(domains[chosen]))},arms=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/tmp/tsfm_moment_native'))
    p.add_argument('--scratch',type=Path,default=Path('/tmp/tsfm_moment_nov22_heads'))
    args=p.parse_args();research=args.root/'research_src'
    rev=subprocess.check_output(['git','-C',str(research),'rev-parse','HEAD'],text=True).strip()
    assert rev=='3ab637e413f35f2c317573c0ace280d825c558de';sys.path.insert(0,str(research));torch.set_num_threads(1)
    with torch.inference_mode():
        model=model_from_original(args.root)
        with np.load(STUDY/'small_0_heads.npz') as f:heads=dict(f)
        states,labels,archived_states,archived_labels,train,extraction=read_fit_data(model,args.scratch)
        print('full original reconstruction passed',extraction,flush=True)
        reference=json.loads((STUDY/'small_0_fit.json').read_text())['interior']
        fits=refit(states,labels,train,model.head.linear,heads,reference)
        fixed=refit(archived_states,archived_labels,train,model.head.linear,heads,reference)
        print('two SVD refits passed',fits,flush=True)
        physical=physical_forecasts(model,heads)
    summary=dict(research_revision=rev,versions={k:version(k) for k in ('torch','transformers','numpy','scipy')},
                 extraction=extraction,independent_library_svd=fits,same_design_svd=fixed,physical=physical)
    (HERE/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('physical replays passed',physical['counts'],flush=True)


if __name__=='__main__':main()
