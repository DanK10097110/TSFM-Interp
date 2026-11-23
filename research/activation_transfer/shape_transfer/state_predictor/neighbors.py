"""Describe whether dense and binary retrieval select the same histories; uses no effect labels."""
import json

import numpy as np
import torch

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.state_predictor.fit import key_material,rank_neighbours
from research.activation_transfer.shape_transfer.state_predictor.memory import HERE


def inspect(model,seed):
    _,labels,key,_,_,_=key_material(model,seed)
    train=labels=="train";test=~np.isin(labels,["train","cal"])
    d,_=rank_neighbours(key["dense"][train],key["dense"][test])
    s,support_distance=rank_neighbours(key["support"][train],key["support"][test],sparse=True)
    train_count=key["support"][train].sum(1);test_count=key["support"][test].sum(1)
    shared=(test_count[:,None]+train_count[None,:]-support_distance)/2
    minimum=support_distance.min(1)
    n_tied=(support_distance==minimum[:,None]).sum(1)
    dense_is_any_nearest=support_distance[np.arange(len(d)),d[:,0]]==minimum
    threshold=support_distance[np.arange(len(s)),s[:,3]]
    tied_overlap=(np.take_along_axis(support_distance,d[:,:4],axis=1)<=threshold[:,None]).mean(1)
    # Mean overlap is the number of the four dense neighbors also in the support set, divided by four.
    overlap=np.any(d[:,:4,None]==s[:,None,:4],axis=2).mean(1)
    output=dict(dense_width=int(key["dense"].shape[1]),dictionary_width=int(key["support"].shape[1]),
                domains={})
    for domain in np.unique(labels[test]):
        use=labels[test]==domain
        n=int(use.sum())
        draws=resampling_counts(n)
        row={}
        for name,val in (("top_one_agreement",(d[:,0]==s[:,0]).astype(float)),("top_four_overlap_fraction",overlap),
                         ("dense_nearest_among_all_support_ties",dense_is_any_nearest),
                         ("dense_top_four_fraction_within_support_kth_distance",tied_overlap)):
            v=val[use]
            row[name]=dict(value=float(v.mean()),series95=np.quantile(draws@v/n,[.025,.975]).tolist())
        count=key["support"][test][use].sum(1)
        row["median_maximum_shared_atoms_with_a_training_history"]=float(np.median(shared[use].max(1)))
        row["median_shared_atoms_with_a_generic_training_history"]=float(np.median(shared[use]))
        row["median_number_tied_for_support_nearest"]=float(np.median(n_tied[use]))
        row["n"]=n;row["positive_code_min_median_max"]=[int(count.min()),float(np.median(count)),int(count.max())]
        output["domains"][domain]=row
    return output


if __name__=="__main__":
    torch.set_num_threads(1)
    results={m:{str(seed):inspect(m,seed) for seed in (0,1)} for m in ("bolt","timesfm")}
    (HERE/"results"/"neighbors.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
