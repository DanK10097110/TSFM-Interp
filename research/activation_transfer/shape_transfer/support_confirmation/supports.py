"""Select dictionary spans without receiving edited forecast measurements."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from research.activation_transfer.atoms.materials import CACHE as ATOM_DATA,dictionary_path
from research.activation_transfer.replication.experiment import read_materials
from tsfm_lens.sae.train import load_sae_checkpoint

SEEDS=(2301,2302,2303,2304)
LADDER=(0,8,16,24)
ARM_NAMES=('native',*(f'swap_{s}' for s in SEEDS),'minimum_overlap','frequent','pca','isotropic',
           *(f'own_{k}' for k in LADDER))


def count_overlap(a,b):
    return np.array([np.intersect1d(u,v,assume_unique=True).size for u,v in zip(a,b)],dtype=np.int16)


def rank_frequency(support,width):
    counts=np.bincount(support.ravel(),minlength=width)
    return np.lexsort((np.arange(width),-counts)),counts


def select_supports(native,ranking):
    n,k=native.shape
    selected={'native':native}
    assigned={}
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(n)
        reassigned=np.empty_like(order)
        reassigned[order]=np.roll(order,1)
        assigned[f'swap_{seed}']=reassigned
    shared=np.array([[len(np.intersect1d(i,j,assume_unique=True)) for j in native] for i in native])
    # All tie-breakers summed together are smaller than any one integer overlap.
    cost=shared+np.random.default_rng(37112).random((n,n))/(n+1)
    np.fill_diagonal(cost,np.inf)
    row,col=linear_sum_assignment(cost)
    np.testing.assert_array_equal(row,np.arange(n))
    assigned['minimum_overlap']=col
    for label,assignment in assigned.items():
        np.testing.assert_array_equal(np.sort(assignment),np.arange(n))
        assert np.all(assignment!=np.arange(n))
        selected[label]=native[assignment]
    selected['frequent']=np.broadcast_to(ranking[:k],(n,k))
    random=np.random.default_rng(20261122014)
    own=np.stack([random.permutation(ids) for ids in native])
    replacement=np.stack([ranking[~np.isin(ranking,ids)][:k] for ids in native])
    for value in LADDER:
        if value>k:continue
        indices=np.concatenate((own[:,:value],replacement[:,:k-value]),axis=1)
        np.testing.assert_equal(count_overlap(indices,native),value)
        selected[f'own_{value}']=indices
    return selected,assigned


def basis(rows):
    # Economy QR of decoder rows. All experimental spans must have rank 32.
    matrix=np.asarray(rows,dtype=float)
    axes,upper=np.linalg.qr(matrix.swapaxes(-2,-1),mode='reduced')
    diag=np.abs(np.diagonal(upper,axis1=-2,axis2=-1))
    assert np.all(np.min(diag,axis=-1)>1e-7*np.max(diag,axis=-1))
    return axes.swapaxes(-2,-1)


def active_ids(sae,states):
    with torch.inference_mode():
        top=sae.encode(torch.from_numpy(states)).topk(sae.k,dim=1)
    assert top.values.min()>0
    return top.indices.numpy()


def fixed_directions(target,sae):
    tokens,_,ids=read_materials(ATOM_DATA,target)
    with np.load(ATOM_DATA/'data.npz') as f:
        np.testing.assert_array_equal(ids,f['ids'])
        train=f['split']=='train'
    assert train.sum()==1024
    pooled=tokens[train].mean(1)
    width,hidden=sae.W_dec.shape
    ranking,counts=rank_frequency(active_ids(sae,pooled),width)
    centered=pooled.astype(float)-pooled.astype(float).mean(0)
    pca=np.linalg.svd(centered,full_matrices=False)[2][:sae.k]
    randomq=np.linalg.qr(np.random.default_rng(2301).normal(size=(hidden,sae.k)))[0].T
    return ranking,counts,{'pca':pca,'isotropic':randomq},ids[train]


def candidates(target,pooled):
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    native=active_ids(sae,pooled)
    ranking,counts,axes,fit_ids=fixed_directions(target,sae)
    selections,assignments=select_supports(native,ranking)
    decoder=sae.W_dec.detach().numpy()
    native_axes=basis(decoder[native])
    for key,labels in selections.items():
        if key=='native':axes[key]=native_axes
        elif key in assignments:axes[key]=native_axes[assignments[key]]
        elif key=='frequent':axes[key]=basis(decoder[labels[0]])
        else:axes[key]=basis(decoder[labels])
    return selections,assignments,axes,counts,fit_ids
