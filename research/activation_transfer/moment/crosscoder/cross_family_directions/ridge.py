"""Select independent clean-state patch probes and an ordinary paired state map on M4 validation only."""
import json
import numpy as np
from research.activation_transfer.moment.crosscoder.mechanism import map_inputs
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import CACHE,OUT,VIEWS,histories
from research.activation_transfer.moment.crosscoder.cross_family_directions.train import arrays

def fit(x,y,valid_x,valid_y):
    queries=np.vstack([valid_x,np.eye(x.shape[1]),np.zeros((1,x.shape[1]))])
    score={}
    best=float('inf')
    for lam,p in map_inputs(x,y,queries):
        loss=float(np.mean((p[:len(valid_x)]-valid_y)**2));score[str(lam)]=loss
        if loss<best:
            best=loss;win=lam;coef=p[len(valid_x):-1]-p[-1];bias=p[-1]
    return coef,bias,dict(chosen=win,validation_mse=score)

def pieces(x):
    shape=x-x.mean(-1,keepdims=True)
    four=x.reshape(*x.shape[:-1],4,8).mean(-1)
    return dict(full=x,shape32=shape,level=x.mean(-1,keepdims=True),shape4=four-four.mean(-1,keepdims=True))

def main():
    xt,it,xe,ie,dom=histories();a=arrays();y=xt[:768].reshape(-1,32)
    out={};details={}
    for name,arr in zip(VIEWS,a):
        train=arr[:768].reshape(-1,arr.shape[-1])
        w,b,record=fit(train,y,arr[768:1024,-1],xt[768:,-32:])
        pred=arr[1024:,-1]@w+b
        data={}
        for d in np.unique(dom):
            take=dom==d;data[str(d)]={}
            for k,goal in pieces(xe[take,-32:].astype(float)).items():
                estimate=pieces(pred[take])[k];center=pieces(y.mean(0,keepdims=True))[k]
                mse=np.mean((estimate-goal)**2)
                data[str(d)][k]=dict(r2=float(1-mse/np.mean((goal-center)**2)),rmse=float(np.sqrt(mse)))
        details[name]=dict(**record,clean_test=data)
        out[name+'_probe']=w;out[name+'_bias']=b;out[name+'_test']=pred
    sx,sy=a
    w,b,record=fit(sx[:768].reshape(-1,sx.shape[-1]),sy[:768].reshape(-1,sy.shape[-1]),sx[768:1024,-1],sy[768:1024,-1])
    details['map']=record;out.update(map=w,map_bias=b,train_ids=it[:768],validation_ids=it[768:],eval_ids=ie)
    np.savez_compressed(OUT/'ridge.npz',**out)
    (OUT/'ridge.json').write_text(json.dumps(details,indent=2)+'\n')

if __name__=='__main__':main()
