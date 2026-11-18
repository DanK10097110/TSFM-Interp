"""Separate quantized frontend jumps from encoder nonlinearity in cached Chronos.

This is an instrumentation control, not a forecasting benchmark. Deterministic
first-decoder-step probabilities avoid Monte Carlo forecast noise. Embedding
interpolations are explicitly off-manifold interventions, not a deployable input
adapter. Run into a fresh directory only after a coordinator compute-slot grant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


def write(path, value):
    """Persist finite audit records after every completed model condition."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    """Hash checkpoint/source bytes without copying large weights."""
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rms(x):
    """FP64 RMS with no arbitrary dimensional dependence."""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))


def boundary_pair(history, direction, tokenize):
    """Locate the first changed token bundle bracketed along a short input ray.

All tokenizer calls see FP32 histories. The returned lower endpoint shares the
original token bundle; the upper differs. No assumption of an exactly solvable
floating-point threshold or of one changed token is made.
"""
    x = np.asarray(history, np.float32)
    d = np.asarray(direction, np.float32)
    ids = tokenize(x)
    lo, hi = 0., 1e-5
    point = lambda a: (x.astype(float)+a*d).astype(np.float32)
    while np.array_equal(tokenize(point(hi)), ids) and hi < .1:
        hi *= 2
    if np.array_equal(tokenize(point(hi)), ids):
        raise ValueError('No token crossing in declared ray')
    for _ in range(48):
        mid = (lo+hi)/2
        if np.array_equal(tokenize(point(mid)), ids): lo = mid
        else: hi = mid
    left, right = point(lo), point(hi)
    if np.array_equal(tokenize(left), tokenize(right)):
        raise ValueError('Boundary endpoints collapsed')
    if not np.array_equal(tokenize(left), ids):
        raise ValueError('Same-token control lost identity')
    return left, right, dict(lower=lo, upper=hi, bracket=hi-lo,
                             raw_rms_distance=rms(right-left),
                             same_token_raw_rms_distance=rms(left-x))


def chord(left, middle, right):
    """Dimensionless departure from a chord; return absolute numerator as well.

This is computed in the actual representation coordinates. Decreasing radius
should reduce it for a differentiable, locally nonconstant map. A zero endpoint
chord is reported as degenerate rather than made significant by an epsilon.
"""
    a, b, c = [np.asarray(x, dtype=float) for x in (left, middle, right)]
    bend, span = rms(b-(a+c)/2), rms(c-a)
    return dict(bend_rms=bend, span_rms=span,
                ratio=None if span == 0 else bend/span)


def prepare(out, checkpoint, count, seed):
    """Freeze architecture, seeds and analysis before observing model responses."""
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(checkpoint, local_files_only=True))
    files = list(snapshot.glob('*.safetensors'))+list(snapshot.glob('pytorch_model*.bin'))
    if not files: raise ValueError('No cached checkpoint weights')
    out.mkdir(parents=True, exist_ok=False)
    p = dict(question='Do raw-input discontinuities originate at quantization, and does learned processing smooth adjacent-token changes?',
             checkpoint=checkpoint, snapshot=str(snapshot), checkpoint_hashes={f.name:sha(f) for f in files},
             source_hash=sha(__file__), seeds=list(range(seed, seed+count)), context=128,
             random_initialization_seed=4956999, radii=[.25,.125,.0625,.03125],
             histories='Fresh independent seeds: persistent sinusoids with seeded phase plus 0.2 white noise; no future prediction task.',
             endpoint='Full encoder states (embedding, block1, block2, final-normalized block4 for tiny) and raw first-step decoder softmax. No sampling, top-k truncation, or multi-step accuracy.',
             controls='Exact repeat; changed raw history with identical complete token bundle; token IDs versus identical input embeddings; pretrained versus one architecture-matched random initialization; decreasing-radius embedding interpolation.',
             units='Encoder changes RMS-normalized by endpoint state RMS; probability total variation; dimensionless embedding-path chord error. Raw distances are actual FP32 input differences.',
             limits='Synthetic8-source instrumentation pilot by default; one random initialization; interpolated embeddings are off manifold; token jumps are expected architecture behavior and do not demonstrate semantic incoherence.',
             resources=dict(threads=1,rss_bytes=3*2**30,seconds=600),
             literature='https://arxiv.org/abs/2403.07815')
    write(out/'protocol.json',p)
    write(out/'knowledge.json',dict(status='prepared_native_held',question=p['question'],design=p,
          exact_findings=[],next_decision='Acquire compute slot, then run pretrained and random conditions sequentially.'))


def run(out):
    """Capture deterministic surfaces and save each completed condition separately."""
    import torch
    from chronos import ChronosPipeline
    from transformers import T5ForConditionalGeneration
    torch.set_num_threads(1)
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if p['source_hash'] != sha(__file__): raise ValueError('Frozen source changed')
    for name, digest in p['checkpoint_hashes'].items():
        if sha(Path(p['snapshot'])/name) != digest: raise ValueError('Checkpoint changed')
    if (out/'pretrained_results.json').exists(): raise ValueError('Refuse completed run overwrite')
    started=time.monotonic()
    pipeline=ChronosPipeline.from_pretrained(p['snapshot'],device_map='cpu',torch_dtype=torch.float32,local_files_only=True)
    tokenizer=pipeline.tokenizer
    model=pipeline.model.model.eval()
    config=model.config
    inputs=[]; meta=[]
    def tokenize(x):
        return tokenizer.context_input_transform(torch.tensor(x[None]))[0][0].numpy()
    for seed in p['seeds']:
        rng=np.random.default_rng(seed); t=np.arange(-128,0)
        x=(np.sin(2*np.pi*t/19.2+rng.uniform(-np.pi,np.pi))+
           .3*np.cos(2*np.pi*t/24+rng.uniform(-np.pi,np.pi))+.2*rng.normal(size=128)).astype(np.float32)
        d=rng.normal(size=128); d/=rms(d)
        left,right,record=boundary_pair(x,d,tokenize)
        inputs.append(np.stack([x,left,right]))
        meta.append(dict(seed=seed,**record,changed_tokens=int(np.sum(tokenize(left)!=tokenize(right)))))
    np.savez_compressed(out/'inputs.npz',contexts=np.stack(inputs))
    write(out/'boundaries.json',meta)
    records=[]
    for condition in ('pretrained','random'):
        if condition=='random':
            del model
            torch.manual_seed(p['random_initialization_seed'])
            model=T5ForConditionalGeneration(config).eval()
        cuts=(0,1,len(model.encoder.block)//2,len(model.encoder.block))
        def capture(ids=None,embeddings=None):
            with torch.inference_mode():
                mask=torch.ones(1,129,dtype=torch.bool)
                enc=model.encoder(input_ids=ids,inputs_embeds=embeddings,attention_mask=mask,
                                  output_hidden_states=True,return_dict=True)
                result=model(encoder_outputs=enc,attention_mask=mask,
                             decoder_input_ids=torch.tensor([[config.decoder_start_token_id]]),return_dict=True)
                values={f'hidden_{i}':enc.hidden_states[i].float().numpy() for i in cuts}
                values['probability']=result.logits[0,0].softmax(-1).float().numpy()
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['resources']['rss_bytes']:
                raise MemoryError('Resident budget exceeded')
            if time.monotonic()-started>p['resources']['seconds']:
                raise TimeoutError('Wall budget exceeded')
            return values
        arrays={};rows=[]
        for j,(bank,description) in enumerate(zip(inputs,meta)):
            token=[torch.tensor(tokenize(v)[None]) for v in bank]
            raw=[capture(ids=tok) for tok in token]
            repeat=capture(ids=token[1])
            embed=[model.encoder.embed_tokens(tok).detach() for tok in token[1:]]
            embedded=capture(embeddings=embed[0])
            for key in raw[0]:
                if not np.array_equal(raw[0][key],raw[1][key]): raise ValueError('Same-token response differs')
                if not np.array_equal(raw[1][key],repeat[key]): raise ValueError('Repeat differs')
                if not np.array_equal(raw[1][key],embedded[key]): raise ValueError('ID/embedding replay differs')
                for endpoint in range(3): arrays[f'{j}_raw{endpoint}_{key}']=raw[endpoint][key]
            center=(embed[0]+embed[1])/2; direction=embed[1]-embed[0]
            mid=capture(embeddings=center)
            for radius in p['radii']:
                low=capture(embeddings=center-radius*direction)
                high=capture(embeddings=center+radius*direction)
                for key in mid:
                    values=chord(low[key],mid[key],high[key])
                    rows.append(dict(seed=description['seed'],surface=key,radius=radius,**values))
                    for tag,value in [('left',low[key]),('middle',mid[key]),('right',high[key])]:
                        arrays[f'{j}_{radius}_{tag}_{key}']=value
            jump={key:rms(raw[2][key]-raw[1][key])/max(rms(raw[1][key]),np.finfo(float).tiny)
                  for key in raw[1] if key.startswith('hidden')}
            rows.append(dict(surface='boundary',**description,relative_hidden_jump=jump,
                probability_tv=float(np.sum(abs(raw[2]['probability']-raw[1]['probability']))/2),
                embedding_relative_jump=rms((embed[1]-embed[0]).numpy())/rms(embed[0].numpy())))
        np.savez_compressed(out/f'{condition}_arrays.npz',**arrays)
        result=dict(condition=condition,rows=rows,cuts=list(cuts),
                    seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    gates=dict(repeat=0.,same_token=0.,embedding_replay=0.))
        write(out/f'{condition}_results.json',result)
        records.append(dict(condition=condition,artifact=f'{condition}_results.json',
                            mean_probability_tv=float(np.mean([r['probability_tv'] for r in rows if r['surface']=='boundary']))))
        write(out/'knowledge.json',dict(status='complete' if condition=='random' else 'pretrained_complete',
              question=p['question'],exact_findings=records,controls=p['controls'],limitations=p['limits'],
              next_decision='Independently audit saved arithmetic and compare frontend jump versus continuous embedding curvature; no forecasting claims.'))
        print(condition,records[-1],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',default='amazon/chronos-t5-tiny')
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--seed',type=int,default=4956000)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.out,args.checkpoint,args.count,args.seed)
    else:
        try: run(args.out)
        except Exception as error:
            write(args.out/'failure.json',dict(error=repr(error)))
            old=json.loads((args.out/'knowledge.json').read_text(encoding='utf-8'))
            old.update(status='failed',failure=repr(error),next_decision='Preserve artifacts; diagnose and rerun in a fresh directory.')
            write(args.out/'knowledge.json',old)
            raise
