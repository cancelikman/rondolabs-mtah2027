"""External-only shared temporal state/event training, independently implemented.

Motivation: ASRF (Ishikawa et al., WACV2021), arxiv.org/abs/2007.06866.
This is not a reproduction: a four-class directed-event head shares our existing
geometry TCN backbone, and our frozen Viterbi replaces their refinement method.
No video, target/public GT, WPY or WR9 is opened. Training is4DFL, selection WOY.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import tcn_external as base
from experiments.transition_external import event_targets, swap_event_targets, decode
from src.bundle import load_manifest
from src.features import blocks_of
from src.train import fingerprint, metrics

VERSION='shared-state-directed-event-tcn-v1'
TRAIN_IDS=('J03WOH','J03WN1','J03WMX','J03WQQ')
PARENT_SHA='b600f85a00b35727df6b9eebf56e57646549bdbbd4eca053e92be4b93bb32dcb'
EVENT_LOSS_WEIGHT=.1


class JointTemporalNet(base.TemporalNet):
    def __init__(self,input_channels,channels=64):
        super().__init__(input_channels,channels)
        self.event_head=nn.Conv1d(channels,4,1)
        nn.init.zeros_(self.event_head.weight)
        nn.init.zeros_(self.event_head.bias)

    def forward(self,x):
        hidden=F.gelu(self.project(x))
        for block in self.blocks:hidden=block(hidden)
        return self.head(hidden),self.event_head(hidden)


def event_batch(sequences,rng,mean,scale,permutation,sign,batch_size,core):
    probability=np.asarray([s[3] for s in sequences],dtype=float);probability/=probability.sum()
    xs,ys,ws,es=[],[],[],[]
    for _ in range(batch_size):
        record,a,b,length=sequences[rng.choice(len(sequences),p=probability)]
        start=rng.integers(max(1,length-core+1))
        positions=np.arange(start-base.CONTEXT,start+core+base.CONTEXT)
        valid=(positions>=0)&(positions<length)
        indices=a+np.clip(positions,0,length-1)*base.STRIDE
        raw=np.array(record['x'][indices]);labels=record['data']['labels'][indices].copy()
        weights=record['weights'][indices].copy();events=record['events'][indices].copy()
        if rng.random()<.5:
            raw=base.exchange(raw,permutation,sign)
            labels=np.where(labels==0,1,np.where(labels==1,0,labels))
            events=swap_event_targets(events)
        x=base.transform(raw,mean,scale);x[~valid]=0
        labels[~valid]=-1;events[~valid]=-1;weights[~valid]=0
        sl=slice(base.CONTEXT,base.CONTEXT+core)
        xs.append(x.T);ys.append(labels[sl]);ws.append(weights[sl]);es.append(events[sl])
    return tuple(torch.from_numpy(np.stack(items)).to(dtype) for items,dtype in
        ((xs,torch.float32),(ys,torch.long),(ws,torch.float32),(es,torch.long)))


@torch.no_grad()
def infer_joint(model,record,mean,scale,permutation,sign,device):
    model.eval();n=len(record['data']['ball'])
    state=np.full((n,3),np.nan,dtype=np.float32);events=np.full((n,4),np.nan,dtype=np.float32)
    for a,b in blocks_of(record['data']):
        indices=np.arange(a,b,base.STRIDE);raw=np.asarray(record['x'][indices])
        normal=base.transform(raw,mean,scale)
        flipped=base.transform(base.exchange(raw,permutation,sign),mean,scale)
        p=np.zeros((len(indices),3),dtype=np.float32);q=np.zeros((len(indices),4),dtype=np.float32)
        for first in range(0,len(indices),1024):
            last=min(first+1024,len(indices));lo=max(0,first-base.CONTEXT);hi=min(len(indices),last+base.CONTEXT)
            window=np.zeros((2,normal.shape[1],last-first+2*base.CONTEXT),dtype=np.float32)
            offset=base.CONTEXT+lo-first
            window[0,:,offset:offset+hi-lo]=normal[lo:hi].T
            window[1,:,offset:offset+hi-lo]=flipped[lo:hi].T
            ps,qs=model(torch.from_numpy(window).to(device))
            ps=ps[:,:,base.CONTEXT:base.CONTEXT+last-first].softmax(1).cpu().numpy().transpose(0,2,1)
            qs=qs[:,:,base.CONTEXT:base.CONTEXT+last-first].softmax(1).cpu().numpy().transpose(0,2,1)
            p[first:last]=.5*(ps[0]+ps[1][:,[1,0,2]])
            q[first:last]=.5*(qs[0]+qs[1][:,[0,2,1,3]])
        state[a:b]=np.repeat(p,base.STRIDE,axis=0)[:b-a]
        events[a:b]=np.repeat(q,base.STRIDE,axis=0)[:b-a]
    return state,events


def evaluate(model,validation,mean,scale,permutation,sign,device,hgb,config):
    scores=[];arrays={}
    for record in validation:
        condition=record['condition']
        state,event=infer_joint(model,record,mean,scale,permutation,sign,device)
        mixed=.5*(hgb[condition]+state)
        scores.append(metrics(record['data'],decode(mixed,event,record['data'],config)))
        arrays.update({condition:state,'event_'+condition:event})
    return {'mean_spell':float(np.mean([r['spell_f1_mean'] for r in scores])),
            'mean_macro':float(np.mean([r['macro_f1'] for r in scores])),
            'clean':scores[0],'occluded':scores[1]},arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--warm',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=ROOT/'models/ahmetcan_motion_checkpoint')
    p.add_argument('--hgb-cache',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--audit-cpu',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.steps,args.patience)<=0:raise ValueError('positive training counts required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(base.SEED);np.random.seed(base.SEED)
    torch.use_deterministic_algorithms(True,warn_only=args.device=='mps')
    if args.device=='mps' and not torch.backends.mps.is_available():raise ValueError('actual MPS device unavailable')
    manifest,_,manifest_sha=load_manifest(args.bundle)
    if manifest_sha!=PARENT_SHA or fingerprint(args.warm)!=manifest['components']['tcn']['source_checkpoint_sha256']:
        raise ValueError('only the reviewed four-train motion parent is supported')
    parity=json.loads((ROOT/'reports/external/motion_bundle_independent_parity.json').read_text())
    if parity['result']!='PASS' or parity['manifest_sha256']!=PARENT_SHA or fingerprint(args.hgb_cache)!=parity['cache_sha256']['hgb']:
        raise ValueError('HGB cache lacks exact canonical source binding')
    if args.out.exists() and not args.audit_cpu:raise FileExistsError('use fresh experiment output; previous evidence stays frozen')
    args.out.mkdir(parents=True,exist_ok=True)
    base.PROFILE='contact_motion'
    records,validation,names,sources=base.prepare(args.data_dir,args.out/'feature_cache',train_ids=TRAIN_IDS,
                                                reuse_cache=args.reuse_cache,segment_power=.5)
    mean,scale,permutation,sign,lineage=base.frozen_normalisation(args.warm,names,sources,TRAIN_IDS,'contact_motion')
    if sources['J03WOY']!=parity['source_sha256']:raise ValueError('WOY source changed')
    with np.load(args.hgb_cache,allow_pickle=False) as saved:hgb={k:saved[k].copy() for k in ('clean','occluded')}
    counts=np.zeros(3);event_counts=np.zeros(4)
    for record in records:
        record['events'],_=event_targets(record['data'])
        y=record['data']['labels'];counts+=np.bincount(y[y>=0],minlength=3)
        e=record['events'];event_counts+=np.bincount(e[e>=0],minlength=4)
    counts[:2]=counts[:2].mean();event_counts[1:3]=event_counts[1:3].mean()
    model=JointTemporalNet(len(names)*2,64)
    warm=torch.load(args.warm,map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(warm['state_dict'],strict=False)
    if set(missing)!={'event_head.weight','event_head.bias'} or unexpected:raise ValueError('warm backbone mismatch')
    with torch.no_grad():model.event_head.bias.copy_(torch.log(torch.tensor(event_counts/event_counts.sum(),dtype=torch.float32)))
    config=manifest['decoder'];device=torch.device(args.device)
    report={'version':VERSION,'method_inspiration':'https://arxiv.org/abs/2007.06866',
        'not_paper_reproduction':True,'code_independently_written':True,
        'train_matches':list(TRAIN_IDS),'validation_match':'J03WOY','source_sha256':sources,
        'parent_manifest_sha256':PARENT_SHA,'warm_checkpoint_sha256':fingerprint(args.warm),
        'hgb_cache_sha256':fingerprint(args.hgb_cache),'normalisation_lineage':lineage,
        'feature_profile':'contact_motion','feature_names':names,'decoder':config,
        'frame_mix':{'HGB':.5,'TCN':.5},'event_source':'joint temporal event head',
        'declared':{'event_loss_weight':EVENT_LOSS_WEIGHT,'event_radius_frames':2,
            'epochs':args.epochs,'steps':args.steps,'batch':8,'core':256,'lr':.0003,'patience':args.patience,
            'promotion_mean_spell_gain':.005,'max_condition_spell_decline':.005},
        'event_training_counts':event_counts.tolist(),'target_opened':False,'test_opened':False,
        'source_code_sha256':fingerprint(__file__),'epochs':[]}
    if args.audit_cpu:
        checkpoint=torch.load(args.out/'best.pt',map_location='cpu',weights_only=True)
        if checkpoint['source_sha256']!=sources or checkpoint['parent_manifest_sha256']!=PARENT_SHA:
            raise ValueError('CPU audit source mismatch')
        model.load_state_dict(checkpoint['state_dict']);model=model.to('cpu')
        score,arrays=evaluate(model,validation,mean,scale,permutation,sign,torch.device('cpu'),hgb,config)
        repeated,again=evaluate(model,validation,mean,scale,permutation,sign,torch.device('cpu'),hgb,config)
        assert score==repeated
        assert all(np.array_equal(arrays[k],again[k],equal_nan=True) for k in arrays)
        np.savez_compressed(args.out/'cpu_validation_probabilities.npz',**arrays)
        audit={'checkpoint_sha256':fingerprint(args.out/'best.pt'),'source_sha256':sources,
            'cpu_repeat_identical':True,'metrics':score,
            'array_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
            'target_or_test_opened':False,'decoder_retuned':False}
        (args.out/'cpu_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps({'CPU_audit':audit}),flush=True);return
    (args.out/'protocol.json').write_text(json.dumps(report,indent=2)+'\n')
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    class_weight=torch.tensor(counts.sum()/(3*counts),dtype=torch.float32,device=device)
    event_weight=torch.tensor(event_counts.sum()/(4*event_counts),dtype=torch.float32,device=device)
    sequences=base.sequence_inventory(records);rng=np.random.default_rng(base.SEED)
    best=None;best_epoch=0;began=monotonic()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for _ in range(args.steps):
            x,y,w,e=(v.to(device) for v in event_batch(sequences,rng,mean,scale,permutation,sign,8,256))
            optimizer.zero_grad(set_to_none=True)
            state,event=model(x);sl=slice(base.CONTEXT,base.CONTEXT+256);state=state[:,:,sl];event=event[:,:,sl]
            ce=F.cross_entropy(state,y,weight=class_weight,ignore_index=-1,reduction='none')
            state_loss=(ce*w).sum()/w.sum().clamp(min=1)
            event_ce=F.cross_entropy(event,e,weight=event_weight,ignore_index=-1,reduction='none')
            event_loss=event_ce.sum()/(e>=0).sum().clamp(min=1)
            logp=state.log_softmax(1);same=(y[:,1:]==y[:,:-1])&(y[:,1:]>=0)
            smooth=(logp[:,:,1:]-logp[:,:,:-1]).square().mean(1).clamp(max=16)
            loss=state_loss+EVENT_LOSS_WEIGHT*event_loss+.01*(smooth*same).sum()/same.sum().clamp(min=1)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();losses.append(float(loss.detach().cpu()))
        score,arrays=evaluate(model,validation,mean,scale,permutation,sign,device,hgb,config)
        row={'epoch':epoch,'loss':float(np.mean(losses)),**score};report['epochs'].append(row)
        key=(score['mean_spell'],score['mean_macro'])
        if best is None or key>best:
            best=key;best_epoch=epoch;report['selected']=row
            checkpoint={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'version':VERSION,'epoch':epoch,'channels':64,'feature_names':names,
                'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),'train_matches':list(TRAIN_IDS),
                'validation_match':'J03WOY','source_sha256':sources,'parent_manifest_sha256':PARENT_SHA,
                'normalisation_lineage':lineage,'decoder':config}
            torch.save(checkpoint,args.out/'best.pt');np.savez_compressed(args.out/'best_validation_probabilities.npz',**arrays)
        report['elapsed_seconds']=monotonic()-began;report['best_checkpoint_sha256']=fingerprint(args.out/'best.pt')
        (args.out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'epoch':epoch,'loss':row['loss'],'mean_spell':key[0],'mean_macro':key[1],
            'best_mean_spell':best[0],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
        if epoch-best_epoch>=args.patience:break


if __name__=='__main__':main()
