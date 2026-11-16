"""Train-only crop sampling: half by time, half by uniformly selected H/A spell.

Loss, labels, geometry and model are unchanged. Short controls get crop exposure
without truncating long spells or inventing a minimum-duration ownership target.
"""
import numpy as np
import torch
from experiments import tcn_external as base
from experiments.transition_external import swap_event_targets

TRAIN_IDS={'J03WOH','J03WN1','J03WMX','J03WQQ','J03WPY'}


def make_event_sampler(sequences,spell_probability=.5,teacher_targets=None,view_rng=None):
    if not 0<=spell_probability<=1:raise ValueError('Invalid mixture')
    spans=[]
    for j,(record,a,b,length) in enumerate(sequences):
        if record['match_id'] not in TRAIN_IDS or str(record['data']['match_id'])!=record['match_id']:
            raise ValueError('Training matches only; no development/reserve access')
        labels=np.asarray(record['data']['labels'])
        edges=np.r_[a,a+1+np.flatnonzero(labels[a+1:b]!=labels[a:b-1]),b]
        for low,high in zip(edges[:-1],edges[1:]):
            if labels[low] in (0,1):spans.append((j,int(low),int(high)))
    if not spans:raise ValueError('No training owner spells')
    probabilities=np.asarray([s[3] for s in sequences],dtype=float);probabilities/=probabilities.sum()

    def batch(current,rng,mean,scale,permutation,sign,batch_size,core):
        if current is not sequences:raise ValueError('Sampler bound to reviewed training inventory')
        xs,ys,ws,es=[],[],[],[]
        soft_targets,soft_masks=[],[]
        for _ in range(batch_size):
            if spell_probability>0 and rng.random()<spell_probability:
                j,low,high=spans[rng.integers(len(spans))]
                record,a,b,length=sequences[j]
                center=int(((low+high-1)/2-a)//base.STRIDE)
                start=int(np.clip(center-rng.integers(core),0,max(0,length-core)))
            else:
                record,a,b,length=sequences[rng.choice(len(sequences),p=probabilities)]
                start=rng.integers(max(1,length-core+1))
            positions=np.arange(start-base.CONTEXT,start+core+base.CONTEXT)
            valid=(positions>=0)&(positions<length)
            indices=a+np.clip(positions,0,length-1)*base.STRIDE
            features = record['x']
            if view_rng is not None and 'feature_views' in record:
                views = record['feature_views']
                features = views[view_rng.integers(len(views))]
            raw=np.array(features[indices]);labels=record['data']['labels'][indices].copy()
            weights=record['weights'][indices].copy();events=record['events'][indices].copy()
            exchanged = rng.random() < .5
            if exchanged:
                raw=base.exchange(raw,permutation,sign)
                labels=np.where(labels==0,1,np.where(labels==1,0,labels))
                events=swap_event_targets(events)
            x=base.transform(raw,mean,scale);x[~valid]=0
            labels[~valid]=-1;events[~valid]=-1;weights[~valid]=0
            sl=slice(base.CONTEXT,base.CONTEXT+core)
            xs.append(x.T);ys.append(labels[sl]);ws.append(weights[sl]);es.append(events[sl])
            if teacher_targets is not None:
                probability, eligible = teacher_targets[record['match_id']]
                soft = probability[indices[sl]].copy()
                if exchanged:
                    soft = soft[:, [1, 0, 2]]
                mask = eligible[indices[sl]] & valid[sl] & (labels[sl] >= 0)
                mask &= record['condition'] == 'occluded'
                soft[~mask] = 0
                soft_targets.append(soft)
                soft_masks.append(mask)
        output = tuple(torch.from_numpy(np.stack(values)).to(dtype) for values,dtype in
            ((xs,torch.float32),(ys,torch.long),(ws,torch.float32),(es,torch.long)))
        if teacher_targets is not None:
            output += (torch.from_numpy(np.stack(soft_targets)), torch.from_numpy(np.stack(soft_masks)))
        return output
    batch.training_spells=len(spans)
    return batch


