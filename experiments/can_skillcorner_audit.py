"""Exploratory frozen full-model transfer to weak SkillCorner ownership labels.

Native 10Hz observations are linearly resampled to the TCN's 25Hz time grid.
Only adjacent finite coordinates interpolate; halves/gaps stay separate. Score
only original native rows after nearest-time prediction sampling (<=20ms).
No dense labels are invented. No SpellF1 or independent-gold accuracy claim.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.skillcorner_external import load_match
from src.bundle import predict_bundle,load_manifest
from src.train import fingerprint


def resample_geometry(data):
    if float(data['fps'])!=10.:raise ValueError('Native 10Hz required')
    balls,players,blocks,mapping=[],[],[],[]
    cursor=0
    for a,b in data['blocks']:
        n=int(b-a)
        # n25 never extrapolates beyond the last native coordinate.
        n25=int(np.floor((n-1)*2.5))+1
        q=np.arange(n25,dtype=float)*.4
        lo=np.floor(q+1e-10).astype(int);hi=np.minimum(lo+1,n-1)
        weight=q-lo
        def interpolate(value):
            x=value[a:b]
            shape=(n25,)+(1,)*(x.ndim-1)
            w=weight.reshape(shape)
            first,last=x[lo],x[hi]
            # At exact native times a missing next endpoint cannot erase data.
            return np.where(w<1e-9,first,first+(last-first)*w).astype(np.float32)
        balls.append(interpolate(data['ball']));players.append(interpolate(data['players']))
        blocks.append((cursor,cursor+n25))
        native_rows=np.arange(a,b)
        output_rows=cursor+np.minimum(np.rint(np.arange(n)*2.5).astype(int),n25-1)
        mapping.extend(zip(native_rows.tolist(),output_rows.tolist()))
        cursor+=n25
    result=dict(ball=np.concatenate(balls),players=np.concatenate(players),fps=np.asarray(25.),
        team_of_track=data['team_of_track'],pitch=data['pitch'],blocks=np.asarray(blocks,dtype=np.int64))
    return result,np.asarray(mapping,dtype=np.int64)


def agreement(labels,prediction,mask):
    mask=mask&np.isin(labels,[0,1])
    truth,pred=labels[mask],prediction[mask]
    confusion=np.zeros((2,4),dtype=int)
    for t,p in zip(truth,pred):confusion[t,p if p>=0 else 3]+=1
    count=len(truth)
    return dict(native_rows=count,weak_H_A_agreement=float(np.mean(truth==pred)) if count else None,
        predicted_D_fraction=float(np.mean(pred==2)) if count else None,
        confusion_rows_H_A_columns_H_A_D_X=confusion.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--match',type=int,choices=[1874553,1886347,1899585,1925299],default=1874553)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Preserve previous audit')
    bundle=ROOT/'models/ahmetcan_joint_checkpoint'
    _,_,digest=load_manifest(bundle)
    protocol=dict(version='can-SC-native-row-transfer-v1',match=args.match,
        parent_manifest_sha256=digest,driver_sha256=fingerprint(Path(__file__)),
        model_training=False,model_selection=False,target_access=False,
        independent_human_gold=False,label_quality='weak automatic group/phase agreement',
        resampling='adjacent-finite linear 10Hz to25Hz, reset each half, native score rows only',
        maximum_sampling_time_error_seconds=.02,spellF1_computed=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2))
    data=load_match(args.data_dir,args.match)
    geometry,mapping=resample_geometry(data)
    pred,metadata=predict_bundle(geometry,bundle)
    native_pred=np.full(len(data['labels']),-1,dtype=np.int8)
    native_pred[mapping[:,0]]=pred[mapping[:,1]]
    visible=np.isfinite(data['ball']).all(axis=1)
    allrows=np.zeros(len(visible),dtype=bool);allrows[mapping[:,0]]=True
    report=dict(protocol=protocol,inference=metadata,
        source_receipt_sha256=fingerprint(args.data_dir/f'SC{args.match}_external_v1.json'),
        native_frames=len(visible),resampled_frames=len(pred),known_label_fraction=float(np.mean(data['labels']>=0)),
        all=agreement(data['labels'],native_pred,allrows),
        ball_detected=agreement(data['labels'],native_pred,allrows&visible),
        ball_missing=agreement(data['labels'],native_pred,allrows&~visible),
        limitations=['Provider labels are automatic, selected and incomplete; clear phases are overrepresented.',
            'Unknown labels are excluded and never converted to dead ball.',
            'Resampling is a declared transfer transform, not new measured observations.',
            'No model or threshold is selected on this match; this is not a comparable dense SpellF1 benchmark.'])
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
