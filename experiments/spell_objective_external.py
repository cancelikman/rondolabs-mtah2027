"""Preregistered external-only continuation/capped-duration/focal ablation.

A: continuation control (power.5,gamma0); B: capped inverse duration (1,0);
C: capped inverse duration plus state focal factor (1,2). The duration cap5
means power1 is NOT exact equal-per-spell weighting. Frozen joint trainer bytes,
archived motion normalization, HGB mixture and decoder remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import tcn_external as base
from experiments import joint_transition_external as joint
from src.features import blocks_of,FEATURE_VERSION
from src.train import fingerprint,metrics,spell_weights
from experiments.transition_external import event_targets,decode

FROZEN_JOINT_SOURCE_SHA='e0b2ce009adc57c48244c7fa65ccbabae2579df2f3b39cd2eed24fd59bd685a8'
JOINT_PARENT_SHA='8ef6e4a13c78bc38d9a95b7ccdc3a0433a6e7917652367ef50f736abcac0d382'
MOTION_PARENT_SHA='ccc821c65b803c250a4ededf6cf97579ec5fa7b366d7e7fb2089c38fbaf4970f'
TRAIN_IDS=('J03WOH','J03WN1','J03WMX','J03WQQ')
VARIANTS=(('A_continuation',.5,0.),('B_capped_inverse',1.,0.),('C_capped_inverse_focal2',1.,2.))
PROMOTION={'minimum_mean_spell_gain':.005,'maximum_condition_spell_drop':.005}
VERSION='external-spell-objective-three-controls-v1'


def assert_frozen_trainer():
    if fingerprint(ROOT/'experiments/joint_transition_external.py')!=FROZEN_JOINT_SOURCE_SHA:
        raise ValueError('Frozen joint trainer bytes changed; refuse this experiment')


def weighted_state_focal(logits,labels,weights,class_weights,gamma):
    """Original weighted CE at gamma0; focal factor affects only state loss.

    The original denominator is retained: sum of sample weights, clamped to1.
    Ignored labels have zero CE and zero focal factor; no dense labels are made.
    """
    if not np.isfinite(gamma) or gamma<0:raise ValueError('gamma must be finite and nonnegative')
    ce=F.cross_entropy(logits,labels,weight=class_weights,ignore_index=-1,reduction='none')
    if gamma:
        safe=labels.clamp(min=0)
        probability=logits.softmax(dim=1).gather(1,safe[:,None,:]).squeeze(1)
        factor=(1.-probability).clamp(min=0).pow(float(gamma))
        factor=torch.where(labels>=0,factor,0.)
        ce=ce*factor
    return (ce*weights).sum()/weights.sum().clamp(min=1)


def objective(state,event,labels,weights,event_labels,class_weights,event_weights,gamma):
    state_loss=weighted_state_focal(state,labels,weights,class_weights,gamma)
    event_ce=F.cross_entropy(event,event_labels,weight=event_weights,ignore_index=-1,reduction='none')
    event_loss=event_ce.sum()/(event_labels>=0).sum().clamp(min=1)
    logp=state.log_softmax(1);same=(labels[:,1:]==labels[:,:-1])&(labels[:,1:]>=0)
    smooth=(logp[:,:,1:]-logp[:,:,:-1]).square().mean(1).clamp(max=16)
    return state_loss+.1*event_loss+.01*(smooth*same).sum()/same.sum().clamp(min=1)


def duration_records(records,power):
    if power not in (.5,1.):raise ValueError('Only preregistered duration powers accepted')
    output=[]
    for record in records:
        data=record['data']
        if record.get('match_id') not in TRAIN_IDS or str(data.get('source',''))!='IDSSE':
            raise ValueError('Duration weights accept external training records only')
        if str(data['match_id'])!=record['match_id']:raise ValueError('External training record identity mismatch')
        if power==.5:
            # Exact continuation control: retain existing validated weights.
            weight=record['weights'].copy()
        else:
            weight=spell_weights(data,power).astype(np.float32)
            for a,b in blocks_of(data):
                changes=a+np.flatnonzero(data['labels'][a+1:b]!=data['labels'][a:b-1])+1
                for change in changes:weight[max(a,change-5):min(b,change+6)]*=2.
        output.append(dict(record,weights=weight))
    return output


def validate_joint_parent(warm,names,sources,mean,scale):
    if (warm.get('version')!=joint.VERSION or warm.get('feature_names')!=names or len(names)!=330
            or warm.get('channels')!=64 or len(warm.get('train_matches',[]))!=4
            or set(warm['train_matches'])!=set(TRAIN_IDS) or warm.get('validation_match')!='J03WOY'
            or warm.get('source_sha256')!=sources or set(sources)!=set(TRAIN_IDS)|{'J03WOY'}
            or not np.array_equal(warm['mean'].numpy(),mean) or not np.array_equal(warm['scale'].numpy(),scale)):
        raise ValueError('Joint warm source/schema/frozen preprocessing mismatch')
    network=joint.JointTemporalNet(660,64);network.load_state_dict(warm['state_dict'])


def baseline_from_cpu(joint_dir,hgb,validation,config,sources):
    audit=json.loads((joint_dir/'cpu_audit.json').read_text())
    if (audit['checkpoint_sha256']!=JOINT_PARENT_SHA or audit['source_sha256']!=sources
            or audit['cpu_repeat_identical'] is not True or audit['target_or_test_opened'] or audit['decoder_retuned']):
        raise ValueError('Reviewed joint CPU baseline evidence required')
    scores={}
    with np.load(joint_dir/'cpu_validation_probabilities.npz',allow_pickle=False) as saved:
        for record in validation:
            c=record['condition'];state,event=saved[c],saved['event_'+c]
            for key,value in ((c,state),('event_'+c,event)):
                if hashlib.sha256(value.tobytes()).hexdigest()!=audit['array_sha256'][key]:raise ValueError('Joint baseline CPU cache changed')
            scores[c]=metrics(record['data'],decode(.5*(hgb[c]+state),event,record['data'],config))
            if scores[c]!=audit['metrics'][c]:raise ValueError('Joint baseline metrics do not reproduce exactly')
    return {'mean_spell':float(np.mean([r['spell_f1_mean'] for r in scores.values()])),
            'mean_macro':float(np.mean([r['macro_f1'] for r in scores.values()])),
            'clean':scores['clean'],'occluded':scores['occluded'],'identity_reproduction_exact':True}


def cpu_audit(folder,validation,mean,scale,perm,sign,hgb,config,sources):
    assert_frozen_trainer()
    c=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    validate_joint_parent(c,c['feature_names'],sources,mean,scale)
    model=joint.JointTemporalNet(660,64);model.load_state_dict(c['state_dict'])
    score,arrays=joint.evaluate(model,validation,mean,scale,perm,sign,torch.device('cpu'),hgb,config)
    repeat,again=joint.evaluate(model,validation,mean,scale,perm,sign,torch.device('cpu'),hgb,config)
    if score!=repeat or not all(np.array_equal(arrays[k],again[k],equal_nan=True) for k in arrays):
        raise ValueError('Frozen objective CPU repeatability failed')
    differences={}
    with np.load(folder/'best_validation_probabilities.npz',allow_pickle=False) as original:
        for key,value in arrays.items():
            valid=np.isfinite(value).all(axis=1)
            if not valid.any():raise ValueError('No finite CPU probabilities')
            differences[key]={'maximum_device_probability_difference':float(np.max(np.abs(value[valid]-original[key][valid]))),
                'argmax_disagreement_fraction':float(np.mean(value[valid].argmax(axis=1)!=original[key][valid].argmax(axis=1)))}
    np.savez_compressed(folder/'cpu_validation_probabilities.npz',**arrays)
    report={'checkpoint_sha256':fingerprint(folder/'best.pt'),'source_sha256':sources,'cpu_repeat_identical':True,
        'actual_device':'cpu','target_or_test_opened':False,'decoder_retuned':False,'metrics':score,
        'array_sha256':{key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in arrays.items()},
        'original_device_comparison':differences}
    (folder/'cpu_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def train_variant(name,power,gamma,args,records,validation,names,sources,mean,scale,perm,sign,lineage,warm,
                  hgb,config,class_weights,event_weights,baseline,registration_hash):
    assert_frozen_trainer();folder=args.out/name;folder.mkdir()
    torch.manual_seed(base.SEED);np.random.seed(base.SEED);random.seed(base.SEED)
    rng=np.random.default_rng(base.SEED);device=torch.device(args.device)
    model=joint.JointTemporalNet(660,64);model.load_state_dict(warm['state_dict']);model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    current=duration_records(records,power);sequences=base.sequence_inventory(current)
    report={'version':VERSION,'variant':name,'segment_power':power,'state_focal_gamma':gamma,
        'inverse_duration_cap':5.,'exact_equal_spell_weighting':False,
        'focal_changes_relative_state_event_loss_scale':bool(gamma),
        'train_matches':list(TRAIN_IDS),'validation_match':'J03WOY','source_sha256':sources,
        'feature_profile':'contact_motion','feature_version':FEATURE_VERSION,'feature_names':names,
        'normalisation_lineage':lineage,'warm_joint_checkpoint_sha256':JOINT_PARENT_SHA,
        'frozen_joint_trainer_sha256':FROZEN_JOINT_SOURCE_SHA,'implementation_sha256':fingerprint(Path(__file__)),
        'registration_sha256':registration_hash,'actual_device':args.device,'torch_version':str(torch.__version__),
        'seed':base.SEED,'fresh_optimizer':True,'learning_rate':.0001,'max_epochs':30,'steps_per_epoch':64,
        'batch_size':8,'core_steps':256,'patience':12,'event_loss_weight':.1,'smoothing_weight':.01,
        'frame_mix':{'HGB':.5,'TCN':.5},'decoder':config,'class_weights':class_weights.cpu().tolist(),
        'event_class_weights':event_weights.cpu().tolist(),'target_opened':False,'test_opened':False,
        'deterministic_mps_retraining_claimed':False,'promotion_rule_declared_before_training':PROMOTION,'epochs':[]}
    (folder/'protocol.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    best=None;started=monotonic()
    for epoch in range(1,31):
        began=monotonic();model.train();losses=[]
        for _ in range(64):
            x,y,w,e=(v.to(device) for v in joint.event_batch(sequences,rng,mean,scale,perm,sign,8,256))
            optimizer.zero_grad(set_to_none=True);state,event=model(x);sl=slice(base.CONTEXT,base.CONTEXT+256)
            loss=objective(state[:,:,sl],event[:,:,sl],y,w,e,class_weights,event_weights,gamma)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();losses.append(float(loss.detach().cpu()))
        score,arrays=joint.evaluate(model,validation,mean,scale,perm,sign,device,hgb,config)
        row={'epoch':epoch,'loss':float(np.mean(losses)),'elapsed_seconds':monotonic()-began,**score};report['epochs'].append(row)
        key=(score['mean_spell'],score['mean_macro'])
        checkpoint={'version':joint.VERSION,'objective_version':VERSION,'variant':name,'segment_power':power,'state_focal_gamma':gamma,
            'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),
            'epoch':epoch,'channels':64,'feature_names':names,'feature_profile':'contact_motion','feature_version':FEATURE_VERSION,
            'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),'train_matches':list(TRAIN_IDS),'validation_match':'J03WOY',
            'source_sha256':sources,'parent_manifest_sha256':warm['parent_manifest_sha256'],'normalisation_lineage':lineage,
            'base_joint_checkpoint_sha256':JOINT_PARENT_SHA,'decoder':config,'registration_sha256':registration_hash,
            'frozen_joint_trainer_sha256':FROZEN_JOINT_SOURCE_SHA,'source_training_code_sha256':report['implementation_sha256']}
        torch.save(checkpoint,folder/'last.pt')
        if best is None or key>best:
            best=key;report['selected']=row;torch.save(checkpoint,folder/'best.pt')
            np.savez_compressed(folder/'best_validation_probabilities.npz',**arrays)
        report['best_checkpoint_sha256']=fingerprint(folder/'best.pt');report['elapsed_seconds']=monotonic()-started
        (folder/'training_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'variant':name,'epoch':epoch,'loss':row['loss'],'mean_spell':key[0],'mean_macro':key[1],
                          'best_mean_spell':best[0],'elapsed_seconds':row['elapsed_seconds']}),flush=True)
        if epoch-report['selected']['epoch']>=12:
            print(json.dumps({'variant':name,'early_stop_epoch':epoch,'best_epoch':report['selected']['epoch']}),flush=True);break
    model=model.to('cpu');del optimizer
    audit=cpu_audit(folder,validation,mean,scale,perm,sign,hgb,config,sources)
    score=audit['metrics'];delta={c:score[c]['spell_f1_mean']-baseline[c]['spell_f1_mean'] for c in ['clean','occluded']}
    result={'variant':name,'segment_power':power,'state_focal_gamma':gamma,'metrics':score,
        'mean_spell_gain':score['mean_spell']-baseline['mean_spell'],'condition_spell_delta':delta,
        'cpu_audit_sha256':fingerprint(folder/'cpu_audit.json'),'checkpoint_sha256':audit['checkpoint_sha256']}
    result['eligible_for_promotion']=(result['mean_spell_gain']>=PROMOTION['minimum_mean_spell_gain']
        and min(delta.values())>=-PROMOTION['maximum_condition_spell_drop'])
    (folder/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    assert_frozen_trainer()
    print(json.dumps({'variant_complete':name,'mean_spell':score['mean_spell'],'mean_macro':score['mean_macro'],
                      'gain':result['mean_spell_gain'],'eligible_for_promotion':result['eligible_for_promotion']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--joint-dir',type=Path,default=ROOT.parent/'models/external_joint_transition_v1')
    parser.add_argument('--preprocessing-parent',type=Path,default=ROOT/'models/development_checkpoints/external_tcn_4train_motion64_v1/best.pt')
    parser.add_argument('--reuse-cache',type=Path,nargs='+',
        default=[ROOT.parent/'models/external_tcn_4train_motion64_v1/feature_cache'],
        help='Optional reviewed motion caches; absent folders trigger a current-provenance rebuild')
    parser.add_argument('--hgb-cache',type=Path,default=ROOT.parent/'models/external_hgb_4train_v1/validation_leaves31.npz')
    parser.add_argument('--device',choices=['cpu','mps'],default='mps');args=parser.parse_args()
    assert_frozen_trainer()
    if args.out.exists():raise FileExistsError('Preserve all previous trials; use new output directory')
    if fingerprint(args.joint_dir/'best.pt')!=JOINT_PARENT_SHA or fingerprint(args.preprocessing_parent)!=MOTION_PARENT_SHA:
        raise ValueError('Only reviewed joint8ef and archived motionccc parents accepted')
    parent_protocol=json.loads((args.joint_dir/'protocol.json').read_text())
    if (parent_protocol['train_matches']!=list(TRAIN_IDS) or parent_protocol['validation_match']!='J03WOY'
            or parent_protocol['target_opened'] or parent_protocol['test_opened']
            or fingerprint(args.hgb_cache)!=parent_protocol['hgb_cache_sha256']):
        raise ValueError('Frozen parent source/HGB protocol mismatch')
    if args.device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable; no silent CPU fallback')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=args.device=='mps')
    args.out.mkdir(parents=True)
    registration={'version':VERSION,'variants_declared_before_training':[{'name':n,'segment_power':p,'gamma':g} for n,p,g in VARIANTS],
        'duration_cap':5.,'exact_equal_spell_weighting':False,'train_matches':list(TRAIN_IDS),'validation_match':'J03WOY',
        'warm_joint_checkpoint_sha256':JOINT_PARENT_SHA,'preprocessing_checkpoint_sha256':MOTION_PARENT_SHA,
        'frozen_joint_trainer_sha256':FROZEN_JOINT_SOURCE_SHA,'implementation_sha256':fingerprint(Path(__file__)),
        'seed_same_for_all_variants':base.SEED,'fresh_AdamW_lr':.0001,'weight_decay':.0001,
        'max_epochs':30,'steps':64,'batch':8,'core':256,'patience':12,'event_loss_weight':.1,'smoothing_weight':.01,
        'frame_mix':parent_protocol['frame_mix'],'decoder':parent_protocol['decoder'],
        'cache_policy':'reuse reviewed external caches when present; otherwise build current features with actual provenance',
        'focal_gamma2_changes_relative_state_CE_event_magnitude':True,
        'promotion_rule':PROMOTION,'target_or_test_opened':False}
    registration_path=args.out/'registration.json';registration_path.write_text(json.dumps(registration,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_three_registered':registration['variants_declared_before_training'],
                      'registration_sha256':fingerprint(registration_path),'actual_device':args.device}),flush=True)
    old_profile=base.PROFILE;base.PROFILE='contact_motion'
    try:
        records,validation,names,sources=base.prepare(args.data_dir,args.out/'feature_cache',
            train_ids=TRAIN_IDS,reuse_cache=args.reuse_cache,
            segment_power=.5,player_augmentation=False)
    finally:base.PROFILE=old_profile
    mean,scale,perm,sign,lineage=base.frozen_normalisation(args.preprocessing_parent,names,sources,TRAIN_IDS,'contact_motion')
    warm=torch.load(args.joint_dir/'best.pt',map_location='cpu',weights_only=True)
    validate_joint_parent(warm,names,sources,mean,scale)
    with np.load(args.hgb_cache,allow_pickle=False) as saved:hgb={c:saved[c].copy() for c in ['clean','occluded']}
    config=parent_protocol['decoder'];baseline=baseline_from_cpu(args.joint_dir,hgb,validation,config,sources)
    if baseline['mean_spell']!=.7664258129656051:raise ValueError('Preregistered selected joint baseline changed')
    counts=np.zeros(3);event_counts=np.zeros(4)
    for record in records:
        record['events'],_=event_targets(record['data'])
        y=record['data']['labels'];counts+=np.bincount(y[y>=0],minlength=3)
        event=record['events'];event_counts+=np.bincount(event[event>=0],minlength=4)
    counts[:2]=counts[:2].mean();event_counts[1:3]=event_counts[1:3].mean()
    if (counts<=0).any() or (event_counts<=0).any():raise ValueError('Original class weights need all training classes')
    class_weights=torch.tensor(counts.sum()/(3*counts),dtype=torch.float32,device=args.device)
    event_weights=torch.tensor(event_counts.sum()/(4*event_counts),dtype=torch.float32,device=args.device)
    summary={'registration_sha256':fingerprint(registration_path),'source_sha256':sources,'baseline':baseline,
        'normalisation_lineage':lineage,'target_or_test_opened':False,'results':[]}
    for name,power,gamma in VARIANTS:
        result=train_variant(name,power,gamma,args,records,validation,names,sources,mean,scale,perm,sign,lineage,warm,
                             hgb,config,class_weights,event_weights,baseline,summary['registration_sha256'])
        summary['results'].append(result)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    eligible=[r for r in summary['results'] if r['eligible_for_promotion']]
    summary['selected']=max(eligible,key=lambda r:(r['metrics']['mean_spell'],r['metrics']['mean_macro'])) if eligible else None
    control=summary['results'][0]['metrics']
    for result in summary['results']:
        result['mean_spell_gain_vs_continuation_control']=result['metrics']['mean_spell']-control['mean_spell']
    summary['decision']='eligible_external_candidate_requires_root_review' if eligible else 'retain_joint_parent_no_integration'
    summary['frozen_joint_source_unchanged']=fingerprint(ROOT/'experiments/joint_transition_external.py')==FROZEN_JOINT_SOURCE_SHA
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_three_complete':True,'decision':summary['decision'],'summary':str(args.out/'summary.json')}),flush=True)


if __name__=='__main__':main()
