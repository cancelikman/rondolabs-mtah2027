"""External-only anonymous track fragmentation and player-specific dropouts.

All parameters are fixed before target inference. This augmenter never examines
labels, status or ball values and never fits a distribution to a provided match.
It preserves geometry as multisets; track slots deliberately lose identity.
"""
from __future__ import annotations
import json
import numpy as np
from .features import blocks_of

TRACK_AUGMENTATION_VERSION="external-anonymous-tracks-v1"


def _runs(mask):
    edges=np.diff(np.r_[0,np.asarray(mask,dtype=np.int8),0])
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def _dropout_mask(length,rng,probability,mean_seconds,fps):
    """Sample a stationary two-state process, resetting at a visible run head.

    The continuous-time hidden dwell has the declared mean; exact sampled
    transition probabilities preserve its stationary hidden probability.
    """
    output=np.zeros(length,dtype=bool)
    if probability==0:return output
    if probability==1:
        output[:]=True
        return output
    rate_out=1./mean_seconds
    rate_in=probability/(1.-probability)*rate_out
    transition=-np.expm1(-(rate_in+rate_out)/fps)
    enter=probability*transition
    leave=(1.-probability)*transition
    hidden=bool(rng.random()<probability)
    cursor=0
    while cursor<length:
        duration=int(rng.geometric(leave if hidden else enter))
        end=min(length,cursor+duration)
        if hidden:output[cursor:end]=True
        hidden=not hidden
        cursor=end
    return output


def fragment_tracks(data,seed,fragment_mean_seconds=30.,dropout_probability=.1,
                    dropout_mean_seconds=.5,n_slots_per_team=23):
    """Return an immutable-source augmentation of canonical external IDSSE.

    Visible coordinate-pair runs are split with seeded exponential durations and
    greedily assigned to non-overlapping same-team slots. With dropout=0 every
    source observation occurs exactly once in its team's output multiset. A
    capacity shortage raises, rather than silently dropping or overwriting it.
    Ball/labels and all non-track arrays are carried through unchanged. Stale
    track-specific identity/speed metadata is removed; geometry features must
    derive bounded speed from the new anonymous slots locally.
    """
    if str(data.get("source",""))!="IDSSE":
        raise ValueError("track augmentation accepts external IDSSE only")
    if not isinstance(seed,(int,np.integer)) or isinstance(seed,bool) or seed<0:
        raise ValueError("seed must be a nonnegative integer")
    if (not isinstance(n_slots_per_team,(int,np.integer)) or isinstance(n_slots_per_team,bool)
            or n_slots_per_team<1):
        raise ValueError("slot capacity must be a positive integer")
    for name,value in (("fragment mean",fragment_mean_seconds),("dropout mean",dropout_mean_seconds)):
        if not np.isfinite(value) or value<=0:raise ValueError(name+" must be finite and positive")
    if not np.isfinite(dropout_probability) or not 0<=dropout_probability<=1:
        raise ValueError("dropout_probability must be in [0,1]")
    fps=float(data["fps"])
    if not np.isfinite(fps) or fps<=0:raise ValueError("fps must be finite and positive")
    original=np.asarray(data["players"])
    team=np.asarray(data["team_of_track"])
    if (original.ndim!=3 or original.shape[2]!=2 or team.shape!=(original.shape[1],)
            or len(data["ball"])!=len(original) or set(np.unique(team))!={0,1}):
        raise ValueError("invalid player/team/recording dimensions")
    if np.isinf(original).any() or not np.array_equal(np.isfinite(original[:,:,0]),np.isfinite(original[:,:,1])):
        raise ValueError("coordinates must be finite pairs or paired NaNs")
    attrs=data.get("attrs")
    if isinstance(attrs,dict) and attrs.get("coordinate_units","metres") not in {"metres","meters","m"}:
        raise ValueError("canonical external metre coordinates required")
    blocks=blocks_of(data)
    covered=np.zeros(len(original),dtype=bool)
    for start,end in blocks:covered[start:end]=True
    if np.isfinite(original[~covered]).any():
        raise ValueError("player observations outside declared blocks would be omitted")
    dtype=original.dtype if np.issubdtype(original.dtype,np.floating) else np.float64
    output=np.full((len(original),2*n_slots_per_team,2),np.nan,dtype=dtype)
    source_count=dropped_count=fragment_count=0
    for block_index,(start,end) in enumerate(blocks):
        source_visible=np.isfinite(original[start:end]).all(axis=2)
        for side in (0,1):
            simultaneous=source_visible[:,team==side].sum(axis=1)
            if (simultaneous>n_slots_per_team).any():
                raise ValueError("anonymous slot capacity insufficient for simultaneous same-team observations")
        split_rng=np.random.default_rng(np.random.SeedSequence([int(seed),block_index,0]))
        slot_rng=np.random.default_rng(np.random.SeedSequence([int(seed),block_index,1]))
        fragments=[]
        hidden=np.zeros(source_visible.shape,dtype=bool)
        for track in range(original.shape[1]):
            dropout_rng=np.random.default_rng(np.random.SeedSequence([int(seed),block_index,2,track]))
            for first,last in _runs(source_visible[:,track]):
                first,last=int(first),int(last)
                hidden[first:last,track]=_dropout_mask(last-first,dropout_rng,
                    dropout_probability,dropout_mean_seconds,fps)
                cursor=first
                while cursor<last:
                    duration=max(1,int(np.ceil(split_rng.exponential(fragment_mean_seconds*fps))))
                    stop=min(last,cursor+duration)
                    fragments.append((cursor,stop,track,int(team[track])))
                    cursor=stop
        available_at=np.zeros((2,n_slots_per_team),dtype=np.int64)
        previous_slot={}
        for first,last,track,side in sorted(fragments):
            candidates=np.flatnonzero(available_at[side]<=first)
            if not len(candidates):raise ValueError("anonymous slot capacity insufficient; no observation was discarded")
            alternative=candidates[candidates!=previous_slot.get(track,-1)]
            if len(alternative):candidates=alternative
            slot=int(slot_rng.choice(candidates));destination=side*n_slots_per_team+slot
            values=original[start+first:start+last,track].copy()
            dropout=hidden[first:last,track]
            values[dropout]=np.nan
            if np.isfinite(output[start+first:start+last,destination]).any():
                raise AssertionError("fragment assignment would overwrite an observation")
            output[start+first:start+last,destination]=values
            available_at[side,slot]=last;previous_slot[track]=slot
        source_count+=int(source_visible.sum())
        dropped_count+=int((hidden & source_visible).sum())
        fragment_count+=len(fragments)
    if int(np.isfinite(output).all(axis=2).sum())!=source_count-dropped_count:
        raise AssertionError("fragmentation did not preserve all retained coordinate pairs")
    result=dict(data)
    stale={"player_speed","players_speed","player_velocity","player_velocities","players_velocity",
           "track_speed","track_velocity","player_ids","players_ids","track_ids",
           "player_metadata","players_metadata","track_metadata"}
    for key in stale:result.pop(key,None)
    result["players"]=output
    result["team_of_track"]=np.repeat(np.array([0,1],dtype=np.int8),n_slots_per_team)
    result["n_tracks"]=np.asarray(2*n_slots_per_team,dtype=np.int64)
    result["track_ids"]=np.arange(2*n_slots_per_team,dtype=np.int64)
    if isinstance(attrs,dict):
        current=dict(attrs)
        for key in stale | {"player_speed_units","speed_channels","team_of_track"}:current.pop(key,None)
        current.update(n_tracks=2*n_slots_per_team,coordinate_units="metres",anonymous_track_slots=True)
        result["attrs"]=current
    result["track_augmentation"]=np.asarray(json.dumps({"version":TRACK_AUGMENTATION_VERSION,
        "seed":int(seed),"fragment_mean_seconds":float(fragment_mean_seconds),
        "dropout_probability":float(dropout_probability),"dropout_mean_seconds":float(dropout_mean_seconds),
        "n_slots_per_team":int(n_slots_per_team),"source_observations":source_count,
        "dropped_observations":dropped_count,"fragments":fragment_count,
        "player_speed_metadata_cleared":True,"labels_inspected":False},sort_keys=True))
    return result
