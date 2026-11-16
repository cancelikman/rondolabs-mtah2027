"""Independent, label-blind team geometry and motion during ball occlusion.

Motivation: permutation-invariant team geometry in possession research, e.g.
Borghesi et al. (2023), doi:10.1016/j.eswa.2023.119780. No third-party source,
weights or fitted constants are copied. Position statistics are pitch fractions;
velocity statistics are metres/second. No global match normalization is fitted.
"""
import warnings
import numpy as np

from .features import blocks_of,smooth_nan
from .motion_features import _velocity

TEAM_SHAPE_FEATURE_VERSION="team-shape-motion-v1"
VELOCITY_LAGS_SECONDS=(.2,.4)
CONTEXT_SECONDS=(.8,3.)
ROSTER_SIZE=11.  # Physical football prior; never divide by the exported track pool.


def _side_means(values,width):
    """Past/future means excluding the current sample, clipped to one block."""
    valid=np.isfinite(values)
    count=np.concatenate([np.zeros((1,values.shape[1]),dtype=np.int64),
                          np.cumsum(valid,axis=0,dtype=np.int64)],axis=0)
    total=np.concatenate([np.zeros((1,values.shape[1])),
                          np.cumsum(np.where(valid,values,0.),axis=0)],axis=0)
    current=np.arange(len(values))
    def mean(first,last):
        denominator=count[last]-count[first]
        numerator=total[last]-total[first]
        return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),
                         where=denominator>0)
    return (mean(np.maximum(0,current-width),current),
            mean(current+1,np.minimum(len(values),current+width+1)))


def _one_block(players,teams,pitch,fps):
    n=len(players)
    velocity={seconds:_velocity(players,fps,max(1,round(seconds*fps)),14.)
              for seconds in VELOCITY_LAGS_SECONDS}
    columns,names=[],[]
    context_indices,side_indices=[],[]
    for team in (0,1):
        p=players[:,teams==team].copy()
        # Empty exported slots carry no observations or information. Pruning them
        # also keeps numerical reduction order unchanged when unused slots append.
        used=np.isfinite(p).all(axis=2).any(axis=0)
        if used.any():
            p=p[:,used]
        valid=np.isfinite(p).all(axis=2)
        p[~valid]=np.nan
        fraction=p/pitch
        count=valid.sum(axis=1)
        prefix="team%d_shape_"%team
        def add(name,value,context=False,side=False):
            index=len(columns)
            columns.append(np.asarray(value,dtype=float));names.append(prefix+name)
            if context:context_indices.append(index)
            if side:side_indices.append(index)
        add("observed_count",count)
        add("roster_fraction",count/ROSTER_SIZE,context=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            centroid=np.nanmean(fraction,axis=1)
            median=np.nanmedian(fraction,axis=1)
            quantiles=np.nanpercentile(fraction,[10,25,75,90],axis=1)
            spread=np.nanstd(fraction,axis=1)
            centered=fraction-centroid[:,None,:]
            covariance=np.nanmean(centered[:,:,0]*centered[:,:,1],axis=1)
            covariance[count<2]=np.nan
        for axis,label in enumerate(("x","y")):
            add("centroid_%s_fraction"%label,centroid[:,axis],context=True,side=True)
            add("median_%s_fraction"%label,median[:,axis],context=True)
            for q_index,percentile in enumerate((10,25,75,90)):
                add("%s_q%d_fraction"%(label,percentile),quantiles[q_index,:,axis])
            add("iqr_%s_fraction"%label,quantiles[2,:,axis]-quantiles[1,:,axis])
            add("width80_%s_fraction"%label,quantiles[3,:,axis]-quantiles[0,:,axis])
            add("spread_%s_fraction"%label,spread[:,axis],context=True)
        add("covariance_xy_fraction2",covariance)
        outside=valid & ((fraction<0)|(fraction>1)).any(axis=2)
        add("outside_fraction",np.divide(outside.sum(axis=1),count,
            out=np.full(n,np.nan),where=count>0))
        for axis,label in enumerate(("longitudinal","lateral")):
            coordinate=fraction[:,:,axis]
            for third in range(3):
                lower,upper=third/3.,(third+1)/3.
                inside=valid & (coordinate>=lower) & ((coordinate<upper) if third<2 else (coordinate<=upper))
                occupancy=np.divide(inside.sum(axis=1),count,out=np.full(n,np.nan),where=count>0)
                add("%s_third%d_fraction"%(label,third),occupancy)
        for seconds in VELOCITY_LAGS_SECONDS:
            v=velocity[seconds][:,teams==team]
            finite=np.isfinite(v).all(axis=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",RuntimeWarning)
                mean_v=np.nanmean(v,axis=1)
                median_v=np.nanmedian(v,axis=1)
            for axis,label in enumerate(("vx","vy")):
                add("mean_%s_dt%g"%(label,seconds),mean_v[:,axis],context=True,side=True)
                add("median_%s_dt%g"%(label,seconds),median_v[:,axis],context=(seconds==.2))
            add("velocity_valid_fraction_dt%g"%seconds,np.divide(finite.sum(axis=1),count,
                out=np.zeros(n),where=count>0))
    base=np.stack(columns,axis=1)
    base[~np.isfinite(base)]=np.nan
    pieces,all_names=[base],list(names)
    for seconds in CONTEXT_SECONDS:
        width=max(1,round(seconds*fps))|1
        pieces.append(smooth_nan(base[:,context_indices],width))
        all_names += [names[i]+"_mean%g"%seconds for i in context_indices]
    before,after=_side_means(base[:,side_indices],max(1,round(2.*fps)))
    pieces.extend([before,after,after-before])
    for suffix in ("_before2s","_after2s","_after_minus_before2s"):
        all_names += [names[i]+suffix for i in side_indices]
    return np.concatenate(pieces,axis=1).astype(np.float32),all_names


def team_shape_features(data):
    """Return (n_frames x n_features, names), using player geometry only.

    Ball values, labels, owning-team/status fields and match identity are not
    inspected. ``ball`` is used by the shared boundary helper only for length.
    Roster fractions divide observed counts by the physical 11-player prior, never
    by the exported pool of anonymous/fragmented tracks. Counts may exceed 11
    under duplicate detections; no target-derived normalization is performed.
    All-unknown shape/motion stays NaN; observed/validity fractions expose its
    quality without equating missing players with a stationary narrow team.
    """
    fps=float(data["fps"])
    players=np.asarray(data["players"],dtype=float)
    teams=np.asarray(data["team_of_track"])
    pitch=np.asarray(data["pitch"],dtype=float)
    if not np.isfinite(fps) or fps<=0:
        raise ValueError("fps must be finite and positive")
    if pitch.shape!=(2,) or not np.isfinite(pitch).all() or (pitch<=0).any():
        raise ValueError("pitch must contain positive finite length/width")
    if players.ndim!=3 or players.shape[2]!=2 or teams.shape!=(players.shape[1],):
        raise ValueError("invalid player/team dimensions")
    if len(data["ball"])!=len(players) or set(np.unique(teams))!={0,1}:
        raise ValueError("expected aligned recording and tracks for both teams")
    result,names=None,None
    for start,end in blocks_of(data):
        feature,current_names=_one_block(players[start:end],teams,pitch,fps)
        if result is None:
            names=current_names
            result=np.full((len(players),feature.shape[1]),np.nan,dtype=np.float32)
        elif names!=current_names:
            raise ValueError("inconsistent team-shape schema")
        result[start:end]=feature
    if result is None:
        raise ValueError("empty recording")
    return result,names
