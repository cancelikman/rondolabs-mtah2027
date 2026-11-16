"""Independent label-blind future geometry and bounded-speed interception cues.

These are features, not a rule that the next reachable player owns a flying pass.
Coordinates are metres, times seconds. All derivative support stays in one block;
no labels/status/video, fitting or target-specific statistics are used.
"""
import numpy as np

from .features import blocks_of, interpolate_short
from .motion_features import _velocity

REACHABILITY_FEATURE_VERSION = "constant-velocity-reach-v1"
HORIZONS_SECONDS = (.2, .5, 1.)
REACH_SPEEDS = (3., 6., 9.)


def _minimum_finite(value):
    """Row minimum, unknown where no finite candidate exists."""
    safe = np.where(np.isfinite(value), value, np.inf)
    result = safe.min(axis=1)
    return np.where(np.isfinite(result), result, np.nan)


def _interception_times(ball, ball_velocity, players, speed):
    """First t>=0 with |ball + v_ball*t - player| <= speed*t.

    This assumes an instantaneous choice of running direction at a fixed speed
    bound. It is an optimistic geometric cue, not an acceleration/contact model.
    An unreachable ray returns NaN, distinguished from unknown by geometry flags.
    """
    relative = ball[:, None, :] - players
    velocity = ball_velocity[:, None, :]
    a = np.sum(velocity * velocity, axis=-1) - speed * speed
    a = np.broadcast_to(a, relative.shape[:-1])
    b = 2. * np.sum(relative * velocity, axis=-1)
    c = np.sum(relative * relative, axis=-1)
    valid = np.isfinite(relative).all(axis=-1) & np.isfinite(velocity).all(axis=-1)
    result = np.full(c.shape, np.nan)
    touching = valid & (c <= 1e-12)
    result[touching] = 0.
    linear = valid & ~touching & (np.abs(a) <= 1e-8) & (b < -1e-8)
    result[linear] = -c[linear] / b[linear]
    discriminant = b*b - 4.*a*c
    quadratic = valid & ~touching & (np.abs(a) > 1e-8) & (discriminant >= 0.)
    root = np.sqrt(np.maximum(discriminant, 0.))
    first = np.full(c.shape, np.nan)
    second = np.full(c.shape, np.nan)
    np.divide(-b-root, 2.*a, out=first, where=quadratic)
    np.divide(-b+root, 2.*a, out=second, where=quadratic)
    first = np.where(quadratic & (first >= 0.), first, np.inf)
    second = np.where(quadratic & (second >= 0.), second, np.inf)
    earliest = np.minimum(first, second)
    result[quadratic & np.isfinite(earliest)] = earliest[quadratic & np.isfinite(earliest)]
    return result


def _block_features(ball, players, teams, fps):
    n = len(ball)
    current_ball = interpolate_short(ball, max(1, round(.2*fps)))
    lag = max(1, round(.2*fps))
    ball_velocity = _velocity(current_ball, fps, lag, 45.)
    player_velocity = _velocity(players, fps, lag, 14.)
    ball_motion_valid = np.isfinite(ball_velocity).all(axis=1)
    columns, names = [], []
    def add(name, value):
        names.append("reach_"+name)
        columns.append(np.asarray(value, dtype=float))
    add("ball_velocity_valid", ball_motion_valid)
    add("ball_speed", np.linalg.norm(ball_velocity, axis=1))
    for team in (0,1):
        present = np.isfinite(players[:, teams == team]).all(axis=2)
        add("team%d_geometry_available" % team, ball_motion_valid & present.any(axis=1))

    for speed in REACH_SPEEDS:
        times = _interception_times(current_ball, ball_velocity, players, speed)
        minima = []
        for team in (0,1):
            team_times = times[:, teams == team]
            minimum = _minimum_finite(team_times)
            minima.append(minimum)
            add("team%d_tti_c%g" % (team,speed), minimum)
            present = np.isfinite(players[:, teams == team]).all(axis=2)
            known = ball_motion_valid & present.any(axis=1)
            fraction = np.divide(np.isfinite(team_times).sum(axis=1),
                np.maximum(1,present.sum(axis=1)))
            add("team%d_intercept_fraction_c%g" % (team,speed), np.where(known,fraction,np.nan))
        add("tti_advantage_home_c%g" % speed, minima[1]-minima[0])

    for horizon in HORIZONS_SECONDS:
        ball_future = current_ball + ball_velocity*horizon
        players_future = players + player_velocity*horizon
        separation = np.linalg.norm(players_future-ball_future[:,None,:],axis=-1)
        target_distance = np.linalg.norm(players-ball_future[:,None,:],axis=-1)
        cv_minima, target_minima = [], []
        for team in (0,1):
            mask = teams == team
            cv = _minimum_finite(separation[:,mask])
            distance = target_distance[:,mask]
            target = _minimum_finite(distance)
            cv_minima.append(cv);target_minima.append(target)
            add("team%d_cv_separation_h%g" % (team,horizon),cv)
            add("team%d_future_target_distance_h%g" % (team,horizon),target)
            known = np.isfinite(distance)
            for speed in REACH_SPEEDS:
                # Signed margin: negative means at least one player can reach
                # the projected ball point within this horizon under this bound.
                add("team%d_reach_margin_h%g_c%g" % (team,horizon,speed),target-speed*horizon)
                fraction = np.divide((known & (distance <= speed*horizon)).sum(axis=1),
                    np.maximum(1,known.sum(axis=1)))
                add("team%d_reach_fraction_h%g_c%g" % (team,horizon,speed),
                    np.where(known.any(axis=1),fraction,np.nan))
        add("cv_advantage_home_h%g" % horizon,cv_minima[1]-cv_minima[0])
        add("target_advantage_home_h%g" % horizon,target_minima[1]-target_minima[0])
    result = np.stack(columns,axis=1)
    result[~np.isfinite(result)] = np.nan
    return result.astype(np.float32),names


def reachability_features(data):
    """Return n-by-f features and names from geometry only.

    Positive advantage_home means home has the smaller distance/interception
    time. A team with missing tracks is unknown, never silently assigned away.
    Future constant-velocity separation also needs valid player velocity, while
    fixed-speed ball-point reachability needs only current player positions.
    """
    fps=float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    ball=np.asarray(data["ball"],dtype=float)
    players=np.asarray(data["players"],dtype=float)
    teams=np.asarray(data["team_of_track"])
    if (ball.ndim != 2 or ball.shape[1] != 2 or players.ndim != 3
        or players.shape[0] != len(ball) or players.shape[2] != 2
        or teams.shape != (players.shape[1],)):
        raise ValueError("invalid ball/player/team dimensions")
    if set(np.unique(teams)) != {0,1}:
        raise ValueError("expected both home=0 and away=1 tracks")
    result,names=None,None
    for start,end in blocks_of(data):
        feature,block_names=_block_features(ball[start:end],players[start:end],teams,fps)
        if result is None:
            names=block_names
            result=np.full((len(ball),feature.shape[1]),np.nan,dtype=np.float32)
        elif names != block_names:
            raise ValueError("inconsistent reachability schema")
        result[start:end]=feature
    if result is None:
        raise ValueError("empty recording")
    return result,names
