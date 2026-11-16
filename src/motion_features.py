"""Label-blind contact/co-motion features; coordinates are metres, fps is Hz.

Unknown velocity remains NaN, stationary velocity is zero. No labels, status,
video, fitting or calibration enter this module. All operations reset at blocks.
"""
import numpy as np
from .features import blocks_of, interpolate_short, smooth_nan

MOTION_FEATURE_VERSION = "contact-motion-v1"
LAGS_SECONDS = (.12, .24, .48)
CONTEXT_SECONDS = (.4, 1.2)


def _velocity(xy, fps, lag, maximum, direction="centered"):
    """Velocity across fully observed support, with physical clipping.

    xy accepts (frames,2) or (frames,tracks,2). Short interpolated ball gaps
    count as observed here; unknown long gaps/player gaps cannot be bridged.
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    out = np.full_like(xy, np.nan)
    if n <= lag:
        return out
    if direction == "incoming":
        current = np.arange(lag, n)
        first, last, elapsed = current-lag, current, lag/fps
    elif direction == "outgoing":
        current = np.arange(n-lag)
        first, last, elapsed = current, current+lag, lag/fps
    elif direction == "centered":
        if n <= 2*lag:
            return out
        current = np.arange(lag, n-lag)
        first, last, elapsed = current-lag, current+lag, 2*lag/fps
    else:
        raise ValueError("invalid velocity direction")
    observed = np.isfinite(xy).all(axis=-1)
    prefix = np.concatenate([np.zeros_like(observed[:1], dtype=np.int64),
                             np.cumsum(observed, axis=0, dtype=np.int64)], axis=0)
    support = prefix[last+1]-prefix[first]
    expected = (last-first+1).reshape((-1,)+(1,)*(observed.ndim-1))
    with np.errstate(invalid="ignore", over="ignore"):
        vector = (xy[last]-xy[first])/elapsed
        speed = np.linalg.norm(vector, axis=-1)
    valid = (support == expected) & np.isfinite(speed) & (speed <= maximum)
    out[current] = np.where(valid[..., None], vector, np.nan)
    return out


def _norm(vector):
    return np.linalg.norm(vector, axis=-1)


def _cosine(a, b):
    denom = _norm(a)*_norm(b)
    cosine = np.divide(np.sum(a*b, axis=-1), denom,
        out=np.full(denom.shape, np.nan), where=np.isfinite(denom) & (denom > 1e-8))
    return np.clip(cosine, -1., 1.)


def _motion_block(ball, players, teams, fps):
    n = len(ball)
    b = interpolate_short(ball, max(1, round(.2*fps)))
    distance = _norm(players-b[:, None, :])
    nearest = {}
    for team in (0, 1):
        indices = np.flatnonzero(teams == team)
        local = np.where(np.isfinite(distance[:, indices]), distance[:, indices], np.inf)
        chosen_local = np.argmin(local, axis=1)
        chosen = indices[chosen_local]
        valid = np.isfinite(local[np.arange(n), chosen_local])
        point = players[np.arange(n), chosen].copy()
        point[~valid] = np.nan
        nearest[team] = chosen, valid, point
    columns, names = [], []
    def add(name, values):
        names.append("motion_"+name)
        columns.append(np.asarray(values, dtype=float))
    add("ball_observed", np.isfinite(ball).all(axis=1))
    add("ball_available", np.isfinite(b).all(axis=1))
    for seconds in LAGS_SECONDS:
        lag = max(1, round(fps*seconds))
        suffix = "_dt%g" % seconds
        incoming = _velocity(b, fps, lag, 45., "incoming")
        outgoing = _velocity(b, fps, lag, 45., "outgoing")
        in_speed, out_speed = _norm(incoming), _norm(outgoing)
        add("ball_in_speed"+suffix, in_speed)
        add("ball_out_speed"+suffix, out_speed)
        add("ball_turn_cosine"+suffix, _cosine(incoming, outgoing))
        add("ball_speed_change"+suffix, out_speed-in_speed)
        add("ball_vector_change"+suffix, _norm(outgoing-incoming))
        add("ball_turn_velocity_valid"+suffix,
            np.isfinite(incoming).all(axis=1) & np.isfinite(outgoing).all(axis=1))
        ball_velocity = _velocity(b, fps, lag, 45.)
        player_velocity = _velocity(players, fps, lag, 14.)
        for team in (0, 1):
            chosen, valid, point = nearest[team]
            velocity = player_velocity[np.arange(n), chosen].copy()
            velocity[~valid] = np.nan
            relative, separation = velocity-ball_velocity, point-b
            length = _norm(separation)
            radial = np.divide(np.sum(relative*separation, axis=1), length,
                out=np.full(n, np.nan), where=np.isfinite(length) & (length > 1e-8))
            name = "team%d_" % team
            add(name+"candidate_relative_speed"+suffix, _norm(relative))
            add(name+"candidate_heading_cosine"+suffix, _cosine(velocity, ball_velocity))
            add(name+"candidate_radial_separation_speed"+suffix, radial)
            add(name+"candidate_speed"+suffix, _norm(velocity))
            add(name+"candidate_velocity_valid"+suffix,
                np.isfinite(velocity).all(axis=1) & np.isfinite(ball_velocity).all(axis=1))
    base = np.stack(columns, axis=1)
    base[~np.isfinite(base)] = np.nan
    pieces, all_names = [base], list(names)
    for seconds in CONTEXT_SECONDS:
        width = max(1, round(seconds*fps)) | 1
        pieces.append(smooth_nan(base, width))
        all_names += [name+"_mean%g" % seconds for name in names]
    return np.concatenate(pieces, axis=1).astype(np.float32), all_names


def motion_features(data):
    """Return (n_frames x n_features array, ordered names), without labels.

    Keys: ball[n,2], players[n,tracks,2], team_of_track[tracks], fps. Boundaries
    follow src.features.blocks_of; non-block rows stay NaN. Relative speed is
    |v_player-v_ball|; radial separation is positive when moving apart. Heading
    cosine is undefined (NaN) for stationary vectors, rather than invented zero.
    """
    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    ball = np.asarray(data["ball"], dtype=float)
    players = np.asarray(data["players"], dtype=float)
    teams = np.asarray(data["team_of_track"])
    if (ball.ndim != 2 or ball.shape[1] != 2 or players.ndim != 3
            or players.shape[0] != len(ball) or players.shape[2] != 2
            or teams.shape != (players.shape[1],)):
        raise ValueError("invalid ball/player/team dimensions")
    if set(np.unique(teams)) != {0, 1}:
        raise ValueError("expected both home=0 and away=1 tracks")
    result, names = None, None
    for start, end in blocks_of(data):
        feature, block_names = _motion_block(ball[start:end], players[start:end], teams, fps)
        if result is None:
            names = block_names
            result = np.full((len(ball), feature.shape[1]), np.nan, dtype=np.float32)
        elif block_names != names:
            raise ValueError("inconsistent motion schema")
        result[start:end] = feature
    if result is None:
        raise ValueError("empty recording")
    return result, names
