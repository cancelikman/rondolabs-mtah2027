"""Common geometric features: no possession/status fields may enter this module.

Coordinates in metres, pitch origin at a corner; NaNs denote unknown observations.
Context windows and interpolation never cross a recording/period boundary.
"""
import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d

FEATURE_VERSION = "geometry-context-v1"


def blocks_of(data):
    n = len(data["ball"])
    if "blocks" in data:
        blocks = np.asarray(data["blocks"], dtype=int).reshape(-1, 2)
    else:
        period = np.asarray(data.get("period", np.zeros(n)))
        times = np.asarray(data.get("timestamps", np.arange(n) / float(data["fps"])))
        cuts = np.flatnonzero((period[1:] != period[:-1]) | (np.diff(times) <= 0)
                              | (np.diff(times) > 1.5 / float(data["fps"]))) + 1
        edges = np.r_[0, cuts, n]
        blocks = np.c_[edges[:-1], edges[1:]]
    if np.any(blocks[:, 0] < 0) or np.any(blocks[:, 1] > n) or np.any(blocks[:, 1] <= blocks[:, 0]):
        raise ValueError("invalid half-open blocks")
    if len(blocks) > 1 and np.any(blocks[1:, 0] < blocks[:-1, 1]):
        raise ValueError("overlapping blocks")
    return blocks


def interpolate_short(values, max_gap):
    out = np.asarray(values, dtype=float).copy()
    good = np.isfinite(out).all(axis=-1)
    indices = np.flatnonzero(good)
    for a, b in zip(indices[:-1], indices[1:]):
        if 1 < b - a <= max_gap + 1:
            weight = np.arange(1, b - a) / float(b - a)
            out[a + 1:b] = out[a] + weight[:, None] * (out[b] - out[a])
    return out


def speed_xy(xy, fps, lag=5, max_speed=45):
    """Centred displacement, missing if either endpoint is absent/implausible."""
    out = np.full(xy.shape[:-1], np.nan)
    if len(xy) > 2 * lag:
        out[lag:-lag] = np.linalg.norm(xy[2 * lag:] - xy[:-2 * lag], axis=-1) * fps / (2 * lag)
    out[out > max_speed] = np.nan
    return out


def smooth_nan(x, width):
    valid = np.isfinite(x)
    denom = uniform_filter1d(valid.astype(float), width, axis=0, mode="nearest")
    numer = uniform_filter1d(np.where(valid, x, 0), width, axis=0, mode="nearest")
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 1e-7)


def _base(ball, players, team, fps, pitch):
    visible = np.isfinite(ball).all(axis=1)
    b = interpolate_short(ball, max(1, round(fps * .2)))
    distance = np.linalg.norm(players - b[:, None, :], axis=-1)
    player_speeds = speed_xy(players, fps, max(1, round(fps * .2)), 14)
    ball_speed = speed_xy(b, fps, max(1, round(fps * .12)), 45)
    columns, names = [], []
    def add(name, values):
        names.append(name)
        columns.append(np.asarray(values, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        add("ball_visible", visible)
        add("ball_interpolated", np.isfinite(b).all(axis=1))
        add("ball_speed", ball_speed)
        add("ball_speed_valid", np.isfinite(ball_speed))
        add("ball_x_fraction", b[:, 0] / pitch[0])
        add("ball_y_fraction", b[:, 1] / pitch[1])
        add("boundary_distance", np.min(np.c_[b[:, 0], pitch[0]-b[:, 0], b[:, 1], pitch[1]-b[:, 1]], axis=1))
        add("ball_outside", np.where(np.isfinite(b).all(axis=1),
                                     ((b < 0) | (b > pitch)).any(axis=1), np.nan))
        mins = []
        for t in (0, 1):
            mask = team == t
            d = distance[:, mask]
            p = players[:, mask]
            s = player_speeds[:, mask]
            ordered = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
            for k in range(3):
                v = ordered[:, k] if ordered.shape[1] > k else np.full(len(b), np.inf)
                add("team%d_distance%d" % (t, k+1), np.where(np.isfinite(v), np.minimum(v, 40), np.nan))
            mins.append(ordered[:, 0])
            add("team%d_visible" % t, np.isfinite(p).all(axis=2).sum(axis=1))
            add("team%d_speed_mean" % t, np.nanmean(s, axis=1))
            add("team%d_speed_median" % t, np.nanmedian(s, axis=1))
            add("team%d_speed_fast_fraction" % t, np.divide(np.sum(s > 3, axis=1),
                np.maximum(1, np.isfinite(s).sum(axis=1))))
            add("team%d_spread_x" % t, np.nanstd(p[:, :, 0], axis=1))
            add("team%d_spread_y" % t, np.nanstd(p[:, :, 1], axis=1))
            for radius in (2, 5, 10):
                add("team%d_near%d" % (t, radius), np.where(np.isfinite(b).all(axis=1), (d < radius).sum(axis=1), np.nan))
        delta = np.clip(mins[1], 0, 40) - np.clip(mins[0], 0, 40)
        delta[~np.isfinite(b).all(axis=1)] = np.nan
        add("distance_advantage_home", delta)
        add("nearest_distance", np.fmin(mins[0], mins[1]))
        add("players_speed_mean", np.nanmean(player_speeds, axis=1))
        add("players_speed_q75", np.nanpercentile(player_speeds, 75, axis=1))
    result = np.stack(columns, axis=1)
    result[~np.isfinite(result)] = np.nan
    return result, names


def build_features(data, profile="geometry"):
    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    ball, players = np.asarray(data["ball"]), np.asarray(data["players"])
    team = np.asarray(data["team_of_track"])
    pitch = np.asarray(data["pitch"])
    if players.shape[:1] != ball.shape[:1] or players.shape[1] != len(team):
        raise ValueError("ball/player/team dimensions disagree")
    if set(np.unique(team)) != {0, 1}:
        raise ValueError("expected both home=0 and away=1 tracks")
    output = None
    all_names = None
    for a, b in blocks_of(data):
        base, names = _base(ball[a:b], players[a:b], team, fps, pitch)
        pieces = [base]
        all_names = list(names)
        # Symmetric context is permitted: submission is offline, not a realtime system.
        for seconds in (.4, 1.2, 3.0):
            width = max(1, round(seconds * fps)) | 1
            pieces.append(smooth_nan(base, width))
            all_names += [name + "_mean%g" % seconds for name in names]
        feature = np.concatenate(pieces, axis=1).astype(np.float32)
        if output is None:
            output = np.full((len(ball), feature.shape[1]), np.nan, dtype=np.float32)
        output[a:b] = feature
    if output is None:
        raise ValueError("empty recording")
    if profile in ("contact", "contact_motion", "contact_reach", "contact_shape", "contact_motion_shape"):
        from .contact_features import contact_history_features
        contact, contact_names = contact_history_features(data)
        output = np.concatenate([output, contact], axis=1)
        all_names += contact_names
        if profile in ("contact_motion", "contact_motion_shape"):
            from .motion_features import motion_features
            motion, motion_names = motion_features(data)
            output = np.concatenate([output, motion], axis=1)
            all_names += motion_names
        if profile == "contact_reach":
            from .reachability_features import reachability_features
            reach, reach_names = reachability_features(data)
            output = np.concatenate([output, reach], axis=1)
            all_names += reach_names
        if profile in ("contact_shape", "contact_motion_shape"):
            from .team_shape_features import team_shape_features
            shape, shape_names = team_shape_features(data)
            output = np.concatenate([output, shape], axis=1)
            all_names += shape_names
    elif profile != "geometry":
        raise ValueError("unknown feature profile")
    return output, all_names


def swap_sides(data):
    result = dict(data)
    result["team_of_track"] = 1 - np.asarray(data["team_of_track"])
    if "labels" in data:
        labels = np.asarray(data["labels"]).copy()
        labels[data["labels"] == 0] = 1
        labels[data["labels"] == 1] = 0
        result["labels"] = labels
    return result


def occlude_external(data, seed, probability=.35, mean_seconds=1.5):
    """External-only synthetic occlusions; never fit to target missingness."""
    rng = np.random.default_rng(seed)
    result = dict(data)
    ball = np.array(data["ball"], copy=True)
    rate = 1 / (float(data["fps"]) * mean_seconds)
    for a, b in blocks_of(data):
        hidden = False
        for i in range(a, b):
            if rng.random() < rate:
                hidden = bool(rng.random() < probability)
            if hidden:
                ball[i] = np.nan
    result["ball"] = ball
    return result
