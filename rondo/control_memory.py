"""Round 10: "last control" memory features (tracking only; no labels, no tags).

A frame is a CONTROL frame of team t when, with the ball visible,
  * the nearest player to the ball is from team t and within `radius` metres,
  * the ball moves with that player: |v_ball - v_player| <= `rel_speed` m/s,
  * the same track satisfies both for at least `min_s` seconds without interruption.
Touches, deflections and a ball passing close by do not count (CHALLENGE.md: possession changes
only on *controlled* possession; while the ball is loose it stays with the last team that had it).

Features per frame (signed: +1 = team 0 / home, -1 = team 1 / away, 0 = none):
  ctrl_now            signed control at this frame
  last_ctrl           signed team of the last control frame at or before this frame (same half)
  t_since_last        seconds since that control frame (capped)
  last_ctrl_dur       duration of that last control run (s, capped)
  next_ctrl           signed team of the next control frame after this frame (same half)
  t_to_next           seconds to it (capped)
  switches_pm2s       control changes of team within +-2 s
  contested           both teams have a player within `contest_m` of the visible ball
  ctrl_past1s/fut1s   mean signed control over the previous / next 1 s
  ctrl_past3s/fut3s   same over 3 s
Offline: future frames are allowed (the whole match is available and labels are defined
retrospectively). SIGNED lists the columns whose sign flips when home and away are exchanged.
"""
from __future__ import annotations

import numpy as np

NAMES = ["ctrl_now", "last_ctrl", "t_since_last", "last_ctrl_dur", "next_ctrl", "t_to_next",
         "switches_pm2s", "contested", "ctrl_past1s", "ctrl_fut1s", "ctrl_past3s", "ctrl_fut3s"]
SIGNED = [0, 1, 4, 8, 9, 10, 11]
DEFAULT = dict(radius=1.5, rel_speed=4.0, min_s=.3, contest_m=2.0)
CAP_S = 10.


def _velocity(xy, lag):
    v = np.full_like(xy, np.nan)
    v[lag:] = (xy[lag:] - xy[:-lag]) / 1.
    return v


def _blocks(geo, n):
    b = geo.get("blocks")
    if b is None or len(b) == 0:
        return [(0, n)]
    return [(int(a), int(e)) for a, e in np.asarray(b)]


def control_frames(geo, radius, rel_speed, min_s, contest_m):
    """(signed control per frame in {-1,0,1}, contested flag)."""
    ball = np.asarray(geo["ball"], float)
    P = np.asarray(geo["players"], float)
    team = np.asarray(geo["team_of_track"]).astype(int)
    fps = float(geo["fps"])
    n, k = P.shape[:2]
    lag = max(1, int(round(.12 * fps)))
    vb = _velocity(ball, lag) * fps / lag
    vp = _velocity(P.reshape(n, -1), lag).reshape(n, k, 2) * fps / lag
    raw_j = np.full(n, -1, np.int64)
    contested = np.zeros(n, bool)
    for s in range(0, n, 20000):
        e = min(n, s + 20000)
        d = np.linalg.norm(P[s:e] - ball[s:e, None, :], axis=2)
        dd = np.where(np.isfinite(d), d, np.inf)
        j = np.argmin(dd, axis=1)
        dmin = dd[np.arange(e - s), j]
        rv = np.linalg.norm(vb[s:e] - vp[s:e][np.arange(e - s), j], axis=1)
        ok = (dmin <= radius) & np.isfinite(rv) & (rv <= rel_speed)
        raw_j[s:e] = np.where(ok, j, -1)
        near0 = np.where(team == 0, dd, np.inf).min(1) if (team == 0).any() else np.full(e - s, np.inf)
        near1 = np.where(team == 1, dd, np.inf).min(1) if (team == 1).any() else np.full(e - s, np.inf)
        contested[s:e] = (near0 <= contest_m) & (near1 <= contest_m)
    # keep runs of the same track that last >= min_s
    kmin = max(1, int(round(min_s * fps)))
    ctrl = np.zeros(n, np.int8)
    edges = np.r_[0, np.flatnonzero(raw_j[1:] != raw_j[:-1]) + 1, n]
    for a, b in zip(edges[:-1], edges[1:]):
        j = raw_j[a]
        if j >= 0 and b - a >= kmin and team[j] in (0, 1):
            ctrl[a:b] = 1 if team[j] == 0 else -1
    return ctrl, contested


def _ffill_index(mask):
    """index of the last True at or before i (-1 if none)."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def _bfill_index(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _window_mean(x, w, future):
    n = len(x)
    c = np.r_[0., np.cumsum(x)]
    i = np.arange(n)
    lo, hi = (i + 1, np.minimum(n, i + 1 + w)) if future else (np.maximum(0, i - w), i)
    return (c[hi] - c[lo]) / np.maximum(1, hi - lo)


def control_memory(geo, radius=DEFAULT["radius"], rel_speed=DEFAULT["rel_speed"], min_s=DEFAULT["min_s"],
                   contest_m=DEFAULT["contest_m"]):
    fps = float(geo["fps"])
    n = len(geo["ball"])
    ctrl, contested = control_frames(geo, radius, rel_speed, min_s, contest_m)
    x = np.zeros((n, len(NAMES)), np.float32)
    x[:, 0] = ctrl
    x[:, 7] = contested
    for a, b in _blocks(geo, n):
        c = ctrl[a:b].astype(float)
        m = c != 0
        if b <= a:
            continue
        li = _ffill_index(m)
        ni = _bfill_index(m)
        i = np.arange(b - a)
        has_l, has_n = li >= 0, ni < (b - a)
        x[a:b, 1] = np.where(has_l, c[np.maximum(li, 0)], 0)
        x[a:b, 2] = np.where(has_l, np.minimum((i - li) / fps, CAP_S), CAP_S)
        x[a:b, 4] = np.where(has_n, c[np.minimum(ni, b - a - 1)], 0)
        x[a:b, 5] = np.where(has_n, np.minimum((ni - i) / fps, CAP_S), CAP_S)
        # duration of the control run that the last control frame belongs to
        run_start = np.zeros(b - a, np.int64)
        e = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1]
        run_start[e] = e
        run_start = np.maximum.accumulate(run_start)
        dur = np.zeros(b - a)
        ends = np.r_[e[1:], b - a]
        for s_, t_ in zip(e, ends):
            dur[s_:t_] = (t_ - s_) / fps
        x[a:b, 3] = np.where(has_l, np.minimum(dur[np.maximum(li, 0)], 5.), 0)
        # team changes between consecutive control runs, counted within +-2 s
        teams = c[m]
        pos = np.flatnonzero(m)
        sw = np.zeros(b - a)
        if len(teams) > 1:
            ch = np.flatnonzero(teams[1:] != teams[:-1]) + 1
            sw[pos[ch]] = 1
        w2 = int(2 * fps)
        cs = np.r_[0., np.cumsum(sw)]
        x[a:b, 6] = cs[np.minimum(b - a, i + 1 + w2)] - cs[np.maximum(0, i - w2)]
        for col, sec, fut in ((8, 1, False), (9, 1, True), (10, 3, False), (11, 3, True)):
            x[a:b, col] = _window_mean(c, int(sec * fps), fut)
    return x


def swap_cm(x):
    y = x.copy()
    y[:, SIGNED] = -x[:, SIGNED]
    return y
