"""G5: ball-gap context features (feature profile "contact_gap" = contact HGB180 + these).

Offline labelling, so frames after the current one may be used. Team-level features keep the
team0_/team1_ naming so src.symmetry.swap_feature_matrix exchanges them exactly. Outside a gap
every value is 0 and gap_inside is 0. src/features.py schemas are untouched.
"""
from __future__ import annotations

import numpy as np

from src.features import blocks_of

NAMES = ["gap_inside", "gap_elapsed_s", "gap_total_s", "gap_remaining_s",
         "team0_gap_dist_before", "team1_gap_dist_before", "team0_gap_dist_after", "team1_gap_dist_after",
         "gap_displacement", "gap_speed_before", "gap_boundary_before", "gap_players_speed"]


def _nearest(players, team, ball, t, which):
    p = players[t][team == which]
    d = np.linalg.norm(p - ball, axis=1)
    d = d[np.isfinite(d)]
    return float(d.min()) if len(d) else 50.


def gap_context(geo):
    ball = np.asarray(geo["ball"], float)
    players = np.asarray(geo["players"], float)
    team = np.asarray(geo["team_of_track"])
    pitch = np.asarray(geo.get("pitch", [105., 68.]), float)
    fps = float(geo["fps"])
    n = len(ball)
    x = np.zeros((n, len(NAMES)), dtype=np.float32)
    missing = ~np.isfinite(ball).all(1)
    step = np.linalg.norm(np.diff(players, axis=0), axis=2) * fps
    with np.errstate(all="ignore"):
        pspeed = np.r_[0., np.nan_to_num(np.nanmean(step, axis=1))]
    lag = max(1, int(.4 * fps))
    for a, b in blocks_of(geo):
        m = missing[a:b].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            s, e = a + s, a + e
            total = (e - s) / fps
            idx = np.arange(s, e)
            x[idx, 0] = 1.
            x[idx, 1] = (idx - s) / fps
            x[idx, 2] = total
            x[idx, 3] = (e - idx) / fps
            before = s - 1 if s > a else None
            after = e if e < b else None
            if before is not None:
                bb = ball[before]
                x[idx, 4] = _nearest(players, team, bb, before, 0)
                x[idx, 5] = _nearest(players, team, bb, before, 1)
                prev = max(a, before - lag)
                if np.isfinite(ball[prev]).all():
                    x[idx, 9] = np.linalg.norm(bb - ball[prev]) / ((before - prev) / fps or 1.)
                x[idx, 10] = min(bb[0], pitch[0] - bb[0], bb[1], pitch[1] - bb[1])
            if after is not None:
                ba = ball[after]
                x[idx, 6] = _nearest(players, team, ba, after, 0)
                x[idx, 7] = _nearest(players, team, ba, after, 1)
            if before is not None and after is not None:
                x[idx, 8] = np.linalg.norm(ball[after] - ball[before])
            else:
                x[idx, 8] = 20.
            x[idx, 11] = float(pspeed[s:e].mean())
    return x, list(NAMES)
