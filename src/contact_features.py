"""Past/future potential contacts, with no labels or identity assumptions.

Multiple radii/speed gates are features, not manually selected target decisions.
The external classifier decides which evidence is useful. Times reset per block.
"""
import numpy as np
from .features import blocks_of, speed_xy, interpolate_short


def contact_history_features(data):
    fps = float(data["fps"])
    team = np.asarray(data["team_of_track"])
    rows, names = [], []
    result = None
    for a, b in blocks_of(data):
        ball = interpolate_short(data["ball"][a:b], max(1, round(.2 * fps)))
        players = data["players"][a:b]
        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
        nearest = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        minimum = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        nearest_team = team[nearest]
        speed = speed_xy(ball, fps, max(1, round(.12 * fps)))
        frames = np.arange(b-a)
        columns, names = [], []
        for radius in (.7, 1.3, 2.2):
            for maximum_speed in (5., 12., 45.):
                contact = (minimum < radius) & np.isfinite(speed) & (speed < maximum_speed)
                for t in (0, 1):
                    is_contact = contact & (nearest_team == t)
                    previous = np.maximum.accumulate(np.where(is_contact, frames, -1000000))
                    following = np.minimum.accumulate(np.where(is_contact, frames, 1000000)[::-1])[::-1]
                    before = np.minimum((frames-previous)/fps, 20.)
                    after = np.minimum((following-frames)/fps, 20.)
                    for label, values in (("since", before), ("until", after)):
                        columns.append(values)
                        names.append("team%d_contact_r%g_v%g_%s" % (t, radius, maximum_speed, label))
        x = np.stack(columns, axis=1).astype(np.float32)
        if result is None:
            result = np.full((len(data["ball"]), x.shape[1]), np.nan, dtype=np.float32)
        result[a:b] = x
    return result, names
