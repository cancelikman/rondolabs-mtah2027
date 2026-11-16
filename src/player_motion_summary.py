"""Ball-free anonymous team movement summaries for the external TCN trial."""
import warnings

import numpy as np

from src.features import blocks_of
from src.motion_features import _velocity


def player_motion_summary(data):
    players = np.asarray(data["players"])
    teams = np.asarray(data["team_of_track"])
    pitch = np.asarray(data["pitch"])
    fps = float(data["fps"])
    names = []
    groups = []
    for team in (0, 1):
        names.extend(f"team{team}_player_motion_{name}" for name in (
            "fast_x", "fast_y", "fast_spread_x", "fast_spread_y",
            "positive_x_fraction", "positive_y_fraction", "velocity_alignment",
            "speed_q25", "speed_q75", "speed_q90",
        ))
        group = np.full((len(players), 10), np.nan, dtype=np.float32)
        for start, stop in blocks_of(data):
            p = players[start:stop, teams == team]
            velocity = _velocity(p, fps, max(1, round(.24 * fps)), 14., direction="incoming")
            speed = np.linalg.norm(velocity, axis=2)
            known = np.isfinite(speed)
            fast = speed >= 3.
            position = np.where(fast[:, :, None], p / pitch - .5, np.nan)
            denominator = np.maximum(1, known.sum(axis=1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                center = np.nanmean(position, axis=1)
                spread = np.nanstd(position, axis=1)
                directional = ((velocity > 0) & known[:, :, None]).sum(axis=1) / denominator[:, None]
                directional[~known.any(axis=1)] = np.nan
                norm = np.nansum(speed, axis=1)
                coherence = np.divide(
                    np.linalg.norm(np.nansum(velocity, axis=1), axis=1), norm,
                    out=np.full(len(norm), np.nan), where=norm > 0,
                )
                percentiles = np.nanpercentile(speed, [25, 75, 90], axis=1).T / 14.
            group[start:stop] = np.column_stack((center, spread, directional, coherence, percentiles))
        groups.append(group)
    return np.column_stack(groups), names
