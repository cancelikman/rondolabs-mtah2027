"""General rule layer (S6/S7). Every rule applies to the whole match, no frame constants.

Evidence rules return a boolean mask of frames where play is probably stopped; the
mask multiplies P(D) before decoding. Post rules edit the decoded H/A/D sequence.
All thresholds are chosen on external matches only.
"""
from __future__ import annotations

import numpy as np

from src.features import blocks_of

H, A, D = 0, 1, 2


def _runs(x):
    x = np.asarray(x)
    if not len(x):
        return []
    edges = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
    return [(int(s), int(e), int(x[s])) for s, e in zip(edges[:-1], edges[1:])]


def _sustain(mask, frames):
    """True only inside runs of `mask` lasting at least `frames`."""
    out = np.zeros(len(mask), dtype=bool)
    for s, e, v in _runs(mask.astype(np.int8)):
        if v and e - s >= frames:
            out[s:e] = True
    return out


# ---------------------------------------------------------------- evidence rules

def kickoff_mask(geo, own_share=.92, centre_radius=2.5, min_players=16, hold_seconds=1.):
    """Both teams almost entirely in their own half and a player on the centre spot."""
    players = np.asarray(geo["players"], float)
    team = np.asarray(geo["team_of_track"])
    pitch = np.asarray(geo.get("pitch", [105., 68.]), float)
    fps = float(geo["fps"])
    cx, cy = pitch[0] / 2, pitch[1] / 2
    out = np.zeros(len(players), dtype=bool)
    for a, b in blocks_of(geo):
        x = players[a:b, :, 0]
        ok = np.isfinite(x)
        # which side each team defends in this block: sign of its mean x
        side = {}
        for t in (0, 1):
            cols = team == t
            side[t] = np.sign(np.nanmean(x[:, cols]) - cx) or 1.
        shares, count = [], ok.sum(1)
        for t in (0, 1):
            cols = team == t
            own = ((x[:, cols] - cx) * side[t] > -1.) & ok[:, cols]   # 1 m tolerance on the line
            shares.append(own.sum(1) / np.maximum(1, ok[:, cols].sum(1)))
        xy = players[a:b]
        dc = np.sqrt((xy[..., 0] - cx) ** 2 + (xy[..., 1] - cy) ** 2)
        near = np.nanmin(np.where(np.isfinite(dc), dc, np.inf), axis=1) < centre_radius
        formed = (shares[0] >= own_share) & (shares[1] >= own_share) & (count >= min_players)
        # keep formation runs that contain a centre-spot player at some point
        m = np.zeros(b - a, dtype=bool)
        for s, e, v in _runs(formed.astype(np.int8)):
            if v and near[s:e].any():
                m[s:e] = True
        out[a:b] = _sustain(m, int(hold_seconds * fps))
    return out


def abandoned_mask(geo, radius=6., seconds=2.):
    """Visible ball with nobody within `radius` m for `seconds`."""
    ball = np.asarray(geo["ball"], float)
    players = np.asarray(geo["players"], float)
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    nearest = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    lonely = np.isfinite(ball).all(1) & (nearest > radius) & np.isfinite(nearest)
    return _sustain(lonely, int(seconds * float(geo["fps"])))


def gap_stoppage_mask(geo, seconds=3., radius=10.):
    """Ball missing >= `seconds` and reappearing within `radius` m of where it vanished."""
    ball = np.asarray(geo["ball"], float)
    fps = float(geo["fps"])
    missing = ~np.isfinite(ball).all(1)
    out = np.zeros(len(ball), dtype=bool)
    for a, b in blocks_of(geo):
        for s, e, v in _runs(missing[a:b].astype(np.int8)):
            s, e = a + s, a + e
            if not v or e - s < seconds * fps or s == a or e >= b:
                continue
            if np.linalg.norm(ball[e] - ball[s - 1]) <= radius:
                out[s:e] = True
    return out


EVIDENCE = {"kickoff": kickoff_mask, "abandoned": abandoned_mask, "gap": gap_stoppage_mask}


def apply_evidence(state, geo, names, factor=20.):
    state = np.array(state, float)
    if not names:
        return state
    mask = np.zeros(len(state), dtype=bool)
    for n in names:
        mask |= EVIDENCE[n](geo)
    mask &= np.isfinite(state).all(1)
    state[mask, 2] *= factor
    state[mask] /= state[mask].sum(1, keepdims=True)
    return state


# ---------------------------------------------------------------- post rules

def merge_short_possessions(pred, geo, seconds=2.):
    """A-B-A with a short B (< seconds) becomes A, unless B directly follows a stoppage."""
    out = np.array(pred, copy=True)
    frames = seconds * float(geo["fps"])
    for a, b in blocks_of(geo):
        changed = True
        while changed:
            changed = False
            runs = _runs(out[a:b])
            for i in range(1, len(runs) - 1):
                (ps, pe, pv), (s, e, v), (ns, ne, nv) = runs[i - 1], runs[i], runs[i + 1]
                if v in (H, A) and pv == nv and pv in (H, A) and pv != v and e - s < frames:
                    out[a + s:a + e] = pv
                    changed = True
                    break
    return out


def min_stoppage(pred, geo, seconds=3.):
    """No D run shorter than `seconds` between two live runs."""
    out = np.array(pred, copy=True)
    frames = seconds * float(geo["fps"])
    for a, b in blocks_of(geo):
        runs = _runs(out[a:b])
        for i in range(1, len(runs) - 1):
            (ps, pe, pv), (s, e, v), (ns, ne, nv) = runs[i - 1], runs[i], runs[i + 1]
            if v == D and e - s < frames and pv in (H, A) and nv in (H, A):
                if pv == nv:
                    out[a + s:a + e] = pv
                else:
                    mid = (s + e) // 2
                    out[a + s:a + mid], out[a + mid:a + e] = pv, nv
    return out


def min_possession(pred, geo, seconds):
    """Round 5 H1: no H/A run shorter than `seconds` inside live play.

    A short run whose previous run is a possession is given back to that previous possession
    (the false switch is undone). A short run right after a stoppage (previous run D, or the
    block start) is the restart possession and is kept."""
    out = np.array(pred, copy=True)
    if not seconds:
        return out
    frames = seconds * float(geo["fps"])
    for a, b in blocks_of(geo):
        changed = True
        while changed:
            changed = False
            r = _runs(out[a:b])
            for i in range(1, len(r)):
                s, e, v = r[i]
                pv = r[i - 1][2]
                if v in (H, A) and pv in (H, A) and pv != v and e - s < frames:
                    out[a + s:a + e] = pv
                    changed = True
                    break
    return out


def gap_dead_runs(pred, geo, seconds):
    """[(start, end, team)] of D runs lying wholly inside one ball gap, flanked by the same team's
    possession on both sides and shorter than `seconds` (round 5 H4 candidates)."""
    ball = np.asarray(geo["ball"], float)
    missing = ~np.isfinite(ball).all(1)
    frames = seconds * float(geo["fps"])
    out = []
    for a, b in blocks_of(geo):
        r = _runs(pred[a:b])
        for i in range(1, len(r) - 1):
            s, e, v = r[i]
            if v != D or e - s >= frames:
                continue
            pv, nv = r[i - 1][2], r[i + 1][2]
            if pv == nv and pv in (H, A) and missing[a + s:a + e].all():
                out.append((a + s, a + e, pv))
    return out


def merge_gap_dead(pred, geo, seconds):
    """Round 5 H4: fill those D runs with the flanking team."""
    out = np.array(pred, copy=True)
    for s, e, team in gap_dead_runs(out, geo, seconds):
        out[s:e] = team
    return out


def published_halves(game="BAR-ATH"):
    """Scored halves in video frames, half-open, from games.py (organiser-published)."""
    from games import GAMES
    g = GAMES[game]
    return [(int(g["halves"][h][0] - g["offsets"][h]), int(g["halves"][h][1] - g["offsets"][h] + 1))
            for h in sorted(g["halves"])]


def tag_frames(tags, type_name, code="set pieces"):
    """round(start_s * 25) of Smart Tagging instances with this code and Type (case-insensitive)."""
    return sorted(int(t["start_frame"]) for t in tags
                  if (t.get("code") or "").strip().lower() == code
                  and (t.get("tags", {}).get("Type") or "").strip().lower() == type_name)


def apply_kickoff_tags(pred, blocks, kickoff_start_frames, max_lead_seconds=300, fps=25):
    """G1a: in each scored block, D from the block start to the first Kick Off (Start) tag start.
    Skipped when no tag falls in the block or it is more than max_lead_seconds after the start.
    A tag window always contains the real kick-off, so its start never lies after it."""
    out = np.array(pred, copy=True)
    for a, b in blocks:
        inside = [f for f in kickoff_start_frames if a <= f < b]
        if not inside:
            continue
        k = min(inside)
        if k - a > max_lead_seconds * fps:
            continue
        out[a:k] = D
    return out


def apply_after_goal(pred, blocks, after_goal_start_frames, seconds, fps=25):
    """G1b: D for `seconds` before each Kick Off (After Goal) tag start (within scored blocks).
    `seconds` is derived from IDSSE goal-to-kickoff times (rondo.derive_goal_kickoff)."""
    out = np.array(pred, copy=True)
    for k in after_goal_start_frames:
        for a, b in blocks:
            if a < k <= b:
                out[max(a, k - int(round(seconds * fps))):k] = D
    return out


def apply_set_piece_windows(pred, blocks, tags, seconds_by_type, fps=25):
    """Round 5 H2: D for T_type seconds before each SET PIECES tag start of that Type.
    T_type comes from IDSSE (rondo.derive_restart_windows); types without a T are skipped."""
    out = np.array(pred, copy=True)
    wanted = {k.strip().lower(): float(v) for k, v in seconds_by_type.items() if v}
    for t in tags:
        if (t.get("code") or "").strip().lower() != "set pieces":
            continue
        T = wanted.get((t.get("tags", {}).get("Type") or "").strip().lower())
        if not T:
            continue
        k = int(t["start_frame"])
        for a, b in blocks:
            if a < k <= b:
                out[max(a, k - int(round(T * fps))):k] = D
    return out


def apply_goal_to_kickoff(pred, blocks, tags, max_gap_seconds=120, fps=25):
    """Round 5 H5: from the end of the closest preceding GOALS tag to each Kick Off (After Goal)
    tag start, D. Without a GOALS tag within max_gap_seconds nothing changes here."""
    out = np.array(pred, copy=True)
    goals = sorted((int(t["start_frame"]), int(t["end_frame"])) for t in tags
                   if (t.get("code") or "").strip().lower() == "goals")
    for k in tag_frames(tags, "kick off (after goal)"):
        prev = [g for g in goals if g[0] < k and k - g[0] <= max_gap_seconds * fps]
        if not prev:
            continue
        ge = prev[-1][1]
        for a, b in blocks:
            if a < k <= b and ge < k:
                out[max(a, ge):k] = D
    return out


def apply_tag_rules(pred, spec, tags, blocks):
    """spec = manifest["post"]["kickoff_tags"], e.g. {"start": true, "after_goal_seconds": 29,
    "set_piece_seconds": {"Goal Kick": 2}, "goal_to_kickoff": true}."""
    if spec.get("set_piece_seconds"):
        pred = apply_set_piece_windows(pred, blocks, tags, spec["set_piece_seconds"])
    if spec.get("goal_to_kickoff"):
        pred = apply_goal_to_kickoff(pred, blocks, tags)
    if spec.get("start", True):
        pred = apply_kickoff_tags(pred, blocks, tag_frames(tags, "kick off (start)"),
                                  spec.get("max_lead_seconds", 300))
    if spec.get("after_goal_seconds"):
        pred = apply_after_goal(pred, blocks, tag_frames(tags, "kick off (after goal)"),
                                float(spec["after_goal_seconds"]))
    return pred


def apply_post(pred, geo, spec):
    """spec: dict like {"merge": 2.0, "minstop": 3.0}. Tag rules are applied by the caller."""
    out = pred
    if spec.get("min_possession"):
        out = min_possession(out, geo, spec["min_possession"])
    if spec.get("gap_dead_merge"):
        out = merge_gap_dead(out, geo, spec["gap_dead_merge"])
    if spec.get("minstop"):
        out = min_stoppage(out, geo, spec["minstop"])
    if spec.get("merge"):
        out = merge_short_possessions(out, geo, spec["merge"])
    return out
