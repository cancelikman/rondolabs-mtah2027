"""Round 6 A2/A3/A4: Smart Tagging phase priors inside the decoder (soft evidence).

phase_signal: per frame, signed weight toward home (+) / away (-) from phases with a prior team,
  weight = confidence(type) with a 1 s linear ramp at both ends; conflicting overlaps -> 0.
apply_phase_bias (A2): P(H) *= exp(+b0*s), P(A) *= exp(-b0*s), team mass renormalised so that
  P(D) is untouched.
A3: around a change of prior team between consecutive phases (+-3 s) the "new possession" event
  for the new team is multiplied by (1 + e); inside phases with a stable prior team the team
  switch cost becomes team_cost * (1 + c) (custom directed Viterbi with a per-step cost).
A4: soft P(D) multiplier in type-specific windows before set-piece tags, and a restart-team
  prior (phase signal for 4 s after the tag) for tags with a team.
Synthetic external tags (for choosing parameters): built from ground-truth possessions of OTHER
matches, never from the provided match.
"""
from __future__ import annotations

import numpy as np

from src.features import blocks_of

H, A, D = 0, 1, 2


def phase_signal(n, phases, fps=25., ramp_s=1.):
    """phases: [(start, end, team, weight)]."""
    w = np.zeros((n, 2))
    r = max(1, int(ramp_s * fps))
    for s, e, team, wt in phases:
        s, e = max(0, int(s)), min(n, int(e))
        if e <= s or wt <= 0:
            continue
        L = e - s
        ramp = np.ones(L)
        k = min(r, L // 2)
        if k > 0:
            ramp[:k] = np.linspace(1. / k, 1., k)
            ramp[-k:] = np.minimum(ramp[-k:], np.linspace(1., 1. / k, k))
        w[s:e, team] += wt * ramp
    conflict = (w[:, 0] > 0) & (w[:, 1] > 0)
    s = w[:, 0] - w[:, 1]
    s[conflict] = 0.
    return s


def apply_phase_bias(state, signal, b0):
    st = np.array(state, float)
    if not b0:
        return st
    ok = np.isfinite(st).all(1) & (signal != 0)
    d = st[ok, 2]
    team = st[ok, :2] * np.exp(np.stack([b0 * signal[ok], -b0 * signal[ok]], 1))
    team = team / team.sum(1, keepdims=True) * (1 - d)[:, None]
    st[ok, :2] = team
    return st


def change_points(phases):
    ph = sorted(phases, key=lambda p: p[0])
    out = []
    for p, q in zip(ph, ph[1:]):
        if p[2] != q[2]:
            out.append((int(q[0]), q[2]))
    return out


def apply_event_bias(event, phases, e, fps=25., window_s=3.):
    ev = np.array(event, float)
    if not e:
        return ev
    w = int(window_s * fps)
    for f, team in change_points(phases):
        a, b = max(0, f - w), min(len(ev), f + w)
        col = 1 + team                      # event columns: none, newH, newA, newD
        ok = np.isfinite(ev[a:b]).all(1)
        seg = ev[a:b][ok]
        seg[:, col] *= (1 + e)
        seg /= seg.sum(1, keepdims=True)
        ev[a:b][ok] = seg
    return ev


def stable_mask(n, phases):
    """Frames inside a phase whose prior team equals the previous and next overlapping phases."""
    m = np.zeros(n, bool)
    for s, e, team, wt in phases:
        if wt > 0:
            m[max(0, int(s)):min(n, int(e))] = True
    return m


def directed_viterbi_varcost(p, q, team_cost, dead_cost, strength, temperature=1., minimum_cost=.1):
    """experiments.transition_external.directed_viterbi with a per-step team cost array."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = len(p)
    if not n:
        return np.zeros(0, np.int8)
    tc = np.broadcast_to(np.asarray(team_cost, float), (n,))
    contrast = np.maximum(0., np.log(np.clip(q[:, 1:], 1e-7, 1)) - np.log(np.clip(q[:, :1], 1e-7, 1)))
    emission = -np.log(np.clip(p, 1e-6, 1)) / temperature
    score = emission[0]
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        base = np.full((3, 3), tc[t])
        base[2, :2] = base[:2, 2] = dead_cost
        np.fill_diagonal(base, 0.)
        cost = np.maximum(minimum_cost, base - strength * contrast[t][None, :]) if strength else base
        np.fill_diagonal(cost, 0.)
        paths = score[:, None] + cost
        parent = np.argmin(paths, axis=0)
        back[t] = parent
        score = paths[parent, np.arange(3)] + emission[t]
        score -= score.min()
    out = np.empty(n, np.int8)
    out[-1] = np.argmin(score)
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def decode_varcost(state, event, geometry, settings, cost_scale):
    """lab.model.decode with dead_bias and a per-frame team-cost multiplier (A3)."""
    st, ev = np.array(state, float), np.array(event, float)
    inside = np.zeros(len(st), bool)
    for a, b in blocks_of(geometry):
        inside[a:b] = True
    k = float(settings.get("dead_bias", 1.))
    if k != 1.:
        st[inside, 2] *= k
        st[inside] /= st[inside].sum(1, keepdims=True)
    st[~inside] = 1 / 3
    ev[~inside] = [1, 0, 0, 0]
    stride = int(settings.get("stride", 2))
    pred = np.full(len(st), -1, np.int8)
    for a, b in blocks_of(geometry):
        starts = np.arange(a, b, stride)
        fr = np.asarray([st[i:min(i + stride, b)].mean(0) for i in starts])
        evs = np.asarray([ev[i:min(i + stride, b)].mean(0) for i in starts])
        tc = settings["team_cost"] * np.asarray([cost_scale[i:min(i + stride, b)].mean() for i in starts])
        lab = directed_viterbi_varcost(fr, evs, tc, settings["dead_cost"], settings["event_strength"],
                                       settings["temperature"])
        pred[a:b] = np.repeat(lab, stride)[:b - a]
    return pred


def dead_windows(state, windows, k):
    """A4: soft P(D) multiplier inside [(start, end)] windows."""
    st = np.array(state, float)
    for s, e in windows:
        s, e = max(0, int(s)), min(len(st), int(e))
        ok = np.isfinite(st[s:e]).all(1)
        seg = st[s:e][ok]
        seg[:, 2] *= k
        seg /= seg.sum(1, keepdims=True)
        st[s:e][ok] = seg
    return st


# --------------------------------------------------------------- synthetic tags (external data)

def gt25(match):
    """Ground-truth labels at 25 Hz for an external match (nearest native row)."""
    lab = np.asarray(match.scorer_data["labels"])
    if match.mapping is None:
        return lab.copy()
    n25 = len(match.geometry["ball"])
    out = np.full(n25, -1, np.int8)
    nat, rows = match.mapping[:, 0], match.mapping[:, 1]
    idx = np.searchsorted(rows, np.arange(n25), side="right") - 1
    idx = np.clip(idx, 0, len(rows) - 1)
    out[:] = lab[nat[idx]]
    return out


def synthetic_phases(labels, fps, seed, wrong, weight, min_s=2., jitter_s=2., keep=.85):
    """Phase-like intervals from ground-truth possessions of an external match."""
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    edges = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1, len(lab)]
    out = []
    for s, e in zip(edges[:-1], edges[1:]):
        v = lab[s]
        if v not in (H, A) or (e - s) < min_s * fps or rng.random() > keep:
            continue
        js, je = rng.uniform(-jitter_s, jitter_s, 2) * fps
        team = int(v) if rng.random() >= wrong else 1 - int(v)
        out.append((int(s + js), int(e + je), team, weight))
    return out
