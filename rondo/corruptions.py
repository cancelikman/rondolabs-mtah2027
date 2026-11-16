"""Synthetic measurement conditions for EXTERNAL matches only.

Labels are used only to place a corruption (e.g. "the ball vanishes during a
stoppage"); they never enter the model input. Constants come from the
organiser's published DATA.md figures (about 41% of frames without a ball) and
from football knowledge, never from the competition match.
"""
from __future__ import annotations

import numpy as np

from src.features import blocks_of, occlude_external


def _segments(mask):
    m = np.asarray(mask, dtype=int)
    edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
    return list(zip(edges[::2], edges[1::2]))


def occlude_stoppage(geo, dead, seed, share=.7):
    """In `share` of dead segments the ball disappears 0.5-1.5 s after the
    stoppage starts and returns 0-1 s before it ends."""
    rng = np.random.default_rng(seed)
    fps = float(geo["fps"])
    ball = np.array(geo["ball"], copy=True)
    for a, b in blocks_of(geo):
        for s, e in _segments(dead[a:b]):
            if rng.random() > share:
                continue
            lo = s + int(rng.uniform(.5, 1.5) * fps)
            hi = e - int(rng.uniform(0., 1.) * fps)
            if hi > lo:
                ball[a + lo:a + hi] = np.nan
    out = dict(geo)
    out["ball"] = ball
    return out


def ball_on_player(geo, dead, seed, noise=.3, mix=None):
    """During stoppages the ball is carried by the nearest player.

    mix=None: every dead segment is carried. mix=(keep, vanish, carry) samples a
    behaviour per segment (e.g. .2/.2/.6, the training-time augmentation).
    """
    rng = np.random.default_rng(seed)
    ball = np.array(geo["ball"], copy=True)
    players = np.asarray(geo["players"])
    for a, b in blocks_of(geo):
        for s, e in _segments(dead[a:b]):
            s, e = a + s, a + e
            if mix is not None:
                u = rng.random()
                if u < mix[0]:
                    continue
                if u < mix[0] + mix[1]:
                    ball[s:e] = np.nan
                    continue
            ref = ball[s] if np.isfinite(ball[s]).all() else None
            frame = players[s]
            ok = np.isfinite(frame).all(1)
            if not ok.any():
                continue
            if ref is None:
                j = int(rng.choice(np.flatnonzero(ok)))
            else:
                d = np.where(ok, np.linalg.norm(frame - ref, axis=1), np.inf)
                j = int(np.argmin(d))
            last = frame[j].copy()
            for t in range(s, e):
                p = players[t, j]
                if np.isfinite(p).all():
                    last = p
                ball[t] = last + rng.normal(0., noise, 2)
    out = dict(geo)
    out["ball"] = ball.astype(np.float32)
    return out


def _player_speed_rank(geo):
    players = np.asarray(geo["players"], dtype=float)
    fps = float(geo["fps"])
    step = np.linalg.norm(np.diff(players, axis=0), axis=2) * fps
    with np.errstate(all="ignore"):
        speed = np.nanmean(step, axis=1)
    speed = np.r_[speed[:1], speed]
    speed = np.where(np.isfinite(speed), speed, np.nanmedian(speed))
    # one-second moving average, then percentile rank in [0, 1]
    k = max(1, int(fps))
    smooth = np.convolve(speed, np.ones(k) / k, mode="same")
    order = smooth.argsort().argsort()
    return order / max(1, len(order) - 1)


def long_gaps(geo, seed, missing_share=.41, median_seconds=10., sigma=.9):
    """Long ball gaps (lognormal durations, median ~10 s) that start more often
    in slow play. Target overall missing share ~41% (DATA.md)."""
    rng = np.random.default_rng(seed)
    fps = float(geo["fps"])
    ball = np.array(geo["ball"], copy=True)
    rank = _player_speed_rank(geo)
    weight = 1.8 - 1.6 * rank           # slow play x9 more likely than fast play
    mean_gap = median_seconds * np.exp(sigma ** 2 / 2)
    mean_visible = mean_gap * (1 - missing_share) / missing_share
    base = 1. / (mean_visible * fps)
    for a, b in blocks_of(geo):
        t = a
        while t < b:
            if rng.random() < base * weight[t]:
                length = int(max(2 * fps, rng.lognormal(np.log(median_seconds), sigma) * fps))
                ball[t:min(b, t + length)] = np.nan
                t += length
            t += 1
    out = dict(geo)
    out["ball"] = ball
    return out


def position_noise(geo, seed, drift=.75, tau=2., jitter=.25, player_sigma=.2):
    """Slowly varying ball drift (OU process) + jitter; small player noise."""
    rng = np.random.default_rng(seed)
    fps = float(geo["fps"])
    ball = np.array(geo["ball"], dtype=np.float64, copy=True)
    n = len(ball)
    alpha = np.exp(-1. / (tau * fps))
    shock = rng.normal(0., drift * np.sqrt(1 - alpha ** 2), (n, 2))
    ou = np.zeros((n, 2))
    for i in range(1, n):
        ou[i] = alpha * ou[i - 1] + shock[i]
    ball += ou + rng.normal(0., jitter, (n, 2))
    players = np.asarray(geo["players"], dtype=np.float64) + rng.normal(0., player_sigma, np.shape(geo["players"]))
    out = dict(geo)
    out["ball"] = ball.astype(np.float32)
    out["players"] = players.astype(np.float32)
    return out


def realistic(geo, dead, seed):
    """Composite: carried/vanished ball in stoppages, long gaps, position noise."""
    g = ball_on_player(geo, dead, seed, mix=(.2, .2, .6))
    g = long_gaps(g, seed + 1)
    return position_noise(g, seed + 2)


# ---------------------------------------------------------------- round 4 (G4): physical corruptions

def _smooth(rng, n, dims, window_frames, amplitude):
    """Gaussian noise low-pass filtered with a Gaussian kernel (sigma = window_frames), rescaled to
    `amplitude` (std, m). A smooth kernel keeps the added velocity physical."""
    sigma = max(1., float(window_frames))
    w = int(4 * sigma)
    raw = rng.normal(0., 1., (n + 2 * w, dims))
    x = np.arange(-w, w + 1)
    kernel = np.exp(-.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.stack([np.convolve(raw[:, d], kernel, mode="same") for d in range(dims)], axis=1)[w:w + n]
    std = out.std(axis=0, keepdims=True)
    return out / np.where(std > 0, std, 1.) * amplitude


def smooth_noise(geo, seed, ball_amp=(.3, 1.), player_amp=(.1, .3), window_s=(.3, .6), jitter=(.02, .05)):
    """Temporally smooth position noise (no 7-10 m/s white-noise velocity spikes)."""
    rng = np.random.default_rng(seed)
    fps = float(geo["fps"])
    ball = np.array(geo["ball"], dtype=np.float64, copy=True)
    players = np.array(geo["players"], dtype=np.float64, copy=True)
    n = len(ball)
    win = rng.uniform(*window_s) * fps
    ball += _smooth(rng, n, 2, win, rng.uniform(*ball_amp)) + _smooth(rng, n, 2, 2, rng.uniform(*jitter))
    pa = rng.uniform(*player_amp)
    for j in range(players.shape[1]):
        players[:, j] += _smooth(rng, n, 2, win, pa)
    out = dict(geo)
    out["ball"], out["players"] = ball.astype(np.float32), players.astype(np.float32)
    return out


def three_stage_dead(geo, dead, seed, mix=(.15, .25, .6), offset=(.3, 1.), rest_share=(.3, .7),
                     vanish_s=(1., 3.), jitter=(.02, .05)):
    """Dead segments without any frozen ball.

    Per segment: stay (ball where it was, with 2-5 cm smooth jitter and a small drift),
    vanish, or three stages: the ball goes to the restart taker (or vanishes for 1-3 s),
    follows the taker with a smooth 0.3-1.0 m offset, and rests at the restart point for the
    last 30-70% of the segment. Restart point = ball position at the first live frame after it."""
    rng = np.random.default_rng(seed)
    fps = float(geo["fps"])
    ball = np.array(geo["ball"], dtype=np.float64, copy=True)
    src = np.asarray(geo["ball"], dtype=np.float64)
    players = np.asarray(geo["players"], dtype=np.float64)
    for a, b in blocks_of(geo):
        for s, e in _segments(dead[a:b]):
            s, e = a + s, a + e
            n = e - s
            if n < 2:
                ball[s:e] = ball[s:e] + rng.normal(0., .03, (n, 2))
                continue
            u = rng.random()
            jit = _smooth(rng, n, 2, 3, rng.uniform(*jitter))
            if u < mix[0]:                                   # stay, never frozen
                drift = np.linspace(0, 1, n)[:, None] * rng.normal(0., .15, 2)
                base = src[s] if np.isfinite(src[s]).all() else np.nanmean(src[s:e], axis=0)
                ball[s:e] = base + drift + jit
                continue
            if u < mix[0] + mix[1]:                          # vanish
                ball[s:e] = np.nan
                continue
            restart = src[e] if e < b and np.isfinite(src[e]).all() else src[e - 1]
            frame = players[s]
            ok = np.isfinite(frame).all(1)
            if not ok.any() or not np.isfinite(restart).all():
                ball[s:e] = np.nan
                continue
            d = np.where(ok, np.linalg.norm(frame - restart, axis=1), np.inf)
            j = int(np.argmin(d))                            # restart taker
            rest_from = s + int(n * (1 - rng.uniform(*rest_share)))
            carry = _smooth(rng, n, 2, max(2, int(.5 * fps)), 1.)
            carry = carry / max(1e-6, np.abs(carry).max()) * rng.uniform(*offset)
            last = frame[j].copy()
            start = src[s] if np.isfinite(src[s]).all() else None
            approach = int(min(fps, max(1, (rest_from - s) // 3)))
            vanish = int(rng.uniform(*vanish_s) * fps) if rng.random() < .5 else 0
            for t in range(s, rest_from):
                p = players[t, j]
                if np.isfinite(p).all():
                    last = p
                target = last + carry[t - s]
                k = t - s
                if vanish and k < vanish:
                    ball[t] = np.nan
                elif not vanish and start is not None and k < approach:
                    w = (k + 1) / approach
                    ball[t] = (1 - w) * start + w * target
                else:
                    ball[t] = target
            ball[rest_from:e] = restart + jit[rest_from - s:]
    out = dict(geo)
    out["ball"] = ball.astype(np.float32)
    return out


def uniform_gaps(geo, seed, missing_share=.2, median_seconds=6., sigma=.9):
    """Ball gaps whose start does not depend on play speed."""
    rng = np.random.default_rng(seed)
    fps = float(geo["fps"])
    ball = np.array(geo["ball"], copy=True)
    mean_gap = median_seconds * np.exp(sigma ** 2 / 2)
    base = 1. / (mean_gap * (1 - missing_share) / missing_share * fps)
    for a, b in blocks_of(geo):
        t = a
        while t < b:
            if rng.random() < base:
                length = int(max(fps, rng.lognormal(np.log(median_seconds), sigma) * fps))
                ball[t:min(b, t + length)] = np.nan
                t += length
            t += 1
    out = dict(geo)
    out["ball"] = ball
    return out


def training_views_v2(data, match_index):
    """Six G4 training views per match (different seeds)."""
    dead = np.asarray(data["labels"]) == 2
    s = 4000 + 31 * match_index
    views = {}
    for v in range(6):
        g = three_stage_dead(data, dead, s + 10 * v)
        kind = v % 3
        if kind == 0:
            g = occlude_external(g, s + 10 * v + 1)
        elif kind == 1:
            g = long_gaps(g, s + 10 * v + 1)
        else:
            g = uniform_gaps(g, s + 10 * v + 1)
        if v >= 3:
            g = smooth_noise(g, s + 10 * v + 2)
        views["v2_%d" % v] = g
    return views


def dead_restart(geo, dead, seed=9001):
    """Test-only: three-stage dead ball with a different seed and parameter ranges."""
    return three_stage_dead(geo, dead, seed, mix=(0., 0., 1.), offset=(.5, 1.5), rest_share=(.2, .8),
                            vanish_s=(.5, 2.), jitter=(.03, .06))


def ball_offset(geo, seed, amp_m, window_s):
    """Round 6 B0: smooth ball-player offset (the experiment's setting)."""
    return smooth_noise(geo, seed, ball_amp=(amp_m, amp_m), player_amp=(.15, .15), window_s=(window_s, window_s),
                        jitter=(.03, .03))


TARGET_LIKE = dict(amp_m=None, window_s=None)   # filled from reports/rondo/b0_offset_choice.json


def _target_like(g, d):
    import json
    from pathlib import Path
    choice = json.loads((Path(__file__).resolve().parents[1] / "reports/rondo/b0_offset_choice.json").read_text())
    return ball_offset(g, 606, choice["chosen"]["amp_m"], choice["chosen"]["window_s"])


CONDITIONS = {
    "clean": lambda g, d: g,
    "random202": lambda g, d: occlude_external(g, 202),
    "stoppage": lambda g, d: occlude_stoppage(g, d, 11),
    "stoppage+random": lambda g, d: occlude_stoppage(occlude_external(g, 202), d, 11),
    "ball_on_player": lambda g, d: ball_on_player(g, d, 7),
    "long_gaps": lambda g, d: long_gaps(g, 31),
    "realistic": lambda g, d: realistic(g, d, 41),
    "dead_restart": lambda g, d: dead_restart(g, d),
    "target_like": _target_like,
}
