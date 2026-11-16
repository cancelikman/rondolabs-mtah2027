"""Build a self-contained HTML pitch viewer from a viewer_dump npz (labelled external matches).

    python tools/viewer_html.py reports/viewer/SC2006229.npz        -> reports/viewer/SC2006229.html

Open the HTML in a browser. Pitch replay (players, ball, ball trail), timelines (truth vs
each package, P(H/A/D), ball visibility) and a clickable list of error episodes with a
category for each one. Error categories are descriptive, for understanding, not rules.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def runs(lab, blocks):
    """[(start, end, value)] maximal runs inside each scored block (end exclusive)."""
    out = []
    for a, b in blocks:
        x = lab[a:b]
        if len(x) == 0:
            continue
        e = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
        out += [(a + int(s), a + int(t), int(x[s])) for s, t in zip(e[:-1], e[1:])]
    return out


def spells(lab, blocks):
    return [r for r in runs(lab, blocks) if r[2] in (0, 1)]


def iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / max(1, max(a[1], b[1]) - min(a[0], b[0]))


def episodes(gt, pred, blocks, vis, fps):
    """Descriptive error episodes.

    Missed true possession (no same-team predicted spell with IoU >= 0.5):
      short<2s / swallowed (a longer same-team prediction covers it) / wrong_team / called_dead /
      boundary (same team, overlapping, IoU 0.3-0.5) / split (several same-team pieces)
    Spurious predicted possession (no same-team true spell with IoU >= 0.5):
      in_other_team / in_dead / split_piece
    """
    G, P = spells(gt, blocks), spells(pred, blocks)
    pi = 0
    res, hit_p = [], set()
    for g in G:
        s, e, v = g
        cands = [(iou(g, p), j) for j, p in enumerate(P) if p[2] == v and p[0] < e and p[1] > s]
        best = max(cands, default=(0., -1))
        if best[0] >= .5:
            hit_p.add(best[1])
            continue
        seg = pred[s:e]
        same, other, dead = (seg == v).mean(), (seg == 1 - v).mean(), (seg == 2).mean()
        dur = (e - s) / fps
        if dur < 2:
            cat = "short<2s"
        elif other >= .5:
            cat = "wrong_team"
        elif dead >= .5:
            cat = "called_dead"
        elif best[1] >= 0 and (P[best[1]][1] - P[best[1]][0]) > 1.8 * (e - s):
            cat = "swallowed"
        elif len(cands) >= 2:
            cat = "split"
        else:
            cat = "boundary"
        res.append(dict(kind="missed", start=s, end=e, team=v, dur=round(dur, 2), cat=cat,
                        iou=round(best[0], 2), same=round(float(same), 2), other=round(float(other), 2),
                        dead=round(float(dead), 2), ball=round(float(vis[s:e].mean()), 2)))
    for j, p in enumerate(P):
        if j in hit_p:
            continue
        s, e, v = p
        cands = [iou(p, g) for g in G if g[2] == v and g[0] < e and g[1] > s]
        if max(cands, default=0.) >= .5:
            continue
        seg = gt[s:e]
        other, dead = (seg == 1 - v).mean(), (seg == 2).mean()
        cat = "in_other_team" if other >= .5 else "in_dead" if dead >= .5 else "split_piece"
        res.append(dict(kind="spurious", start=s, end=e, team=v, dur=round((e - s) / fps, 2), cat=cat,
                        iou=round(max(cands, default=0.), 2), same=round(float((seg == v).mean()), 2),
                        other=round(float(other), 2), dead=round(float(dead), 2), ball=round(float(vis[s:e].mean()), 2)))
    res.sort(key=lambda r: r["start"])
    return res, len(G), len(P)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def build(npz_path, out=None, gt_key="labels"):
    d = np.load(npz_path)
    fps = float(d["fps"])
    ball = np.asarray(d["ball"], float)
    players = np.asarray(d["players"], float)
    n, k = players.shape[:2]
    vis = np.isfinite(ball).all(1)
    blocks = [tuple(map(int, b)) for b in d["blocks"]]
    gt = np.asarray(d[gt_key]).astype(int)
    pkgs = sorted(k_[5:] for k_ in d.files if k_.startswith("pred_"))
    meta = json.loads(Path(npz_path).with_suffix(".json").read_text()) if Path(npz_path).with_suffix(".json").exists() else {}

    def q(x):  # decimetres, int16, missing -> -32768
        y = np.where(np.isfinite(x), np.round(x * 10), -32768)
        return y.astype(np.int16)

    # nearest player distance per team (for the strip and episode context)
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    team = np.asarray(d["team_of_track"]).astype(int)
    near = np.full((n, 2), np.nan)
    for t in (0, 1):
        dd = dist[:, team == t]
        with np.errstate(all="ignore"):
            near[:, t] = np.where(np.isfinite(dd).any(1), np.nanmin(np.where(np.isfinite(dd), dd, np.inf), 1), np.nan)
    data = dict(
        name=Path(npz_path).stem, fps=fps, n=n, k=k, pitch=[float(v) for v in d["pitch"]], blocks=blocks,
        team=team.tolist(), packages=pkgs, scores=meta.get("scores", {}), gt_key=gt_key,
        ball=b64(q(ball)), players=b64(q(players).reshape(n, k * 2)), gt=b64(gt.astype(np.int8)),
        near=b64(q(np.nan_to_num(near, nan=99.9))),
        preds={p: b64(np.asarray(d["pred_" + p]).astype(np.int8)) for p in pkgs},
        probs={p: b64(np.clip(np.round(np.nan_to_num(np.asarray(d["prob_" + p], float)) * 255), 0, 255).astype(np.uint8))
               for p in pkgs},
        episodes={}, counts={})
    for p in pkgs:
        ep, ng, np_ = episodes(gt, np.asarray(d["pred_" + p]).astype(int), blocks, vis, fps)
        data["episodes"][p] = ep
        data["counts"][p] = dict(gt=ng, pred=np_)
    html = (HERE / "viewer_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    out = Path(out) if out else Path(npz_path).with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out, data


if __name__ == "__main__":
    for path in sys.argv[1:]:
        o, _ = build(path)
        print("wrote", o)
