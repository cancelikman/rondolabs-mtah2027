"""Per-match report on labelled external matches (from tools/viewer_dump.py npz files).

    python tools/match_report.py reports/viewer/SC*.npz --package rondo_r7 --out reports/rondo/r10_match_report.md

Per match: spellF1, macroF1, true vs predicted possession count, recall of true possessions by
duration (IoU >= 0.5, same team), where the wrong frames are (ball visible / missing, contested /
open play) and which error type costs most (spellF1 gained if that type alone were fixed).
Diagnostic only; open data only.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from src.train import metrics  # noqa: E402
from viewer_html import episodes, spells, iou  # noqa: E402

BUCKETS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 1e9)]
BNAMES = ["<1s", "1-2s", "2-5s", "5-10s", ">10s"]


def score(d, pred):
    m = metrics(dict(labels=np.asarray(d["labels"]), blocks=np.asarray(d["blocks"]), fps=float(d["fps"]),
                     ball=d["ball"]), np.asarray(pred))
    return float(m["spell_f1_mean"]), float(m["macro_f1"])


def recall_by_duration(gt, pred, blocks, fps):
    G, P = spells(gt, blocks), spells(pred, blocks)
    hit, tot = np.zeros(len(BUCKETS)), np.zeros(len(BUCKETS))
    for g in G:
        dur = (g[1] - g[0]) / fps
        b = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= dur < hi)
        tot[b] += 1
        if max((iou(g, p) for p in P if p[2] == g[2] and p[0] < g[1] and p[1] > g[0]), default=0) >= .5:
            hit[b] += 1
    return hit, tot


def contested(d):
    ball = np.asarray(d["ball"], float)
    P = np.asarray(d["players"], float)
    team = np.asarray(d["team_of_track"]).astype(int)
    dist = np.linalg.norm(P - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    n0 = dist[:, team == 0].min(1) if (team == 0).any() else np.full(len(ball), np.inf)
    n1 = dist[:, team == 1].min(1) if (team == 1).any() else np.full(len(ball), np.inf)
    nearest = np.minimum(n0, n1)
    return (n0 <= 2) & (n1 <= 2), nearest > 3      # contested; nobody near (loose / in flight)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+")
    p.add_argument("--package", default="rondo_r7")
    p.add_argument("--out")
    a = p.parse_args()
    lines, agg_gain, agg_hit, agg_tot = [], defaultdict(list), np.zeros(5), np.zeros(5)
    agg_where = defaultdict(float)
    head = ("| match | spellF1 | macroF1 | possessions true / pred | recall <1s | 1-2s | 2-5s | 5-10s | >10s "
            "| wrong frames: ball missing | contested | loose | top error type (gain) |")
    lines += [head, "|" + "---|" * 14]
    for path in a.npz:
        d = np.load(path)
        name = Path(path).stem
        fps = float(d["fps"])
        gt = np.asarray(d["labels"]).astype(int)
        pred = np.asarray(d["pred_" + a.package]).astype(int)
        blocks = [tuple(map(int, b)) for b in d["blocks"]]
        vis = np.isfinite(np.asarray(d["ball"], float)).all(1)
        sp, mc = score(d, pred)
        hit, tot = recall_by_duration(gt, pred, blocks, fps)
        agg_hit += hit; agg_tot += tot
        cont, loose = contested(d)
        scored = gt >= 0
        wrong = scored & (pred != gt)
        nw = max(1, wrong.sum())
        where = dict(missing=(wrong & ~vis).sum() / nw, contested=(wrong & vis & cont).sum() / nw,
                     loose=(wrong & vis & loose & ~cont).sum() / nw)
        for k, v in where.items():
            agg_where[k] += v / len(a.npz)
        eps, ng, npd = episodes(gt, pred, blocks, vis, fps)
        gains = {}
        for key in sorted({"%s:%s" % (e["kind"], e["cat"]) for e in eps}):
            fixed = pred.copy()
            for e in eps:
                if "%s:%s" % (e["kind"], e["cat"]) == key:
                    fixed[e["start"]:e["end"]] = gt[e["start"]:e["end"]]
            gains[key] = 100 * (score(d, fixed)[0] - sp)
            agg_gain[key].append(gains[key])
        top = max(gains, key=gains.get)
        rec = ["%.0f%%" % (100 * h / t) if t else "-" for h, t in zip(hit, tot)]
        lines.append("| %s | %.1f | %.1f | %d / %d | %s | %.0f%% | %.0f%% | %.0f%% | %s (+%.1f) |" % (
            name, 100 * sp, 100 * mc, ng, npd, " | ".join(rec), 100 * where["missing"], 100 * where["contested"],
            100 * where["loose"], top, gains[top]))
        print(lines[-1], flush=True)
    lines += ["", "**All matches**", "",
              "Recall of true possessions by duration: " + ", ".join(
                  "%s %.0f%% (n=%d)" % (b, 100 * h / max(1, t), t) for b, h, t in zip(BNAMES, agg_hit, agg_tot)),
              "", "Where the wrong frames are (mean share): ball missing %.0f%%, contested (both teams within 2 m) %.0f%%, "
              "loose (nobody within 3 m, ball visible) %.0f%%, other %.0f%%" % (
                  100 * agg_where["missing"], 100 * agg_where["contested"], 100 * agg_where["loose"],
                  100 * (1 - sum(agg_where.values()))),
              "", "| error type | mean spellF1 gain if fixed | matches |", "|---|---|---|"]
    for k, v in sorted(agg_gain.items(), key=lambda x: -np.mean(x[1])):
        lines.append("| %s | +%.1f | %d |" % (k, np.mean(v), len(v)))
    text = "\n".join(lines)
    print("\n".join(lines[len(a.npz) + 2:]))
    if a.out:
        Path(a.out).write_text("# Per-match report (%s)\n\n" % a.package + text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
