"""Error budget on labelled external matches: spellF1 gained if one error category were fixed.

    python tools/error_budget.py reports/viewer/SC*.npz --package rondo_r7

For each category the frames of its episodes are set to the truth and the match is rescored
with the official metric. Gains overlap (fixing a missed and a spurious episode on the same
frames counts in both), so they are not additive. Diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from src.train import metrics  # noqa: E402
from viewer_html import episodes  # noqa: E402


def score(d, pred):
    data = dict(labels=np.asarray(d["labels"]), blocks=np.asarray(d["blocks"]), fps=float(d["fps"]), ball=d["ball"])
    m = metrics(data, np.asarray(pred))
    return float(m["spell_f1_mean"]), float(m["macro_f1"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+")
    p.add_argument("--package", default="rondo_r7")
    p.add_argument("--out")
    a = p.parse_args()
    rows, agg = {}, defaultdict(list)
    for path in a.npz:
        d = np.load(path)
        gt = np.asarray(d["labels"]).astype(int)
        pred = np.asarray(d["pred_" + a.package]).astype(int)
        vis = np.isfinite(np.asarray(d["ball"], float)).all(1)
        blocks = [tuple(map(int, b)) for b in d["blocks"]]
        eps, ng, npred = episodes(gt, pred, blocks, vis, float(d["fps"]))
        base = score(d, pred)
        row = dict(spell=round(base[0], 4), macro=round(base[1], 4), n_gt=ng, n_pred=npred, cats={})
        cats = sorted({(e["kind"], e["cat"]) for e in eps})
        groups = {"%s:%s" % c: [e for e in eps if (e["kind"], e["cat"]) == c] for c in cats}
        groups["ALL:ball_missing_frames"] = None
        for key, es in groups.items():
            fixed = pred.copy()
            if es is None:
                m = (~vis) & (gt >= 0)
                fixed[m] = gt[m]
            else:
                for e in es:
                    fixed[e["start"]:e["end"]] = gt[e["start"]:e["end"]]
            s = score(d, fixed)
            row["cats"][key] = dict(n=0 if es is None else len(es), gain_spell=round(100 * (s[0] - base[0]), 2))
            agg[key].append(row["cats"][key]["gain_spell"])
        rows[Path(path).stem] = row
        print(Path(path).stem, row["spell"], row["n_pred"], "/", row["n_gt"], flush=True)
    summary = {k: round(float(np.sum(v)) / len(rows), 2) for k, v in agg.items()}
    summary = dict(sorted(summary.items(), key=lambda x: -x[1]))
    print(json.dumps(dict(package=a.package, mean_spell=round(float(np.mean([r["spell"] for r in rows.values()])), 4),
                          mean_gain_if_fixed=summary), indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(package=a.package, matches=rows, mean_gain_if_fixed=summary), indent=1))


if __name__ == "__main__":
    main()
