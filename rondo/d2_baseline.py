"""Round 8 D2: reproduce the official baseline fingerprint on external matches.

starter/baseline.py logic (nearest player's team, majority window 0/25/50/125 frames, D where
nobody is near a visible ball) on target-like external matches, each scored against its own
closest label variant from D1 (WOY: T*, Metrica: raw, SkillCorner: <0.5 s merged, Groningen H2:
raw). Official full match: raw 0.0486, smooth25 0.1996, smooth50 0.2433, smooth125 0.2242.

    python -m rondo.d2_baseline --out reports/rondo/d2_baseline.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import corruptions
from .h1_grid import SC
from .matches import Match, ROOT, load

OFFICIAL = {"raw": .0486, "smooth25": .1996, "smooth50": .2433, "smooth125": .2242}
WINDOWS = {"raw": 0, "smooth25": 25, "smooth50": 50, "smooth125": 125}


def variant_match(m, variant):
    from src.features import blocks_of
    from .matches import merged_labels
    if variant == "raw":
        return m
    s = float(variant.replace("merge<", "").replace("s", ""))
    data = dict(m.scorer_data)
    blocks = [tuple(map(int, b)) for b in blocks_of(data)]
    data["labels"] = merged_labels(data["labels"], blocks, s, float(data["fps"]))
    mm = Match(m.name, m.geometry, m.dead25, data, m.mapping)
    if hasattr(m, "cache_name"):
        mm.cache_name = m.cache_name
    return mm


def closest():
    d1 = json.loads((ROOT / "reports/rondo/d1_fingerprint.json").read_text())
    c = d1["closest_variant_per_source"]
    return {"WOY": d1["T_star"], "MT": c["Metrica_control"], "SC": c["SkillCorner"], "GRO": c["Groningen"]}


SETS = {"WOY": ["WOY"], "MT": ["MT1_control", "MT2_control"], "SC": SC, "GRO": ["WCSF_H2"]}


def baseline_pred(geo, window):
    from starter.baseline import AWAY, HOME, NOBODY, smooth
    from .tags import nearest_team
    nt, _ = nearest_team(geo)
    codes = np.where(nt == 0, HOME, np.where(nt == 1, AWAY, NOBODY)).astype(np.int8)
    lab = smooth(codes, window)
    return np.where(lab == "H", 0, np.where(lab == "A", 1, 2)).astype(np.int8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    var = closest()
    rows = []
    for g, names in SETS.items():
        for name in names:
            m = load(name)
            mm = variant_match(m, var[g])
            geo = corruptions.CONDITIONS["target_like"](m.geometry, m.dead25)
            for b, w in WINDOWS.items():
                s = mm.score(baseline_pred(geo, w))
                rows.append(dict(set=g, match=name, baseline=b, spell=s["spell"], macro=s["macro"]))
            print(name, "done", flush=True)
    summary = {g: {b: float(np.mean([r["spell"] for r in rows if r["set"] == g and r["baseline"] == b])) for b in WINDOWS}
               for g in SETS}
    order = {g: sorted(WINDOWS, key=lambda b: -summary[g][b]) for g in SETS}
    res = dict(official=OFFICIAL, variants=var, summary=summary, ranking=order, rows=rows,
               pattern_holds={g: order[g][0] == "smooth50" for g in SETS})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(dict(summary=summary, ranking=order), indent=1))


if __name__ == "__main__":
    main()
