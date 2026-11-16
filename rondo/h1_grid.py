"""H1 (round 5): team-switch cost, temperature and minimum possession for F3 posteriors.

Tuning data (from this round on, NOT independent tests): Metrica games 1-2 (both label
definitions, clean and long_gaps) and SkillCorner (10 matches, clean). Guard: IDSSE WOY, seven
conditions, raw labels and the 2 s merged target.

Grid: team_cost {4,6,8,10} x temperature {1.4,2.0} x minimum possession {none, 1.0, 1.5 s}.
Selection rule (fixed in the round-5 note before measuring):
 1. beat F3 on the Metrica mean and on the SkillCorner mean, and by >= +1.0 on their average;
 2. WOY 2 s target mean not below F3;  3. WOY raw mean at most 1.5 points below F3;
 4. among passing variants, predicted / true possession ratio over Metrica + SkillCorner
    closest to 1.

    python -m rondo.h1_grid --set MT --out reports/rondo/h1_MT.json
    python -m rondo.h1_grid --select reports/rondo/h1_MT.json reports/rondo/h1_SC.json reports/rondo/h1_WOY.json --out reports/rondo/h1_selection.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import candidates, corruptions, rules
from .evaluate import posteriors
from .matches import load, score_targets

CAND = "c3w_seeds_blend_ref"          # = F3 posteriors (ref 0.5 + three C3w seeds 1/6 each)
SC = ["SC1874553", "SC1886347", "SC1899585", "SC1925299", "SC1927964", "SC1953632", "SC1959846",
      "SC1986691", "SC1996435", "SC1996436"]
SETS = {
    "MT": [(m, c) for m in ("MT1_control", "MT1_duel", "MT2_control", "MT2_duel") for c in ("clean", "long_gaps")],
    "SC": [(m, "clean") for m in SC],
    "WOY": [("WOY", c) for c in ("clean", "random202", "stoppage", "stoppage+random", "ball_on_player",
                                 "long_gaps", "realistic")],
}
F3 = dict(team_cost=4., temperature=1.4, min_possession=0.)


def grid():
    return [dict(team_cost=t, temperature=T, min_possession=m)
            for t, T, m in itertools.product((4., 6., 8., 10.), (1.4, 2.), (0., 1., 1.5))]


def vname(v):
    return "team%g_temp%g_min%g" % (v["team_cost"], v["temperature"], v["min_possession"])


def run(set_name):
    from lab.model import decode
    cand = candidates.get(CAND)
    rows = []
    for mname, cname in SETS[set_name]:
        match = load(mname)
        geo = corruptions.CONDITIONS[cname](match.geometry, match.dead25)
        state, event = posteriors(cand, match, cname, geo)
        for v in grid():
            settings = dict(cand.decoder, team_cost=v["team_cost"], temperature=v["temperature"], dead_bias=4.)
            pred = rules.min_possession(decode(state, event, geo, settings), geo, v["min_possession"])
            targets = (0, 2) if set_name == "WOY" else (0,)
            sc = score_targets(match, pred, targets)
            row = dict(set=set_name, match=mname, condition=cname, variant=vname(v), **v)
            for t, s in sc.items():
                row.update({"%s_spell" % t: s["spell"], "%s_macro" % t: s["macro"],
                            "%s_n_pred" % t: s["n_pred"], "%s_n_gt" % t: s["n_gt"]})
            rows.append(row)
            print("%-11s %-15s %-24s raw %.4f n %d/%d%s" % (mname, cname, vname(v), sc["raw"]["spell"],
                  sc["raw"]["n_pred"], sc["raw"]["n_gt"], "  N2 %.4f" % sc["N2"]["spell"] if "N2" in sc else ""),
                  flush=True)
    return rows


def select(rows):
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)
    f3 = vname(F3)

    def mean(rs, s, key):
        v = [r[key] for r in rs if r["set"] == s]
        return float(np.mean(v)) if v else float("nan")

    base = {k: mean(by[f3], *k) for k in (("MT", "raw_spell"), ("SC", "raw_spell"), ("WOY", "raw_spell"), ("WOY", "N2_spell"))}
    summary, passing = {}, []
    for name, rs in by.items():
        d = {"%s/%s" % k: mean(rs, *k) - base[k] for k in base}
        avg = (d["MT/raw_spell"] + d["SC/raw_spell"]) / 2
        npred = sum(r["raw_n_pred"] for r in rs if r["set"] in ("MT", "SC"))
        ngt = sum(r["raw_n_gt"] for r in rs if r["set"] in ("MT", "SC"))
        ok = (d["MT/raw_spell"] > 0 and d["SC/raw_spell"] > 0 and avg >= .01 and d["WOY/N2_spell"] >= 0
              and d["WOY/raw_spell"] >= -.015)
        summary[name] = dict(delta_vs_F3=d, mt_sc_avg_gain=avg, pred_true_ratio=npred / max(1, ngt), passes=ok)
        if ok and name != f3:
            passing.append((abs(npred / max(1, ngt) - 1), name))
    chosen = min(passing)[1] if passing else f3
    return dict(F3=f3, base_means={"%s/%s" % k: v for k, v in base.items()}, chosen=chosen, n_passing=len(passing), summary=summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", choices=list(SETS))
    p.add_argument("--select", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.select:
        rows = [r for f in a.select for r in json.loads(f.read_text(encoding="utf-8"))["rows"]]
        res = select(rows)
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("chosen", res["chosen"], "passing", res["n_passing"])
        for k, v in sorted(res["summary"].items(), key=lambda kv: -kv[1]["mt_sc_avg_gain"])[:10]:
            print("%-24s avg %+.4f ratio %.3f pass %s %s" % (k, v["mt_sc_avg_gain"], v["pred_true_ratio"], v["passes"],
                                                            {c: round(x, 4) for c, x in v["delta_vs_F3"].items()}))
        return
    rows = run(a.set)
    a.out.write_text(json.dumps(dict(grid=grid(), rows=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
