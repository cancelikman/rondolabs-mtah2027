"""Round 6 A2/A3: choose phase-prior parameters with synthetic phase tags on external data.

Data (target-like condition, F4 posteriors, F4 decoder): Metrica 1-2, SkillCorner 10, WOY.
Synthetic phases from each external match's ground-truth possessions (>= 2 s, 85% kept, ends
jittered +-2 s), 10 / 20 / 30% wrong team, weight = mean confidence of the non-demoted phase
types in A0. Nothing here reads the provided match.

A2 grid: b0 in {0, 0.5, 1, 2, 3, 4}. Acceptance: mean spellF1 gain over the three error rates
>= +3 and no loss at 30% wrong.
A3 grid (on the chosen b0): event multiplier e in {0, 1} x stable-phase cost scale c in {0, .5, 1}.
Acceptance: >= +1 over A2 and fewer "spurious possession inside an opponent possession" errors.

    python -m rondo.a2_phase --stage a2 --out reports/rondo/a2_phase_prior.json
    python -m rondo.a2_phase --stage a3 --b0 <b0> --out reports/rondo/a3_phase_change.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import candidates, corruptions, phase_prior as pp
from .evaluate import posteriors
from .h1_grid import SC
from .matches import load

SETS = ["MT1_control", "MT2_control", *SC, "WOY"]
WRONG = (.1, .2, .3)
F4_DECODER = dict(team_cost=8., temperature=1.4, dead_bias=4.)


def a0_weight():
    inv = json.loads((Path(__file__).resolve().parents[1] / "reports/rondo/a0_tag_inventory.json").read_text())
    c = [v["confidence"] for v in inv["consistency"].values() if not v["demoted_to_unknown"]]
    return float(np.mean(c))


def codes(match, pred25):
    """(gt, pred) code arrays with half separators, as the official scorer sees them."""
    import sys
    from src.features import blocks_of
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
    import possession_metrics as pm
    data = match.scorer_data
    pred = np.asarray(pred25)
    if match.mapping is not None:
        pred = pred[match.mapping[:, 1]]
    gt = np.asarray(data["labels"]).copy()
    gt[gt < 0] = pm.X
    p = np.where(pred < 0, pm.X, pred)
    cuts = blocks_of(data)[1:, 0]
    if len(cuts):
        gt, p = np.insert(gt, cuts, pm.X), np.insert(p, cuts, pm.X)
    return gt, p


def spurious_in_opp(match, pred25):
    from lab.errors import analyse
    gt, p = codes(match, pred25)
    res = analyse(gt, p)
    return int(sum(1 for e in res["errors"] if e["kind"] == "spurious_in_opp")) if isinstance(res, dict) and "errors" in res else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["a2", "a3"], required=True)
    p.add_argument("--b0", type=float, default=0.)
    p.add_argument("--sets", nargs="+", default=SETS)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from lab.model import decode
    cand = candidates.get("c3w_blend_ref")
    settings = dict(cand.decoder, **F4_DECODER)
    weight = a0_weight()
    rows = []
    for k, mname in enumerate(a.sets):
        match = load(mname)
        geo = corruptions.CONDITIONS["target_like"](match.geometry, match.dead25)
        state, event = posteriors(cand, match, "target_like", geo)
        lab25 = pp.gt25(match)
        base_pred = decode(state, event, geo, settings)
        base = match.score(base_pred)
        base_sp = spurious_in_opp(match, base_pred)
        for wrong in WRONG:
            phases = pp.synthetic_phases(lab25, float(geo["fps"]), 100 + k, wrong, weight)
            sig = pp.phase_signal(len(state), phases, float(geo["fps"]))
            if a.stage == "a2":
                grid = [dict(b0=b) for b in (0., .5, 1., 2., 3., 4.)]
            else:
                grid = [dict(b0=a.b0, e=e, c=c) for e in (0., 1.) for c in (0., .5, 1.)]
            for g in grid:
                st = pp.apply_phase_bias(state, sig, g["b0"])
                if a.stage == "a2":
                    pred = decode(st, event, geo, settings)
                else:
                    ev = pp.apply_event_bias(event, phases, g["e"], float(geo["fps"]))
                    scale = 1. + g["c"] * pp.stable_mask(len(st), phases)
                    pred = pp.decode_varcost(st, ev, geo, settings, scale)
                s = match.score(pred)
                row = dict(match=mname, wrong=wrong, **g, spell=s["spell"], macro=s["macro"], team_acc=s["team_acc"],
                           base_spell=base["spell"], delta=s["spell"] - base["spell"], n_pred=s["n_pred"], n_gt=s["n_gt"])
                if a.stage == "a3":
                    row.update(spurious_in_opp=spurious_in_opp(match, pred), base_spurious_in_opp=base_sp)
                rows.append(row)
                print(mname, wrong, g, "spell %.4f delta %+.4f" % (s["spell"], row["delta"]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(stage=a.stage, weight=weight, rows=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
