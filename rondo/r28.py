"""Round 28: multi-league, multi-provider evaluation with a ranking-consistency check (new design).

Why (team review, 2026-11-17 20:05): R14-R27 were accepted or rejected mainly on N10 (10 SkillCorner
matches, one league, one provider, one labeller); the guards were only a floor. Across six candidates
the guards disagreed with N10 in sign every time. The validity of a single-family proxy was assumed,
never tested. R28 fixes the design before any R28 number:

FAMILIES (equal weight; within a family, equal weight per evaluation set):
  SC        N10, 10 matches (A-League, SkillCorner), variant merge<0.5s, clean
  IDSSE     WOY and WR9 (Bundesliga, DFL optical; WR9 never used before), merge<1s, clean + target-like
  METRICA   MT1, MT2 (Metrica sample games, control labels), clean + target-like
  GRONINGEN WCSF second half, clean + target-like
  PFF       8 held-out World Cup matches (game id % 8 == 0), v0|raw labels, target-like, filled ball
            (PFF-trained candidates are retrained here without them: R15h, R16h, R17h)
  The provided match and its public 5 minutes are not used.

CANDIDATES (everything tried today that can be evaluated everywhere; R21/R22 dropped for time):
  F7 (baseline); decoder/mix settings on F7's network: R23 (team_cost 8, w 0.65), R25 (dead_bias 2),
  R26 (alpha 0.8, w 0.6), R27 (dead_bias 3); networks: R14, R15h, R16h, R17h (each = its two fold
  networks), R18h (F7's + R14 + R15h + R16h fold networks).
  On SC every fold network is scored only on the N10 half it did not see; elsewhere a candidate's fold
  networks are averaged.

CONSISTENCY (tested first): for each family, Spearman correlation across candidates between that
family's delta vs F7 and the mean delta of the other families. A family with rho < 0 disagrees with
the consensus of the others and is excluded from the decision (reported either way).
DECISION: aggregate = mean delta vs F7 (spellF1) over the included families. Adopt the best candidate
iff aggregate >= +0.5, it is ahead of F7 in a majority of the included families, and no included
family is below -1.0. Otherwise F7 stays.

Pre-registration: reports/rondo/r10_preregistration.md (addendum R28, before any R28 number).
Caveat stated there: the candidates and the design were chosen after today's N10/guard results were
known; WR9 and the 8 held-out PFF matches are the genuinely new evidence.

    python -m rondo.r28 train                  (~60 min: R15h, R16h, R17h fold networks without the PFF holdout)
    python -m rondo.r28 eval --out reports/rondo/r28_eval.json
    python -m rondo.r28 pack --out models/rondo_r10
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import gnn
from .matches import ROOT
from .r10 import N10A, N10B, decoder, f6_estimator, features, item, state
from .r12 import ART as R12ART, THREADS, gnn_item
from .r14 import ART as R14ART
from .r15 import compact, pff_variant
from .r16 import HEADS
from .twin import GUARDS, IDSSE_TRAIN, SC10, TWIN_VARIANT

SEED = 20261116
ART = ROOT / "data/artifacts/rondo/r28"
F7 = ROOT / "models/rondo_r8"


def holdout():
    from .pff_data import match_ids
    h = [g for g in match_ids() if int(g) % 8 == 0]
    assert len(h) == 8, h
    return h


# ------------------------------------------------------------------ training (PFF without the holdout)

def pff_items_train():
    from . import corruptions
    from .pff_data import load_pff, match_ids
    from .d2_baseline import variant_match
    ev, _, merge = pff_variant().partition("|")
    import os
    out = []
    hold = set(holdout())
    train_ids = [g for g in match_ids() if g not in hold]
    n = int(os.environ.get("RONDO_PFF_N", len(train_ids)))   # pre-registered RAM fallback (R15): fewer PFF matches
    for gid in train_ids[:n]:
        t = time.time()
        m = load_pff("%s@%s" % (gid, ev))
        mm = variant_match(m, merge or "raw")
        geo = corruptions.CONDITIONS["target_like"](m.geometry, m.dead25)
        out.append(compact(gnn.Item(geo, mm.scorer_data["labels"])))
        del m, mm, geo
        gc.collect()
        print(json.dumps(dict(stage="pff_item", game=gid, s=round(time.time() - t, 1))), flush=True)
    print(json.dumps(dict(stage="pff_train_set", n=len(out), holdout=sorted(hold))), flush=True)
    return out


def _train(sc, path, pff, head_of=None, init=None, lr=2e-3, epochs=4, with_pff=True):
    if path.is_file():
        return gnn.load(path)
    groups = {"twin": [gnn_item(n, TWIN_VARIANT, "clean") for n in sc],
              "idsse": [gnn_item(n, "merge<1s", c) for n in IDSSE_TRAIN for c in ("clean", "target_like")]}
    if with_pff:
        groups["pff"] = pff
    net = gnn.train(groups, epochs=epochs, per_epoch=300000, batch=512, seed=SEED, threads=THREADS,
                    log=lambda m: print(m, flush=True), head_of=head_of, init=init, lr=lr)
    ART.mkdir(parents=True, exist_ok=True)
    gnn.save(net, path, dict(sc=sc, pff=len(pff) if with_pff else 0, holdout=holdout(), head_of=head_of, seed=SEED))
    print(json.dumps(dict(stage="saved", path=str(path))), flush=True)
    return net


def task_train():
    pff = pff_items_train()
    for fold, half in (("A", N10A), ("B", N10B)):
        sc = SC10 + half
        r15 = _train(sc, ART / ("r15h_fold%s.pt" % fold), pff)
        _train(sc, ART / ("r16h_fold%s.pt" % fold), pff, head_of=HEADS)
        _train(sc, ART / ("r17h_fold%s.pt" % fold), pff, init=r15, lr=5e-4, epochs=1, with_pff=False)


# ------------------------------------------------------------------ evaluation

def families():
    fam = {"SC": [(n, TWIN_VARIANT, "clean") for n in N10A + N10B],
           "IDSSE": [(n, "merge<1s", c) for n in ("WOY", "WR9") for c in ("clean", "target_like")],
           "METRICA": [(GUARDS[g][0], GUARDS[g][1], c) for g in ("MT1", "MT2") for c in ("clean", "target_like")],
           "GRONINGEN": [(GUARDS["GRO"][0], GUARDS["GRO"][1], c) for c in ("clean", "target_like")],
           "PFF": [("PF%s@v0" % g, "raw", "target_like") for g in holdout()]}
    return fam


def net_paths():
    p = {"F7": R12ART / "gnn.pt"}
    for f in "AB":
        p["r14" + f] = R14ART / ("gnn_fold%s.pt" % f)
        for r in ("r15h", "r16h", "r17h"):
            p[r + f] = ART / ("%s_fold%s.pt" % (r, f))
    return p


def candidates():
    """name -> dict(nets=[member prefixes], dec={decoder overrides}, w=, alpha=)."""
    base = dict(nets=["F7"], dec={}, w=.5, alpha=.7)
    c = {"F7": base,
         "R23": dict(base, dec=dict(team_cost=8., temperature=1.4), w=.65),
         "R25": dict(base, dec=dict(dead_bias=2., dead_cost=6.)),
         "R26": dict(base, alpha=.8, w=.6),
         "R27": dict(base, dec=dict(dead_bias=3., dead_cost=6.))}
    for r in ("r14", "r15h", "r16h", "r17h"):
        c[r.upper()] = dict(base, nets=[r])
    c["R18H"] = dict(base, nets=["F7", "r14", "r15h", "r16h"])
    return c


def members(spec_nets, name_of_set):
    """Network keys used for one evaluation set: fold networks only on the N10 half they did not see."""
    keys = []
    for m in spec_nets:
        if m == "F7":
            keys.append("F7")
        elif name_of_set in N10A:
            keys.append(m + "B")
        elif name_of_set in N10B:
            keys.append(m + "A")
        else:
            keys += [m + "A", m + "B"]
    return keys


def task_eval(out):
    from scipy.stats import spearmanr
    from lab.model import decode as dec
    paths = net_paths()
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        raise SystemExit("missing networks %s: run `python -m rondo.r28 train` first" % missing)
    nets = {k: gnn.load(p) for k, p in paths.items()}
    est = f6_estimator()
    cands = candidates()
    raw = {}
    for fam, specs in families().items():
        for spec in specs:
            key = "%s|%s|%s" % (fam, spec[0], spec[2])
            t = time.time()
            try:
                it = item(*spec)
            except Exception as e:                     # pre-specified: a set that cannot load is skipped
                print(json.dumps(dict(stage="skip", set=key, error=repr(e))), flush=True)
                continue
            S = np.asarray(it["S"], float)
            st = state(est, features(it["geo"], S, it["E"], False), False)
            P = {}
            raw[key] = {}
            for cname, c in cands.items():
                ks = members(c["nets"], spec[0])
                for k in ks:
                    if k not in P:
                        P[k] = gnn.predict(nets[k], it["geo"], threads=THREADS)
                Pm = np.mean([P[k] for k in ks], axis=0)
                S6 = c["alpha"] * st + (1 - c["alpha"]) * S
                cfg = dict(decoder(), **c["dec"])
                m = it["mm"].score(dec((1 - c["w"]) * S6 + c["w"] * Pm, it["E"], it["geo"], cfg))
                raw[key][cname] = dict(spell=m["spell"], macro=m["macro"])
            print(key, "s=%d" % (time.time() - t), {k: round(v["spell"] - raw[key]["F7"]["spell"], 4)
                                                    for k, v in raw[key].items() if k != "F7"}, flush=True)
    fams = list(families())
    delta = {c: {} for c in cands}
    for c in cands:
        for f in fams:
            ks = [k for k in raw if k.startswith(f + "|")]
            if ks:
                delta[c][f] = float(np.mean([raw[k][c]["spell"] - raw[k]["F7"]["spell"] for k in ks]))
    others = [c for c in cands if c != "F7"]
    consistency = {}
    for f in fams:
        a = [delta[c][f] for c in others]
        b = [np.mean([delta[c][g] for g in fams if g != f and g in delta[c]]) for c in others]
        consistency[f] = float(spearmanr(a, b).correlation)
    included = [f for f in fams if consistency[f] >= 0]
    agg = {c: float(np.mean([delta[c][f] for f in included])) for c in others}
    ok = {c: (agg[c] >= .005 and sum(delta[c][f] > 0 for f in included) > len(included) / 2
              and min(delta[c][f] for f in included) >= -.01) for c in others}
    passing = [c for c in others if ok[c]]
    chosen = max(passing, key=lambda c: agg[c]) if passing else "F7"
    res = dict(design="R28 multi-family", families=fams, consistency_spearman=consistency, included=included,
               delta_spell=delta, aggregate=agg, passes=ok, chosen=chosen, raw=raw)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print("\nconsistency (Spearman vs the other families):", {f: round(v, 2) for f, v in consistency.items()})
    print("included families:", included)
    print("\ncandidate  " + "  ".join("%9s" % f for f in fams) + "   aggregate  pass")
    for c in sorted(others, key=lambda c: -agg[c]):
        print("%-9s  " % c + "  ".join("%+9.4f" % delta[c].get(f, float("nan")) for f in fams) +
              "   %+.4f    %s" % (agg[c], ok[c]))
    print("\nchosen:", chosen)


def task_pack(out, ev):
    r = json.loads(Path(ev).read_text())
    c = r["chosen"]
    if c == "F7":
        raise SystemExit("R28 chose F7; nothing to pack")
    spec = candidates()[c]
    out = Path(out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    shutil.copytree(F7, out)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    paths = net_paths()
    files = []
    for m in spec["nets"]:
        if m == "F7":
            files.append("gnn.pt")
            continue
        for f in "AB":
            name = "gnn_%s_fold%s.pt" % (m, f)
            shutil.copyfile(paths[m + f], out / name)
            files.append(name)
    if "gnn.pt" not in files:
        (out / "gnn.pt").unlink()
    man["name"] = "rondolabs-r10-F10"
    man["gnn"] = dict(files=files, weight=spec["w"],
                      sha256={f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files})
    man["decoder"] = dict(man["decoder"], **spec["dec"])
    man["stack"]["alpha"] = spec["alpha"]
    man["description"] += " Round 28 F10: candidate %s chosen by the multi-league, multi-provider evaluation." % c
    man["selection"] = "Round 28 (reports/rondo/r28_eval.json)."
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out, c, files, man["decoder"], man["gnn"]["weight"], man["stack"]["alpha"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=["train", "eval", "pack"])
    p.add_argument("--out")
    p.add_argument("--eval", default=str(ROOT / "reports/rondo/r28_eval.json"))
    a = p.parse_args()
    if a.task == "train":
        task_train()
    elif a.task == "eval":
        task_eval(a.out or a.eval)
    else:
        task_pack(a.out, a.eval)


if __name__ == "__main__":
    main()
