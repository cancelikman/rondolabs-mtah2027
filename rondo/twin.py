"""Round 9 A line: the SkillCorner twin (training data from this round on).

Fixed split of the ten SkillCorner matches used so far, sorted by id: first 6 train, next 2
development, last 2 test (test opened once, in A6). No synthetic corruption on the twin; twin
labels = <0.5 s merged (raw also reported). Guards: WOY (T*), Metrica 1-2 (raw), Groningen H2
(raw), clean and target-like.

    python -m rondo.twin a1 --out reports/rondo/a1_twin.json
    python -m rondo.twin a2 --out reports/rondo/a2_stack.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import candidates, corruptions
from .d2_baseline import WINDOWS, baseline_pred, variant_match
from .evaluate import posteriors
from .matches import ROOT, load

SC10 = ["SC1874553", "SC1886347", "SC1899585", "SC1925299", "SC1927964", "SC1953632", "SC1959846",
        "SC1986691", "SC1996435", "SC1996436"]
TRAIN, DEV, TEST = SC10[:6], SC10[6:8], SC10[8:]
TWIN_VARIANT = "merge<0.5s"
GUARDS = {"WOY": ("WOY", "merge<1s"), "MT1": ("MT1_control", "raw"), "MT2": ("MT2_control", "raw"),
          "GRO": ("WCSF_H2", "raw")}
F4_SETTINGS = dict(dead_bias=4., team_cost=8., temperature=1.4)
WINS = (.25, .5, 1., 2., 4.)


def f4_parts(m, geo, cond):
    r = posteriors(candidates.get("ref"), m, cond, geo)
    c = posteriors(candidates.get("c3w"), m, cond, geo)
    return .5 * (np.asarray(r[0], float) + np.asarray(c[0], float)), .5 * (np.asarray(r[1], float) + np.asarray(c[1], float))


def decode(S, E, geo, **over):
    from lab.model import decode as dec
    return dec(S, E, geo, {**candidates.get("ref").decoder, **F4_SETTINGS, **over})


# ------------------------------------------------------------------ A1

def a1(out):
    from lab.errors import analyse
    from .a2_phase import codes
    from .tags import nearest_team
    rows = {}
    for name in TRAIN + DEV:      # test matches stay closed until A6
        m = load(name)
        mm = variant_match(m, TWIN_VARIANT)
        geo = m.geometry
        vis = np.isfinite(np.asarray(geo["ball"], float)).all(1)
        _, near = nearest_team(geo)
        S, E = f4_parts(m, geo, "clean")
        pred = decode(S, E, geo)
        s, sraw = mm.score(pred), m.score(pred)
        gt, pc = codes(mm, pred)
        res = analyse(gt, pc)
        rows[name] = dict(
            split="train" if name in TRAIN else "dev" if name in DEV else "test",
            ball_visible=round(float(vis.mean()), 4),
            nearest_m=dict(median=round(float(np.nanmedian(near[vis])), 2), p75=round(float(np.nanpercentile(near[vis], 75)), 2)),
            dead_share=round(float(np.mean(np.asarray(mm.scorer_data["labels"]) == 2)), 4),
            baselines={b: round(mm.score(baseline_pred(geo, w))["spell"], 4) for b, w in WINDOWS.items()},
            F4=dict(spell=round(s["spell"], 4), macro=round(s["macro"], 4), team_acc=round(s["team_acc"], 4),
                    spell_raw_labels=round(sraw["spell"], 4)),
            taxonomy={t["key"]: t["count"] for t in res["taxonomy"]},
            oracles={o["name"]: round(o["spell"], 4) for o in res["oracles"]})
        print(name, rows[name]["F4"], flush=True)
    band = all(.40 <= r["F4"]["spell"] <= .45 for r in rows.values())
    res = dict(variant=TWIN_VARIANT, split=dict(train=TRAIN, dev=DEV, test=TEST), matches=rows,
               F4_mean_spell=float(np.mean([r["F4"]["spell"] for r in rows.values()])),
               F4_in_band_0_40_0_45=band)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "matches"}, indent=1))


# ------------------------------------------------------------------ A2 stacking

def _window_means(x, fps, sec, future):
    w = max(1, int(sec * fps))
    c = np.r_[np.zeros((1,) + x.shape[1:]), np.cumsum(np.nan_to_num(x), axis=0)]
    n = len(x)
    i = np.arange(n)
    if future:
        lo, hi = i + 1, np.minimum(n, i + 1 + w)
    else:
        lo, hi = np.maximum(0, i - w), i
    cnt = np.maximum(1, hi - lo)
    return (c[hi] - c[lo]) / (cnt[:, None] if x.ndim == 2 else cnt)


def stack_features(geo, S, E):
    from .tags import nearest_team
    ball = np.asarray(geo["ball"], float)
    P = np.asarray(geo["players"], float)
    team = np.asarray(geo["team_of_track"])
    fps = float(geo["fps"])
    vis = np.isfinite(ball).all(1).astype(float)
    d = np.linalg.norm(P - ball[:, None, :], axis=2)
    dd = np.where(np.isfinite(d), d, np.inf)
    dh = dd[:, team == 0].min(1)
    da = dd[:, team == 1].min(1)
    ok = np.isfinite(dh) & np.isfinite(da)
    rel = np.where(ok, (da - dh) / np.maximum(1e-6, da + dh), 0.)
    nt, near = nearest_team(geo)
    sign = np.where(nt == 0, 1., np.where(nt == 1, -1., 0.))
    near = np.nan_to_num(near, nan=50.)
    v = np.r_[0., np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps]
    v = np.nan_to_num(np.minimum(v, 40.))
    pos = np.nan_to_num(ball, nan=-1.)
    S = np.nan_to_num(np.asarray(S, float), nan=1 / 3)
    E = np.nan_to_num(np.asarray(E, float), nan=.25)
    base = np.column_stack([S, E, vis, rel, sign, near, v, pos])
    sig = np.column_stack([S, rel, sign, vis])
    feats = [base]
    for sec in WINS:
        feats += [_window_means(sig, fps, sec, False), _window_means(sig, fps, sec, True)]
    return np.column_stack(feats).astype(np.float32)


def swap(x):
    """Exchange home/away in the stacking features (state H/A, event newH/newA, rel, sign)."""
    y = x.copy()
    y[:, [0, 1]] = x[:, [1, 0]]
    y[:, [4, 5]] = x[:, [5, 4]]
    y[:, 8] = -x[:, 8]
    y[:, 9] = -x[:, 9]
    k = 14
    for _ in WINS:
        for _ in range(2):
            y[:, [k, k + 1]] = x[:, [k + 1, k]]
            y[:, k + 3] = -x[:, k + 3]
            y[:, k + 4] = -x[:, k + 4]
            k += 6
    return y


def stack_data(names, cond="clean", variant=TWIN_VARIANT):
    from .phase_prior import gt25
    xs, ys = [], []
    for name in names:
        m = load(name)
        mm = variant_match(m, variant)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        S, E = f4_parts(m, geo, cond)
        x = stack_features(geo, S, E)
        y = gt25(mm)
        rows = np.arange(0, len(y), 3)
        rows = rows[y[rows] >= 0]
        xs += [x[rows], swap(x[rows])]
        yy = y[rows]
        ys += [yy, np.where(yy == 0, 1, np.where(yy == 1, 0, yy))]
    return np.concatenate(xs), np.concatenate(ys).astype(int)


def fit_stack(names, team_only=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    x, y = stack_data(names)
    if team_only:
        keep = y < 2
        x, y = x[keep], y[keep]
    est = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=.08, l2_regularization=1.,
                                         random_state=20261116, early_stopping=False)
    est.fit(x, y)
    return est


def stack_state(est, x, S, team_only=False):
    p = est.predict_proba(x)
    q = est.predict_proba(swap(x))
    if team_only:
        team = .5 * (p + q[:, [1, 0]])
        dead = np.asarray(S, float)[:, 2]
        st = np.column_stack([team * (1 - dead)[:, None], dead])
    else:
        st = .5 * (p + q[:, [1, 0, 2]])
    return st


def evaluate_stack(est, team_only, alpha, team_cost, names_conds):
    from .phase_prior import gt25  # noqa: F401
    rows = []
    for key, (name, variant, cond) in names_conds.items():
        m = load(name)
        mm = variant_match(m, variant)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        S, E = f4_parts(m, geo, cond)
        st = stack_state(est, stack_features(geo, S, E), S, team_only)
        mix = alpha * st + (1 - alpha) * S
        s = mm.score(decode(mix, E, geo, team_cost=team_cost))
        rows.append(dict(key=key, spell=s["spell"], macro=s["macro"], team_acc=s["team_acc"],
                         d_recall=s["d_recall"], d_precision=s["d_precision"]))
    return rows


def eval_sets():
    sets = {("dev", n): (n, TWIN_VARIANT, "clean") for n in DEV}
    for g, (n, v) in GUARDS.items():
        for cond in ("clean", "target_like"):
            sets[(g, cond)] = (n, v, cond)
    return sets


def a2(out):
    sets = eval_sets()
    keyed = {"%s|%s" % k: v for k, v in sets.items()}
    base = {}
    for k, (n, v, cond) in keyed.items():
        m = load(n)
        mm = variant_match(m, v)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        S, E = f4_parts(m, geo, cond)
        for tc in (6., 8.):
            s = mm.score(decode(S, E, geo, team_cost=tc))
            base[(k, tc)] = s["spell"]
    results = {}
    for team_only in (False, True):
        est = fit_stack(TRAIN, team_only)
        for alpha in (.5, .7, 1.):
            for tc in (6., 8.):
                rows = evaluate_stack(est, team_only, alpha, tc, keyed)
                name = "%s_a%g_team%g" % ("team" if team_only else "3cls", alpha, tc)
                dev = float(np.mean([r["spell"] for r in rows if r["key"].startswith("dev")]))
                dev_base = float(np.mean([base[(k, 8.)] for k in keyed if k.startswith("dev")]))
                guards = {r["key"]: r["spell"] - base[(r["key"], 8.)] for r in rows if not r["key"].startswith("dev")}
                results[name] = dict(dev_spell=dev, dev_delta=dev - dev_base, guards=guards,
                                     passes=dev - dev_base >= .03 and all(x >= -.01 for x in guards.values()),
                                     rows=rows)
                print(name, "dev %+.4f" % (dev - dev_base), "worst guard %+.4f" % min(guards.values()), flush=True)
    ok = [k for k, v in results.items() if v["passes"]]
    chosen = max(ok, key=lambda k: results[k]["dev_delta"]) if ok else None
    Path(out).write_text(json.dumps(dict(base={"%s@%g" % k: v for k, v in base.items()}, results=results,
                                         chosen=chosen), indent=1))
    print("chosen", chosen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["a1", "a2", "a3", "a5", "a2m", "a5s", "a6", "stackfit", "hyb"])
    p.add_argument("--cand", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.task == "hyb":
        hyb(a.out)
    elif a.task == "stackfit":
        fit_and_save_stack()
    elif a.task == "a5s":
        a5s(a.out)
    elif a.task == "a6":
        a6(a.out)
    elif a.task == "a2m":
        a2m(a.out)
    elif a.task == "a5":
        a5(a.out)
    elif a.task == "a3":
        a3_eval(a.cand, a.out)
    else:
        {"a1": a1, "a2": a2}[a.task](a.out)




def a3_eval(cand_name, out):
    """A3 acceptance: twin dev >= +1.5 over F4 (no F6a), guards <= 1 loss (clean and target-like)."""
    rows = []
    for key, (name, variant, cond) in {"%s|%s" % k: v for k, v in eval_sets().items()}.items():
        m = load(name)
        mm = variant_match(m, variant)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        S4, E4 = f4_parts(m, geo, cond)
        S, E = posteriors(candidates.get(cand_name), m, cond, geo)
        s4 = mm.score(decode(S4, E4, geo))
        s = mm.score(decode(np.asarray(S, float), np.asarray(E, float), geo))
        rows.append(dict(key=key, F4=s4["spell"], new=s["spell"], delta=s["spell"] - s4["spell"],
                         team_acc_delta=s["team_acc"] - s4["team_acc"]))
        print(key, "%+.4f" % rows[-1]["delta"], flush=True)
    dev = float(np.mean([r["delta"] for r in rows if r["key"].startswith("dev")]))
    guards = {r["key"]: r["delta"] for r in rows if not r["key"].startswith("dev")}
    res = dict(candidate=cand_name, dev_delta=dev, guards=guards,
               passes=dev >= .015 and all(x >= -.01 for x in guards.values()), rows=rows)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))

def a5(out):
    """A5: reference weight {0, .25, .5} x team_cost {6, 8} x event_strength {1, 1.5, 2} on the last
    package's members (F4: reference + C3w). Accept >= +1 on twin dev, guards <= 1 loss."""
    import itertools
    sets = {"%s|%s" % k: v for k, v in eval_sets().items()}
    cache = {}
    for key, (name, variant, cond) in sets.items():
        m = load(name)
        mm = variant_match(m, variant)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        r = posteriors(candidates.get("ref"), m, cond, geo)
        c = posteriors(candidates.get("c3w"), m, cond, geo)
        cache[key] = (mm, geo, [np.asarray(x, float) for x in r], [np.asarray(x, float) for x in c])
    def score(key, w, tc, ev):
        mm, geo, r, c = cache[key]
        S, E = w * r[0] + (1 - w) * c[0], w * r[1] + (1 - w) * c[1]
        return mm.score(decode(S, E, geo, team_cost=tc, event_strength=ev))["spell"]
    base = {k: score(k, .5, 8., 1.) for k in sets}
    res = {}
    for w, tc, ev in itertools.product((0., .25, .5), (6., 8.), (1., 1.5, 2.)):
        d = {k: score(k, w, tc, ev) - base[k] for k in sets}
        dev = float(np.mean([v for k, v in d.items() if k.startswith("dev")]))
        g = {k: v for k, v in d.items() if not k.startswith("dev")}
        res["wref%g_team%g_ev%g" % (w, tc, ev)] = dict(dev_delta=dev, worst_guard=min(g.values()), guards=g,
                                                       passes=dev >= .01 and all(x >= -.01 for x in g.values()))
        print("wref%g team%g ev%g dev %+.4f worst %+.4f" % (w, tc, ev, dev, min(g.values())), flush=True)
    ok = [k for k, v in res.items() if v["passes"]]
    Path(out).write_text(json.dumps(dict(results=res, chosen=max(ok, key=lambda k: res[k]["dev_delta"]) if ok else None), indent=1))

# ------------------------------------------------------------------ A2m: multi-source stack on the A3 base

BASE = "a3_w50_blend_ref"
IDSSE_TRAIN = ["WOH", "WN1", "WMX", "WQQ", "WPY"]


def base_parts(m, geo, cond):
    S, E = posteriors(candidates.get(BASE), m, cond, geo)
    return np.asarray(S, float), np.asarray(E, float)


def stack_rows(name, variant, cond):
    from .phase_prior import gt25
    m = load(name)
    mm = variant_match(m, variant)
    geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
    S, E = base_parts(m, geo, cond)
    x = stack_features(geo, S, E)
    y = gt25(mm)
    rows = np.arange(0, len(y), 3)
    rows = rows[y[rows] >= 0]
    xx = np.concatenate([x[rows], swap(x[rows])])
    yy = y[rows]
    return xx, np.concatenate([yy, np.where(yy == 0, 1, np.where(yy == 1, 0, yy))]).astype(int)


def fit_multisource(team_only):
    from sklearn.ensemble import HistGradientBoostingClassifier
    parts = {"twin": [stack_rows(n, TWIN_VARIANT, "clean") for n in TRAIN],
             "idsse": [stack_rows(n, "merge<1s", c) for n in IDSSE_TRAIN for c in ("clean", "target_like")]}
    xs, ys, ws = [], [], []
    for src, lst in parts.items():
        x = np.concatenate([p[0] for p in lst]); y = np.concatenate([p[1] for p in lst])
        if team_only:
            k = y < 2; x, y = x[k], y[k]
        xs.append(x); ys.append(y); ws.append(np.full(len(y), 1. / len(y)))   # equal total weight per source
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    est = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=.08, l2_regularization=1.,
                                         random_state=20261116, early_stopping=False)
    est.fit(x, y, sample_weight=w * len(w))
    return est


def a2m(out):
    """Acceptance (unchanged): twin dev >= +3 over F4; WOY, Metrica, Groningen H2 loss <= 1 (clean, target-like)."""
    sets = {"%s|%s" % k: v for k, v in eval_sets().items()}
    cache = {}
    for key, (name, variant, cond) in sets.items():
        m = load(name)
        mm = variant_match(m, variant)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        S4, E4 = f4_parts(m, geo, cond)
        S, E = base_parts(m, geo, cond)
        cache[key] = (mm, geo, S, E, stack_features(geo, S, E), mm.score(decode(S4, E4, geo))["spell"])
    results = {}
    for team_only in (False, True):
        est = fit_multisource(team_only)
        for alpha in (.5, .7, 1.):
            for tc in (6., 8.):
                d = {}
                for key, (mm, geo, S, E, x, f4s) in cache.items():
                    st = stack_state(est, x, S, team_only)
                    d[key] = mm.score(decode(alpha * st + (1 - alpha) * S, E, geo, team_cost=tc))["spell"] - f4s
                dev = float(np.mean([v for k, v in d.items() if k.startswith("dev")]))
                g = {k: v for k, v in d.items() if not k.startswith("dev")}
                name = "%s_a%g_team%g" % ("team" if team_only else "3cls", alpha, tc)
                results[name] = dict(dev_delta=dev, guards=g, worst_guard=min(g.values()),
                                     passes=dev >= .03 and all(v >= -.01 for v in g.values()))
                print(name, "dev %+.4f worst %+.4f" % (dev, min(g.values())), flush=True)
    base_d = {k: v[0].score(decode(v[2], v[3], v[1]))["spell"] - v[5] for k, v in cache.items()}
    ok = [k for k, v in results.items() if v["passes"]]
    Path(out).write_text(json.dumps(dict(base="A3 (" + BASE + ")", A3_vs_F4=base_d, results=results,
                                         chosen=max(ok, key=lambda k: results[k]["dev_delta"]) if ok else None,
                                         note="IDSSE training matches are in-sample for the A3 members"), indent=1))


# ------------------------------------------------------------------ persisted stack (A2m choice) + A5 + A6

STACK_DIR = ROOT / "data/artifacts/rondo/stack_a2m"
STACK_CHOICE = dict(team_only=False, alpha=.7, team_cost=6.)


def fit_and_save_stack():
    import joblib
    est = fit_multisource(STACK_CHOICE["team_only"])
    STACK_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(estimator=est, base=BASE, **STACK_CHOICE, windows=WINS,
                     provenance=dict(twin=TRAIN, idsse=IDSSE_TRAIN, conds=["clean", "target_like"],
                                     labels="twin <0.5 s merged, IDSSE <1 s merged", weights="equal total per source")),
                STACK_DIR / "stack.joblib", compress=3)
    return est


def load_stack():
    import joblib
    path = STACK_DIR / "stack.joblib"
    return joblib.load(path)["estimator"] if path.is_file() else fit_and_save_stack()


def stacked(est, geo, S, E, alpha=STACK_CHOICE["alpha"]):
    st = stack_state(est, stack_features(geo, S, E), S, STACK_CHOICE["team_only"])
    return alpha * st + (1 - alpha) * S


def a5s(out):
    """A5 on the new candidate (ref + A3 member, stack on top): reference weight {0,.25,.5} x team_cost
    {6,8} x event_strength {1,1.5,2}. Accept >= +1 twin dev over the candidate's own setting, guards <= 1 loss."""
    import itertools
    est = load_stack()
    sets = {"%s|%s" % k: v for k, v in eval_sets().items()}
    cache = {}
    for key, (name, variant, cond) in sets.items():
        m = load(name)
        mm = variant_match(m, variant)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        r = posteriors(candidates.get("ref"), m, cond, geo)
        c = posteriors(candidates.get("a3_w50"), m, cond, geo)
        cache[key] = (mm, geo, [np.asarray(x, float) for x in r], [np.asarray(x, float) for x in c])
    mixed = {}

    def score(key, w, tc, ev):
        mm, geo, r, c = cache[key]
        if (key, w) not in mixed:
            S, E = w * r[0] + (1 - w) * c[0], w * r[1] + (1 - w) * c[1]
            mixed[(key, w)] = (stacked(est, geo, S, E), E)
        S, E = mixed[(key, w)]
        return mm.score(decode(S, E, geo, team_cost=tc, event_strength=ev))["spell"]
    base = {k: score(k, .5, 6., 1.) for k in sets}
    res = {}
    for w, tc, ev in itertools.product((0., .25, .5), (6., 8.), (1., 1.5, 2.)):
        d = {k: score(k, w, tc, ev) - base[k] for k in sets}
        dev = float(np.mean([v for k, v in d.items() if k.startswith("dev")]))
        g = {k: v for k, v in d.items() if not k.startswith("dev")}
        res["wref%g_team%g_ev%g" % (w, tc, ev)] = dict(dev_delta=dev, worst_guard=min(g.values()), guards=g,
                                                       passes=dev >= .01 and all(x >= -.01 for x in g.values()))
        print("wref%g team%g ev%g dev %+.4f worst %+.4f" % (w, tc, ev, dev, min(g.values())), flush=True)
    ok = [k for k, v in res.items() if v["passes"]]
    Path(out).write_text(json.dumps(dict(candidate="ref+A3 (w .5) + A2m stack (a .7, team 6)", base=base, results=res,
                                         chosen=max(ok, key=lambda k: res[k]["dev_delta"]) if ok else None), indent=1))


def a6(out, w=.5, tc=6., ev=1.):
    """A6 (opened once): candidate vs F4 on the twin test matches. New package iff >= +2 over F4."""
    est = load_stack()
    rows = []
    for name in TEST:
        m = load(name)
        mm = variant_match(m, TWIN_VARIANT)
        geo = m.geometry
        S4, E4 = f4_parts(m, geo, "clean")
        r = [np.asarray(x, float) for x in posteriors(candidates.get("ref"), m, "clean", geo)]
        c = [np.asarray(x, float) for x in posteriors(candidates.get("a3_w50"), m, "clean", geo)]
        S, E = w * r[0] + (1 - w) * c[0], w * r[1] + (1 - w) * c[1]
        s4 = mm.score(decode(S4, E4, geo))
        s = mm.score(decode(stacked(est, geo, S, E), E, geo, team_cost=tc, event_strength=ev))
        rows.append(dict(match=name, F4_spell=s4["spell"], new_spell=s["spell"], F4_macro=s4["macro"], new_macro=s["macro"],
                         delta_spell=s["spell"] - s4["spell"], delta_macro=s["macro"] - s4["macro"]))
        print(json.dumps(rows[-1]), flush=True)
    d = float(np.mean([r["delta_spell"] for r in rows]))
    res = dict(settings={**STACK_CHOICE, "ref_weight": w, "team_cost": tc, "event_strength": ev}, rows=rows,
               mean_delta_spell=d, mean_delta_macro=float(np.mean([r["delta_macro"] for r in rows])),
               new_package=d >= .02)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))



# ------------------------------------------------------------------ hybrid (can, no training)

def hybrid_state(S4, st):
    """Dead probability from the F4 mixture; the rest split H:A by the F6 correction model's ratio."""
    d = np.asarray(S4, float)[:, 2]
    h = st[:, 0] / np.maximum(1e-9, st[:, 0] + st[:, 1])
    return np.column_stack([(1 - d) * h, (1 - d) * (1 - h), d])


def hyb(out):
    """Pass: twin dev mean spellF1 above F4 and no guard set losing more than 1 point vs F4."""
    est = load_stack()
    rows = []
    for key, (name, variant, cond) in {"%s|%s" % k: v for k, v in eval_sets().items()}.items():
        m = load(name)
        mm = variant_match(m, variant)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        S4, E4 = f4_parts(m, geo, cond)
        S, E = base_parts(m, geo, cond)
        st = stack_state(est, stack_features(geo, S, E), S, False)
        r = dict(key=key)
        for tag, (SS, EE, tc) in {"F4": (S4, E4, 8.), "F6": (.7 * st + .3 * S, E, 6.),
                                  "hybrid": (hybrid_state(S4, st), E, 6.)}.items():
            sc = mm.score(decode(SS, EE, geo, team_cost=tc))
            r[tag] = dict(spell=sc["spell"], macro=sc["macro"])
        rows.append(r)
        print(key, {t: "%.4f/%.4f" % (r[t]["spell"], r[t]["macro"]) for t in ("F4", "F6", "hybrid")}, flush=True)
    dev = float(np.mean([r["hybrid"]["spell"] - r["F4"]["spell"] for r in rows if r["key"].startswith("dev")]))
    g = {r["key"]: r["hybrid"]["spell"] - r["F4"]["spell"] for r in rows if not r["key"].startswith("dev")}
    res = dict(rule="dev mean spellF1 > F4 and every guard >= -1 point vs F4", dev_delta=dev, guards=g,
               passes=dev > 0 and all(v >= -.01 for v in g.values()), rows=rows,
               note="event posteriors and decoder as F6 (team_cost 6); kick-off rules unchanged")
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
