"""Round 10: multi-source correction model with "last control" memory features.

Pre-registration: reports/rondo/r10_preregistration.md (rule R10-2).
  N10   = SkillCorner matches never used before; N10-A (first 5 ids) selects, N10-B (last 5) tests once.
  F6    = models/rondo_r7 (reference for every delta).
  C0    = F6 recipe, stack refit on all 10 old twin matches + IDSSE (data only, same features).
  C1    = C0 + control_memory features (rondo/control_memory.py, fixed defaults) + agreement column.
  Select: N10-A mean spellF1 >= F6 + 1.5 and every guard (WOY, MT1, MT2, Groningen H2;
          clean and target-like) loses <= 1.0 against F6. Best passing N10-A delta wins.
  Test:   chosen candidate on N10-B, opened once; new package iff mean delta >= +1.5.

    python -m rondo.r10 select --out reports/rondo/r10_select.json
    python -m rondo.r10 test   --out reports/rondo/r10_test.json      (once, after select)
    python -m rondo.r10 pack   --out models/rondo_r8                   (only if test passes)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import corruptions
from .control_memory import DEFAULT as CM_DEFAULT, NAMES as CM_NAMES, SIGNED, control_memory
from .d2_baseline import variant_match
from .matches import ROOT, load
from .twin import GUARDS, IDSSE_TRAIN, SC10, TWIN_VARIANT, base_parts, stack_features, swap

N10 = ["SC2006229", "SC2006363", "SC2007448", "SC2007721", "SC2010085",
       "SC2011166", "SC2013725", "SC2015213", "SC2016236", "SC2017461"]
N10A, N10B = N10[:5], N10[5:]
F6 = ROOT / "models/rondo_r7"
ALPHA, TEAM_COST = .7, 6.
NBASE = None   # set on first feature build


def features(geo, S, E, use_cm, cm=None):
    base = stack_features(geo, S, E)
    if not use_cm:
        return base
    x = control_memory(geo, **(cm or CM_DEFAULT))
    S = np.nan_to_num(np.asarray(S, float), nan=1 / 3)
    agree = x[:, 1] * (S[:, 0] - S[:, 1])        # invariant under home/away exchange
    return np.column_stack([base, x, agree]).astype(np.float32)


def swap_all(x, use_cm):
    nb = x.shape[1] - (len(CM_NAMES) + 1 if use_cm else 0)
    y = np.empty_like(x)
    y[:, :nb] = swap(x[:, :nb])
    if use_cm:
        cm = x[:, nb:nb + len(CM_NAMES)].copy()
        cm[:, SIGNED] = -cm[:, SIGNED]
        y[:, nb:nb + len(CM_NAMES)] = cm
        y[:, -1] = x[:, -1]
    return y


def state(est, x, use_cm):
    p = est.predict_proba(x)
    q = est.predict_proba(swap_all(x, use_cm))
    return .5 * (p + q[:, [1, 0, 2]])


def decoder():
    return json.loads((F6 / "manifest.json").read_text(encoding="utf-8"))["decoder"]


def decode(S, E, geo):
    from lab.model import decode as dec
    return dec(S, E, geo, decoder())


# ------------------------------------------------------------------ data

_CACHE = {}


def item(name, variant, cond):
    key = (name, variant, cond)
    if key not in _CACHE:
        t = time.time()
        m = load(name)
        mm = variant_match(m, variant)
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        S, E = base_parts(m, geo, cond)
        _CACHE[key] = dict(mm=mm, geo=geo, S=S, E=E, x={})
        print(json.dumps(dict(stage="posteriors", match=name, cond=cond, s=round(time.time() - t))), flush=True)
    return _CACHE[key]


def feats_of(it, use_cm):
    if use_cm not in it["x"]:
        it["x"][use_cm] = features(it["geo"], it["S"], it["E"], use_cm)
    return it["x"][use_cm]


def train_rows(name, variant, cond, use_cm):
    from .phase_prior import gt25
    it = item(name, variant, cond)
    x = feats_of(it, use_cm)
    y = gt25(it["mm"])
    rows = np.arange(0, len(y), 3)
    rows = rows[y[rows] >= 0]
    yy = y[rows]
    return (np.concatenate([x[rows], swap_all(x[rows], use_cm)]),
            np.concatenate([yy, np.where(yy == 0, 1, np.where(yy == 1, 0, yy))]).astype(int))


def fit(use_cm):
    from sklearn.ensemble import HistGradientBoostingClassifier
    parts = {"twin": [train_rows(n, TWIN_VARIANT, "clean", use_cm) for n in SC10],
             "idsse": [train_rows(n, "merge<1s", c, use_cm) for n in IDSSE_TRAIN for c in ("clean", "target_like")]}
    xs, ys, ws = [], [], []
    for lst in parts.values():
        x = np.concatenate([p[0] for p in lst]); y = np.concatenate([p[1] for p in lst])
        xs.append(x); ys.append(y); ws.append(np.full(len(y), 1. / len(y)))   # equal total weight per source
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    est = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=.08, l2_regularization=1.,
                                         random_state=20261116, early_stopping=False)
    t = time.time()
    est.fit(x, y, sample_weight=w * len(w))
    print(json.dumps(dict(stage="fit", use_cm=use_cm, rows=int(len(y)), cols=int(x.shape[1]), s=round(time.time() - t))), flush=True)
    return est


def f6_estimator():
    import joblib
    return joblib.load(F6 / "stack.joblib")["estimator"]


def score(it, est, use_cm):
    S = np.asarray(it["S"], float)
    if est is None:
        return it["mm"].score(decode(S, it["E"], it["geo"]))
    st = state(est, feats_of(it, use_cm), use_cm)
    return it["mm"].score(decode(ALPHA * st + (1 - ALPHA) * S, it["E"], it["geo"]))


def eval_sets(names):
    sets = {"N|%s" % n: (n, TWIN_VARIANT, "clean") for n in names}
    for g, (n, v) in GUARDS.items():
        for cond in ("clean", "target_like"):
            sets["G|%s|%s" % (g, cond)] = (n, v, cond)
    return sets


def compare(cands, names):
    """{set: {cand: spell}} with F6 (old stack, old features) as 'F6'."""
    out = {}
    f6 = f6_estimator()
    for key, spec in eval_sets(names).items():
        it = item(*spec)
        row = {"F6": score(it, f6, False)["spell"]}
        for cname, (est, use_cm) in cands.items():
            row[cname] = score(it, est, use_cm)["spell"]
        out[key] = row
        print(key, {k: round(v, 4) for k, v in row.items()}, flush=True)
        ref = ROOT / "reports/viewer" / ("%s.json" % spec[0])
        if key.startswith("N|") and ref.is_file():      # sanity: same F6 as tools/viewer_dump.py
            want = json.loads(ref.read_text())["scores"].get("rondo_r7", {}).get("spell")
            if want is not None and abs(want - row["F6"]) > .002:
                print("WARNING F6 mismatch on %s: %.4f here vs %.4f in viewer dump" % (spec[0], row["F6"], want), flush=True)
    return out


def summarize(table, cname):
    n = [r[cname] - r["F6"] for k, r in table.items() if k.startswith("N|")]
    g = {k: r[cname] - r["F6"] for k, r in table.items() if k.startswith("G|")}
    return dict(n_delta=float(np.mean(n)), n_wins=int(sum(d > 0 for d in n)), n_matches=len(n),
                worst_guard=float(min(g.values())), guards=g)


def save_est(est, use_cm, name):
    import joblib
    d = ROOT / "data/artifacts/rondo/r10"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(estimator=est, use_cm=use_cm, cm=CM_DEFAULT if use_cm else None, alpha=ALPHA,
                     provenance=dict(twin=SC10, idsse=IDSSE_TRAIN, conds=["clean", "target_like"],
                                     labels="twin <0.5 s merged, IDSSE <1 s merged", weights="equal total per source")),
                d / ("%s.joblib" % name), compress=3)


def load_est(name):
    import joblib
    a = joblib.load(ROOT / "data/artifacts/rondo/r10" / ("%s.joblib" % name))
    return a["estimator"], a["use_cm"]


# ------------------------------------------------------------------ tasks

def task_select(out):
    cands = {}
    for cname, use_cm in (("C0", False), ("C1", True)):
        est = fit(use_cm)
        save_est(est, use_cm, cname)
        cands[cname] = (est, use_cm)
    table = compare(cands, N10A)
    res = {c: summarize(table, c) for c in cands}
    for c, r in res.items():
        r["passes"] = r["n_delta"] >= .015 and r["worst_guard"] >= -.01
        print(c, "N10-A %+.4f (%d/%d wins)  worst guard %+.4f  passes %s" % (r["n_delta"], r["n_wins"], r["n_matches"],
                                                                           r["worst_guard"], r["passes"]), flush=True)
    ok = [c for c, r in res.items() if r["passes"]]
    chosen = max(ok, key=lambda c: res[c]["n_delta"]) if ok else None
    Path(out).write_text(json.dumps(dict(rule="N10-A >= F6 + 1.5, guards >= -1.0", results=res, table=table,
                                         chosen=chosen, cm=CM_DEFAULT), indent=1))
    print("chosen", chosen)


def task_test(out, select_path):
    sel = json.loads(Path(select_path).read_text())
    chosen = sel["chosen"]
    if chosen is None:
        raise SystemExit("no candidate passed selection; N10-B stays closed")
    if Path(out).exists():
        raise SystemExit("%s exists: N10-B has already been opened" % out)
    est, use_cm = load_est(chosen)
    rows = {}
    f6 = f6_estimator()
    for n in N10B:
        it = item(n, TWIN_VARIANT, "clean")
        a, b = score(it, f6, False), score(it, est, use_cm)
        rows[n] = dict(F6=a["spell"], cand=b["spell"], F6_macro=a["macro"], cand_macro=b["macro"],
                       n_pred_F6=a["n_pred"], n_pred_cand=b["n_pred"], n_gt=a["n_gt"])
        print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[n].items()}, flush=True)
    d = float(np.mean([r["cand"] - r["F6"] for r in rows.values()]))
    dm = float(np.mean([r["cand_macro"] - r["F6_macro"] for r in rows.values()]))
    res = dict(candidate=chosen, rows=rows, mean_delta_spell=d, mean_delta_macro=dm, new_package=d >= .015)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def task_pack(out, test_path):
    import joblib
    import shutil
    res = json.loads(Path(test_path).read_text())
    if not res.get("new_package"):
        raise SystemExit("test did not pass; no package")
    est, use_cm = load_est(res["candidate"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((F6 / "manifest.json").read_text(encoding="utf-8"))
    for m in man["members"]:
        for f in (m["tcn"], m["hgb"]):
            shutil.copyfile(F6 / f, out / f)
    joblib.dump(dict(estimator=est), out / "stack.joblib", compress=3)
    man["name"] = "rondolabs-r8-F7"
    man["description"] = ("Round 10 F7: F6 with the multi-source correction model refit on all 10 SkillCorner twin "
                          "matches + 5 IDSSE" + (" and last-control memory features (rondo/control_memory.py)" if use_cm else "") + ".")
    man["stack"] = dict(file="stack.joblib", sha256=hashlib.sha256((out / "stack.joblib").read_bytes()).hexdigest(),
                        alpha=ALPHA, team_only=False)
    if use_cm:
        man["stack"]["control_memory"] = CM_DEFAULT
    man["selection"] = "Round 10: selected on N10-A, tested once on N10-B (%+.4f spellF1 vs F6). See reports/rondo/r10_*.json." % res["mean_delta_spell"]
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["select", "test", "pack"])
    p.add_argument("--out", required=True)
    p.add_argument("--select", default=str(ROOT / "reports/rondo/r10_select.json"))
    p.add_argument("--test", default=str(ROOT / "reports/rondo/r10_test.json"))
    a = p.parse_args()
    if a.task == "select":
        task_select(a.out)
    elif a.task == "test":
        task_test(a.out, a.select)
    else:
        task_pack(a.out, a.test)


if __name__ == "__main__":
    main()
