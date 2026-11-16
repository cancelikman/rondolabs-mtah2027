"""Round 11: possession-change detector as event evidence for the decoder.

Pre-registration: reports/rondo/r10_preregistration.md (addendum R11, written before any result).
  Target: the existing transition-head target (experiments.transition_external.event_targets:
          none / new H / new A / new D within +-2 frames of a label change), same class balancing.
  Inputs: F6's final posteriors (after the correction model), F6's event probabilities, the stack
          geometry/window features and the control-memory features (rondo.r10.features).
  Data:   10 old SkillCorner twin matches (clean, <0.5 s merged labels) + 5 IDSSE training matches
          (clean and target-like, <1 s merged), equal total weight per source, home/away swap.
  Use:    event' = w * detector + (1 - w) * F6 event, decoded with F6's decoder; w in {0.5, 1.0}.
  Select: N10-A mean spellF1 >= F6 + 1.5 and every guard loses <= 1.0. Test once on N10-B,
          new package iff mean delta >= +1.5.

    python -m rondo.r11 select --out reports/rondo/r11_select.json
    python -m rondo.r11 test   --out reports/rondo/r11_test.json
    python -m rondo.r11 pack   --out models/rondo_r8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .matches import ROOT
from .r10 import (ALPHA, F6, N10A, N10B, decode, eval_sets, f6_estimator, features, item, state, swap_all)
from .twin import IDSSE_TRAIN, SC10, TWIN_VARIANT

WEIGHTS = (.5, 1.)
ART = ROOT / "data/artifacts/rondo/r11"


def f6_parts(it, est):
    S = np.asarray(it["S"], float)
    if "S6" not in it:
        it["S6"] = ALPHA * state(est, features(it["geo"], S, it["E"], False), False) + (1 - ALPHA) * S
    return it["S6"], np.asarray(it["E"], float)


def det_features(it, est):
    if "xd" not in it:
        S6, E = f6_parts(it, est)
        it["xd"] = features(it["geo"], S6, E, True)
    return it["xd"]


def det_proba(det, x):
    p = det.predict_proba(x)
    q = det.predict_proba(swap_all(x, True))
    return .5 * (p + q[:, [0, 2, 1, 3]])


def targets(it):
    from experiments.transition_external import event_targets
    from .phase_prior import gt25
    y = gt25(it["mm"])
    geo = it["geo"]
    data = dict(labels=np.asarray(y), ball=np.asarray(geo["ball"]), fps=float(geo["fps"]))
    if geo.get("blocks") is not None:
        data["blocks"] = np.asarray(geo["blocks"])
    t, centers = event_targets(data)
    return np.asarray(t), centers


def rows_of(name, variant, cond, est):
    it = item(name, variant, cond)
    x = det_features(it, est)
    t, centers = targets(it)
    near = np.clip(centers[:, None] + np.arange(-3, 4), 0, len(t) - 1).ravel() if len(centers) else np.zeros(0, int)
    idx = np.unique(np.r_[np.arange(0, len(t), 3), near])
    idx = idx[t[idx] >= 0]
    y = t[idx].astype(int)
    ys = np.where(y == 1, 2, np.where(y == 2, 1, y))
    return np.concatenate([x[idx], swap_all(x[idx], True)]), np.concatenate([y, ys])


def fit(est):
    from sklearn.ensemble import HistGradientBoostingClassifier
    groups = {"twin": [rows_of(n, TWIN_VARIANT, "clean", est) for n in SC10],
              "idsse": [rows_of(n, "merge<1s", c, est) for n in IDSSE_TRAIN for c in ("clean", "target_like")]}
    xs, ys, ws = [], [], []
    for g in groups.values():
        x = np.concatenate([r[0] for r in g]); y = np.concatenate([r[1] for r in g])
        cls = len(y) / (4 * np.maximum(np.bincount(y, minlength=4), 1))      # same balancing as the TCN head
        w = cls[y]
        xs.append(x); ys.append(y); ws.append(w / w.sum())                    # equal total weight per source
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    det = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=.08, l2_regularization=1.,
                                         random_state=20261116, early_stopping=False)
    t = time.time()
    det.fit(x, y, sample_weight=w * len(w))
    print(json.dumps(dict(stage="fit", rows=int(len(y)), cols=int(x.shape[1]),
                          classes=np.bincount(y, minlength=4).tolist(), s=round(time.time() - t))), flush=True)
    return det


def scores(it, est, det):
    S6, E = f6_parts(it, est)
    out = {"F6": it["mm"].score(decode(S6, E, it["geo"]))}
    Ed = det_proba(det, det_features(it, est))
    for w in WEIGHTS:
        out["W%g" % w] = it["mm"].score(decode(S6, w * Ed + (1 - w) * E, it["geo"]))
    return out


def task_select(out):
    import joblib
    est = f6_estimator()
    det = fit(est)
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(estimator=det, provenance=dict(twin=SC10, idsse=IDSSE_TRAIN, target="event_targets radius 2")),
                ART / "detector.joblib", compress=3)
    table = {}
    for key, spec in eval_sets(N10A).items():
        sc = scores(item(*spec), est, det)
        table[key] = {k: v["spell"] for k, v in sc.items()}
        table[key].update({"n_pred_" + k: v["n_pred"] for k, v in sc.items()})
        table[key]["n_gt"] = sc["F6"]["n_gt"]
        print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in table[key].items()}, flush=True)
    res = {}
    for w in WEIGHTS:
        c = "W%g" % w
        n = [r[c] - r["F6"] for k, r in table.items() if k.startswith("N|")]
        g = {k: r[c] - r["F6"] for k, r in table.items() if k.startswith("G|")}
        res[c] = dict(n_delta=float(np.mean(n)), n_wins=int(sum(d > 0 for d in n)), worst_guard=float(min(g.values())),
                      guards=g, passes=bool(np.mean(n) >= .015 and min(g.values()) >= -.01))
        print(c, "N10-A %+.4f (%d/5 wins)  worst guard %+.4f  passes %s" % (res[c]["n_delta"], res[c]["n_wins"],
                                                                         res[c]["worst_guard"], res[c]["passes"]), flush=True)
    ok = [c for c, r in res.items() if r["passes"]]
    chosen = max(ok, key=lambda c: res[c]["n_delta"]) if ok else None
    Path(out).write_text(json.dumps(dict(rule="N10-A >= F6 + 1.5, guards >= -1.0", results=res, table=table,
                                         chosen=chosen), indent=1))
    print("chosen", chosen)


def task_test(out, sel):
    import joblib
    chosen = json.loads(Path(sel).read_text())["chosen"]
    if chosen is None:
        raise SystemExit("no candidate passed selection; N10-B stays closed")
    if Path(out).exists():
        raise SystemExit("%s exists: N10-B has already been opened" % out)
    w = float(chosen[1:])
    est, det = f6_estimator(), joblib.load(ART / "detector.joblib")["estimator"]
    rows = {}
    for n in N10B:
        it = item(n, TWIN_VARIANT, "clean")
        S6, E = f6_parts(it, est)
        a = it["mm"].score(decode(S6, E, it["geo"]))
        Ed = det_proba(det, det_features(it, est))
        b = it["mm"].score(decode(S6, w * Ed + (1 - w) * E, it["geo"]))
        rows[n] = dict(F6=a["spell"], cand=b["spell"], F6_macro=a["macro"], cand_macro=b["macro"],
                       n_pred_F6=a["n_pred"], n_pred_cand=b["n_pred"], n_gt=a["n_gt"])
        print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[n].items()}, flush=True)
    d = float(np.mean([r["cand"] - r["F6"] for r in rows.values()]))
    res = dict(candidate=chosen, rows=rows, mean_delta_spell=d,
               mean_delta_macro=float(np.mean([r["cand_macro"] - r["F6_macro"] for r in rows.values()])),
               new_package=d >= .015)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def task_pack(out, test):
    res = json.loads(Path(test).read_text())
    if not res.get("new_package"):
        raise SystemExit("test did not pass; no package")
    out = Path(out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    shutil.copytree(F6, out)
    shutil.copyfile(ART / "detector.joblib", out / "detector.joblib")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["name"] = "rondolabs-r8-F7"
    man["event_detector"] = dict(file="detector.joblib", weight=float(res["candidate"][1:]),
                                 sha256=hashlib.sha256((out / "detector.joblib").read_bytes()).hexdigest())
    man["description"] += (" Round 11 F7: a possession-change detector (HGB on F6 posteriors, events, geometry and "
                           "control-memory features; 10 SkillCorner + 5 IDSSE matches) is mixed into the event "
                           "probabilities at weight %g." % man["event_detector"]["weight"])
    man["selection"] = "Round 11: selected on N10-A, tested once on N10-B (%+.4f spellF1 vs F6)." % res["mean_delta_spell"]
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out)


def apply_detector(state, event, geometry, package, cfg):
    """Used by rondo.predict for packages with an `event_detector` block (F6 and earlier never call it)."""
    import joblib
    path = Path(package) / cfg["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["sha256"]:
        raise ValueError("detector file does not match manifest sha256")
    det = joblib.load(path)["estimator"]
    x = features(geometry, np.asarray(state, float), np.asarray(event, float), True)
    w = float(cfg["weight"])
    return (w * det_proba(det, x) + (1 - w) * np.asarray(event, float)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["select", "test", "pack"])
    p.add_argument("--out", required=True)
    p.add_argument("--select", default=str(ROOT / "reports/rondo/r11_select.json"))
    p.add_argument("--test", default=str(ROOT / "reports/rondo/r11_test.json"))
    a = p.parse_args()
    {"select": lambda: task_select(a.out), "test": lambda: task_test(a.out, a.select),
     "pack": lambda: task_pack(a.out, a.test)}[a.task]()


if __name__ == "__main__":
    main()
