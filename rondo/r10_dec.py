"""Round 10 E2: "control start" evidence for the decoder (no training).

Pre-registration: reports/rondo/r10_preregistration.md (addendum E2).
  F6 unchanged up to the decoder. Where a team starts a control run (rondo/control_memory.py,
  fixed defaults) after the last control was the other team's, the event model's P(new team)
  is raised to `strength` within +-0.2 s. Strength in {0.3, 0.5}; nothing else is tuned.
  Select: N10-A mean spellF1 >= F6 + 1.5 and every guard loses <= 1.0. Test once on N10-B,
  new package iff mean delta >= +1.5.

    python -m rondo.r10_dec select --out reports/rondo/r10e2_select.json
    python -m rondo.r10_dec test   --out reports/rondo/r10e2_test.json    (once)
    python -m rondo.r10_dec pack   --out models/rondo_r8                  (only if test passes)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .control_memory import control_event_boost
from .matches import ROOT
from .r10 import F6, N10A, N10B, ALPHA, decode, eval_sets, f6_estimator, item, state, feats_of
from .twin import TWIN_VARIANT

STRENGTHS = (.3, .5)


def f6_state(it, est):
    S = np.asarray(it["S"], float)
    st = state(est, feats_of(it, False), False)
    return ALPHA * st + (1 - ALPHA) * S


def scores(it, est):
    S6 = f6_state(it, est)
    out = {"F6": it["mm"].score(decode(S6, it["E"], it["geo"]))}
    for s in STRENGTHS:
        E = control_event_boost(it["E"], it["geo"], s)
        out["E%g" % s] = it["mm"].score(decode(S6, E, it["geo"]))
    return out


def task_select(out):
    est = f6_estimator()
    table = {}
    for key, spec in eval_sets(N10A).items():
        sc = scores(item(*spec), est)
        table[key] = {k: v["spell"] for k, v in sc.items()}
        table[key].update({"n_pred_" + k: v["n_pred"] for k, v in sc.items()})
        table[key]["n_gt"] = sc["F6"]["n_gt"]
        print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in table[key].items()}, flush=True)
    res = {}
    for s in STRENGTHS:
        c = "E%g" % s
        n = [r[c] - r["F6"] for k, r in table.items() if k.startswith("N|")]
        g = {k: r[c] - r["F6"] for k, r in table.items() if k.startswith("G|")}
        res[c] = dict(n_delta=float(np.mean(n)), n_wins=int(sum(d > 0 for d in n)), worst_guard=float(min(g.values())),
                      guards=g, passes=bool(np.mean(n) >= .015 and min(g.values()) >= -.01))
        print(c, "N10-A %+.4f (%d/5 wins)  worst guard %+.4f  passes %s" % (res[c]["n_delta"], res[c]["n_wins"],
                                                                         res[c]["worst_guard"], res[c]["passes"]), flush=True)
    ok = [c for c, r in res.items() if r["passes"]]
    chosen = max(ok, key=lambda c: res[c]["n_delta"]) if ok else None
    Path(out).write_text(json.dumps(dict(rule="N10-A >= F6 + 1.5, guards >= -1.0", results=res, table=table, chosen=chosen), indent=1))
    print("chosen", chosen)


def task_test(out, sel):
    chosen = json.loads(Path(sel).read_text())["chosen"]
    if chosen is None:
        raise SystemExit("no candidate passed selection; N10-B stays closed")
    if Path(out).exists():
        raise SystemExit("%s exists: N10-B has already been opened" % out)
    s = float(chosen[1:])
    est = f6_estimator()
    rows = {}
    for n in N10B:
        it = item(n, TWIN_VARIANT, "clean")
        S6 = f6_state(it, est)
        a = it["mm"].score(decode(S6, it["E"], it["geo"]))
        b = it["mm"].score(decode(S6, control_event_boost(it["E"], it["geo"], s), it["geo"]))
        rows[n] = dict(F6=a["spell"], cand=b["spell"], F6_macro=a["macro"], cand_macro=b["macro"],
                       n_pred_F6=a["n_pred"], n_pred_cand=b["n_pred"], n_gt=a["n_gt"])
        print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[n].items()}, flush=True)
    d = float(np.mean([r["cand"] - r["F6"] for r in rows.values()]))
    res = dict(candidate=chosen, rows=rows, mean_delta_spell=d,
               mean_delta_macro=float(np.mean([r["cand_macro"] - r["F6_macro"] for r in rows.values()])), new_package=d >= .015)
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
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["name"] = "rondolabs-r8-F7"
    man["decoder"]["control_event"] = dict(strength=float(res["candidate"][1:]))
    man["description"] = man["description"] + (" Round 10 F7: decoder gets control-start evidence "
                                               "(rondo/control_memory.control_event_boost, strength %g)." % float(res["candidate"][1:]))
    man["selection"] = "Round 10 E2: selected on N10-A, tested once on N10-B (%+.4f spellF1 vs F6)." % res["mean_delta_spell"]
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["select", "test", "pack"])
    p.add_argument("--out", required=True)
    p.add_argument("--select", default=str(ROOT / "reports/rondo/r10e2_select.json"))
    p.add_argument("--test", default=str(ROOT / "reports/rondo/r10e2_test.json"))
    a = p.parse_args()
    {"select": lambda: task_select(a.out), "test": lambda: task_test(a.out, a.select),
     "pack": lambda: task_pack(a.out, a.test)}[a.task]()


if __name__ == "__main__":
    main()
