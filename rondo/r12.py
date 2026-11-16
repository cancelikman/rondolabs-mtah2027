"""Round 12: set/graph model (rondo/gnn.py) as a third opinion mixed into F6.

Pre-registration: reports/rondo/r10_preregistration.md (addendum R12, written before any result).
  Model:  rondo.gnn SetNet, trained from scratch on 10 old SkillCorner twin matches (clean,
          <0.5 s merged) + 5 IDSSE training matches (clean and target-like, <1 s merged), equal
          samples per source, random home/away swap; fixed seed, 4 epochs x 300k windows.
  Use:    state = (1 - w) * F6 final posteriors + w * GNN, F6 events and decoder; w in {0.25, 0.5}.
  Select: N10-A mean spellF1 >= F6 + 1.5 and every guard loses <= 1.0. Test once on N10-B,
          new package iff mean delta >= +1.5.

    python -m rondo.r12 select --out reports/rondo/r12_select.json
    python -m rondo.r12 test   --out reports/rondo/r12_test.json
    python -m rondo.r12 pack   --out models/rondo_r8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from . import gnn
from .matches import ROOT
from .r10 import F6, N10A, N10B, decode, eval_sets, f6_estimator, item
from .r11 import f6_parts
from .twin import IDSSE_TRAIN, SC10, TWIN_VARIANT

WEIGHTS = (.25, .5)
ART = ROOT / "data/artifacts/rondo/r12"
THREADS = int(os.environ.get("RONDO_GNN_THREADS", "16"))


def gnn_item(name, variant, cond):
    from .phase_prior import gt25
    it = item(name, variant, cond)
    return gnn.Item(it["geo"], gt25(it["mm"]))


def task_select(out):
    ART.mkdir(parents=True, exist_ok=True)
    groups = {"twin": [gnn_item(n, TWIN_VARIANT, "clean") for n in SC10],
              "idsse": [gnn_item(n, "merge<1s", c) for n in IDSSE_TRAIN for c in ("clean", "target_like")]}
    net = gnn.train(groups, epochs=4, per_epoch=300000, batch=512, threads=THREADS,
                    log=lambda s: print(s, flush=True))
    del groups
    gnn.save(net, ART / "gnn.pt", dict(twin=SC10, idsse=IDSSE_TRAIN, epochs=4, per_epoch=300000))
    est = f6_estimator()
    table = {}
    for key, spec in eval_sets(N10A).items():
        it = item(*spec)
        S6, E = f6_parts(it, est)
        P = gnn.predict(net, it["geo"], threads=THREADS)
        row = {"F6": it["mm"].score(decode(S6, E, it["geo"]))}
        for w in WEIGHTS:
            row["W%g" % w] = it["mm"].score(decode((1 - w) * S6 + w * P, E, it["geo"]))
        table[key] = {k: v["spell"] for k, v in row.items()}
        table[key].update({"n_pred_" + k: v["n_pred"] for k, v in row.items()})
        table[key]["n_gt"] = row["F6"]["n_gt"]
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


def task_test(out, sel, candidate=None):
    chosen = candidate or json.loads(Path(sel).read_text())["chosen"]
    if chosen is None:
        raise SystemExit("no candidate passed selection; N10-B stays closed")
    if Path(out).exists():
        raise SystemExit("%s exists: N10-B has already been opened" % out)
    w = float(chosen[1:])
    est, net = f6_estimator(), gnn.load(ART / "gnn.pt")
    rows = {}
    for n in N10B:
        it = item(n, TWIN_VARIANT, "clean")
        S6, E = f6_parts(it, est)
        P = gnn.predict(net, it["geo"], threads=THREADS)
        a = it["mm"].score(decode(S6, E, it["geo"]))
        b = it["mm"].score(decode((1 - w) * S6 + w * P, E, it["geo"]))
        rows[n] = dict(F6=a["spell"], cand=b["spell"], F6_macro=a["macro"], cand_macro=b["macro"],
                       n_pred_F6=a["n_pred"], n_pred_cand=b["n_pred"], n_gt=a["n_gt"])
        print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[n].items()}, flush=True)
    d = float(np.mean([r["cand"] - r["F6"] for r in rows.values()]))
    wins = int(sum(r["cand"] > r["F6"] for r in rows.values()))
    rule = "R12-B: mean >= +1.5 and >= 4/5 wins" if candidate else "R12: mean >= +1.5"
    res = dict(candidate=chosen, rule=rule, rows=rows, mean_delta_spell=d, wins=wins,
               mean_delta_macro=float(np.mean([r["cand_macro"] - r["F6_macro"] for r in rows.values()])),
               new_package=bool(d >= .015 and (wins >= 4 or not candidate)))
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
    shutil.copyfile(ART / "gnn.pt", out / "gnn.pt")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["name"] = "rondolabs-r8-F7"
    man["gnn"] = dict(file="gnn.pt", weight=float(res["candidate"][1:]),
                      sha256=hashlib.sha256((out / "gnn.pt").read_bytes()).hexdigest())
    man["description"] += (" Round 12 F7: a set/graph model over the ball and all players (rondo/gnn.py; 10 SkillCorner "
                           "+ 5 IDSSE matches) is mixed into the final posteriors at weight %g." % man["gnn"]["weight"])
    man["selection"] = "Round 12: selected on N10-A, tested once on N10-B (%+.4f spellF1 vs F6)." % res["mean_delta_spell"]
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out)


def apply_gnn(state, geometry, package, cfg):
    """Used by rondo.predict for packages with a `gnn` block (F6 and earlier never call it)."""
    import torch
    torch.use_deterministic_algorithms(True)
    files = cfg.get("files") or [cfg["file"]]                  # F7: one file; F8 (Round 13): three seeds
    shas = cfg["sha256"] if isinstance(cfg["sha256"], dict) else {files[0]: cfg["sha256"]}
    if cfg.get("clean"):                        # Round 20 packages only: the network sees cleaned tracking
        from .clean import clean
        geometry = clean(geometry)
    Ps = []
    for f in files:
        path = Path(package) / f
        if hashlib.sha256(path.read_bytes()).hexdigest() != shas[f]:
            raise ValueError("gnn file %s does not match manifest sha256" % f)
        Ps.append(gnn.predict(gnn.load(path), geometry, threads=2))
    P = Ps[0] if len(Ps) == 1 else np.mean(Ps, axis=0)
    w = float(cfg["weight"])
    return ((1 - w) * np.asarray(state, float) + w * P).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["select", "test", "pack"])
    p.add_argument("--out", required=True)
    p.add_argument("--select", default=str(ROOT / "reports/rondo/r12_select.json"))
    p.add_argument("--test", default=str(ROOT / "reports/rondo/r12_test.json"))
    p.add_argument("--candidate", choices=["W0.5"], help="R12-B: the team-chosen candidate (see pre-registration)")
    a = p.parse_args()
    {"select": lambda: task_select(a.out), "test": lambda: task_test(a.out, a.select, a.candidate),
     "pack": lambda: task_pack(a.out, a.test)}[a.task]()


if __name__ == "__main__":
    main()
