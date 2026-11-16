"""Round 13: seed ensemble of the R12 graph network (variance reduction, no new tunable setting).

Pre-registration: reports/rondo/r10_preregistration.md (addendum R13, written before any result).
  Train two more GNNs with the exact R12 recipe, seeds 20261117 and 20261118. F8 = F6 posteriors
  mixed at w = 0.5 with the mean of the three GNNs (R12 seed + the two new ones).
  Adopt F8 iff, against F7: mean spellF1 over all 10 N10 matches >= F7, and every guard
  (clean, target-like) loses <= 1.0. Otherwise F7 stays. Each seed's own delta vs F6 is reported.

    python -m rondo.r13 run  --out reports/rondo/r13.json
    python -m rondo.r13 pack --out models/rondo_r9      (only if run says adopt)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import gnn
from .matches import ROOT
from .r10 import N10, decode, eval_sets, f6_estimator, item
from .r11 import f6_parts
from .r12 import ART as R12ART, THREADS, gnn_item
from .twin import IDSSE_TRAIN, SC10, TWIN_VARIANT

SEEDS = (20261117, 20261118)
W = .5
ART = ROOT / "data/artifacts/rondo/r13"
F7 = ROOT / "models/rondo_r8"


def train_seeds():
    ART.mkdir(parents=True, exist_ok=True)
    todo = [s for s in SEEDS if not (ART / ("gnn_%d.pt" % s)).is_file()]
    if not todo:
        return
    groups = {"twin": [gnn_item(n, TWIN_VARIANT, "clean") for n in SC10],
              "idsse": [gnn_item(n, "merge<1s", c) for n in IDSSE_TRAIN for c in ("clean", "target_like")]}
    for s in todo:
        net = gnn.train(groups, epochs=4, per_epoch=300000, batch=512, seed=s, threads=THREADS,
                        log=lambda m: print(m, flush=True))
        gnn.save(net, ART / ("gnn_%d.pt" % s), dict(seed=s, twin=SC10, idsse=IDSSE_TRAIN))
        print(json.dumps(dict(stage="saved", seed=s)), flush=True)


def task_run(out):
    train_seeds()
    nets = [gnn.load(R12ART / "gnn.pt")] + [gnn.load(ART / ("gnn_%d.pt" % s)) for s in SEEDS]
    est = f6_estimator()
    sets = {"N|%s" % n: (n, TWIN_VARIANT, "clean") for n in N10}
    sets.update({k: v for k, v in eval_sets([]).items()})
    table = {}
    for key, spec in sets.items():
        it = item(*spec)
        S6, E = f6_parts(it, est)
        Ps = [gnn.predict(net, it["geo"], threads=THREADS) for net in nets]
        sc = lambda S: it["mm"].score(decode(S, E, it["geo"]))["spell"]
        row = {"F6": sc(S6), "F7": sc((1 - W) * S6 + W * Ps[0])}
        for i, s in enumerate(SEEDS):
            row["seed%d" % s] = sc((1 - W) * S6 + W * Ps[i + 1])
        row["F8"] = sc((1 - W) * S6 + W * np.mean(Ps, axis=0))
        table[key] = row
        print(key, {k: round(v, 4) for k, v in row.items()}, flush=True)
    n = {c: float(np.mean([r[c] for k, r in table.items() if k.startswith("N|")])) for c in next(iter(table.values()))}
    g = {k: r["F8"] - r["F7"] for k, r in table.items() if k.startswith("G|")}
    adopt = bool(n["F8"] >= n["F7"] and min(g.values()) >= -.01)
    res = dict(rule="F8 adopted iff N10 mean >= F7 and every guard >= F7 - 1.0", n10_mean=n,
               seed_deltas_vs_F6={c: n[c] - n["F6"] for c in n if c != "F6"},
               f8_minus_f7_guards=g, worst_guard=float(min(g.values())), adopt=adopt, table=table)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=1))


def task_pack(out, run):
    res = json.loads(Path(run).read_text())
    if not res.get("adopt"):
        raise SystemExit("rule not met; F7 stays")
    out = Path(out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    shutil.copytree(F7, out)
    files = ["gnn.pt"]
    for s in SEEDS:
        shutil.copyfile(ART / ("gnn_%d.pt" % s), out / ("gnn_%d.pt" % s))
        files.append("gnn_%d.pt" % s)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["name"] = "rondolabs-r9-F8"
    man["gnn"] = dict(files=files, weight=W,
                      sha256={f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files})
    man["description"] += " Round 13 F8: the graph network is the mean of three seeds (same recipe)."
    man["selection"] = "Round 13: three-seed mean adopted under the pre-registered no-worse rule (see reports/rondo/r13.json)."
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["run", "pack"])
    p.add_argument("--out", required=True)
    p.add_argument("--run", default=str(ROOT / "reports/rondo/r13.json"))
    a = p.parse_args()
    task_run(a.out) if a.task == "run" else task_pack(a.out, a.run)


if __name__ == "__main__":
    main()
