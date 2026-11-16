"""Round 16: R15 with a separate output head for PFF (shared trunk, own label definition).

Why: R15 showed PFF makes the network better on other providers (+1 to +4.5 on the guards) but
worse on SkillCorner (-0.6 to -0.9). PFF's possession comes from events (team of the last
on-the-ball event); the target labels follow control. One shared head has to average the two
definitions. Here the trunk (player/ball encoders, pooling, temporal convolutions) learns from all
sources, but PFF windows train their own H/A/D head; SkillCorner + IDSSE train head 0, the only
head used at inference.

Pre-registration: reports/rondo/r10_preregistration.md (addendum R16, before any R16 number).
Everything else identical to R15 (seed, 4 x 300k windows, equal share per source, 64 PFF matches
target-like with filled ball, variant v0|raw, folds, baseline = F7's network, rule).

    python -m rondo.r16 cv    --out reports/rondo/r16_cv.json
    python -m rondo.r16 final
    python -m rondo.r16 pack  --out models/rondo_r10
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import gnn
from .matches import ROOT
from .r10 import N10A, N10B, eval_sets, f6_estimator
from .r12 import ART as R12ART, THREADS, gnn_item
from .r14 import ART as R14ART, score
from .r15 import ART as R15ART, F7, pff_items, pff_variant
from .twin import IDSSE_TRAIN, SC10, TWIN_VARIANT

W = .5
SEED = 20261116
HEADS = {"twin": 0, "idsse": 0, "pff": 1}
ART = ROOT / "data/artifacts/rondo/r16"


def train_net(sc, path, pff):
    if path.is_file():
        return gnn.load(path)
    groups = {"twin": [gnn_item(n, TWIN_VARIANT, "clean") for n in sc],
              "idsse": [gnn_item(n, "merge<1s", c) for n in IDSSE_TRAIN for c in ("clean", "target_like")],
              "pff": pff}
    net = gnn.train(groups, epochs=4, per_epoch=300000, batch=512, seed=SEED, threads=THREADS,
                    log=lambda m: print(m, flush=True), head_of=HEADS)
    ART.mkdir(parents=True, exist_ok=True)
    gnn.save(net, path, dict(sc=sc, idsse=IDSSE_TRAIN, pff=len(pff), pff_variant=pff_variant(), seed=SEED,
                             head_of=HEADS))
    print(json.dumps(dict(stage="saved", path=str(path))), flush=True)
    return net


def task_cv(out):
    f7net = gnn.load(R12ART / "gnn.pt")
    pff = pff_items(pff_variant())
    netA = train_net(SC10 + N10A, ART / "gnn_foldA.pt", pff)
    netB = train_net(SC10 + N10B, ART / "gnn_foldB.pt", pff)
    del pff
    gc.collect()
    est = f6_estimator()
    folds = {}
    for fold, net, test in (("A", netA, N10B), ("B", netB, N10A)):
        rows = {}
        for n in test:
            nets = {"base": f7net, "new": net}
            for tag, art in (("r14_same_fold", R14ART), ("r15_same_fold", R15ART)):   # information only
                f = art / ("gnn_fold%s.pt" % fold)
                if f.is_file():
                    nets[tag] = gnn.load(f)
            rows[n] = score((n, TWIN_VARIANT, "clean"), nets, est)
            print(fold, n, {k: round(v, 4) for k, v in rows[n].items()}, flush=True)
        d = [r["new"] - r["base"] for r in rows.values()]
        folds[fold] = dict(rows=rows, mean_delta=float(np.mean(d)), wins=int(sum(x > 0 for x in d)))
    guards = {}
    for key, spec in eval_sets([]).items():
        guards[key] = score(spec, {"base": f7net, "A": netA, "B": netB}, est)
        print(key, {k: round(v, 4) for k, v in guards[key].items()}, flush=True)
    worst = float(min(min(g["A"], g["B"]) - g["base"] for g in guards.values()))
    adopt = bool(all(f["mean_delta"] >= .005 for f in folds.values()) and worst >= -.01)
    res = dict(rule="both folds >= F7 + 0.5 and every guard (both fold models) >= F7 - 1.0",
               champion="F7", pff_variant=pff_variant(), heads=HEADS,
               fold_A_on_N10B=folds["A"]["mean_delta"], fold_B_on_N10A=folds["B"]["mean_delta"],
               wins={k: f["wins"] for k, f in folds.items()}, worst_guard=worst, adopt=adopt,
               folds=folds, guards=guards)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("folds", "guards")}, indent=1))


def _adopted(cv):
    if not json.loads(Path(cv).read_text()).get("adopt"):
        raise SystemExit("cv rule not met; F7 stays")


def task_final(cv):
    _adopted(cv)
    train_net(SC10 + N10A + N10B, ART / "gnn_sc20_pff_2head.pt", pff_items(pff_variant()))


def task_pack(out, cv):
    _adopted(cv)
    out = Path(out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    shutil.copytree(F7, out)
    shutil.copyfile(ART / "gnn_sc20_pff_2head.pt", out / "gnn.pt")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["name"] = "rondolabs-r10-F10"
    man["gnn"] = dict(file="gnn.pt", weight=W, sha256=hashlib.sha256((out / "gnn.pt").read_bytes()).hexdigest())
    man["description"] += (" Round 16 F10: the graph network's trunk is trained on all 20 SkillCorner matches,"
                           " IDSSE and the 64 PFF FC World Cup 2022 matches; PFF trains its own output head,"
                           " only the SkillCorner/IDSSE head is used.")
    man["selection"] = "Round 16: two-fold cross-validation over the 10 newest SkillCorner matches (reports/rondo/r16_cv.json)."
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=["cv", "final", "pack"])
    p.add_argument("--out")
    p.add_argument("--cv", default=str(ROOT / "reports/rondo/r16_cv.json"))
    a = p.parse_args()
    if a.task == "cv":
        task_cv(a.out or a.cv)
    elif a.task == "final":
        task_final(a.cv)
    else:
        task_pack(a.out, a.cv)


if __name__ == "__main__":
    main()
