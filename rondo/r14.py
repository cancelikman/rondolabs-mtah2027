"""Round 14: the R12 graph network trained with all 20 SkillCorner matches (two-fold cross-validation).

Pre-registration: reports/rondo/r10_preregistration.md (addendum R14, written before any result).
  Same recipe as R12 (seed 20261116, 4 x 300k windows, equal samples per source, IDSSE clean +
  target-like), only the SkillCorner training set grows:
    fold A: 10 old + N10-A  -> scored on N10-B;   fold B: 10 old + N10-B -> scored on N10-A.
  Each fold model replaces F7's network (w = 0.5 with F6's final posteriors, F6 events/decoder).
  Adopt iff the mean spellF1 delta vs F7 is >= +0.5 on BOTH folds and every guard (clean,
  target-like), scored with both fold models, is within 1.0 of F7. Then the final network is
  trained on all 20 SkillCorner matches + IDSSE with the same recipe -> F9.

    python -m rondo.r14 cv    --out reports/rondo/r14_cv.json
    python -m rondo.r14 final                        (only if cv says adopt; trains the 20-match net)
    python -m rondo.r14 pack  --out models/rondo_r9
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
from .r10 import N10A, N10B, decode, eval_sets, f6_estimator, item
from .r11 import f6_parts
from .r12 import ART as R12ART, THREADS, gnn_item
from .twin import IDSSE_TRAIN, SC10, TWIN_VARIANT

W = .5
SEED = 20261116
ART = ROOT / "data/artifacts/rondo/r14"
F7 = ROOT / "models/rondo_r8"


def train_net(sc, path):
    if path.is_file():
        return gnn.load(path)
    groups = {"twin": [gnn_item(n, TWIN_VARIANT, "clean") for n in sc],
              "idsse": [gnn_item(n, "merge<1s", c) for n in IDSSE_TRAIN for c in ("clean", "target_like")]}
    net = gnn.train(groups, epochs=4, per_epoch=300000, batch=512, seed=SEED, threads=THREADS,
                    log=lambda m: print(m, flush=True))
    ART.mkdir(parents=True, exist_ok=True)
    gnn.save(net, path, dict(sc=sc, idsse=IDSSE_TRAIN, seed=SEED))
    print(json.dumps(dict(stage="saved", path=str(path))), flush=True)
    return net


def score(spec, nets, est):
    it = item(*spec)
    S6, E = f6_parts(it, est)
    out = {}
    for name, net in nets.items():
        P = gnn.predict(net, it["geo"], threads=THREADS)
        out[name] = it["mm"].score(decode((1 - W) * S6 + W * P, E, it["geo"]))["spell"]
    return out


def task_cv(out):
    f7net = gnn.load(R12ART / "gnn.pt")
    netA = train_net(SC10 + N10A, ART / "gnn_foldA.pt")
    netB = train_net(SC10 + N10B, ART / "gnn_foldB.pt")
    est = f6_estimator()
    folds = {}
    for fold, net, test in (("A", netA, N10B), ("B", netB, N10A)):
        rows = {}
        for n in test:
            rows[n] = score((n, TWIN_VARIANT, "clean"), {"F7": f7net, "new": net}, est)
            print(fold, n, {k: round(v, 4) for k, v in rows[n].items()}, flush=True)
        d = [r["new"] - r["F7"] for r in rows.values()]
        folds[fold] = dict(rows=rows, mean_delta=float(np.mean(d)), wins=int(sum(x > 0 for x in d)))
    guards = {}
    for key, spec in eval_sets([]).items():
        guards[key] = score(spec, {"F7": f7net, "A": netA, "B": netB}, est)
        print(key, {k: round(v, 4) for k, v in guards[key].items()}, flush=True)
    worst = float(min(min(g["A"], g["B"]) - g["F7"] for g in guards.values()))
    adopt = bool(all(f["mean_delta"] >= .005 for f in folds.values()) and worst >= -.01)
    res = dict(rule="both folds >= F7 + 0.5 and every guard (both fold models) >= F7 - 1.0",
               fold_A_on_N10B=folds["A"]["mean_delta"], fold_B_on_N10A=folds["B"]["mean_delta"],
               wins={k: f["wins"] for k, f in folds.items()}, worst_guard=worst, adopt=adopt,
               folds=folds, guards=guards)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("folds", "guards")}, indent=1))


def task_final(cv):
    if not json.loads(Path(cv).read_text()).get("adopt"):
        raise SystemExit("cv rule not met; F7 stays")
    train_net(SC10 + N10A + N10B, ART / "gnn_sc20.pt")


def task_pack(out, cv):
    if not json.loads(Path(cv).read_text()).get("adopt"):
        raise SystemExit("cv rule not met; F7 stays")
    out = Path(out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    shutil.copytree(F7, out)
    shutil.copyfile(ART / "gnn_sc20.pt", out / "gnn.pt")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["name"] = "rondolabs-r9-F9"
    man["gnn"] = dict(file="gnn.pt", weight=W, sha256=hashlib.sha256((out / "gnn.pt").read_bytes()).hexdigest())
    man["description"] += " Round 14 F9: the graph network is retrained on all 20 SkillCorner matches + IDSSE."
    man["selection"] = "Round 14: two-fold cross-validation over the 10 newest SkillCorner matches (reports/rondo/r14_cv.json)."
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["cv", "final", "pack"])
    p.add_argument("--out")
    p.add_argument("--cv", default=str(ROOT / "reports/rondo/r14_cv.json"))
    a = p.parse_args()
    if a.task == "cv":
        task_cv(a.out or a.cv)
    elif a.task == "final":
        task_final(a.cv)
    else:
        task_pack(a.out, a.cv)


if __name__ == "__main__":
    main()
