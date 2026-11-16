"""Round 15: the R12/R14 graph network with PFF FC World Cup 2022 as a third training source.

Pre-registration: reports/rondo/r10_preregistration.md (addendum R15, written before any result).
  Recipe identical to R14 (seed 20261116, 4 x 300k windows, batch 512, equal samples per source);
  the only change is a third source "pff": the 64 PFF matches (event-derived labels, variant fixed
  by `python -m rondo.pff_data fingerprint` = closest to the official self-shift curve), every match
  target-like (PFF snaps the ball onto the carrier during on-ball events, 0.004 m at event starts;
  the target-like offset removes that too-clean cue), ball gaps filled by interpolation (PFF drops
  the ball in ~98% of dead frames, which would leak the label). Same folds as R14:
    fold A: SC10 + N10-A + IDSSE + PFF -> scored on N10-B;   fold B: SC10 + N10-B + IDSSE + PFF -> N10-A.
  Baseline per fold = the current champion's network for that fold: R14's fold model if R14 was
  adopted (F9), otherwise F7's network. Adopt iff the mean spellF1 delta vs that baseline is
  >= +0.5 on BOTH folds and every guard (clean, target-like; both fold models) is within 1.0 of
  the champion. Then the final network = SC20 + IDSSE + PFF, packed as F10.

    python -m rondo.r15 cv    --out reports/rondo/r15_cv.json
    python -m rondo.r15 final
    python -m rondo.r15 pack  --out models/rondo_r10

RONDO_PFF_N limits the number of PFF matches (default all 64; set 32 if RAM is short).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import gnn
from .matches import ROOT
from .r10 import N10A, N10B, eval_sets, f6_estimator
from .r12 import ART as R12ART, THREADS, gnn_item
from .r14 import ART as R14ART, score
from .twin import IDSSE_TRAIN, SC10, TWIN_VARIANT

W = .5
SEED = 20261116
SLOTS = 13
ART = ROOT / "data/artifacts/rondo/r15"
FP = ROOT / "reports/rondo/pff_fingerprint.json"
F7 = ROOT / "models/rondo_r8"
F9 = ROOT / "models/rondo_r9"


def pff_variant():
    if not FP.is_file():
        raise SystemExit("run `python -m rondo.pff_data fingerprint --out %s` first" % FP)
    return json.loads(FP.read_text(encoding="utf-8"))["closest"]          # e.g. "v1|merge<0.5s"


def compact(it, slots=SLOTS):
    """Keep only present tracks, `slots` per team per frame (the network pools per frame)."""
    present = it.X[..., 0] > .5
    Xs, teams = [], []
    rows = np.arange(it.n)[:, None]
    for t in (0, 1):
        cols = np.flatnonzero(it.team == t)
        order = np.argsort(~present[:, cols], axis=1, kind="stable")[:, :slots]
        x = it.X[:, cols][rows, order]
        if x.shape[1] < slots:
            x = np.concatenate([x, np.zeros((it.n, slots - x.shape[1], x.shape[2]), x.dtype)], 1)
        Xs.append(x)
        teams += [t] * slots
    it.X = np.concatenate(Xs, 1)
    it.team = np.asarray(teams)
    present = it.X[..., 0] > .5
    it.M0 = present & (it.team == 0)[None, :]
    it.M1 = present & (it.team == 1)[None, :]
    return it


def pff_items(variant):
    from . import corruptions
    from .d2_baseline import variant_match
    from .pff_data import load_pff, match_ids
    ev, _, merge = variant.partition("|")
    ids = match_ids()
    n = int(os.environ.get("RONDO_PFF_N", len(ids)))
    out = []
    for gid in ids[:n]:
        t = time.time()
        m = load_pff("%s@%s" % (gid, ev))
        mm = variant_match(m, merge or "raw")
        cond = "target_like"
        geo = corruptions.CONDITIONS[cond](m.geometry, m.dead25)
        out.append(compact(gnn.Item(geo, mm.scorer_data["labels"])))
        del m, mm, geo
        gc.collect()
        print(json.dumps(dict(stage="pff_item", game=gid, cond=cond, s=round(time.time() - t, 1))), flush=True)
    return out


def train_net(sc, path, pff):
    if path.is_file():
        return gnn.load(path)
    groups = {"twin": [gnn_item(n, TWIN_VARIANT, "clean") for n in sc],
              "idsse": [gnn_item(n, "merge<1s", c) for n in IDSSE_TRAIN for c in ("clean", "target_like")],
              "pff": pff}
    net = gnn.train(groups, epochs=4, per_epoch=300000, batch=512, seed=SEED, threads=THREADS,
                    log=lambda m: print(m, flush=True))
    ART.mkdir(parents=True, exist_ok=True)
    gnn.save(net, path, dict(sc=sc, idsse=IDSSE_TRAIN, pff=len(pff), pff_variant=pff_variant(), seed=SEED))
    print(json.dumps(dict(stage="saved", path=str(path))), flush=True)
    return net


def champion():
    """'F9' iff R14's cv adopted it, else 'F7'. Needs only R14's fold networks, not the F9 package."""
    cv = ROOT / "reports/rondo/r14_cv.json"
    if not cv.is_file():
        raise SystemExit("R14 cv must finish first (reports/rondo/r14_cv.json)")
    return "F9" if json.loads(cv.read_text()).get("adopt") else "F7"


def task_cv(out):
    champ = champion()
    f7net = gnn.load(R12ART / "gnn.pt")
    base = {"A": gnn.load(R14ART / "gnn_foldA.pt"), "B": gnn.load(R14ART / "gnn_foldB.pt")} if champ == "F9" \
        else {"A": f7net, "B": f7net}
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
            nets = {"base": base[fold], "new": net, "F7": f7net}
            r14f = R14ART / ("gnn_fold%s.pt" % fold)
            if r14f.is_file():                       # information only: same SC set without PFF
                nets["r14_same_fold"] = gnn.load(r14f)
            rows[n] = score((n, TWIN_VARIANT, "clean"), nets, est)
            print(fold, n, {k: round(v, 4) for k, v in rows[n].items()}, flush=True)
        d = [r["new"] - r["base"] for r in rows.values()]
        folds[fold] = dict(rows=rows, mean_delta=float(np.mean(d)), wins=int(sum(x > 0 for x in d)))
    guards = {}
    for key, spec in eval_sets([]).items():
        guards[key] = score(spec, {"baseA": base["A"], "baseB": base["B"], "A": netA, "B": netB}, est)
        print(key, {k: round(v, 4) for k, v in guards[key].items()}, flush=True)
    worst = float(min(min(g["A"] - g["baseA"], g["B"] - g["baseB"]) for g in guards.values()))
    adopt = bool(all(f["mean_delta"] >= .005 for f in folds.values()) and worst >= -.01)
    res = dict(rule="both folds >= champion + 0.5 and every guard (each fold model vs its fold baseline) >= -1.0",
               champion=champ, pff_variant=pff_variant(),
               fold_A_on_N10B=folds["A"]["mean_delta"], fold_B_on_N10A=folds["B"]["mean_delta"],
               wins={k: f["wins"] for k, f in folds.items()}, worst_guard=worst, adopt=adopt,
               folds=folds, guards=guards)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("folds", "guards")}, indent=1))


def _adopted(cv):
    if not json.loads(Path(cv).read_text()).get("adopt"):
        raise SystemExit("cv rule not met; champion stays")


def task_final(cv):
    _adopted(cv)
    train_net(SC10 + N10A + N10B, ART / "gnn_sc20_pff.pt", pff_items(pff_variant()))


def task_pack(out, cv):
    _adopted(cv)
    out = Path(out)
    if out.exists():
        raise SystemExit("%s exists" % out)
    shutil.copytree(F7, out)
    shutil.copyfile(ART / "gnn_sc20_pff.pt", out / "gnn.pt")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["name"] = "rondolabs-r10-F10"
    man["gnn"] = dict(file="gnn.pt", weight=W, sha256=hashlib.sha256((out / "gnn.pt").read_bytes()).hexdigest())
    man["description"] += (" Round 15 F10: the graph network is retrained on all 20 SkillCorner matches + IDSSE"
                           " + the 64 PFF FC World Cup 2022 matches (event-derived labels).")
    man["selection"] = "Round 15: two-fold cross-validation over the 10 newest SkillCorner matches (reports/rondo/r15_cv.json)."
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=["cv", "final", "pack"])
    p.add_argument("--out")
    p.add_argument("--cv", default=str(ROOT / "reports/rondo/r15_cv.json"))
    a = p.parse_args()
    if a.task == "cv":
        task_cv(a.out or a.cv)
    elif a.task == "final":
        task_final(a.cv)
    else:
        task_pack(a.out, a.cv)


if __name__ == "__main__":
    main()
