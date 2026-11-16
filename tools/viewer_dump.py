"""Dump model outputs on a LABELLED EXTERNAL match for the error viewer (no competition data).

    python tools/viewer_dump.py SC2006229 --packages models/rondo_r5 models/rondo_r7

Writes reports/viewer/<match>.npz at the match's native frame rate:
ball, players, team_of_track, labels (raw and merged<0.5s), and for every package its
decoded labels and P(H,A,D). Members shared by packages (same weight files) run once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/viewer"


def member_posteriors(geometry, package, member, cache):
    from rondo.candidates import Candidate
    key = tuple(sorted(member.get("sha256", {}).values())) + (member.get("tcn_weight", .5),)
    if key not in cache:
        cand = Candidate(member.get("name", "member"), checkpoint=package / member["tcn"], parent=package,
                         hgb=package / member["hgb"], tcn_weight=float(member.get("tcn_weight", .5)),
                         hgb_when_hidden=float(member.get("hgb_when_hidden", 0.)))
        cache[key] = cand.combine(cand.networks(geometry), geometry)
    return cache[key]


def run_package(geometry, package, cache):
    from lab.model import decode
    from rondo.predict import apply_stack
    man = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    os.environ["RONDO_THREADS"] = str(int(man.get("threads", 2)))
    members = man.get("members") or [man]
    w = np.array([float(m.get("weight", 1.)) for m in members])
    w = w / w.sum()
    parts = [member_posteriors(geometry, package, m, cache) for m in members]
    S = sum(wi * np.asarray(p[0], float) for wi, p in zip(w, parts))
    E = sum(wi * np.asarray(p[1], float) for wi, p in zip(w, parts))
    S0 = np.asarray(S, np.float32)
    if man.get("stack"):
        S = apply_stack(S, E, geometry, package, man["stack"])
    pred = decode(S, E, geometry, man["decoder"])
    return np.asarray(S, np.float32), np.asarray(pred, np.int8), S0, np.asarray(E, np.float32)


def main():
    from rondo.d2_baseline import variant_match
    from rondo.matches import load
    p = argparse.ArgumentParser()
    p.add_argument("match")
    p.add_argument("--packages", nargs="+", type=Path, default=[ROOT / "models/rondo_r5", ROOT / "models/rondo_r7"])
    a = p.parse_args()
    t = time.time()
    m = load(a.match)
    mm = variant_match(m, "merge<0.5s")
    nat = m.scorer_data
    rows = m.mapping[:, 1] if m.mapping is not None else np.arange(len(nat["labels"]))
    out = dict(ball=np.asarray(nat["ball"], np.float32), players=np.asarray(nat["players"], np.float16),
               team_of_track=np.asarray(nat["team_of_track"]), fps=float(nat["fps"]),
               pitch=np.asarray(nat["pitch"], np.float32), blocks=np.asarray(nat["blocks"]),
               labels_raw=np.asarray(nat["labels"], np.int8), labels=np.asarray(mm.scorer_data["labels"], np.int8))
    scores, cache = {}, {}
    for pkg in a.packages:
        S, pred, S0, E = run_package(m.geometry, pkg, cache)
        tag = pkg.name
        out["S0_25_" + tag] = S0.astype(np.float16)   # 25 Hz, before the stack (for refits)
        out["E_25_" + tag] = E.astype(np.float16)
        out["pred_" + tag] = pred[rows]
        out["prob_" + tag] = S[rows].astype(np.float16)
        scores[tag] = mm.score(pred)
        print(a.match, tag, {k: round(v, 4) if isinstance(v, float) else v for k, v in scores[tag].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / ("%s.npz" % a.match), **out)
    (OUT / ("%s.json" % a.match)).write_text(json.dumps(dict(match=a.match, scores=scores, seconds=round(time.time() - t)), indent=1))


if __name__ == "__main__":
    main()
