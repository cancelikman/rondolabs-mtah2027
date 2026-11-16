"""Measurement table: candidate x {match} x {condition} x {dead-ball prior k}.

External matches only. Posteriors are cached under data/artifacts/rondo/ (git-ignored),
the aggregate table is written as JSON (aggregate numbers only).

    python -m rondo.evaluate --candidate ref --matches WOY WCSF --out reports/rondo/ref.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import candidates, corruptions
from .matches import ROOT, load

CACHE = ROOT / "data/artifacts/rondo/cache"


def posteriors(cand, match, cond_name, geo):
    if hasattr(cand, "members"):
        outs = [posteriors(m, match, cond_name, geo) for m in cand.members]
        return (np.mean([o[0] for o in outs], axis=0).astype(np.float32),
                np.mean([o[1] for o in outs], axis=0).astype(np.float32))
    path = CACHE / cand.network_key / getattr(match, "cache_name", match.name) / ("%s_parts.npz" % cond_name)
    legacy = CACHE / cand.network_key / getattr(match, "cache_name", match.name) / ("%s.npz" % cond_name)
    if not path.is_file() and legacy.is_file() and cand.tcn_weight == .5 and not cand.hgb_when_hidden:
        with np.load(legacy) as f:   # combined 50:50 state from the first S3 run
            return f["state"], f["event"]
    if path.is_file():
        with np.load(path) as f:
            parts = {k: f[k] for k in f.files}
    else:
        parts = cand.networks(geo)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **parts)
    return cand.combine(parts, geo)


def run(cand, match_names, cond_names, ks, extra=None):
    from lab.model import decode
    rows = []
    for mname in match_names:
        match = load(mname)
        for cname in cond_names:
            t = time.time()
            geo = corruptions.CONDITIONS[cname](match.geometry, match.dead25)
            state, event = posteriors(cand, match, cname, geo)
            for k in ks:
                settings = dict(cand.decoder, dead_bias=float(k), **(extra or {}))
                pred = cand.postprocess(decode(state, event, geo, settings), geo)
                s = match.score(pred)
                rows.append(dict(candidate=cand.name, match=mname, condition=cname, k=float(k), **s))
                print("%-10s %-5s %-16s k=%-5g spell %.4f macro %.4f Drec %.3f Dprec %.3f n %d/%d (%.0fs)" % (
                    cand.name, mname, cname, k, s["spell"], s["macro"], s["d_recall"], s["d_precision"],
                    s["n_pred"], s["n_gt"], time.time() - t), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", default="ref")
    p.add_argument("--matches", nargs="+", default=["WOY", "WCSF"])
    p.add_argument("--conditions", nargs="+", default=list(corruptions.CONDITIONS))
    p.add_argument("--k", nargs="+", type=float, default=[1, 4, 8, 20])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cand = candidates.get(a.candidate)
    rows = run(cand, a.matches, a.conditions, a.k)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(candidate=cand.describe(), rows=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
