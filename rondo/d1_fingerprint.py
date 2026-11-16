"""Round 8 D1: label granularity from the official self-shift fingerprint (external labels only).

EVALUATION.md gives the full-match ground truth scored against itself shifted by 3 / 8 / 25
frames: 0.9818 / 0.8933 / 0.6539. For every candidate label set we compute the same curve with
the official possession_metrics.spell_scores (inside each valid run, pred[t] = gt[t - s]),
plus live possessions per minute, mean / median duration and the shares under 0.7 / 1.8 / 3 /
5.7 s. Distance = |d3| + |d8| + |d25|. The IDSSE-based variant with the smallest distance is the
target T*; every external source uses its own closest variant. 10 Hz sources are upsampled to
25 Hz by nearest row first.

    python -m rondo.d1_fingerprint --out reports/rondo/d1_fingerprint.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

OFFICIAL = {1: 1.0, 3: .9818, 8: .8933, 25: .6539}
SHIFTS = (1, 3, 8, 25)
MERGES = (.5, 1., 2.)


def to25(labels, blocks, fps):
    lab = np.asarray(labels)
    if fps == 25:
        return lab.copy()
    out = []
    for a, b in blocks:
        n = b - a
        n25 = int(np.floor((n - 1) * 25 / fps)) + 1
        idx = np.minimum(np.rint(np.arange(n25) * fps / 25).astype(int), n - 1)
        out.append(lab[a:b][idx])
        out.append(np.array([-1], np.int8))
    return np.concatenate(out)


def codes(lab25):
    c = np.asarray(lab25).astype(np.int64)
    return np.where(c < 0, pm.X, c)


def shifted(c, runs, s):
    p = c.copy()
    for a, b in runs:
        if b - a > s:
            p[a + s:b] = c[a:b - s]
            p[a:a + s] = c[a]
    return p


def fingerprint(lab25):
    c = codes(lab25)
    runs = pm.valid_runs(c)
    curve = {s: float(pm.spell_scores(c, shifted(c, runs, s), runs)["mean_f1"]) for s in SHIFTS}
    spells = pm.possession_spells(c, runs)
    d = np.array([(e - s) / 25. for s, e, _ in spells])
    live_min = np.isin(c, [0, 1]).sum() / 25 / 60
    return dict(curve=curve, distance=float(sum(abs(curve[s] - OFFICIAL[s]) for s in (3, 8, 25))),
                possessions_per_live_min=float(len(d) / live_min), mean_s=float(d.mean()), median_s=float(np.median(d)),
                share_lt={str(t): float((d < t).mean()) for t in (.7, 1.8, 3., 5.7)})


def variants(data):
    from src.features import blocks_of
    from .matches import merged_labels
    from .relabel import VARIANTS, apply
    blocks = [tuple(map(int, b)) for b in blocks_of(data)]
    fps = float(data["fps"])
    out = {"raw": to25(data["labels"], blocks, fps)}
    for s in MERGES:
        out["merge<%gs" % s] = to25(merged_labels(data["labels"], blocks, s, fps), blocks, fps)
    for v in VARIANTS:
        out[v] = to25(apply(data, v)["labels"], blocks, fps)
    return out


def average(fps_list):
    keys = fps_list[0].keys()
    out = {}
    for k in keys:
        if k == "curve":
            out[k] = {s: float(np.mean([f[k][s] for f in fps_list])) for s in SHIFTS}
        elif k == "share_lt":
            out[k] = {t: float(np.mean([f[k][t] for f in fps_list])) for t in fps_list[0][k]}
        else:
            out[k] = float(np.mean([f[k] for f in fps_list]))
    out["distance"] = float(sum(abs(out["curve"][s] - OFFICIAL[s]) for s in (3, 8, 25)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from src.train import load_external
    from .matches import load_wcsf
    from .open_data import load_metrica_native, load_skillcorner_native, skillcorner_ids
    sources = {}
    for code in ("J03WOH", "J03WN1", "J03WMX", "J03WQQ", "J03WPY", "J03WOY"):
        sources.setdefault("IDSSE", []).append((code, load_external(ROOT / "data/idsse_external" / ("%s_external_v1.npz" % code))))
    sources["Metrica_control"] = [("MT%d" % g, load_metrica_native(g, "control")) for g in (1, 2)]
    sources["Metrica_duel"] = [("MT%d" % g, load_metrica_native(g, "duel")) for g in (1, 2)]
    sources["SkillCorner"] = [(i, load_skillcorner_native(i)) for i in skillcorner_ids()[:10]]
    sources["Groningen"] = [("WCSF", load_wcsf(corrected=True).scorer_data)]
    report = dict(official=OFFICIAL, sources={})
    for src, items in sources.items():
        per = {}
        for name, data in items:
            for v, lab in variants(data).items():
                per.setdefault(v, []).append(fingerprint(lab))
            print(src, name, "done", flush=True)
        report["sources"][src] = {v: average(f) for v, f in per.items()}
        if src == "IDSSE":
            report["IDSSE_per_match_raw"] = {n: fingerprint(variants(d)["raw"]) for n, d in items}
    idsse = report["sources"]["IDSSE"]
    t_star = min(idsse, key=lambda v: idsse[v]["distance"])
    report["T_star"] = t_star
    report["closest_variant_per_source"] = {s: min(v, key=lambda k: v[k]["distance"]) for s, v in report["sources"].items()}
    report["distance_gap_T_star_vs_IDSSE_raw"] = idsse["raw"]["distance"] - idsse[t_star]["distance"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for s, v in report["sources"].items():
        for k, f in v.items():
            print("%-16s %-10s d=%.4f curve %s per_min %.2f mean %.1f med %.1f <1.8s %.2f <3s %.2f <5.7s %.2f" % (
                s, k, f["distance"], [round(f["curve"][x], 4) for x in SHIFTS], f["possessions_per_live_min"],
                f["mean_s"], f["median_s"], f["share_lt"]["1.8"], f["share_lt"]["3.0"], f["share_lt"]["5.7"]))
    print("T*", t_star, "closest per source", report["closest_variant_per_source"])


if __name__ == "__main__":
    main()
